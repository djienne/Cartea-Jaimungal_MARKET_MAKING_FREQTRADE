#!/usr/bin/env python3
"""Quoting core shared by the live engine and the replay harness.

Every piece of quote arithmetic in this project existed two or three times over:
``_assemble_half_spread`` in the strategy and ``assemble_half_spread`` in the
replay, ``_maker_safe`` / ``post_only_check`` / ``maker_safe``, ``_inventory_level``
/ ``inventory_q``, two tick-rounding implementations plus a third in the ALO
executor, and three separate ``finite_float_or_none``. Parity was maintained by
hand and asserted in one test (tests/test_quote_assembly.py). That is the wrong
place for the guarantee: a backtest whose quoting differs from the live path
tells you nothing, and the drift is silent until it costs money.

This module is the single implementation. The replay imports it so that what it
simulates is literally the code that will quote, and the engine will import it in
turn.

Conventions worth knowing before reading further:

- A half-spread of ``None`` means the side is DISABLED, not zero. The HJB returns
  an infinite depth at the inventory boundary (no bid at +q_max, no ask at
  -q_max), and that must never be clamped into a real quote.
- Depths are measured from the MID, in price units, on the same coordinate the
  estimators calibrate in.
- ``q`` is signed. The retired freqtrade strategy was long-only and clamped
  inventory to [0, q_max], which quietly discarded half of the HJB's own grid;
  the engine quotes both sides and uses the full [-q_max, +q_max].
- Tick and lot rounding delegate to hyperliquid_alo_executor, which does it in
  Decimal. The float floor/ceil copies elsewhere are subject to binary
  representation error at exactly the wrong moment -- a bid rounded one ULP up
  can cross the ask and turn a post-only order into a reject.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hjb import compute_h_asymmetric, compute_h_symmetric
from hyperliquid_alo_executor import round_amount_down, round_price_for_side

# Parameter snapshot schema this module consumes (scripts/param_utils.py).
SUPPORTED_PARAM_SCHEMA_VERSION = 3

# Keys every snapshot must carry before a quote may be built from it.
REQUIRED_PARAM_KEYS = ("kappa+", "kappa-", "lambda+", "lambda-", "epsilon+", "epsilon-")


def finite_float_or_none(value: Any) -> float | None:
    """Float, or None for anything non-numeric / non-finite."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_utc_timestamp(value: Any) -> datetime | None:
    """Parse an ISO string, epoch seconds, or epoch milliseconds into aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    try:
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1_000_000_000_000:  # milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        return as_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class QuoteConfig:
    """Everything that shapes a quote, in one place.

    Defaults mirror the values the freqtrade strategy shipped with, so replay
    results stay comparable across the migration.
    """

    maker_fee_rate: float = 0.00015
    spread_multiplier: float = 1.0
    min_half_spread_bps: float = 3.0
    max_half_spread_bps: float = 80.0

    inventory_unit_base: float = 0.01
    q_max: int = 3
    # The engine quotes both sides, so inventory runs the full signed grid. Set
    # False to reproduce the long-only strategy's [0, q_max] clamp.
    allow_short: bool = True

    hjb_alpha: float = 0.001
    hjb_phi: float = 0.0001
    hjb_horizon_seconds: float = 60.0
    gamma_inventory_risk: float = 0.05
    use_asymmetric_kappa: bool = True

    max_toxicity: float = 1.5
    max_param_age_seconds: float = 90.0
    max_future_timestamp_skew_seconds: float = 10.0
    min_kappa_fit_points: int = 6
    min_kappa_r2: float = 0.30
    min_epsilon_events: int = 50

    def __post_init__(self) -> None:
        if self.min_half_spread_bps >= self.max_half_spread_bps:
            raise ValueError(
                f"min_half_spread_bps ({self.min_half_spread_bps}) must be below "
                f"max_half_spread_bps ({self.max_half_spread_bps})"
            )
        if self.spread_multiplier <= 0 or not np.isfinite(self.spread_multiplier):
            raise ValueError("spread_multiplier must be a positive finite number")
        if int(self.q_max) < 1:
            raise ValueError("q_max must be >= 1")


@dataclass(frozen=True)
class HalfSpread:
    """A single side's final half-spread plus the diagnostics behind it."""

    delta: float
    delta_model: float
    delta_pre_clamp: float
    fee_cushion: float
    bps: float | None
    clamped: str | None  # "floor" | "cap" | None
    depth_p95: float | None
    outside_calibrated_range: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta_total": self.delta,
            "delta_model": self.delta_model,
            "delta_pre_clamp": self.delta_pre_clamp,
            "fee_cushion": self.fee_cushion,
            "bps": self.bps,
            "clamped": self.clamped,
            "depth_p95": self.depth_p95,
            "quote_outside_calibrated_range": self.outside_calibrated_range,
        }


def assemble_half_spread(
    delta_model: float,
    mid: float,
    config: QuoteConfig,
    *,
    depth_p95: float | None = None,
) -> HalfSpread | None:
    """Final half-spread for one side:

        delta_total = clamp(delta_model * spread_multiplier + maker_fee*mid,
                            min_half_spread_bps, max_half_spread_bps)

    The fee cushion is ONE maker fee per side: a round trip pays two fees and
    collects the cushion twice. The multiplier scales only the MODEL term, so
    widening quotes defensively never inflates the fee compensation.

    Returns None when ``delta_model`` is not finite -- that is the HJB's way of
    saying this side is disabled at this inventory, and clamping it into the
    [floor, cap] band would turn "do not quote" into "quote at the floor".

    ``depth_p95`` is the kappa fit's 95th-percentile market-order depth for this
    side. When the assembled quote sits beyond it the fill model is extrapolating
    past its own data, which is exactly what ETH did (0.565 vs a 0.05 p95), so
    the flag is surfaced rather than silently ignored.
    """
    model = finite_float_or_none(delta_model)
    if model is None:
        return None
    mid = float(mid)
    if not np.isfinite(mid) or mid <= 0:
        return None

    fee_cushion = float(config.maker_fee_rate) * mid
    delta_pre_clamp = model * float(config.spread_multiplier) + fee_cushion
    floor = float(config.min_half_spread_bps) / 10_000.0 * mid
    cap = float(config.max_half_spread_bps) / 10_000.0 * mid
    delta_total = min(max(delta_pre_clamp, floor), cap)

    clamped = None
    if delta_pre_clamp < floor:
        clamped = "floor"
    elif delta_pre_clamp > cap:
        clamped = "cap"

    p95 = finite_float_or_none(depth_p95)
    outside = None
    if p95 is not None and p95 > 0:
        outside = bool(delta_total > p95)

    return HalfSpread(
        delta=float(delta_total),
        delta_model=model,
        delta_pre_clamp=float(delta_pre_clamp),
        fee_cushion=float(fee_cushion),
        bps=(delta_total / mid) * 10_000.0,
        clamped=clamped,
        depth_p95=p95,
        outside_calibrated_range=outside,
    )


def maker_safe(
    side: str,
    price: float,
    best_bid: float,
    best_ask: float,
) -> tuple[bool, str]:
    """Would this price rest passively, or cross and take?

    Checked locally before submission because a post-only order that crosses is
    rejected by the exchange, and a stream of rejects is indistinguishable from a
    dead quoter unless it is caught here first.
    """
    price_f = finite_float_or_none(price)
    bid_f = finite_float_or_none(best_bid)
    ask_f = finite_float_or_none(best_ask)
    if price_f is None or price_f <= 0:
        return False, "invalid_rate"
    if bid_f is None or ask_f is None or bid_f <= 0 or ask_f <= 0 or bid_f >= ask_f:
        return False, "crossed_or_invalid_book"
    if side == "bid" and price_f >= ask_f:
        return False, "bid_crosses_ask"
    if side == "ask" and price_f <= bid_f:
        return False, "ask_crosses_bid"
    return True, "ok"


def inventory_to_q(signed_base: float, config: QuoteConfig) -> int:
    """Map a signed base position onto the HJB's integer inventory grid.

    With ``allow_short`` the result spans [-q_max, +q_max]; otherwise it is
    clamped to [0, q_max] to reproduce the long-only strategy.
    """
    value = finite_float_or_none(signed_base)
    if value is None:
        return 0
    unit = max(float(config.inventory_unit_base), 1e-12)
    q_max = int(config.q_max)
    if not config.allow_short:
        value = max(0.0, value)
    q = int(round(value / unit))
    return max(-q_max if config.allow_short else 0, min(q_max, q))


def effective_phi(config: QuoteConfig, sigma2_per_sec: Any = None) -> tuple[float, str]:
    """Volatility-aware running inventory penalty.

        phi_effective = hjb_phi + gamma * sigma2_per_sec * inventory_unit_base

    A missing or invalid sigma2 falls back to the static phi: a volatility
    channel going quiet must never stop the quoter.
    """
    base = float(config.hjb_phi)
    sigma2 = finite_float_or_none(sigma2_per_sec)
    if sigma2 is None or sigma2 < 0:
        return base, "phi_base_fallback"
    phi = base + float(config.gamma_inventory_risk) * sigma2 * float(config.inventory_unit_base)
    return float(phi), "sigma2_channel"


def solve_hjb(
    params: dict[str, Any],
    config: QuoteConfig,
    *,
    sigma2_per_sec: Any = None,
) -> dict[str, Any]:
    """Solve the HJB for the current parameter snapshot.

    ``params`` is the flat merged kappa/epsilon/lambda dict for one symbol.
    Raises whatever the solver raises -- callers keep their last known-good
    surface rather than quoting off a failed solve.
    """
    phi, phi_source = effective_phi(config, sigma2_per_sec)
    solver = compute_h_asymmetric if config.use_asymmetric_kappa else compute_h_symmetric
    result = solver(
        lambda_plus=float(params["lambda+"]),
        lambda_minus=float(params["lambda-"]),
        epsilon_plus=float(params["epsilon+"]),
        epsilon_minus=float(params["epsilon-"]),
        kappa_plus=float(params["kappa+"]),
        kappa_minus=float(params["kappa-"]),
        alpha=float(config.hjb_alpha),
        phi=phi,
        T_seconds=float(config.hjb_horizon_seconds),
        q_max=int(config.q_max),
    )
    result = dict(result)
    result["phi_effective"] = phi
    result["phi_source"] = phi_source
    result["phi_base"] = float(config.hjb_phi)
    result["sigma2_per_sec"] = finite_float_or_none(sigma2_per_sec)
    return result


def _grid_index(hjb: dict[str, Any], q: int) -> int:
    q_grid = np.asarray(hjb["q_grid"])
    if q <= q_grid[0]:
        return 0
    if q >= q_grid[-1]:
        return int(len(q_grid) - 1)
    return int(np.argmin(np.abs(q_grid - q)))


def select_delta(hjb: dict[str, Any], q: int, side: str) -> float | None:
    """Model depth for one side at inventory ``q``.

    Returns None for a disabled side. delta_minus prices the BID (a bid fill
    increases inventory), delta_plus prices the ASK.
    """
    if not hjb:
        return None
    idx = _grid_index(hjb, int(q))
    key = "delta_minus" if side == "bid" else "delta_plus"
    value = finite_float_or_none(np.asarray(hjb[key])[idx])
    return value


@dataclass
class QuotePair:
    """Both sides of a quote at one instant. ``None`` means the side is off."""

    mid: float
    q: int
    bid_price: float | None = None
    ask_price: float | None = None
    bid: HalfSpread | None = None
    ask: HalfSpread | None = None
    hjb_generation: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mid": self.mid,
            "q": self.q,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid": self.bid.as_dict() if self.bid else None,
            "ask": self.ask.as_dict() if self.ask else None,
            "hjb_generation": self.hjb_generation,
        }


def compute_quotes(
    mid: float,
    q: int,
    hjb: dict[str, Any],
    config: QuoteConfig,
    *,
    depth_p95_plus: float | None = None,
    depth_p95_minus: float | None = None,
    price_tick_size: float | None = None,
) -> QuotePair:
    """Both sides at once, which is the whole point of the rework.

    The freqtrade strategy could only ever have one resting order per pair, so it
    alternated: bid while flat, ask while long. That is not market making -- it
    halves the spread capture and turns the inventory dimension of the model into
    a two-state toggle. Here both sides are priced from the same mid, the same
    inventory and the same HJB surface, and either may be disabled independently
    at the inventory boundary.

    Prices are rounded maker-safe when a tick size is given: bids down, asks up,
    never toward the touch.
    """
    pair = QuotePair(mid=float(mid), q=int(q))

    bid_spread = assemble_half_spread(
        select_delta(hjb, q, "bid"), mid, config, depth_p95=depth_p95_minus
    )
    ask_spread = assemble_half_spread(
        select_delta(hjb, q, "ask"), mid, config, depth_p95=depth_p95_plus
    )

    pair.bid = bid_spread
    pair.ask = ask_spread
    if bid_spread is not None:
        price = float(mid) - bid_spread.delta
        pair.bid_price = round_price_for_side(
            side="bid", price=price, price_tick_size=price_tick_size
        )
    if ask_spread is not None:
        price = float(mid) + ask_spread.delta
        pair.ask_price = round_price_for_side(
            side="ask", price=price, price_tick_size=price_tick_size
        )
    return pair


def merged_params(
    kappa: dict[str, Any] | None,
    epsilon: dict[str, Any] | None,
    lambda_: dict[str, Any] | None,
) -> dict[str, Any]:
    """Flatten the three per-symbol snapshot entries into one dict."""
    merged: dict[str, Any] = {}
    for source in (kappa, epsilon, lambda_):
        if isinstance(source, dict):
            merged.update(source)
    return merged


def validate_param_snapshot(
    kappa: dict[str, Any] | None,
    epsilon: dict[str, Any] | None,
    lambda_: dict[str, Any] | None,
    config: QuoteConfig,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Fail-closed validation of a parameter snapshot before it prices anything.

    Returns (ok, reason). Every rejection reason is a stable string so the health
    stream and the gate evaluators can count them.

    The checks are deliberately paranoid because the failure mode is silent: a
    stale or half-written snapshot still produces perfectly plausible-looking
    quotes, priced off a market that no longer exists.
    """
    entries = [entry for entry in (kappa, epsilon, lambda_) if isinstance(entry, dict)]
    if len(entries) < 3:
        return False, "param_schema_unsupported"

    for entry in entries:
        version = entry.get("schema_version")
        if version is None:
            return False, "param_schema_unsupported"
        try:
            if int(version) != SUPPORTED_PARAM_SCHEMA_VERSION:
                return False, "param_schema_unsupported"
        except (TypeError, ValueError):
            return False, "param_schema_unsupported"

    # lambda must come from the survival fit, not the raw trades/sec monitor:
    # they differ by orders of magnitude and only one is the model's arrival rate.
    if not isinstance(lambda_, dict) or lambda_.get("lambda_source") != "mo_survival_fit":
        return False, "invalid_lambda_source"

    reference = as_utc(now) or utc_now()
    for entry in entries:
        if str(entry.get("status", "")).lower() != "ok":
            return False, "param_status_not_ok"
        generated_at = entry.get("generated_at")
        if not generated_at:
            return False, "missing_param_timestamp"
        parsed = parse_utc_timestamp(generated_at)
        if parsed is None:
            return False, "invalid_param_timestamp"
        age = (reference - parsed).total_seconds()
        if age < -float(config.max_future_timestamp_skew_seconds):
            return False, "param_timestamp_future"
        if age > float(config.max_param_age_seconds):
            return False, "stale_params"

    params = merged_params(kappa, epsilon, lambda_)
    for key in REQUIRED_PARAM_KEYS:
        if key not in params:
            return False, f"missing_{key}"
        if finite_float_or_none(params[key]) is None:
            return False, f"nonfinite_{key}"

    if float(params["kappa+"]) <= 0 or float(params["kappa-"]) <= 0:
        return False, "invalid_kappa"
    if float(params["lambda+"]) < 0 or float(params["lambda-"]) < 0:
        return False, "invalid_lambda"
    if float(params["epsilon+"]) < 0 or float(params["epsilon-"]) < 0:
        return False, "invalid_epsilon"

    toxicity = max(
        float(params["kappa+"]) * float(params["epsilon+"]),
        float(params["kappa-"]) * float(params["epsilon-"]),
    )
    if toxicity > float(config.max_toxicity):
        return False, "toxicity_too_high"

    for key in ("n_points_plus", "n_points_minus"):
        try:
            if int(params.get(key) or 0) < int(config.min_kappa_fit_points):
                return False, "insufficient_kappa_diagnostics"
        except (TypeError, ValueError):
            return False, "insufficient_kappa_diagnostics"

    for key in ("r2_plus", "r2_minus"):
        r2 = finite_float_or_none(params.get(key))
        if r2 is None or r2 < float(config.min_kappa_r2):
            return False, "insufficient_kappa_diagnostics"

    try:
        buy_events = int(params.get("n_buy_events") or 0)
        sell_events = int(params.get("n_sell_events") or 0)
    except (TypeError, ValueError):
        return False, "insufficient_epsilon_diagnostics"
    if buy_events < int(config.min_epsilon_events) or sell_events < int(config.min_epsilon_events):
        return False, "insufficient_epsilon_diagnostics"

    return True, "ok"


class JsonlEventLogger:
    """Append-only JSONL audit stream with size-based rotation.

    The gate evaluators (verify_live_canary.py, verify_fee_evidence.py,
    verify_dry_run_quality.py) read this file, so the envelope -- a "ts" and an
    "event" key with the payload flattened alongside -- is a contract, not a
    preference.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        max_bytes: int = 2_000_000,
    ) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()

    def _rotate_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                backup = self.path.with_suffix(self.path.suffix + ".1")
                backup.unlink(missing_ok=True)
                self.path.replace(backup)
        except OSError:
            return

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        record = {
            "ts": utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event": event,
            **(payload or {}),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"), default=str)
                    handle.write("\n")
        except OSError:
            # Never let audit logging take the quoter down.
            return


__all__ = [
    "HalfSpread",
    "JsonlEventLogger",
    "QuoteConfig",
    "QuotePair",
    "SUPPORTED_PARAM_SCHEMA_VERSION",
    "assemble_half_spread",
    "compute_quotes",
    "effective_phi",
    "finite_float_or_none",
    "inventory_to_q",
    "maker_safe",
    "merged_params",
    "parse_utc_timestamp",
    "round_amount_down",
    "round_price_for_side",
    "select_delta",
    "solve_hjb",
    "validate_param_snapshot",
]
