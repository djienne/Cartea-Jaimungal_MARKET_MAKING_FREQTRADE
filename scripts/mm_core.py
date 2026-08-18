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

This module is the single implementation. Both the replay harness and the live
strategy import it, so what a backtest simulates is literally the code that
quotes.

Conventions worth knowing before reading further:

- A half-spread of ``None`` means the side is DISABLED, not zero. The HJB returns
  an infinite depth at the inventory boundary (no bid at +q_max, no ask at
  -q_max), and that must never be clamped into a real quote.
- Depths are measured from the MID, in price units, on the same coordinate the
  estimators calibrate in.
- ``q`` is signed and spans the full [-q_max, +q_max]. Freqtrade rests one order
  per pair, so the two sides are run as two cooperating instances (a long leg
  and a short leg on separate sub-accounts); ``route_sides`` decides which leg
  owns which side, and ``allow_short=False`` reproduces a single long-only leg.
- ``phi`` and ``alpha`` are NOT kappa-invariant -- eq. 10.28 uses -phi*kappa*q^2
  -- so ``solve_hjb`` derives them from live kappa via the dimensionless targets
  ``hjb_phi_kappa_t`` / ``hjb_alpha_kappa``. Tuning the raw values per symbol is
  how the quotes ended up pinned to the floor and the cap.
- Tick and lot rounding delegate to hyperliquid_alo_executor, which does it in
  Decimal. The float floor/ceil copies elsewhere are subject to binary
  representation error at exactly the wrong moment -- a bid rounded one ULP up
  can cross the ask and turn a post-only order into a reject.
"""

from __future__ import annotations

import json
import math
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

# How the (t,q) control surface is read. See QuoteConfig.hjb_time_mode.
HJB_TIME_MODES = frozenset({"episodic", "stationary"})


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

    The strategy builds one of these from its own attributes, and the replay
    constructs one directly, so a backtest and the live path are configured by
    the same object rather than by two lists of constants kept in step by hand.
    """

    maker_fee_rate: float = 0.00015
    # The book's optimum is delta* itself; a multiplier scales 1/kappa, which IS
    # the optimum, and scales the inventory skew along with it. Kept at 1.0 and
    # retained only so an operator can reproduce a historical run.
    spread_multiplier: float = 1.0
    # Defensive widening, in bps of mid, added AFTER the model term. Additive
    # rather than multiplicative so it shifts both sides equally and leaves the
    # HJB's inventory skew intact. Preferred over raising min_half_spread_bps,
    # which is a clamp and therefore flattens skew for every q beneath it.
    extra_cushion_bps: float = 0.0
    min_half_spread_bps: float = 1.5
    max_half_spread_bps: float = 80.0

    inventory_unit_base: float = 0.01
    q_max: int = 3
    # The engine quotes both sides, so inventory runs the full signed grid. Set
    # False to reproduce the long-only strategy's [0, q_max] clamp.
    allow_short: bool = True

    # phi and alpha are NOT kappa-invariant: eq. 10.28 puts them in the
    # transition matrix as -phi*kappa*q^2 and exp(-alpha*kappa*q^2), so a value
    # tuned at one kappa is meaningless at another. Moving from ETH (kappa~2)
    # to CASHCAT (kappa~10000) took phi*kappa*T from 0.03 to 153 and drove every
    # quote onto the floor or the cap.
    #
    # So express the risk preference in the DIMENSIONLESS products the model
    # actually responds to and derive phi/alpha from live kappa. These carry
    # across symbols; the raw values below are only the fallback when the
    # targets are disabled (set to 0).
    # 10.0 is NOT a measured optimum -- do not treat it as one.
    #
    # An early sweep on a pinned 9.8h CASHCAT tape reported an inverted U with
    # the optimum at 10-20. That did NOT replicate. Re-swept 2026-08-17 over
    # 0.05 -> 400 on one pinned 18.67h tape (docs/spread_calculation.tex
    # sec:phisweep): the curve improves MONOTONICALLY past ~1 and is still
    # improving at 400, eight times hjb_phi_kappa_t_max, where the loss is the
    # smallest of the sweep (-24.83 vs -135.94 at the worst point).
    #
    # The reason it has no optimum: USDC per fill stays between -0.06 and -0.11
    # across a factor of 8000 in this parameter, while fills fall 1959 -> 338.
    # Each fill costs about the same however the model is tuned; phi only
    # changes how many you take. So the sweep is measuring negative expected
    # value per maker fill at this latency, not an inventory trade-off, and the
    # limit of "raise phi" is "stop quoting".
    #
    # 10.0 is kept as a deliberately mid-range setting that still trades enough
    # to exercise and measure the model, which is this project's stated goal.
    # Raising it would lose less money and demonstrate less.
    hjb_phi_kappa_t: float = 10.0
    hjb_alpha_kappa: float = 0.05
    # Ceiling on the SAME dimensionless product, so the volatility channel --
    # which is still in absolute price units -- cannot quietly undo the
    # normalisation. Its contribution scales as gamma*sigma2*unit*kappa*T,
    # which at CASHCAT scale is ~1.8e8 per unit of sigma2 against a 0.05
    # target, so a volatile stretch would otherwise re-pin quotes to the
    # floor and cap through the other channel.
    hjb_phi_kappa_t_max: float = 50.0
    hjb_alpha: float = 0.001
    hjb_phi: float = 0.0001
    hjb_horizon_seconds: float = 60.0
    gamma_inventory_risk: float = 0.05
    use_asymmetric_kappa: bool = True

    # The book's control is delta*(t,q) on [0,T] with terminal condition
    # h(T,q) = -alpha*q^2 (eq. 10.26). "stationary" reads only the t=0 slice,
    # which is the approximation this project ran until now: the agent never
    # approaches T, so alpha is inert and the model's flattening pressure never
    # appears. "episodic" solves the whole surface once and reads the slice at
    # the caller's actual time-to-go.
    hjb_time_mode: str = "episodic"
    # Backward Euler is first order and its error grows as time-to-go shrinks,
    # so cap dt rather than fixing the step count: n_steps scales with T.
    hjb_max_dt_seconds: float = 0.25
    hjb_n_steps_min: int = 200
    hjb_n_steps_max: int = 2000

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
        if self.hjb_time_mode not in HJB_TIME_MODES:
            raise ValueError(
                f"hjb_time_mode must be one of {sorted(HJB_TIME_MODES)}, "
                f"got {self.hjb_time_mode!r}"
            )


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

        delta_total = clamp(delta_model * spread_multiplier
                              + maker_fee*mid + extra_cushion_bps/1e4*mid,
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
    extra_cushion = float(config.extra_cushion_bps) / 10_000.0 * mid
    delta_pre_clamp = model * float(config.spread_multiplier) + fee_cushion + extra_cushion
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


def inventory_to_q_exact(signed_base: float, config: QuoteConfig) -> float:
    """Position in inventory units, clamped to the grid but NOT rounded.

    Eq. 10.2 has dq = dN(bid) - dN(ask): unit Poisson jumps, so the book's h and
    delta* are defined on integer q only. Real fills are partial, so the actual
    position lands between grid points. Rounding that away (as ``inventory_to_q``
    must, to index the grid) discards up to half a unit of live risk -- 0.49
    units reads as flat. This returns what is really held so the caller can
    price it and report the residual.
    """
    value = finite_float_or_none(signed_base)
    if value is None:
        return 0.0
    unit = max(float(config.inventory_unit_base), 1e-12)
    q_max = float(config.q_max)
    if not config.allow_short:
        value = max(0.0, value)
    q = value / unit
    return max(-q_max if config.allow_short else 0.0, min(q_max, q))


def inventory_to_q(signed_base: float, config: QuoteConfig) -> int:
    """Map a signed base position onto the HJB's integer inventory grid.

    With ``allow_short`` the result spans [-q_max, +q_max]; otherwise it is
    clamped to [0, q_max] to reproduce the long-only strategy.

    Integer q stays the currency of routing, boundary tests and the two-leg
    heartbeat. Use :func:`inventory_to_q_exact` for pricing.
    """
    return int(round(inventory_to_q_exact(signed_base, config)))


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


def hjb_n_steps(config: QuoteConfig) -> int:
    """Backward-Euler step count that keeps dt under ``hjb_max_dt_seconds``.

    Fixing the step count instead makes dt scale with T, and the solver's error
    grows as time-to-go shrinks, so a longer horizon would quietly degrade
    exactly the slices the terminal condition lives on.
    """
    horizon = float(config.hjb_horizon_seconds)
    max_dt = float(config.hjb_max_dt_seconds)
    lo = int(config.hjb_n_steps_min)
    hi = int(config.hjb_n_steps_max)
    if max_dt <= 0 or horizon <= 0:
        return lo
    return int(max(lo, min(hi, math.ceil(horizon / max_dt))))


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
    # The volatility channel is an additive risk term, independent of how the
    # base phi is chosen, so separate it once here rather than recomputing it.
    vol_delta = phi - float(config.hjb_phi) if phi_source == "sigma2_channel" else 0.0

    kappa_avg = 0.5 * (float(params["kappa+"]) + float(params["kappa-"]))
    horizon = float(config.hjb_horizon_seconds)
    alpha = float(config.hjb_alpha)
    if kappa_avg > 0:
        # Derive from the dimensionless targets so the same config expresses the
        # same risk trade-off at any kappa.
        if float(config.hjb_phi_kappa_t) > 0 and horizon > 0:
            phi = float(config.hjb_phi_kappa_t) / (kappa_avg * horizon) + vol_delta
            phi_source = "kappa_relative+sigma2" if vol_delta else "kappa_relative"
            ceiling = float(config.hjb_phi_kappa_t_max)
            if ceiling > 0 and phi * kappa_avg * horizon > ceiling:
                phi = ceiling / (kappa_avg * horizon)
                phi_source += "+capped"
        if float(config.hjb_alpha_kappa) > 0:
            alpha = float(config.hjb_alpha_kappa) / kappa_avg
    solver = compute_h_asymmetric if config.use_asymmetric_kappa else compute_h_symmetric
    episodic = config.hjb_time_mode == "episodic"
    result = solver(
        lambda_plus=float(params["lambda+"]),
        lambda_minus=float(params["lambda-"]),
        epsilon_plus=float(params["epsilon+"]),
        epsilon_minus=float(params["epsilon-"]),
        kappa_plus=float(params["kappa+"]),
        kappa_minus=float(params["kappa-"]),
        alpha=alpha,
        phi=phi,
        T_seconds=float(config.hjb_horizon_seconds),
        q_max=int(config.q_max),
        n_steps=hjb_n_steps(config),
        return_surface=episodic,
    )
    result = dict(result)
    result["hjb_time_mode"] = config.hjb_time_mode
    result["phi_effective"] = phi
    result["phi_source"] = phi_source
    result["phi_base"] = float(config.hjb_phi)
    result["alpha_effective"] = alpha
    result["kappa_avg"] = kappa_avg
    result["sigma2_per_sec"] = finite_float_or_none(sigma2_per_sec)
    return result


def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    """(lower index, upper index, weight on upper) for ``value`` on ``axis``.

    ``axis`` must be ascending. Values outside it clamp to the end, which is
    what the boundary policy already does for q and what an expired episode
    needs for t.

    A value sitting exactly ON a node returns that node twice rather than
    straddling it. That matters: straddling q=2 on a [-3..3] grid would pull in
    the disabled q=3 neighbour and switch the bid off at an inventory the model
    quotes perfectly happily.
    """
    n = len(axis)
    if n == 1 or value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        return n - 1, n - 1, 0.0
    hi = int(np.searchsorted(axis, value, side="left"))
    if float(axis[hi]) == float(value):
        return hi, hi, 0.0
    lo = hi - 1
    span = float(axis[hi] - axis[lo])
    weight = 0.0 if span <= 0 else (float(value) - float(axis[lo])) / span
    return lo, hi, weight


def _blend(low: float, high: float, weight: float) -> float:
    """Linear blend that keeps a disabled endpoint disabling.

    An infinite depth means "this side is off at this node" (the inventory
    boundary), not "very deep". Averaging it into a finite neighbour would
    invent a quotable price at a state the model refuses to quote, so any
    non-finite endpoint disables the blend.

    Consequence worth knowing: with q_max=6, a fractional q anywhere above 5.0
    disables the bid, not just q above 5.5 as rounding did. That is the
    conservative direction -- a bid at q=5.9 would permit a jump to 6.9, past
    the boundary -- and it only bites once partial fills make q fractional at
    all.
    """
    if not np.isfinite(low) or not np.isfinite(high):
        return float("inf")
    if weight <= 0.0:
        return float(low)
    if weight >= 1.0:
        return float(high)
    return float(low) + weight * (float(high) - float(low))


def select_delta(
    hjb: dict[str, Any],
    q: float,
    side: str,
    *,
    tau_remaining: float | None = None,
) -> float | None:
    """Model depth for one side at inventory ``q`` and time-to-go ``tau_remaining``.

    Returns None for a disabled side. delta_minus prices the BID (a bid fill
    increases inventory), delta_plus prices the ASK.

    Two departures from a plain grid lookup, both deliberate:

    - **Time.** The book's control is delta*(t,q) with a terminal condition at
      T. When the solve carried a surface (``hjb_time_mode="episodic"``) and the
      caller supplies its time-to-go, the slice at t = T - tau is used, so the
      depths tighten on the reducing side as the episode runs out. Without a
      surface, or without tau, this is the t=0 slice -- the stationary
      approximation -- exactly as before.
    - **Inventory.** Partial fills put q between grid points. Depths are blended
      linearly between the bracketing integers, which reproduces the book
      exactly at every integer q and interpolates over a domain the book does
      not define. Note this must interpolate DEPTHS, not h: delta reads a
      difference of h, so a piecewise-linear h yields piecewise-CONSTANT depths
      that jump at the integers -- no better than rounding.
    """
    if not hjb:
        return None

    key = "delta_minus" if side == "bid" else "delta_plus"
    surface = hjb.get(f"{key}_surface")
    t_grid = hjb.get("t_grid")
    if surface is not None and t_grid is not None and tau_remaining is not None:
        t_grid = np.asarray(t_grid, dtype=float)
        elapsed = float(hjb.get("T_seconds", t_grid[-1])) - float(tau_remaining)
        t_lo, t_hi, t_w = _bracket(t_grid, elapsed)
        row = np.asarray(surface, dtype=float)
        # The disabled-side columns are +inf at EVERY time node (the boundary is
        # structural, not time-varying), so a weight of exactly 0 or 1 has to
        # short-circuit: 0.0 * inf is NaN, which would turn a disabled side into
        # a nonsense depth instead of leaving it disabled.
        if t_lo == t_hi or t_w <= 0.0:
            depths = row[t_lo]
        elif t_w >= 1.0:
            depths = row[t_hi]
        else:
            depths = (1.0 - t_w) * row[t_lo] + t_w * row[t_hi]
    else:
        depths = np.asarray(hjb[key], dtype=float)

    q_grid = np.asarray(hjb["q_grid"], dtype=float)
    q_lo, q_hi, q_w = _bracket(q_grid, float(q))
    value = _blend(depths[q_lo], depths[q_hi], q_w)
    return finite_float_or_none(value)


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
    # State the depths were actually priced from, as opposed to the integer q
    # the routing and boundary checks agree on. q_residual is the risk the
    # integer grid cannot represent; a persistent non-zero value means partial
    # fills are leaving unmodelled inventory on the book.
    q_exact: float | None = None
    tau_remaining: float | None = None

    @property
    def q_residual(self) -> float | None:
        if self.q_exact is None:
            return None
        return float(self.q_exact) - float(self.q)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mid": self.mid,
            "q": self.q,
            "q_exact": self.q_exact,
            "q_residual": self.q_residual,
            "tau_remaining": self.tau_remaining,
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
    tau_remaining: float | None = None,
    q_exact: float | None = None,
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

    ``q`` stays integer because it is what the boundary tests and the two-leg
    routing agree on; ``q_exact`` is the unrounded position and is what the
    depths are priced from when given. ``tau_remaining`` is the episode's
    time-to-go; see :func:`select_delta`.
    """
    pair = QuotePair(mid=float(mid), q=int(q))
    q_price = float(q) if q_exact is None else float(q_exact)
    pair.q_exact = q_price
    pair.tau_remaining = (
        None if tau_remaining is None else float(tau_remaining)
    )

    bid_spread = assemble_half_spread(
        select_delta(hjb, q_price, "bid", tau_remaining=tau_remaining),
        mid,
        config,
        depth_p95=depth_p95_minus,
    )
    ask_spread = assemble_half_spread(
        select_delta(hjb, q_price, "ask", tau_remaining=tau_remaining),
        mid,
        config,
        depth_p95=depth_p95_plus,
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


def route_sides(q_long: int, q_short: int, q_max: int) -> dict[str, str | None]:
    """Decide which leg of the two-instance pair rests which side this cycle.

    Freqtrade rests one order per trade, so each instance can hold only ONE
    side. The naive split -- long always bids, short always asks -- deadlocks:
    the long leg ratchets to +q_max and the short leg to -q_max and both stop.

    Each leg has two possible actions:
        long leg   bid = add long,    ask = reduce long
        short leg  ask = add short,   bid = cover short

    So exactly two assignments keep both sides live: (long ask, short bid),
    which reduces gross inventory, and (long bid, short ask), which adds. We
    prefer reducing whenever both legs hold something, which is what keeps gross
    inventory from drifting while net stays near zero.

    At the net boundary both legs want the same side. That is correct rather
    than exceptional -- the model itself disables the other side there (delta
    = inf at +/-q_max) -- so the side is posted once, by whichever leg is
    reducing its own position.

    This is a pure function of (q_long, q_short, q_max), so both instances
    derive the same answer from the shared heartbeat with no negotiation.

    Returns {"long": "bid"|"ask"|None, "short": "bid"|"ask"|None}.
    """
    q_long = int(q_long)
    q_short = int(q_short)
    q_max = int(q_max)
    q_net = q_long + q_short

    # What the model permits at this net inventory.
    allows_bid = q_net < q_max
    allows_ask = q_net > -q_max

    if q_long > 0 and q_short < 0:
        # Both legs hold something: unwind gross.
        long_side = "ask" if allows_ask else None
        short_side = "bid" if allows_bid else None
    else:
        long_side = "bid" if (allows_bid and q_long < q_max) else None
        short_side = "ask" if (allows_ask and -q_short < q_max) else None
        # A leg that cannot add may still be able to reduce.
        if long_side is None and allows_ask and q_long > 0:
            long_side = "ask"
        if short_side is None and allows_bid and q_short < 0:
            short_side = "bid"

    if long_side is not None and long_side == short_side:
        # Both legs converged on the same side; the reducing leg posts it.
        # ask reduces for the long leg, bid reduces for the short leg.
        if long_side == "ask":
            short_side = None if q_long > 0 else short_side
            long_side = long_side if q_long > 0 else None
        else:
            long_side = None if q_short < 0 else long_side
            short_side = short_side if q_short < 0 else None

    return {"long": long_side, "short": short_side}


def resolve_net_inventory(
    q_own: int,
    peer: dict[str, Any] | None,
    own_fingerprint: str | None,
    *,
    now: datetime | None = None,
    max_peer_age_seconds: float = 30.0,
) -> tuple[int | None, str]:
    """Net inventory across both legs, or None with a reason to stop quoting.

    Fail closed rather than assuming a missing peer is flat: "flat" is a
    perfectly plausible value that would silently mis-price every quote.

    Callers must keep publishing their own heartbeat and keep quoting EXITS
    even when this returns None -- otherwise a peer outage strands both books
    with no way to unwind, and a mutual stall never recovers.
    """
    if peer is None:
        return None, "peer_inventory_missing"

    published = parse_utc_timestamp(peer.get("published_at"))
    if published is None:
        return None, "peer_inventory_timestamp_invalid"
    reference = as_utc(now) or utc_now()
    age = (reference - published).total_seconds()
    if age > float(max_peer_age_seconds):
        return None, "peer_inventory_stale"

    peer_fingerprint = peer.get("param_fingerprint")
    if own_fingerprint is not None and peer_fingerprint is not None:
        if str(peer_fingerprint) != str(own_fingerprint):
            # Content-addressed, so a mismatch means the two legs are genuinely
            # pricing off different parameter snapshots.
            return None, "peer_param_fingerprint_mismatch"

    try:
        peer_q = int(peer["q_own"])
    except (KeyError, TypeError, ValueError):
        return None, "peer_inventory_invalid"

    return int(q_own) + peer_q, "ok"


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
        backup_count: int = 1,
    ) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.max_bytes = int(max_bytes)
        self.backup_count = max(1, int(backup_count))
        self._lock = threading.Lock()

    def _rotate_if_needed(self) -> None:
        """Roll .1 -> .2 -> ... -> .backup_count, dropping the oldest.

        This kept exactly ONE backup, which silently bounded how much history
        could ever exist. At the measured 3.7 MB/hour a 2 MB cap retained about
        66 minutes across both files, so a week-long dry run would have kept
        0.7% of its own fill evidence -- and accumulating that evidence is the
        entire reason the run exists. Every consumer of this stream
        (calibrate_replay_from_logs, verify_dry_run_quality, verify_live_canary,
        verify_fee_evidence) reads whatever survives, so the cap was quietly
        setting the sample size of the gates too.
        """
        if self.max_bytes <= 0:
            return
        try:
            if not self.path.exists() or self.path.stat().st_size <= self.max_bytes:
                return
            oldest = self.path.with_suffix(f"{self.path.suffix}.{self.backup_count}")
            oldest.unlink(missing_ok=True)
            for index in range(self.backup_count - 1, 0, -1):
                source = self.path.with_suffix(f"{self.path.suffix}.{index}")
                if source.exists():
                    source.replace(self.path.with_suffix(f"{self.path.suffix}.{index + 1}"))
            self.path.replace(self.path.with_suffix(f"{self.path.suffix}.1"))
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
    "hjb_n_steps",
    "inventory_to_q",
    "inventory_to_q_exact",
    "maker_safe",
    "merged_params",
    "parse_utc_timestamp",
    "round_amount_down",
    "round_price_for_side",
    "select_delta",
    "solve_hjb",
    "validate_param_snapshot",
]
