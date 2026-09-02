#!/usr/bin/env python3
"""Decide whether an instrument can pay a passive market maker at all.

This gate exists because the Cartea-Jaimungal machinery answers "how wide should
I quote", never "can quoting this instrument ever be profitable". ETH perp on
Hyperliquid is the worked counter-example: a one-tick-wide book whose touch sits
0.27 bps from mid against a 1.5 bps per-side maker fee, so a round trip at the
touch loses ~2.5 bps. The model produced quotes anyway -- floor-clamped, 11x past
the calibrated depth -- and the replay recorded no maker fills in any variant.

The verdict comes from an EMPIRICAL PROFIT CURVE, not from the fitted model, so a
degenerate kappa cannot hide the answer. For each side and each candidate quote
depth delta, using only measured market orders:

    edge(delta)   = delta - maker_fee*mid - E[markout | depth >= delta]
    volume(delta) = sum of sizes of market orders reaching delta, per hour
    pnl(delta)    = volume(delta) * edge(delta)                    [USDC/hour]

and the instrument is viable when max over delta of the two-sided pnl is
positive on a non-trivial amount of flow. Four things make this the honest test:

- It searches ALL depths, not just the touch. Quoting deeper to cover the fee is
  allowed, and the search finds the best such depth if one exists.
- P(depth >= delta) is the empirical survival function -- the fill probability
  straight from the sample, with no lambda*exp(-kappa*delta) assumption.
- The adverse-selection term is CONDITIONAL on being reached. An unconditional
  epsilon (what get_epsilon.py publishes) is dominated by touch-takers that move
  nothing and reads ~0; but a market order that sweeps deep enough to fill a
  resting quote is exactly the informed one. Conditioning is what stops the curve
  from looking profitable purely because the tail was averaged away.
- Fill opportunity is measured in TRADED VOLUME, not arrivals. It is still an
  upper bound -- 100% participation, no queue ahead of the quote -- but one
  tethered to flow that actually exists, so compare the reported reachable
  notional against the instrument's real volume before believing the P&L.

A separate KAPPA IDENTIFIABILITY block reports whether the fill model the strategy
actually uses is estimable here: the survival fit needs enough distinct depth
points, an acceptable R^2, an intercept near 1.0 (the model's law requires
S(0)=1), and no single depth atom dominating. A one-tick book fails the last two
-- a touch-taking market order mechanically has depth = half the spread, so the
"distribution" is one spike at half a tick. This is reported as a warning, not a
veto: it says the HJB is guessing, not that the market is unprofitable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from estimator_common import (
    MIN_KAPPA_FIT_POINTS,
    MIN_KAPPA_R2,
    aggregate_market_orders,
    attach_pre_mid,
    fit_kappa_survival,
    load_market_window,
)
from param_utils import atomic_write_json, finite_or_none, utc_now_iso


# Resolved against the repo root, not the CWD: this script must be importable
# from scripts/ (estimator_common lives there) while writing evidence to docs/
# alongside every other verify_*.py report.
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "market_viability_report.json"
DEFAULT_MAKER_FEE = 0.00015
DEFAULT_MINUTES = 30
# Permanent-impact horizon, matching get_epsilon.py's primary horizon so the
# conditional markout here is comparable to the published epsilon.
DEFAULT_MARKOUT_HORIZON_MS = 5000

# The optimum must clear break-even by this much before it is worth quoting:
# below it the edge is inside the measurement error of the spread itself.
DEFAULT_MIN_NET_EDGE_BPS = 0.5
# Fewer fills than this at the optimum means the quote is decorative.
DEFAULT_MIN_FILLS_PER_HOUR = 1.0
# A verdict may not be issued on a window shorter than this. Spreads and flow on
# thin instruments are wildly regime-dependent: CASHCAT was measured on
# 2026-08-17 running 19.3x its own daily average volume with a 200-tick spread,
# and over that 15-minute burst the profit curve reported +$3,279/h. The other
# candidates were simultaneously at 0.18-0.68x their daily rate. Any window short
# enough to sit inside one regime will confidently describe that regime and
# nothing else.
DEFAULT_MIN_WINDOW_HOURS = 6.0
# A conditional markout estimated on fewer market orders than this is noise, so
# the depth is not offered as a candidate optimum.
MIN_SAMPLES_PER_DEPTH = 20
# ...and the winning depth needs materially more than that before its edge is
# allowed to declare an instrument viable. Without this floor the optimum lands
# on the deepest, smallest-sample point every time: the conditional markout falls
# monotonically as the sample shrinks, which reads as edge but is just the tail
# running out of observations.
MIN_SAMPLES_AT_OPTIMUM = 50

# Survival-fit intercept must sit near 1.0 or the exponential fill law is
# misspecified: S(0) = 1 by construction for a real survival function.
MIN_SURVIVAL_INTERCEPT = 0.5
MAX_SURVIVAL_INTERCEPT = 2.0
# Above this share of depths sitting on one value the sample is an atom, not a
# distribution, and the fitted kappa is measuring tick granularity.
MAX_DEPTH_ATOM_SHARE = 0.60

# Cap on curve points written to the report. The curve has one point per distinct
# observed depth, so it grows with the dataset; the verdict only needs the
# optimum and the shape survives subsampling.
CURVE_POINTS_IN_REPORT = 25


def _percentile_or_none(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return finite_or_none(float(np.percentile(values, q)))


def conditional_markout(values: np.ndarray) -> float | None:
    """Mean adverse move, deliberately UNTRIMMED.

    get_epsilon.py trims 10% of each tail for a stable central estimate, which is
    right for a location parameter but wrong-signed for a risk one: the informed
    trades that make market making lose money ARE the upper tail, so trimming
    them flatters the maker exactly where the answer matters. On a short window
    the trimmed version falls monotonically as the sample shrinks and manufactures
    an edge out of nothing. A viability gate must not produce false positives, so
    it takes the full mean and leans on MIN_SAMPLES_AT_OPTIMUM for stability.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return finite_or_none(float(np.mean(values)))


def depth_atom_share(depths: np.ndarray) -> float | None:
    """Largest share of the sample sitting on a single depth value.

    A one-tick book puts nearly every touch-taking market order on exactly half a
    tick, so this approaches 1.0 and the survival fit degenerates.
    """
    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths)]
    if depths.size == 0:
        return None
    _, counts = np.unique(depths, return_counts=True)
    return float(counts.max()) / float(depths.size)


def infer_tick_size(prices: np.ndarray) -> float | None:
    """Smallest positive gap between distinct quoted prices."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 2:
        return None
    unique = np.unique(prices)
    if unique.size < 2:
        return None
    diffs = np.diff(unique)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    # Float noise makes the raw minimum unreliable; round to the price scale.
    return finite_or_none(float(np.round(diffs.min(), 10)))


def mo_depth_impact_frame(
    mos_with_pre_mid: pd.DataFrame,
    mids: pd.DataFrame,
    horizon_ms: int,
    window_end_ms: float,
) -> pd.DataFrame:
    """Per market order: (side, depth, markout) on one shared clock.

    depth   = how far past the prevailing mid the order walked (the depth a
              resting quote would have needed to be filled by it).
    markout = signed mid move over ``horizon_ms`` after the order, in the
              direction that hurts the maker who filled it.

    Market orders whose horizon runs past the end of the data window are dropped
    rather than truncated, matching get_epsilon.py, so the tail is not biased
    toward zero impact.
    """
    empty = pd.DataFrame(columns=["side", "depth", "markout"])
    if mos_with_pre_mid.empty or mids.empty:
        return empty

    frame = mos_with_pre_mid.dropna(subset=["pre_mid"]).copy()
    frame = frame[frame["ts_ms"] + float(horizon_ms) <= float(window_end_ms)]
    if frame.empty:
        return empty

    frame = frame.assign(target_ms=frame["ts_ms"] + float(horizon_ms)).sort_values("target_ms")
    post = pd.merge_asof(
        frame,
        mids[["ts_ms", "mid"]]
        .sort_values("ts_ms")
        .rename(columns={"ts_ms": "mid_ts_ms", "mid": "post_mid"}),
        left_on="target_ms",
        right_on="mid_ts_ms",
        direction="backward",
        allow_exact_matches=True,
    ).dropna(subset=["post_mid"])
    if post.empty:
        return empty

    is_buy = post["side"] == "buy"
    # Buy MOs lift the ask: they fill a resting ASK, and hurt it when the mid
    # rises. Sell MOs are the mirror. Negative depths are stale-feed noise and
    # truncate to 0, the same convention mo_depths() uses.
    depth = np.where(
        is_buy,
        post["price_extreme"] - post["pre_mid"],
        post["pre_mid"] - post["price_extreme"],
    )
    markout = np.where(
        is_buy,
        post["post_mid"] - post["pre_mid"],
        post["pre_mid"] - post["post_mid"],
    )
    return pd.DataFrame(
        {
            "side": post["side"].to_numpy(),
            "depth": np.maximum(depth.astype(float), 0.0),
            "markout": markout.astype(float),
            # Base size of the aggregated market order. Fill opportunity scales
            # with traded volume, not with the number of events: 600 one-lot
            # prints an hour is not the same opportunity as 600 large sweeps.
            "size": post["size"].astype(float).to_numpy(),
        }
    )


def profit_curve(
    depths: np.ndarray,
    markouts: np.ndarray,
    *,
    covered_seconds: float,
    fee_cost: float,
    mid: float,
    sizes: np.ndarray | None = None,
    min_samples: int = MIN_SAMPLES_PER_DEPTH,
) -> list[dict[str, Any]]:
    """Expected P&L rate for a resting quote at each candidate depth.

    Only depths above the fee cost are considered -- below it the edge is
    negative by construction whatever the fill rate.

    Everything here is an UPPER BOUND on what a real maker earns: it credits the
    quote with the entire size of every market order that reaches its depth,
    i.e. 100% participation with no queue ahead of it. Counting events alone was
    worse still -- it implied filling a fixed size on every arrival, which for an
    illiquid coin works out to a multiple of the instrument's whole daily volume.
    Sizing by traded volume keeps the bound tethered to reality. The replay
    harness, which models queue position and latency, is what turns this into an
    achievable number; this function only decides what is worth replaying.
    """
    depths = np.asarray(depths, dtype=float)
    markouts = np.asarray(markouts, dtype=float)
    if sizes is None:
        sizes = np.ones_like(depths)
    sizes = np.asarray(sizes, dtype=float)
    finite = np.isfinite(depths) & np.isfinite(markouts) & np.isfinite(sizes)
    depths, markouts, sizes = depths[finite], markouts[finite], sizes[finite]
    if depths.size == 0:
        return []

    covered_seconds = max(float(covered_seconds), 1e-6)
    candidates = np.unique(depths)
    candidates = candidates[candidates > float(fee_cost)]
    # Beyond the 99th percentile there is no sample left to estimate a
    # conditional markout from.
    if candidates.size:
        candidates = candidates[candidates <= float(np.percentile(depths, 99))]
    if candidates.size == 0:
        return []

    curve: list[dict[str, Any]] = []
    for delta in candidates:
        reached = depths >= delta
        n = int(np.count_nonzero(reached))
        if n < int(min_samples):
            continue
        markout = conditional_markout(markouts[reached])
        if markout is None:
            continue
        edge = float(delta) - float(fee_cost) - float(markout)
        fill_rate = float(n) / covered_seconds
        reachable_base = float(np.sum(sizes[reached]))
        base_per_hour = reachable_base / covered_seconds * 3600.0
        curve.append(
            {
                "depth": float(delta),
                "depth_bps": float(delta) / mid * 10_000.0 if mid > 0 else None,
                "survival": float(n) / float(depths.size),
                "n_market_orders": n,
                "conditional_markout": float(markout),
                "edge_per_fill": edge,
                "edge_bps": edge / mid * 10_000.0 if mid > 0 else None,
                "fills_per_hour": fill_rate * 3600.0,
                "reachable_base_per_hour": base_per_hour,
                "reachable_notional_per_hour": base_per_hour * mid,
                # USDC/hour at 100% participation -- see the docstring.
                "pnl_per_hour_upper_bound": base_per_hour * edge,
            }
        )
    return curve


def best_depth(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not curve:
        return None
    return max(curve, key=lambda point: point["pnl_per_hour_upper_bound"])


def subsample_curve(
    curve: list[dict[str, Any]],
    limit: int = CURVE_POINTS_IN_REPORT,
) -> list[dict[str, Any]]:
    """Thin a curve for the persisted report, preserving shape and the optimum.

    One candidate depth per distinct observed depth means the curve grows with
    the dataset: three days of an active instrument produced hundreds of points
    per side and a 472 KB tracked evidence file. The verdict only needs the
    optimum; the rest is there to show the shape, which survives subsampling.
    """
    if len(curve) <= limit:
        return curve
    step = (len(curve) - 1) / float(limit - 1)
    keep_indices = {int(round(i * step)) for i in range(limit)}
    keep_indices.add(0)
    keep_indices.add(len(curve) - 1)
    best = best_depth(curve)
    if best is not None:
        keep_indices.add(curve.index(best))
    return [curve[i] for i in sorted(keep_indices)]


def build_market_viability_report(
    *,
    crypto: str,
    mids: pd.DataFrame,
    trades: pd.DataFrame,
    covered_seconds: float,
    window_end_ms: float,
    maker_fee_rate: float = DEFAULT_MAKER_FEE,
    markout_horizon_ms: int = DEFAULT_MARKOUT_HORIZON_MS,
    min_net_edge_bps: float = DEFAULT_MIN_NET_EDGE_BPS,
    min_fills_per_hour: float = DEFAULT_MIN_FILLS_PER_HOUR,
    min_window_hours: float = DEFAULT_MIN_WINDOW_HOURS,
    window_start: str | None = None,
    window_end: str | None = None,
    mid_source: str | None = None,
    ts_source: str | None = None,
) -> dict[str, Any]:
    """Pure verdict builder -- takes already-loaded frames so it is unit-testable."""
    reasons: list[str] = []
    warnings: list[str] = []

    mid_values = mids["mid"].to_numpy(dtype=float) if not mids.empty else np.array([])
    mid_values = mid_values[np.isfinite(mid_values) & (mid_values > 0)]
    if mid_values.size == 0:
        return {
            "generated_at": utc_now_iso(),
            "ok": False,
            "viable": False,
            "reasons": ["no_mid_data"],
            "warnings": [],
            "symbol": crypto,
        }

    mid_median = float(np.median(mid_values))
    spread = mids["ask"].to_numpy(dtype=float) - mids["bid"].to_numpy(dtype=float)
    spread_bps = np.where(mid_values > 0, spread / mid_values, np.nan) * 10_000.0
    spread_bps = spread_bps[np.isfinite(spread_bps) & (spread_bps > 0)]

    tick_size = infer_tick_size(
        np.concatenate([mids["bid"].to_numpy(dtype=float), mids["ask"].to_numpy(dtype=float)])
    )
    spread_median_bps = _percentile_or_none(spread_bps, 50) or 0.0
    spread_ticks_median = None
    if tick_size and tick_size > 0:
        positive_spread = spread[np.isfinite(spread) & (spread > 0)]
        spread_ticks_median = _percentile_or_none(positive_spread / tick_size, 50)

    maker_fee_bps = float(maker_fee_rate) * 10_000.0
    fee_cost = float(maker_fee_rate) * mid_median  # one maker fee per side, price units
    covered_seconds = max(float(covered_seconds), 1e-6)
    window_hours = covered_seconds / 3600.0

    # Traded notional per hour over this window. Compare it against the
    # instrument's usual daily volume / 24 before believing anything below: a
    # ratio far from 1 means the window caught a burst or a lull, not normality.
    observed_notional_per_hour = None
    if not trades.empty and {"price", "size"}.issubset(trades.columns):
        traded = float((trades["price"] * trades["size"]).sum())
        observed_notional_per_hour = traded / covered_seconds * 3600.0

    # --- headline diagnostic: resting at the touch -----------------------
    # Captures the whole quoted spread, pays two maker fees. Intuitive, and an
    # UPPER BOUND (assumes both sides fill, ignores queue priority).
    round_trip_fee_bps = 2.0 * maker_fee_bps
    net_edge_at_touch_bps = spread_median_bps - round_trip_fee_bps

    # --- verdict: empirical profit curve over all depths -----------------
    mos = attach_pre_mid(aggregate_market_orders(trades), mids)
    events = mo_depth_impact_frame(mos, mids, markout_horizon_ms, window_end_ms)

    curves: dict[str, list[dict[str, Any]]] = {}
    optima: dict[str, dict[str, Any] | None] = {}
    for label, mo_side in (("plus", "buy"), ("minus", "sell")):
        side_events = events[events["side"] == mo_side] if not events.empty else events
        curve = profit_curve(
            side_events["depth"].to_numpy(dtype=float) if not side_events.empty else np.array([]),
            side_events["markout"].to_numpy(dtype=float) if not side_events.empty else np.array([]),
            sizes=side_events["size"].to_numpy(dtype=float) if not side_events.empty else np.array([]),
            covered_seconds=covered_seconds,
            fee_cost=fee_cost,
            mid=mid_median,
        )
        curves[label] = curve
        optima[label] = best_depth(curve)

    # A two-sided maker eats BOTH sides, so the total sums each side's best
    # achievable P&L including a negative one. Summing only the profitable side
    # would describe a directional trader, not a market maker, and on a noisy
    # window it reliably reports a dead instrument as viable.
    present = [point for point in optima.values() if point]
    total_pnl_per_hour = sum(point["pnl_per_hour_upper_bound"] for point in present)
    total_fills_per_hour = sum(point["fills_per_hour"] for point in present)
    total_base_per_hour = sum(point["reachable_base_per_hour"] for point in present)
    total_notional_per_hour = sum(point["reachable_notional_per_hour"] for point in present)
    blended_edge_bps = None
    if total_base_per_hour > 0 and mid_median > 0:
        blended_edge_bps = (total_pnl_per_hour / total_base_per_hour) / mid_median * 10_000.0

    if events.empty:
        reasons.append("no_market_orders_measured")
    elif len(present) < 2:
        missing = [label for label, point in optima.items() if not point]
        reasons.append(f"no_profitable_depth_on_side:{','.join(missing)}")
    else:
        if total_pnl_per_hour <= 0:
            reasons.append(
                f"two_sided_pnl_not_positive:{total_pnl_per_hour:+.4f}_usdc_per_hour_upper_bound "
                f"(plus {optima['plus']['pnl_per_hour_upper_bound']:+.4f}, "
                f"minus {optima['minus']['pnl_per_hour_upper_bound']:+.4f})"
            )
        if blended_edge_bps is not None and blended_edge_bps < float(min_net_edge_bps):
            reasons.append(
                f"blended_edge_below_threshold:{blended_edge_bps:.3f}bps"
                f"<min_{float(min_net_edge_bps):g}bps"
            )
        if total_fills_per_hour < float(min_fills_per_hour):
            reasons.append(
                f"optimum_unreachable:{total_fills_per_hour:.4f}_fills_per_hour"
                f"<min_{float(min_fills_per_hour):g}"
            )
        if window_hours < float(min_window_hours):
            reasons.append(
                f"window_too_short_for_a_verdict:{window_hours:.2f}h"
                f"<min_{float(min_window_hours):g}h (a short window describes one "
                f"regime, not the instrument)"
            )
        thin = {
            label: point["n_market_orders"]
            for label, point in optima.items()
            if point["n_market_orders"] < MIN_SAMPLES_AT_OPTIMUM
        }
        if thin:
            reasons.append(
                "optimum_sample_too_thin:"
                + ",".join(f"{label}={n}<{MIN_SAMPLES_AT_OPTIMUM}" for label, n in thin.items())
                + " (collect a longer window before trusting this verdict)"
            )

    # --- kappa identifiability (warning only) ----------------------------
    kappa_diagnostics: dict[str, Any] = {}
    kappa_identifiable = True
    for label, mo_side in (("plus", "buy"), ("minus", "sell")):
        side_events = events[events["side"] == mo_side] if not events.empty else events
        depths = (
            side_events["depth"].to_numpy(dtype=float) if not side_events.empty else np.array([])
        )
        fit = fit_kappa_survival(depths)
        atom_share = depth_atom_share(depths)
        intercept = finite_or_none(fit.get("survival_intercept"))
        r_squared = finite_or_none(fit.get("r_squared"))
        n_points = int(fit.get("n_points", 0) or 0)
        side_ok = True
        if n_points < MIN_KAPPA_FIT_POINTS:
            side_ok = False
            warnings.append(f"kappa_{label}_too_few_fit_points:{n_points}<{MIN_KAPPA_FIT_POINTS}")
        if r_squared is None or r_squared < MIN_KAPPA_R2:
            side_ok = False
            warnings.append(f"kappa_{label}_poor_fit_r2:{r_squared}")
        # Since schema v5 the intercept is applied to lambda rather than
        # assumed to be one, so an off-unity value is a fit-quality warning,
        # not a failure of the model's premise.
        if intercept is None or not (MIN_SURVIVAL_INTERCEPT <= intercept <= MAX_SURVIVAL_INTERCEPT):
            warnings.append(
                f"kappa_{label}_survival_intercept_off_unity:{intercept}"
                f" (expected {MIN_SURVIVAL_INTERCEPT}-{MAX_SURVIVAL_INTERCEPT})"
            )
        if atom_share is not None and atom_share > MAX_DEPTH_ATOM_SHARE:
            side_ok = False
            warnings.append(
                f"kappa_{label}_depth_is_single_atom:{atom_share:.2f}>{MAX_DEPTH_ATOM_SHARE}"
            )
        kappa_identifiable = kappa_identifiable and side_ok
        kappa_diagnostics[label] = {
            "kappa": finite_or_none(fit.get("kappa")),
            "r_squared": r_squared,
            "n_points": n_points,
            "survival_intercept": intercept,
            "depth_atom_share": atom_share,
            "n_unique_depths": int(np.unique(depths).size) if depths.size else 0,
            "identifiable": side_ok,
        }
    if not kappa_identifiable:
        warnings.append("kappa_not_identifiable_hjb_output_is_unreliable")

    viable = not reasons
    return {
        "generated_at": utc_now_iso(),
        "ok": viable,
        "viable": viable,
        "reasons": reasons,
        "warnings": warnings,
        "symbol": crypto,
        "window": {
            "start": window_start,
            "end": window_end,
            "covered_seconds": covered_seconds,
            "hours": window_hours,
            "min_window_hours": float(min_window_hours),
            "observed_notional_per_hour": finite_or_none(observed_notional_per_hour)
            if observed_notional_per_hour is not None
            else None,
            "mid_source": mid_source,
            "ts_source": ts_source,
            "n_mid_updates": int(len(mids)),
            "n_trade_prints": int(len(trades)),
            "n_market_orders": int(len(events)),
            "markout_horizon_ms": int(markout_horizon_ms),
        },
        "book": {
            "mid_median": mid_median,
            "tick_size": tick_size,
            "quoted_spread_median_bps": finite_or_none(spread_median_bps),
            "quoted_spread_p95_bps": _percentile_or_none(spread_bps, 95),
            "quoted_spread_median_ticks": spread_ticks_median,
            "touch_half_spread_median_bps": finite_or_none(spread_median_bps / 2.0),
        },
        "at_touch": {
            "maker_fee_rate": float(maker_fee_rate),
            "maker_fee_bps": maker_fee_bps,
            "round_trip_fee_bps": round_trip_fee_bps,
            "max_capture_bps": finite_or_none(spread_median_bps),
            "net_edge_bps": finite_or_none(net_edge_at_touch_bps),
            "note": (
                "Upper bound on resting at both touches: assumes both sides fill "
                "and ignores queue priority. The verdict comes from the profit "
                "curve, which also considers quoting deeper than the touch."
            ),
        },
        "profit_curve": {
            "optimum_plus": optima["plus"],
            "optimum_minus": optima["minus"],
            "total_pnl_per_hour_upper_bound": finite_or_none(total_pnl_per_hour),
            "total_fills_per_hour": finite_or_none(total_fills_per_hour),
            "total_reachable_base_per_hour": finite_or_none(total_base_per_hour),
            "total_reachable_notional_per_hour": finite_or_none(total_notional_per_hour),
            "blended_edge_bps": finite_or_none(blended_edge_bps)
            if blended_edge_bps is not None
            else None,
            "min_net_edge_bps": float(min_net_edge_bps),
            "min_samples_at_optimum": MIN_SAMPLES_AT_OPTIMUM,
            "note": (
                "total sums BOTH sides' best achievable P&L, including a negative "
                "side: a two-sided maker cannot decline to be filled on the side "
                "that loses. P&L is an UPPER BOUND at 100% participation with no "
                "queue ahead of the quote -- compare total_reachable_notional_per_hour "
                "against the instrument's real volume before believing it."
            ),
            "min_fills_per_hour": float(min_fills_per_hour),
            "n_candidate_depths_plus": len(curves["plus"]),
            "n_candidate_depths_minus": len(curves["minus"]),
            "curve_plus": subsample_curve(curves["plus"]),
            "curve_minus": subsample_curve(curves["minus"]),
            "curve_subsampled_to": CURVE_POINTS_IN_REPORT,
        },
        "kappa_identifiability": {"identifiable": kappa_identifiable, **kappa_diagnostics},
    }


def evaluate_symbol(
    crypto: str,
    *,
    minutes: int = DEFAULT_MINUTES,
    maker_fee_rate: float = DEFAULT_MAKER_FEE,
    markout_horizon_ms: int = DEFAULT_MARKOUT_HORIZON_MS,
    min_net_edge_bps: float = DEFAULT_MIN_NET_EDGE_BPS,
    min_fills_per_hour: float = DEFAULT_MIN_FILLS_PER_HOUR,
    min_window_hours: float = DEFAULT_MIN_WINDOW_HOURS,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        window = load_market_window(crypto, minutes, data_dir=data_dir)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "generated_at": utc_now_iso(),
            "ok": False,
            "viable": False,
            "reasons": [f"no_data:{exc}"],
            "warnings": [],
            "symbol": crypto,
        }

    # Arrival rates divide by the span both streams actually cover, the same
    # convention get_kappa.py uses, or lambda is understated on a cold start.
    if window.mids.empty:
        covered_seconds = 0.0
    else:
        data_start_ms = max(window.window_start_ms or 0.0, float(window.mids["ts_ms"].min()))
        covered_seconds = max(((window.window_end_ms or 0.0) - data_start_ms) / 1000.0, 1e-6)

    return build_market_viability_report(
        crypto=crypto,
        mids=window.mids,
        trades=window.trades,
        covered_seconds=covered_seconds,
        window_end_ms=window.window_end_ms or 0.0,
        maker_fee_rate=maker_fee_rate,
        markout_horizon_ms=markout_horizon_ms,
        min_net_edge_bps=min_net_edge_bps,
        min_fills_per_hour=min_fills_per_hour,
        min_window_hours=min_window_hours,
        window_start=window.window_start_iso(),
        window_end=window.window_end_iso(),
        mid_source=window.mid_source,
        ts_source=window.ts_source,
    )


def available_symbols(data_dir: str | Path | None = None) -> list[str]:
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "HL_data"
    if not base.is_dir():
        return []
    symbols = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        has_mid = (entry / "prices").is_dir() or (entry / "orderbooks").is_dir()
        if has_mid and (entry / "trades").is_dir():
            symbols.append(entry.name)
    return symbols


def print_summary(report: dict[str, Any]) -> None:
    symbol = report.get("symbol", "?")
    if not report.get("book"):
        print(f"  {symbol:<10} NO DATA   {'; '.join(report.get('reasons') or [])}")
        return
    book = report["book"]
    touch = report["at_touch"]
    curve = report["profit_curve"]
    window = report.get("window", {})
    ticks = book["quoted_spread_median_ticks"]
    print(
        f"  {symbol:<10} {'VIABLE' if report.get('viable') else 'NOT VIABLE':<10} "
        f"spread={book['quoted_spread_median_bps']:.2f}bps"
        f"{f' ({ticks:.1f} ticks)' if ticks else ''}  "
        f"fee_round_trip={touch['round_trip_fee_bps']:.2f}bps  "
        f"touch_net={touch['net_edge_bps']:+.2f}bps"
    )
    blended = curve.get("blended_edge_bps")
    for label in ("plus", "minus"):
        point = curve.get(f"optimum_{label}")
        if not point:
            print(f"       {label:<5}: no depth clears the fee")
            continue
        print(
            f"       {label:<5}: depth={point['depth_bps']:.2f}bps "
            f"edge={point['edge_bps']:+.2f}bps  markout={point['conditional_markout']:+.6g}  "
            f"n={point['n_market_orders']}  fills/h={point['fills_per_hour']:.1f}  "
            f"reach=${point['reachable_notional_per_hour']:,.0f}/h  "
            f"pnl<=${point['pnl_per_hour_upper_bound']:+,.2f}/h"
        )
    print(
        f"       two-sided: pnl<=${curve['total_pnl_per_hour_upper_bound']:+,.2f}/h "
        f"on ${curve['total_reachable_notional_per_hour']:,.0f}/h of reachable flow"
        + (f"  blended_edge={blended:+.2f}bps" if blended is not None else "")
    )
    flow = window.get("observed_notional_per_hour")
    print(
        f"       window={window.get('hours', 0):.2f}h"
        + (f"  observed_flow=${flow:,.0f}/h" if flow else "")
        + f"  kappa_identifiable={report['kappa_identifiability']['identifiable']}"
    )
    for reason in report.get("reasons") or []:
        print(f"       - {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decide whether an instrument can pay a passive market maker."
    )
    parser.add_argument(
        "--crypto",
        "-c",
        default="CASHCAT",
        help="Symbol to evaluate, or ALL to scan every symbol with collector data",
    )
    parser.add_argument("--minutes", "-m", type=int, default=DEFAULT_MINUTES)
    parser.add_argument("--maker-fee", type=float, default=DEFAULT_MAKER_FEE)
    parser.add_argument("--markout-horizon-ms", type=int, default=DEFAULT_MARKOUT_HORIZON_MS)
    parser.add_argument("--min-net-edge-bps", type=float, default=DEFAULT_MIN_NET_EDGE_BPS)
    parser.add_argument("--min-fills-per-hour", type=float, default=DEFAULT_MIN_FILLS_PER_HOUR)
    parser.add_argument(
        "--min-window-hours",
        type=float,
        default=DEFAULT_MIN_WINDOW_HOURS,
        help="Refuse a verdict on a window shorter than this (0 disables)",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.crypto).strip().upper() == "ALL":
        symbols = available_symbols(args.data_dir)
        if not symbols:
            print("No symbols with collector data found.")
            return 1
    else:
        symbols = [str(args.crypto).strip().upper()]

    reports = [
        evaluate_symbol(
            symbol,
            minutes=args.minutes,
            maker_fee_rate=float(args.maker_fee),
            markout_horizon_ms=int(args.markout_horizon_ms),
            min_net_edge_bps=float(args.min_net_edge_bps),
            min_fills_per_hour=float(args.min_fills_per_hour),
            min_window_hours=float(args.min_window_hours),
            data_dir=args.data_dir,
        )
        for symbol in symbols
    ]

    print("\n" + "=" * 72)
    print("MARKET VIABILITY FOR PASSIVE MARKET MAKING")
    print("=" * 72)
    for report in reports:
        print_summary(report)

    viable_symbols = [r["symbol"] for r in reports if r.get("viable")]
    payload = {
        "generated_at": utc_now_iso(),
        "ok": bool(viable_symbols),
        "viable_symbols": viable_symbols,
        "criteria": {
            "maker_fee_rate": float(args.maker_fee),
            "markout_horizon_ms": int(args.markout_horizon_ms),
            "min_net_edge_bps": float(args.min_net_edge_bps),
            "min_fills_per_hour": float(args.min_fills_per_hour),
            "min_window_hours": float(args.min_window_hours),
            "window_minutes": int(args.minutes),
        },
        "symbols": {r["symbol"]: r for r in reports},
    }
    atomic_write_json(args.output, payload)
    print(f"\n[save] viability report -> {args.output}")
    if not viable_symbols:
        print("No symbol clears the bar: passive making cannot pay here at this fee tier.")
    return 0 if viable_symbols else 1


if __name__ == "__main__":
    raise SystemExit(main())
