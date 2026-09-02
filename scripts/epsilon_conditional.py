#!/usr/bin/env python3
"""Fill-CONDITIONAL adverse selection: E[mid jump | the sweep reached depth d].

WHY THIS EXISTS. `get_epsilon.py` estimates the quantity Cartea-Jaimungal eq.
10.22 asks for: the mean jump `eps_bar` of an *average* market order. That is
correct for the book, because the book assumes the jump size is independent of
whether the order reached our quote -- in the DPE (10.23) the fill probability
`e^{-kappa*delta}` multiplies the expectation as a constant.

The data violates that assumption badly. A maker resting behind the touch is not
filled by an average market order; it is filled selectively, by the tail of
orders large enough to sweep to its price. Conditioning on "we were filled"
therefore selects the toxic subset, and the unconditional mean understates what
our fills actually suffer by roughly 7x at the depths this bot quotes.

WHAT THIS COMPUTES. For a depth grid, `E[jump | reach >= d]` per side, plus a
linear fit `eps(d) = a + b*d` with a bootstrap CI on the slope. `b` is the number
that matters: the per-fill edge of a maker quoting at depth `d` is

    (1 - b) * d - a - fee

so `b -> 1` means the mid ends up where the sweep reached, the spread capture is
handed straight back, and NO depth is profitable. `b` is a makeability statistic,
not a constant of the instrument: it is measurably time-varying.

THE REACH VARIABLE IS THE ONE KAPPA IS ALREADY FITTED ON. `mo_depths` computes
`price_extreme - pre_mid` (buys, truncated at 0) and `fit_kappa_survival` fits
`P(depth >= d) = A*exp(-kappa*d)` on exactly that. Reusing it is what makes
`eps(d)` and `e^{-kappa*d}` two views of one quantity rather than two models
bolted together.

Research/diagnostic only -- this writes no parameters and changes no behaviour.
The live estimator path is `get_epsilon.py` / Rust `estimate_epsilon`.

    python scripts/epsilon_conditional.py --minutes 0 \
        --window-start 2026-08-28T10:56:50Z --window-end 2026-09-02T09:39:00Z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from estimator_common import (  # noqa: E402
    aggregate_market_orders,
    attach_pre_mid,
    load_market_window,
)

# Horizons to report. 200 ms is what the shipped estimator uses; 1 s and 5 s are
# the diagnostic pair `get_epsilon.py` already computes. Nothing longer: at 300 s
# the markouts on this instrument are trend-dominated (measured +276 bps on buys
# against -47 on sells at reach >= 26 bps), which measures drift, not adverse
# selection, and would make `b` a statement about the price path.
DEFAULT_HORIZONS_MS = (200, 1000, 5000)

# Depth grid in bps. Fitted in bps rather than price units so the slope is
# comparable across a tape whose mid moved 0.105 -> 0.251; `b` is dimensionless
# either way, and `a` is reported in both.
DEFAULT_GRID_BPS = tuple(float(d) for d in range(5, 61, 5))

# Enough observations in the conditioning set for a mean to mean anything.
MIN_CONDITIONAL_EVENTS = 200


def reach_and_jump(
    mos: pd.DataFrame, mids: pd.DataFrame, horizon_ms: float
) -> dict[str, np.ndarray]:
    """Per-MO sweep reach and subsequent mid jump, both in bps of the pre-mid.

    Sign convention matches `get_epsilon.compute_mo_impacts`: for a buy market
    order the jump is `post_mid - pre_mid`, for a sell it is `pre_mid - post_mid`,
    so a positive jump is always adverse to the maker who was filled. Reach uses
    `price_extreme` against `pre_mid` and is truncated at 0, exactly as
    `estimator_common.mo_depths` does for the kappa fit.
    """
    frame = mos[mos["pre_mid"].notna()]
    if frame.empty or mids.empty:
        empty = np.array([])
        return {"reach_bps": empty, "jump_bps": empty, "is_buy": empty.astype(bool)}

    side_is_buy = frame["side"].astype(str).str.lower().to_numpy() == "buy"
    extreme = frame["price_extreme"].to_numpy(dtype=float)
    pre = frame["pre_mid"].to_numpy(dtype=float)
    ts = frame["ts_ms"].to_numpy(dtype=float)

    reach = np.where(side_is_buy, extreme - pre, pre - extreme)
    reach_bps = np.clip(reach / pre, 0.0, None) * 1e4

    mid_ts = mids["ts_ms"].to_numpy(dtype=float)
    mid_px = mids["mid"].to_numpy(dtype=float)
    # Last mid at or before ts+horizon, matching the inclusive backward
    # `merge_asof` in compute_mo_impacts and the `partition_point` in Rust.
    index = np.searchsorted(mid_ts, ts + float(horizon_ms), side="right") - 1
    usable = index >= 0
    post = np.where(usable, mid_px[np.clip(index, 0, len(mid_px) - 1)], np.nan)
    jump_bps = np.where(side_is_buy, post - pre, pre - post) / pre * 1e4

    # Truncation-bias guard, same rule the shipped estimator applies: an MO whose
    # horizon runs past the end of the window would otherwise contribute a jump
    # measured over a shorter interval than every other row.
    within = ts + float(horizon_ms) <= mid_ts[-1]
    keep = usable & within & np.isfinite(jump_bps)
    return {
        "reach_bps": reach_bps[keep],
        "jump_bps": jump_bps[keep],
        "is_buy": side_is_buy[keep],
    }


def conditional_curve(
    reach_bps: np.ndarray, jump_bps: np.ndarray, grid_bps: tuple[float, ...]
) -> list[dict[str, Any]]:
    """E[jump | reach >= d] for each d, with the size of the conditioning set."""
    rows: list[dict[str, Any]] = []
    for depth in grid_bps:
        mask = reach_bps >= depth
        count = int(mask.sum())
        rows.append({
            "depth_bps": float(depth),
            "n": count,
            "mean_jump_bps": float(np.mean(jump_bps[mask])) if count else None,
            "median_jump_bps": float(np.median(jump_bps[mask])) if count else None,
            "usable": count >= MIN_CONDITIONAL_EVENTS,
        })
    return rows


def fit_slope(
    reach_bps: np.ndarray,
    jump_bps: np.ndarray,
    grid_bps: tuple[float, ...],
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Least squares `eps(d) = a + b*d` over the conditional means, with a CI on b.

    The CI is the point of this function. At `b = 1` the optimal-depth problem is
    degenerate -- the objective becomes independent of depth -- and the measured
    values straddle exactly that. A point estimate either side of 1 is therefore
    not evidence of anything, so the gate that eventually consumes this must read
    the interval, never the point.

    Resampling is over MARKET ORDERS, not over grid points: the grid points are
    nested conditioning sets built from the same events and are strongly
    dependent, so resampling them would understate the true spread.
    """
    usable = [
        d for d in grid_bps
        if int((reach_bps >= d).sum()) >= MIN_CONDITIONAL_EVENTS
    ]
    if len(usable) < 3:
        return {"ok": False, "reason": f"only {len(usable)} usable grid points"}

    def one(sample_reach: np.ndarray, sample_jump: np.ndarray) -> tuple[float, float] | None:
        ys, xs = [], []
        for depth in usable:
            mask = sample_reach >= depth
            if not mask.any():
                continue
            ys.append(float(np.mean(sample_jump[mask])))
            xs.append(depth)
        if len(xs) < 3:
            return None
        slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
        return float(slope), float(intercept)

    point = one(reach_bps, jump_bps)
    if point is None:
        return {"ok": False, "reason": "degenerate fit"}
    slope, intercept = point

    slopes: list[float] = []
    n = len(reach_bps)
    for _ in range(max(0, int(bootstrap))):
        pick = rng.integers(0, n, size=n)
        drawn = one(reach_bps[pick], jump_bps[pick])
        if drawn is not None:
            slopes.append(drawn[0])

    result: dict[str, Any] = {
        "ok": True,
        "a_bps": intercept,
        "b": slope,
        "grid_points_used": len(usable),
        "n_events": int(n),
    }
    if slopes:
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        result["b_ci95"] = [float(lo), float(hi)]
        # The decision-relevant quantity: (1-b) multiplies depth in the per-fill
        # edge, so if its lower bound is <= 0 no depth is defensibly profitable.
        result["one_minus_b_ci95"] = [float(1.0 - hi), float(1.0 - lo)]
        result["makeable"] = bool(1.0 - hi > 0.0)
    return result


def edge_bps(depth_bps: float, a_bps: float, b: float, fee_bps: float) -> float:
    """Per-fill maker edge at a quoted depth, in bps: capture minus markout minus fee."""
    return (1.0 - b) * depth_bps - a_bps - fee_bps


def analyse(
    mos: pd.DataFrame,
    mids: pd.DataFrame,
    *,
    horizons_ms: tuple[int, ...],
    grid_bps: tuple[float, ...],
    fee_bps: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"horizons": {}}
    for horizon in horizons_ms:
        data = reach_and_jump(mos, mids, horizon)
        per_side: dict[str, Any] = {}
        for label, mask in (
            ("epsilon+", data["is_buy"]),
            ("epsilon-", ~data["is_buy"]),
            ("pooled", np.ones_like(data["is_buy"], dtype=bool)),
        ):
            reach, jump = data["reach_bps"][mask], data["jump_bps"][mask]
            if reach.size == 0:
                per_side[label] = {"ok": False, "reason": "no events"}
                continue
            fit = fit_slope(reach, jump, grid_bps, bootstrap=bootstrap, rng=rng)
            entry: dict[str, Any] = {
                "n_events": int(reach.size),
                # d = 0 is the unconditional mean: what the shipped estimator
                # computes, and the value of eps(d) at the origin.
                "unconditional_mean_bps": float(np.mean(jump)),
                "curve": conditional_curve(reach, jump, grid_bps),
                "fit": fit,
            }
            if fit.get("ok"):
                entry["edge_bps_at"] = {
                    str(int(d)): edge_bps(d, fit["a_bps"], fit["b"], fee_bps)
                    for d in (10.0, 26.0, 40.0, 60.0)
                }
            per_side[label] = entry
        out["horizons"][str(horizon)] = per_side
    return out


def to_markdown(payload: dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        f"# Fill-conditional adverse selection — {args.symbol}",
        "",
        "`E[mid jump | the sweep reached depth d]`, the quantity the fill term of the",
        "Cartea-Jaimungal DPE actually needs. The shipped estimator computes the `d = 0`",
        "row of this table and uses it at every depth.",
        "",
        f"- window: `{payload['window']['start']}` → `{payload['window']['end']}` "
        f"({payload['window']['hours']:.1f} h)",
        f"- market orders: {payload['window']['n_market_orders']}",
        f"- maker fee assumed: {args.fee_bps:g} bps",
        "",
        "Per-fill maker edge at depth `d` is `(1 - b) * d - a - fee`, so `b` decides",
        "whether ANY depth is profitable. Read the CI on `b`, not the point estimate:",
        "at `b = 1` the optimal depth is degenerate.",
        "",
    ]
    for horizon, sides in payload["horizons"].items():
        lines += [f"## Markout horizon {int(horizon)} ms", ""]
        pooled = sides.get("pooled", {})
        if pooled.get("curve"):
            lines += [
                "| reach d (bps) | E[jump given reach >= d] | N | usable |",
                "| ---: | ---: | ---: | :--- |",
            ]
            for row in pooled["curve"]:
                mean = row["mean_jump_bps"]
                lines.append(
                    f"| {row['depth_bps']:g} | "
                    f"{'—' if mean is None else format(mean, '.2f')} | "
                    f"{row['n']} | {row['usable']} |"
                )
            lines.append("")
        lines += ["| side | unconditional | a (bps) | b | 95% CI on b | makeable | edge@26bps |",
                  "| :--- | ---: | ---: | ---: | :--- | :--- | ---: |"]
        for label in ("epsilon+", "epsilon-", "pooled"):
            side = sides.get(label, {})
            fit = side.get("fit", {})
            if not fit.get("ok"):
                lines.append(f"| `{label}` | — | — | — | — | — | — |")
                continue
            ci = fit.get("b_ci95")
            ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
            edge = side.get("edge_bps_at", {}).get("26")
            lines.append(
                f"| `{label}` | {side['unconditional_mean_bps']:.2f} | {fit['a_bps']:+.2f} | "
                f"{fit['b']:.3f} | {ci_text} | {fit.get('makeable')} | "
                f"{'—' if edge is None else format(edge, '+.2f')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fill-conditional epsilon diagnostic")
    parser.add_argument("--symbol", default="CASHCAT")
    parser.add_argument("--data-dir", type=Path, default=SCRIPTS / "HL_data")
    parser.add_argument("--minutes", type=int, default=0,
                        help="window length; 0 with explicit bounds uses the bounds")
    parser.add_argument("--window-start", default=None)
    parser.add_argument("--window-end", default=None)
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=list(DEFAULT_HORIZONS_MS))
    parser.add_argument("--grid-bps", type=float, nargs="+", default=list(DEFAULT_GRID_BPS))
    parser.add_argument("--fee-bps", type=float, default=1.5,
                        help="maker fee in bps; 0.00015 as a fraction is 1.5 bps")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="resamples for the CI on b; 0 disables")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window = load_market_window(
        args.symbol,
        minutes=args.minutes,
        data_dir=str(args.data_dir),
        window_start=args.window_start,
        window_end=args.window_end,
    )
    mos = attach_pre_mid(aggregate_market_orders(window.trades), window.mids)
    mos = mos[mos["pre_mid"].notna()]
    if mos.empty:
        raise SystemExit("no market orders with a pre-trade mid in this window")

    payload = analyse(
        mos,
        window.mids,
        horizons_ms=tuple(int(h) for h in args.horizons_ms),
        grid_bps=tuple(float(d) for d in args.grid_bps),
        fee_bps=float(args.fee_bps),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    span_ms = float(window.mids["ts_ms"].iloc[-1] - window.mids["ts_ms"].iloc[0])
    payload["window"] = {
        "symbol": args.symbol,
        "start": args.window_start,
        "end": args.window_end,
        "hours": span_ms / 3_600_000.0,
        "n_market_orders": int(len(mos)),
        "n_mids": int(len(window.mids)),
    }
    payload["settings"] = {
        "horizons_ms": list(args.horizons_ms),
        "grid_bps": list(args.grid_bps),
        "fee_bps": args.fee_bps,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "min_conditional_events": MIN_CONDITIONAL_EVENTS,
    }

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(to_markdown(payload, args), encoding="utf-8")

    for horizon, sides in payload["horizons"].items():
        pooled = sides.get("pooled", {})
        fit = pooled.get("fit", {})
        if not fit.get("ok"):
            print(f"  {int(horizon):>6} ms: {fit.get('reason')}", flush=True)
            continue
        ci = fit.get("b_ci95")
        ci_text = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "n/a"
        print(
            f"  {int(horizon):>6} ms: uncond={pooled['unconditional_mean_bps']:5.2f} bps  "
            f"a={fit['a_bps']:+6.2f}  b={fit['b']:.3f} CI95={ci_text}  "
            f"makeable={fit.get('makeable')}  "
            f"edge@26bps={pooled.get('edge_bps_at', {}).get('26', float('nan')):+.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
