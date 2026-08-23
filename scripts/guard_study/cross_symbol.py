"""Candidate 4, step 1: was the 08-22 05:11 cascade CASHCAT-idiosyncratic?

The oracle-dislocation / OI-drop signal is only worth building if the crash
was a single-instrument liquidation cascade (perp dislocates alone, mean
reverts) rather than a market-wide repricing. The sibling tapes cover the
event; this measures each symbol's worst 5-second and 60-second mid move in
05:00-05:30 and its net move over the cascade minute, next to CASHCAT's.
"""

from __future__ import annotations

import numpy as np

import tape as T

WINDOW = (1_787_374_800_000, 1_787_376_600_000)  # 08-22 05:00 -> 05:30
CASCADE_MINUTE = (1_787_375_460_000, 1_787_375_520_000)  # 05:11:00 -> 05:12:00


def worst_moves(tape_dir, symbol) -> dict | None:
    prices = T.load_stream(tape_dir, "prices", *WINDOW, symbol=symbol)
    if prices.empty:
        return None
    bbo = T.build_bbo_full(prices)
    if len(bbo) < 10:
        return None
    ts = bbo["ts_ms"].to_numpy(dtype=float)
    mid = bbo["mid"].to_numpy(dtype=float)
    m5 = T.fast_move_bps(ts, mid, 5_000).max()
    m60 = T.fast_move_bps(ts, mid, 60_000).max()
    lo, hi = CASCADE_MINUTE
    inside = (ts >= lo) & (ts <= hi)
    if inside.any():
        first, last = mid[inside][0], mid[inside][-1]
        cascade_pct = (last - first) / first * 100.0
    else:
        cascade_pct = float("nan")
    return {"max_5s_bps": float(m5), "max_60s_bps": float(m60), "cascade_min_pct": float(cascade_pct)}


def main() -> None:
    rows = {"CASHCAT": worst_moves(T.TAPES_DIR / "full_tape", "CASHCAT")}
    for sym in ("ETH", "PENGU", "ACE", "CHIP", "NIL"):
        rows[sym] = worst_moves(T.TAPES_DIR / "siblings", sym)
    print(f"{'symbol':8s} {'max 5s move':>12s} {'max 60s move':>13s} {'05:11 minute':>13s}")
    for sym, row in rows.items():
        if row is None:
            print(f"{sym:8s} {'no data':>12s}")
            continue
        print(
            f"{sym:8s} {row['max_5s_bps']:>10.0f}bp {row['max_60s_bps']:>11.0f}bp "
            f"{row['cascade_min_pct']:>12.2f}%"
        )


if __name__ == "__main__":
    main()
