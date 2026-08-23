"""Candidate 1: spread-blowout / depth-withdrawal tripwire.

Thesis: makers pull before price moves -- spread widens and top-of-book size
collapses hundreds of ms to seconds before the mid has moved far. If true,
a rule on spread or L1 depth fires EARLIER than the shipped 8%/5s breaker
(05:11:15). That earliness is the only reason to keep this candidate.

Rules under test, each against its own trailing median (lookbacks 15m/1h/4h,
a rule that only separates at one lookback is overfit):

    spread_bps > k_s * trailing_median(spread_bps)
    min(bid_sz, ask_sz) < trailing_median(min_sz) / k_d

Depth beyond L1 (5-level / 20-level sums from the 5 s orderbooks stream) is
computed as a diagnostic only: at 0.2 Hz it cannot demonstrate sub-5s
earliness and could not be faithfully replayed.

Self-check printed first: the breaker's first trip recomputed from the same
BBO series must be 05:11:15.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import tape as T

LOOKBACKS_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
KS_GRID = (3.0, 5.0, 10.0, 20.0)
KD_GRID = (3.0, 5.0, 10.0, 20.0)


def trailing_median(ts: np.ndarray, values: np.ndarray, lookback_ms: float) -> np.ndarray:
    """Median of values in [t - lookback, t), per event. Computed on a 1-minute
    grid then forward-mapped -- exact per-event medians are O(n^2) and the
    signal question is minutes-scale, so a minute grid loses nothing."""
    frame = pd.DataFrame({"ts": pd.to_datetime(ts, unit="ms"), "v": values}).set_index("ts")
    grid = frame["v"].resample("1min").median()
    rolled = grid.rolling(f"{int(lookback_ms / 60_000)}min", min_periods=5).median()
    # value at minute m uses data up to m's end; shift so each event sees only
    # strictly earlier minutes (no lookahead through the current minute).
    rolled = rolled.shift(1)
    idx = np.searchsorted(rolled.index.view("int64") // 1_000_000, ts, side="right") - 1
    out = np.full(len(ts), np.nan)
    valid = idx >= 0
    out[valid] = rolled.to_numpy()[idx[valid]]
    return out


def score_rule(ts: np.ndarray, fired: np.ndarray, label: str) -> dict:
    zone_lo, zone_hi = T.CASCADE_ZONE_MS
    in_zone = (ts >= zone_lo) & (ts <= zone_hi)
    inside = fired & in_zone
    outside = fired & ~in_zone
    first_inside = float(ts[inside][0]) if inside.any() else None
    # Count outside fires as EPISODES (gap > 5 min = new episode), not events:
    # one widening that lasts a minute is one withdrawal, not sixty FPs.
    ep_times = ts[outside]
    episodes = 0
    episode_starts: list[float] = []
    if len(ep_times):
        gaps = np.diff(ep_times)
        episodes = 1 + int((gaps > 300_000).sum())
        starts = np.concatenate(([ep_times[0]], ep_times[1:][gaps > 300_000]))
        episode_starts = [float(s) for s in starts[:50]]
    return {
        "rule": label,
        "fires": int(fired.sum()),
        "inside": int(inside.sum()),
        "outside_events": int(outside.sum()),
        "outside_episodes": episodes,
        "episode_starts": [T.iso(s) for s in episode_starts],
        "first_inside_ms": first_inside,
        "first_inside": T.iso(first_inside) if first_inside else None,
        "earliness_vs_breaker_s": (
            (T.BREAKER_FIRST_TRIP_MS - first_inside) / 1_000.0 if first_inside else None
        ),
    }


def main() -> None:
    prices = T.load_stream(T.TAPES_DIR / "full_tape", "prices")
    bbo = T.build_bbo_full(prices)
    ts = bbo["ts_ms"].to_numpy(dtype=float)
    mid = bbo["mid"].to_numpy(dtype=float)
    spread = bbo["spread_bps"].to_numpy(dtype=float)
    min_sz = np.minimum(bbo["bid_sz"].to_numpy(dtype=float), bbo["ask_sz"].to_numpy(dtype=float))

    # Self-check against the shipped breaker on the identical series.
    move = T.fast_move_bps(ts, mid, T.FAST_MOVE_WINDOW_MS)
    fired = move >= T.FAST_MOVE_THRESHOLD_BPS
    first = ts[fired][0] if fired.any() else None
    print(f"self-check breaker first trip: {T.iso(first)} (must be {T.iso(T.BREAKER_FIRST_TRIP_MS)})")

    results: dict[str, list[dict]] = {}
    for lb_name, lb_ms in LOOKBACKS_MS.items():
        med_spread = trailing_median(ts, spread, lb_ms)
        med_depth = trailing_median(ts, min_sz, lb_ms)
        rows: list[dict] = []
        print(f"\n=== lookback {lb_name} ===")
        for k in KS_GRID:
            row = score_rule(ts, spread > k * med_spread, f"spread>{k}x_med_{lb_name}")
            rows.append(row)
            e = row["earliness_vs_breaker_s"]
            print(
                f"  spread>{k:>4}x: fires={row['fires']:>6} out_ep={row['outside_episodes']:>3} "
                f"first={row['first_inside']} early={e if e is not None else '-'}s"
            )
        for k in KD_GRID:
            row = score_rule(ts, min_sz < med_depth / k, f"depth<med/{k}_{lb_name}")
            rows.append(row)
            e = row["earliness_vs_breaker_s"]
            print(
                f"  depth<1/{k:>3}x: fires={row['fires']:>6} out_ep={row['outside_episodes']:>3} "
                f"first={row['first_inside']} early={e if e is not None else '-'}s"
            )
        results[lb_name] = rows

    # Cascade-morning microscope: what the raw series did around the event.
    lo = T.BREAKER_FIRST_TRIP_MS - 120_000
    hi = T.BREAKER_FIRST_TRIP_MS + 60_000
    win = (ts >= lo) & (ts <= hi)
    print("\n=== 05:09:15 -> 05:12:15, per BBO event (spread_bps / min_sz / mid) ===")
    for t, s, z, m2 in list(zip(ts[win], spread[win], min_sz[win], mid[win]))[::5]:
        print(f"  {T.iso(t)}  spread={s:8.1f}  min_sz={z:12.0f}  mid={m2:.5f}")

    out = Path(__file__).with_name("spread_depth.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
