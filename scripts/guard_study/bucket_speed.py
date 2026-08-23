"""Candidate 2: bucket-completion-speed alarm.

VPIN's volume clock accelerating is itself a signal: a bucket that normally
takes ~30 min completing in seconds, one-sided, is an alarm long before the
30-bucket VPIN average crosses 0.40. Rule under test:

    trip when duration_ms < median_duration / k  AND  |imbalance| / V >= m

over a grid of (k, m). Reported per rule: first trip on the cascade vs the
shipped breaker (05:11:15) and shipped VPIN (05:11:31), and every trip
outside the canonical cascade zone with its timestamp (for forgone-P&L
costing against the baseline replay).

Sensitivity: the whole grid is recomputed at bucket sizes x0.5 and x2 --
bucket volume is ADV-derived and ADV moves, so a rule that only works at one
bucket size is overfit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tape as T

K_GRID = (5.0, 10.0, 20.0, 50.0)
M_GRID = (0.5, 0.7, 0.9)
BUCKET_SCALES = (1.0, 0.5, 2.0)


def evaluate(vp, bucket_volume: float, k: float, m: float) -> dict:
    ts = vp["ts_ms"].to_numpy(dtype=float)
    dur = vp["duration_ms"].to_numpy(dtype=float)
    imb = vp["imbalance"].to_numpy(dtype=float)
    median = float(np.median(dur))
    fired = (dur < median / k) & (imb / bucket_volume >= m)
    zone_lo, zone_hi = T.CASCADE_ZONE_MS
    inside = fired & (ts >= zone_lo) & (ts <= zone_hi)
    outside = fired & ~((ts >= zone_lo) & (ts <= zone_hi))
    first_inside = float(ts[inside][0]) if inside.any() else None
    return {
        "k": k,
        "m": m,
        "median_duration_s": median / 1_000.0,
        "fires": int(fired.sum()),
        "inside": int(inside.sum()),
        "outside": int(outside.sum()),
        "first_inside_ms": first_inside,
        "first_inside": T.iso(first_inside) if first_inside else None,
        "outside_times": [T.iso(t) for t in ts[outside][:50]],
        "earliness_vs_breaker_s": (
            (T.BREAKER_FIRST_TRIP_MS - first_inside) / 1_000.0 if first_inside else None
        ),
    }


def main() -> None:
    trades = T.load_stream(T.TAPES_DIR / "full_tape", "trades")
    base_bucket = T.bucket_volume_from(trades)
    print(f"trades: {len(trades)}  base bucket volume: {base_bucket:.0f} units")

    results = {}
    for scale in BUCKET_SCALES:
        bucket = base_bucket * scale
        vp = T.vpin_series(trades, bucket)
        dur = vp["duration_ms"].to_numpy(dtype=float)
        print(
            f"\n=== bucket x{scale} ({bucket:.0f} units): {len(vp)} buckets, "
            f"median completion {np.median(dur) / 1_000.0:.0f} s, p1 {np.percentile(dur, 1) / 1_000.0:.1f} s ==="
        )
        rows = []
        for k in K_GRID:
            for m in M_GRID:
                row = evaluate(vp, bucket, k, m)
                rows.append(row)
                flag = " <-- beats breaker" if (row["earliness_vs_breaker_s"] or 0) >= 5 else ""
                print(
                    f"  k={k:>4} m={m}: fires={row['fires']:>4} outside={row['outside']:>4} "
                    f"first_inside={row['first_inside']}"
                    f" (+{row['earliness_vs_breaker_s']}s vs breaker){flag}"
                    if row["first_inside"]
                    else f"  k={k:>4} m={m}: fires={row['fires']:>4} outside={row['outside']:>4} never fires inside zone"
                )
        results[f"bucket_x{scale}"] = rows

    out = Path(__file__).with_name("bucket_speed.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
