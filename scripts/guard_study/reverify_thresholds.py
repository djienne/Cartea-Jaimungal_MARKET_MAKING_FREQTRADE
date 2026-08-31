"""Phase 0: re-verify the shipped guard thresholds on the extended freeze.

The 8%/5s and VPIN 0.40 zero-false-positive claims were made on 6.8 days of
tape; the full freeze now covers more. This recomputes both statistics over
the whole freeze and reports every breach with its timestamp, split into
inside/outside the canonical cascade zone (08-22 03:57 -> 09:57).

Self-check (mandatory for every study script): the recomputed first trips
must land on 05:11:15 (breaker) and 05:11:31 (VPIN) -- if they do not, the
script cannot be trusted on anything else.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import tape as T


def main() -> None:
    tape_dir = T.TAPES_DIR / "full_tape"
    prices = T.load_stream(tape_dir, "prices")
    trades = T.load_stream(tape_dir, "trades")
    bbo = T.build_bbo_full(prices)
    print(
        f"freeze: {T.iso(bbo['ts_ms'].iloc[0])} -> {T.iso(bbo['ts_ms'].iloc[-1])} "
        f"({(bbo['ts_ms'].iloc[-1] - bbo['ts_ms'].iloc[0]) / 3_600_000.0:.2f} h), "
        f"{len(bbo)} bbo events, {len(trades)} trades"
    )

    zone_lo, zone_hi = T.CASCADE_ZONE_MS
    ts = bbo["ts_ms"].to_numpy(dtype=float)
    mid = bbo["mid"].to_numpy(dtype=float)

    # --- fast breaker ---
    move = T.fast_move_bps(ts, mid, T.FAST_MOVE_WINDOW_MS)
    fired = move >= T.FAST_MOVE_THRESHOLD_BPS
    inside = fired & (ts >= zone_lo) & (ts <= zone_hi)
    outside = fired & ~((ts >= zone_lo) & (ts <= zone_hi))
    first = ts[fired][0] if fired.any() else None
    print("\n[fast breaker >= 800 bps / 5 s]")
    print(f"  fires: {int(fired.sum())} (inside zone {int(inside.sum())}, outside {int(outside.sum())})")
    print(f"  first trip: {T.iso(first) if first else 'never'}  (self-check: {T.iso(T.BREAKER_FIRST_TRIP_MS)})")
    if outside.any():
        for t, m in zip(ts[outside][:20], move[outside][:20]):
            print(f"    OUTSIDE-ZONE FIRE {T.iso(t)}  {m:.0f} bps")
    calm_mask = ~((ts >= zone_lo) & (ts <= zone_hi))
    print(f"  max move outside zone: {move[calm_mask].max():.1f} bps (threshold 800)")

    # --- VPIN ---
    bucket = T.bucket_volume_from(trades)
    vp = T.vpin_series(trades, bucket)
    complete = vp.dropna(subset=["vpin"])
    vts = complete["ts_ms"].to_numpy(dtype=float)
    vval = complete["vpin"].to_numpy(dtype=float)
    vfired = vval >= T.VPIN_THRESHOLD
    vzone = (vts >= zone_lo) & (vts <= zone_hi)
    vfirst = vts[vfired][0] if vfired.any() else None
    print(f"\n[VPIN >= {T.VPIN_THRESHOLD} | bucket={bucket:.0f} units, {T.VPIN_WINDOW_BUCKETS} buckets]")
    print(f"  buckets: {len(vp)} total, {len(complete)} with full window")
    print(f"  breaches: {int(vfired.sum())} (inside zone {int((vfired & vzone).sum())}, outside {int((vfired & ~vzone).sum())})")
    print(f"  first breach: {T.iso(vfirst) if vfirst else 'never'}  (self-check: {T.iso(T.VPIN_FIRST_TRIP_MS)})")
    print(f"  max VPIN outside zone: {vval[~vzone].max():.3f}   peak inside: {vval[vzone].max() if vzone.any() else float('nan'):.3f}")
    out = vfired & ~vzone
    for t, v in zip(vts[out][:20], vval[out][:20]):
        print(f"    OUTSIDE-ZONE BREACH {T.iso(t)}  vpin={v:.3f}")
    max_benign = float(vval[~vzone].max())
    peak_cascade = float(vval[vzone].max()) if vzone.any() else float("nan")
    recommended = math.floor(max_benign / 0.05 + 1.0) * 0.05
    tier_valid = recommended < 1.0 and peak_cascade >= recommended
    print(
        f"  evidence threshold: {recommended:.2f} "
        f"({'retain VPIN tier' if tier_valid else 'disable VPIN tier'})"
    )

    result = {
        "freeze_span_h": float((ts[-1] - ts[0]) / 3_600_000.0),
        "breaker": {
            "fires": int(fired.sum()),
            "outside_zone": int(outside.sum()),
            "first_trip_ms": float(first) if first else None,
            "max_outside_bps": float(move[calm_mask].max()),
        },
        "vpin": {
            "bucket_volume": float(bucket),
            "breaches": int(vfired.sum()),
            "outside_zone": int((vfired & ~vzone).sum()),
            "first_breach_ms": float(vfirst) if vfirst else None,
            "max_outside": float(vval[~vzone].max()),
            "peak_inside": peak_cascade,
            "recommended_threshold": recommended,
            "tier_valid": tier_valid,
        },
    }
    out_path = Path(__file__).with_name("reverify_thresholds.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
