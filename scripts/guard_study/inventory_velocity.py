"""Candidate 3: inventory-velocity guard.

Trip when |inventory change| over a trailing window Y exceeds X units -- a
one-sided fill burst. Fires only when WE are being run over, so it has no
market-noise false positives by construction; the question is whether a
threshold exists that clears every benign window with margin while tripping
early in the cascade accumulation.

Sources: the Phase-0 baseline replay JSONLs (crash/calm/full guardoff legs --
guardoff so the cascade accumulation is visible un-truncated) plus the live
grid JSONLs as a second benign source. Inventory is reconstructed by cumsum
of execution_event:fill (+/- qty by side); quote_decision.q_exact is NOT used
because empty decisions hardcode q=0.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

import tape as T

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]

WINDOWS_S = (10, 30, 60, 120)
RUNS = T.TAPES_DIR / "runs"


def fill_trajectory(jsonl_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(ts_ms, inventory_units) after each fill. Handles .jsonl and .jsonl.zst."""
    from grid_pnl_curve import open_log  # noqa: PLC0415

    ts: list[float] = []
    inv: list[float] = []
    running = 0.0
    with open_log(Path(jsonl_path)) as handle:
        for line in handle:
            if '"execution_event"' not in line or '"fill"' not in line:
                continue
            rec = json.loads(line)
            payload = rec.get("payload", {})
            if payload.get("kind") != "fill":
                continue
            qty = float(payload.get("qty_units", 0))
            side = str(payload.get("side", "")).lower()
            running += qty if side == "buy" else -qty
            when = rec.get("exchange_ms") or payload.get("exchange_ms")
            if when is None:
                continue
            ts.append(float(when))
            inv.append(running)
    return np.asarray(ts, dtype=float), np.asarray(inv, dtype=float)


def max_velocity(ts: np.ndarray, inv: np.ndarray, window_ms: float) -> tuple[float, float | None]:
    """Max |inventory(t) - inventory(t - window)| and when it first PEAKS.
    Inventory between fills is a step function; evaluating at fill times is
    exact because extremes of the difference occur at fill events."""
    if len(ts) == 0:
        return 0.0, None
    # inventory just before the window start: value of the last fill at or
    # before (t - window); zero if none.
    idx = np.searchsorted(ts, ts - window_ms, side="right") - 1
    prev = np.where(idx >= 0, inv[np.clip(idx, 0, None)], 0.0)
    delta = np.abs(inv - prev)
    peak = float(delta.max())
    return peak, float(ts[int(delta.argmax())])


def velocity_first_cross(ts, inv, window_ms, threshold) -> float | None:
    idx = np.searchsorted(ts, ts - window_ms, side="right") - 1
    prev = np.where(idx >= 0, inv[np.clip(idx, 0, None)], 0.0)
    crossed = np.abs(inv - prev) >= threshold
    return float(ts[crossed][0]) if crossed.any() else None


def main() -> None:
    sources: dict[str, str] = {}
    for leg in ("crash_guardoff", "calm_guardoff", "full_guardoff"):
        files = glob.glob(str(RUNS / leg / "replay-*.jsonl"))
        if files:
            sources[leg] = max(files)
    # Grid logs are zstd now and keep a stable stem; match both schemes.
    sys.path.insert(0, str(SCRIPTS_ROOT))
    from grid_pnl_curve import variant_files  # noqa: PLC0415

    for variant, path in variant_files(None).items():
        sources[f"grid:{variant}"] = str(path)

    zone_lo, zone_hi = T.CASCADE_ZONE_MS
    print(f"{'source':22s} " + "  ".join(f"maxV({w}s)" for w in WINDOWS_S) + "   (benign = outside cascade zone)")
    table: dict[str, dict] = {}
    crash_data = None
    for name, path in sorted(sources.items()):
        ts, inv = fill_trajectory(path)
        if name == "crash_guardoff":
            crash_data = (ts, inv)
        row = {}
        for w in WINDOWS_S:
            if len(ts):
                in_zone = (ts >= zone_lo) & (ts <= zone_hi)
                benign_ts, benign_inv = ts, inv
                if in_zone.any():
                    # benign = velocity evaluated only at fills outside the zone
                    idx = np.searchsorted(ts, ts - w * 1_000.0, side="right") - 1
                    prev = np.where(idx >= 0, inv[np.clip(idx, 0, None)], 0.0)
                    delta = np.abs(inv - prev)
                    benign_peak = float(delta[~in_zone].max()) if (~in_zone).any() else 0.0
                    zone_peak = float(delta[in_zone].max())
                    row[w] = {"benign": benign_peak, "zone": zone_peak}
                else:
                    peak, _ = max_velocity(ts, inv, w * 1_000.0)
                    row[w] = {"benign": peak, "zone": None}
            else:
                row[w] = {"benign": 0.0, "zone": None}
        table[name] = {"fills": len(ts), "velocity": row}
        cells = "  ".join(
            f"{row[w]['benign']:9.0f}" + (f"/{row[w]['zone']:.0f}z" if row[w]["zone"] else "")
            for w in WINDOWS_S
        )
        print(f"{name:22s} {cells}   fills={len(ts)}")

    # Threshold design: for each window, the smallest X that is >= 3x every
    # benign peak; then when does the crash leg first cross it?
    print("\n=== candidate thresholds (3x max benign peak) and crash first-cross ===")
    design = {}
    for w in WINDOWS_S:
        benign_max = max(t["velocity"][w]["benign"] for t in table.values())
        threshold = 3.0 * benign_max
        first = None
        if crash_data is not None:
            first = velocity_first_cross(*crash_data, w * 1_000.0, threshold)
        lateness = (first - T.BREAKER_FIRST_TRIP_MS) / 1_000.0 if first else None
        design[w] = {
            "benign_max": benign_max,
            "threshold": threshold,
            "crash_first_cross_ms": first,
            "vs_breaker_s": lateness,
        }
        print(
            f"  Y={w:>4}s: benign_max={benign_max:9.0f} -> X={threshold:9.0f}  "
            f"crash first-cross={T.iso(first) if first else 'never'}"
            + (f"  ({lateness:+.0f}s vs breaker)" if lateness is not None else "")
        )

    out = Path(__file__).with_name("inventory_velocity.json")
    out.write_text(
        json.dumps({"sources": {k: v["fills"] for k, v in table.items()},
                    "velocity": {k: v["velocity"] for k, v in table.items()},
                    "design": design}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
