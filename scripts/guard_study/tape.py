"""Shared tape access for the guard-candidate study.

Reads the frozen shard sets under scripts/guard_study_tapes/ directly with
pandas/pyarrow, reusing the alignment conventions from estimator_common.py
(prices = one row per side per BBO event; exchange_timestamp is the time
base, matching the replay guard which runs on exchange time).

Everything here is measurement-side: it sees all 20 book levels and the BBO
sizes that the Rust replay loader drops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import estimator_common as ec  # noqa: E402

TAPES_DIR = SCRIPTS_DIR / "guard_study_tapes"

# Study windows (UTC, epoch ms) -- identical to TOXIC_FLOW_GUARD.md.
CRASH_WINDOW_MS = (1_787_342_400_000, 1_787_400_000_000)  # 08-21 20:00 -> 08-22 12:00
CALM_WINDOW_MS = (1_787_126_400_000, 1_787_184_000_000)  # 08-19 08:00 -> 08-20 00:00

# Canonical cascade exclusion zone for false-positive counting: the losing
# six-hour window the 161.95h staged sweep localised the entire loss to
# (08-22 03:57 -> 09:57). "Outside the cascade" always means outside this.
CASCADE_ZONE_MS = (1_787_371_020_000, 1_787_392_620_000)

# Shipped-guard reference points, for the self-check every script must print:
# fast breaker (8%/5s) first trip and VPIN >= 0.40 first breach on the cascade.
BREAKER_FIRST_TRIP_MS = 1_787_375_475_000  # 2026-08-22 05:11:15Z
VPIN_FIRST_TRIP_MS = 1_787_375_491_000  # 2026-08-22 05:11:31Z

# Shipped FlowGuardConfig values (mm-settings defaults == cashcat.toml).
FAST_MOVE_WINDOW_MS = 5_000
FAST_MOVE_THRESHOLD_BPS = 800.0
VPIN_BUCKETS_PER_DAY = 50
VPIN_WINDOW_BUCKETS = 30
VPIN_THRESHOLD = 0.40


def _ts_ms(frame: pd.DataFrame) -> pd.Series:
    """Exchange time in ms, falling back to local recv time (seconds float)."""
    ts = pd.to_numeric(frame.get("exchange_timestamp"), errors="coerce").astype(float)
    if ts.isna().all():
        ts = pd.to_numeric(frame["timestamp"], errors="coerce").astype(float) * 1_000.0
    return ts


def load_stream(
    tape_dir: Path | str,
    stream: str,
    start_ms: float | None = None,
    end_ms: float | None = None,
    symbol: str = "CASHCAT",
) -> pd.DataFrame:
    """Load one shard stream (prices|trades|orderbooks) for an absolute window."""
    root = Path(tape_dir) / symbol / stream
    files = sorted(root.glob("*.parquet"))
    files = ec.select_shards_for_bounds(files, start_ms, end_ms)
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001 - one bad shard must not kill a study pass
            print(f"WARN unreadable shard {path.name}: {exc}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["ts_ms"] = _ts_ms(frame)
    frame = frame.dropna(subset=["ts_ms"]).sort_values("ts_ms", kind="stable")
    if start_ms is not None:
        frame = frame[frame["ts_ms"] >= float(start_ms)]
    if end_ms is not None:
        frame = frame[frame["ts_ms"] <= float(end_ms)]
    return frame.reset_index(drop=True)


def build_bbo_full(prices: pd.DataFrame) -> pd.DataFrame:
    """BBO events with px AND size per side: ts_ms, bid, ask, bid_sz, ask_sz,
    mid, spread_bps. This is the study's version of estimator_common's
    build_bbo_mid -- the difference is that sizes survive."""
    if prices.empty:
        return pd.DataFrame(
            columns=["ts_ms", "bid", "ask", "bid_sz", "ask_sz", "mid", "spread_bps"]
        )
    frame = prices.copy()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame[frame["side"].isin({"bid", "ask"})].dropna(subset=["ts_ms", "price"])
    px = frame.pivot_table(index="ts_ms", columns="side", values="price", aggfunc="last")
    sz = frame.pivot_table(index="ts_ms", columns="side", values="size", aggfunc="last")
    bbo = pd.DataFrame(
        {
            "bid": px.get("bid"),
            "ask": px.get("ask"),
            "bid_sz": sz.get("bid"),
            "ask_sz": sz.get("ask"),
        }
    ).sort_index()
    bbo = bbo.ffill().dropna(subset=["bid", "ask"])
    bbo = bbo[(bbo["bid"] > 0) & (bbo["ask"] > 0) & (bbo["bid"] < bbo["ask"])]
    bbo = bbo.reset_index().rename(columns={"index": "ts_ms"})
    bbo["mid"] = (bbo["bid"] + bbo["ask"]) / 2.0
    bbo["spread_bps"] = (bbo["ask"] - bbo["bid"]) / bbo["mid"] * 10_000.0
    return bbo


def fast_move_bps(ts_ms: np.ndarray, mid: np.ndarray, window_ms: float) -> np.ndarray:
    """Replicates MidWindow::observe: |move| in bps against the OLDEST sample
    still inside the trailing window, per event. Vectorised."""
    cutoff = ts_ms - window_ms
    oldest = np.searchsorted(ts_ms, cutoff, side="left")
    ref = mid[oldest]
    return np.abs(mid - ref) / ref * 10_000.0


def vpin_series(
    trades: pd.DataFrame,
    bucket_volume: float,
    window_buckets: int = VPIN_WINDOW_BUCKETS,
) -> pd.DataFrame:
    """Replicates VpinTracker: consecutive-duplicate trade_id dedup, bucket
    accumulation with dominant-side overflow carry, rolling |imbalance| sum.
    Returns one row per COMPLETED bucket: ts_ms (completion time), imbalance,
    vpin (NaN until window_buckets buckets exist), duration_ms."""
    if trades.empty:
        return pd.DataFrame(columns=["ts_ms", "imbalance", "vpin", "duration_ms"])
    tid = trades["trade_id"].astype(str).to_numpy()
    keep = np.ones(len(tid), dtype=bool)
    keep[1:] = tid[1:] != tid[:-1]  # single-entry memory, exactly like the Rust
    ts = trades["ts_ms"].to_numpy(dtype=float)[keep]
    qty = trades["size"].to_numpy(dtype=float).clip(min=0.0)[keep]
    is_buy = (trades["side"].astype(str).str.lower() == "buy").to_numpy()[keep]

    bucket_volume = max(bucket_volume, 1e-12)
    rows: list[tuple[float, float, float]] = []
    buy = sell = 0.0
    started_ms = ts[0] if len(ts) else 0.0
    for i in range(len(ts)):
        if is_buy[i]:
            buy += qty[i]
        else:
            sell += qty[i]
        while buy + sell >= bucket_volume:
            rows.append((ts[i], abs(buy - sell), ts[i] - started_ms))
            overflow = buy + sell - bucket_volume
            if buy >= sell:
                buy, sell = overflow, 0.0
            else:
                buy, sell = 0.0, overflow
            started_ms = ts[i]
    out = pd.DataFrame(rows, columns=["ts_ms", "imbalance", "duration_ms"])
    if out.empty:
        return out.assign(vpin=pd.Series(dtype=float))
    rolling = out["imbalance"].rolling(window_buckets).sum()
    out["vpin"] = rolling / (window_buckets * bucket_volume)
    return out


def bucket_volume_from(trades: pd.DataFrame, buckets_per_day: int = VPIN_BUCKETS_PER_DAY) -> float:
    """vpin_bucket_units equivalent: ADV over the loaded window / buckets_per_day."""
    if trades.empty:
        return 1.0
    span_days = max((trades["ts_ms"].iloc[-1] - trades["ts_ms"].iloc[0]) / 86_400_000.0, 1e-9)
    total = float(trades["size"].clip(lower=0).sum())
    return max(total / span_days / max(buckets_per_day, 1), 1e-12)


def iso(ts_ms: float) -> str:
    return pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").strftime("%m-%d %H:%M:%S")
