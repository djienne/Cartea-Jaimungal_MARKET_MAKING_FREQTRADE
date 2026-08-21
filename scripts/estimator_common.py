"""Shared data pipeline for the kappa/lambda/epsilon estimators.

Methodology (schema v4):
- Mid series comes from the dense BBO ``prices/`` stream (one row per side per
  update), falling back to ``orderbooks/`` top-of-book for old datasets.
- Trades and mids are aligned on exchange timestamps (ms) when both streams
  carry them, otherwise on local receive timestamps for both — never mixed,
  so cross-stream clock offsets cannot skew depth/impact measurements.
- Trade prints are aggregated into market orders (same side + same exchange
  timestamp) before any depth or impact measurement.
- Depths are measured from the prevailing MID (strictly before the MO), not
  from the touch: that is the coordinate the strategy quotes in.
- kappa is fitted on the empirical survival function P(depth >= delta), whose
  slope matches the model fill intensity lambda(delta) = Lambda * exp(-kappa
  * delta). lambda is the raw per-side MO arrival rate (count / window), NOT
  a regression intercept — the binned-density intercept equals
  Lambda*kappa*binwidth and is bin-width dependent (kept only as diagnostic).
- The survival fit runs over a configurable depth support [lower, upper], both
  expressed as quantiles of that side's depth distribution. Defaults are
  [0.0, 0.99] — i.e. the whole distribution up to the 99th percentile, exactly
  the fit that shipped before the lower bound existed.
- The window is normally the trailing ``minutes``; explicit window_start /
  window_end bounds select a historical slice instead (used by calibration
  sweeps that must fit on a train slice, never by the live estimator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from param_utils import atomic_write_json


# Validation floors shared with the strategy (Market_Making.py mirrors these as
# class attributes min_kappa_fit_points / min_kappa_r2 / min_epsilon_events).
MIN_KAPPA_FIT_POINTS = 6
MIN_KAPPA_R2 = 0.30
MIN_EPSILON_EVENTS = 50

# Fraction of rows that must carry a finite exchange_timestamp before a stream
# is trusted to be on exchange time.
_EXCHANGE_TS_MIN_COVERAGE = 0.99

# Extra slack when selecting shards for a window: a shard flushed at time T holds
# rows from roughly T-FLUSH_INTERVAL_SEC to T, and clocks between the collector
# and the estimator need not agree exactly. Two minutes covers both with room to
# spare while still bounding the work per cycle.
SHARD_WINDOW_MARGIN_MS = 120_000.0

# Above this share of unreadable shards in one stream, fail the cycle instead of
# returning short data. Post-atomic-write any failure means real corruption.
MAX_SHARD_READ_FAILURE_RATE = 0.05


class ShardReadError(RuntimeError):
    """Too large a share of a stream's shards failed to read.

    Raised instead of silently returning short data: a dropped shard biases
    n_trades and lambda downward with no signal that anything went missing.
    """


def shard_timestamp_ms(path: Path) -> float | None:
    """Flush timestamp embedded in a shard name ("<dtype>_<ms>.parquet")."""
    stem = path.stem
    if "_" not in stem:
        return None
    try:
        return float(stem.rsplit("_", 1)[1])
    except ValueError:
        return None


def select_shards_for_window(
    files: list[Path],
    window_minutes: float | None,
    *,
    margin_ms: float = SHARD_WINDOW_MARGIN_MS,
) -> list[Path]:
    """Drop shards that cannot contain data inside the trailing window.

    Cost used to scale with RETENTION_MINUTES rather than with the window the
    estimator actually needs: at 60 min retention and a 10 s flush that is ~360
    shards per stream re-read every cycle, and README advises disabling pruning
    entirely to capture replay datasets, which makes it unbounded.

    The cutoff is relative to the newest shard rather than wall-clock time, so
    this behaves identically on a live directory and on a frozen historical one.
    Shards whose name cannot be parsed are kept -- fail open, never drop data
    because of a naming surprise.
    """
    if window_minutes is None or window_minutes <= 0 or not files:
        return files
    stamped = [(path, shard_timestamp_ms(path)) for path in files]
    known = [ts for _, ts in stamped if ts is not None]
    if not known:
        return files
    cutoff = max(known) - (float(window_minutes) * 60_000.0 + float(margin_ms))
    return [path for path, ts in stamped if ts is None or ts >= cutoff]


def select_shards_for_bounds(
    files: list[Path],
    start_ms: float | None,
    end_ms: float | None,
    *,
    margin_ms: float = SHARD_WINDOW_MARGIN_MS,
) -> list[Path]:
    """Drop shards that cannot contain data inside an ABSOLUTE [start, end] window.

    select_shards_for_window() is relative to the newest shard, which is right
    for the live trailing window but wrong for a historical slice: asking for
    yesterday's train window would keep only today's shards and silently return
    nothing. This filters on the shard's own flush timestamp instead.

    A shard flushed at T holds rows from roughly T-FLUSH_INTERVAL_SEC to T, so a
    shard is relevant when T >= start (its newest rows may be in range) and
    T - flush <= end; the same two-minute margin used for the trailing window
    covers the flush interval and any clock disagreement. Unparseable names are
    kept — fail open, never drop data because of a naming surprise.
    """
    if not files or (start_ms is None and end_ms is None):
        return files
    kept: list[Path] = []
    for path in files:
        ts = shard_timestamp_ms(path)
        if ts is None:
            kept.append(path)
            continue
        if start_ms is not None and ts < float(start_ms) - float(margin_ms):
            continue
        if end_ms is not None and ts > float(end_ms) + float(margin_ms):
            continue
        kept.append(path)
    return kept


def parse_window_bound_ms(value: Any) -> float | None:
    """ISO-8601 (or epoch seconds/ms) -> epoch milliseconds; None passes through.

    Window bounds are compared against ts_ms, which is epoch milliseconds on
    whichever clock the window chose (exchange or local receive), so a bound has
    to land on the same scale. Naive ISO strings are read as UTC because every
    timestamp this stack emits — the snapshots' window_start/window_end, the
    collector's shard names — is UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        # Epoch seconds are ~1.8e9 today, epoch ms ~1.8e12; 1e11 separates them
        # for any date this stack can plausibly be pointed at.
        return numeric if abs(numeric) >= 1e11 else numeric * 1000.0
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        pass
    else:
        return numeric if abs(numeric) >= 1e11 else numeric * 1000.0
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Unparseable window bound {value!r}: expected ISO-8601 or epoch") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def _load_parquet_dir(
    directory: Path,
    window_minutes: float | None = None,
    bounds: tuple[float | None, float | None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Concatenate a stream's shards, reporting what was skipped or unreadable."""
    stats: dict[str, Any] = {
        "files_present": 0,
        "files_considered": 0,
        "files_read": 0,
        "files_failed": 0,
        "files_vanished": 0,
        "errors": [],
    }
    if not directory.is_dir():
        return pd.DataFrame(), stats
    files = sorted(directory.glob("*.parquet"))
    stats["files_present"] = len(files)
    if not files:
        return pd.DataFrame(), stats

    if bounds is None:
        files = select_shards_for_window(files, window_minutes)
    else:
        start_ms, end_ms = bounds
        files = select_shards_for_bounds(files, start_ms, end_ms)
        if start_ms is None:
            # Only an end was pinned, so the span is still the trailing
            # ``window_minutes`` -- but measured back from the newest shard that
            # SURVIVED the end cutoff, not from an end that may sit past the data.
            # Anchoring on the requested end instead would return nothing at all
            # for "the last hour ending at <a time after the last shard>".
            files = select_shards_for_window(files, window_minutes)
    stats["files_considered"] = len(files)

    frames = []
    for file in files:
        try:
            frames.append(pd.read_parquet(file))
        except (FileNotFoundError, OSError) as exc:
            # The collector compacts settled shards into one file per hour and
            # deletes the sources once the merged file is in place, so a shard
            # listed a moment ago can be gone by the time we open it. That is
            # normal housekeeping, not corruption: the same rows are in the
            # compacted file this same scan picked up. Counted separately so it
            # never trips the corruption threshold.
            if isinstance(exc, FileNotFoundError) or not file.exists():
                stats["files_vanished"] += 1
                continue
            stats["files_failed"] += 1
            if len(stats["errors"]) < 5:
                stats["errors"].append(f"{file.name}: {exc}")
            continue
        except Exception as exc:
            stats["files_failed"] += 1
            if len(stats["errors"]) < 5:
                stats["errors"].append(f"{file.name}: {exc}")
            continue
    stats["files_read"] = len(frames)

    considered = max(stats["files_considered"], 1)
    failure_rate = stats["files_failed"] / considered
    if stats["files_failed"] and failure_rate > MAX_SHARD_READ_FAILURE_RATE:
        raise ShardReadError(
            f"{stats['files_failed']}/{stats['files_considered']} shards unreadable in "
            f"{directory} (>{MAX_SHARD_READ_FAILURE_RATE:.0%}): {'; '.join(stats['errors'])}"
        )

    if not frames:
        return pd.DataFrame(), stats
    return pd.concat(frames, ignore_index=True), stats


def _exchange_ts_coverage(frame: pd.DataFrame) -> float:
    if frame.empty or "exchange_timestamp" not in frame.columns:
        return 0.0
    series = pd.to_numeric(frame["exchange_timestamp"], errors="coerce")
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    positive = finite[finite > 0]
    return float(len(positive)) / float(len(frame))


def _ts_ms_from(frame: pd.DataFrame, source: str) -> pd.Series:
    """Millisecond timestamps (float64) from the chosen clock ('exchange' or 'local')."""
    if source == "exchange":
        return pd.to_numeric(frame["exchange_timestamp"], errors="coerce").astype(float)
    return pd.to_numeric(frame["timestamp"], errors="coerce").astype(float) * 1000.0


def choose_time_source(*frames: pd.DataFrame) -> str:
    """Use exchange time only when every stream has near-full coverage."""
    for frame in frames:
        if _exchange_ts_coverage(frame) < _EXCHANGE_TS_MIN_COVERAGE:
            return "local"
    return "exchange"


def build_bbo_mid(prices: pd.DataFrame, ts_source: str) -> pd.DataFrame:
    """Mid series (ts_ms, mid, bid, ask) from the one-row-per-side BBO stream."""
    if prices.empty or not {"timestamp", "price", "side"}.issubset(prices.columns):
        return pd.DataFrame(columns=["ts_ms", "bid", "ask", "mid"])
    frame = prices.copy()
    frame["ts_ms"] = _ts_ms_from(frame, ts_source)
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame[frame["side"].isin({"bid", "ask"})]
    frame = frame.dropna(subset=["ts_ms", "price"])
    if frame.empty:
        return pd.DataFrame(columns=["ts_ms", "bid", "ask", "mid"])
    pivot = frame.pivot_table(index="ts_ms", columns="side", values="price", aggfunc="last")
    bbo = pivot.sort_index().ffill().dropna(subset=[c for c in ("bid", "ask") if c in pivot.columns])
    if "bid" not in bbo.columns or "ask" not in bbo.columns:
        return pd.DataFrame(columns=["ts_ms", "bid", "ask", "mid"])
    bbo = bbo[(bbo["bid"] > 0) & (bbo["ask"] > 0) & (bbo["bid"] < bbo["ask"])]
    bbo = bbo.reset_index()
    bbo["mid"] = (bbo["bid"] + bbo["ask"]) / 2.0
    return bbo[["ts_ms", "bid", "ask", "mid"]]


def build_orderbook_mid(orderbooks: pd.DataFrame, ts_source: str) -> pd.DataFrame:
    """Fallback mid series from top-of-book orderbook snapshots."""
    needed = {"timestamp", "bid_price_0", "ask_price_0"}
    if orderbooks.empty or not needed.issubset(orderbooks.columns):
        return pd.DataFrame(columns=["ts_ms", "bid", "ask", "mid"])
    frame = orderbooks.copy()
    frame["ts_ms"] = _ts_ms_from(frame, ts_source)
    frame = frame.dropna(subset=["ts_ms", "bid_price_0", "ask_price_0"])
    frame = frame[(frame["bid_price_0"] > 0) & (frame["ask_price_0"] > 0) & (frame["bid_price_0"] < frame["ask_price_0"])]
    if frame.empty:
        return pd.DataFrame(columns=["ts_ms", "bid", "ask", "mid"])
    out = pd.DataFrame(
        {
            "ts_ms": frame["ts_ms"].astype(float),
            "bid": frame["bid_price_0"].astype(float),
            "ask": frame["ask_price_0"].astype(float),
        }
    ).sort_values("ts_ms")
    out = out.drop_duplicates(subset="ts_ms", keep="last").reset_index(drop=True)
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out


# Placeholders the collector can write into trade_id when the feed message
# carried no id: it stores str(trade.get("tid")), so a missing id lands as the
# literal "None". Those rows are DISTINCT trades that happen to share a
# placeholder, so they must never be collapsed into one.
_MISSING_TRADE_ID_TOKENS = frozenset({"", "None", "nan", "NaT", "<NA>", "none", "null"})


def duplicate_trade_id_mask(trades: pd.DataFrame) -> pd.Series | None:
    """Boolean mask of rows that repeat an already-seen trade_id, or None.

    Returns None when the frame carries no usable id column at all, so callers
    can tell "nothing duplicated" apart from "could not check".
    """
    if trades.empty or "trade_id" not in trades.columns:
        return None
    tid = trades["trade_id"].astype(str)
    has_id = ~tid.isin(_MISSING_TRADE_ID_TOKENS)
    return has_id & tid.duplicated(keep="first")


def normalize_trades(trades: pd.DataFrame, ts_source: str) -> pd.DataFrame:
    """Trades as (ts_ms, side, price, size), side lowercased buy/sell."""
    needed = {"timestamp", "price", "side"}
    if trades.empty or not needed.issubset(trades.columns):
        return pd.DataFrame(columns=["ts_ms", "side", "price", "size"])
    frame = trades.copy()
    # De-duplicate on the exchange's own trade id before anything counts rows.
    # _load_parquet_dir() concatenates every shard in the directory blindly, so
    # if a second collector is ever pointed at the same output dir both record
    # the same public feed under different shard names and every trade is
    # counted twice — silently doubling n_trades and lambda+/-. That happened on
    # 2026-08-16 (hl-collector + hl-collector2 sharing scripts/HL_data). tid is
    # unique per trade, so this is exact, and a no-op for a correct single-writer
    # directory. Only local receive `timestamp` differs between the two copies,
    # which is why nothing upstream catches it. The drop count is surfaced in
    # MarketWindow.meta["duplicate_trade_ids_dropped"]: the failure mode was
    # silent, so it should never be silent again even once it is corrected for.
    duplicates = duplicate_trade_id_mask(frame)
    if duplicates is not None and bool(duplicates.any()):
        frame = frame[~duplicates]
    frame["ts_ms"] = _ts_ms_from(frame, ts_source)
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame[frame["side"].isin({"buy", "sell"})]
    frame = frame.dropna(subset=["ts_ms", "price"])
    if "size" not in frame.columns:
        frame["size"] = 0.0
    out = frame[["ts_ms", "side", "price", "size"]].sort_values("ts_ms").reset_index(drop=True)
    return out


@dataclass
class MarketWindow:
    mids: pd.DataFrame
    trades: pd.DataFrame
    mid_source: str  # "prices" | "orderbooks_fallback"
    ts_source: str  # "exchange" | "local"
    window_start_ms: float | None = None
    window_end_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def window_seconds(self) -> float:
        """Wall-clock span. NOT the right denominator for an arrival rate."""
        if self.window_start_ms is None or self.window_end_ms is None:
            return 0.0
        return max(0.0, (self.window_end_ms - self.window_start_ms) / 1000.0)

    @property
    def observed_seconds(self) -> float:
        """Wall clock minus time the collector was down -- the rate denominator.

        lambda is a count divided by the time we were WATCHING. Dividing by wall
        clock understates it by exactly the missing fraction, which was 6.6% on
        the window holding the 7.9 min outage and would be 50% inside the 1 h
        gap the acceptance gate now tolerates.
        """
        return max(0.0, self.window_seconds - float((self.meta or {}).get("outage_seconds", 0.0)))

    def window_start_iso(self) -> str | None:
        return _ms_to_iso(self.window_start_ms)

    def window_end_iso(self) -> str | None:
        return _ms_to_iso(self.window_end_ms)


def _ms_to_iso(ts_ms: float | None) -> str | None:
    if ts_ms is None or not np.isfinite(ts_ms):
        return None
    try:
        dt = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        return dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def load_market_window(
    crypto: str,
    minutes: int,
    data_dir: str | Path | None = None,
    window_start: Any = None,
    window_end: Any = None,
) -> MarketWindow:
    """Load mid series + trades for one window on one shared clock.

    Default (both bounds None, what the live estimator uses): the window ends at
    the earlier of the two streams' latest timestamps (so both streams cover it)
    and spans ``minutes`` back from there.

    ``window_start`` / ``window_end`` (ISO-8601 or epoch) pin the window to a
    historical slice instead, so a calibration sweep can fit on a train slice and
    score on a held-out one. Bounds EXTEND the trailing selection rather than
    replacing it: ``minutes`` still sets the span when only one bound is given,
    and an explicit end is still clamped to what the data actually covers, so a
    slice can never claim more history than exists.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "HL_data"
    base = Path(data_dir) / crypto

    requested_start_ms = parse_window_bound_ms(window_start)
    requested_end_ms = parse_window_bound_ms(window_end)
    if (
        requested_start_ms is not None
        and requested_end_ms is not None
        and requested_start_ms >= requested_end_ms
    ):
        raise ValueError(
            f"window_start ({window_start}) must be strictly before window_end ({window_end})"
        )

    # Shard pre-selection. With no bounds this is the trailing-window rule that
    # keeps a live cycle cheap. With bounds it becomes absolute, because the
    # trailing rule is anchored on the NEWEST shard and would discard the whole
    # requested slice.
    shard_bounds: tuple[float | None, float | None] | None = None
    if requested_start_ms is not None or requested_end_ms is not None:
        shard_bounds = (requested_start_ms, requested_end_ms)

    shard_stats: dict[str, Any] = {}
    prices_raw, shard_stats["prices"] = _load_parquet_dir(base / "prices", minutes, shard_bounds)
    trades_raw, shard_stats["trades"] = _load_parquet_dir(base / "trades", minutes, shard_bounds)
    orderbooks_raw = pd.DataFrame()

    mid_source = "prices"
    if prices_raw.empty:
        orderbooks_raw, shard_stats["orderbooks"] = _load_parquet_dir(
            base / "orderbooks", minutes, shard_bounds
        )
        mid_source = "orderbooks_fallback"
        if orderbooks_raw.empty:
            raise FileNotFoundError(f"No prices or orderbooks data under {base}")

    if mid_source == "prices":
        ts_source = choose_time_source(prices_raw, trades_raw) if not trades_raw.empty else choose_time_source(prices_raw)
        mids = build_bbo_mid(prices_raw, ts_source)
        if mids.empty:
            orderbooks_raw, shard_stats["orderbooks"] = _load_parquet_dir(
                base / "orderbooks", minutes, shard_bounds
            )
            if not orderbooks_raw.empty:
                mid_source = "orderbooks_fallback"
    if mid_source == "orderbooks_fallback":
        ts_source = choose_time_source(orderbooks_raw, trades_raw) if not trades_raw.empty else choose_time_source(orderbooks_raw)
        mids = build_orderbook_mid(orderbooks_raw, ts_source)

    if mids.empty:
        raise ValueError(f"No usable mid data for {crypto} under {base}")

    duplicate_trade_ids = duplicate_trade_id_mask(trades_raw)
    n_duplicate_trade_ids = int(duplicate_trade_ids.sum()) if duplicate_trade_ids is not None else 0
    trades = normalize_trades(trades_raw, ts_source)

    mid_end = float(mids["ts_ms"].max())
    if not trades.empty:
        data_end = min(mid_end, float(trades["ts_ms"].max()))
    else:
        data_end = mid_end
    # An explicit end is clamped to the data: a slice must never report coverage
    # it does not have, or lambda (count / window seconds) is understated.
    window_end_ms = min(data_end, requested_end_ms) if requested_end_ms is not None else data_end
    if requested_start_ms is not None:
        window_start = requested_start_ms
    else:
        window_start = window_end_ms - float(minutes) * 60_000.0
    window_end = window_end_ms

    mids = mids[(mids["ts_ms"] >= window_start) & (mids["ts_ms"] <= window_end)].reset_index(drop=True)
    if not trades.empty:
        trades = trades[(trades["ts_ms"] >= window_start) & (trades["ts_ms"] <= window_end)].reset_index(drop=True)

    # Interior collector downtime, asked of both streams at once -- see
    # outage_seconds() for why the mid stream alone cannot answer it.
    n_outages, outage_total = outage_seconds(
        [
            mids["ts_ms"].to_numpy(dtype=float) if not mids.empty else np.array([]),
            trades["ts_ms"].to_numpy(dtype=float) if not trades.empty else np.array([]),
        ],
        window_start_ms=window_start,
        window_end_ms=window_end,
    )

    return MarketWindow(
        mids=mids,
        trades=trades,
        mid_source=mid_source,
        ts_source=ts_source,
        window_start_ms=window_start,
        window_end_ms=window_end,
        meta={
            "n_mid_updates": int(len(mids)),
            "n_trade_prints": int(len(trades)),
            "duplicate_trade_ids_dropped": n_duplicate_trade_ids,
            "outage_seconds": outage_total,
            "n_outages": n_outages,
            "outage_threshold_seconds": float(DEFAULT_OUTAGE_THRESHOLD_SECONDS),
            "requested_window_start_ms": requested_start_ms,
            "requested_window_end_ms": requested_end_ms,
            "shard_stats": shard_stats,
            "shards_read": sum(int(s.get("files_read", 0)) for s in shard_stats.values()),
            "shards_failed": sum(int(s.get("files_failed", 0)) for s in shard_stats.values()),
            "shards_skipped_outside_window": sum(
                int(s.get("files_present", 0)) - int(s.get("files_considered", 0))
                for s in shard_stats.values()
            ),
        },
    )


# --------------------------------------------------------------------------
# Emit mode (calibration sweeps)
# --------------------------------------------------------------------------
# A sweep needs parameter sets computed at many different calibration settings.
# It must NOT reach the live snapshots: user_data/strategies/{kappa,epsilon,
# lambda}.json are read by two running freqtrade legs, and scripts/*.json are
# what periodic_test_runner copies there. Emit mode therefore writes ONE file at
# a caller-chosen path and performs no other write at all — it also never takes
# param_update.lock, because the estimator scripts never take it in the first
# place (the runner owns it), so a sweep cannot stall quoting by holding it.
#
# The emitted blocks are byte-for-byte the entries that WOULD have been written
# to the live files, so anything that can read a snapshot can read a sweep
# result, plus a "calibration" block recording the settings that produced them.
EMIT_PAYLOAD_KIND = "estimator_params_emit"


def build_emit_window_block(window: "MarketWindow", minutes: int | float | None) -> dict[str, Any]:
    """Provenance for an emitted parameter set: which slice produced it."""
    meta = dict(window.meta or {})
    return {
        "start": window.window_start_iso(),
        "end": window.window_end_iso(),
        "seconds": window.window_seconds,
        "observed_seconds": window.observed_seconds,
        "outage_seconds": meta.get("outage_seconds"),
        "n_outages": meta.get("n_outages"),
        "minutes_requested": float(minutes) if minutes is not None else None,
        "requested_start_ms": meta.get("requested_window_start_ms"),
        "requested_end_ms": meta.get("requested_window_end_ms"),
        "mid_source": window.mid_source,
        "ts_source": window.ts_source,
        "n_mid_updates": int(len(window.mids)),
        "n_trade_prints": int(len(window.trades)),
        "duplicate_trade_ids_dropped": meta.get("duplicate_trade_ids_dropped"),
    }


def write_emit_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one emitted parameter set. Never touches the live snapshots."""
    out = Path(path)
    atomic_write_json(out, payload)
    return out


def aggregate_market_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse prints sharing (side, ts_ms) into one market order.

    One MO sweeping several levels emits several prints with the same exchange
    timestamp; counting them individually overweights shallow depths and
    overstates the arrival rate. Price extreme = deepest print (max for buys,
    min for sells).
    """
    if trades.empty:
        return pd.DataFrame(columns=["ts_ms", "side", "price_extreme", "size", "n_prints"])
    grouped = trades.groupby(["ts_ms", "side"], sort=True)
    mos = grouped.agg(
        price_max=("price", "max"),
        price_min=("price", "min"),
        size=("size", "sum"),
        n_prints=("price", "size"),
    ).reset_index()
    mos["price_extreme"] = np.where(mos["side"] == "buy", mos["price_max"], mos["price_min"])
    return mos[["ts_ms", "side", "price_extreme", "size", "n_prints"]].sort_values("ts_ms").reset_index(drop=True)


def attach_pre_mid(mos: pd.DataFrame, mids: pd.DataFrame) -> pd.DataFrame:
    """Attach the last mid strictly BEFORE each MO (no stale-lookback window)."""
    if mos.empty or mids.empty:
        out = mos.copy()
        out["pre_mid"] = np.nan
        return out
    left = mos.sort_values("ts_ms").reset_index(drop=True)
    right = mids[["ts_ms", "mid"]].sort_values("ts_ms").rename(columns={"mid": "pre_mid"})
    merged = pd.merge_asof(
        left,
        right,
        on="ts_ms",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged


def mo_depths(mos_with_mid: pd.DataFrame) -> dict[str, Any]:
    """Mid-relative depths per side; negatives truncated to 0 (counted)."""
    result: dict[str, Any] = {
        "buy_depths": np.array([]),
        "sell_depths": np.array([]),
        "skipped_no_pre_mid": 0,
        "negative_depth_truncated_buy": 0,
        "negative_depth_truncated_sell": 0,
    }
    if mos_with_mid.empty:
        return result
    frame = mos_with_mid.copy()
    missing = frame["pre_mid"].isna()
    result["skipped_no_pre_mid"] = int(missing.sum())
    frame = frame[~missing]
    if frame.empty:
        return result

    buys = frame[frame["side"] == "buy"]
    sells = frame[frame["side"] == "sell"]
    raw_buy = (buys["price_extreme"] - buys["pre_mid"]).to_numpy(dtype=float)
    raw_sell = (sells["pre_mid"] - sells["price_extreme"]).to_numpy(dtype=float)
    result["negative_depth_truncated_buy"] = int(np.sum(raw_buy < 0))
    result["negative_depth_truncated_sell"] = int(np.sum(raw_sell < 0))
    result["buy_depths"] = np.maximum(raw_buy, 0.0)
    result["sell_depths"] = np.maximum(raw_sell, 0.0)
    return result


# Default fit support: everything up to the 99th percentile of that side's
# depths. DEFAULT_SUPPORT_QUANTILE_LOWER = 0.0 means "no lower bound", which is
# the fit that shipped before the bound existed — see fit_kappa_survival.
DEFAULT_SUPPORT_QUANTILE_LOWER = 0.0
DEFAULT_SUPPORT_QUANTILE_UPPER = 0.99


def fit_kappa_survival(
    depths: np.ndarray,
    support_quantile: float = DEFAULT_SUPPORT_QUANTILE_UPPER,
    support_quantile_lower: float = DEFAULT_SUPPORT_QUANTILE_LOWER,
) -> dict[str, Any]:
    """Fit kappa from the empirical survival function of MO depths.

    Model: a resting order at depth delta fills when an MO walks at least that
    deep, so its fill intensity is Lambda * P(depth >= delta). With exponential
    depths, log S(delta) = -kappa * delta. Weighted LS with weights
    sqrt(tail count) approximates the binomial precision of each survival
    estimate.

    The fit runs over the depth support [Q(support_quantile_lower),
    Q(support_quantile)] of that side's own distribution. The upper bound has
    always been there (it drops the fat right tail that a handful of sweeps
    would otherwise dominate). The LOWER bound exists because the depth
    distribution is dominated by shallow market orders, and the ask/bid depths
    where CASHCAT's measured edge is actually positive live far out in the tail
    (docs/market_viability_report.json: curve_plus only turns positive past ~14
    bps). Fitting one exponential across the whole distribution therefore sets
    the slope from the region that pays nothing, and 1/kappa comes out too tight
    for the region that pays. Raising the lower bound refits the same model over
    only the depths that matter.

    The survival values themselves are ALWAYS the full-sample tail fractions
    P(depth >= delta): the bounds select which points to regress, not which
    trades exist. log S is linear in delta over any sub-range, so restricting
    the support leaves kappa on the same scale; only the extrapolated intercept
    (survival_intercept) is more of an extrapolation than it was.

    Defaults reproduce the pre-bound fit exactly: with support_quantile_lower =
    0.0 the lower mask is not applied at all.
    """
    support_quantile_upper = float(support_quantile)
    support_quantile_lower = float(support_quantile_lower)
    if not 0.0 <= support_quantile_lower < support_quantile_upper <= 1.0:
        raise ValueError(
            "kappa fit support must satisfy 0 <= lower < upper <= 1, got "
            f"[{support_quantile_lower}, {support_quantile_upper}]"
        )

    def _empty(depth_lower: float = float("nan"), depth_upper: float = float("nan")) -> dict[str, Any]:
        return {
            "kappa": float("nan"),
            "r_squared": float("nan"),
            "n_points": 0,
            "depth_p95": float("nan"),
            "depth_max_fitted": float("nan"),
            "survival_intercept": float("nan"),
            "depth_min_fitted": float("nan"),
            "support_quantile_lower": support_quantile_lower,
            "support_quantile_upper": support_quantile_upper,
            "support_depth_lower": depth_lower,
            "support_depth_upper": depth_upper,
        }

    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths)]
    n_total = len(depths)
    if n_total < 10:
        return _empty()

    depth_p95 = float(np.percentile(depths, 95))
    support_cap = float(np.quantile(depths, support_quantile_upper))
    sorted_depths = np.sort(depths)

    in_support = sorted_depths <= support_cap
    support_floor = float("nan")
    if support_quantile_lower > 0.0:
        support_floor = float(np.quantile(depths, support_quantile_lower))
        in_support &= sorted_depths >= support_floor

    grid = np.unique(sorted_depths[in_support])
    if len(grid) < 3:
        return _empty(support_floor, support_cap)

    # Tail counts via searchsorted: count of depths >= delta.
    tail_counts = n_total - np.searchsorted(sorted_depths, grid, side="left")
    survival = tail_counts.astype(float) / float(n_total)
    mask = (survival > 0) & (tail_counts >= 2)
    grid = grid[mask]
    survival = survival[mask]
    tail_counts = tail_counts[mask]
    if len(grid) < 3:
        return _empty(support_floor, support_cap)

    y = np.log(survival)
    weights = np.sqrt(tail_counts.astype(float))
    design = np.column_stack((np.ones_like(grid), -grid))
    try:
        coef, _, _, _ = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)
    except np.linalg.LinAlgError:
        return _empty(support_floor, support_cap)
    intercept, kappa = float(coef[0]), float(coef[1])

    y_pred = intercept - kappa * grid
    ss_res = float(np.sum((weights * (y - y_pred)) ** 2))
    y_mean = float(np.average(y, weights=weights**2))
    ss_tot = float(np.sum((weights * (y - y_mean)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "kappa": max(kappa, 0.0),
        "r_squared": r_squared,
        "n_points": int(len(grid)),
        "depth_p95": depth_p95,
        "depth_max_fitted": float(grid.max()),
        "survival_intercept": float(np.exp(intercept)),
        "depth_min_fitted": float(grid.min()),
        "support_quantile_lower": support_quantile_lower,
        "support_quantile_upper": support_quantile_upper,
        "support_depth_lower": support_floor,
        "support_depth_upper": support_cap,
    }


# A collector outage and a quiet market look identical on the price stream
# alone: CASHCAT's BBO simply does not move for minutes at a time. Measured over
# 61.2 h to 2026-08-19, the price stream had 91.5 min of gaps > 60 s, but only
# 43.2 min of those were the collector actually being down -- the other 48.3 min
# had trades or book updates arriving normally. Treating all 91.5 min as missing
# would over-correct lambda by more than the bug it fixes, so "was the collector
# alive" is asked of the UNION of the streams, not of the mids.
#
# prices+trades is used rather than all three because both are already loaded on
# the live path, and the estimator container re-runs every 30 s. Adding the
# orderbook stream would tighten the estimate from 42.4 to 36.5 min over that
# same 61.2 h -- 0.16% of the span -- which does not pay for the extra I/O.
DEFAULT_OUTAGE_THRESHOLD_SECONDS = 60.0


def outage_seconds(
    streams: "list[np.ndarray] | tuple[np.ndarray, ...]",
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
    threshold_seconds: float = DEFAULT_OUTAGE_THRESHOLD_SECONDS,
) -> tuple[int, float]:
    """(count, seconds) of interior stretches where EVERY stream went silent.

    Only interior stretches count. A window that merely starts late is handled
    by the caller trimming its start; extending this to the edges would charge
    the same missing time twice.
    """
    stamps = [np.asarray(a, dtype=float).ravel() for a in streams if a is not None and len(a)]
    if not stamps:
        return 0, 0.0
    alive = np.sort(np.concatenate(stamps))
    if window_start_ms is not None:
        alive = alive[alive >= float(window_start_ms)]
    if window_end_ms is not None:
        alive = alive[alive <= float(window_end_ms)]
    if len(alive) < 2:
        return 0, 0.0
    deltas = np.diff(alive) / 1000.0
    over = deltas > float(threshold_seconds)
    return int(over.sum()), float(deltas[over].sum())


def realized_sigma2_per_sec(
    mids: pd.DataFrame,
    sample_seconds: float = 1.0,
    max_gap_seconds: float = 5.0,
    min_samples: int = 60,
) -> float | None:
    """Realized mid variance in USDC^2/sec from sample_seconds-spaced increments.

    Increments spanning data gaps longer than max_gap_seconds are dropped so a
    collector outage does not masquerade as volatility.
    """
    if mids.empty or "ts_ms" not in mids.columns or "mid" not in mids.columns:
        return None
    frame = mids[["ts_ms", "mid"]].dropna().sort_values("ts_ms")
    if frame.empty:
        return None
    ts_sec = frame["ts_ms"].to_numpy(dtype=float) / 1000.0
    mid = frame["mid"].to_numpy(dtype=float)

    step = max(float(sample_seconds), 1e-3)
    grid = np.arange(math.ceil(ts_sec[0] / step) * step, ts_sec[-1] + 1e-9, step)
    if len(grid) < min_samples + 1:
        return None
    idx = np.searchsorted(ts_sec, grid, side="right") - 1
    valid = idx >= 0
    # Drop grid points whose last observation is older than max_gap_seconds.
    staleness = grid[valid] - ts_sec[idx[valid]]
    fresh = np.zeros_like(valid)
    fresh[valid] = staleness <= max_gap_seconds
    sampled = np.where(fresh, mid[np.clip(idx, 0, None)], np.nan)

    diffs = np.diff(sampled)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < min_samples:
        return None
    sigma2 = float(np.var(diffs)) / step
    if not np.isfinite(sigma2) or sigma2 < 0:
        return None
    return sigma2
