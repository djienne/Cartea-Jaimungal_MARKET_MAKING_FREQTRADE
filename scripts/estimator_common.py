"""Shared data pipeline for the kappa/lambda/epsilon estimators.

Methodology (schema v3):
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
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from param_utils import PARAM_SCHEMA_VERSION


# Validation floors shared with the strategy (Market_Making.py mirrors these as
# class attributes min_kappa_fit_points / min_kappa_r2 / min_epsilon_events).
MIN_KAPPA_FIT_POINTS = 6
MIN_KAPPA_R2 = 0.30
MIN_EPSILON_EVENTS = 50

DEFAULT_EMA_TAU_SECONDS = 300.0
EMA_TAU_ENV_VAR = "PARAM_EMA_TAU_SECONDS"

# Fraction of rows that must carry a finite exchange_timestamp before a stream
# is trusted to be on exchange time.
_EXCHANGE_TS_MIN_COVERAGE = 0.99


def resolve_ema_tau(cli_value: float | None = None) -> float:
    """CLI flag > PARAM_EMA_TAU_SECONDS env > default. tau <= 0 disables."""
    if cli_value is not None:
        try:
            return float(cli_value)
        except (TypeError, ValueError):
            pass
    env_value = os.getenv(EMA_TAU_ENV_VAR)
    if env_value is not None:
        try:
            return float(env_value)
        except (TypeError, ValueError):
            pass
    return DEFAULT_EMA_TAU_SECONDS


def _load_parquet_dir(directory: Path) -> pd.DataFrame:
    if not directory.is_dir():
        return pd.DataFrame()
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for file in files:
        try:
            frames.append(pd.read_parquet(file))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    # which is why nothing upstream catches it.
    if "trade_id" in frame.columns:
        tid = frame["trade_id"].astype(str)
        # The collector stores str(trade.get("tid")), so a feed message with no
        # id lands as the literal "None". Those rows are distinct trades and
        # must never be collapsed into one, so only de-duplicate real ids.
        has_id = ~tid.isin({"", "None", "nan", "NaT", "<NA>"})
        frame = frame[~(has_id & tid.duplicated(keep="first"))]
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
        if self.window_start_ms is None or self.window_end_ms is None:
            return 0.0
        return max(0.0, (self.window_end_ms - self.window_start_ms) / 1000.0)

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


def load_market_window(crypto: str, minutes: int, data_dir: str | Path | None = None) -> MarketWindow:
    """Load mid series + trades for the trailing window on one shared clock.

    The window ends at the earlier of the two streams' latest timestamps (so
    both streams cover it) and spans ``minutes`` back from there.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "HL_data"
    base = Path(data_dir) / crypto

    prices_raw = _load_parquet_dir(base / "prices")
    trades_raw = _load_parquet_dir(base / "trades")
    orderbooks_raw = pd.DataFrame()

    mid_source = "prices"
    if prices_raw.empty:
        orderbooks_raw = _load_parquet_dir(base / "orderbooks")
        mid_source = "orderbooks_fallback"
        if orderbooks_raw.empty:
            raise FileNotFoundError(f"No prices or orderbooks data under {base}")

    if mid_source == "prices":
        ts_source = choose_time_source(prices_raw, trades_raw) if not trades_raw.empty else choose_time_source(prices_raw)
        mids = build_bbo_mid(prices_raw, ts_source)
        if mids.empty:
            orderbooks_raw = _load_parquet_dir(base / "orderbooks")
            if not orderbooks_raw.empty:
                mid_source = "orderbooks_fallback"
    if mid_source == "orderbooks_fallback":
        ts_source = choose_time_source(orderbooks_raw, trades_raw) if not trades_raw.empty else choose_time_source(orderbooks_raw)
        mids = build_orderbook_mid(orderbooks_raw, ts_source)

    if mids.empty:
        raise ValueError(f"No usable mid data for {crypto} under {base}")

    trades = normalize_trades(trades_raw, ts_source)

    mid_end = float(mids["ts_ms"].max())
    if not trades.empty:
        window_end = min(mid_end, float(trades["ts_ms"].max()))
    else:
        window_end = mid_end
    window_start = window_end - float(minutes) * 60_000.0

    mids = mids[(mids["ts_ms"] >= window_start) & (mids["ts_ms"] <= window_end)].reset_index(drop=True)
    if not trades.empty:
        trades = trades[(trades["ts_ms"] >= window_start) & (trades["ts_ms"] <= window_end)].reset_index(drop=True)

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
        },
    )


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


def fit_kappa_survival(depths: np.ndarray, support_quantile: float = 0.99) -> dict[str, Any]:
    """Fit kappa from the empirical survival function of MO depths.

    Model: a resting order at depth delta fills when an MO walks at least that
    deep, so its fill intensity is Lambda * P(depth >= delta). With exponential
    depths, log S(delta) = -kappa * delta. Weighted LS with weights
    sqrt(tail count) approximates the binomial precision of each survival
    estimate.
    """
    empty = {
        "kappa": float("nan"),
        "r_squared": float("nan"),
        "n_points": 0,
        "depth_p95": float("nan"),
        "depth_max_fitted": float("nan"),
        "survival_intercept": float("nan"),
    }
    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths)]
    n_total = len(depths)
    if n_total < 10:
        return empty

    depth_p95 = float(np.percentile(depths, 95))
    support_cap = float(np.quantile(depths, support_quantile))
    sorted_depths = np.sort(depths)

    grid = np.unique(sorted_depths[sorted_depths <= support_cap])
    if len(grid) < 3:
        return empty

    # Tail counts via searchsorted: count of depths >= delta.
    tail_counts = n_total - np.searchsorted(sorted_depths, grid, side="left")
    survival = tail_counts.astype(float) / float(n_total)
    mask = (survival > 0) & (tail_counts >= 2)
    grid = grid[mask]
    survival = survival[mask]
    tail_counts = tail_counts[mask]
    if len(grid) < 3:
        return empty

    y = np.log(survival)
    weights = np.sqrt(tail_counts.astype(float))
    design = np.column_stack((np.ones_like(grid), -grid))
    try:
        coef, _, _, _ = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)
    except np.linalg.LinAlgError:
        return empty
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
    }


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


def _parse_generated_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ema_update(
    raw: float | None,
    prev_entry: dict[str, Any] | None,
    key: str,
    tau_seconds: float,
    now: datetime | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Time-aware EMA of one parameter against the previous snapshot entry.

    Seeds from raw (no blending) when: tau disabled, no/invalid previous entry,
    previous status != ok, previous schema_version != current (NEVER blend
    across the v2->v3 semantic change), non-finite previous value, or
    unparsable previous generated_at.
    """
    meta: dict[str, Any] = {
        "ema_tau_seconds": float(tau_seconds),
        "ema_seeded": True,
        "ema_alpha": None,
    }
    if raw is None or not np.isfinite(float(raw)):
        return None, meta
    raw = float(raw)

    if tau_seconds <= 0:
        return raw, meta
    if not isinstance(prev_entry, dict):
        return raw, meta
    if prev_entry.get("status") != "ok":
        return raw, meta
    if prev_entry.get("schema_version") != PARAM_SCHEMA_VERSION:
        return raw, meta
    try:
        prev_value = float(prev_entry.get(key))
    except (TypeError, ValueError):
        return raw, meta
    if not np.isfinite(prev_value):
        return raw, meta
    prev_dt = _parse_generated_at(prev_entry.get("generated_at"))
    if prev_dt is None:
        return raw, meta

    if now is None:
        now = datetime.now(timezone.utc)
    delta_t = max(0.0, (now - prev_dt).total_seconds())
    alpha = 1.0 - math.exp(-delta_t / float(tau_seconds))
    smoothed = alpha * raw + (1.0 - alpha) * prev_value
    meta["ema_seeded"] = False
    meta["ema_alpha"] = alpha
    return float(smoothed), meta
