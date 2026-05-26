#!/usr/bin/env python3
"""Event replay harness for the market-making model.

This is intentionally conservative. It is not a candle backtest: quotes that
cross the book are post-only rejects, quotes away from touch wait for the book to
move through them, and quotes at touch need observed traded volume before fill.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hjb import compute_h_asymmetric


MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
MARKOUT_HORIZONS_MS = (100, 1_000, 5_000, 30_000)
PARAMETER_SERIES_UNIT = {
    "kappa": "1/USDC",
    "lambda": "events/second",
    "epsilon": "USDC",
}


@dataclass
class ReplayConfig:
    symbol: str
    data_dir: Path
    mid_fallback: float
    inventory_unit_base: float = 0.01
    q_max: int = 3
    decision_latency_ms: int = 250
    order_ack_latency_ms: int = 250
    cancel_latency_ms: int = 250
    maker_fee: float = MAKER_FEE
    taker_fee: float = TAKER_FEE
    funding_rate_per_hour: float = 0.0
    starting_equity_usdc: float = 1000.0
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.05
    queue_decay_per_second: float = 0.05
    newest_per_stream: int | None = None
    max_price_events: int | None = None


@dataclass
class ReplayMetrics:
    input_files: dict[str, int] = field(default_factory=dict)
    input_rows: dict[str, int] = field(default_factory=dict)
    data_start: str | None = None
    data_end: str | None = None
    data_span_seconds: float = 0.0
    price_event_count: int = 0
    price_events_per_day: float = 0.0
    max_price_gap_seconds: float | None = None
    p95_price_gap_seconds: float | None = None
    quote_attempts: int = 0
    post_only_rejects: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    stale_quote_cancels: int = 0
    realized_spread_usdc: float = 0.0
    fees_usdc: float = 0.0
    cash_usdc: float = 0.0
    inventory_base: float = 0.0
    final_mid: float | None = None
    mark_to_market_pnl_usdc: float = 0.0
    funding_usdc: float = 0.0
    starting_equity_usdc: float = 0.0
    equity_usdc: float = 0.0
    min_equity_usdc: float | None = None
    notional_exposure_usdc: float = 0.0
    margin_used_usdc: float = 0.0
    max_margin_used_usdc: float = 0.0
    maintenance_margin_usdc: float = 0.0
    min_liquidation_buffer_usdc: float | None = None
    liquidation_breach_events: int = 0
    queue_decay_base: float = 0.0
    time_at_q_boundary: dict[str, int] = field(default_factory=lambda: {"q_min": 0, "q_max": 0})
    inventory_histogram: dict[int, int] = field(default_factory=dict)
    pnl_by_side: dict[str, float] = field(default_factory=lambda: {"bid": 0.0, "ask": 0.0})
    quote_attempts_by_depth: dict[str, int] = field(default_factory=dict)
    fills_by_depth: dict[str, int] = field(default_factory=dict)
    markout_samples: list[dict[str, Any]] = field(default_factory=list)
    parameter_series: list[dict[str, Any]] = field(default_factory=list)
    toxicity_series: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        attempts = max(self.quote_attempts, 1)
        fills = self.maker_fills + self.taker_fills
        fill_ratio_by_depth = {
            key: self.fills_by_depth.get(key, 0) / max(attempts_at_depth, 1)
            for key, attempts_at_depth in sorted(self.quote_attempts_by_depth.items())
        }
        return {
            "input_files": self.input_files,
            "input_rows": self.input_rows,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "data_span_seconds": self.data_span_seconds,
            "price_event_count": self.price_event_count,
            "price_events_per_day": self.price_events_per_day,
            "max_price_gap_seconds": self.max_price_gap_seconds,
            "p95_price_gap_seconds": self.p95_price_gap_seconds,
            "quote_attempts": self.quote_attempts,
            "post_only_rejects": self.post_only_rejects,
            "post_only_reject_ratio": self.post_only_rejects / attempts,
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
            "maker_ratio": self.maker_fills / max(fills, 1),
            "stale_quote_cancels": self.stale_quote_cancels,
            "stale_quote_cancel_ratio": self.stale_quote_cancels / attempts,
            "realized_spread_usdc": self.realized_spread_usdc,
            "fees_usdc": self.fees_usdc,
            "cash_usdc": self.cash_usdc,
            "inventory_base": self.inventory_base,
            "final_mid": self.final_mid,
            "mark_to_market_pnl_usdc": self.mark_to_market_pnl_usdc,
            "funding_usdc": self.funding_usdc,
            "starting_equity_usdc": self.starting_equity_usdc,
            "equity_usdc": self.equity_usdc,
            "min_equity_usdc": self.min_equity_usdc,
            "notional_exposure_usdc": self.notional_exposure_usdc,
            "margin_used_usdc": self.margin_used_usdc,
            "max_margin_used_usdc": self.max_margin_used_usdc,
            "maintenance_margin_usdc": self.maintenance_margin_usdc,
            "min_liquidation_buffer_usdc": self.min_liquidation_buffer_usdc,
            "liquidation_breach_events": self.liquidation_breach_events,
            "queue_decay_base": self.queue_decay_base,
            "time_at_q_boundary": self.time_at_q_boundary,
            "inventory_histogram": self.inventory_histogram,
            "pnl_by_side": self.pnl_by_side,
            "quote_attempts_by_depth": self.quote_attempts_by_depth,
            "fills_by_depth": self.fills_by_depth,
            "fill_ratio_by_depth": fill_ratio_by_depth,
            "markout_samples": self.markout_samples,
            "parameter_series": self.parameter_series,
            "toxicity_series": self.toxicity_series,
        }


def selected_parquet_files(path: Path, newest_per_stream: int | None = None) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if newest_per_stream is not None and newest_per_stream > 0:
        files = sorted(sorted(files, key=lambda file: file.stat().st_mtime, reverse=True)[:newest_per_stream])
    return files


def load_parquet_dir(path: Path, newest_per_stream: int | None = None) -> tuple[pd.DataFrame, int]:
    files = selected_parquet_files(path, newest_per_stream)
    if not files:
        return pd.DataFrame(), 0
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True), len(files)


def normalize_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame:
        return frame
    frame = frame.copy()
    series = frame["timestamp"]
    if pd.api.types.is_numeric_dtype(series):
        finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            frame["timestamp"] = pd.to_datetime(series, utc=True, errors="coerce")
            return frame
        median_abs = float(finite.abs().median())
        if median_abs >= 1e18:
            unit = "ns"
        elif median_abs >= 1e15:
            unit = "us"
        elif median_abs >= 1e12:
            unit = "ms"
        else:
            unit = "s"
        frame["timestamp"] = pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(series, utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"])


def normalize_price_bbo(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    if {"bid", "ask"}.issubset(prices.columns):
        return prices
    if not {"timestamp", "side", "price"}.issubset(prices.columns):
        return prices

    frame = prices.copy()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame[frame["side"].isin({"bid", "ask"})]
    if frame.empty:
        return prices

    price_pivot = frame.pivot_table(index="timestamp", columns="side", values="price", aggfunc="last")
    size_pivot = frame.pivot_table(index="timestamp", columns="side", values="size", aggfunc="last") if "size" in frame else pd.DataFrame(index=price_pivot.index)
    bbo = pd.DataFrame(index=price_pivot.index)
    if "bid" in price_pivot:
        bbo["bid"] = price_pivot["bid"]
    if "ask" in price_pivot:
        bbo["ask"] = price_pivot["ask"]
    if "bid" in size_pivot:
        bbo["bid_size"] = size_pivot["bid"]
    if "ask" in size_pivot:
        bbo["ask_size"] = size_pivot["ask"]
    bbo = bbo.sort_index().ffill().dropna(subset=["bid", "ask"]).reset_index()
    return bbo


def load_symbol_data(config: ReplayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    base = config.data_dir / config.symbol
    prices, price_files = load_parquet_dir(base / "prices", config.newest_per_stream)
    trades, trade_files = load_parquet_dir(base / "trades", config.newest_per_stream)
    orderbooks, orderbook_files = load_parquet_dir(base / "orderbooks", config.newest_per_stream)
    for frame in (prices, trades, orderbooks):
        if not frame.empty and "timestamp" in frame:
            frame = normalize_timestamp_column(frame)
            frame.sort_values("timestamp", inplace=True)
    prices = normalize_price_bbo(normalize_timestamp_column(prices))
    trades = normalize_timestamp_column(trades)
    orderbooks = normalize_timestamp_column(orderbooks)
    for frame in (prices, trades, orderbooks):
        if not frame.empty and "timestamp" in frame:
            frame.sort_values("timestamp", inplace=True)
    return prices, trades, orderbooks, {
        "prices": price_files,
        "trades": trade_files,
        "orderbooks": orderbook_files,
    }


def post_only_check(side: str, price: float, best_bid: float, best_ask: float) -> tuple[bool, str]:
    try:
        price, best_bid, best_ask = (float(price), float(best_bid), float(best_ask))
    except (TypeError, ValueError):
        return False, "crossed_or_invalid_book"
    if not all(np.isfinite(value) for value in (price, best_bid, best_ask)):
        return False, "crossed_or_invalid_book"
    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        return False, "crossed_or_invalid_book"
    if side == "bid" and price >= best_ask:
        return False, "bid_crosses_ask"
    if side == "ask" and price <= best_bid:
        return False, "ask_crosses_bid"
    return True, "ok"


def inventory_q(inventory_base: float, unit: float, q_max: int) -> int:
    q = int(round(max(0.0, inventory_base) / max(unit, 1e-12)))
    return max(0, min(q_max, q))


def compute_hjb_cache(params: dict[str, float], q_max: int) -> dict:
    return compute_h_asymmetric(
        lambda_plus=params["lambda+"],
        lambda_minus=params["lambda-"],
        epsilon_plus=params["epsilon+"],
        epsilon_minus=params["epsilon-"],
        kappa_plus=params["kappa+"],
        kappa_minus=params["kappa-"],
        alpha=params.get("alpha", 0.001),
        phi=params.get("phi", 0.0001),
        T_seconds=params.get("T_seconds", 60.0),
        q_max=q_max,
    )


def finite_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def timestamp_iso(value: Any) -> str | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.isoformat()


def replay_parameter_snapshot(
    config: ReplayConfig,
    params: dict[str, float],
    ts: Any,
    *,
    data_start: str | None,
    data_end: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ts": timestamp_iso(ts),
        "source": "static_replay_params",
        "symbol": config.symbol,
        "unit": dict(PARAMETER_SERIES_UNIT),
        "data_start": data_start,
        "data_end": data_end,
        "kappa_plus": finite_float_or_none(params.get("kappa+")),
        "kappa_minus": finite_float_or_none(params.get("kappa-")),
        "lambda_plus": finite_float_or_none(params.get("lambda+")),
        "lambda_minus": finite_float_or_none(params.get("lambda-")),
        "epsilon_plus": finite_float_or_none(params.get("epsilon+")),
        "epsilon_minus": finite_float_or_none(params.get("epsilon-")),
    }


def replay_toxicity_snapshot(
    config: ReplayConfig,
    params: dict[str, float],
    ts: Any,
    *,
    data_start: str | None,
    data_end: str | None,
) -> dict[str, Any]:
    kappa_plus = finite_float_or_none(params.get("kappa+")) or 0.0
    kappa_minus = finite_float_or_none(params.get("kappa-")) or 0.0
    epsilon_plus = finite_float_or_none(params.get("epsilon+")) or 0.0
    epsilon_minus = finite_float_or_none(params.get("epsilon-")) or 0.0
    return {
        "schema_version": 1,
        "ts": timestamp_iso(ts),
        "source": "static_replay_params",
        "symbol": config.symbol,
        "unit": "kappa_times_epsilon",
        "formula": "kappa * epsilon",
        "data_start": data_start,
        "data_end": data_end,
        "toxicity_plus": kappa_plus * epsilon_plus,
        "toxicity_minus": kappa_minus * epsilon_minus,
    }


def compute_quotes(
    mid: float,
    q: int,
    params: dict[str, float],
    q_max: int,
    hjb: dict | None = None,
    *,
    maker_fee: float = MAKER_FEE,
) -> tuple[float | None, float | None, dict]:
    if hjb is None:
        hjb = compute_hjb_cache(params, q_max)
    q_grid = hjb["q_grid"]
    idx = int(np.argmin(np.abs(q_grid - q)))
    bid_delta = hjb["delta_minus"][idx]
    ask_delta = hjb["delta_plus"][idx]
    fee_cushion = float(maker_fee) * mid * 2.0
    bid = None if not np.isfinite(bid_delta) else mid - float(bid_delta + fee_cushion)
    ask = None if not np.isfinite(ask_delta) else mid + float(ask_delta + fee_cushion)
    return bid, ask, hjb


def mid_from_price_row(row: pd.Series, fallback: float) -> tuple[float, float, float]:
    best_bid = float(row.get("bid", row.get("best_bid", np.nan)))
    best_ask = float(row.get("ask", row.get("best_ask", np.nan)))
    if not np.isfinite(best_bid) or not np.isfinite(best_ask):
        return fallback, np.nan, np.nan
    return (best_bid + best_ask) / 2.0, best_bid, best_ask


def future_mid(prices: pd.DataFrame, start_ts: pd.Timestamp, horizon_ms: int, fallback: float) -> float | None:
    target = start_ts + pd.Timedelta(milliseconds=horizon_ms)
    idx = prices["timestamp"].searchsorted(target, side="left")
    if idx >= len(prices):
        return None
    mid, _, _ = mid_from_price_row(prices.iloc[int(idx)], fallback)
    return mid


def markout_value(side: str, fill_price: float, future_mid_price: float) -> float:
    if side == "bid":
        return future_mid_price - fill_price
    return fill_price - future_mid_price


def quote_depth_key(side: str, mid: float, price: float) -> str:
    depth_bps = 0.0 if mid <= 0 else abs(float(mid) - float(price)) / float(mid) * 10_000.0
    return f"{side}:{depth_bps:.2f}bps"


def update_margin_metrics(metrics: ReplayMetrics, config: ReplayConfig, mid: float) -> None:
    notional = abs(float(metrics.inventory_base)) * float(mid)
    leverage = max(float(config.leverage), 1e-12)
    maintenance_rate = max(float(config.maintenance_margin_rate), 0.0)
    equity = float(config.starting_equity_usdc) + float(metrics.cash_usdc) + float(metrics.inventory_base) * float(mid)
    margin_used = notional / leverage
    maintenance_margin = notional * maintenance_rate
    liquidation_buffer = equity - maintenance_margin

    metrics.starting_equity_usdc = float(config.starting_equity_usdc)
    metrics.equity_usdc = equity
    metrics.min_equity_usdc = equity if metrics.min_equity_usdc is None else min(metrics.min_equity_usdc, equity)
    metrics.notional_exposure_usdc = notional
    metrics.margin_used_usdc = margin_used
    metrics.max_margin_used_usdc = max(float(metrics.max_margin_used_usdc), margin_used)
    metrics.maintenance_margin_usdc = maintenance_margin
    metrics.min_liquidation_buffer_usdc = (
        liquidation_buffer
        if metrics.min_liquidation_buffer_usdc is None
        else min(metrics.min_liquidation_buffer_usdc, liquidation_buffer)
    )
    if notional > 0 and liquidation_buffer < 0:
        metrics.liquidation_breach_events += 1


def active_orderbook_row(orderbooks: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if orderbooks.empty or "timestamp" not in orderbooks:
        return None
    idx = orderbooks["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return orderbooks.iloc[int(idx)]


def first_level_size(row: pd.Series | None, side: str) -> float:
    if row is None:
        return 0.0
    keys = ("bid_size", "best_bid_size", "bid_size_0", "bids") if side == "bid" else ("ask_size", "best_ask_size", "ask_size_0", "asks")
    for key in keys:
        value = row.get(key, None)
        if value is None:
            continue
        if key in {"bids", "asks"}:
            try:
                return float(value[0][1])
            except Exception:
                continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def is_joining_best(side: str, price: float, best_bid: float, best_ask: float) -> bool:
    if side == "bid":
        return abs(float(price) - float(best_bid)) <= max(1e-9, abs(float(best_bid)) * 1e-9)
    return abs(float(price) - float(best_ask)) <= max(1e-9, abs(float(best_ask)) * 1e-9)


def matching_trade_with_queue_decay(
    side: str,
    price: float,
    window: pd.DataFrame,
    queue_ahead: float,
    *,
    queue_decay_per_second: float = 0.0,
    active_at: pd.Timestamp | None = None,
) -> tuple[pd.Series | None, float]:
    remaining_queue = max(0.0, float(queue_ahead))
    initial_queue = remaining_queue
    decay_rate = max(0.0, float(queue_decay_per_second))
    queue_decay_base = 0.0
    last_ts = pd.Timestamp(active_at) if active_at is not None else None
    for _, trade in window.iterrows():
        trade_ts = trade.get("timestamp", None)
        if last_ts is None and trade_ts is not None:
            last_ts = pd.Timestamp(trade_ts)
        if decay_rate > 0 and last_ts is not None and trade_ts is not None and remaining_queue > 0:
            elapsed_seconds = max(0.0, (pd.Timestamp(trade_ts) - last_ts).total_seconds())
            decay = min(remaining_queue, initial_queue * decay_rate * elapsed_seconds)
            remaining_queue -= decay
            queue_decay_base += decay
            last_ts = pd.Timestamp(trade_ts)

        trade_price = float(trade.get("price", np.nan))
        if not np.isfinite(trade_price):
            continue
        crosses = (side == "bid" and trade_price <= price) or (side == "ask" and trade_price >= price)
        if not crosses:
            continue
        trade_size = float(trade.get("size", 0.0) or 0.0)
        if remaining_queue > 0:
            remaining_queue -= max(0.0, trade_size)
            if remaining_queue >= 0:
                continue
        return trade, queue_decay_base
    return None, queue_decay_base


def matching_trade(
    side: str,
    price: float,
    window: pd.DataFrame,
    queue_ahead: float,
    *,
    queue_decay_per_second: float = 0.0,
    active_at: pd.Timestamp | None = None,
) -> pd.Series | None:
    trade, _queue_decay_base = matching_trade_with_queue_decay(
        side,
        price,
        window,
        queue_ahead,
        queue_decay_per_second=queue_decay_per_second,
        active_at=active_at,
    )
    return trade


def run_replay(config: ReplayConfig, params: dict[str, float]) -> ReplayMetrics:
    prices, trades, orderbooks, input_files = load_symbol_data(config)
    metrics = ReplayMetrics()
    metrics.input_files = input_files
    metrics.starting_equity_usdc = float(config.starting_equity_usdc)
    metrics.equity_usdc = float(config.starting_equity_usdc)
    metrics.min_equity_usdc = float(config.starting_equity_usdc)
    metrics.min_liquidation_buffer_usdc = float(config.starting_equity_usdc)

    if prices.empty:
        return metrics

    prices = prices.reset_index(drop=True)
    trades = trades.reset_index(drop=True)
    orderbooks = orderbooks.reset_index(drop=True)
    if config.max_price_events is not None and config.max_price_events > 0 and len(prices) > config.max_price_events:
        start_ts = prices.iloc[-config.max_price_events]["timestamp"]
        prices = prices[prices["timestamp"] >= start_ts].reset_index(drop=True)
        if not trades.empty and "timestamp" in trades:
            trades = trades[trades["timestamp"] >= start_ts].reset_index(drop=True)
        if not orderbooks.empty and "timestamp" in orderbooks:
            orderbooks = orderbooks[orderbooks["timestamp"] >= start_ts].reset_index(drop=True)
    metrics.input_rows = {
        "prices": int(len(prices)),
        "trades": int(len(trades)),
        "orderbooks": int(len(orderbooks)),
    }
    metrics.data_start = prices.iloc[0]["timestamp"].isoformat()
    metrics.data_end = prices.iloc[-1]["timestamp"].isoformat()
    metrics.price_event_count = int(len(prices))
    metrics.data_span_seconds = max(
        0.0,
        float((prices.iloc[-1]["timestamp"] - prices.iloc[0]["timestamp"]).total_seconds()),
    )
    if metrics.data_span_seconds > 0:
        metrics.price_events_per_day = float(metrics.price_event_count) / (metrics.data_span_seconds / 86_400.0)
    gaps = prices["timestamp"].diff().dt.total_seconds().dropna()
    if not gaps.empty:
        metrics.max_price_gap_seconds = float(gaps.max())
        metrics.p95_price_gap_seconds = float(gaps.quantile(0.95))
    snapshot_ts = prices.iloc[0]["timestamp"]
    metrics.parameter_series.append(
        replay_parameter_snapshot(
            config,
            params,
            snapshot_ts,
            data_start=metrics.data_start,
            data_end=metrics.data_end,
        )
    )
    metrics.toxicity_series.append(
        replay_toxicity_snapshot(
            config,
            params,
            snapshot_ts,
            data_start=metrics.data_start,
            data_end=metrics.data_end,
        )
    )

    total_quote_latency_ms = config.decision_latency_ms + config.order_ack_latency_ms
    hjb_cache = compute_hjb_cache(params, config.q_max)
    for row_idx, row in prices.iterrows():
        mid, best_bid, best_ask = mid_from_price_row(row, config.mid_fallback)
        metrics.final_mid = mid

        q = inventory_q(metrics.inventory_base, config.inventory_unit_base, config.q_max)
        if q == 0:
            metrics.time_at_q_boundary["q_min"] += 1
        if q == config.q_max:
            metrics.time_at_q_boundary["q_max"] += 1
        metrics.inventory_histogram[q] = metrics.inventory_histogram.get(q, 0) + 1

        if config.funding_rate_per_hour and row_idx > 0:
            prev_ts = prices.loc[row_idx - 1, "timestamp"]
            elapsed_hours = max(0.0, (row["timestamp"] - prev_ts).total_seconds() / 3600.0)
            funding = -metrics.inventory_base * mid * float(config.funding_rate_per_hour) * elapsed_hours
            metrics.funding_usdc += funding
            metrics.cash_usdc += funding

        bid, ask, _ = compute_quotes(mid, q, params, config.q_max, hjb_cache, maker_fee=config.maker_fee)
        inventory_at_decision = metrics.inventory_base
        active_at = row["timestamp"] + pd.Timedelta(milliseconds=total_quote_latency_ms)
        if row_idx + 1 < len(prices):
            stale_at = prices.loc[row_idx + 1, "timestamp"] + pd.Timedelta(milliseconds=config.cancel_latency_ms)
        else:
            stale_at = active_at + pd.Timedelta(milliseconds=config.cancel_latency_ms)

        for side, price in (("bid", bid), ("ask", ask)):
            if price is None:
                continue
            if side == "ask" and inventory_at_decision <= 0:
                continue
            metrics.quote_attempts += 1
            depth_key = quote_depth_key(side, mid, price)
            metrics.quote_attempts_by_depth[depth_key] = metrics.quote_attempts_by_depth.get(depth_key, 0) + 1
            ok, _reason = post_only_check(side, price, best_bid, best_ask)
            if not ok:
                metrics.post_only_rejects += 1
                continue

            if trades.empty:
                metrics.stale_quote_cancels += 1
                continue

            queue_row = active_orderbook_row(orderbooks, active_at)
            queue_ahead = first_level_size(queue_row, side) if is_joining_best(side, price, best_bid, best_ask) else 0.0
            window = trades[(trades["timestamp"] >= active_at) & (trades["timestamp"] <= stale_at)]
            fill_trade, queue_decay_base = matching_trade_with_queue_decay(
                side,
                price,
                window,
                queue_ahead,
                queue_decay_per_second=config.queue_decay_per_second,
                active_at=active_at,
            )
            metrics.queue_decay_base += queue_decay_base

            if fill_trade is None:
                metrics.stale_quote_cancels += 1
                continue

            trade_price = float(fill_trade.get("price", np.nan))
            trade_size = float(fill_trade.get("size", config.inventory_unit_base) or config.inventory_unit_base)
            if side == "ask":
                fill_size = min(trade_size, config.inventory_unit_base, max(0.0, metrics.inventory_base))
            else:
                fill_size = min(trade_size, config.inventory_unit_base)
            if fill_size <= 0:
                continue
            notional = fill_size * price
            fee = notional * config.maker_fee
            gross_spread = abs(mid - price) * fill_size
            side_pnl = gross_spread - fee
            metrics.maker_fills += 1
            metrics.fees_usdc += fee
            if side == "bid":
                metrics.inventory_base += fill_size
                metrics.cash_usdc -= notional + fee
                metrics.pnl_by_side["bid"] += side_pnl
            else:
                metrics.inventory_base -= fill_size
                metrics.cash_usdc += notional - fee
                metrics.pnl_by_side["ask"] += side_pnl
            metrics.realized_spread_usdc += gross_spread
            metrics.fills_by_depth[depth_key] = metrics.fills_by_depth.get(depth_key, 0) + 1
            fill_ts = fill_trade["timestamp"]
            for horizon_ms in MARKOUT_HORIZONS_MS:
                future = future_mid(prices, fill_ts, horizon_ms, config.mid_fallback)
                if future is None:
                    continue
                markout = markout_value(side, price, future)
                metrics.markout_samples.append({
                    "fill_ts": fill_ts.isoformat(),
                    "side": side,
                    "horizon_ms": horizon_ms,
                    "fill_price": price,
                    "future_mid": future,
                    "markout_usdc_per_base": markout,
                    "markout_usdc": markout * fill_size,
                })

        update_margin_metrics(metrics, config, mid)

    if metrics.final_mid is not None:
        metrics.mark_to_market_pnl_usdc = metrics.cash_usdc + metrics.inventory_base * metrics.final_mid

    return metrics


def synthetic_symbol_data(config: ReplayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": config.mid_fallback - 0.5, "ask": config.mid_fallback + 0.5},
            {
                "timestamp": ts0 + pd.Timedelta(seconds=1),
                "bid": config.mid_fallback - 0.4,
                "ask": config.mid_fallback + 0.6,
            },
            {
                "timestamp": ts0 + pd.Timedelta(seconds=2),
                "bid": config.mid_fallback - 0.3,
                "ask": config.mid_fallback + 0.7,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": ts0 + pd.Timedelta(milliseconds=600),
                "price": config.mid_fallback - 3.0,
                "size": config.inventory_unit_base,
            }
        ]
    )
    return prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay HJB market-making quotes over collected Hyperliquid data.")
    parser.add_argument("--symbol", default="ETH")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "HL_data")
    parser.add_argument("--mid", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run against built-in synthetic BBO/trade data.")
    parser.add_argument("--kappa-plus", type=float, required=True)
    parser.add_argument("--kappa-minus", type=float, required=True)
    parser.add_argument("--lambda-plus", type=float, required=True)
    parser.add_argument("--lambda-minus", type=float, required=True)
    parser.add_argument("--epsilon-plus", type=float, default=0.0)
    parser.add_argument("--epsilon-minus", type=float, default=0.0)
    parser.add_argument("--maker-fee", type=float, default=MAKER_FEE)
    parser.add_argument("--taker-fee", type=float, default=TAKER_FEE)
    parser.add_argument("--funding-rate-per-hour", type=float, default=0.0)
    parser.add_argument("--starting-equity-usdc", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-rate", type=float, default=0.05)
    parser.add_argument(
        "--queue-decay-per-second",
        type=float,
        default=0.05,
        help="Conservative queue-ahead cancellation decay as a fraction of initial queue per second.",
    )
    parser.add_argument("--newest-per-stream", type=int, default=None)
    parser.add_argument("--max-price-events", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = {
        "kappa+": args.kappa_plus,
        "kappa-": args.kappa_minus,
        "lambda+": args.lambda_plus,
        "lambda-": args.lambda_minus,
        "epsilon+": args.epsilon_plus,
        "epsilon-": args.epsilon_minus,
    }
    config = ReplayConfig(
        symbol=args.symbol,
        data_dir=args.data_dir,
        mid_fallback=args.mid,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        funding_rate_per_hour=args.funding_rate_per_hour,
        starting_equity_usdc=args.starting_equity_usdc,
        leverage=args.leverage,
        maintenance_margin_rate=args.maintenance_margin_rate,
        queue_decay_per_second=args.queue_decay_per_second,
        newest_per_stream=args.newest_per_stream,
        max_price_events=args.max_price_events,
    )
    if args.synthetic_smoke:
        original_loader = load_symbol_data
        try:
            globals()["load_symbol_data"] = synthetic_symbol_data
            metrics = run_replay(config, params)
        finally:
            globals()["load_symbol_data"] = original_loader
    else:
        metrics = run_replay(config, params)
    payload = metrics.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
