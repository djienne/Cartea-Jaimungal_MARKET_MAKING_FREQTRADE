from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import replay_market_maker  # noqa: E402
from replay_market_maker import (  # noqa: E402
    ReplayConfig,
    compute_quotes,
    first_level_size,
    inventory_q,
    matching_trade,
    matching_trade_with_queue_decay,
    normalize_price_bbo,
    normalize_timestamp_column,
    post_only_check,
    quote_depth_bucket_key,
    quote_depth_key,
    round_amount_down,
    round_price_for_side,
    run_replay,
    selected_parquet_files,
)


def test_post_only_check_rejects_crossing_quotes():
    assert post_only_check("bid", 101.0, 99.0, 101.0) == (False, "bid_crosses_ask")
    assert post_only_check("ask", 99.0, 99.0, 101.0) == (False, "ask_crosses_bid")


def test_post_only_check_accepts_passive_quotes():
    assert post_only_check("bid", 100.0, 99.0, 101.0) == (True, "ok")
    assert post_only_check("ask", 100.5, 99.0, 101.0) == (True, "ok")


def test_long_only_inventory_q_is_clipped_nonnegative():
    assert inventory_q(-0.03, 0.01, 3) == 0
    assert inventory_q(0.02, 0.01, 3) == 2
    assert inventory_q(0.08, 0.01, 3) == 3


def test_compute_quotes_uses_configurable_maker_fee_cushion():
    hjb = {
        "q_grid": np.array([-1, 0, 1]),
        "delta_plus": np.array([np.inf, 0.5, 0.4]),
        "delta_minus": np.array([0.4, 0.5, np.inf]),
    }
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    bid_base, ask_base, _ = compute_quotes(100.0, 0, params, 1, hjb, maker_fee=0.001)
    bid_wide, ask_wide, _ = compute_quotes(100.0, 0, params, 1, hjb, maker_fee=0.002)

    assert round(float(bid_base), 6) == 99.3
    assert round(float(ask_base), 6) == 100.7
    assert round(float(bid_wide), 6) == 99.1
    assert round(float(ask_wide), 6) == 100.9


def test_replay_rounds_prices_in_maker_safe_direction():
    assert round_price_for_side("bid", 100.09, 0.1) == 100.0
    assert round_price_for_side("ask", 100.01, 0.1) == 100.1
    assert round_price_for_side("bid", 100.09, 0.0) == 100.09


def test_replay_rounds_amount_down_to_step():
    assert round_amount_down(0.019, 0.01) == 0.01
    assert round_amount_down(0.009, 0.01) == 0.0
    assert round_amount_down(0.019, 0.0) == 0.019


def test_replay_applies_latency_and_records_markouts(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.2, "ask": 101.2},
            {"timestamp": ts0 + pd.Timedelta(seconds=2), "bid": 100.3, "ask": 101.3},
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=100), "price": 90.0, "size": 0.01},
            {"timestamp": ts0 + pd.Timedelta(milliseconds=600), "price": 100.4, "size": 0.01},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )

    params = {
        "kappa+": 100.0,
        "kappa-": 100.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }
    metrics = run_replay(
        ReplayConfig(symbol="ETH", data_dir=Path("."), mid_fallback=100.5, inventory_unit_base=0.01, q_max=3),
        params,
    )

    assert metrics.maker_fills == 1
    assert metrics.taker_fills == 0
    assert metrics.inventory_base == 0.01
    assert metrics.markout_samples
    assert metrics.mark_to_market_pnl_usdc != 0.0
    assert metrics.input_rows["prices"] == 3
    assert metrics.price_event_count == 3
    assert metrics.data_span_seconds == 2.0
    assert metrics.price_events_per_day == 129600.0
    assert metrics.max_price_gap_seconds == 1.0
    assert metrics.p95_price_gap_seconds == 1.0
    metrics_payload = metrics.to_dict()
    assert metrics_payload["price_event_count"] == 3
    assert metrics_payload["max_price_gap_seconds"] == 1.0
    parameter_snapshot = metrics.parameter_series[0]
    toxicity_snapshot = metrics.toxicity_series[0]
    assert parameter_snapshot["ts"] == ts0.isoformat()
    assert parameter_snapshot["source"] == "static_replay_params"
    assert parameter_snapshot["schema_version"] == 1
    assert parameter_snapshot["unit"]["kappa"] == "1/USDC"
    assert parameter_snapshot["data_start"] == ts0.isoformat()
    assert parameter_snapshot["data_end"] == (ts0 + pd.Timedelta(seconds=2)).isoformat()
    assert parameter_snapshot["kappa_plus"] == 100.0
    assert toxicity_snapshot["ts"] == ts0.isoformat()
    assert toxicity_snapshot["source"] == "static_replay_params"
    assert toxicity_snapshot["unit"] == "kappa_times_epsilon"
    assert toxicity_snapshot["toxicity_plus"] == 0.0


def test_matching_trade_waits_for_queue_ahead_volume():
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    trades = pd.DataFrame(
        [
            {"timestamp": ts0, "price": 100.0, "size": 0.4},
            {"timestamp": ts0 + pd.Timedelta(milliseconds=1), "price": 100.0, "size": 0.4},
            {"timestamp": ts0 + pd.Timedelta(milliseconds=2), "price": 100.0, "size": 0.4},
        ]
    )

    fill = matching_trade("bid", 100.0, trades, queue_ahead=0.8)

    assert fill is not None
    assert fill["timestamp"] == trades.iloc[2]["timestamp"]


def test_matching_trade_applies_conservative_queue_decay():
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=500), "price": 100.0, "size": 0.6},
        ]
    )

    no_decay = matching_trade("bid", 100.0, trades, queue_ahead=1.0, active_at=ts0)
    fill, queue_decay = matching_trade_with_queue_decay(
        "bid",
        100.0,
        trades,
        queue_ahead=1.0,
        queue_decay_per_second=1.0,
        active_at=ts0,
    )

    assert no_decay is None
    assert fill is not None
    assert fill["timestamp"] == trades.iloc[0]["timestamp"]
    assert queue_decay == 0.5


def test_replay_records_queue_decay_metric(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=500), "price": 100.0, "size": 0.6},
        ]
    )
    orderbooks = pd.DataFrame(
        [
            {"timestamp": ts0, "bid_size_0": 1.0, "ask_size_0": 1.0},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, orderbooks, {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=1000,
            queue_decay_per_second=1.0,
        ),
        params,
    )

    assert metrics.maker_fills == 1
    assert metrics.queue_decay_base == 0.5
    assert metrics.to_dict()["queue_decay_base"] == 0.5


def test_replay_keeps_final_quote_active_for_refresh_interval(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame([{"timestamp": ts0, "bid": 100.0, "ask": 101.0}])
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=700), "price": 100.0, "size": 0.01},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=0,
            quote_refresh_interval_ms=1000,
        ),
        params,
    )

    payload = metrics.to_dict()
    assert metrics.maker_fills == 1
    assert metrics.quote_decision_events == 1
    assert payload["quote_refresh_interval_ms"] == 1000


def test_replay_consumes_trade_events_once_across_overlapping_windows(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=1500), "price": 100.0, "size": 0.01},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=1000,
            quote_refresh_interval_ms=1000,
        ),
        params,
    )

    assert metrics.quote_decision_events == 2
    assert metrics.quote_attempts == 2
    assert metrics.maker_fills == 1
    assert metrics.consumed_trade_events == 1
    assert metrics.stale_quote_cancels == 1


def test_replay_applies_usable_fill_calibration_conservatively(monkeypatch, tmp_path):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=500), "price": 100.0, "size": 0.01},
            {"timestamp": ts0 + pd.Timedelta(milliseconds=1500), "price": 100.0, "size": 0.01},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_key = quote_depth_bucket_key("bid", 100.5, 100.0, 100.0)
    calibration_path.write_text(
        """{
  "usable_for_calibration": true,
  "generated_at": "2026-05-25T10:00:00Z",
  "inputs": {"bucket_bps": 100.0, "accepted_quotes": 2},
  "maker_fills": 1,
  "maker_ratio": 1.0,
  "reasons": [],
  "fill_probability_by_depth": {"%s": 0.5},
  "fill_probability_by_side": {}
}
"""
        % calibration_key,
        encoding="utf-8",
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=0,
            quote_refresh_interval_ms=1000,
            fill_calibration_path=calibration_path,
        ),
        params,
    )

    payload = metrics.to_dict()
    assert metrics.quote_attempts == 2
    assert metrics.maker_fills == 1
    assert metrics.calibration_rejected_fills == 1
    assert metrics.consumed_trade_events == 2
    assert metrics.calibration_attempts_by_key == {calibration_key: 2}
    assert metrics.calibration_fills_by_key == {calibration_key: 1}
    assert payload["fill_calibration"]["applied"] is True


def test_replay_uses_rounded_price_and_amount_for_fills(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame([{"timestamp": ts0, "bid": 100.0, "ask": 101.0}])
    trades = pd.DataFrame(
        [
            {"timestamp": ts0 + pd.Timedelta(milliseconds=500), "price": 100.0, "size": 0.02},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.09, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.019,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=0,
            quote_refresh_interval_ms=1000,
            price_tick_size=0.1,
            amount_step_size=0.01,
        ),
        params,
    )

    payload = metrics.to_dict()
    depth_key = quote_depth_key("bid", 100.5, 100.0)
    assert metrics.maker_fills == 1
    assert metrics.price_rounding_adjustments == 1
    assert metrics.inventory_base == 0.01
    assert metrics.fills_by_depth == {depth_key: 1}
    assert round(metrics.realized_spread_usdc, 8) == 0.005
    assert payload["price_tick_size"] == 0.1
    assert payload["amount_step_size"] == 0.01


def test_replay_rejects_quote_when_amount_rounds_to_zero(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame([{"timestamp": ts0, "bid": 100.0, "ask": 101.0}])
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, pd.DataFrame(), pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.009,
            q_max=3,
            amount_step_size=0.01,
        ),
        params,
    )

    assert metrics.quote_attempts == 0
    assert metrics.amount_rounding_rejects == 1


def test_replay_tracks_margin_equity_and_liquidation_buffer(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": ts0, "price": 100.0, "size": 0.01},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=1000,
            starting_equity_usdc=10.0,
            leverage=2.0,
            maintenance_margin_rate=0.1,
        ),
        params,
    )

    assert metrics.maker_fills == 1
    assert round(metrics.notional_exposure_usdc, 6) == 1.005
    assert round(metrics.margin_used_usdc, 6) == 0.5025
    assert round(metrics.maintenance_margin_usdc, 6) == 0.1005
    assert metrics.max_margin_used_usdc == metrics.margin_used_usdc
    assert metrics.min_liquidation_buffer_usdc > 0
    assert metrics.liquidation_breach_events == 0
    payload = metrics.to_dict()
    assert payload["starting_equity_usdc"] == 10.0
    assert payload["equity_usdc"] == metrics.equity_usdc


def test_replay_counts_maintenance_margin_breaches(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame([{"timestamp": ts0, "price": 100.0, "size": 0.01}])
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=1000,
            starting_equity_usdc=0.0,
            leverage=2.0,
            maintenance_margin_rate=0.5,
        ),
        params,
    )

    assert metrics.liquidation_breach_events > 0
    assert metrics.min_liquidation_buffer_usdc < 0


def test_replay_reports_fill_ratio_by_depth_as_ratio(monkeypatch):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": 100.0, "ask": 101.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "bid": 100.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": ts0, "price": 100.0, "size": 0.01},
        ]
    )
    monkeypatch.setattr(
        replay_market_maker,
        "load_symbol_data",
        lambda _config: (prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}),
    )
    monkeypatch.setattr(
        replay_market_maker,
        "compute_quotes",
        lambda *_args, **_kwargs: (100.0, None, {}),
    )
    params = {
        "kappa+": 2.0,
        "kappa-": 2.0,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "epsilon+": 0.0,
        "epsilon-": 0.0,
    }

    metrics = run_replay(
        ReplayConfig(
            symbol="ETH",
            data_dir=Path("."),
            mid_fallback=100.5,
            inventory_unit_base=0.01,
            q_max=3,
            decision_latency_ms=0,
            order_ack_latency_ms=0,
            cancel_latency_ms=1000,
        ),
        params,
    )

    depth_key = quote_depth_key("bid", 100.5, 100.0)
    payload = metrics.to_dict()
    assert metrics.quote_attempts_by_depth == {depth_key: 2}
    assert metrics.fills_by_depth == {depth_key: 1}
    assert round(metrics.pnl_by_side["bid"], 8) == 0.00485
    assert round(sum(metrics.pnl_by_side.values()), 8) == round(
        metrics.realized_spread_usdc - metrics.fees_usdc,
        8,
    )
    assert payload["quote_attempts_by_depth"] == {depth_key: 2}
    assert payload["fills_by_depth"] == {depth_key: 1}
    assert payload["fill_ratio_by_depth"] == {depth_key: 0.5}
    assert payload["pnl_by_side"]["bid"] == metrics.pnl_by_side["bid"]


def test_first_level_size_reads_nested_orderbook_levels():
    row = pd.Series({"bids": [[100.0, 1.25]], "asks": [[101.0, 2.5]]})

    assert first_level_size(row, "bid") == 1.25
    assert first_level_size(row, "ask") == 2.5


def test_first_level_size_reads_collector_orderbook_columns():
    row = pd.Series({"bid_size_0": 1.25, "ask_size_0": 2.5})

    assert first_level_size(row, "bid") == 1.25
    assert first_level_size(row, "ask") == 2.5


def test_normalize_timestamp_column_infers_seconds():
    frame = pd.DataFrame({"timestamp": [1779713847.3817592]})

    out = normalize_timestamp_column(frame)

    assert out.iloc[0]["timestamp"].year == 2026


def test_normalize_price_bbo_pivots_collector_side_rows():
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    frame = pd.DataFrame(
        [
            {"timestamp": ts0, "side": "bid", "price": 100.0, "size": 1.0},
            {"timestamp": ts0, "side": "ask", "price": 101.0, "size": 2.0},
            {"timestamp": ts0 + pd.Timedelta(seconds=1), "side": "bid", "price": 100.5, "size": 1.5},
        ]
    )

    out = normalize_price_bbo(frame)

    assert list(out.columns) == ["timestamp", "bid", "ask", "bid_size", "ask_size"]
    assert out.iloc[-1]["bid"] == 100.5
    assert out.iloc[-1]["ask"] == 101.0


def test_selected_parquet_files_can_limit_to_newest(tmp_path):
    old = tmp_path / "old.parquet"
    new = tmp_path / "new.parquet"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    old_mtime = 1_700_000_000
    new_mtime = 1_700_000_100
    old.touch()
    new.touch()
    import os

    os.utime(old, (old_mtime, old_mtime))
    os.utime(new, (new_mtime, new_mtime))

    assert selected_parquet_files(tmp_path, newest_per_stream=1) == [new]
