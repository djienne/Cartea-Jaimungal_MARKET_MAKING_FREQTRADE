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
    normalize_price_bbo,
    normalize_timestamp_column,
    post_only_check,
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
