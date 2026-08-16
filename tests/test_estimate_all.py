from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import estimate_all  # noqa: E402
from estimator_common import (  # noqa: E402
    MAX_SHARD_READ_FAILURE_RATE,
    MarketWindow,
    ShardReadError,
    _load_parquet_dir,
    select_shards_for_window,
    shard_timestamp_ms,
)


def _window(n_trades: int = 4) -> MarketWindow:
    mids = pd.DataFrame(
        {
            "ts_ms": [0.0, 1000.0, 2000.0],
            "bid": [99.9, 99.9, 99.9],
            "ask": [100.1, 100.1, 100.1],
            "mid": [100.0, 100.0, 100.0],
        }
    )
    trades = pd.DataFrame(
        {
            "ts_ms": np.arange(n_trades, dtype=float) * 500.0,
            "side": ["buy", "sell"] * (n_trades // 2),
            "price": [100.0] * n_trades,
            "size": [1.0] * n_trades,
        }
    )
    return MarketWindow(
        mids=mids,
        trades=trades,
        mid_source="prices",
        ts_source="exchange",
        window_start_ms=0.0,
        window_end_ms=2000.0,
        meta={"n_mid_updates": 3, "n_trade_prints": n_trades, "shards_read": 7},
    )


# --- shard selection ---------------------------------------------------------


def test_shard_timestamp_ms_parses_collector_names():
    assert shard_timestamp_ms(Path("trades_1786915691544.parquet")) == 1786915691544.0
    assert shard_timestamp_ms(Path("noseparator.parquet")) is None
    assert shard_timestamp_ms(Path("trades_notanumber.parquet")) is None


def test_select_shards_for_window_keeps_only_the_window():
    newest = 10_000_000.0
    files = [Path(f"trades_{int(newest - i * 60_000)}.parquet") for i in range(60)]
    # 10-minute window: everything older than 10 min (+ margin) is irrelevant.
    kept = select_shards_for_window(files, window_minutes=10, margin_ms=0.0)
    assert len(kept) == 11  # newest plus the ten one-minute steps inside the window
    assert all(shard_timestamp_ms(p) >= newest - 600_000 for p in kept)


def test_select_shards_for_window_is_relative_to_newest_not_wall_clock():
    """A frozen historical directory must behave like a live one."""
    files = [Path(f"trades_{1_000_000 + i * 1000}.parquet") for i in range(100)]
    kept = select_shards_for_window(files, window_minutes=1, margin_ms=0.0)
    assert Path("trades_1099000.parquet") in kept  # newest shard survives
    assert Path("trades_1000000.parquet") not in kept


def test_select_shards_for_window_fails_open_on_unparseable_names():
    files = [Path("weird.parquet"), Path("trades_5000.parquet")]
    assert select_shards_for_window(files, window_minutes=1) == files
    # No parseable timestamps at all: keep everything rather than drop data.
    only_weird = [Path("a.parquet"), Path("b.parquet")]
    assert select_shards_for_window(only_weird, window_minutes=1) == only_weird


def test_select_shards_for_window_noop_without_a_window():
    files = [Path("trades_1.parquet"), Path("trades_2.parquet")]
    assert select_shards_for_window(files, window_minutes=None) == files
    assert select_shards_for_window(files, window_minutes=0) == files


# --- shard read failures -----------------------------------------------------


def _write_shard(directory: Path, name: str, rows: int = 2) -> Path:
    path = directory / name
    pd.DataFrame({"timestamp": [1.0] * rows, "price": [1.0] * rows}).to_parquet(path, index=False)
    return path


def test_load_parquet_dir_reports_shard_counts(tmp_path):
    for i in range(4):
        _write_shard(tmp_path, f"trades_{1000 + i}.parquet")
    frame, stats = _load_parquet_dir(tmp_path)
    assert len(frame) == 8
    assert stats["files_present"] == 4
    assert stats["files_read"] == 4
    assert stats["files_failed"] == 0


def test_load_parquet_dir_raises_when_too_many_shards_are_corrupt(tmp_path):
    """Regression: a torn shard used to be swallowed with a bare `continue`.

    That silently shortened the window -- biasing n_trades and lambda down with
    no signal anywhere that data had gone missing.
    """
    _write_shard(tmp_path, "trades_1000.parquet")
    (tmp_path / "trades_1001.parquet").write_bytes(b"not a parquet file at all")
    with pytest.raises(ShardReadError) as excinfo:
        _load_parquet_dir(tmp_path)
    assert "unreadable" in str(excinfo.value)


def test_load_parquet_dir_tolerates_a_failure_rate_under_the_threshold(tmp_path):
    good = int(1 / MAX_SHARD_READ_FAILURE_RATE) + 5
    for i in range(good):
        _write_shard(tmp_path, f"trades_{1000 + i}.parquet")
    (tmp_path / f"trades_{2000}.parquet").write_bytes(b"garbage")
    frame, stats = _load_parquet_dir(tmp_path)
    assert stats["files_failed"] == 1
    assert stats["files_read"] == good
    assert len(frame) == good * 2


def test_load_parquet_dir_ignores_in_progress_tmp_files(tmp_path):
    """The collector writes <name>.parquet.tmp then renames; readers must not see it."""
    _write_shard(tmp_path, "trades_1000.parquet")
    (tmp_path / "trades_1001.parquet.tmp").write_bytes(b"half written")
    frame, stats = _load_parquet_dir(tmp_path)
    assert stats["files_present"] == 1
    assert stats["files_failed"] == 0
    assert len(frame) == 2


# --- single-pass cycle -------------------------------------------------------


def test_run_all_loads_the_window_once_and_shares_it(monkeypatch):
    """The whole point of estimate_all: one parquet scan per cycle, not three."""
    window = _window()
    loads = []
    seen = {}

    def fake_load(crypto, minutes, data_dir=None):
        loads.append((crypto, minutes))
        return window

    def fake_kappa(crypto, minutes=30, ema_tau=None, window=None, **kwargs):
        seen["kappa"] = window

    def fake_epsilon(crypto, minutes=30, post_horizon_ms=None, ema_tau=None, window=None, **kwargs):
        seen["epsilon"] = window

    monkeypatch.setattr(estimate_all, "load_market_window", fake_load)
    monkeypatch.setattr(estimate_all, "run_kappa_for_crypto", fake_kappa)
    monkeypatch.setattr(estimate_all, "run_epsilon_for_crypto", fake_epsilon)
    monkeypatch.setattr(estimate_all, "_lambda_trades_monitor", lambda *a, **k: None)

    result = estimate_all.run_all("ETH", minutes=30)

    assert loads == [("ETH", 30)], "the window must be loaded exactly once per cycle"
    assert seen["kappa"] is window
    assert seen["epsilon"] is window
    assert result["crypto"] == "ETH"
    assert result["cycle_seconds"] >= 0


def test_run_all_runs_kappa_before_epsilon(monkeypatch):
    """get_epsilon reads kappa.json for its toxicity diagnostic, so order matters."""
    order = []
    monkeypatch.setattr(estimate_all, "load_market_window", lambda *a, **k: _window())
    monkeypatch.setattr(
        estimate_all, "run_kappa_for_crypto", lambda *a, **k: order.append("kappa")
    )
    monkeypatch.setattr(
        estimate_all, "run_epsilon_for_crypto", lambda *a, **k: order.append("epsilon")
    )
    monkeypatch.setattr(
        estimate_all, "_lambda_trades_monitor", lambda *a, **k: order.append("lambda")
    )

    estimate_all.run_all("ETH", minutes=30)

    assert order == ["kappa", "epsilon", "lambda"]


def test_run_all_propagates_a_load_failure(monkeypatch):
    """A cycle that cannot read its data must fail loudly, not write stale params."""

    def boom(*args, **kwargs):
        raise ShardReadError("3/4 shards unreadable")

    monkeypatch.setattr(estimate_all, "load_market_window", boom)
    with pytest.raises(ShardReadError):
        estimate_all.run_all("ETH", minutes=30)
