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

# NOTE: do not stub the hyperliquid SDK here. It is installed, and injecting fake
# modules into sys.modules leaks across the whole pytest session -- a stubbed
# hyperliquid.utils shadows the real Cloid that hyperliquid_alo_executor imports
# inside order_args_for_submit, breaking three unrelated tests depending only on
# collection order.
from hyperliquid_data_collector import (  # noqa: E402
    FLOAT32_EXACT_INT_LIMIT,
    HyperliquidDataCollector,
)
from estimator_common import _load_parquet_dir  # noqa: E402


def _collector(tmp_path: Path, *, compact_after_minutes: float = 0.001):
    """A collector instance with no network, for storage behaviour only."""
    collector = HyperliquidDataCollector.__new__(HyperliquidDataCollector)
    collector.symbols = ["TEST"]
    collector.output_dir = str(tmp_path)
    collector.compact_after_minutes = compact_after_minutes
    collector.retention_minutes = 0.0
    return collector


def _write_shards(directory: Path, dtype: str, n_shards: int, rows_per_shard: int = 5):
    directory.mkdir(parents=True, exist_ok=True)
    base_ms = 1_700_000_000_000
    for shard in range(n_shards):
        ts_ms = base_ms + shard * 10_000
        frame = pd.DataFrame(
            {
                "timestamp": [(ts_ms + i) / 1000.0 for i in range(rows_per_shard)],
                "price": [100.0 + i * 0.1 for i in range(rows_per_shard)],
                "size": [1.5 + i for i in range(rows_per_shard)],
                "side": ["buy" if i % 2 else "sell" for i in range(rows_per_shard)],
                "exchange_timestamp": [ts_ms + i for i in range(rows_per_shard)],
            }
        )
        frame.to_parquet(directory / f"{dtype}_{ts_ms}.parquet", index=False)


# --- dtype narrowing ---------------------------------------------------------


def test_narrow_dtypes_shrinks_size_and_side_only():
    frame = pd.DataFrame(
        {
            "timestamp": [1.5, 2.5],
            "price": [62929.5, 62929.6],
            "size": [1.25, 2.5],
            "side": ["buy", "sell"],
        }
    )
    out = HyperliquidDataCollector.narrow_dtypes(frame.copy())
    assert out["size"].dtype == "float32"
    assert str(out["side"].dtype) == "category"
    # Prices must stay float64: float32 cannot hold a 0.1 tick at BTC scale.
    assert out["price"].dtype == "float64"
    assert out["price"].iloc[0] == 62929.5
    # timestamp stays float seconds -- readers multiply this column by 1000.
    assert out["timestamp"].dtype == "float64"
    assert out["timestamp"].iloc[0] == 1.5


def test_narrow_dtypes_keeps_float64_for_sizes_beyond_float32_integer_range():
    """Regression: cheap assets trade integer sizes near the 2**24 ceiling.

    PENGU already prints 7.0M-unit orders. Above 2**24 float32 stops representing
    integers exactly, and size feeds the notional the viability gate reports.
    """
    frame = pd.DataFrame({"size": [1.0, float(FLOAT32_EXACT_INT_LIMIT) + 8.0]})
    out = HyperliquidDataCollector.narrow_dtypes(frame.copy())
    assert out["size"].dtype == "float64"
    assert out["size"].iloc[1] == float(FLOAT32_EXACT_INT_LIMIT) + 8.0

    safe = pd.DataFrame({"size": [1.0, 7_037_672.0]})
    narrowed = HyperliquidDataCollector.narrow_dtypes(safe.copy())
    assert narrowed["size"].dtype == "float32"
    # Integers below the ceiling round-trip exactly.
    assert float(narrowed["size"].iloc[1]) == 7_037_672.0


def test_narrow_dtypes_survives_a_non_numeric_size_column():
    frame = pd.DataFrame({"size": ["oops", "values"]})
    out = HyperliquidDataCollector.narrow_dtypes(frame.copy())
    assert list(out["size"]) == ["oops", "values"]


# --- compaction --------------------------------------------------------------


def test_compaction_merges_shards_and_preserves_every_row(tmp_path):
    directory = tmp_path / "TEST" / "trades"
    _write_shards(directory, "trades", n_shards=40, rows_per_shard=5)
    before_frame, _ = _load_parquet_dir(directory)
    before_files = list(directory.glob("*.parquet"))
    assert len(before_files) == 40

    _collector(tmp_path)._compact_old_shards()

    after_files = list(directory.glob("*.parquet"))
    after_frame, _ = _load_parquet_dir(directory)
    assert len(after_files) == 1, "one file per hour bucket"
    assert "_compact_" in after_files[0].name
    assert len(after_frame) == len(before_frame)
    # Content is preserved exactly, modulo row order and the narrowed size dtype.
    key = ["timestamp", "price", "exchange_timestamp"]
    a = before_frame.sort_values(key).reset_index(drop=True)
    b = after_frame.sort_values(key).reset_index(drop=True)
    for col in key:
        assert np.allclose(a[col].astype(float), b[col].astype(float), rtol=0, atol=0)
    assert np.allclose(a["size"].astype(float), b["size"].astype(float), rtol=1e-6, atol=0)
    assert list(a["side"].astype(str)) == list(b["side"].astype(str))


def test_compacted_name_carries_the_newest_timestamp(tmp_path):
    """select_shards_for_window and the pruner both key off the filename stamp.

    Naming the merged file for its NEWEST source keeps it selected while any of
    its rows are inside the window, and pruned only once all of them are stale.
    """
    directory = tmp_path / "TEST" / "prices"
    _write_shards(directory, "prices", n_shards=6)
    _collector(tmp_path)._compact_old_shards()
    merged = list(directory.glob("*.parquet"))[0]
    newest_source_ms = 1_700_000_000_000 + 5 * 10_000
    assert merged.stem.rsplit("_", 1)[1] == str(newest_source_ms)


def test_compaction_leaves_the_live_tail_alone(tmp_path):
    """Shards inside the estimator's window must never be rewritten under it."""
    directory = tmp_path / "TEST" / "prices"
    directory.mkdir(parents=True)
    import time as _time

    now_ms = int(_time.time() * 1000)
    # Two settled shards inside ONE hour bucket (compaction merges per hour, and
    # a bucket holding a single shard is skipped as pointless), plus two shards
    # young enough to still be inside the estimator's window.
    settled_hour = ((now_ms - 2 * 3_600_000) // 3_600_000) * 3_600_000
    stamps = [settled_hour + 10 * 60_000, settled_hour + 20 * 60_000]
    stamps += [now_ms - 2 * 60_000, now_ms - 60_000]
    for ts_ms in stamps:
        pd.DataFrame({"timestamp": [ts_ms / 1000.0], "price": [1.0], "size": [1.0]}).to_parquet(
            directory / f"prices_{ts_ms}.parquet", index=False
        )

    _collector(tmp_path, compact_after_minutes=15.0)._compact_old_shards()

    names = sorted(p.name for p in directory.glob("*.parquet"))
    compacted = [n for n in names if "_compact_" in n]
    untouched = [n for n in names if "_compact_" not in n]
    assert len(compacted) == 1, "the two settled shards merge"
    assert len(untouched) == 2, "the two recent shards are left in place"


def test_compaction_skips_a_single_shard_bucket(tmp_path):
    """Nothing to gain from rewriting one file as one file."""
    directory = tmp_path / "TEST" / "trades"
    _write_shards(directory, "trades", n_shards=1)
    _collector(tmp_path)._compact_old_shards()
    names = [p.name for p in directory.glob("*.parquet")]
    assert names == ["trades_1700000000000.parquet"]


def test_compaction_is_idempotent(tmp_path):
    """A second pass must not re-compact its own output into a new file."""
    directory = tmp_path / "TEST" / "trades"
    _write_shards(directory, "trades", n_shards=10)
    collector = _collector(tmp_path)
    collector._compact_old_shards()
    first = sorted(p.name for p in directory.glob("*.parquet"))
    collector._compact_old_shards()
    assert sorted(p.name for p in directory.glob("*.parquet")) == first


def test_loader_treats_a_vanished_shard_as_housekeeping_not_corruption(tmp_path):
    """Compaction deletes sources after writing the merged file.

    A reader mid-scan can list a shard and find it gone a moment later. That must
    not count toward the corruption threshold, or routine compaction would start
    failing estimator cycles.
    """
    directory = tmp_path / "trades"
    _write_shards(directory, "trades", n_shards=30)
    files = sorted(directory.glob("*.parquet"))

    real_read = pd.read_parquet
    deleted = {"done": False}

    def racy_read(path, *args, **kwargs):
        # Delete a later shard the first time we read, mimicking compaction
        # landing between the directory listing and the open.
        if not deleted["done"]:
            deleted["done"] = True
            files[-1].unlink()
        return real_read(path, *args, **kwargs)

    import estimator_common

    original = estimator_common.pd.read_parquet
    estimator_common.pd.read_parquet = racy_read
    try:
        frame, stats = _load_parquet_dir(directory)
    finally:
        estimator_common.pd.read_parquet = original

    assert stats["files_vanished"] == 1
    assert stats["files_failed"] == 0, "a vanished shard is not corruption"
    assert len(frame) > 0


def test_loader_still_raises_on_real_corruption_after_the_vanish_carve_out(tmp_path):
    directory = tmp_path / "trades"
    _write_shards(directory, "trades", n_shards=2)
    (directory / "trades_1700000000999.parquet").write_bytes(b"not parquet")
    from estimator_common import ShardReadError

    with pytest.raises(ShardReadError):
        _load_parquet_dir(directory)


# --------------------------------------------------------------------------
# Websocket health probe. Hyperliquid expires a session every few hours and
# sends a close frame; the SDK's manager thread then exits without reconnecting.
# Recovery used to wait out INACTIVITY_TIMEOUT_SEC, costing ~1.8% of the CASHCAT
# tape in clockwork 3.1-3.5 min gaps.
# --------------------------------------------------------------------------


class _FakeSock:
    def __init__(self, connected: bool):
        self.connected = connected


class _FakeWs:
    def __init__(self, sock):
        self.sock = sock


class _FakeManager:
    def __init__(self, alive: bool, ws):
        self._alive = alive
        self.ws = ws

    def is_alive(self) -> bool:
        return self._alive


class _FakeInfo:
    def __init__(self, ws_manager):
        self.ws_manager = ws_manager


def _probe(ws_manager):
    collector = HyperliquidDataCollector.__new__(HyperliquidDataCollector)
    collector.info = _FakeInfo(ws_manager)
    return collector._websocket_is_down()


def test_healthy_socket_is_not_down():
    assert _probe(_FakeManager(True, _FakeWs(_FakeSock(True)))) is False


def test_dead_manager_thread_is_down():
    """The observed failure: the close frame ends WebsocketManager.run()."""
    assert _probe(_FakeManager(False, _FakeWs(_FakeSock(True)))) is True


def test_missing_socket_is_down():
    assert _probe(_FakeManager(True, _FakeWs(None))) is True


def test_disconnected_socket_is_down():
    assert _probe(_FakeManager(True, _FakeWs(_FakeSock(False)))) is True


def test_skip_ws_mode_is_never_down():
    """Without a ws_manager there is no socket to watch; must not reconnect-loop."""
    collector = HyperliquidDataCollector.__new__(HyperliquidDataCollector)
    collector.info = _FakeInfo(None)
    collector.info.ws_manager = None
    assert collector._websocket_is_down() is False
