"""Regression tests for documentation/report helpers that summarize live evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import archive_period  # noqa: E402
import grid_pnl_curve  # noqa: E402
import sweep_replay  # noqa: E402


def test_sweep_markdown_discloses_execution_and_scoring_limits():
    text = sweep_replay.to_markdown({
        "status": "ok",
        "stage_c": [{"key": "calibration|risk", "held_out": {}, "train": {}}],
        "latency_ladder": [{"scenario": {}}],
    })
    assert r"calibration\|risk" in text
    assert "including training" in text
    assert "quoted spread capture" in text
    assert "not measured host execution capabilities" in text
    assert "not solvency or promotion readiness" in text
    assert "max(0, refresh" not in text


def test_sweep_headline_reads_the_current_nested_schema(tmp_path):
    payload = {
        "status": "ok",
        "search_scenario": {"name": "good", "latency_ms": 100, "refresh_ms": 250},
        "split_at": "2026-08-26T10:35:45Z",
        "stage_c": [
            {
                "key": "winner",
                "train": {"pnl_usdc": -83.54},
                "held_out": {"pnl_usdc": -99.12, "maker_fills": 2753},
            }
        ],
        "latency_ladder": [
            {
                "scenario": {"name": "colocated", "latency_ms": 50, "refresh_ms": 100},
                "pnl_usdc": 130.74,
                "maker_fills": 3046,
            }
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    text = "\n".join(archive_period.sweep_headline(path))
    assert "good (100 ms latency, 250 ms refresh)" in text
    assert "| `winner` | -83.54 | **-99.12** | 2753 |" in text
    assert "| colocated | +130.74 | 3046 |" in text


def test_sweep_headline_refuses_schema_drift_instead_of_printing_zeros(tmp_path):
    path = tmp_path / "stale-sweep.json"
    path.write_text(
        json.dumps({"status": "ok", "stage_c": [{"obsolete_key": 1}]}),
        encoding="utf-8",
    )
    text = "\n".join(archive_period.sweep_headline(path))
    assert "schema mismatch" in text
    assert "+0.00" not in text


def test_history_discovery_and_downsampling_preserve_run_boundaries(tmp_path):
    root = tmp_path / "grid"
    first = root / "runs" / "run-1" / "equity_history.csv"
    second = root / "runs" / "run-2" / "equity_history.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    columns = ["ts_ms", "run_started_ms", "variant", "net_pnl_usdc", "fills", "mid"]
    pd.DataFrame([[1_000, 100, "baseline", 1.0, 1, 0.1]], columns=columns).to_csv(first, index=False)
    pd.DataFrame([[1_100, 200, "baseline", 2.0, 2, 0.1]], columns=columns).to_csv(second, index=False)

    histories = archive_period.grid_history_paths(root)
    assert histories == [first, second]
    output = tmp_path / "period.csv"
    stats = archive_period.slice_and_downsample(histories, output, 0, 15)
    frame = pd.read_csv(output)
    assert stats["rows"] == 2
    assert set(frame["run_started_ms"]) == {100, 200}


def test_plotter_resolves_the_active_run_from_grid_state(tmp_path):
    root = tmp_path / "grid"
    active = root / "runs" / "run-active"
    active.mkdir(parents=True)
    (root / "grid_state.json").write_text(
        json.dumps({"run_id": "run-active"}), encoding="utf-8"
    )
    assert grid_pnl_curve.resolve_run_dir(root) == active
    assert grid_pnl_curve.resolve_run_dir(active) == active


def test_plotter_reads_rotated_compressed_fills_even_after_damage(tmp_path):
    import pytest
    import zstandard

    def write(generation, stamp):
        row = {"kind": "fill", "exchange_ms": stamp, "payload": {
            "side": "buy", "qty_units": 1, "px": 100000, "fee_usdc": 0.01}}
        path = tmp_path / ("grid-baseline.jsonl.zst" + generation)
        path.write_bytes(zstandard.ZstdCompressor().compress((json.dumps(row, separators=(",", ":")) + "\n").encode()))
        return path

    oldest = write(".2", 1_000)
    damaged = tmp_path / "grid-baseline.jsonl.zst.1"
    damaged.write_bytes(b"broken compressed frame")
    current = write("", 3_000)
    files = grid_pnl_curve.variant_files(["baseline"], tmp_path)
    assert files == {"baseline": [oldest, damaged, current]}
    with pytest.warns(RuntimeWarning, match="Incomplete event log"):
        fills = grid_pnl_curve.read_fills(files["baseline"])
    assert fills["ts_ms"].tolist() == [1_000, 3_000]
    assert fills["signed_qty"].sum() == 2
