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


def _replay_board(**overrides):
    board = {
        "generated_at_ms": 1_788_000_000_000,
        "started_at_ms": 1_787_000_000_000,
        "elapsed_seconds": 3_600,
        "symbol": "CASHCAT",
        "feed_health": {"gaps": 0, "downtime_fraction": 0.0, "event_loss": False},
        "resumes": 0,
        "rows": [
            {
                "name": "flatten300",
                "net_pnl_usdc": 11.2,
                "fills": 217,
                "inventory_units": 0,
            }
        ],
        "replay": {
            "training_start_ms": 1_786_000_000_000,
            "training_end_ms": 1_787_000_000_000,
            "scoring_start_ms": 1_787_000_000_000,
            "scoring_end_ms": 1_788_000_000_000,
            "train_fraction": 0.05,
            "latency_ms": 150,
        },
    }
    board.update(overrides)
    return board


def test_replay_headline_reports_the_window_and_the_rows(tmp_path):
    path = tmp_path / "replay_leaderboard.json"
    path.write_text(json.dumps(_replay_board()), encoding="utf-8")

    text = "\n".join(archive_period.replay_headline(path))
    assert "150 ms assumed latency" in text
    assert "first 5% fits and sizes only" in text
    assert "| flatten300 | +11.20 | 217 | 0 |" in text


def test_replay_headline_refuses_a_live_board_instead_of_relabelling_it(tmp_path):
    """A live board and a replay board are the same schema on purpose.

    That is what makes them comparable and what makes mislabelling one as the
    other easy, so the archive refuses rather than presenting a live ranking as
    offline evidence for a window nobody replayed.
    """
    board = _replay_board()
    del board["replay"]
    path = tmp_path / "grid_leaderboard.json"
    path.write_text(json.dumps(board), encoding="utf-8")

    text = "\n".join(archive_period.replay_headline(path))
    assert "not a replay board" in text
    assert "flatten300" not in text


def test_replay_headline_survives_an_unreadable_artifact(tmp_path):
    path = tmp_path / "replay_leaderboard.json"
    path.write_text("{ truncated", encoding="utf-8")
    assert "unreadable" in "\n".join(archive_period.replay_headline(path))


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


def test_the_live_supervisor_makes_no_economic_decision():
    """The supervisor is a liveness watchdog, not a strategy selector.

    It once stopped live unless the best `eligible_for_promotion` leaderboard row
    had positive `promotion_pnl_usdc`. That was coherent only while `promote-best`
    generated the live config from that row. With a hand-edited config the two are
    unrelated -- the shipped `sweep1_flat300` is permanently ineligible -- so the
    gate judged one strategy by another's P&L. Re-introducing any leaderboard read
    here brings that back.
    """
    script = (ROOT / "scripts" / "Manage-CashcatLive.ps1").read_text(encoding="utf-8")
    tick = script.split("function Invoke-SupervisorTick {", 1)[1].split("\nswitch (", 1)[0]
    body = "\n".join(line for line in tick.splitlines() if not line.lstrip().startswith("#"))
    assert "leaderboard" not in body.lower()
    assert "promotion_pnl" not in body
    # It may still stop live, but only to restart an unhealthy container.
    assert body.count("stop cashcat-live") == 1
    assert "up -d --no-deps cashcat-live" in body
