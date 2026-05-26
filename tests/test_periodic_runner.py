from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "user_data" / "strategies"
if str(STRATEGIES) not in sys.path:
    sys.path.insert(0, str(STRATEGIES))

import periodic_test_runner  # noqa: E402
from periodic_test_runner import _validate_symbol_snapshot  # noqa: E402


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_snapshots(tmp_path: Path) -> dict[str, Path]:
    return {
        "kappa.json": write(
            tmp_path / "kappa.json",
            {
                "ETH": {
                    "schema_version": 2,
                    "status": "ok",
                    "generated_at": "2026-05-25T10:00:00Z",
                    "window_start": "2026-05-25T09:30:00Z",
                    "window_end": "2026-05-25T10:00:00Z",
                    "kappa+": 2.0,
                    "kappa-": 2.5,
                    "lambda+": 0.1,
                    "lambda-": 0.2,
                    "n_quotes": 100,
                    "n_trades": 50,
                    "n_points_plus": 3,
                    "n_points_minus": 3,
                    "r2_plus": 0.5,
                    "r2_minus": 0.6,
                }
            },
        ),
        "lambda.json": write(
            tmp_path / "lambda.json",
            {
                "ETH": {
                    "schema_version": 2,
                    "status": "ok",
                    "generated_at": "2026-05-25T10:00:00Z",
                    "window_start": "2026-05-25T09:30:00Z",
                    "window_end": "2026-05-25T10:00:00Z",
                    "lambda+": 0.1,
                    "lambda-": 0.2,
                    "lambda_source": "lambda0_fit",
                    "n_trades": 50,
                }
            },
        ),
        "epsilon.json": write(
            tmp_path / "epsilon.json",
            {
                "ETH": {
                    "schema_version": 2,
                    "status": "ok",
                    "generated_at": "2026-05-25T10:00:00Z",
                    "window_start": "2026-05-25T09:30:00Z",
                    "window_end": "2026-05-25T10:00:00Z",
                    "epsilon+": 0.01,
                    "epsilon-": 0.02,
                    "window_ms": 200,
                    "estimator": "trimmed_mean",
                    "unit": "USDC",
                    "n_buy_events": 3,
                    "n_sell_events": 3,
                    "toxicity_plus": 0.02,
                    "toxicity_minus": 0.05,
                }
            },
        ),
    }


def test_runner_accepts_valid_schema_v2_snapshots(tmp_path):
    assert _validate_symbol_snapshot(valid_snapshots(tmp_path), "ETH") == (True, "ok")


def test_runner_rejects_non_ok_snapshot_status(tmp_path):
    snapshots = valid_snapshots(tmp_path)
    payload = json.loads(snapshots["epsilon.json"].read_text(encoding="utf-8"))
    payload["ETH"]["status"] = "insufficient_data"
    snapshots["epsilon.json"].write_text(json.dumps(payload), encoding="utf-8")

    ok, reason = _validate_symbol_snapshot(snapshots, "ETH")

    assert not ok
    assert reason.startswith("status_not_ok_epsilon.json")


def test_runner_rejects_missing_kappa_fit_diagnostics(tmp_path):
    snapshots = valid_snapshots(tmp_path)
    payload = json.loads(snapshots["kappa.json"].read_text(encoding="utf-8"))
    del payload["ETH"]["r2_plus"]
    snapshots["kappa.json"].write_text(json.dumps(payload), encoding="utf-8")

    ok, reason = _validate_symbol_snapshot(snapshots, "ETH")

    assert not ok
    assert reason == "missing_r2_plus_in_kappa.json"


def test_runner_rejects_raw_lambda_as_hjb_input(tmp_path):
    snapshots = valid_snapshots(tmp_path)
    payload = json.loads(snapshots["lambda.json"].read_text(encoding="utf-8"))
    payload["ETH"]["lambda_source"] = "lambda_raw"
    snapshots["lambda.json"].write_text(json.dumps(payload), encoding="utf-8")

    ok, reason = _validate_symbol_snapshot(snapshots, "ETH")

    assert not ok
    assert reason == "lambda_json_must_be_lambda0_fit"


def test_runner_executes_estimators_in_dependency_order(monkeypatch, tmp_path):
    calls = []

    async def fake_stream_process(name, cmd, cwd):
        calls.append((name, cmd[-2:], cwd))
        return 0

    found = {}
    for name in periodic_test_runner.TEST_FILES:
        path = tmp_path / name
        path.write_text("pass", encoding="utf-8")
        found[name] = path

    monkeypatch.setattr(periodic_test_runner, "_stream_process", fake_stream_process)

    asyncio.run(periodic_test_runner._run_once(found, "ETH"))

    assert [name for name, _cmd_tail, _cwd in calls] == ["get_kappa.py", "get_epsilon.py", "get_lambda.py"]
    assert all(cmd_tail == ["--crypto", "ETH"] for _name, cmd_tail, _cwd in calls)


def test_runner_copies_strategy_snapshots_atomically(tmp_path):
    src = tmp_path / "source.json"
    dst = tmp_path / "strategy" / "kappa.json"
    src.write_text(json.dumps({"ETH": {"schema_version": 2}}), encoding="utf-8")
    dst.parent.mkdir()
    dst.write_text("old", encoding="utf-8")

    assert periodic_test_runner._copy_if_needed(src, dst)

    assert json.loads(dst.read_text(encoding="utf-8")) == {"ETH": {"schema_version": 2}}
    assert not dst.with_suffix(".json.tmp").exists()
