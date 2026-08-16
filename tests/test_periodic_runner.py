from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
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
                    "schema_version": 3,
                    "status": "ok",
                    "generated_at": "2026-05-25T10:00:00Z",
                    "window_start": "2026-05-25T09:30:00Z",
                    "window_end": "2026-05-25T10:00:00Z",
                    "kappa+": 2.0,
                    "kappa-": 2.5,
                    "lambda+": 0.1,
                    "lambda-": 0.2,
                    "kappa+_raw": 2.1,
                    "kappa-_raw": 2.6,
                    "lambda+_raw": 0.11,
                    "lambda-_raw": 0.21,
                    "depth_p95_plus": 0.4,
                    "depth_p95_minus": 0.3,
                    "depth_max_fitted_plus": 0.6,
                    "depth_max_fitted_minus": 0.5,
                    "sigma2_per_sec": 0.02,
                    "n_quotes": 100,
                    "n_trades": 50,
                    "n_points_plus": 8,
                    "n_points_minus": 8,
                    "r2_plus": 0.5,
                    "r2_minus": 0.6,
                }
            },
        ),
        "lambda.json": write(
            tmp_path / "lambda.json",
            {
                "ETH": {
                    "schema_version": 3,
                    "status": "ok",
                    "generated_at": "2026-05-25T10:00:00Z",
                    "window_start": "2026-05-25T09:30:00Z",
                    "window_end": "2026-05-25T10:00:00Z",
                    "lambda+": 0.1,
                    "lambda-": 0.2,
                    "lambda+_raw": 0.11,
                    "lambda-_raw": 0.21,
                    "lambda_source": "mo_survival_fit",
                    "n_trades": 50,
                }
            },
        ),
        "epsilon.json": write(
            tmp_path / "epsilon.json",
            {
                "ETH": {
                    "schema_version": 3,
                    "status": "ok",
                    "generated_at": "2026-05-25T10:00:00Z",
                    "window_start": "2026-05-25T09:30:00Z",
                    "window_end": "2026-05-25T10:00:00Z",
                    "epsilon+": 0.01,
                    "epsilon-": 0.02,
                    "epsilon+_raw": 0.011,
                    "epsilon-_raw": 0.021,
                    "window_ms": 5000,
                    "estimator": "trimmed_mean",
                    "unit": "USDC",
                    "n_buy_events": 60,
                    "n_sell_events": 60,
                    "toxicity_plus": 0.02,
                    "toxicity_minus": 0.05,
                }
            },
        ),
    }


def test_runner_accepts_valid_schema_v3_snapshots(tmp_path):
    assert _validate_symbol_snapshot(valid_snapshots(tmp_path), "ETH") == (True, "ok")


def test_runner_rejects_schema_v2_snapshots(tmp_path):
    snapshots = valid_snapshots(tmp_path)
    payload = json.loads(snapshots["kappa.json"].read_text(encoding="utf-8"))
    payload["ETH"]["schema_version"] = 2
    snapshots["kappa.json"].write_text(json.dumps(payload), encoding="utf-8")

    ok, reason = _validate_symbol_snapshot(snapshots, "ETH")

    assert not ok
    assert reason == "unsupported_schema_kappa.json"


def test_runner_rejects_missing_raw_keys(tmp_path):
    snapshots = valid_snapshots(tmp_path)
    payload = json.loads(snapshots["kappa.json"].read_text(encoding="utf-8"))
    del payload["ETH"]["kappa+_raw"]
    snapshots["kappa.json"].write_text(json.dumps(payload), encoding="utf-8")

    ok, reason = _validate_symbol_snapshot(snapshots, "ETH")

    assert not ok
    assert reason == "missing_kappa+_raw_in_kappa.json"


def test_runner_accepts_null_sigma2_but_rejects_negative(tmp_path):
    snapshots = valid_snapshots(tmp_path)
    payload = json.loads(snapshots["kappa.json"].read_text(encoding="utf-8"))
    payload["ETH"]["sigma2_per_sec"] = None
    snapshots["kappa.json"].write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_symbol_snapshot(snapshots, "ETH") == (True, "ok")

    payload["ETH"]["sigma2_per_sec"] = -0.5
    snapshots["kappa.json"].write_text(json.dumps(payload), encoding="utf-8")
    ok, reason = _validate_symbol_snapshot(snapshots, "ETH")
    assert not ok
    assert reason == "invalid_sigma2_per_sec_in_kappa.json"


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
    assert reason == "lambda_json_must_be_mo_survival_fit"


def test_runner_drives_a_single_estimator_process_per_cycle(monkeypatch, tmp_path):
    """One process per cycle, not three.

    The three estimators each re-read the same parquet shards, so a cycle paid
    for the same directory scan three times and could outrun its own interval.
    estimate_all.py loads the window once and drives all three from it; the
    kappa-before-epsilon ordering that used to live here is now asserted in
    test_estimate_all.py, where it actually lives.
    """
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

    assert [name for name, _cmd_tail, _cwd in calls] == ["estimate_all.py"]
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


def test_process_lock_is_exclusive_and_released(tmp_path):
    lock_path = tmp_path / "param_update.lock"

    token = periodic_test_runner._acquire_process_lock(lock_path, "ETH")
    assert token
    assert periodic_test_runner._acquire_process_lock(lock_path, "ETH") is None
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["crypto"] == "ETH"
    assert payload["pid"] > 0
    assert payload["started_at"]
    assert payload["token"] == token

    periodic_test_runner._release_process_lock(lock_path, token)

    assert not lock_path.exists()


def test_stale_process_lock_is_replaced(tmp_path):
    lock_path = tmp_path / "param_update.lock"
    stale_started_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat().replace("+00:00", "Z")
    lock_path.write_text(
        json.dumps({"pid": 123, "crypto": "ETH", "started_at": stale_started_at}),
        encoding="utf-8",
    )

    token = periodic_test_runner._acquire_process_lock(lock_path, "ETH", stale_after_seconds=60)
    assert token

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] != 123
    assert payload["crypto"] == "ETH"

    periodic_test_runner._release_process_lock(lock_path, token)
    assert not lock_path.exists()


def test_release_with_wrong_token_leaves_foreign_lock(tmp_path):
    lock_path = tmp_path / "param_update.lock"
    token = periodic_test_runner._acquire_process_lock(lock_path, "ETH")
    assert token

    periodic_test_runner._release_process_lock(lock_path, "not-the-owner")
    assert lock_path.exists()

    periodic_test_runner._release_process_lock(lock_path, None)
    assert lock_path.exists()

    periodic_test_runner._release_process_lock(lock_path, token)
    assert not lock_path.exists()


def test_sigterm_handler_raises_keyboard_interrupt():
    # Docker stops the sidecar with SIGTERM; the handler must convert it into
    # KeyboardInterrupt so the lock-release/status finally paths run instead of
    # the process dying with param_update.lock left behind.
    try:
        periodic_test_runner._sigterm_to_keyboard_interrupt(15, None)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt from SIGTERM handler")


def test_run_once_skip_does_not_delete_foreign_lock(tmp_path, monkeypatch):
    # A --once invocation that cannot acquire the lock (another runner is
    # mid-cycle) must exit WITHOUT deleting that runner's active lock —
    # otherwise the strategy loses its estimator_running gate during the
    # foreign runner's snapshot copy.
    lock_path = tmp_path / "param_update.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 123,
                "crypto": "ETH",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "token": "foreign-owner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(periodic_test_runner, "_runner_lock_path", lambda: lock_path)

    periodic_test_runner.schedule_tests(run_once=True, crypto="ETH")

    assert lock_path.exists()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "foreign-owner"


def test_fresh_process_lock_is_not_replaced(tmp_path):
    lock_path = tmp_path / "param_update.lock"
    lock_path.write_text(
        json.dumps({"pid": 123, "crypto": "ETH", "started_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )

    assert not periodic_test_runner._acquire_process_lock(lock_path, "ETH", stale_after_seconds=60)
    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == 123
