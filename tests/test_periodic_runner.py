from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "user_data" / "strategies"
if str(STRATEGIES) not in sys.path:
    sys.path.insert(0, str(STRATEGIES))

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
                    "kappa+": 2.0,
                    "kappa-": 2.5,
                    "lambda+": 0.1,
                    "lambda-": 0.2,
                    "n_points_plus": 3,
                    "n_points_minus": 3,
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
                    "lambda+": 0.1,
                    "lambda-": 0.2,
                    "lambda_source": "lambda0_fit",
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
                    "epsilon+": 0.01,
                    "epsilon-": 0.02,
                    "n_buy_events": 3,
                    "n_sell_events": 3,
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
