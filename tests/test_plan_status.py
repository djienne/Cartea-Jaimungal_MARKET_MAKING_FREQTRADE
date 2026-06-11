from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_plan_status import build_plan_status_audit  # noqa: E402


def gate_payload(pytest_count: int = 173) -> dict:
    required_local = [
        "compileall",
        "pytest_core",
        "config_safety_report",
        "strategy_safety_report",
        "compute_spreads_boundary_smoke",
        "replay_smoke",
        "adapter_plans",
        "post_only_evidence_report",
        "promotion_evidence_manifest",
        "docker_compose_config",
        "freqtrade_runtime_load",
        "freqtrade_callback_surface",
        "freqtrade_tif_runtime",
        "dry_run_disabled_smoke",
        "dry_run_enabled_smoke",
        "dry_run_quality_report",
        "replay_log_calibration_artifact",
        "fee_evidence_report",
        "hl_data_validation_report",
        "replay_latest_data_smoke",
        "replay_acceptance_report_artifact",
        "live_canary_evidence_report",
    ]
    manual_gates = [
        {"name": "hyperliquid_post_only_mapping", "passed": False},
        {"name": "multi_day_event_replay", "passed": False},
        {"name": "hyperliquid_fee_tier", "passed": False},
        {"name": "live_canary", "passed": False},
    ]
    deployment_blockers = [gate["name"] for gate in manual_gates]
    return {
        "all_local_passed": True,
        "all_automated_passed": True,
        "deployment_ready": False,
        "deployment_blockers": deployment_blockers,
        "manual_gates_remaining": len(manual_gates),
        "runtime_gates_included": True,
        "local_gates": [
            {
                "name": name,
                "passed": True,
                "stdout_tail": f"{pytest_count} passed" if name == "pytest_core" else "",
            }
            for name in required_local
        ],
        "manual_gates": manual_gates,
    }


def status_text(test_count: int = 173) -> str:
    return f"""
- Unit/integration tests: `python -m pytest tests` passed with {test_count} tests.
- Checked-in config safety:
  `scripts/verify_config_safety.py` writes a report.
- Checked-in strategy safety:
  `scripts/verify_strategy_safety.py` writes a report.
- Runtime load gate:
  `freqtrade_runtime_load` proves the strategy loads in the Freqtrade runtime.
- Runtime callback surface gate:
  `freqtrade_callback_surface` imports the strategy in the Freqtrade container and checks callback signatures.
- Runtime TIF compatibility gate:
  `freqtrade_tif_runtime` records whether GTC, PO, and Alo configs load in the Freqtrade container.
- Locked dry-run evidence:
  the disabled dry-run smoke proves zero orders while locked.
- Enabled dry-run evidence:
  the enabled dry-run smoke proves model-valid dry-run order creation.
- Dry-run quality report:
  `scripts/verify_dry_run_quality.py` checks quote distance, order size, runtime, and PnL bounds.
- Replay log calibration:
  runtime logs produce calibration artifacts for replay fill assumptions.
- Hyperliquid fee evidence capture:
  read-only capture normalizes userFees/userFills evidence.
- Promotion evidence manifest:
  `docs/promotion_evidence_manifest.json` summarizes dry-run quality and remaining external evidence commands.
- Hyperliquid risk flatten scaffold:
  direct reduce-only IOC flatten plans are generated without network access.
## Remaining Required Gates
- Hyperliquid post-only/Alo integration:
- Hyperliquid account fee tier:
- Multi-day event replay:
- Live canary:
"""


def audit_for(status: str, payload: dict):
    return build_plan_status_audit(
        status,
        payload,
        status_path=Path("docs/PLAN_IMPLEMENTATION_STATUS.md"),
        gates_path=Path("docs/last_safety_gates.json"),
    )


def test_plan_status_audit_passes_when_status_matches_gate_evidence():
    report = audit_for(status_text(), gate_payload())

    assert report["ok"] is True
    assert report["reasons"] == []


def test_plan_status_audit_rejects_stale_test_count():
    report = audit_for(status_text(test_count=163), gate_payload(pytest_count=173))

    assert report["ok"] is False
    assert "pytest_count_mismatch:163!=173" in report["reasons"]


def test_plan_status_audit_rejects_missing_required_gate():
    payload = gate_payload()
    payload["local_gates"] = [
        gate for gate in payload["local_gates"] if gate["name"] != "dry_run_enabled_smoke"
    ]

    report = audit_for(status_text(), payload)

    assert report["ok"] is False
    assert "missing_local_gate:dry_run_enabled_smoke" in report["reasons"]


def test_plan_status_audit_rejects_missing_remaining_gate_text():
    report = audit_for(status_text().replace("- Live canary:", ""), gate_payload())

    assert report["ok"] is False
    assert "status_missing_phrase:live_canary_gate" in report["reasons"]


def test_plan_status_audit_rejects_deployment_ready_with_manual_blockers():
    payload = gate_payload()
    payload["deployment_ready"] = True

    report = audit_for(status_text(), payload)

    assert report["ok"] is False
    assert "deployment_ready_true_before_manual_gates" in report["reasons"]


def test_plan_status_audit_rejects_manual_gate_count_drift():
    payload = gate_payload()
    payload["manual_gates_remaining"] = 0

    report = audit_for(status_text(), payload)

    assert report["ok"] is False
    assert "manual_gates_remaining_mismatch:0!=4" in report["reasons"]


def test_plan_status_audit_accepts_deployment_ready_when_manual_gates_pass():
    payload = gate_payload()
    for gate in payload["manual_gates"]:
        gate["passed"] = True
    payload["deployment_blockers"] = []
    payload["manual_gates_remaining"] = 0
    payload["deployment_ready"] = True

    report = audit_for(status_text(), payload)

    assert report["ok"] is True
    assert report["deployment"]["ready"] is True
    assert report["deployment"]["blockers"] == []


def test_plan_status_audit_rejects_blocker_status_drift():
    payload = gate_payload()
    payload["manual_gates"][0]["passed"] = True

    report = audit_for(status_text(), payload)

    assert report["ok"] is False
    assert "deployment_blockers_do_not_match_failed_manual_gates" in report["reasons"]
