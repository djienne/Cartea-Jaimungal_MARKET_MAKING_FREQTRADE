from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_safety_gates import local_gates, plan_status_audit_command, render_markdown  # noqa: E402


def gate_names(*, include_runtime: bool) -> list[str]:
    return [name for name, _, _ in local_gates(include_runtime=include_runtime)]


def test_live_canary_gate_runs_after_dependency_artifacts_with_runtime_gates():
    names = gate_names(include_runtime=True)

    assert names.index("post_only_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("fee_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("replay_acceptance_report_artifact") < names.index("live_canary_evidence_report")


def test_live_canary_gate_is_still_available_without_runtime_gates():
    names = gate_names(include_runtime=False)

    assert "live_canary_evidence_report" in names


def test_plan_status_audit_command_uses_latest_gate_json_and_output():
    command = plan_status_audit_command(Path("docs/last_safety_gates.json"), Path("docs/plan_status_audit.json"))
    normalized = [item.replace("\\", "/") for item in command]

    assert "scripts/verify_plan_status.py" in command
    assert "--gates" in command
    assert "docs/last_safety_gates.json" in normalized
    assert "--output" in command
    assert "docs/plan_status_audit.json" in normalized


def test_render_markdown_includes_post_run_audits():
    markdown = render_markdown(
        {
            "all_automated_passed": True,
            "deployment_ready": False,
            "manual_gates_remaining": 1,
            "local_gates": [
                {
                    "name": "compileall",
                    "passed": True,
                    "returncode": 0,
                    "elapsed_seconds": 0.1,
                    "stderr_tail": "",
                }
            ],
            "post_run_audits": [
                {
                    "name": "plan_status_audit",
                    "passed": True,
                    "returncode": 0,
                    "elapsed_seconds": 0.2,
                    "stderr_tail": "",
                }
            ],
            "deployment_blockers": ["live_canary"],
            "manual_gates": [{"name": "live_canary", "reason": "external evidence required", "passed": False}],
        }
    )

    assert "Automated gates: PASS" in markdown
    assert "Deployment ready: NO" in markdown
    assert "Manual gates remaining: 1" in markdown
    assert "Manual/external gate evidence:" in markdown
    assert "WAIT `live_canary`" in markdown
    assert "Post-run audits:" in markdown
    assert "PASS `plan_status_audit`" in markdown
