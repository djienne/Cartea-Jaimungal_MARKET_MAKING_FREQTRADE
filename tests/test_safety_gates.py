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


def gate_expected_returncodes(name: str, *, include_runtime: bool) -> list[int]:
    for gate_name, _, expected_returncodes in local_gates(include_runtime=include_runtime):
        if gate_name == name:
            return expected_returncodes
    raise AssertionError(f"missing gate {name}")


def gate_command(
    name: str,
    *,
    include_runtime: bool,
    audit_log_input: Path = Path("user_data/logs/mm_debug.jsonl"),
    manual_monitoring_ack: bool = False,
    post_only_crossing_result: Path | None = None,
    post_only_passive_result: Path | None = None,
    use_default_post_only_artifacts: bool = False,
    replay_acceptance_newest_per_stream: int | None = 25,
    replay_acceptance_max_price_events: int | None = 2000,
    replay_acceptance_allow_incomplete: bool = True,
) -> list[str]:
    for gate_name, command, _ in local_gates(
        include_runtime=include_runtime,
        audit_log_input=audit_log_input,
        manual_monitoring_ack=manual_monitoring_ack,
        post_only_crossing_result=post_only_crossing_result,
        post_only_passive_result=post_only_passive_result,
        use_default_post_only_artifacts=use_default_post_only_artifacts,
        replay_acceptance_newest_per_stream=replay_acceptance_newest_per_stream,
        replay_acceptance_max_price_events=replay_acceptance_max_price_events,
        replay_acceptance_allow_incomplete=replay_acceptance_allow_incomplete,
    ):
        if gate_name == name:
            return command
    raise AssertionError(f"missing gate {name}")


def test_live_canary_gate_runs_after_dependency_artifacts_with_runtime_gates():
    names = gate_names(include_runtime=True)

    assert names.index("post_only_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("fee_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("hyperliquid_fee_capture_plan") < names.index("fee_evidence_report")
    assert names.index("replay_acceptance_report_artifact") < names.index("live_canary_evidence_report")


def test_live_canary_gate_is_still_available_without_runtime_gates():
    names = gate_names(include_runtime=False)

    assert "live_canary_evidence_report" in names


def test_external_evidence_report_gates_accept_incomplete_and_complete_returncodes():
    assert gate_expected_returncodes("post_only_evidence_report", include_runtime=True) == [0, 1]
    assert gate_expected_returncodes("fee_evidence_report", include_runtime=True) == [0, 1]
    assert gate_expected_returncodes("live_canary_evidence_report", include_runtime=True) == [0, 1]


def test_post_only_gate_can_pass_external_evidence_artifacts():
    command = gate_command(
        "post_only_evidence_report",
        include_runtime=True,
        post_only_crossing_result=Path("docs/post_only_crossing_result.json"),
        post_only_passive_result=Path("docs/post_only_passive_result.json"),
    )
    normalized = [item.replace("\\", "/") for item in command]

    assert "--crossing-result" in command
    assert "docs/post_only_crossing_result.json" in normalized
    assert "--passive-result" in command
    assert "docs/post_only_passive_result.json" in normalized


def test_post_only_gate_keeps_incomplete_artifact_check_when_no_evidence_is_supplied():
    command = gate_command("post_only_evidence_report", include_runtime=True)

    assert "--crossing-result" not in command
    assert "--passive-result" not in command


def test_replay_acceptance_gate_defaults_to_short_artifact_mode():
    command = gate_command("replay_acceptance_report_artifact", include_runtime=True)

    assert "--newest-per-stream" in command
    assert command[command.index("--newest-per-stream") + 1] == "25"
    assert "--max-price-events" in command
    assert command[command.index("--max-price-events") + 1] == "2000"
    assert "--allow-incomplete" in command


def test_replay_acceptance_gate_can_run_uncapped_promotion_mode():
    command = gate_command(
        "replay_acceptance_report_artifact",
        include_runtime=True,
        replay_acceptance_newest_per_stream=None,
        replay_acceptance_max_price_events=None,
        replay_acceptance_allow_incomplete=False,
    )

    assert "--newest-per-stream" not in command
    assert "--max-price-events" not in command
    assert "--allow-incomplete" not in command


def test_runtime_log_artifact_gates_can_use_external_audit_log_input():
    audit_log = Path("docs/testnet_mm_debug.jsonl")

    for gate_name in ("replay_log_calibration_artifact", "fee_evidence_report", "live_canary_evidence_report"):
        command = gate_command(gate_name, include_runtime=True, audit_log_input=audit_log)
        normalized = [item.replace("\\", "/") for item in command]

        assert "--input" in command
        assert "docs/testnet_mm_debug.jsonl" in normalized


def test_fee_capture_plan_gate_is_read_only_plan_mode():
    command = gate_command("hyperliquid_fee_capture_plan", include_runtime=True)
    normalized = [item.replace("\\", "/") for item in command]

    assert "scripts/capture_hyperliquid_fee_evidence.py" in command
    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "plan"
    assert "--output" in command
    assert "docs/hyperliquid_fee_capture_plan.json" in normalized


def test_non_runtime_live_canary_gate_can_use_external_audit_log_input():
    command = gate_command(
        "live_canary_evidence_report",
        include_runtime=False,
        audit_log_input=Path("docs/live_canary_mm_debug.jsonl"),
    )
    normalized = [item.replace("\\", "/") for item in command]

    assert "--input" in command
    assert "docs/live_canary_mm_debug.jsonl" in normalized


def test_live_canary_gate_does_not_ack_manual_monitoring_by_default():
    command = gate_command("live_canary_evidence_report", include_runtime=True)

    assert "--manual-monitoring-ack" not in command


def test_live_canary_gate_can_pass_manual_monitoring_acknowledgement():
    command = gate_command(
        "live_canary_evidence_report",
        include_runtime=True,
        manual_monitoring_ack=True,
    )

    assert "--manual-monitoring-ack" in command


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
