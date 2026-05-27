from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_safety_gates  # noqa: E402
from run_safety_gates import local_gates, manual_gate_statuses, plan_status_audit_command, render_markdown  # noqa: E402


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
    max_evidence_age_seconds: float = 86_400.0,
    max_canary_event_age_seconds: float = 604_800.0,
    replay_acceptance_newest_per_stream: int | None = 25,
    replay_acceptance_max_price_events: int | None = 2000,
    replay_acceptance_min_price_events_per_day: float = 1000.0,
    replay_acceptance_max_price_gap_seconds: float = 300.0,
    replay_acceptance_allow_incomplete: bool = True,
    enabled_dry_run_seconds: int = 600,
    enabled_dry_run_min_runtime_seconds: float = 540.0,
    enabled_dry_run_min_event_span_seconds: float = 300.0,
    enabled_dry_run_min_accepted_quotes: int = 2,
    enabled_dry_run_min_order_attempts: int = 2,
    enabled_dry_run_max_loss_rate_usdc_per_hour: float = 6.0,
) -> list[str]:
    for gate_name, command, _ in local_gates(
        include_runtime=include_runtime,
        audit_log_input=audit_log_input,
        manual_monitoring_ack=manual_monitoring_ack,
        post_only_crossing_result=post_only_crossing_result,
        post_only_passive_result=post_only_passive_result,
        use_default_post_only_artifacts=use_default_post_only_artifacts,
        max_evidence_age_seconds=max_evidence_age_seconds,
        max_canary_event_age_seconds=max_canary_event_age_seconds,
        replay_acceptance_newest_per_stream=replay_acceptance_newest_per_stream,
        replay_acceptance_max_price_events=replay_acceptance_max_price_events,
        replay_acceptance_min_price_events_per_day=replay_acceptance_min_price_events_per_day,
        replay_acceptance_max_price_gap_seconds=replay_acceptance_max_price_gap_seconds,
        replay_acceptance_allow_incomplete=replay_acceptance_allow_incomplete,
        enabled_dry_run_seconds=enabled_dry_run_seconds,
        enabled_dry_run_min_runtime_seconds=enabled_dry_run_min_runtime_seconds,
        enabled_dry_run_min_event_span_seconds=enabled_dry_run_min_event_span_seconds,
        enabled_dry_run_min_accepted_quotes=enabled_dry_run_min_accepted_quotes,
        enabled_dry_run_min_order_attempts=enabled_dry_run_min_order_attempts,
        enabled_dry_run_max_loss_rate_usdc_per_hour=enabled_dry_run_max_loss_rate_usdc_per_hour,
    ):
        if gate_name == name:
            return command
    raise AssertionError(f"missing gate {name}")


def test_live_canary_gate_runs_after_dependency_artifacts_with_runtime_gates():
    names = gate_names(include_runtime=True)

    assert names.index("freqtrade_runtime_load") < names.index("freqtrade_callback_surface")
    assert names.index("freqtrade_callback_surface") < names.index("freqtrade_tif_runtime")
    assert names.index("freqtrade_tif_runtime") < names.index("dry_run_disabled_smoke")
    assert names.index("dry_run_enabled_smoke") < names.index("dry_run_quality_report")
    assert names.index("dry_run_quality_report") < names.index("replay_log_calibration_artifact")
    assert names.index("post_only_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("fee_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("hyperliquid_fee_capture_plan") < names.index("fee_evidence_report")
    assert names.index("direct_risk_flatten_plan") < names.index("live_canary_evidence_report")
    assert names.index("replay_acceptance_report_artifact") < names.index("live_canary_evidence_report")
    assert names.index("live_canary_evidence_report") < names.index("promotion_evidence_manifest")


def test_live_canary_gate_is_still_available_without_runtime_gates():
    names = gate_names(include_runtime=False)

    assert "live_canary_evidence_report" in names
    assert names.index("live_canary_evidence_report") < names.index("promotion_evidence_manifest")


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


def test_post_only_gate_passes_evidence_age_window_to_checker():
    command = gate_command(
        "post_only_evidence_report",
        include_runtime=True,
        max_evidence_age_seconds=3600.0,
    )

    assert "--max-evidence-age-seconds" in command
    assert command[command.index("--max-evidence-age-seconds") + 1] == "3600.0"


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
    assert "--min-price-events-per-day" in command
    assert command[command.index("--min-price-events-per-day") + 1] == "1000.0"
    assert "--max-price-gap-seconds" in command
    assert command[command.index("--max-price-gap-seconds") + 1] == "300.0"
    assert "--allow-incomplete" in command


def test_replay_acceptance_gate_can_run_uncapped_promotion_mode_with_custom_coverage_thresholds():
    command = gate_command(
        "replay_acceptance_report_artifact",
        include_runtime=True,
        replay_acceptance_newest_per_stream=None,
        replay_acceptance_max_price_events=None,
        replay_acceptance_min_price_events_per_day=2000.0,
        replay_acceptance_max_price_gap_seconds=120.0,
        replay_acceptance_allow_incomplete=False,
    )

    assert "--newest-per-stream" not in command
    assert "--max-price-events" not in command
    assert command[command.index("--min-price-events-per-day") + 1] == "2000.0"
    assert command[command.index("--max-price-gap-seconds") + 1] == "120.0"
    assert "--allow-incomplete" not in command


def test_runtime_log_artifact_gates_can_use_external_audit_log_input():
    audit_log = Path("docs/testnet_mm_debug.jsonl")

    for gate_name in (
        "dry_run_quality_report",
        "replay_log_calibration_artifact",
        "fee_evidence_report",
        "live_canary_evidence_report",
    ):
        command = gate_command(gate_name, include_runtime=True, audit_log_input=audit_log)
        normalized = [item.replace("\\", "/") for item in command]

        assert "--input" in command
        assert "docs/testnet_mm_debug.jsonl" in normalized


def test_dry_run_quality_gate_checks_quotes_amounts_and_pnl():
    enabled_command = gate_command("dry_run_enabled_smoke", include_runtime=True)
    command = gate_command("dry_run_quality_report", include_runtime=True)
    normalized = [item.replace("\\", "/") for item in command]

    assert enabled_command[enabled_command.index("--seconds") + 1] == "600"
    assert "scripts/verify_dry_run_quality.py" in normalized
    assert "--gate-report" in command
    assert "docs/dry_run_enabled_gate.json" in normalized
    assert command[command.index("--min-runtime-seconds") + 1] == "540.0"
    assert command[command.index("--min-event-span-seconds") + 1] == "300.0"
    assert command[command.index("--min-accepted-quotes") + 1] == "2"
    assert command[command.index("--min-order-attempts") + 1] == "2"
    assert "--max-order-notional-usdc" in command
    assert command[command.index("--max-order-notional-usdc") + 1] == "25"
    assert "--max-order-amount-units" in command
    assert command[command.index("--max-order-amount-units") + 1] == "0.01"
    assert "--max-loss-usdc" in command
    assert command[command.index("--max-loss-usdc") + 1] == "1"
    assert "--max-loss-rate-usdc-per-hour" in command
    assert command[command.index("--max-loss-rate-usdc-per-hour") + 1] == "6.0"


def test_runtime_dry_run_gate_can_be_extended_for_promotion_evidence():
    enabled_command = gate_command(
        "dry_run_enabled_smoke",
        include_runtime=True,
        enabled_dry_run_seconds=1800,
    )
    quality_command = gate_command(
        "dry_run_quality_report",
        include_runtime=True,
        enabled_dry_run_min_runtime_seconds=1620.0,
        enabled_dry_run_min_event_span_seconds=1200.0,
        enabled_dry_run_min_accepted_quotes=5,
        enabled_dry_run_min_order_attempts=5,
        enabled_dry_run_max_loss_rate_usdc_per_hour=2.0,
    )

    assert enabled_command[enabled_command.index("--seconds") + 1] == "1800"
    assert quality_command[quality_command.index("--min-runtime-seconds") + 1] == "1620.0"
    assert quality_command[quality_command.index("--min-event-span-seconds") + 1] == "1200.0"
    assert quality_command[quality_command.index("--min-accepted-quotes") + 1] == "5"
    assert quality_command[quality_command.index("--min-order-attempts") + 1] == "5"
    assert quality_command[quality_command.index("--max-loss-rate-usdc-per-hour") + 1] == "2.0"


def test_freqtrade_tif_runtime_gate_writes_runtime_artifact():
    command = gate_command("freqtrade_tif_runtime", include_runtime=True)
    normalized = [item.replace("\\", "/") for item in command]

    assert "scripts/verify_freqtrade_tif_runtime.py" in normalized
    assert "--output" in command
    assert "docs/freqtrade_tif_runtime_report.json" in normalized


def test_promotion_evidence_manifest_gate_writes_manifest_after_reports():
    command = gate_command("promotion_evidence_manifest", include_runtime=True)
    normalized = [item.replace("\\", "/") for item in command]

    assert "scripts/build_promotion_evidence_manifest.py" in normalized
    assert "--output" in command
    assert "docs/promotion_evidence_manifest.json" in normalized
    assert gate_expected_returncodes("promotion_evidence_manifest", include_runtime=True) == [0]


def test_runtime_evidence_age_windows_are_passed_to_fee_and_canary_checks():
    fee_command = gate_command("fee_evidence_report", include_runtime=True, max_evidence_age_seconds=1800.0)
    canary_command = gate_command(
        "live_canary_evidence_report",
        include_runtime=True,
        max_evidence_age_seconds=1800.0,
        max_canary_event_age_seconds=7200.0,
    )

    assert "--max-evidence-age-seconds" in fee_command
    assert fee_command[fee_command.index("--max-evidence-age-seconds") + 1] == "1800.0"
    assert "--max-dependency-report-age-seconds" in canary_command
    assert canary_command[canary_command.index("--max-dependency-report-age-seconds") + 1] == "1800.0"
    assert "--max-canary-event-age-seconds" in canary_command
    assert canary_command[canary_command.index("--max-canary-event-age-seconds") + 1] == "7200.0"


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


def test_manual_gate_status_rejects_stale_ok_reports(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    stale_report = {"ok": True, "generated_at": "2000-01-01T00:00:00Z", "reasons": []}
    for name in (
        "post_only_evidence_report.json",
        "replay_acceptance_report.json",
        "fee_evidence_report.json",
        "live_canary_report.json",
    ):
        (docs / name).write_text(json.dumps(stale_report), encoding="utf-8")
    monkeypatch.setattr(run_safety_gates, "ROOT", tmp_path)

    statuses = manual_gate_statuses(include_runtime=True)

    assert statuses
    assert all(status["passed"] is False for status in statuses)
    assert all(str(status["freshness_reason"]).startswith("report_stale:") for status in statuses)
    assert all(str(status["freshness_reason"]).endswith(">max_86400.0s") for status in statuses)


def test_manual_gate_status_uses_custom_report_age_window(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    report = {"ok": True, "generated_at": "2000-01-01T00:00:00Z", "reasons": []}
    for name in (
        "post_only_evidence_report.json",
        "replay_acceptance_report.json",
        "fee_evidence_report.json",
        "live_canary_report.json",
    ):
        (docs / name).write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(run_safety_gates, "ROOT", tmp_path)

    statuses = manual_gate_statuses(include_runtime=True, max_report_age_seconds=60.0)

    assert all(status["passed"] is False for status in statuses)
    assert all(str(status["freshness_reason"]).startswith("report_stale:") for status in statuses)
    assert all(str(status["freshness_reason"]).endswith(">max_60.0s") for status in statuses)


def test_post_only_manual_gate_reason_mentions_direct_sdk_fallback():
    statuses = manual_gate_statuses(include_runtime=True)
    post_only = next(status for status in statuses if status["name"] == "hyperliquid_post_only_mapping")

    assert "direct SDK Alo fallback" in str(post_only["reason"])


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
