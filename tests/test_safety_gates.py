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
    replay_acceptance_price_tick_size: float = 0.0,
    replay_acceptance_amount_step_size: float = 0.0,
    replay_acceptance_fill_calibration: Path | None = Path("docs/replay_log_calibration.json"),
    replay_acceptance_require_fill_calibration: bool = False,
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
        replay_acceptance_price_tick_size=replay_acceptance_price_tick_size,
        replay_acceptance_amount_step_size=replay_acceptance_amount_step_size,
        replay_acceptance_fill_calibration=replay_acceptance_fill_calibration,
        replay_acceptance_require_fill_calibration=replay_acceptance_require_fill_calibration,
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
    assert names.index("adapter_plans") < names.index("fee_evidence_report")
    assert names.index("adapter_plans") < names.index("live_canary_evidence_report")
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
    assert "--price-tick-size" in command
    assert command[command.index("--price-tick-size") + 1] == "0.0"
    assert "--amount-step-size" in command
    assert command[command.index("--amount-step-size") + 1] == "0.0"
    assert "--fill-calibration" in command
    assert command[command.index("--fill-calibration") + 1].replace("\\", "/") == "docs/replay_log_calibration.json"
    assert "--allow-incomplete" in command


def test_replay_acceptance_gate_can_run_uncapped_promotion_mode_with_custom_coverage_and_calibration_thresholds():
    command = gate_command(
        "replay_acceptance_report_artifact",
        include_runtime=True,
        replay_acceptance_newest_per_stream=None,
        replay_acceptance_max_price_events=None,
        replay_acceptance_min_price_events_per_day=2000.0,
        replay_acceptance_max_price_gap_seconds=120.0,
        replay_acceptance_price_tick_size=0.1,
        replay_acceptance_amount_step_size=0.001,
        replay_acceptance_require_fill_calibration=True,
        replay_acceptance_allow_incomplete=False,
    )

    assert "--newest-per-stream" not in command
    assert "--max-price-events" not in command
    assert command[command.index("--min-price-events-per-day") + 1] == "2000.0"
    assert command[command.index("--max-price-gap-seconds") + 1] == "120.0"
    assert command[command.index("--price-tick-size") + 1] == "0.1"
    assert command[command.index("--amount-step-size") + 1] == "0.001"
    assert "--require-fill-calibration" in command
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


def test_adapter_plans_gate_runs_consolidated_plan_script():
    command = gate_command("adapter_plans", include_runtime=False)
    normalized = [item.replace("\\", "/") for item in command]

    assert "scripts/run_adapter_plans.py" in normalized


def test_adapter_plans_covers_all_former_plan_gates_read_only():
    from run_adapter_plans import sub_plan_commands

    plans = dict(sub_plan_commands("python"))
    assert set(plans) == {
        "post_only_probe_plan",
        "direct_alo_adapter_plan",
        "direct_alo_probe_preparation_plan",
        # The ask path has no exchange evidence of its own; a two-sided
        # maker rests an ask continuously, so it gets its own probe.
        "direct_alo_probe_preparation_plan_ask",
        "direct_risk_flatten_plan",
        "hyperliquid_fee_capture_plan",
    }

    fee_plan = plans["hyperliquid_fee_capture_plan"]
    assert "scripts/capture_hyperliquid_fee_evidence.py" in fee_plan
    assert fee_plan[fee_plan.index("--mode") + 1] == "plan"
    assert "docs/hyperliquid_fee_capture_plan.json" in [item.replace("\\", "/") for item in fee_plan]

    probe_plan = plans["direct_alo_probe_preparation_plan"]
    assert probe_plan[probe_plan.index("--mode") + 1] == "prepare-probes"
    assert "--acknowledge-real-orders" not in probe_plan
    assert "--best-bid" in probe_plan and "--best-ask" in probe_plan
    assert "--price-tick-size" in probe_plan and "--amount-step-size" in probe_plan
    assert "docs/direct_alo_probe_commands.json" in [item.replace("\\", "/") for item in probe_plan]

    # Every sub-plan stays read-only plan/preparation mode.
    for name, plan_command in plans.items():
        assert "--acknowledge-real-orders" not in plan_command, name
        assert "submit" not in " ".join(plan_command), name


def test_non_runtime_live_canary_gate_can_use_external_audit_log_input():
    command = gate_command(
        "live_canary_evidence_report",
        include_runtime=False,
        audit_log_input=Path("docs/live_canary_mm_debug.jsonl"),
    )
    normalized = [item.replace("\\", "/") for item in command]

    assert "--input" in command
    assert "docs/live_canary_mm_debug.jsonl" in normalized


# ---------------------------------------------------------------------------
# --reuse-smoke-artifacts: evaluator reruns against prior smoke evidence
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

from run_safety_gates import (  # noqa: E402
    production_restore_commands,
    reused_smoke_results,
)


def test_reuse_mode_skips_only_the_two_long_smokes():
    full = [name for name, _, _ in local_gates(include_runtime=True)]
    reuse = [name for name, _, _ in local_gates(include_runtime=True, reuse_smoke_artifacts=True)]

    assert set(full) - set(reuse) == {"dry_run_disabled_smoke", "dry_run_enabled_smoke"}
    # Everything cheap still runs live, including the quality evaluator.
    assert "dry_run_quality_report" in reuse
    assert "pytest_core" in reuse


def _write_previous_gates(path: Path, *, disabled_passed=True, enabled_passed=True) -> None:
    payload = {
        "local_gates": [
            {
                "name": "dry_run_disabled_smoke",
                "command": ["python", "scripts/verify_dry_run_disabled.py"],
                "passed": disabled_passed,
                "returncode": 0 if disabled_passed else 1,
                "expected_returncodes": [0],
                "elapsed_seconds": 225.0,
                "stdout_tail": "",
                "stderr_tail": "",
            },
            {
                "name": "dry_run_enabled_smoke",
                "command": ["python", "scripts/verify_dry_run_enabled.py"],
                "passed": enabled_passed,
                "returncode": 0 if enabled_passed else 1,
                "expected_returncodes": [0],
                "elapsed_seconds": 674.0,
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_smoke_artifacts(tmp_path: Path, monkeypatch, *, age_seconds=60.0, passed=True) -> None:
    started = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat().replace("+00:00", "Z")
    artifacts = {}
    for name in ("dry_run_disabled_smoke", "dry_run_enabled_smoke"):
        artifact_path = tmp_path / f"{name}.json"
        artifact_path.write_text(json.dumps({"passed": passed, "started_at": started}), encoding="utf-8")
        artifacts[name] = artifact_path
    monkeypatch.setattr(run_safety_gates, "SMOKE_GATE_ARTIFACTS", artifacts)


def test_reused_smoke_results_happy_path(tmp_path, monkeypatch):
    previous = tmp_path / "last_safety_gates.json"
    _write_previous_gates(previous)
    _write_smoke_artifacts(tmp_path, monkeypatch, age_seconds=60.0)

    entries = reused_smoke_results(previous, max_age_seconds=21_600.0)

    assert [entry["name"] for entry in entries] == ["dry_run_disabled_smoke", "dry_run_enabled_smoke"]
    assert all(entry["passed"] is True for entry in entries)
    assert all(entry["reused"] is True for entry in entries)
    assert all(str(entry["stdout_tail"]).startswith("reused_from:") for entry in entries)


def test_reused_smoke_results_fail_closed_on_stale_artifact(tmp_path, monkeypatch):
    previous = tmp_path / "last_safety_gates.json"
    _write_previous_gates(previous)
    _write_smoke_artifacts(tmp_path, monkeypatch, age_seconds=10_000.0)

    entries = reused_smoke_results(previous, max_age_seconds=3_600.0)

    assert all(entry["passed"] is False for entry in entries)
    assert all("artifact_stale" in entry["stderr_tail"] for entry in entries)


def test_reused_smoke_results_fail_closed_on_missing_previous_gates(tmp_path, monkeypatch):
    _write_smoke_artifacts(tmp_path, monkeypatch, age_seconds=60.0)

    entries = reused_smoke_results(tmp_path / "nope.json", max_age_seconds=21_600.0)

    assert all(entry["passed"] is False for entry in entries)
    assert all("previous_gates_unreadable" in entry["stderr_tail"] for entry in entries)


def test_reused_smoke_results_fail_closed_on_failed_prior_gate(tmp_path, monkeypatch):
    previous = tmp_path / "last_safety_gates.json"
    _write_previous_gates(previous, enabled_passed=False)
    _write_smoke_artifacts(tmp_path, monkeypatch, age_seconds=60.0)

    entries = {entry["name"]: entry for entry in reused_smoke_results(previous, max_age_seconds=21_600.0)}

    assert entries["dry_run_disabled_smoke"]["passed"] is True
    assert entries["dry_run_enabled_smoke"]["passed"] is False
    assert "previous_gate_not_passed" in entries["dry_run_enabled_smoke"]["stderr_tail"]


def test_reused_smoke_results_fail_closed_on_failed_artifact(tmp_path, monkeypatch):
    previous = tmp_path / "last_safety_gates.json"
    _write_previous_gates(previous)
    _write_smoke_artifacts(tmp_path, monkeypatch, age_seconds=60.0, passed=False)

    entries = reused_smoke_results(previous, max_age_seconds=21_600.0)

    assert all(entry["passed"] is False for entry in entries)
    assert all("artifact_passed_not_true" in entry["stderr_tail"] for entry in entries)


def test_production_restore_commands_match_documented_recovery():
    commands = production_restore_commands()

    # The smokes replace MM_ADV and stop the collector on teardown; restore
    # brings both back.
    assert commands == [
        ["docker", "rm", "-f", "MM_ADV"],
        ["docker", "compose", "up", "-d", "freqtrade", "hl-collector2"],
    ]


def test_reuse_mode_can_point_quality_gate_at_archived_audit_log():
    for gate_name, command, _expected in local_gates(
        include_runtime=True,
        reuse_smoke_artifacts=True,
        quality_audit_log_input=Path("user_data/logs/mm_gate_enabled_audit.jsonl"),
    ):
        if gate_name == "dry_run_quality_report":
            normalized = [item.replace("\\", "/") for item in command]
            assert "user_data/logs/mm_gate_enabled_audit.jsonl" in normalized
            return
    raise AssertionError("missing dry_run_quality_report gate")


def test_reuse_mode_drops_quality_gate_for_legacy_artifacts_without_archive():
    names = [
        name
        for name, _, _ in local_gates(
            include_runtime=True,
            reuse_smoke_artifacts=True,
            reuse_quality_report=True,
        )
    ]

    assert "dry_run_quality_report" not in names
    assert "dry_run_disabled_smoke" not in names
    assert "dry_run_enabled_smoke" not in names


def test_reused_quality_result_validates_report_artifact(tmp_path, monkeypatch):
    from run_safety_gates import reused_quality_result

    previous = tmp_path / "last_safety_gates.json"
    previous.write_text(
        json.dumps(
            {
                "local_gates": [
                    {
                        "name": "dry_run_quality_report",
                        "command": ["python", "scripts/verify_dry_run_quality.py"],
                        "passed": True,
                        "returncode": 0,
                        "expected_returncodes": [0],
                        "elapsed_seconds": 0.1,
                        "stdout_tail": "",
                        "stderr_tail": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "dry_run_quality_report.json"
    fresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report.write_text(json.dumps({"ok": True, "generated_at": fresh}), encoding="utf-8")
    monkeypatch.setattr(run_safety_gates, "QUALITY_REPORT_ARTIFACT", report)

    entry = reused_quality_result(previous, max_age_seconds=21_600.0)
    assert entry["passed"] is True
    assert entry["reused"] is True

    report.write_text(json.dumps({"ok": False, "generated_at": fresh}), encoding="utf-8")
    entry = reused_quality_result(previous, max_age_seconds=21_600.0)
    assert entry["passed"] is False
    assert "artifact_ok_not_true" in entry["stderr_tail"]


def test_render_markdown_reports_reuse_and_restore_status():
    payload = {
        "all_automated_passed": True,
        "deployment_ready": False,
        "manual_gates_remaining": 4,
        "smoke_artifacts_reused": True,
        "production_restore": {"attempted": True, "ok": True},
        "local_gates": [
            {
                "name": "dry_run_enabled_smoke",
                "passed": True,
                "elapsed_seconds": 0.0,
                "returncode": 0,
                "stderr_tail": "",
                "reused": True,
            }
        ],
        "manual_gates": [],
    }

    markdown = render_markdown(payload)

    assert "Smoke artifacts: REUSED" in markdown
    assert "Production restore: OK" in markdown
    assert "(reused)" in markdown


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
