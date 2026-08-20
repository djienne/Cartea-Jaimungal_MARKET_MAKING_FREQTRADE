#!/usr/bin/env python3
"""Verify PLAN implementation status matches the latest safety-gate evidence."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATUS = Path("docs/PLAN_IMPLEMENTATION_STATUS.md")
DEFAULT_GATES = Path("docs/last_safety_gates.json")
DEFAULT_OUTPUT = Path("docs/plan_status_audit.json")


REQUIRED_LOCAL_GATES = {
    "compileall",
    "pytest_core",
    "config_safety_report",
    "strategy_safety_report",
    # Added 2026-08-19 with scripts/verify_strategy_attributes.py. It runs and
    # passes; without it listed here the audit would not notice the gate
    # disappearing, which is the whole job of this set.
    "strategy_attribute_report",
    "compute_spreads_boundary_smoke",
    "replay_smoke",
    # adapter_plans consolidates the former post_only_probe_plan,
    # direct_alo_adapter_plan, direct_alo_probe_preparation_plan,
    # direct_risk_flatten_plan and hyperliquid_fee_capture_plan gates
    # (see scripts/run_adapter_plans.py).
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
}

REQUIRED_MANUAL_GATES = {
    "hyperliquid_post_only_mapping",
    "multi_day_event_replay",
    "hyperliquid_fee_tier",
    "live_canary",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing_json"
    except Exception as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    return payload, None


def read_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return "", "missing_status"
    except Exception as exc:
        return "", f"status_read_error:{exc}"


def extract_pytest_count(gates_payload: dict[str, Any]) -> int | None:
    for gate in gates_payload.get("local_gates", []):
        if not isinstance(gate, dict) or gate.get("name") != "pytest_core":
            continue
        text = str(gate.get("stdout_tail") or "")
        match = re.search(r"(\d+)\s+passed", text)
        if match:
            return int(match.group(1))
    return None


def extract_status_test_count(status_text: str) -> int | None:
    match = re.search(r"passed with\s+(\d+)\s+tests", status_text)
    return int(match.group(1)) if match else None


def build_plan_status_audit(
    status_text: str,
    gates_payload: dict[str, Any] | None,
    *,
    status_path: Path,
    gates_path: Path,
    status_error: str | None = None,
    gates_error: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if status_error:
        reasons.append(status_error)
    if gates_error:
        reasons.append(gates_error)
    gates_payload = gates_payload or {}

    if gates_payload.get("all_local_passed") is not True:
        reasons.append("latest_safety_gates_not_all_passed")
    if gates_payload.get("all_automated_passed") is not True:
        reasons.append("latest_automated_gates_not_all_passed")
    if gates_payload.get("runtime_gates_included") is not True:
        reasons.append("latest_safety_gates_missing_runtime")

    local_gate_names = {
        str(gate.get("name"))
        for gate in gates_payload.get("local_gates", [])
        if isinstance(gate, dict)
    }
    missing_local_gates = sorted(REQUIRED_LOCAL_GATES - local_gate_names)
    for name in missing_local_gates:
        reasons.append(f"missing_local_gate:{name}")

    failed_local_gates = sorted(
        str(gate.get("name"))
        for gate in gates_payload.get("local_gates", [])
        if isinstance(gate, dict) and gate.get("passed") is not True
    )
    for name in failed_local_gates:
        reasons.append(f"local_gate_not_passed:{name}")

    manual_gate_names = {
        str(gate.get("name"))
        for gate in gates_payload.get("manual_gates", [])
        if isinstance(gate, dict)
    }
    manual_gates_remaining = gates_payload.get("manual_gates_remaining")
    deployment_blockers = {
        str(name)
        for name in gates_payload.get("deployment_blockers", [])
        if name not in (None, "")
    }
    failed_manual_gates = {
        str(gate.get("name"))
        for gate in gates_payload.get("manual_gates", [])
        if isinstance(gate, dict) and gate.get("passed") is not True
    }
    if manual_gates_remaining != len(deployment_blockers):
        reasons.append(f"manual_gates_remaining_mismatch:{manual_gates_remaining}!={len(deployment_blockers)}")
    if deployment_blockers != failed_manual_gates:
        reasons.append("deployment_blockers_do_not_match_failed_manual_gates")
    if gates_payload.get("deployment_ready") is True and deployment_blockers:
        reasons.append("deployment_ready_true_before_manual_gates")
    if (
        gates_payload.get("deployment_ready") is not True
        and gates_payload.get("all_automated_passed") is True
        and not deployment_blockers
    ):
        reasons.append("deployment_ready_false_without_blockers")
    missing_manual_gates = sorted(REQUIRED_MANUAL_GATES - manual_gate_names)
    for name in missing_manual_gates:
        reasons.append(f"missing_manual_gate:{name}")
    unexpected_deployment_blockers = sorted(deployment_blockers - REQUIRED_MANUAL_GATES)
    for name in unexpected_deployment_blockers:
        reasons.append(f"unexpected_deployment_blocker:{name}")

    pytest_count = extract_pytest_count(gates_payload)
    status_count = extract_status_test_count(status_text)
    if pytest_count is None:
        reasons.append("pytest_count_missing_from_gates")
    if status_count is None:
        reasons.append("pytest_count_missing_from_status")
    if pytest_count is not None and status_count is not None and pytest_count != status_count:
        reasons.append(f"pytest_count_mismatch:{status_count}!={pytest_count}")

    required_status_phrases = {
        "config_safety_report": "`scripts/verify_config_safety.py`",
        "strategy_safety_report": "`scripts/verify_strategy_safety.py`",
        "locked_dry_run_gate": "Locked dry-run evidence",
        "enabled_dry_run_gate": "Enabled dry-run evidence",
        "dry_run_quality_report": "`scripts/verify_dry_run_quality.py`",
        "runtime_load_gate": "`freqtrade_runtime_load`",
        "callback_surface_gate": "`freqtrade_callback_surface`",
        "tif_runtime_gate": "`freqtrade_tif_runtime`",
        "post_only_gate": "Hyperliquid post-only/Alo integration",
        "fee_gate": "Hyperliquid account fee tier",
        "fee_capture_gate": "Hyperliquid fee evidence capture",
        "promotion_manifest": "`docs/promotion_evidence_manifest.json`",
        "risk_flatten_gate": "Hyperliquid risk flatten scaffold",
        "replay_calibration_gate": "Replay log calibration",
        "multi_day_replay_gate": "Multi-day event replay",
        "live_canary_gate": "Live canary",
    }
    missing_phrases = []
    for key, phrase in required_status_phrases.items():
        if phrase not in status_text:
            missing_phrases.append(key)
            reasons.append(f"status_missing_phrase:{key}")

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "status_path": str(status_path),
        "gates_path": str(gates_path),
        "pytest_count": {
            "status": status_count,
            "latest_gates": pytest_count,
        },
        "local_gates": {
            "required": sorted(REQUIRED_LOCAL_GATES),
            "present": sorted(local_gate_names),
            "missing": missing_local_gates,
            "failed": failed_local_gates,
        },
        "manual_gates": {
            "required": sorted(REQUIRED_MANUAL_GATES),
            "present": sorted(manual_gate_names),
            "missing": missing_manual_gates,
            "failed": sorted(failed_manual_gates),
        },
        "deployment": {
            "ready": bool(gates_payload.get("deployment_ready")),
            "blockers": sorted(deployment_blockers),
            "unexpected_blockers": unexpected_deployment_blockers,
            "manual_gates_remaining": manual_gates_remaining,
        },
        "status_phrases": {
            "missing": missing_phrases,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PLAN implementation status against latest gate evidence.")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_text, status_error = read_text(args.status)
    gates_payload, gates_error = load_json(args.gates)
    report = build_plan_status_audit(
        status_text,
        gates_payload,
        status_path=args.status,
        gates_path=args.gates,
        status_error=status_error,
        gates_error=gates_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
