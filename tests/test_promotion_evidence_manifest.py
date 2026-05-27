from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_promotion_evidence_manifest import build_manifest  # noqa: E402


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def dry_run_report(**overrides):
    payload = {
        "ok": True,
        "gate_report": {"runtime_seconds": 600.0},
        "event_span_seconds": 540.0,
        "accepted_quotes": 12,
        "accepted_order_attempts": 11,
        "dry_run_fills": 0,
        "quote_quality": {"max_depth_bps": 6.0, "avg_depth_bps": 5.5},
        "order_sizing": {
            "max_amount": 0.01,
            "max_notional_usdc": 20.0,
            "avg_notional_usdc": 19.5,
        },
        "pnl": {"min_total_pnl_usdc": 0.0, "final_total_pnl_usdc": 0.0},
        "loss_velocity_usdc_per_hour": 0.0,
    }
    payload.update(overrides)
    return payload


def tif_runtime_report(**overrides):
    payload = {
        "ok": True,
        "research_gtc_supported": True,
        "freqtrade_post_only_config_supported": True,
        "freqtrade_post_only_supported_tifs": ["PO"],
        "freqtrade_post_only_unsupported_tifs": ["Alo"],
        "live_safe_via_freqtrade": False,
        "requires_exchange_submit_evidence": True,
        "live_safe_reason": "startup_config_acceptance_is_not_exchange_alo_submit_evidence",
    }
    payload.update(overrides)
    return payload


def test_manifest_keeps_good_dry_run_separate_from_live_readiness(tmp_path):
    dry = write_json(tmp_path / "dry.json", dry_run_report())
    tif = write_json(tmp_path / "tif.json", tif_runtime_report())
    reports = {
        "hyperliquid_post_only_mapping": write_json(tmp_path / "post.json", {"ok": False, "reasons": ["missing_crossing_result"]}),
        "hyperliquid_fee_tier": write_json(tmp_path / "fee.json", {"ok": False, "reasons": ["exchange_fee_not_proven"]}),
        "multi_day_event_replay": write_json(tmp_path / "replay.json", {"ok": False, "reasons": ["insufficient_coverage_days"]}),
        "live_canary": write_json(tmp_path / "canary.json", {"ok": False, "reasons": ["manual_monitoring_not_acknowledged"]}),
    }

    manifest = build_manifest(dry_run_quality_path=dry, tif_runtime_path=tif, external_gate_reports=reports)

    assert manifest["ok"] is True
    assert manifest["deployment_ready"] is False
    assert manifest["manual_gates_remaining"] == 4
    assert manifest["dry_run_quality"]["ok"] is True
    assert manifest["dry_run_quality"]["conclusion"] == "dry_run_quotes_and_sizing_passed_but_no_fill_profit_evidence"
    assert manifest["dry_run_quality"]["loss_velocity_usdc_per_hour"] == 0.0
    assert manifest["freqtrade_tif_runtime"]["freqtrade_post_only_supported_tifs"] == ["PO"]
    assert manifest["freqtrade_tif_runtime"]["live_safe_via_freqtrade"] is False
    assert manifest["real_money_policy"]["dry_run_alone_is_sufficient_for_live"] is False


def test_manifest_marks_ready_only_when_external_reports_are_green(tmp_path):
    dry = write_json(tmp_path / "dry.json", dry_run_report(dry_run_fills=1))
    tif = write_json(tmp_path / "tif.json", tif_runtime_report())
    reports = {
        name: write_json(tmp_path / f"{name}.json", {"ok": True, "generated_at": "2026-05-27T00:00:00Z"})
        for name in (
            "hyperliquid_post_only_mapping",
            "hyperliquid_fee_tier",
            "multi_day_event_replay",
            "live_canary",
        )
    }

    manifest = build_manifest(dry_run_quality_path=dry, tif_runtime_path=tif, external_gate_reports=reports)

    assert manifest["deployment_ready"] is True
    assert manifest["manual_gates_remaining"] == 0
    assert manifest["deployment_blockers"] == []
    assert manifest["dry_run_quality"]["conclusion"] == "dry_run_quotes_sizing_and_bounded_pnl_passed_with_fills"


def test_manifest_includes_guarded_promotion_commands(tmp_path):
    dry = write_json(tmp_path / "dry.json", dry_run_report())
    tif = write_json(tmp_path / "tif.json", tif_runtime_report())
    reports = {
        "hyperliquid_post_only_mapping": write_json(tmp_path / "post.json", {"ok": False}),
        "hyperliquid_fee_tier": write_json(tmp_path / "fee.json", {"ok": False}),
        "multi_day_event_replay": write_json(tmp_path / "replay.json", {"ok": False}),
        "live_canary": write_json(tmp_path / "canary.json", {"ok": False}),
    }

    manifest = build_manifest(dry_run_quality_path=dry, tif_runtime_path=tif, external_gate_reports=reports)

    post_only_commands = manifest["promotion_commands"]["post_only"]
    assert any(cmd["requires_real_orders"] for cmd in post_only_commands)
    assert any(any("verify_post_only_mapping.py" in part for part in cmd["command"]) for cmd in post_only_commands)
    extended_command = manifest["promotion_commands"]["extended_dry_run"][0]["command"]
    assert extended_command[extended_command.index("--enabled-dry-run-seconds") + 1] == "1800"
    assert extended_command[extended_command.index("--enabled-dry-run-max-loss-rate-usdc-per-hour") + 1] == "2"
    assert manifest["promotion_commands"]["multi_day_replay"][0]["command"][-4:] == [
        "--json-output",
        "docs/last_safety_gates.json",
        "--markdown-output",
        "docs/LAST_SAFETY_GATES.md",
    ]
