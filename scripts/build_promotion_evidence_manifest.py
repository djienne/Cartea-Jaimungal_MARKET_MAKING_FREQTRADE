#!/usr/bin/env python3
"""Build a single promotion evidence manifest from gate artifacts.

The manifest does not unlock live trading. It summarizes which MEGA_PLAN
deployment gates are already supported by local evidence, which external
artifacts are still missing, and the exact guarded commands an operator should
run to collect those artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("docs/promotion_evidence_manifest.json")
DEFAULT_DRY_RUN_QUALITY = Path("docs/dry_run_quality_report.json")
DEFAULT_TIF_RUNTIME = Path("docs/freqtrade_tif_runtime_report.json")
DEFAULT_POST_ONLY = Path("docs/post_only_evidence_report.json")
DEFAULT_FEE = Path("docs/fee_evidence_report.json")
DEFAULT_REPLAY = Path("docs/replay_acceptance_report.json")
DEFAULT_LIVE_CANARY = Path("docs/live_canary_report.json")


EXTERNAL_GATE_REPORTS = {
    "hyperliquid_post_only_mapping": DEFAULT_POST_ONLY,
    "hyperliquid_fee_tier": DEFAULT_FEE,
    "multi_day_event_replay": DEFAULT_REPLAY,
    "live_canary": DEFAULT_LIVE_CANARY,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"ok": False, "missing": True, "path": str(path), "reasons": ["missing_report"]}
    except Exception as exc:
        return {"ok": False, "invalid": True, "path": str(path), "reasons": [f"invalid_report:{exc}"]}
    if not isinstance(payload, dict):
        return {"ok": False, "invalid": True, "path": str(path), "reasons": ["report_not_object"]}
    payload = dict(payload)
    payload.setdefault("path", str(path))
    return payload


def report_status(name: str, path: Path) -> dict[str, Any]:
    payload = load_json(path)
    reasons = payload.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "name": name,
        "path": str(path),
        "ok": payload.get("ok") is True,
        "generated_at": payload.get("generated_at"),
        "reasons": [str(reason) for reason in reasons],
    }


def dry_run_assessment(report: dict[str, Any]) -> dict[str, Any]:
    order_sizing = report.get("order_sizing") if isinstance(report.get("order_sizing"), dict) else {}
    quote_quality = report.get("quote_quality") if isinstance(report.get("quote_quality"), dict) else {}
    pnl = report.get("pnl") if isinstance(report.get("pnl"), dict) else {}
    fills = int(report.get("dry_run_fills") or 0)
    ok = report.get("ok") is True
    report_conclusion = report.get("conclusion") if isinstance(report.get("conclusion"), str) else None
    if report_conclusion:
        conclusion = report_conclusion
    elif ok and fills > 0:
        conclusion = "dry_run_quotes_sizing_and_bounded_pnl_passed_with_fills"
    elif ok:
        conclusion = "dry_run_quotes_and_sizing_passed_but_no_fill_profit_evidence"
    else:
        conclusion = "dry_run_quality_gate_not_passed"
    return {
        "ok": ok,
        "conclusion": conclusion,
        "runtime_seconds": (report.get("gate_report") or {}).get("runtime_seconds")
        if isinstance(report.get("gate_report"), dict)
        else None,
        "event_span_seconds": report.get("event_span_seconds"),
        "accepted_quotes": report.get("accepted_quotes"),
        "accepted_order_attempts": report.get("accepted_order_attempts"),
        "dry_run_fills": fills,
        "max_quote_depth_bps": quote_quality.get("max_depth_bps"),
        "avg_quote_depth_bps": quote_quality.get("avg_depth_bps"),
        "max_order_amount": order_sizing.get("max_amount"),
        "max_order_notional_usdc": order_sizing.get("max_notional_usdc"),
        "avg_order_notional_usdc": order_sizing.get("avg_notional_usdc"),
        "min_total_pnl_usdc": pnl.get("min_total_pnl_usdc"),
        "final_total_pnl_usdc": pnl.get("final_total_pnl_usdc"),
        "loss_velocity_usdc_per_hour": report.get("loss_velocity_usdc_per_hour"),
    }


def tif_runtime_assessment(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok") is True,
        "research_gtc_supported": report.get("research_gtc_supported") is True,
        "freqtrade_post_only_config_supported": report.get("freqtrade_post_only_config_supported") is True,
        "freqtrade_post_only_supported_tifs": report.get("freqtrade_post_only_supported_tifs") or [],
        "freqtrade_post_only_unsupported_tifs": report.get("freqtrade_post_only_unsupported_tifs") or [],
        "live_safe_via_freqtrade": report.get("live_safe_via_freqtrade") is True,
        "requires_exchange_submit_evidence": report.get("requires_exchange_submit_evidence") is True,
        "live_safe_reason": report.get("live_safe_reason"),
    }


def command(label: str, args: list[str], *, requires_real_orders: bool = False, note: str | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "command": args,
        "requires_real_orders": bool(requires_real_orders),
        "note": note,
    }


def promotion_commands() -> dict[str, list[dict[str, Any]]]:
    return {
        "extended_dry_run": [
            command(
                "run_30_minute_enabled_dry_run_quality_gate",
                [
                    "python",
                    "scripts/run_safety_gates.py",
                    "--include-runtime",
                    "--enabled-dry-run-seconds",
                    "1800",
                    "--enabled-dry-run-min-runtime-seconds",
                    "1620",
                    "--enabled-dry-run-min-event-span-seconds",
                    "1200",
                    "--enabled-dry-run-min-accepted-quotes",
                    "5",
                    "--enabled-dry-run-min-order-attempts",
                    "5",
                    "--enabled-dry-run-max-loss-rate-usdc-per-hour",
                    "2",
                    "--enabled-dry-run-min-final-total-pnl-usdc",
                    "0",
                    "--json-output",
                    "docs/last_safety_gates.json",
                    "--markdown-output",
                    "docs/LAST_SAFETY_GATES.md",
                ],
                note=(
                    "Promotion-grade dry-run evidence: longer runtime, multiple model-valid quotes/orders, "
                    "tiny order size, bounded drawdown, and bounded loss velocity. This remains dry-run evidence only."
                ),
            )
        ],
        "post_only": [
            command(
                "write_no_network_plan",
                ["python", "scripts/verify_post_only_mapping.py", "--mode", "plan", "--output", "docs/post_only_probe_plan.json"],
            ),
            command(
                "prepare_direct_sdk_alo_probe_commands",
                [
                    "python",
                    "scripts/hyperliquid_alo_executor.py",
                    "--mode",
                    "prepare-probes",
                    "--testnet",
                    "--symbol",
                    "ETH/USDC:USDC",
                    "--side",
                    "bid",
                    "--size",
                    "<min_size>",
                    "--best-bid",
                    "<best_bid>",
                    "--best-ask",
                    "<best_ask>",
                    "--price-tick-size",
                    "<tick_size>",
                    "--amount-step-size",
                    "<amount_step>",
                    "--quote-id",
                    "<QUOTE_ID>",
                    "--session-id",
                    "<CANARY_SESSION_ID>",
                    "--hjb-generation",
                    "<HJB_GENERATION>",
                    "--output",
                    "docs/direct_alo_probe_commands.json",
                ],
                note=(
                    "No order submission. Converts an observed BBO into exact crossing/passive ALO submit commands "
                    "with quote-linked client order id and cloid evidence."
                ),
            ),
            command(
                "direct_sdk_crossing_alo_reject_probe",
                [
                    "python",
                    "scripts/hyperliquid_alo_executor.py",
                    "--mode",
                    "submit-crossing-alo",
                    "--testnet",
                    "--symbol",
                    "ETH/USDC:USDC",
                    "--side",
                    "bid",
                    "--size",
                    "<min_size>",
                    "--best-bid",
                    "<best_bid>",
                    "--best-ask",
                    "<best_ask>",
                    "--price-tick-size",
                    "<tick_size>",
                    "--amount-step-size",
                    "<amount_step>",
                    "--allow-crossing-probe",
                    "--acknowledge-real-orders",
                    "--output",
                    "docs/direct_alo_reject_result.json",
                ],
                requires_real_orders=True,
                note="Requires HYPERLIQUID_DIRECT_ALO_ALLOW=1 and testnet/tiny order sizing.",
            ),
            command(
                "direct_sdk_passive_alo_resting_probe",
                [
                    "python",
                    "scripts/hyperliquid_alo_executor.py",
                    "--mode",
                    "submit-passive-alo",
                    "--testnet",
                    "--symbol",
                    "ETH/USDC:USDC",
                    "--side",
                    "bid",
                    "--size",
                    "<min_size>",
                    "--price",
                    "<passive_bid>",
                    "--best-bid",
                    "<best_bid>",
                    "--best-ask",
                    "<best_ask>",
                    "--price-tick-size",
                    "<tick_size>",
                    "--amount-step-size",
                    "<amount_step>",
                    "--allow-passive-probe",
                    "--acknowledge-real-orders",
                    "--output",
                    "docs/direct_alo_passive_result.json",
                ],
                requires_real_orders=True,
                note="Cancels resting order ids reported by the SDK after classification.",
            ),
            command(
                "evaluate_post_only_evidence",
                [
                    "python",
                    "scripts/verify_post_only_mapping.py",
                    "--mode",
                    "evaluate-evidence",
                    "--crossing-result",
                    "docs/direct_alo_reject_result.json",
                    "--passive-result",
                    "docs/direct_alo_passive_result.json",
                    "--output",
                    "docs/post_only_evidence_report.json",
                ],
            ),
        ],
        "fee": [
            command(
                "capture_read_only_fee_and_fill_evidence",
                [
                    "python",
                    "scripts/capture_hyperliquid_fee_evidence.py",
                    "--mode",
                    "fetch",
                    "--testnet",
                    "--account-address",
                    "<account_address>",
                    "--order-evidence-json",
                    "docs/direct_alo_passive_result.json",
                    "--acknowledge-account-read",
                    "--output",
                    "docs/hyperliquid_fee_evidence_capture.jsonl",
                ],
                note="Requires HYPERLIQUID_FEE_EVIDENCE_ALLOW=1; read-only account request.",
            ),
            command(
                "verify_fee_evidence",
                [
                    "python",
                    "scripts/verify_fee_evidence.py",
                    "--input",
                    "docs/hyperliquid_fee_evidence_capture.jsonl",
                    "--output",
                    "docs/fee_evidence_report.json",
                ],
            ),
        ],
        "multi_day_replay": [
            command(
                "run_full_replay_acceptance",
                [
                    "python",
                    "scripts/run_safety_gates.py",
                    "--include-runtime",
                    "--audit-log-input",
                    "docs/testnet_mm_debug.jsonl",
                    "--replay-acceptance-newest-per-stream",
                    "0",
                    "--replay-acceptance-max-price-events",
                    "0",
                    "--replay-acceptance-require-fill-calibration",
                    "--replay-acceptance-require-pass",
                    "--json-output",
                    "docs/last_safety_gates.json",
                    "--markdown-output",
                    "docs/LAST_SAFETY_GATES.md",
                ],
                note=(
                    "Requires several days of HL_data shards with sufficient price-event density. "
                    "Set --replay-acceptance-price-tick-size and --replay-acceptance-amount-step-size "
                    "to the exchange constraints being validated."
                ),
            ),
        ],
        "live_canary": [
            command(
                "record_manual_monitoring_ack",
                [
                    "python",
                    "scripts/record_manual_monitoring_ack.py",
                    "--output",
                    "docs/live_canary_mm_debug.jsonl",
                    "--session-id",
                    "<CANARY_SESSION_ID>",
                    "--operator",
                    "<NAME>",
                    "--acknowledge-risk",
                ],
                note="Append this to the retained canary audit log only during a monitored canary.",
            ),
            command(
                "verify_live_canary_bundle",
                [
                    "python",
                    "scripts/run_safety_gates.py",
                    "--include-runtime",
                    "--audit-log-input",
                    "docs/live_canary_mm_debug.jsonl",
                    "--manual-monitoring-ack",
                    "--json-output",
                    "docs/last_safety_gates.json",
                    "--markdown-output",
                    "docs/LAST_SAFETY_GATES.md",
                ],
                requires_real_orders=True,
                note="Only after post-only, fee, and multi-day replay reports are ok=true.",
            ),
        ],
    }


def build_manifest(
    *,
    dry_run_quality_path: Path = DEFAULT_DRY_RUN_QUALITY,
    tif_runtime_path: Path = DEFAULT_TIF_RUNTIME,
    external_gate_reports: dict[str, Path] | None = None,
) -> dict[str, Any]:
    external_gate_reports = external_gate_reports or EXTERNAL_GATE_REPORTS
    dry_run_quality = load_json(dry_run_quality_path)
    tif_runtime = load_json(tif_runtime_path)
    external = [report_status(name, path) for name, path in external_gate_reports.items()]
    blockers = [item for item in external if item["ok"] is not True]
    return {
        "generated_at": utc_now_iso(),
        "ok": True,
        "deployment_ready": not blockers,
        "safe_default": "manifest_only_no_orders_no_network",
        "dry_run_quality": dry_run_assessment(dry_run_quality),
        "freqtrade_tif_runtime": tif_runtime_assessment(tif_runtime),
        "external_gates": external,
        "deployment_blockers": [item["name"] for item in blockers],
        "manual_gates_remaining": len(blockers),
        "promotion_commands": promotion_commands(),
        "real_money_policy": {
            "dry_run_is_required": True,
            "dry_run_alone_is_sufficient_for_live": False,
            "requires_post_only_fee_replay_and_canary_evidence": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build promotion evidence manifest from existing reports.")
    parser.add_argument("--dry-run-quality", type=Path, default=DEFAULT_DRY_RUN_QUALITY)
    parser.add_argument("--tif-runtime", type=Path, default=DEFAULT_TIF_RUNTIME)
    parser.add_argument("--post-only-report", type=Path, default=DEFAULT_POST_ONLY)
    parser.add_argument("--fee-report", type=Path, default=DEFAULT_FEE)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--live-canary-report", type=Path, default=DEFAULT_LIVE_CANARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        dry_run_quality_path=args.dry_run_quality,
        tif_runtime_path=args.tif_runtime,
        external_gate_reports={
            "hyperliquid_post_only_mapping": args.post_only_report,
            "hyperliquid_fee_tier": args.fee_report,
            "multi_day_event_replay": args.replay_report,
            "live_canary": args.live_canary_report,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
