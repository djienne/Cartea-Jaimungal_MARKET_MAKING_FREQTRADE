#!/usr/bin/env python3
"""Run the local safety gates for the market-making implementation.

This runner intentionally separates local deterministic checks from the
deployment gates that need Freqtrade runtime wiring or Hyperliquid testnet.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateResult:
    name: str
    command: list[str]
    passed: bool
    returncode: int
    expected_returncodes: list[int]
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str


def tail(text: str | None, max_chars: int = 4_000) -> str:
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_command(name: str, command: Sequence[str], expected_returncodes: Sequence[int] = (0,)) -> GateResult:
    start = time.perf_counter()
    proc = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start
    expected = list(expected_returncodes)
    return GateResult(
        name=name,
        command=list(command),
        passed=proc.returncode in expected,
        returncode=proc.returncode,
        expected_returncodes=expected,
        elapsed_seconds=round(elapsed, 3),
        stdout_tail=tail(proc.stdout),
        stderr_tail=tail(proc.stderr),
    )


def local_gates(*, include_runtime: bool = False) -> list[tuple[str, list[str], list[int]]]:
    py = sys.executable
    gates = [
        ("compileall", [py, "-m", "compileall", "scripts", "user_data/strategies"], [0]),
        (
            "pytest_core",
            [
                py,
                "-m",
                "pytest",
                "tests/test_hjb.py",
                "tests/test_hyperliquid_alo_executor.py",
                "tests/test_config_safety.py",
                "tests/test_fee_evidence.py",
                "tests/test_live_canary.py",
                "tests/test_plan_status.py",
                "tests/test_strategy_guards.py",
                "tests/test_strategy_safety.py",
                "tests/test_params.py",
                "tests/test_periodic_runner.py",
                "tests/test_replay_market_maker.py",
                "tests/test_replay_log_calibration.py",
                "tests/test_replay_report.py",
                "tests/test_safety_gates.py",
            ],
            [0],
        ),
        (
            "config_safety_report",
            [
                py,
                "scripts/verify_config_safety.py",
                "--config",
                "user_data/config.json",
                "--output",
                "docs/config_safety_report.json",
            ],
            [0],
        ),
        (
            "strategy_safety_report",
            [
                py,
                "scripts/verify_strategy_safety.py",
                "--strategy",
                "user_data/strategies/Market_Making.py",
                "--output",
                "docs/strategy_safety_report.json",
            ],
            [0],
        ),
        (
            "compute_spreads_boundary_smoke",
            [
                py,
                "scripts/compute_spreads.py",
                "--crypto",
                "ETH",
                "--mid",
                "4322.05",
                "--qmax",
                "3",
                "--asym-kappa",
                "--skip-refresh",
            ],
            [0],
        ),
        (
            "replay_smoke",
            [
                py,
                "scripts/replay_market_maker.py",
                "--symbol",
                "ETH",
                "--mid",
                "4322.05",
                "--synthetic-smoke",
                "--kappa-plus",
                "2.8815386649988977",
                "--kappa-minus",
                "1.0839082467471117",
                "--lambda-plus",
                "0.1364138970843998",
                "--lambda-minus",
                "0.09516227408346505",
            ],
            [0],
        ),
        (
            "post_only_probe_plan",
            [
                py,
                "scripts/verify_post_only_mapping.py",
                "--mode",
                "plan",
                "--output",
                "docs/post_only_probe_plan.json",
            ],
            [0],
        ),
        (
            "post_only_evidence_report",
            [
                py,
                "scripts/verify_post_only_mapping.py",
                "--mode",
                "evaluate-evidence",
                "--output",
                "docs/post_only_evidence_report.json",
            ],
            [1],
        ),
        (
            "direct_alo_adapter_plan",
            [
                py,
                "scripts/hyperliquid_alo_executor.py",
                "--mode",
                "plan",
                "--output",
                "docs/direct_alo_adapter_plan.json",
            ],
            [0],
        ),
    ]
    if include_runtime:
        gates.extend(
            [
                ("docker_compose_config", ["docker", "compose", "config"], [0]),
                (
                    "freqtrade_runtime_load",
                    [
                        "docker",
                        "compose",
                        "run",
                        "--rm",
                        "--no-deps",
                        "freqtrade",
                        "list-strategies",
                        "--config",
                        "/freqtrade/user_data/config.json",
                    ],
                    [0],
                ),
                (
                    "dry_run_disabled_smoke",
                    [
                        sys.executable,
                        "scripts/verify_dry_run_disabled.py",
                        "--seconds",
                        "150",
                        "--require-health",
                        "--start-collector",
                        "--collector-warmup-seconds",
                        "70",
                        "--json-output",
                        "docs/dry_run_disabled_gate.json",
                    ],
                    [0],
                ),
                (
                    "dry_run_enabled_smoke",
                    [
                        sys.executable,
                        "scripts/verify_dry_run_enabled.py",
                        "--seconds",
                        "240",
                        "--collector-warmup-seconds",
                        "70",
                        "--json-output",
                        "docs/dry_run_enabled_gate.json",
                    ],
                    [0],
                ),
                (
                    "replay_log_calibration_artifact",
                    [
                        sys.executable,
                        "scripts/calibrate_replay_from_logs.py",
                        "--input",
                        "user_data/logs/mm_debug.jsonl",
                        "--output",
                        "docs/replay_log_calibration.json",
                    ],
                    [0],
                ),
                (
                    "fee_evidence_report",
                    [
                        sys.executable,
                        "scripts/verify_fee_evidence.py",
                        "--input",
                        "user_data/logs/mm_debug.jsonl",
                        "--output",
                        "docs/fee_evidence_report.json",
                    ],
                    [0, 1],
                ),
                (
                    "hl_data_validation_report",
                    [
                        sys.executable,
                        "scripts/validate_hl_data.py",
                        "--symbol",
                        "ETH",
                        "--newest-per-stream",
                        "25",
                        "--max-age-seconds",
                        "180",
                        "--output",
                        "docs/hl_data_validation.json",
                        "--fail-on-bad-data",
                    ],
                    [0],
                ),
                (
                    "replay_latest_data_smoke",
                    [
                        sys.executable,
                        "scripts/replay_market_maker.py",
                        "--symbol",
                        "ETH",
                        "--mid",
                        "2117.35",
                        "--newest-per-stream",
                        "10",
                        "--max-price-events",
                        "1000",
                        "--kappa-plus",
                        "2.0",
                        "--kappa-minus",
                        "2.0",
                        "--lambda-plus",
                        "0.1",
                        "--lambda-minus",
                        "0.1",
                        "--output",
                        "docs/replay_latest_smoke.json",
                    ],
                    [0],
                ),
                (
                    "replay_acceptance_report_artifact",
                    [
                        sys.executable,
                        "scripts/run_replay_report.py",
                        "--symbol",
                        "ETH",
                        "--mid",
                        "2116.95",
                        "--newest-per-stream",
                        "25",
                        "--max-price-events",
                        "2000",
                        "--kappa-plus",
                        "2.0",
                        "--kappa-minus",
                        "2.0",
                        "--lambda-plus",
                        "0.1",
                        "--lambda-minus",
                        "0.1",
                        "--output",
                        "docs/replay_acceptance_report.json",
                        "--markdown-output",
                        "docs/replay_acceptance_report.md",
                        "--allow-incomplete",
                    ],
                    [0],
                ),
                (
                    "live_canary_evidence_report",
                    [
                        sys.executable,
                        "scripts/verify_live_canary.py",
                        "--input",
                        "user_data/logs/mm_debug.jsonl",
                        "--output",
                        "docs/live_canary_report.json",
                    ],
                    [0, 1],
                ),
            ]
        )
    else:
        gates.append(
            (
                "live_canary_evidence_report",
                [
                    py,
                    "scripts/verify_live_canary.py",
                    "--input",
                    "user_data/logs/mm_debug.jsonl",
                    "--output",
                    "docs/live_canary_report.json",
                ],
                [0, 1],
            )
        )
    return gates


def manual_gates(*, include_runtime: bool = False) -> list[dict[str, str]]:
    gates = [
        {
            "name": "hyperliquid_post_only_mapping",
            "reason": "Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.",
        },
        {
            "name": "multi_day_event_replay",
            "reason": "Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.",
        },
        {
            "name": "hyperliquid_fee_tier",
            "reason": "Requires exchange/account maker-fee evidence and actual maker fill fee rates.",
        },
        {
            "name": "live_canary",
            "reason": "Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.",
        },
    ]
    if not include_runtime:
        gates.insert(
            0,
            {
                "name": "deterministic_dry_run_trading_disabled",
                "reason": "Requires running the bot loop and confirming zero orders plus health logs in Freqtrade logs.",
            },
        )
        gates.insert(
            1,
            {
                "name": "freqtrade_runtime_load",
                "reason": "Requires a Freqtrade environment with exchange/config plugins installed.",
            },
        )
    return gates


def plan_status_audit_command(gates_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/verify_plan_status.py",
        "--status",
        "docs/PLAN_IMPLEMENTATION_STATUS.md",
        "--gates",
        str(gates_path),
        "--output",
        str(output_path),
    ]


def run_plan_status_audit(payload: dict, gates_path: Path, output_path: Path) -> GateResult:
    gates_path.parent.mkdir(parents=True, exist_ok=True)
    gates_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return run_command("plan_status_audit", plan_status_audit_command(gates_path, output_path), [0])


def render_markdown(payload: dict) -> str:
    lines = ["# Safety Gate Results", ""]
    automated_status = "PASS" if payload.get("all_automated_passed") else "FAIL"
    deployment_status = "YES" if payload.get("deployment_ready") else "NO"
    lines.append(f"Automated gates: {automated_status}")
    lines.append(f"Deployment ready: {deployment_status}")
    lines.append(f"Manual gates remaining: {payload.get('manual_gates_remaining', len(payload.get('manual_gates', [])))}")
    lines.append("")
    for result in payload["local_gates"]:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"- {status} `{result['name']}` ({result['elapsed_seconds']}s)")
        if not result["passed"]:
            lines.append(f"  - returncode: `{result['returncode']}`")
            if result["stderr_tail"]:
                lines.append("  - stderr tail:")
                lines.append("```text")
                lines.append(result["stderr_tail"])
                lines.append("```")
    post_run_audits = payload.get("post_run_audits", [])
    if post_run_audits:
        lines.append("")
        lines.append("Post-run audits:")
        for result in post_run_audits:
            status = "PASS" if result["passed"] else "FAIL"
            lines.append(f"- {status} `{result['name']}` ({result['elapsed_seconds']}s)")
            if not result["passed"]:
                lines.append(f"  - returncode: `{result['returncode']}`")
                if result["stderr_tail"]:
                    lines.append("  - stderr tail:")
                    lines.append("```text")
                    lines.append(result["stderr_tail"])
                    lines.append("```")
    lines.append("")
    lines.append("Manual gates still required:")
    for gate in payload["manual_gates"]:
        lines.append(f"- `{gate['name']}`: {gate['reason']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local market-making safety gates.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for full JSON results.")
    parser.add_argument("--markdown-output", type=Path, default=None, help="Optional path for markdown summary.")
    parser.add_argument("--include-runtime", action="store_true", help="Also run non-trading Docker/Freqtrade load gates.")
    parser.add_argument(
        "--plan-status-audit-output",
        type=Path,
        default=Path("docs/plan_status_audit.json"),
        help="Path for the post-run PLAN status audit artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        run_command(name, command, expected_returncodes)
        for name, command, expected_returncodes in local_gates(include_runtime=args.include_runtime)
    ]
    manual_gate_list = manual_gates(include_runtime=args.include_runtime)
    deployment_blockers = [gate["name"] for gate in manual_gate_list]
    all_local_passed = all(result.passed for result in results)
    payload = {
        "local_gates": [asdict(result) for result in results],
        "manual_gates": manual_gate_list,
        "manual_gates_remaining": len(deployment_blockers),
        "all_local_passed": all_local_passed,
        "post_run_audits": [],
        "all_automated_passed": all_local_passed,
        "deployment_ready": bool(all_local_passed and not deployment_blockers),
        "deployment_blockers": deployment_blockers,
        "runtime_gates_included": bool(args.include_runtime),
    }

    if args.include_runtime:
        if args.json_output:
            audit_gate_path = args.json_output
            audit_result = run_plan_status_audit(payload, audit_gate_path, args.plan_status_audit_output)
        else:
            with tempfile.TemporaryDirectory(prefix="mm-safety-gates-") as tmp_dir:
                audit_gate_path = Path(tmp_dir) / "last_safety_gates.json"
                audit_result = run_plan_status_audit(payload, audit_gate_path, args.plan_status_audit_output)

        payload["post_run_audits"] = [asdict(audit_result)]
        payload["all_automated_passed"] = bool(payload["all_local_passed"] and audit_result.passed)
        payload["deployment_ready"] = bool(payload["all_automated_passed"] and not deployment_blockers)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    markdown = render_markdown(payload)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    print(markdown)

    return 0 if payload["all_automated_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
