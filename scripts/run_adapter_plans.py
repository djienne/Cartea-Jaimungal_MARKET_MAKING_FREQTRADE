#!/usr/bin/env python3
"""Regenerate all no-network adapter planning artifacts as one safety gate.

Consolidates what used to be five separate ~0.1s gates (post_only_probe_plan,
direct_alo_adapter_plan, direct_alo_probe_preparation_plan,
direct_risk_flatten_plan, hyperliquid_fee_capture_plan) into the single
``adapter_plans`` gate. Each sub-plan still runs its own CLI exactly as
before; all sub-plans run even after a failure so one battery surfaces every
broken plan at once.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sub_plan_commands(py: str) -> list[tuple[str, list[str]]]:
    return [
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
        ),
        (
            "direct_alo_probe_preparation_plan",
            [
                py,
                "scripts/hyperliquid_alo_executor.py",
                "--mode",
                "prepare-probes",
                "--testnet",
                "--symbol",
                "ETH/USDC:USDC",
                "--side",
                "bid",
                "--size",
                "0.01",
                "--best-bid",
                "1999",
                "--best-ask",
                "2001",
                "--price-tick-size",
                "0.1",
                "--amount-step-size",
                "0.001",
                "--quote-id",
                "quote-safety-gate",
                "--session-id",
                "safety-gate",
                "--hjb-generation",
                "1",
                "--output",
                "docs/direct_alo_probe_commands.json",
            ],
        ),
        (
            # The ask side has never been exercised against the exchange -- every
            # probe artefact in this repo was --side bid. A two-sided maker rests
            # an ask continuously, and instance S opens with one, so the ask path
            # needs its own evidence rather than inheriting the bid's.
            "direct_alo_probe_preparation_plan_ask",
            [
                py,
                "scripts/hyperliquid_alo_executor.py",
                "--mode",
                "prepare-probes",
                "--testnet",
                "--symbol",
                "ETH/USDC:USDC",
                "--side",
                "ask",
                "--size",
                "0.01",
                "--best-bid",
                "1999",
                "--best-ask",
                "2001",
                "--price-tick-size",
                "0.1",
                "--amount-step-size",
                "0.001",
                "--quote-id",
                "quote-safety-gate-ask",
                "--session-id",
                "safety-gate",
                "--hjb-generation",
                "1",
                "--output",
                "docs/direct_alo_probe_commands_ask.json",
            ],
        ),
        (
            "direct_risk_flatten_plan",
            [
                py,
                "scripts/hyperliquid_risk_executor.py",
                "--mode",
                "plan",
                "--output",
                "docs/hyperliquid_risk_flatten_plan.json",
            ],
        ),
        (
            "hyperliquid_fee_capture_plan",
            [
                py,
                "scripts/capture_hyperliquid_fee_evidence.py",
                "--mode",
                "plan",
                "--output",
                "docs/hyperliquid_fee_capture_plan.json",
            ],
        ),
    ]


def main() -> int:
    failures: list[str] = []
    for name, command in sub_plan_commands(sys.executable):
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        status = "ok" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"[adapter_plans] {name}: {status}")
        if proc.returncode != 0:
            failures.append(name)
            stderr_tail = (proc.stderr or "").strip()[-2000:]
            if stderr_tail:
                print(stderr_tail, file=sys.stderr)
    if failures:
        print(f"[adapter_plans] failed sub-plans: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
