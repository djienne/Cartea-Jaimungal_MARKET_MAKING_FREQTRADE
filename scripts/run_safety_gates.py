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
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_LOG_INPUT = Path("user_data/logs/mm_debug.jsonl")
DEFAULT_REPLAY_MIN_PRICE_EVENTS_PER_DAY = 1_000.0
DEFAULT_REPLAY_MAX_PRICE_GAP_SECONDS = 300.0
DEFAULT_REPORT_MAX_AGE_SECONDS = 86_400.0
DEFAULT_LIVE_CANARY_MAX_EVENT_AGE_SECONDS = 604_800.0
DEFAULT_ENABLED_DRY_RUN_SECONDS = 600
DEFAULT_ENABLED_DRY_RUN_MIN_RUNTIME_SECONDS = 540
DEFAULT_ENABLED_DRY_RUN_MIN_EVENT_SPAN_SECONDS = 300
DEFAULT_ENABLED_DRY_RUN_MIN_ACCEPTED_QUOTES = 2
DEFAULT_ENABLED_DRY_RUN_MIN_ORDER_ATTEMPTS = 2
DEFAULT_ENABLED_DRY_RUN_MAX_LOSS_RATE_USDC_PER_HOUR = 6.0
DEFAULT_ENABLED_DRY_RUN_MIN_FINAL_TOTAL_PNL_USDC = -1.0
DEFAULT_POST_ONLY_CROSSING_RESULT = Path("docs/post_only_crossing_result.json")
DEFAULT_POST_ONLY_PASSIVE_RESULT = Path("docs/post_only_passive_result.json")
DEFAULT_REPLAY_FILL_CALIBRATION = Path("docs/replay_log_calibration.json")
DEFAULT_PREVIOUS_GATES = Path("docs/last_safety_gates.json")
DEFAULT_MAX_SMOKE_ARTIFACT_AGE_SECONDS = 21_600.0  # 6 hours

# The two long runtime smokes and the artifacts they write. --reuse-smoke-artifacts
# skips re-running them and instead injects the previous run's results, validated
# against these artifacts' freshness (fail-closed: missing/stale/failed inputs
# inject FAILED gate results).
SMOKE_GATE_ARTIFACTS: dict[str, Path] = {
    "dry_run_disabled_smoke": Path("docs/dry_run_disabled_gate.json"),
    "dry_run_enabled_smoke": Path("docs/dry_run_enabled_gate.json"),
}

# The quality evaluator's stable input: the audit-event slice the enabled
# smoke archives (the live mm_debug.jsonl keeps mutating after the smoke).
# Legacy gate artifacts without an archive fall back to reusing the previous
# battery's quality result, validated against this report.
QUALITY_REPORT_ARTIFACT = Path("docs/dry_run_quality_report.json")

# The disabled smoke runs its gate container under the production name and the
# smokes stop the collector on teardown, so a runtime battery leaves both
# stopped; the runner restores them afterwards.
PRODUCTION_FREQTRADE_CONTAINER = "MM_ADV"
PRODUCTION_FREQTRADE_SERVICE = "freqtrade"
COLLECTOR_SERVICE = "hl-collector2"


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


@dataclass(frozen=True)
class ManualGateSpec:
    name: str
    reason: str
    report_path: str | None
    ok_field: str = "ok"
    max_age_seconds: float | None = DEFAULT_REPORT_MAX_AGE_SECONDS


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


def load_json_report(path: Path) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing_report"
    except Exception as exc:
        return None, f"invalid_report:{exc}"
    if not isinstance(payload, dict):
        return None, "report_not_object"
    return payload, None


def parse_report_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def report_freshness_status(
    payload: dict,
    *,
    max_age_seconds: float | None,
    now: datetime | None = None,
) -> tuple[bool, str, float | None]:
    if max_age_seconds is None:
        return True, "not_required", None
    generated_at = parse_report_timestamp(payload.get("generated_at"))
    if generated_at is None:
        return False, "missing_generated_at", None
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_seconds = max(0.0, (reference - generated_at).total_seconds())
    if age_seconds > float(max_age_seconds):
        return False, f"report_stale:{age_seconds:.1f}s>max_{float(max_age_seconds):.1f}s", age_seconds
    return True, "fresh", age_seconds


def default_evidence_artifact(path: Path) -> Path | None:
    return path if (ROOT / path).exists() else None


def post_only_evidence_command(
    py: str,
    *,
    crossing_result: Path | None = None,
    passive_result: Path | None = None,
    use_default_artifacts: bool = True,
    max_evidence_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> list[str]:
    command = [
        py,
        "scripts/verify_post_only_mapping.py",
        "--mode",
        "evaluate-evidence",
        "--max-evidence-age-seconds",
        str(float(max_evidence_age_seconds)),
        "--output",
        "docs/post_only_evidence_report.json",
    ]
    crossing = crossing_result
    passive = passive_result
    if use_default_artifacts:
        crossing = crossing or default_evidence_artifact(DEFAULT_POST_ONLY_CROSSING_RESULT)
        passive = passive or default_evidence_artifact(DEFAULT_POST_ONLY_PASSIVE_RESULT)
    if crossing is not None:
        command.extend(["--crossing-result", str(crossing)])
    if passive is not None:
        command.extend(["--passive-result", str(passive)])
    return command


def replay_log_calibration_command(
    py: str,
    *,
    audit_log_input: Path = DEFAULT_AUDIT_LOG_INPUT,
) -> list[str]:
    return [
        py,
        "scripts/calibrate_replay_from_logs.py",
        "--input",
        str(audit_log_input),
        "--output",
        "docs/replay_log_calibration.json",
    ]


def fee_evidence_command(
    py: str,
    *,
    audit_log_input: Path = DEFAULT_AUDIT_LOG_INPUT,
    max_evidence_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> list[str]:
    return [
        py,
        "scripts/verify_fee_evidence.py",
        "--input",
        str(audit_log_input),
        "--max-evidence-age-seconds",
        str(float(max_evidence_age_seconds)),
        "--output",
        "docs/fee_evidence_report.json",
    ]


def promotion_evidence_manifest_command(py: str) -> list[str]:
    return [
        py,
        "scripts/build_promotion_evidence_manifest.py",
        "--output",
        "docs/promotion_evidence_manifest.json",
    ]


def freqtrade_callback_surface_command() -> list[str]:
    return [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "python",
        "freqtrade",
        "/freqtrade/scripts/verify_freqtrade_callback_surface.py",
        "--strategy-path",
        "/freqtrade/user_data/strategies/Market_Making.py",
        "--output",
        "/freqtrade/docs/freqtrade_callback_surface_report.json",
    ]


def freqtrade_tif_runtime_command(py: str) -> list[str]:
    return [
        py,
        "scripts/verify_freqtrade_tif_runtime.py",
        "--output",
        "docs/freqtrade_tif_runtime_report.json",
    ]


def live_canary_evidence_command(
    py: str,
    *,
    audit_log_input: Path = DEFAULT_AUDIT_LOG_INPUT,
    manual_monitoring_ack: bool = False,
    max_dependency_report_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
    max_canary_event_age_seconds: float = DEFAULT_LIVE_CANARY_MAX_EVENT_AGE_SECONDS,
) -> list[str]:
    command = [
        py,
        "scripts/verify_live_canary.py",
        "--input",
        str(audit_log_input),
        "--max-dependency-report-age-seconds",
        str(float(max_dependency_report_age_seconds)),
        "--max-canary-event-age-seconds",
        str(float(max_canary_event_age_seconds)),
        "--output",
        "docs/live_canary_report.json",
    ]
    if manual_monitoring_ack:
        command.append("--manual-monitoring-ack")
    return command


def optional_positive_int(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def replay_acceptance_report_command(
    py: str,
    *,
    symbol: str = "CASHCAT",
    # Fallback mid only (used when a price row carries no bid/ask). Keep it on the
    # same scale as `symbol` -- CASHCAT trades near 0.10, not 2116.
    mid: float = 0.1015,
    newest_per_stream: int | None = 25,
    max_price_events: int | None = 2000,
    min_price_events_per_day: float = DEFAULT_REPLAY_MIN_PRICE_EVENTS_PER_DAY,
    max_price_gap_seconds: float = DEFAULT_REPLAY_MAX_PRICE_GAP_SECONDS,
    price_tick_size: float = 0.0,
    amount_step_size: float = 0.0,
    fill_calibration: Path | None = DEFAULT_REPLAY_FILL_CALIBRATION,
    require_fill_calibration: bool = False,
    allow_incomplete: bool = True,
) -> list[str]:
    command = [
        py,
        "scripts/run_replay_report.py",
        "--symbol",
        str(symbol),
        "--mid",
        str(float(mid)),
        "--kappa-plus",
        "2.0",
        "--kappa-minus",
        "2.0",
        "--lambda-plus",
        "0.1",
        "--lambda-minus",
        "0.1",
        "--spread-multiplier",
        "3.0",
        "--output",
        "docs/replay_acceptance_report.json",
        "--markdown-output",
        "docs/replay_acceptance_report.md",
        "--min-price-events-per-day",
        str(float(min_price_events_per_day)),
        "--max-price-gap-seconds",
        str(float(max_price_gap_seconds)),
        "--price-tick-size",
        str(float(price_tick_size)),
        "--amount-step-size",
        str(float(amount_step_size)),
    ]
    newest = optional_positive_int(newest_per_stream)
    max_events = optional_positive_int(max_price_events)
    if newest is not None:
        command.extend(["--newest-per-stream", str(newest)])
    if max_events is not None:
        command.extend(["--max-price-events", str(max_events)])
    if fill_calibration is not None:
        command.extend(["--fill-calibration", str(fill_calibration)])
    if require_fill_calibration:
        command.append("--require-fill-calibration")
    if allow_incomplete:
        command.append("--allow-incomplete")
    return command


def local_gates(
    *,
    include_runtime: bool = False,
    reuse_smoke_artifacts: bool = False,
    reuse_quality_report: bool = False,
    quality_audit_log_input: Path | None = None,
    audit_log_input: Path = DEFAULT_AUDIT_LOG_INPUT,
    manual_monitoring_ack: bool = False,
    post_only_crossing_result: Path | None = None,
    post_only_passive_result: Path | None = None,
    use_default_post_only_artifacts: bool = True,
    max_evidence_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
    max_canary_event_age_seconds: float = DEFAULT_LIVE_CANARY_MAX_EVENT_AGE_SECONDS,
    replay_acceptance_symbol: str = "CASHCAT",
    replay_acceptance_mid: float = 0.1015,
    replay_acceptance_newest_per_stream: int | None = 25,
    replay_acceptance_max_price_events: int | None = 2000,
    replay_acceptance_min_price_events_per_day: float = DEFAULT_REPLAY_MIN_PRICE_EVENTS_PER_DAY,
    replay_acceptance_max_price_gap_seconds: float = DEFAULT_REPLAY_MAX_PRICE_GAP_SECONDS,
    replay_acceptance_price_tick_size: float = 0.0,
    replay_acceptance_amount_step_size: float = 0.0,
    replay_acceptance_fill_calibration: Path | None = DEFAULT_REPLAY_FILL_CALIBRATION,
    replay_acceptance_require_fill_calibration: bool = False,
    replay_acceptance_allow_incomplete: bool = True,
    enabled_dry_run_seconds: int = DEFAULT_ENABLED_DRY_RUN_SECONDS,
    enabled_dry_run_min_runtime_seconds: float = DEFAULT_ENABLED_DRY_RUN_MIN_RUNTIME_SECONDS,
    enabled_dry_run_min_event_span_seconds: float = DEFAULT_ENABLED_DRY_RUN_MIN_EVENT_SPAN_SECONDS,
    enabled_dry_run_min_accepted_quotes: int = DEFAULT_ENABLED_DRY_RUN_MIN_ACCEPTED_QUOTES,
    enabled_dry_run_min_order_attempts: int = DEFAULT_ENABLED_DRY_RUN_MIN_ORDER_ATTEMPTS,
    enabled_dry_run_max_loss_rate_usdc_per_hour: float = DEFAULT_ENABLED_DRY_RUN_MAX_LOSS_RATE_USDC_PER_HOUR,
    enabled_dry_run_min_final_total_pnl_usdc: float = DEFAULT_ENABLED_DRY_RUN_MIN_FINAL_TOTAL_PNL_USDC,
) -> list[tuple[str, list[str], list[int]]]:
    py = sys.executable
    gates = [
        ("compileall", [py, "-m", "compileall", "scripts", "user_data/strategies"], [0]),
        (
            "pytest_core",
            [
                py,
                "-m",
                "pytest",
                "tests/test_estimator_common.py",
                "tests/test_hjb.py",
                "tests/test_quote_assembly.py",
                "tests/test_hyperliquid_alo_executor.py",
                "tests/test_hyperliquid_risk_executor.py",
                "tests/test_hyperliquid_fee_capture.py",
                "tests/test_promotion_evidence_manifest.py",
                "tests/test_config_safety.py",
                "tests/test_freqtrade_callback_surface.py",
                "tests/test_freqtrade_tif_runtime.py",
                "tests/test_dry_run_quality.py",
                "tests/test_fee_evidence.py",
                "tests/test_live_canary.py",
                "tests/test_manual_monitoring_ack.py",
                "tests/test_plan_status.py",
                "tests/test_strategy_guards.py",
                "tests/test_strategy_safety.py",
                "tests/test_strategy_attributes.py",
                "tests/test_param_store.py",
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
            "strategy_attribute_report",
            [
                py,
                "scripts/verify_strategy_attributes.py",
                "--strategy",
                "user_data/strategies/Market_Making.py",
                "--output",
                "docs/strategy_attribute_report.json",
            ],
            [0],
        ),
        (
            "compute_spreads_boundary_smoke",
            [
                py,
                "scripts/compute_spreads.py",
                # The traded symbol. --mid must stay on the same scale as the
                # symbol: it is only used to turn model depth into bps and to
                # apply the clamps, but an ETH-scale mid on CASHCAT params makes
                # every printed row meaningless.
                "--crypto",
                "CASHCAT",
                "--mid",
                "0.1015",
                "--qmax",
                "3",
                "--asym-kappa",
                "--skip-refresh",
            ],
            [0],
        ),
        (
            # DELIBERATELY STILL ETH, unlike every other gate here. --synthetic-smoke
            # reads nothing from disk: it builds its own tape out of --mid using
            # ABSOLUTE offsets (bid = mid - 0.5, a trade at mid - 3.0), so the symbol
            # name, the mid and the ETH-fitted kappas/lambdas below are one matched
            # fixture. Renaming the symbol to CASHCAT without rescaling would be a lie;
            # rescaling the mid to 0.1015 would generate negative bids and break it.
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
                "--spread-multiplier",
                "3.0",
            ],
            [0],
        ),
        (
            "adapter_plans",
            [
                py,
                "scripts/run_adapter_plans.py",
            ],
            [0],
        ),
        (
            "post_only_evidence_report",
            post_only_evidence_command(
                py,
                crossing_result=post_only_crossing_result,
                passive_result=post_only_passive_result,
                use_default_artifacts=use_default_post_only_artifacts,
                max_evidence_age_seconds=max_evidence_age_seconds,
            ),
            [0, 1],
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
                    "freqtrade_callback_surface",
                    freqtrade_callback_surface_command(),
                    [0],
                ),
                (
                    "freqtrade_tif_runtime",
                    freqtrade_tif_runtime_command(py),
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
                        str(enabled_dry_run_seconds),
                        "--collector-warmup-seconds",
                        "70",
                        "--json-output",
                        "docs/dry_run_enabled_gate.json",
                    ],
                    [0],
                ),
                (
                    "dry_run_quality_report",
                    [
                        sys.executable,
                        "scripts/verify_dry_run_quality.py",
                        "--input",
                        str(quality_audit_log_input or audit_log_input),
                        "--gate-report",
                        "docs/dry_run_enabled_gate.json",
                        "--output",
                        "docs/dry_run_quality_report.json",
                        "--min-runtime-seconds",
                        str(enabled_dry_run_min_runtime_seconds),
                        "--min-event-span-seconds",
                        str(enabled_dry_run_min_event_span_seconds),
                        "--min-accepted-quotes",
                        str(enabled_dry_run_min_accepted_quotes),
                        "--min-order-attempts",
                        str(enabled_dry_run_min_order_attempts),
                        "--max-quote-distance-ratio",
                        "0.01",
                        "--max-quote-depth-bps",
                        "80",
                        "--min-quote-depth-bps",
                        "3",
                        "--max-order-notional-usdc",
                        "25",
                        "--max-order-amount-units",
                        "0.01",
                        "--max-loss-usdc",
                        "1",
                        "--max-loss-rate-usdc-per-hour",
                        str(enabled_dry_run_max_loss_rate_usdc_per_hour),
                        "--min-final-total-pnl-usdc",
                        str(enabled_dry_run_min_final_total_pnl_usdc),
                    ],
                    [0],
                ),
                (
                    "replay_log_calibration_artifact",
                    replay_log_calibration_command(
                        sys.executable,
                        audit_log_input=audit_log_input,
                    ),
                    [0],
                ),
                (
                    "fee_evidence_report",
                    fee_evidence_command(
                        sys.executable,
                        audit_log_input=audit_log_input,
                        max_evidence_age_seconds=max_evidence_age_seconds,
                    ),
                    [0, 1],
                ),
                (
                    "hl_data_validation_report",
                    [
                        sys.executable,
                        "scripts/validate_hl_data.py",
                        "--symbol",
                        "CASHCAT",
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
                        # Real on-disk tape, so this must be the symbol the
                        # collector is actually writing. --mid is only the
                        # fallback for a price row with no bid/ask, but it has to
                        # be on the tape's scale or a single fallback row would
                        # inject a 2117 mid into a 0.1 tape.
                        "--symbol",
                        "CASHCAT",
                        "--mid",
                        "0.1015",
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
                        "--spread-multiplier",
                        "3.0",
                        "--output",
                        "docs/replay_latest_smoke.json",
                    ],
                    [0],
                ),
                (
                    "replay_acceptance_report_artifact",
                    replay_acceptance_report_command(
                        sys.executable,
                        symbol=replay_acceptance_symbol,
                        mid=replay_acceptance_mid,
                        newest_per_stream=replay_acceptance_newest_per_stream,
                        max_price_events=replay_acceptance_max_price_events,
                        min_price_events_per_day=replay_acceptance_min_price_events_per_day,
                        max_price_gap_seconds=replay_acceptance_max_price_gap_seconds,
                        price_tick_size=replay_acceptance_price_tick_size,
                        amount_step_size=replay_acceptance_amount_step_size,
                        fill_calibration=replay_acceptance_fill_calibration,
                        require_fill_calibration=replay_acceptance_require_fill_calibration,
                        allow_incomplete=replay_acceptance_allow_incomplete,
                    ),
                    [0],
                ),
                (
                    "live_canary_evidence_report",
                    live_canary_evidence_command(
                        sys.executable,
                        audit_log_input=audit_log_input,
                        manual_monitoring_ack=manual_monitoring_ack,
                        max_dependency_report_age_seconds=max_evidence_age_seconds,
                        max_canary_event_age_seconds=max_canary_event_age_seconds,
                    ),
                    [0, 1],
                ),
                (
                    "promotion_evidence_manifest",
                    promotion_evidence_manifest_command(sys.executable),
                    [0],
                ),
            ]
        )
    else:
        gates.append(
            (
                "live_canary_evidence_report",
                live_canary_evidence_command(
                    py,
                    audit_log_input=audit_log_input,
                    manual_monitoring_ack=manual_monitoring_ack,
                    max_dependency_report_age_seconds=max_evidence_age_seconds,
                    max_canary_event_age_seconds=max_canary_event_age_seconds,
                ),
                [0, 1],
            )
        )
        gates.append(
            (
                "promotion_evidence_manifest",
                promotion_evidence_manifest_command(py),
                [0],
            )
        )
    if reuse_smoke_artifacts:
        # The smokes are injected from the previous run's artifacts instead
        # (see reused_smoke_results); everything cheap still runs live.
        skipped = set(SMOKE_GATE_ARTIFACTS)
        if reuse_quality_report:
            # Legacy enabled-smoke artifact without an audit_log_archive: the
            # quality verdict is injected too (see reused_quality_result).
            skipped.add("dry_run_quality_report")
        gates = [gate for gate in gates if gate[0] not in skipped]
    return gates


def manual_gate_specs(
    *,
    include_runtime: bool = False,
    max_report_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> list[ManualGateSpec]:
    gates = [
        ManualGateSpec(
            name="hyperliquid_post_only_mapping",
            reason=(
                "Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to "
                "Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely."
            ),
            report_path="docs/post_only_evidence_report.json",
            max_age_seconds=max_report_age_seconds,
        ),
        ManualGateSpec(
            name="multi_day_event_replay",
            reason="Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.",
            report_path="docs/replay_acceptance_report.json",
            max_age_seconds=max_report_age_seconds,
        ),
        ManualGateSpec(
            name="hyperliquid_fee_tier",
            reason="Requires exchange/account maker-fee evidence and actual maker fill fee rates.",
            report_path="docs/fee_evidence_report.json",
            max_age_seconds=max_report_age_seconds,
        ),
        ManualGateSpec(
            name="live_canary",
            reason="Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.",
            report_path="docs/live_canary_report.json",
            max_age_seconds=max_report_age_seconds,
        ),
    ]
    if not include_runtime:
        gates.insert(
            0,
            ManualGateSpec(
                name="deterministic_dry_run_trading_disabled",
                reason="Requires running the bot loop and confirming zero orders plus health logs in Freqtrade logs.",
                report_path="docs/dry_run_disabled_gate.json",
                ok_field="passed",
                max_age_seconds=None,
            ),
        )
        gates.insert(
            1,
            ManualGateSpec(
                name="freqtrade_runtime_load",
                reason="Requires a Freqtrade environment with exchange/config plugins installed.",
                report_path=None,
            ),
        )
    return gates


def manual_gate_statuses(
    *,
    include_runtime: bool = False,
    max_report_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    for spec in manual_gate_specs(include_runtime=include_runtime, max_report_age_seconds=max_report_age_seconds):
        passed = False
        status_reason = "no_machine_check"
        report_reasons: list[str] = []
        freshness_reason: str | None = None
        report_age_seconds: float | None = None
        report_path = spec.report_path
        if report_path:
            payload, load_error = load_json_report(ROOT / report_path)
            if load_error:
                status_reason = load_error
            else:
                field_passed = payload.get(spec.ok_field) is True
                fresh, freshness_reason, report_age_seconds = report_freshness_status(
                    payload,
                    max_age_seconds=spec.max_age_seconds,
                )
                passed = bool(field_passed and fresh)
                if not field_passed:
                    status_reason = f"{spec.ok_field}_not_true"
                elif not fresh:
                    status_reason = freshness_reason
                else:
                    status_reason = "ok"
                raw_reasons = payload.get("reasons", [])
                if isinstance(raw_reasons, list):
                    report_reasons = [str(reason) for reason in raw_reasons]
        statuses.append(
            {
                "name": spec.name,
                "reason": spec.reason,
                "report_path": report_path,
                "ok_field": spec.ok_field if report_path else None,
                "passed": passed,
                "status_reason": status_reason,
                "report_reasons": report_reasons,
                "freshness_reason": freshness_reason,
                "report_age_seconds": report_age_seconds,
                "max_report_age_seconds": spec.max_age_seconds,
            }
        )
    return statuses


def manual_gates(
    *,
    include_runtime: bool = False,
    max_report_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> list[dict[str, object]]:
    return manual_gate_statuses(include_runtime=include_runtime, max_report_age_seconds=max_report_age_seconds)


def _previous_gate_entries(previous_gates_path: Path) -> tuple[dict[str, dict], str | None]:
    previous, previous_error = load_json_report(ROOT / previous_gates_path)
    previous_gates: dict[str, dict] = {}
    if previous is not None:
        for entry in previous.get("local_gates", []):
            if isinstance(entry, dict) and entry.get("name"):
                previous_gates[str(entry["name"])] = entry
    return previous_gates, previous_error


def _reused_gate_entry(
    name: str,
    prior: dict | None,
    previous_error: str | None,
    artifact_path: Path,
    *,
    ok_field: str,
    ts_field: str,
    max_age_seconds: float,
    now: datetime,
) -> dict:
    """One payload-ready reused gate entry, fail-closed on any doubt."""
    failure_reason: str | None = None
    artifact_timestamp: str | None = None

    if previous_error:
        failure_reason = f"previous_gates_unreadable:{previous_error}"
    elif prior is None:
        failure_reason = "missing_previous_gate_result"
    elif prior.get("passed") is not True:
        failure_reason = "previous_gate_not_passed"
    else:
        artifact, artifact_error = load_json_report(ROOT / artifact_path)
        if artifact_error:
            failure_reason = f"artifact_unreadable:{artifact_error}"
        elif artifact.get(ok_field) is not True:
            failure_reason = f"artifact_{ok_field}_not_true"
        else:
            timestamp = parse_report_timestamp(artifact.get(ts_field))
            if timestamp is None:
                failure_reason = f"artifact_missing_{ts_field}"
            else:
                artifact_timestamp = str(artifact.get(ts_field))
                age_seconds = (now - timestamp).total_seconds()
                if age_seconds < 0 or age_seconds > float(max_age_seconds):
                    failure_reason = (
                        f"artifact_stale:age_{age_seconds:.0f}s>max_{float(max_age_seconds):.0f}s"
                    )

    if failure_reason is None:
        return {
            "name": name,
            "command": list(prior.get("command") or []),
            "passed": True,
            "returncode": int(prior.get("returncode", 0)),
            "expected_returncodes": list(prior.get("expected_returncodes") or [0]),
            "elapsed_seconds": 0.0,
            "stdout_tail": f"reused_from:{artifact_timestamp}",
            "stderr_tail": "",
            "reused": True,
            "reused_from": artifact_timestamp,
        }
    return {
        "name": name,
        "command": [],
        "passed": False,
        "returncode": -1,
        "expected_returncodes": [0],
        "elapsed_seconds": 0.0,
        "stdout_tail": "",
        "stderr_tail": f"reuse_rejected:{failure_reason}",
        "reused": True,
        "reused_from": None,
    }


def reused_smoke_results(
    previous_gates_path: Path,
    max_age_seconds: float,
    now: datetime | None = None,
) -> list[dict]:
    """Payload-ready gate entries for the two smokes, reused from a prior run.

    A smoke result is reused only when the previous battery recorded it as
    passed AND its artifact still exists, reports passed, and its started_at
    is within max_age_seconds. Anything else injects a FAILED entry — a reuse
    run can never look greener than its inputs.
    """
    now = now or datetime.now(timezone.utc)
    previous_gates, previous_error = _previous_gate_entries(previous_gates_path)
    return [
        _reused_gate_entry(
            name,
            previous_gates.get(name),
            previous_error,
            artifact_rel,
            ok_field="passed",
            ts_field="started_at",
            max_age_seconds=max_age_seconds,
            now=now,
        )
        for name, artifact_rel in SMOKE_GATE_ARTIFACTS.items()
    ]


def reused_quality_result(
    previous_gates_path: Path,
    max_age_seconds: float,
    now: datetime | None = None,
) -> dict:
    """Reused dry_run_quality_report entry for legacy smoke artifacts.

    Used only when the enabled-smoke gate report carries no audit_log_archive
    (pre-archive artifacts): the live audit log can no longer reproduce the
    smoke window, so the previous battery's quality verdict is reused,
    validated against docs/dry_run_quality_report.json (ok + generated_at).
    """
    now = now or datetime.now(timezone.utc)
    previous_gates, previous_error = _previous_gate_entries(previous_gates_path)
    return _reused_gate_entry(
        "dry_run_quality_report",
        previous_gates.get("dry_run_quality_report"),
        previous_error,
        QUALITY_REPORT_ARTIFACT,
        ok_field="ok",
        ts_field="generated_at",
        max_age_seconds=max_age_seconds,
        now=now,
    )


def production_restore_commands() -> list[list[str]]:
    """The exact documented manual recovery for the post-battery stopped stack."""
    return [
        ["docker", "rm", "-f", PRODUCTION_FREQTRADE_CONTAINER],
        ["docker", "compose", "up", "-d", PRODUCTION_FREQTRADE_SERVICE, COLLECTOR_SERVICE],
    ]


def restore_production_container() -> dict:
    outcomes: list[dict] = []
    for command in production_restore_commands():
        try:
            proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            outcomes.append({"command": command, "returncode": proc.returncode, "stderr_tail": tail(proc.stderr, 500)})
        except Exception as exc:  # docker missing etc. — never crash the battery
            outcomes.append({"command": command, "returncode": -1, "stderr_tail": str(exc)})
    ok = bool(outcomes) and outcomes[-1]["returncode"] == 0
    return {"attempted": True, "ok": ok, "commands": outcomes}


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
    if payload.get("smoke_artifacts_reused"):
        lines.append("Smoke artifacts: REUSED from previous battery (freshness-validated)")
    restore = payload.get("production_restore") or {}
    if restore.get("attempted"):
        lines.append(f"Production restore: {'OK' if restore.get('ok') else 'FAILED (restart MM_ADV manually)'}")
    lines.append("")
    for result in payload["local_gates"]:
        status = "PASS" if result["passed"] else "FAIL"
        reused_marker = " (reused)" if result.get("reused") else ""
        lines.append(f"- {status} `{result['name']}` ({result['elapsed_seconds']}s){reused_marker}")
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
    manual_gates = payload.get("manual_gates", [])
    if manual_gates:
        lines.append("")
        lines.append("Manual/external gate evidence:")
        for gate in manual_gates:
            status = "PASS" if gate.get("passed") else "WAIT"
            lines.append(f"- {status} `{gate['name']}`: {gate['reason']}")
            if not gate.get("passed") and gate.get("status_reason"):
                lines.append(f"  - reason: `{gate['status_reason']}`")
    blockers = set(str(name) for name in payload.get("deployment_blockers", []))
    if blockers:
        lines.append("")
        lines.append("Manual gates still required:")
        for gate in manual_gates:
            if str(gate.get("name")) in blockers:
                lines.append(f"- `{gate['name']}`: {gate['reason']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local market-making safety gates.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for full JSON results.")
    parser.add_argument("--markdown-output", type=Path, default=None, help="Optional path for markdown summary.")
    parser.add_argument("--include-runtime", action="store_true", help="Also run non-trading Docker/Freqtrade load gates.")
    parser.add_argument(
        "--reuse-smoke-artifacts",
        action="store_true",
        help=(
            "With --include-runtime: skip re-running the two long dry-run smokes and reuse the "
            "previous battery's results, validated against the smoke artifacts' freshness. "
            "Everything cheap (pytest, probes, evaluators, replays) still runs live."
        ),
    )
    parser.add_argument(
        "--previous-gates",
        type=Path,
        default=DEFAULT_PREVIOUS_GATES,
        help="Previous battery JSON to reuse smoke results from (default docs/last_safety_gates.json).",
    )
    parser.add_argument(
        "--max-smoke-artifact-age-seconds",
        type=float,
        default=DEFAULT_MAX_SMOKE_ARTIFACT_AGE_SECONDS,
        help="Maximum age of the reused smoke artifacts' started_at (default 21600 = 6h).",
    )
    parser.add_argument(
        "--no-restore-production",
        action="store_true",
        help="Skip restarting the production freqtrade container after the runtime smokes.",
    )
    parser.add_argument(
        "--audit-log-input",
        type=Path,
        default=DEFAULT_AUDIT_LOG_INPUT,
        help=(
            "JSONL audit log for replay calibration, fee evidence, and live canary evidence. "
            "Defaults to user_data/logs/mm_debug.jsonl."
        ),
    )
    parser.add_argument(
        "--manual-monitoring-ack",
        action="store_true",
        help=(
            "Pass the live-canary manual monitoring acknowledgement through to "
            "scripts/verify_live_canary.py. Use only after actual monitored canary sessions."
        ),
    )
    parser.add_argument(
        "--enabled-dry-run-seconds",
        type=int,
        default=DEFAULT_ENABLED_DRY_RUN_SECONDS,
        help=(
            "Seconds for the trading_enabled=True dry-run runtime gate. "
            "Use a larger value, such as 1800, for promotion evidence."
        ),
    )
    parser.add_argument(
        "--enabled-dry-run-min-runtime-seconds",
        type=float,
        default=DEFAULT_ENABLED_DRY_RUN_MIN_RUNTIME_SECONDS,
        help="Minimum reported runtime accepted by the enabled dry-run quality gate.",
    )
    parser.add_argument(
        "--enabled-dry-run-min-event-span-seconds",
        type=float,
        default=DEFAULT_ENABLED_DRY_RUN_MIN_EVENT_SPAN_SECONDS,
        help="Minimum audit-log event span accepted by the enabled dry-run quality gate.",
    )
    parser.add_argument(
        "--enabled-dry-run-min-accepted-quotes",
        type=int,
        default=DEFAULT_ENABLED_DRY_RUN_MIN_ACCEPTED_QUOTES,
        help="Minimum accepted quote decisions for the enabled dry-run quality gate.",
    )
    parser.add_argument(
        "--enabled-dry-run-min-order-attempts",
        type=int,
        default=DEFAULT_ENABLED_DRY_RUN_MIN_ORDER_ATTEMPTS,
        help="Minimum quote-linked accepted order attempts for the enabled dry-run quality gate.",
    )
    parser.add_argument(
        "--enabled-dry-run-max-loss-rate-usdc-per-hour",
        type=float,
        default=DEFAULT_ENABLED_DRY_RUN_MAX_LOSS_RATE_USDC_PER_HOUR,
        help="Maximum allowed negative PnL velocity for the enabled dry-run quality gate.",
    )
    parser.add_argument(
        "--enabled-dry-run-min-final-total-pnl-usdc",
        type=float,
        default=DEFAULT_ENABLED_DRY_RUN_MIN_FINAL_TOTAL_PNL_USDC,
        help=(
            "Minimum final total PnL accepted by the enabled dry-run quality gate. "
            "Use 0 for promotion runs that require break-even-or-better dry-run evidence."
        ),
    )
    parser.add_argument(
        "--max-evidence-age-seconds",
        type=float,
        default=DEFAULT_REPORT_MAX_AGE_SECONDS,
        help=(
            "Maximum age for external evidence reports and underlying post-only/fee/dependency "
            "evidence consumed by the gate runner. Defaults to 86400 seconds."
        ),
    )
    parser.add_argument(
        "--max-canary-event-age-seconds",
        type=float,
        default=DEFAULT_LIVE_CANARY_MAX_EVENT_AGE_SECONDS,
        help="Maximum age for live-canary audit events. Defaults to 604800 seconds.",
    )
    parser.add_argument(
        "--post-only-crossing-result",
        type=Path,
        default=None,
        help=(
            "Crossing Alo evidence artifact for scripts/verify_post_only_mapping.py. "
            "Defaults to docs/post_only_crossing_result.json when that file exists."
        ),
    )
    parser.add_argument(
        "--post-only-passive-result",
        type=Path,
        default=None,
        help=(
            "Passive Alo evidence artifact for scripts/verify_post_only_mapping.py. "
            "Defaults to docs/post_only_passive_result.json when that file exists."
        ),
    )
    parser.add_argument("--replay-acceptance-symbol", default="CASHCAT", help="Symbol for the replay acceptance report.")
    parser.add_argument(
        "--replay-acceptance-mid",
        type=float,
        default=0.1015,
        help="Mid-price fallback for the replay acceptance report (must match --replay-acceptance-symbol's scale).",
    )
    parser.add_argument(
        "--replay-acceptance-newest-per-stream",
        type=int,
        default=25,
        help="Newest shards per stream for the replay acceptance report. Use 0 to scan all available shards.",
    )
    parser.add_argument(
        "--replay-acceptance-max-price-events",
        type=int,
        default=2000,
        help="Maximum price events for the replay acceptance report. Use 0 for no event cap.",
    )
    parser.add_argument(
        "--replay-acceptance-min-price-events-per-day",
        type=float,
        default=DEFAULT_REPLAY_MIN_PRICE_EVENTS_PER_DAY,
        help="Minimum replay price events per day required by the acceptance report.",
    )
    parser.add_argument(
        "--replay-acceptance-max-price-gap-seconds",
        type=float,
        default=DEFAULT_REPLAY_MAX_PRICE_GAP_SECONDS,
        help="Maximum allowed gap between replay price events in the acceptance report.",
    )
    parser.add_argument(
        "--replay-acceptance-price-tick-size",
        type=float,
        default=0.0,
        help="Price tick size used by replay acceptance rounding. Defaults to disabled.",
    )
    parser.add_argument(
        "--replay-acceptance-amount-step-size",
        type=float,
        default=0.0,
        help="Base amount step size used by replay acceptance sizing. Defaults to disabled.",
    )
    parser.add_argument(
        "--replay-acceptance-require-pass",
        action="store_true",
        help="Omit --allow-incomplete so the replay acceptance command fails when report.ok is false.",
    )
    parser.add_argument(
        "--replay-acceptance-fill-calibration",
        type=Path,
        default=DEFAULT_REPLAY_FILL_CALIBRATION,
        help="Replay-log calibration artifact to pass into replay acceptance.",
    )
    parser.add_argument(
        "--replay-acceptance-require-fill-calibration",
        action="store_true",
        help="Require replay acceptance variants to apply a usable fill calibration artifact.",
    )
    parser.add_argument(
        "--plan-status-audit-output",
        type=Path,
        default=Path("docs/plan_status_audit.json"),
        help="Path for the post-run PLAN status audit artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reuse_smoke_artifacts and not args.include_runtime:
        print("--reuse-smoke-artifacts requires --include-runtime", file=sys.stderr)
        return 2
    # In reuse mode the quality evaluator needs a stable input: the audit-event
    # slice archived by the enabled smoke. Legacy gate artifacts without an
    # archive fall back to reusing the previous quality verdict (fail-closed).
    reuse_quality_report = False
    quality_audit_log_input: Path | None = None
    if args.reuse_smoke_artifacts:
        enabled_gate_report, _enabled_error = load_json_report(
            ROOT / SMOKE_GATE_ARTIFACTS["dry_run_enabled_smoke"]
        )
        archive_raw = (enabled_gate_report or {}).get("audit_log_archive")
        if archive_raw and Path(str(archive_raw)).exists():
            quality_audit_log_input = Path(str(archive_raw))
        else:
            reuse_quality_report = True
    results = [
        run_command(name, command, expected_returncodes)
        for name, command, expected_returncodes in local_gates(
            include_runtime=args.include_runtime,
            reuse_smoke_artifacts=args.reuse_smoke_artifacts,
            reuse_quality_report=reuse_quality_report,
            quality_audit_log_input=quality_audit_log_input,
            audit_log_input=args.audit_log_input,
            manual_monitoring_ack=args.manual_monitoring_ack,
            post_only_crossing_result=args.post_only_crossing_result,
            post_only_passive_result=args.post_only_passive_result,
            max_evidence_age_seconds=args.max_evidence_age_seconds,
            max_canary_event_age_seconds=args.max_canary_event_age_seconds,
            replay_acceptance_symbol=args.replay_acceptance_symbol,
            replay_acceptance_mid=args.replay_acceptance_mid,
            replay_acceptance_newest_per_stream=args.replay_acceptance_newest_per_stream,
            replay_acceptance_max_price_events=args.replay_acceptance_max_price_events,
            replay_acceptance_min_price_events_per_day=args.replay_acceptance_min_price_events_per_day,
            replay_acceptance_max_price_gap_seconds=args.replay_acceptance_max_price_gap_seconds,
            replay_acceptance_price_tick_size=args.replay_acceptance_price_tick_size,
            replay_acceptance_amount_step_size=args.replay_acceptance_amount_step_size,
            replay_acceptance_fill_calibration=args.replay_acceptance_fill_calibration,
            replay_acceptance_require_fill_calibration=args.replay_acceptance_require_fill_calibration,
            replay_acceptance_allow_incomplete=not args.replay_acceptance_require_pass,
            enabled_dry_run_seconds=int(args.enabled_dry_run_seconds),
            enabled_dry_run_min_runtime_seconds=float(args.enabled_dry_run_min_runtime_seconds),
            enabled_dry_run_min_event_span_seconds=float(args.enabled_dry_run_min_event_span_seconds),
            enabled_dry_run_min_accepted_quotes=int(args.enabled_dry_run_min_accepted_quotes),
            enabled_dry_run_min_order_attempts=int(args.enabled_dry_run_min_order_attempts),
            enabled_dry_run_max_loss_rate_usdc_per_hour=float(args.enabled_dry_run_max_loss_rate_usdc_per_hour),
            enabled_dry_run_min_final_total_pnl_usdc=float(args.enabled_dry_run_min_final_total_pnl_usdc),
        )
    ]
    reused_entries: list[dict] = []
    if args.reuse_smoke_artifacts:
        reused_entries = reused_smoke_results(
            args.previous_gates,
            float(args.max_smoke_artifact_age_seconds),
        )
        if reuse_quality_report:
            reused_entries.append(
                reused_quality_result(
                    args.previous_gates,
                    float(args.max_smoke_artifact_age_seconds),
                )
            )
    manual_gate_list = manual_gates(
        include_runtime=args.include_runtime,
        max_report_age_seconds=float(args.max_evidence_age_seconds),
    )
    deployment_blockers = [str(gate["name"]) for gate in manual_gate_list if gate.get("passed") is not True]
    all_local_passed = all(result.passed for result in results) and all(
        entry.get("passed") is True for entry in reused_entries
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "local_gates": [asdict(result) for result in results] + reused_entries,
        "smoke_artifacts_reused": bool(args.reuse_smoke_artifacts),
        "manual_gates": manual_gate_list,
        "manual_gates_remaining": len(deployment_blockers),
        "all_local_passed": all_local_passed,
        "post_run_audits": [],
        "all_automated_passed": all_local_passed,
        "deployment_ready": bool(all_local_passed and not deployment_blockers),
        "deployment_blockers": deployment_blockers,
        "runtime_gates_included": bool(args.include_runtime),
        "audit_log_input": str(args.audit_log_input),
        "manual_monitoring_ack": bool(args.manual_monitoring_ack),
        "evidence_freshness": {
            "max_evidence_age_seconds": float(args.max_evidence_age_seconds),
            "max_canary_event_age_seconds": float(args.max_canary_event_age_seconds),
        },
        "post_only_crossing_result": str(args.post_only_crossing_result)
        if args.post_only_crossing_result
        else None,
        "post_only_passive_result": str(args.post_only_passive_result)
        if args.post_only_passive_result
        else None,
        "replay_acceptance": {
            "symbol": str(args.replay_acceptance_symbol),
            "mid": float(args.replay_acceptance_mid),
            "newest_per_stream": optional_positive_int(args.replay_acceptance_newest_per_stream),
            "max_price_events": optional_positive_int(args.replay_acceptance_max_price_events),
            "min_price_events_per_day": float(args.replay_acceptance_min_price_events_per_day),
            "max_price_gap_seconds": float(args.replay_acceptance_max_price_gap_seconds),
            "price_tick_size": float(args.replay_acceptance_price_tick_size),
            "amount_step_size": float(args.replay_acceptance_amount_step_size),
            "fill_calibration": str(args.replay_acceptance_fill_calibration)
            if args.replay_acceptance_fill_calibration
            else None,
            "require_fill_calibration": bool(args.replay_acceptance_require_fill_calibration),
            "allow_incomplete": not bool(args.replay_acceptance_require_pass),
        },
        "enabled_dry_run": {
            "seconds": int(args.enabled_dry_run_seconds),
            "min_runtime_seconds": float(args.enabled_dry_run_min_runtime_seconds),
            "min_event_span_seconds": float(args.enabled_dry_run_min_event_span_seconds),
            "min_accepted_quotes": int(args.enabled_dry_run_min_accepted_quotes),
            "min_order_attempts": int(args.enabled_dry_run_min_order_attempts),
            "max_loss_rate_usdc_per_hour": float(args.enabled_dry_run_max_loss_rate_usdc_per_hour),
            "min_final_total_pnl_usdc": float(args.enabled_dry_run_min_final_total_pnl_usdc),
        },
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

    # The disabled smoke replaces the production freqtrade container; bring it
    # back so quoting does not stay silently halted after a runtime battery.
    payload["production_restore"] = {"attempted": False}
    if args.include_runtime and not args.reuse_smoke_artifacts and not args.no_restore_production:
        payload["production_restore"] = restore_production_container()
        if not payload["production_restore"].get("ok"):
            print(
                "WARNING: failed to restore the production freqtrade container; "
                "run 'docker rm -f MM_ADV' then 'docker compose up -d freqtrade' manually.",
                file=sys.stderr,
            )

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
