#!/usr/bin/env python3
"""Generate a replay acceptance report over collected Hyperliquid data.

The report is intentionally stricter than the latest-data smoke. It is the
operator-facing Gate 5 harness: enough data, baseline replay, fee/latency/param
sensitivity, and explicit pass/fail reasons. Use `--allow-incomplete` when you
want an artifact even though the local data window is too short.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from replay_market_maker import MAKER_FEE, TAKER_FEE, ReplayConfig, ReplayMetrics, run_replay  # noqa: E402


@dataclass(frozen=True)
class ReplayVariant:
    name: str
    params_multiplier: float = 1.0
    epsilon_add: float = 0.0
    maker_fee_multiplier: float = 1.0
    latency_multiplier: float = 1.0


DEFAULT_VARIANTS = (
    ReplayVariant("baseline"),
    ReplayVariant("fee_2x", maker_fee_multiplier=2.0),
    ReplayVariant("latency_2x", latency_multiplier=2.0),
    ReplayVariant("params_soft", params_multiplier=0.8, epsilon_add=0.01),
    ReplayVariant("params_hard", params_multiplier=1.2, epsilon_add=0.02),
)

REQUIRED_MARKOUT_HORIZONS_MS = (100, 1_000, 5_000, 30_000)


def parse_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(value)


def coverage_days(metrics: dict[str, Any]) -> float:
    start = parse_ts(metrics.get("data_start"))
    end = parse_ts(metrics.get("data_end"))
    if start is None or end is None:
        return 0.0
    return max(0.0, float((end - start).total_seconds()) / 86_400.0)


def markout_summary(markout_samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_horizon: dict[str, list[float]] = {}
    for sample in markout_samples:
        horizon = str(sample.get("horizon_ms"))
        try:
            value = float(sample.get("markout_usdc", 0.0))
        except Exception:
            continue
        by_horizon.setdefault(horizon, []).append(value)
    return {
        horizon: {
            "count": float(len(values)),
            "mean_usdc": float(np.mean(values)) if values else 0.0,
            "min_usdc": float(np.min(values)) if values else 0.0,
        }
        for horizon, values in sorted(by_horizon.items(), key=lambda item: int(item[0]))
    }


def variant_params(base: dict[str, float], variant: ReplayVariant) -> dict[str, float]:
    params = dict(base)
    for key in ("kappa+", "kappa-", "lambda+", "lambda-"):
        params[key] = float(params[key]) * float(variant.params_multiplier)
    params["epsilon+"] = float(params.get("epsilon+", 0.0)) + float(variant.epsilon_add)
    params["epsilon-"] = float(params.get("epsilon-", 0.0)) + float(variant.epsilon_add)
    return params


def variant_config(base: ReplayConfig, variant: ReplayVariant) -> ReplayConfig:
    multiplier = float(variant.latency_multiplier)
    return ReplayConfig(
        symbol=base.symbol,
        data_dir=base.data_dir,
        mid_fallback=base.mid_fallback,
        inventory_unit_base=base.inventory_unit_base,
        q_max=base.q_max,
        decision_latency_ms=int(round(base.decision_latency_ms * multiplier)),
        order_ack_latency_ms=int(round(base.order_ack_latency_ms * multiplier)),
        cancel_latency_ms=int(round(base.cancel_latency_ms * multiplier)),
        maker_fee=float(base.maker_fee) * float(variant.maker_fee_multiplier),
        taker_fee=base.taker_fee,
        funding_rate_per_hour=base.funding_rate_per_hour,
        starting_equity_usdc=base.starting_equity_usdc,
        leverage=base.leverage,
        maintenance_margin_rate=base.maintenance_margin_rate,
        queue_decay_per_second=base.queue_decay_per_second,
        newest_per_stream=base.newest_per_stream,
        max_price_events=base.max_price_events,
    )


def net_realized_spread_usdc(metrics: dict[str, Any]) -> float:
    return float(metrics.get("realized_spread_usdc") or 0.0) - float(metrics.get("fees_usdc") or 0.0)


def directional_pnl_proxy_usdc(metrics: dict[str, Any]) -> float:
    """Approximate PnL not explained by realized spread capture after fees."""
    mark_to_market = float(metrics.get("mark_to_market_pnl_usdc") or 0.0)
    return mark_to_market - net_realized_spread_usdc(metrics)


def directional_drift_ratio(metrics: dict[str, Any]) -> float:
    directional = abs(directional_pnl_proxy_usdc(metrics))
    scale = max(
        abs(float(metrics.get("mark_to_market_pnl_usdc") or 0.0)),
        abs(net_realized_spread_usdc(metrics)),
        1e-12,
    )
    return directional / scale


def replay_param_guard(params: dict[str, Any], *, max_toxicity: float = 1.5) -> tuple[bool, str]:
    required = ("kappa+", "kappa-", "lambda+", "lambda-", "epsilon+", "epsilon-")
    parsed: dict[str, float] = {}
    for key in required:
        if key not in params:
            return False, f"missing_{key}"
        try:
            value = float(params[key])
        except Exception:
            return False, f"nonfinite_{key}"
        if not np.isfinite(value):
            return False, f"nonfinite_{key}"
        parsed[key] = value

    if parsed["kappa+"] <= 0 or parsed["kappa-"] <= 0:
        return False, "invalid_kappa"
    if parsed["lambda+"] < 0 or parsed["lambda-"] < 0:
        return False, "invalid_lambda"
    if parsed["epsilon+"] < 0 or parsed["epsilon-"] < 0:
        return False, "invalid_epsilon"

    toxicity = max(parsed["kappa+"] * parsed["epsilon+"], parsed["kappa-"] * parsed["epsilon-"])
    if toxicity > float(max_toxicity):
        return False, "toxicity_too_high"

    return True, "ok"


PARAMETER_SERIES_FIELDS = (
    "kappa_plus",
    "kappa_minus",
    "lambda_plus",
    "lambda_minus",
    "epsilon_plus",
    "epsilon_minus",
)
TOXICITY_SERIES_FIELDS = ("toxicity_plus", "toxicity_minus")


def is_valid_timestamp(value: Any) -> bool:
    if not value:
        return False
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return False
    return not bool(pd.isna(ts))


def is_finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed))


def add_series_evidence_reasons(
    metrics: dict[str, Any],
    reasons: list[str],
    *,
    series_key: str,
    numeric_fields: tuple[str, ...],
) -> None:
    series = metrics.get(series_key) or []
    if not isinstance(series, list) or not series:
        reasons.append(f"missing_{series_key}")
        return

    for idx, row in enumerate(series):
        if not isinstance(row, dict):
            reasons.append(f"invalid_{series_key}_row:{idx}")
            continue
        if not is_valid_timestamp(row.get("ts")):
            reasons.append(f"{series_key}_missing_or_invalid_ts:{idx}")
        if not row.get("source"):
            reasons.append(f"{series_key}_missing_source:{idx}")
        if not row.get("schema_version"):
            reasons.append(f"{series_key}_missing_schema_version:{idx}")
        if not row.get("symbol"):
            reasons.append(f"{series_key}_missing_symbol:{idx}")
        for key in numeric_fields:
            if not is_finite_number(row.get(key)):
                reasons.append(f"{series_key}_nonfinite:{idx}:{key}")


def add_depth_ratio_reasons(metrics: dict[str, Any], reasons: list[str]) -> None:
    quote_attempts = int(metrics.get("quote_attempts") or 0)
    maker_fills = int(metrics.get("maker_fills") or 0)
    attempts_by_depth = metrics.get("quote_attempts_by_depth") or {}
    fills_by_depth = metrics.get("fills_by_depth") or {}
    ratios_by_depth = metrics.get("fill_ratio_by_depth") or {}

    if quote_attempts > 0 and not isinstance(attempts_by_depth, dict):
        reasons.append("invalid_quote_attempts_by_depth")
        attempts_by_depth = {}
    if maker_fills > 0 and not isinstance(fills_by_depth, dict):
        reasons.append("invalid_fills_by_depth")
        fills_by_depth = {}
    if quote_attempts > 0 and not isinstance(ratios_by_depth, dict):
        reasons.append("invalid_fill_ratio_by_depth")
        ratios_by_depth = {}

    if quote_attempts > 0 and not attempts_by_depth:
        reasons.append("missing_quote_attempts_by_depth")
    if quote_attempts > 0 and not ratios_by_depth:
        reasons.append("missing_fill_ratio_by_depth")
    if maker_fills > 0 and not fills_by_depth:
        reasons.append("missing_fills_by_depth")

    parsed_attempts: dict[str, int] = {}
    parsed_fills: dict[str, int] = {}
    for key, value in attempts_by_depth.items():
        try:
            attempts = int(value)
        except Exception:
            reasons.append(f"noninteger_quote_attempts_by_depth:{key}")
            continue
        if attempts <= 0:
            reasons.append(f"nonpositive_quote_attempts_by_depth:{key}")
            continue
        parsed_attempts[str(key)] = attempts

    for key, value in fills_by_depth.items():
        try:
            fills = int(value)
        except Exception:
            reasons.append(f"noninteger_fills_by_depth:{key}")
            continue
        if fills < 0:
            reasons.append(f"negative_fills_by_depth:{key}")
            continue
        parsed_fills[str(key)] = fills
        if str(key) not in parsed_attempts:
            reasons.append(f"fills_depth_without_quote_attempts:{key}")
        elif fills > parsed_attempts[str(key)]:
            reasons.append(f"fills_exceed_quote_attempts_by_depth:{key}")

    if maker_fills > 0 and sum(parsed_fills.values()) != maker_fills:
        reasons.append(f"fills_by_depth_total_mismatch:{sum(parsed_fills.values())}!={maker_fills}")

    for key, value in ratios_by_depth.items():
        ratio = float(value) if is_finite_number(value) else None
        if ratio is None:
            reasons.append(f"nonfinite_fill_ratio_by_depth:{key}")
            continue
        if ratio < 0.0 or ratio > 1.0:
            reasons.append(f"fill_ratio_by_depth_out_of_range:{key}:{ratio:.6f}")
            continue
        attempts = parsed_attempts.get(str(key))
        if attempts is None:
            reasons.append(f"ratio_depth_without_quote_attempts:{key}")
            continue
        expected = parsed_fills.get(str(key), 0) / max(attempts, 1)
        if abs(ratio - expected) > 1e-12:
            reasons.append(f"fill_ratio_by_depth_mismatch:{key}:{ratio:.12f}!={expected:.12f}")

    if maker_fills > 0 and not any(float(value) > 0.0 for value in ratios_by_depth.values() if is_finite_number(value)):
        reasons.append("no_positive_fill_ratio_by_depth")


def replay_data_fresh_guard(
    metrics: dict[str, Any],
    *,
    reference_time: pd.Timestamp,
    max_age_seconds: float,
) -> tuple[bool, str, float | None]:
    data_end = parse_ts(metrics.get("data_end"))
    if data_end is None:
        return False, "no_collector_data", None
    age_seconds = max(0.0, float((reference_time - data_end).total_seconds()))
    if age_seconds > float(max_age_seconds):
        return False, "stale_collector_data", age_seconds
    return True, "ok", age_seconds


def build_refusal_checks(
    *,
    params: dict[str, float],
    baseline_metrics: dict[str, Any],
    max_toxicity: float = 1.5,
    max_data_age_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    bad_kappa_params = dict(params)
    bad_kappa_params["kappa+"] = 0.0
    ok, reason = replay_param_guard(bad_kappa_params, max_toxicity=max_toxicity)
    checks.append(
        {
            "name": "bad_params_nonpositive_kappa",
            "expected_decision": "reject",
            "decision": "accept" if ok else "reject",
            "reason": reason,
            "ok": (not ok and reason == "invalid_kappa"),
        }
    )

    toxic_params = dict(params)
    toxic_params["epsilon+"] = max(float(toxic_params.get("epsilon+", 0.0)), float(max_toxicity) + 0.1)
    ok, reason = replay_param_guard(toxic_params, max_toxicity=max_toxicity)
    checks.append(
        {
            "name": "bad_params_toxicity",
            "expected_decision": "reject",
            "decision": "accept" if ok else "reject",
            "reason": reason,
            "ok": (not ok and reason == "toxicity_too_high"),
        }
    )

    data_end = parse_ts(baseline_metrics.get("data_end"))
    reference = (
        data_end + pd.Timedelta(seconds=float(max_data_age_seconds) + 1.0)
        if data_end is not None
        else pd.Timestamp.now(tz="UTC")
    )
    ok, reason, age_seconds = replay_data_fresh_guard(
        baseline_metrics,
        reference_time=reference,
        max_age_seconds=max_data_age_seconds,
    )
    checks.append(
        {
            "name": "stale_collector_data",
            "expected_decision": "reject",
            "decision": "accept" if ok else "reject",
            "reason": reason,
            "age_seconds": age_seconds,
            "max_age_seconds": float(max_data_age_seconds),
            "ok": (not ok and reason == "stale_collector_data"),
        }
    )

    return checks


def evaluate_metrics(
    metrics: dict[str, Any],
    *,
    min_days: float,
    q_max: int,
    min_quote_attempts: int,
    min_maker_ratio: float,
    min_net_realized_spread: float,
    min_mean_markout_usdc: float,
    max_directional_drift_ratio: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    add_series_evidence_reasons(
        metrics,
        reasons,
        series_key="parameter_series",
        numeric_fields=PARAMETER_SERIES_FIELDS,
    )
    add_series_evidence_reasons(
        metrics,
        reasons,
        series_key="toxicity_series",
        numeric_fields=TOXICITY_SERIES_FIELDS,
    )
    add_depth_ratio_reasons(metrics, reasons)

    days = coverage_days(metrics)
    if days < float(min_days):
        reasons.append(f"insufficient_coverage_days:{days:.6f}<min_{float(min_days):.6f}")

    quote_attempts = int(metrics.get("quote_attempts") or 0)
    if quote_attempts < int(min_quote_attempts):
        reasons.append(f"insufficient_quote_attempts:{quote_attempts}<min_{int(min_quote_attempts)}")

    maker_fills = int(metrics.get("maker_fills") or 0)
    taker_fills = int(metrics.get("taker_fills") or 0)
    if taker_fills != 0:
        reasons.append(f"taker_fills_nonzero:{taker_fills}")
    if maker_fills <= 0:
        reasons.append("no_maker_fills")

    liquidation_breaches = int(metrics.get("liquidation_breach_events") or 0)
    if liquidation_breaches:
        reasons.append(f"maintenance_margin_breached:{liquidation_breaches}")

    maker_ratio = float(metrics.get("maker_ratio") or 0.0)
    if maker_fills > 0 and maker_ratio < float(min_maker_ratio):
        reasons.append(f"maker_ratio_below_threshold:{maker_ratio:.6f}<min_{float(min_maker_ratio):.6f}")

    net_spread = net_realized_spread_usdc(metrics)
    if maker_fills > 0 and net_spread < float(min_net_realized_spread):
        reasons.append(f"net_realized_spread_below_threshold:{net_spread:.8f}<min_{float(min_net_realized_spread):.8f}")

    drift_ratio = directional_drift_ratio(metrics)
    if maker_fills > 0 and drift_ratio > float(max_directional_drift_ratio):
        reasons.append(
            f"directional_drift_dominates_pnl:{drift_ratio:.6f}>max_{float(max_directional_drift_ratio):.6f}"
        )

    inventory_histogram = metrics.get("inventory_histogram") or {}
    for raw_q in inventory_histogram.keys():
        try:
            q = int(raw_q)
        except Exception:
            reasons.append(f"invalid_inventory_bucket:{raw_q}")
            continue
        if q < 0 or q > int(q_max):
            reasons.append(f"inventory_out_of_bounds:{q}")

    summary = markout_summary(metrics.get("markout_samples") or [])
    if maker_fills > 0 and not summary:
        reasons.append("missing_markout_samples")
    if maker_fills > 0:
        for horizon_ms in REQUIRED_MARKOUT_HORIZONS_MS:
            if str(horizon_ms) not in summary:
                reasons.append(f"missing_markout_horizon:{horizon_ms}ms")
    for horizon, stats in summary.items():
        if float(stats["mean_usdc"]) < float(min_mean_markout_usdc):
            reasons.append(
                f"markout_mean_below_threshold:{horizon}ms:{float(stats['mean_usdc']):.8f}<min_{float(min_mean_markout_usdc):.8f}"
            )

    return not reasons, reasons


def build_report(
    *,
    config: ReplayConfig,
    params: dict[str, float],
    min_days: float,
    min_quote_attempts: int,
    min_maker_ratio: float,
    min_net_realized_spread: float,
    min_mean_markout_usdc: float,
    max_directional_drift_ratio: float,
    variants: tuple[ReplayVariant, ...] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    variant_payloads: list[dict[str, Any]] = []
    all_reasons: list[str] = []
    for variant in variants:
        metrics: ReplayMetrics = run_replay(variant_config(config, variant), variant_params(params, variant))
        metrics_payload = metrics.to_dict()
        ok, reasons = evaluate_metrics(
            metrics_payload,
            min_days=min_days,
            q_max=config.q_max,
            min_quote_attempts=min_quote_attempts,
            min_maker_ratio=min_maker_ratio,
            min_net_realized_spread=min_net_realized_spread,
            min_mean_markout_usdc=min_mean_markout_usdc,
            max_directional_drift_ratio=max_directional_drift_ratio,
        )
        variant_payloads.append(
            {
                "variant": asdict(variant),
                "ok": ok,
                "reasons": reasons,
                "coverage_days": coverage_days(metrics_payload),
                "net_realized_spread_usdc": net_realized_spread_usdc(metrics_payload),
                "directional_pnl_proxy_usdc": directional_pnl_proxy_usdc(metrics_payload),
                "directional_drift_ratio": directional_drift_ratio(metrics_payload),
                "markout_summary": markout_summary(metrics_payload.get("markout_samples") or []),
                "metrics": metrics_payload,
            }
        )
        all_reasons.extend(f"{variant.name}:{reason}" for reason in reasons)

    baseline_metrics = variant_payloads[0]["metrics"] if variant_payloads else {}
    refusal_checks = build_refusal_checks(params=params, baseline_metrics=baseline_metrics)
    for check in refusal_checks:
        if not check["ok"]:
            all_reasons.append(f"refusal:{check['name']}:{check['reason']}")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "symbol": config.symbol,
        "ok": not all_reasons,
        "reasons": all_reasons,
        "acceptance": {
            "min_days": float(min_days),
            "min_quote_attempts": int(min_quote_attempts),
            "min_maker_ratio": float(min_maker_ratio),
            "min_net_realized_spread": float(min_net_realized_spread),
            "min_mean_markout_usdc": float(min_mean_markout_usdc),
            "max_directional_drift_ratio": float(max_directional_drift_ratio),
            "q_max": int(config.q_max),
        },
        "config": {
            "data_dir": str(config.data_dir),
            "newest_per_stream": config.newest_per_stream,
            "max_price_events": config.max_price_events,
            "decision_latency_ms": config.decision_latency_ms,
            "order_ack_latency_ms": config.order_ack_latency_ms,
            "cancel_latency_ms": config.cancel_latency_ms,
            "maker_fee": config.maker_fee,
            "taker_fee": config.taker_fee,
            "funding_rate_per_hour": config.funding_rate_per_hour,
            "starting_equity_usdc": config.starting_equity_usdc,
            "leverage": config.leverage,
            "maintenance_margin_rate": config.maintenance_margin_rate,
            "queue_decay_per_second": config.queue_decay_per_second,
        },
        "base_params": params,
        "refusal_checks": refusal_checks,
        "variants": variant_payloads,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Replay Acceptance Report",
        "",
        f"- status: {'PASS' if report['ok'] else 'FAIL'}",
        f"- symbol: `{report['symbol']}`",
        f"- generated_at: `{report['generated_at']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Net spread | Directional ratio | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["variants"]:
        metrics = item["metrics"]
        reasons = ", ".join(item["reasons"]) if item["reasons"] else "ok"
        lines.append(
            "| {name} | {status} | {days:.6f} | {quotes} | {maker} | {taker} | {spread:.8f} | {drift:.6f} | {reasons} |".format(
                name=item["variant"]["name"],
                status="PASS" if item["ok"] else "FAIL",
                days=float(item["coverage_days"]),
                quotes=int(metrics.get("quote_attempts") or 0),
                maker=int(metrics.get("maker_fills") or 0),
                taker=int(metrics.get("taker_fills") or 0),
                spread=float(item["net_realized_spread_usdc"]),
                drift=float(item.get("directional_drift_ratio") or 0.0),
                reasons=reasons,
            )
        )
    if report.get("refusal_checks"):
        lines.extend(
            [
                "",
                "## Refusal Checks",
                "",
                "| Check | Status | Expected | Decision | Reason |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for check in report["refusal_checks"]:
            lines.append(
                "| {name} | {status} | {expected} | {decision} | {reason} |".format(
                    name=check["name"],
                    status="PASS" if check["ok"] else "FAIL",
                    expected=check.get("expected_decision", ""),
                    decision=check.get("decision", ""),
                    reason=check.get("reason", ""),
                )
            )
    if report["reasons"]:
        lines.extend(["", "## Blocking Reasons", ""])
        lines.extend(f"- `{reason}`" for reason in report["reasons"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-variant replay acceptance report.")
    parser.add_argument("--symbol", default="ETH")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "HL_data")
    parser.add_argument("--mid", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/replay_acceptance_report.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/replay_acceptance_report.md"))
    parser.add_argument("--allow-incomplete", action="store_true", help="Write the report and return 0 even when acceptance fails.")
    parser.add_argument("--min-days", type=float, default=3.0)
    parser.add_argument("--min-quote-attempts", type=int, default=1000)
    parser.add_argument("--min-maker-ratio", type=float, default=0.99)
    parser.add_argument("--min-net-realized-spread", type=float, default=0.0)
    parser.add_argument("--min-mean-markout-usdc", type=float, default=-0.01)
    parser.add_argument("--max-directional-drift-ratio", type=float, default=0.75)
    parser.add_argument("--maker-fee", type=float, default=MAKER_FEE)
    parser.add_argument("--taker-fee", type=float, default=TAKER_FEE)
    parser.add_argument("--funding-rate-per-hour", type=float, default=0.0)
    parser.add_argument("--starting-equity-usdc", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-rate", type=float, default=0.05)
    parser.add_argument("--newest-per-stream", type=int, default=None)
    parser.add_argument("--max-price-events", type=int, default=None)
    parser.add_argument("--kappa-plus", type=float, required=True)
    parser.add_argument("--kappa-minus", type=float, required=True)
    parser.add_argument("--lambda-plus", type=float, required=True)
    parser.add_argument("--lambda-minus", type=float, required=True)
    parser.add_argument("--epsilon-plus", type=float, default=0.0)
    parser.add_argument("--epsilon-minus", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = {
        "kappa+": args.kappa_plus,
        "kappa-": args.kappa_minus,
        "lambda+": args.lambda_plus,
        "lambda-": args.lambda_minus,
        "epsilon+": args.epsilon_plus,
        "epsilon-": args.epsilon_minus,
    }
    config = ReplayConfig(
        symbol=args.symbol,
        data_dir=args.data_dir,
        mid_fallback=args.mid,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        funding_rate_per_hour=args.funding_rate_per_hour,
        starting_equity_usdc=args.starting_equity_usdc,
        leverage=args.leverage,
        maintenance_margin_rate=args.maintenance_margin_rate,
        newest_per_stream=args.newest_per_stream,
        max_price_events=args.max_price_events,
    )
    report = build_report(
        config=config,
        params=params,
        min_days=args.min_days,
        min_quote_attempts=args.min_quote_attempts,
        min_maker_ratio=args.min_maker_ratio,
        min_net_realized_spread=args.min_net_realized_spread,
        min_mean_markout_usdc=args.min_mean_markout_usdc,
        max_directional_drift_ratio=args.max_directional_drift_ratio,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(text)
    return 0 if report["ok"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
