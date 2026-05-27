#!/usr/bin/env python3
"""Evaluate whether an enabled dry-run produced reasonable market-maker evidence.

This report is intentionally stricter than the basic enabled dry-run smoke. The
smoke proves the bot can start and create an order; this checker inspects the
JSONL audit trail to verify quote quality, order size, and loss/risk behavior
over the observed dry-run window.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_LOG = Path("user_data/logs/mm_debug.jsonl")
DEFAULT_GATE_REPORT = Path("docs/dry_run_enabled_gate.json")
DEFAULT_OUTPUT = Path("docs/dry_run_quality_report.json")


ERROR_EVENTS = {
    "kill_switch",
    "param_update_failed",
    "param_update_rejected",
    "collector_data_read_error",
    "collector_data_timestamp_error",
    "hjb_refresh_failed",
    "hjb_refresh_skipped",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def event_time(event: dict[str, Any]) -> datetime | None:
    for key in ("ts", "timestamp", "current_time", "generated_at", "time"):
        dt = parse_time(event.get(key))
        if dt is not None:
            return dt
    return None


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"missing": True, "path": str(path)}
    except Exception as exc:
        return {"invalid": True, "path": str(path), "error": str(exc)}
    return payload if isinstance(payload, dict) else {"invalid": True, "path": str(path)}


def read_jsonl_events(path: Path, *, since: datetime | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    threshold = since.astimezone(timezone.utc) if since is not None else None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        ts = event_time(payload)
        if threshold is not None and (ts is None or ts < threshold):
            continue
        events.append(payload)
    return events


def event_span_seconds(events: list[dict[str, Any]]) -> float:
    times = [dt for dt in (event_time(event) for event in events) if dt is not None]
    if len(times) < 2:
        return 0.0
    return max(0.0, (max(times) - min(times)).total_seconds())


def gate_runtime_seconds(gate_report: dict[str, Any]) -> float:
    waited = gate_report.get("container_wait", {})
    if isinstance(waited, dict):
        waited_seconds = finite_float(waited.get("waited_seconds"))
        if waited_seconds is not None:
            return waited_seconds
    duration = finite_float(gate_report.get("duration_seconds"))
    return float(duration or 0.0)


def health_pnl_stats(health_events: list[dict[str, Any]]) -> dict[str, Any]:
    totals: list[float] = []
    realized: list[float] = []
    unrealized: list[float] = []
    for event in health_events:
        realized_value = finite_float(event.get("realized_pnl"))
        unrealized_value = finite_float(event.get("unrealized_pnl"))
        if realized_value is not None:
            realized.append(realized_value)
        if unrealized_value is not None:
            unrealized.append(unrealized_value)
        if realized_value is not None or unrealized_value is not None:
            totals.append(float(realized_value or 0.0) + float(unrealized_value or 0.0))
    return {
        "samples": len(totals),
        "min_total_pnl_usdc": min(totals) if totals else None,
        "max_total_pnl_usdc": max(totals) if totals else None,
        "final_total_pnl_usdc": totals[-1] if totals else None,
        "min_realized_pnl_usdc": min(realized) if realized else None,
        "final_realized_pnl_usdc": realized[-1] if realized else None,
        "min_unrealized_pnl_usdc": min(unrealized) if unrealized else None,
        "final_unrealized_pnl_usdc": unrealized[-1] if unrealized else None,
    }


def build_dry_run_quality_report(
    events: list[dict[str, Any]],
    *,
    gate_report: dict[str, Any],
    min_runtime_seconds: float = 180.0,
    min_event_span_seconds: float = 60.0,
    min_accepted_quotes: int = 1,
    min_order_attempts: int = 1,
    max_quote_distance_ratio: float = 0.01,
    max_quote_depth_bps: float = 50.0,
    max_order_notional_usdc: float = 25.0,
    max_order_amount_units: float = 1.0,
    max_loss_usdc: float = 1.0,
    max_loss_rate_usdc_per_hour: float = 6.0,
    min_final_total_pnl_usdc: float = -1.0,
    event_span_tolerance_seconds: float = 1.0,
) -> dict[str, Any]:
    reasons: list[str] = []

    runtime_seconds = gate_runtime_seconds(gate_report)
    if gate_report.get("passed") is not True:
        reasons.append("dry_run_enabled_gate_not_passed")
    if runtime_seconds < float(min_runtime_seconds):
        reasons.append(f"runtime_too_short:{runtime_seconds:.1f}<min_{float(min_runtime_seconds):.1f}")

    span_seconds = event_span_seconds(events)
    if span_seconds + float(event_span_tolerance_seconds) < float(min_event_span_seconds):
        reasons.append(f"event_span_too_short:{span_seconds:.1f}<min_{float(min_event_span_seconds):.1f}")

    quote_events = [event for event in events if event.get("event") == "quote_decision"]
    accepted_quotes = [
        event
        for event in quote_events
        if event.get("decision") == "accept"
        and bool_value(event.get("dry_run")) is True
        and bool_value(event.get("trading_enabled")) is True
    ]
    dry_run_fills = [
        event
        for event in events
        if event.get("event") == "fill"
        and bool_value(event.get("dry_run")) is True
        and bool_value(event.get("trading_enabled")) is True
    ]
    fill_observed = bool(dry_run_fills)
    quote_minimum_met = len(accepted_quotes) >= int(min_accepted_quotes) or (
        fill_observed and len(accepted_quotes) >= 1
    )
    if not quote_minimum_met:
        reasons.append(f"insufficient_accepted_quotes:{len(accepted_quotes)}<min_{int(min_accepted_quotes)}")

    quote_failures: dict[str, int] = {}
    quote_depths: list[float] = []
    quote_distances: list[float] = []
    for quote in accepted_quotes:
        for key, reason in (
            ("params_fresh", "accepted_quote_stale_params"),
            ("collector_fresh", "accepted_quote_stale_collector"),
            ("book_fresh", "accepted_quote_stale_book"),
        ):
            if bool_value(quote.get(key)) is not True:
                quote_failures[reason] = quote_failures.get(reason, 0) + 1
        fee_snapshot = quote.get("fee_snapshot") if isinstance(quote.get("fee_snapshot"), dict) else {}
        if bool_value(fee_snapshot.get("fee_agreement_ok")) is not True:
            quote_failures["accepted_quote_missing_fee_agreement"] = (
                quote_failures.get("accepted_quote_missing_fee_agreement", 0) + 1
            )

        depth_bps = finite_float(quote.get("bps"))
        if depth_bps is None:
            quote_failures["accepted_quote_missing_depth_bps"] = quote_failures.get("accepted_quote_missing_depth_bps", 0) + 1
        else:
            quote_depths.append(abs(depth_bps))
            if abs(depth_bps) > float(max_quote_depth_bps):
                quote_failures["accepted_quote_depth_too_wide"] = quote_failures.get("accepted_quote_depth_too_wide", 0) + 1

        distance_ratio = finite_float(quote.get("custom_price_distance_ratio"))
        if distance_ratio is None:
            quote_failures["accepted_quote_missing_distance_ratio"] = (
                quote_failures.get("accepted_quote_missing_distance_ratio", 0) + 1
            )
        else:
            quote_distances.append(abs(distance_ratio))
            if abs(distance_ratio) > float(max_quote_distance_ratio):
                quote_failures["accepted_quote_too_far_from_proposed"] = (
                    quote_failures.get("accepted_quote_too_far_from_proposed", 0) + 1
                )

        rounded = finite_float(quote.get("rounded_price"))
        best_bid = finite_float(quote.get("best_bid"))
        best_ask = finite_float(quote.get("best_ask"))
        side = str(quote.get("side") or "").lower()
        if rounded is None or best_bid is None or best_ask is None:
            quote_failures["accepted_quote_missing_bbo"] = quote_failures.get("accepted_quote_missing_bbo", 0) + 1
        elif side == "bid" and rounded >= best_ask:
            quote_failures["accepted_bid_crosses_ask"] = quote_failures.get("accepted_bid_crosses_ask", 0) + 1
        elif side == "ask" and rounded <= best_bid:
            quote_failures["accepted_ask_crosses_bid"] = quote_failures.get("accepted_ask_crosses_bid", 0) + 1

    reasons.extend(f"{key}:{value}" for key, value in sorted(quote_failures.items()) if value)

    order_attempts_all = [
        event
        for event in events
        if event.get("event") == "order_attempt_accepted"
        and bool_value(event.get("dry_run")) is True
        and bool_value(event.get("trading_enabled")) is True
    ]

    accepted_quote_ids = {str(event.get("quote_id")) for event in accepted_quotes if event.get("quote_id") not in (None, "")}
    order_failures: dict[str, int] = {}
    ignored_order_attempts: dict[str, int] = {}
    order_attempts: list[dict[str, Any]] = []
    for order in order_attempts_all:
        quote_id = order.get("quote_id")
        if quote_id in (None, ""):
            order_failures["order_missing_quote_id"] = order_failures.get("order_missing_quote_id", 0) + 1
            continue
        if str(quote_id) not in accepted_quote_ids:
            ignored_order_attempts["order_quote_id_outside_quality_window"] = (
                ignored_order_attempts.get("order_quote_id_outside_quality_window", 0) + 1
            )
            continue
        order_attempts.append(order)

    order_minimum_met = len(order_attempts) >= int(min_order_attempts) or (
        fill_observed and len(order_attempts) >= 1
    )
    if not order_minimum_met:
        reasons.append(f"insufficient_quote_linked_order_attempts:{len(order_attempts)}<min_{int(min_order_attempts)}")

    notionals: list[float] = []
    amounts: list[float] = []
    for order in order_attempts:
        amount = finite_float(order.get("amount"))
        rate = finite_float(order.get("rate"))
        if amount is None or rate is None:
            order_failures["order_missing_amount_or_rate"] = order_failures.get("order_missing_amount_or_rate", 0) + 1
            continue
        amounts.append(amount)
        notional = amount * rate
        notionals.append(notional)
        if float(max_order_amount_units) > 0 and amount > float(max_order_amount_units):
            order_failures["order_amount_too_large"] = order_failures.get("order_amount_too_large", 0) + 1
        if float(max_order_notional_usdc) > 0 and notional > float(max_order_notional_usdc):
            order_failures["order_notional_too_large"] = order_failures.get("order_notional_too_large", 0) + 1

    reasons.extend(f"{key}:{value}" for key, value in sorted(order_failures.items()) if value)

    health_events = [event for event in events if event.get("event") == "health"]
    if not health_events:
        reasons.append("no_health_events")
    health_bad = {
        "health_trading_not_enabled": sum(1 for event in health_events if bool_value(event.get("trading_enabled")) is not True),
        "health_not_dry_run": sum(1 for event in health_events if bool_value(event.get("dry_run")) is not True),
        "health_params_stale": sum(1 for event in health_events if bool_value(event.get("params_fresh")) is not True),
        "health_collector_stale": sum(1 for event in health_events if bool_value(event.get("collector_fresh")) is not True),
        "health_book_stale": sum(1 for event in health_events if bool_value(event.get("book_fresh")) is not True),
        "health_hjb_stale": sum(1 for event in health_events if bool_value(event.get("hjb_fresh")) is not True),
    }
    reasons.extend(f"{key}:{value}" for key, value in sorted(health_bad.items()) if value)

    pnl = health_pnl_stats(health_events)
    min_total = pnl.get("min_total_pnl_usdc")
    final_total = pnl.get("final_total_pnl_usdc")
    if min_total is None:
        reasons.append("missing_pnl_samples")
    elif float(min_total) < -abs(float(max_loss_usdc)):
        reasons.append(f"loss_too_large:{float(min_total):.6f}<min_{-abs(float(max_loss_usdc)):.6f}")
    if final_total is not None and float(final_total) < float(min_final_total_pnl_usdc):
        reasons.append(
            f"final_pnl_below_threshold:{float(final_total):.6f}<min_{float(min_final_total_pnl_usdc):.6f}"
        )
    loss_velocity_usdc_per_hour: float | None = None
    if min_total is not None and span_seconds > 0:
        loss_velocity_usdc_per_hour = max(0.0, -float(min_total)) * 3600.0 / float(span_seconds)
        if loss_velocity_usdc_per_hour > float(max_loss_rate_usdc_per_hour):
            reasons.append(
                "loss_rate_too_fast:"
                f"{loss_velocity_usdc_per_hour:.6f}>max_{float(max_loss_rate_usdc_per_hour):.6f}_usdc_per_hour"
            )

    error_counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event") or "")
        if name in ERROR_EVENTS:
            error_counts[name] = error_counts.get(name, 0) + 1
    reasons.extend(f"{key}:{value}" for key, value in sorted(error_counts.items()))

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "criteria": {
            "min_runtime_seconds": float(min_runtime_seconds),
            "min_event_span_seconds": float(min_event_span_seconds),
            "min_accepted_quotes": int(min_accepted_quotes),
            "min_order_attempts": int(min_order_attempts),
            "max_quote_distance_ratio": float(max_quote_distance_ratio),
            "max_quote_depth_bps": float(max_quote_depth_bps),
            "max_order_notional_usdc": float(max_order_notional_usdc),
            "max_order_amount_units": float(max_order_amount_units),
            "max_loss_usdc": float(max_loss_usdc),
            "max_loss_rate_usdc_per_hour": float(max_loss_rate_usdc_per_hour),
            "min_final_total_pnl_usdc": float(min_final_total_pnl_usdc),
            "event_span_tolerance_seconds": float(event_span_tolerance_seconds),
        },
        "gate_report": {
            "path": gate_report.get("path"),
            "passed": bool(gate_report.get("passed")),
            "reason": gate_report.get("reason"),
            "runtime_seconds": runtime_seconds,
            "started_at": gate_report.get("started_at"),
        },
        "event_count": len(events),
        "event_span_seconds": span_seconds,
        "health_events": len(health_events),
        "quote_decisions": len(quote_events),
        "accepted_quotes": len(accepted_quotes),
        "accepted_order_attempts": len(order_attempts),
        "total_order_attempts": len(order_attempts_all),
        "dry_run_fills": len(dry_run_fills),
        "quote_quality": {
            "failures": quote_failures,
            "max_depth_bps": max(quote_depths) if quote_depths else None,
            "avg_depth_bps": sum(quote_depths) / len(quote_depths) if quote_depths else None,
            "max_distance_ratio": max(quote_distances) if quote_distances else None,
            "avg_distance_ratio": sum(quote_distances) / len(quote_distances) if quote_distances else None,
        },
        "order_sizing": {
            "failures": order_failures,
            "ignored": ignored_order_attempts,
            "max_amount": max(amounts) if amounts else None,
            "max_notional_usdc": max(notionals) if notionals else None,
            "avg_notional_usdc": sum(notionals) / len(notionals) if notionals else None,
        },
        "pnl": pnl,
        "loss_velocity_usdc_per_hour": loss_velocity_usdc_per_hour,
        "error_events": error_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify enabled dry-run quote/order/PnL quality.")
    parser.add_argument("--input", type=Path, default=DEFAULT_AUDIT_LOG, help="JSONL audit log to inspect.")
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-runtime-seconds", type=float, default=180.0)
    parser.add_argument("--min-event-span-seconds", type=float, default=60.0)
    parser.add_argument("--min-accepted-quotes", type=int, default=1)
    parser.add_argument("--min-order-attempts", type=int, default=1)
    parser.add_argument("--max-quote-distance-ratio", type=float, default=0.01)
    parser.add_argument("--max-quote-depth-bps", type=float, default=50.0)
    parser.add_argument("--max-order-notional-usdc", type=float, default=25.0)
    parser.add_argument("--max-order-amount-units", type=float, default=1.0)
    parser.add_argument("--max-loss-usdc", type=float, default=1.0)
    parser.add_argument("--max-loss-rate-usdc-per-hour", type=float, default=6.0)
    parser.add_argument("--min-final-total-pnl-usdc", type=float, default=-1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate_report = read_json(args.gate_report)
    gate_report["path"] = str(args.gate_report)
    since = parse_time(gate_report.get("started_at"))
    report = build_dry_run_quality_report(
        read_jsonl_events(args.input, since=since),
        gate_report=gate_report,
        min_runtime_seconds=float(args.min_runtime_seconds),
        min_event_span_seconds=float(args.min_event_span_seconds),
        min_accepted_quotes=int(args.min_accepted_quotes),
        min_order_attempts=int(args.min_order_attempts),
        max_quote_distance_ratio=float(args.max_quote_distance_ratio),
        max_quote_depth_bps=float(args.max_quote_depth_bps),
        max_order_notional_usdc=float(args.max_order_notional_usdc),
        max_order_amount_units=float(args.max_order_amount_units),
        max_loss_usdc=float(args.max_loss_usdc),
        max_loss_rate_usdc_per_hour=float(args.max_loss_rate_usdc_per_hour),
        min_final_total_pnl_usdc=float(args.min_final_total_pnl_usdc),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
