#!/usr/bin/env python3
"""Evaluate live canary evidence before considering production use.

PLAN.md requires a final tiny live canary only after post-only, fee-tier, and
multi-day replay gates have passed. This checker turns JSONL audit logs and the
prior gate artifacts into an explicit pass/fail report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("user_data/logs/mm_debug.jsonl")
DEFAULT_OUTPUT = Path("docs/live_canary_report.json")
DEFAULT_POST_ONLY_REPORT = Path("docs/post_only_evidence_report.json")
DEFAULT_FEE_REPORT = Path("docs/fee_evidence_report.json")
DEFAULT_REPLAY_REPORT = Path("docs/replay_acceptance_report.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "missing": True, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "invalid": True, "path": str(path), "error": str(exc)}
    if not isinstance(payload, dict):
        return {"ok": False, "invalid": True, "path": str(path)}
    return payload


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


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


def canonical_liquidity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"maker", "m", "add", "a", "add_liquidity", "added_liquidity", "post_only", "true"}:
        return "maker"
    if text in {"taker", "t", "remove", "r", "remove_liquidity", "removed_liquidity", "false"}:
        return "taker"
    return text or None


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


def event_session_key(event: dict[str, Any]) -> str | None:
    for key in ("session_id", "run_id", "canary_session"):
        value = event.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def group_sessions(events: Iterable[dict[str, Any]], *, session_gap_minutes: float) -> list[dict[str, Any]]:
    timed = [(event_time(event), event) for event in events]
    timed = [(dt, event) for dt, event in timed if dt is not None]
    if not timed:
        return []

    explicit: dict[str, list[tuple[datetime, dict[str, Any]]]] = {}
    implicit: list[tuple[datetime, dict[str, Any]]] = []
    for dt, event in timed:
        key = event_session_key(event)
        if key is None:
            implicit.append((dt, event))
        else:
            explicit.setdefault(key, []).append((dt, event))

    sessions: list[dict[str, Any]] = []
    for key, items in explicit.items():
        sessions.append(summarize_session(key, items))

    implicit.sort(key=lambda item: item[0])
    current: list[tuple[datetime, dict[str, Any]]] = []
    gap_seconds = max(0.0, float(session_gap_minutes)) * 60.0
    previous: datetime | None = None
    for dt, event in implicit:
        if previous is not None and (dt - previous).total_seconds() > gap_seconds and current:
            sessions.append(summarize_session(f"implicit-{len(sessions) + 1}", current))
            current = []
        current.append((dt, event))
        previous = dt
    if current:
        sessions.append(summarize_session(f"implicit-{len(sessions) + 1}", current))

    sessions.sort(key=lambda item: item["start_ts"])
    return sessions


def summarize_session(key: str, items: list[tuple[datetime, dict[str, Any]]]) -> dict[str, Any]:
    items = sorted(items, key=lambda item: item[0])
    start = items[0][0]
    end = items[-1][0]
    events = [event for _, event in items]
    fills = [event for event in events if event.get("event") == "fill"]
    quote_decisions = [event for event in events if event.get("event") == "quote_decision"]
    accepted_quotes = [event for event in quote_decisions if event.get("decision") == "accept"]
    health = [event for event in events if event.get("event") == "health"]
    return {
        "session_id": key,
        "start_ts": start.isoformat().replace("+00:00", "Z"),
        "end_ts": end.isoformat().replace("+00:00", "Z"),
        "duration_minutes": (end - start).total_seconds() / 60.0,
        "events": len(events),
        "health_events": len(health),
        "quote_decisions": len(quote_decisions),
        "accepted_quotes": len(accepted_quotes),
        "maker_fills": sum(1 for event in fills if canonical_liquidity(event.get("liquidity")) == "maker"),
        "taker_fills": sum(1 for event in fills if canonical_liquidity(event.get("liquidity")) == "taker"),
        "kill_switches": sum(1 for event in events if event.get("event") == "kill_switch"),
        "live_health_events": sum(
            1
            for event in health
            if bool_value(event.get("trading_enabled")) is True and bool_value(event.get("dry_run")) is False
        ),
    }


def dependency_status(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "missing": bool(report.get("missing")),
        "invalid": bool(report.get("invalid")),
        "reasons": report.get("reasons", []),
    }


def accepted_quote_failures(events: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "accepted_quote_stale_params": 0,
        "accepted_quote_stale_collector": 0,
        "accepted_quote_stale_orderbook": 0,
        "accepted_quote_not_post_only": 0,
        "accepted_quote_post_only_unverified": 0,
        "accepted_quote_missing_fee_agreement": 0,
    }
    for event in events:
        if event.get("event") != "quote_decision" or event.get("decision") != "accept":
            continue
        if bool_value(event.get("params_fresh")) is not True:
            counters["accepted_quote_stale_params"] += 1
        if bool_value(event.get("collector_fresh")) is not True:
            counters["accepted_quote_stale_collector"] += 1
        if bool_value(event.get("book_fresh")) is not True:
            counters["accepted_quote_stale_orderbook"] += 1
        if bool_value(event.get("post_only")) is not True:
            counters["accepted_quote_not_post_only"] += 1
        if bool_value(event.get("post_only_verified")) is not True:
            counters["accepted_quote_post_only_unverified"] += 1
        fee_snapshot = event.get("fee_snapshot") if isinstance(event.get("fee_snapshot"), dict) else {}
        if bool_value(fee_snapshot.get("fee_agreement_ok")) is not True:
            counters["accepted_quote_missing_fee_agreement"] += 1
    return {key: value for key, value in counters.items() if value}


def error_event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    error_names = {
        "param_update_failed",
        "param_update_rejected",
        "collector_data_read_error",
        "collector_data_timestamp_error",
        "hjb_refresh_failed",
        "hjb_refresh_skipped",
    }
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event") or "")
        if name in error_names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def health_failures(
    events: list[dict[str, Any]],
    *,
    max_stake_amount: float,
    max_symbols: int,
    max_daily_loss_usdc: float,
) -> dict[str, Any]:
    health = [event for event in events if event.get("event") == "health"]
    live_health = [
        event
        for event in health
        if bool_value(event.get("trading_enabled")) is True and bool_value(event.get("dry_run")) is False
    ]
    symbols = {
        str(event.get("symbol") or event.get("pair") or "")
        for event in live_health
        if event.get("symbol") or event.get("pair")
    }
    stakes = [finite_float(event.get("stake_amount")) for event in live_health]
    stakes = [stake for stake in stakes if stake is not None]
    daily_loss_limits = [finite_float(event.get("max_daily_loss_usdc")) for event in live_health]
    daily_loss_limits = [limit for limit in daily_loss_limits if limit is not None]

    failures: dict[str, Any] = {}
    if not live_health:
        failures["no_live_health_events"] = True
    if any(bool_value(event.get("post_only_verified")) is not True for event in live_health):
        failures["live_health_post_only_unverified"] = True
    if any(bool_value(event.get("kill_on_taker_fill")) is not True for event in live_health):
        failures["live_health_kill_on_taker_fill_disabled"] = True
    if any(stake > float(max_stake_amount) for stake in stakes):
        failures["stake_amount_above_canary_limit"] = max(stakes)
    if live_health and not stakes:
        failures["missing_stake_amount"] = True
    if len(symbols) > int(max_symbols):
        failures["too_many_symbols"] = sorted(symbols)
    if any(abs(limit) > abs(float(max_daily_loss_usdc)) for limit in daily_loss_limits):
        failures["daily_loss_limit_above_canary_limit"] = max(daily_loss_limits)
    if live_health and not daily_loss_limits:
        failures["missing_daily_loss_limit"] = True
    return failures


def build_live_canary_report(
    events: list[dict[str, Any]],
    *,
    post_only_report: dict[str, Any],
    fee_report: dict[str, Any],
    replay_report: dict[str, Any],
    min_sessions: int = 3,
    min_session_minutes: float = 30.0,
    session_gap_minutes: float = 60.0,
    max_stake_amount: float = 25.0,
    max_symbols: int = 1,
    max_daily_loss_usdc: float = 20.0,
    manual_monitoring_ack: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    dependencies = {
        "post_only": dependency_status(post_only_report),
        "fee": dependency_status(fee_report),
        "replay": dependency_status(replay_report),
    }
    if not dependencies["post_only"]["ok"]:
        reasons.append("post_only_gate_not_passed")
    if not dependencies["fee"]["ok"]:
        reasons.append("fee_gate_not_passed")
    if not dependencies["replay"]["ok"]:
        reasons.append("replay_gate_not_passed")
    if not bool(manual_monitoring_ack):
        reasons.append("manual_monitoring_not_acknowledged")

    if not events:
        reasons.append("no_canary_events")

    sessions = group_sessions(events, session_gap_minutes=session_gap_minutes)
    eligible_sessions = [
        session
        for session in sessions
        if session["duration_minutes"] >= float(min_session_minutes)
        and session["live_health_events"] > 0
        and (session["accepted_quotes"] > 0 or session["maker_fills"] > 0)
    ]
    if len(eligible_sessions) < int(min_sessions):
        reasons.append(f"insufficient_canary_sessions:{len(eligible_sessions)}<min_{int(min_sessions)}")

    fill_events = [event for event in events if event.get("event") == "fill"]
    taker_fills = [event for event in fill_events if canonical_liquidity(event.get("liquidity")) == "taker"]
    unknown_fill_liquidity = [
        event for event in fill_events if canonical_liquidity(event.get("liquidity")) not in {"maker", "taker"}
    ]
    if taker_fills:
        reasons.append(f"taker_fills_seen:{len(taker_fills)}")
    if unknown_fill_liquidity:
        reasons.append(f"unknown_fill_liquidity:{len(unknown_fill_liquidity)}")

    kill_switches = [event for event in events if event.get("event") == "kill_switch"]
    if kill_switches:
        reasons.append(f"kill_switches_seen:{len(kill_switches)}")

    quote_failures = accepted_quote_failures(events)
    reasons.extend(f"{key}:{value}" for key, value in sorted(quote_failures.items()))

    error_counts = error_event_counts(events)
    reasons.extend(f"{key}:{value}" for key, value in sorted(error_counts.items()))

    health_issues = health_failures(
        events,
        max_stake_amount=max_stake_amount,
        max_symbols=max_symbols,
        max_daily_loss_usdc=max_daily_loss_usdc,
    )
    reasons.extend(sorted(health_issues))

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "criteria": {
            "min_sessions": int(min_sessions),
            "min_session_minutes": float(min_session_minutes),
            "session_gap_minutes": float(session_gap_minutes),
            "max_stake_amount": float(max_stake_amount),
            "max_symbols": int(max_symbols),
            "max_daily_loss_usdc": float(max_daily_loss_usdc),
            "manual_monitoring_ack": bool(manual_monitoring_ack),
        },
        "dependencies": dependencies,
        "event_count": len(events),
        "sessions": sessions,
        "eligible_sessions": len(eligible_sessions),
        "fills": {
            "count": len(fill_events),
            "maker": sum(1 for event in fill_events if canonical_liquidity(event.get("liquidity")) == "maker"),
            "taker": len(taker_fills),
            "unknown_liquidity": len(unknown_fill_liquidity),
        },
        "quote_failures": quote_failures,
        "health_failures": health_issues,
        "error_events": error_counts,
        "kill_switches": [
            {
                "ts": event_time(event).isoformat().replace("+00:00", "Z") if event_time(event) else None,
                "reason": event.get("reason"),
            }
            for event in kill_switches[:50]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate live canary evidence.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL audit log to inspect.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--post-only-report", type=Path, default=DEFAULT_POST_ONLY_REPORT)
    parser.add_argument("--fee-report", type=Path, default=DEFAULT_FEE_REPORT)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY_REPORT)
    parser.add_argument("--min-sessions", type=int, default=3)
    parser.add_argument("--min-session-minutes", type=float, default=30.0)
    parser.add_argument("--session-gap-minutes", type=float, default=60.0)
    parser.add_argument("--max-stake-amount", type=float, default=25.0)
    parser.add_argument("--max-symbols", type=int, default=1)
    parser.add_argument("--max-daily-loss-usdc", type=float, default=20.0)
    parser.add_argument(
        "--manual-monitoring-ack",
        action="store_true",
        help="Acknowledge that the canary sessions were manually monitored.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_live_canary_report(
        read_jsonl_events(args.input),
        post_only_report=load_report(args.post_only_report),
        fee_report=load_report(args.fee_report),
        replay_report=load_report(args.replay_report),
        min_sessions=int(args.min_sessions),
        min_session_minutes=float(args.min_session_minutes),
        session_gap_minutes=float(args.session_gap_minutes),
        max_stake_amount=float(args.max_stake_amount),
        max_symbols=int(args.max_symbols),
        max_daily_loss_usdc=float(args.max_daily_loss_usdc),
        manual_monitoring_ack=bool(args.manual_monitoring_ack),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
