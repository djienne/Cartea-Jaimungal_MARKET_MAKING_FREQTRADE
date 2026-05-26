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


def text_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


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


def event_symbol(event: dict[str, Any]) -> str | None:
    for key in ("symbol", "pair"):
        value = event.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def normalized_symbol(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).split("/", 1)[0].split(":", 1)[0].strip().upper() or None


def quote_side(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"bid", "buy", "long", "entry", "open_long", "close_short"}:
        return "bid"
    if text in {"ask", "sell", "short", "exit", "open_short", "close_long"}:
        return "ask"
    return None


def event_price(event: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = finite_float(event.get(key))
        if value is not None and value > 0:
            return value
    return None


def is_live_enabled_event(event: dict[str, Any]) -> bool:
    return (
        bool_value(event.get("trading_enabled")) is True
        and bool_value(event.get("dry_run")) is False
    )


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
    live_accepted_quotes = [event for event in accepted_quotes if is_live_enabled_event(event)]
    live_maker_fills = [
        event
        for event in fills
        if canonical_liquidity(event.get("liquidity")) == "maker"
        and is_live_enabled_event(event)
    ]
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
        "live_accepted_quotes": len(live_accepted_quotes),
        "maker_fills": sum(1 for event in fills if canonical_liquidity(event.get("liquidity")) == "maker"),
        "live_maker_fills": len(live_maker_fills),
        "taker_fills": sum(1 for event in fills if canonical_liquidity(event.get("liquidity")) == "taker"),
        "kill_switches": sum(1 for event in events if event.get("event") == "kill_switch"),
        "live_health_events": sum(
            1
            for event in health
            if bool_value(event.get("trading_enabled")) is True and bool_value(event.get("dry_run")) is False
        ),
    }


def evidence_age_meta(
    timestamp: Any,
    *,
    now: datetime,
    max_age_seconds: float,
) -> dict[str, Any]:
    parsed = parse_time(timestamp)
    age_seconds = (
        max(0.0, (now - parsed).total_seconds())
        if parsed is not None
        else None
    )
    ok = parsed is not None
    reason = "ok" if ok else "missing_timestamp"
    if ok and max_age_seconds > 0 and age_seconds is not None and age_seconds > float(max_age_seconds):
        ok = False
        reason = "stale"
    return {
        "ok": ok,
        "reason": reason,
        "timestamp": timestamp,
        "age_seconds": age_seconds,
        "max_age_seconds": float(max_age_seconds),
    }


def dependency_status(
    report: dict[str, Any],
    *,
    now: datetime,
    max_report_age_seconds: float,
) -> dict[str, Any]:
    report_ok = bool(report.get("ok"))
    freshness = evidence_age_meta(
        report.get("generated_at"),
        now=now,
        max_age_seconds=max_report_age_seconds,
    )
    ok = bool(report_ok and freshness["ok"])
    return {
        "ok": ok,
        "report_ok": report_ok,
        "missing": bool(report.get("missing")),
        "invalid": bool(report.get("invalid")),
        "reasons": report.get("reasons", []),
        "generated_at": report.get("generated_at"),
        "age_seconds": freshness["age_seconds"],
        "max_age_seconds": freshness["max_age_seconds"],
        "freshness_reason": freshness["reason"],
    }


def canary_event_freshness(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    max_event_age_seconds: float,
) -> dict[str, Any]:
    timestamp_missing = 0
    stale = 0
    fresh = 0
    ages: list[float] = []
    for event in events:
        dt = event_time(event)
        if dt is None:
            timestamp_missing += 1
            continue
        age_seconds = max(0.0, (now - dt).total_seconds())
        ages.append(age_seconds)
        if max_event_age_seconds > 0 and age_seconds > float(max_event_age_seconds):
            stale += 1
        else:
            fresh += 1
    return {
        "fresh": fresh,
        "stale": stale,
        "timestamp_missing": timestamp_missing,
        "max_age_seconds": max(ages) if ages else None,
        "max_allowed_age_seconds": float(max_event_age_seconds),
    }


def accepted_quote_failures(events: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "accepted_quote_stale_params": 0,
        "accepted_quote_stale_collector": 0,
        "accepted_quote_stale_orderbook": 0,
        "accepted_quote_not_post_only": 0,
        "accepted_quote_post_only_unverified": 0,
        "accepted_quote_missing_fee_agreement": 0,
        "accepted_quote_not_live_enabled": 0,
    }
    for event in events:
        if event.get("event") != "quote_decision" or event.get("decision") != "accept":
            continue
        if not is_live_enabled_event(event):
            counters["accepted_quote_not_live_enabled"] += 1
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


def accepted_order_attempt_failures(events: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "accepted_order_attempt_not_live_enabled": 0,
        "accepted_order_attempt_not_post_only": 0,
        "accepted_order_attempt_post_only_unverified": 0,
        "accepted_order_attempt_missing_quote_id": 0,
        "accepted_order_attempt_missing_match_fields": 0,
    }
    for event in events:
        if event.get("event") != "order_attempt_accepted":
            continue
        if not is_live_enabled_event(event):
            counters["accepted_order_attempt_not_live_enabled"] += 1
        if bool_value(event.get("post_only")) is not True:
            counters["accepted_order_attempt_not_post_only"] += 1
        if bool_value(event.get("post_only_verified")) is not True:
            counters["accepted_order_attempt_post_only_unverified"] += 1
        if text_or_none(event.get("quote_id")) is None:
            counters["accepted_order_attempt_missing_quote_id"] += 1
        ts = event_time(event)
        side = quote_side(event.get("quote_side") or event.get("side"))
        price = event_price(event, ("rate", "price", "rounded_price"))
        if ts is None or side is None or price is None:
            counters["accepted_order_attempt_missing_match_fields"] += 1
    return {key: value for key, value in counters.items() if value}


def live_maker_fill_reconciliation(
    events: list[dict[str, Any]],
    *,
    max_quote_to_fill_seconds: float,
    price_tolerance: float,
) -> dict[str, Any]:
    counters = {
        "live_maker_fills_missing_order_id": 0,
        "live_maker_fills_missing_quote_side": 0,
        "live_maker_fills_missing_price": 0,
        "live_maker_fills_missing_amount": 0,
        "live_maker_fills_missing_quote_id": 0,
        "live_maker_fills_without_matching_quote": 0,
        "live_maker_fills_without_matching_order_attempt": 0,
    }
    accepted_quotes: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "quote_decision" or event.get("decision") != "accept":
            continue
        if not is_live_enabled_event(event):
            continue
        ts = event_time(event)
        side = quote_side(event.get("quote_side") or event.get("side"))
        price = event_price(event, ("rounded_price", "rate", "proposed_rate", "raw_price"))
        if ts is None or side is None or price is None:
            continue
        accepted_quotes.append(
            {
                "ts": ts,
                "session": event_session_key(event),
                "symbol": normalized_symbol(event_symbol(event)),
                "side": side,
                "price": price,
                "quote_id": text_or_none(event.get("quote_id")),
            }
        )

    accepted_order_attempts: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "order_attempt_accepted":
            continue
        if not is_live_enabled_event(event):
            continue
        if bool_value(event.get("post_only")) is not True:
            continue
        if bool_value(event.get("post_only_verified")) is not True:
            continue
        quote_id = text_or_none(event.get("quote_id"))
        if quote_id is None:
            continue
        ts = event_time(event)
        side = quote_side(event.get("quote_side") or event.get("side"))
        price = event_price(event, ("rate", "price", "rounded_price"))
        if ts is None or side is None or price is None:
            continue
        accepted_order_attempts.append(
            {
                "ts": ts,
                "session": event_session_key(event),
                "symbol": normalized_symbol(event_symbol(event)),
                "side": side,
                "price": price,
                "quote_id": quote_id,
            }
        )

    fills = [
        event
        for event in events
        if event.get("event") == "fill"
        and canonical_liquidity(event.get("liquidity")) == "maker"
        and is_live_enabled_event(event)
    ]
    unmatched: list[dict[str, Any]] = []
    unmatched_order_attempts: list[dict[str, Any]] = []
    matched = 0
    matched_order_attempts = 0
    fills_with_quote_id = 0

    def match_prior(
        candidates: list[dict[str, Any]],
        *,
        fill_ts: datetime,
        fill_session: str | None,
        fill_symbol: str | None,
        fill_side: str,
        fill_price: float,
        fill_quote_id: str | None,
    ) -> dict[str, Any] | None:
        filtered = (
            [item for item in candidates if item.get("quote_id") == fill_quote_id]
            if fill_quote_id is not None
            else candidates
        )
        for candidate in filtered:
            if fill_session is not None and candidate["session"] is not None and fill_session != candidate["session"]:
                continue
            if fill_symbol is not None and candidate["symbol"] is not None and fill_symbol != candidate["symbol"]:
                continue
            if candidate["side"] != fill_side:
                continue
            age_seconds = (fill_ts - candidate["ts"]).total_seconds()
            if age_seconds < 0 or age_seconds > float(max_quote_to_fill_seconds):
                continue
            if abs(float(fill_price) - float(candidate["price"])) > float(price_tolerance):
                continue
            return candidate
        return None

    for fill in fills:
        fill_ts = event_time(fill)
        fill_side = quote_side(fill.get("quote_side") or fill.get("side"))
        fill_price = event_price(fill, ("price", "fill_price", "rate"))
        fill_amount = event_price(fill, ("amount", "filled", "fill_size", "size"))
        fill_order_id = fill.get("order_id")
        fill_quote_id = text_or_none(fill.get("quote_id"))
        if fill_quote_id is not None:
            fills_with_quote_id += 1
        else:
            counters["live_maker_fills_missing_quote_id"] += 1
        if fill_order_id in (None, ""):
            counters["live_maker_fills_missing_order_id"] += 1
        if fill_side is None:
            counters["live_maker_fills_missing_quote_side"] += 1
        if fill_price is None:
            counters["live_maker_fills_missing_price"] += 1
        if fill_amount is None:
            counters["live_maker_fills_missing_amount"] += 1
        if fill_ts is None or fill_side is None or fill_price is None:
            counters["live_maker_fills_without_matching_quote"] += 1
            counters["live_maker_fills_without_matching_order_attempt"] += 1
            unmatched.append({"order_id": fill_order_id, "reason": "missing_required_fill_fields"})
            unmatched_order_attempts.append({"order_id": fill_order_id, "reason": "missing_required_fill_fields"})
            continue

        fill_session = event_session_key(fill)
        fill_symbol = normalized_symbol(event_symbol(fill))
        matched_quote = match_prior(
            accepted_quotes,
            fill_ts=fill_ts,
            fill_session=fill_session,
            fill_symbol=fill_symbol,
            fill_side=fill_side,
            fill_price=fill_price,
            fill_quote_id=fill_quote_id,
        )
        matched_attempt = match_prior(
            accepted_order_attempts,
            fill_ts=fill_ts,
            fill_session=fill_session,
            fill_symbol=fill_symbol,
            fill_side=fill_side,
            fill_price=fill_price,
            fill_quote_id=fill_quote_id,
        )
        if matched_quote is None:
            counters["live_maker_fills_without_matching_quote"] += 1
            unmatched.append(
                {
                    "order_id": fill_order_id,
                    "session_id": fill_session,
                    "symbol": fill_symbol,
                    "quote_side": fill_side,
                    "price": fill_price,
                    "quote_id": fill_quote_id,
                }
            )
        else:
            matched += 1
        if matched_attempt is None:
            counters["live_maker_fills_without_matching_order_attempt"] += 1
            unmatched_order_attempts.append(
                {
                    "order_id": fill_order_id,
                    "session_id": fill_session,
                    "symbol": fill_symbol,
                    "quote_side": fill_side,
                    "price": fill_price,
                    "quote_id": fill_quote_id,
                }
            )
        else:
            matched_order_attempts += 1

    return {
        "live_maker_fills": len(fills),
        "accepted_live_quotes_with_match_fields": len(accepted_quotes),
        "accepted_live_quotes_with_quote_id": sum(1 for quote in accepted_quotes if quote.get("quote_id")),
        "accepted_live_order_attempts_with_match_fields": len(accepted_order_attempts),
        "accepted_live_order_attempts_with_quote_id": sum(
            1 for attempt in accepted_order_attempts if attempt.get("quote_id")
        ),
        "live_maker_fills_with_quote_id": int(fills_with_quote_id),
        "matched_live_maker_fills": matched,
        "matched_live_maker_fills_to_order_attempt": matched_order_attempts,
        "failures": {key: value for key, value in counters.items() if value},
        "unmatched": unmatched[:50],
        "unmatched_order_attempts": unmatched_order_attempts[:50],
        "max_quote_to_fill_seconds": float(max_quote_to_fill_seconds),
        "price_tolerance": float(price_tolerance),
    }


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
    symbols = {symbol for symbol in (normalized_symbol(event_symbol(event)) for event in events) if symbol}
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
    if live_health and not symbols:
        failures["missing_symbol"] = True
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
    max_dependency_report_age_seconds: float = 86_400.0,
    max_canary_event_age_seconds: float = 604_800.0,
    max_quote_to_fill_seconds: float = 900.0,
    fill_match_price_tolerance: float = 1e-8,
    now: datetime | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    reference_time = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    dependencies = {
        "post_only": dependency_status(
            post_only_report,
            now=reference_time,
            max_report_age_seconds=float(max_dependency_report_age_seconds),
        ),
        "fee": dependency_status(
            fee_report,
            now=reference_time,
            max_report_age_seconds=float(max_dependency_report_age_seconds),
        ),
        "replay": dependency_status(
            replay_report,
            now=reference_time,
            max_report_age_seconds=float(max_dependency_report_age_seconds),
        ),
    }
    if not dependencies["post_only"]["ok"]:
        reasons.append("post_only_gate_not_passed")
    if not dependencies["fee"]["ok"]:
        reasons.append("fee_gate_not_passed")
    if not dependencies["replay"]["ok"]:
        reasons.append("replay_gate_not_passed")
    for name, status in dependencies.items():
        if bool(status.get("report_ok")) and status.get("freshness_reason") == "missing_timestamp":
            reasons.append(f"{name}_report_missing_generated_at")
        elif bool(status.get("report_ok")) and status.get("freshness_reason") == "stale":
            reasons.append(f"{name}_report_stale")
    if not bool(manual_monitoring_ack):
        reasons.append("manual_monitoring_not_acknowledged")

    if not events:
        reasons.append("no_canary_events")
    event_freshness = canary_event_freshness(
        events,
        now=reference_time,
        max_event_age_seconds=float(max_canary_event_age_seconds),
    )
    if event_freshness["timestamp_missing"]:
        reasons.append(f"canary_event_timestamp_missing:{event_freshness['timestamp_missing']}")
    if event_freshness["stale"]:
        reasons.append(f"canary_event_stale:{event_freshness['stale']}")

    sessions = group_sessions(events, session_gap_minutes=session_gap_minutes)
    eligible_sessions = [
        session
        for session in sessions
        if session["duration_minutes"] >= float(min_session_minutes)
        and session["live_health_events"] > 0
        and (session["live_accepted_quotes"] > 0 or session["live_maker_fills"] > 0)
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
    non_live_fills = [event for event in fill_events if not is_live_enabled_event(event)]
    if non_live_fills:
        reasons.append(f"fill_not_live_enabled:{len(non_live_fills)}")

    kill_switches = [event for event in events if event.get("event") == "kill_switch"]
    if kill_switches:
        reasons.append(f"kill_switches_seen:{len(kill_switches)}")

    quote_failures = accepted_quote_failures(events)
    reasons.extend(f"{key}:{value}" for key, value in sorted(quote_failures.items()))
    order_attempt_failures = accepted_order_attempt_failures(events)
    reasons.extend(f"{key}:{value}" for key, value in sorted(order_attempt_failures.items()))

    fill_reconciliation = live_maker_fill_reconciliation(
        events,
        max_quote_to_fill_seconds=float(max_quote_to_fill_seconds),
        price_tolerance=float(fill_match_price_tolerance),
    )
    reasons.extend(f"{key}:{value}" for key, value in sorted(fill_reconciliation["failures"].items()))

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
            "max_dependency_report_age_seconds": float(max_dependency_report_age_seconds),
            "max_canary_event_age_seconds": float(max_canary_event_age_seconds),
            "max_quote_to_fill_seconds": float(max_quote_to_fill_seconds),
            "fill_match_price_tolerance": float(fill_match_price_tolerance),
        },
        "dependencies": dependencies,
        "event_count": len(events),
        "event_freshness": event_freshness,
        "sessions": sessions,
        "eligible_sessions": len(eligible_sessions),
        "fills": {
            "count": len(fill_events),
            "maker": sum(1 for event in fill_events if canonical_liquidity(event.get("liquidity")) == "maker"),
            "taker": len(taker_fills),
            "unknown_liquidity": len(unknown_fill_liquidity),
        },
        "quote_failures": quote_failures,
        "order_attempt_failures": order_attempt_failures,
        "fill_reconciliation": fill_reconciliation,
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
    parser.add_argument("--max-dependency-report-age-seconds", type=float, default=86_400.0)
    parser.add_argument("--max-canary-event-age-seconds", type=float, default=604_800.0)
    parser.add_argument("--max-quote-to-fill-seconds", type=float, default=900.0)
    parser.add_argument("--fill-match-price-tolerance", type=float, default=1e-8)
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
        max_dependency_report_age_seconds=float(args.max_dependency_report_age_seconds),
        max_canary_event_age_seconds=float(args.max_canary_event_age_seconds),
        max_quote_to_fill_seconds=float(args.max_quote_to_fill_seconds),
        fill_match_price_tolerance=float(args.fill_match_price_tolerance),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
