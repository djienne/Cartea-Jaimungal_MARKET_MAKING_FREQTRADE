#!/usr/bin/env python3
"""Evaluate fee-tier evidence before any live canary.

The strategy can prove local config/strategy agreement by itself, but PLAN.md
requires stronger evidence before real capital: the configured maker fee, the
exchange/account fee snapshot, and actual maker fill fee rates must agree.
This checker turns JSONL audit logs into a small explicit artifact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("user_data/logs/mm_debug.jsonl")
DEFAULT_OUTPUT = Path("docs/fee_evidence_report.json")
VALID_QUOTE_SIDES = {"bid", "ask"}
POST_ONLY_TIFS = {"alo", "po", "post_only", "postonly"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def parse_utc_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
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


def event_timestamp(event: dict[str, Any]) -> datetime | None:
    for key in ("ts", "timestamp", "generated_at", "time"):
        parsed = parse_utc_timestamp(event.get(key))
        if parsed is not None:
            return parsed
    return None


def evidence_age_status(
    event: dict[str, Any],
    *,
    now: datetime,
    max_evidence_age_seconds: float,
) -> tuple[bool, str, float | None]:
    timestamp = event_timestamp(event)
    if timestamp is None:
        return False, "missing_timestamp", None
    age_seconds = max(0.0, (now - timestamp).total_seconds())
    if max_evidence_age_seconds > 0 and age_seconds > float(max_evidence_age_seconds):
        return False, "stale", age_seconds
    return True, "ok", age_seconds


def fee_rate_matches(expected: float | None, observed: float | None, tolerance: float) -> bool | None:
    if expected is None or observed is None:
        return None
    return abs(float(expected) - float(observed)) <= float(tolerance)


def first_finite_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = finite_float(payload.get(key))
        if value is not None:
            return value
    return None


def normalized_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_post_only_tif(value: Any) -> bool:
    return normalized_text(value) in POST_ONLY_TIFS


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


def iter_fee_snapshot_events(events: Iterable[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for event in events:
        snapshot = event.get("fee_snapshot")
        if isinstance(snapshot, dict):
            yield event, snapshot


def build_fee_evidence_report(
    events: list[dict[str, Any]],
    *,
    expected_maker_fee_rate: float = 0.00015,
    expected_taker_fee_rate: float | None = 0.00045,
    tolerance: float = 1e-9,
    min_maker_fills: int = 1,
    max_evidence_age_seconds: float = 86_400.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    mismatches: list[dict[str, Any]] = []
    reference_time = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)

    snapshot_events = list(iter_fee_snapshot_events(events))
    strategy_matches = 0
    config_matches = 0
    exchange_matches = 0
    exchange_unavailable = 0
    exchange_sources: dict[str, int] = {}
    snapshot_timestamp_missing = 0
    snapshot_stale = 0
    snapshot_fresh = 0
    snapshot_age_seconds: list[float] = []

    for event, snapshot in snapshot_events:
        age_ok, age_reason, age_seconds = evidence_age_status(
            event,
            now=reference_time,
            max_evidence_age_seconds=float(max_evidence_age_seconds),
        )
        if age_seconds is not None:
            snapshot_age_seconds.append(float(age_seconds))
        if age_ok:
            snapshot_fresh += 1
        elif age_reason == "missing_timestamp":
            snapshot_timestamp_missing += 1
        elif age_reason == "stale":
            snapshot_stale += 1

        strategy_fee = finite_float(snapshot.get("strategy_maker_fee_rate"))
        config_fee = finite_float(snapshot.get("config_fee_rate"))
        exchange_maker_fee = finite_float(snapshot.get("exchange_maker_fee_rate"))
        exchange_taker_fee = finite_float(snapshot.get("exchange_taker_fee_rate"))
        source = str(snapshot.get("exchange_fee_source") or "unknown")
        exchange_sources[source] = exchange_sources.get(source, 0) + 1

        if age_ok:
            if fee_rate_matches(expected_maker_fee_rate, strategy_fee, tolerance) is True:
                strategy_matches += 1
            elif strategy_fee is not None:
                mismatches.append({"field": "strategy_maker_fee_rate", "observed": strategy_fee})

        if age_ok:
            if snapshot.get("config_fee_matches_strategy") is True and fee_rate_matches(
                expected_maker_fee_rate,
                config_fee,
                tolerance,
            ) is True:
                config_matches += 1
            elif config_fee is not None and snapshot.get("config_fee_matches_strategy") is False:
                mismatches.append({"field": "config_fee_rate", "observed": config_fee})

        if exchange_maker_fee is None:
            exchange_unavailable += 1
        elif age_ok:
            if snapshot.get("exchange_maker_fee_matches_strategy") is True and fee_rate_matches(
                expected_maker_fee_rate,
                exchange_maker_fee,
                tolerance,
            ) is True:
                if expected_taker_fee_rate is None or fee_rate_matches(expected_taker_fee_rate, exchange_taker_fee, tolerance) is not False:
                    exchange_matches += 1
            else:
                mismatches.append(
                    {
                        "field": "exchange_maker_fee_rate",
                        "observed": exchange_maker_fee,
                        "source": source,
                    }
                )

    fill_events = [event for event in events if event.get("event") == "fill"]
    maker_fills = [event for event in fill_events if str(event.get("liquidity") or "").lower() == "maker"]
    taker_fills = [event for event in fill_events if str(event.get("liquidity") or "").lower() == "taker"]
    maker_actual_fee_observed = 0
    maker_actual_fee_matches = 0
    maker_actual_fee_mismatches = 0
    maker_actual_fee_timestamp_missing = 0
    maker_actual_fee_stale = 0
    maker_actual_fee_fresh = 0
    maker_actual_fee_age_seconds: list[float] = []
    maker_fill_quote_side_invalid = 0
    maker_fill_order_type_invalid = 0
    maker_fill_tif_invalid = 0
    maker_fill_expected_fee_invalid = 0
    maker_fill_actual_fee_paid_missing = 0
    maker_fill_actual_fee_paid_invalid = 0
    maker_fill_price_invalid = 0
    maker_fill_amount_invalid = 0
    maker_fill_actual_fee_paid_mismatch = 0

    for fill in maker_fills:
        quote_side = normalized_text(fill.get("quote_side"))
        if quote_side not in VALID_QUOTE_SIDES:
            maker_fill_quote_side_invalid += 1

        if normalized_text(fill.get("order_type")) != "limit":
            maker_fill_order_type_invalid += 1

        if not is_post_only_tif(fill.get("tif") or fill.get("time_in_force")):
            maker_fill_tif_invalid += 1

        expected_fee_rate = finite_float(fill.get("expected_fee_rate"))
        if fee_rate_matches(expected_maker_fee_rate, expected_fee_rate, tolerance) is not True:
            maker_fill_expected_fee_invalid += 1

        actual_fee_paid = finite_float(fill.get("actual_fee_paid"))
        if actual_fee_paid is None:
            maker_fill_actual_fee_paid_missing += 1
        elif actual_fee_paid < 0:
            maker_fill_actual_fee_paid_invalid += 1

        fill_price = first_finite_float(fill, ("price", "fill_price", "rate"))
        fill_amount = first_finite_float(fill, ("amount", "filled", "fill_size", "size"))
        if fill_price is None or fill_price <= 0:
            maker_fill_price_invalid += 1
        if fill_amount is None or fill_amount <= 0:
            maker_fill_amount_invalid += 1

        actual_fee_rate = finite_float(fill.get("actual_fee_rate"))
        if actual_fee_rate is None:
            continue

        if actual_fee_paid is not None and fill_price is not None and fill_amount is not None:
            expected_paid = abs(float(fill_price) * float(fill_amount) * float(actual_fee_rate))
            paid_tolerance = max(float(tolerance), expected_paid * 1e-6)
            if abs(float(actual_fee_paid) - expected_paid) > paid_tolerance:
                maker_fill_actual_fee_paid_mismatch += 1
                mismatches.append(
                    {
                        "field": "actual_maker_fill_fee_paid",
                        "observed": actual_fee_paid,
                        "expected": expected_paid,
                        "order_id": fill.get("order_id"),
                    }
                )

        maker_actual_fee_observed += 1
        age_ok, age_reason, age_seconds = evidence_age_status(
            fill,
            now=reference_time,
            max_evidence_age_seconds=float(max_evidence_age_seconds),
        )
        if age_seconds is not None:
            maker_actual_fee_age_seconds.append(float(age_seconds))
        if age_ok:
            maker_actual_fee_fresh += 1
        elif age_reason == "missing_timestamp":
            maker_actual_fee_timestamp_missing += 1
        elif age_reason == "stale":
            maker_actual_fee_stale += 1

        if age_ok:
            if fee_rate_matches(expected_maker_fee_rate, actual_fee_rate, tolerance) is True:
                maker_actual_fee_matches += 1
            else:
                maker_actual_fee_mismatches += 1
                mismatches.append(
                    {
                        "field": "actual_maker_fill_fee_rate",
                        "observed": actual_fee_rate,
                        "order_id": fill.get("order_id"),
                    }
                )

    if not snapshot_events:
        reasons.append("no_fee_snapshots")
    if strategy_matches == 0:
        reasons.append("strategy_fee_not_proven")
    if config_matches == 0:
        reasons.append("config_fee_not_proven")
    if exchange_matches == 0:
        reasons.append("exchange_fee_not_proven")
    if len(maker_fills) < int(min_maker_fills):
        reasons.append(f"insufficient_maker_fills:{len(maker_fills)}<min_{int(min_maker_fills)}")
    if maker_actual_fee_matches < int(min_maker_fills):
        reasons.append(
            f"insufficient_actual_maker_fee_matches:{maker_actual_fee_matches}<min_{int(min_maker_fills)}"
        )
    if taker_fills:
        reasons.append(f"taker_fills_seen:{len(taker_fills)}")
    if snapshot_timestamp_missing:
        reasons.append(f"fee_snapshot_timestamp_missing:{snapshot_timestamp_missing}")
    if snapshot_stale:
        reasons.append(f"fee_snapshot_stale:{snapshot_stale}")
    if maker_actual_fee_timestamp_missing:
        reasons.append(f"actual_maker_fee_timestamp_missing:{maker_actual_fee_timestamp_missing}")
    if maker_actual_fee_stale:
        reasons.append(f"actual_maker_fee_stale:{maker_actual_fee_stale}")
    if maker_actual_fee_mismatches:
        reasons.append(f"actual_maker_fee_mismatches:{maker_actual_fee_mismatches}")
    if maker_fill_quote_side_invalid:
        reasons.append(f"maker_fill_quote_side_invalid:{maker_fill_quote_side_invalid}")
    if maker_fill_order_type_invalid:
        reasons.append(f"maker_fill_order_type_invalid:{maker_fill_order_type_invalid}")
    if maker_fill_tif_invalid:
        reasons.append(f"maker_fill_tif_invalid:{maker_fill_tif_invalid}")
    if maker_fill_expected_fee_invalid:
        reasons.append(f"maker_fill_expected_fee_invalid:{maker_fill_expected_fee_invalid}")
    if maker_fill_actual_fee_paid_missing:
        reasons.append(f"maker_fill_actual_fee_paid_missing:{maker_fill_actual_fee_paid_missing}")
    if maker_fill_actual_fee_paid_invalid:
        reasons.append(f"maker_fill_actual_fee_paid_invalid:{maker_fill_actual_fee_paid_invalid}")
    if maker_fill_price_invalid:
        reasons.append(f"maker_fill_price_invalid:{maker_fill_price_invalid}")
    if maker_fill_amount_invalid:
        reasons.append(f"maker_fill_amount_invalid:{maker_fill_amount_invalid}")
    if maker_fill_actual_fee_paid_mismatch:
        reasons.append(f"maker_fill_actual_fee_paid_mismatch:{maker_fill_actual_fee_paid_mismatch}")
    if any(item.get("field") in {"config_fee_rate", "exchange_maker_fee_rate"} for item in mismatches):
        reasons.append("fee_snapshot_mismatch")

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "expected_maker_fee_rate": float(expected_maker_fee_rate),
        "expected_taker_fee_rate": float(expected_taker_fee_rate) if expected_taker_fee_rate is not None else None,
        "tolerance": float(tolerance),
        "max_evidence_age_seconds": float(max_evidence_age_seconds),
        "event_count": len(events),
        "fee_snapshots": {
            "count": len(snapshot_events),
            "fresh": snapshot_fresh,
            "timestamp_missing": snapshot_timestamp_missing,
            "stale": snapshot_stale,
            "max_age_seconds": max(snapshot_age_seconds) if snapshot_age_seconds else None,
            "strategy_matches": strategy_matches,
            "config_matches": config_matches,
            "exchange_matches": exchange_matches,
            "exchange_unavailable": exchange_unavailable,
            "exchange_sources": exchange_sources,
        },
        "fills": {
            "count": len(fill_events),
            "maker": len(maker_fills),
            "taker": len(taker_fills),
            "maker_actual_fee_observed": maker_actual_fee_observed,
            "maker_actual_fee_matches": maker_actual_fee_matches,
            "maker_actual_fee_mismatches": maker_actual_fee_mismatches,
            "maker_actual_fee_fresh": maker_actual_fee_fresh,
            "maker_actual_fee_timestamp_missing": maker_actual_fee_timestamp_missing,
            "maker_actual_fee_stale": maker_actual_fee_stale,
            "maker_actual_fee_max_age_seconds": max(maker_actual_fee_age_seconds) if maker_actual_fee_age_seconds else None,
            "maker_fill_quote_side_invalid": maker_fill_quote_side_invalid,
            "maker_fill_order_type_invalid": maker_fill_order_type_invalid,
            "maker_fill_tif_invalid": maker_fill_tif_invalid,
            "maker_fill_expected_fee_invalid": maker_fill_expected_fee_invalid,
            "maker_fill_actual_fee_paid_missing": maker_fill_actual_fee_paid_missing,
            "maker_fill_actual_fee_paid_invalid": maker_fill_actual_fee_paid_invalid,
            "maker_fill_price_invalid": maker_fill_price_invalid,
            "maker_fill_amount_invalid": maker_fill_amount_invalid,
            "maker_fill_actual_fee_paid_mismatch": maker_fill_actual_fee_paid_mismatch,
            "min_maker_fills": int(min_maker_fills),
        },
        "mismatches": mismatches[:50],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate maker fee tier and fill-fee evidence.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL audit log to inspect.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-maker-fee-rate", type=float, default=0.00015)
    parser.add_argument("--expected-taker-fee-rate", type=float, default=0.00045)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--min-maker-fills", type=int, default=1)
    parser.add_argument("--max-evidence-age-seconds", type=float, default=86_400.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_fee_evidence_report(
        read_jsonl_events(args.input),
        expected_maker_fee_rate=float(args.expected_maker_fee_rate),
        expected_taker_fee_rate=float(args.expected_taker_fee_rate),
        tolerance=float(args.tolerance),
        min_maker_fills=int(args.min_maker_fills),
        max_evidence_age_seconds=float(args.max_evidence_age_seconds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
