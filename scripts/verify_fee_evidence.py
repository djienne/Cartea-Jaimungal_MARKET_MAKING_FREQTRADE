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


def fee_rate_matches(expected: float | None, observed: float | None, tolerance: float) -> bool | None:
    if expected is None or observed is None:
        return None
    return abs(float(expected) - float(observed)) <= float(tolerance)


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


def iter_fee_snapshots(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for event in events:
        snapshot = event.get("fee_snapshot")
        if isinstance(snapshot, dict):
            yield snapshot


def build_fee_evidence_report(
    events: list[dict[str, Any]],
    *,
    expected_maker_fee_rate: float = 0.00015,
    expected_taker_fee_rate: float | None = 0.00045,
    tolerance: float = 1e-9,
    min_maker_fills: int = 1,
) -> dict[str, Any]:
    reasons: list[str] = []
    mismatches: list[dict[str, Any]] = []

    snapshots = list(iter_fee_snapshots(events))
    strategy_matches = 0
    config_matches = 0
    exchange_matches = 0
    exchange_unavailable = 0
    exchange_sources: dict[str, int] = {}

    for snapshot in snapshots:
        strategy_fee = finite_float(snapshot.get("strategy_maker_fee_rate"))
        config_fee = finite_float(snapshot.get("config_fee_rate"))
        exchange_maker_fee = finite_float(snapshot.get("exchange_maker_fee_rate"))
        exchange_taker_fee = finite_float(snapshot.get("exchange_taker_fee_rate"))
        source = str(snapshot.get("exchange_fee_source") or "unknown")
        exchange_sources[source] = exchange_sources.get(source, 0) + 1

        if fee_rate_matches(expected_maker_fee_rate, strategy_fee, tolerance) is True:
            strategy_matches += 1
        elif strategy_fee is not None:
            mismatches.append({"field": "strategy_maker_fee_rate", "observed": strategy_fee})

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
        elif snapshot.get("exchange_maker_fee_matches_strategy") is True and fee_rate_matches(
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

    for fill in maker_fills:
        actual_fee_rate = finite_float(fill.get("actual_fee_rate"))
        if actual_fee_rate is None:
            continue
        maker_actual_fee_observed += 1
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

    if not snapshots:
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
    if maker_actual_fee_mismatches:
        reasons.append(f"actual_maker_fee_mismatches:{maker_actual_fee_mismatches}")
    if any(item.get("field") in {"config_fee_rate", "exchange_maker_fee_rate"} for item in mismatches):
        reasons.append("fee_snapshot_mismatch")

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "expected_maker_fee_rate": float(expected_maker_fee_rate),
        "expected_taker_fee_rate": float(expected_taker_fee_rate) if expected_taker_fee_rate is not None else None,
        "tolerance": float(tolerance),
        "event_count": len(events),
        "fee_snapshots": {
            "count": len(snapshots),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_fee_evidence_report(
        read_jsonl_events(args.input),
        expected_maker_fee_rate=float(args.expected_maker_fee_rate),
        expected_taker_fee_rate=float(args.expected_taker_fee_rate),
        tolerance=float(args.tolerance),
        min_maker_fills=int(args.min_maker_fills),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
