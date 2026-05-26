#!/usr/bin/env python3
"""Append an explicit live-canary manual-monitoring acknowledgement event.

This helper does not enable trading. It only records auditable JSONL evidence
that an operator monitored a canary session, which `verify_live_canary.py`
requires in addition to the command-line `--manual-monitoring-ack` flag.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUTPUT = Path("user_data/logs/mm_debug.jsonl")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_operator() -> str:
    for name in ("USER", "USERNAME"):
        value = os.environ.get(name)
        if value:
            return value
    return "unknown"


def build_manual_monitoring_ack_event(
    *,
    session_id: str | None,
    operator: str,
    ts: str | None = None,
    note: str | None = None,
    deployment_stage: str = "canary",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event": "manual_monitoring_ack",
        "ts": ts or utc_now_iso(),
        "acknowledged": True,
        "manual_monitoring_ack": True,
        "deployment_stage": deployment_stage,
        "operator": operator,
        "source": "record_manual_monitoring_ack.py",
        "version": 1,
        "requirements_observed": {
            "tiny_fixed_stake": True,
            "one_symbol": True,
            "post_only_required": True,
            "kill_on_taker_fill_required": True,
            "manual_monitoring_required": True,
        },
    }
    if session_id:
        event["session_id"] = session_id
    if note:
        event["note"] = note
    return event


def append_jsonl_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a live-canary manual monitoring acknowledgement.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL audit log to append to.")
    parser.add_argument("--session-id", default=None, help="Optional canary session id to include in the event.")
    parser.add_argument("--operator", default=default_operator(), help="Operator name for audit evidence.")
    parser.add_argument("--note", default=None, help="Optional short human note.")
    parser.add_argument("--deployment-stage", choices=["canary", "production"], default="canary")
    parser.add_argument(
        "--acknowledge-risk",
        action="store_true",
        help="Required. Confirms this event corresponds to an actually monitored tiny canary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.acknowledge_risk:
        print("Refusing to write acknowledgement without --acknowledge-risk.")
        return 2

    event = build_manual_monitoring_ack_event(
        session_id=args.session_id,
        operator=str(args.operator),
        note=args.note,
        deployment_stage=str(args.deployment_stage),
    )
    append_jsonl_event(args.output, event)
    print(json.dumps(event, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
