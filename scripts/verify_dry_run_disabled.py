#!/usr/bin/env python3
"""Run a bounded dry-run start and verify the fail-closed strategy places no orders."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import subprocess
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = ROOT / "user_data" / "logs" / "freqtrade_gate_disabled.log"
DEFAULT_MM_DEBUG_PATH = ROOT / "user_data" / "logs" / "mm_debug.jsonl"
ORDER_PATTERNS = (
    "order dry_run",
    "order was created",
    "entering trade",
    "executing buy",
    "executing sell",
    "create_order",
)
START_PATTERNS = (
    "runmode set to dry_run",
    "strategy using order_time_in_force",
    "using resolved strategy market_making",
)
FATAL_PATTERNS = (
    "configuration error",
    "operationalexception",
    "traceback",
    "time in force policies are not supported",
)


def run(command: Sequence[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def evaluate_logs(log_text: str) -> tuple[bool, str, list[str]]:
    lower = log_text.lower()
    fatal_lines = [
        line
        for line in log_text.splitlines()
        if any(pattern in line.lower() for pattern in FATAL_PATTERNS)
    ]
    if fatal_lines:
        return False, "runtime_error_detected", fatal_lines[:20]
    order_lines = [
        line
        for line in log_text.splitlines()
        if any(pattern in line.lower() for pattern in ORDER_PATTERNS)
    ]
    if order_lines:
        return False, "order_creation_detected", order_lines[:20]
    if not any(pattern in lower for pattern in START_PATTERNS):
        return False, "no_start_evidence", log_text.splitlines()[-80:]
    return True, "ok", []


def parse_event_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def recent_health_events(mm_debug_path: Path, since: datetime) -> list[dict]:
    if not mm_debug_path.exists():
        return []

    threshold = since.astimezone(timezone.utc) - timedelta(seconds=2)
    events: list[dict] = []
    for line in mm_debug_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "health":
            continue
        ts = parse_event_timestamp(payload.get("ts"))
        if ts is None or ts < threshold:
            continue
        events.append(payload)
    return events


def run_gate(
    seconds: int,
    log_path: Path,
    mm_debug_path: Path,
    require_health: bool,
    *,
    start_collector: bool = False,
    collector_service: str = "hl-collector2",
    collector_warmup_seconds: int = 70,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    collector_start: subprocess.CompletedProcess[str] | None = None
    collector_stop: subprocess.CompletedProcess[str] | None = None
    if start_collector:
        collector_start = run(["docker", "compose", "up", "-d", "--no-deps", collector_service], timeout=180)
        time.sleep(max(0, int(collector_warmup_seconds)))

    started_at = datetime.now(timezone.utc)
    cleanup_before = run(["docker", "rm", "-f", "MM_ADV"], timeout=60)
    start = run(["docker", "compose", "up", "-d", "--no-deps", "freqtrade"], timeout=180)
    time.sleep(seconds)
    logs = run(["docker", "logs", "MM_ADV"], timeout=60)
    stop = run(["docker", "stop", "MM_ADV"], timeout=60)
    if start_collector:
        collector_stop = run(["docker", "compose", "stop", collector_service], timeout=60)
    log_text = logs.stdout or ""
    log_path.write_text(log_text, encoding="utf-8")

    ok, reason, evidence = evaluate_logs(log_text)
    health_events = recent_health_events(mm_debug_path, started_at)
    latest_health = health_events[-1] if health_events else None
    if ok and require_health and not health_events:
        ok = False
        reason = "no_recent_health_event"
        evidence = [f"No health event found in {mm_debug_path} since {started_at.isoformat()}"]

    collector_ok = True
    if start_collector:
        collector_ok = (
            collector_start is not None
            and collector_start.returncode == 0
            and collector_stop is not None
            and collector_stop.returncode == 0
        )

    return {
        "passed": ok and start.returncode == 0 and stop.returncode == 0 and collector_ok,
        "reason": reason,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": seconds,
        "log_path": str(log_path),
        "mm_debug_path": str(mm_debug_path),
        "cleanup_before_returncode": cleanup_before.returncode,
        "docker_compose_up_returncode": start.returncode,
        "docker_stop_returncode": stop.returncode,
        "collector_service": collector_service if start_collector else None,
        "collector_warmup_seconds": int(collector_warmup_seconds) if start_collector else None,
        "collector_start_returncode": collector_start.returncode if collector_start is not None else None,
        "collector_stop_returncode": collector_stop.returncode if collector_stop is not None else None,
        "health_events": len(health_events),
        "latest_health": latest_health,
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify trading_enabled=False dry-run starts without order creation.")
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--mm-debug-path", type=Path, default=DEFAULT_MM_DEBUG_PATH)
    parser.add_argument("--require-health", action="store_true")
    parser.add_argument("--start-collector", action="store_true")
    parser.add_argument("--collector-service", default="hl-collector2")
    parser.add_argument("--collector-warmup-seconds", type=int, default=70)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_gate(
        max(1, int(args.seconds)),
        args.log_path,
        args.mm_debug_path,
        bool(args.require_health),
        start_collector=bool(args.start_collector),
        collector_service=str(args.collector_service),
        collector_warmup_seconds=max(0, int(args.collector_warmup_seconds)),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
