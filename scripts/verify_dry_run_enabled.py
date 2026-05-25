#!/usr/bin/env python3
"""Run a bounded dry-run with trading_enabled=True using temporary safe params.

This is a Gate 3 smoke test. It does not enable live trading and does not touch
the checked-in strategy parameter snapshots. Temporary config/params/DB files
are written under user_data/logs, which is ignored and mounted into Docker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from param_utils import PARAM_SCHEMA_VERSION, atomic_write_json  # noqa: E402
from verify_dry_run_disabled import (  # noqa: E402
    FATAL_PATTERNS,
    parse_event_timestamp,
    run,
)


DEFAULT_LOG_DIR = ROOT / "user_data" / "logs"
DEFAULT_CONTAINER_NAME = "MM_GATE_ENABLED"
ORDER_PATTERNS = (
    "order dry_run",
    "order was created",
    "entering trade",
    "create_order",
    "long signal found: about create a new trade",
    "about create a new trade",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_gate_params(param_dir: Path) -> None:
    param_dir.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    common = {
        "schema_version": PARAM_SCHEMA_VERSION,
        "status": "ok",
        "generated_at": now,
        "window_start": now,
        "window_end": now,
    }
    kappa_payload = {
        "ETH": {
            **common,
            "kappa+": 2.0,
            "kappa-": 2.0,
            "lambda+": 0.1,
            "lambda-": 0.1,
            "lambda_source": "lambda0_fit",
            "unit": {"kappa": "1/USDC", "lambda": "events_per_second"},
            "n_quotes": 100,
            "n_trades": 50,
            "n_points_plus": 5,
            "n_points_minus": 5,
            "r2_plus": 0.5,
            "r2_minus": 0.5,
        }
    }
    lambda_payload = {
        "ETH": {
            **common,
            "lambda+": 0.1,
            "lambda-": 0.1,
            "lambda_source": "lambda0_fit",
            "unit": "events_per_second",
            "n_trades": 50,
        }
    }
    epsilon_payload = {
        "ETH": {
            **common,
            "epsilon+": 0.0,
            "epsilon-": 0.0,
            "unit": "USDC",
            "estimator": "gate_static",
            "window_ms": 200,
            "n_buy_events": 10,
            "n_sell_events": 10,
            "toxicity_plus": 0.0,
            "toxicity_minus": 0.0,
        }
    }
    atomic_write_json(param_dir / "kappa.json", kappa_payload)
    atomic_write_json(param_dir / "lambda.json", lambda_payload)
    atomic_write_json(param_dir / "epsilon.json", epsilon_payload)


def write_gate_config(base_config: Path, output_config: Path, container_param_dir: str) -> None:
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["dry_run"] = True
    config["force_entry_enable"] = False
    config["stake_amount"] = 25
    config["tradable_balance_ratio"] = 0.10
    config["fee"] = 0.00015
    config["custom_price_max_distance_ratio"] = 0.05
    config["market_making"] = {
        "trading_enabled": True,
        "post_only_verified": False,
        "disable_param_refresh": True,
        "param_dir": container_param_dir,
        "max_param_age_seconds": 300,
        "max_collector_age_seconds": 180,
    }
    api_server = config.get("api_server")
    if isinstance(api_server, dict):
        api_server["forcebuy_enable"] = False
        api_server["force_entry_enable"] = False
        api_server["Force_entry"] = False
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl_events(path: Path, since: datetime, event_name: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    threshold = since.astimezone(timezone.utc)
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event_name is not None and payload.get("event") != event_name:
            continue
        ts = parse_event_timestamp(payload.get("ts"))
        if ts is None or ts < threshold:
            continue
        events.append(payload)
    return events


def evaluate_enabled_gate(log_text: str, debug_events: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    fatal_lines = [
        line
        for line in log_text.splitlines()
        if any(pattern in line.lower() for pattern in FATAL_PATTERNS)
    ]
    if fatal_lines:
        return False, "runtime_error_detected", fatal_lines[:20]

    health_events = [event for event in debug_events if event.get("event") == "health"]
    if not health_events:
        return False, "no_health_event", []
    latest_health = health_events[-1]
    for key in ("trading_enabled", "collector_fresh", "params_fresh", "hjb_fresh"):
        if latest_health.get(key) is not True:
            return False, f"health_{key}_not_true", [json.dumps(latest_health, sort_keys=True, default=str)]

    quote_events = [event for event in debug_events if event.get("event") == "quote_decision"]
    accepted_quotes = [event for event in quote_events if event.get("decision") == "accept"]
    if not accepted_quotes:
        evidence = [json.dumps(event, sort_keys=True, default=str) for event in quote_events[-10:]]
        return False, "no_accepted_quote_decision", evidence

    return True, "ok", []


def remove_gate_database_files(log_dir: Path) -> None:
    for path in log_dir.glob("mm_gate_enabled_trades.sqlite*"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_gate(
    *,
    seconds: int,
    collector_warmup_seconds: int,
    log_dir: Path,
    container_name: str,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    param_dir = log_dir / "mm_gate_enabled_params"
    config_path = log_dir / "mm_gate_enabled_config.json"
    log_path = log_dir / "freqtrade_gate_enabled.log"
    db_path = "/freqtrade/user_data/logs/mm_gate_enabled_trades.sqlite"
    mm_debug_path = ROOT / "user_data" / "logs" / "mm_debug.jsonl"

    remove_gate_database_files(log_dir)
    write_gate_params(param_dir)
    write_gate_config(
        ROOT / "user_data" / "config.json",
        config_path,
        "/freqtrade/user_data/logs/mm_gate_enabled_params",
    )

    run(["docker", "rm", "-f", container_name], timeout=60)
    collector_start = run(["docker", "compose", "up", "-d", "--no-deps", "hl-collector2"], timeout=180)
    time.sleep(max(0, int(collector_warmup_seconds)))
    write_gate_params(param_dir)

    started_at = datetime.now(timezone.utc)
    start = run(
        [
            "docker",
            "compose",
            "run",
            "-d",
            "--name",
            container_name,
            "--no-deps",
            "freqtrade",
            "trade",
            "--logfile",
            "/freqtrade/user_data/logs/freqtrade_gate_enabled.log",
            "--db-url",
            f"sqlite:///{db_path}",
            "--config",
            "/freqtrade/user_data/logs/mm_gate_enabled_config.json",
            "--strategy",
            "Market_Making",
        ],
        timeout=180,
    )
    time.sleep(max(1, int(seconds)))
    logs = run(["docker", "logs", container_name], timeout=60)
    stop = run(["docker", "rm", "-f", container_name], timeout=60)
    collector_stop = run(["docker", "compose", "stop", "hl-collector2"], timeout=60)

    log_text = logs.stdout or ""
    log_path.write_text(log_text, encoding="utf-8")
    debug_events = read_jsonl_events(mm_debug_path, started_at)
    ok, reason, evidence = evaluate_enabled_gate(log_text, debug_events)
    order_evidence = [
        line
        for line in log_text.splitlines()
        if any(pattern in line.lower() for pattern in ORDER_PATTERNS)
    ][:20]
    quote_events = [event for event in debug_events if event.get("event") == "quote_decision"]
    accepted_quotes = [event for event in quote_events if event.get("decision") == "accept"]
    if ok and not order_evidence:
        ok = False
        reason = "no_dry_run_order_evidence"
        evidence = [json.dumps(event, sort_keys=True, default=str) for event in accepted_quotes[-3:]]

    return {
        "passed": (
            ok
            and collector_start.returncode == 0
            and collector_stop.returncode == 0
            and start.returncode == 0
            and stop.returncode == 0
        ),
        "reason": reason,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": seconds,
        "collector_warmup_seconds": collector_warmup_seconds,
        "collector_start_returncode": collector_start.returncode,
        "collector_stop_returncode": collector_stop.returncode,
        "docker_start_returncode": start.returncode,
        "docker_stop_returncode": stop.returncode,
        "log_path": str(log_path),
        "mm_debug_path": str(mm_debug_path),
        "health_events": len([event for event in debug_events if event.get("event") == "health"]),
        "quote_decisions": len(quote_events),
        "accepted_quote_decisions": len(accepted_quotes),
        "latest_health": [event for event in debug_events if event.get("event") == "health"][-1]
        if any(event.get("event") == "health" for event in debug_events)
        else None,
        "latest_accepted_quote": accepted_quotes[-1] if accepted_quotes else None,
        "order_evidence": order_evidence,
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dry-run enabled quote path with temporary safe params.")
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--collector-warmup-seconds", type=int, default=70)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_gate(
        seconds=max(1, int(args.seconds)),
        collector_warmup_seconds=max(0, int(args.collector_warmup_seconds)),
        log_dir=args.log_dir,
        container_name=str(args.container_name),
    )
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
