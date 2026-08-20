#!/usr/bin/env python3
"""Run a bounded dry-run start and verify the fail-closed strategy places no orders."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from param_utils import PARAM_SCHEMA_VERSION, atomic_write_json  # noqa: E402

DEFAULT_LOG_PATH = ROOT / "user_data" / "logs" / "freqtrade_gate_disabled.log"
DEFAULT_MM_DEBUG_PATH = ROOT / "user_data" / "logs" / "mm_debug.jsonl"
DEFAULT_CONFIG_PATH = ROOT / "user_data" / "logs" / "mm_gate_disabled_config.json"
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


# A name that collides with neither live leg (MM_ADV_LONG / MM_ADV_SHORT).
GATE_CONTAINER = "MM_GATE_SMOKE"
# `compose run` on this service starts a NEW ephemeral container.
DEFAULT_FREQTRADE_SERVICE = "mm-long"

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
    atomic_write_json(
        param_dir / "kappa.json",
        {
            "ETH": {
                **common,
                "kappa+": 2.0,
                "kappa-": 2.0,
                "lambda+": 0.1,
                "lambda-": 0.1,
                "kappa+_raw": 2.0,
                "kappa-_raw": 2.0,
                "lambda+_raw": 0.1,
                "lambda-_raw": 0.1,
                "lambda_source": "mo_survival_fit",
                "unit": {"kappa": "1/USDC", "lambda": "events_per_second"},
                "n_quotes": 100,
                "n_trades": 50,
                "n_points_plus": 8,
                "n_points_minus": 8,
                "r2_plus": 0.5,
                "r2_minus": 0.5,
                "depth_p95_plus": 2.5,
                "depth_p95_minus": 2.5,
                "depth_max_fitted_plus": 3.0,
                "depth_max_fitted_minus": 3.0,
                "sigma2_per_sec": 0.02,
            }
        },
    )
    atomic_write_json(
        param_dir / "lambda.json",
        {
            "ETH": {
                **common,
                "lambda+": 0.1,
                "lambda-": 0.1,
                "lambda+_raw": 0.1,
                "lambda-_raw": 0.1,
                "lambda_source": "mo_survival_fit",
                "unit": "events_per_second",
                "n_trades": 50,
            }
        },
    )
    atomic_write_json(
        param_dir / "epsilon.json",
        {
            "ETH": {
                **common,
                "epsilon+": 0.0,
                "epsilon-": 0.0,
                "epsilon+_raw": 0.0,
                "epsilon-_raw": 0.0,
                "unit": "USDC",
                "estimator": "gate_static",
                "window_ms": 5000,
                "n_buy_events": 60,
                "n_sell_events": 60,
                "toxicity_plus": 0.0,
                "toxicity_minus": 0.0,
            }
        },
    )


def write_gate_config(
    base_config: Path,
    output_config: Path,
    *,
    container_param_dir: str,
    max_collector_age_seconds: int,
) -> None:
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["dry_run"] = True
    config["force_entry_enable"] = False
    config["stake_amount"] = 25
    config["tradable_balance_ratio"] = 0.10
    config["fee"] = 0.00015
    mm_config = config.get("market_making")
    if not isinstance(mm_config, dict):
        mm_config = {}
    mm_config.update(
        {
            "trading_enabled": False,
            "post_only_verified": False,
            "disable_param_refresh": True,
            "param_dir": container_param_dir,
            "max_param_age_seconds": int(max_collector_age_seconds),
            "max_collector_age_seconds": int(max_collector_age_seconds),
        }
    )
    config["market_making"] = mm_config
    api_server = config.get("api_server")
    if isinstance(api_server, dict):
        api_server["forcebuy_enable"] = False
        api_server["force_entry_enable"] = False
        api_server["Force_entry"] = False
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def remove_gate_database_files(log_dir: Path) -> None:
    for path in log_dir.glob("mm_gate_disabled_trades.sqlite*"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_gate(
    seconds: int,
    log_path: Path,
    mm_debug_path: Path,
    require_health: bool,
    *,
    start_collector: bool = False,
    collector_service: str | None = None,
    freqtrade_service: str = DEFAULT_FREQTRADE_SERVICE,
    collector_warmup_seconds: int = 70,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    remove_gate_database_files(log_path.parent)

    collector_start: subprocess.CompletedProcess[str] | None = None
    collector_stop: subprocess.CompletedProcess[str] | None = None
    if start_collector:
        collector_start = run(["docker", "compose", "up", "-d", "--no-deps", collector_service], timeout=180)
        time.sleep(max(0, int(collector_warmup_seconds)))

    config_path = DEFAULT_CONFIG_PATH if log_path == DEFAULT_LOG_PATH else log_path.with_name("mm_gate_disabled_config.json")
    param_dir = log_path.parent / "mm_gate_disabled_params"
    container_log_path = "/freqtrade/user_data/logs/freqtrade_gate_disabled.log"
    container_config_path = "/freqtrade/user_data/logs/mm_gate_disabled_config.json"
    container_param_dir = "/freqtrade/user_data/logs/mm_gate_disabled_params"
    db_path = "/freqtrade/user_data/logs/mm_gate_disabled_trades.sqlite"
    max_collector_age = max(90, int(seconds) + int(collector_warmup_seconds) + 120)
    write_gate_params(param_dir)
    write_gate_config(
        ROOT / "user_data" / "config.json",
        config_path,
        container_param_dir=container_param_dir,
        max_collector_age_seconds=max_collector_age,
    )

    started_at = datetime.now(timezone.utc)
    # Throwaway name, NOT a live leg. This used to be `MM_ADV`, so running the
    # battery destroyed the production bot and needed a restore step; and once
    # the deployment split into MM_ADV_LONG / MM_ADV_SHORT the name matched
    # nothing at all and the gate simply stopped working.
    cleanup_before = run(["docker", "rm", "-f", GATE_CONTAINER], timeout=60)
    start = run(
        [
            "docker",
            "compose",
            "run",
            "-d",
            "--name",
            GATE_CONTAINER,
            "--no-deps",
            freqtrade_service,
            "trade",
            "--logfile",
            container_log_path,
            "--db-url",
            f"sqlite:///{db_path}",
            "--config",
            container_config_path,
            "--strategy",
            "Market_Making",
        ],
        timeout=180,
    )
    time.sleep(seconds)
    logs = run(["docker", "logs", GATE_CONTAINER], timeout=60)
    stop = run(["docker", "stop", GATE_CONTAINER], timeout=60)
    if start_collector and collector_service:
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
        "config_path": str(config_path),
        "max_collector_age_seconds": max_collector_age,
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
    # The collectors moved to HYPERLIQUID_DATA/docker-compose.yml and run under
    # restart: unless-stopped, so they are always up. Managing them from here
    # would punch a hole in the tape every consumer reads. Off unless asked.
    parser.add_argument("--collector-service", default=None)
    parser.add_argument("--freqtrade-service", default=DEFAULT_FREQTRADE_SERVICE)
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
