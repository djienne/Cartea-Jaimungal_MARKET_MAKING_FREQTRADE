#!/usr/bin/env python3
"""Probe Freqtrade runtime support for configured time-in-force values.

This is a no-network/container-startup compatibility artifact. It does not
submit orders and it does not prove Hyperliquid maker safety. Its purpose is to
record whether the exact Freqtrade image can load configs using GTC, PO, and Alo
so live execution cannot rely on an undocumented assumption.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = Path("user_data/config.json")
DEFAULT_OUTPUT = Path("docs/freqtrade_tif_runtime_report.json")
DEFAULT_WORK_DIR = Path("user_data/logs/freqtrade_tif_runtime")
DEFAULT_VARIANTS = ("GTC", "PO", "Alo")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def tail(text: str | None, max_chars: int = 2_500) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return text if len(text) <= max_chars else text[-max_chars:]


def build_variant_config(base_config: dict[str, Any], tif: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["dry_run"] = True
    config["force_entry_enable"] = False
    config["order_time_in_force"] = {"entry": tif, "exit": tif}
    market_making = dict(config.get("market_making") or {})
    market_making["trading_enabled"] = False
    market_making["post_only_verified"] = False
    market_making["deployment_stage"] = "research"
    config["market_making"] = market_making
    api_server = config.get("api_server")
    if isinstance(api_server, dict):
        api_server["forcebuy_enable"] = False
        api_server["force_entry_enable"] = False
        api_server["Force_entry"] = False
    return config


def write_variant_configs(
    base_config_path: Path,
    work_dir: Path,
    variants: Sequence[str],
) -> dict[str, Path]:
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for tif in variants:
        name = str(tif).lower().replace("/", "_").replace(" ", "_")
        path = work_dir / f"config_tif_{name}.json"
        path.write_text(
            json.dumps(build_variant_config(base_config, str(tif)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths[str(tif)] = path
    return paths


def container_config_path(host_path: Path) -> str:
    resolved = host_path.resolve()
    user_data = (ROOT / "user_data").resolve()
    try:
        relative = resolved.relative_to(user_data)
    except ValueError as exc:
        raise ValueError(f"config must be under {user_data}") from exc
    return "/freqtrade/user_data/" + relative.as_posix()


def freqtrade_list_strategies_command(container_config: str) -> list[str]:
    return [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "freqtrade",
        "list-strategies",
        "--config",
        container_config,
    ]


def run_command(command: Sequence[str], *, timeout: int) -> dict[str, Any]:
    proc = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "command": list(command),
        "returncode": int(proc.returncode),
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


def post_only_candidate(tif: str) -> bool:
    return str(tif).strip().lower() in {"po", "post_only", "postonly", "alo"}


def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_tif = {str(result.get("time_in_force")): result for result in results}
    reasons: list[str] = []
    gtc = by_tif.get("GTC")
    if gtc is None:
        reasons.append("missing_gtc_result")
    elif int(gtc.get("returncode") or 0) != 0:
        reasons.append("gtc_research_config_rejected")

    for result in results:
        if "returncode" not in result:
            reasons.append(f"{result.get('time_in_force')}_missing_returncode")

    post_only_results = [result for result in results if post_only_candidate(str(result.get("time_in_force")))]
    post_only_supported = any(int(result.get("returncode") or 0) == 0 for result in post_only_results)
    unsupported = [
        str(result.get("time_in_force"))
        for result in post_only_results
        if int(result.get("returncode") or 0) != 0
    ]
    supported = [
        str(result.get("time_in_force"))
        for result in post_only_results
        if int(result.get("returncode") or 0) == 0
    ]

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "safe_default": "no order submission; config-load probe only",
        "research_gtc_supported": bool(gtc and int(gtc.get("returncode") or 0) == 0),
        "freqtrade_post_only_config_supported": bool(post_only_supported),
        "freqtrade_post_only_supported_tifs": supported,
        "freqtrade_post_only_unsupported_tifs": unsupported,
        "live_safe_via_freqtrade": False,
        "live_safe_reason": "startup_config_acceptance_is_not_exchange_alo_submit_evidence",
        "requires_exchange_submit_evidence": True,
        "variants": results,
    }


def probe_runtime(
    *,
    base_config: Path,
    work_dir: Path,
    variants: Sequence[str],
    timeout: int,
) -> dict[str, Any]:
    paths = write_variant_configs(base_config, work_dir, variants)
    results: list[dict[str, Any]] = []
    for tif, path in paths.items():
        container_path = container_config_path(path)
        result = run_command(freqtrade_list_strategies_command(container_path), timeout=timeout)
        result.update(
            {
                "time_in_force": tif,
                "host_config": str(path),
                "container_config": container_path,
                "loads": int(result.get("returncode") or 0) == 0,
            }
        )
        results.append(result)
    return build_report(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Freqtrade TIF config support in the Docker runtime.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--variant", action="append", dest="variants", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = tuple(args.variants or DEFAULT_VARIANTS)
    report = probe_runtime(
        base_config=args.base_config,
        work_dir=args.work_dir,
        variants=variants,
        timeout=max(1, int(args.timeout)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
