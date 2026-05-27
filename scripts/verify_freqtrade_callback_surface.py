#!/usr/bin/env python3
"""Verify strategy callbacks in the actual Freqtrade runtime.

This complements the AST safety check by importing the strategy in the runtime
environment and inspecting the callable signatures Freqtrade will see.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY_PATH = Path("user_data/strategies/Market_Making.py")
DEFAULT_OUTPUT = Path("docs/freqtrade_callback_surface_report.json")


EXPECTED_CALLBACKS: dict[str, list[str]] = {
    "custom_entry_price": ["self", "pair", "trade", "current_time", "proposed_rate", "entry_tag", "side"],
    "custom_exit_price": ["self", "pair", "trade", "current_time", "proposed_rate", "current_profit", "exit_tag"],
    "custom_stake_amount": [
        "self",
        "pair",
        "current_time",
        "current_rate",
        "proposed_stake",
        "min_stake",
        "max_stake",
        "leverage",
        "entry_tag",
        "side",
    ],
    "leverage": [
        "self",
        "pair",
        "current_time",
        "current_rate",
        "proposed_leverage",
        "max_leverage",
        "entry_tag",
        "side",
    ],
    "confirm_trade_entry": [
        "self",
        "pair",
        "order_type",
        "amount",
        "rate",
        "time_in_force",
        "current_time",
        "entry_tag",
        "side",
    ],
    "confirm_trade_exit": [
        "self",
        "pair",
        "trade",
        "order_type",
        "amount",
        "rate",
        "time_in_force",
        "exit_reason",
        "current_time",
    ],
    "adjust_entry_price": [
        "self",
        "trade",
        "order",
        "pair",
        "current_time",
        "proposed_rate",
        "current_order_rate",
        "entry_tag",
        "side",
    ],
    "adjust_exit_price": [
        "self",
        "trade",
        "order",
        "pair",
        "current_time",
        "proposed_rate",
        "current_order_rate",
        "exit_tag",
        "side",
    ],
    "order_filled": ["self", "pair", "trade", "order", "current_time"],
}

CALLBACKS_REQUIRING_KWARGS = set(EXPECTED_CALLBACKS)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def import_strategy_class(strategy_path: Path) -> type:
    resolved = strategy_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    for path in (resolved.parent, resolved.parents[2] / "scripts" if len(resolved.parents) > 2 else None):
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("callback_surface_strategy", str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load strategy module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_class = getattr(module, "Market_Making", None)
    if strategy_class is None:
        raise RuntimeError("Market_Making class not found")
    return strategy_class


def callback_signature_payload(strategy_class: type, callback: str) -> dict[str, Any]:
    method = getattr(strategy_class, callback, None)
    if method is None:
        return {"present": False, "parameters": [], "has_var_keyword": False}
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    return {
        "present": True,
        "parameters": [param.name for param in parameters],
        "kinds": {param.name: str(param.kind) for param in parameters},
        "defaults": {
            param.name: (
                None
                if param.default is inspect.Parameter.empty
                else repr(param.default)
            )
            for param in parameters
        },
        "has_var_keyword": any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters),
    }


def build_callback_surface_report(
    strategy_class: type | None,
    *,
    strategy_path: Path,
    import_error: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    callbacks: dict[str, Any] = {}
    if import_error is not None:
        reasons.append(import_error)

    if strategy_class is not None:
        for callback, expected_prefix in EXPECTED_CALLBACKS.items():
            payload = callback_signature_payload(strategy_class, callback)
            callbacks[callback] = payload
            if not payload["present"]:
                reasons.append(f"{callback}_missing")
                continue
            observed = payload["parameters"]
            if observed[: len(expected_prefix)] != expected_prefix:
                reasons.append(f"{callback}_signature_mismatch")
            if callback in CALLBACKS_REQUIRING_KWARGS and not payload["has_var_keyword"]:
                reasons.append(f"{callback}_missing_kwargs")

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "strategy_path": str(strategy_path),
        "runtime": {
            "python": sys.version.split()[0],
            "freqtrade": package_version("freqtrade"),
            "ccxt": package_version("ccxt"),
        },
        "expected_prefixes": EXPECTED_CALLBACKS,
        "callbacks": callbacks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify imported strategy callback signatures.")
    parser.add_argument("--strategy-path", type=Path, default=DEFAULT_STRATEGY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        strategy_class = import_strategy_class(args.strategy_path)
        import_error = None
    except Exception as exc:
        strategy_class = None
        import_error = f"strategy_import_error:{exc}"
    report = build_callback_surface_report(
        strategy_class,
        strategy_path=args.strategy_path,
        import_error=import_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
