#!/usr/bin/env python3
"""Verify checked-in strategy defaults keep the fail-closed PLAN.md posture."""

from __future__ import annotations

import argparse
import ast
import json
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY = Path("user_data/strategies/Market_Making.py")
DEFAULT_OUTPUT = Path("docs/strategy_safety_report.json")


BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -safe_eval(node.operand)
    if isinstance(node, ast.BinOp):
        op = BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported_binop")
        return op(safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.Dict):
        return {safe_eval(key): safe_eval(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Tuple):
        return tuple(safe_eval(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [safe_eval(item) for item in node.elts]
    raise ValueError(f"unsupported_node:{type(node).__name__}")


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def parse_strategy_source(source: str) -> tuple[dict[str, Any], dict[str, list[str]], str | None]:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        return {}, {}, f"strategy_syntax_error:{exc}"

    strategy_class: ast.ClassDef | None = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Market_Making":
            strategy_class = node
            break
    if strategy_class is None:
        return {}, {}, "market_making_class_missing"

    defaults: dict[str, Any] = {}
    signatures: dict[str, list[str]] = {}
    for node in strategy_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        defaults[target.id] = safe_eval(node.value)
                    except Exception:
                        pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                defaults[node.target.id] = safe_eval(node.value) if node.value is not None else None
            except Exception:
                pass
        elif isinstance(node, ast.FunctionDef):
            signatures[node.name] = [arg.arg for arg in node.args.args]
    return defaults, signatures, None


def build_strategy_safety_report(
    source: str | None,
    *,
    strategy_path: Path,
    load_error: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    defaults: dict[str, Any] = {}
    signatures: dict[str, list[str]] = {}
    parse_error = None
    if load_error is not None:
        reasons.append(load_error)
    else:
        defaults, signatures, parse_error = parse_strategy_source(source or "")
        if parse_error is not None:
            reasons.append(parse_error)

    def require_value(key: str, expected: Any, reason: str) -> None:
        if defaults.get(key) != expected:
            reasons.append(reason)

    require_value("trading_enabled", False, "trading_enabled_default_not_false")
    require_value("fail_closed_reason", "initial_safety_lock", "fail_closed_reason_default_changed")
    require_value("post_only_verified", False, "post_only_verified_default_not_false")
    require_value("can_short", False, "can_short_default_not_false")
    require_value("use_exit_signal", True, "use_exit_signal_not_true")
    require_value("position_adjustment_enable", False, "position_adjustment_enable_not_false")

    minimal_roi = defaults.get("minimal_roi")
    if not isinstance(minimal_roi, dict):
        reasons.append("minimal_roi_missing")
    elif finite_float(minimal_roi.get("0")) == -1:
        reasons.append("minimal_roi_force_exit_minus_one")
    elif finite_float(minimal_roi.get("0")) != 10:
        reasons.append("minimal_roi_not_disabled_default")

    order_types = defaults.get("order_types")
    if not isinstance(order_types, dict):
        reasons.append("order_types_missing")
    else:
        if str(order_types.get("entry") or "").lower() != "limit":
            reasons.append("entry_order_type_not_limit")
        if str(order_types.get("exit") or "").lower() != "limit":
            reasons.append("exit_order_type_not_limit")

    for key in ("hjb_alpha", "hjb_phi", "inventory_unit_base"):
        value = finite_float(defaults.get(key))
        if value is None or value <= 0:
            reasons.append(f"{key}_not_positive")

    if finite_float(defaults.get("hjb_q_max")) != 3:
        reasons.append("hjb_q_max_not_three")
    if finite_float(defaults.get("max_abs_inventory_units")) != 3:
        reasons.append("max_abs_inventory_units_not_three")
    if finite_float(defaults.get("max_daily_loss_usdc")) is None or float(defaults.get("max_daily_loss_usdc")) > 20:
        reasons.append("max_daily_loss_above_plan_limit")

    for key in ("kill_on_taker_fill", "kill_on_time_in_force_mismatch", "kill_on_unknown_liquidity_fill"):
        if defaults.get(key) is not True:
            reasons.append(f"{key}_not_true")

    expected_entry_signature = ["self", "pair", "trade", "current_time", "proposed_rate"]
    if signatures.get("custom_entry_price", [])[:5] != expected_entry_signature:
        reasons.append("custom_entry_price_signature_mismatch")
    for callback in ("confirm_trade_entry", "confirm_trade_exit", "custom_exit_price", "adjust_entry_price", "adjust_exit_price"):
        if callback not in signatures:
            reasons.append(f"{callback}_missing")

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "strategy_path": str(strategy_path),
        "observed": {
            "defaults": {
                key: defaults.get(key)
                for key in (
                    "trading_enabled",
                    "fail_closed_reason",
                    "post_only_verified",
                    "can_short",
                    "use_exit_signal",
                    "position_adjustment_enable",
                    "minimal_roi",
                    "order_types",
                    "hjb_alpha",
                    "hjb_phi",
                    "hjb_q_max",
                    "inventory_unit_base",
                    "max_abs_inventory_units",
                    "max_daily_loss_usdc",
                    "kill_on_taker_fill",
                    "kill_on_time_in_force_mismatch",
                    "kill_on_unknown_liquidity_fill",
                )
            },
            "signatures": {
                key: signatures.get(key)
                for key in (
                    "custom_entry_price",
                    "custom_exit_price",
                    "confirm_trade_entry",
                    "confirm_trade_exit",
                    "adjust_entry_price",
                    "adjust_exit_price",
                )
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checked-in strategy fail-closed safety defaults.")
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = args.strategy.read_text(encoding="utf-8")
        load_error = None
    except FileNotFoundError:
        source = None
        load_error = "strategy_missing"
    except Exception as exc:
        source = None
        load_error = f"strategy_read_error:{exc}"

    report = build_strategy_safety_report(source, strategy_path=args.strategy, load_error=load_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
