#!/usr/bin/env python3
"""Verify the strategy never reads a ``self.X`` it does not define.

Why this exists: from d20b27d until 2026-08-19 ``adjust_trade_position`` read
``self._kill_switch_active``, an attribute assigned nowhere in the file. The
callback therefore raised ``AttributeError`` on *every* invocation. Freqtrade's
``strategy_wrapper`` swallows that into a log line and returns ``None``, which is
indistinguishable from the callback deliberately declining -- so the inventory
adjustment path was dead for weeks while the bot looked healthy.

A phantom attribute is invisible to the existing tests because none of them
import the strategy (that would need freqtrade installed), so this is a static
AST check in the same spirit as ``verify_strategy_safety.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY = Path("user_data/strategies/Market_Making.py")
DEFAULT_OUTPUT = Path("docs/strategy_attribute_report.json")
DEFAULT_CLASS = "Market_Making"

# Attributes freqtrade's IStrategy sets on the instance before any callback
# runs. Reading these is correct; they will never be assigned in our subclass.
INHERITED_ATTRIBUTES = frozenset(
    {
        "config",
        "dp",
        "wallets",
        "ft_bot_start",
        "stoploss",
        "trades_close",
        "lock_pair",
        "unlock_pair",
        "unlock_reason",
        "is_pair_locked",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def find_class(tree: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def collect_defined(cls: ast.ClassDef) -> set[str]:
    """Every name the class binds: methods, class attributes, and self.X stores."""
    defined: set[str] = set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
    for node in ast.walk(cls):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.attr)
        # setattr(self, "name", ...) binds too.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            defined.add(node.args[1].value)
    return defined


def collect_loaded(cls: ast.ClassDef) -> dict[str, int]:
    """Bare ``self.X`` reads, mapped to the first line each appears on.

    ``getattr(self, "X", default)`` is deliberately excluded: a default makes the
    read total, which is how the file already guards genuinely optional state.
    """
    guarded: set[int] = set()
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
        ):
            for arg in node.args[1:]:
                for sub in ast.walk(arg):
                    guarded.add(id(sub))
    loaded: dict[str, int] = {}
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Load)
            and id(node) not in guarded
        ):
            loaded.setdefault(node.attr, node.lineno)
    return loaded


def build_strategy_attribute_report(
    source: str | None,
    *,
    strategy_path: Path = DEFAULT_STRATEGY,
    class_name: str = DEFAULT_CLASS,
    load_error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "strategy_path": str(strategy_path),
        "class_name": class_name,
        "undefined_attributes": [],
        "ok": False,
    }
    if load_error is not None or source is None:
        report["error"] = load_error or "strategy_missing"
        return report
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report["error"] = f"syntax_error:{exc}"
        return report
    cls = find_class(tree, class_name)
    if cls is None:
        report["error"] = f"class_not_found:{class_name}"
        return report

    defined = collect_defined(cls)
    loaded = collect_loaded(cls)
    undefined = [
        {"attribute": name, "line": line}
        for name, line in sorted(loaded.items())
        if name not in defined and name not in INHERITED_ATTRIBUTES
    ]
    report["attributes_defined"] = len(defined)
    report["attributes_read"] = len(loaded)
    report["undefined_attributes"] = undefined
    report["ok"] = not undefined
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the strategy defines every self attribute it reads.")
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--class-name", default=DEFAULT_CLASS)
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
    except Exception as exc:  # pragma: no cover - defensive
        source = None
        load_error = f"strategy_read_error:{exc}"

    report = build_strategy_attribute_report(
        source, strategy_path=args.strategy, class_name=args.class_name, load_error=load_error
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
