#!/usr/bin/env python3
"""Verify checked-in config keeps the PLAN.md fail-closed safety posture."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("user_data/config.json")
DEFAULT_OUTPUT = Path("docs/config_safety_report.json")


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


def bool_is_false_or_missing(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    return value is None or value is False


def load_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "config_missing"
    except Exception as exc:
        return None, f"config_invalid:{exc}"
    if not isinstance(payload, dict):
        return None, "config_not_object"
    return payload, None


def build_config_safety_report(
    config: dict[str, Any] | None,
    *,
    config_path: Path,
    # stake_amount must comfortably EXCEED one inventory unit, or
    # custom_stake_amount's proposed-stake term binds before the one-unit cap
    # and a 'one unit' fill is quietly smaller than one unit of q.
    max_stake_amount: float = 600.0,
    max_tradable_balance_ratio: float = 0.10,
    max_available_capital: float = 1000.0,
    expected_fee: float = 0.00015,
    max_custom_price_distance_ratio: float = 0.05,
    load_error: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, Any] = {
        "max_stake_amount": float(max_stake_amount),
        "max_tradable_balance_ratio": float(max_tradable_balance_ratio),
        "max_available_capital": float(max_available_capital),
        "expected_fee": float(expected_fee),
        "max_custom_price_distance_ratio": float(max_custom_price_distance_ratio),
    }
    if load_error is not None:
        reasons.append(load_error)
        config = {}

    config = config or {}
    api_server = config.get("api_server") if isinstance(config.get("api_server"), dict) else {}
    market_making = config.get("market_making") if isinstance(config.get("market_making"), dict) else {}
    order_types = config.get("order_types") if isinstance(config.get("order_types"), dict) else {}
    tif = config.get("order_time_in_force") if isinstance(config.get("order_time_in_force"), dict) else {}

    if config.get("dry_run") is not True:
        reasons.append("dry_run_not_true")

    if config.get("force_entry_enable") is not False:
        reasons.append("force_entry_enable_not_false")
    for key in ("force_entry_enable", "forcebuy_enable", "Force_entry"):
        if not bool_is_false_or_missing(api_server, key):
            reasons.append(f"api_server_{key}_not_false")

    stake_amount = config.get("stake_amount")
    stake_float = finite_float(stake_amount)
    if str(stake_amount).strip().lower() == "unlimited":
        reasons.append("stake_amount_unlimited")
    elif stake_float is None or stake_float <= 0:
        reasons.append("stake_amount_invalid")
    elif stake_float > float(max_stake_amount):
        reasons.append(f"stake_amount_above_limit:{stake_float:g}>max_{float(max_stake_amount):g}")

    # Capital is bounded by EITHER available_capital or tradable_balance_ratio.
    # Freqtrade documents them as incompatible -- setting available_capital
    # "will replace any configuration of tradable_balance_ratio" -- so requiring
    # the ratio once available_capital is set would be demanding a value the bot
    # silently ignores. Exactly one of them must bound the capital.
    tradable_ratio = finite_float(config.get("tradable_balance_ratio"))
    available_capital = finite_float(config.get("available_capital"))
    if available_capital is not None and available_capital > 0:
        if tradable_ratio is not None:
            # Not fatal, but it reads as a live limit and is not one.
            reasons.append("tradable_balance_ratio_ignored_because_available_capital_set")
        if available_capital > float(max_available_capital):
            reasons.append(
                f"available_capital_above_limit:{available_capital:g}>max_{float(max_available_capital):g}"
            )
    elif tradable_ratio is None or tradable_ratio <= 0:
        reasons.append("capital_bound_missing")
    elif tradable_ratio > float(max_tradable_balance_ratio):
        reasons.append(
            f"tradable_balance_ratio_above_limit:{tradable_ratio:g}>max_{float(max_tradable_balance_ratio):g}"
        )

    fee = finite_float(config.get("fee"))
    effective_expected_fee = float(expected_fee)
    if "maker_fee_rate" in market_making:
        # Mirror the strategy's runtime gate: market_making.maker_fee_rate is the
        # maker fee the quotes assume (config override for what the exchange/ccxt
        # actually reports), so the freqtrade accounting fee must match it.
        # Bound it so a zero/absurd fee cannot be smuggled in via the override.
        maker_fee_rate = finite_float(market_making.get("maker_fee_rate"))
        if maker_fee_rate is None or maker_fee_rate <= 0 or maker_fee_rate > 0.001:
            reasons.append("maker_fee_rate_invalid")
        else:
            effective_expected_fee = maker_fee_rate
    checks["expected_fee"] = effective_expected_fee
    if fee is None:
        reasons.append("fee_missing")
    elif abs(fee - effective_expected_fee) > 1e-12:
        reasons.append(f"fee_mismatch:{fee:g}!=expected_{effective_expected_fee:g}")

    custom_price_distance = finite_float(config.get("custom_price_max_distance_ratio"))
    if custom_price_distance is None or custom_price_distance <= 0:
        reasons.append("custom_price_max_distance_ratio_invalid")
    elif custom_price_distance > float(max_custom_price_distance_ratio):
        reasons.append(
            "custom_price_max_distance_ratio_above_limit:"
            f"{custom_price_distance:g}>max_{float(max_custom_price_distance_ratio):g}"
        )

    max_open_trades = finite_float(config.get("max_open_trades"))
    if max_open_trades is None or max_open_trades > 1:
        reasons.append("max_open_trades_above_one")

    if str(order_types.get("entry") or "").strip().lower() != "limit":
        reasons.append("entry_order_type_not_limit")
    if str(order_types.get("exit") or "").strip().lower() != "limit":
        reasons.append("exit_order_type_not_limit")

    if str(tif.get("entry") or "").strip().lower() not in {"gtc", "po", "alo"}:
        reasons.append("entry_time_in_force_unsupported")
    if str(tif.get("exit") or "").strip().lower() not in {"gtc", "po", "alo"}:
        reasons.append("exit_time_in_force_unsupported")

    # trading_enabled=true is the dry-run operating mode (quotes in paper
    # trading; the strategy fail-closes live use behind post_only_verified and
    # deployment-stage evidence). Only flag it as live enablement when the same
    # config also drops dry_run.
    if market_making.get("trading_enabled") is True and config.get("dry_run") is not True:
        reasons.append("checked_in_trading_enabled_without_dry_run")
    if market_making.get("post_only_verified") is True:
        reasons.append("checked_in_post_only_verified_true")
    if str(market_making.get("deployment_stage") or "research").strip().lower() != "research":
        reasons.append("checked_in_deployment_stage_not_research")
    if market_making.get("run_estimators_in_strategy") is True:
        reasons.append("checked_in_internal_param_estimator_true")

    return {
        "generated_at": utc_now_iso(),
        "ok": not reasons,
        "reasons": reasons,
        "config_path": str(config_path),
        "checks": checks,
        "observed": {
            "dry_run": config.get("dry_run"),
            "force_entry_enable": config.get("force_entry_enable"),
            "api_force_entry_enable": api_server.get("force_entry_enable"),
            "api_forcebuy_enable": api_server.get("forcebuy_enable"),
            "stake_amount": config.get("stake_amount"),
            "tradable_balance_ratio": config.get("tradable_balance_ratio"),
            "available_capital": config.get("available_capital"),
            "fee": config.get("fee"),
            "custom_price_max_distance_ratio": config.get("custom_price_max_distance_ratio"),
            "max_open_trades": config.get("max_open_trades"),
            "order_types": {
                "entry": order_types.get("entry"),
                "exit": order_types.get("exit"),
            },
            "order_time_in_force": {
                "entry": tif.get("entry"),
                "exit": tif.get("exit"),
            },
            "market_making": {
                "trading_enabled": market_making.get("trading_enabled"),
                "maker_fee_rate": market_making.get("maker_fee_rate"),
                "post_only_verified": market_making.get("post_only_verified"),
                "deployment_stage": market_making.get("deployment_stage", "research"),
                "run_estimators_in_strategy": market_making.get("run_estimators_in_strategy"),
                "param_snapshot_reload_interval_seconds": market_making.get("param_snapshot_reload_interval_seconds"),
                "param_update_lock_stale_seconds": market_making.get("param_update_lock_stale_seconds"),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checked-in fail-closed config safety defaults.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-stake-amount", type=float, default=600.0)
    parser.add_argument("--max-available-capital", type=float, default=1000.0)
    parser.add_argument("--max-tradable-balance-ratio", type=float, default=0.10)
    parser.add_argument("--expected-fee", type=float, default=0.00015)
    parser.add_argument("--max-custom-price-distance-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, load_error = load_config(args.config)
    report = build_config_safety_report(
        config,
        config_path=args.config,
        max_stake_amount=float(args.max_stake_amount),
        max_available_capital=float(args.max_available_capital),
        max_tradable_balance_ratio=float(args.max_tradable_balance_ratio),
        expected_fee=float(args.expected_fee),
        max_custom_price_distance_ratio=float(args.max_custom_price_distance_ratio),
        load_error=load_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
