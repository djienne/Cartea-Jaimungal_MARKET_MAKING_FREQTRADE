#!/usr/bin/env python3
"""Guarded Hyperliquid risk-reduction executor.

This is intentionally separate from the maker ALO adapter. It builds only
reduce-only IOC flatten orders for emergency inventory reduction after quoting
has been disabled and open maker orders have been cancelled.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from hyperliquid_alo_executor import (
    build_hyperliquid_cloid,
    extract_statuses,
    load_sdk_exchange,
    normalize_coin,
    order_args_for_submit,
    utc_now_iso,
)


RISK_FLATTEN_ALLOW_ENV = "HYPERLIQUID_RISK_FLATTEN_ALLOW"
DEFAULT_MAX_FLATTEN_NOTIONAL_USDC = 150.0
DEFAULT_FLATTEN_SLIPPAGE_BPS = 50.0


@dataclass(frozen=True)
class FlattenOrderIntent:
    symbol: str
    signed_position_base: float
    size: float
    price: float
    side: str
    reason: str
    slippage_bps: float = DEFAULT_FLATTEN_SLIPPAGE_BPS
    reduce_only: bool = True
    cloid: str | None = None
    client_order_id: str | None = None

    @property
    def coin(self) -> str:
        return normalize_coin(self.symbol)

    @property
    def is_buy(self) -> bool:
        if self.side == "buy":
            return True
        if self.side == "sell":
            return False
        raise ValueError(f"unsupported flatten side: {self.side}")


def flatten_order_type() -> dict[str, dict[str, str]]:
    return {"limit": {"tif": "Ioc"}}


def flatten_side_from_position(signed_position_base: float) -> str:
    position = float(signed_position_base)
    if position > 0:
        return "sell"
    if position < 0:
        return "buy"
    raise ValueError("signed_position_base must be non-zero for flattening")


def flatten_ioc_price(reference_price: float, side: str, slippage_bps: float) -> float:
    reference = float(reference_price)
    slippage = float(slippage_bps)
    if reference <= 0:
        raise ValueError("reference_price must be positive")
    if slippage < 0:
        raise ValueError("slippage_bps must be non-negative")
    if side == "buy":
        price = reference * (1.0 + slippage / 10_000.0)
    elif side == "sell":
        price = reference * (1.0 - slippage / 10_000.0)
    else:
        raise ValueError(f"unsupported flatten side: {side}")
    if price <= 0:
        raise ValueError("flatten IOC price must be positive")
    return price


def build_risk_client_order_id(
    *,
    risk_action_id: str | None,
    session_id: str | None,
    side: str,
    reason: str,
) -> str:
    return "|".join(
        [
            "risk",
            f"sess={session_id or 'unknown'}",
            f"rid={risk_action_id or 'unknown'}",
            "mode=flatten",
            f"side={side}",
            f"reason={reason or 'unspecified'}",
        ]
    )


def build_flatten_intent(
    *,
    symbol: str,
    signed_position_base: float,
    reference_price: float,
    reason: str,
    slippage_bps: float = DEFAULT_FLATTEN_SLIPPAGE_BPS,
    max_size_base: float | None = None,
    risk_action_id: str | None = None,
    session_id: str | None = None,
    client_order_id: str | None = None,
    cloid: str | None = None,
) -> FlattenOrderIntent:
    position = float(signed_position_base)
    side = flatten_side_from_position(position)
    size = abs(position)
    if max_size_base is not None and float(max_size_base) > 0:
        size = min(size, float(max_size_base))
    if size <= 0:
        raise ValueError("flatten size must be positive")
    price = flatten_ioc_price(reference_price, side, slippage_bps)
    resolved_client_order_id = client_order_id or build_risk_client_order_id(
        risk_action_id=risk_action_id,
        session_id=session_id,
        side=side,
        reason=reason,
    )
    resolved_cloid = cloid or build_hyperliquid_cloid(resolved_client_order_id)
    return FlattenOrderIntent(
        symbol=symbol,
        signed_position_base=position,
        size=float(size),
        price=float(price),
        side=side,
        reason=reason,
        slippage_bps=float(slippage_bps),
        reduce_only=True,
        cloid=resolved_cloid,
        client_order_id=resolved_client_order_id,
    )


def build_sdk_order_args(intent: FlattenOrderIntent) -> dict[str, Any]:
    if not intent.reduce_only:
        raise ValueError("risk flatten orders must be reduce-only")
    if intent.size <= 0:
        raise ValueError("size must be positive")
    if intent.price <= 0:
        raise ValueError("price must be positive")
    order_args = {
        "name": intent.coin,
        "is_buy": intent.is_buy,
        "sz": float(intent.size),
        "limit_px": float(intent.price),
        "order_type": flatten_order_type(),
        "reduce_only": True,
    }
    if intent.cloid:
        order_args["cloid"] = intent.cloid
    return order_args


def notional_limit_check(
    intent: FlattenOrderIntent,
    max_notional_usdc: float | None,
) -> tuple[bool, str, dict[str, Any]]:
    notional = abs(float(intent.size) * float(intent.price))
    payload = {
        "notional_usdc": notional,
        "max_notional_usdc": float(max_notional_usdc) if max_notional_usdc is not None else None,
    }
    if max_notional_usdc is not None and float(max_notional_usdc) > 0 and notional > float(max_notional_usdc):
        return False, "notional_limit_exceeded", payload
    return True, "ok", payload


def classify_flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    statuses = extract_statuses(result)
    reasons: list[str] = []
    saw_filled = False
    saw_cancelled = False
    saw_resting = False
    filled_total = 0.0
    error_messages: list[str] = []

    for status in statuses:
        if not isinstance(status, dict):
            continue
        if "resting" in status:
            saw_resting = True
        filled = status.get("filled")
        if isinstance(filled, dict):
            saw_filled = True
            try:
                filled_total += float(filled.get("totalSz") or filled.get("sz") or 0.0)
            except Exception:
                pass
        if "canceled" in status or "cancelled" in status:
            saw_cancelled = True
        error = status.get("error")
        if error:
            error_messages.append(str(error))

    if not statuses:
        reasons.append("missing_order_statuses")
    if saw_resting:
        reasons.append("ioc_order_resting")
    if error_messages:
        reasons.append("exchange_error_status")
    if not saw_filled and not saw_cancelled:
        reasons.append("no_fill_or_cancel_status")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "statuses_seen": len(statuses),
        "saw_filled": saw_filled,
        "saw_cancelled": saw_cancelled,
        "saw_resting": saw_resting,
        "filled_total": filled_total,
        "error_messages": error_messages,
    }


def render_plan(symbol: str) -> dict[str, Any]:
    return {
        "component": "direct_hyperliquid_risk_executor",
        "safe_default": "no network call or order submission",
        "symbol": symbol,
        "purpose": "emergency inventory reduction after maker quoting is disabled and open maker orders are cancelled",
        "sdk_method": "Exchange.order(name, is_buy, sz, limit_px, order_type, reduce_only, cloid)",
        "order_type": flatten_order_type(),
        "reduce_only": True,
        "not_maker_strategy": True,
        "submit_guards": [
            f"{RISK_FLATTEN_ALLOW_ENV}=1",
            "--acknowledge-risk-reducing-taker",
            "--reason is required",
            "--signed-position-base must be non-zero",
            "--reference-price must be positive",
            f"--max-notional-usdc defaults to {DEFAULT_MAX_FLATTEN_NOTIONAL_USDC}",
            "--testnet, or --allow-mainnet-flatten for tiny live emergency/canary use",
        ],
        "side_mapping": {
            "positive_signed_position": "sell reduce-only IOC",
            "negative_signed_position": "buy reduce-only IOC",
        },
        "client_order_id_fields": ["session_id", "risk_action_id", "side", "reason"],
        "cloid_format": "0x + 16-byte hex",
        "response_acceptance": [
            "IOC filled status is acceptable",
            "IOC canceled status is acceptable when no liquidity is available",
            "resting order status is a failure",
            "exchange error status is a failure",
        ],
        "generated_at": utc_now_iso(),
    }


def submit_flatten_order(args: argparse.Namespace) -> dict[str, Any]:
    if os.getenv(RISK_FLATTEN_ALLOW_ENV) != "1":
        raise SystemExit(f"Set {RISK_FLATTEN_ALLOW_ENV}=1 to permit direct risk flatten submit mode")
    if not args.acknowledge_risk_reducing_taker:
        raise SystemExit("--acknowledge-risk-reducing-taker is required for flatten submit mode")
    if not args.reason:
        raise SystemExit("--reason is required for flatten submit mode")
    if not args.testnet and not args.allow_mainnet_flatten:
        raise SystemExit("--testnet is required unless --allow-mainnet-flatten is explicitly supplied")

    intent = build_flatten_intent(
        symbol=args.symbol,
        signed_position_base=args.signed_position_base,
        reference_price=args.reference_price,
        reason=args.reason,
        slippage_bps=args.slippage_bps,
        max_size_base=args.max_size_base,
        risk_action_id=args.risk_action_id,
        session_id=args.session_id,
        client_order_id=args.client_order_id,
        cloid=args.cloid,
    )
    notional_ok, notional_reason, notional_payload = notional_limit_check(intent, args.max_notional_usdc)
    if not notional_ok:
        raise SystemExit(
            f"notional guard failed: {notional_reason} "
            f"({notional_payload['notional_usdc']:.8f} > {notional_payload['max_notional_usdc']:.8f} USDC)"
        )

    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args_for_submit(order_args))
    classification = classify_flatten_result(result if isinstance(result, dict) else {"raw_result": result})
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-flatten",
        "intent": asdict(intent),
        "notional_check": {"ok": notional_ok, "reason": notional_reason, **notional_payload},
        "sdk_order_args": order_args,
        "classification": classification,
        "raw_result": result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or submit guarded Hyperliquid reduce-only flatten orders.")
    parser.add_argument("--mode", choices=["plan", "build-flatten", "submit-flatten"], default="plan")
    parser.add_argument("--symbol", default="ETH/USDC:USDC")
    parser.add_argument("--signed-position-base", type=float, default=None)
    parser.add_argument("--reference-price", type=float, default=None)
    parser.add_argument("--reason", default="")
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_FLATTEN_SLIPPAGE_BPS)
    parser.add_argument("--max-size-base", type=float, default=None)
    parser.add_argument("--max-notional-usdc", type=float, default=DEFAULT_MAX_FLATTEN_NOTIONAL_USDC)
    parser.add_argument("--risk-action-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--client-order-id", default=None)
    parser.add_argument("--cloid", default=None)
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--allow-mainnet-flatten", action="store_true")
    parser.add_argument("--acknowledge-risk-reducing-taker", action="store_true")
    parser.add_argument("--private-key", default=None)
    parser.add_argument("--account-address", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def require_build_inputs(args: argparse.Namespace) -> None:
    if args.signed_position_base is None:
        raise SystemExit("--signed-position-base is required")
    if args.reference_price is None:
        raise SystemExit("--reference-price is required")
    if not args.reason:
        raise SystemExit("--reason is required")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        payload = render_plan(args.symbol)
    elif args.mode == "build-flatten":
        require_build_inputs(args)
        intent = build_flatten_intent(
            symbol=args.symbol,
            signed_position_base=args.signed_position_base,
            reference_price=args.reference_price,
            reason=args.reason,
            slippage_bps=args.slippage_bps,
            max_size_base=args.max_size_base,
            risk_action_id=args.risk_action_id,
            session_id=args.session_id,
            client_order_id=args.client_order_id,
            cloid=args.cloid,
        )
        notional_ok, notional_reason, notional_payload = notional_limit_check(intent, args.max_notional_usdc)
        payload = {
            "generated_at": utc_now_iso(),
            "mode": "build-flatten",
            "intent": asdict(intent),
            "sdk_order_args": build_sdk_order_args(intent),
            "notional_check": {"ok": notional_ok, "reason": notional_reason, **notional_payload},
        }
    else:
        require_build_inputs(args)
        payload = submit_flatten_order(args)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
