#!/usr/bin/env python3
"""Guarded Hyperliquid post-only verification probe and evidence checker.

Default mode is documentation-only and performs no network calls. The submit
mode is intentionally hard to invoke because it can place real exchange orders
depending on the supplied credentials and endpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any


def alo_order_params() -> dict[str, Any]:
    """Return the native Hyperliquid add-liquidity-only order parameters."""
    return {
        "timeInForce": "Alo",
        "postOnly": True,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def render_plan(symbol: str) -> str:
    payload = {
        "gate": "hyperliquid_post_only_mapping",
        "symbol": symbol,
        "native_time_in_force": "Alo",
        "expected_crossing_order_result": "reject_or_cancel_without_fill",
        "expected_passive_order_result": "resting_maker_order",
        "required_evidence": [
            "submitted params contain timeInForce=Alo",
            "actual exchange/order time-in-force is confirmed as Alo",
            "crossing ALO order has zero filled amount",
            "crossing ALO order rejects/cancels/expires rather than rests",
            "passive ALO order rests or fills as maker only",
            "exchange fill/liquidity flag is maker for any fill",
            "actual order status, filled amount, and raw exchange result are retained",
            "crossing/passive submit artifacts include fresh generated_at timestamps",
        ],
        "safe_default": "no network call or order submission",
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def find_nested_value(payload: Any, keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in keys:
                return value
        for value in payload.values():
            found = find_nested_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_nested_value(value, keys)
            if found is not None:
                return found
    return None


def filled_amount(payload: dict[str, Any]) -> float:
    classification = payload.get("classification")
    if isinstance(classification, dict) and classification.get("filled_total") is not None:
        try:
            return float(classification["filled_total"])
        except Exception:
            pass
    for key in ("filled", "filledAmount", "filledSz", "totalSz"):
        value = payload.get(key)
        if value is None and isinstance(payload.get("raw_result"), dict):
            value = payload["raw_result"].get(key)
        if value is None:
            value = find_nested_value(payload, {key.lower()})
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def order_status(payload: dict[str, Any]) -> str:
    classification = payload.get("classification")
    if isinstance(classification, dict):
        if classification.get("alo_rejected"):
            return "rejected"
        if classification.get("saw_resting"):
            return "open"
        if classification.get("saw_filled"):
            return "filled"
    for key in ("order_status", "status"):
        value = payload.get(key)
        if value is None and isinstance(payload.get("raw_result"), dict):
            value = payload["raw_result"].get(key)
        if value is not None:
            return str(value).lower()
    value = find_nested_value(payload, {"status"})
    return str(value).lower() if value is not None else "unknown"


def liquidity_flag(payload: dict[str, Any]) -> str | None:
    classification = payload.get("classification")
    if isinstance(classification, dict):
        reasons = {str(reason) for reason in classification.get("reasons", [])}
        if "taker_liquidity_seen" in reasons:
            return "taker"
        if classification.get("saw_filled") and classification.get("ok"):
            return "maker"
    value = find_nested_value(payload, {"liquidity", "liquiditytype", "maker"})
    if value is None:
        return None
    if isinstance(value, bool):
        return "maker" if value else "taker"
    return str(value).lower()


def submitted_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("submitted_params")
    if isinstance(params, dict):
        return params
    params = payload.get("params")
    if isinstance(params, dict):
        return params
    sdk_args = payload.get("sdk_order_args")
    if isinstance(sdk_args, dict):
        order_type = sdk_args.get("order_type")
        if isinstance(order_type, dict):
            limit_type = order_type.get("limit")
            if isinstance(limit_type, dict):
                tif = limit_type.get("tif") or limit_type.get("timeInForce")
                if tif:
                    return {
                        "timeInForce": tif,
                        "postOnly": str(tif).lower() == "alo",
                        "source": "hyperliquid_sdk_order_type",
                    }
    return {}


def actual_time_in_force(payload: dict[str, Any]) -> str | None:
    for key in ("actual_time_in_force", "actual_tif", "timeInForce", "time_in_force", "tif"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    raw_result = payload.get("raw_result")
    if isinstance(raw_result, dict):
        value = find_nested_value(raw_result, {"actual_time_in_force", "actual_tif", "timeinforce", "time_in_force", "tif"})
        if value is not None:
            return str(value)
    classification = payload.get("classification")
    if isinstance(classification, dict):
        value = find_nested_value(classification, {"actual_time_in_force", "actual_tif", "timeinforce", "time_in_force", "tif"})
        if value is not None:
            return str(value)
    return None


def has_actual_alo_tif(payload: dict[str, Any]) -> bool:
    actual = actual_time_in_force(payload)
    return str(actual or "").lower() == "alo"


def has_alo_params(payload: dict[str, Any]) -> bool:
    params = submitted_params(payload)
    tif = str(params.get("timeInForce") or params.get("time_in_force") or "").lower()
    post_only = params.get("postOnly")
    return tif == "alo" and bool(post_only) is True


def evaluate_crossing_result(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not has_alo_params(payload):
        reasons.append("crossing_missing_alo_params")
    if not has_actual_alo_tif(payload):
        reasons.append("crossing_actual_tif_not_alo")
    filled = filled_amount(payload)
    if abs(filled) > 1e-12:
        reasons.append(f"crossing_filled_nonzero:{filled}")
    status = order_status(payload)
    if status not in {"canceled", "cancelled", "rejected", "expired"}:
        reasons.append(f"crossing_status_not_rejected_or_cancelled:{status}")
    liquidity = liquidity_flag(payload)
    if liquidity == "taker":
        reasons.append("crossing_liquidity_taker")
    return not reasons, reasons


def evaluate_passive_result(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not has_alo_params(payload):
        reasons.append("passive_missing_alo_params")
    if not has_actual_alo_tif(payload):
        reasons.append("passive_actual_tif_not_alo")
    filled = filled_amount(payload)
    status = order_status(payload)
    if status not in {"open", "new", "canceled", "cancelled", "closed", "filled", "partially_filled", "expired"}:
        reasons.append(f"passive_status_unexpected:{status}")
    liquidity = liquidity_flag(payload)
    if liquidity == "taker":
        reasons.append("passive_liquidity_taker")
    if filled > 0 and liquidity not in {"maker", "add", "a", "true"}:
        reasons.append("passive_fill_without_maker_liquidity_flag")
    if filled <= 0 and status not in {"open", "new"}:
        reasons.append(f"passive_not_resting_or_maker_fill:{status}")
    return not reasons, reasons


def evidence_age_status(
    label: str,
    payload: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    max_evidence_age_seconds: float = 86_400.0,
) -> tuple[bool, list[str], dict[str, Any]]:
    if payload is None:
        return False, [], {"age_seconds": None, "generated_at": None, "max_age_seconds": float(max_evidence_age_seconds)}
    generated_at = payload.get("generated_at")
    generated_at_dt = parse_utc_timestamp(generated_at)
    meta = {
        "generated_at": generated_at,
        "age_seconds": None,
        "max_age_seconds": float(max_evidence_age_seconds),
    }
    if generated_at_dt is None:
        return False, [f"{label}_missing_generated_at"], meta
    reference = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    age_seconds = max(0.0, (reference - generated_at_dt).total_seconds())
    meta["age_seconds"] = age_seconds
    if max_evidence_age_seconds > 0 and age_seconds > float(max_evidence_age_seconds):
        return False, [f"{label}_evidence_stale:{age_seconds:.1f}s>max_{float(max_evidence_age_seconds):.1f}s"], meta
    return True, [], meta


def evaluate_evidence(
    crossing_payload: dict[str, Any] | None,
    passive_payload: dict[str, Any] | None,
    *,
    max_evidence_age_seconds: float = 86_400.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    crossing_ok = False
    passive_ok = False
    crossing_age_ok = False
    passive_age_ok = False
    crossing_age_meta: dict[str, Any] = {"age_seconds": None, "generated_at": None, "max_age_seconds": float(max_evidence_age_seconds)}
    passive_age_meta: dict[str, Any] = {"age_seconds": None, "generated_at": None, "max_age_seconds": float(max_evidence_age_seconds)}
    if crossing_payload is None:
        reasons.append("missing_crossing_result")
    else:
        crossing_ok, crossing_reasons = evaluate_crossing_result(crossing_payload)
        reasons.extend(crossing_reasons)
        crossing_age_ok, crossing_age_reasons, crossing_age_meta = evidence_age_status(
            "crossing",
            crossing_payload,
            now=now,
            max_evidence_age_seconds=max_evidence_age_seconds,
        )
        reasons.extend(crossing_age_reasons)
    if passive_payload is None:
        reasons.append("missing_passive_result")
    else:
        passive_ok, passive_reasons = evaluate_passive_result(passive_payload)
        reasons.extend(passive_reasons)
        passive_age_ok, passive_age_reasons, passive_age_meta = evidence_age_status(
            "passive",
            passive_payload,
            now=now,
            max_evidence_age_seconds=max_evidence_age_seconds,
        )
        reasons.extend(passive_age_reasons)
    return {
        "generated_at": utc_now_iso(),
        "gate": "hyperliquid_post_only_mapping",
        "ok": crossing_ok and passive_ok and crossing_age_ok and passive_age_ok and not reasons,
        "reasons": reasons,
        "max_evidence_age_seconds": float(max_evidence_age_seconds),
        "crossing": {
            "present": crossing_payload is not None,
            "ok": crossing_ok,
            "filled": filled_amount(crossing_payload) if crossing_payload else None,
            "status": order_status(crossing_payload) if crossing_payload else None,
            "liquidity": liquidity_flag(crossing_payload) if crossing_payload else None,
            "has_alo_params": has_alo_params(crossing_payload) if crossing_payload else False,
            "actual_time_in_force": actual_time_in_force(crossing_payload) if crossing_payload else None,
            "has_actual_alo_tif": has_actual_alo_tif(crossing_payload) if crossing_payload else False,
            "age_ok": crossing_age_ok,
            **crossing_age_meta,
        },
        "passive": {
            "present": passive_payload is not None,
            "ok": passive_ok,
            "filled": filled_amount(passive_payload) if passive_payload else None,
            "status": order_status(passive_payload) if passive_payload else None,
            "liquidity": liquidity_flag(passive_payload) if passive_payload else None,
            "has_alo_params": has_alo_params(passive_payload) if passive_payload else False,
            "actual_time_in_force": actual_time_in_force(passive_payload) if passive_payload else None,
            "has_actual_alo_tif": has_actual_alo_tif(passive_payload) if passive_payload else False,
            "age_ok": passive_age_ok,
            **passive_age_meta,
        },
    }


def load_ccxt_exchange(args: argparse.Namespace):
    try:
        import ccxt  # type: ignore
    except Exception as exc:
        raise SystemExit(f"ccxt is required for --mode ccxt-check/submit-crossing-alo: {exc}")

    api_key = args.api_key or os.getenv("HYPERLIQUID_API_KEY")
    secret = args.secret or os.getenv("HYPERLIQUID_SECRET")
    wallet = args.wallet or os.getenv("HYPERLIQUID_WALLET_ADDRESS")
    config: dict[str, Any] = {"enableRateLimit": True}
    if api_key:
        config["apiKey"] = api_key
    if secret:
        config["secret"] = secret
    if wallet:
        config.setdefault("walletAddress", wallet)
    exchange = ccxt.hyperliquid(config)
    if args.sandbox and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
    return exchange


def ccxt_check(args: argparse.Namespace) -> dict[str, Any]:
    exchange = load_ccxt_exchange(args)
    markets = exchange.load_markets()
    market = markets.get(args.symbol)
    if market is None:
        raise SystemExit(f"symbol not found after load_markets: {args.symbol}")
    return {
        "exchange": getattr(exchange, "id", "hyperliquid"),
        "symbol": args.symbol,
        "market_precision": market.get("precision"),
        "market_limits": market.get("limits"),
        "params_to_submit": alo_order_params(),
        "note": "This check does not prove orders are post-only; submit mode or exchange logs are required.",
    }


def submit_crossing_alo(args: argparse.Namespace) -> dict[str, Any]:
    if not args.acknowledge_real_orders:
        raise SystemExit("--acknowledge-real-orders is required for submit mode")
    if os.getenv("HYPERLIQUID_POST_ONLY_PROBE_ALLOW") != "1":
        raise SystemExit("Set HYPERLIQUID_POST_ONLY_PROBE_ALLOW=1 to permit submit mode")
    if args.amount <= 0:
        raise SystemExit("--amount must be positive")

    exchange = load_ccxt_exchange(args)
    exchange.load_markets()
    orderbook = exchange.fetch_order_book(args.symbol, limit=1)
    if not orderbook.get("asks"):
        raise SystemExit("cannot submit crossing bid without a best ask")
    crossing_bid = float(orderbook["asks"][0][0])
    params = alo_order_params()
    result = exchange.create_order(args.symbol, "limit", "buy", args.amount, crossing_bid, params)
    filled = float(result.get("filled") or 0.0)
    status = str(result.get("status") or "unknown")
    order_id = result.get("id")
    if order_id and status in {"open", "new"}:
        try:
            exchange.cancel_order(order_id, args.symbol)
        except Exception:
            pass
    payload = {
        "generated_at": utc_now_iso(),
        "probe": "crossing_alo",
        "symbol": args.symbol,
        "submitted_side": "buy",
        "submitted_price": crossing_bid,
        "submitted_amount": args.amount,
        "submitted_params": params,
        "actual_time_in_force": actual_time_in_force({"raw_result": result}),
        "order_status": status,
        "filled": filled,
        "raw_result": result,
    }
    ok, reasons = evaluate_crossing_result(payload)
    payload["passed"] = ok
    payload["reasons"] = reasons
    return payload


def submit_passive_alo(args: argparse.Namespace) -> dict[str, Any]:
    if not args.acknowledge_real_orders:
        raise SystemExit("--acknowledge-real-orders is required for submit mode")
    if os.getenv("HYPERLIQUID_POST_ONLY_PROBE_ALLOW") != "1":
        raise SystemExit("Set HYPERLIQUID_POST_ONLY_PROBE_ALLOW=1 to permit submit mode")
    if args.amount <= 0:
        raise SystemExit("--amount must be positive")

    exchange = load_ccxt_exchange(args)
    exchange.load_markets()
    orderbook = exchange.fetch_order_book(args.symbol, limit=1)
    if not orderbook.get("bids"):
        raise SystemExit("cannot submit passive bid without a best bid")
    passive_bid = float(orderbook["bids"][0][0])
    params = alo_order_params()
    result = exchange.create_order(args.symbol, "limit", "buy", args.amount, passive_bid, params)
    filled = float(result.get("filled") or 0.0)
    status = str(result.get("status") or "unknown").lower()
    order_id = result.get("id")
    cancel_result = None
    if order_id and status in {"open", "new", "unknown"}:
        try:
            cancel_result = exchange.cancel_order(order_id, args.symbol)
            status = str((cancel_result or {}).get("status") or status).lower()
        except Exception as exc:
            cancel_result = {"cancel_error": str(exc)}
    payload = {
        "generated_at": utc_now_iso(),
        "probe": "passive_alo",
        "symbol": args.symbol,
        "submitted_side": "buy",
        "submitted_price": passive_bid,
        "submitted_amount": args.amount,
        "submitted_params": params,
        "actual_time_in_force": actual_time_in_force({"raw_result": result}),
        "order_status": status,
        "filled": filled,
        "cancel_result": cancel_result,
        "raw_result": result,
    }
    ok, reasons = evaluate_passive_result(payload)
    payload["passed"] = ok
    payload["reasons"] = reasons
    return payload


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Hyperliquid Alo/post-only behavior with hard safety gates.")
    parser.add_argument(
        "--mode",
        choices=["plan", "ccxt-check", "submit-crossing-alo", "submit-passive-alo", "evaluate-evidence"],
        default="plan",
    )
    parser.add_argument("--symbol", default="ETH/USDC:USDC")
    parser.add_argument("--sandbox", action="store_true", help="Ask ccxt to use sandbox mode if supported.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--secret", default=None)
    parser.add_argument("--wallet", default=None)
    parser.add_argument("--amount", type=float, default=0.0)
    parser.add_argument("--acknowledge-real-orders", action="store_true")
    parser.add_argument("--crossing-result", type=Path, default=None)
    parser.add_argument("--passive-result", type=Path, default=None)
    parser.add_argument("--max-evidence-age-seconds", type=float, default=86_400.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "plan":
        text = render_plan(args.symbol)
    elif args.mode == "ccxt-check":
        text = json.dumps(ccxt_check(args), indent=2, sort_keys=True, default=str)
    elif args.mode == "submit-passive-alo":
        text = json.dumps(submit_passive_alo(args), indent=2, sort_keys=True, default=str)
    elif args.mode == "evaluate-evidence":
        report = evaluate_evidence(
            load_optional_json(args.crossing_result),
            load_optional_json(args.passive_result),
            max_evidence_age_seconds=float(args.max_evidence_age_seconds),
        )
        text = json.dumps(report, indent=2, sort_keys=True, default=str)
    else:
        text = json.dumps(submit_crossing_alo(args), indent=2, sort_keys=True, default=str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.mode == "evaluate-evidence":
        return 0 if json.loads(text)["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
