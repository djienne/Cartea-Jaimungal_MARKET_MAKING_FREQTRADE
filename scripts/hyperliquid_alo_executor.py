#!/usr/bin/env python3
"""Direct Hyperliquid ALO execution adapter.

This is the fallback path from PLAN.md for live maker execution if Freqtrade
cannot prove a safe PO -> Hyperliquid Alo mapping. The module is no-network by
default. Submit mode requires explicit environment and CLI acknowledgements.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


DIRECT_ALO_ALLOW_ENV = "HYPERLIQUID_DIRECT_ALO_ALLOW"


@dataclass(frozen=True)
class AloOrderIntent:
    symbol: str
    side: str
    size: float
    price: float
    reduce_only: bool = False

    @property
    def coin(self) -> str:
        return normalize_coin(self.symbol)

    @property
    def is_buy(self) -> bool:
        side = self.side.lower()
        if side in {"buy", "bid", "long"}:
            return True
        if side in {"sell", "ask", "short"}:
            return False
        raise ValueError(f"unsupported side: {self.side}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_coin(symbol: str) -> str:
    return str(symbol).split("/", 1)[0].split(":", 1)[0]


def alo_order_type() -> dict[str, dict[str, str]]:
    return {"limit": {"tif": "Alo"}}


def build_sdk_order_args(intent: AloOrderIntent) -> dict[str, Any]:
    if intent.size <= 0:
        raise ValueError("size must be positive")
    if intent.price <= 0:
        raise ValueError("price must be positive")
    return {
        "name": intent.coin,
        "is_buy": intent.is_buy,
        "sz": float(intent.size),
        "limit_px": float(intent.price),
        "order_type": alo_order_type(),
        "reduce_only": bool(intent.reduce_only),
    }


def maker_safe(intent: AloOrderIntent, best_bid: float, best_ask: float) -> tuple[bool, str]:
    try:
        price = float(intent.price)
        best_bid = float(best_bid)
        best_ask = float(best_ask)
    except Exception:
        return False, "invalid_book_or_price"
    if price <= 0 or best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        return False, "crossed_or_invalid_book"
    if intent.is_buy and price >= best_ask:
        return False, "bid_crosses_ask"
    if not intent.is_buy and price <= best_bid:
        return False, "ask_crosses_bid"
    return True, "ok"


def crossing_probe_intent(symbol: str, side: str, size: float, best_bid: float, best_ask: float) -> AloOrderIntent:
    """Build an intentionally crossing ALO probe intent from an observed BBO."""
    if size <= 0:
        raise ValueError("size must be positive")
    try:
        best_bid = float(best_bid)
        best_ask = float(best_ask)
    except Exception as exc:
        raise ValueError("best_bid and best_ask must be numeric") from exc
    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        raise ValueError("best_bid/best_ask must describe a valid uncrossed book")
    side_l = side.lower()
    if side_l in {"buy", "bid", "long"}:
        return AloOrderIntent(symbol=symbol, side=side, size=float(size), price=best_ask)
    if side_l in {"sell", "ask", "short"}:
        return AloOrderIntent(symbol=symbol, side=side, size=float(size), price=best_bid)
    raise ValueError(f"unsupported side: {side}")


def crossing_probe_check(intent: AloOrderIntent, best_bid: float, best_ask: float) -> tuple[bool, str]:
    try:
        price = float(intent.price)
        best_bid = float(best_bid)
        best_ask = float(best_ask)
    except Exception:
        return False, "invalid_book_or_price"
    if price <= 0 or best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        return False, "crossed_or_invalid_book"
    if intent.is_buy and price >= best_ask:
        return True, "bid_crosses_ask_for_alo_probe"
    if not intent.is_buy and price <= best_bid:
        return True, "ask_crosses_bid_for_alo_probe"
    return False, "probe_does_not_cross"


def nested_values(payload: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.append(value)
            values.extend(nested_values(value))
    elif isinstance(payload, list):
        for value in payload:
            values.append(value)
            values.extend(nested_values(value))
    return values


def extract_statuses(result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        result.get("statuses"),
        (((result.get("response") or {}).get("data") or {}).get("statuses") if isinstance(result.get("response"), dict) else None),
        (((result.get("data") or {}).get("statuses")) if isinstance(result.get("data"), dict) else None),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [status for status in candidate if isinstance(status, dict)]
    return []


def has_taker_liquidity(payload: Any) -> bool:
    for value in nested_values(payload):
        if isinstance(value, str) and value.lower() == "taker":
            return True
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in {"liquidity", "liquiditytype"} and str(nested).lower() == "taker":
                    return True
    return False


def has_maker_liquidity(payload: Any) -> bool:
    for value in nested_values(payload):
        if isinstance(value, str) and value.lower() == "maker":
            return True
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in {"liquidity", "liquiditytype"} and str(nested).lower() == "maker":
                    return True
    return False


def classify_order_result(result: dict[str, Any]) -> dict[str, Any]:
    statuses = extract_statuses(result)
    reasons: list[str] = []
    resting_oids: list[Any] = []
    filled_total = 0.0
    error_messages: list[str] = []
    saw_filled = False
    saw_resting = False

    for status in statuses:
        if "resting" in status:
            saw_resting = True
            resting = status.get("resting") or {}
            if isinstance(resting, dict) and "oid" in resting:
                resting_oids.append(resting["oid"])
        if "filled" in status:
            saw_filled = True
            filled = status.get("filled") or {}
            if isinstance(filled, dict):
                for key in ("totalSz", "sz", "filledSz"):
                    if key in filled:
                        try:
                            filled_total += float(filled[key])
                            break
                        except Exception:
                            continue
        if "error" in status:
            error_messages.append(str(status.get("error")))

    if str(result.get("status", "")).lower() not in {"", "ok"}:
        reasons.append(f"top_level_status_not_ok:{result.get('status')}")
    if has_taker_liquidity(result):
        reasons.append("taker_liquidity_seen")
    if saw_filled and not has_maker_liquidity(result):
        reasons.append("filled_without_maker_liquidity_flag")

    alo_rejected = any("post" in msg.lower() or "alo" in msg.lower() or "would immediately match" in msg.lower() for msg in error_messages)
    ok = (saw_resting or alo_rejected or (saw_filled and has_maker_liquidity(result))) and not reasons
    if not statuses:
        ok = False
        reasons.append("missing_order_statuses")
    return {
        "ok": ok,
        "reasons": reasons,
        "statuses_seen": len(statuses),
        "saw_resting": saw_resting,
        "resting_oids": resting_oids,
        "saw_filled": saw_filled,
        "filled_total": filled_total,
        "alo_rejected": alo_rejected,
        "error_messages": error_messages,
    }


def cancel_resting_orders(exchange: Any, coin: str, resting_oids: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for oid in resting_oids:
        try:
            cancel_result = exchange.cancel(coin, int(oid))
            results.append({"oid": oid, "ok": True, "raw_result": cancel_result})
        except Exception as exc:
            results.append({"oid": oid, "ok": False, "cancel_error": str(exc)})
    return results


def load_sdk_exchange(args: argparse.Namespace):
    try:
        import eth_account  # type: ignore
        from hyperliquid.exchange import Exchange  # type: ignore
        from hyperliquid.utils import constants  # type: ignore
    except Exception as exc:
        raise SystemExit(f"hyperliquid-python-sdk and eth-account are required for submit mode: {exc}")

    private_key = args.private_key or os.getenv("HYPERLIQUID_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("HYPERLIQUID_PRIVATE_KEY or --private-key is required for submit mode")
    base_url = constants.TESTNET_API_URL if args.testnet else constants.MAINNET_API_URL
    wallet = eth_account.Account.from_key(private_key)
    return Exchange(wallet, base_url, account_address=args.account_address or os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS"))


def submit_alo_order(args: argparse.Namespace) -> dict[str, Any]:
    if os.getenv(DIRECT_ALO_ALLOW_ENV) != "1":
        raise SystemExit(f"Set {DIRECT_ALO_ALLOW_ENV}=1 to permit direct ALO submit mode")
    if not args.acknowledge_real_orders:
        raise SystemExit("--acknowledge-real-orders is required for submit mode")
    if args.best_bid is None or args.best_ask is None:
        raise SystemExit("--best-bid and --best-ask are required so local maker-safety is explicit")

    intent = AloOrderIntent(
        symbol=args.symbol,
        side=args.side,
        size=float(args.size),
        price=float(args.price),
        reduce_only=bool(args.reduce_only),
    )
    ok, reason = maker_safe(intent, float(args.best_bid), float(args.best_ask))
    if not ok:
        raise SystemExit(f"local maker-safety failed: {reason}")

    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args)
    classification = classify_order_result(result if isinstance(result, dict) else {"raw_result": result})
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-alo",
        "intent": asdict(intent),
        "local_maker_check": {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask},
        "sdk_order_args": order_args,
        "classification": classification,
        "raw_result": result,
    }


def submit_crossing_alo_probe(args: argparse.Namespace) -> dict[str, Any]:
    if os.getenv(DIRECT_ALO_ALLOW_ENV) != "1":
        raise SystemExit(f"Set {DIRECT_ALO_ALLOW_ENV}=1 to permit direct ALO submit mode")
    if not args.acknowledge_real_orders:
        raise SystemExit("--acknowledge-real-orders is required for submit mode")
    if not args.allow_crossing_probe:
        raise SystemExit("--allow-crossing-probe is required because this intentionally submits a crossing ALO order")
    if not args.testnet and not args.allow_mainnet_crossing_probe:
        raise SystemExit("--allow-mainnet-crossing-probe is required for non-testnet crossing probes")
    if args.best_bid is None or args.best_ask is None:
        raise SystemExit("--best-bid and --best-ask are required so crossing probe evidence is explicit")

    try:
        intent = crossing_probe_intent(
            symbol=args.symbol,
            side=args.side,
            size=float(args.size),
            best_bid=float(args.best_bid),
            best_ask=float(args.best_ask),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ok, reason = crossing_probe_check(intent, float(args.best_bid), float(args.best_ask))
    if not ok:
        raise SystemExit(f"crossing probe check failed: {reason}")

    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args)
    classification = classify_order_result(result if isinstance(result, dict) else {"raw_result": result})
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-crossing-alo",
        "intent": asdict(intent),
        "crossing_probe_check": {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask},
        "sdk_order_args": order_args,
        "classification": classification,
        "raw_result": result,
    }


def submit_passive_alo_probe(args: argparse.Namespace) -> dict[str, Any]:
    if os.getenv(DIRECT_ALO_ALLOW_ENV) != "1":
        raise SystemExit(f"Set {DIRECT_ALO_ALLOW_ENV}=1 to permit direct ALO submit mode")
    if not args.acknowledge_real_orders:
        raise SystemExit("--acknowledge-real-orders is required for submit mode")
    if not args.allow_passive_probe:
        raise SystemExit("--allow-passive-probe is required for passive evidence probes")
    if not args.testnet and not args.allow_mainnet_passive_probe:
        raise SystemExit("--allow-mainnet-passive-probe is required for non-testnet passive probes")
    if args.best_bid is None or args.best_ask is None:
        raise SystemExit("--best-bid and --best-ask are required so local maker-safety is explicit")

    intent = AloOrderIntent(
        symbol=args.symbol,
        side=args.side,
        size=float(args.size),
        price=float(args.price),
        reduce_only=bool(args.reduce_only),
    )
    ok, reason = maker_safe(intent, float(args.best_bid), float(args.best_ask))
    if not ok:
        raise SystemExit(f"local maker-safety failed: {reason}")

    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args)
    classification = classify_order_result(result if isinstance(result, dict) else {"raw_result": result})
    cancel_results = cancel_resting_orders(exchange, intent.coin, classification.get("resting_oids", []))
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-passive-alo",
        "intent": asdict(intent),
        "local_maker_check": {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask},
        "sdk_order_args": order_args,
        "classification": classification,
        "cancel_results": cancel_results,
        "raw_result": result,
    }


def render_plan(symbol: str) -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "component": "direct_hyperliquid_alo_executor",
        "safe_default": "no network call or order submission",
        "symbol": symbol,
        "sdk_method": "Exchange.order(name, is_buy, sz, limit_px, order_type, reduce_only)",
        "order_type": alo_order_type(),
        "submit_guards": [
            f"{DIRECT_ALO_ALLOW_ENV}=1",
            "--acknowledge-real-orders",
            "--best-bid and --best-ask local maker-safety inputs",
            "bid price must be below best ask",
            "ask price must be above best bid",
        ],
        "crossing_probe_guards": [
            f"{DIRECT_ALO_ALLOW_ENV}=1",
            "--acknowledge-real-orders",
            "--allow-crossing-probe",
            "--testnet, or --allow-mainnet-crossing-probe for tiny mainnet evidence",
            "--best-bid and --best-ask crossing evidence inputs",
            "bid probe price is best ask; ask probe price is best bid",
        ],
        "passive_probe_guards": [
            f"{DIRECT_ALO_ALLOW_ENV}=1",
            "--acknowledge-real-orders",
            "--allow-passive-probe",
            "--testnet, or --allow-mainnet-passive-probe for tiny mainnet evidence",
            "--best-bid and --best-ask local maker-safety inputs",
            "resting order ids are canceled after evidence capture",
        ],
        "response_acceptance": [
            "resting order status is acceptable",
            "ALO post-only rejection/cancel without fill is acceptable",
            "fills require explicit maker liquidity evidence",
            "any taker liquidity fails",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded direct Hyperliquid SDK Alo executor.")
    parser.add_argument(
        "--mode",
        choices=["plan", "build-order", "classify-result", "submit-alo", "submit-crossing-alo", "submit-passive-alo"],
        default="plan",
    )
    parser.add_argument("--symbol", default="ETH/USDC:USDC")
    parser.add_argument("--side", default="bid", choices=["bid", "ask", "buy", "sell", "long", "short"])
    parser.add_argument("--size", type=float, default=0.0)
    parser.add_argument("--price", type=float, default=0.0)
    parser.add_argument("--reduce-only", action="store_true")
    parser.add_argument("--best-bid", type=float, default=None)
    parser.add_argument("--best-ask", type=float, default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--private-key", default=None)
    parser.add_argument("--account-address", default=None)
    parser.add_argument("--acknowledge-real-orders", action="store_true")
    parser.add_argument("--allow-crossing-probe", action="store_true")
    parser.add_argument("--allow-mainnet-crossing-probe", action="store_true")
    parser.add_argument("--allow-passive-probe", action="store_true")
    parser.add_argument("--allow-mainnet-passive-probe", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "plan":
        payload = render_plan(args.symbol)
    elif args.mode == "build-order":
        intent = AloOrderIntent(args.symbol, args.side, args.size, args.price, args.reduce_only)
        maker_check = None
        if args.best_bid is not None and args.best_ask is not None:
            ok, reason = maker_safe(intent, args.best_bid, args.best_ask)
            maker_check = {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask}
        payload = {
            "generated_at": utc_now_iso(),
            "intent": asdict(intent),
            "sdk_order_args": build_sdk_order_args(intent),
            "local_maker_check": maker_check,
        }
    elif args.mode == "classify-result":
        if args.result_json is None:
            raise SystemExit("--result-json is required for classify-result")
        payload = classify_order_result(json.loads(args.result_json.read_text(encoding="utf-8")))
    elif args.mode == "submit-alo":
        payload = submit_alo_order(args)
    elif args.mode == "submit-crossing-alo":
        payload = submit_crossing_alo_probe(args)
    else:
        payload = submit_passive_alo_probe(args)

    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
