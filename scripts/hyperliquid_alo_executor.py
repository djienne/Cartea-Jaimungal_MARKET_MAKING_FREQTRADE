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
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DIRECT_ALO_ALLOW_ENV = "HYPERLIQUID_DIRECT_ALO_ALLOW"
DEFAULT_MAX_SUBMIT_NOTIONAL_USDC = 25.0


@dataclass(frozen=True)
class AloOrderIntent:
    symbol: str
    side: str
    size: float
    price: float
    reduce_only: bool = False
    cloid: str | None = None
    client_order_id: str | None = None

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


def build_client_order_id(
    *,
    quote_id: str | None,
    side: str | None,
    hjb_generation: int | None,
    session_id: str | None,
) -> str:
    parts = [
        "mm",
        f"sess={session_id or 'unknown'}",
        f"qid={quote_id or 'unknown'}",
        f"side={side or 'unknown'}",
        f"hjb={hjb_generation if hjb_generation is not None else 'unknown'}",
    ]
    return "|".join(parts)


def build_hyperliquid_cloid(client_order_id: str) -> str:
    """Build a Hyperliquid CLOID string: 0x plus 16 bytes of deterministic hex."""
    digest = hashlib.sha256(str(client_order_id).encode("utf-8")).hexdigest()[:32]
    return "0x" + digest


def quote_link_payload(
    *,
    quote_id: str | None = None,
    side: str | None = None,
    hjb_generation: int | None = None,
    session_id: str | None = None,
    client_order_id: str | None = None,
    cloid: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "quote_id": quote_id or None,
        "side": side or None,
        "hjb_generation": int(hjb_generation) if hjb_generation is not None else None,
        "session_id": session_id or None,
        "client_order_id": client_order_id or None,
        "cloid": cloid or None,
    }
    if payload["client_order_id"] is None and any(
        payload.get(key) is not None for key in ("quote_id", "side", "hjb_generation", "session_id")
    ):
        payload["client_order_id"] = build_client_order_id(
            quote_id=payload["quote_id"],
            side=payload["side"],
            hjb_generation=payload["hjb_generation"],
            session_id=payload["session_id"],
        )
    if payload["cloid"] is None and payload["client_order_id"] is not None:
        payload["cloid"] = build_hyperliquid_cloid(payload["client_order_id"])
    return payload


def build_sdk_order_args(intent: AloOrderIntent) -> dict[str, Any]:
    if intent.size <= 0:
        raise ValueError("size must be positive")
    if intent.price <= 0:
        raise ValueError("price must be positive")
    order_args = {
        "name": intent.coin,
        "is_buy": intent.is_buy,
        "sz": float(intent.size),
        "limit_px": float(intent.price),
        "order_type": alo_order_type(),
        "reduce_only": bool(intent.reduce_only),
    }
    if intent.cloid:
        order_args["cloid"] = intent.cloid
    return order_args


def actual_tif_from_sdk_order_args(order_args: dict[str, Any]) -> str | None:
    order_type = order_args.get("order_type")
    if not isinstance(order_type, dict):
        return None
    limit = order_type.get("limit")
    if not isinstance(limit, dict):
        return None
    tif = limit.get("tif") or limit.get("timeInForce")
    return str(tif) if tif else None


def order_args_for_submit(order_args: dict[str, Any]) -> dict[str, Any]:
    submit_args = dict(order_args)
    raw_cloid = submit_args.get("cloid")
    if raw_cloid:
        try:
            from hyperliquid.utils.types import Cloid  # type: ignore
        except Exception as exc:
            raise SystemExit(f"hyperliquid-python-sdk is required to submit a CLOID: {exc}")
        submit_args["cloid"] = Cloid.from_str(str(raw_cloid))
    return submit_args


def notional_usdc(intent: AloOrderIntent) -> float:
    return abs(float(intent.size) * float(intent.price))


def notional_limit_check(intent: AloOrderIntent, max_notional_usdc: float | None) -> tuple[bool, str, dict[str, Any]]:
    notional = notional_usdc(intent)
    payload = {
        "notional_usdc": notional,
        "max_notional_usdc": float(max_notional_usdc) if max_notional_usdc is not None else None,
    }
    if max_notional_usdc is not None and float(max_notional_usdc) > 0 and notional > float(max_notional_usdc):
        return False, "notional_limit_exceeded", payload
    return True, "ok", payload


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


def format_cli_number(value: float) -> str:
    return f"{float(value):.12g}"


def level_price(level: Any) -> float | None:
    if isinstance(level, dict):
        for key in ("px", "price", "p"):
            if key in level:
                try:
                    return float(level[key])
                except Exception:
                    return None
    if isinstance(level, (list, tuple)) and level:
        try:
            return float(level[0])
        except Exception:
            return None
    return None


def bbo_from_l2_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    levels = snapshot.get("levels") if isinstance(snapshot, dict) else None
    bids: list[Any] | None = None
    asks: list[Any] | None = None
    if isinstance(levels, list) and len(levels) >= 2:
        bids = levels[0] if isinstance(levels[0], list) else None
        asks = levels[1] if isinstance(levels[1], list) else None
    if bids is None and isinstance(snapshot.get("bids"), list):
        bids = snapshot.get("bids")
    if asks is None and isinstance(snapshot.get("asks"), list):
        asks = snapshot.get("asks")
    best_bid = level_price(bids[0]) if bids else None
    best_ask = level_price(asks[0]) if asks else None
    if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        raise ValueError("Hyperliquid l2Book snapshot did not contain a valid uncrossed BBO")
    return {
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "raw_snapshot": snapshot,
    }


def fetch_public_bbo(symbol: str, *, testnet: bool) -> dict[str, Any]:
    try:
        from hyperliquid.info import Info  # type: ignore
        from hyperliquid.utils import constants  # type: ignore
    except Exception as exc:
        raise SystemExit(f"hyperliquid-python-sdk is required to fetch public BBO: {exc}") from exc
    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(base_url, skip_ws=True)
    snapshot = info.l2_snapshot(normalize_coin(symbol))
    return bbo_from_l2_snapshot(snapshot)


def quote_link_cli_args(quote_link: dict[str, Any]) -> list[str]:
    args: list[str] = []
    mapping = {
        "quote_id": "--quote-id",
        "session_id": "--session-id",
        "hjb_generation": "--hjb-generation",
        "client_order_id": "--client-order-id",
        "cloid": "--cloid",
    }
    for key, flag in mapping.items():
        value = quote_link.get(key)
        if value not in (None, ""):
            args.extend([flag, str(value)])
    return args


def build_probe_preparation(
    *,
    symbol: str,
    side: str,
    size: float,
    best_bid: float,
    best_ask: float,
    passive_price: float | None = None,
    testnet: bool = True,
    max_notional_usdc: float = DEFAULT_MAX_SUBMIT_NOTIONAL_USDC,
    quote_id: str | None = None,
    session_id: str | None = None,
    hjb_generation: int | None = None,
    client_order_id: str | None = None,
    cloid: str | None = None,
    crossing_output: str = "docs/direct_alo_reject_result.json",
    passive_output: str = "docs/direct_alo_passive_result.json",
    evidence_output: str = "docs/post_only_evidence_report.json",
) -> dict[str, Any]:
    if float(size) <= 0:
        raise ValueError("size must be positive")
    best_bid = float(best_bid)
    best_ask = float(best_ask)
    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        raise ValueError("best_bid/best_ask must describe a valid uncrossed book")

    quote_link = quote_link_payload(
        quote_id=quote_id,
        side=side,
        hjb_generation=hjb_generation,
        session_id=session_id,
        client_order_id=client_order_id,
        cloid=cloid,
    )
    crossing_base = crossing_probe_intent(symbol, side, float(size), best_bid, best_ask)
    crossing_intent = AloOrderIntent(
        symbol=crossing_base.symbol,
        side=crossing_base.side,
        size=crossing_base.size,
        price=crossing_base.price,
        reduce_only=crossing_base.reduce_only,
        cloid=quote_link.get("cloid"),
        client_order_id=quote_link.get("client_order_id"),
    )
    crossing_ok, crossing_reason = crossing_probe_check(crossing_intent, best_bid, best_ask)
    if not crossing_ok:
        raise ValueError(f"crossing probe check failed: {crossing_reason}")

    side_is_buy = crossing_intent.is_buy
    default_passive_price = best_bid if side_is_buy else best_ask
    passive_intent = AloOrderIntent(
        symbol=symbol,
        side=side,
        size=float(size),
        price=float(passive_price) if passive_price is not None and float(passive_price) > 0 else default_passive_price,
        cloid=quote_link.get("cloid"),
        client_order_id=quote_link.get("client_order_id"),
    )
    passive_ok, passive_reason = maker_safe(passive_intent, best_bid, best_ask)
    if not passive_ok:
        raise ValueError(f"passive probe maker-safety failed: {passive_reason}")

    crossing_notional_ok, crossing_notional_reason, crossing_notional = notional_limit_check(
        crossing_intent,
        max_notional_usdc,
    )
    passive_notional_ok, passive_notional_reason, passive_notional = notional_limit_check(
        passive_intent,
        max_notional_usdc,
    )
    if not crossing_notional_ok:
        raise ValueError(f"crossing probe notional guard failed: {crossing_notional_reason}")
    if not passive_notional_ok:
        raise ValueError(f"passive probe notional guard failed: {passive_notional_reason}")

    common = [
        "python",
        "scripts/hyperliquid_alo_executor.py",
        "--symbol",
        symbol,
        "--side",
        side,
        "--size",
        format_cli_number(size),
        "--best-bid",
        format_cli_number(best_bid),
        "--best-ask",
        format_cli_number(best_ask),
        "--max-notional-usdc",
        format_cli_number(max_notional_usdc),
        *quote_link_cli_args(quote_link),
    ]
    if testnet:
        common.insert(2, "--testnet")

    crossing_command = [
        *common[:2],
        "--mode",
        "submit-crossing-alo",
        *common[2:],
        "--allow-crossing-probe",
        "--acknowledge-real-orders",
        "--output",
        crossing_output,
    ]
    passive_command = [
        *common[:2],
        "--mode",
        "submit-passive-alo",
        *common[2:],
        "--price",
        format_cli_number(passive_intent.price),
        "--allow-passive-probe",
        "--acknowledge-real-orders",
        "--output",
        passive_output,
    ]
    if not testnet:
        crossing_command.append("--allow-mainnet-crossing-probe")
        passive_command.append("--allow-mainnet-passive-probe")

    return {
        "generated_at": utc_now_iso(),
        "mode": "prepare-probes",
        "safe_default": "no order submission",
        "env_required_for_submit": f"{DIRECT_ALO_ALLOW_ENV}=1",
        "symbol": symbol,
        "side": side,
        "testnet": bool(testnet),
        "bbo": {"best_bid": best_bid, "best_ask": best_ask},
        "quote_link": quote_link,
        "crossing_probe": {
            "intent": asdict(crossing_intent),
            "check": {"ok": crossing_ok, "reason": crossing_reason},
            "notional_check": {"ok": crossing_notional_ok, "reason": crossing_notional_reason, **crossing_notional},
            "command": crossing_command,
        },
        "passive_probe": {
            "intent": asdict(passive_intent),
            "check": {"ok": passive_ok, "reason": passive_reason},
            "notional_check": {"ok": passive_notional_ok, "reason": passive_notional_reason, **passive_notional},
            "command": passive_command,
        },
        "evaluate_command": [
            "python",
            "scripts/verify_post_only_mapping.py",
            "--mode",
            "evaluate-evidence",
            "--crossing-result",
            crossing_output,
            "--passive-result",
            passive_output,
            "--output",
            evidence_output,
        ],
    }


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

    quote_link = quote_link_payload(
        quote_id=args.quote_id,
        side=args.side,
        hjb_generation=args.hjb_generation,
        session_id=args.session_id,
        client_order_id=args.client_order_id,
        cloid=args.cloid,
    )
    intent = AloOrderIntent(
        symbol=args.symbol,
        side=args.side,
        size=float(args.size),
        price=float(args.price),
        reduce_only=bool(args.reduce_only),
        cloid=quote_link.get("cloid"),
        client_order_id=quote_link.get("client_order_id"),
    )
    ok, reason = maker_safe(intent, float(args.best_bid), float(args.best_ask))
    if not ok:
        raise SystemExit(f"local maker-safety failed: {reason}")
    notional_ok, notional_reason, notional_payload = notional_limit_check(intent, args.max_notional_usdc)
    if not notional_ok:
        raise SystemExit(
            f"notional guard failed: {notional_reason} "
            f"({notional_payload['notional_usdc']:.8f} > {notional_payload['max_notional_usdc']:.8f} USDC)"
        )

    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args_for_submit(order_args))
    classification = classify_order_result(result if isinstance(result, dict) else {"raw_result": result})
    actual_tif = actual_tif_from_sdk_order_args(order_args)
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-alo",
        "intent": asdict(intent),
        "quote_link": quote_link,
        "local_maker_check": {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask},
        "notional_check": {"ok": notional_ok, "reason": notional_reason, **notional_payload},
        "sdk_order_args": order_args,
        "actual_time_in_force": actual_tif,
        "actual_time_in_force_source": "hyperliquid_sdk_order_type",
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
    notional_ok, notional_reason, notional_payload = notional_limit_check(intent, args.max_notional_usdc)
    if not notional_ok:
        raise SystemExit(
            f"notional guard failed: {notional_reason} "
            f"({notional_payload['notional_usdc']:.8f} > {notional_payload['max_notional_usdc']:.8f} USDC)"
        )

    quote_link = quote_link_payload(
        quote_id=args.quote_id,
        side=args.side,
        hjb_generation=args.hjb_generation,
        session_id=args.session_id,
        client_order_id=args.client_order_id,
        cloid=args.cloid,
    )
    intent = AloOrderIntent(
        symbol=intent.symbol,
        side=intent.side,
        size=intent.size,
        price=intent.price,
        reduce_only=intent.reduce_only,
        cloid=quote_link.get("cloid"),
        client_order_id=quote_link.get("client_order_id"),
    )
    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args_for_submit(order_args))
    classification = classify_order_result(result if isinstance(result, dict) else {"raw_result": result})
    actual_tif = actual_tif_from_sdk_order_args(order_args)
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-crossing-alo",
        "intent": asdict(intent),
        "quote_link": quote_link,
        "crossing_probe_check": {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask},
        "notional_check": {"ok": notional_ok, "reason": notional_reason, **notional_payload},
        "sdk_order_args": order_args,
        "actual_time_in_force": actual_tif,
        "actual_time_in_force_source": "hyperliquid_sdk_order_type",
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

    quote_link = quote_link_payload(
        quote_id=args.quote_id,
        side=args.side,
        hjb_generation=args.hjb_generation,
        session_id=args.session_id,
        client_order_id=args.client_order_id,
        cloid=args.cloid,
    )
    intent = AloOrderIntent(
        symbol=args.symbol,
        side=args.side,
        size=float(args.size),
        price=float(args.price),
        reduce_only=bool(args.reduce_only),
        cloid=quote_link.get("cloid"),
        client_order_id=quote_link.get("client_order_id"),
    )
    ok, reason = maker_safe(intent, float(args.best_bid), float(args.best_ask))
    if not ok:
        raise SystemExit(f"local maker-safety failed: {reason}")
    notional_ok, notional_reason, notional_payload = notional_limit_check(intent, args.max_notional_usdc)
    if not notional_ok:
        raise SystemExit(
            f"notional guard failed: {notional_reason} "
            f"({notional_payload['notional_usdc']:.8f} > {notional_payload['max_notional_usdc']:.8f} USDC)"
        )

    exchange = load_sdk_exchange(args)
    order_args = build_sdk_order_args(intent)
    result = exchange.order(**order_args_for_submit(order_args))
    classification = classify_order_result(result if isinstance(result, dict) else {"raw_result": result})
    cancel_results = cancel_resting_orders(exchange, intent.coin, classification.get("resting_oids", []))
    actual_tif = actual_tif_from_sdk_order_args(order_args)
    return {
        "generated_at": utc_now_iso(),
        "mode": "submit-passive-alo",
        "intent": asdict(intent),
        "quote_link": quote_link,
        "local_maker_check": {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask},
        "notional_check": {"ok": notional_ok, "reason": notional_reason, **notional_payload},
        "sdk_order_args": order_args,
        "actual_time_in_force": actual_tif,
        "actual_time_in_force_source": "hyperliquid_sdk_order_type",
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
        "sdk_method": "Exchange.order(name, is_buy, sz, limit_px, order_type, reduce_only, cloid)",
        "order_type": alo_order_type(),
        "quote_linking": {
            "sdk_arg": "cloid",
            "cloid_format": "0x + 16-byte hex",
            "client_order_id_fields": ["session_id", "quote_id", "side", "hjb_generation"],
            "raw_client_order_id_stored_in_artifact": True,
        },
        "post_only_evidence_fields": {
            "actual_time_in_force": "Alo",
            "actual_time_in_force_source": "hyperliquid_sdk_order_type",
            "submitted_params_source": "sdk_order_args.order_type.limit.tif",
        },
        "submit_guards": [
            f"{DIRECT_ALO_ALLOW_ENV}=1",
            "--acknowledge-real-orders",
            f"--max-notional-usdc defaults to {DEFAULT_MAX_SUBMIT_NOTIONAL_USDC}",
            "--best-bid and --best-ask local maker-safety inputs",
            "bid price must be below best ask",
            "ask price must be above best bid",
            "--quote-id/--session-id/--hjb-generation produce deterministic CLOID evidence",
        ],
        "crossing_probe_guards": [
            f"{DIRECT_ALO_ALLOW_ENV}=1",
            "--acknowledge-real-orders",
            "--allow-crossing-probe",
            f"--max-notional-usdc defaults to {DEFAULT_MAX_SUBMIT_NOTIONAL_USDC}",
            "--testnet, or --allow-mainnet-crossing-probe for tiny mainnet evidence",
            "--best-bid and --best-ask crossing evidence inputs",
            "bid probe price is best ask; ask probe price is best bid",
            "--quote-id/--session-id/--hjb-generation produce deterministic CLOID evidence",
        ],
        "passive_probe_guards": [
            f"{DIRECT_ALO_ALLOW_ENV}=1",
            "--acknowledge-real-orders",
            "--allow-passive-probe",
            f"--max-notional-usdc defaults to {DEFAULT_MAX_SUBMIT_NOTIONAL_USDC}",
            "--testnet, or --allow-mainnet-passive-probe for tiny mainnet evidence",
            "--best-bid and --best-ask local maker-safety inputs",
            "resting order ids are canceled after evidence capture",
            "--quote-id/--session-id/--hjb-generation produce deterministic CLOID evidence",
        ],
        "response_acceptance": [
            "resting order status is acceptable",
            "ALO post-only rejection/cancel without fill is acceptable",
            "fills require explicit maker liquidity evidence",
            "any taker liquidity fails",
        ],
        "prepare_probes": {
            "mode": "prepare-probes",
            "safe_default": "no order submission",
            "purpose": "turn an observed BBO into exact guarded crossing/passive ALO probe commands",
            "optional_public_bbo_fetch": "--fetch-bbo requires --acknowledge-public-market-read",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded direct Hyperliquid SDK Alo executor.")
    parser.add_argument(
        "--mode",
        choices=[
            "plan",
            "build-order",
            "classify-result",
            "prepare-probes",
            "submit-alo",
            "submit-crossing-alo",
            "submit-passive-alo",
        ],
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
    parser.add_argument("--max-notional-usdc", type=float, default=DEFAULT_MAX_SUBMIT_NOTIONAL_USDC)
    parser.add_argument("--allow-crossing-probe", action="store_true")
    parser.add_argument("--allow-mainnet-crossing-probe", action="store_true")
    parser.add_argument("--allow-passive-probe", action="store_true")
    parser.add_argument("--allow-mainnet-passive-probe", action="store_true")
    parser.add_argument("--fetch-bbo", action="store_true")
    parser.add_argument("--acknowledge-public-market-read", action="store_true")
    parser.add_argument("--crossing-output", default="docs/direct_alo_reject_result.json")
    parser.add_argument("--passive-output", default="docs/direct_alo_passive_result.json")
    parser.add_argument("--evidence-output", default="docs/post_only_evidence_report.json")
    parser.add_argument("--quote-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--hjb-generation", type=int, default=None)
    parser.add_argument("--client-order-id", default=None)
    parser.add_argument("--cloid", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "plan":
        payload = render_plan(args.symbol)
    elif args.mode == "build-order":
        quote_link = quote_link_payload(
            quote_id=args.quote_id,
            side=args.side,
            hjb_generation=args.hjb_generation,
            session_id=args.session_id,
            client_order_id=args.client_order_id,
            cloid=args.cloid,
        )
        intent = AloOrderIntent(
            args.symbol,
            args.side,
            args.size,
            args.price,
            args.reduce_only,
            cloid=quote_link.get("cloid"),
            client_order_id=quote_link.get("client_order_id"),
        )
        maker_check = None
        if args.best_bid is not None and args.best_ask is not None:
            ok, reason = maker_safe(intent, args.best_bid, args.best_ask)
            maker_check = {"ok": ok, "reason": reason, "best_bid": args.best_bid, "best_ask": args.best_ask}
        payload = {
            "generated_at": utc_now_iso(),
            "intent": asdict(intent),
            "quote_link": quote_link,
            "sdk_order_args": build_sdk_order_args(intent),
            "local_maker_check": maker_check,
        }
    elif args.mode == "classify-result":
        if args.result_json is None:
            raise SystemExit("--result-json is required for classify-result")
        payload = classify_order_result(json.loads(args.result_json.read_text(encoding="utf-8")))
    elif args.mode == "prepare-probes":
        fetched_bbo = None
        if args.fetch_bbo:
            if not args.acknowledge_public_market_read:
                raise SystemExit("--acknowledge-public-market-read is required with --fetch-bbo")
            fetched_bbo = fetch_public_bbo(args.symbol, testnet=bool(args.testnet))
            best_bid = fetched_bbo["best_bid"]
            best_ask = fetched_bbo["best_ask"]
        else:
            if args.best_bid is None or args.best_ask is None:
                raise SystemExit("--best-bid and --best-ask are required unless --fetch-bbo is used")
            best_bid = args.best_bid
            best_ask = args.best_ask
        payload = build_probe_preparation(
            symbol=args.symbol,
            side=args.side,
            size=float(args.size),
            best_bid=float(best_bid),
            best_ask=float(best_ask),
            passive_price=float(args.price) if args.price and args.price > 0 else None,
            testnet=bool(args.testnet),
            max_notional_usdc=float(args.max_notional_usdc),
            quote_id=args.quote_id,
            session_id=args.session_id,
            hjb_generation=args.hjb_generation,
            client_order_id=args.client_order_id,
            cloid=args.cloid,
            crossing_output=str(args.crossing_output),
            passive_output=str(args.passive_output),
            evidence_output=str(args.evidence_output),
        )
        payload["bbo"]["source"] = "hyperliquid_l2_snapshot" if fetched_bbo is not None else "cli"
        if fetched_bbo is not None:
            payload["bbo"]["raw_snapshot"] = fetched_bbo.get("raw_snapshot")
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
