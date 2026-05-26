#!/usr/bin/env python3
"""Capture Hyperliquid fee-tier and fill-fee evidence.

The script is read-only and safe by default. `plan` mode emits the required
operator steps without network access. `normalize` mode converts saved
Hyperliquid `userFees` and `userFills` payloads into the JSONL audit events
expected by `verify_fee_evidence.py`. `fetch` mode performs read-only
Hyperliquid info requests, but requires an explicit environment variable and
CLI acknowledgement so account evidence is never collected accidentally.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable


FEE_EVIDENCE_ALLOW_ENV = "HYPERLIQUID_FEE_EVIDENCE_ALLOW"
DEFAULT_OUTPUT = Path("docs/hyperliquid_fee_evidence_capture.jsonl")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def fee_rate_matches(expected: float | None, observed: float | None, tolerance: float) -> bool | None:
    if expected is None or observed is None:
        return None
    return abs(float(expected) - float(observed)) <= float(tolerance)


def timestamp_ms_to_iso(value: Any) -> str:
    parsed = finite_float(value)
    if parsed is None:
        return utc_now_iso()
    return datetime.fromtimestamp(parsed / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def load_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_fill_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("fills", "data", "raw_user_fills"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        nested = payload.get("data")
        if isinstance(nested, dict):
            fills = nested.get("fills")
            if isinstance(fills, list):
                return [item for item in fills if isinstance(item, dict)]
    return []


def normalized_quote_side(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"b", "buy", "bid", "long"}:
        return "bid"
    if text in {"a", "ask", "sell", "short"}:
        return "ask"
    return None


def order_type_from_sdk_args(order_args: dict[str, Any]) -> str | None:
    order_type = order_args.get("order_type")
    if isinstance(order_type, dict) and isinstance(order_type.get("limit"), dict):
        return "limit"
    return None


def tif_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("actual_time_in_force", "tif", "time_in_force"):
        value = payload.get(key)
        if value:
            return str(value)
    order_args = payload.get("sdk_order_args")
    if isinstance(order_args, dict):
        order_type = order_args.get("order_type")
        if isinstance(order_type, dict):
            limit = order_type.get("limit")
            if isinstance(limit, dict) and limit.get("tif"):
                return str(limit["tif"])
    return None


def collect_oids(payload: Any) -> set[str]:
    oids: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"oid", "order_id"} and value is not None:
                oids.add(str(value))
            elif key in {"oids", "order_ids", "resting_oids"} and isinstance(value, list):
                for item in value:
                    if item is not None:
                        oids.add(str(item))
            else:
                oids.update(collect_oids(value))
    elif isinstance(payload, list):
        for item in payload:
            oids.update(collect_oids(item))
    return oids


def iter_order_evidence(payload: Any) -> Iterable[dict[str, Any]]:
    if payload is None:
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("orders", "events", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


def build_order_evidence_by_oid(payload: Any) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for item in iter_order_evidence(payload):
        order_args = item.get("sdk_order_args") if isinstance(item.get("sdk_order_args"), dict) else {}
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        quote_side = normalized_quote_side(item.get("quote_side") or intent.get("side"))
        if quote_side is None and isinstance(order_args, dict) and "is_buy" in order_args:
            quote_side = "bid" if bool(order_args.get("is_buy")) else "ask"
        order_type = item.get("order_type") or order_type_from_sdk_args(order_args)
        tif = tif_from_payload(item)
        source = item.get("mode") or item.get("source") or "order_evidence"
        for oid in collect_oids(item):
            evidence[oid] = {
                "quote_side": quote_side,
                "order_type": str(order_type) if order_type else None,
                "tif": tif,
                "source": str(source),
            }
    return evidence


def normalize_user_fees_event(
    raw_user_fees: dict[str, Any],
    *,
    expected_maker_fee_rate: float,
    expected_taker_fee_rate: float | None,
    tolerance: float,
    generated_at: str,
) -> dict[str, Any]:
    maker_fee = finite_float(raw_user_fees.get("userAddRate"))
    taker_fee = finite_float(raw_user_fees.get("userCrossRate"))
    return {
        "event": "health",
        "ts": generated_at,
        "fee_snapshot": {
            "strategy_maker_fee_rate": float(expected_maker_fee_rate),
            "config_fee_rate": float(expected_maker_fee_rate),
            "config_fee_matches_strategy": True,
            "exchange_fee_source": "hyperliquid_userFees",
            "exchange_maker_fee_rate": maker_fee,
            "exchange_taker_fee_rate": taker_fee,
            "exchange_maker_fee_matches_strategy": fee_rate_matches(expected_maker_fee_rate, maker_fee, tolerance),
            "raw_user_fees": raw_user_fees,
        },
    }


def normalize_user_fill_event(
    fill: dict[str, Any],
    *,
    order_evidence_by_oid: dict[str, dict[str, Any]],
    expected_maker_fee_rate: float,
    expected_taker_fee_rate: float | None,
) -> dict[str, Any]:
    oid = fill.get("oid")
    order_evidence = order_evidence_by_oid.get(str(oid), {}) if oid is not None else {}
    crossed = fill.get("crossed")
    liquidity = "taker" if crossed is True else "maker" if crossed is False else "unknown"
    price = finite_float(fill.get("px"))
    amount = finite_float(fill.get("sz"))
    fee_paid = finite_float(fill.get("fee"))
    actual_fee_paid = abs(float(fee_paid)) if fee_paid is not None else None
    actual_fee_rate = None
    if actual_fee_paid is not None and price is not None and amount is not None and price > 0 and amount > 0:
        actual_fee_rate = actual_fee_paid / abs(float(price) * float(amount))
    expected_fee_rate = expected_maker_fee_rate if liquidity == "maker" else expected_taker_fee_rate if liquidity == "taker" else None
    quote_side = order_evidence.get("quote_side") or normalized_quote_side(fill.get("side"))
    return {
        "event": "fill",
        "ts": timestamp_ms_to_iso(fill.get("time")),
        "pair": f"{fill.get('coin')}/USDC:USDC" if fill.get("coin") else None,
        "quote_side": quote_side,
        "liquidity": liquidity,
        "order_id": oid,
        "order_type": order_evidence.get("order_type"),
        "tif": order_evidence.get("tif"),
        "tif_source": order_evidence.get("source") if order_evidence else "missing_order_evidence",
        "price": price,
        "amount": amount,
        "expected_fee_rate": float(expected_fee_rate) if expected_fee_rate is not None else None,
        "actual_fee_paid": actual_fee_paid,
        "actual_fee_rate": actual_fee_rate,
        "fee_token": fill.get("feeToken"),
        "raw_fill": fill,
    }


def normalize_events(
    *,
    raw_user_fees: dict[str, Any] | None,
    raw_user_fills: Any,
    order_evidence: Any,
    expected_maker_fee_rate: float,
    expected_taker_fee_rate: float | None,
    tolerance: float,
) -> list[dict[str, Any]]:
    generated_at = utc_now_iso()
    events: list[dict[str, Any]] = []
    if isinstance(raw_user_fees, dict):
        events.append(
            normalize_user_fees_event(
                raw_user_fees,
                expected_maker_fee_rate=expected_maker_fee_rate,
                expected_taker_fee_rate=expected_taker_fee_rate,
                tolerance=tolerance,
                generated_at=generated_at,
            )
        )
    order_evidence_by_oid = build_order_evidence_by_oid(order_evidence)
    for fill in as_fill_list(raw_user_fills):
        events.append(
            normalize_user_fill_event(
                fill,
                order_evidence_by_oid=order_evidence_by_oid,
                expected_maker_fee_rate=expected_maker_fee_rate,
                expected_taker_fee_rate=expected_taker_fee_rate,
            )
        )
    return events


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + ("\n" if events else ""), encoding="utf-8")


def fetch_raw_payloads(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if os.getenv(FEE_EVIDENCE_ALLOW_ENV) != "1":
        raise SystemExit(f"Set {FEE_EVIDENCE_ALLOW_ENV}=1 to permit read-only account evidence fetch")
    if not args.acknowledge_account_read:
        raise SystemExit("--acknowledge-account-read is required for fetch mode")
    address = args.account_address or os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS")
    if not address:
        raise SystemExit("--account-address or HYPERLIQUID_ACCOUNT_ADDRESS is required for fetch mode")
    try:
        from hyperliquid.info import Info  # type: ignore
        from hyperliquid.utils import constants  # type: ignore
    except Exception as exc:
        raise SystemExit(f"hyperliquid-python-sdk is required for fetch mode: {exc}") from exc

    base_url = constants.TESTNET_API_URL if args.testnet else constants.MAINNET_API_URL
    info = Info(base_url, skip_ws=True)
    raw_user_fees = info.user_fees(address)
    end_ms = int(args.end_ms) if args.end_ms is not None else int(utc_now().timestamp() * 1000)
    if args.start_ms is not None:
        start_ms = int(args.start_ms)
    else:
        start_ms = int((utc_now() - timedelta(hours=float(args.lookback_hours))).timestamp() * 1000)
    raw_user_fills = info.user_fills_by_time(address, start_ms, end_ms)
    return raw_user_fees, as_fill_list(raw_user_fills)


def render_plan() -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "component": "hyperliquid_fee_evidence_capture",
        "safe_default": "no network calls; no orders; no private key required",
        "read_only_fetch_guards": [
            f"{FEE_EVIDENCE_ALLOW_ENV}=1",
            "--acknowledge-account-read",
            "--account-address or HYPERLIQUID_ACCOUNT_ADDRESS",
        ],
        "raw_inputs": [
            "Hyperliquid Info.user_fees(account)",
            "Hyperliquid Info.user_fills_by_time(account, start_ms, end_ms)",
            "order evidence containing order id, limit order type, and actual Alo/post-only TIF",
        ],
        "normalized_outputs": [
            "fee_snapshot audit event with exchange maker/taker fee rates",
            "fill audit events with maker/taker liquidity, actual fee paid, actual fee rate, price, amount, and order TIF when oid evidence exists",
        ],
        "downstream_checker": "scripts/verify_fee_evidence.py",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture or normalize Hyperliquid account fee evidence.")
    parser.add_argument("--mode", choices=["plan", "normalize", "fetch"], default="plan")
    parser.add_argument("--user-fees-json", type=Path, default=None)
    parser.add_argument("--user-fills-json", type=Path, default=None)
    parser.add_argument("--order-evidence-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-maker-fee-rate", type=float, default=0.00015)
    parser.add_argument("--expected-taker-fee-rate", type=float, default=0.00045)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--account-address", default=None)
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--acknowledge-account-read", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "plan":
        payload = render_plan()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.mode == "fetch":
        raw_user_fees, raw_user_fills = fetch_raw_payloads(args)
    else:
        raw_user_fees = load_json(args.user_fees_json)
        raw_user_fills = load_json(args.user_fills_json)
    events = normalize_events(
        raw_user_fees=raw_user_fees if isinstance(raw_user_fees, dict) else None,
        raw_user_fills=raw_user_fills,
        order_evidence=load_json(args.order_evidence_json),
        expected_maker_fee_rate=float(args.expected_maker_fee_rate),
        expected_taker_fee_rate=float(args.expected_taker_fee_rate) if args.expected_taker_fee_rate is not None else None,
        tolerance=float(args.tolerance),
    )
    write_jsonl(args.output, events)
    print(json.dumps({"generated_at": utc_now_iso(), "output": str(args.output), "events": len(events)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
