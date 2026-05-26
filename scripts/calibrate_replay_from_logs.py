#!/usr/bin/env python3
"""Summarize dry-run/testnet logs for replay fill calibration.

This does not make live trading safer by itself. It turns quote/fill audit logs
into a small artifact that says whether the sample is good enough to tune replay
fill probabilities, and if so what the observed fill rates look like by side and
depth bucket.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LOG = Path("user_data/logs/mm_debug.jsonl")
DEFAULT_OUTPUT = Path("docs/replay_log_calibration.json")


@dataclass(frozen=True)
class AcceptedQuote:
    ts: pd.Timestamp
    pair: str
    side: str
    price: float
    depth_bucket: str
    quote_id: str | None = None


def parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if np.isfinite(parsed) else None


def text_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def read_jsonl_events(paths: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def depth_bucket(event: dict[str, Any], bucket_bps: float) -> str:
    bps = finite_float(event.get("bps"))
    if bps is None:
        mid = finite_float(event.get("mid"))
        rounded = finite_float(event.get("rounded_price"))
        if mid is not None and rounded is not None and mid > 0:
            bps = abs(mid - rounded) / mid * 10_000.0
    if bps is None:
        return "unknown"
    bucket = max(0.0, float(bucket_bps))
    if bucket <= 0:
        return f"{bps:.2f}bps"
    lower = np.floor(float(bps) / bucket) * bucket
    upper = lower + bucket
    return f"{lower:.2f}-{upper:.2f}bps"


def quote_from_event(event: dict[str, Any], *, bucket_bps: float) -> AcceptedQuote | None:
    if event.get("event") != "quote_decision":
        return None
    if event.get("decision") != "accept":
        return None
    side = str(event.get("side") or "").lower()
    if side not in {"bid", "ask"}:
        return None
    ts = parse_ts(event.get("ts"))
    price = finite_float(event.get("rounded_price"))
    pair = str(event.get("pair") or "")
    if ts is None or price is None or not pair:
        return None
    return AcceptedQuote(
        ts=ts,
        pair=pair,
        side=side,
        price=price,
        depth_bucket=depth_bucket(event, bucket_bps),
        quote_id=text_or_none(event.get("quote_id")),
    )


def increment(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = int(mapping.get(key, 0)) + int(amount)


def match_quote(
    fill_event: dict[str, Any],
    accepted_quotes: list[AcceptedQuote],
    *,
    match_window_seconds: float,
    price_tolerance: float,
) -> AcceptedQuote | None:
    fill_ts = parse_ts(fill_event.get("ts"))
    fill_price = finite_float(fill_event.get("price"))
    fill_side = str(fill_event.get("quote_side") or "").lower()
    pair = str(fill_event.get("pair") or "")
    if fill_ts is None or fill_price is None or fill_side not in {"bid", "ask"} or not pair:
        return None

    fill_quote_id = text_or_none(fill_event.get("quote_id"))
    if fill_quote_id is not None:
        quote_id_candidates = [
            quote
            for quote in accepted_quotes
            if quote.quote_id == fill_quote_id
            and quote.pair == pair
            and quote.side == fill_side
            and abs(float(quote.price) - float(fill_price)) <= float(price_tolerance)
        ]
        if quote_id_candidates:
            return max(quote_id_candidates, key=lambda quote: quote.ts)

    earliest = fill_ts - pd.Timedelta(seconds=float(match_window_seconds))
    candidates = [
        quote
        for quote in accepted_quotes
        if quote.pair == pair
        and quote.side == fill_side
        and earliest <= quote.ts <= fill_ts
        and abs(float(quote.price) - float(fill_price)) <= float(price_tolerance)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda quote: quote.ts)


def summarize_markouts(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_horizon: dict[str, list[float]] = {}
    for event in events:
        if event.get("event") != "fill_markout":
            continue
        horizon = event.get("horizon_ms")
        value = finite_float(event.get("markout_usdc"))
        if horizon is None or value is None:
            continue
        by_horizon.setdefault(str(int(horizon)), []).append(value)
    return {
        horizon: {
            "count": float(len(values)),
            "mean_usdc": float(np.mean(values)) if values else 0.0,
            "min_usdc": float(np.min(values)) if values else 0.0,
            "max_usdc": float(np.max(values)) if values else 0.0,
        }
        for horizon, values in sorted(by_horizon.items(), key=lambda item: int(item[0]))
    }


def fill_probability(accepted: dict[str, int], fills: dict[str, int]) -> dict[str, float]:
    keys = sorted(set(accepted) | set(fills))
    return {
        key: float(fills.get(key, 0)) / float(accepted[key])
        for key in keys
        if int(accepted.get(key, 0)) > 0
    }


def build_calibration_report(
    events: list[dict[str, Any]],
    *,
    bucket_bps: float = 1.0,
    match_window_seconds: float = 300.0,
    price_tolerance: float = 1e-8,
    max_fee_rate: float = 0.01,
    min_quotes: int = 100,
    min_fills: int = 10,
) -> dict[str, Any]:
    accepted_quotes = [quote for event in events if (quote := quote_from_event(event, bucket_bps=bucket_bps))]
    accepted_by_side: dict[str, int] = {}
    accepted_by_depth: dict[str, int] = {}
    for quote in accepted_quotes:
        increment(accepted_by_side, quote.side)
        increment(accepted_by_depth, f"{quote.side}:{quote.depth_bucket}")

    fills = [event for event in events if event.get("event") == "fill"]
    fills_by_side: dict[str, int] = {}
    maker_fills = 0
    taker_fills = 0
    unknown_liquidity_fills = 0
    matched_fills_by_depth: dict[str, int] = {}
    unmatched_fills = 0
    fills_with_quote_id = 0
    matched_fills_by_quote_id: dict[str, int] = {}
    actual_fee_rates: list[float] = []
    actual_fee_rate_outliers = 0

    for fill in fills:
        side = str(fill.get("quote_side") or "unknown").lower()
        increment(fills_by_side, side)

        liquidity = str(fill.get("liquidity") or "").lower()
        if liquidity == "maker":
            maker_fills += 1
        elif liquidity == "taker":
            taker_fills += 1
        else:
            unknown_liquidity_fills += 1

        if text_or_none(fill.get("quote_id")) is not None:
            fills_with_quote_id += 1

        actual_fee_rate = finite_float(fill.get("actual_fee_rate"))
        if actual_fee_rate is not None:
            if 0.0 <= actual_fee_rate <= float(max_fee_rate):
                actual_fee_rates.append(actual_fee_rate)
            else:
                actual_fee_rate_outliers += 1

        matched = match_quote(
            fill,
            accepted_quotes,
            match_window_seconds=match_window_seconds,
            price_tolerance=price_tolerance,
        )
        if matched is None:
            unmatched_fills += 1
        else:
            increment(matched_fills_by_depth, f"{matched.side}:{matched.depth_bucket}")
            if matched.quote_id is not None:
                increment(matched_fills_by_quote_id, matched.quote_id)

    reasons: list[str] = []
    if len(accepted_quotes) < int(min_quotes):
        reasons.append(f"insufficient_accepted_quotes:{len(accepted_quotes)}<min_{int(min_quotes)}")
    if maker_fills < int(min_fills):
        reasons.append(f"insufficient_maker_fills:{maker_fills}<min_{int(min_fills)}")
    if taker_fills:
        reasons.append(f"taker_fills_seen:{taker_fills}")
    if unknown_liquidity_fills:
        reasons.append(f"unknown_liquidity_fills:{unknown_liquidity_fills}")
    if fills and unmatched_fills == len(fills):
        reasons.append("no_fills_matched_to_quotes")

    maker_ratio = float(maker_fills) / float(len(fills)) if fills else 0.0
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "usable_for_calibration": not reasons,
        "reasons": reasons,
        "inputs": {
            "events": len(events),
            "accepted_quotes": len(accepted_quotes),
            "accepted_quotes_with_quote_id": sum(1 for quote in accepted_quotes if quote.quote_id is not None),
            "fills": len(fills),
            "fills_with_quote_id": int(fills_with_quote_id),
            "bucket_bps": float(bucket_bps),
            "match_window_seconds": float(match_window_seconds),
            "price_tolerance": float(price_tolerance),
            "max_fee_rate": float(max_fee_rate),
            "min_quotes": int(min_quotes),
            "min_fills": int(min_fills),
        },
        "accepted_quotes_by_side": accepted_by_side,
        "accepted_quotes_by_depth": accepted_by_depth,
        "fills_by_side": fills_by_side,
        "matched_fills_by_depth": matched_fills_by_depth,
        "matched_fills_by_quote_id": matched_fills_by_quote_id,
        "unmatched_fills": int(unmatched_fills),
        "maker_fills": int(maker_fills),
        "taker_fills": int(taker_fills),
        "unknown_liquidity_fills": int(unknown_liquidity_fills),
        "maker_ratio": maker_ratio,
        "fill_probability_by_side": fill_probability(accepted_by_side, fills_by_side),
        "fill_probability_by_depth": fill_probability(accepted_by_depth, matched_fills_by_depth),
        "actual_fee_rate": {
            "count": len(actual_fee_rates),
            "mean": float(np.mean(actual_fee_rates)) if actual_fee_rates else None,
            "min": float(np.min(actual_fee_rates)) if actual_fee_rates else None,
            "max": float(np.max(actual_fee_rates)) if actual_fee_rates else None,
            "outlier_count": int(actual_fee_rate_outliers),
            "outlier_threshold": float(max_fee_rate),
        },
        "markout_summary": summarize_markouts(events),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replay fill calibration metrics from JSONL audit logs.")
    parser.add_argument("--input", type=Path, action="append", default=None, help="JSONL audit log path. May be repeated.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bucket-bps", type=float, default=1.0)
    parser.add_argument("--match-window-seconds", type=float, default=300.0)
    parser.add_argument("--price-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-fee-rate", type=float, default=0.01)
    parser.add_argument("--min-quotes", type=int, default=100)
    parser.add_argument("--min-fills", type=int, default=10)
    parser.add_argument("--require-usable", action="store_true", help="Return nonzero when the sample is not usable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = args.input or [DEFAULT_LOG]
    events = read_jsonl_events(inputs)
    report = build_calibration_report(
        events,
        bucket_bps=args.bucket_bps,
        match_window_seconds=args.match_window_seconds,
        price_tolerance=args.price_tolerance,
        max_fee_rate=args.max_fee_rate,
        min_quotes=args.min_quotes,
        min_fills=args.min_fills,
    )
    report["input_paths"] = [str(path) for path in inputs]
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["usable_for_calibration"] or not args.require_usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
