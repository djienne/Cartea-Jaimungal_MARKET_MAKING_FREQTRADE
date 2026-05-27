from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from capture_hyperliquid_fee_evidence import (  # noqa: E402
    build_order_evidence_by_oid,
    normalize_events,
    render_plan,
    write_jsonl,
)
from verify_fee_evidence import build_fee_evidence_report, read_jsonl_events  # noqa: E402


def user_fees_payload() -> dict:
    return {
        "userAddRate": "0.00015",
        "userCrossRate": "0.00045",
        "feeSchedule": {"add": "0.00015", "cross": "0.00045"},
    }


def user_fill_payload(**overrides) -> dict:
    payload = {
        "coin": "ETH",
        "px": "100.0",
        "sz": "1.0",
        "side": "B",
        "time": 1_779_638_400_000,
        "oid": 123,
        "crossed": False,
        "fee": "0.015",
        "feeToken": "USDC",
    }
    payload.update(overrides)
    return payload


def order_evidence_payload(*, quote_link: bool = False) -> dict:
    payload = {
        "mode": "submit-passive-alo",
        "intent": {"symbol": "ETH/USDC:USDC", "side": "bid"},
        "sdk_order_args": {
            "name": "ETH",
            "is_buy": True,
            "sz": 1.0,
            "limit_px": 100.0,
            "order_type": {"limit": {"tif": "Alo"}},
            "reduce_only": False,
        },
        "classification": {"resting_oids": [123]},
    }
    if quote_link:
        payload["quote_link"] = {
            "quote_id": "quote-000000000123",
            "side": "bid",
            "hjb_generation": 42,
            "session_id": "session-a",
            "client_order_id": "mm|sess=session-a|qid=quote-000000000123|side=bid|hjb=42",
            "cloid": "0x0123456789abcdef0123456789abcdef",
        }
    return payload


def test_build_order_evidence_maps_sdk_alo_order_by_oid():
    evidence = build_order_evidence_by_oid(order_evidence_payload())

    assert evidence["123"] == {
        "quote_side": "bid",
        "order_type": "limit",
        "tif": "Alo",
        "source": "submit-passive-alo",
    }


def test_order_evidence_preserves_quote_linkage_for_fill_reconciliation():
    evidence = build_order_evidence_by_oid(order_evidence_payload(quote_link=True))

    assert evidence["123"]["quote_side"] == "bid"
    assert evidence["123"]["quote_id"] == "quote-000000000123"
    assert evidence["123"]["session_id"] == "session-a"
    assert evidence["123"]["hjb_generation"] == 42
    assert evidence["123"]["client_order_id"].startswith("mm|sess=session-a")
    assert evidence["123"]["cloid"] == "0x0123456789abcdef0123456789abcdef"
    assert evidence["123"]["quote_link_source"] == "order_evidence_quote_link"


def test_quote_link_preserves_zero_hjb_generation():
    payload = order_evidence_payload(quote_link=True)
    payload["quote_link"]["hjb_generation"] = 0

    evidence = build_order_evidence_by_oid(payload)

    assert evidence["123"]["hjb_generation"] == 0


def test_normalized_hyperliquid_fee_evidence_can_satisfy_fee_checker(tmp_path):
    events = normalize_events(
        raw_user_fees=user_fees_payload(),
        raw_user_fills=[user_fill_payload()],
        order_evidence=order_evidence_payload(),
        expected_maker_fee_rate=0.00015,
        expected_taker_fee_rate=0.00045,
        tolerance=1e-9,
    )
    path = tmp_path / "fee_evidence.jsonl"
    write_jsonl(path, events)

    report = build_fee_evidence_report(
        read_jsonl_events(path),
        now=None,
        max_evidence_age_seconds=0,
    )

    assert report["ok"] is True
    assert report["fee_snapshots"]["exchange_matches"] == 1
    assert report["fills"]["maker_actual_fee_matches"] == 1
    assert report["fills"]["maker_fill_tif_invalid"] == 0


def test_normalized_fill_carries_quote_linkage_and_post_only_evidence():
    events = normalize_events(
        raw_user_fees=user_fees_payload(),
        raw_user_fills=[user_fill_payload()],
        order_evidence=order_evidence_payload(quote_link=True),
        expected_maker_fee_rate=0.00015,
        expected_taker_fee_rate=0.00045,
        tolerance=1e-9,
    )

    fill = [event for event in events if event["event"] == "fill"][0]

    assert fill["quote_id"] == "quote-000000000123"
    assert fill["session_id"] == "session-a"
    assert fill["hjb_generation"] == 42
    assert fill["client_order_id"].startswith("mm|sess=session-a")
    assert fill["cloid"] == "0x0123456789abcdef0123456789abcdef"
    assert fill["quote_link_source"] == "order_evidence_quote_link"
    assert fill["post_only"] is True
    assert fill["post_only_verified"] is True
    assert fill["tif"] == "Alo"


def test_missing_order_evidence_keeps_fee_capture_from_proving_tif():
    events = normalize_events(
        raw_user_fees=user_fees_payload(),
        raw_user_fills=[user_fill_payload()],
        order_evidence=None,
        expected_maker_fee_rate=0.00015,
        expected_taker_fee_rate=0.00045,
        tolerance=1e-9,
    )

    fill = [event for event in events if event["event"] == "fill"][0]
    report = build_fee_evidence_report(events, max_evidence_age_seconds=0)

    assert fill["tif"] is None
    assert fill["order_type"] is None
    assert fill["tif_source"] == "missing_order_evidence"
    assert "maker_fill_tif_invalid:1" in report["reasons"]
    assert "maker_fill_order_type_invalid:1" in report["reasons"]


def test_fee_capture_plan_is_read_only_and_mentions_downstream_checker():
    plan = render_plan()

    assert plan["safe_default"] == "no network calls; no orders; no private key required"
    assert "scripts/verify_fee_evidence.py" == plan["downstream_checker"]
    assert json.dumps(plan)
