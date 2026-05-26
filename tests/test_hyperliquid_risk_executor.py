from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hyperliquid_risk_executor import (  # noqa: E402
    build_flatten_intent,
    build_risk_client_order_id,
    build_sdk_order_args,
    classify_flatten_result,
    flatten_ioc_price,
    flatten_order_type,
    flatten_side_from_position,
    notional_limit_check,
    render_plan,
)


def test_flatten_side_reduces_signed_position():
    assert flatten_side_from_position(0.03) == "sell"
    assert flatten_side_from_position(-0.02) == "buy"


def test_flatten_ioc_price_crosses_with_configured_slippage():
    assert flatten_ioc_price(2000.0, "sell", 50.0) == 1990.0
    assert round(flatten_ioc_price(2000.0, "buy", 50.0), 8) == 2010.0


def test_flatten_ioc_price_rejects_nonpositive_sell_limit():
    try:
        flatten_ioc_price(2000.0, "sell", 10_000.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected nonpositive flatten price to fail")


def test_build_flatten_intent_for_long_position_is_reduce_only_ioc_sell():
    intent = build_flatten_intent(
        symbol="ETH/USDC:USDC",
        signed_position_base=0.03,
        reference_price=2000.0,
        reason="position_limit_reached",
        risk_action_id="risk-1",
        session_id="session-a",
    )

    assert intent.coin == "ETH"
    assert intent.side == "sell"
    assert intent.is_buy is False
    assert intent.size == 0.03
    assert intent.price == 1990.0
    assert intent.reduce_only is True
    assert intent.client_order_id == "risk|sess=session-a|rid=risk-1|mode=flatten|side=sell|reason=position_limit_reached"
    assert intent.cloid and intent.cloid.startswith("0x")

    args = build_sdk_order_args(intent)
    assert args["order_type"] == {"limit": {"tif": "Ioc"}}
    assert args["reduce_only"] is True
    assert args["is_buy"] is False


def test_build_flatten_intent_for_short_position_is_reduce_only_ioc_buy():
    intent = build_flatten_intent(
        symbol="ETH/USDC:USDC",
        signed_position_base=-0.02,
        reference_price=2000.0,
        reason="unexpected_short_position",
        max_size_base=0.01,
    )

    assert intent.side == "buy"
    assert intent.is_buy is True
    assert intent.size == 0.01
    assert round(intent.price, 8) == 2010.0


def test_notional_limit_check_rejects_oversized_flatten_probe():
    intent = build_flatten_intent(
        symbol="ETH",
        signed_position_base=0.03,
        reference_price=2000.0,
        reason="manual_test",
    )

    ok, reason, payload = notional_limit_check(intent, 25.0)

    assert ok is False
    assert reason == "notional_limit_exceeded"
    assert round(payload["notional_usdc"], 8) == 59.7
    assert payload["max_notional_usdc"] == 25.0


def test_classify_flatten_result_accepts_filled_or_cancelled_ioc():
    filled = {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": "0.01"}}]}}}
    cancelled = {"status": "ok", "response": {"data": {"statuses": [{"canceled": {"oid": 123}}]}}}

    assert classify_flatten_result(filled)["ok"] is True
    assert classify_flatten_result(cancelled)["ok"] is True


def test_classify_flatten_result_rejects_resting_or_error_status():
    resting = {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}}
    error = {"status": "ok", "response": {"data": {"statuses": [{"error": "reduce only rejected"}]}}}

    assert "ioc_order_resting" in classify_flatten_result(resting)["reasons"]
    assert "exchange_error_status" in classify_flatten_result(error)["reasons"]


def test_plan_documents_risk_flatten_guards():
    plan = render_plan("ETH/USDC:USDC")

    assert plan["order_type"] == flatten_order_type()
    assert plan["reduce_only"] is True
    assert plan["not_maker_strategy"] is True
    assert any("HYPERLIQUID_RISK_FLATTEN_ALLOW" in item for item in plan["submit_guards"])
    assert any("--acknowledge-risk-reducing-taker" in item for item in plan["submit_guards"])
    assert plan["side_mapping"]["positive_signed_position"] == "sell reduce-only IOC"


def test_risk_client_order_id_is_readable_before_cloid_hashing():
    client_id = build_risk_client_order_id(
        risk_action_id="risk-1",
        session_id="session-a",
        side="sell",
        reason="drawdown_limit_reached",
    )

    assert client_id == "risk|sess=session-a|rid=risk-1|mode=flatten|side=sell|reason=drawdown_limit_reached"
