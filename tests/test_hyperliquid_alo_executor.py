from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hyperliquid_alo_executor import (  # noqa: E402
    AloOrderIntent,
    alo_order_type,
    build_sdk_order_args,
    classify_order_result,
    crossing_probe_check,
    crossing_probe_intent,
    maker_safe,
    normalize_coin,
    render_plan,
)


def test_normalize_coin_from_freqtrade_pair():
    assert normalize_coin("ETH/USDC:USDC") == "ETH"
    assert normalize_coin("BTC") == "BTC"


def test_build_sdk_order_args_uses_alo_order_type():
    intent = AloOrderIntent("ETH/USDC:USDC", "bid", 0.01, 2000.0)

    args = build_sdk_order_args(intent)

    assert args == {
        "name": "ETH",
        "is_buy": True,
        "sz": 0.01,
        "limit_px": 2000.0,
        "order_type": {"limit": {"tif": "Alo"}},
        "reduce_only": False,
    }
    assert alo_order_type()["limit"]["tif"] == "Alo"


def test_maker_safe_rejects_crossing_bid_and_ask():
    assert maker_safe(AloOrderIntent("ETH", "bid", 0.01, 101.0), 99.0, 101.0) == (False, "bid_crosses_ask")
    assert maker_safe(AloOrderIntent("ETH", "ask", 0.01, 99.0), 99.0, 101.0) == (False, "ask_crosses_bid")


def test_maker_safe_accepts_passive_quotes():
    assert maker_safe(AloOrderIntent("ETH", "bid", 0.01, 100.0), 99.0, 101.0) == (True, "ok")
    assert maker_safe(AloOrderIntent("ETH", "ask", 0.01, 100.5), 99.0, 101.0) == (True, "ok")


def test_crossing_probe_intent_uses_opposite_best_price():
    bid_probe = crossing_probe_intent("ETH/USDC:USDC", "bid", 0.01, 99.0, 101.0)
    ask_probe = crossing_probe_intent("ETH/USDC:USDC", "ask", 0.01, 99.0, 101.0)

    assert bid_probe.price == 101.0
    assert ask_probe.price == 99.0
    assert crossing_probe_check(bid_probe, 99.0, 101.0) == (True, "bid_crosses_ask_for_alo_probe")
    assert crossing_probe_check(ask_probe, 99.0, 101.0) == (True, "ask_crosses_bid_for_alo_probe")


def test_crossing_probe_check_rejects_passive_intent():
    intent = AloOrderIntent("ETH", "bid", 0.01, 100.0)

    assert crossing_probe_check(intent, 99.0, 101.0) == (False, "probe_does_not_cross")


def test_classify_resting_sdk_result_as_ok():
    result = {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}}

    classified = classify_order_result(result)

    assert classified["ok"] is True
    assert classified["saw_resting"] is True
    assert classified["resting_oids"] == [123]


def test_classify_post_only_error_as_ok_rejection():
    result = {"status": "ok", "response": {"data": {"statuses": [{"error": "Post only order would immediately match"}]}}}

    classified = classify_order_result(result)

    assert classified["ok"] is True
    assert classified["alo_rejected"] is True


def test_classify_taker_fill_as_failure():
    result = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "filled": {"totalSz": "0.01", "avgPx": "2000"},
                        "liquidity": "taker",
                    }
                ]
            }
        },
    }

    classified = classify_order_result(result)

    assert classified["ok"] is False
    assert "taker_liquidity_seen" in classified["reasons"]


def test_classify_fill_without_maker_flag_as_failure():
    result = {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": "0.01"}}]}}}

    classified = classify_order_result(result)

    assert classified["ok"] is False
    assert "filled_without_maker_liquidity_flag" in classified["reasons"]


def test_plan_documents_submit_guards():
    plan = render_plan("ETH/USDC:USDC")

    assert plan["order_type"] == {"limit": {"tif": "Alo"}}
    assert any("HYPERLIQUID_DIRECT_ALO_ALLOW" in item for item in plan["submit_guards"])
    assert any("--allow-crossing-probe" in item for item in plan["crossing_probe_guards"])
