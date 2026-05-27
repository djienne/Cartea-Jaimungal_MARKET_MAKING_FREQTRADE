from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hyperliquid_alo_executor import (  # noqa: E402
    AloOrderIntent,
    actual_tif_from_sdk_order_args,
    alo_order_type,
    bbo_from_l2_snapshot,
    build_client_order_id,
    build_probe_preparation,
    build_hyperliquid_cloid,
    build_sdk_order_args,
    cancel_resting_orders,
    classify_order_result,
    crossing_probe_check,
    crossing_probe_intent,
    maker_safe,
    notional_limit_check,
    notional_usdc,
    normalize_coin,
    quote_link_payload,
    render_plan,
    submit_crossing_alo_probe,
    submit_passive_alo_probe,
)
from verify_post_only_mapping import evaluate_crossing_result, evaluate_passive_result  # noqa: E402


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
    assert actual_tif_from_sdk_order_args(args) == "Alo"


def test_quote_link_payload_builds_deterministic_hyperliquid_cloid():
    link = quote_link_payload(
        quote_id="quote-000000000123",
        side="bid",
        hjb_generation=42,
        session_id="session-a",
    )

    assert link["client_order_id"] == "mm|sess=session-a|qid=quote-000000000123|side=bid|hjb=42"
    assert link["cloid"].startswith("0x")
    assert len(link["cloid"]) == 34
    assert link["cloid"] == build_hyperliquid_cloid(link["client_order_id"])
    assert quote_link_payload(client_order_id=link["client_order_id"])["cloid"] == link["cloid"]


def test_build_sdk_order_args_includes_cloid_when_quote_linked():
    client_order_id = build_client_order_id(
        quote_id="quote-000000000123",
        side="bid",
        hjb_generation=42,
        session_id="session-a",
    )
    intent = AloOrderIntent(
        "ETH/USDC:USDC",
        "bid",
        0.01,
        2000.0,
        cloid=build_hyperliquid_cloid(client_order_id),
        client_order_id=client_order_id,
    )

    args = build_sdk_order_args(intent)

    assert args["cloid"] == build_hyperliquid_cloid(client_order_id)


def test_maker_safe_rejects_crossing_bid_and_ask():
    assert maker_safe(AloOrderIntent("ETH", "bid", 0.01, 101.0), 99.0, 101.0) == (False, "bid_crosses_ask")
    assert maker_safe(AloOrderIntent("ETH", "ask", 0.01, 99.0), 99.0, 101.0) == (False, "ask_crosses_bid")


def test_maker_safe_accepts_passive_quotes():
    assert maker_safe(AloOrderIntent("ETH", "bid", 0.01, 100.0), 99.0, 101.0) == (True, "ok")
    assert maker_safe(AloOrderIntent("ETH", "ask", 0.01, 100.5), 99.0, 101.0) == (True, "ok")


def test_notional_limit_check_rejects_oversized_submit_intent():
    intent = AloOrderIntent("ETH", "bid", 0.02, 2000.0)

    ok, reason, payload = notional_limit_check(intent, 25.0)

    assert notional_usdc(intent) == 40.0
    assert ok is False
    assert reason == "notional_limit_exceeded"
    assert payload["notional_usdc"] == 40.0
    assert payload["max_notional_usdc"] == 25.0


def test_notional_limit_check_accepts_tiny_submit_intent():
    intent = AloOrderIntent("ETH", "bid", 0.01, 2000.0)

    ok, reason, payload = notional_limit_check(intent, 25.0)

    assert ok is True
    assert reason == "ok"
    assert payload["notional_usdc"] == 20.0


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


def test_cancel_resting_orders_calls_sdk_cancel():
    class FakeExchange:
        def __init__(self):
            self.calls = []

        def cancel(self, coin, oid):
            self.calls.append((coin, oid))
            return {"status": "ok", "oid": oid}

    exchange = FakeExchange()

    results = cancel_resting_orders(exchange, "ETH", [123, "456"])

    assert exchange.calls == [("ETH", 123), ("ETH", 456)]
    assert results == [
        {"oid": 123, "ok": True, "raw_result": {"status": "ok", "oid": 123}},
        {"oid": "456", "ok": True, "raw_result": {"status": "ok", "oid": 456}},
    ]


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
    assert plan["post_only_evidence_fields"]["actual_time_in_force"] == "Alo"
    assert plan["post_only_evidence_fields"]["actual_time_in_force_source"] == "hyperliquid_sdk_order_type"
    assert plan["quote_linking"]["sdk_arg"] == "cloid"
    assert "quote_id" in plan["quote_linking"]["client_order_id_fields"]
    assert any("HYPERLIQUID_DIRECT_ALO_ALLOW" in item for item in plan["submit_guards"])
    assert any("--max-notional-usdc" in item for item in plan["submit_guards"])
    assert any("--quote-id" in item for item in plan["submit_guards"])
    assert any("--allow-crossing-probe" in item for item in plan["crossing_probe_guards"])
    assert any("--allow-passive-probe" in item for item in plan["passive_probe_guards"])
    assert plan["prepare_probes"]["mode"] == "prepare-probes"
    assert plan["prepare_probes"]["safe_default"] == "no order submission"


def test_bbo_from_l2_snapshot_accepts_hyperliquid_levels_shape():
    snapshot = {"levels": [[{"px": "99.5", "sz": "1.0"}], [{"px": "100.5", "sz": "2.0"}]]}

    bbo = bbo_from_l2_snapshot(snapshot)

    assert bbo["best_bid"] == 99.5
    assert bbo["best_ask"] == 100.5


def test_prepare_probe_plan_builds_guarded_commands_from_bbo():
    plan = build_probe_preparation(
        symbol="ETH/USDC:USDC",
        side="bid",
        size=0.01,
        best_bid=99.0,
        best_ask=101.0,
        testnet=True,
        quote_id="quote-000000000123",
        session_id="session-a",
        hjb_generation=42,
    )

    crossing = plan["crossing_probe"]
    passive = plan["passive_probe"]

    assert plan["safe_default"] == "no order submission"
    assert crossing["intent"]["price"] == 101.0
    assert crossing["intent"]["client_order_id"] == "mm|sess=session-a|qid=quote-000000000123|side=bid|hjb=42"
    assert crossing["intent"]["cloid"] == plan["quote_link"]["cloid"]
    assert crossing["check"] == {"ok": True, "reason": "bid_crosses_ask_for_alo_probe"}
    assert passive["intent"]["price"] == 99.0
    assert passive["intent"]["client_order_id"] == "mm|sess=session-a|qid=quote-000000000123|side=bid|hjb=42"
    assert passive["intent"]["cloid"] == plan["quote_link"]["cloid"]
    assert passive["check"] == {"ok": True, "reason": "ok"}
    assert crossing["notional_check"]["notional_usdc"] == 1.01
    assert passive["notional_check"]["notional_usdc"] == 0.99
    assert "--testnet" in crossing["command"]
    assert "--allow-crossing-probe" in crossing["command"]
    assert "--acknowledge-real-orders" in crossing["command"]
    assert "--allow-passive-probe" in passive["command"]
    assert passive["command"][passive["command"].index("--price") + 1] == "99"
    assert plan["quote_link"]["client_order_id"] == "mm|sess=session-a|qid=quote-000000000123|side=bid|hjb=42"
    assert plan["evaluate_command"][-1] == "docs/post_only_evidence_report.json"


def test_prepare_probe_plan_rejects_oversized_probe():
    try:
        build_probe_preparation(
            symbol="ETH/USDC:USDC",
            side="bid",
            size=1.0,
            best_bid=99.0,
            best_ask=101.0,
            testnet=True,
            max_notional_usdc=25.0,
        )
    except ValueError as exc:
        assert "notional guard failed" in str(exc)
    else:
        raise AssertionError("expected oversized probe to fail")


def args_for_submit(**overrides):
    payload = {
        "symbol": "ETH/USDC:USDC",
        "side": "bid",
        "size": 0.01,
        "price": 100.0,
        "reduce_only": False,
        "best_bid": 99.0,
        "best_ask": 101.0,
        "testnet": True,
        "acknowledge_real_orders": True,
        "allow_crossing_probe": False,
        "allow_mainnet_crossing_probe": False,
        "allow_passive_probe": False,
        "allow_mainnet_passive_probe": False,
        "quote_id": "quote-000000000123",
        "session_id": "session-a",
        "hjb_generation": 42,
        "client_order_id": None,
        "cloid": None,
        "max_notional_usdc": 25.0,
        "private_key": None,
        "account_address": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class FakeExchange:
    def __init__(self, result):
        self.result = result
        self.orders = []
        self.cancels = []

    def order(self, **kwargs):
        self.orders.append(kwargs)
        return self.result

    def cancel(self, coin, oid):
        self.cancels.append((coin, oid))
        return {"status": "ok", "oid": oid}


def test_direct_passive_probe_artifact_carries_actual_alo_tif(monkeypatch):
    result = {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}}
    exchange = FakeExchange(result)
    monkeypatch.setenv("HYPERLIQUID_DIRECT_ALO_ALLOW", "1")
    monkeypatch.setattr("hyperliquid_alo_executor.load_sdk_exchange", lambda args: exchange)

    payload = submit_passive_alo_probe(args_for_submit(allow_passive_probe=True))
    ok, reasons = evaluate_passive_result(payload)

    assert payload["actual_time_in_force"] == "Alo"
    assert payload["actual_time_in_force_source"] == "hyperliquid_sdk_order_type"
    assert payload["sdk_order_args"]["order_type"] == {"limit": {"tif": "Alo"}}
    assert payload["quote_link"]["quote_id"] == "quote-000000000123"
    assert payload["cancel_results"][0]["oid"] == 123
    assert exchange.cancels == [("ETH", 123)]
    assert ok is True
    assert reasons == []


def test_direct_crossing_probe_artifact_carries_actual_alo_tif(monkeypatch):
    result = {
        "status": "ok",
        "response": {"data": {"statuses": [{"error": "Post only order would immediately match"}]}},
    }
    exchange = FakeExchange(result)
    monkeypatch.setenv("HYPERLIQUID_DIRECT_ALO_ALLOW", "1")
    monkeypatch.setattr("hyperliquid_alo_executor.load_sdk_exchange", lambda args: exchange)

    payload = submit_crossing_alo_probe(
        args_for_submit(price=0.0, allow_crossing_probe=True, best_bid=99.0, best_ask=101.0)
    )
    ok, reasons = evaluate_crossing_result(payload)

    assert payload["actual_time_in_force"] == "Alo"
    assert payload["actual_time_in_force_source"] == "hyperliquid_sdk_order_type"
    assert payload["sdk_order_args"]["limit_px"] == 101.0
    assert payload["classification"]["alo_rejected"] is True
    assert ok is True
    assert reasons == []
