from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from get_epsilon import save_epsilon_to_json  # noqa: E402
from get_kappa import save_kappa_lambda_to_json  # noqa: E402
from get_lambda import save_lambda_to_json  # noqa: E402
from param_utils import PARAM_SCHEMA_VERSION, atomic_write_json, load_json_object  # noqa: E402
from verify_dry_run_disabled import evaluate_logs, recent_health_events  # noqa: E402
from verify_post_only_mapping import (  # noqa: E402
    alo_order_params,
    evaluate_crossing_result,
    evaluate_evidence,
    evaluate_passive_result,
    render_plan,
)
from validate_hl_data import latest_parquet_timestamp, validate_parquet_file, validate_symbol  # noqa: E402
from verify_dry_run_enabled import evaluate_enabled_gate, write_gate_config, write_gate_params  # noqa: E402


def test_atomic_write_json_round_trip(tmp_path):
    path = tmp_path / "params.json"
    atomic_write_json(path, {"ETH": {"schema_version": PARAM_SCHEMA_VERSION}})

    assert load_json_object(path) == {"ETH": {"schema_version": PARAM_SCHEMA_VERSION}}
    assert not path.with_suffix(".json.tmp").exists()


def test_kappa_lambda_writer_uses_schema_v2_and_lambda0_fit(tmp_path):
    kappa_path = tmp_path / "kappa.json"
    lambda_path = tmp_path / "lambda.json"

    save_kappa_lambda_to_json(
        2.0,
        3.0,
        0.1,
        0.2,
        "ETH",
        kappa_file=str(kappa_path),
        lambda_file=str(lambda_path),
        metadata={"generated_at": "2026-05-25T10:00:00Z", "n_quotes": 10, "n_trades": 5},
    )

    kappa = json.loads(kappa_path.read_text(encoding="utf-8"))["ETH"]
    lambdas = json.loads(lambda_path.read_text(encoding="utf-8"))["ETH"]

    assert kappa["schema_version"] == PARAM_SCHEMA_VERSION
    assert kappa["status"] == "ok"
    assert kappa["lambda_source"] == "lambda0_fit"
    assert kappa["unit"]["kappa"] == "1/USDC"
    assert lambdas["schema_version"] == PARAM_SCHEMA_VERSION
    assert lambdas["status"] == "ok"
    assert lambdas["lambda_source"] == "lambda0_fit"


def test_epsilon_writer_includes_diagnostics(tmp_path):
    path = tmp_path / "epsilon.json"
    save_epsilon_to_json(
        0.01,
        0.02,
        "ETH",
        filename=str(path),
        metadata={"generated_at": "2026-05-25T10:00:00Z", "n_buy_events": 4, "n_sell_events": 5},
    )

    data = json.loads(path.read_text(encoding="utf-8"))["ETH"]
    assert data["schema_version"] == PARAM_SCHEMA_VERSION
    assert data["status"] == "ok"
    assert data["estimator"] == "trimmed_mean"
    assert data["unit"] == "USDC"
    assert data["n_buy_events"] == 4


def test_raw_lambda_writer_is_monitoring_only(tmp_path):
    path = tmp_path / "lambda_trades.json"
    save_lambda_to_json(
        0.3,
        0.4,
        "ETH",
        filename=str(path),
        metadata={"generated_at": "2026-05-25T10:00:00Z", "n_trades_total": 9},
    )

    data = json.loads(path.read_text(encoding="utf-8"))["ETH"]
    assert data["schema_version"] == PARAM_SCHEMA_VERSION
    assert data["status"] == "ok"
    assert data["lambda_source"] == "lambda_raw"
    assert data["unit"] == "events_per_second"


def test_post_only_probe_defaults_to_alo_without_network():
    params = alo_order_params()

    assert params["timeInForce"] == "Alo"
    assert params["postOnly"] is True
    assert "safe_default" in render_plan("ETH/USDC:USDC")


def test_post_only_crossing_evidence_requires_cancel_or_reject():
    ok, reasons = evaluate_crossing_result(
        {
            "submitted_params": alo_order_params(),
            "order_status": "open",
            "filled": 0.0,
        }
    )

    assert not ok
    assert "crossing_status_not_rejected_or_cancelled:open" in reasons


def test_post_only_crossing_evidence_accepts_zero_fill_rejection():
    ok, reasons = evaluate_crossing_result(
        {
            "submitted_params": alo_order_params(),
            "order_status": "rejected",
            "filled": 0.0,
        }
    )

    assert ok
    assert reasons == []


def test_post_only_passive_evidence_rejects_taker_liquidity():
    ok, reasons = evaluate_passive_result(
        {
            "submitted_params": alo_order_params(),
            "order_status": "closed",
            "filled": 0.01,
            "raw_result": {"info": {"liquidity": "taker"}},
        }
    )

    assert not ok
    assert "passive_liquidity_taker" in reasons


def test_post_only_evidence_report_requires_both_results():
    report = evaluate_evidence(
        {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "submitted_params": alo_order_params(),
            "order_status": "rejected",
            "filled": 0.0,
        },
        None,
    )

    assert not report["ok"]
    assert "missing_passive_result" in report["reasons"]


def test_post_only_evidence_report_passes_complete_safe_results():
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = evaluate_evidence(
        {
            "generated_at": generated_at,
            "submitted_params": alo_order_params(),
            "order_status": "rejected",
            "filled": 0.0,
        },
        {
            "generated_at": generated_at,
            "submitted_params": alo_order_params(),
            "order_status": "canceled",
            "filled": 0.0,
        },
    )

    assert report["ok"]
    assert report["reasons"] == []
    assert report["crossing"]["age_ok"] is True
    assert report["passive"]["age_ok"] is True


def test_post_only_evidence_report_rejects_missing_artifact_timestamps():
    report = evaluate_evidence(
        {
            "submitted_params": alo_order_params(),
            "order_status": "rejected",
            "filled": 0.0,
        },
        {
            "submitted_params": alo_order_params(),
            "order_status": "canceled",
            "filled": 0.0,
        },
    )

    assert not report["ok"]
    assert "crossing_missing_generated_at" in report["reasons"]
    assert "passive_missing_generated_at" in report["reasons"]


def test_post_only_evidence_report_rejects_stale_artifacts():
    report = evaluate_evidence(
        {
            "generated_at": "2026-05-24T11:59:00Z",
            "submitted_params": alo_order_params(),
            "order_status": "rejected",
            "filled": 0.0,
        },
        {
            "generated_at": "2026-05-25T11:59:59Z",
            "submitted_params": alo_order_params(),
            "order_status": "canceled",
            "filled": 0.0,
        },
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
        max_evidence_age_seconds=86_400,
    )

    assert not report["ok"]
    assert "crossing_evidence_stale:86460.0s>max_86400.0s" in report["reasons"]
    assert report["crossing"]["age_ok"] is False
    assert report["passive"]["age_ok"] is True


def test_post_only_evaluator_accepts_direct_sdk_alo_rejection():
    ok, reasons = evaluate_crossing_result(
        {
            "sdk_order_args": {"order_type": {"limit": {"tif": "Alo"}}},
            "classification": {
                "ok": True,
                "alo_rejected": True,
                "filled_total": 0.0,
                "reasons": [],
            },
        }
    )

    assert ok
    assert reasons == []


def test_post_only_evaluator_accepts_direct_sdk_resting_order():
    ok, reasons = evaluate_passive_result(
        {
            "sdk_order_args": {"order_type": {"limit": {"tif": "Alo"}}},
            "classification": {
                "ok": True,
                "saw_resting": True,
                "filled_total": 0.0,
                "reasons": [],
            },
        }
    )

    assert ok
    assert reasons == []


def test_post_only_evaluator_rejects_direct_sdk_taker_fill():
    ok, reasons = evaluate_passive_result(
        {
            "sdk_order_args": {"order_type": {"limit": {"tif": "Alo"}}},
            "classification": {
                "ok": False,
                "saw_filled": True,
                "filled_total": 0.01,
                "reasons": ["taker_liquidity_seen"],
            },
        }
    )

    assert not ok
    assert "passive_liquidity_taker" in reasons


def test_config_uses_runtime_supported_tif_and_keeps_force_entry_off():
    config = json.loads((ROOT / "user_data" / "config.json").read_text(encoding="utf-8"))

    assert config["order_time_in_force"] == {"entry": "GTC", "exit": "GTC"}
    assert config["dry_run"] is True
    assert config["force_entry_enable"] is False


def test_dry_run_disabled_log_evaluator_detects_order_creation():
    ok, reason, evidence = evaluate_logs(
        "Runmode set to dry_run\n"
        "Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}\n"
        "Order dry_run_buy_ETH was created\n"
    )

    assert not ok
    assert reason == "order_creation_detected"
    assert evidence


def test_dry_run_disabled_log_evaluator_accepts_locked_start():
    ok, reason, evidence = evaluate_logs(
        "Runmode set to dry_run\n"
        "Using resolved strategy Market_Making\n"
        "Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}\n"
    )

    assert ok
    assert reason == "ok"
    assert evidence == []


def test_dry_run_disabled_log_evaluator_rejects_runtime_errors():
    ok, reason, evidence = evaluate_logs(
        "Runmode set to dry_run\n"
        "Configuration error: Time in force policies are not supported for Hyperliquid yet.\n"
    )

    assert not ok
    assert reason == "runtime_error_detected"
    assert evidence


def test_recent_health_events_filters_by_gate_start(tmp_path):
    log = tmp_path / "mm_debug.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-05-25T09:59:59Z", "event": "health", "trading_enabled": False}),
                json.dumps({"ts": "2026-05-25T10:00:01Z", "event": "quote_decision"}),
                json.dumps({"ts": "2026-05-25T10:00:02.123Z", "event": "health", "trading_enabled": False}),
                "not json",
            ]
        ),
        encoding="utf-8",
    )

    events = recent_health_events(log, datetime(2026, 5, 25, 10, 0, 2, tzinfo=timezone.utc))

    assert len(events) == 1
    assert events[0]["ts"] == "2026-05-25T10:00:02.123Z"


def test_dry_run_enabled_gate_writes_safe_temp_config_and_params(tmp_path):
    base_config = tmp_path / "base_config.json"
    output_config = tmp_path / "gate_config.json"
    param_dir = tmp_path / "params"
    base_config.write_text(
        json.dumps(
            {
                "dry_run": True,
                "force_entry_enable": True,
                "stake_amount": "unlimited",
                "api_server": {"forcebuy_enable": True, "force_entry_enable": True, "Force_entry": True},
            }
        ),
        encoding="utf-8",
    )

    write_gate_params(param_dir)
    write_gate_config(base_config, output_config, "/freqtrade/user_data/logs/mm_gate_enabled_params")

    config = json.loads(output_config.read_text(encoding="utf-8"))
    kappa = json.loads((param_dir / "kappa.json").read_text(encoding="utf-8"))["ETH"]
    epsilon = json.loads((param_dir / "epsilon.json").read_text(encoding="utf-8"))["ETH"]
    lambdas = json.loads((param_dir / "lambda.json").read_text(encoding="utf-8"))["ETH"]

    assert config["dry_run"] is True
    assert config["force_entry_enable"] is False
    assert config["stake_amount"] == 25
    assert config["market_making"]["trading_enabled"] is True
    assert config["market_making"]["disable_param_refresh"] is True
    assert config["market_making"]["param_dir"] == "/freqtrade/user_data/logs/mm_gate_enabled_params"
    assert config["market_making"]["max_param_age_seconds"] == 300
    assert config["market_making"]["max_collector_age_seconds"] == 180
    assert config["api_server"]["forcebuy_enable"] is False
    assert kappa["schema_version"] == PARAM_SCHEMA_VERSION
    assert kappa["status"] == "ok"
    assert epsilon["n_buy_events"] >= 1
    assert lambdas["lambda_source"] == "lambda0_fit"


def test_dry_run_enabled_gate_requires_health_and_accepted_quote():
    health = {
        "event": "health",
        "trading_enabled": True,
        "collector_fresh": True,
        "params_fresh": True,
        "hjb_fresh": True,
    }
    quote = {
        "event": "quote_decision",
        "decision": "accept",
        "reason": "ok",
        "side": "bid",
    }

    ok, reason, evidence = evaluate_enabled_gate("Runmode set to dry_run", [health, quote])

    assert ok
    assert reason == "ok"
    assert evidence == []


def test_dry_run_enabled_gate_rejects_missing_accept():
    health = {
        "event": "health",
        "trading_enabled": True,
        "collector_fresh": True,
        "params_fresh": True,
        "hjb_fresh": True,
    }

    ok, reason, evidence = evaluate_enabled_gate("Runmode set to dry_run", [health])

    assert not ok
    assert reason == "no_accepted_quote_decision"
    assert evidence == []


def test_hl_data_validator_reports_bad_parquet(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet", encoding="utf-8")

    result = validate_parquet_file(bad, {"timestamp"})

    assert not result.ok
    assert result.error


def test_hl_data_validator_reads_row_timestamp(tmp_path):
    shard = tmp_path / "prices.parquet"
    ts = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    pd.DataFrame({"timestamp": [ts.timestamp()]}).to_parquet(shard, index=False)

    result = validate_parquet_file(shard, {"timestamp"})

    assert result.ok
    assert result.latest_timestamp == "2026-05-25T12:00:00Z"
    assert latest_parquet_timestamp(shard) == ts


def test_hl_data_validator_reports_missing_streams(tmp_path):
    payload = validate_symbol(tmp_path, "ETH")

    assert not payload["ok"]
    assert set(payload["missing_streams"]) == {"prices", "trades", "orderbooks"}


def test_hl_data_validator_can_fail_stale_data(tmp_path):
    payload = validate_symbol(tmp_path, "ETH", max_age_seconds=30)

    assert not payload["fresh"]
    assert not payload["ok"]


def test_hl_data_validator_uses_data_timestamp_not_file_mtime(tmp_path):
    now = datetime.now(timezone.utc)
    stale_ts = now - timedelta(seconds=120)
    symbol_dir = tmp_path / "ETH"
    for stream in ("prices", "trades", "orderbooks"):
        stream_dir = symbol_dir / stream
        stream_dir.mkdir(parents=True)
        payload = {"timestamp": [stale_ts.timestamp()]}
        if stream == "trades":
            payload["price"] = [100.0]
        shard = stream_dir / f"{stream}.parquet"
        pd.DataFrame(payload).to_parquet(shard, index=False)
        os.utime(shard, None)

    result = validate_symbol(tmp_path, "ETH", max_age_seconds=30)

    assert not result["fresh"]
    assert not result["ok"]
    assert result["freshness_age_seconds"] >= 100
    assert result["latest_file_age_seconds"] < 30
