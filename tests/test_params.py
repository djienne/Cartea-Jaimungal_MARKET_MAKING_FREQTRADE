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
from validate_hl_data import latest_parquet_timestamp, validate_parquet_file, validate_symbol  # noqa: E402


def test_atomic_write_json_round_trip(tmp_path):
    path = tmp_path / "params.json"
    atomic_write_json(path, {"ETH": {"schema_version": PARAM_SCHEMA_VERSION}})

    assert load_json_object(path) == {"ETH": {"schema_version": PARAM_SCHEMA_VERSION}}
    assert not path.with_suffix(".json.tmp").exists()


def test_kappa_lambda_writer_uses_schema_v4_and_mo_survival_fit(tmp_path):
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
        raw_values={"kappa+_raw": 2.5, "kappa-_raw": 3.5, "lambda+_raw": 0.15, "lambda-_raw": 0.25},
    )

    kappa = json.loads(kappa_path.read_text(encoding="utf-8"))["ETH"]
    lambdas = json.loads(lambda_path.read_text(encoding="utf-8"))["ETH"]

    assert kappa["schema_version"] == PARAM_SCHEMA_VERSION
    assert kappa["status"] == "ok"
    assert kappa["lambda_source"] == "mo_survival_fit"
    assert kappa["unit"]["kappa"] == "1/USDC"
    assert kappa["kappa+"] == 2.0
    assert kappa["kappa+_raw"] == 2.5
    assert kappa["lambda+_raw"] == 0.15
    assert lambdas["schema_version"] == PARAM_SCHEMA_VERSION
    assert lambdas["status"] == "ok"
    assert lambdas["lambda_source"] == "mo_survival_fit"
    assert lambdas["lambda-_raw"] == 0.25


def test_kappa_lambda_writer_defaults_raw_to_primary(tmp_path):
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
        metadata={"generated_at": "2026-05-25T10:00:00Z"},
    )

    kappa = json.loads(kappa_path.read_text(encoding="utf-8"))["ETH"]
    assert kappa["kappa+_raw"] == 2.0
    assert kappa["kappa-_raw"] == 3.0
    assert kappa["lambda+_raw"] == 0.1
    assert kappa["lambda-_raw"] == 0.2


def test_epsilon_writer_includes_diagnostics(tmp_path):
    path = tmp_path / "epsilon.json"
    save_epsilon_to_json(
        0.01,
        0.02,
        "ETH",
        filename=str(path),
        metadata={"generated_at": "2026-05-25T10:00:00Z", "n_buy_events": 4, "n_sell_events": 5},
        raw_values={"epsilon+_raw": 0.015, "epsilon-_raw": 0.025},
    )

    data = json.loads(path.read_text(encoding="utf-8"))["ETH"]
    assert data["schema_version"] == PARAM_SCHEMA_VERSION
    assert data["status"] == "ok"
    assert data["estimator"] == "mean_at_arrival"
    assert data["unit"] == "USDC"
    assert data["n_buy_events"] == 4
    assert data["epsilon+_raw"] == 0.015
    assert data["epsilon-_raw"] == 0.025


def test_epsilon_writer_defaults_raw_to_primary(tmp_path):
    path = tmp_path / "epsilon.json"
    save_epsilon_to_json(0.01, 0.02, "ETH", filename=str(path), metadata={"generated_at": "2026-05-25T10:00:00Z"})
    data = json.loads(path.read_text(encoding="utf-8"))["ETH"]
    assert data["epsilon+_raw"] == 0.01
    assert data["epsilon-_raw"] == 0.02


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
