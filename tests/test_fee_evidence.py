from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_fee_evidence import build_fee_evidence_report, read_jsonl_events  # noqa: E402


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fresh_ts() -> str:
    return iso_utc(datetime.now(timezone.utc))


def fee_snapshot_event(**overrides):
    ts = overrides.pop("ts", fresh_ts())
    snapshot = {
        "strategy_maker_fee_rate": 0.00015,
        "config_fee_rate": 0.00015,
        "config_fee_matches_strategy": True,
        "exchange_fee_source": "fetch_trading_fee",
        "exchange_maker_fee_rate": 0.00015,
        "exchange_taker_fee_rate": 0.00045,
        "exchange_maker_fee_matches_strategy": True,
    }
    snapshot.update(overrides)
    event = {"event": "health", "fee_snapshot": snapshot}
    if ts is not None:
        event["ts"] = ts
    return event


def maker_fill_event(**overrides):
    event = {
        "event": "fill",
        "ts": fresh_ts(),
        "liquidity": "maker",
        "quote_side": "bid",
        "order_type": "limit",
        "tif": "Alo",
        "expected_fee_rate": 0.00015,
        "actual_fee_paid": 0.015,
        "actual_fee_rate": 0.00015,
        "order_id": "maker-1",
    }
    event.update(overrides)
    return event


def test_fee_evidence_passes_with_exchange_and_maker_fill_fee():
    report = build_fee_evidence_report(
        [
            fee_snapshot_event(),
            maker_fill_event(),
        ]
    )

    assert report["ok"] is True
    assert report["reasons"] == []
    assert report["fee_snapshots"]["exchange_matches"] == 1
    assert report["fills"]["maker_actual_fee_matches"] == 1


def test_fee_evidence_requires_exchange_fee_and_actual_maker_fill():
    report = build_fee_evidence_report(
        [
            fee_snapshot_event(
                exchange_fee_source="unavailable",
                exchange_maker_fee_rate=None,
                exchange_taker_fee_rate=None,
                exchange_maker_fee_matches_strategy=None,
            )
        ]
    )

    assert report["ok"] is False
    assert "exchange_fee_not_proven" in report["reasons"]
    assert "insufficient_maker_fills:0<min_1" in report["reasons"]
    assert "insufficient_actual_maker_fee_matches:0<min_1" in report["reasons"]


def test_fee_evidence_rejects_taker_or_mismatched_fee():
    report = build_fee_evidence_report(
        [
            fee_snapshot_event(exchange_maker_fee_rate=0.001, exchange_maker_fee_matches_strategy=False),
            maker_fill_event(actual_fee_rate=0.001, order_id="bad-maker"),
            {
                "event": "fill",
                "ts": fresh_ts(),
                "liquidity": "taker",
                "actual_fee_rate": 0.00045,
                "order_id": "taker-1",
            },
        ]
    )

    assert report["ok"] is False
    assert "exchange_fee_not_proven" in report["reasons"]
    assert "fee_snapshot_mismatch" in report["reasons"]
    assert "actual_maker_fee_mismatches:1" in report["reasons"]
    assert "taker_fills_seen:1" in report["reasons"]


def test_fee_evidence_requires_timestamps_on_proof_events():
    report = build_fee_evidence_report(
        [
            fee_snapshot_event(ts=None),
            maker_fill_event(ts=None, order_id="timestampless-maker"),
        ]
    )

    assert report["ok"] is False
    assert "fee_snapshot_timestamp_missing:1" in report["reasons"]
    assert "actual_maker_fee_timestamp_missing:1" in report["reasons"]
    assert "exchange_fee_not_proven" in report["reasons"]
    assert "insufficient_actual_maker_fee_matches:0<min_1" in report["reasons"]
    assert report["fee_snapshots"]["exchange_matches"] == 0
    assert report["fills"]["maker_actual_fee_matches"] == 0


def test_fee_evidence_rejects_stale_proof_events():
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
    stale_ts = iso_utc(now - timedelta(seconds=86_401))

    report = build_fee_evidence_report(
        [
            fee_snapshot_event(ts=stale_ts),
            maker_fill_event(ts=stale_ts, order_id="stale-maker"),
        ],
        now=now,
        max_evidence_age_seconds=86_400,
    )

    assert report["ok"] is False
    assert "fee_snapshot_stale:1" in report["reasons"]
    assert "actual_maker_fee_stale:1" in report["reasons"]
    assert "exchange_fee_not_proven" in report["reasons"]
    assert "insufficient_actual_maker_fee_matches:0<min_1" in report["reasons"]
    assert report["fee_snapshots"]["stale"] == 1
    assert report["fills"]["maker_actual_fee_stale"] == 1
    assert report["mismatches"] == []


def test_fee_evidence_requires_fill_accounting_fields():
    report = build_fee_evidence_report(
        [
            fee_snapshot_event(),
            {
                "event": "fill",
                "ts": fresh_ts(),
                "liquidity": "maker",
                "actual_fee_rate": 0.00015,
                "order_id": "partial-maker",
            },
        ]
    )

    assert report["ok"] is False
    assert "maker_fill_quote_side_invalid:1" in report["reasons"]
    assert "maker_fill_order_type_invalid:1" in report["reasons"]
    assert "maker_fill_tif_invalid:1" in report["reasons"]
    assert "maker_fill_expected_fee_invalid:1" in report["reasons"]
    assert "maker_fill_actual_fee_paid_missing:1" in report["reasons"]


def test_read_jsonl_events_skips_invalid_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "health"}),
                "not-json",
                json.dumps(["not", "an", "event"]),
                json.dumps({"event": "fill"}),
            ]
        ),
        encoding="utf-8",
    )

    assert read_jsonl_events(path) == [{"event": "health"}, {"event": "fill"}]
