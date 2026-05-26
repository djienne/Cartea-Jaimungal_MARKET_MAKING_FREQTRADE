from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_replay_from_logs import build_calibration_report, read_jsonl_events  # noqa: E402


def test_replay_log_calibration_matches_fills_to_quote_depths(tmp_path):
    log_path = tmp_path / "mm_debug.jsonl"
    events = [
        {
            "event": "quote_decision",
            "quote_id": "q-bid-1",
            "ts": "2026-05-25T10:00:00Z",
            "pair": "ETH/USDC:USDC",
            "side": "bid",
            "decision": "accept",
            "rounded_price": 99.0,
            "mid": 100.0,
            "bps": 100.0,
        },
        {
            "event": "quote_decision",
            "ts": "2026-05-25T10:00:01Z",
            "pair": "ETH/USDC:USDC",
            "side": "ask",
            "decision": "accept",
            "rounded_price": 101.0,
            "mid": 100.0,
            "bps": 100.0,
        },
        {
            "event": "fill",
            "quote_id": "q-bid-1",
            "ts": "2026-05-25T10:00:10Z",
            "pair": "ETH/USDC:USDC",
            "quote_side": "bid",
            "price": 99.0,
            "liquidity": "maker",
            "actual_fee_rate": 0.00015,
        },
        {
            "event": "fill_markout",
            "horizon_ms": 100,
            "markout_usdc": 0.01,
        },
        {
            "event": "fill_markout",
            "horizon_ms": 100,
            "markout_usdc": 0.03,
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    report = build_calibration_report(
        read_jsonl_events([log_path]),
        bucket_bps=25.0,
        min_quotes=2,
        min_fills=1,
    )

    assert report["usable_for_calibration"] is True
    assert report["accepted_quotes_by_side"] == {"bid": 1, "ask": 1}
    assert report["inputs"]["accepted_quotes_with_quote_id"] == 1
    assert report["inputs"]["fills_with_quote_id"] == 1
    assert report["fills_by_side"] == {"bid": 1}
    assert report["matched_fills_by_depth"] == {"bid:100.00-125.00bps": 1}
    assert report["matched_fills_by_quote_id"] == {"q-bid-1": 1}
    assert report["fill_probability_by_side"]["bid"] == 1.0
    assert report["fill_probability_by_side"]["ask"] == 0.0
    assert report["fill_probability_by_depth"]["bid:100.00-125.00bps"] == 1.0
    assert report["maker_ratio"] == 1.0
    assert report["actual_fee_rate"]["mean"] == 0.00015
    assert report["actual_fee_rate"]["outlier_count"] == 0
    assert report["markout_summary"]["100"]["mean_usdc"] == 0.02


def test_replay_log_calibration_marks_small_samples_unusable():
    report = build_calibration_report([], min_quotes=2, min_fills=1)

    assert report["usable_for_calibration"] is False
    assert "insufficient_accepted_quotes:0<min_2" in report["reasons"]
    assert "insufficient_maker_fills:0<min_1" in report["reasons"]


def test_replay_log_calibration_rejects_taker_and_unmatched_fills():
    events = [
        {
            "event": "quote_decision",
            "ts": "2026-05-25T10:00:00Z",
            "pair": "ETH/USDC:USDC",
            "side": "bid",
            "decision": "accept",
            "rounded_price": 99.0,
            "mid": 100.0,
        },
        {
            "event": "fill",
            "ts": "2026-05-25T10:10:00Z",
            "pair": "ETH/USDC:USDC",
            "quote_side": "bid",
            "price": 90.0,
            "liquidity": "taker",
            "actual_fee_rate": 1.0,
        },
    ]

    report = build_calibration_report(events, min_quotes=1, min_fills=0)

    assert report["usable_for_calibration"] is False
    assert "taker_fills_seen:1" in report["reasons"]
    assert "no_fills_matched_to_quotes" in report["reasons"]
    assert report["actual_fee_rate"]["count"] == 0
    assert report["actual_fee_rate"]["outlier_count"] == 1


def test_replay_log_calibration_prefers_exact_quote_id_match():
    events = [
        {
            "event": "quote_decision",
            "quote_id": "older-quote",
            "ts": "2026-05-25T10:00:00Z",
            "pair": "ETH/USDC:USDC",
            "side": "bid",
            "decision": "accept",
            "rounded_price": 99.0,
            "mid": 100.0,
            "bps": 100.0,
        },
        {
            "event": "quote_decision",
            "quote_id": "newer-quote",
            "ts": "2026-05-25T10:00:05Z",
            "pair": "ETH/USDC:USDC",
            "side": "bid",
            "decision": "accept",
            "rounded_price": 99.0,
            "mid": 99.1,
            "bps": 10.0,
        },
        {
            "event": "fill",
            "quote_id": "older-quote",
            "ts": "2026-05-25T10:00:10Z",
            "pair": "ETH/USDC:USDC",
            "quote_side": "bid",
            "price": 99.0,
            "liquidity": "maker",
        },
    ]

    report = build_calibration_report(events, bucket_bps=25.0, min_quotes=2, min_fills=1)

    assert report["usable_for_calibration"] is True
    assert report["matched_fills_by_depth"] == {"bid:100.00-125.00bps": 1}
    assert report["matched_fills_by_quote_id"] == {"older-quote": 1}
