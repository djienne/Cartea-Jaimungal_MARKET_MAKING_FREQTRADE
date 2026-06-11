from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_dry_run_quality import build_dry_run_quality_report, read_jsonl_events  # noqa: E402


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def gate_report(start: datetime, *, passed: bool = True, runtime: float = 240.0) -> dict:
    return {
        "passed": passed,
        "reason": "ok" if passed else "failed",
        "started_at": iso(start),
        "container_wait": {"waited_seconds": runtime},
    }


def dry_run_events(start: datetime) -> list[dict]:
    quote_id = "quote-1"
    return [
        {
            "event": "health",
            "ts": iso(start + timedelta(seconds=5)),
            "trading_enabled": True,
            "dry_run": True,
            "params_fresh": True,
            "collector_fresh": True,
            "book_fresh": True,
            "hjb_fresh": True,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
        },
        {
            "event": "quote_decision",
            "quote_id": quote_id,
            "ts": iso(start + timedelta(seconds=70)),
            "decision": "accept",
            "side": "bid",
            "trading_enabled": True,
            "dry_run": True,
            "params_fresh": True,
            "collector_fresh": True,
            "book_fresh": True,
            "fee_snapshot": {"fee_agreement_ok": True},
            "bps": 5.5,
            "custom_price_distance_ratio": 0.00055,
            "rounded_price": 99.5,
            "best_bid": 99.0,
            "best_ask": 100.0,
        },
        {
            "event": "order_attempt_accepted",
            "quote_id": quote_id,
            "ts": iso(start + timedelta(seconds=75)),
            "trading_enabled": True,
            "dry_run": True,
            "rate": 99.5,
            "amount": 0.01,
        },
        {
            "event": "health",
            "ts": iso(start + timedelta(seconds=180)),
            "trading_enabled": True,
            "dry_run": True,
            "params_fresh": True,
            "collector_fresh": True,
            "book_fresh": True,
            "hjb_fresh": True,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.05,
        },
    ]


def test_dry_run_quality_passes_reasonable_quotes_size_and_pnl():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)

    report = build_dry_run_quality_report(
        dry_run_events(start),
        gate_report=gate_report(start),
        min_runtime_seconds=180,
        min_event_span_seconds=60,
    )

    assert report["ok"] is True
    assert report["conclusion"] == "dry_run_quotes_and_sizing_passed_but_no_fill_profit_evidence"
    assert report["reasons"] == []
    assert report["accepted_quotes"] == 1
    assert report["accepted_order_attempts"] == 1
    assert report["quality_verdict"]["break_even_or_profitable"] is True
    assert report["quality_verdict"]["small_profit_observed"] is True
    assert report["quality_verdict"]["dry_run_alone_is_live_safe"] is False
    assert report["quote_quality"]["max_depth_bps"] == 5.5
    assert report["order_sizing"]["max_notional_usdc"] == 0.995
    assert report["pnl"]["final_total_pnl_usdc"] == 0.05
    assert report["loss_velocity_usdc_per_hour"] == 0.0


def test_dry_run_quality_accepts_single_quote_when_it_fills_and_pnl_is_bounded():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    events.insert(
        3,
        {
            "event": "fill",
            "ts": iso(start + timedelta(seconds=76)),
            "trading_enabled": True,
            "dry_run": True,
            "quote_side": "bid",
            "amount": 0.01,
            "price": 99.5,
            "liquidity": "unknown",
        },
    )

    report = build_dry_run_quality_report(
        events,
        gate_report=gate_report(start),
        min_accepted_quotes=2,
        min_order_attempts=2,
    )

    assert report["ok"] is True
    assert report["conclusion"] == "dry_run_quotes_sizing_and_small_profit_evidence_with_fills"
    assert report["dry_run_fills"] == 1
    assert report["accepted_quotes"] == 1
    assert report["accepted_order_attempts"] == 1


def test_dry_run_quality_rejects_short_runtime_and_missing_orders():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = [event for event in dry_run_events(start) if event["event"] != "order_attempt_accepted"]

    report = build_dry_run_quality_report(
        events,
        gate_report=gate_report(start, runtime=30.0),
        min_runtime_seconds=180,
        min_order_attempts=1,
    )

    assert report["ok"] is False
    assert report["conclusion"] == "dry_run_quality_gate_not_passed"
    assert "runtime_too_short:30.0<min_180.0" in report["reasons"]
    assert "insufficient_quote_linked_order_attempts:0<min_1" in report["reasons"]


def test_dry_run_quality_ignores_boundary_orders_when_enough_matched_orders():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    events.insert(
        1,
        {
            "event": "order_attempt_accepted",
            "quote_id": "quote-before-window",
            "ts": iso(start + timedelta(seconds=20)),
            "trading_enabled": True,
            "dry_run": True,
            "rate": 99.5,
            "amount": 0.01,
        },
    )

    report = build_dry_run_quality_report(events, gate_report=gate_report(start))

    assert report["ok"] is True
    assert report["accepted_order_attempts"] == 1
    assert report["total_order_attempts"] == 2
    assert report["order_sizing"]["ignored"] == {"order_quote_id_outside_quality_window": 1}


def test_dry_run_quality_rejects_unreasonable_quote_distance_and_depth():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    quote = next(event for event in events if event["event"] == "quote_decision")
    quote["custom_price_distance_ratio"] = 0.02
    quote["bps"] = 95.0  # beyond the 80 bps cap

    report = build_dry_run_quality_report(events, gate_report=gate_report(start))

    assert report["ok"] is False
    assert "accepted_quote_depth_too_wide:1" in report["reasons"]
    assert "accepted_quote_too_far_from_proposed:1" in report["reasons"]


def test_dry_run_quality_rejects_quote_tighter_than_floor():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    quote = next(event for event in events if event["event"] == "quote_decision")
    quote["bps"] = 1.5  # below the 3 bps floor: clamps not applied / fee-losing

    report = build_dry_run_quality_report(events, gate_report=gate_report(start))

    assert report["ok"] is False
    assert "accepted_quote_depth_too_tight:1" in report["reasons"]


def test_dry_run_quality_tolerates_order_amount_float_dust():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    order = next(event for event in events if event["event"] == "order_attempt_accepted")
    order["amount"] = 0.010000000000000002  # freqtrade stake/rate float dust

    report = build_dry_run_quality_report(
        events, gate_report=gate_report(start), max_order_amount_units=0.01, max_order_notional_usdc=25.0
    )

    assert "order_amount_too_large:1" not in report["reasons"]

    order["amount"] = 0.02  # a real violation still fails
    report = build_dry_run_quality_report(
        events, gate_report=gate_report(start), max_order_amount_units=0.01, max_order_notional_usdc=25.0
    )
    assert "order_amount_too_large:1" in report["reasons"]


def test_dry_run_quality_tolerates_rare_collector_read_races():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    events.append({"event": "collector_data_read_error", "ts": iso(start + timedelta(seconds=80)), "path": "x.parquet"})

    report = build_dry_run_quality_report(events, gate_report=gate_report(start))

    assert report["ok"] is True
    assert report["tolerated_error_events"] == {"collector_data_read_error": 1}
    assert "collector_data_read_error:1" not in report["reasons"]

    # Persistent read errors still fail (above the tolerance).
    for i in (3, 4, 5):
        events.append({"event": "collector_data_read_error", "ts": iso(start + timedelta(seconds=80 + i)), "path": "x.parquet"})
    report = build_dry_run_quality_report(events, gate_report=gate_report(start))
    assert report["ok"] is False
    assert "collector_data_read_error:4" in report["reasons"]


def test_dry_run_quality_collector_read_error_fails_when_collector_went_stale():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    events.append({"event": "collector_data_read_error", "ts": iso(start + timedelta(seconds=80)), "path": "x.parquet"})
    health = next(event for event in events if event["event"] == "health")
    health["collector_fresh"] = False

    report = build_dry_run_quality_report(events, gate_report=gate_report(start))

    assert report["ok"] is False
    assert "collector_data_read_error:1" in report["reasons"]


def test_dry_run_quality_reports_clamp_and_calibration_diagnostics_informational():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    quote = next(event for event in events if event["event"] == "quote_decision")
    quote["clamped"] = "floor"
    quote["quote_outside_calibrated_range"] = True

    report = build_dry_run_quality_report(events, gate_report=gate_report(start))

    quality = report["quote_quality"]
    assert quality["clamp_counts"]["floor"] == 1
    assert quality["outside_calibrated_range_count"] == 1
    assert quality["outside_calibrated_range_fraction"] == 1.0
    # Diagnostics never produce failure reasons by themselves.
    assert not any("calibrated" in reason or "clamp" in reason for reason in report["reasons"])


def test_dry_run_quality_rejects_large_order_and_bad_pnl():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    order = next(event for event in events if event["event"] == "order_attempt_accepted")
    order["rate"] = 3000.0
    order["amount"] = 0.02
    events[-1]["unrealized_pnl"] = -2.5

    report = build_dry_run_quality_report(
        events,
        gate_report=gate_report(start),
        max_order_notional_usdc=25.0,
        max_order_amount_units=0.01,
        max_loss_usdc=1.0,
    )

    assert report["ok"] is False
    assert "order_amount_too_large:1" in report["reasons"]
    assert "order_notional_too_large:1" in report["reasons"]
    assert "loss_too_large:-2.500000<min_-1.000000" in report["reasons"]
    assert "loss_rate_too_fast:51.428571>max_6.000000_usdc_per_hour" in report["reasons"]


def test_dry_run_quality_rejects_fast_loss_even_when_absolute_loss_is_small():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    events = dry_run_events(start)
    events[-1]["unrealized_pnl"] = -0.40

    report = build_dry_run_quality_report(
        events,
        gate_report=gate_report(start),
        max_loss_usdc=1.0,
        max_loss_rate_usdc_per_hour=5.0,
    )

    assert report["ok"] is False
    assert "loss_too_large" not in " ".join(report["reasons"])
    assert "loss_rate_too_fast:8.228571>max_5.000000_usdc_per_hour" in report["reasons"]


def test_read_jsonl_events_filters_since_and_bad_lines(tmp_path):
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    path = tmp_path / "mm_debug.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event":"health","ts":"2026-05-24T00:00:00Z"}',
                "not-json",
                '{"event":"health","ts":"2026-05-25T00:00:01Z"}',
            ]
        ),
        encoding="utf-8",
    )

    assert read_jsonl_events(path, since=start) == [{"event": "health", "ts": "2026-05-25T00:00:01Z"}]
