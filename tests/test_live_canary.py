from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_live_canary import build_live_canary_report, read_jsonl_events  # noqa: E402


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def ok_report(now: datetime) -> dict:
    return {"ok": True, "reasons": [], "generated_at": iso(now)}


def canary_session(session_id: str, start: datetime, *, minutes: int = 2) -> list[dict]:
    common = {"session_id": session_id}
    return [
        {
            "event": "health",
            "ts": iso(start),
            "trading_enabled": True,
            "dry_run": False,
            "post_only_verified": True,
            "kill_on_taker_fill": True,
            "stake_amount": 25,
            "max_daily_loss_usdc": 20,
            "symbol": "ETH",
            **common,
        },
        {
            "event": "quote_decision",
            "ts": iso(start + timedelta(seconds=10)),
            "decision": "accept",
            "params_fresh": True,
            "collector_fresh": True,
            "book_fresh": True,
            "post_only": True,
            "post_only_verified": True,
            "fee_snapshot": {"fee_agreement_ok": True},
            **common,
        },
        {
            "event": "fill",
            "ts": iso(start + timedelta(seconds=20)),
            "liquidity": "maker",
            "actual_fee_rate": 0.00015,
            **common,
        },
        {
            "event": "health",
            "ts": iso(start + timedelta(minutes=minutes)),
            "trading_enabled": True,
            "dry_run": False,
            "post_only_verified": True,
            "kill_on_taker_fill": True,
            "stake_amount": 25,
            "max_daily_loss_usdc": 20,
            "symbol": "ETH",
            **common,
        },
    ]


def test_live_canary_report_passes_when_all_evidence_is_present():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=3)
    events = []
    for idx in range(3):
        events.extend(canary_session(f"s{idx}", start + timedelta(hours=idx), minutes=2))

    report = build_live_canary_report(
        events,
        post_only_report=ok_report(now),
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=3,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        now=now,
    )

    assert report["ok"] is True
    assert report["reasons"] == []
    assert report["eligible_sessions"] == 3
    assert report["fills"]["maker"] == 3
    assert report["fills"]["taker"] == 0


def test_live_canary_report_requires_prior_gates_and_sessions():
    report = build_live_canary_report(
        [],
        post_only_report={"ok": False, "reasons": ["missing_crossing_result"]},
        fee_report={"ok": False, "reasons": ["exchange_fee_not_proven"]},
        replay_report={"ok": False, "reasons": ["no_maker_fills"]},
        min_sessions=3,
        min_session_minutes=1,
    )

    assert report["ok"] is False
    assert "post_only_gate_not_passed" in report["reasons"]
    assert "fee_gate_not_passed" in report["reasons"]
    assert "replay_gate_not_passed" in report["reasons"]
    assert "manual_monitoring_not_acknowledged" in report["reasons"]
    assert "no_canary_events" in report["reasons"]
    assert "insufficient_canary_sessions:0<min_3" in report["reasons"]


def test_live_canary_report_rejects_taker_and_stale_accepted_quote():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=1)
    events = canary_session("s1", start, minutes=2)
    events.append(
        {
            "event": "quote_decision",
            "ts": iso(start + timedelta(seconds=30)),
            "session_id": "s1",
            "decision": "accept",
            "params_fresh": False,
            "collector_fresh": True,
            "book_fresh": True,
            "post_only": True,
            "post_only_verified": True,
            "fee_snapshot": {"fee_agreement_ok": True},
        }
    )
    events.append(
        {
            "event": "fill",
            "ts": iso(start + timedelta(seconds=40)),
            "session_id": "s1",
            "liquidity": "taker",
        }
    )

    report = build_live_canary_report(
        events,
        post_only_report=ok_report(now),
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=1,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        now=now,
    )

    assert report["ok"] is False
    assert "taker_fills_seen:1" in report["reasons"]
    assert "accepted_quote_stale_params:1" in report["reasons"]


def test_live_canary_report_rejects_unsafe_live_health():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=1)
    events = canary_session("s1", start, minutes=2)
    events[0]["stake_amount"] = 100
    events[0]["kill_on_taker_fill"] = False
    events[0]["post_only_verified"] = False

    report = build_live_canary_report(
        events,
        post_only_report=ok_report(now),
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=1,
        min_session_minutes=1,
        max_stake_amount=25,
        manual_monitoring_ack=True,
        now=now,
    )

    assert report["ok"] is False
    assert "live_health_post_only_unverified" in report["reasons"]
    assert "live_health_kill_on_taker_fill_disabled" in report["reasons"]
    assert "stake_amount_above_canary_limit" in report["reasons"]


def test_live_canary_report_rejects_param_or_hjb_errors():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=1)
    events = canary_session("s1", start, minutes=2)
    events.append({"event": "param_update_failed", "ts": iso(start + timedelta(seconds=30)), "session_id": "s1"})

    report = build_live_canary_report(
        events,
        post_only_report=ok_report(now),
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=1,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        now=now,
    )

    assert report["ok"] is False
    assert "param_update_failed:1" in report["reasons"]
    assert report["error_events"] == {"param_update_failed": 1}


def test_live_canary_report_rejects_stale_dependency_reports_and_events():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(days=8)
    old_report = ok_report(start)

    report = build_live_canary_report(
        canary_session("s1", start, minutes=2),
        post_only_report=old_report,
        fee_report=old_report,
        replay_report=old_report,
        min_sessions=1,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        max_dependency_report_age_seconds=86_400,
        max_canary_event_age_seconds=604_800,
        now=now,
    )

    assert report["ok"] is False
    assert "post_only_report_stale" in report["reasons"]
    assert "fee_report_stale" in report["reasons"]
    assert "replay_report_stale" in report["reasons"]
    assert "canary_event_stale:4" in report["reasons"]
    assert report["dependencies"]["post_only"]["freshness_reason"] == "stale"
    assert report["event_freshness"]["stale"] == 4


def test_live_canary_report_requires_dependency_generated_at():
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=1)

    report = build_live_canary_report(
        canary_session("s1", start, minutes=2),
        post_only_report={"ok": True, "reasons": []},
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=1,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        now=now,
    )

    assert report["ok"] is False
    assert "post_only_report_missing_generated_at" in report["reasons"]
    assert report["dependencies"]["post_only"]["freshness_reason"] == "missing_timestamp"


def test_read_jsonl_events_skips_bad_lines(tmp_path):
    path = tmp_path / "canary.jsonl"
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
