from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from record_manual_monitoring_ack import (  # noqa: E402
    append_jsonl_event,
    build_manual_monitoring_ack_event,
    main,
)
from test_live_canary import canary_session, iso, ok_report  # noqa: E402
from verify_live_canary import build_live_canary_report, read_jsonl_events  # noqa: E402


def test_build_manual_monitoring_ack_event_is_explicit():
    event = build_manual_monitoring_ack_event(
        session_id="canary-1",
        operator="tester",
        ts="2026-05-25T10:00:00Z",
        note="watched exchange fills",
    )

    assert event["event"] == "manual_monitoring_ack"
    assert event["acknowledged"] is True
    assert event["manual_monitoring_ack"] is True
    assert event["session_id"] == "canary-1"
    assert event["operator"] == "tester"
    assert event["requirements_observed"]["post_only_required"] is True
    assert event["requirements_observed"]["kill_on_taker_fill_required"] is True


def test_record_manual_monitoring_ack_requires_explicit_risk_ack(tmp_path, capsys):
    output = tmp_path / "mm_debug.jsonl"

    result = main(["--output", str(output), "--session-id", "canary-1", "--operator", "tester"])

    assert result == 2
    assert not output.exists()
    assert "Refusing to write acknowledgement" in capsys.readouterr().out


def test_recorded_manual_monitoring_ack_satisfies_canary_ack_requirement(tmp_path):
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=1)
    output = tmp_path / "mm_debug.jsonl"
    events = [
        event
        for event in canary_session("canary-1", start, minutes=2)
        if event.get("event") != "manual_monitoring_ack"
    ]
    for event in events:
        append_jsonl_event(output, event)

    result = main(
        [
            "--output",
            str(output),
            "--session-id",
            "canary-1",
            "--operator",
            "tester",
            "--timestamp",
            iso(start + timedelta(minutes=3)),
            "--note",
            "monitored quotes and fills",
            "--acknowledge-risk",
        ]
    )

    assert result == 0
    stored = read_jsonl_events(output)
    ack_events = [event for event in stored if event.get("event") == "manual_monitoring_ack"]
    assert len(ack_events) == 1
    assert ack_events[0]["operator"] == "tester"
    assert ack_events[0]["note"] == "monitored quotes and fills"

    report = build_live_canary_report(
        stored,
        post_only_report=ok_report(now),
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=1,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        now=now,
    )

    assert "manual_monitoring_ack_event_missing" not in report["reasons"]
    assert report["manual_monitoring"]["fresh"] == 1


def test_recorded_future_manual_monitoring_ack_is_rejected(tmp_path):
    start = datetime(2026, 5, 25, tzinfo=timezone.utc)
    now = start + timedelta(hours=1)
    output = tmp_path / "mm_debug.jsonl"
    events = [
        event
        for event in canary_session("canary-1", start, minutes=2)
        if event.get("event") != "manual_monitoring_ack"
    ]
    for event in events:
        append_jsonl_event(output, event)

    result = main(
        [
            "--output",
            str(output),
            "--session-id",
            "canary-1",
            "--operator",
            "tester",
            "--timestamp",
            iso(now + timedelta(minutes=5)),
            "--acknowledge-risk",
        ]
    )

    assert result == 0
    report = build_live_canary_report(
        read_jsonl_events(output),
        post_only_report=ok_report(now),
        fee_report=ok_report(now),
        replay_report=ok_report(now),
        min_sessions=1,
        min_session_minutes=1,
        manual_monitoring_ack=True,
        now=now,
    )

    assert report["ok"] is False
    assert "manual_monitoring_ack_event_future_timestamp:1" in report["reasons"]
    assert report["manual_monitoring"]["future"] == 1


def test_append_jsonl_event_writes_one_json_object_per_line(tmp_path):
    output = tmp_path / "audit.jsonl"
    event = build_manual_monitoring_ack_event(session_id=None, operator="tester", ts=iso(datetime(2026, 5, 25, tzinfo=timezone.utc)))

    append_jsonl_event(output, event)

    line = output.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["event"] == "manual_monitoring_ack"
    assert payload["operator"] == "tester"
