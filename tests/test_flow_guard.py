"""The Python toxic-flow guard must match the shipped Rust one.

The live bot and the Rust replay both guard; until 2026-09-02 the Python replay
did not, which biased every sweep it fed. These tests pin the semantics that
port carries over from ``rust_live/crates/mm-runtime/src/flow_guard.rs``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from flow_guard import (  # noqa: E402
    FlowGuard,
    FlowGuardConfig,
    MidWindow,
    VpinTracker,
    bucket_volume_from_totals,
)


def config(**overrides) -> FlowGuardConfig:
    base = dict(cooldown_ms=60_000.0, warmup_reentry_cooldown_ms=1_800_000.0)
    base.update(overrides)
    return FlowGuardConfig(**base)


# --------------------------------------------------------------------------- MidWindow
def test_mid_window_measures_against_the_oldest_retained_sample():
    window = MidWindow(5_000.0)
    assert window.observe(0.0, 100.0) == 0.0
    # 10% up against the 5 s-old reference.
    assert window.observe(1_000.0, 110.0) == 1_000.0
    # Once the old sample falls out of the window the reference moves with it.
    assert window.observe(7_000.0, 110.0) == 0.0


def test_mid_window_never_empties_and_rejects_nonpositive_prices():
    window = MidWindow(1_000.0)
    window.observe(0.0, 100.0)
    # Eviction keeps at least one sample, so the reference stays defined.
    assert window.observe(10_000.0, 100.0) == 0.0
    assert window.observe(10_001.0, 0.0) == 0.0


# --------------------------------------------------------------------------- VPIN
def test_vpin_is_none_until_a_full_window_of_buckets_exists():
    tracker = VpinTracker(bucket_volume=10.0, window_buckets=3)
    assert tracker.observe(True, 10.0, "a") is None
    assert tracker.observe(True, 10.0, "b") is None
    # Third bucket completes the window; three one-sided buckets are maximally
    # imbalanced, so VPIN saturates at 1.0.
    assert tracker.observe(True, 10.0, "c") == 1.0
    assert tracker.buckets_seen() == 3


def test_balanced_flow_reads_zero_and_vpin_stays_within_the_unit_interval():
    tracker = VpinTracker(bucket_volume=10.0, window_buckets=2)
    for index in range(4):
        tracker.observe(index % 2 == 0, 5.0, str(index))
    value = tracker.value()
    assert value is not None
    assert 0.0 <= value <= 1.0
    assert value == 0.0


def test_duplicate_trade_ids_are_ignored_but_anonymous_trades_are_not():
    tracker = VpinTracker(bucket_volume=10.0, window_buckets=1)
    assert tracker.observe(True, 5.0, "dup") is None
    # The same id again must not add volume.
    assert tracker.observe(True, 5.0, "dup") is None
    assert tracker.buckets_seen() == 0
    # A falsy id carries no identity, so two of them are distinct trades.
    tracker.observe(True, 5.0, "0")
    assert tracker.observe(True, 5.0, "0") == 1.0


def test_one_print_larger_than_the_window_saturates_without_looping():
    tracker = VpinTracker(bucket_volume=1.0, window_buckets=4)
    # 10_000 buckets' worth in a single print collapses to the retained window.
    assert tracker.observe(True, 10_000.0, "whale") == 1.0
    assert tracker.buckets_seen() == 4


def test_resizing_the_bucket_restarts_warm_up():
    tracker = VpinTracker(bucket_volume=10.0, window_buckets=2)
    tracker.observe(True, 20.0, "a")
    assert tracker.buckets_seen() == 2
    tracker.resize_bucket(25.0)
    assert tracker.buckets_seen() == 0
    assert tracker.value() is None


def test_bucket_volume_tracks_daily_volume():
    # 480 units over 12 h is 960/day; at 50 buckets/day that is 19.2 per bucket.
    assert bucket_volume_from_totals(480.0, 12 * 3_600_000.0, 50) == 19.2
    assert bucket_volume_from_totals(0.0, 1.0, 50) == 1.0


# --------------------------------------------------------------------------- FlowGuard
def test_a_fast_move_trips_and_the_cooldown_holds_it_closed():
    guard = FlowGuard(config())
    assert guard.evaluate(0, 900.0, 0.1) is True
    assert guard.trips == 1
    # Calm VPIN alone is not enough before the cooldown elapses.
    assert guard.evaluate(59_999, 10.0, 0.1) is True
    assert guard.evaluate(60_000, 10.0, 0.1) is False
    assert guard.is_tripped is False


def test_reentry_needs_vpin_under_threshold_as_well_as_the_cooldown():
    guard = FlowGuard(config())
    guard.evaluate(0, 900.0, 0.1)
    # Cooled, but flow is still toxic: stays closed (and re-trips on VPIN).
    assert guard.evaluate(600_000, 10.0, 0.9) is True
    assert guard.evaluate(600_001, 10.0, 0.1) is True
    assert guard.evaluate(660_001, 10.0, 0.1) is False


def test_a_fresh_breach_rearms_the_clock_but_not_the_trip_count():
    guard = FlowGuard(config())
    guard.evaluate(0, 900.0, 0.1)
    guard.evaluate(59_000, 900.0, 0.1)
    # The re-arm at 59_000 means the cooldown now expires at 119_000.
    assert guard.evaluate(60_000, 10.0, 0.1) is True
    assert guard.evaluate(119_000, 10.0, 0.1) is False
    assert guard.trips == 1, "a re-arm inside an open trip is not a new trip"


def test_warmup_reentry_needs_the_longer_cooldown_and_a_calm_window():
    guard = FlowGuard(config())
    assert guard.evaluate(0, 900.0, None) is True
    # The ordinary cooldown does not apply without the statistic.
    assert guard.evaluate(60_000, 10.0, None) is True
    # Longer cooldown alone is not enough: 500 bps is above the 0.5 x 800 line.
    assert guard.evaluate(1_800_000, 500.0, None) is True
    assert guard.evaluate(1_800_000, 10.0, None) is False


def test_a_live_cascade_cannot_expire_its_own_warmup_cooldown():
    guard = FlowGuard(config())
    guard.evaluate(0, 900.0, None)
    now = 0
    for _ in range(5):
        now += 1_799_999
        assert guard.evaluate(now, 900.0, None) is True
        assert guard.evaluate(now + 1, 10.0, None) is True
    assert guard.is_tripped


def test_a_disabled_guard_never_withholds():
    guard = FlowGuard(config(enabled=False))
    assert guard.evaluate(0, 10_000.0, 0.99) is False
    assert guard.is_tripped is False
    assert guard.trips == 0
