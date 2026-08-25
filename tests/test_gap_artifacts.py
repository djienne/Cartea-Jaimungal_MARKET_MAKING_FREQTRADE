"""Gaps must not bias the estimates.

The acceptance gate now tolerates a ~1 h data gap (raised from 300 s on
2026-08-19). That only makes sense if a gap cannot move the numbers, so these
tests pin the three places it could and the two places it was measured not to.

Measured on 61.2 h of real CASHCAT to 2026-08-19: the price stream had 91.5 min
of gaps > 60 s, but only 43.2 min were the collector actually down -- the other
48.3 min were a quiet market with trades and book updates still arriving. That
distinction is the whole design: over-correcting is as wrong as not correcting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from estimator_common import (  # noqa: E402
    DEFAULT_OUTAGE_THRESHOLD_SECONDS,
    MarketWindow,
    outage_seconds,
)
from replay_market_maker import (  # noqa: E402
    MARKOUT_MAX_STALENESS_MS,
    future_mid_from_arrays,
)
import pandas as pd  # noqa: E402


S = 1000.0  # ms per second


def _stream(*seconds):
    return np.array([s * S for s in seconds], dtype=float)


# ---------------------------------------------------------------- outages ---


def test_no_outage_when_the_stream_is_continuous():
    assert outage_seconds([_stream(0, 10, 20, 30)]) == (0, 0.0)


def test_interior_outage_is_counted():
    n, secs = outage_seconds([_stream(0, 10, 400, 410)])
    assert n == 1
    assert secs == pytest.approx(390.0)


def test_a_quiet_mid_stream_is_not_an_outage_when_trades_kept_arriving():
    """The 48.3 min case. CASHCAT's BBO can sit still for minutes while the
    collector is perfectly healthy; charging that to downtime would inflate
    lambda, which is the opposite of the bug being fixed."""
    mids = _stream(0, 10, 400, 410)
    trades = _stream(60, 120, 180, 240, 300, 360)  # steady flow, none over threshold
    n, secs = outage_seconds([mids, trades])
    assert (n, secs) == (0, 0.0)


def test_a_real_outage_survives_the_union_test():
    """Both streams silent together -- the collector was down."""
    mids = _stream(0, 10, 400, 410)
    trades = _stream(5, 405)
    n, secs = outage_seconds([mids, trades])
    assert n == 1
    assert secs == pytest.approx(390.0)


def test_only_interior_time_counts():
    """A window that merely starts late is the caller's problem; counting it
    here as well would subtract the same missing time twice."""
    n, secs = outage_seconds([_stream(500, 510, 520)], window_start_ms=0.0, window_end_ms=520 * S)
    assert (n, secs) == (0, 0.0)


def test_threshold_is_respected():
    assert outage_seconds([_stream(0, 30)], threshold_seconds=60.0) == (0, 0.0)
    assert outage_seconds([_stream(0, 90)], threshold_seconds=60.0)[0] == 1


def test_empty_and_single_sample_streams_are_safe():
    assert outage_seconds([]) == (0, 0.0)
    assert outage_seconds([np.array([])]) == (0, 0.0)
    assert outage_seconds([_stream(0)]) == (0, 0.0)


# ------------------------------------------------------- rate denominator ---


def _window(seconds: float, outage: float) -> MarketWindow:
    return MarketWindow(
        mids=pd.DataFrame({"ts_ms": [0.0], "mid": [1.0]}),
        trades=pd.DataFrame(),
        mid_source="prices",
        ts_source="exchange",
        window_start_ms=0.0,
        window_end_ms=seconds * S,
        meta={"outage_seconds": outage},
    )


def test_observed_seconds_subtracts_outage():
    w = _window(7200.0, 405.5)
    assert w.window_seconds == pytest.approx(7200.0)
    assert w.observed_seconds == pytest.approx(6794.5)


def test_observed_seconds_equals_wall_clock_without_outage():
    w = _window(7200.0, 0.0)
    assert w.observed_seconds == pytest.approx(w.window_seconds)


def test_observed_seconds_never_goes_negative():
    assert _window(60.0, 999.0).observed_seconds == 0.0


def test_lambda_denominator_matters_at_the_new_gate_threshold():
    """The reason this work was needed: the gate now tolerates a 1 h gap, and a
    1 h outage inside a 2 h window halves the arrival rate if unhandled."""
    w = _window(7200.0, 3600.0)
    n_events = 720
    assert n_events / w.window_seconds == pytest.approx(0.10)
    assert n_events / w.observed_seconds == pytest.approx(0.20)


# -------------------------------------------------------------- markouts ---


def _prices(*seconds):
    return np.array([int(s * 1e9) for s in seconds], dtype=np.int64)


def test_markout_uses_the_first_mid_at_or_after_the_horizon():
    ts = _prices(0, 1, 2, 3)
    mid = np.array([10.0, 11.0, 12.0, 13.0])
    assert future_mid_from_arrays(ts, mid, 0, 2000) == pytest.approx(12.0)


def test_markout_declines_across_an_outage():
    """A fill just before an outage must not have the outage's drift booked as
    its markout."""
    ts = _prices(0, 1, 500)
    mid = np.array([10.0, 11.0, 99.0])
    assert future_mid_from_arrays(ts, mid, 0, 2000) is None


def test_markout_accepts_a_mid_inside_the_staleness_budget():
    ts = _prices(0, 1, 2 + MARKOUT_MAX_STALENESS_MS / 1000.0 - 0.5)
    mid = np.array([10.0, 11.0, 12.0])
    assert future_mid_from_arrays(ts, mid, 0, 2000) == pytest.approx(12.0)


def test_markout_still_declines_past_the_end_of_the_tape():
    ts = _prices(0, 1)
    mid = np.array([10.0, 11.0])
    assert future_mid_from_arrays(ts, mid, 0, 60_000) is None


def test_markout_staleness_can_be_disabled_for_reproducing_old_artifacts():
    ts = _prices(0, 1, 500)
    mid = np.array([10.0, 11.0, 99.0])
    assert future_mid_from_arrays(ts, mid, 0, 2000, max_staleness_ms=None) == pytest.approx(99.0)


# ------------------------------------------------------------- the gate ----


def test_gate_threshold_tolerates_an_hour():
    # This used to cross-check run_safety_gates.DEFAULT_REPLAY_MAX_PRICE_GAP_SECONDS
    # against the replay's own default. That gate pipeline retired with the
    # freqtrade trader, so the replay's default is now the only one there is.
    from run_replay_report import DEFAULT_MAX_PRICE_GAP_SECONDS

    assert DEFAULT_MAX_PRICE_GAP_SECONDS == pytest.approx(3600.0)
    # The 474 s outage that failed the old gate must now pass.
    assert 474.0 <= DEFAULT_MAX_PRICE_GAP_SECONDS


def test_outage_threshold_default_is_the_documented_one():
    assert DEFAULT_OUTAGE_THRESHOLD_SECONDS == pytest.approx(60.0)
