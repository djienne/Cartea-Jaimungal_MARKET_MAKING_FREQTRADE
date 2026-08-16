from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_market_viability import (  # noqa: E402
    MIN_SAMPLES_AT_OPTIMUM,
    best_depth,
    build_market_viability_report,
    conditional_markout,
    depth_atom_share,
    infer_tick_size,
    mo_depth_impact_frame,
    profit_curve,
)


def _mids(n: int = 4000, mid: float = 1000.0, spread: float = 0.1) -> pd.DataFrame:
    ts = np.arange(n, dtype=float) * 100.0
    bid = np.full(n, mid - spread / 2.0)
    ask = np.full(n, mid + spread / 2.0)
    return pd.DataFrame({"ts_ms": ts, "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0})


def test_infer_tick_size_reads_the_price_grid():
    prices = np.array([100.0, 100.1, 100.2, 100.5, 101.0])
    assert infer_tick_size(prices) == 0.1


def test_depth_atom_share_flags_a_one_tick_book():
    # Every touch-taking market order lands on exactly half a tick.
    depths = np.array([0.05] * 95 + [0.15, 0.25, 0.35, 0.45, 0.55])
    assert depth_atom_share(depths) == 0.95
    assert depth_atom_share(np.arange(10, dtype=float)) == 0.1


def test_conditional_markout_keeps_the_informed_tail():
    """The untrimmed mean is the point: trimming deletes adverse selection.

    A 10% trimmed mean would drop the two extreme observations and report a much
    smaller markout, which is what previously manufactured a phantom edge.
    """
    markouts = np.array([0.0] * 18 + [5.0, 5.0])
    assert conditional_markout(markouts) == pytest.approx(0.5)


def test_profit_curve_skips_depths_below_the_fee():
    depths = np.array([0.5] * 60)
    markouts = np.zeros(60)
    # fee_cost above every observed depth leaves nothing to consider.
    assert profit_curve(depths, markouts, covered_seconds=600.0, fee_cost=1.0, mid=1000.0) == []


def test_profit_curve_edge_nets_fee_and_conditional_markout():
    depths = np.array([1.0] * 100)
    markouts = np.full(100, 0.2)
    curve = profit_curve(depths, markouts, covered_seconds=100.0, fee_cost=0.3, mid=1000.0)
    assert len(curve) == 1
    point = curve[0]
    assert point["edge_per_fill"] == pytest.approx(1.0 - 0.3 - 0.2)
    assert point["n_market_orders"] == 100
    # Default sizes of 1.0: 100 base over 100s == 3600 base/hour at 0.5 edge.
    assert point["reachable_base_per_hour"] == pytest.approx(3600.0)
    assert point["pnl_per_hour_upper_bound"] == pytest.approx(3600.0 * 0.5)
    assert best_depth(curve) is point


def test_profit_curve_scales_with_traded_size_not_event_count():
    """Fill opportunity is volume, not arrivals.

    Counting events alone credited the quote with a fixed size on every arrival,
    which on an illiquid coin implied filling a large multiple of the whole
    instrument's daily volume.
    """
    depths = np.array([1.0] * 100)
    markouts = np.zeros(100)
    small = profit_curve(
        depths, markouts, sizes=np.full(100, 0.5),
        covered_seconds=100.0, fee_cost=0.3, mid=1000.0,
    )[0]
    large = profit_curve(
        depths, markouts, sizes=np.full(100, 5.0),
        covered_seconds=100.0, fee_cost=0.3, mid=1000.0,
    )[0]
    assert large["pnl_per_hour_upper_bound"] == pytest.approx(
        10.0 * small["pnl_per_hour_upper_bound"]
    )
    # Same number of arrivals either way.
    assert small["fills_per_hour"] == large["fills_per_hour"]


def test_mo_depth_impact_frame_signs_both_sides_against_the_maker():
    mids = pd.DataFrame(
        {
            "ts_ms": [0.0, 1000.0, 6000.0],
            "bid": [999.9, 999.9, 1001.9],
            "ask": [1000.1, 1000.1, 1002.1],
            "mid": [1000.0, 1000.0, 1002.0],
        }
    )
    mos = pd.DataFrame(
        {
            "ts_ms": [1000.0],
            "side": ["buy"],
            "price_extreme": [1000.5],
            "size": [1.0],
            "n_prints": [1],
            "pre_mid": [1000.0],
        }
    )
    frame = mo_depth_impact_frame(mos, mids, horizon_ms=5000, window_end_ms=6000.0)
    assert len(frame) == 1
    # Buy MO walked 0.5 past the mid, and the mid then rose 2.0 against the ask.
    assert frame.iloc[0]["depth"] == pytest.approx(0.5)
    assert frame.iloc[0]["markout"] == pytest.approx(2.0)


def _trades(depth: float, per_side: int) -> pd.DataFrame:
    """Synthetic prints: one market order per side per timestamp, at a fixed depth."""
    rows = []
    for index in range(per_side):
        ts = 1000.0 + index * 100.0
        rows.append({"ts_ms": ts, "side": "buy", "price": 1000.0 + depth, "size": 1.0})
        rows.append({"ts_ms": ts, "side": "sell", "price": 1000.0 - depth, "size": 1.0})
    return pd.DataFrame(rows)


def test_one_tick_book_is_reported_not_viable():
    report = build_market_viability_report(
        crypto="TEST",
        mids=_mids(spread=0.1),
        trades=_trades(0.05, per_side=200),
        covered_seconds=1800.0,
        window_end_ms=400_000.0,
        maker_fee_rate=0.00015,
    )
    assert report["viable"] is False
    # A one-tick book cannot cover two maker fees at the touch.
    assert report["at_touch"]["net_edge_bps"] < 0
    assert report["book"]["quoted_spread_median_ticks"] == pytest.approx(1.0)


def _informed_sell_market(
    *,
    n_steps: int = 1500,
    step_ms: float = 100.0,
    drift_until: int = 700,
    drift: float = 0.06,
    depth: float = 1.0,
    per_side: int = 60,
    spread: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """A market where sell market orders are informed and buy ones are not.

    The mid falls steadily through the first phase, where every sell MO lands, so
    whoever filled those bids is immediately underwater. Buy MOs land in the
    second, flat phase and carry no follow-through. That asymmetry is what a real
    toxic book looks like, and a symmetric trend would not produce it: under pure
    drift a two-sided maker gains on one side exactly what it loses on the other.
    """
    index = np.arange(n_steps, dtype=float)
    mid = np.where(
        index < drift_until,
        1000.0 - drift * index,
        1000.0 - drift * (drift_until - 1),
    )
    mids = pd.DataFrame(
        {
            "ts_ms": index * step_ms,
            "bid": mid - spread / 2.0,
            "ask": mid + spread / 2.0,
            "mid": mid,
        }
    )

    # MOs sit between mid rows so attach_pre_mid (strictly-before) picks the row
    # exactly one horizon ahead of the post-mid lookup.
    rows = []
    for k in range(per_side):  # informed sells, entirely inside the drift phase
        rows.append(
            {
                "ts_ms": k * 1000.0 + 50.0,
                "side": "sell",
                "price": mid[k * 10] - depth,
                "size": 1.0,
            }
        )
    for k in range(75, 75 + per_side):  # uninformed buys, in the flat phase
        rows.append(
            {
                "ts_ms": k * 1000.0 + 50.0,
                "side": "buy",
                "price": mid[k * 10] + depth,
                "size": 1.0,
            }
        )
    window_end_ms = (n_steps - 1) * step_ms
    return mids, pd.DataFrame(rows), window_end_ms


def test_losing_side_is_not_discarded_from_the_two_sided_total():
    """Regression: summing only the profitable side declared a dead market viable.

    Sell MOs here are informed (mid keeps falling 3.0 against the filled bid) and
    buy MOs are not. The buy side's optimum is genuinely positive, so a total that
    kept only profitable sides would call this viable. A maker cannot decline to be
    filled on the side that loses, so the verdict must be NOT viable.
    """
    mids, trades, window_end_ms = _informed_sell_market()
    report = build_market_viability_report(
        crypto="TEST",
        mids=mids,
        trades=trades,
        covered_seconds=window_end_ms / 1000.0,
        window_end_ms=window_end_ms,
        maker_fee_rate=0.00015,
    )

    plus = report["profit_curve"]["optimum_plus"]
    minus = report["profit_curve"]["optimum_minus"]
    assert plus is not None and minus is not None
    assert plus["pnl_per_hour_upper_bound"] > 0, "buy side should look profitable on its own"
    assert minus["pnl_per_hour_upper_bound"] < 0, "informed sells should make the bid side lose"
    assert report["profit_curve"]["total_pnl_per_hour_upper_bound"] < 0

    assert report["viable"] is False
    assert any("two_sided_pnl_not_positive" in reason for reason in report["reasons"])


def test_thin_optimum_sample_blocks_a_viable_verdict():
    """A handful of deep market orders must not be enough to declare viability."""
    mids = _mids(spread=2.0)  # wide book, so the touch itself clears the fee
    rows = []
    for index in range(MIN_SAMPLES_AT_OPTIMUM - 10):
        ts = 1000.0 + index * 100.0
        rows.append({"ts_ms": ts, "side": "buy", "price": 1002.0, "size": 1.0})
        rows.append({"ts_ms": ts, "side": "sell", "price": 998.0, "size": 1.0})
    report = build_market_viability_report(
        crypto="TEST",
        mids=mids,
        trades=pd.DataFrame(rows),
        covered_seconds=1800.0,
        window_end_ms=400_000.0,
        maker_fee_rate=0.00015,
    )
    assert report["viable"] is False
    assert any("optimum_sample_too_thin" in reason for reason in report["reasons"])


def test_wide_book_with_deep_sample_is_viable():
    """The gate must be able to say yes, or it is not a gate.

    min_window_hours is disabled here: this exercises the economics, and the
    window-length guard has its own test below.
    """
    mids = _mids(spread=4.0)
    rows = []
    for index in range(400):
        ts = 1000.0 + index * 100.0
        rows.append({"ts_ms": ts, "side": "buy", "price": 1002.0, "size": 1.0})
        rows.append({"ts_ms": ts, "side": "sell", "price": 998.0, "size": 1.0})
    report = build_market_viability_report(
        crypto="TEST",
        mids=mids,
        trades=pd.DataFrame(rows),
        covered_seconds=1800.0,
        window_end_ms=400_000.0,
        maker_fee_rate=0.00015,
        min_window_hours=0.0,
    )
    assert report["viable"] is True, report["reasons"]
    assert report["profit_curve"]["total_pnl_per_hour_upper_bound"] > 0
    assert report["profit_curve"]["total_reachable_notional_per_hour"] > 0
    assert report["at_touch"]["net_edge_bps"] > 0


def test_no_mid_data_fails_closed():
    report = build_market_viability_report(
        crypto="TEST",
        mids=pd.DataFrame(columns=["ts_ms", "bid", "ask", "mid"]),
        trades=pd.DataFrame(columns=["ts_ms", "side", "price", "size"]),
        covered_seconds=1800.0,
        window_end_ms=0.0,
    )
    assert report["viable"] is False
    assert report["reasons"] == ["no_mid_data"]


def test_short_window_cannot_issue_a_viable_verdict():
    """Regression: a 15-minute burst must not be allowed to declare viability.

    CASHCAT on 2026-08-17 ran 19.3x its own daily average volume with a 200-tick
    spread; over that burst the profit curve reported +$3,279/h off ~20 samples,
    while every other candidate sat at 0.18-0.68x its daily rate. A window short
    enough to sit inside one regime describes that regime, not the instrument.
    """
    mids = _mids(spread=4.0)
    rows = []
    for index in range(400):
        ts = 1000.0 + index * 100.0
        rows.append({"ts_ms": ts, "side": "buy", "price": 1002.0, "size": 1.0})
        rows.append({"ts_ms": ts, "side": "sell", "price": 998.0, "size": 1.0})
    trades = pd.DataFrame(rows)

    common = dict(
        crypto="TEST",
        mids=mids,
        trades=trades,
        covered_seconds=1800.0,  # half an hour
        window_end_ms=400_000.0,
        maker_fee_rate=0.00015,
    )
    # Identical economics; only the required window length differs.
    assert build_market_viability_report(**common, min_window_hours=0.0)["viable"] is True
    short = build_market_viability_report(**common, min_window_hours=6.0)
    assert short["viable"] is False
    assert any("window_too_short_for_a_verdict" in r for r in short["reasons"])
    assert short["window"]["hours"] == pytest.approx(0.5)


def test_report_exposes_observed_flow_rate():
    """The report must show the window's own flow so a burst is visible."""
    mids = _mids(spread=4.0)
    rows = [
        {"ts_ms": 1000.0 + i * 100.0, "side": "buy", "price": 1000.0, "size": 2.0}
        for i in range(100)
    ]
    report = build_market_viability_report(
        crypto="TEST",
        mids=mids,
        trades=pd.DataFrame(rows),
        covered_seconds=3600.0,
        window_end_ms=400_000.0,
        maker_fee_rate=0.00015,
    )
    # 100 prints x 2.0 size x 1000.0 price over exactly one hour.
    assert report["window"]["observed_notional_per_hour"] == pytest.approx(200_000.0)
