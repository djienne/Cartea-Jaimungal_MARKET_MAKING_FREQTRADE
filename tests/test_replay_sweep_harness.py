"""What a parameter sweep needs from the replay harness before it can be believed.

Three properties, each of which was violated at some point and cost a result:

1. A pinned tape scores the same bytes twice. The collector writes a shard every
   ~10 s, so an unpinned sweep compares settings AND data. That is not a
   hypothetical: one phi setting came back at both -3.98 and -19.39 USDC.
2. The replay and direct ``mm_core`` paths derive phi and alpha from dimensionless
   targets and the fitted kappa, not from raw values that are not kappa-invariant
   (eq. 10.28). Rust/Python parity is tested separately in the Rust workspace.
3. Fills are attributed to a side, and the sides add up to the total.
"""

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

import mm_core  # noqa: E402
import replay_market_maker  # noqa: E402
from mm_core import QuoteConfig  # noqa: E402
from replay_market_maker import (  # noqa: E402
    MARKOUT_HORIZONS_MS,
    ReplayConfig,
    ReplayTape,
    compute_hjb_cache,
    hjb_quote_config,
    load_tape,
    parse_args,
    resolve_latency_ms,
    run_replay,
    solve_replay_hjb,
)


PARAMS = {
    "kappa+": 6200.0,
    "kappa-": 8500.0,
    "lambda+": 0.23,
    "lambda-": 0.49,
    "epsilon+": 2.97e-5,
    "epsilon-": 1.19e-5,
}

TS0 = pd.Timestamp("2026-08-17T10:00:00Z")


def _price_rows(start: pd.Timestamp, count: int, step_ms: int = 1000) -> pd.DataFrame:
    """Collector-shaped BBO rows: one bid row and one ask row per update, both
    clocks present, so the exchange-clock path is the one under test."""
    rows = []
    for i in range(count):
        ts = start + pd.Timedelta(milliseconds=step_ms * i)
        epoch_ms = int(ts.value // 1_000_000)
        for side, price in (("bid", 99.0), ("ask", 101.0)):
            rows.append(
                {
                    # The receive clock lags the exchange by a jittery ~400 ms on
                    # the real tape; reproduce a lag here so a regression to the
                    # local clock changes the answer.
                    "timestamp": (epoch_ms + 400 + (i % 3) * 60) / 1000.0,
                    "price": price,
                    "size": 1000.0,
                    "side": side,
                    "exchange_timestamp": epoch_ms,
                }
            )
    return pd.DataFrame(rows)


def _trade_rows(start: pd.Timestamp, count: int, step_ms: int = 1000) -> pd.DataFrame:
    """Trades deep enough to reach either side's quote, alternating direction so
    both a bid and an ask can fill."""
    rows = []
    for i in range(count):
        ts = start + pd.Timedelta(milliseconds=step_ms * i + 100)
        epoch_ms = int(ts.value // 1_000_000)
        for offset, price in ((0, 98.0), (10, 102.0)):
            rows.append(
                {
                    "timestamp": (epoch_ms + offset + 400) / 1000.0,
                    "price": price,
                    "size": 5.0,
                    "side": "sell" if price < 100.0 else "buy",
                    "trade_id": f"t{i}-{price}",
                    "exchange_timestamp": epoch_ms + offset,
                }
            )
    return pd.DataFrame(rows)


def _write_tape(data_dir: Path, symbol: str, name: str, prices: pd.DataFrame, trades: pd.DataFrame) -> None:
    for stream, frame in (("prices", prices), ("trades", trades), ("orderbooks", pd.DataFrame())):
        target = data_dir / symbol / stream
        target.mkdir(parents=True, exist_ok=True)
        if frame.empty:
            continue
        frame.to_parquet(target / f"{stream}_{name}.parquet", index=False)


def _config(data_dir: Path, **overrides) -> ReplayConfig:
    base = dict(
        symbol="CASHCAT",
        data_dir=data_dir,
        mid_fallback=100.0,
        inventory_unit_base=1.0,
        q_max=3,
        q_min=-3,
        decision_latency_ms=0,
        order_ack_latency_ms=0,
        cancel_latency_ms=1000,
        quote_refresh_interval_ms=1000,
        hjb_phi_kappa_t=10.0,
        hjb_alpha_kappa=0.05,
        hjb_horizon_seconds=150.0,
    )
    base.update(overrides)
    return ReplayConfig(**base)


# ---------------------------------------------------------------------------
# 1. Tape pinning
# ---------------------------------------------------------------------------


def test_pinned_tape_scores_identically_even_after_the_collector_writes_more(tmp_path):
    """The regression this whole mechanism exists for: a sweep row must not move
    because a shard landed between two runs."""
    data_dir = tmp_path / "HL_data"
    _write_tape(data_dir, "CASHCAT", "0001", _price_rows(TS0, 12), _trade_rows(TS0, 12))
    config = _config(data_dir)

    tape = load_tape(config)
    first = run_replay(config, PARAMS, tape=tape).to_dict()

    # The collector keeps running.
    later = TS0 + pd.Timedelta(seconds=60)
    _write_tape(data_dir, "CASHCAT", "0002", _price_rows(later, 12), _trade_rows(later, 12))

    pinned_again = run_replay(config, PARAMS, tape=tape).to_dict()
    reloaded = run_replay(config, PARAMS).to_dict()

    assert first == pinned_again
    assert first["maker_fills"] > 0
    # And the unpinned run really did score different data, which is what makes
    # the guarantee above worth having.
    assert reloaded["input_rows"]["prices"] > first["input_rows"]["prices"]


def test_tape_none_still_loads_and_matches_a_tape_built_from_the_same_files(tmp_path):
    """tape=None must be the same run as tape=load_tape(config), or the pinned
    and unpinned paths are two different backtests."""
    data_dir = tmp_path / "HL_data"
    _write_tape(data_dir, "CASHCAT", "0001", _price_rows(TS0, 8), _trade_rows(TS0, 8))
    config = _config(data_dir)

    assert run_replay(config, PARAMS).to_dict() == run_replay(config, PARAMS, tape=load_tape(config)).to_dict()


def test_replay_keys_off_the_exchange_clock_not_the_collector_receive_clock(tmp_path):
    """The receive clock carries ~200 ms of jitter on the real tape -- more than
    a latency rung -- so which column orders the events has to be explicit."""
    data_dir = tmp_path / "HL_data"
    _write_tape(data_dir, "CASHCAT", "0001", _price_rows(TS0, 8), _trade_rows(TS0, 8))

    exchange = run_replay(_config(data_dir), PARAMS).to_dict()
    local = run_replay(_config(data_dir, event_clock="local"), PARAMS).to_dict()

    assert exchange["event_clock_by_stream"]["prices"] == "exchange"
    assert local["event_clock_by_stream"]["prices"] == "local"
    # The synthetic lag is a constant 400 ms plus a 0/60/120 ms jitter, so the
    # first event lands one lag later on the receive clock. (Compared loosely:
    # the receive column is float seconds and does not survive to the nanosecond.)
    lag = pd.Timestamp(local["data_start"]) - pd.Timestamp(exchange["data_start"])
    assert abs(lag.total_seconds() - 0.400) < 1e-3


# ---------------------------------------------------------------------------
# 2. HJB parity with an independently assembled mm_core configuration
# ---------------------------------------------------------------------------


def _independent_quote_config(**overrides) -> QuoteConfig:
    """A QuoteConfig assembled independently of hjb_quote_config, so a field
    dropped there fails here. This construction path is deliberately separate
    from the replay adapter under test."""
    base = dict(
        inventory_unit_base=2092.0,
        q_max=6,
        allow_short=True,
        hjb_phi_kappa_t=10.0,
        hjb_alpha_kappa=0.05,
        hjb_alpha=0.001,
        hjb_phi=0.0001,
        hjb_horizon_seconds=150.0,
        hjb_time_mode="episodic",
        gamma_inventory_risk=0.05,
    )
    base.update(overrides)
    return QuoteConfig(**base)


@pytest.mark.parametrize("sigma2_per_sec", [None, 1e-6])
def test_dimensionless_path_reproduces_mm_core_solve_hjb(tmp_path, sigma2_per_sec):
    config = _config(
        tmp_path,
        inventory_unit_base=2092.0,
        q_max=6,
        q_min=-6,
        sigma2_per_sec=sigma2_per_sec,
    )

    replayed = solve_replay_hjb(config, PARAMS)
    independent = mm_core.solve_hjb(
        PARAMS, _independent_quote_config(), sigma2_per_sec=sigma2_per_sec
    )

    assert replayed["phi_effective"] == independent["phi_effective"]
    assert replayed["alpha_effective"] == independent["alpha_effective"]
    assert replayed["phi_source"] == independent["phi_source"]
    assert list(replayed["q_grid"]) == list(independent["q_grid"])
    for key in ("delta_plus", "delta_minus"):
        np.testing.assert_allclose(replayed[key], independent[key], rtol=0, atol=0)
    np.testing.assert_allclose(
        replayed["delta_plus_surface"], independent["delta_plus_surface"], rtol=0, atol=0
    )


def test_dimensionless_targets_are_not_the_raw_values(tmp_path):
    """The point of routing through mm_core: at CASHCAT kappa the derived phi is
    two orders of magnitude off the raw default, so a replay still using the raw
    value would be scoring a different model."""
    config = _config(tmp_path, inventory_unit_base=2092.0, q_max=6, q_min=-6)
    kappa_avg = 0.5 * (PARAMS["kappa+"] + PARAMS["kappa-"])

    derived = solve_replay_hjb(config, PARAMS)
    raw = solve_replay_hjb(_config(tmp_path, hjb_phi_kappa_t=0.0, hjb_alpha_kappa=0.0, q_max=6, q_min=-6), PARAMS)

    assert derived["phi_effective"] == pytest.approx(10.0 / (kappa_avg * 150.0))
    assert derived["alpha_effective"] == pytest.approx(0.05 / kappa_avg)
    assert raw["phi_effective"] == pytest.approx(0.0001)
    assert raw["alpha_effective"] == pytest.approx(0.001)
    assert derived["phi_effective"] != raw["phi_effective"]


def test_phi_kappa_t_ceiling_binds_in_replay_and_direct_core(tmp_path):
    """The volatility channel is in absolute price units, so at CASHCAT scale it
    would blow straight past the dimensionless target if the ceiling were not
    applied. The replay adapter must clamp exactly where the direct core does."""
    config = _config(
        tmp_path,
        inventory_unit_base=2092.0,
        q_max=6,
        q_min=-6,
        sigma2_per_sec=1.0,
    )
    kappa_avg = 0.5 * (PARAMS["kappa+"] + PARAMS["kappa-"])

    solved = solve_replay_hjb(config, PARAMS)

    assert "capped" in solved["phi_source"]
    assert solved["phi_effective"] == pytest.approx(50.0 / (kappa_avg * 150.0))


def test_hjb_quote_config_carries_the_sweepable_knobs(tmp_path):
    config = _config(
        tmp_path,
        inventory_unit_base=2092.0,
        q_max=6,
        q_min=-6,
        hjb_phi_kappa_t_max=25.0,
        gamma_inventory_risk=0.2,
        hjb_time_mode="stationary",
    )

    solver_config = hjb_quote_config(config)

    assert solver_config.q_max == 6
    assert solver_config.allow_short is True
    assert solver_config.inventory_unit_base == 2092.0
    assert solver_config.hjb_phi_kappa_t == 10.0
    assert solver_config.hjb_alpha_kappa == 0.05
    assert solver_config.hjb_phi_kappa_t_max == 25.0
    assert solver_config.gamma_inventory_risk == 0.2
    assert solver_config.hjb_horizon_seconds == 150.0
    assert solver_config.hjb_time_mode == "stationary"


def test_asymmetric_domain_keeps_the_penalties_mm_core_derived(tmp_path):
    """mm_core.solve_hjb cannot express q_min, so the long-only solve is done
    here -- but on the SAME derived phi/alpha, never on re-derived ones."""
    config = _config(tmp_path, inventory_unit_base=2092.0, q_max=6, q_min=0, hjb_q_min=0)

    long_only = solve_replay_hjb(config, PARAMS)
    symmetric = solve_replay_hjb(_config(tmp_path, inventory_unit_base=2092.0, q_max=6, q_min=-6), PARAMS)

    assert list(long_only["q_grid"]) == [0, 1, 2, 3, 4, 5, 6]
    assert long_only["phi_effective"] == symmetric["phi_effective"]
    assert long_only["alpha_effective"] == symmetric["alpha_effective"]
    # At the bottom of a long-only grid the disabled side is the ask.
    assert np.isinf(long_only["delta_plus"][0])


def test_raw_entry_point_still_solves_the_requested_domain():
    """compute_hjb_cache is now a wrapper over mm_core; the historical raw-value
    behaviour it documents must not have moved."""
    cache = compute_hjb_cache(PARAMS, 3, 0, alpha=0.002, phi=0.0003, T_seconds=90.0)

    assert list(cache["q_grid"]) == [0, 1, 2, 3]
    assert cache["phi_effective"] == pytest.approx(0.0003)
    assert cache["alpha_effective"] == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# 3. Per-side attribution
# ---------------------------------------------------------------------------


def test_per_side_fill_counts_sum_to_the_total(tmp_path):
    data_dir = tmp_path / "HL_data"
    _write_tape(data_dir, "CASHCAT", "0001", _price_rows(TS0, 20), _trade_rows(TS0, 20))
    config = _config(data_dir)

    payload = run_replay(config, PARAMS, tape=load_tape(config)).to_dict()

    assert payload["maker_fills"] > 0
    assert sum(payload["fills_by_side"].values()) == payload["maker_fills"]
    # Two-sided tape, two-sided agent: both sides must actually be exercised or
    # the assertion above is vacuous.
    assert payload["fills_by_side"]["bid"] > 0
    assert payload["fills_by_side"]["ask"] > 0
    assert sum(payload["fills_by_depth"].values()) == payload["maker_fills"]


def test_per_side_depth_and_markout_are_reported_for_every_horizon(tmp_path):
    data_dir = tmp_path / "HL_data"
    _write_tape(data_dir, "CASHCAT", "0001", _price_rows(TS0, 20), _trade_rows(TS0, 20))
    config = _config(data_dir)

    payload = run_replay(config, PARAMS, tape=load_tape(config)).to_dict()

    for side in ("bid", "ask"):
        assert payload["mean_fill_depth_bps_by_side"][side] > 0
        horizons = payload["markout_by_side"][side]
        assert sorted(horizons) == sorted(str(h) for h in MARKOUT_HORIZONS_MS)
        for horizon_ms in MARKOUT_HORIZONS_MS:
            entry = horizons[str(horizon_ms)]
            # A markout is only recorded when the tape reaches that far past the
            # fill, so counts fall with the horizon but can never exceed fills.
            assert entry["count"] <= payload["fills_by_side"][side]
            if entry["count"]:
                assert entry["mean_usdc"] == pytest.approx(entry["sum_usdc"] / entry["count"])

    # The per-side markout sums must be a partition of the sampled markouts.
    for horizon_ms in MARKOUT_HORIZONS_MS:
        sampled = sum(
            float(sample["markout_usdc"])
            for sample in payload["markout_samples"]
            if int(sample["horizon_ms"]) == horizon_ms
        )
        aggregated = sum(payload["markout_by_side"][side][str(horizon_ms)]["sum_usdc"] for side in ("bid", "ask"))
        assert aggregated == pytest.approx(sampled)


# ---------------------------------------------------------------------------
# Latency as a first-class axis
# ---------------------------------------------------------------------------


def test_latency_total_splits_evenly_and_explicit_halves_win():
    args = parse_args_for(["--latency-ms", "50"])
    assert resolve_latency_ms(args) == (25, 25)

    args = parse_args_for(["--latency-ms", "101"])
    assert resolve_latency_ms(args) == (51, 50)

    # The shipped default is the slow end, kept for reproducibility.
    assert resolve_latency_ms(parse_args_for([])) == (250, 250)

    args = parse_args_for(["--latency-ms", "50", "--order-ack-latency-ms", "300"])
    assert resolve_latency_ms(args) == (25, 300)


def parse_args_for(extra: list[str]):
    argv = [
        "replay_market_maker.py",
        "--kappa-plus",
        "6200",
        "--kappa-minus",
        "8500",
        "--lambda-plus",
        "0.23",
        "--lambda-minus",
        "0.49",
        *extra,
    ]
    original = sys.argv
    try:
        sys.argv = argv
        return parse_args()
    finally:
        sys.argv = original


def test_cli_exposes_the_knobs_a_sweep_needs():
    args = parse_args_for(
        [
            "--q-min",
            "-6",
            "--q-max",
            "6",
            "--hjb-q-min",
            "0",
            "--inventory-unit-base",
            "2092",
            "--hjb-phi-kappa-t",
            "10",
            "--hjb-alpha-kappa",
            "0.05",
            "--sigma2-per-sec",
            "1e-6",
            "--latency-ms",
            "50",
            "--cancel-latency-ms",
            "40",
        ]
    )

    assert args.symbol == "CASHCAT"  # the symbol this project is sweeping
    assert (args.q_min, args.q_max, args.hjb_q_min) == (-6, 6, 0)
    assert args.inventory_unit_base == 2092.0
    assert (args.hjb_phi_kappa_t, args.hjb_alpha_kappa) == (10.0, 0.05)
    assert args.sigma2_per_sec == 1e-6
    assert (args.latency_ms, args.cancel_latency_ms) == (50, 40)


def test_metrics_report_the_latency_and_the_cadence_that_bounds_it(tmp_path):
    data_dir = tmp_path / "HL_data"
    _write_tape(data_dir, "CASHCAT", "0001", _price_rows(TS0, 12), _trade_rows(TS0, 12))
    config = _config(data_dir, decision_latency_ms=25, order_ack_latency_ms=25, cancel_latency_ms=40)

    payload = run_replay(config, PARAMS, tape=load_tape(config)).to_dict()

    assert payload["latency"]["total_quote_latency_ms"] == 50
    assert payload["latency"]["cancel_latency_ms"] == 40
    # Exposure = the wait for the refresh to replace it, plus the cancel tail.
    assert payload["latency"]["quote_exposure_ms"] == 1000 - 50 + 40
    # The tape is on a 1 s grid, so a 50 ms rung is below what it can resolve.
    assert payload["price_event_cadence_ms"]["p50"] == pytest.approx(1000.0)
    assert payload["latency"]["below_tape_resolution"] is True


def test_tape_can_be_built_directly_from_frames(tmp_path):
    """Sweep drivers hold frames, not directories; ReplayTape has to accept them
    so a caller never has to re-enter the loader to pin data."""
    prices = pd.DataFrame(
        [
            {"timestamp": TS0, "bid": 99.0, "ask": 101.0},
            {"timestamp": TS0 + pd.Timedelta(seconds=1), "bid": 99.0, "ask": 101.0},
        ]
    )
    trades = pd.DataFrame([{"timestamp": TS0 + pd.Timedelta(milliseconds=100), "price": 98.0, "size": 5.0}])
    tape = ReplayTape(
        prices=prices, trades=trades, orderbooks=pd.DataFrame(), input_files={"prices": 1, "trades": 1, "orderbooks": 0}
    )

    payload = run_replay(_config(tmp_path), PARAMS, tape=tape).to_dict()

    assert payload["maker_fills"] == 1
    assert payload["fills_by_side"]["bid"] == 1
    assert payload["input_files"] == {"prices": 1, "trades": 1, "orderbooks": 0}
