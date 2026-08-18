"""The replay's hot loop was rewritten for speed. These tests say it still scores
the same tape the same way, to the last bit.

WHY A WHOLE FILE FOR THIS. ``run_replay`` used to read every field through
pandas scalar access and took about 1.8 ms per price event, which made one pass
over a day of data ~5 minutes and a 300-point sweep unaffordable. The rewrite
hoists the columns into numpy arrays and cuts the trade window with a binary
search instead of a full-frame boolean mask. That is a 30-40x speedup and
exactly the kind of change that can move a fill by one without anyone noticing
-- and a backtest that quietly changes its fills invalidates every number the
project has published while still looking healthy.

So the guarantee is BIT identity, not approximate agreement:

- ``test_reference_grid_is_bit_identical`` replays a pinned synthetic tape at 41
  configurations spanning both latency extremes, both requote intervals, a
  clamping and a non-clamping phi, the two-sided and long-only inventory
  domains, both time modes, and the funding / calibration-throttle / asymmetric
  solve / sigma2 / amount-step / raw-phi branches. Every float is compared by
  repr, so a last-bit difference fails.
- The reference was recorded from the PRE-optimisation replay (git 85bb564) and
  must not be regenerated from the current one; see tests/replay_reference_matrix.py.
- The rest pin each hoisted column view against the row-wise function it
  replaced, since those are where a rewrite would go wrong quietly.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import replay_market_maker  # noqa: E402
from replay_market_maker import (  # noqa: E402
    ReplayConfig,
    ReplayTape,
    first_level_size,
    first_level_size_array,
    future_mid,
    future_mid_from_arrays,
    is_monotonic_ns,
    matching_trade_with_queue_decay,
    mid_from_price_row,
    price_event_arrays,
    run_replay,
    scan_for_matching_trade,
    timestamps_as_ns,
    total_seconds_from_ns,
    trade_event_arrays,
)
from replay_reference_matrix import (  # noqa: E402
    REFERENCE_PARAMS,
    load_reference,
    reference_configs,
    reference_tape,
)


# ---------------------------------------------------------------------------
# The headline guarantee
# ---------------------------------------------------------------------------


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}", value[key], out)
    elif isinstance(value, list):
        out[f"{prefix}.__len__"] = len(value)
        for index, item in enumerate(value):
            _flatten(f"{prefix}[{index}]", item, out)
    elif isinstance(value, float):
        # repr, not ==: it distinguishes the last bit and it makes NaN compare
        # equal to NaN, which is what "the run did the same thing" means here.
        out[prefix] = f"float:{value!r}"
    else:
        out[prefix] = value


@pytest.fixture(scope="module")
def reference_payload():
    return load_reference()


@pytest.fixture(scope="module")
def pinned_tape():
    return reference_tape(replay_market_maker)


def test_reference_was_recorded_before_the_rewrite(reference_payload):
    """Guards against the reference being refreshed from the code it checks."""
    assert reference_payload["generated_from"] == "replay_baseline.py"
    assert reference_payload["params"] == REFERENCE_PARAMS
    assert len(reference_payload["results"]) == len(reference_configs())


def test_reference_grid_is_bit_identical(reference_payload, pinned_tape):
    expected_all = reference_payload["results"]
    differences: list[str] = []
    for name, kwargs in reference_configs():
        assert name in expected_all, f"{name} missing from the recorded reference"
        actual = run_replay(ReplayConfig(**kwargs), dict(REFERENCE_PARAMS), tape=pinned_tape).to_dict()
        flat_actual: dict = {}
        flat_expected: dict = {}
        _flatten(name, actual, flat_actual)
        _flatten(name, expected_all[name], flat_expected)
        for key in sorted(set(flat_actual) | set(flat_expected)):
            if flat_actual.get(key, "<absent>") != flat_expected.get(key, "<absent>"):
                differences.append(
                    f"{key}: reference {flat_expected.get(key, '<absent>')!r} "
                    f"-> now {flat_actual.get(key, '<absent>')!r}"
                )
    assert not differences, "replay output moved:\n" + "\n".join(differences[:40])


def test_grid_actually_exercises_the_fill_path(reference_payload):
    """A grid that never fills would pass the identity check and prove nothing."""
    fills = [row["maker_fills"] for row in reference_payload["results"].values()]
    assert min(fills) > 0
    assert sum(fills) > 1_000


def test_unsorted_trade_window_agrees_with_the_binary_search(monkeypatch, pinned_tape):
    """The two window paths must select the same rows.

    Sorted timestamps take a searchsorted range; anything else falls back to the
    boolean mask the original always used. Forcing the fallback on a sorted tape
    is the direct test that they agree -- if they did not, a monkeypatched
    loader in some other test would silently score a different tape.
    """
    config = ReplayConfig(**dict(reference_configs())["lat500_ref100_phi10_twosided6_episodic"])
    fast = run_replay(config, dict(REFERENCE_PARAMS), tape=pinned_tape).to_dict()
    monkeypatch.setattr(replay_market_maker, "is_monotonic_ns", lambda _values: False)
    slow = run_replay(config, dict(REFERENCE_PARAMS), tape=pinned_tape).to_dict()
    flat_fast: dict = {}
    flat_slow: dict = {}
    _flatten("run", fast, flat_fast)
    _flatten("run", slow, flat_slow)
    assert flat_fast == flat_slow


# ---------------------------------------------------------------------------
# The hoisted column views, each against the row-wise function it replaced
# ---------------------------------------------------------------------------


def test_total_seconds_from_ns_matches_pandas():
    """pandas truncates to microseconds and then adds in floating point.

    Neither ``ns / 1e9`` nor the exact ``ns // 1000 / 1e6`` reproduces it, and
    the episode clock, the funding accrual and the queue decay all read this
    number, so getting it merely close would move fills.
    """
    rng = np.random.default_rng(11)
    samples = [0, 1, -1, 999, -999, 1000, -1000, 86_400 * 10**9, -86_400 * 10**9 - 1]
    for magnitude in (10**3, 10**6, 10**9, 10**11, 10**13, 10**15, 10**17):
        samples += [int(value) for value in rng.integers(-magnitude, magnitude, size=4000)]
    for delta_ns in samples:
        expected = pd.Timedelta(delta_ns, unit="ns").total_seconds()
        assert repr(total_seconds_from_ns(delta_ns)) == repr(expected), delta_ns


def test_total_seconds_from_ns_rejects_the_obvious_shortcuts():
    """A regression guard with teeth: these two are what someone would 'simplify' to."""
    delta_ns = 1_904_529_860
    exact = total_seconds_from_ns(delta_ns)
    assert exact != delta_ns / 1e9
    assert exact != (delta_ns // 1000) / 1e6


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"bid": [100.0, 101.5, np.nan, 3.0], "ask": [101.0, np.nan, 102.0, 3.5]}),
        pd.DataFrame({"best_bid": [1.0, 2.0], "best_ask": [1.5, np.inf]}),
        pd.DataFrame({"bid": [1.0, 2.0]}),
        pd.DataFrame({"ask": [1.0, 2.0]}),
        pd.DataFrame({"other": [1.0, 2.0]}),
        pd.DataFrame({"bid": np.array([0.1015, 0.10151], dtype="float32"),
                      "ask": np.array([0.1016, 0.10161], dtype="float32")}),
        # Both spellings present and DISAGREEING, which is the only way to pin
        # the preference order -- a frame carrying only one resolves the same
        # either way.
        pd.DataFrame(
            {
                "bid": [10.0, 11.0],
                "best_bid": [20.0, 21.0],
                "ask": [10.5, 11.5],
                "best_ask": [20.5, 21.5],
            }
        ),
    ],
)
def test_price_event_arrays_match_mid_from_price_row(frame):
    fallback = 7.25
    mid, best_bid, best_ask = price_event_arrays(frame, fallback)
    for i in range(len(frame)):
        expected = mid_from_price_row(frame.iloc[i], fallback)
        got = (float(mid[i]), float(best_bid[i]), float(best_ask[i]))
        for value, reference in zip(got, expected):
            assert repr(value) == repr(float(reference)), (i, got, expected)


def test_first_level_size_array_matches_the_row_wise_reader():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([0, 1, 2, 3, 4], unit="s", utc=True),
            "bid_size": [None, 1.5, None, None, None],
            "best_bid_size": [2.5, None, None, np.nan, None],
            "bid_size_0": [3.5, 3.5, 3.5, 3.5, np.nan],
            "bids": [[[1.0, 9.0]], None, [[1.0, 8.0]], "junk", [[1.0, 7.0]]],
            "ask_size_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "asks": [[[1.0, 9.0]]] * 5,
        }
    )
    for side in ("bid", "ask"):
        column = first_level_size_array(frame, side)
        for i in range(len(frame)):
            expected = first_level_size(frame.iloc[i], side)
            assert repr(float(column[i])) == repr(expected), (side, i)


def test_first_level_size_array_handles_absent_columns():
    frame = pd.DataFrame({"timestamp": pd.to_datetime([0, 1], unit="s", utc=True)})
    for side in ("bid", "ask"):
        column = first_level_size_array(frame, side)
        assert list(column) == [0.0, 0.0]
        assert [first_level_size(frame.iloc[i], side) for i in range(2)] == [0.0, 0.0]


def test_first_level_size_array_reads_nested_levels_when_scalars_are_absent():
    frame = pd.DataFrame({"bids": [[[1.0, 1.25]]], "asks": [[[2.0, 2.5]]]})
    assert first_level_size_array(frame, "bid")[0] == 1.25
    assert first_level_size_array(frame, "ask")[0] == 2.5


def test_trade_event_arrays_preserve_the_falsy_or_default_idiom():
    """``float(x or default)`` treats 0.0 and None as absent and NaN as present.

    The two call sites pass different defaults, so the mask has to say WHICH
    cells are falsy rather than substituting anything here.
    """
    frame = pd.DataFrame({"price": [1.0, 2.0, 3.0], "size": [0.0, np.nan, 5.0]})
    _price, size, falsy = trade_event_arrays(frame)
    assert list(falsy) == [True, False, False]
    assert math.isnan(size[1])

    objects = pd.DataFrame({"price": [1.0, 2.0], "size": pd.Series([None, 4.0], dtype=object)})
    _price, size, falsy = trade_event_arrays(objects)
    assert list(falsy) == [True, False]
    # None becomes 0.0 rather than NaN, which is what makes the falsy guard in
    # scan_for_matching_trade belt-and-braces rather than load-bearing. If this
    # ever becomes NaN the guard starts carrying the behaviour, so pin it here.
    assert size[0] == 0.0

    missing = pd.DataFrame({"price": [1.0, 2.0]})
    _price, _size, falsy = trade_event_arrays(missing)
    assert list(falsy) == [True, True]


def test_future_mid_from_arrays_matches_the_frame_search():
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        {
            "timestamp": [ts0 + pd.Timedelta(milliseconds=step) for step in (0, 300, 900, 4000, 40000)],
            "bid": [100.0, 100.4, 100.9, np.nan, 101.5],
            "ask": [101.0, 101.4, 101.9, 102.0, 102.5],
        }
    )
    ts_ns = timestamps_as_ns(prices)
    mid, _bid, _ask = price_event_arrays(prices, 99.0)
    for horizon_ms in (0, 100, 1_000, 5_000, 30_000, 120_000):
        expected = future_mid(prices, ts0, horizon_ms, 99.0)
        actual = future_mid_from_arrays(ts_ns, mid, int(ts_ns[0]), horizon_ms)
        assert (expected is None) == (actual is None), horizon_ms
        if expected is not None:
            assert repr(actual) == repr(float(expected)), horizon_ms


def _row_wise_round_price(side, price, tick_size):
    """The pre-rewrite ``round_price_for_side``, copied verbatim.

    The tick resolution was hoisted out of the loop, which is only safe if the
    remaining arithmetic is untouched -- and it is deliberately numpy's:
    ``round`` on a numpy scalar is a scale-and-rint, not Python's decimal round,
    so swapping np.floor for math.floor would change quoted prices.
    """
    tick = replay_market_maker.finite_float_or_none(tick_size)
    if tick is None or tick <= 0:
        return float(price)
    scaled = float(price) / tick
    rounded = (np.floor(scaled) if side == "bid" else np.ceil(scaled)) * tick
    return float(round(rounded, 12))


def test_round_price_for_side_matches_the_original_formula():
    rng = np.random.default_rng(99)
    ticks = [0.0, -1.0, np.nan, 1e-6, 1e-5, 0.01, 0.1, 0.5, 1.0, 1e-8, 1 / 3]
    for tick in ticks:
        for _ in range(400):
            price = float(rng.choice([1e-4, 0.1015, 3.5, 100.0, 98765.4321])) * float(
                rng.uniform(0.5, 2.0)
            )
            for side in ("bid", "ask"):
                assert repr(replay_market_maker.round_price_for_side(side, price, tick)) == repr(
                    _row_wise_round_price(side, price, tick)
                ), (side, price, tick)


def test_round_price_to_tick_keeps_twelve_digits():
    """A tick and price where rounding at 11 digits and at 12 disagree.

    Contrived on purpose -- a 1e-6 tick on a ten-cent coin never exposes the
    difference -- but the digit count is part of the quoting arithmetic and this
    is what stops it being 'simplified'.
    """
    tick = 1 / 3
    price = 166.35730288843064
    twelve = replay_market_maker.round_price_to_tick("bid", price, tick)
    eleven = float(round(np.floor(price / tick) * tick, 11))
    assert twelve != eleven
    assert repr(twelve) == repr(_row_wise_round_price("bid", price, tick))


def test_post_only_check_rejects_a_locked_book():
    """bid == ask is not a quotable book, and `>=` is what says so."""
    assert replay_market_maker.post_only_check("bid", 99.5, 100.0, 100.0) == (
        False,
        "crossed_or_invalid_book",
    )
    assert replay_market_maker.post_only_check("ask", 100.5, 100.0, 100.0) == (
        False,
        "crossed_or_invalid_book",
    )
    assert replay_market_maker.post_only_check("bid", 99.5, 100.0, np.nan) == (
        False,
        "crossed_or_invalid_book",
    )
    assert replay_market_maker.post_only_check("bid", 99.5, 100.0, 100.5) == (True, "ok")


def test_is_monotonic_ns():
    assert is_monotonic_ns(None)
    assert is_monotonic_ns(np.array([], dtype="int64"))
    assert is_monotonic_ns(np.array([5], dtype="int64"))
    assert is_monotonic_ns(np.array([1, 1, 2, 9], dtype="int64"))
    assert not is_monotonic_ns(np.array([1, 9, 2], dtype="int64"))


# ---------------------------------------------------------------------------
# The fill rule, against a transcription of the loop it replaced
# ---------------------------------------------------------------------------


def _row_wise_matching_trade(side, price, window, queue_ahead, queue_decay_per_second, active_at):
    """The pre-rewrite ``matching_trade_with_queue_decay``, copied verbatim.

    Kept here rather than imported so this stays a comparison against the OLD
    code even after the old code is gone.
    """
    remaining_queue = max(0.0, float(queue_ahead))
    initial_queue = remaining_queue
    decay_rate = max(0.0, float(queue_decay_per_second))
    queue_decay_base = 0.0
    last_ts = pd.Timestamp(active_at) if active_at is not None else None
    for _, trade in window.iterrows():
        trade_ts = trade.get("timestamp", None)
        if last_ts is None and trade_ts is not None:
            last_ts = pd.Timestamp(trade_ts)
        if decay_rate > 0 and last_ts is not None and trade_ts is not None and remaining_queue > 0:
            elapsed_seconds = max(0.0, (pd.Timestamp(trade_ts) - last_ts).total_seconds())
            decay = min(remaining_queue, initial_queue * decay_rate * elapsed_seconds)
            remaining_queue -= decay
            queue_decay_base += decay
            last_ts = pd.Timestamp(trade_ts)

        trade_price = float(trade.get("price", np.nan))
        if not np.isfinite(trade_price):
            continue
        crosses = (side == "bid" and trade_price <= price) or (side == "ask" and trade_price >= price)
        if not crosses:
            continue
        trade_size = float(trade.get("size", 0.0) or 0.0)
        if remaining_queue > 0:
            remaining_queue -= max(0.0, trade_size)
            if remaining_queue >= 0:
                continue
        return trade, queue_decay_base
    return None, queue_decay_base


def _random_window(rng, rows):
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z").value
    offsets = np.sort(rng.integers(0, 5_000_000_000, size=rows))
    sizes = rng.choice([0.0, 0.25, 0.5, 1.0, np.nan], size=rows, p=[0.15, 0.25, 0.25, 0.25, 0.10])
    prices = rng.choice([99.0, 100.0, 101.0, np.nan], size=rows, p=[0.3, 0.3, 0.3, 0.1])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts0 + offsets, utc=True),
            "price": prices,
            "size": sizes,
            "side": ["buy" if value else "sell" for value in rng.integers(0, 2, size=rows)],
        }
    )


def test_scan_matches_the_row_wise_fill_rule_over_random_windows():
    rng = np.random.default_rng(4242)
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    for trial in range(200):
        window = _random_window(rng, int(rng.integers(0, 12)))
        side = "bid" if trial % 2 else "ask"
        price = float(rng.choice([99.0, 100.0, 100.5, 101.0]))
        queue_ahead = float(rng.choice([0.0, 0.4, 1.2, np.nan]))
        decay = float(rng.choice([0.0, 0.05, 1.0]))
        active_at = ts0 if trial % 3 else None

        expected_trade, expected_decay = _row_wise_matching_trade(
            side, price, window, queue_ahead, decay, active_at
        )
        actual_trade, actual_decay = matching_trade_with_queue_decay(
            side,
            price,
            window,
            queue_ahead,
            queue_decay_per_second=decay,
            active_at=active_at,
        )
        assert repr(actual_decay) == repr(expected_decay), trial
        assert (expected_trade is None) == (actual_trade is None), trial
        if expected_trade is not None:
            assert actual_trade.name == expected_trade.name, trial


def test_scan_skips_consumed_trades_without_filtering_the_frame():
    """The used-trade mask must remove rows the old code removed with isin()."""
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    window = pd.DataFrame(
        {
            "timestamp": [ts0, ts0 + pd.Timedelta(milliseconds=1), ts0 + pd.Timedelta(milliseconds=2)],
            "price": [100.0, 100.0, 100.0],
            "size": [1.0, 1.0, 1.0],
        }
    )
    price_arr, size_arr, falsy = trade_event_arrays(window)
    ts_ns = timestamps_as_ns(window)
    used = [True, False, False]
    index, _decay = scan_for_matching_trade(
        "bid",
        100.0,
        range(3),
        trade_ts_ns=ts_ns,
        trade_price=price_arr,
        trade_size=size_arr,
        trade_size_falsy=falsy,
        used=used,
        queue_ahead=0.0,
    )
    assert index == 1
    expected, _ = _row_wise_matching_trade("bid", 100.0, window.iloc[1:], 0.0, 0.0, None)
    assert expected.name == 1


def test_scan_treats_an_unknown_side_as_crossing_nothing():
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    window = pd.DataFrame({"timestamp": [ts0], "price": [100.0], "size": [1.0]})
    trade, decay = matching_trade_with_queue_decay("sideways", 100.0, window, 0.0)
    assert trade is None and decay == 0.0


# ---------------------------------------------------------------------------
# The sweep's tape pinning
# ---------------------------------------------------------------------------


def test_freeze_thaw_round_trip_preserves_the_tape_exactly():
    """Pool workers score a parquet copy; the inline path scores the frames.

    If the round trip narrowed a float64 to float32 or dropped the timezone the
    two would disagree and nobody would see it, because the sweep never compares
    them. This is that comparison.
    """
    import sweep_replay

    tape = reference_tape(replay_market_maker)
    with tempfile.TemporaryDirectory() as tmp:
        sweep_replay.freeze_tape(tape, Path(tmp) / "tape")
        thawed = sweep_replay.thaw_tape(Path(tmp) / "tape")

    for name in ("prices", "trades", "orderbooks"):
        before = getattr(tape, name)
        after = getattr(thawed, name)
        assert list(before.columns) == list(after.columns), name
        assert before.dtypes.equals(after.dtypes), f"{name}: {before.dtypes} vs {after.dtypes}"
        assert before.index.equals(after.index), name
        for column in before.columns:
            left, right = before[column], after[column]
            if pd.api.types.is_float_dtype(left):
                assert np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True), f"{name}.{column}"
            else:
                assert left.equals(right), f"{name}.{column}"


def test_frozen_and_in_memory_tapes_score_identically():
    """The guarantee the round trip exists to provide, stated as an outcome."""
    import sweep_replay

    tape = reference_tape(replay_market_maker)
    config = ReplayConfig(**dict(reference_configs())["lat500_ref30000_phi10_twosided6_episodic"])
    in_memory = run_replay(config, dict(REFERENCE_PARAMS), tape=tape).to_dict()
    with tempfile.TemporaryDirectory() as tmp:
        sweep_replay.freeze_tape(tape, Path(tmp) / "tape")
        thawed = sweep_replay.thaw_tape(Path(tmp) / "tape")
    from_disk = run_replay(config, dict(REFERENCE_PARAMS), tape=thawed).to_dict()

    flat_memory: dict = {}
    flat_disk: dict = {}
    _flatten("run", in_memory, flat_memory)
    _flatten("run", from_disk, flat_disk)
    assert flat_memory == flat_disk


def test_calibration_cache_key_covers_every_input():
    import sweep_replay

    base = dict(
        symbol="CASHCAT",
        data_dir=Path("."),
        window_start=pd.Timestamp("2026-08-17T00:00:00Z"),
        window_end=pd.Timestamp("2026-08-17T12:00:00Z"),
    )
    calibration = sweep_replay.Calibration()
    key = sweep_replay.calibration_cache_key(calibration, **base)
    assert key == sweep_replay.calibration_cache_key(calibration, **base)
    for other in (
        sweep_replay.Calibration(epsilon_ms_plus=500),
        sweep_replay.Calibration(epsilon_ms_minus=500),
        sweep_replay.Calibration(kappa_support_lower_plus=0.5),
        sweep_replay.Calibration(kappa_support_lower_minus=0.5),
    ):
        assert sweep_replay.calibration_cache_key(other, **base) != key
    assert sweep_replay.calibration_cache_key(calibration, **{**base, "symbol": "ETH"}) != key
    assert (
        sweep_replay.calibration_cache_key(
            calibration, **{**base, "window_end": pd.Timestamp("2026-08-17T13:00:00Z")}
        )
        != key
    )


def test_data_fingerprint_tracks_shard_content(tmp_path):
    import sweep_replay

    stream = tmp_path / "CASHCAT" / "prices"
    stream.mkdir(parents=True)
    (stream / "a.parquet").write_bytes(b"0123")
    first = sweep_replay.data_fingerprint(tmp_path, "CASHCAT")
    assert first == sweep_replay.data_fingerprint(tmp_path, "CASHCAT")

    (stream / "a.parquet").write_bytes(b"01234")
    assert sweep_replay.data_fingerprint(tmp_path, "CASHCAT") != first

    (stream / "b.parquet").write_bytes(b"9")
    grown = sweep_replay.data_fingerprint(tmp_path, "CASHCAT")
    assert grown != first
    (stream / "b.parquet").unlink()
    assert sweep_replay.data_fingerprint(tmp_path, "CASHCAT") != grown


def test_calibration_cache_survives_appends_but_not_rewrites(tmp_path):
    """The exact rule the calibration cache reuses a fit under.

    A collector appending new shards must NOT invalidate a fit -- otherwise the
    cache never fires on the host it exists for -- but a shard the fit actually
    read being rewritten, truncated or deleted must.
    """
    import sweep_replay

    stream = tmp_path / "CASHCAT" / "trades"
    stream.mkdir(parents=True)
    (stream / "a.parquet").write_bytes(b"0123")
    (stream / "b.parquet").write_bytes(b"45")
    read = sweep_replay.shard_index(tmp_path, "CASHCAT")
    assert read == {"trades/a.parquet": 4, "trades/b.parquet": 2}

    assert sweep_replay.shard_index_still_holds(read, sweep_replay.shard_index(tmp_path, "CASHCAT"))

    (stream / "c.parquet").write_bytes(b"new flush")
    assert sweep_replay.shard_index_still_holds(read, sweep_replay.shard_index(tmp_path, "CASHCAT"))

    (stream / "b.parquet").write_bytes(b"456")
    assert not sweep_replay.shard_index_still_holds(read, sweep_replay.shard_index(tmp_path, "CASHCAT"))

    (stream / "b.parquet").write_bytes(b"45")
    assert sweep_replay.shard_index_still_holds(read, sweep_replay.shard_index(tmp_path, "CASHCAT"))
    (stream / "a.parquet").unlink()
    assert not sweep_replay.shard_index_still_holds(read, sweep_replay.shard_index(tmp_path, "CASHCAT"))

    assert not sweep_replay.shard_index_still_holds(read, None)


def test_empty_streams_still_replay():
    """A tape with no trades and no book must not reach the array paths at all."""
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        {
            "timestamp": [ts0, ts0 + pd.Timedelta(seconds=1)],
            "bid": [100.0, 100.1],
            "ask": [101.0, 101.1],
        }
    )
    tape = ReplayTape(
        prices=prices,
        trades=pd.DataFrame(),
        orderbooks=pd.DataFrame(),
        input_files={"prices": 0, "trades": 0, "orderbooks": 0},
    )
    metrics = run_replay(
        ReplayConfig(symbol="X", data_dir=Path("."), mid_fallback=100.5, inventory_unit_base=0.01, q_max=3),
        dict(REFERENCE_PARAMS),
        tape=tape,
    ).to_dict()
    assert metrics["maker_fills"] == 0
    assert metrics["stale_quote_cancels"] > 0
