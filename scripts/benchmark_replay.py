#!/usr/bin/env python3
"""Time ``run_replay`` on a pinned tape, so a change to it can be defended.

WHY THIS EXISTS. The replay is the research programme's inner loop: the sweep
driver runs 100-400 passes over one tape, so a constant factor here is the
difference between an overnight grid and an unaffordable one. It used to cost
about 1.8 ms per price event -- roughly 300 s for a 148k-row day -- and almost
all of that was pandas scalar access, not arithmetic. Anyone touching that hot
loop again needs a number, not an impression.

WHAT IT MEASURES, AND WHAT IT DOES NOT. It times ``run_replay`` only, with the
tape already loaded and the frames already normalised, because that is the part
a sweep pays per grid point (``load_tape`` is paid once). The HJB solve is
inside the measurement because run_replay does one per call; ``--report-solve``
splits it out, since a config whose replay is fast and whose solve is slow needs
a different fix.

    # deterministic synthetic tape, runs anywhere, no collected data needed
    python scripts/benchmark_replay.py --rows 50000

    # the real thing
    python scripts/benchmark_replay.py --data-dir scripts/HL_data --symbol CASHCAT

Timings are reported as the MINIMUM over the repeats, not the mean: this host
runs the Rust dry-run grid and several collectors, so the spread
between repeats is contention and the minimum is the closest available estimate
of the work actually done. Run it with --repeats 1 and you are measuring the
machine's mood as much as the code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from replay_market_maker import (  # noqa: E402
    ReplayConfig,
    ReplayTape,
    load_tape,
    run_replay,
    solve_replay_hjb,
)

# The live CASHCAT calibration, so the synthetic tape is scored at plausible
# depths rather than ones that never fill.
BENCH_PARAMS = {
    "kappa+": 10179.501680696987,
    "kappa-": 10131.639846298318,
    "lambda+": 0.18384454888412738,
    "lambda-": 0.2720996837612637,
    "epsilon+": 2.0594489447746147e-05,
    "epsilon-": 1.5018482890699448e-05,
}

# Three historical replay scenarios. A short refresh interval quotes at nearly
# every price event and is the expensive case; the 30 s case is retained to
# reproduce the former slow-refresh comparison, not as a current runtime claim.
BENCH_SCENARIOS = {
    "fast_requote": dict(
        decision_latency_ms=50,
        order_ack_latency_ms=50,
        cancel_latency_ms=250,
        quote_refresh_interval_ms=100,
    ),
    "this_stack": dict(
        decision_latency_ms=250,
        order_ack_latency_ms=250,
        cancel_latency_ms=250,
        quote_refresh_interval_ms=1000,
    ),
    "reality": dict(
        decision_latency_ms=250,
        order_ack_latency_ms=250,
        cancel_latency_ms=250,
        quote_refresh_interval_ms=30000,
    ),
}


# Half-spreads the synthetic book quotes at, in bps. Discrete because a real
# book's spread is a whole number of ticks, and because JOIN_HALF_SPREAD_BPS
# has to be reachable: a quote pinned to that half-spread then lands ON the
# touch, which is the only way the queue-ahead and queue-decay branches run.
SYNTHETIC_HALF_SPREAD_BPS = (8.0, 12.0, 16.0, 20.0, 26.0, 34.0, 44.0, 60.0)
JOIN_HALF_SPREAD_BPS = 20.0
SYNTHETIC_TICK = 1e-6
_NS_PER_MS = 1_000_000


def synthetic_tape(rows: int, seed: int = 20260817) -> ReplayTape:
    """A deterministic tape with the shape of the collected data.

    Not a market model and not meant to be one -- it exists so the benchmark
    runs on a machine with no collected data and still exercises every branch
    the hot loop has: irregular inter-arrival gaps, trades far enough from the
    touch to fill a quote resting tens of bps out, a top-of-book size series,
    and a book that is tick-aligned so a quote can join the touch exactly.

    That last property is the fiddly one and it is deliberate. Quotes here rest
    15-50 bps out, so with a continuously-distributed book they would never sit
    exactly at the touch, ``is_joining_best`` would never fire, and the whole
    queue-ahead and queue-decay path -- a third of the fill rule -- would go
    unexercised by anything built on this tape.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-08-17T00:00:00Z").value
    # Log-normal gaps, so the cadence is bursty like the real BBO stream, and
    # quantised to WHOLE MILLISECONDS because that is the resolution
    # Hyperliquid stamps its messages at. The quantisation is not cosmetic: the
    # replay's latency and staleness offsets are whole milliseconds too, so on a
    # real tape trades land exactly ON a window boundary all the time. With
    # nanosecond-random timestamps they never would, and every boundary rule in
    # the fill logic -- which searchsorted side, whether stale_at is inclusive --
    # would be untestable by anything built on this tape.
    gaps = np.maximum(1, (np.exp(rng.normal(np.log(2e8), 0.9, size=rows)) // 1e6).astype("int64"))
    price_ns = start + np.cumsum(gaps) * _NS_PER_MS
    mid = 0.1015 * np.exp(np.cumsum(rng.normal(0.0, 2e-4, size=rows)))
    half_bps = rng.choice(SYNTHETIC_HALF_SPREAD_BPS, size=rows)
    half = mid * (half_bps / 1e4)
    # Rounded the way the harness rounds a quote -- bids down, asks up -- so a
    # quote clamped to the same half-spread lands on the same tick.
    bid = np.floor((mid - half) / SYNTHETIC_TICK) * SYNTHETIC_TICK
    ask = np.ceil((mid + half) / SYNTHETIC_TICK) * SYNTHETIC_TICK
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(price_ns, utc=True),
            "bid": np.round(bid, 12),
            "ask": np.round(ask, 12),
        }
    )
    mid = (prices["bid"].to_numpy() + prices["ask"].to_numpy()) / 2.0

    n_trades = max(1, rows // 2)
    trade_ns = np.sort(
        rng.choice(price_ns, size=n_trades, replace=True)
        + rng.integers(0, 100, size=n_trades) * _NS_PER_MS
    )
    trade_mid = np.interp(trade_ns, price_ns, mid)
    # Heavy-tailed offsets from mid. A thin Gaussian would put every trade
    # inside the touch and nothing would ever fill -- the model's own depths sit
    # 15-50 bps out at this calibration -- so the benchmark would be timing a
    # replay that never takes the fill path. A Student-t at 2.5 degrees of
    # freedom reaches those depths often enough to exercise it.
    offsets = rng.standard_t(2.5, size=n_trades) * 8e-4
    sizes = np.abs(rng.normal(3000.0, 1500.0, size=n_trades))
    # A few exact zeros. The fill rule reads size as ``float(x or default)``, so
    # a zero takes the default rather than the value, and the two call sites use
    # different defaults -- a rule no tape of strictly positive sizes can check.
    sizes[rng.random(n_trades) < 0.02] = 0.0
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(trade_ns, utc=True),
            "price": trade_mid * (1.0 + offsets),
            "size": sizes.astype("float32"),
        }
    )

    n_books = max(1, rows // 10)
    book_ns = np.sort(rng.choice(price_ns, size=n_books, replace=False) if n_books <= rows else price_ns)
    orderbooks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(book_ns, utc=True),
            "bid_size_0": np.abs(rng.normal(50000.0, 20000.0, size=len(book_ns))).astype("float32"),
            "ask_size_0": np.abs(rng.normal(50000.0, 20000.0, size=len(book_ns))).astype("float32"),
        }
    )
    for frame in (prices, trades, orderbooks):
        frame.attrs["event_clock"] = "synthetic"
    return ReplayTape(
        prices=prices,
        trades=trades,
        orderbooks=orderbooks,
        input_files={"prices": 0, "trades": 0, "orderbooks": 0},
    )


def truncate(tape: ReplayTape, rows: int | None) -> ReplayTape:
    """A contiguous prefix, cut on the price stream and applied to all three."""
    if not rows or len(tape.prices) <= rows:
        return tape
    prices = tape.prices.iloc[:rows]
    lo, hi = prices["timestamp"].iloc[0], prices["timestamp"].iloc[-1]

    def cut(frame: pd.DataFrame) -> pd.DataFrame:
        if not len(frame) or "timestamp" not in frame.columns:
            return frame
        kept = frame[(frame["timestamp"] >= lo) & (frame["timestamp"] <= hi)].reset_index(drop=True)
        kept.attrs = dict(frame.attrs)
        return kept

    kept_prices = prices.reset_index(drop=True)
    kept_prices.attrs = dict(tape.prices.attrs)
    return ReplayTape(
        prices=kept_prices,
        trades=cut(tape.trades),
        orderbooks=cut(tape.orderbooks),
        input_files=dict(tape.input_files),
    )


def base_config(args: argparse.Namespace) -> ReplayConfig:
    return ReplayConfig(
        symbol=args.symbol,
        data_dir=Path(args.data_dir) if args.data_dir else SCRIPTS / "HL_data",
        mid_fallback=args.mid,
        inventory_unit_base=args.inventory_unit_base,
        price_tick_size=args.price_tick_size,
        q_max=args.q_max,
        q_min=-args.q_max,
        hjb_phi_kappa_t=args.phi_kappa_t,
        hjb_phi_kappa_t_max=max(50.0, args.phi_kappa_t),
        hjb_alpha_kappa=0.05,
        hjb_horizon_seconds=args.horizon_seconds,
        hjb_time_mode=args.time_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Collected data root. Omitted uses a deterministic synthetic tape.")
    parser.add_argument("--symbol", default="CASHCAT")
    parser.add_argument("--rows", type=int, default=None,
                        help="Price rows to score. Required for the synthetic tape; truncates a real one.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenario", action="append", choices=sorted(BENCH_SCENARIOS),
                        help="Repeatable; default runs all three.")
    parser.add_argument("--time-mode", default="episodic", choices=("episodic", "stationary"))
    parser.add_argument("--horizon-seconds", type=float, default=150.0)
    parser.add_argument("--phi-kappa-t", type=float, default=10.0)
    parser.add_argument("--q-max", type=int, default=6)
    parser.add_argument("--mid", type=float, default=0.1015)
    parser.add_argument("--price-tick-size", type=float, default=1e-6)
    parser.add_argument("--inventory-unit-base", type=float, default=2092.0)
    parser.add_argument("--report-solve", action="store_true",
                        help="Time the HJB solve separately and net it out of the replay.")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.data_dir:
        config = base_config(args)
        loaded_at = time.perf_counter()
        tape = truncate(load_tape(config), args.rows)
        load_seconds = time.perf_counter() - loaded_at
        source = f"{args.symbol} from {args.data_dir}"
    else:
        rows = args.rows or 25_000
        loaded_at = time.perf_counter()
        tape = synthetic_tape(rows)
        load_seconds = time.perf_counter() - loaded_at
        source = f"synthetic({rows} rows)"

    base = base_config(args)
    price_rows = len(tape.prices)
    print(f"tape: {source} -- {price_rows} price rows, {len(tape.trades)} trades, "
          f"{len(tape.orderbooks)} book snapshots (loaded in {load_seconds:.2f} s)")
    print(f"config: q_max={args.q_max} phi_kappa_t={args.phi_kappa_t} T={args.horizon_seconds} "
          f"mode={args.time_mode}, {args.repeats} repeats, reporting the minimum\n")

    solve_seconds = None
    if args.report_solve:
        timings = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            solve_replay_hjb(base, dict(BENCH_PARAMS))
            timings.append(time.perf_counter() - started)
        solve_seconds = min(timings)
        print(f"{'hjb solve':<16}{solve_seconds:>10.3f} s\n")

    names = args.scenario or list(BENCH_SCENARIOS)
    header = f"{'scenario':<16}{'seconds':>10}{'us/event':>11}{'events/s':>12}{'fills':>8}{'decisions':>11}"
    print(header)
    print("-" * len(header))
    results = {}
    for name in names:
        config = ReplayConfig(**{**base.__dict__, **BENCH_SCENARIOS[name]})
        timings = []
        metrics = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            metrics = run_replay(config, dict(BENCH_PARAMS), tape=tape)
            timings.append(time.perf_counter() - started)
        seconds = min(timings)
        payload = metrics.to_dict()
        per_event_us = seconds / max(price_rows, 1) * 1e6
        results[name] = {
            "seconds": seconds,
            "seconds_all_repeats": timings,
            "us_per_price_event": per_event_us,
            "price_events": price_rows,
            "maker_fills": payload["maker_fills"],
            "quote_decision_events": payload["quote_decision_events"],
            "quote_attempts": payload["quote_attempts"],
        }
        print(f"{name:<16}{seconds:>10.3f}{per_event_us:>11.2f}{price_rows / max(seconds, 1e-9):>12,.0f}"
              f"{payload['maker_fills']:>8}{payload['quote_decision_events']:>11}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "source": source,
                    "price_rows": price_rows,
                    "trade_rows": len(tape.trades),
                    "orderbook_rows": len(tape.orderbooks),
                    "repeats": args.repeats,
                    "hjb_solve_seconds": solve_seconds,
                    "scenarios": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\n[json] {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
