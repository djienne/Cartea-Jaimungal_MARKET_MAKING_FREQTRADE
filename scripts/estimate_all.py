#!/usr/bin/env python3
"""Run the kappa, epsilon and lambda estimators over ONE shared market window.

periodic_test_runner.py used to spawn get_kappa.py, get_epsilon.py and
get_lambda.py as three separate processes. Each re-read the same parquet shards
from scratch, so a cycle paid for the same directory scan three times -- at 60
minute retention and a 10 s flush that is ~1800 parquet opens per 30 s cycle
under a 0.25 CPU limit. Cycles ran long, overlapped, and on 2026-08-16 the
strategy went 87 minutes with no fresh parameters while the collector was
perfectly healthy.

This loads the window once and hands it to each estimator. The three CLIs still
work standalone for manual use; they take an optional preloaded window.

Ordering matters: kappa runs first because get_epsilon reads kappa.json for its
toxicity diagnostic.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from estimator_common import load_market_window
from get_epsilon import run_epsilon_for_crypto
from get_kappa import run_kappa_for_crypto
from get_lambda import compute_lambda_from_trades, save_lambda_to_json
from param_utils import timestamp_to_iso, utc_now_iso

import pandas as pd


def _lambda_trades_monitor(window, crypto: str, output: str = "lambda_trades.json") -> None:
    """Raw trades/sec sanity file, computed from the already-loaded window.

    get_lambda.py's own loader re-reads the trades directory and works in local
    receive time; this uses the shared window (already de-duplicated on trade_id
    and on the window's chosen clock), so the monitor agrees with what the
    estimators actually saw.
    """
    trades = window.trades
    if trades is None or trades.empty:
        print("[lambda-monitor] no trades in window; skipping.")
        return
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(trades["ts_ms"], unit="ms"),
            "side": trades["side"],
        }
    )
    results = compute_lambda_from_trades(frame)
    if results is None:
        print("[lambda-monitor] unable to compute lambda; skipping.")
        return
    save_lambda_to_json(
        results.lambda_plus,
        results.lambda_minus,
        crypto,
        filename=output,
        metadata={
            "status": "ok" if results.n_trades_total > 0 else "insufficient_data",
            "window_start": timestamp_to_iso(results.start_time),
            "window_end": timestamp_to_iso(results.end_time),
            "generated_at": utc_now_iso(),
            "n_trades_buy": int(results.n_trades_buy),
            "n_trades_sell": int(results.n_trades_sell),
            "n_trades_total": int(results.n_trades_total),
        },
    )


def run_all(
    crypto: str,
    *,
    minutes: int = 30,
    ema_tau: float | None = None,
    post_horizon_ms: int | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """One cycle: load the window once, then kappa -> epsilon -> lambda monitor."""
    started = time.monotonic()
    window = load_market_window(crypto, minutes, data_dir=data_dir)
    load_seconds = time.monotonic() - started
    meta = dict(window.meta or {})
    print(
        f"[window] {crypto}: {meta.get('n_mid_updates', 0)} mids, "
        f"{meta.get('n_trade_prints', 0)} trade prints, "
        f"{meta.get('shards_read', 0)} shards read "
        f"({meta.get('shards_skipped_outside_window', 0)} skipped outside window, "
        f"{meta.get('shards_failed', 0)} unreadable) in {load_seconds:.2f}s"
    )

    run_kappa_for_crypto(crypto, minutes=minutes, ema_tau=ema_tau, window=window)
    run_epsilon_for_crypto(
        crypto,
        minutes=minutes,
        post_horizon_ms=post_horizon_ms,
        ema_tau=ema_tau,
        window=window,
    )
    _lambda_trades_monitor(window, crypto)

    total_seconds = time.monotonic() - started
    print(f"[cycle] {crypto}: estimators completed in {total_seconds:.2f}s")
    return {
        "crypto": crypto,
        "window_load_seconds": load_seconds,
        "cycle_seconds": total_seconds,
        "shards_read": meta.get("shards_read"),
        "shards_failed": meta.get("shards_failed"),
        "n_mid_updates": meta.get("n_mid_updates"),
        "n_trade_prints": meta.get("n_trade_prints"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run kappa/epsilon/lambda estimators over one shared market window."
    )
    parser.add_argument("--crypto", "-c", default=os.getenv("CRYPTO_NAME", "ETH"))
    parser.add_argument("--minutes", "-m", type=int, default=30)
    parser.add_argument("--ema-tau", type=float, default=None)
    parser.add_argument("--post-horizon-ms", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_all(
        str(args.crypto).strip().upper(),
        minutes=args.minutes,
        ema_tau=args.ema_tau,
        post_horizon_ms=args.post_horizon_ms,
        data_dir=args.data_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
