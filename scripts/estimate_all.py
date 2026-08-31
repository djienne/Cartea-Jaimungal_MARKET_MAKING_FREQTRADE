#!/usr/bin/env python3
"""Run the kappa, epsilon and lambda estimators over ONE shared market window.

Loading the window once avoids three independent Parquet scans per cycle. At a
10 s flush cadence, repeated scans previously dominated estimator runtime and
could overlap even while collection itself was healthy.

This loads the window once and hands it to each estimator. The three CLIs still
work standalone for manual use; they take an optional preloaded window.

Ordering matters: kappa runs first because get_epsilon reads kappa.json for its
toxicity diagnostic.

--emit-params-json turns the cycle into an isolated calibration run: one window
load, kappa/epsilon computed at the requested settings, and one JSON written to
the specified path. Normal snapshot files under ``scripts/`` are not modified,
so sweeps cannot contaminate later manual analyses.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from estimator_common import EMIT_PAYLOAD_KIND, build_emit_window_block, load_market_window, write_emit_payload
from get_epsilon import run_epsilon_for_crypto
from get_kappa import run_kappa_for_crypto
from get_lambda import compute_lambda_from_trades, save_lambda_to_json
from param_utils import PARAM_SCHEMA_VERSION, timestamp_to_iso, utc_now_iso

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
    post_horizon_ms: int | None = None,
    data_dir: str | None = None,
    post_horizon_ms_plus: int | None = None,
    post_horizon_ms_minus: int | None = None,
    support_quantile_plus: float | None = None,
    support_quantile_minus: float | None = None,
    support_quantile_lower_plus: float | None = None,
    support_quantile_lower_minus: float | None = None,
    window_start: Any = None,
    window_end: Any = None,
    emit_params_json: str | Path | None = None,
) -> dict[str, Any]:
    """One cycle: load the window once, then kappa -> epsilon -> lambda monitor.

    With ``emit_params_json`` the cycle computes the same numbers but writes them
    only to that path -- no snapshot file is touched and the lambda_trades
    monitor is skipped, since it too writes into the live tree.
    """
    started = time.monotonic()
    window = load_market_window(
        crypto, minutes, data_dir=data_dir, window_start=window_start, window_end=window_end
    )
    load_seconds = time.monotonic() - started
    meta = dict(window.meta or {})
    print(
        f"[window] {crypto}: {meta.get('n_mid_updates', 0)} mids, "
        f"{meta.get('n_trade_prints', 0)} trade prints, "
        f"{meta.get('shards_read', 0)} shards read "
        f"({meta.get('shards_skipped_outside_window', 0)} skipped outside window, "
        f"{meta.get('shards_failed', 0)} unreadable) in {load_seconds:.2f}s"
    )

    emitting = emit_params_json is not None
    kappa_sink: dict[str, Any] | None = {} if emitting else None
    epsilon_sink: dict[str, Any] | None = {} if emitting else None

    run_kappa_for_crypto(
        crypto,
        minutes=minutes,
        window=window,
        support_quantile_plus=support_quantile_plus,
        support_quantile_minus=support_quantile_minus,
        support_quantile_lower_plus=support_quantile_lower_plus,
        support_quantile_lower_minus=support_quantile_lower_minus,
        emit_sink=kappa_sink,
    )

    # In emit mode epsilon's toxicity diagnostic uses the kappa just computed on
    # THIS slice rather than whatever kappa.json holds; that is the only way the
    # emitted kappa*epsilon belongs to a single window.
    kappa_override = None
    if emitting and isinstance(kappa_sink, dict):
        kappa_entry = kappa_sink.get("kappa") or {}
        kp, km = kappa_entry.get("kappa+"), kappa_entry.get("kappa-")
        if kp is not None and km is not None:
            kappa_override = (float(kp), float(km))

    run_epsilon_for_crypto(
        crypto,
        minutes=minutes,
        post_horizon_ms=post_horizon_ms,
        post_horizon_ms_plus=post_horizon_ms_plus,
        post_horizon_ms_minus=post_horizon_ms_minus,
        window=window,
        emit_sink=epsilon_sink,
        kappa_override=kappa_override,
    )

    if emitting:
        payload = {
            "kind": EMIT_PAYLOAD_KIND,
            "schema_version": PARAM_SCHEMA_VERSION,
            "crypto": crypto,
            "generated_at": utc_now_iso(),
            "window": build_emit_window_block(window, minutes),
            "calibration": {
                **dict((kappa_sink or {}).get("calibration", {})),
                **dict((epsilon_sink or {}).get("calibration", {})),
            },
            "kappa": (kappa_sink or {}).get("kappa"),
            "lambda": (kappa_sink or {}).get("lambda"),
            "epsilon": (epsilon_sink or {}).get("epsilon"),
        }
        written = write_emit_payload(emit_params_json, payload)
        print(f"[emit] kappa/lambda/epsilon params -> {written}")
    else:
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
        "emitted_to": str(emit_params_json) if emitting else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run kappa/epsilon/lambda estimators over one shared market window."
    )
    parser.add_argument("--crypto", "-c", default=os.getenv("CRYPTO_NAME", "CASHCAT"))
    parser.add_argument("--minutes", "-m", type=int, default=30)
    parser.add_argument("--post-horizon-ms", type=int, default=None,
                        help="Shorthand: epsilon post-trade horizon in ms for BOTH sides")
    parser.add_argument("--post-horizon-ms-plus", type=int, default=None,
                        help="epsilon+ post-trade horizon in ms (buy MOs / our resting ask)")
    parser.add_argument("--post-horizon-ms-minus", type=int, default=None,
                        help="epsilon- post-trade horizon in ms (sell MOs / our resting bid)")
    parser.add_argument("--support-quantile", type=float, default=None,
                        help="Shorthand: upper quantile of the kappa fit support on both sides")
    parser.add_argument("--support-quantile-plus", type=float, default=None)
    parser.add_argument("--support-quantile-minus", type=float, default=None)
    parser.add_argument("--support-quantile-lower", type=float, default=None,
                        help="Shorthand: lower quantile of the kappa fit support on both sides")
    parser.add_argument("--support-quantile-lower-plus", type=float, default=None)
    parser.add_argument("--support-quantile-lower-minus", type=float, default=None)
    parser.add_argument("--window-start", default=None,
                        help="ISO-8601 (or epoch) start of the window; selects a historical slice")
    parser.add_argument("--window-end", default=None,
                        help="ISO-8601 (or epoch) end of the window (clamped to the data that exists)")
    parser.add_argument("--emit-params-json", default=None,
                        help="Write kappa/lambda/epsilon to this path and write NOTHING else: the "
                             "live snapshots are left untouched. Use for calibration sweeps.")
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.window_start or args.window_end) and not args.emit_params_json:
        print("WARNING: --window-start/--window-end without --emit-params-json will OVERWRITE the "
              "live snapshots with parameters fitted on a historical slice.")
    run_all(
        str(args.crypto).strip().upper(),
        minutes=args.minutes,
        post_horizon_ms=args.post_horizon_ms,
        post_horizon_ms_plus=args.post_horizon_ms_plus,
        post_horizon_ms_minus=args.post_horizon_ms_minus,
        support_quantile_plus=(
            args.support_quantile_plus if args.support_quantile_plus is not None else args.support_quantile
        ),
        support_quantile_minus=(
            args.support_quantile_minus if args.support_quantile_minus is not None else args.support_quantile
        ),
        support_quantile_lower_plus=(
            args.support_quantile_lower_plus
            if args.support_quantile_lower_plus is not None
            else args.support_quantile_lower
        ),
        support_quantile_lower_minus=(
            args.support_quantile_lower_minus
            if args.support_quantile_lower_minus is not None
            else args.support_quantile_lower
        ),
        window_start=args.window_start,
        window_end=args.window_end,
        emit_params_json=args.emit_params_json,
        data_dir=args.data_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
