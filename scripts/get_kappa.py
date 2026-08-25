#!/usr/bin/env python3
"""Joint kappa/lambda estimation from Hyperliquid market data (schema v4).

Methodology:
- Trade prints are aggregated into market orders (same side + same exchange
  timestamp); one MO sweeping several levels is ONE arrival whose depth is its
  deepest print.
- Depths are measured from the prevailing MID (last BBO update strictly before
  the MO, exchange-timestamp aligned) — the same coordinate the strategy
  quotes in. Negative depths (stale-feed noise) are truncated to 0, not
  dropped.
- kappa± comes from a weighted log-linear fit of the empirical survival
  function P(depth >= delta): a resting order at depth delta fills when an MO
  walks at least that deep, so the model fill intensity is
  lambda(delta) = lambda± * exp(-kappa± * delta).
- The fit support is per side and configurable as quantiles of that side's own
  depth distribution: [--support-quantile-lower-{plus,minus},
  --support-quantile-{plus,minus}], defaulting to [0.0, 0.99] — the whole
  distribution up to the 99th percentile, which is the fit that has always
  shipped. Raising the lower bound refits kappa over only the deeper region
  where the measured edge is positive (see fit_kappa_survival for the argument).
- lambda± is the raw per-side MO arrival rate (count / covered window
  seconds). The old binned-density regression intercept equals
  lambda*kappa*binwidth (bin-width dependent) and is kept only as the
  lambda0_intercept_± diagnostic.
- Primary kappa±/lambda± values are the direct validated estimates from the
  selected market-data window; no temporal smoothing is applied.
- Realized mid variance sigma2_per_sec (USDC^2/s) is published for the
  strategy's volatility-aware inventory penalty.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

from estimator_common import (
    DEFAULT_SUPPORT_QUANTILE_LOWER,
    DEFAULT_SUPPORT_QUANTILE_UPPER,
    EMIT_PAYLOAD_KIND,
    MIN_KAPPA_FIT_POINTS,
    MIN_KAPPA_R2,
    aggregate_market_orders,
    attach_pre_mid,
    build_emit_window_block,
    fit_kappa_survival,
    load_market_window,
    mo_depths,
    realized_sigma2_per_sec,
    write_emit_payload,
)
from param_utils import (
    PARAM_SCHEMA_VERSION,
    atomic_write_json,
    finite_or_none,
    load_json_object,
    utc_now_iso,
)

# Verbosity control: 0=minimal, 1=verbose
VERBOSITY = 0

def _set_verbosity(v: int):
    global VERBOSITY
    try:
        VERBOSITY = 0 if v is None else max(0, int(v))
    except Exception:
        VERBOSITY = 0

def vprint(*args, level: int = 1, **kwargs):
    if VERBOSITY >= level:
        print(*args, **kwargs)

warnings.filterwarnings('ignore')

def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def list_available_cryptos(data_dir: str = None):
    """Return sorted list of crypto symbols that have data directories."""
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'HL_data')

    if not os.path.isdir(data_dir):
        return []
    try:
        candidates = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        valid = []
        for sym in candidates:
            has_mid = os.path.exists(os.path.join(data_dir, sym, 'prices')) or \
                os.path.exists(os.path.join(data_dir, sym, 'orderbooks'))
            if has_mid and os.path.exists(os.path.join(data_dir, sym, 'trades')):
                valid.append(sym)
        return sorted(valid)
    except Exception:
        return []


def build_kappa_lambda_entries(kappa_plus, kappa_minus, lambda_plus, lambda_minus,
                               metadata: dict | None = None,
                               raw_values: dict | None = None) -> tuple[dict, dict]:
    """The (kappa.json, lambda.json) entries for one symbol.

    Split out of save_kappa_lambda_to_json so emit mode can produce the identical
    entries without writing anywhere near the live snapshots.
    """
    metadata = dict(metadata or {})
    status = str(metadata.pop("status", "ok"))
    metadata.setdefault("generated_at", utc_now_iso())
    raw = dict(raw_values or {})

    raw_block = {
        "kappa+_raw": finite_or_none(raw.get("kappa+_raw", kappa_plus)),
        "kappa-_raw": finite_or_none(raw.get("kappa-_raw", kappa_minus)),
        "lambda+_raw": finite_or_none(raw.get("lambda+_raw", lambda_plus)),
        "lambda-_raw": finite_or_none(raw.get("lambda-_raw", lambda_minus)),
    }

    kappa_entry = {
        "schema_version": PARAM_SCHEMA_VERSION,
        "status": status,
        "kappa+": finite_or_none(kappa_plus),
        "kappa-": finite_or_none(kappa_minus),
        "lambda+": finite_or_none(lambda_plus),
        "lambda-": finite_or_none(lambda_minus),
        **raw_block,
        "lambda_source": "mo_survival_fit",
        "unit": {
            "kappa": "1/USDC",
            "lambda": "events_per_second",
        },
        **metadata,
    }
    lambda_entry = {
        "schema_version": PARAM_SCHEMA_VERSION,
        "status": status,
        "lambda+": finite_or_none(lambda_plus),
        "lambda-": finite_or_none(lambda_minus),
        "lambda+_raw": raw_block["lambda+_raw"],
        "lambda-_raw": raw_block["lambda-_raw"],
        "lambda_source": "mo_survival_fit",
        "unit": "events_per_second",
        **metadata,
    }
    return kappa_entry, lambda_entry


def save_kappa_lambda_to_json(kappa_plus, kappa_minus, lambda_plus, lambda_minus,
                              crypto: str, kappa_file: str = "kappa.json", lambda_file: str = "lambda.json",
                              metadata: dict | None = None, raw_values: dict | None = None):
    """Persist kappa and per-side MO arrival rates (per second) to JSON files.

    Primary keys hold the direct per-window estimates. *_raw aliases are retained
    for diagnostic/backward tooling and are identical to the primaries in v4.
    """
    kappa_entry, lambda_entry = build_kappa_lambda_entries(
        kappa_plus, kappa_minus, lambda_plus, lambda_minus,
        metadata=metadata, raw_values=raw_values,
    )
    kappa_data = load_json_object(kappa_file)
    lambda_data = load_json_object(lambda_file)
    kappa_data[crypto] = kappa_entry
    lambda_data[crypto] = lambda_entry

    atomic_write_json(kappa_file, kappa_data)
    atomic_write_json(lambda_file, lambda_data)

    print(f"[save] kappa -> {kappa_file}")
    print(f"[save] lambda (MO arrival rate) -> {lambda_file}")


def run_kappa_for_crypto(crypto: str, minutes: int = 30,
                         kappa_file: str = "kappa.json", lambda_file: str = "lambda.json",
                         data_dir=None, window=None,
                         support_quantile_plus: float | None = None,
                         support_quantile_minus: float | None = None,
                         support_quantile_lower_plus: float | None = None,
                         support_quantile_lower_minus: float | None = None,
                         window_start=None, window_end=None,
                         emit_params_json: str | Path | None = None,
                         emit_sink: dict | None = None):
    """Run the full kappa/lambda estimation flow for a single crypto symbol.

    ``window`` lets a caller that already loaded the market window (estimate_all.py)
    hand it over instead of paying for a second full parquet scan of the same data.

    The four support_* arguments set the per-side depth range the survival fit
    runs over (quantiles of that side's own depths). None means "use the shipped
    default", so the live estimator keeps the [0.0, 0.99] fit unchanged.

    ``emit_params_json`` / ``emit_sink`` switch on emit mode: the computed
    entries go to the given path (and/or into the given dict) and NOTHING is
    written to kappa.json or lambda.json, so a sweep can run alongside the live
    estimator container without disturbing any snapshot a live consumer reads.
    """
    support_upper_plus = (
        DEFAULT_SUPPORT_QUANTILE_UPPER if support_quantile_plus is None else float(support_quantile_plus)
    )
    support_upper_minus = (
        DEFAULT_SUPPORT_QUANTILE_UPPER if support_quantile_minus is None else float(support_quantile_minus)
    )
    support_lower_plus = (
        DEFAULT_SUPPORT_QUANTILE_LOWER
        if support_quantile_lower_plus is None
        else float(support_quantile_lower_plus)
    )
    support_lower_minus = (
        DEFAULT_SUPPORT_QUANTILE_LOWER
        if support_quantile_lower_minus is None
        else float(support_quantile_lower_minus)
    )
    emitting = emit_params_json is not None or emit_sink is not None

    log_section(f"KAPPA/LAMBDA FROM MARKET DATA - {crypto} (last {minutes} min)")
    if emitting:
        target = emit_params_json if emit_params_json is not None else "caller (combined emit)"
        print(f"[emit] kappa.json/lambda.json will NOT be written (emit mode); target: {target}")
    if window is None:
        try:
            window = load_market_window(
                crypto, minutes, data_dir=data_dir,
                window_start=window_start, window_end=window_end,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

    vprint(f"Mid source: {window.mid_source} ({window.ts_source} time)")
    vprint(f"Loaded {len(window.mids):,} mid updates and {len(window.trades):,} trade prints")

    mos = attach_pre_mid(aggregate_market_orders(window.trades), window.mids)
    depth_info = mo_depths(mos)

    n_buy_mos = int((mos["side"] == "buy").sum()) if not mos.empty else 0
    n_sell_mos = int((mos["side"] == "sell").sum()) if not mos.empty else 0

    buy_fit = fit_kappa_survival(
        depth_info["buy_depths"],
        support_quantile=support_upper_plus,
        support_quantile_lower=support_lower_plus,
    )
    sell_fit = fit_kappa_survival(
        depth_info["sell_depths"],
        support_quantile=support_upper_minus,
        support_quantile_lower=support_lower_minus,
    )

    # Arrival rate over the span both streams actually cover. Two corrections,
    # and they are different: trimming the START drops a window that begins
    # before the data does, while subtracting OUTAGE drops interior stretches
    # where the collector was down. Only the first existed before 2026-08-19, so
    # lambda was understated by exactly the missing interior fraction -- 5.6% on
    # the window holding the 7.9 min outage, and up to 50% inside the 1 h gap
    # the acceptance gate now tolerates. See estimator_common.outage_seconds for
    # why a silent mid stream on its own is not evidence of an outage.
    if window.mids.empty:
        covered_seconds = 0.0
        outage_excluded = 0.0
    else:
        data_start_ms = max(window.window_start_ms or 0.0, float(window.mids["ts_ms"].min()))
        wall_seconds = max(((window.window_end_ms or 0.0) - data_start_ms) / 1000.0, 1e-6)
        outage_excluded = float((window.meta or {}).get("outage_seconds", 0.0) or 0.0)
        covered_seconds = max(wall_seconds - outage_excluded, 1e-6)

    lambda_plus_raw = n_buy_mos / covered_seconds if covered_seconds > 0 else float("nan")
    lambda_minus_raw = n_sell_mos / covered_seconds if covered_seconds > 0 else float("nan")
    kappa_plus_raw = buy_fit["kappa"]
    kappa_minus_raw = sell_fit["kappa"]

    sigma2 = realized_sigma2_per_sec(window.mids)

    log_section(f"KAPPA/LAMBDA ESTIMATES - {crypto}")
    print(
        f"Window: {window.window_start_iso()} -> {window.window_end_iso()} "
        f"(covered {covered_seconds:.0f}s, {outage_excluded:.0f}s of outage excluded)"
    )
    if (
        support_lower_plus != DEFAULT_SUPPORT_QUANTILE_LOWER
        or support_lower_minus != DEFAULT_SUPPORT_QUANTILE_LOWER
        or support_upper_plus != DEFAULT_SUPPORT_QUANTILE_UPPER
        or support_upper_minus != DEFAULT_SUPPORT_QUANTILE_UPPER
    ):
        # A non-default support silently changes what kappa means; never let one
        # appear in the log without saying so.
        def _depth(value) -> str:
            return "none" if value is None or not np.isfinite(value) else f"{float(value):.6g}"

        print(
            f"  fit support (quantiles): + [{support_lower_plus:g}, {support_upper_plus:g}] "
            f"-> depths [{_depth(buy_fit.get('support_depth_lower'))}, "
            f"{_depth(buy_fit.get('support_depth_upper'))}]; "
            f"- [{support_lower_minus:g}, {support_upper_minus:g}] "
            f"-> depths [{_depth(sell_fit.get('support_depth_lower'))}, "
            f"{_depth(sell_fit.get('support_depth_upper'))}]"
        )
    for label, fit, lam_raw, n_mos in (
        ("kappa+ (ask side, buy MOs)", buy_fit, lambda_plus_raw, n_buy_mos),
        ("kappa- (bid side, sell MOs)", sell_fit, lambda_minus_raw, n_sell_mos),
    ):
        if np.isfinite(fit["kappa"]):
            print(
                f"  {label}: kappa={fit['kappa']:.4f}, lambda={lam_raw:.4f} MO/sec (n={n_mos}), "
                f"R^2={fit['r_squared']:.3f}, fit_points={fit['n_points']}, depth_p95={fit['depth_p95']:.4f}"
            )
        else:
            print(f"  {label}: unavailable (insufficient data)")
    if sigma2 is not None:
        print(f"  sigma2_per_sec: {sigma2:.6f} USDC^2/s")

    generated_at = utc_now_iso()

    metadata = {
        "window_start": window.window_start_iso(),
        "window_end": window.window_end_iso(),
        "generated_at": generated_at,
        "n_quotes": int(len(window.mids)),
        "n_trades": int(len(window.trades)),
        "n_market_orders_plus": n_buy_mos,
        "n_market_orders_minus": n_sell_mos,
        "lambda_covered_seconds": float(covered_seconds),
        "lambda_outage_seconds_excluded": float(outage_excluded),
        "r2_plus": finite_or_none(buy_fit.get("r_squared")),
        "r2_minus": finite_or_none(sell_fit.get("r_squared")),
        "n_points_plus": int(buy_fit.get("n_points", 0) or 0),
        "n_points_minus": int(sell_fit.get("n_points", 0) or 0),
        "depth_p95_plus": finite_or_none(buy_fit.get("depth_p95")),
        "depth_p95_minus": finite_or_none(sell_fit.get("depth_p95")),
        "depth_max_fitted_plus": finite_or_none(buy_fit.get("depth_max_fitted")),
        "depth_max_fitted_minus": finite_or_none(sell_fit.get("depth_max_fitted")),
        # Which depth range each side's fit actually used. Two kappa values are
        # only comparable across cycles if they were fitted over the same
        # support, so the support travels with the number.
        "depth_min_fitted_plus": finite_or_none(buy_fit.get("depth_min_fitted")),
        "depth_min_fitted_minus": finite_or_none(sell_fit.get("depth_min_fitted")),
        "support_quantile_lower_plus": float(support_lower_plus),
        "support_quantile_lower_minus": float(support_lower_minus),
        "support_quantile_upper_plus": float(support_upper_plus),
        "support_quantile_upper_minus": float(support_upper_minus),
        "support_depth_lower_plus": finite_or_none(buy_fit.get("support_depth_lower")),
        "support_depth_lower_minus": finite_or_none(sell_fit.get("support_depth_lower")),
        "support_depth_upper_plus": finite_or_none(buy_fit.get("support_depth_upper")),
        "support_depth_upper_minus": finite_or_none(sell_fit.get("support_depth_upper")),
        "sigma2_per_sec": finite_or_none(sigma2) if sigma2 is not None else None,
        "mid_source": window.mid_source,
        "ts_source": window.ts_source,
        "negative_depth_truncated_plus": int(depth_info["negative_depth_truncated_buy"]),
        "negative_depth_truncated_minus": int(depth_info["negative_depth_truncated_sell"]),
        "skipped_no_pre_mid": int(depth_info["skipped_no_pre_mid"]),
        "lambda0_intercept_plus": finite_or_none(
            lambda_plus_raw * buy_fit["survival_intercept"]
            if np.isfinite(buy_fit["survival_intercept"]) and np.isfinite(lambda_plus_raw)
            else None
        ),
        "lambda0_intercept_minus": finite_or_none(
            lambda_minus_raw * sell_fit["survival_intercept"]
            if np.isfinite(sell_fit["survival_intercept"]) and np.isfinite(lambda_minus_raw)
            else None
        ),
    }

    if (
        finite_or_none(kappa_plus_raw) is None
        or finite_or_none(kappa_minus_raw) is None
        or finite_or_none(lambda_plus_raw) is None
        or finite_or_none(lambda_minus_raw) is None
        or n_buy_mos <= 0
        or n_sell_mos <= 0
        or metadata["n_points_plus"] < MIN_KAPPA_FIT_POINTS
        or metadata["n_points_minus"] < MIN_KAPPA_FIT_POINTS
    ):
        metadata["status"] = "insufficient_data"
    elif (
        finite_or_none(buy_fit.get("r_squared")) is None
        or finite_or_none(sell_fit.get("r_squared")) is None
        or float(buy_fit["r_squared"]) < MIN_KAPPA_R2
        or float(sell_fit["r_squared"]) < MIN_KAPPA_R2
    ):
        metadata["status"] = "poor_fit"
    else:
        metadata["status"] = "ok"

    raw_values = {
        "kappa+_raw": finite_or_none(kappa_plus_raw),
        "kappa-_raw": finite_or_none(kappa_minus_raw),
        "lambda+_raw": finite_or_none(lambda_plus_raw),
        "lambda-_raw": finite_or_none(lambda_minus_raw),
    }

    kappa_plus_out, kappa_minus_out = kappa_plus_raw, kappa_minus_raw
    lambda_plus_out, lambda_minus_out = lambda_plus_raw, lambda_minus_raw

    if emitting:
        kappa_entry, lambda_entry = build_kappa_lambda_entries(
            kappa_plus_out, kappa_minus_out, lambda_plus_out, lambda_minus_out,
            metadata=metadata, raw_values=raw_values,
        )
        payload = {
            "kind": EMIT_PAYLOAD_KIND,
            "schema_version": PARAM_SCHEMA_VERSION,
            "crypto": crypto,
            "generated_at": generated_at,
            "window": build_emit_window_block(window, minutes),
            "calibration": {
                "kappa_support_quantile_lower_plus": float(support_lower_plus),
                "kappa_support_quantile_lower_minus": float(support_lower_minus),
                "kappa_support_quantile_upper_plus": float(support_upper_plus),
                "kappa_support_quantile_upper_minus": float(support_upper_minus),
            },
            "kappa": kappa_entry,
            "lambda": lambda_entry,
        }
        if emit_sink is not None:
            emit_sink.update(payload)
        if emit_params_json is not None:
            written = write_emit_payload(emit_params_json, payload)
            print(f"[emit] kappa/lambda params -> {written}")
        return

    save_kappa_lambda_to_json(
        kappa_plus_out,
        kappa_minus_out,
        lambda_plus_out,
        lambda_minus_out,
        crypto,
        kappa_file=kappa_file,
        lambda_file=lambda_file,
        metadata=metadata,
        raw_values=raw_values,
    )


# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joint kappa/lambda estimation from market data (survival fit + MO arrival rates)')
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'CASHCAT'),
                        help='Cryptocurrency symbol (e.g., ETH) or ALL to process every symbol in HL_data')
    parser.add_argument('--minutes', '-m', type=int, default=30,
                        help='Number of minutes from most recent data to analyze')
    parser.add_argument('--support-quantile', type=float, default=None,
                        help='Shorthand: upper quantile of the kappa fit support on BOTH sides '
                             f'(default {DEFAULT_SUPPORT_QUANTILE_UPPER})')
    parser.add_argument('--support-quantile-plus', type=float, default=None,
                        help='Upper quantile of the fit support for kappa+ (buy MO depths). '
                             'Overrides --support-quantile.')
    parser.add_argument('--support-quantile-minus', type=float, default=None,
                        help='Upper quantile of the fit support for kappa- (sell MO depths). '
                             'Overrides --support-quantile.')
    parser.add_argument('--support-quantile-lower', type=float, default=None,
                        help='Shorthand: lower quantile of the kappa fit support on BOTH sides '
                             f'(default {DEFAULT_SUPPORT_QUANTILE_LOWER} = no lower bound). Raise it '
                             'to fit kappa only over the deeper market orders that actually pay.')
    parser.add_argument('--support-quantile-lower-plus', type=float, default=None,
                        help='Lower quantile of the fit support for kappa+ (buy MO depths). '
                             'Overrides --support-quantile-lower.')
    parser.add_argument('--support-quantile-lower-minus', type=float, default=None,
                        help='Lower quantile of the fit support for kappa- (sell MO depths). '
                             'Overrides --support-quantile-lower.')
    parser.add_argument('--window-start', type=str, default=None,
                        help='ISO-8601 (or epoch) start of the window; selects a historical slice '
                             'instead of the trailing --minutes window')
    parser.add_argument('--window-end', type=str, default=None,
                        help='ISO-8601 (or epoch) end of the window (clamped to the data that exists)')
    parser.add_argument('--emit-params-json', type=str, default=None,
                        help='Write the computed kappa/lambda set to this path and write NOTHING '
                             'else: kappa.json and lambda.json are left untouched. Use for '
                             'calibration sweeps while the live estimator is running. With '
                             '--crypto ALL the symbol is inserted into the filename.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Root of the collector output (default: scripts/HL_data). A sweep can '
                             'point this at a frozen copy so it never contends with the collector.')
    parser.add_argument('--verbosity', '-v', type=int, choices=[0, 1], default=0,
                        help='Verbosity: 0=minimal (default), 1=verbose')

    args = parser.parse_args()

    _set_verbosity(args.verbosity)

    if isinstance(args.crypto, str) and args.crypto.strip().upper() == 'ALL':
        symbols = list_available_cryptos(args.data_dir)
        if not symbols:
            print("No crypto data found in HL_data (need prices/orderbooks and trades parquet).")
            raise SystemExit(1)
    else:
        symbols = [args.crypto.strip().upper()]

    # Per-side flag wins over the both-sides shorthand, which wins over the
    # default. Same precedence the epsilon horizons use.
    support_upper_plus = args.support_quantile_plus if args.support_quantile_plus is not None else args.support_quantile
    support_upper_minus = args.support_quantile_minus if args.support_quantile_minus is not None else args.support_quantile
    support_lower_plus = (
        args.support_quantile_lower_plus if args.support_quantile_lower_plus is not None else args.support_quantile_lower
    )
    support_lower_minus = (
        args.support_quantile_lower_minus if args.support_quantile_lower_minus is not None else args.support_quantile_lower
    )

    if (args.window_start or args.window_end) and not args.emit_params_json:
        # Not forbidden -- someone may legitimately want to backfill a snapshot --
        # but it is worth one loud line, because writing a historical slice into
        # kappa.json hands the live strategy parameters from another hour.
        print("WARNING: --window-start/--window-end without --emit-params-json will OVERWRITE "
              "kappa.json/lambda.json with parameters fitted on a historical slice.")

    for sym in symbols:
        emit_path = args.emit_params_json
        if emit_path and len(symbols) > 1:
            p = Path(emit_path)
            emit_path = str(p.with_name(f"{p.stem}.{sym}{p.suffix}"))
        try:
            run_kappa_for_crypto(
                sym,
                minutes=args.minutes,
                support_quantile_plus=support_upper_plus,
                support_quantile_minus=support_upper_minus,
                support_quantile_lower_plus=support_lower_plus,
                support_quantile_lower_minus=support_lower_minus,
                window_start=args.window_start,
                window_end=args.window_end,
                emit_params_json=emit_path,
                data_dir=args.data_dir,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing {sym}: {e}")
