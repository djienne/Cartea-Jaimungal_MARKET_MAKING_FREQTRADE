#!/usr/bin/env python3
"""Joint kappa/lambda estimation from Hyperliquid market data (schema v3).

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
- lambda± is the raw per-side MO arrival rate (count / covered window
  seconds). The old binned-density regression intercept equals
  lambda*kappa*binwidth (bin-width dependent) and is kept only as the
  lambda0_intercept_± diagnostic.
- Primary kappa±/lambda± values are EMA-smoothed across estimator cycles
  (time-aware, tau via --ema-tau / PARAM_EMA_TAU_SECONDS, default 300 s);
  unsmoothed values are stored in the *_raw keys.
- Realized mid variance sigma2_per_sec (USDC^2/s) is published for the
  strategy's volatility-aware inventory penalty.
"""

import argparse
import os
import warnings
from datetime import datetime, timezone

import numpy as np

from estimator_common import (
    MIN_KAPPA_FIT_POINTS,
    MIN_KAPPA_R2,
    aggregate_market_orders,
    attach_pre_mid,
    fit_kappa_survival,
    load_market_window,
    mo_depths,
    realized_sigma2_per_sec,
    resolve_ema_tau,
    ema_update,
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


def save_kappa_lambda_to_json(kappa_plus, kappa_minus, lambda_plus, lambda_minus,
                              crypto: str, kappa_file: str = "kappa.json", lambda_file: str = "lambda.json",
                              metadata: dict | None = None, raw_values: dict | None = None):
    """Persist kappa and per-side MO arrival rates (per second) to JSON files.

    Primary keys hold the EMA-smoothed values; *_raw keys hold this window's
    unsmoothed estimates (defaulting to the primaries when not supplied so
    every v3 snapshot carries the full key set).
    """
    metadata = dict(metadata or {})
    status = str(metadata.pop("status", "ok"))
    metadata.setdefault("generated_at", utc_now_iso())
    raw = dict(raw_values or {})
    kappa_data = load_json_object(kappa_file)
    lambda_data = load_json_object(lambda_file)

    raw_block = {
        "kappa+_raw": finite_or_none(raw.get("kappa+_raw", kappa_plus)),
        "kappa-_raw": finite_or_none(raw.get("kappa-_raw", kappa_minus)),
        "lambda+_raw": finite_or_none(raw.get("lambda+_raw", lambda_plus)),
        "lambda-_raw": finite_or_none(raw.get("lambda-_raw", lambda_minus)),
    }

    kappa_data[crypto] = {
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
    lambda_data[crypto] = {
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

    atomic_write_json(kappa_file, kappa_data)
    atomic_write_json(lambda_file, lambda_data)

    print(f"[save] kappa -> {kappa_file}")
    print(f"[save] lambda (MO arrival rate) -> {lambda_file}")


def run_kappa_for_crypto(crypto: str, minutes: int = 30, ema_tau: float | None = None,
                         kappa_file: str = "kappa.json", lambda_file: str = "lambda.json",
                         data_dir=None, window=None):
    """Run the full kappa/lambda estimation flow for a single crypto symbol.

    ``window`` lets a caller that already loaded the market window (estimate_all.py)
    hand it over instead of paying for a second full parquet scan of the same data.
    """
    log_section(f"KAPPA/LAMBDA FROM MARKET DATA - {crypto} (last {minutes} min)")
    if window is None:
        try:
            window = load_market_window(crypto, minutes, data_dir=data_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

    vprint(f"Mid source: {window.mid_source} ({window.ts_source} time)")
    vprint(f"Loaded {len(window.mids):,} mid updates and {len(window.trades):,} trade prints")

    mos = attach_pre_mid(aggregate_market_orders(window.trades), window.mids)
    depth_info = mo_depths(mos)

    n_buy_mos = int((mos["side"] == "buy").sum()) if not mos.empty else 0
    n_sell_mos = int((mos["side"] == "sell").sum()) if not mos.empty else 0

    buy_fit = fit_kappa_survival(depth_info["buy_depths"])
    sell_fit = fit_kappa_survival(depth_info["sell_depths"])

    # Arrival rate over the span both streams actually cover.
    if window.mids.empty:
        covered_seconds = 0.0
    else:
        data_start_ms = max(window.window_start_ms or 0.0, float(window.mids["ts_ms"].min()))
        covered_seconds = max(((window.window_end_ms or 0.0) - data_start_ms) / 1000.0, 1e-6)

    lambda_plus_raw = n_buy_mos / covered_seconds if covered_seconds > 0 else float("nan")
    lambda_minus_raw = n_sell_mos / covered_seconds if covered_seconds > 0 else float("nan")
    kappa_plus_raw = buy_fit["kappa"]
    kappa_minus_raw = sell_fit["kappa"]

    sigma2 = realized_sigma2_per_sec(window.mids)

    log_section(f"KAPPA/LAMBDA ESTIMATES - {crypto}")
    print(f"Window: {window.window_start_iso()} -> {window.window_end_iso()} (covered {covered_seconds:.0f}s)")
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

    generated_at_dt = datetime.now(timezone.utc)
    generated_at = generated_at_dt.isoformat(timespec="seconds").replace("+00:00", "Z")

    metadata = {
        "window_start": window.window_start_iso(),
        "window_end": window.window_end_iso(),
        "generated_at": generated_at,
        "n_quotes": int(len(window.mids)),
        "n_trades": int(len(window.trades)),
        "n_market_orders_plus": n_buy_mos,
        "n_market_orders_minus": n_sell_mos,
        "r2_plus": finite_or_none(buy_fit.get("r_squared")),
        "r2_minus": finite_or_none(sell_fit.get("r_squared")),
        "n_points_plus": int(buy_fit.get("n_points", 0) or 0),
        "n_points_minus": int(sell_fit.get("n_points", 0) or 0),
        "depth_p95_plus": finite_or_none(buy_fit.get("depth_p95")),
        "depth_p95_minus": finite_or_none(sell_fit.get("depth_p95")),
        "depth_max_fitted_plus": finite_or_none(buy_fit.get("depth_max_fitted")),
        "depth_max_fitted_minus": finite_or_none(sell_fit.get("depth_max_fitted")),
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

    tau = resolve_ema_tau(ema_tau)
    metadata["ema_tau_seconds"] = float(tau)
    if metadata["status"] == "ok":
        prev_entry = load_json_object(kappa_file).get(crypto)
        prev_entry = prev_entry if isinstance(prev_entry, dict) else None
        smoothed = {}
        seeded_flags = []
        for key, raw_value in (
            ("kappa+", kappa_plus_raw),
            ("kappa-", kappa_minus_raw),
            ("lambda+", lambda_plus_raw),
            ("lambda-", lambda_minus_raw),
        ):
            value, ema_meta = ema_update(raw_value, prev_entry, key, tau, now=generated_at_dt)
            smoothed[key] = value
            seeded_flags.append(bool(ema_meta["ema_seeded"]))
        metadata["ema_seeded"] = all(seeded_flags)
        kappa_plus_out, kappa_minus_out = smoothed["kappa+"], smoothed["kappa-"]
        lambda_plus_out, lambda_minus_out = smoothed["lambda+"], smoothed["lambda-"]
    else:
        # Bad windows ship raw (consumers reject on status); the next ok cycle
        # re-seeds rather than blending across the bad snapshot.
        metadata["ema_seeded"] = True
        kappa_plus_out, kappa_minus_out = raw_values["kappa+_raw"], raw_values["kappa-_raw"]
        lambda_plus_out, lambda_minus_out = raw_values["lambda+_raw"], raw_values["lambda-_raw"]

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
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'ETH'),
                        help='Cryptocurrency symbol (e.g., ETH) or ALL to process every symbol in HL_data')
    parser.add_argument('--minutes', '-m', type=int, default=30,
                        help='Number of minutes from most recent data to analyze')
    parser.add_argument('--ema-tau', type=float, default=None,
                        help='EMA time constant in seconds for smoothing primary values '
                             '(default: PARAM_EMA_TAU_SECONDS env or 300; 0 disables smoothing)')
    parser.add_argument('--verbosity', '-v', type=int, choices=[0, 1], default=0,
                        help='Verbosity: 0=minimal (default), 1=verbose')

    args = parser.parse_args()

    _set_verbosity(args.verbosity)

    if isinstance(args.crypto, str) and args.crypto.strip().upper() == 'ALL':
        symbols = list_available_cryptos('HL_data')
        if not symbols:
            print("No crypto data found in HL_data (need prices/orderbooks and trades parquet).")
            raise SystemExit(1)
    else:
        symbols = [args.crypto.strip().upper()]

    for sym in symbols:
        try:
            run_kappa_for_crypto(sym, minutes=args.minutes, ema_tau=args.ema_tau)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing {sym}: {e}")
