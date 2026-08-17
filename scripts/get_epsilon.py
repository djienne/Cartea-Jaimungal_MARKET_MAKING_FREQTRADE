#!/usr/bin/env python3
"""Event-level epsilon (adverse-selection impact) estimation (schema v3).

Methodology:
- Impacts are measured per MARKET ORDER (prints sharing side + exchange
  timestamp are one MO), not per print, against the mid series built from the
  dense BBO ``prices/`` stream on exchange timestamps.
- pre-mid = last mid strictly BEFORE the MO; post-mid = last mid at or before
  MO time + horizon. MOs whose horizon extends past the data window are
  dropped (no truncation bias).
- The primary horizon is 200 ms (--post-horizon-ms / EPSILON_POST_HORIZON_MS).
  The book's epsilon (eq. 10.22) is the jump AT the MO arrival, so a long
  markout measures a different quantity: it absorbs every other MO in the
  window, and the HJB already carries that net-flow drift separately in the
  q(lambda+ eps+ - lambda- eps-) term of eq. 10.26. 1 s and 5 s means are
  recorded as diagnostics (epsilon_1s_±, epsilon_5s_±).
- The shipped statistic is the plain MEAN of outlier-cleaned impacts, because
  the model wants E[eps]. A 10% trimmed mean is still reported as a diagnostic
  but is not the model's moment -- on a right-skewed jump distribution it
  understates the mean, i.e. understates adverse selection.
- The C-J model defines epsilon >= 0; slightly negative means
  (mean-reversion noise) floor at 0.
- Primary epsilon± values are EMA-smoothed across estimator cycles
  (time-aware, tau via --ema-tau / PARAM_EMA_TAU_SECONDS, default 300 s);
  unsmoothed values live in epsilon±_raw.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from estimator_common import (
    MIN_EPSILON_EVENTS,
    aggregate_market_orders,
    attach_pre_mid,
    ema_update,
    load_market_window,
    resolve_ema_tau,
)
from param_utils import (
    PARAM_SCHEMA_VERSION,
    atomic_write_json,
    finite_or_none,
    load_json_object,
    utc_now_iso,
)

# The book (eq. 10.22) defines epsilon as the midprice jump AT the market-order
# arrival: dS = sigma dW + eps+ dM+ - eps- dM-. A long markout is a different
# estimand -- it absorbs the impact of every OTHER MO arriving inside the window,
# so each side picks up ~horizon * (lambda+ eps+ - lambda- eps-) of net-flow
# drift. The HJB then adds that drift back itself via the q(lambda+ eps+ -
# lambda- eps-) term of eq. 10.26, so a long horizon double-counts the trend.
# 200 ms is short enough to be the arrival jump and long enough to clear the
# mechanical BBO reaction. The former 5 s default is kept as a diagnostic.
DEFAULT_POST_HORIZON_MS = 200
DIAGNOSTIC_HORIZONS_MS = (1000, 5000)
POST_HORIZON_ENV_VAR = "EPSILON_POST_HORIZON_MS"

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

def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def resolve_post_horizon_ms(cli_value: int | None = None) -> int:
    if cli_value is not None:
        try:
            return max(1, int(cli_value))
        except (TypeError, ValueError):
            pass
    env_value = os.getenv(POST_HORIZON_ENV_VAR)
    if env_value is not None:
        try:
            return max(1, int(float(env_value)))
        except (TypeError, ValueError):
            pass
    return DEFAULT_POST_HORIZON_MS


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


# ============================================================================
# EPSILON CALCULATION
# ============================================================================

def compute_mo_impacts(mos_with_pre_mid: pd.DataFrame, mids: pd.DataFrame,
                       horizon_ms: int, window_end_ms: float) -> dict:
    """Per-MO mid impacts at one horizon (vectorized merge_asof).

    Returns the same shape the trimmed-mean estimator consumes:
    {'buy_impacts': [{'impact', 'size'}...], 'sell_impacts': [...],
     'trades_analyzed': int, 'trades_skipped': int}.
    """
    results = {
        'buy_impacts': [],
        'sell_impacts': [],
        'trades_analyzed': 0,
        'trades_skipped': 0,
    }
    if mos_with_pre_mid.empty or mids.empty:
        results['trades_skipped'] = int(len(mos_with_pre_mid))
        return results

    frame = mos_with_pre_mid.copy()
    n_total = len(frame)
    frame = frame.dropna(subset=['pre_mid'])
    # Require the data window to cover the full horizon (truncation bias).
    frame = frame[frame['ts_ms'] + float(horizon_ms) <= float(window_end_ms)]
    if frame.empty:
        results['trades_skipped'] = n_total
        return results

    frame = frame.assign(target_ms=frame['ts_ms'] + float(horizon_ms)).sort_values('target_ms')
    post = pd.merge_asof(
        frame,
        mids[['ts_ms', 'mid']].sort_values('ts_ms').rename(columns={'ts_ms': 'mid_ts_ms', 'mid': 'post_mid'}),
        left_on='target_ms',
        right_on='mid_ts_ms',
        direction='backward',
        allow_exact_matches=True,
    )
    post = post.dropna(subset=['post_mid'])

    results['trades_analyzed'] = int(len(post))
    results['trades_skipped'] = int(n_total - len(post))

    buys = post[post['side'] == 'buy']
    sells = post[post['side'] == 'sell']
    results['buy_impacts'] = [
        {'impact': float(impact), 'size': float(size)}
        for impact, size in zip(buys['post_mid'] - buys['pre_mid'], buys['size'])
    ]
    results['sell_impacts'] = [
        {'impact': float(impact), 'size': float(size)}
        for impact, size in zip(sells['pre_mid'] - sells['post_mid'], sells['size'])
    ]
    return results


def estimate_epsilon_parameters(results):
    """
    Estimate epsilon+ and epsilon- from impact measurements (event-level).
    Uses trimmed mean (10%) and median for robustness.
    """

    estimates = {}

    for side in ['buy', 'sell']:
        impacts_data = results[f'{side}_impacts']

        if len(impacts_data) == 0:
            estimates[f'epsilon_{side}'] = {
                'mean': 0, 'median': 0, 'trimmed_mean': 0, 'std': 0, 'n_trades': 0
            }
            continue

        impacts = np.array([d['impact'] for d in impacts_data])

        # Remove outliers (impacts > 3 std)
        mean_imp = np.mean(impacts)
        std_imp = np.std(impacts)
        mask = np.abs(impacts - mean_imp) < 3 * std_imp if std_imp > 0 else np.ones_like(impacts, dtype=bool)
        impacts_clean = impacts[mask]

        # Trim 10% tails
        if len(impacts_clean) > 2:
            sorted_impacts = np.sort(impacts_clean)
            lower = int(0.1 * len(sorted_impacts))
            upper = int(0.9 * len(sorted_impacts))
            trimmed = sorted_impacts[lower:upper] if upper > lower else sorted_impacts
        else:
            trimmed = impacts_clean

        estimates[f'epsilon_{side}'] = {
            'mean': float(np.mean(impacts_clean)) if len(impacts_clean) > 0 else 0.0,
            'median': float(np.median(impacts_clean)) if len(impacts_clean) > 0 else 0.0,
            'trimmed_mean': float(np.mean(trimmed)) if len(trimmed) > 0 else 0.0,
            'std': float(np.std(impacts_clean)) if len(impacts_clean) > 0 else 0.0,
            'n_trades': int(len(impacts_clean))
        }

    return estimates


def _floored_trimmed_means(results) -> tuple[float | None, float | None, dict]:
    """Mean impact per side, floored at 0 (C-J epsilon >= 0).

    The book asks for epsbar = E[eps], so the shipped figure is the plain mean
    of the outlier-cleaned impacts. The 10% two-sided trim that used to be
    shipped is still computed as a diagnostic, but it is not the model's moment:
    on a right-skewed jump distribution trimming systematically understates the
    mean, i.e. understates the adverse selection the quote is meant to recover.
    The 3-sigma clip is kept -- that removes bad ticks rather than reshaping the
    distribution.
    """
    estimates = estimate_epsilon_parameters(results)
    eps_plus = estimates['epsilon_buy']['mean']
    eps_minus = estimates['epsilon_sell']['mean']
    if eps_plus is not None and np.isfinite(eps_plus):
        eps_plus = max(0.0, float(eps_plus))
    if eps_minus is not None and np.isfinite(eps_minus):
        eps_minus = max(0.0, float(eps_minus))
    return eps_plus, eps_minus, estimates


# ============================================================================
# RUN ANALYSIS
# ============================================================================

def load_kappa_from_json(crypto: str, filename: str = 'kappa.json'):
    """Load kappa+ and kappa- values for a crypto from kappa.json."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"kappa file not found: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
    if crypto not in data:
        raise KeyError(f"crypto '{crypto}' not found in {filename}")
    entry = data[crypto]
    # Support keys 'kappa+'/'kappa-' and 'kappa_plus'/'kappa_minus'
    kappa_plus = entry.get('kappa+') if isinstance(entry, dict) else None
    if kappa_plus is None:
        kappa_plus = entry.get('kappa_plus')
    kappa_minus = entry.get('kappa-') if isinstance(entry, dict) else None
    if kappa_minus is None:
        kappa_minus = entry.get('kappa_minus')
    if kappa_plus is None or kappa_minus is None:
        raise ValueError(f"kappa values missing for '{crypto}' in {filename}")
    return float(kappa_plus), float(kappa_minus)


def save_epsilon_to_json(eps_plus: float, eps_minus: float, crypto: str, filename: str = "epsilon.json",
                         metadata: dict | None = None, raw_values: dict | None = None):
    """Save epsilon estimates to JSON. Primary keys are EMA-smoothed; *_raw
    holds this window's unsmoothed means -- 3-sigma-clipped and floored at zero,
    NOT trimmed. The trimmed mean is computed alongside but deliberately not
    shipped: trimming a right-skewed jump distribution understates adverse
    selection. *_raw defaults to the primaries when not supplied, so every v3
    snapshot carries the full key set."""
    metadata = dict(metadata or {})
    status = str(metadata.pop("status", "ok"))
    metadata.setdefault("generated_at", utc_now_iso())
    raw = dict(raw_values or {})
    epsilon_plus_val = finite_or_none(eps_plus)
    epsilon_minus_val = finite_or_none(eps_minus)
    data = load_json_object(filename)

    data[crypto] = {
        "schema_version": PARAM_SCHEMA_VERSION,
        "status": status,
        "epsilon+": epsilon_plus_val,
        "epsilon-": epsilon_minus_val,
        "epsilon+_raw": finite_or_none(raw.get("epsilon+_raw", eps_plus)),
        "epsilon-_raw": finite_or_none(raw.get("epsilon-_raw", eps_minus)),
        "unit": "USDC",
        "estimator": "mean_at_arrival",
        **metadata,
    }

    atomic_write_json(filename, data)

    print(f"[save] epsilon -> {filename}")
    print(f"[save] {crypto}: epsilon+={epsilon_plus_val}, epsilon-={epsilon_minus_val}")


def load_mid_price_from_json(crypto: str, filename: str = 'mid_price.json') -> float:
    """Load mid-price for a crypto from mid_price.json."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"mid price file not found: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
    if crypto not in data:
        raise KeyError(f"crypto '{crypto}' not found in {filename}")
    return float(data[crypto])


def run_epsilon_for_crypto(crypto: str, minutes: int = 30, post_horizon_ms: int | None = None,
                           ema_tau: float | None = None, epsilon_file: str = "epsilon.json",
                           data_dir=None, window=None):
    """Run the full epsilon analysis and persistence for a single crypto symbol.

    ``window`` lets a caller that already loaded the market window (estimate_all.py)
    hand it over instead of paying for a second full parquet scan of the same data.
    """
    horizon_ms = resolve_post_horizon_ms(post_horizon_ms)
    log_section(f"EPSILON FROM MARKET DATA - {crypto} (last {minutes} min, horizon {horizon_ms} ms)")
    if window is None:
        try:
            window = load_market_window(crypto, minutes, data_dir=data_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

    vprint(f"Mid source: {window.mid_source} ({window.ts_source} time)")
    vprint(f"Loaded {len(window.mids):,} mid updates and {len(window.trades):,} trade prints")

    mos = attach_pre_mid(aggregate_market_orders(window.trades), window.mids)
    window_end_ms = window.window_end_ms or 0.0

    log_section(f"CALCULATING PRICE IMPACTS FOR {crypto}")

    impact_results = compute_mo_impacts(mos, window.mids, horizon_ms, window_end_ms)
    eps_plus_raw, eps_minus_raw, final_estimates = _floored_trimmed_means(impact_results)

    diagnostics: dict[str, float | None] = {}
    for diag_horizon in DIAGNOSTIC_HORIZONS_MS:
        if diag_horizon == horizon_ms:
            diag_plus, diag_minus = eps_plus_raw, eps_minus_raw
        else:
            diag_results = compute_mo_impacts(mos, window.mids, diag_horizon, window_end_ms)
            diag_plus, diag_minus, _ = _floored_trimmed_means(diag_results)
        label = (
            f"{diag_horizon // 1000}s" if diag_horizon % 1000 == 0 else f"{diag_horizon}ms"
        )
        diagnostics[f"epsilon_{label}_plus"] = finite_or_none(diag_plus)
        diagnostics[f"epsilon_{label}_minus"] = finite_or_none(diag_minus)

    log_section(f"EPSILON ESTIMATES (event-level, {horizon_ms} ms horizon) - {crypto}")
    print(f"{crypto}: epsilon+={eps_plus_raw:.8f}, epsilon-={eps_minus_raw:.8f} (raw)")

    n_buy_events = int(final_estimates["epsilon_buy"].get("n_trades", 0) or 0)
    n_sell_events = int(final_estimates["epsilon_sell"].get("n_trades", 0) or 0)

    # Check toxicity using kappa from kappa.json (per-crypto)
    toxicity_plus = None
    toxicity_minus = None
    try:
        kappa_plus, kappa_minus = load_kappa_from_json(crypto)
        log_section(f"TOXICITY CHECK - {crypto}")
        toxicity_plus = kappa_plus * eps_plus_raw
        toxicity_minus = kappa_minus * eps_minus_raw
        print(f"  kappa+ x epsilon+ = {toxicity_plus:.4f}")
        print(f"  kappa- x epsilon- = {toxicity_minus:.4f}")

        kappa_epsilon_max = max(toxicity_plus, toxicity_minus)
        if kappa_epsilon_max >= 2:
            print("  WARNING: Market appears very toxic (kappa x epsilon >= 2)")
        elif kappa_epsilon_max >= 1:
            print("  CAUTION: Market is competitive (1 <= kappa x epsilon < 2)")
        else:
            print("  Market toxicity appears manageable (kappa x epsilon < 1)")
    except Exception as e:
        print(f"Toxicity check skipped: {e}")

    # Model-level delta diagnostic (1/kappa + epsilon; the strategy adds the
    # spread multiplier, fee cushion and bps clamps on top of this).
    try:
        mid_price = load_mid_price_from_json(crypto)
        eps_plus_val = 0.0 if (eps_plus_raw is None or np.isnan(eps_plus_raw)) else float(eps_plus_raw)
        eps_minus_val = 0.0 if (eps_minus_raw is None or np.isnan(eps_minus_raw)) else float(eps_minus_raw)

        try:
            kappa_plus
            kappa_minus
        except NameError:
            kappa_plus, kappa_minus = load_kappa_from_json(crypto)

        delta_plus_price = (1.0 / float(kappa_plus)) + eps_plus_val
        delta_minus_price = (1.0 / float(kappa_minus)) + eps_minus_val

        delta_plus_pct = (delta_plus_price / mid_price) * 100.0 if mid_price != 0 else float('nan')
        delta_minus_pct = (delta_minus_price / mid_price) * 100.0 if mid_price != 0 else float('nan')

        log_section(f"DELTA ESTIMATES (model term only) - {crypto}")
        print(f"  mid price [{crypto}]: {mid_price:.6f}")
        print(f"  delta+ = 1/kappa+ + epsilon+ = {delta_plus_price:.8f}  ({delta_plus_pct:.6f}% of mid)")
        print(f"  delta- = 1/kappa- + epsilon- = {delta_minus_price:.8f}  ({delta_minus_pct:.6f}% of mid)")
    except Exception as e:
        print(f"Delta calculation skipped: {e}")

    if (
        eps_plus_raw is None
        or eps_minus_raw is None
        or not np.isfinite(float(eps_plus_raw))
        or not np.isfinite(float(eps_minus_raw))
        or n_buy_events < MIN_EPSILON_EVENTS
        or n_sell_events < MIN_EPSILON_EVENTS
    ):
        status = "insufficient_data"
    else:
        status = "ok"

    generated_at_dt = datetime.now(timezone.utc)
    generated_at = generated_at_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    tau = resolve_ema_tau(ema_tau)

    raw_values = {
        "epsilon+_raw": finite_or_none(eps_plus_raw),
        "epsilon-_raw": finite_or_none(eps_minus_raw),
    }
    if status == "ok":
        prev_entry = load_json_object(epsilon_file).get(crypto)
        prev_entry = prev_entry if isinstance(prev_entry, dict) else None
        eps_plus_out, meta_plus = ema_update(eps_plus_raw, prev_entry, "epsilon+", tau, now=generated_at_dt)
        eps_minus_out, meta_minus = ema_update(eps_minus_raw, prev_entry, "epsilon-", tau, now=generated_at_dt)
        ema_seeded = bool(meta_plus["ema_seeded"]) and bool(meta_minus["ema_seeded"])
    else:
        eps_plus_out, eps_minus_out = eps_plus_raw, eps_minus_raw
        ema_seeded = True

    save_epsilon_to_json(
        eps_plus_out,
        eps_minus_out,
        crypto,
        filename=epsilon_file,
        metadata={
            "status": status,
            "window_ms": int(horizon_ms),
            "window_start": window.window_start_iso(),
            "window_end": window.window_end_iso(),
            "generated_at": generated_at,
            "n_quotes": int(len(window.mids)),
            "n_trades": int(len(window.trades)),
            "n_buy_events": n_buy_events,
            "n_sell_events": n_sell_events,
            "trades_analyzed": int(impact_results.get("trades_analyzed", 0) or 0),
            "trades_skipped": int(impact_results.get("trades_skipped", 0) or 0),
            "toxicity_plus": finite_or_none(toxicity_plus),
            "toxicity_minus": finite_or_none(toxicity_minus),
            "mid_source": window.mid_source,
            "ts_source": window.ts_source,
            "ema_tau_seconds": float(tau),
            "ema_seeded": ema_seeded,
            **diagnostics,
        },
        raw_values=raw_values,
    )

    vprint("\nNotes:")
    vprint("- epsilon is measured per market order against the BBO mid stream")
    vprint("- the primary 200ms horizon targets the jump AT arrival (eq 10.22), not permanent impact;")
    vprint("  the 1s/5s markouts are diagnostics only -- see the rationale at the top of this file")
    vprint("- monitor epsilon over time as market conditions change")

    vprint("\n" + "-"*60 + "\n")

    return eps_plus_out, eps_minus_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate epsilon (permanent price impact) from market data')
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'ETH'),
                        help='Cryptocurrency symbol (e.g., ETH) or ALL for all available in HL_data')
    parser.add_argument('--minutes', '-m', type=int, default=30,
                        help='Number of minutes from most recent data to analyze')
    parser.add_argument('--post-horizon-ms', type=int, default=None,
                        help='Primary post-trade horizon in ms for the permanent-impact measurement '
                             f'(default: {POST_HORIZON_ENV_VAR} env or {DEFAULT_POST_HORIZON_MS})')
    parser.add_argument('--ema-tau', type=float, default=None,
                        help='EMA time constant in seconds for smoothing primary values '
                             '(default: PARAM_EMA_TAU_SECONDS env or 300; 0 disables smoothing)')
    parser.add_argument('--verbosity', '-v', type=int, choices=[0, 1], default=0,
                        help='Verbosity: 0=minimal (default), 1=verbose')

    args = parser.parse_args()

    _set_verbosity(args.verbosity)

    # Determine cryptos to process
    if isinstance(args.crypto, str) and args.crypto.strip().upper() == 'ALL':
        symbols = list_available_cryptos('HL_data')
        if not symbols:
            print("No crypto data found in HL_data (need <SYMBOL>/prices or orderbooks plus trades parquet)")
            raise SystemExit(1)
    else:
        symbols = [args.crypto.strip().upper()]

    for sym in symbols:
        try:
            run_epsilon_for_crypto(sym, minutes=args.minutes, post_horizon_ms=args.post_horizon_ms, ema_tau=args.ema_tau)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing {sym}: {e}")
