import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
import os
import json
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings

# Verbosity control: 0=minimal, 1=verbose (previous default)
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

def load_market_data(crypto='BTC', time_range_minutes=60):
    """Load market data for specified cryptocurrency and time range from Parquet files."""
    
    vprint(f"Loading market data for {crypto} (last {time_range_minutes} minutes)...")
    
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'HL_data')
    
    orderbook_dir = os.path.join(base_dir, crypto, 'orderbooks')
    trades_dir = os.path.join(base_dir, crypto, 'trades')
    
    if not os.path.exists(orderbook_dir):
        raise FileNotFoundError(f"Orderbook directory not found: {orderbook_dir}")
    if not os.path.exists(trades_dir):
        raise FileNotFoundError(f"Trades directory not found: {trades_dir}")
    
    # Helper to load all parquet files in a directory
    def load_parquet_dir(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    # Load data
    vprint("Loading orderbook data...")
    df_orderbook = load_parquet_dir(orderbook_dir)
    if df_orderbook.empty:
        raise ValueError(f"No orderbook data found in {orderbook_dir}")
    # Parquet saves timestamps as float seconds (from data collector), convert to datetime
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='s')
    
    vprint("Loading trade data...")
    df_trades_raw = load_parquet_dir(trades_dir)
    if df_trades_raw.empty:
         # It's possible to have no trades but valid orderbook data, return empty trades df
         vprint("Warning: No trade data found.")
         df_trades_raw = pd.DataFrame(columns=['timestamp', 'side', 'size', 'price'])
    else:
        df_trades_raw['timestamp'] = pd.to_datetime(df_trades_raw['timestamp'], unit='s')
    
    # Find overlapping time window
    most_recent_quotes = df_orderbook['timestamp'].max()
    if not df_trades_raw.empty:
        most_recent_trades = df_trades_raw['timestamp'].max()
        most_recent = min(most_recent_quotes, most_recent_trades)
    else:
        most_recent = most_recent_quotes

    time_window_start = most_recent - timedelta(minutes=time_range_minutes)
    
    vprint(f"Most recent data: {most_recent}")
    vprint(f"Time window: {time_window_start} to {most_recent}")
    
    # Filter to time range
    df_orderbook = df_orderbook[
        (df_orderbook['timestamp'] >= time_window_start) & 
        (df_orderbook['timestamp'] <= most_recent)
    ].sort_values('timestamp').reset_index(drop=True)
    
    if not df_trades_raw.empty:
        df_trades_raw = df_trades_raw[
            (df_trades_raw['timestamp'] >= time_window_start) & 
            (df_trades_raw['timestamp'] <= most_recent)
        ].sort_values('timestamp').reset_index(drop=True)
    
    # Process trades
    if not df_trades_raw.empty:
        df_trades = df_trades_raw[['timestamp', 'side', 'size', 'price']].copy()
    else:
        df_trades = df_trades_raw # Empty DataFrame
    
    # Final overlap filtering if we have both
    if not df_trades.empty:
        start_time = max(df_orderbook['timestamp'].min(), df_trades['timestamp'].min())
        end_time = min(df_orderbook['timestamp'].max(), df_trades['timestamp'].max())
        
        df_quotes = df_orderbook[
            (df_orderbook['timestamp'] >= start_time) & 
            (df_orderbook['timestamp'] <= end_time)
        ].reset_index(drop=True)
        
        df_trades = df_trades[
            (df_trades['timestamp'] >= start_time) & 
            (df_trades['timestamp'] <= end_time)
        ].reset_index(drop=True)
    else:
         df_quotes = df_orderbook

    return df_quotes, df_trades


def list_available_cryptos(data_dir: str = None):
    """Return sorted list of crypto symbols that have data directories."""
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'HL_data')

    if not os.path.isdir(data_dir):
        return []
    try:
        # Check subdirectories in HL_data
        candidates = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        # Filter for those having 'orderbooks' and 'trades' subdirs
        valid = []
        for sym in candidates:
            if os.path.exists(os.path.join(data_dir, sym, 'orderbooks')) and \
               os.path.exists(os.path.join(data_dir, sym, 'trades')):
                valid.append(sym)
        return sorted(valid)
    except Exception:
        return []

def calculate_trade_execution_rates(df_quotes, df_trades, lookback_ms=500):
    """
    Calculate actual trade execution rates at different depth levels.
    
    This improved method:
    1. For each trade, finds the orderbook state just before
    2. Calculates the depth at which the trade occurred
    3. Bins trades by depth and calculates execution rates
    """
    
    vprint("Calculating trade execution rates by depth...")
    
    # Sort by timestamp
    df_quotes = df_quotes.sort_values('timestamp').reset_index(drop=True)
    df_trades = df_trades.sort_values('timestamp').reset_index(drop=True)
    
    # Store trade depths
    buy_depths = []   # When market order hits ask (positive depth from best ask)
    sell_depths = []  # When market order hits bid (positive depth from best bid)
    
    trades_analyzed = 0
    trades_skipped = 0
    
    for idx, trade in df_trades.iterrows():
        trade_time = trade['timestamp']
        
        # Find orderbook state just before this trade
        pre_trade_quotes = df_quotes[df_quotes['timestamp'] <= trade_time - pd.Timedelta(milliseconds=lookback_ms)]
        
        if len(pre_trade_quotes) == 0:
            trades_skipped += 1
            continue
            
        # Get the most recent quote before trade
        recent_quote = pre_trade_quotes.iloc[-1]
        
        if trade['side'] == 'buy':
            # Buy order hits ask side
            best_ask = recent_quote['ask_price_0']
            if pd.notna(best_ask) and trade['price'] >= best_ask:
                depth = trade['price'] - best_ask
                buy_depths.append(depth)
                trades_analyzed += 1
        else:
            # Sell order hits bid side  
            best_bid = recent_quote['bid_price_0']
            if pd.notna(best_bid) and trade['price'] <= best_bid:
                depth = best_bid - trade['price']
                sell_depths.append(depth)
                trades_analyzed += 1
    
    vprint(f"Analyzed {trades_analyzed} trades, skipped {trades_skipped}")
    
    return np.array(buy_depths), np.array(sell_depths)

def bin_and_calculate_intensity(depths, n_bins=15, total_time_minutes=30):
    """
    Bin depths and calculate trade intensity (trades per second) for each bin.
    """
    
    if len(depths) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Remove outliers (beyond 99th percentile)
    depth_99 = np.percentile(depths, 99)
    depths_clean = depths[depths <= depth_99]
    
    if len(depths_clean) < 10:  # Need minimum data
        return np.array([]), np.array([]), np.array([])
    
    # Create bins
    max_depth = np.max(depths_clean) 
    bins = np.linspace(0, max_depth, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Count trades in each bin
    trade_counts, _ = np.histogram(depths_clean, bins=bins)
    
    # Calculate intensity (trades per second)
    duration_seconds = total_time_minutes * 60.0
    intensities = trade_counts / duration_seconds
    
    return bin_centers, intensities, trade_counts

def fit_lambda_kappa(depths, counts, duration_seconds):
    """
    Fit lambda0 and kappa from binned counts using weighted log-linear regression
    for lambda(delta) = lambda0 * exp(-kappa * delta). Approximates Poisson MLE.
    """
    mask = (counts > 0) & np.isfinite(depths)
    if np.sum(mask) < 3:
        return {
            'lambda_0': np.nan,
            'kappa': np.nan,
            'r_squared': np.nan,
            'n_points': 0
        }
    
    delta = depths[mask]
    counts_sel = counts[mask].astype(float)
    intensity = counts_sel / duration_seconds
    
    if np.any(intensity <= 0):
        return {
            'lambda_0': np.nan,
            'kappa': np.nan,
            'r_squared': np.nan,
            'n_points': 0
        }
    
    # Weighted least squares on log intensity
    y = np.log(intensity)
    X = np.column_stack((np.ones_like(delta), -delta))
    w = np.sqrt(counts_sel)
    Xt = X * w[:, None]
    yt = y * w
    try:
        coef, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
        intercept, slope_neg_delta = coef
    except Exception:
        return {
            'lambda_0': np.nan,
            'kappa': np.nan,
            'r_squared': np.nan,
            'n_points': 0
        }
    
    lambda_0 = np.exp(intercept)
    kappa = max(slope_neg_delta, 0.0)
    
    # R^2 on weighted log space
    y_pred = intercept + slope_neg_delta * (-delta)
    ss_res = np.sum(w * w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * w * (y - np.average(y, weights=w*w)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    return {
        'lambda_0': lambda_0,
        'kappa': kappa,
        'r_squared': r_squared,
        'n_points': len(delta)
    }

def estimate_kappa_linear_method(depths, intensities):
    """
    Estimate kappa using linear regression on log(intensity) vs depth.
    
    From lambda(delta) = lambda0 * exp(-kappa * delta), taking log:
    log(lambda(delta)) = log(lambda0) - kappa * delta
    
    So slope of log(intensity) vs depth gives -kappa.
    """
    
    # Filter valid data points
    valid_mask = (intensities > 0) & np.isfinite(intensities) & np.isfinite(depths)
    
    if np.sum(valid_mask) < 3:
        return {
            'kappa': np.nan,
            'lambda_0': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'n_points': 0
        }
    
    valid_depths = depths[valid_mask]
    valid_intensities = intensities[valid_mask]
    
    # Linear regression on log scale
    log_intensities = np.log(valid_intensities)
    
    slope, intercept, r_value, p_value, std_err = linregress(valid_depths, log_intensities)
    
    # Extract parameters
    kappa_estimate = -slope  # Since slope = -kappa
    lambda_0_estimate = np.exp(intercept)
    r_squared = r_value ** 2
    
    return {
        'kappa': max(kappa_estimate, 0),  # Ensure positive
        'lambda_0': lambda_0_estimate,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_error': std_err,
        'n_points': len(valid_depths)
    }

def estimate_kappa_nonlinear_method(depths, intensities):
    """
    Estimate kappa using nonlinear curve fitting.
    """
    
    def exponential_decay(delta, lambda_0, kappa):
        return lambda_0 * np.exp(-kappa * delta)
    
    # Filter valid data
    valid_mask = (intensities > 0) & np.isfinite(intensities) & np.isfinite(depths)
    
    if np.sum(valid_mask) < 4:
        return {
            'kappa': np.nan,
            'lambda_0': np.nan, 
            'r_squared': np.nan,
            'n_points': 0
        }
    
    valid_depths = depths[valid_mask]
    valid_intensities = intensities[valid_mask]
    
    try:
        # Initial guess
        p0 = [np.max(valid_intensities), 50]
        
        # Fit curve
        popt, pcov = curve_fit(
            exponential_decay, 
            valid_depths, 
            valid_intensities,
            p0=p0,
            bounds=([0, 1], [np.inf, 500]),
            maxfev=3000
        )
        
        lambda_0_est, kappa_est = popt
        
        # Calculate R-squared
        y_pred = exponential_decay(valid_depths, *popt)
        ss_res = np.sum((valid_intensities - y_pred) ** 2)
        ss_tot = np.sum((valid_intensities - np.mean(valid_intensities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'kappa': kappa_est,
            'lambda_0': lambda_0_est,
            'r_squared': r_squared,
            'std_error': np.sqrt(np.diag(pcov))[1] if len(pcov) > 1 else np.nan,
            'n_points': len(valid_depths)
        }
        
    except:
        return {
            'kappa': np.nan,
            'lambda_0': np.nan,
            'r_squared': np.nan,
            'std_error': np.nan,
            'n_points': 0
        }

def plot_kappa_analysis_improved(buy_depths, buy_intensities, sell_depths, sell_intensities, 
                                buy_estimates_linear, buy_estimates_nonlinear,
                                sell_estimates_linear, sell_estimates_nonlinear,
                                crypto, time_minutes):
    """Create improved diagnostic plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Add main title with crypto and time window info
    fig.suptitle(f'Kappa Estimation Analysis - {crypto} (Last {time_minutes} minutes)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Plot 1: Buy side linear scale
    ax1.scatter(buy_depths, buy_intensities, alpha=0.7, color='blue', s=50, label='Observed')
    
    if not np.isnan(buy_estimates_linear['kappa']):
        x_fit = np.linspace(0, np.max(buy_depths), 100)
        y_fit_linear = buy_estimates_linear['lambda_0'] * np.exp(-buy_estimates_linear['kappa'] * x_fit)
        ax1.plot(x_fit, y_fit_linear, 'r--', linewidth=2, 
                label=f'Linear fit: k={buy_estimates_linear["kappa"]:.1f} (R^2={buy_estimates_linear["r_squared"]:.3f})')
    
    if not np.isnan(buy_estimates_nonlinear['kappa']):
        x_fit = np.linspace(0, np.max(buy_depths), 100)
        y_fit_nonlinear = buy_estimates_nonlinear['lambda_0'] * np.exp(-buy_estimates_nonlinear['kappa'] * x_fit)
        ax1.plot(x_fit, y_fit_nonlinear, 'g:', linewidth=2,
                label=f'Nonlinear fit: k={buy_estimates_nonlinear["kappa"]:.1f} (R^2={buy_estimates_nonlinear["r_squared"]:.3f})')
    
    ax1.set_xlabel('Buy Depth (price units)')
    ax1.set_ylabel('Trade Intensity (trades/min)')
    ax1.set_title('Buy Side - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Buy side log scale
    ax2.scatter(buy_depths, buy_intensities, alpha=0.7, color='blue', s=50)
    
    if not np.isnan(buy_estimates_linear['kappa']):
        x_fit = np.linspace(0.01, np.max(buy_depths), 100)
        y_fit = buy_estimates_linear['lambda_0'] * np.exp(-buy_estimates_linear['kappa'] * x_fit)
        ax2.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'k={buy_estimates_linear["kappa"]:.1f}')
    
    ax2.set_xlabel('Buy Depth (price units)')
    ax2.set_ylabel('Trade Intensity (trades/min)')
    ax2.set_title('Buy Side - Log Scale')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sell side linear scale
    ax3.scatter(sell_depths, sell_intensities, alpha=0.7, color='red', s=50, label='Observed')
    
    if not np.isnan(sell_estimates_linear['kappa']):
        x_fit = np.linspace(0, np.max(sell_depths), 100)
        y_fit_linear = sell_estimates_linear['lambda_0'] * np.exp(-sell_estimates_linear['kappa'] * x_fit)
        ax3.plot(x_fit, y_fit_linear, 'b--', linewidth=2,
                label=f'Linear fit: k={sell_estimates_linear["kappa"]:.1f} (R^2={sell_estimates_linear["r_squared"]:.3f})')
    
    if not np.isnan(sell_estimates_nonlinear['kappa']):
        x_fit = np.linspace(0, np.max(sell_depths), 100)
        y_fit_nonlinear = sell_estimates_nonlinear['lambda_0'] * np.exp(-sell_estimates_nonlinear['kappa'] * x_fit)
        ax3.plot(x_fit, y_fit_nonlinear, 'g:', linewidth=2,
                label=f'Nonlinear fit: k={sell_estimates_nonlinear["kappa"]:.1f} (R^2={sell_estimates_nonlinear["r_squared"]:.3f})')
    
    ax3.set_xlabel('Sell Depth (price units)')
    ax3.set_ylabel('Trade Intensity (trades/min)')
    ax3.set_title('Sell Side - Linear Scale')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sell side log scale
    ax4.scatter(sell_depths, sell_intensities, alpha=0.7, color='red', s=50)
    
    if not np.isnan(sell_estimates_linear['kappa']):
        x_fit = np.linspace(0.01, np.max(sell_depths), 100)
        y_fit = sell_estimates_linear['lambda_0'] * np.exp(-sell_estimates_linear['kappa'] * x_fit)
        ax4.plot(x_fit, y_fit, 'b--', linewidth=2, label=f'k={sell_estimates_linear["kappa"]:.1f}')
    
    ax4.set_xlabel('Sell Depth (price units)')
    ax4.set_ylabel('Trade Intensity (trades/min)')
    ax4.set_title('Sell Side - Log Scale')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Adjust top margin for suptitle
    plt.show()

def save_kappa_lambda_to_json(kappa_plus, kappa_minus, lambda_plus, lambda_minus,
                              crypto: str, kappa_file: str = "kappa.json", lambda_file: str = "lambda.json"):
    """Persist kappa and base lambda0 estimates (per second) to JSON files."""
    def _load(path):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    kappa_data = _load(kappa_file)
    lambda_data = _load(lambda_file)

    kappa_data[crypto] = {
        "kappa+": float(kappa_plus) if kappa_plus is not None else None,
        "kappa-": float(kappa_minus) if kappa_minus is not None else None
    }
    lambda_data[crypto] = {
        "lambda+": float(lambda_plus) if lambda_plus is not None else None,
        "lambda-": float(lambda_minus) if lambda_minus is not None else None,
        "unit": "trades_per_second"
    }

    with open(kappa_file, 'w') as f:
        json.dump(kappa_data, f, indent=4)
    with open(lambda_file, 'w') as f:
        json.dump(lambda_data, f, indent=4)

    print(f"[save] kappa -> {kappa_file}")
    print(f"[save] lambda0 -> {lambda_file}")


def run_kappa_for_crypto(crypto: str, minutes: int = 30, bins: int = 20, do_plot: bool = False):
    """Run the full kappa estimation flow for a single crypto symbol."""
    log_section(f"KAPPA/LAMBDA FROM MARKET DATA - {crypto} (last {minutes} min)")
    # Load market data
    try:
        df_quotes, df_trades = load_market_data(crypto, minutes)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    vprint(f"Loaded {len(df_quotes):,} quote records")
    vprint(f"Loaded {len(df_trades):,} trade records")
    total_time_minutes = minutes

    # Calculate trade execution depths
    vprint("\n" + "="*60)
    vprint("ANALYZING TRADE EXECUTION DEPTHS...")
    vprint("="*60)

    buy_depths, sell_depths = calculate_trade_execution_rates(df_quotes, df_trades)

    vprint(f"Found {len(buy_depths)} buy trades and {len(sell_depths)} sell trades with valid depths")

    if len(buy_depths) == 0 and len(sell_depths) == 0:
        print("No valid trades found for depth analysis!")
        return

    # Bin depths and calculate intensities
    buy_depth_bins, buy_intensities, buy_counts = bin_and_calculate_intensity(buy_depths, bins, total_time_minutes)
    sell_depth_bins, sell_intensities, sell_counts = bin_and_calculate_intensity(sell_depths, bins, total_time_minutes)

    duration_seconds = total_time_minutes * 60.0

    buy_est = fit_lambda_kappa(buy_depth_bins, buy_counts, duration_seconds) if len(buy_depth_bins) > 0 else {'lambda_0': np.nan, 'kappa': np.nan, 'r_squared': np.nan, 'n_points': 0}
    sell_est = fit_lambda_kappa(sell_depth_bins, sell_counts, duration_seconds) if len(sell_depth_bins) > 0 else {'lambda_0': np.nan, 'kappa': np.nan, 'r_squared': np.nan, 'n_points': 0}

    log_section(f"KAPPA/LAMBDA ESTIMATES - {crypto}")
    print(f"Time window: {minutes} minutes ({duration_seconds:.0f} seconds)")
    if not np.isnan(buy_est['kappa']):
        print(f"  kappa+ (ask): {buy_est['kappa']:.4f}, lambda+ (delta=0)={buy_est['lambda_0']:.6f} trades/sec, R^2={buy_est['r_squared']:.3f}, n={buy_est['n_points']}")
    else:
        print("  kappa+/lambda+ unavailable (insufficient data)")
    if not np.isnan(sell_est['kappa']):
        print(f"  kappa- (bid): {sell_est['kappa']:.4f}, lambda- (delta=0)={sell_est['lambda_0']:.6f} trades/sec, R^2={sell_est['r_squared']:.3f}, n={sell_est['n_points']}")
    else:
        print("  kappa-/lambda- unavailable (insufficient data)")

    save_kappa_lambda_to_json(
        buy_est['kappa'] if not np.isnan(buy_est['kappa']) else None,
        sell_est['kappa'] if not np.isnan(sell_est['kappa']) else None,
        buy_est['lambda_0'] if not np.isnan(buy_est['lambda_0']) else None,
        sell_est['lambda_0'] if not np.isnan(sell_est['lambda_0']) else None,
        crypto
    )

    # Create visualization (optional)
    if (len(buy_depth_bins) > 0 or len(sell_depth_bins) > 0) and do_plot:
        buy_est_linear = {'kappa': buy_est['kappa'], 'lambda_0': buy_est['lambda_0'], 'r_squared': buy_est['r_squared']}
        sell_est_linear = {'kappa': sell_est['kappa'], 'lambda_0': sell_est['lambda_0'], 'r_squared': sell_est['r_squared']}
        plot_kappa_analysis_improved(
            buy_depth_bins, buy_intensities,
            sell_depth_bins, sell_intensities,
            buy_est_linear, {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan},
            sell_est_linear, {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan},
            crypto, minutes
        )

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joint kappa/lambda estimation from market data')
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'ETH'), 
                        help='Cryptocurrency symbol (e.g., ETH) or ALL to process every symbol in HL_data')
    parser.add_argument('--minutes', '-m', type=int, default=30,
                        help='Number of minutes from most recent data to analyze')
    parser.add_argument('--bins', '-b', type=int, default=20,
                        help='Number of depth bins for analysis')
    parser.add_argument('--plot', '-p', action='store_true', default=False,
                        help='Show diagnostic plots (disabled by default)')
    parser.add_argument('--verbosity', '-v', type=int, choices=[0, 1], default=0,
                        help='Verbosity: 0=minimal (default), 1=verbose')

    args = parser.parse_args()

    _set_verbosity(args.verbosity)

    if isinstance(args.crypto, str) and args.crypto.strip().upper() == 'ALL':
        symbols = list_available_cryptos('HL_data')
        if not symbols:
            print("No crypto data found in HL_data (need orderbooks and trades parquet).")
            raise SystemExit(1)
    else:
        symbols = [args.crypto.strip().upper()]

    for sym in symbols:
        try:
            run_kappa_for_crypto(sym, minutes=args.minutes, bins=args.bins, do_plot=args.plot)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing {sym}: {e}")
