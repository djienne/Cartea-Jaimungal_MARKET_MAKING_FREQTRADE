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

def load_market_data(crypto='BTC', time_range_minutes=60):
    """Load market data for specified cryptocurrency and time range."""
    
    vprint(f"Loading market data for {crypto} (last {time_range_minutes} minutes)...")
    
    # Check if files exist
    orderbook_file = f'HL_data/orderbooks_{crypto}.csv'
    trades_file = f'HL_data/trades_{crypto}.csv'
    
    if not os.path.exists(orderbook_file):
        raise FileNotFoundError(f"Orderbook file not found: {orderbook_file}")
    if not os.path.exists(trades_file):
        raise FileNotFoundError(f"Trades file not found: {trades_file}")
    
    # Load data
    vprint("Loading orderbook data...")
    df_orderbook = pd.read_csv(orderbook_file)
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='s')
    
    vprint("Loading trade data...")
    df_trades_raw = pd.read_csv(trades_file)
    df_trades_raw['timestamp'] = pd.to_datetime(df_trades_raw['timestamp'], unit='s')
    
    # Find overlapping time window
    most_recent_quotes = df_orderbook['timestamp'].max()
    most_recent_trades = df_trades_raw['timestamp'].max()
    most_recent = min(most_recent_quotes, most_recent_trades)
    time_window_start = most_recent - timedelta(minutes=time_range_minutes)
    
    vprint(f"Most recent data: {most_recent}")
    vprint(f"Time window: {time_window_start} to {most_recent}")
    
    # Filter to time range
    df_orderbook = df_orderbook[
        (df_orderbook['timestamp'] >= time_window_start) & 
        (df_orderbook['timestamp'] <= most_recent)
    ].reset_index(drop=True)
    
    df_trades_raw = df_trades_raw[
        (df_trades_raw['timestamp'] >= time_window_start) & 
        (df_trades_raw['timestamp'] <= most_recent)
    ].reset_index(drop=True)
    
    # Process trades
    df_trades = df_trades_raw[['timestamp', 'side', 'size', 'price']].copy()
    
    # Final overlap filtering
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
    
    return df_quotes, df_trades


def list_available_cryptos(data_dir: str = 'HL_data'):
    """Return sorted list of crypto symbols that have both orderbooks_*.csv and trades_*.csv"""
    if not os.path.isdir(data_dir):
        return []
    try:
        files = os.listdir(data_dir)
    except Exception:
        return []
    ob = {f.split('_', 1)[1].split('.', 1)[0] for f in files if f.startswith('orderbooks_') and f.endswith('.csv') and '_' in f}
    tr = {f.split('_', 1)[1].split('.', 1)[0] for f in files if f.startswith('trades_') and f.endswith('.csv') and '_' in f}
    syms = sorted(ob.intersection(tr))
    return syms

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
    Bin depths and calculate trade intensity (trades per minute) for each bin.
    """
    
    if len(depths) == 0:
        return np.array([]), np.array([])
    
    # Remove outliers (beyond 99th percentile)
    depth_99 = np.percentile(depths, 99)
    depths_clean = depths[depths <= depth_99]
    
    if len(depths_clean) < 10:  # Need minimum data
        return np.array([]), np.array([])
    
    # Create bins
    max_depth = np.max(depths_clean) 
    bins = np.linspace(0, max_depth, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Count trades in each bin
    trade_counts, _ = np.histogram(depths_clean, bins=bins)
    
    # Calculate intensity (trades per minute)
    intensities = trade_counts / total_time_minutes
    
    return bin_centers, intensities

def estimate_kappa_linear_method(depths, intensities):
    """
    Estimate kappa using linear regression on log(intensity) vs depth.
    
    From λ(δ) = λ₀ * exp(-κ * δ), taking log:
    log(λ(δ)) = log(λ₀) - κ * δ
    
    So slope of log(intensity) vs depth gives -κ.
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
    kappa_estimate = -slope  # Since slope = -κ
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
                label=f'Linear fit: k={buy_estimates_linear["kappa"]:.1f} (R²={buy_estimates_linear["r_squared"]:.3f})')
    
    if not np.isnan(buy_estimates_nonlinear['kappa']):
        x_fit = np.linspace(0, np.max(buy_depths), 100)
        y_fit_nonlinear = buy_estimates_nonlinear['lambda_0'] * np.exp(-buy_estimates_nonlinear['kappa'] * x_fit)
        ax1.plot(x_fit, y_fit_nonlinear, 'g:', linewidth=2,
                label=f'Nonlinear fit: k={buy_estimates_nonlinear["kappa"]:.1f} (R²={buy_estimates_nonlinear["r_squared"]:.3f})')
    
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
                label=f'Linear fit: k={sell_estimates_linear["kappa"]:.1f} (R²={sell_estimates_linear["r_squared"]:.3f})')
    
    if not np.isnan(sell_estimates_nonlinear['kappa']):
        x_fit = np.linspace(0, np.max(sell_depths), 100)
        y_fit_nonlinear = sell_estimates_nonlinear['lambda_0'] * np.exp(-sell_estimates_nonlinear['kappa'] * x_fit)
        ax3.plot(x_fit, y_fit_nonlinear, 'g:', linewidth=2,
                label=f'Nonlinear fit: k={sell_estimates_nonlinear["kappa"]:.1f} (R²={sell_estimates_nonlinear["r_squared"]:.3f})')
    
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

def save_kappa_to_json(kappa_estimates: dict, crypto: str, filename: str = "kappa.json"):
    """Save the best kappa estimates from linear method to JSON file"""
    
    # Extract linear method estimates (most reliable)
    kappa_plus = None
    kappa_minus = None
    
    # Get the best linear estimates from bid and ask sides
    if 'kappa_bid' in kappa_estimates and not np.isnan(kappa_estimates['kappa_bid']['kappa']):
        kappa_minus = float(kappa_estimates['kappa_bid']['kappa'])  # Bid side = kappa-
        
    if 'kappa_ask' in kappa_estimates and not np.isnan(kappa_estimates['kappa_ask']['kappa']):
        kappa_plus = float(kappa_estimates['kappa_ask']['kappa'])   # Ask side = kappa+
    
    # Load existing data if file exists
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    else:
        data = {}
    
    # Update with new estimates
    data[crypto] = {
        "kappa+": kappa_plus,
        "kappa-": kappa_minus
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    vprint(f"\nKappa estimates saved to {filename}")
    vprint(f"  {crypto}: kappa+ = {kappa_plus}, kappa- = {kappa_minus}")


def run_kappa_for_crypto(crypto: str, minutes: int = 30, bins: int = 20, do_plot: bool = False):
    """Run the full kappa estimation flow for a single crypto symbol."""
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
    buy_depth_bins, buy_intensities = bin_and_calculate_intensity(buy_depths, bins, total_time_minutes)
    sell_depth_bins, sell_intensities = bin_and_calculate_intensity(sell_depths, bins, total_time_minutes)

    # Estimate kappa using both methods
    buy_est_linear = estimate_kappa_linear_method(buy_depth_bins, buy_intensities) if len(buy_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}
    buy_est_nonlinear = estimate_kappa_nonlinear_method(buy_depth_bins, buy_intensities) if len(buy_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}

    sell_est_linear = estimate_kappa_linear_method(sell_depth_bins, sell_intensities) if len(sell_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}
    sell_est_nonlinear = estimate_kappa_nonlinear_method(sell_depth_bins, sell_intensities) if len(sell_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}

    # Display results
    vprint("\n" + "="*60)
    vprint(f"IMPROVED KAPPA ESTIMATION RESULTS FOR {crypto}")
    vprint("="*60)

    vprint("\nBuy Side (kappa+):")
    if not np.isnan(buy_est_linear['kappa']):
        vprint(f"  Linear Method:")
        vprint(f"    kappa: {buy_est_linear['kappa']:.2f}")
        vprint(f"    RA�: {buy_est_linear['r_squared']:.3f}")
        vprint(f"    p-value: {buy_est_linear['p_value']:.6f}")

    if not np.isnan(buy_est_nonlinear['kappa']):
        vprint(f"  Nonlinear Method:")
        vprint(f"    kappa: {buy_est_nonlinear['kappa']:.2f}")
        vprint(f"    RA�: {buy_est_nonlinear['r_squared']:.3f}")

    vprint("\nSell Side (kappa-):")
    if not np.isnan(sell_est_linear['kappa']):
        vprint(f"  Linear Method:")
        vprint(f"    kappa: {sell_est_linear['kappa']:.2f}")
        vprint(f"    RA�: {sell_est_linear['r_squared']:.3f}")
        vprint(f"    p-value: {sell_est_linear['p_value']:.6f}")

    if not np.isnan(sell_est_nonlinear['kappa']):
        vprint(f"  Nonlinear Method:")
        vprint(f"    kappa: {sell_est_nonlinear['kappa']:.2f}")
        vprint(f"    RA�: {sell_est_nonlinear['r_squared']:.3f}")

    # Best estimates
    valid_kappas = []
    best_estimates = []

    if not np.isnan(buy_est_linear['kappa']) and buy_est_linear['r_squared'] > 0.1:
        valid_kappas.append(buy_est_linear['kappa'])
        best_estimates.append(('Buy Linear', buy_est_linear))

    if not np.isnan(buy_est_nonlinear['kappa']) and buy_est_nonlinear['r_squared'] > buy_est_linear.get('r_squared', -1):
        valid_kappas.append(buy_est_nonlinear['kappa'])
        best_estimates.append(('Buy Nonlinear', buy_est_nonlinear))

    if not np.isnan(sell_est_linear['kappa']) and sell_est_linear['r_squared'] > 0.1:
        valid_kappas.append(sell_est_linear['kappa'])
        best_estimates.append(('Sell Linear', sell_est_linear))

    if not np.isnan(sell_est_nonlinear['kappa']) and sell_est_nonlinear['r_squared'] > sell_est_linear.get('r_squared', -1):
        valid_kappas.append(sell_est_nonlinear['kappa'])
        best_estimates.append(('Sell Nonlinear', sell_est_nonlinear))

    if valid_kappas:
        avg_kappa = np.mean(valid_kappas)
        vprint(f"\nRecommended kappa estimate: {avg_kappa:.2f}")

        # Show which estimates were used
        vprint("Based on:")
        for name, est in best_estimates:
            vprint(f"  - {name}: kappa={est['kappa']:.2f} (RA�={est['r_squared']:.3f})")

        # Market interpretation
        vprint(f"\nMarket Interpretation:")
        if avg_kappa > 200:
            vprint("  Very liquid market - high depth sensitivity")
        elif avg_kappa > 100:
            vprint("  Moderately liquid market")
        else:
            vprint("  Lower liquidity - less depth sensitivity")
    else:
        vprint("\nNo reliable kappa estimates obtained.")
        vprint("This could indicate:")
        vprint("  - Insufficient data in the time window")
        vprint("  - Market structure doesn't follow exponential decay")
        vprint("  - Need longer time window or different crypto")

    # Save kappa estimates to JSON file
    kappa_estimates = {
        'kappa_ask': buy_est_linear if not np.isnan(buy_est_linear['kappa']) and buy_est_linear.get('r_squared', 0) > 0.1 else {'kappa': np.nan},
        'kappa_bid': sell_est_linear if not np.isnan(sell_est_linear['kappa']) and sell_est_linear.get('r_squared', 0) > 0.1 else {'kappa': np.nan}
    }
    save_kappa_to_json(kappa_estimates, crypto)

    # Minimal summary line when verbosity=0
    if VERBOSITY == 0:
        kp = kappa_estimates['kappa_ask'].get('kappa') if isinstance(kappa_estimates['kappa_ask'], dict) else None
        km = kappa_estimates['kappa_bid'].get('kappa') if isinstance(kappa_estimates['kappa_bid'], dict) else None
        try:
            kp_s = f"{float(kp):.2f}" if kp is not None and not np.isnan(kp) else "NA"
        except Exception:
            kp_s = "NA"
        try:
            km_s = f"{float(km):.2f}" if km is not None and not np.isnan(km) else "NA"
        except Exception:
            km_s = "NA"
        try:
            avg_s = f"{avg_kappa:.2f}" if 'avg_kappa' in locals() else "NA"
        except Exception:
            avg_s = "NA"
        print(f"{crypto}: kappa+={kp_s}, kappa-={km_s}, avg={avg_s}")

    # Create visualization (optional)
    if (len(buy_depth_bins) > 0 or len(sell_depth_bins) > 0) and do_plot:
        plot_kappa_analysis_improved(
            buy_depth_bins, buy_intensities,
            sell_depth_bins, sell_intensities,
            buy_est_linear, buy_est_nonlinear,
            sell_est_linear, sell_est_nonlinear,
            crypto, minutes
        )

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved kappa estimation from market data')
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'ALL'), 
                        help='Cryptocurrency symbol (e.g., BTC) or ALL to process every symbol in HL_data')
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

    # Determine symbols to process and run them, then exit to skip legacy block
    if isinstance(args.crypto, str) and args.crypto.strip().upper() == 'ALL':
        symbols = list_available_cryptos('HL_data')
        if not symbols:
            print("No crypto data found in HL_data (need orderbooks_*.csv and trades_*.csv)")
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

    raise SystemExit(0)
    
    # Load market data
    try:
        df_quotes, df_trades = load_market_data(args.crypto, args.minutes)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    print(f"Loaded {len(df_quotes):,} quote records")
    print(f"Loaded {len(df_trades):,} trade records")
    total_time_minutes = args.minutes
    
    # Calculate trade execution depths
    print("\n" + "="*60)
    print("ANALYZING TRADE EXECUTION DEPTHS...")
    print("="*60)
    
    buy_depths, sell_depths = calculate_trade_execution_rates(df_quotes, df_trades)
    
    print(f"Found {len(buy_depths)} buy trades and {len(sell_depths)} sell trades with valid depths")
    
    if len(buy_depths) == 0 and len(sell_depths) == 0:
        print("No valid trades found for depth analysis!")
        exit(1)
    
    # Bin depths and calculate intensities
    buy_depth_bins, buy_intensities = bin_and_calculate_intensity(buy_depths, args.bins, total_time_minutes)
    sell_depth_bins, sell_intensities = bin_and_calculate_intensity(sell_depths, args.bins, total_time_minutes)
    
    # Estimate kappa using both methods
    buy_est_linear = estimate_kappa_linear_method(buy_depth_bins, buy_intensities) if len(buy_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}
    buy_est_nonlinear = estimate_kappa_nonlinear_method(buy_depth_bins, buy_intensities) if len(buy_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}
    
    sell_est_linear = estimate_kappa_linear_method(sell_depth_bins, sell_intensities) if len(sell_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}
    sell_est_nonlinear = estimate_kappa_nonlinear_method(sell_depth_bins, sell_intensities) if len(sell_depth_bins) > 0 else {'kappa': np.nan, 'lambda_0': np.nan, 'r_squared': np.nan, 'n_points': 0}
    
    # Display results
    print("\n" + "="*60)
    print(f"IMPROVED KAPPA ESTIMATION RESULTS FOR {args.crypto}")
    print("="*60)
    
    print("\nBuy Side (kappa+):")
    if not np.isnan(buy_est_linear['kappa']):
        print(f"  Linear Method:")
        print(f"    kappa: {buy_est_linear['kappa']:.2f}")
        print(f"    R²: {buy_est_linear['r_squared']:.3f}")
        print(f"    p-value: {buy_est_linear['p_value']:.6f}")
    
    if not np.isnan(buy_est_nonlinear['kappa']):
        print(f"  Nonlinear Method:")
        print(f"    kappa: {buy_est_nonlinear['kappa']:.2f}")
        print(f"    R²: {buy_est_nonlinear['r_squared']:.3f}")
    
    print("\nSell Side (kappa-):")
    if not np.isnan(sell_est_linear['kappa']):
        print(f"  Linear Method:")
        print(f"    kappa: {sell_est_linear['kappa']:.2f}")
        print(f"    R²: {sell_est_linear['r_squared']:.3f}")
        print(f"    p-value: {sell_est_linear['p_value']:.6f}")
    
    if not np.isnan(sell_est_nonlinear['kappa']):
        print(f"  Nonlinear Method:")
        print(f"    kappa: {sell_est_nonlinear['kappa']:.2f}")
        print(f"    R²: {sell_est_nonlinear['r_squared']:.3f}")
    
    # Best estimates
    valid_kappas = []
    best_estimates = []
    
    if not np.isnan(buy_est_linear['kappa']) and buy_est_linear['r_squared'] > 0.1:
        valid_kappas.append(buy_est_linear['kappa'])
        best_estimates.append(('Buy Linear', buy_est_linear))
        
    if not np.isnan(buy_est_nonlinear['kappa']) and buy_est_nonlinear['r_squared'] > buy_est_linear.get('r_squared', -1):
        valid_kappas.append(buy_est_nonlinear['kappa'])
        best_estimates.append(('Buy Nonlinear', buy_est_nonlinear))
        
    if not np.isnan(sell_est_linear['kappa']) and sell_est_linear['r_squared'] > 0.1:
        valid_kappas.append(sell_est_linear['kappa'])
        best_estimates.append(('Sell Linear', sell_est_linear))
        
    if not np.isnan(sell_est_nonlinear['kappa']) and sell_est_nonlinear['r_squared'] > sell_est_linear.get('r_squared', -1):
        valid_kappas.append(sell_est_nonlinear['kappa'])
        best_estimates.append(('Sell Nonlinear', sell_est_nonlinear))
    
    if valid_kappas:
        avg_kappa = np.mean(valid_kappas)
        print(f"\nRecommended kappa estimate: {avg_kappa:.2f}")
        
        # Show which estimates were used
        print("Based on:")
        for name, est in best_estimates:
            print(f"  - {name}: kappa={est['kappa']:.2f} (R²={est['r_squared']:.3f})")
        
        # Market interpretation
        print(f"\nMarket Interpretation:")
        if avg_kappa > 200:
            print("  Very liquid market - high depth sensitivity")
        elif avg_kappa > 100:
            print("  Moderately liquid market")  
        else:
            print("  Lower liquidity - less depth sensitivity")
    else:
        print("\nNo reliable kappa estimates obtained.")
        print("This could indicate:")
        print("  - Insufficient data in the time window")
        print("  - Market structure doesn't follow exponential decay")
        print("  - Need longer time window or different crypto")
    
    # Save kappa estimates to JSON file
    # Create a structure compatible with the save function
    kappa_estimates = {
        'kappa_ask': buy_est_linear if not np.isnan(buy_est_linear['kappa']) and buy_est_linear.get('r_squared', 0) > 0.1 else {'kappa': np.nan},
        'kappa_bid': sell_est_linear if not np.isnan(sell_est_linear['kappa']) and sell_est_linear.get('r_squared', 0) > 0.1 else {'kappa': np.nan}
    }
    save_kappa_to_json(kappa_estimates, args.crypto)
    
    # Create visualization (optional)
    if (len(buy_depth_bins) > 0 or len(sell_depth_bins) > 0) and args.plot:
        plot_kappa_analysis_improved(
            buy_depth_bins, buy_intensities, 
            sell_depth_bins, sell_intensities,
            buy_est_linear, buy_est_nonlinear,
            sell_est_linear, sell_est_nonlinear,
            args.crypto, args.minutes
        )
