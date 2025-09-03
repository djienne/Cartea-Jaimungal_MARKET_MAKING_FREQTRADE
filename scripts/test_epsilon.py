import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
import os
import json

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

def load_market_data(crypto='BTC', time_range_minutes=60):
    """
    Load market data for specified cryptocurrency and time range.
    
    Parameters:
    -----------
    crypto : str
        Cryptocurrency symbol (BTC, ETH, SOL, WLFI)
    time_range_minutes : int
        Number of minutes from most recent data to include
    
    Returns:
    --------
    df_quotes, df_trades : DataFrames with market data
    """
    
    vprint(f"Loading market data for {crypto} (last {time_range_minutes} minutes)...")
    
    # Check if files exist
    orderbook_file = f'HL_data/orderbooks_{crypto}.csv'
    trades_file = f'HL_data/trades_{crypto}.csv'
    
    if not os.path.exists(orderbook_file):
        raise FileNotFoundError(f"Orderbook file not found: {orderbook_file}")
    if not os.path.exists(trades_file):
        raise FileNotFoundError(f"Trades file not found: {trades_file}")
    
    # Load orderbook data
    vprint("Loading orderbook data...")
    df_orderbook = pd.read_csv(orderbook_file)
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='s')
    
    # Load trade data  
    vprint("Loading trade data...")
    df_trades_raw = pd.read_csv(trades_file)
    df_trades_raw['timestamp'] = pd.to_datetime(df_trades_raw['timestamp'], unit='s')
    
    # Find the most recent timestamp across both datasets
    most_recent_quotes = df_orderbook['timestamp'].max()
    most_recent_trades = df_trades_raw['timestamp'].max()
    most_recent = min(most_recent_quotes, most_recent_trades)  # Use the earlier of the two
    
    # Calculate time window
    time_window_start = most_recent - timedelta(minutes=time_range_minutes)
    
    vprint(f"Most recent data: {most_recent}")
    vprint(f"Time window: {time_window_start} to {most_recent}")
    
    # Filter data to specified time range
    df_orderbook = df_orderbook[
        (df_orderbook['timestamp'] >= time_window_start) & 
        (df_orderbook['timestamp'] <= most_recent)
    ].reset_index(drop=True)
    
    df_trades_raw = df_trades_raw[
        (df_trades_raw['timestamp'] >= time_window_start) & 
        (df_trades_raw['timestamp'] <= most_recent)
    ].reset_index(drop=True)
    
    # Create quotes DataFrame with bid/ask/mid prices
    df_quotes = pd.DataFrame({
        'timestamp': df_orderbook['timestamp'],
        'bid': df_orderbook['bid_price_0'],
        'ask': df_orderbook['ask_price_0']
    })
    df_quotes['mid'] = (df_quotes['bid'] + df_quotes['ask']) / 2
    
    # Process trades data to match expected format
    df_trades = df_trades_raw[['timestamp', 'side', 'size', 'price']].copy()
    
    # Final filtering to ensure overlapping time range
    start_time = max(df_quotes['timestamp'].min(), df_trades['timestamp'].min())
    end_time = min(df_quotes['timestamp'].max(), df_trades['timestamp'].max())
    
    df_quotes = df_quotes[
        (df_quotes['timestamp'] >= start_time) & 
        (df_quotes['timestamp'] <= end_time)
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

# ============================================================================
# EPSILON CALCULATION
# ============================================================================

def calculate_epsilon(df_quotes, df_trades, 
                     lookback_window_ms=1000,  # Time before trade to measure initial price
                     impact_windows_ms=[100, 500, 1000, 5000, 10000]):  # Multiple windows to measure permanent impact
    """
    Calculate epsilon (permanent price impact) from market data.
    
    Parameters:
    -----------
    df_quotes : DataFrame with columns ['timestamp', 'bid', 'ask', 'mid']
    df_trades : DataFrame with columns ['timestamp', 'side', 'size', 'price']
    lookback_window_ms : milliseconds before trade to establish pre-trade price
    impact_windows_ms : list of milliseconds after trade to measure impact
    
    Returns:
    --------
    Dictionary with epsilon estimates and diagnostic information
    """
    
    results = {
        'buy_impacts': {window: [] for window in impact_windows_ms},
        'sell_impacts': {window: [] for window in impact_windows_ms},
        'trades_analyzed': 0,
        'trades_skipped': 0
    }
    
    # Sort data by timestamp
    df_quotes = df_quotes.sort_values('timestamp')
    df_trades = df_trades.sort_values('timestamp')
    
    for idx, trade in df_trades.iterrows():
        trade_time = trade['timestamp']
        trade_side = trade['side']
        trade_size = trade['size']
        
        # Find pre-trade mid price (lookback window)
        pre_trade_window = (
            df_quotes['timestamp'] >= trade_time - pd.Timedelta(milliseconds=lookback_window_ms),
            df_quotes['timestamp'] < trade_time
        )
        pre_trade_quotes = df_quotes.loc[pre_trade_window[0] & pre_trade_window[1]]
        
        if len(pre_trade_quotes) == 0:
            results['trades_skipped'] += 1
            continue
            
        # Use the last quote before trade as reference
        pre_trade_mid = pre_trade_quotes['mid'].iloc[-1]
        
        # Measure impact at different time horizons
        for window_ms in impact_windows_ms:
            post_trade_window = (
                df_quotes['timestamp'] > trade_time,
                df_quotes['timestamp'] <= trade_time + pd.Timedelta(milliseconds=window_ms)
            )
            post_trade_quotes = df_quotes.loc[post_trade_window[0] & post_trade_window[1]]
            
            if len(post_trade_quotes) == 0:
                continue
            
            # Use the last quote in the window as the "permanent" price
            post_trade_mid = post_trade_quotes['mid'].iloc[-1]
            
            # Calculate signed impact (positive for buy, negative for sell)
            if trade_side == 'buy':
                impact = post_trade_mid - pre_trade_mid
                results['buy_impacts'][window_ms].append({
                    'impact': impact,
                    'size': trade_size,
                    'normalized_impact': impact / np.log(1 + trade_size/1000)  # Size-normalized
                })
            else:
                impact = pre_trade_mid - post_trade_mid
                results['sell_impacts'][window_ms].append({
                    'impact': impact,
                    'size': trade_size,
                    'normalized_impact': impact / np.log(1 + trade_size/1000)
                })
        
        results['trades_analyzed'] += 1
    
    return results

def estimate_epsilon_parameters(results, window_ms=5000):
    """
    Estimate epsilon+ and epsilon- from impact measurements.
    
    Uses multiple methods:
    1. Simple mean of all impacts
    2. Size-weighted mean
    3. Median (robust to outliers)
    4. Trimmed mean (remove top/bottom 10%)
    """
    
    estimates = {}
    
    for side in ['buy', 'sell']:
        impacts_data = results[f'{side}_impacts'][window_ms]
        
        if len(impacts_data) == 0:
            estimates[f'epsilon_{side}'] = {
                'mean': 0, 'weighted_mean': 0, 'median': 0, 
                'trimmed_mean': 0, 'std': 0, 'n_trades': 0
            }
            continue
        
        impacts = np.array([d['impact'] for d in impacts_data])
        sizes = np.array([d['size'] for d in impacts_data])
        normalized = np.array([d['normalized_impact'] for d in impacts_data])
        
        # Remove outliers (impacts > 3 std)
        mean_imp = np.mean(impacts)
        std_imp = np.std(impacts)
        mask = np.abs(impacts - mean_imp) < 3 * std_imp
        impacts_clean = impacts[mask]
        sizes_clean = sizes[mask]
        
        estimates[f'epsilon_{side}'] = {
            'mean': np.mean(impacts_clean) if len(impacts_clean) > 0 else 0,
            'weighted_mean': np.average(impacts_clean, weights=sizes_clean) if len(impacts_clean) > 0 else 0,
            'median': np.median(impacts_clean) if len(impacts_clean) > 0 else 0,
            'trimmed_mean': np.mean(np.sort(impacts_clean)[int(0.1*len(impacts_clean)):int(0.9*len(impacts_clean))]) if len(impacts_clean) > 2 else 0,
            'normalized_mean': np.mean(normalized[mask]) if np.sum(mask) > 0 else 0,
            'std': np.std(impacts_clean) if len(impacts_clean) > 0 else 0,
            'n_trades': len(impacts_clean)
        }
    
    return estimates

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

def save_epsilon_to_json(eps_plus: float, eps_minus: float, crypto: str, filename: str = "epsilon.json"):
    """Save the epsilon estimates from trimmed mean 5000ms to JSON file"""
    
    # Handle NaN values
    epsilon_plus_val = float(eps_plus) if not np.isnan(eps_plus) else None
    epsilon_minus_val = float(eps_minus) if not np.isnan(eps_minus) else None
    
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
        "epsilon+": epsilon_plus_val,
        "epsilon-": epsilon_minus_val
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    vprint(f"\nEpsilon estimates saved to {filename}")
    vprint(f"  {crypto}: epsilon+ = {epsilon_plus_val}, epsilon- = {epsilon_minus_val}")

def load_mid_price_from_json(crypto: str, filename: str = 'mid_price.json') -> float:
    """Load mid-price for a crypto from mid_price.json."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"mid price file not found: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
    if crypto not in data:
        raise KeyError(f"crypto '{crypto}' not found in {filename}")
    return float(data[crypto])

def run_epsilon_for_crypto(crypto: str, minutes: int = 30, do_plot: bool = False):
    """Run the full epsilon analysis and persistence for a single crypto symbol."""
    # Load market data with specified parameters
    try:
        df_quotes, df_trades = load_market_data(crypto, minutes)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Available crypto files in HL_data/:")
        for file in os.listdir('HL_data'):
            if file.startswith('orderbooks_') or file.startswith('trades_'):
                print(f"  {file}")
        return

    vprint(f"Loaded {len(df_quotes):,} quote records")
    vprint(f"Loaded {len(df_trades):,} trade records") 
    vprint(f"Duration: {df_quotes['timestamp'].max() - df_quotes['timestamp'].min()}")

    vprint("\nSample Quotes Data:")
    if VERBOSITY >= 1:
        vprint(df_quotes.head(), level=1)
    vprint("\nSample Trades Data:")
    if VERBOSITY >= 1:
        vprint(df_trades.head(), level=1)

    # Calculate impacts
    vprint("\n" + "="*60)
    vprint(f"CALCULATING PRICE IMPACTS FOR {crypto}...")
    vprint("="*60)

    impact_results = calculate_epsilon(df_quotes, df_trades)

    # Estimate epsilon for different time windows
    for window in [100, 500, 1000, 5000, 10000]:
        vprint(f"\n--- Impact Window: {window}ms ---")
        epsilon_est = estimate_epsilon_parameters(impact_results, window)
        
        for side in ['buy', 'sell']:
            eps_data = epsilon_est[f'epsilon_{side}']
            vprint(f"\nEpsilon {side.upper()}:")
            vprint(f"  Simple Mean:    {eps_data['mean']:.8f}")
            vprint(f"  Weighted Mean:  {eps_data['weighted_mean']:.8f}")
            vprint(f"  Median:         {eps_data['median']:.8f}")
            vprint(f"  Trimmed Mean:   {eps_data['trimmed_mean']:.8f}")
            vprint(f"  Std Dev:        {eps_data['std']:.8f}")
            vprint(f"  N Trades:       {eps_data['n_trades']}")

    # Final recommendations and persistence
    vprint("\n" + "="*60)
    vprint(f"RECOMMENDED EPSILON VALUES FOR {crypto} MARKET MAKING")
    vprint("="*60)

    # Use 5-second window as most stable estimate
    final_estimates = estimate_epsilon_parameters(impact_results, 5000)
    eps_plus = final_estimates['epsilon_buy']['trimmed_mean']
    eps_minus = final_estimates['epsilon_sell']['trimmed_mean']

    vprint(f"\nRecommended values for {crypto} (using 5-second window, trimmed mean):")
    vprint(f"  epsilon+ (buy impact):  {eps_plus:.8f}")
    vprint(f"  epsilon- (sell impact): {eps_minus:.8f}")
    if VERBOSITY == 0:
        print(f"{crypto}: epsilon+={eps_plus:.8f}, epsilon-={eps_minus:.8f}")

    # Check toxicity using kappa from kappa.json (per-crypto)
    try:
        kappa_plus, kappa_minus = load_kappa_from_json(crypto)
        vprint("\nToxicity check (from kappa.json):")
        vprint(f"  kappa+ x epsilon+ = {kappa_plus * eps_plus:.4f}")
        vprint(f"  kappa- x epsilon- = {kappa_minus * eps_minus:.4f}")

        kappa_epsilon_max = max(kappa_plus * eps_plus, kappa_minus * eps_minus)
        if kappa_epsilon_max >= 2:
            vprint("  WARNING: Market appears very toxic (kappa x epsilon >= 2)")
        elif kappa_epsilon_max >= 1:
            vprint("  CAUTION: Market is competitive (1 <= kappa x epsilon < 2)")
        else:
            vprint("  Market toxicity appears manageable (kappa x epsilon < 1)")
    except Exception as e:
        vprint(f"\nToxicity check skipped: {e}")

    # Calculate delta and express as % of mid-price
    try:
        mid_price = load_mid_price_from_json(crypto)
        # Guard against NaN epsilons
        eps_plus_val = 0.0 if (eps_plus is None or np.isnan(eps_plus)) else float(eps_plus)
        eps_minus_val = 0.0 if (eps_minus is None or np.isnan(eps_minus)) else float(eps_minus)

        # Reuse kappa from prior block if available; otherwise load again
        try:
            kappa_plus
            kappa_minus
        except NameError:
            kappa_plus, kappa_minus = load_kappa_from_json(crypto)

        delta_plus_price = (1.0 / float(kappa_plus)) + eps_plus_val
        delta_minus_price = (1.0 / float(kappa_minus)) + eps_minus_val

        delta_plus_pct = (delta_plus_price / mid_price) * 100.0 if mid_price != 0 else float('nan')
        delta_minus_pct = (delta_minus_price / mid_price) * 100.0 if mid_price != 0 else float('nan')

        vprint("\nDelta estimates (half-spread + skew):")
        vprint(f"  mid price [{crypto}]: {mid_price:.6f}")
        vprint(f"  delta+ = 1/kappa+ + epsilon+ = {delta_plus_price:.8f}  ({delta_plus_pct:.6f}% of mid)")
        vprint(f"  delta- = 1/kappa- + epsilon- = {delta_minus_price:.8f}  ({delta_minus_pct:.6f}% of mid)")
    except Exception as e:
        vprint(f"\nDelta calculation skipped: {e}")

    # Save epsilon estimates to JSON file
    save_epsilon_to_json(eps_plus, eps_minus, crypto)

    vprint("\nNotes:")
    vprint("- Use longer windows (5-10s) for more stable permanent impact")
    vprint("- Consider using trimmed mean or median to handle outliers")
    vprint("- Monitor epsilon over time as market conditions change")
    vprint("- Adjust for your specific latency and execution capabilities")

    vprint("\n" + "-"*60 + "\n")

    return eps_plus, eps_minus

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_impact_analysis(impact_results, window_ms=5000):
    """Create diagnostic plots for epsilon estimation"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Price Impact Analysis (Window: {window_ms}ms)', fontsize=14)
    
    for i, side in enumerate(['buy', 'sell']):
        impacts_data = impact_results[f'{side}_impacts'][window_ms]
        
        if len(impacts_data) == 0:
            continue
            
        impacts = [d['impact'] for d in impacts_data]
        sizes = [d['size'] for d in impacts_data]
        normalized = [d['normalized_impact'] for d in impacts_data]
        
        # 1. Impact Distribution
        axes[i, 0].hist(impacts, bins=20, alpha=0.7, color='blue' if side == 'buy' else 'red')
        axes[i, 0].axvline(np.mean(impacts), color='black', linestyle='--', label=f'Mean: {np.mean(impacts):.8f}')
        axes[i, 0].axvline(np.median(impacts), color='green', linestyle='--', label=f'Median: {np.median(impacts):.8f}')
        axes[i, 0].set_xlabel('Price Impact')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].set_title(f'{side.upper()} Impact Distribution')
        axes[i, 0].legend()
        
        # 2. Impact vs Size
        axes[i, 1].scatter(sizes, impacts, alpha=0.5, color='blue' if side == 'buy' else 'red')
        axes[i, 1].set_xlabel('Trade Size')
        axes[i, 1].set_ylabel('Price Impact')
        axes[i, 1].set_title(f'{side.upper()} Impact vs Size')
        
        # Fit sqrt relationship
        if len(sizes) > 3:
            z = np.polyfit(np.sqrt(sizes), impacts, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(sizes), max(sizes), 100)
            axes[i, 1].plot(x_fit, p(np.sqrt(x_fit)), "k--", alpha=0.5, label='√size fit')
            axes[i, 1].legend()
        
        # 3. Time series of impacts
        axes[i, 2].plot(impacts, marker='o', alpha=0.5, color='blue' if side == 'buy' else 'red')
        axes[i, 2].axhline(np.mean(impacts), color='black', linestyle='--', alpha=0.5)
        axes[i, 2].set_xlabel('Trade Number')
        axes[i, 2].set_ylabel('Price Impact')
        axes[i, 2].set_title(f'{side.upper()} Impact Time Series')
    
    plt.tight_layout()
    plt.show()

""" Disabled legacy single-crypto block
# Create plots (optional)
if args.plot:
    plot_impact_analysis(impact_results, window_ms=5000)

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print("\n" + "="*60)
print(f"RECOMMENDED EPSILON VALUES FOR {args.crypto} MARKET MAKING")
print("="*60)

# Use 5-second window as most stable estimate
final_estimates = estimate_epsilon_parameters(impact_results, 5000)

eps_plus = final_estimates['epsilon_buy']['trimmed_mean']
eps_minus = final_estimates['epsilon_sell']['trimmed_mean']

print(f"\nRecommended values for {args.crypto} (using 5-second window, trimmed mean):")
print(f"  epsilon+ (buy impact):  {eps_plus:.8f}")
print(f"  epsilon- (sell impact): {eps_minus:.8f}")

# Check toxicity using kappa from kappa.json (per-crypto)
try:
    kappa_plus, kappa_minus = load_kappa_from_json(args.crypto)
    print("\nToxicity check (from kappa.json):")
    print(f"  kappa+ x epsilon+ = {kappa_plus * eps_plus:.4f}")
    print(f"  kappa- x epsilon- = {kappa_minus * eps_minus:.4f}")

    kappa_epsilon_max = max(kappa_plus * eps_plus, kappa_minus * eps_minus)
    if kappa_epsilon_max >= 2:
        print("  WARNING: Market appears very toxic (kappa x epsilon >= 2)")
    elif kappa_epsilon_max >= 1:
        print("  CAUTION: Market is competitive (1 <= kappa x epsilon < 2)")
    else:
        print("  Market toxicity appears manageable (kappa x epsilon < 1)")
except Exception as e:
    print(f"\nToxicity check skipped: {e}")

# Calculate delta± = 1/kappa± + epsilon± and express as % of mid-price
try:
    mid_price = load_mid_price_from_json(args.crypto)
    # Guard against NaN epsilons
    eps_plus_val = 0.0 if (eps_plus is None or np.isnan(eps_plus)) else float(eps_plus)
    eps_minus_val = 0.0 if (eps_minus is None or np.isnan(eps_minus)) else float(eps_minus)

    # Reuse kappa from prior block if available; otherwise load again
    try:
        kappa_plus
        kappa_minus
    except NameError:
        kappa_plus, kappa_minus = load_kappa_from_json(args.crypto)

    delta_plus_price = (1.0 / float(kappa_plus)) + eps_plus_val
    delta_minus_price = (1.0 / float(kappa_minus)) + eps_minus_val

    delta_plus_pct = (delta_plus_price / mid_price) * 100.0 if mid_price != 0 else float('nan')
    delta_minus_pct = (delta_minus_price / mid_price) * 100.0 if mid_price != 0 else float('nan')

    print("\nDelta estimates (half-spread + skew):")
    print(f"  mid price [{args.crypto}]: {mid_price:.6f}")
    print(f"  delta+ = 1/kappa+ + epsilon+ = {delta_plus_price:.8f}  ({delta_plus_pct:.6f}% of mid)")
    print(f"  delta- = 1/kappa- + epsilon- = {delta_minus_price:.8f}  ({delta_minus_pct:.6f}% of mid)")
except Exception as e:
    print(f"\nDelta calculation skipped: {e}")

# Save epsilon estimates to JSON file
save_epsilon_to_json(eps_plus, eps_minus, args.crypto)

print("\nNotes:")
print("- Use longer windows (5-10s) for more stable permanent impact")
print("- Consider using trimmed mean or median to handle outliers")
print("- Monitor epsilon over time as market conditions change")
print("- Adjust for your specific latency and execution capabilities")
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate epsilon (permanent price impact) from market data')
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'ALL'),
                        help='Cryptocurrency symbol (e.g., BTC) or ALL for all available in HL_data')
    parser.add_argument('--minutes', '-m', type=int, default=30,
                        help='Number of minutes from most recent data to analyze')
    parser.add_argument('--plot', '-p', action='store_true', default=False,
                        help='Show diagnostic plots (disabled by default)')
    parser.add_argument('--verbosity', '-v', type=int, choices=[0, 1], default=0,
                        help='Verbosity: 0=minimal (default), 1=verbose')

    args = parser.parse_args()

    _set_verbosity(args.verbosity)

    # Determine cryptos to process
    if isinstance(args.crypto, str) and args.crypto.strip().upper() == 'ALL':
        symbols = list_available_cryptos('HL_data')
        if not symbols:
            print("No crypto data found in HL_data (need orderbooks_*.csv and trades_*.csv)")
            raise SystemExit(1)
    else:
        symbols = [args.crypto.strip().upper()]

    for sym in symbols:
        try:
            run_epsilon_for_crypto(sym, minutes=args.minutes, do_plot=args.plot)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing {sym}: {e}")
