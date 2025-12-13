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

def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def load_market_data(crypto='BTC', time_range_minutes=60):
    """
    Load market data for specified cryptocurrency and time range from Parquet files.
    """
    
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
    
    # Helper to load all parquet files
    def load_parquet_dir(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    # Load orderbook data
    vprint("Loading orderbook data...")
    df_orderbook = load_parquet_dir(orderbook_dir)
    if df_orderbook.empty:
        raise ValueError(f"No orderbook data found in {orderbook_dir}")
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='s')
    
    # Load trade data  
    vprint("Loading trade data...")
    df_trades_raw = load_parquet_dir(trades_dir)
    if df_trades_raw.empty:
         vprint("Warning: No trade data found.")
         df_trades_raw = pd.DataFrame(columns=['timestamp', 'side', 'size', 'price'])
    else:
        df_trades_raw['timestamp'] = pd.to_datetime(df_trades_raw['timestamp'], unit='s')
    
    # Find the most recent timestamp across both datasets
    most_recent_quotes = df_orderbook['timestamp'].max()
    if not df_trades_raw.empty:
        most_recent_trades = df_trades_raw['timestamp'].max()
        most_recent = min(most_recent_quotes, most_recent_trades)  # Use the earlier of the two
    else:
        most_recent = most_recent_quotes
    
    # Calculate time window
    time_window_start = most_recent - timedelta(minutes=time_range_minutes)
    
    vprint(f"Most recent data: {most_recent}")
    vprint(f"Time window: {time_window_start} to {most_recent}")
    
    # Filter data to specified time range
    df_orderbook = df_orderbook[
        (df_orderbook['timestamp'] >= time_window_start) & 
        (df_orderbook['timestamp'] <= most_recent)
    ].sort_values('timestamp').reset_index(drop=True)
    
    if not df_trades_raw.empty:
        df_trades_raw = df_trades_raw[
            (df_trades_raw['timestamp'] >= time_window_start) & 
            (df_trades_raw['timestamp'] <= most_recent)
        ].sort_values('timestamp').reset_index(drop=True)
    
    # Create quotes DataFrame with bid/ask/mid prices
    df_quotes = pd.DataFrame({
        'timestamp': df_orderbook['timestamp'],
        'bid': df_orderbook['bid_price_0'],
        'ask': df_orderbook['ask_price_0']
    })
    df_quotes['mid'] = (df_quotes['bid'] + df_quotes['ask']) / 2
    
    # Process trades data to match expected format
    if not df_trades_raw.empty:
        df_trades = df_trades_raw[['timestamp', 'side', 'size', 'price']].copy()
    else:
        df_trades = df_trades_raw
    
    # Final filtering to ensure overlapping time range
    if not df_trades.empty:
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

# ============================================================================
# EPSILON CALCULATION
# ============================================================================

def calculate_epsilon(df_quotes, df_trades, 
                     lookback_window_ms=200,  # Time before trade to measure initial price
                     post_window_ms=200):      # Immediate window after trade to capture permanent jump
    """
    Calculate event-level epsilon (instant permanent price impact) from market data.
    Uses the last mid before the trade and the first mid after the trade within a short window.
    """
    
    results = {
        'buy_impacts': [],
        'sell_impacts': [],
        'trades_analyzed': 0,
        'trades_skipped': 0
    }
    
    # Sort data by timestamp
    df_quotes = df_quotes.sort_values('timestamp')
    df_trades = df_trades.sort_values('timestamp')
    
    for _, trade in df_trades.iterrows():
        trade_time = trade['timestamp']
        trade_side = trade['side']
        trade_size = trade['size']
        
        # Find pre-trade mid price (lookback window)
        pre_mask = (df_quotes['timestamp'] >= trade_time - pd.Timedelta(milliseconds=lookback_window_ms)) & (df_quotes['timestamp'] < trade_time)
        pre_trade_quotes = df_quotes.loc[pre_mask]
        
        if len(pre_trade_quotes) == 0:
            results['trades_skipped'] += 1
            continue
            
        pre_trade_mid = pre_trade_quotes['mid'].iloc[-1]
        
        # First quote after trade within post window
        post_mask = (df_quotes['timestamp'] > trade_time) & (df_quotes['timestamp'] <= trade_time + pd.Timedelta(milliseconds=post_window_ms))
        post_trade_quotes = df_quotes.loc[post_mask]
        
        if len(post_trade_quotes) == 0:
            results['trades_skipped'] += 1
            continue
        
        post_trade_mid = post_trade_quotes['mid'].iloc[0]
        
        if trade_side == 'buy':
            impact = post_trade_mid - pre_trade_mid
            results['buy_impacts'].append({'impact': impact, 'size': trade_size})
        else:
            impact = pre_trade_mid - post_trade_mid
            results['sell_impacts'].append({'impact': impact, 'size': trade_size})
        
        results['trades_analyzed'] += 1
    
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

def run_epsilon_for_crypto(crypto: str, minutes: int = 30, do_plot: bool = False):
    """Run the full epsilon analysis and persistence for a single crypto symbol."""
    log_section(f"EPSILON FROM MARKET DATA - {crypto} (last {minutes} min)")
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
    log_section(f"CALCULATING PRICE IMPACTS FOR {crypto}")

    impact_results = calculate_epsilon(df_quotes, df_trades)

    # Final recommendations and persistence
    log_section(f"EPSILON ESTIMATES (event-level) - {crypto}")

    final_estimates = estimate_epsilon_parameters(impact_results)
    eps_plus = final_estimates['epsilon_buy']['trimmed_mean']
    eps_minus = final_estimates['epsilon_sell']['trimmed_mean']

    print(f"{crypto}: epsilon+={eps_plus:.8f}, epsilon-={eps_minus:.8f}")

    # Check toxicity using kappa from kappa.json (per-crypto)
    try:
        kappa_plus, kappa_minus = load_kappa_from_json(crypto)
        log_section(f"TOXICITY CHECK - {crypto}")
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
        print(f"Toxicity check skipped: {e}")

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

        log_section(f"DELTA ESTIMATES - {crypto}")
        print(f"  mid price [{crypto}]: {mid_price:.6f}")
        print(f"  delta+ = 1/kappa+ + epsilon+ = {delta_plus_price:.8f}  ({delta_plus_pct:.6f}% of mid)")
        print(f"  delta- = 1/kappa- + epsilon- = {delta_minus_price:.8f}  ({delta_minus_pct:.6f}% of mid)")
    except Exception as e:
        print(f"Delta calculation skipped: {e}")

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
            axes[i, 1].plot(x_fit, p(np.sqrt(x_fit)), "k--", alpha=0.5, label='sqrt(size) fit')
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

# Calculate delta +/- = 1/kappa +/- epsilon +/- and express as % of mid-price
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
    parser.add_argument('--crypto', '-c', type=str, default=os.getenv('CRYPTO_NAME', 'ETH'),
                        help='Cryptocurrency symbol (e.g., ETH) or ALL for all available in HL_data')
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
            print("No crypto data found in HL_data (need <SYMBOL>/orderbooks/*.parquet and <SYMBOL>/trades/*.parquet)")
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
