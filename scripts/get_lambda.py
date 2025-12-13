#!/usr/bin/env python3
"""
Calibrate raw trade arrival rates from HL_data trades only.

Definition used here (unconditional):
- lambda+ (buy): number of buy-side market trades per second
- lambda- (sell): number of sell-side market trades per second

These values are intended as a sanity-check / monitoring signal.
Baseline λ₀± for the HJB (used by the strategy) comes from the joint κ/λ₀
regression in `get_kappa.py` and is stored in `lambda.json`.
To avoid overwriting λ₀, this script writes to `lambda_trades.json` by default.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd


def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


@dataclass
class LambdaResults:
    lambda_plus: float
    lambda_minus: float
    lambda_total: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    n_trades_buy: int
    n_trades_sell: int
    n_trades_total: int


def load_trades_only(crypto: str = 'ETH', time_range_minutes: int = 15) -> pd.DataFrame:
    """Load trades for the chosen crypto and filter to the last time_range_minutes from Parquet files."""
    
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trades_dir = os.path.join(script_dir, 'HL_data', crypto, 'trades')
    
    if not os.path.exists(trades_dir):
        raise FileNotFoundError(f"Trades directory not found: {trades_dir}")

    # Load all parquet files in the directory
    files = [os.path.join(trades_dir, f) for f in os.listdir(trades_dir) if f.endswith('.parquet')]
    if not files:
        raise ValueError(f"No parquet files found in {trades_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df_trades = pd.concat(dfs, ignore_index=True)

    if 'timestamp' not in df_trades.columns or 'side' not in df_trades.columns:
        raise ValueError('Expected columns: timestamp, side in trades data')

    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='s')
    most_recent = df_trades['timestamp'].max()
    start = most_recent - pd.Timedelta(minutes=time_range_minutes)

    df_trades = df_trades[(df_trades['timestamp'] >= start) & (df_trades['timestamp'] <= most_recent)].copy()
    df_trades['side'] = df_trades['side'].str.lower()
    return df_trades


def compute_lambda_from_trades(df_trades: pd.DataFrame) -> Optional[LambdaResults]:
    """Compute lambda+ and lambda- as trades per second in the filtered window."""
    if df_trades.empty:
        return None

    start_time = df_trades['timestamp'].min()
    end_time = df_trades['timestamp'].max()
    total_seconds = max((end_time - start_time).total_seconds(), 1e-6)

    n_buy = int((df_trades['side'] == 'buy').sum())
    n_sell = int((df_trades['side'] == 'sell').sum())
    n_total = int(len(df_trades))

    lam_plus = n_buy / total_seconds
    lam_minus = n_sell / total_seconds
    lam_total = n_total / total_seconds

    return LambdaResults(
        lambda_plus=lam_plus,
        lambda_minus=lam_minus,
        lambda_total=lam_total,
        start_time=start_time,
        end_time=end_time,
        n_trades_buy=n_buy,
        n_trades_sell=n_sell,
        n_trades_total=n_total,
    )


def save_lambda_to_json(lam_plus: float, lam_minus: float, crypto: str, filename: str = "lambda_trades.json"):
    """Save raw trade lambda estimates (per second) to JSON, overwriting or appending the symbol entry."""
    # Prepare values as plain floats
    lam_plus_val = float(lam_plus) if pd.notna(lam_plus) else None
    lam_minus_val = float(lam_minus) if pd.notna(lam_minus) else None

    # Load existing file if present
    data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}

    # Update
    data[crypto] = {
        "lambda+": lam_plus_val,
        "lambda-": lam_minus_val,
        "unit": "trades_per_second"
    }

    # Write back
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"[save] lambda -> {filename}")
    print(f"[save] {crypto}: lambda+={lam_plus_val}, lambda-={lam_minus_val}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate raw trade lambda (trades/sec) from HL_data trades only')
    parser.add_argument('--crypto', '-c', type=str, default='ETH', help='Crypto symbol (default ETH)')
    parser.add_argument('--minutes', '-m', type=int, default=30, help='Minutes from most recent to analyze')
    parser.add_argument('--output', '-o', type=str, default='lambda_trades.json',
                        help='Output JSON filename (default lambda_trades.json)')
    args = parser.parse_args()

    log_section(f'LAMBDA FROM TRADES - {args.crypto} (last {args.minutes} min)')

    try:
        df_trades = load_trades_only(args.crypto, args.minutes)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(script_dir, 'HL_data')
            print('Available symbols with trade data in HL_data:')
            candidates = [
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
            valid = []
            for sym in sorted(candidates):
                trades_dir = os.path.join(base_dir, sym, 'trades')
                if os.path.isdir(trades_dir) and any(name.endswith('.parquet') for name in os.listdir(trades_dir)):
                    valid.append(sym)
            if valid:
                for sym in valid:
                    print(f'  - {sym}')
            else:
                print('  (none found)')
        except Exception:
            pass
        return
    except ValueError as e:
        print(f'Error: {e}')
        return

    if df_trades.empty:
        print('No trades found in the selected window.')
        return

    results = compute_lambda_from_trades(df_trades)
    if results is None:
        print('Unable to compute lambda (no data after filtering).')
        return

    duration_min = (results.end_time - results.start_time).total_seconds() / 60.0

    log_section('LAMBDA ESTIMATES (trades per second)')
    print(f"Window: {results.start_time} -> {results.end_time}  ({duration_min:.2f} min)")
    print(f"Trades: total={results.n_trades_total}, buy={results.n_trades_buy}, sell={results.n_trades_sell}")
    print(f"lambda+ (buy):  {results.lambda_plus:.6f} trades/sec")
    print(f"lambda- (sell): {results.lambda_minus:.6f} trades/sec")
    print(f"lambda (total): {results.lambda_total:.6f} trades/sec")

    # Save to monitoring file (does not overwrite lambda.json baseline)
    save_lambda_to_json(results.lambda_plus, results.lambda_minus, args.crypto, filename=args.output)


if __name__ == '__main__':
    main()
