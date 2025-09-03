#!/usr/bin/env python3
"""
Calibrate lambda (order arrival intensity) from HL_data trades only.

Definition used here:
- lambda+ (buy): number of buy-side market trades per minute
- lambda- (sell): number of sell-side market trades per minute

No quotes/orderbook, kappa, or epsilon are used in this script.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd


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


def load_trades_only(crypto: str = 'WLFI', time_range_minutes: int = 15) -> pd.DataFrame:
    """Load trades for the chosen crypto and filter to the last time_range_minutes."""
    trades_file = f'HL_data/trades_{crypto}.csv'
    if not os.path.exists(trades_file):
        raise FileNotFoundError(f"Trades file not found: {trades_file}")

    df_trades = pd.read_csv(trades_file)
    if 'timestamp' not in df_trades.columns or 'side' not in df_trades.columns:
        raise ValueError('Expected columns: timestamp, side in trades CSV')

    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='s')
    most_recent = df_trades['timestamp'].max()
    start = most_recent - pd.Timedelta(minutes=time_range_minutes)

    df_trades = df_trades[(df_trades['timestamp'] >= start) & (df_trades['timestamp'] <= most_recent)].copy()
    df_trades['side'] = df_trades['side'].str.lower()
    return df_trades


def compute_lambda_from_trades(df_trades: pd.DataFrame) -> Optional[LambdaResults]:
    """Compute λ+ and λ- as trades per minute in the filtered window."""
    if df_trades.empty:
        return None

    start_time = df_trades['timestamp'].min()
    end_time = df_trades['timestamp'].max()
    total_minutes = max((end_time - start_time).total_seconds() / 60.0, 1e-6)

    n_buy = int((df_trades['side'] == 'buy').sum())
    n_sell = int((df_trades['side'] == 'sell').sum())
    n_total = int(len(df_trades))

    lam_plus = n_buy / total_minutes
    lam_minus = n_sell / total_minutes
    lam_total = n_total / total_minutes

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


def save_lambda_to_json(lam_plus: float, lam_minus: float, crypto: str, filename: str = "lambda.json"):
    """Save lambda estimates to lambda.json, overwriting or appending the symbol entry."""
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
    }

    # Write back
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"\nLambda estimates saved to {filename}")
    print(f"  {crypto}: lambda+ = {lam_plus_val}, lambda- = {lam_minus_val}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate lambda (trades/min) from HL_data trades only')
    parser.add_argument('--crypto', '-c', type=str, default='WLFI', help='Crypto symbol (BTC, ETH, SOL, WLFI, etc.)')
    parser.add_argument('--minutes', '-m', type=int, default=30, help='Minutes from most recent to analyze')
    args = parser.parse_args()

    print('=' * 60)
    print(f'LAMBDA FROM TRADES - {args.crypto} (last {args.minutes} min)')
    print('=' * 60)

    try:
        df_trades = load_trades_only(args.crypto, args.minutes)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print('Available trade files in HL_data:')
        for f in os.listdir('HL_data'):
            if f.startswith('trades_') and f.endswith('.csv'):
                print(f'  - {f}')
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

    print('\n' + '=' * 60)
    print('LAMBDA ESTIMATES (trades per minute)')
    print('=' * 60)
    print(f"Window: {results.start_time} -> {results.end_time}  ({duration_min:.2f} min)")
    print(f"Trades: total={results.n_trades_total}, buy={results.n_trades_buy}, sell={results.n_trades_sell}")
    print(f"lambda+ (buy):  {results.lambda_plus:.6f} trades/min")
    print(f"lambda- (sell): {results.lambda_minus:.6f} trades/min")
    print(f"lambda (total): {results.lambda_total:.6f} trades/min")

    # Save to lambda.json
    save_lambda_to_json(results.lambda_plus, results.lambda_minus, args.crypto)


if __name__ == '__main__':
    main()
