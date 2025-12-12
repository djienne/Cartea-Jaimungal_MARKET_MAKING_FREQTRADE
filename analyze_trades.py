import sqlite3
import pandas as pd
import datetime
import argparse
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Default DB path, can be overridden by arguments
DEFAULT_DB_PATH = 'user_data/tradesv3.sqlite'

def color_pnl(val):
    """Returns a string with color codes based on the value."""
    try:
        if isinstance(val, str):
            # Try removing % or currency symbols if present
            clean_val = val.replace('%', '').replace('USDC', '').strip()
            v = float(clean_val)
        else:
            v = float(val)

        if v > 0:
            return f"{Fore.GREEN}{val}{Style.RESET_ALL}"
        elif v < 0:
            return f"{Fore.RED}{val}{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{val}{Style.RESET_ALL}"
    except ValueError:
        return val

def format_duration(td):
    """Formats a timedelta object into a readable string."""
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts) if parts else "0s"

def analyze_trades(db_path):
    try:
        conn = sqlite3.connect(db_path)
        
        # Load trades into a DataFrame
        query = "SELECT * FROM trades"
        df = pd.read_sql_query(query, conn)
        
        conn.close()

        if df.empty:
            print(f"{Fore.YELLOW}No trades found in the database at {db_path}.{Style.RESET_ALL}")
            return

        # Pre-processing
        df['open_date'] = pd.to_datetime(df['open_date'])
        df['close_date'] = pd.to_datetime(df['close_date'])
        
        # Filter for closed trades for PnL analysis
        closed_trades = df[df['is_open'] == 0].copy()
        open_trades = df[df['is_open'] == 1].copy()
        
        # Determine currency
        currency = df['stake_currency'].iloc[0] if 'stake_currency' in df.columns else 'Units'

        print(f"\n{Style.BRIGHT}========================================{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}       FREQTRADE DATA ANALYSIS          {Style.RESET_ALL}")
        print(f"{Style.BRIGHT}========================================{Style.RESET_ALL}")
        print(f"Database: {db_path}\n")

        # --- Summary Statistics ---
        summary_data = [
            ["Total Trades", len(df)],
            ["Closed Trades", len(closed_trades)],
            ["Open Trades", len(open_trades)]
        ]

        if not closed_trades.empty:
            total_profit_abs = closed_trades['close_profit_abs'].sum()
            total_profit_pct = closed_trades['close_profit'].sum() * 100
            
            winning_trades = closed_trades[closed_trades['close_profit_abs'] > 0]
            losing_trades = closed_trades[closed_trades['close_profit_abs'] <= 0]
            win_rate = (len(winning_trades) / len(closed_trades)) * 100
            
            closed_trades['duration'] = closed_trades['close_date'] - closed_trades['open_date']
            avg_duration = closed_trades['duration'].mean()
            
            best_trade = closed_trades.loc[closed_trades['close_profit_abs'].idxmax()]
            worst_trade = closed_trades.loc[closed_trades['close_profit_abs'].idxmin()]

            summary_data.extend([
                ["Total PnL (Abs)", color_pnl(f"{total_profit_abs:.2f} {currency}")],
                ["Total Profit %", color_pnl(f"{total_profit_pct:.2f}%")],
                ["Win Rate", f"{win_rate:.2f}% ({len(winning_trades)} W / {len(losing_trades)} L)"],
                ["Avg Duration", format_duration(avg_duration)]
            ])
        
        print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

        if not closed_trades.empty:
            # --- Best & Worst Trades ---
            print(f"\n{Style.BRIGHT}--- Extremes ---{Style.RESET_ALL}")
            extremes_data = [
                ["Best Trade", best_trade['pair'], color_pnl(f"{best_trade['close_profit_abs']:.2f}"), str(best_trade['close_date'])],
                ["Worst Trade", worst_trade['pair'], color_pnl(f"{worst_trade['close_profit_abs']:.2f}"), str(worst_trade['close_date'])]
            ]
            print(tabulate(extremes_data, headers=["Type", "Pair", f"Profit ({currency})", "Close Date"], tablefmt="simple"))

            # --- PnL by Pair ---
            print(f"\n{Style.BRIGHT}--- Performance by Pair ---{Style.RESET_ALL}")
            pair_stats = closed_trades.groupby('pair').agg({
                'close_profit_abs': 'sum',
                'id': 'count',
                'close_profit': 'mean' # Average profit % per trade
            }).rename(columns={'id': 'Count', 'close_profit_abs': 'Total Profit', 'close_profit': 'Avg Profit %'})
            
            pair_stats = pair_stats.sort_values(by='Total Profit', ascending=False)
            
            # Format columns for display
            pair_table = []
            for pair, row in pair_stats.iterrows():
                pair_table.append([
                    pair, 
                    row['Count'], 
                    color_pnl(f"{row['Total Profit']:.2f}"),
                    color_pnl(f"{row['Avg Profit %']*100:.2f}%")
                ])
                
            print(tabulate(pair_table, headers=["Pair", "Count", f"Total Profit ({currency})", "Avg Profit %"], tablefmt="fancy_grid"))
            
            # --- PnL by Strategy ---
            print(f"\n{Style.BRIGHT}--- Performance by Strategy ---{Style.RESET_ALL}")
            strategy_stats = closed_trades.groupby('strategy').agg({
                'close_profit_abs': 'sum',
                'id': 'count'
            }).rename(columns={'id': 'Count', 'close_profit_abs': 'Total Profit'}).sort_values(by='Total Profit', ascending=False)
            
            strategy_table = []
            for strat, row in strategy_stats.iterrows():
                strategy_table.append([strat, row['Count'], color_pnl(f"{row['Total Profit']:.2f}")])
            
            print(tabulate(strategy_table, headers=["Strategy", "Count", f"Total Profit ({currency})"], tablefmt="fancy_grid"))

        # --- Open Trades ---
        if not open_trades.empty:
            print(f"\n{Style.BRIGHT}--- Open Trades ---{Style.RESET_ALL}")
            columns_to_show = ['id', 'pair', 'open_date', 'stake_amount', 'open_rate', 'is_short']
            # Map column names to nicer headers
            header_map = {
                'id': 'ID', 'pair': 'Pair', 'open_date': 'Open Date', 
                'stake_amount': 'Stake', 'open_rate': 'Open Rate', 'is_short': 'Short?'
            }
            
            available_cols = [c for c in columns_to_show if c in open_trades.columns]
            display_df = open_trades[available_cols].copy()
            
            # Format specific columns if needed
            display_df['is_short'] = display_df['is_short'].apply(lambda x: "Yes" if x else "No")
            display_df['stake_amount'] = display_df['stake_amount'].apply(lambda x: f"{x:.2f}")
            display_df['open_rate'] = display_df['open_rate'].apply(lambda x: f"{x:.4f}")
            
            headers = [header_map.get(c, c) for c in available_cols]
            
            print(tabulate(display_df, headers=headers, tablefmt="fancy_grid", showindex=False))
        else:
             print(f"\n{Fore.CYAN}No open trades.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Freqtrade sqlite database.")
    parser.add_argument("db_path", nargs='?', default=DEFAULT_DB_PATH, help="Path to the sqlite database file")
    args = parser.parse_args()
    
    analyze_trades(args.db_path)