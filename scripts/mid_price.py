import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

def find_latest_bid_ask_parquet(prices_dir: Path, max_files: int = 10) -> Optional[Tuple[float, float]]:
    """
    Scan recent Parquet shards and return the most recent (bid, ask) prices.

    Expected columns (from hyperliquid_data_collector.py):
    - timestamp (float seconds)
    - price (float)
    - side ("bid"/"ask")
    """
    latest_bid: Optional[float] = None
    latest_ask: Optional[float] = None

    files = sorted(prices_dir.glob("prices_*.parquet"), key=lambda p: p.name, reverse=True)
    for parquet_path in files[: max(1, int(max_files))]:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            continue

        if df is None or df.empty:
            continue
        if not {"timestamp", "price", "side"}.issubset(df.columns):
            continue

        df = df[["timestamp", "price", "side"]].copy()
        df["side"] = df["side"].astype(str).str.lower()

        if latest_bid is None:
            bids = df[df["side"] == "bid"]
            if not bids.empty:
                latest_bid = float(bids.loc[bids["timestamp"].idxmax(), "price"])

        if latest_ask is None:
            asks = df[df["side"] == "ask"]
            if not asks.empty:
                latest_ask = float(asks.loc[asks["timestamp"].idxmax(), "price"])

        if latest_bid is not None and latest_ask is not None:
            return latest_bid, latest_ask

    return None


def compute_mid_price_from_prices_dir(prices_dir: Path) -> Optional[float]:
    pair = find_latest_bid_ask_parquet(prices_dir)
    if pair is None:
        return None
    bid, ask = pair
    return (bid + ask) / 2.0


def load_existing_json(json_path: Path) -> Dict[str, float]:
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure it is a dict of str -> number
        if isinstance(data, dict):
            clean: Dict[str, float] = {}
            for k, v in data.items():
                try:
                    clean[str(k)] = float(v)
                except Exception:
                    continue
            return clean
        return {}
    except Exception:
        return {}


def main() -> None:
    root = Path(__file__).resolve().parent
    hl_dir = root / "HL_data"
    json_path = root / "mid_price.json"

    output = load_existing_json(json_path)

    if not hl_dir.is_dir():
        print("HL_data directory not found (expected scripts/HL_data).")
        return

    symbol_dirs = sorted([p for p in hl_dir.iterdir() if p.is_dir()])
    if not symbol_dirs:
        print("No symbol folders found in HL_data (expected HL_data/<SYMBOL>/prices/*.parquet)")
        return

    for sym_dir in symbol_dirs:
        prices_dir = sym_dir / "prices"
        if not prices_dir.is_dir():
            continue
        mid = compute_mid_price_from_prices_dir(prices_dir)
        if mid is None:
            continue
        output[sym_dir.name] = mid

    # Persist results
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    # Optional: print summary to stdout
    for k, v in output.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
