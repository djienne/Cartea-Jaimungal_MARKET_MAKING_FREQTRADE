import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def find_latest_bid_ask(csv_path: Path) -> Optional[Tuple[float, float]]:
    """
    Scan the CSV from the end and return the most recent (bid, ask) prices.
    Assumes columns: timestamp,price,size,side,exchange_timestamp
    Returns None if one side is missing.
    """
    try:
        # Read all lines and iterate in reverse for simplicity and robustness
        lines = csv_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except Exception:
        # Fall back to system default encoding if utf-8 fails
        try:
            lines = csv_path.read_text().splitlines()
        except Exception:
            return None

    if not lines:
        return None

    # Skip header if present
    start_idx = 1 if lines[0].lower().startswith("timestamp,") else 0
    if start_idx >= len(lines):
        return None

    latest_bid: Optional[float] = None
    latest_ask: Optional[float] = None

    # Traverse from end to start to find the freshest entries
    for line in reversed(lines[start_idx:]):
        # Some lines may be empty; skip safely
        if not line:
            continue
        try:
            # Manual split is fine (no embedded commas expected in schema)
            parts = line.split(",")
            # Expected columns: timestamp, price, size, side, exchange_timestamp
            if len(parts) < 4:
                continue
            price_str = parts[1]
            side = parts[3].strip().lower()
            price = float(price_str)

            if side == "bid" and latest_bid is None:
                latest_bid = price
            elif side == "ask" and latest_ask is None:
                latest_ask = price

            if latest_bid is not None and latest_ask is not None:
                return latest_bid, latest_ask
        except Exception:
            # Skip malformed lines
            continue

    return None


def compute_mid_price(csv_path: Path) -> Optional[float]:
    pair = find_latest_bid_ask(csv_path)
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

    prices_files = sorted(hl_dir.glob("prices_*.csv"))
    if not prices_files:
        print("No price files found in HL_data (pattern prices_*.csv)")
        return

    output = load_existing_json(json_path)

    for csv_path in prices_files:
        symbol = csv_path.stem.replace("prices_", "", 1)
        mid = compute_mid_price(csv_path)
        if mid is None:
            # Skip symbols where a mid can't be computed
            continue
        # Overwrite or append
        output[symbol] = mid

    # Persist results
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    # Optional: print summary to stdout
    for k, v in output.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

