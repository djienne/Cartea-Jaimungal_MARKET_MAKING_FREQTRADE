#!/usr/bin/env python3
"""
Compute bid/ask spreads (bps from mid) using the same logic as user_data/strategies/Market_Making.py.

Inputs:
- Refreshes κ/ε/λ by calling get_kappa.py, get_epsilon.py, and get_lambda.py before computing spreads.
- kappa.json, epsilon.json, lambda.json (expected in working directory or parent)
- Mid price (via --mid) or fallback to mid_price.json if present, else 1.0
- Inventory level q (optional, default 0)

The script:
1. Loads κ, ε, λ for the symbol.
2. Runs the symmetric-κ HJB solver (from scripts/hjb.py) to get δ* with inventory skew.
3. Adds maker-fee cushion identical to the strategy (0.015% maker fee → fee * mid * 2).
4. Prints bid/ask prices and spreads in bps from mid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import subprocess
import sys

import numpy as np

from hjb import compute_h_symmetric


MAKER_FEE = 0.0150 / 100.0  # 1.5 bps as fraction


def find_upwards(filename: str, start: Path, max_up: int = 5) -> Optional[Path]:
    p = start.resolve()
    for _ in range(max_up + 1):
        candidate = p / filename
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


def load_json(name: str, start_dir: Path) -> dict:
    path = find_upwards(name, start_dir)
    if not path:
        raise FileNotFoundError(f"Could not find {name} upward from {start_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_delta_from_hjb(hjb_res: dict, side: str, q: int, q_max: int) -> float:
    q_grid = hjb_res["q_grid"]
    q = max(-q_max, min(q_max, q))
    if q < q_grid[0]:
        idx = 0
    elif q > q_grid[-1]:
        idx = -1
    else:
        idx = int(np.argmin(np.abs(q_grid - q)))
    if side == "bid":
        return float(hjb_res["delta_minus"][idx])
    return float(hjb_res["delta_plus"][idx])


def load_mid_price(symbol: str, start_dir: Path) -> Optional[float]:
    """Try to load a mid price from mid_price.json; return None on failure."""
    path = find_upwards("mid_price.json", start_dir)
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get(symbol)
        return float(val) if val is not None else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compute bid/ask spreads (bps from mid) using HJB deltas.")
    parser.add_argument("--crypto", "-c", default="ETH", help="Symbol key in JSON files (default ETH)")
    parser.add_argument("--mid", "-m", type=float, default=None, help="Mid price to evaluate (falls back to mid_price.json or 1.0)")
    parser.add_argument("--inventory", "-q", type=int, default=0, help="Inventory level q (clipped to HJB grid)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Terminal inventory penalty (alpha)")
    parser.add_argument("--phi", type=float, default=0.0, help="Running inventory penalty (phi)")
    parser.add_argument("--qmax", type=int, default=3, help="Inventory grid radius (q_max)")
    parser.add_argument("--horizon", type=float, default=60.0, help="Horizon in seconds for HJB (default 60s, tuned for λ in trades/sec)")
    parser.add_argument("--minutes", "-t", type=int, default=30, help="Minutes of data to use when refreshing κ/ε/λ")
    args = parser.parse_args()

    start_dir = Path(__file__).resolve().parent

    # Refresh κ/ε/λ
    for script_name in ("get_kappa.py", "get_epsilon.py", "get_lambda.py"):
        script_path = start_dir / script_name
        if not script_path.exists():
            raise SystemExit(f"Required script not found: {script_path}")
        cmd = [
            sys.executable,
            str(script_path),
            "--crypto",
            args.crypto,
            "--minutes",
            str(args.minutes),
        ]
        result = subprocess.run(cmd, cwd=start_dir)
        if result.returncode != 0:
            raise SystemExit(f"{script_name} failed with exit code {result.returncode}")
    kappa = load_json("kappa.json", start_dir)
    epsilon = load_json("epsilon.json", start_dir)
    lambdas = load_json("lambda.json", start_dir)

    sym = args.crypto
    try:
        kappa_p = float(kappa[sym]["kappa+"])
        kappa_m = float(kappa[sym]["kappa-"])
        eps_p = float(epsilon[sym]["epsilon+"])
        eps_m = float(epsilon[sym]["epsilon-"])
        lam_p = float(lambdas.get(sym, {}).get("lambda+", 0.0))
        lam_m = float(lambdas.get(sym, {}).get("lambda-", 0.0))
    except Exception as e:
        raise SystemExit(f"Missing parameters for {sym}: {e}")

    hjb_res = compute_h_symmetric(
        lambda_plus=lam_p,
        lambda_minus=lam_m,
        epsilon_plus=eps_p,
        epsilon_minus=eps_m,
        kappa_plus=kappa_p,
        kappa_minus=kappa_m,
        alpha=args.alpha,
        phi=args.phi,
        T_seconds=args.horizon,
        q_max=args.qmax,
    )

    # Resolve mid price (arg -> mid_price.json -> 1.0)
    mid = args.mid
    if mid is None:
        mid = load_mid_price(sym, start_dir)
    if mid is None:
        mid = 1.0

    fee_cushion = MAKER_FEE * mid * 2.0

    print(f"Symbol: {sym}")
    print(f"Mid: {mid:.8f}")
    print(f"Inventory grid: q in [-{args.qmax}, {args.qmax}]")
    print(f"Parameters: kappa+={kappa_p}, kappa-={kappa_m}, epsilon+={eps_p}, epsilon-={eps_m}, lambda+={lam_p}, lambda-={lam_m}")
    print("\nq\tbid_px\t\task_px\t\tbid_bps\t\task_bps")

    for q in range(-args.qmax, args.qmax + 1):
        delta_bid = select_delta_from_hjb(hjb_res, "bid", q, args.qmax) + fee_cushion
        delta_ask = select_delta_from_hjb(hjb_res, "ask", q, args.qmax) + fee_cushion
        bid_px = mid - delta_bid
        ask_px = mid + delta_ask
        bid_bps = (delta_bid / mid) * 1e4
        ask_bps = (delta_ask / mid) * 1e4
        print(f"{q:+d}\t{bid_px:.8f}\t{ask_px:.8f}\t{bid_bps:.4f}\t\t{ask_bps:.4f}")


if __name__ == "__main__":
    main()
