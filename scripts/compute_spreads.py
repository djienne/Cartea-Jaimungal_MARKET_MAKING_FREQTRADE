#!/usr/bin/env python3
"""
Compute bid/ask spreads (bps from mid) using the same logic as user_data/strategies/Market_Making.py.

Inputs:
- Refreshes κ/ε and baseline λ₀ by calling get_kappa.py and get_epsilon.py.
- Also runs get_lambda.py to compute unconditional trade rates into lambda_trades.json (monitoring only).
- kappa.json, epsilon.json, lambda.json (baseline λ₀; expected in working directory or parent)
- Mid price (via --mid) or fallback to mid_price.json if present, else 1.0
- Inventory level q (optional, default 0)

The script:
1. Loads κ, ε, and per-side MO arrival rates λ for the symbol.
2. Runs the HJB solver (symmetric closed-form by default, optional asymmetric-κ backward Euler) to get δ* with inventory skew.
3. Assembles the final half-spread identically to the strategy:
   delta_total = clamp(δ* × spread_multiplier + fee × mid, min/max half-spread bps).
4. Prints bid/ask prices and spreads in bps from mid, with (floor)/(cap) markers when clamped.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import subprocess
import sys

import numpy as np

from hjb import compute_h_symmetric, compute_h_asymmetric
from replay_market_maker import MAX_HALF_SPREAD_BPS, MIN_HALF_SPREAD_BPS, assemble_half_spread


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


def param_metadata_lines(symbol: str, *sources: dict) -> list[str]:
    lines: list[str] = []
    now = datetime.now(timezone.utc)
    for name, source in zip(("kappa", "epsilon", "lambda"), sources):
        entry = source.get(symbol, {}) if isinstance(source, dict) else {}
        if not isinstance(entry, dict):
            continue
        schema = entry.get("schema_version", "v1_or_missing")
        status = entry.get("status", "missing")
        generated_at = entry.get("generated_at")
        age_label = "unknown"
        if generated_at:
            try:
                parsed = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_label = f"{(now - parsed.astimezone(timezone.utc)).total_seconds():.1f}s"
            except Exception:
                age_label = "invalid_timestamp"
        extra = ""
        if name == "lambda":
            extra = f", lambda_source={entry.get('lambda_source', 'unknown')}"
        lines.append(f"{name}: schema={schema}, status={status}, generated_at={generated_at}, age={age_label}{extra}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Compute bid/ask spreads (bps from mid) using HJB deltas.")
    parser.add_argument("--crypto", "-c", default="CASHCAT", help="Symbol key in JSON files (default CASHCAT)")
    parser.add_argument("--mid", "-m", type=float, default=None, help="Mid price to evaluate (falls back to mid_price.json or 1.0)")
    parser.add_argument("--inventory", "-q", type=int, default=0, help="Inventory level q (clipped to HJB grid)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Terminal inventory penalty (alpha)")
    parser.add_argument("--phi", type=float, default=0.0, help="Running inventory penalty (phi)")
    # Bounded deliberately. compute_h_symmetric reconstructs omega densely and
    # floors it at 1e-300 before the log; on a very wide grid the smallest
    # entries are pure roundoff (possibly negative), get floored to log ~ -690,
    # and the corner depths come back wrong by ~0.1 with no error raised. The
    # live path cannot reach this (it forces the asymmetric solver at q_max=6),
    # so this flag is the only way in -- cap it rather than silently mislead.
    parser.add_argument(
        "--qmax",
        type=int,
        default=3,
        choices=range(1, 21),
        metavar="{1..20}",
        help="Inventory grid radius (q_max), 1-20",
    )
    parser.add_argument("--horizon", type=float, default=60.0, help="Horizon in seconds for HJB (default 60s, tuned for λ in trades/sec)")
    parser.add_argument("--asym-kappa", action="store_true", help="Use asymmetric-κ backward-Euler solver instead of symmetric closed form")
    parser.add_argument("--steps", type=int, default=200, help="Time steps for asymmetric solver (ignored for symmetric)")
    parser.add_argument("--minutes", "-t", type=int, default=30, help="Minutes of data to use when refreshing κ/ε/λ")
    parser.add_argument("--skip-refresh", action="store_true", help="Use existing JSON params without running estimator scripts first")
    parser.add_argument("--spread-multiplier", type=float, default=1.0,
                        help="Scales the HJB model depth only (fee added separately); pass 3.0 to mirror the production config")
    parser.add_argument("--min-half-spread-bps", type=float, default=MIN_HALF_SPREAD_BPS)
    parser.add_argument("--max-half-spread-bps", type=float, default=MAX_HALF_SPREAD_BPS)
    args = parser.parse_args()

    start_dir = Path(__file__).resolve().parent

    # Refresh κ/ε/λ
    if not args.skip_refresh:
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

    if args.asym_kappa:
        hjb_res = compute_h_asymmetric(
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
            n_steps=args.steps,
        )
    else:
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

    print(f"Symbol: {sym}")
    print(f"Mid: {mid:.8f}")
    print(f"Inventory grid: q in [-{args.qmax}, {args.qmax}]")
    print(f"Parameters: kappa+={kappa_p}, kappa-={kappa_m}, epsilon+={eps_p}, epsilon-={eps_m}, lambda+={lam_p}, lambda-={lam_m}")
    print(
        f"Assembly: multiplier={args.spread_multiplier}, fee={MAKER_FEE} per side, "
        f"clamps=[{args.min_half_spread_bps}, {args.max_half_spread_bps}] bps"
    )
    metadata = param_metadata_lines(sym, kappa, epsilon, lambdas)
    if metadata:
        print("Parameter metadata:")
        for line in metadata:
            print(f"  {line}")
    print("\nq\tbid_px\t\task_px\t\tbid_bps\t\task_bps")

    def render_side(delta_model: float, side_sign: float) -> tuple[str, str]:
        delta_total = assemble_half_spread(
            delta_model,
            mid,
            spread_multiplier=args.spread_multiplier,
            maker_fee=MAKER_FEE,
            min_half_spread_bps=args.min_half_spread_bps,
            max_half_spread_bps=args.max_half_spread_bps,
        )
        if delta_total is None:
            return "DISABLED", "DISABLED"
        pre_clamp = delta_model * args.spread_multiplier + MAKER_FEE * mid
        marker = ""
        if pre_clamp < delta_total:
            marker = " (floor)"
        elif pre_clamp > delta_total:
            marker = " (cap)"
        px = f"{mid + side_sign * delta_total:.8f}"
        bps = f"{(delta_total / mid) * 1e4:.4f}{marker}"
        return px, bps

    for q in range(-args.qmax, args.qmax + 1):
        delta_bid_model = select_delta_from_hjb(hjb_res, "bid", q, args.qmax)
        delta_ask_model = select_delta_from_hjb(hjb_res, "ask", q, args.qmax)

        bid_px, bid_bps = render_side(delta_bid_model, -1.0)
        ask_px, ask_bps = render_side(delta_ask_model, +1.0)

        print(f"{q:+d}\t{bid_px}\t{ask_px}\t{bid_bps}\t\t{ask_bps}")

    print("\nBoundary check:")
    print(f"q = -{args.qmax} => ask {'DISABLED' if not np.isfinite(select_delta_from_hjb(hjb_res, 'ask', -args.qmax, args.qmax)) else 'ENABLED'}")
    print(f"q = +{args.qmax} => bid {'DISABLED' if not np.isfinite(select_delta_from_hjb(hjb_res, 'bid', args.qmax, args.qmax)) else 'ENABLED'}")


if __name__ == "__main__":
    main()
