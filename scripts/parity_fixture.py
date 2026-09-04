#!/usr/bin/env python3
"""Regenerate the golden numbers pinned by ``rust_live/tests/python_parity.rs``.

The Rust test builds a deterministic market window in-process and compares
its calibration, HJB surface and final quote prices against literals that
were produced once by the Python reference modules. This script rebuilds the
same window here and prints those literals from the untouched oracles
(``estimator_common.py``, ``get_epsilon.py``, ``hjb.py`` via ``mm_core.py``),
so the goldens can be regenerated whenever the shared semantics change.

The fixture MUST stay byte-for-byte equivalent to
``deterministic_python_fixture()`` in the Rust test; the two are the contract.

Usage (from the repository root)::

    python scripts/parity_fixture.py            # current semantics (v5)
    python scripts/parity_fixture.py --unscaled # pre-v5 lambda, self-check

``--unscaled`` reproduces the schema-v4 goldens (lambda not scaled by the
survival intercept) and exists so the script can be checked against the
literals that shipped before the change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimator_common import (  # noqa: E402
    aggregate_market_orders,
    attach_pre_mid,
    fit_kappa_survival,
    mo_depths,
    realized_sigma2_per_sec,
)
from get_epsilon import _floored_mean_estimates, compute_mo_impacts  # noqa: E402
from mm_core import QuoteConfig, compute_quotes, select_delta, solve_hjb  # noqa: E402

WINDOW_END_MS = 200_000.0
EPSILON_HORIZON_MS = 200
INVENTORY_UNIT_BASE = 1_868.0
PRICE_TICK = 1.0e-5  # 5 significant figures at a 0.13 price, 6 max decimals


def fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror of ``deterministic_python_fixture()`` in the Rust parity test."""
    mids = []
    for index in range(0, 2_001):
        ts_ms = index * 100.0
        mid = 0.132 + 0.000_05 * math.sin(index / 37.0) + 0.000_000_02 * index
        mids.append({"ts_ms": ts_ms, "bid": mid - 0.000_01, "ask": mid + 0.000_01, "mid": mid})
    mids = pd.DataFrame(mids)
    mid_values = mids["mid"].to_numpy()
    trades = []
    for index in range(1, 400):
        ts_ms = index * 500.0 + 50.0
        pre_index = int(math.floor((ts_ms - 1.0) / 100.0))
        pre_mid = mid_values[pre_index]
        buy = index % 2 == 0
        u = ((index * 73) % 397) / 398.0 + 0.5 / 398.0
        depth = -math.log(max(1.0 - u, 1.0e-12)) / 9_000.0
        trades.append(
            {
                "ts_ms": ts_ms,
                "side": "buy" if buy else "sell",
                "price": pre_mid + depth if buy else pre_mid - depth,
                "size": float(10 + index % 7),
                "trade_id": str(index),
            }
        )
    return mids, pd.DataFrame(trades)


def calibrate(unscaled: bool) -> dict:
    mids, trades = fixture()
    mos = attach_pre_mid(aggregate_market_orders(trades), mids)
    depths = mo_depths(mos)
    plus = fit_kappa_survival(depths["buy_depths"], 0.99, 0.0)
    minus = fit_kappa_survival(depths["sell_depths"], 0.99, 0.0)
    covered_seconds = (WINDOW_END_MS - max(0.0, float(mids["ts_ms"].min()))) / 1000.0
    n_plus = int((mos["side"] == "buy").sum())
    n_minus = int((mos["side"] == "sell").sum())
    lambda_plus_raw = n_plus / covered_seconds
    lambda_minus_raw = n_minus / covered_seconds
    intercept_plus = float(plus["survival_intercept"])
    intercept_minus = float(minus["survival_intercept"])
    if unscaled:
        lambda_plus, lambda_minus = lambda_plus_raw, lambda_minus_raw
    else:
        lambda_plus = lambda_plus_raw * intercept_plus
        lambda_minus = lambda_minus_raw * intercept_minus
    impacts = compute_mo_impacts(mos, mids, EPSILON_HORIZON_MS, WINDOW_END_MS)
    epsilon_plus, epsilon_minus, estimates = _floored_mean_estimates(impacts)
    sigma2 = realized_sigma2_per_sec(mids)
    return {
        "kappa_plus": float(plus["kappa"]),
        "kappa_minus": float(minus["kappa"]),
        "lambda_plus": float(lambda_plus),
        "lambda_minus": float(lambda_minus),
        "lambda_plus_raw": float(lambda_plus_raw),
        "lambda_minus_raw": float(lambda_minus_raw),
        "survival_intercept_plus": intercept_plus,
        "survival_intercept_minus": intercept_minus,
        "epsilon_plus": float(epsilon_plus),
        "epsilon_minus": float(epsilon_minus),
        "sigma2_per_second": float(sigma2),
        "market_orders_plus": n_plus,
        "market_orders_minus": n_minus,
        "n_points_plus": int(plus["n_points"]),
        "n_points_minus": int(minus["n_points"]),
        "epsilon_events_plus": int(estimates["epsilon_buy"]["n_trades"]),
        "epsilon_events_minus": int(estimates["epsilon_sell"]["n_trades"]),
        "r_squared_plus": float(plus["r_squared"]),
        "r_squared_minus": float(minus["r_squared"]),
    }


def model(parameters: dict) -> dict:
    config = QuoteConfig(
        inventory_unit_base=INVENTORY_UNIT_BASE,
        q_max=6,
        # FROZEN FIXTURE VALUES, not the shipped ones -- config/cashcat.toml
        # moved to 400/600 on 2026-09-02 and this pair deliberately did not.
        # tests/python_parity.rs::parity_model_config holds the same two and
        # explains why: a golden that tracks a tunable turns every retune into a
        # parity failure, and regenerating it is how a real drift slips through.
        # Change both sides together or neither.
        hjb_phi_kappa_t=200.0,
        hjb_phi_kappa_t_max=300.0,
        hjb_alpha_kappa=0.05,
        hjb_alpha=0.001,
        hjb_phi=0.0001,
        hjb_horizon_seconds=150.0,
        gamma_inventory_risk=0.05,
        hjb_max_dt_seconds=0.25,
        hjb_n_steps_min=200,
        hjb_n_steps_max=2000,
    )
    params = {
        "kappa+": parameters["kappa_plus"],
        "kappa-": parameters["kappa_minus"],
        "lambda+": parameters["lambda_plus"],
        "lambda-": parameters["lambda_minus"],
        "epsilon+": parameters["epsilon_plus"],
        "epsilon-": parameters["epsilon_minus"],
    }
    hjb = solve_hjb(params, config, sigma2_per_sec=parameters["sigma2_per_second"])
    points = [(150.0, -5.0), (150.0, 0.0), (75.0, 1.5), (1.0, 5.0)]
    depths = [
        {
            "tau": tau,
            "q": q,
            "bid": select_delta(hjb, q, "bid", tau_remaining=tau),
            "ask": select_delta(hjb, q, "ask", tau_remaining=tau),
        }
        for tau, q in points
    ]
    mid = (0.132_08 + 0.132_09) / 2.0
    flat = compute_quotes(mid, 0, hjb, config, price_tick_size=PRICE_TICK, tau_remaining=150.0, q_exact=0.0)
    fractional = compute_quotes(
        mid, 2, hjb, config, price_tick_size=PRICE_TICK, tau_remaining=75.0, q_exact=2_802 / 1_868
    )
    return {
        "n_steps": int(len(hjb["t_grid"]) - 1),
        "phi_effective": float(hjb["phi_effective"]),
        "alpha_effective": float(hjb["alpha_effective"]),
        "depths": depths,
        "flat_bid_units": int(round(flat.bid_price * 1e6)),
        "flat_ask_units": int(round(flat.ask_price * 1e6)),
        "fractional_bid_units": int(round(fractional.bid_price * 1e6)),
        "fractional_ask_units": int(round(fractional.ask_price * 1e6)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unscaled", action="store_true", help="pre-v5 lambda (self-check)")
    args = parser.parse_args()
    parameters = calibrate(args.unscaled)
    output = {"semantics": "direct_window_v1" if args.unscaled else "direct_window_v2", "parameters": parameters, "model": model(parameters)}
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
