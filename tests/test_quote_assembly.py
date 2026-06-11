"""Range guarantee for the final quote assembly.

The user-facing contract: every assembled half-spread (model depth x
multiplier + one maker fee per side, clamped) lands in [3, 80] bps of mid,
fees included, for any plausible parameter combination. The clamps are the
hard guarantee; this sweep proves the wiring end-to-end (HJB solve ->
_assemble_half_spread) and the parity between the strategy implementation and
the replay harness's copy.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for path in (SCRIPTS, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hjb import compute_h_asymmetric  # noqa: E402
from replay_market_maker import assemble_half_spread as replay_assemble  # noqa: E402

# Installs the freqtrade stubs as an import side effect and provides the bot
# factory with a schema-v3 parameter snapshot.
from test_strategy_guards import make_bot  # noqa: E402

SPREAD_MULTIPLIER = 3.0
MAKER_FEE = 0.00015
MIN_BPS = 3.0
MAX_BPS = 80.0
GAMMA = 0.05
PHI_BASE = 0.0001
INVENTORY_UNIT = 0.01

KAPPAS = (1.0, 5.0, 20.0)
EPSILONS = (0.0, 0.1, 0.3)
LAMBDAS = (0.1, 1.0, 3.0)
MIDS = (1000.0, 4300.0, 10000.0)
SIGMA2S = (None, 1.0, 25.0)


def _configured_bot():
    bot = make_bot()
    bot.spread_multiplier = SPREAD_MULTIPLIER
    bot.fees_maker_HL = MAKER_FEE
    bot.min_half_spread_bps = MIN_BPS
    bot.max_half_spread_bps = MAX_BPS
    return bot


def _phi_effective(sigma2: float | None) -> float:
    if sigma2 is None:
        return PHI_BASE
    return PHI_BASE + GAMMA * sigma2 * INVENTORY_UNIT


def test_assembled_half_spread_always_lands_in_3_to_80_bps():
    bot = _configured_bot()
    pair = "ETH/USDC:USDC"

    for kappa, eps, lam, sigma2 in itertools.product(KAPPAS, EPSILONS, LAMBDAS, SIGMA2S):
        hjb = compute_h_asymmetric(
            lambda_plus=lam,
            lambda_minus=lam,
            epsilon_plus=eps,
            epsilon_minus=eps,
            kappa_plus=kappa,
            kappa_minus=kappa,
            alpha=0.001,
            phi=_phi_effective(sigma2),
            T_seconds=60.0,
            q_max=3,
            n_steps=50,
        )
        deltas = [float(d) for d in (*hjb["delta_plus"], *hjb["delta_minus"]) if np.isfinite(d)]
        assert deltas, "every interior q must have at least one finite delta"
        for mid in MIDS:
            for side, delta_model in (("ask", deltas[0]), ("bid", deltas[-1])):
                for delta in deltas:
                    delta_total, info = bot._assemble_half_spread(pair, side, delta, mid)
                    bps = info["bps"]
                    assert bps is not None
                    assert MIN_BPS - 1e-9 <= bps <= MAX_BPS + 1e-9, (
                        f"bps {bps} out of range for kappa={kappa} eps={eps} lam={lam} "
                        f"sigma2={sigma2} mid={mid} delta={delta}"
                    )
                    assert abs(delta_total / mid * 10_000.0 - bps) < 1e-9


def test_asymmetric_kappa_combo_also_stays_in_range():
    bot = _configured_bot()
    hjb = compute_h_asymmetric(
        lambda_plus=3.0,
        lambda_minus=0.1,
        epsilon_plus=0.3,
        epsilon_minus=0.0,
        kappa_plus=1.0,
        kappa_minus=20.0,
        alpha=0.001,
        phi=_phi_effective(25.0),
        T_seconds=60.0,
        q_max=3,
    )
    for mid in MIDS:
        for delta in (*hjb["delta_plus"], *hjb["delta_minus"]):
            if not np.isfinite(delta):
                continue
            _, info = bot._assemble_half_spread("ETH/USDC:USDC", "ask", float(delta), mid)
            assert MIN_BPS - 1e-9 <= info["bps"] <= MAX_BPS + 1e-9


def test_central_live_combo_ask_lands_in_band_before_clamping():
    """The pre-clamp value (not just the clamped one) should sit inside the
    band for realistic live parameters — the clamps are a guarantee, not the
    intended operating mode on the ask side."""
    bot = _configured_bot()
    hjb = compute_h_asymmetric(
        lambda_plus=0.4226,
        lambda_minus=0.3529,
        epsilon_plus=0.059,
        epsilon_minus=0.0,
        kappa_plus=3.27,
        kappa_minus=8.09,
        alpha=0.001,
        phi=PHI_BASE,
        T_seconds=60.0,
        q_max=3,
    )
    mid = 4300.0
    q_grid = list(hjb["q_grid"])
    ask_delta = float(hjb["delta_plus"][q_grid.index(1)])
    delta_total, info = bot._assemble_half_spread("ETH/USDC:USDC", "ask", ask_delta, mid)
    assert info["clamped"] is None
    assert MIN_BPS <= info["bps"] <= 10.0  # ~4 bps expected

    bid_delta = float(hjb["delta_minus"][q_grid.index(0)])
    _, bid_info = bot._assemble_half_spread("ETH/USDC:USDC", "bid", bid_delta, mid)
    # The bid side is tighter than 3 bps pre-clamp and rides the floor.
    assert bid_info["clamped"] in (None, "floor")
    assert MIN_BPS - 1e-9 <= bid_info["bps"] <= MAX_BPS + 1e-9


def test_strategy_and_replay_assembly_are_identical():
    bot = _configured_bot()
    sample_deltas = (0.0, 0.005, 0.05, 0.2, 0.5, 2.0, 10.0)
    sample_mids = (1000.0, 4300.0, 10000.0)
    for delta_model, mid in itertools.product(sample_deltas, sample_mids):
        strategy_total, _ = bot._assemble_half_spread("ETH/USDC:USDC", "ask", delta_model, mid)
        replay_total = replay_assemble(
            delta_model,
            mid,
            spread_multiplier=SPREAD_MULTIPLIER,
            maker_fee=MAKER_FEE,
            min_half_spread_bps=MIN_BPS,
            max_half_spread_bps=MAX_BPS,
        )
        assert replay_total is not None
        assert abs(strategy_total - replay_total) < 1e-12, (
            f"strategy/replay divergence at delta={delta_model} mid={mid}: "
            f"{strategy_total} vs {replay_total}"
        )

    assert replay_assemble(float("inf"), 4300.0) is None
