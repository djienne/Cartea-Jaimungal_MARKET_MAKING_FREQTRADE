from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hjb import compute_h_asymmetric, compute_h_symmetric  # noqa: E402


def test_boundary_depths_are_infinite():
    res = compute_h_asymmetric(
        lambda_plus=0.1,
        lambda_minus=0.1,
        epsilon_plus=0.0,
        epsilon_minus=0.0,
        kappa_plus=2.0,
        kappa_minus=2.0,
        alpha=0.01,
        phi=0.01,
        T_seconds=60,
        q_max=3,
    )

    assert np.isinf(res["delta_plus"][0])
    assert np.isinf(res["delta_minus"][-1])
    assert np.all(np.isfinite(res["delta_plus"][1:]))
    assert np.all(np.isfinite(res["delta_minus"][:-1]))
    assert res["boundary_policy"] == "disabled_side_is_inf"


def test_asymmetric_matches_symmetric_when_kappas_equal():
    kwargs = dict(
        lambda_plus=0.2,
        lambda_minus=0.15,
        epsilon_plus=0.01,
        epsilon_minus=0.008,
        kappa_plus=2.0,
        kappa_minus=2.0,
        alpha=0.01,
        phi=0.001,
        T_seconds=60,
        q_max=3,
    )

    sym = compute_h_symmetric(**kwargs)
    asym = compute_h_asymmetric(**kwargs, n_steps=500)

    mask_p = np.isfinite(sym["delta_plus"]) & np.isfinite(asym["delta_plus"])
    mask_m = np.isfinite(sym["delta_minus"]) & np.isfinite(asym["delta_minus"])

    assert np.allclose(sym["delta_plus"][mask_p], asym["delta_plus"][mask_p], atol=1e-3)
    assert np.allclose(sym["delta_minus"][mask_m], asym["delta_minus"][mask_m], atol=1e-3)


def test_no_risk_center_depth_matches_inverse_kappa():
    res = compute_h_asymmetric(
        lambda_plus=0.2,
        lambda_minus=0.2,
        epsilon_plus=0.0,
        epsilon_minus=0.0,
        kappa_plus=2.0,
        kappa_minus=4.0,
        alpha=0.0,
        phi=0.0,
        T_seconds=60,
        q_max=8,
        n_steps=300,
    )
    q_grid = res["q_grid"]
    center = int(np.where(q_grid == 0)[0][0])

    assert res["delta_plus"][center] == pytest.approx(0.5, abs=2e-3)
    assert res["delta_minus"][center] == pytest.approx(0.25, abs=2e-3)


def test_inventory_skew_direction_with_symmetric_params():
    res = compute_h_asymmetric(
        lambda_plus=0.2,
        lambda_minus=0.2,
        epsilon_plus=0.0,
        epsilon_minus=0.0,
        kappa_plus=2.0,
        kappa_minus=2.0,
        alpha=0.01,
        phi=0.001,
        T_seconds=60,
        q_max=4,
        n_steps=300,
    )
    q_grid = res["q_grid"]

    for idx, q in enumerate(q_grid):
        if q > 0 and np.isfinite(res["delta_plus"][idx]) and np.isfinite(res["delta_minus"][idx]):
            assert res["delta_plus"][idx] < res["delta_minus"][idx]
        if q < 0 and np.isfinite(res["delta_plus"][idx]) and np.isfinite(res["delta_minus"][idx]):
            assert res["delta_minus"][idx] < res["delta_plus"][idx]


@pytest.mark.parametrize(
    "override",
    [
        {"q_max": 0},
        {"T_seconds": 0},
        {"kappa_plus": 0},
        {"kappa_minus": -1},
        {"lambda_plus": -0.1},
        {"lambda_minus": -0.1},
        {"epsilon_plus": np.nan},
        {"alpha": -0.1},
        {"phi": -0.1},
    ],
)
def test_invalid_inputs_raise_value_error(override):
    kwargs = dict(
        lambda_plus=0.1,
        lambda_minus=0.1,
        epsilon_plus=0.0,
        epsilon_minus=0.0,
        kappa_plus=2.0,
        kappa_minus=2.0,
        alpha=0.01,
        phi=0.01,
        T_seconds=60,
        q_max=3,
    )
    kwargs.update(override)

    with pytest.raises(ValueError):
        compute_h_asymmetric(**kwargs)


# ---------------------------------------------------------------------------
# Asymmetric inventory domain [q_min, q_max]
# ---------------------------------------------------------------------------


BASE = dict(
    lambda_plus=0.1,
    lambda_minus=0.1,
    epsilon_plus=0.05,
    epsilon_minus=0.05,
    kappa_plus=2.0,
    kappa_minus=2.0,
    alpha=0.001,
    phi=0.0001,
    T_seconds=60.0,
)


def test_q_min_defaults_to_the_symmetric_grid():
    """Omitting q_min must be byte-for-byte the old behaviour."""
    implicit = compute_h_symmetric(**BASE, q_max=3)
    explicit = compute_h_symmetric(**BASE, q_max=3, q_min=-3)

    assert list(implicit["q_grid"]) == list(explicit["q_grid"]) == [-3, -2, -1, 0, 1, 2, 3]
    assert np.allclose(implicit["h"], explicit["h"])
    finite = np.isfinite(implicit["delta_plus"])
    assert np.allclose(implicit["delta_plus"][finite], explicit["delta_plus"][finite])


@pytest.mark.parametrize("solver", [compute_h_symmetric, compute_h_asymmetric])
def test_long_only_grid_disables_the_ask_when_flat(solver):
    """With q_min=0 the bottom of the grid is q=0, so the disabled side there is
    the ask: you cannot sell inventory you do not hold. This is the boundary the
    live long-only strategy actually has, and it differs from the q=0 row of a
    symmetric solve."""
    res = solver(**BASE, q_max=3, q_min=0)

    assert list(res["q_grid"]) == [0, 1, 2, 3]
    assert res["q_min"] == 0 and res["q_max"] == 3
    assert np.isinf(res["delta_plus"][0])       # no ask when flat
    assert np.isfinite(res["delta_minus"][0])   # but the bid is live
    assert np.isinf(res["delta_minus"][-1])     # no bid at max long
    assert np.all(np.isfinite(res["delta_plus"][1:]))


def test_long_only_solve_differs_from_the_symmetric_slice():
    """Solving on the wrong domain is not a harmless truncation -- the boundary
    is part of the optimisation, so h differs everywhere."""
    long_only = compute_h_symmetric(**BASE, q_max=3, q_min=0)
    symmetric = compute_h_symmetric(**BASE, q_max=3)

    sym_at_q1 = symmetric["delta_plus"][list(symmetric["q_grid"]).index(1)]
    lo_at_q1 = long_only["delta_plus"][list(long_only["q_grid"]).index(1)]
    assert not np.isclose(sym_at_q1, lo_at_q1, rtol=1e-3)


def test_asymmetric_ramp_domain_is_supported():
    res = compute_h_asymmetric(**BASE, q_max=3, q_min=-1)
    assert list(res["q_grid"]) == [-1, 0, 1, 2, 3]
    assert np.isinf(res["delta_plus"][0])
    assert np.isinf(res["delta_minus"][-1])


@pytest.mark.parametrize("q_min,q_max", [(3, 3), (5, 2), (0, 0)])
def test_incoherent_bounds_raise(q_min, q_max):
    with pytest.raises(ValueError):
        compute_h_symmetric(**BASE, q_max=q_max, q_min=q_min)


# ---------------------------------------------------------------------------
# Regression: expm overflow used to return a silent all-NaN surface
# ---------------------------------------------------------------------------


# Drift-asymmetric and high kappa: q_max*kappa*|lam+*eps+ - lam-*eps-|*T ran past
# the float exponent limit, expm overflowed, and the old np.maximum(omega, 1e-300)
# guard only caught underflow -- so every depth came back NaN with no error.
NAN_REGRESSION = dict(
    lambda_plus=2.0,
    lambda_minus=0.1,
    epsilon_plus=0.05,
    epsilon_minus=0.05,
    kappa_plus=100.0,
    kappa_minus=100.0,
    alpha=0.0001,
    phi=0.0002,
    q_max=3,
)


@pytest.mark.parametrize("T_seconds", [60.0, 300.0, 1800.0])
def test_drift_asymmetric_params_do_not_return_nan(T_seconds):
    res = compute_h_symmetric(**NAN_REGRESSION, T_seconds=T_seconds)

    assert np.all(np.isfinite(res["h"]))
    assert np.all(np.isfinite(res["delta_plus"][1:]))
    assert np.all(np.isfinite(res["delta_minus"][:-1]))


def test_drift_asymmetric_result_is_correct_not_merely_finite():
    """Cross-check the eigen-shift against the independent backward-Euler
    solver, which never went through expm."""
    sym = compute_h_symmetric(**NAN_REGRESSION, T_seconds=60.0)
    asym = compute_h_asymmetric(**NAN_REGRESSION, T_seconds=60.0, n_steps=4000)

    finite = np.isfinite(sym["delta_plus"]) & np.isfinite(asym["delta_plus"])
    assert np.max(np.abs(sym["delta_plus"][finite] - asym["delta_plus"][finite])) < 1e-4


def test_benign_params_unchanged_by_the_eigen_shift():
    """Values captured before the change. An eigendecomposition is not the same
    arithmetic as a Pade expm, so this is a tolerance, not bit-equality."""
    res = compute_h_symmetric(**BASE, q_max=3)
    expected = np.array([0.843943, 0.668927, 0.583217, 0.516783, 0.431073, 0.256057])
    assert np.allclose(res["delta_plus"][1:], expected, atol=1e-5)
