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


# --- time-dependent control surface (delta*(t,q), eq. 10.26-10.27) ---
#
# The solvers integrate backward from h(T,q) = -alpha*q^2 and used to return
# only the t=0 slice. Everything below pins the surface that integration was
# already producing and throwing away.

SURFACE_PARAMS = dict(
    lambda_plus=1.2,
    lambda_minus=0.9,
    epsilon_plus=0.002,
    epsilon_minus=0.003,
    alpha=0.01,
    phi=0.0005,
    T_seconds=150.0,
    q_max=4,
)


def test_return_surface_leaves_the_legacy_result_untouched():
    """The stationary path must be bit-identical, or every existing gate moves."""
    plain = compute_h_asymmetric(kappa_plus=80.0, kappa_minus=120.0, **SURFACE_PARAMS)
    with_surface = compute_h_asymmetric(
        kappa_plus=80.0, kappa_minus=120.0, **SURFACE_PARAMS, return_surface=True
    )
    added = {"t_grid", "h_surface", "delta_plus_surface", "delta_minus_surface", "T_seconds"}
    assert set(with_surface) - set(plain) == added
    assert np.array_equal(plain["h"], with_surface["h"])
    assert np.array_equal(plain["delta_plus"], with_surface["delta_plus"])
    assert np.array_equal(plain["delta_minus"], with_surface["delta_minus"])


@pytest.mark.parametrize(
    "solver, kappas",
    [
        (compute_h_asymmetric, dict(kappa_plus=80.0, kappa_minus=120.0)),
        (compute_h_symmetric, dict(kappa_plus=100.0, kappa_minus=100.0)),
    ],
)
def test_surface_spans_zero_to_T_and_ends_on_the_terminal_condition(solver, kappas):
    res = solver(**kappas, **SURFACE_PARAMS, return_surface=True)
    t_grid = res["t_grid"]
    q_grid = res["q_grid"]

    assert t_grid[0] == pytest.approx(0.0)
    assert t_grid[-1] == pytest.approx(SURFACE_PARAMS["T_seconds"])
    assert np.all(np.diff(t_grid) > 0)
    assert res["h_surface"].shape == (len(t_grid), len(q_grid))

    # h(T,q) = -alpha*q^2 exactly. Only true up to an additive constant in
    # general -- the solvers renormalise per slice -- so compare differences.
    terminal = res["h_surface"][-1]
    expected = -SURFACE_PARAMS["alpha"] * q_grid.astype(float) ** 2
    assert np.allclose(terminal - terminal[0], expected - expected[0])

    # The t=0 slice is what the stationary path returns.
    assert np.allclose(res["h_surface"][0], res["h"])
    assert np.allclose(res["delta_plus_surface"][0], res["delta_plus"], equal_nan=True)


def test_terminal_depths_match_the_closed_form_from_the_terminal_condition():
    """delta+ = 1/k+ + eps+ + alpha(1-2q), delta- = 1/k- + eps- + alpha(1+2q).

    Substituting h(T,q) = -alpha*q^2 into eq. 10.27 gives these directly, so
    they pin the terminal slice against algebra rather than against the solver.
    """
    res = compute_h_asymmetric(
        kappa_plus=80.0, kappa_minus=120.0, **SURFACE_PARAMS, return_surface=True
    )
    q = res["q_grid"].astype(float)
    alpha = SURFACE_PARAMS["alpha"]
    expected_plus = 1 / 80.0 + SURFACE_PARAMS["epsilon_plus"] + alpha * (1 - 2 * q)
    expected_minus = 1 / 120.0 + SURFACE_PARAMS["epsilon_minus"] + alpha * (1 + 2 * q)

    # Boundary columns stay disabled at every time node: no ask at q_min, no bid
    # at q_max. That is structural, not time-varying.
    assert np.isinf(res["delta_plus_surface"][:, 0]).all()
    assert np.isinf(res["delta_minus_surface"][:, -1]).all()
    assert np.allclose(res["delta_plus_surface"][-1][1:], expected_plus[1:])
    assert np.allclose(res["delta_minus_surface"][-1][:-1], expected_minus[:-1])


def test_solvers_agree_across_the_whole_surface_when_kappa_is_symmetric():
    """The exact matrix solution is the ground truth for the Newton solver.

    Checked at a FIXED time-to-go, not as a max over slices: backward Euler is
    first order, so the error at tau falls like O(dt) but with a constant that
    grows as tau shrinks. A max over all slices is dominated by tau = dt and
    does not converge -- which is a property of the scheme, not a defect.
    """
    base = dict(SURFACE_PARAMS, kappa_plus=100.0, kappa_minus=100.0)
    errors = {}
    for n_steps in (200, 800):
        approx = compute_h_asymmetric(**base, n_steps=n_steps, return_surface=True)
        exact = compute_h_symmetric(**base, n_steps=n_steps, return_surface=True)
        assert np.allclose(approx["t_grid"], exact["t_grid"])
        tau = base["T_seconds"] - approx["t_grid"]
        idx = int(np.argmin(np.abs(tau - 10.0)))
        a = approx["delta_plus_surface"][idx]
        b = exact["delta_plus_surface"][idx]
        # Skip the disabled boundary column: inf - inf is NaN, not an error.
        finite = np.isfinite(a) & np.isfinite(b)
        errors[n_steps] = float(np.max(np.abs(a[finite] - b[finite])))

    assert errors[200] < 1e-3
    # Four times the resolution, at least three times the accuracy: first order.
    assert errors[800] < errors[200] / 3.0
