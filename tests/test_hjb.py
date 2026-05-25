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
