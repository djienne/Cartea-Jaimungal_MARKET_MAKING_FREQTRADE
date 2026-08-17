from __future__ import annotations

import numpy as np
from scipy.linalg import eig, solve_banded


def _resolve_q_bounds(q_min, q_max) -> tuple[int, int]:
    """Resolve the inventory grid bounds.

    ``q_min=None`` keeps the symmetric [-q_max, q_max] grid the model shipped
    with. Passing it explicitly allows an asymmetric domain -- notably the
    long-only [0, q_max], where delta_plus at the bottom of the grid becomes
    "no ask when flat" rather than "no ask when maximally short".
    """
    q_max_i = int(q_max)
    if q_min is None:
        if q_max_i < 1:
            raise ValueError("q_max must be >= 1")
        return -q_max_i, q_max_i
    q_min_i = int(q_min)
    if q_min_i >= q_max_i:
        raise ValueError("q_min must be < q_max")
    return q_min_i, q_max_i


def _validate_common_inputs(
    lambda_plus: float,
    lambda_minus: float,
    epsilon_plus: float,
    epsilon_minus: float,
    kappa_plus: float,
    kappa_minus: float,
    alpha: float,
    phi: float,
    T_seconds: float,
    q_max: int,
    q_min: int | None = None,
) -> None:
    values = {
        "lambda_plus": lambda_plus,
        "lambda_minus": lambda_minus,
        "epsilon_plus": epsilon_plus,
        "epsilon_minus": epsilon_minus,
        "kappa_plus": kappa_plus,
        "kappa_minus": kappa_minus,
        "alpha": alpha,
        "phi": phi,
        "T_seconds": T_seconds,
    }
    for name, value in values.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    _resolve_q_bounds(q_min, q_max)
    if float(T_seconds) <= 0:
        raise ValueError("T_seconds must be > 0")
    if float(kappa_plus) <= 0 or float(kappa_minus) <= 0:
        raise ValueError("kappa_plus and kappa_minus must be > 0")
    if float(lambda_plus) < 0 or float(lambda_minus) < 0:
        raise ValueError("lambda_plus and lambda_minus must be >= 0")
    if float(alpha) < 0 or float(phi) < 0:
        raise ValueError("alpha and phi must be >= 0")


def _optimal_delta_and_value(
    lam: float,
    kappa: float,
    eps: float,
    dh: float,
    *,
    clip_at_zero: bool = False,
) -> tuple[float, float, float]:
    """
    One-side optimal depth and maximized HJB contribution.

    Args:
        lam: Baseline arrival intensity (trades/sec).
        kappa: Book depth sensitivity.
        eps: Adverse-selection jump magnitude.
        dh: h(t, q_next) - h(t, q) where q_next = q-1 (ask hit) or q+1 (bid hit).

    Returns:
        (delta_star, value_star, dvalue_ddh) where value_star is the maximized
        arrival term and dvalue_ddh is its exact derivative w.r.t. ``dh``.

    The derivative is what makes an analytic Jacobian possible. At the interior
    optimum value = (lam/kappa)*exp(-kappa*delta*) with delta* = 1/kappa + eps - dh,
    so d(value)/d(dh) = kappa * value -- no finite differences needed.
    """
    lam = max(float(lam), 0.0)
    kappa = max(1e-12, float(kappa))
    eps = float(eps)
    dh = float(dh)

    # Unconstrained optimum from FOC:
    # delta* = 1/kappa + eps - dh  (cf. infos_MM.ipynb)
    delta_star = (1.0 / kappa) + eps - dh

    # HJB gain bracket at delta=0 is c = -eps + dh.
    c = -eps + dh

    if clip_at_zero and delta_star <= 0.0:
        # Best is to quote at the touch (delta=0) if c>0,
        # otherwise to not quote (delta -> +inf gives value 0).
        # value = max(lam*c, 0) with dc/d(dh) = 1, so the slope is lam or 0.
        if lam * c > 0.0:
            return 0.0, lam * c, lam
        return 0.0, 0.0, 0.0

    # At the interior optimum, bracket equals 1/kappa (unconstrained model).
    raw_exponent = -kappa * delta_star
    exponent = float(np.clip(raw_exponent, -700.0, 700.0))
    value_star = (lam / kappa) * np.exp(exponent)
    if raw_exponent > 700.0:
        # The value saturated, so it no longer varies with dh. Reporting the
        # unsaturated slope kappa*value here would hand Newton a ~1e304
        # derivative for a locally constant function. Only reachable at
        # implausible h-jumps (dh >= 7 + eps at kappa=100), but the Jacobian
        # should describe the function actually being evaluated.
        return float(delta_star), float(value_star), 0.0
    return float(delta_star), float(value_star), float(kappa * value_star)


def compute_h_symmetric(
    lambda_plus: float,
    lambda_minus: float,
    epsilon_plus: float,
    epsilon_minus: float,
    kappa_plus: float,
    kappa_minus: float,
    *,
    alpha: float = 0.0,
    phi: float = 0.0,
    T_seconds: float = 30 * 60,
    q_max: int = 3,
    q_min: int | None = None,
):
    """
    Closed-form matrix solution from fq_market_making_introduction.ipynb
    under symmetric κ (use average of κ+/κ-). Returns h(t=0, q) vector and
    corresponding δ+ / δ- optimal depths for each inventory state.

    ``q_min`` defaults to -q_max (the symmetric grid). Pass it to solve on an
    asymmetric inventory domain, e.g. q_min=0 for a long-only agent.
    """
    _validate_common_inputs(
        lambda_plus,
        lambda_minus,
        epsilon_plus,
        epsilon_minus,
        kappa_plus,
        kappa_minus,
        alpha,
        phi,
        T_seconds,
        q_max,
        q_min,
    )

    q_min, q_max = _resolve_q_bounds(q_min, q_max)
    kappa = 0.5 * (float(kappa_plus) + float(kappa_minus))
    lam_p = float(lambda_plus)
    lam_m = float(lambda_minus)
    eps_p = float(epsilon_plus)
    eps_m = float(epsilon_minus)

    lam_tilde_p = lam_p * np.exp(-1.0 - kappa * eps_p)
    lam_tilde_m = lam_m * np.exp(-1.0 - kappa * eps_m)

    q_grid = np.arange(q_min, q_max + 1)
    d = len(q_grid)
    A = np.zeros((d, d))

    for i, q in enumerate(q_grid):
        A[i, i] = q * kappa * (lam_p * eps_p - lam_m * eps_m) - phi * kappa * (q ** 2)
        if i > 0:
            A[i, i - 1] = lam_tilde_p
        if i < d - 1:
            A[i, i + 1] = lam_tilde_m

    z = np.exp(-alpha * kappa * (q_grid ** 2))

    # ω(0) = expm(A·T)·z, evaluated via an eigen-decomposition with an exact
    # max-shift. Computing expm(A·T) in linear space overflows once the drift
    # diagonal grows -- q_max·κ·|λ⁺ε⁺ − λ⁻ε⁻|·T past ~709 -- and the old
    # np.maximum(ω, 1e-300) guard only caught underflow, so the surface came
    # back silently all-NaN. Factoring exp(θ) = exp(θ−m)·exp(m) and adding m
    # back after the log is algebraically exact and cannot overflow.
    eigval, V = eig(A)
    imag_scale = float(np.max(np.abs(eigval.imag)))
    real_scale = max(float(np.max(np.abs(eigval.real))), 1e-300)
    if imag_scale > 1e-8 * real_scale:
        # A is tridiagonal with non-negative off-diagonal products, so it is
        # similar to a symmetric matrix and its spectrum is real. Complex
        # eigenvalues mean the matrix is not what the model assumes; fail loudly
        # rather than silently discarding the imaginary part.
        raise ValueError(
            "HJB transition matrix has a complex spectrum "
            f"(max|Im|={imag_scale:.3e} vs max|Re|={real_scale:.3e})"
        )
    theta = eigval.real * float(T_seconds)
    shift = float(np.max(theta))
    omega = np.real(V @ (np.exp(theta - shift) * np.linalg.solve(V, z)))
    log_omega = np.log(np.maximum(omega, 1e-300)) + shift
    # Normalize log-omega to reduce spread before dividing by kappa (invariant up to additive const)
    log_omega = log_omega - np.max(log_omega)
    h = log_omega / kappa

    # Boundary sides are disabled, not clamped: no ask at q_min, no bid at q_max.
    delta_plus = np.full(d, np.inf, dtype=float)
    delta_minus = np.full(d, np.inf, dtype=float)
    for i, q in enumerate(q_grid):
        h_q = h[i]
        if i > 0:
            h_qm1 = h[i - 1]
            delta_plus[i] = (1.0 / kappa) + eps_p - (h_qm1 - h_q)
        if i < d - 1:
            h_qp1 = h[i + 1]
            delta_minus[i] = (1.0 / kappa) + eps_m - (h_qp1 - h_q)

    return {
        "q_grid": q_grid,
        "h": h,
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "kappa_sym": kappa,
        "method": "matrix_exponential",
        "boundary_policy": "disabled_side_is_inf",
        "q_min": int(q_grid[0]),
        "q_max": int(q_grid[-1]),
    }


def compute_h_asymmetric(
    lambda_plus: float,
    lambda_minus: float,
    epsilon_plus: float,
    epsilon_minus: float,
    kappa_plus: float,
    kappa_minus: float,
    *,
    alpha: float = 0.0,
    phi: float = 0.0,
    T_seconds: float = 30 * 60,
    q_max: int = 3,
    q_min: int | None = None,
    n_steps: int = 200,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 0.7,
    clip_deltas: bool = False,
):
    """
    Backward-Euler solver for the asymmetric-κ HJB (κ+ != κ-).

    This follows the nonlinear HJB described in infos_MM.ipynb §4.1. The closed-form
    matrix exponential only applies under κ+ = κ-; otherwise we solve h(t,q) on a
    (t,q) grid and return the t=0 surface and optimal depths:

        δ+*(t,q) = 1/κ+ + ε+ - (h(t,q-1) - h(t,q))
        δ-*(t,q) = 1/κ- + ε- - (h(t,q+1) - h(t,q))

    The scheme uses damped Newton iterations for each implicit step, with an
    exact analytic Jacobian. G(h) depends on h only through the neighbouring
    differences, so dG/dh is tridiagonal and the Newton step is a banded solve.

    ``q_min`` defaults to -q_max (the symmetric grid). Pass it to solve on an
    asymmetric inventory domain, e.g. q_min=0 for a long-only agent.
    """
    _validate_common_inputs(
        lambda_plus,
        lambda_minus,
        epsilon_plus,
        epsilon_minus,
        kappa_plus,
        kappa_minus,
        alpha,
        phi,
        T_seconds,
        q_max,
        q_min,
    )

    q_min, q_max = _resolve_q_bounds(q_min, q_max)
    kappa_p = float(kappa_plus)
    kappa_m = float(kappa_minus)
    lam_p = float(lambda_plus)
    lam_m = float(lambda_minus)
    eps_p = float(epsilon_plus)
    eps_m = float(epsilon_minus)

    q_grid = np.arange(q_min, q_max + 1)
    d = len(q_grid)

    # Terminal condition h(T,q) = -alpha q^2
    h = -float(alpha) * (q_grid.astype(float) ** 2)

    n_steps = int(max(n_steps, 1))
    dt = float(T_seconds) / float(n_steps)
    dt = max(dt, 1e-6)

    def _compute_g_and_jac(h_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """G(h) plus its exact Jacobian, returned as tridiagonal bands.

        g[i] depends only on h[i-1], h[i], h[i+1], so dG/dh is tridiagonal. With
        s_p = d(val_p)/d(dh) and dh = h[i-1] - h[i]:

            dg[i]/dh[i-1] = +s_p      dg[i]/dh[i] = -s_p - s_m
            dg[i]/dh[i+1] = +s_m

        Building this costs O(d) instead of the O(d) full G-evaluations a
        finite-difference Jacobian needs, and lets the Newton step use a banded
        solve. That is the whole speedup.
        """
        g_vec = np.zeros_like(h_vec)
        jac_lower = np.zeros(d)  # dg[i]/dh[i-1]
        jac_diag = np.zeros(d)   # dg[i]/dh[i]
        jac_upper = np.zeros(d)  # dg[i]/dh[i+1]

        for i, q in enumerate(q_grid):
            h_q = h_vec[i]
            value_total = -float(phi) * (float(q) ** 2)

            if i > 0:
                _, val_p, slope_p = _optimal_delta_and_value(
                    lam_p,
                    kappa_p,
                    eps_p,
                    h_vec[i - 1] - h_q,
                    clip_at_zero=clip_deltas,
                )
                value_total += val_p
                jac_lower[i] = slope_p
                jac_diag[i] -= slope_p

            if i < d - 1:
                _, val_m, slope_m = _optimal_delta_and_value(
                    lam_m,
                    kappa_m,
                    eps_m,
                    h_vec[i + 1] - h_q,
                    clip_at_zero=clip_deltas,
                )
                value_total += val_m
                jac_upper[i] = slope_m
                jac_diag[i] -= slope_m

            drift = float(q) * (lam_p * eps_p - lam_m * eps_m)
            g_vec[i] = value_total + drift

        return g_vec, jac_lower, jac_diag, jac_upper


    for _ in range(n_steps):
        h_old = h.copy()
        h_old = h_old - np.max(h_old)
        h_new = h_old.copy()

        for _it in range(int(max_iter)):
            g, jac_lower, jac_diag, jac_upper = _compute_g_and_jac(h_new)
            # Implicit backward step for ∂_t h + G = 0:
            # h(t-dt) = h(t) + dt * G(h(t-dt))
            F = h_new - h_old - dt * g
            # Note: once the profile reaches its ergodic steady state, G(h) is a
            # nonzero constant vector (the strategy earns at a constant rate) and
            # the per-step max-normalisation quotients that constant out. F is
            # then a constant vector of magnitude dt*mean(G), so this test does
            # not fire and convergence is detected by the step-size break below.
            # That is correct -- depths depend only on differences of h -- but do
            # not read a non-triggering residual here as non-convergence.
            if np.max(np.abs(F)) < tol:
                break

            # JF = I - dt*Jg, tridiagonal. solve_banded wants three rows:
            # row 0 = superdiagonal (offset +1), row 1 = diagonal, row 2 = subdiagonal.
            ab = np.zeros((3, d))
            ab[0, 1:] = -dt * jac_upper[:-1]
            ab[1, :] = 1.0 - dt * jac_diag
            ab[2, :-1] = -dt * jac_lower[1:]
            try:
                step = solve_banded((1, 1), ab, -F)
            except (ValueError, np.linalg.LinAlgError):
                # Fall back to a damped fixed-point step.
                step = -F

            h_trial = h_new + damping * step
            h_trial = h_trial - np.max(h_trial)

            if np.max(np.abs(h_trial - h_new)) < tol:
                h_new = h_trial
                break
            h_new = h_trial

        h = h_new

    # Boundary sides are disabled, not clamped: no ask at q_min, no bid at q_max.
    delta_plus = np.full(d, np.inf, dtype=float)
    delta_minus = np.full(d, np.inf, dtype=float)
    for i, q in enumerate(q_grid):
        h_q = h[i]
        if i > 0:
            raw_plus, _, _ = _optimal_delta_and_value(
                lam_p,
                kappa_p,
                eps_p,
                h[i - 1] - h_q,
                clip_at_zero=clip_deltas,
            )
            delta_plus[i] = max(0.0, raw_plus) if clip_deltas else raw_plus
        if i < d - 1:
            raw_minus, _, _ = _optimal_delta_and_value(
                lam_m,
                kappa_m,
                eps_m,
                h[i + 1] - h_q,
                clip_at_zero=clip_deltas,
            )
            delta_minus[i] = max(0.0, raw_minus) if clip_deltas else raw_minus

    return {
        "q_grid": q_grid,
        "h": h,
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "kappa_plus": kappa_p,
        "kappa_minus": kappa_m,
        "method": "backward_euler",
        "boundary_policy": "disabled_side_is_inf",
        "q_min": int(q_grid[0]),
        "q_max": int(q_grid[-1]),
        "dt": dt,
        "n_steps": n_steps,
    }
