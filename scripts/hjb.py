from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _optimal_delta_and_value(
    lam: float,
    kappa: float,
    eps: float,
    dh: float,
    *,
    clip_at_zero: bool = False,
) -> tuple[float, float]:
    """
    One-side optimal depth and maximized HJB contribution.

    Args:
        lam: Baseline arrival intensity (trades/sec).
        kappa: Book depth sensitivity.
        eps: Adverse-selection jump magnitude.
        dh: h(t, q_next) - h(t, q) where q_next = q-1 (ask hit) or q+1 (bid hit).

    Returns:
        (delta_star, value_star) where value_star is the maximized arrival term.
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
        return 0.0, max(lam * c, 0.0)

    # At the interior optimum, bracket equals 1/kappa (unconstrained model).
    exponent = -kappa * delta_star
    exponent = float(np.clip(exponent, -700.0, 700.0))
    value_star = (lam / kappa) * np.exp(exponent)
    return float(delta_star), float(value_star)


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
):
    """
    Closed-form matrix solution from fq_market_making_introduction.ipynb
    under symmetric κ (use average of κ+/κ-). Returns h(t=0, q) vector and
    corresponding δ+ / δ- optimal depths for each inventory state.
    """
    kappa = max(1e-8, 0.5 * (float(kappa_plus) + float(kappa_minus)))
    lam_p = max(float(lambda_plus), 0.0)
    lam_m = max(float(lambda_minus), 0.0)
    eps_p = float(epsilon_plus)
    eps_m = float(epsilon_minus)

    lam_tilde_p = lam_p * np.exp(-1.0 - kappa * eps_p)
    lam_tilde_m = lam_m * np.exp(-1.0 - kappa * eps_m)

    q_grid = np.arange(-q_max, q_max + 1)
    d = len(q_grid)
    A = np.zeros((d, d))

    for i, q in enumerate(q_grid):
        A[i, i] = q * kappa * (lam_p * eps_p - lam_m * eps_m) - phi * kappa * (q ** 2)
        if i > 0:
            A[i, i - 1] = lam_tilde_p
        if i < d - 1:
            A[i, i + 1] = lam_tilde_m

    z = np.exp(-alpha * kappa * (q_grid ** 2))
    # Solve in log-domain for stability: log(ω) = log(expm(A*T)·z).
    # We compute expm(A*T) in linear space but stabilize the multiplication via max/clip before log.
    omega = expm(A * T_seconds).dot(z)
    omega = np.maximum(omega, 1e-300)  # guard against log(0)
    log_omega = np.log(omega)
    # Normalize log-omega to reduce spread before dividing by kappa (invariant up to additive const)
    log_omega = log_omega - np.max(log_omega)
    h = log_omega / kappa

    # Optimal depths at t=0 for each q
    delta_plus = []
    delta_minus = []
    for i, q in enumerate(q_grid):
        h_q = h[i]
        h_qm1 = h[i - 1] if i > 0 else h_q  # clamp at edge
        h_qp1 = h[i + 1] if i < d - 1 else h_q
        delta_plus.append((1.0 / kappa) + eps_p - (h_qm1 - h_q))
        delta_minus.append((1.0 / kappa) + eps_m - (h_qp1 - h_q))

    return {
        "q_grid": q_grid,
        "h": h,
        "delta_plus": np.array(delta_plus),
        "delta_minus": np.array(delta_minus),
        "kappa_sym": kappa,
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

    The scheme uses damped Newton iterations (finite-difference Jacobian)
    for each implicit step.
    """
    kappa_p = max(1e-8, float(kappa_plus))
    kappa_m = max(1e-8, float(kappa_minus))
    lam_p = max(float(lambda_plus), 0.0)
    lam_m = max(float(lambda_minus), 0.0)
    eps_p = float(epsilon_plus)
    eps_m = float(epsilon_minus)

    q_grid = np.arange(-q_max, q_max + 1)
    d = len(q_grid)

    # Terminal condition h(T,q) = -alpha q^2
    h = -float(alpha) * (q_grid.astype(float) ** 2)

    n_steps = int(max(n_steps, 1))
    dt = float(T_seconds) / float(n_steps)
    dt = max(dt, 1e-6)

    def _compute_g(h_vec: np.ndarray) -> np.ndarray:
        g_vec = np.zeros_like(h_vec)
        for i, q in enumerate(q_grid):
            h_q = h_vec[i]
            h_qm1 = h_vec[i - 1] if i > 0 else h_q
            h_qp1 = h_vec[i + 1] if i < d - 1 else h_q

            _, val_p = _optimal_delta_and_value(
                lam_p, kappa_p, eps_p, h_qm1 - h_q, clip_at_zero=clip_deltas
            )
            _, val_m = _optimal_delta_and_value(
                lam_m, kappa_m, eps_m, h_qp1 - h_q, clip_at_zero=clip_deltas
            )

            drift = float(q) * (lam_p * eps_p - lam_m * eps_m)
            g_vec[i] = val_p + val_m + drift - float(phi) * (float(q) ** 2)
        return g_vec

    fd_eps = 1e-6

    for _ in range(n_steps):
        h_old = h.copy()
        h_old = h_old - np.max(h_old)
        h_new = h_old.copy()

        for _it in range(int(max_iter)):
            g = _compute_g(h_new)
            # Implicit backward step for ∂_t h + G = 0:
            # h(t-dt) = h(t) + dt * G(h(t-dt))
            F = h_new - h_old - dt * g
            if np.max(np.abs(F)) < tol:
                break

            # Finite-difference Jacobian of g
            Jg = np.zeros((d, d))
            for j in range(d):
                h_pert = h_new.copy()
                h_pert[j] += fd_eps
                g_pert = _compute_g(h_pert)
                Jg[:, j] = (g_pert - g) / fd_eps

            JF = np.eye(d) - dt * Jg
            try:
                step = np.linalg.solve(JF, -F)
            except np.linalg.LinAlgError:
                # Fall back to a damped fixed-point step.
                step = -F

            h_trial = h_new + damping * step
            h_trial = h_trial - np.max(h_trial)

            if np.max(np.abs(h_trial - h_new)) < tol:
                h_new = h_trial
                break
            h_new = h_trial

        h = h_new

    # Optimal depths at t=0 for each q
    delta_plus = np.zeros(d)
    delta_minus = np.zeros(d)
    for i, q in enumerate(q_grid):
        h_q = h[i]
        h_qm1 = h[i - 1] if i > 0 else h_q
        h_qp1 = h[i + 1] if i < d - 1 else h_q
        raw_plus = (1.0 / kappa_p) + eps_p - (h_qm1 - h_q)
        raw_minus = (1.0 / kappa_m) + eps_m - (h_qp1 - h_q)
        delta_plus[i] = max(0.0, raw_plus) if clip_deltas else raw_plus
        delta_minus[i] = max(0.0, raw_minus) if clip_deltas else raw_minus

    return {
        "q_grid": q_grid,
        "h": h,
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "kappa_plus": kappa_p,
        "kappa_minus": kappa_m,
        "method": "backward_euler",
        "dt": dt,
        "n_steps": n_steps,
    }
