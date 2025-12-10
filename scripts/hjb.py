from __future__ import annotations

import numpy as np
from scipy.linalg import expm


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
