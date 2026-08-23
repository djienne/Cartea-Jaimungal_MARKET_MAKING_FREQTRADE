use crate::config::ModelConfig;
use crate::types::Side;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CjParameters {
    pub lambda_plus: f64,
    pub lambda_minus: f64,
    pub kappa_plus: f64,
    pub kappa_minus: f64,
    pub epsilon_plus: f64,
    pub epsilon_minus: f64,
    pub sigma2_per_second: Option<f64>,
}

impl CjParameters {
    pub fn validate(self) -> Result<()> {
        for (name, value) in [
            ("lambda_plus", self.lambda_plus),
            ("lambda_minus", self.lambda_minus),
            ("kappa_plus", self.kappa_plus),
            ("kappa_minus", self.kappa_minus),
            ("epsilon_plus", self.epsilon_plus),
            ("epsilon_minus", self.epsilon_minus),
        ] {
            if !value.is_finite() {
                bail!("{name} must be finite");
            }
        }
        if self.lambda_plus < 0.0 || self.lambda_minus < 0.0 {
            bail!("lambda values must be non-negative");
        }
        if self.kappa_plus <= 0.0 || self.kappa_minus <= 0.0 {
            bail!("kappa values must be strictly positive");
        }
        if self
            .sigma2_per_second
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            bail!("sigma2_per_second must be finite and non-negative when present");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HjbSurface {
    pub revision: u64,
    pub q_min: i64,
    pub q_max: i64,
    pub horizon_seconds: f64,
    pub dt: f64,
    pub n_steps: usize,
    pub phi_effective: f64,
    pub alpha_effective: f64,
    pub kappa_average: f64,
    pub t_grid: Vec<f64>,
    pub h_surface: Box<[f64]>,
    pub delta_plus_surface: Box<[f64]>,
    pub delta_minus_surface: Box<[f64]>,
    row_width: usize,
}

/// Both Cartea-Jaimungal quote depths selected from one `(t, q)` lookup.
/// `bid` maps to delta-minus and `ask` maps to delta-plus.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DepthPair {
    pub bid: Option<f64>,
    pub ask: Option<f64>,
}

impl HjbSurface {
    /// Select both quote depths without allocation. Time and inventory
    /// brackets are shared so bid/ask selection cannot drift independently.
    #[inline]
    pub fn depth_pair(&self, q_exact: f64, tau_remaining: f64) -> DepthPair {
        let elapsed = (self.horizon_seconds - tau_remaining).clamp(0.0, self.horizon_seconds);
        let (t_lo, t_hi, t_weight) = self.bracket_time(elapsed);
        let (q_lo, q_hi, q_weight) = bracket_inventory(self.q_min, self.q_max, q_exact);

        let select = |surface: &[f64]| {
            let low_row = interpolate_time(surface, self.row_width, t_lo, t_hi, t_weight, q_lo);
            let high_row = interpolate_time(surface, self.row_width, t_lo, t_hi, t_weight, q_hi);
            blend_depth(low_row, high_row, q_weight)
        };

        DepthPair {
            bid: select(&self.delta_minus_surface),
            ask: select(&self.delta_plus_surface),
        }
    }

    /// Select the Cartea-Jaimungal depth at physical inventory and episode time.
    /// A non-finite bracketing boundary disables the side conservatively.
    #[inline]
    pub fn depth(&self, side: Side, q_exact: f64, tau_remaining: f64) -> Option<f64> {
        let pair = self.depth_pair(q_exact, tau_remaining);
        match side {
            Side::Buy => pair.bid,
            Side::Sell => pair.ask,
        }
    }

    pub fn t0_depths(&self) -> (&[f64], &[f64]) {
        (
            &self.delta_plus_surface[..self.row_width],
            &self.delta_minus_surface[..self.row_width],
        )
    }

    /// O(1) time bracketing over the uniform grid `t_grid[i] = dt * i`.
    ///
    /// The grid is uniform by construction (asserted in [`solve_asymmetric`]),
    /// so the bracket is a division — not a binary search over ~600 points on
    /// every quote decision, as it was when this went through
    /// `partition_point`. Mirrors `bracket_inventory`.
    #[inline]
    fn bracket_time(&self, elapsed: f64) -> (usize, usize, f64) {
        if elapsed <= 0.0 || self.n_steps == 0 {
            return (0, 0, 0.0);
        }
        let position = elapsed / self.dt;
        let last = self.n_steps;
        if position >= last as f64 {
            return (last, last, 0.0);
        }
        let low = position.floor() as usize;
        let weight = position - low as f64;
        if weight == 0.0 {
            (low, low, 0.0)
        } else {
            (low, low.min(last - 1) + 1, weight)
        }
    }
}

pub fn solve_asymmetric(
    parameters: CjParameters,
    config: &ModelConfig,
    inventory_unit_base: f64,
    revision: u64,
) -> Result<HjbSurface> {
    parameters.validate()?;
    validate_model_config(config, inventory_unit_base)?;

    let q_min = -config.q_max;
    let q_max = config.q_max;
    let q_grid: Vec<i64> = (q_min..=q_max).collect();
    let dimension = q_grid.len();
    let n_steps = ((config.horizon_seconds / config.max_dt_seconds).ceil() as usize)
        .clamp(config.min_steps, config.max_steps)
        .max(1);
    let dt = (config.horizon_seconds / n_steps as f64).max(1.0e-6);
    let kappa_average = 0.5 * (parameters.kappa_plus + parameters.kappa_minus);
    let volatility_delta = parameters.sigma2_per_second.map_or(0.0, |sigma2| {
        config.volatility_risk_coefficient * sigma2 * inventory_unit_base
    });
    let mut phi_effective = if config.phi_kappa_t > 0.0 {
        config.phi_kappa_t / (kappa_average * config.horizon_seconds) + volatility_delta
    } else {
        config.raw_phi_fallback + volatility_delta
    };
    if config.phi_kappa_t_max > 0.0
        && phi_effective * kappa_average * config.horizon_seconds > config.phi_kappa_t_max
    {
        phi_effective = config.phi_kappa_t_max / (kappa_average * config.horizon_seconds);
    }
    let alpha_effective = if config.alpha_kappa > 0.0 {
        config.alpha_kappa / kappa_average
    } else {
        config.raw_alpha_fallback
    };

    let mut h: Vec<f64> = q_grid
        .iter()
        .map(|q| -alpha_effective * (*q as f64).powi(2))
        .collect();
    let mut reverse_slices = Vec::with_capacity(n_steps + 1);
    reverse_slices.push(h.clone());

    for _ in 0..n_steps {
        normalize_slice(&mut h);
        let h_old = h.clone();
        let mut h_new = h_old.clone();
        for _ in 0..config.newton_max_iterations {
            let (g, jac_lower, jac_diag, jac_upper) =
                compute_g_and_jac(&h_new, &q_grid, parameters, phi_effective);
            let residual: Vec<f64> = h_new
                .iter()
                .zip(&h_old)
                .zip(&g)
                .map(|((new, old), value)| new - old - dt * value)
                .collect();
            if max_abs(&residual) < config.newton_tolerance {
                break;
            }

            let mut lower = vec![0.0; dimension];
            let mut diagonal = vec![0.0; dimension];
            let mut upper = vec![0.0; dimension];
            for index in 0..dimension {
                diagonal[index] = 1.0 - dt * jac_diag[index];
                if index > 0 {
                    lower[index] = -dt * jac_lower[index];
                }
                if index + 1 < dimension {
                    upper[index] = -dt * jac_upper[index];
                }
            }
            let rhs: Vec<f64> = residual.iter().map(|value| -value).collect();
            let step = solve_tridiagonal(&lower, &diagonal, &upper, &rhs)
                .unwrap_or_else(|| residual.iter().map(|value| -value).collect());
            let mut trial: Vec<f64> = h_new
                .iter()
                .zip(step)
                .map(|(value, change)| value + config.newton_damping * change)
                .collect();
            normalize_slice(&mut trial);
            let change = trial
                .iter()
                .zip(&h_new)
                .map(|(next, previous)| next - previous)
                .collect::<Vec<_>>();
            h_new = trial;
            if max_abs(&change) < config.newton_tolerance {
                break;
            }
        }
        if h_new.iter().any(|value| !value.is_finite()) {
            bail!("asymmetric HJB solve produced a non-finite value");
        }
        h = h_new;
        reverse_slices.push(h.clone());
    }

    reverse_slices.reverse();
    let surface_capacity = reverse_slices.len().saturating_mul(dimension);
    let mut delta_plus_surface = Vec::with_capacity(surface_capacity);
    let mut delta_minus_surface = Vec::with_capacity(surface_capacity);
    for slice in &reverse_slices {
        let (plus, minus) = depths_from_h(slice, parameters);
        delta_plus_surface.extend(plus);
        delta_minus_surface.extend(minus);
    }
    let t_grid: Vec<f64> = (0..=n_steps)
        .map(|index| config.horizon_seconds - dt * (n_steps - index) as f64)
        .collect();
    // depth_pair brackets time as `index = elapsed / dt`, which is only valid
    // on an exactly uniform grid. The construction above is algebraically
    // `dt * index`; this guards against that ever changing.
    for (index, value) in t_grid.iter().enumerate() {
        if (value - dt * index as f64).abs() > 1.0e-9 * config.horizon_seconds.max(1.0) {
            bail!("HJB time grid is not uniform; O(1) bracketing would misread it");
        }
    }

    Ok(HjbSurface {
        revision,
        q_min,
        q_max,
        horizon_seconds: config.horizon_seconds,
        dt,
        n_steps,
        phi_effective,
        alpha_effective,
        kappa_average,
        t_grid,
        h_surface: reverse_slices
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        delta_plus_surface: delta_plus_surface.into_boxed_slice(),
        delta_minus_surface: delta_minus_surface.into_boxed_slice(),
        row_width: dimension,
    })
}

fn validate_model_config(config: &ModelConfig, inventory_unit_base: f64) -> Result<()> {
    if config.q_max < 1 {
        bail!("q_max must be positive");
    }
    for (name, value) in [
        ("horizon_seconds", config.horizon_seconds),
        ("phi_kappa_t", config.phi_kappa_t),
        ("phi_kappa_t_max", config.phi_kappa_t_max),
        ("alpha_kappa", config.alpha_kappa),
        (
            "volatility_risk_coefficient",
            config.volatility_risk_coefficient,
        ),
        ("max_dt_seconds", config.max_dt_seconds),
        ("newton_tolerance", config.newton_tolerance),
        ("newton_damping", config.newton_damping),
        ("inventory_unit_base", inventory_unit_base),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be finite and non-negative");
        }
    }
    if config.horizon_seconds <= 0.0
        || config.max_dt_seconds <= 0.0
        || config.newton_tolerance <= 0.0
        || inventory_unit_base <= 0.0
    {
        bail!("positive model scales must be strictly greater than zero");
    }
    if !(0.0..=1.0).contains(&config.newton_damping) || config.newton_damping == 0.0 {
        bail!("newton_damping must be in (0,1]");
    }
    Ok(())
}

fn optimal_delta_and_value(lambda: f64, kappa: f64, epsilon: f64, dh: f64) -> (f64, f64) {
    let delta = 1.0 / kappa + epsilon - dh;
    let raw_exponent = -kappa * delta;
    let value = lambda / kappa * raw_exponent.clamp(-700.0, 700.0).exp();
    let slope = if raw_exponent > 700.0 {
        0.0
    } else {
        kappa * value
    };
    (value, slope)
}

fn compute_g_and_jac(
    h: &[f64],
    q_grid: &[i64],
    parameters: CjParameters,
    phi: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let dimension = h.len();
    let mut g = vec![0.0; dimension];
    let mut lower = vec![0.0; dimension];
    let mut diagonal = vec![0.0; dimension];
    let mut upper = vec![0.0; dimension];
    for index in 0..dimension {
        let q = q_grid[index] as f64;
        let mut value = -phi * q.powi(2);
        if index > 0 {
            let (gain, slope) = optimal_delta_and_value(
                parameters.lambda_plus,
                parameters.kappa_plus,
                parameters.epsilon_plus,
                h[index - 1] - h[index],
            );
            value += gain;
            lower[index] = slope;
            diagonal[index] -= slope;
        }
        if index + 1 < dimension {
            let (gain, slope) = optimal_delta_and_value(
                parameters.lambda_minus,
                parameters.kappa_minus,
                parameters.epsilon_minus,
                h[index + 1] - h[index],
            );
            value += gain;
            upper[index] = slope;
            diagonal[index] -= slope;
        }
        value += q
            * (parameters.lambda_plus * parameters.epsilon_plus
                - parameters.lambda_minus * parameters.epsilon_minus);
        g[index] = value;
    }
    (g, lower, diagonal, upper)
}

fn depths_from_h(h: &[f64], parameters: CjParameters) -> (Vec<f64>, Vec<f64>) {
    let dimension = h.len();
    let mut plus = vec![f64::INFINITY; dimension];
    let mut minus = vec![f64::INFINITY; dimension];
    for index in 0..dimension {
        if index > 0 {
            plus[index] =
                1.0 / parameters.kappa_plus + parameters.epsilon_plus - (h[index - 1] - h[index]);
        }
        if index + 1 < dimension {
            minus[index] =
                1.0 / parameters.kappa_minus + parameters.epsilon_minus - (h[index + 1] - h[index]);
        }
    }
    (plus, minus)
}

fn normalize_slice(values: &mut [f64]) {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for value in values {
        *value -= maximum;
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn solve_tridiagonal(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let dimension = diagonal.len();
    if dimension == 0
        || lower.len() != dimension
        || upper.len() != dimension
        || rhs.len() != dimension
    {
        return None;
    }
    let mut c_prime = vec![0.0; dimension];
    let mut d_prime = vec![0.0; dimension];
    let first = diagonal[0];
    if !first.is_finite() || first.abs() < 1.0e-300 {
        return None;
    }
    c_prime[0] = upper[0] / first;
    d_prime[0] = rhs[0] / first;
    for index in 1..dimension {
        let denominator = diagonal[index] - lower[index] * c_prime[index - 1];
        if !denominator.is_finite() || denominator.abs() < 1.0e-300 {
            return None;
        }
        c_prime[index] = if index + 1 < dimension {
            upper[index] / denominator
        } else {
            0.0
        };
        d_prime[index] = (rhs[index] - lower[index] * d_prime[index - 1]) / denominator;
    }
    let mut solution = vec![0.0; dimension];
    solution[dimension - 1] = d_prime[dimension - 1];
    for index in (0..dimension - 1).rev() {
        solution[index] = d_prime[index] - c_prime[index] * solution[index + 1];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

#[inline]
fn bracket_inventory(q_min: i64, q_max: i64, value: f64) -> (usize, usize, f64) {
    let minimum = q_min as f64;
    let maximum = q_max as f64;
    if value <= minimum {
        return (0, 0, 0.0);
    }
    let last = (q_max - q_min) as usize;
    if value >= maximum {
        return (last, last, 0.0);
    }
    let position = value - minimum;
    let low = position.floor() as usize;
    let weight = position - low as f64;
    if weight == 0.0 {
        (low, low, 0.0)
    } else {
        (low, low + 1, weight)
    }
}

#[inline]
fn interpolate_time(
    surface: &[f64],
    row_width: usize,
    low: usize,
    high: usize,
    weight: f64,
    q_index: usize,
) -> f64 {
    let low_value = surface[low * row_width + q_index];
    if low == high || weight <= 0.0 {
        low_value
    } else if weight >= 1.0 {
        surface[high * row_width + q_index]
    } else {
        let high_value = surface[high * row_width + q_index];
        if low_value.is_finite() && high_value.is_finite() {
            low_value + weight * (high_value - low_value)
        } else {
            f64::INFINITY
        }
    }
}

fn blend_depth(low: f64, high: f64, weight: f64) -> Option<f64> {
    if !low.is_finite() || !high.is_finite() {
        return None;
    }
    let value = if weight <= 0.0 {
        low
    } else if weight >= 1.0 {
        high
    } else {
        low + weight * (high - low)
    };
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn parameters() -> CjParameters {
        CjParameters {
            lambda_plus: 0.117,
            lambda_minus: 0.103,
            kappa_plus: 10_538.0,
            kappa_minus: 9_161.0,
            epsilon_plus: 2.38e-5,
            epsilon_minus: 3.42e-5,
            sigma2_per_second: Some(3.8e-9),
        }
    }

    #[test]
    fn boundaries_disable_only_the_adding_side() {
        let config = ModelConfig::default();
        let surface = solve_asymmetric(parameters(), &config, 2_430.0, 1).unwrap();
        assert!(surface.depth(Side::Buy, 6.0, 150.0).is_none());
        assert!(surface.depth(Side::Sell, 6.0, 150.0).is_some());
        assert!(surface.depth(Side::Sell, -6.0, 150.0).is_none());
        assert!(surface.depth(Side::Buy, -6.0, 150.0).is_some());
    }

    /// The retired binary-search bracket, kept as the reference the O(1)
    /// division-based bracket must agree with.
    fn reference_bracket(axis: &[f64], value: f64) -> (usize, usize, f64) {
        if axis.len() == 1 || value <= axis[0] {
            return (0, 0, 0.0);
        }
        let last = axis.len() - 1;
        if value >= axis[last] {
            return (last, last, 0.0);
        }
        let high = axis.partition_point(|candidate| *candidate < value);
        if axis[high] == value {
            return (high, high, 0.0);
        }
        let low = high - 1;
        let span = axis[high] - axis[low];
        let weight = if span <= 0.0 {
            0.0
        } else {
            (value - axis[low]) / span
        };
        (low, high, weight)
    }

    #[test]
    fn division_bracket_matches_binary_search_across_the_surface() {
        let surface = solve_asymmetric(parameters(), &ModelConfig::default(), 2_430.0, 1).unwrap();
        let steps = 1_500_u32;
        let row_width = (surface.q_max - surface.q_min + 1) as usize;
        for index in 0..=steps {
            let elapsed = surface.horizon_seconds * f64::from(index) / f64::from(steps);
            let reference = reference_bracket(&surface.t_grid, elapsed);
            // Bracket indices may legitimately differ at exact grid points
            // (exact hit versus an epsilon weight); what must agree are the
            // interpolated depths.
            for q in [-5.5_f64, -2.0, 0.0, 0.25, 3.75, 5.5] {
                let tau = surface.horizon_seconds - elapsed;
                let via_pair = surface.depth_pair(q, tau);
                let (q_lo, q_hi, q_weight) = bracket_inventory(surface.q_min, surface.q_max, q);
                let reference_depth = |values: &[f64]| {
                    let low_row = interpolate_time(
                        values,
                        row_width,
                        reference.0,
                        reference.1,
                        reference.2,
                        q_lo,
                    );
                    let high_row = interpolate_time(
                        values,
                        row_width,
                        reference.0,
                        reference.1,
                        reference.2,
                        q_hi,
                    );
                    blend_depth(low_row, high_row, q_weight)
                };
                let reference_bid = reference_depth(&surface.delta_minus_surface);
                let reference_ask = reference_depth(&surface.delta_plus_surface);
                match (reference_bid, via_pair.bid) {
                    (Some(a), Some(b)) => assert_relative_eq!(a, b, epsilon = 1.0e-9),
                    (a, b) => assert_eq!(a.is_some(), b.is_some(), "bid at q={q} t={elapsed}"),
                }
                match (reference_ask, via_pair.ask) {
                    (Some(a), Some(b)) => assert_relative_eq!(a, b, epsilon = 1.0e-9),
                    (a, b) => assert_eq!(a.is_some(), b.is_some(), "ask at q={q} t={elapsed}"),
                }
            }
        }
    }

    #[test]
    fn fractional_inventory_interpolates_depths() {
        let config = ModelConfig::default();
        let surface = solve_asymmetric(parameters(), &config, 2_430.0, 1).unwrap();
        let low = surface.depth(Side::Sell, 1.0, 75.0).unwrap();
        let high = surface.depth(Side::Sell, 2.0, 75.0).unwrap();
        let middle = surface.depth(Side::Sell, 1.5, 75.0).unwrap();
        assert_relative_eq!(middle, 0.5 * (low + high), epsilon = 1.0e-12);
    }

    #[test]
    fn fractional_inventory_next_to_boundary_is_disabled() {
        let config = ModelConfig::default();
        let surface = solve_asymmetric(parameters(), &config, 2_430.0, 1).unwrap();
        assert!(surface.depth(Side::Buy, 5.01, 75.0).is_none());
    }

    #[test]
    fn volatility_channel_respects_phi_ceiling() {
        let config = ModelConfig::default();
        let mut stressed = parameters();
        stressed.sigma2_per_second = Some(1.0);
        let surface = solve_asymmetric(stressed, &config, 2_430.0, 1).unwrap();
        assert_relative_eq!(
            surface.phi_effective * surface.kappa_average * surface.horizon_seconds,
            config.phi_kappa_t_max,
            epsilon = 1.0e-9
        );
    }

    #[test]
    fn symmetric_parameter_limit_is_symmetric_at_flat_inventory() {
        let symmetric = CjParameters {
            lambda_plus: 0.2,
            lambda_minus: 0.2,
            kappa_plus: 100.0,
            kappa_minus: 100.0,
            epsilon_plus: 0.001,
            epsilon_minus: 0.001,
            sigma2_per_second: None,
        };
        let surface = solve_asymmetric(symmetric, &ModelConfig::default(), 10.0, 1).unwrap();
        assert_relative_eq!(
            surface.depth(Side::Buy, 0.0, 150.0).unwrap(),
            surface.depth(Side::Sell, 0.0, 150.0).unwrap(),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn backward_euler_depth_converges_when_dt_is_halved() {
        let mut coarse_config = ModelConfig::default();
        coarse_config.max_dt_seconds = 1.0;
        coarse_config.min_steps = 1;
        let mut medium_config = coarse_config.clone();
        medium_config.max_dt_seconds = 0.5;
        let mut fine_config = coarse_config.clone();
        fine_config.max_dt_seconds = 0.25;
        let coarse = solve_asymmetric(parameters(), &coarse_config, 2_430.0, 1).unwrap();
        let medium = solve_asymmetric(parameters(), &medium_config, 2_430.0, 1).unwrap();
        let fine = solve_asymmetric(parameters(), &fine_config, 2_430.0, 1).unwrap();
        let coarse_error = (coarse.depth(Side::Sell, 2.0, 20.0).unwrap()
            - fine.depth(Side::Sell, 2.0, 20.0).unwrap())
        .abs();
        let medium_error = (medium.depth(Side::Sell, 2.0, 20.0).unwrap()
            - fine.depth(Side::Sell, 2.0, 20.0).unwrap())
        .abs();
        assert!(medium_error < coarse_error);
    }
}
