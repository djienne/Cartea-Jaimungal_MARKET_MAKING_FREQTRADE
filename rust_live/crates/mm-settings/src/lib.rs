#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub q_max: i64,
    pub horizon_seconds: f64,
    pub phi_kappa_t: f64,
    pub phi_kappa_t_max: f64,
    pub alpha_kappa: f64,
    pub raw_phi_fallback: f64,
    pub raw_alpha_fallback: f64,
    pub volatility_risk_coefficient: f64,
    pub max_dt_seconds: f64,
    pub min_steps: usize,
    pub max_steps: usize,
    pub newton_max_iterations: usize,
    pub newton_tolerance: f64,
    pub newton_damping: f64,
    pub episode_reset_on_flat: bool,
    pub episode_min_elapsed_fraction: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            q_max: 6,
            horizon_seconds: 150.0,
            phi_kappa_t: 200.0,
            phi_kappa_t_max: 300.0,
            alpha_kappa: 0.05,
            raw_phi_fallback: 0.0001,
            raw_alpha_fallback: 0.001,
            volatility_risk_coefficient: 0.05,
            max_dt_seconds: 0.25,
            min_steps: 200,
            max_steps: 2_000,
            newton_max_iterations: 50,
            newton_tolerance: 1.0e-8,
            newton_damping: 0.7,
            episode_reset_on_flat: true,
            episode_min_elapsed_fraction: 0.25,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct QuotingConfig {
    pub available_capital_usdc: f64,
    pub target_capital_utilisation: f64,
    pub leverage: f64,
    pub maker_fee_rate: f64,
    pub spread_multiplier: f64,
    pub extra_cushion_bps: f64,
    pub min_half_spread_bps: f64,
    pub max_half_spread_bps: f64,
    pub replace_threshold_ticks: i64,
    pub min_order_lifetime_ms: u64,
    pub reduce_only_threshold_q: f64,
}

impl Default for QuotingConfig {
    fn default() -> Self {
        Self {
            available_capital_usdc: 1_000.0,
            target_capital_utilisation: 0.74,
            leverage: 2.0,
            maker_fee_rate: 0.00015,
            spread_multiplier: 1.0,
            extra_cushion_bps: 0.0,
            min_half_spread_bps: 1.5,
            max_half_spread_bps: 80.0,
            replace_threshold_ticks: 1,
            min_order_lifetime_ms: 75,
            reduce_only_threshold_q: 5.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiskConfig {
    pub kill_switch: bool,
    pub max_notional_usdc: f64,
    pub max_margin_usdc: f64,
    pub min_liquidation_buffer_usdc: f64,
    pub maintenance_margin_rate: f64,
    pub max_daily_loss_usdc: f64,
    pub max_consecutive_losses: u32,
    pub max_market_spread_bps: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            kill_switch: false,
            max_notional_usdc: 3_200.0,
            max_margin_usdc: 1_600.0,
            min_liquidation_buffer_usdc: 100.0,
            maintenance_margin_rate: 0.05,
            max_daily_loss_usdc: 200.0,
            max_consecutive_losses: 25,
            max_market_spread_bps: 100.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CalibrationConfig {
    pub window_minutes: u64,
    pub interval_seconds: u64,
    pub epsilon_horizon_ms_plus: u64,
    pub epsilon_horizon_ms_minus: u64,
    pub support_quantile_lower_plus: f64,
    pub support_quantile_lower_minus: f64,
    pub support_quantile_upper_plus: f64,
    pub support_quantile_upper_minus: f64,
    pub outage_threshold_seconds: f64,
    pub max_age_seconds: u64,
    pub max_data_age_seconds: u64,
    pub max_future_skew_seconds: u64,
    pub min_kappa_fit_points: usize,
    pub min_kappa_r2: f64,
    pub min_epsilon_events: usize,
    pub max_toxicity: f64,
    pub shard_window_margin_seconds: u64,
    pub max_shard_failure_rate: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            window_minutes: 120,
            interval_seconds: 30,
            epsilon_horizon_ms_plus: 200,
            epsilon_horizon_ms_minus: 200,
            support_quantile_lower_plus: 0.0,
            support_quantile_lower_minus: 0.0,
            support_quantile_upper_plus: 0.99,
            support_quantile_upper_minus: 0.99,
            outage_threshold_seconds: 60.0,
            max_age_seconds: 120,
            max_data_age_seconds: 120,
            max_future_skew_seconds: 10,
            min_kappa_fit_points: 6,
            min_kappa_r2: 0.30,
            min_epsilon_events: 50,
            max_toxicity: 1.5,
            shard_window_margin_seconds: 120,
            max_shard_failure_rate: 0.05,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DryRunConfig {
    pub starting_equity_usdc: f64,
    pub decision_latency_ms: u64,
    pub acknowledgement_latency_ms: u64,
    pub cancel_latency_ms: u64,
    pub queue_decay_per_second: f64,
    pub funding_rate_per_hour: f64,
    pub markout_horizons_ms: Vec<u64>,
}

impl Default for DryRunConfig {
    fn default() -> Self {
        Self {
            starting_equity_usdc: 1_000.0,
            decision_latency_ms: 250,
            acknowledgement_latency_ms: 250,
            cancel_latency_ms: 250,
            queue_decay_per_second: 0.05,
            funding_rate_per_hour: 0.0,
            markout_horizons_ms: vec![100, 1_000, 5_000, 30_000],
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LatencyConfig {
    pub gate_enabled: bool,
    #[serde(alias = "max_acceptable_p99_ms")]
    pub max_acceptable_p95_ms: f64,
    pub minimum_samples: u64,
    pub minimum_network_samples: u64,
    pub max_sample_age_ms: u64,
    pub hot_sample_every: u64,
    pub window_seconds: u64,
    pub queue_capacity: usize,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            gate_enabled: true,
            max_acceptable_p95_ms: 150.0,
            minimum_samples: 20,
            minimum_network_samples: 20,
            max_sample_age_ms: 15_000,
            hot_sample_every: 16,
            window_seconds: 60,
            queue_capacity: 4_096,
        }
    }
}
