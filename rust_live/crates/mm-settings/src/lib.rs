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
    /// Requote hold window as a multiple of the venue price quantum. A resting
    /// order whose price is within the hold window of the new target (same
    /// size and reduce-only flag) is kept, preserving queue position.
    pub replace_threshold_ticks: i64,
    /// Price-relative half of the hold window, in basis points of the target
    /// price. The effective window is the larger of the two thresholds, which
    /// keeps the behavior portable across instruments whose quantum differs
    /// in bps terms.
    pub replace_threshold_bps: f64,
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
            replace_threshold_bps: 2.0,
            // 100ms keeps the worst-case replace rate inside the WebSocket
            // message budget with room for pings and dead-man refreshes; 75ms
            // consumed the entire budget by itself.
            min_order_lifetime_ms: 100,
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
    /// Deterministic empirical tail applied every `tail_latency_every` exchange
    /// seconds. CASHCAT measurements put p95 near 2.35x the median.
    pub tail_latency_multiplier: f64,
    pub tail_latency_every: u64,
    pub queue_decay_per_second: f64,
    /// Taker fee used when comparing variants after conservatively flattening
    /// their residual inventory at the executable side of the book.
    pub promotion_flatten_fee_rate: f64,
    /// Additional adverse price movement charged to the virtual flatten.
    pub promotion_flatten_slippage_bps: f64,
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
            tail_latency_multiplier: 2.35,
            tail_latency_every: 20,
            // Hyperliquid exposes only aggregate depth, not order identities.
            // Time alone is therefore not evidence that queue ahead vanished.
            queue_decay_per_second: 0.0,
            promotion_flatten_fee_rate: 0.00035,
            promotion_flatten_slippage_bps: 25.0,
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
            // A 60s window with 5s pings can hold at most 12 samples, so 20
            // was unsatisfiable and production could never finish warming up.
            // AppConfig::validate now rejects impossible combinations.
            minimum_network_samples: 10,
            max_sample_age_ms: 15_000,
            hot_sample_every: 16,
            window_seconds: 60,
            // Sized so per-event dispatch sampling cannot overflow between two
            // observer drains even during bursts; overflow now blocks only
            // the window it happened in, but staying out of it is cheaper.
            queue_capacity: 16_384,
        }
    }
}

/// Guard against toxic order flow — the regime that produced the only losing
/// window in the 161.95 h tape.
///
/// Two tiers, because they fail differently and were measured separately
/// (`docs/TOXIC_FLOW_GUARD.md`):
///
/// - a **fast mid-move breaker**, which trips on a large adverse move inside a
///   few seconds. On the 2026-08-22 cascade it fired at −14% where VPIN needed
///   −45%, and had zero false positives across the whole tape;
/// - **VPIN**, the volume-synchronised probability of informed trading, which
///   identifies the toxic *regime* rather than the instant. 86% of the losing
///   window's volume arrived after it fired, so it is what keeps the bot out of
///   the aftermath.
///
/// Neither predicts a flash crash. They bound how much of one gets ridden down.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlowGuardConfig {
    pub enabled: bool,
    /// Trailing window for the fast breaker.
    pub fast_move_window_ms: u64,
    /// Adverse mid move within that window that trips the breaker, in bps.
    /// 800 (8%) was selected as the tightest threshold with zero false positives
    /// on 6.8 days and re-verified with zero outside-cascade breaches over
    /// 165.11 h; 5% produced four on the selection tape.
    pub fast_move_threshold_bps: f64,
    /// Volume buckets per day, which sets bucket size from observed volume.
    /// 50 is the VPIN literature default and the reference implementation's.
    pub vpin_buckets_per_day: u32,
    /// Rolling bucket count in the VPIN numerator.
    pub vpin_window_buckets: u32,
    /// VPIN level treated as toxic. Re-verification over 165.11 h measured a
    /// Corrected exact-volume bucketing measured a 0.367 maximum outside the
    /// cascade and a 0.655 cascade peak over the frozen 165.11 h tape.
    pub vpin_threshold: f64,
    /// Minimum time quoting stays withdrawn after a trip. Re-entry additionally
    /// requires VPIN to have fallen back under `vpin_threshold`, so this is a
    /// floor and not the whole rule.
    pub cooldown_ms: u64,
}

impl Default for FlowGuardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fast_move_window_ms: 5_000,
            fast_move_threshold_bps: 800.0,
            vpin_buckets_per_day: 50,
            vpin_window_buckets: 30,
            vpin_threshold: 0.40,
            cooldown_ms: 900_000,
        }
    }
}
