use anyhow::{bail, Context, Result};
pub use mm_settings::{
    CalibrationConfig, DryRunConfig, FlowGuardConfig, LatencyConfig, ModelConfig, QuotingConfig,
    RiskConfig,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Network {
    #[default]
    Mainnet,
    Testnet,
}

impl Network {
    pub const fn ws_url(self) -> &'static str {
        match self {
            Self::Mainnet => "wss://api.hyperliquid.xyz/ws",
            Self::Testnet => "wss://api.hyperliquid-testnet.xyz/ws",
        }
    }

    pub const fn info_url(self) -> &'static str {
        match self {
            Self::Mainnet => "https://api.hyperliquid.xyz/info",
            Self::Testnet => "https://api.hyperliquid-testnet.xyz/info",
        }
    }

    pub const fn api_url(self) -> &'static str {
        match self {
            Self::Mainnet => "https://api.hyperliquid.xyz",
            Self::Testnet => "https://api.hyperliquid-testnet.xyz",
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    pub instrument: InstrumentProfile,
    pub runtime: RuntimeConfig,
    pub latency: LatencyConfig,
    pub storage: StorageConfig,
    pub calibration: CalibrationConfig,
    pub model: ModelConfig,
    pub quoting: QuotingConfig,
    pub risk: RiskConfig,
    pub dry_run: DryRunConfig,
    pub flow_guard: FlowGuardConfig,
    pub live: LiveConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct InstrumentProfile {
    pub profile: String,
    pub symbol: String,
    pub dex: String,
    pub validated: bool,
    pub expected_sz_decimals: u32,
    pub max_significant_figures: u32,
    pub minimum_notional: f64,
    pub evidence_path: PathBuf,
}

impl Default for InstrumentProfile {
    fn default() -> Self {
        Self {
            profile: "cashcat".to_owned(),
            symbol: "CASHCAT".to_owned(),
            dex: String::new(),
            validated: true,
            expected_sz_decimals: 0,
            max_significant_figures: 5,
            minimum_notional: 10.0,
            evidence_path: PathBuf::from("cashcat.validation.json"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub network: Network,
    pub hot_path_cpu: Option<usize>,
    pub market_stale_ms: u64,
    pub ws_ping_interval_ms: u64,
    pub ws_idle_timeout_ms: u64,
    pub market_event_capacity: usize,
    pub execution_event_capacity: usize,
    pub stats_interval_ms: u64,
    pub log_json: bool,
    /// Share of a run's wall time that may be lost to public-feed gaps before
    /// the evidence is called invalid.
    ///
    /// A gap is missing data and is always recorded, but a binary latch made
    /// the verdict useless: past about three hours a run is guaranteed at least
    /// one venue connection recycle, so `scientifically_valid` was false for
    /// every long run and therefore said nothing. The counters carry the truth;
    /// this only decides where to draw the line.
    pub max_feed_downtime_fraction: f64,
    /// A single gap this long disqualifies a run regardless of the total: a
    /// ten-minute hole is a different kind of problem from sixty short blips.
    pub max_feed_gap_ms: u64,
    /// How late a *genuinely new* trade print may arrive before the public
    /// stream is treated as broken and reconnected.
    ///
    /// Separate from `market_stale_ms`, which it used to share a value with.
    /// That answers "is the top-of-book fresh enough to quote from"; this
    /// answers "is a new trade so late the feed is broken". Replayed prints
    /// that predate the connection are filtered out before this applies, so the
    /// threshold only ever sees live delivery -- whose body is fast (p50
    /// 378 ms, p99 2.4 s over 183,344 CASHCAT trades).
    pub max_trade_lag_ms: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            network: Network::Mainnet,
            hot_path_cpu: None,
            market_stale_ms: 5_000,
            ws_ping_interval_ms: 5_000,
            ws_idle_timeout_ms: 15_000,
            market_event_capacity: 65_536,
            execution_event_capacity: 16_384,
            stats_interval_ms: 5_000,
            log_json: false,
            max_feed_downtime_fraction: 0.05,
            max_feed_gap_ms: 60_000,
            max_trade_lag_ms: 5_000,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveMode {
    #[default]
    Production,
    AcceptanceTest,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
// Config structs mirror the TOML surface; independent feature toggles are
// clearer as named bools than as a state enum here.
#[allow(clippy::struct_excessive_bools)]
pub struct LiveConfig {
    pub enabled: bool,
    pub mode: LiveMode,
    pub credentials_path: PathBuf,
    pub state_path: PathBuf,
    pub action_timeout_ms: u64,
    pub action_expiry_ms: u64,
    pub reconcile_interval_ms: u64,
    pub startup_warmup_seconds: u64,
    pub deadman_enabled: bool,
    pub deadman_deadline_ms: u64,
    pub deadman_refresh_ms: u64,
    pub max_maker_fee_rate: f64,
    pub flatten_on_stop: bool,
    pub emergency_flatten_max_slippage_bps: f64,
    pub max_rest_weight_per_minute: u64,
    pub max_ws_messages_per_minute: u64,
    pub max_inflight_posts: usize,
    pub cancel_reserve_fraction: f64,
    pub acceptance_max_order_notional_usdc: f64,
    pub acceptance_max_directional_notional_usdc: f64,
    pub acceptance_max_working_gross_usdc: f64,
    pub acceptance_max_turnover_usdc: f64,
    pub acceptance_max_realized_loss_usdc: f64,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: LiveMode::Production,
            credentials_path: PathBuf::from("../hyperliquid.env"),
            state_path: PathBuf::from("run/cashcat-live.redb"),
            action_timeout_ms: 2_000,
            action_expiry_ms: 5_000,
            reconcile_interval_ms: 30_000,
            startup_warmup_seconds: 120,
            deadman_enabled: true,
            deadman_deadline_ms: 30_000,
            deadman_refresh_ms: 10_000,
            max_maker_fee_rate: 0.0002,
            flatten_on_stop: false,
            emergency_flatten_max_slippage_bps: 250.0,
            max_rest_weight_per_minute: 1_000,
            max_ws_messages_per_minute: 1_600,
            max_inflight_posts: 64,
            cancel_reserve_fraction: 0.25,
            acceptance_max_order_notional_usdc: 12.0,
            acceptance_max_directional_notional_usdc: 12.0,
            acceptance_max_working_gross_usdc: 24.0,
            acceptance_max_turnover_usdc: 60.0,
            acceptance_max_realized_loss_usdc: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub state_path: PathBuf,
    pub calibration_path: PathBuf,
    pub latency_path: PathBuf,
    pub report_dir: PathBuf,
    pub writer_lock_path: PathBuf,
    pub flush_interval_seconds: u64,
    pub compact_after_minutes: u64,
    pub retention_minutes: u64,
    pub write_parquet: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("../scripts/HL_data"),
            state_path: PathBuf::from("run/cashcat-dry-state.json"),
            calibration_path: PathBuf::from("run/cashcat-calibration.json"),
            latency_path: PathBuf::from("run/cashcat-latency.json"),
            report_dir: PathBuf::from("reports"),
            writer_lock_path: PathBuf::from("run/cashcat-collector.lock"),
            flush_interval_seconds: 10,
            compact_after_minutes: 15,
            retention_minutes: 180,
            write_parquet: true,
        }
    }
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config {}", path.display()))?;
        let mut config: Self = toml::from_str(&raw)
            .with_context(|| format!("cannot parse TOML config {}", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        for value in [
            &mut config.storage.data_dir,
            &mut config.storage.state_path,
            &mut config.storage.calibration_path,
            &mut config.storage.latency_path,
            &mut config.storage.report_dir,
            &mut config.storage.writer_lock_path,
            &mut config.instrument.evidence_path,
            &mut config.live.credentials_path,
            &mut config.live.state_path,
        ] {
            if value.is_relative() {
                *value = base.join(&*value);
            }
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_profile()?;
        self.validate_flow_guard()?;
        for (name, capacity) in [
            (
                "runtime.market_event_capacity",
                self.runtime.market_event_capacity,
            ),
            (
                "runtime.execution_event_capacity",
                self.runtime.execution_event_capacity,
            ),
        ] {
            if capacity == 0 || !capacity.is_power_of_two() {
                bail!("{name} must be a positive power of two");
            }
        }
        if !(0.0..=1.0).contains(&self.runtime.max_feed_downtime_fraction)
            || !self.runtime.max_feed_downtime_fraction.is_finite()
        {
            bail!("runtime.max_feed_downtime_fraction must be finite and inside [0, 1]");
        }
        if self.runtime.max_feed_gap_ms == 0 {
            bail!("runtime.max_feed_gap_ms must be greater than zero");
        }
        if self.runtime.max_trade_lag_ms < 50 {
            bail!("runtime.max_trade_lag_ms must be >= 50");
        }
        if self.runtime.market_stale_ms < 50 {
            bail!("runtime.market_stale_ms must be >= 50");
        }
        if self.runtime.ws_ping_interval_ms < 1_000 {
            bail!("runtime.ws_ping_interval_ms must be >= 1000");
        }
        if self.runtime.ws_idle_timeout_ms <= self.runtime.ws_ping_interval_ms {
            bail!("runtime.ws_idle_timeout_ms must exceed ws_ping_interval_ms");
        }
        if self.runtime.stats_interval_ms < 100 {
            bail!("runtime.stats_interval_ms must be >= 100");
        }
        if !self.latency.max_acceptable_p95_ms.is_finite()
            || self.latency.max_acceptable_p95_ms <= 0.0
        {
            bail!("latency.max_acceptable_p95_ms must be finite and positive");
        }
        if self.latency.minimum_samples == 0 {
            bail!("latency.minimum_samples must be positive");
        }
        if self.latency.minimum_network_samples == 0 {
            bail!("latency.minimum_network_samples must be positive");
        }
        // Network samples come from application pings, so the rolling window
        // physically bounds how many can ever exist. Requiring more than that
        // makes the production gate unsatisfiable and warm-up never completes.
        let achievable_network_samples = self
            .latency
            .window_seconds
            .saturating_mul(1_000)
            .checked_div(self.runtime.ws_ping_interval_ms.max(1))
            .unwrap_or(0);
        if self.latency.minimum_network_samples > achievable_network_samples {
            bail!(
                "latency.minimum_network_samples ({}) exceeds the {achievable_network_samples}                  samples a {}s window can hold at a {}ms ping interval; the production gate                  could never open",
                self.latency.minimum_network_samples,
                self.latency.window_seconds,
                self.runtime.ws_ping_interval_ms
            );
        }
        if self.latency.max_sample_age_ms
            < self
                .runtime
                .stats_interval_ms
                .max(self.runtime.ws_ping_interval_ms)
        {
            bail!("latency.max_sample_age_ms must cover stats_interval_ms and ws_ping_interval_ms");
        }
        if self.latency.hot_sample_every == 0 || !self.latency.hot_sample_every.is_power_of_two() {
            bail!("latency.hot_sample_every must be a positive power of two");
        }
        if self.latency.window_seconds == 0 {
            bail!("latency.window_seconds must be positive");
        }
        if self.latency.queue_capacity == 0 || !self.latency.queue_capacity.is_power_of_two() {
            bail!("latency.queue_capacity must be a positive power of two");
        }
        if self.storage.retention_minutes < self.calibration.window_minutes + 30 {
            bail!("storage retention must exceed the calibration window by at least 30 minutes");
        }
        if self.live.action_timeout_ms == 0 || self.live.action_expiry_ms == 0 {
            bail!("live action timeout and expiry must be positive");
        }
        if self.live.reconcile_interval_ms < 500 {
            bail!("live.reconcile_interval_ms must be at least 500");
        }
        if self.live.deadman_enabled
            && (self.live.deadman_deadline_ms < 5_000
                || self.live.deadman_refresh_ms == 0
                || self.live.deadman_refresh_ms >= self.live.deadman_deadline_ms)
        {
            bail!("live dead-man refresh must be positive and below a deadline of at least 5s");
        }
        if !self.live.max_maker_fee_rate.is_finite() || self.live.max_maker_fee_rate < 0.0 {
            bail!("live.max_maker_fee_rate must be finite and non-negative");
        }
        if !self.live.emergency_flatten_max_slippage_bps.is_finite()
            || !(0.0..=250.0).contains(&self.live.emergency_flatten_max_slippage_bps)
        {
            bail!("live emergency flatten slippage must be in [0,250] bps");
        }
        if self.live.max_rest_weight_per_minute == 0
            || self.live.max_ws_messages_per_minute == 0
            || self.live.max_inflight_posts == 0
            || !(0.0..1.0).contains(&self.live.cancel_reserve_fraction)
        {
            bail!("invalid live rate-limit configuration");
        }
        for (name, value, hard_maximum) in [
            (
                "acceptance_max_order_notional_usdc",
                self.live.acceptance_max_order_notional_usdc,
                12.0,
            ),
            (
                "acceptance_max_directional_notional_usdc",
                self.live.acceptance_max_directional_notional_usdc,
                12.0,
            ),
            (
                "acceptance_max_working_gross_usdc",
                self.live.acceptance_max_working_gross_usdc,
                24.0,
            ),
            (
                "acceptance_max_turnover_usdc",
                self.live.acceptance_max_turnover_usdc,
                60.0,
            ),
            (
                "acceptance_max_realized_loss_usdc",
                self.live.acceptance_max_realized_loss_usdc,
                0.5,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 || value > hard_maximum {
                bail!("live.{name} must be positive and <= {hard_maximum}");
            }
        }
        if self.model.q_max < 1 || self.model.q_max > 1_000 {
            bail!("model.q_max must be in 1..=1000");
        }
        if !self.model.horizon_seconds.is_finite() || self.model.horizon_seconds <= 0.0 {
            bail!("model.horizon_seconds must be finite and positive");
        }
        if self.quoting.min_half_spread_bps >= self.quoting.max_half_spread_bps {
            bail!("minimum half-spread must be below maximum half-spread");
        }
        if self.quoting.replace_threshold_ticks < 0 {
            bail!("quoting.replace_threshold_ticks must be non-negative");
        }
        if !self.quoting.replace_threshold_bps.is_finite()
            || self.quoting.replace_threshold_bps < 0.0
            || self.quoting.replace_threshold_bps >= self.quoting.max_half_spread_bps
        {
            bail!(
                "quoting.replace_threshold_bps must be finite, non-negative, and below \
                 max_half_spread_bps: a hold window wider than the widest permitted quote \
                 is nonsense"
            );
        }
        if self.quoting.min_order_lifetime_ms == 0 {
            bail!("quoting.min_order_lifetime_ms must be positive");
        }
        // The WebSocket budget must cover the worst-case replace rate plus
        // protocol overhead. Shipping a config where sustained requoting alone
        // exhausts the budget starves pings and dead-man refreshes.
        let replace_messages_per_minute = 120_000 / self.quoting.min_order_lifetime_ms;
        let ping_messages_per_minute = 60_000 / self.runtime.ws_ping_interval_ms.max(1);
        let deadman_messages_per_minute = if self.live.deadman_enabled {
            60_000 / self.live.deadman_refresh_ms.max(1)
        } else {
            0
        };
        if replace_messages_per_minute + ping_messages_per_minute + deadman_messages_per_minute
            > self.live.max_ws_messages_per_minute
        {
            bail!(
                "WebSocket budget insufficient: {replace_messages_per_minute} worst-case \
                 replace messages + {ping_messages_per_minute} pings + \
                 {deadman_messages_per_minute} dead-man refreshes per minute exceed \
                 live.max_ws_messages_per_minute ({}); raise the budget or \
                 quoting.min_order_lifetime_ms",
                self.live.max_ws_messages_per_minute
            );
        }
        if !(0.0..=1.0).contains(&self.quoting.target_capital_utilisation) {
            bail!("target_capital_utilisation must be in [0,1]");
        }
        if !self.quoting.leverage.is_finite() || self.quoting.leverage <= 0.0 {
            bail!("quoting.leverage must be finite and positive");
        }
        for (lower, upper, side) in [
            (
                self.calibration.support_quantile_lower_plus,
                self.calibration.support_quantile_upper_plus,
                "plus",
            ),
            (
                self.calibration.support_quantile_lower_minus,
                self.calibration.support_quantile_upper_minus,
                "minus",
            ),
        ] {
            if !(0.0 <= lower && lower < upper && upper <= 1.0) {
                bail!("invalid {side} calibration support quantiles");
            }
        }
        Ok(())
    }

    /// The flow guard withdraws quoting, so a misconfigured one is not a
    /// performance problem but a trading outage. Reject the shapes that would
    /// either fire constantly or never fire at all.
    fn validate_flow_guard(&self) -> Result<()> {
        let guard = &self.flow_guard;
        if !guard.enabled {
            return Ok(());
        }
        if guard.fast_move_window_ms == 0 {
            bail!("flow_guard.fast_move_window_ms must be greater than zero");
        }
        if !guard.fast_move_threshold_bps.is_finite() || guard.fast_move_threshold_bps <= 0.0 {
            bail!("flow_guard.fast_move_threshold_bps must be finite and greater than zero");
        }
        // A breaker that trips inside the spread would fire on every tick.
        if guard.fast_move_threshold_bps <= self.quoting.max_half_spread_bps {
            bail!(
                "flow_guard.fast_move_threshold_bps ({}) must exceed quoting.max_half_spread_bps ({});                  a breaker inside the widest quote would fire continuously",
                guard.fast_move_threshold_bps,
                self.quoting.max_half_spread_bps
            );
        }
        if !guard.vpin_threshold.is_finite()
            || guard.vpin_threshold <= 0.0
            || guard.vpin_threshold >= 1.0
        {
            bail!("flow_guard.vpin_threshold must be finite and inside (0, 1)");
        }
        if guard.vpin_window_buckets == 0 {
            bail!("flow_guard.vpin_window_buckets must be greater than zero");
        }
        if guard.vpin_buckets_per_day == 0 {
            bail!("flow_guard.vpin_buckets_per_day must be greater than zero");
        }
        Ok(())
    }

    fn validate_profile(&self) -> Result<()> {
        if !self.instrument.validated {
            bail!("instrument profile is not scientifically validated");
        }
        let raw = std::fs::read_to_string(&self.instrument.evidence_path).with_context(|| {
            format!(
                "cannot read instrument validation evidence {}",
                self.instrument.evidence_path.display()
            )
        })?;
        let evidence: ValidationEvidence = serde_json::from_str(&raw)?;
        if evidence.schema_version != 1
            || evidence.profile != self.instrument.profile
            || evidence.symbol != self.instrument.symbol
            || !evidence.scientifically_valid
            || !evidence.rounded_quotes_exact
            || evidence.parameter_relative_tolerance > 0.001
            || evidence.hjb_relative_tolerance > 0.001
        {
            bail!("instrument validation evidence does not satisfy the release contract");
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        Ok(hex::encode(Sha256::digest(serde_json::to_vec(self)?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cashcat_config_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("config/cashcat.toml")
    }

    #[test]
    fn validated_cashcat_defaults_pass() {
        let path = cashcat_config_path();
        assert!(AppConfig::load(&path).is_ok());
    }

    #[test]
    fn unvalidated_symbol_fails_closed() {
        let mut config = AppConfig::default();
        config.instrument.profile = "synthetic".to_owned();
        config.instrument.symbol = "SYN".to_owned();
        assert!(config.validate().is_err());
    }

    #[test]
    fn future_symbol_needs_only_matching_evidence_and_profile_data() {
        let directory = tempfile::tempdir().unwrap();
        let evidence = directory.path().join("synthetic.validation.json");
        std::fs::write(
            &evidence,
            r#"{
                "schema_version": 1,
                "profile": "synthetic",
                "symbol": "SYN",
                "scientifically_valid": true,
                "parameter_relative_tolerance": 0.00001,
                "hjb_relative_tolerance": 0.00001,
                "rounded_quotes_exact": true
            }"#,
        )
        .unwrap();
        let mut config = AppConfig::default();
        config.instrument.profile = "synthetic".to_owned();
        config.instrument.symbol = "SYN".to_owned();
        config.instrument.expected_sz_decimals = 3;
        config.instrument.evidence_path = evidence;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn acceptance_caps_and_deadman_timing_cannot_be_relaxed_past_contract() {
        let path = cashcat_config_path();
        let mut config = AppConfig::load(&path).unwrap();
        config.live.acceptance_max_directional_notional_usdc = 12.01;
        assert!(config.validate().is_err());
        config.live.acceptance_max_directional_notional_usdc = 12.0;
        config.live.deadman_refresh_ms = config.live.deadman_deadline_ms;
        assert!(config.validate().is_err());
    }
}

#[derive(Debug, Deserialize)]
struct ValidationEvidence {
    schema_version: u32,
    profile: String,
    symbol: String,
    scientifically_valid: bool,
    parameter_relative_tolerance: f64,
    hjb_relative_tolerance: f64,
    rounded_quotes_exact: bool,
}

/// How much of a run the public feed was missing, and whether that disqualifies
/// the evidence.
///
/// Feed gaps used to latch `scientifically_valid` false forever, which made the
/// flag meaningless for any run long enough to see a venue connection recycle —
/// i.e. every run past about three hours. The counters below are the durable
/// artefact; the verdict is a convenience derived from them, so a reader can
/// see "9 gaps, 31 s total, worst 3.8 s over 72 h" instead of a bare `false`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeedHealth {
    pub gaps: u64,
    pub downtime_ms: u64,
    pub longest_gap_ms: u64,
    pub run_ms: u64,
    pub downtime_fraction: f64,
    /// True when the causal event ring saturated. Unlike a gap, this means the
    /// simulation processed the wrong sequence, so it is always disqualifying.
    pub event_loss: bool,
}

impl FeedHealth {
    pub fn new(
        gaps: u64,
        downtime_ms: u64,
        longest_gap_ms: u64,
        run_ms: u64,
        event_loss: bool,
    ) -> Self {
        let downtime_fraction = if run_ms == 0 {
            0.0
        } else {
            downtime_ms as f64 / run_ms as f64
        };
        Self {
            gaps,
            downtime_ms,
            longest_gap_ms,
            run_ms,
            downtime_fraction,
            event_loss,
        }
    }

    /// Reasons this run's feed disqualifies its evidence. Empty means healthy.
    pub fn failures(&self, runtime: &RuntimeConfig) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.event_loss {
            reasons.push(
                "causal market-event ring saturated: events were dropped, so the simulated \
                 sequence is wrong rather than merely incomplete"
                    .to_owned(),
            );
        }
        if self.downtime_fraction > runtime.max_feed_downtime_fraction {
            reasons.push(format!(
                "public feed was down for {:.2}% of the run ({} gaps, {} ms total), over the {:.2}% limit",
                self.downtime_fraction * 100.0,
                self.gaps,
                self.downtime_ms,
                runtime.max_feed_downtime_fraction * 100.0
            ));
        }
        if self.longest_gap_ms > runtime.max_feed_gap_ms {
            reasons.push(format!(
                "longest public feed gap was {} ms, over the {} ms limit",
                self.longest_gap_ms, runtime.max_feed_gap_ms
            ));
        }
        reasons
    }

    pub fn is_valid(&self, runtime: &RuntimeConfig) -> bool {
        self.failures(runtime).is_empty()
    }
}

#[cfg(test)]
mod feed_health_tests {
    use super::*;

    fn runtime() -> RuntimeConfig {
        RuntimeConfig::default()
    }

    #[test]
    fn short_blips_over_a_long_run_stay_valid() {
        // The case that made the old latch useless: a multi-day grid always
        // sees venue connection recycles, so this must not disqualify it.
        let health = FeedHealth::new(9, 31_402, 3_812, 72 * 60 * 60 * 1_000, false);
        assert!(health.downtime_fraction < 0.001);
        assert!(
            health.is_valid(&runtime()),
            "{:?}",
            health.failures(&runtime())
        );
    }

    #[test]
    fn event_loss_is_always_disqualifying() {
        // Zero downtime, but the ring dropped events: the sequence is wrong.
        let health = FeedHealth::new(0, 0, 0, 3_600_000, true);
        assert!(!health.is_valid(&runtime()));
        assert!(health.failures(&runtime())[0].contains("ring saturated"));
    }

    #[test]
    fn too_much_cumulative_downtime_is_disqualifying() {
        // 10% of a one-hour run, against the 5% default.
        let health = FeedHealth::new(20, 360_000, 30_000, 3_600_000, false);
        assert!(!health.is_valid(&runtime()));
        assert!(health.failures(&runtime())[0].contains("down for"));
    }

    #[test]
    fn one_long_gap_is_disqualifying_even_when_the_total_is_small() {
        // 90 s missing from a 24 h run is only 0.1% of it, but a gap that long
        // is a different kind of problem from many short blips.
        let health = FeedHealth::new(1, 90_000, 90_000, 24 * 60 * 60 * 1_000, false);
        assert!(health.downtime_fraction < 0.05);
        assert!(!health.is_valid(&runtime()));
        assert!(health.failures(&runtime())[0].contains("longest"));
    }

    #[test]
    fn a_zero_length_run_does_not_divide_by_zero() {
        let health = FeedHealth::new(0, 0, 0, 0, false);
        assert_eq!(health.downtime_fraction, 0.0);
        assert!(health.is_valid(&runtime()));
    }
}
