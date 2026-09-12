//! Independent paper accounts sharing one public market feed.
//! Variant overrides isolate policy choices while latency, fees, funding and
//! capital remain common controls.

use anyhow::{bail, Context, Result};
use fs2::FileExt;
use mm_live::config::{AppConfig, FeedHealth};
use mm_live::hjb::CjParameters;
use mm_live::types::{Bbo, MarketEvent};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct PaperMarketState {
    pub bbo: Option<Bbo>,
    pub pause_reason: Option<&'static str>,
    /// Set on the observation that ends a pause: how long quoting was blind.
    pub resumed_after_ms: Option<u64>,
    connected_ns: u64,
    ready_after_ns: u64,
    paused_since_ns: u64,
}

impl PaperMarketState {
    pub fn observe(
        &mut self,
        event: Option<&MarketEvent>,
        now_ns: u64,
        now_ms: u64,
        connected_ns: u64,
        disconnected: bool,
        max_age_ms: u64,
    ) -> bool {
        let was_paused = self.pause_reason.is_some();
        let connection_changed = connected_ns != self.connected_ns;
        self.connected_ns = connected_ns;
        if connection_changed {
            self.bbo = None;
            self.ready_after_ns = connected_ns;
        }
        if disconnected {
            self.bbo = None;
            self.ready_after_ns = now_ns;
        }
        if let Some(MarketEvent::Bbo(bbo)) = event {
            if !disconnected
                && bbo.is_valid()
                && bbo.recv_ns >= self.ready_after_ns
                && now_ms.saturating_sub(bbo.exchange_ms) <= max_age_ms
            {
                self.bbo = Some(*bbo);
            }
        }
        if self.bbo.is_some_and(|bbo| {
            now_ns.saturating_sub(bbo.recv_ns) > max_age_ms.saturating_mul(1_000_000)
        }) {
            self.bbo = None;
            self.ready_after_ns = now_ns;
        }
        self.pause_reason = if disconnected {
            Some("public feed disconnected; retrying")
        } else if self.bbo.is_none() {
            Some("waiting for a fresh BBO")
        } else {
            None
        };
        self.resumed_after_ms = match (was_paused, self.pause_reason.is_some()) {
            (false, true) => {
                self.paused_since_ns = now_ns;
                None
            }
            (true, false) => Some(now_ns.saturating_sub(self.paused_since_ns) / 1_000_000),
            _ => None,
        };
        connection_changed || (!was_paused && self.pause_reason.is_some())
    }
}

pub struct GridRunLock {
    file: std::fs::File,
    _path: PathBuf,
}

impl GridRunLock {
    pub fn acquire(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)?;
        let path = root.join(".grid.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        file.try_lock_exclusive()
            .with_context(|| format!("another dry-run grid owns {}", path.display()))?;
        Ok(Self { file, _path: path })
    }
}

impl Drop for GridRunLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Sparse overrides applied on top of the base config.
///
/// Every field is optional and `None` means "inherit". A variant that sets
/// nothing is the grid's base config, which is what makes `baseline` a
/// meaningful control rather than a separately-maintained copy. Since
/// 2026-09-10 that base is no longer the shipped live configuration: the live
/// profile moved to `sweep1_flat300` and the base deliberately stayed put.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VariantOverrides {
    /// `model.q_max` — inventory cap. Replay and early live runs disagreed on
    /// whether lowering it helped, so the grid retains it as a measured axis.
    pub q_max: Option<i64>,
    pub horizon_seconds: Option<f64>,
    pub parameter_profile: Option<String>,
    /// `model.phi_kappa_t` — running inventory penalty. Higher pushes back to
    /// flat harder.
    pub phi_kappa_t: Option<f64>,
    /// `model.phi_kappa_t_max` — the ceiling that `hjb.rs` rescales φ against.
    /// Without raising this, a `phi_kappa_t` above the base ceiling (450) is
    /// silently clamped; `phi1000` therefore raises both fields.
    pub phi_kappa_t_max: Option<f64>,
    /// `quoting.min_half_spread_bps` — floor on each quoted side's depth.
    pub min_half_spread_bps: Option<f64>,
    /// Lot-age exit deadline before decision/acknowledgement latency. Zero
    /// disables it; see `docs/CAUSAL_EXECUTION_REVIEW.md`.
    ///
    /// Paper-only as a GRID override -- a variant carrying it is refused by
    /// promotion -- but the policy itself now exists live as
    /// `live.flatten_after_ms`, which the shipped profile sets.
    pub flatten_after_ms: Option<u64>,
    /// `quoting.min_order_lifetime_ms` — requote cadence. Its interaction with
    /// spread width is non-monotone, so slow rows are controls rather than a
    /// presumed improvement.
    pub min_order_lifetime_ms: Option<u64>,
    /// `quoting.reduce_only_threshold_q`; must not exceed the variant q-range.
    pub reduce_only_threshold_q: Option<f64>,
    /// `quoting.replace_threshold_bps` — requote hold window, the other half of
    /// cadence.
    pub replace_threshold_bps: Option<f64>,
    /// `flow_guard.enabled` — the toxic-flow guard. Exposing it as a lever is
    /// what makes a guarded/unguarded A/B on one shared live feed possible.
    pub flow_guard_enabled: Option<bool>,
    /// `flow_guard.vpin_threshold`.
    pub vpin_threshold: Option<f64>,
    /// `flow_guard.fast_move_threshold_bps`.
    pub fast_move_threshold_bps: Option<f64>,
}

impl VariantOverrides {
    /// Apply to a base config, then validate.
    ///
    /// Each variant is validated independently: an override can push a config
    /// into a combination the base never had (a hold window wider than the
    /// widest permitted quote, a requote rate that cannot fit the message
    /// budget), and that must fail at startup rather than halfway through a
    /// multi-hour run.
    pub fn apply(&self, base: &AppConfig) -> Result<AppConfig> {
        let mut config = base.clone();
        if let Some(value) = self.q_max {
            config.model.q_max = value;
        }
        if let Some(value) = self.horizon_seconds {
            config.model.horizon_seconds = value;
        }
        if let Some(value) = self.phi_kappa_t {
            config.model.phi_kappa_t = value;
        }
        if let Some(value) = self.phi_kappa_t_max {
            config.model.phi_kappa_t_max = value;
        }
        if let Some(value) = self.min_half_spread_bps {
            config.quoting.min_half_spread_bps = value;
        }
        if let Some(value) = self.flatten_after_ms {
            config.dry_run.flatten_after_ms = value;
        }
        if let Some(value) = self.min_order_lifetime_ms {
            config.quoting.min_order_lifetime_ms = value;
        }
        if let Some(value) = self.reduce_only_threshold_q {
            config.quoting.reduce_only_threshold_q = value;
        }
        if let Some(value) = self.replace_threshold_bps {
            config.quoting.replace_threshold_bps = value;
        }
        if let Some(value) = self.flow_guard_enabled {
            config.flow_guard.enabled = value;
        }
        if let Some(value) = self.vpin_threshold {
            config.flow_guard.vpin_threshold = value;
        }
        if let Some(value) = self.fast_move_threshold_bps {
            config.flow_guard.fast_move_threshold_bps = value;
        }
        config.validate()?;
        Ok(config)
    }

    /// Human-readable summary of what this variant changes, for the leaderboard.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(value) = self.q_max {
            parts.push(format!("q_max={value}"));
        }
        if let Some(value) = self.horizon_seconds {
            parts.push(format!("T={value}s"));
        }
        if let Some(value) = &self.parameter_profile {
            parts.push(format!("parameters={value}"));
        }
        if let Some(value) = self.phi_kappa_t {
            parts.push(format!("phiKT={value}"));
        }
        if let Some(value) = self.phi_kappa_t_max {
            parts.push(format!("phiKTmax={value}"));
        }
        if let Some(value) = self.min_half_spread_bps {
            parts.push(format!("minHalf={value}bps"));
        }
        if let Some(value) = self.flatten_after_ms {
            parts.push(format!("flatten={value}ms"));
        }
        if let Some(value) = self.min_order_lifetime_ms {
            parts.push(format!("lifetime={value}ms"));
        }
        if let Some(value) = self.reduce_only_threshold_q {
            parts.push(format!("reduceAt={value}q"));
        }
        if let Some(value) = self.replace_threshold_bps {
            parts.push(format!("hold={value}bps"));
        }
        if let Some(value) = self.flow_guard_enabled {
            parts.push(format!("guard={value}"));
        }
        if let Some(value) = self.vpin_threshold {
            parts.push(format!("vpin={value}"));
        }
        if let Some(value) = self.fast_move_threshold_bps {
            parts.push(format!("fastMove={value}bps"));
        }
        if parts.is_empty() {
            "shipped defaults".to_owned()
        } else {
            parts.join(" ")
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VariantSpec {
    pub name: String,
    #[serde(flatten)]
    pub overrides: VariantOverrides,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GridSpec {
    #[serde(default)]
    pub parameter_profiles: BTreeMap<String, CjParameters>,
    #[serde(rename = "variant")]
    pub variants: Vec<VariantSpec>,
}

impl GridSpec {
    pub fn resolve_variant(
        &self,
        entry: &VariantSpec,
        base: &AppConfig,
    ) -> Result<(AppConfig, Option<CjParameters>, String)> {
        let config = entry.overrides.apply(base)?;
        let parameters = entry
            .overrides
            .parameter_profile
            .as_ref()
            .map(|name| {
                self.parameter_profiles
                    .get(name)
                    .copied()
                    .with_context(|| format!("unknown parameter profile {name:?}"))
            })
            .transpose()?;
        let mut fingerprint = config.fingerprint()?;
        fingerprint.push_str(";execution=");
        fingerprint.push_str(EXECUTION_REVISION);
        if let Some(parameters) = parameters {
            parameters.validate()?;
            fingerprint.push_str(";parameters=");
            fingerprint.push_str(&serde_json::to_string(&parameters)?);
        }
        Ok((config, parameters, fingerprint))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read grid spec {}", path.display()))?;
        let spec: Self = toml::from_str(&text)
            .with_context(|| format!("cannot parse grid spec {}", path.display()))?;
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<()> {
        if self.variants.is_empty() {
            bail!("grid spec defines no variants");
        }
        for (name, parameters) in &self.parameter_profiles {
            parameters
                .validate()
                .with_context(|| format!("invalid parameter profile {name:?}"))?;
        }
        let mut seen = BTreeSet::new();
        for variant in &self.variants {
            if let Some(profile) = &variant.overrides.parameter_profile {
                if !self.parameter_profiles.contains_key(profile) {
                    bail!(
                        "grid variant {:?} references unknown parameter profile {profile:?}",
                        variant.name
                    );
                }
            }
            if variant.name.trim().is_empty() {
                bail!("grid variant names must be non-empty");
            }
            // Names become directory and file names, and they key the
            // leaderboard, so a duplicate would silently overwrite a peer's
            // report and produce a leaderboard that quietly lost a row.
            if !seen.insert(variant.name.clone()) {
                bail!("duplicate grid variant name {:?}", variant.name);
            }
            if !variant
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                bail!(
                    "grid variant name {:?} must be alphanumeric, '-' or '_'",
                    variant.name
                );
            }
        }
        Ok(())
    }
}

/// A frozen paper control, with the live operator's capital and safeguards.
pub struct LiveStrategy {
    pub config: AppConfig,
    pub parameters: CjParameters,
    pub reference_unit: i64,
    pub reference_capital: f64,
}

impl LiveStrategy {
    pub fn scaled_unit(&self, capital: f64) -> Result<i64> {
        let unit = (self.reference_unit as f64 * capital / self.reference_capital).floor() as i64;
        if unit <= 0 {
            bail!("live allocation is too small for one proportional inventory unit");
        }
        Ok(unit)
    }
}

pub fn live_strategy(config: &AppConfig) -> Result<Option<LiveStrategy>> {
    let Some(source) = &config.live.paper_strategy else {
        return Ok(None);
    };
    let base = AppConfig::load(&source.base_config)?;
    if base.live.paper_strategy.is_some() {
        bail!("paper strategy must reference the grid base, not another linked live profile");
    }
    let spec = GridSpec::load(&source.grid)?;
    let row = spec
        .variants
        .iter()
        .find(|row| row.name == source.variant)
        .with_context(|| format!("unknown paper strategy {:?}", source.variant))?;
    let (paper, parameters, fingerprint) = spec.resolve_variant(row, &base)?;
    let parameters =
        parameters.context("live paper reference must select a frozen parameter profile")?;
    let state = PersistedGridState::load(&source.checkpoint, |state| {
        state.validate_variants()?;
        if state.symbol != config.instrument.symbol || state.symbol != paper.instrument.symbol {
            bail!("paper strategy and live instrument differ");
        }
        let saved = state
            .variants
            .iter()
            .find(|row| row.name == source.variant)
            .context("paper strategy is missing from the existing checkpoint")?;
        // The full fingerprint also contains host paths. Compare the saved fit
        // itself so a Docker checkpoint can be inspected from another host path.
        let encoded = saved
            .config_fingerprint
            .split_once(";parameters=")
            .context("paper checkpoint does not identify its frozen parameters")?
            .1;
        if serde_json::from_str::<CjParameters>(encoded)? != parameters {
            bail!("configured frozen fit differs from the running paper checkpoint");
        }
        Ok(())
    })?;
    let reference_unit = state
        .variants
        .iter()
        .find(|row| row.name == source.variant)
        .context("validated paper row disappeared")?
        .inventory_unit;
    let capital = config.quoting.available_capital_usdc;
    let reference_capital = paper.quoting.available_capital_usdc;
    if reference_capital <= 0.0 || capital <= 0.0 {
        bail!("strategy capital must be positive");
    }
    let scale = capital / reference_capital;
    let mut resolved = config.clone();
    resolved.model = paper.model;
    resolved.calibration = paper.calibration;
    resolved.flow_guard = paper.flow_guard;
    resolved.quoting = paper.quoting;
    resolved.quoting.available_capital_usdc = capital;
    resolved.quoting.leverage = config.quoting.leverage;
    resolved.risk = paper.risk;
    resolved.risk.kill_switch |= config.risk.kill_switch;
    resolved.risk.max_notional_usdc = (resolved.risk.max_notional_usdc * scale)
        .min(config.risk.max_notional_usdc)
        .min(capital * resolved.quoting.leverage);
    resolved.risk.max_margin_usdc = (resolved.risk.max_margin_usdc * scale)
        .min(config.risk.max_margin_usdc)
        .min(capital);
    resolved.risk.min_liquidation_buffer_usdc *= scale;
    resolved.risk.max_daily_loss_usdc =
        (resolved.risk.max_daily_loss_usdc * scale).min(config.risk.max_daily_loss_usdc);
    resolved.live.flatten_after_ms = paper.dry_run.flatten_after_ms;
    resolved.live.strategy_fingerprint =
        Some(format!("{fingerprint};inventory_unit={reference_unit}"));
    resolved.validate()?;
    Ok(Some(LiveStrategy {
        config: resolved,
        parameters,
        reference_unit,
        reference_capital,
    }))
}

/// One row of the leaderboard.
///
/// Ordered by executable-side, fee-adjusted flatten P&L. This prevents a large
/// directional inventory from winning merely because the market moved with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardRow {
    pub name: String,
    pub description: String,
    pub net_pnl_usdc: f64,
    pub promotion_pnl_usdc: Option<f64>,
    pub equity_usdc: f64,
    pub realized_pnl_usdc: f64,
    pub mark_to_market_pnl_usdc: f64,
    pub fees_usdc: f64,
    pub funding_usdc: f64,
    pub inventory_units: i64,
    pub fills: u64,
    pub working_orders: usize,
    pub max_drawdown_usdc: f64,
    #[serde(default)]
    pub config_changes: u32,
    pub scientifically_valid: bool,
    /// Why the row was disqualified, from the variant's own failure or the
    /// backend diagnostics; `None` while it is valid.
    ///
    /// The board is what gets read, and `scientifically_valid: false` alone
    /// cannot tell a variant that blew through its liquidation buffer from one
    /// that merely printed a bad number -- a different scientific claim.
    #[serde(default)]
    pub invalid_reason: Option<String>,
    /// False for invalid rows and for dry-run-only policies with no live
    /// equivalent, such as `flatten_after_ms > 0`.
    pub eligible_for_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leaderboard {
    pub generated_at_ms: u64,
    pub started_at_ms: u64,
    pub elapsed_seconds: u64,
    pub symbol: String,
    /// Feed counters for this run so far, recomputed on every write so a run
    /// killed before teardown still leaves an honest artifact.
    pub feed_health: FeedHealth,
    /// How long the feed has been down *at this instant*; 0 when it is up.
    ///
    /// Folded into `feed_health` as well, but kept separate because it answers
    /// a different question: `feed_health` is "was this run's evidence any
    /// good", this is "is it blind right now". The container healthcheck reads
    /// this one -- a liveness check alone passes happily through a blackout,
    /// since the process stays healthy and keeps rewriting this very file.
    pub feed_down_for_ms: u64,
    #[serde(default)]
    pub quote_pause_reason: Option<String>,
    /// How many times this run has been resumed from a checkpoint.
    ///
    /// Non-zero means the numbers below span more than one process lifetime.
    /// Visible because a stitched run is a different object from a continuous
    /// one and a reader must not have to guess which they are holding.
    pub resumes: u32,
    /// Total wall time the grid was *not running* across those resumes.
    ///
    /// Not counted inside `feed_health.downtime_ms` — that budget measures
    /// blindness *while quoting*, and a stopped process quotes nothing — and
    /// kept separately because the two have
    /// different risk. A feed gap means the grid was quoting into the dark; a
    /// restart gap means it was not quoting at all.
    pub resumed_downtime_ms: u64,
    /// Promotable rows first, ordered by executable-side flatten P&L.
    pub rows: Vec<LeaderboardRow>,
    /// Present only when this board came from `replay`; absent for a live grid.
    ///
    /// Check it first. The two boards are deliberately the same shape so rows
    /// can be compared field for field, which is exactly what makes them easy
    /// to confuse. It also warns that `feed_health` above is not a measurement
    /// here: a replay consumes a tape slice and cannot observe a gap in it, so
    /// those counters are zero meaning "not measured", not "the feed was
    /// perfect".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<ReplayWindow>,
}

/// What a replay board scored: the window, the split and the assumed latency.
///
/// Enough for another run to be reproduced or refused as incomparable. A live
/// row and a replay row of the same variant differ by the tape window, the
/// latency assumption and whether the run was stitched across restarts; the
/// first two are here and the third is `resumes` above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayWindow {
    pub training_start_ms: f64,
    pub training_end_ms: f64,
    pub scoring_start_ms: f64,
    pub scoring_end_ms: f64,
    pub train_fraction: f64,
    /// The decision/acknowledgement/cancel latency every variant assumed.
    ///
    /// Load-bearing for the flatten family: the exit deadline is
    /// `flatten_after_ms + decision + acknowledgement`, so a latency rung
    /// retunes the strategy rather than merely handicapping it.
    pub latency_ms: u64,
}

/// One variant's accounting, checkpointed so a restart can carry it forward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVariant {
    pub name: String,
    /// Guards against resuming a variant whose parameters were edited in the
    /// meantime, which would silently splice two different strategies into one
    /// P&L curve.
    pub config_fingerprint: String,
    /// How many times this row's parameters changed while its history continued.
    #[serde(default)]
    pub config_changes: u32,
    pub inventory_unit: i64,
    pub last_bbo: Option<mm_live::types::Bbo>,
    pub account: mm_live::types::DryRunAccountState,
    pub diagnostics: mm_live::execution::DryRunDiagnostics,
    pub fills: u64,
    pub peak_equity_usdc: f64,
    pub max_drawdown_usdc: f64,
    pub failure: Option<String>,
    pub current_day: Option<u64>,
    pub daily_realized_pnl_usdc: f64,
}

/// The whole grid's accounting at one instant, written every stats tick.
///
/// This exists so a reboot costs a gap rather than the run. Before it, the grid
/// started from zero equity and a zero clock on every launch, so the 2026-08-27
/// Windows-update reboot did not merely interrupt a 46 h measurement -- it
/// meant any relaunch would have discarded it.
///
/// What it deliberately does *not* do is hide the interruption. The wall time
/// between the last checkpoint and the resume is added to feed downtime, and a
/// gap longer than the caller's threshold refuses to resume at all: carrying
/// inventory across an unobserved price move is how the 46 h leaderboard came
/// to report a 13.2% rally as trading profit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedGridState {
    pub schema_version: u32,
    pub symbol: String,
    /// `key=value;...` identity of the run: execution model, estimator schema,
    /// starting equity. Incompatible identities refuse recovery; they never reset it.
    pub grid_fingerprint: String,
    pub run_id: String,
    /// The *original* start, carried across every resume. This is what makes
    /// elapsed time and the downtime fraction continuous.
    pub started_at_ms: u64,
    pub checkpoint_ms: u64,
    pub resumes: u32,
    pub resumed_downtime_ms: u64,
    pub feed_health: FeedHealth,
    pub trade_prints: u64,
    pub replayed_trades_ignored: u64,
    pub variants: Vec<PersistedVariant>,
}

pub const EXECUTION_REVISION: &str = "causal-v5";

impl PersistedGridState {
    pub const SCHEMA_VERSION: u32 = 3;

    /// Recover the current checkpoint or its backup, validating each candidate
    /// before accepting it. Failure never authorizes a new experiment.
    pub fn load(path: &Path, validate: impl Fn(&Self) -> Result<()>) -> Result<Self> {
        let mut failures = Vec::new();
        for candidate in [path.to_owned(), Self::backup_path(path)] {
            match Self::read_one(&candidate).and_then(|state| {
                validate(&state)?;
                Ok(state)
            }) {
                Ok(state) => {
                    if candidate != path {
                        tracing::warn!(path = %candidate.display(), "resuming checkpoint backup");
                    }
                    return Ok(state);
                }
                Err(error) => failures.push(format!("{}: {error:#}", candidate.display())),
            }
        }
        bail!(
            "cannot resume grid; no fresh run will be started: {}",
            failures.join("; ")
        )
    }

    fn read_one(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let state: Self = serde_json::from_slice(&bytes)?;
        if state.schema_version != Self::SCHEMA_VERSION {
            bail!("unsupported checkpoint schema {}", state.schema_version);
        }
        Ok(state)
    }

    pub fn validate_roster(&self, names: &[&str]) -> Result<()> {
        let saved: BTreeSet<_> = self.variants.iter().map(|row| row.name.as_str()).collect();
        if saved != names.iter().copied().collect() || names.len() != self.variants.len() {
            bail!("checkpoint roster differs from the grid; no accounts may be added, removed or reset");
        }
        Ok(())
    }

    /// Refuse corrupt accounting; parameter changes preserve it and are marked.
    pub fn validate_variants(&self) -> Result<()> {
        if self
            .variants
            .iter()
            .map(|entry| &entry.name)
            .collect::<BTreeSet<_>>()
            .len()
            != self.variants.len()
            || self.started_at_ms > self.checkpoint_ms
            || self
                .run_id
                .strip_prefix("run-")
                .and_then(|suffix| suffix.parse::<u64>().ok())
                .is_none()
        {
            bail!("checkpoint has duplicate variants or a bad run ID");
        }
        for entry in &self.variants {
            let name = &entry.name;
            let account = entry.account;
            if entry.inventory_unit <= 0
                || ![
                    account.cash_usdc,
                    account.equity_usdc,
                    account.average_entry_px,
                    account.realized_pnl_usdc,
                    account.fees_usdc,
                    account.funding_usdc,
                    account.mark_to_market_pnl_usdc,
                    account.position_notional_usdc,
                    account.margin_used_usdc,
                    account.maintenance_margin_usdc,
                    account.liquidation_buffer_usdc,
                    entry.daily_realized_pnl_usdc,
                    entry.peak_equity_usdc,
                    entry.max_drawdown_usdc,
                ]
                .iter()
                .all(|value| value.is_finite())
                || entry.last_bbo.is_some_and(|book| !book.is_valid())
                || (account.inventory_units != 0 && entry.last_bbo.is_none())
            {
                bail!("checkpoint variant {name:?} has incompatible or invalid state");
            }
        }
        Ok(())
    }

    fn backup_path(path: &Path) -> std::path::PathBuf {
        path.with_extension("json.bak")
    }

    /// Write the checkpoint, keeping the previous one as `.bak`.
    ///
    /// Two fixed-schema generations, overwritten in place. Their size depends
    /// on variant count and bounded diagnostic maps, not run duration.
    pub fn write_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Demote the current generation before replacing it. Best-effort: on
        // the first write there is nothing to demote, and a failure here must
        // not stop the checkpoint that matters from being written.
        let _ = std::fs::rename(path, Self::backup_path(path));
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Why this checkpoint cannot be resumed into the grid as it now stands, or
    /// `None` if it can.
    #[must_use]
    pub fn rejection(&self, symbol: &str, grid_fingerprint: &str) -> Option<String> {
        if self.symbol != symbol {
            return Some(format!(
                "checkpoint is for {} but this grid trades {symbol}",
                self.symbol
            ));
        }
        // Key-by-key, so a checkpoint written before a key existed still resumes.
        let stored: BTreeMap<&str, &str> = self
            .grid_fingerprint
            .split(';')
            .filter_map(|part| part.split_once('='))
            .collect();
        for (key, value) in grid_fingerprint
            .split(';')
            .filter_map(|part| part.split_once('='))
        {
            // v5 changes prospective controls/execution, not the schema-3 ledger.
            // This is the only supported algorithm transition, not a fresh account.
            if key == "execution"
                && stored.get(key) == Some(&"causal-v4")
                && value == EXECUTION_REVISION
                && self.schema_version == Self::SCHEMA_VERSION
            {
                continue;
            }
            if stored.get(key).is_some_and(|was| *was != value) {
                return Some(format!(
                    "{key} changed since the checkpoint ({} -> {value}); the old accounting has \
                     no meaning under it",
                    stored[key]
                ));
            }
        }
        None
    }
}

/// Append-only equity history — the time axis `leaderboard.json` does not have.
///
/// The leaderboard is rewritten in place every stats tick, so it only ever
/// shows the *current* state. Reconstructing a P&L curve afterwards meant
/// replaying every variant's fill log and joining it against the collector's
/// price tape, which is slow, needs the tape to still exist, and cannot
/// recover a variant's own view of its equity. One CSV row per variant per
/// interval removes all of that.
///
/// Design notes, all of them about surviving a long run:
///
/// - **Append-only, never rewritten.** A crash loses at most the last row; the
///   rest of the file is already durable. The leaderboard's write-temp-then-
///   rename is right for a small snapshot and wrong for a growing log.
/// - **Flushed every write.** The grid is expected to run for days and be
///   killed abruptly; an unflushed `BufWriter` would silently discard the tail.
/// - **CSV, not JSONL.** This is a dense numeric series whose only consumer is
///   a plotting script. CSV is roughly a third the size of the equivalent JSON
///   and `pandas.read_csv` reads it directly.
/// - **Its own interval, coarser than the stats tick.** At the 5 s stats
///   cadence ten variants would write ~15 MB/day; the default 60 s keeps a
///   year under half a gigabyte while still resolving every move that matters
///   at this strategy's timescale.
/// - **`run_started_ms` on every row.** Restarting the grid appends to the same
///   file, so a consumer needs to be able to tell two runs apart; `elapsed_s`
///   alone resets and would silently splice them.
/// - **`mid` on every row.** Without it the curve cannot be plotted against
///   price unless the tape is still on disk, which retention does not
///   guarantee.
#[derive(Debug)]
pub struct EquityHistory {
    writer: BufWriter<std::fs::File>,
    interval_ms: u64,
    last_write_ms: u64,
}

impl EquityHistory {
    pub const HEADER: &'static str = "ts_ms,run_started_ms,elapsed_s,variant,net_pnl_usdc,equity_usdc,realized_pnl_usdc,fees_usdc,funding_usdc,inventory_units,fills,working_orders,max_drawdown_usdc,mid,valid";

    /// Open (or create) the history file. An existing file is appended to, not
    /// truncated, so a restart extends the same series.
    pub fn create(path: &Path, interval_ms: u64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("cannot open equity history {}", path.display()))?;
        let is_new = file.metadata().map(|meta| meta.len() == 0).unwrap_or(true);
        let mut writer = BufWriter::new(file);
        if is_new {
            writeln!(writer, "{}", Self::HEADER)?;
            writer.flush()?;
        }
        Ok(Self {
            writer,
            interval_ms,
            last_write_ms: 0,
        })
    }

    /// Write one row per variant when the interval has elapsed. Returns whether
    /// anything was written, so callers can log at the same cadence.
    pub fn record(&mut self, board: &Leaderboard, mid: Option<f64>) -> Result<bool> {
        if board.generated_at_ms.saturating_sub(self.last_write_ms) < self.interval_ms {
            return Ok(false);
        }
        self.force_record(board, mid)
    }

    /// Write a sample regardless of the interval. Used at shutdown so the curve
    /// ends at the run's true end.
    pub fn force_record(&mut self, board: &Leaderboard, mid: Option<f64>) -> Result<bool> {
        self.last_write_ms = board.generated_at_ms;
        for row in &board.rows {
            // A variant name is validated against a strict character set at
            // spec load, so it can never contain a comma or a quote.
            writeln!(
                self.writer,
                "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{},{}",
                board.generated_at_ms,
                board.started_at_ms,
                board.elapsed_seconds,
                row.name,
                row.net_pnl_usdc,
                row.equity_usdc,
                row.realized_pnl_usdc,
                row.fees_usdc,
                row.funding_usdc,
                row.inventory_units,
                row.fills,
                row.working_orders,
                row.max_drawdown_usdc,
                mid.map_or_else(String::new, |value| format!("{value:.10}")),
                u8::from(row.scientifically_valid),
            )?;
        }
        self.writer.flush()?;
        Ok(true)
    }
}

impl Leaderboard {
    /// Rank by promotion P&L alone, best first, unscored rows last.
    ///
    /// This sorted every `eligible_for_promotion` row ahead of every
    /// ineligible one until 2026-09-10, because `promote-best` read `rows[0]`
    /// and needed the first row to be promotable. Nothing selects on that flag
    /// now, and it is false for every flatten row, so the board simply opened
    /// with whichever promotable row happened to exist: the 2026-09-10 archive
    /// heads its table with `baseline` at -193.50, above `sweep1_flat300` at
    /// +953.05. The flag stays on the row as a fact about it. It is not a rank.
    pub fn sort_by_promotion_pnl(&mut self) {
        self.rows.sort_by(|a, b| {
            b.promotion_pnl_usdc
                .partial_cmp(&a.promotion_pnl_usdc)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn write_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Append one sample per variant to the equity history.
    pub fn append_history(&self, history: &mut EquityHistory, mid: Option<f64>) -> Result<bool> {
        history.record(self, mid)
    }

    /// Fixed-width table for the terminal.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "\n{} grid — {} variants, {}s elapsed (ranked by flatten P&L)\n",
            self.symbol,
            self.rows.len(),
            self.elapsed_seconds
        );
        out.push_str(
            "variant          exit value  net P&L   realized   fills     inv     maxDD  overrides\n",
        );
        for row in &self.rows {
            let _ = writeln!(
                out,
                "{:<12} {:>10} {:>10.4} {:>10.4} {:>7} {:>7} {:>9.4}  {}{}",
                row.name,
                row.promotion_pnl_usdc
                    .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.4}")),
                row.net_pnl_usdc,
                row.realized_pnl_usdc,
                row.fills,
                row.inventory_units,
                row.max_drawdown_usdc,
                row.description,
                if !row.scientifically_valid {
                    "  [STOPPED MARK]"
                } else if row.eligible_for_promotion {
                    ""
                } else {
                    "  [INELIGIBLE]"
                },
            );
        }
        if self.resumes > 0 {
            let _ = writeln!(
                out,
                "\n  [RESUMED] {} restart(s), {:.1} min not running — these totals span more \
                 than one process lifetime",
                self.resumes,
                self.resumed_downtime_ms as f64 / 60_000.0
            );
        }
        let reconfigured: Vec<String> = self
            .rows
            .iter()
            .filter(|row| row.config_changes > 0)
            .map(|row| format!("{} x{}", row.name, row.config_changes))
            .collect();
        if !reconfigured.is_empty() {
            let _ = writeln!(
                out,
                "\n  [RECONFIGURED] parameters changed mid-run, history continued: {}",
                reconfigured.join(", ")
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> AppConfig {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat.toml");
        AppConfig::load(&path).expect("cashcat.toml must load")
    }

    /// The shipped grid spec is now the only record of these parameters.
    ///
    /// It used to be cross-checked against `docs/cashcat_sweep.json`, the
    /// Python sweep's artifact. That engine is gone, so a value pin here would
    /// only compare the config with a copy of itself. What is still worth
    /// asserting is structural: the rows are distinct, every frozen profile
    /// still solves, and the flatten family is exactly `sweep1` plus the two
    /// overrides that define it.
    #[test]
    fn shipped_paper_candidates_are_distinct_and_every_frozen_profile_solves() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let spec = GridSpec::load(&root.join("config/grid_cashcat.toml")).unwrap();
        let config = AppConfig::load(&root.join("config/cashcat_dryrun_realistic.toml")).unwrap();
        assert_eq!(spec.variants.len(), 22);
        let fingerprints: BTreeSet<_> = spec
            .variants
            .iter()
            .map(|entry| spec.resolve_variant(entry, &config).unwrap().2)
            .collect();
        assert_eq!(fingerprints.len(), 22);

        let mut profiled = 0;
        for entry in &spec.variants {
            let (applied, parameters, _) = spec.resolve_variant(entry, &config).unwrap();
            let Some(actual) = parameters else { continue };
            let payload = serde_json::to_string(&actual).unwrap();
            assert!(!payload.contains("price_drift_per_second"));
            assert_eq!(
                serde_json::from_str::<CjParameters>(&payload).unwrap(),
                actual
            );
            profiled += 1;
            for (key, value) in [
                ("lambda+", actual.lambda_plus),
                ("lambda-", actual.lambda_minus),
                ("kappa+", actual.kappa_plus),
                ("kappa-", actual.kappa_minus),
            ] {
                assert!(value.is_finite() && value > 0.0, "{}: {key}", entry.name);
            }
            assert!(actual.epsilon_plus.is_finite() && actual.epsilon_plus >= 0.0);
            assert!(actual.epsilon_minus.is_finite() && actual.epsilon_minus >= 0.0);
            // A frozen fit is not a live calibration, so it must never be
            // offered to promotion; `sigma2` is likewise the calibrator's.
            assert!(actual.sigma2_per_second.is_none(), "{}", entry.name);
            let surface = mm_live::hjb::solve_asymmetric(actual, &applied.model, 306.0, 1)
                .unwrap_or_else(|error| panic!("{}: {error}", entry.name));
            assert!(surface.max_final_residual <= applied.model.newton_tolerance);
        }
        assert!(
            profiled >= 8,
            "expected the sweep and contender rows, got {profiled}"
        );

        let first = spec
            .variants
            .iter()
            .find(|entry| entry.name == "sweep1")
            .unwrap();
        let guarded = spec
            .resolve_variant(first, &config)
            .unwrap()
            .0
            .flow_guard
            .enabled;
        let guard_row = if guarded {
            "sweep1_unguarded"
        } else {
            "sweep1_guarded"
        };
        for (name, spread, deadline, guard) in [
            (guard_row, None, None, !guarded),
            ("sweep1_wide60", Some(60.0), None, guarded),
            ("sweep1_flat300", Some(60.0), Some(1), guarded),
            ("sweep1_flat550", Some(60.0), Some(250), guarded),
        ] {
            let mut expected = first.clone();
            expected.name = name.into();
            expected.overrides.flow_guard_enabled = Some(guard);
            expected.overrides.min_half_spread_bps = spread;
            expected.overrides.flatten_after_ms = deadline;
            let actual = spec
                .variants
                .iter()
                .find(|entry| entry.name == name)
                .unwrap();
            assert_eq!(
                spec.resolve_variant(actual, &config).unwrap().2,
                spec.resolve_variant(&expected, &config).unwrap().2,
                "{name}"
            );
        }
    }

    #[test]
    fn fixed_parameters_participate_in_resume_identity_and_must_validate() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut spec = GridSpec::load(&root.join("config/grid_cashcat.toml")).unwrap();
        let entry = spec
            .variants
            .iter()
            .find(|entry| entry.name == "sweep1")
            .unwrap()
            .clone();
        let before = spec.resolve_variant(&entry, &base()).unwrap().2;
        spec.parameter_profiles
            .get_mut("sweep_a")
            .unwrap()
            .epsilon_plus += 0.000_001;
        assert_ne!(before, spec.resolve_variant(&entry, &base()).unwrap().2);
        spec.parameter_profiles
            .get_mut("sweep_a")
            .unwrap()
            .kappa_plus = 0.0;
        assert!(spec.validate().is_err());
        assert!(spec.resolve_variant(&entry, &base()).is_err());
        spec.parameter_profiles.remove("sweep_a");
        assert!(spec.validate().is_err());
        assert!(spec.resolve_variant(&entry, &base()).is_err());
    }

    fn healthy_feed() -> FeedHealth {
        FeedHealth::new(0, 0, 0, 3_600_000, false)
    }

    #[test]
    fn sparse_overrides_land_on_the_right_fields_and_leave_the_rest_alone() {
        let config = base();
        let overrides = VariantOverrides {
            q_max: Some(2),
            reduce_only_threshold_q: Some(2.0),
            min_half_spread_bps: Some(4.0),
            ..VariantOverrides::default()
        };
        let applied = overrides.apply(&config).unwrap();
        assert_eq!(applied.model.q_max, 2);
        assert!((applied.quoting.min_half_spread_bps - 4.0).abs() < f64::EPSILON);
        // Untouched fields inherit, rather than silently resetting to defaults.
        assert!(
            (applied.model.phi_kappa_t - config.model.phi_kappa_t).abs() < f64::EPSILON,
            "phi_kappa_t must inherit when not overridden"
        );
        assert_eq!(
            applied.quoting.min_order_lifetime_ms,
            config.quoting.min_order_lifetime_ms
        );
    }

    #[test]
    fn an_empty_override_set_reproduces_the_base_configuration() {
        let config = base();
        let applied = VariantOverrides::default().apply(&config).unwrap();
        assert_eq!(
            applied.fingerprint().unwrap(),
            config.fingerprint().unwrap()
        );
    }

    #[test]
    fn a_variant_that_violates_validation_is_refused_at_parse_time() {
        let config = base();
        // A hold window wider than the widest permitted quote is nonsense, and
        // validate() rejects it. The point is that an override can create a
        // combination the base never had.
        let overrides = VariantOverrides {
            replace_threshold_bps: Some(config.quoting.max_half_spread_bps + 1.0),
            ..VariantOverrides::default()
        };
        assert!(overrides.apply(&config).is_err());
    }

    #[test]
    fn duplicate_and_malformed_variant_names_are_refused() {
        let duplicate =
            toml::from_str::<GridSpec>("[[variant]]\nname = \"a\"\n\n[[variant]]\nname = \"a\"\n")
                .unwrap();
        assert!(duplicate.validate().is_err());

        let malformed = toml::from_str::<GridSpec>("[[variant]]\nname = \"a b/c\"\n").unwrap();
        assert!(malformed.validate().is_err());

        let empty = toml::from_str::<GridSpec>("variant = []\n").unwrap();
        assert!(empty.validate().is_err());
    }

    #[test]
    fn leaderboard_ranks_by_flatten_pnl_and_puts_invalid_rows_last() {
        let row = |name: &str, pnl: f64| LeaderboardRow {
            name: name.to_owned(),
            description: String::new(),
            net_pnl_usdc: pnl,
            promotion_pnl_usdc: Some(pnl),
            equity_usdc: 0.0,
            realized_pnl_usdc: 0.0,
            mark_to_market_pnl_usdc: 0.0,
            fees_usdc: 0.0,
            funding_usdc: 0.0,
            inventory_units: 0,
            fills: 0,
            working_orders: 0,
            max_drawdown_usdc: 0.0,
            config_changes: 0,
            scientifically_valid: true,
            invalid_reason: None,
            eligible_for_promotion: true,
        };
        let mut board = Leaderboard {
            generated_at_ms: 0,
            started_at_ms: 0,
            elapsed_seconds: 0,
            symbol: "CASHCAT".to_owned(),
            feed_health: healthy_feed(),
            feed_down_for_ms: 0,
            quote_pause_reason: None,
            resumes: 0,
            resumed_downtime_ms: 0,
            rows: vec![
                row("a", -1.0),
                row("b", 2.0),
                row("c", 0.5),
                row("unscored", 0.0),
            ],
            replay: None,
        };
        // Ineligibility is not a demerit: `b` is every flatten row, which has
        // no live equivalent and is still the best measurement on the board.
        board.rows[1].eligible_for_promotion = false;
        // Disqualified rows have no executable exit price, so they cannot be
        // ranked against rows that do -- last, not worst.
        board.rows[3].promotion_pnl_usdc = None;
        board.rows[3].scientifically_valid = false;
        board.sort_by_promotion_pnl();
        let order: Vec<&str> = board.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(order, vec!["b", "c", "a", "unscored"]);
    }

    #[test]
    fn market_pause_requires_post_reconnect_fresh_data() {
        let mut market = PaperMarketState::default();
        let mut book = Bbo {
            bid_px: 99,
            ask_px: 101,
            bid_sz: 1,
            ask_sz: 1,
            exchange_ms: 1_000,
            recv_ns: 1_000_000,
        };
        assert!(market.observe(None, 0, 1_000, 0, false, 5_000));
        assert!(market.pause_reason.is_some());
        market.observe(
            Some(&MarketEvent::Bbo(book)),
            1_000_000,
            1_000,
            0,
            false,
            5_000,
        );
        assert!(market.pause_reason.is_none());
        assert!(market.observe(None, 2_000_000, 1_001, 0, true, 5_000));
        assert!(market.bbo.is_none());
        market.observe(
            Some(&MarketEvent::Bbo(book)),
            3_000_000,
            1_002,
            3_000_000,
            false,
            5_000,
        );
        assert!(market.bbo.is_none());
        book.recv_ns = 3_000_000;
        book.exchange_ms = 1_002;
        market.observe(
            Some(&MarketEvent::Bbo(book)),
            3_000_000,
            1_002,
            3_000_000,
            false,
            5_000,
        );
        assert!(market.pause_reason.is_none());
        assert!(market.observe(None, 6_000_000_000, 7_000, 3_000_000, false, 5_000));
        assert!(market.bbo.is_none());
        book.recv_ns = 6_100_000_000;
        market.observe(
            Some(&MarketEvent::Bbo(book)),
            6_100_000_000,
            7_001,
            3_000_000,
            false,
            5_000,
        );
        assert!(
            market.bbo.is_none(),
            "fresh delivery of an old quote must not resume trading"
        );
        book.exchange_ms = 7_001;
        market.observe(
            Some(&MarketEvent::Bbo(book)),
            6_100_000_000,
            7_001,
            3_000_000,
            false,
            5_000,
        );
        assert!(market.pause_reason.is_none());
        book.recv_ns = 7_000_000_000;
        book.exchange_ms = 8_000;
        market.observe(
            Some(&MarketEvent::Bbo(book)),
            book.recv_ns + 1,
            8_000,
            book.recv_ns,
            false,
            5_000,
        );
        assert!(
            market.pause_reason.is_none(),
            "the first fresh BBO must recover even if a short gap happened between timer ticks"
        );
    }

    fn board_at(now_ms: u64, pnl: f64) -> Leaderboard {
        Leaderboard {
            generated_at_ms: now_ms,
            started_at_ms: 1_000,
            elapsed_seconds: (now_ms - 1_000) / 1_000,
            symbol: "CASHCAT".to_owned(),
            feed_health: healthy_feed(),
            feed_down_for_ms: 0,
            quote_pause_reason: None,
            resumes: 0,
            resumed_downtime_ms: 0,
            rows: vec![LeaderboardRow {
                name: "wide8".to_owned(),
                description: "minHalf=8bps".to_owned(),
                net_pnl_usdc: pnl,
                promotion_pnl_usdc: Some(pnl - 0.5),
                equity_usdc: 297.88 + pnl,
                realized_pnl_usdc: pnl,
                mark_to_market_pnl_usdc: pnl,
                fees_usdc: 1.5,
                funding_usdc: 0.0,
                inventory_units: 640,
                fills: 12,
                working_orders: 2,
                max_drawdown_usdc: 3.25,
                config_changes: 0,
                scientifically_valid: true,
                invalid_reason: None,
                eligible_for_promotion: true,
            }],
            replay: None,
        }
    }

    #[test]
    fn equity_history_writes_a_header_once_and_then_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equity_history.csv");
        let mut history = EquityHistory::create(&path, 60_000).unwrap();
        assert!(history
            .record(&board_at(61_000, -1.0), Some(0.1234))
            .unwrap());
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], EquityHistory::HEADER);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].split(',').count(), lines[1].split(',').count());
        assert!(
            lines[1].starts_with("61000,1000,60,wide8,"),
            "got {}",
            lines[1]
        );
        assert!(lines[1].ends_with(",0.1234000000,1"), "got {}", lines[1]);
    }

    #[test]
    fn equity_history_respects_its_interval() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equity_history.csv");
        let mut history = EquityHistory::create(&path, 60_000).unwrap();
        assert!(history.record(&board_at(60_000, -1.0), None).unwrap());
        // Too soon: silently skipped, not an error.
        assert!(!history.record(&board_at(90_000, -2.0), None).unwrap());
        assert!(history.record(&board_at(120_000, -3.0), None).unwrap());
        // ... but shutdown always gets its sample.
        assert!(history
            .force_record(&board_at(130_000, -4.0), None)
            .unwrap());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text.lines().count(),
            4,
            "header + the two samples an interval apart + the forced one"
        );
    }

    #[test]
    fn equity_history_appends_across_restarts_without_a_second_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equity_history.csv");
        {
            let mut first = EquityHistory::create(&path, 0).unwrap();
            first.record(&board_at(2_000, -1.0), None).unwrap();
        }
        // A restart must extend the same series: the file is opened for append,
        // and the header is written only when the file is new.
        let mut second = EquityHistory::create(&path, 0).unwrap();
        second.record(&board_at(3_000, -2.0), None).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines
                .iter()
                .filter(|l| **l == EquityHistory::HEADER)
                .count(),
            1
        );
    }

    #[test]
    fn equity_history_leaves_mid_empty_when_the_book_is_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equity_history.csv");
        let mut history = EquityHistory::create(&path, 0).unwrap();
        history.record(&board_at(2_000, -1.0), None).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        // An empty field, never a zero: a zero mid would plot as a real price.
        assert!(text.lines().nth(1).unwrap().ends_with(",,1"));
    }

    fn checkpoint() -> PersistedGridState {
        PersistedGridState {
            schema_version: PersistedGridState::SCHEMA_VERSION,
            symbol: "CASHCAT".to_owned(),
            grid_fingerprint: "execution=causal-v4;estimator=v5:direct".to_owned(),
            run_id: "run-1000".to_owned(),
            started_at_ms: 1_000,
            checkpoint_ms: 3_600_000,
            resumes: 0,
            resumed_downtime_ms: 0,
            feed_health: FeedHealth::new(2, 500, 400, 3_599_000, false),
            trade_prints: 9_000,
            replayed_trades_ignored: 30,
            variants: Vec::new(),
        }
    }

    #[test]
    fn live_reuses_the_paper_fit_and_scales_its_unit_without_relaxing_live_caps() {
        use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
        use mm_live::types::QuoteReason;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut live = AppConfig::load(&root.join("config/cashcat.toml")).unwrap();
        let source = live.live.paper_strategy.as_ref().unwrap();
        let base = AppConfig::load(&source.base_config).unwrap();
        let spec = GridSpec::load(&source.grid).unwrap();
        let row = spec
            .variants
            .iter()
            .find(|row| row.name == "sweep1_flat300")
            .unwrap();
        let (paper, parameters, fingerprint) = spec.resolve_variant(row, &base).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.json");
        let mut saved = checkpoint();
        saved.variants.push(PersistedVariant {
            name: row.name.clone(),
            config_fingerprint: fingerprint,
            config_changes: 3,
            inventory_unit: 636,
            last_bbo: None,
            account: mm_live::types::DryRunAccountState::default(),
            diagnostics: mm_live::execution::DryRunDiagnostics::default(),
            fills: 7,
            peak_equity_usdc: 297.88,
            max_drawdown_usdc: 1.0,
            failure: None,
            current_day: None,
            daily_realized_pnl_usdc: 0.0,
        });
        saved.write_atomic(&path).unwrap();
        live.live.paper_strategy.as_mut().unwrap().checkpoint = path.clone();
        let before = std::fs::read(&path).unwrap();
        let resolved = live_strategy(&live).unwrap().unwrap();
        assert_eq!(resolved.parameters, parameters.unwrap());
        assert_eq!(resolved.scaled_unit(100.0).unwrap(), 213);
        assert_eq!(resolved.scaled_unit(50.0).unwrap(), 106);
        assert!(resolved.scaled_unit(0.0).is_err());
        assert!(!resolved.config.live.enabled);
        assert_eq!(resolved.config.live.flatten_after_ms, 1);
        assert_eq!(resolved.config.risk.max_daily_loss_usdc, 1.0);
        assert_eq!(resolved.config.risk.max_notional_usdc, 200.0);
        assert_eq!(resolved.config.risk.max_margin_usdc, 100.0);
        assert_eq!(
            resolved.config.live.address_action_reserve,
            live.live.address_action_reserve
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let instrument = mm_live::InstrumentSpec {
            symbol: "CASHCAT".to_owned(),
            dex: String::new(),
            asset_id: 231,
            sz_decimals: 0,
            max_price_decimals: 6,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 10.0,
            margin_table_id: 3,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        };
        let paper_policy = CarteaJaimungalPolicy::new(
            instrument.clone(),
            paper.quoting.clone(),
            paper.risk.clone(),
        )
        .unwrap();
        let live_policy = CarteaJaimungalPolicy::new(
            instrument.clone(),
            resolved.config.quoting.clone(),
            resolved.config.risk.clone(),
        )
        .unwrap();
        let surface =
            mm_live::hjb::solve_asymmetric(resolved.parameters, &paper.model, 636.0, 1).unwrap();
        let book = Bbo {
            bid_px: 169_900,
            ask_px: 170_100,
            bid_sz: 1_000,
            ask_sz: 1_000,
            exchange_ms: 1,
            recv_ns: 1,
        };
        for q in -3..=3 {
            let decision = |policy: &CarteaJaimungalPolicy, unit, equity| {
                policy
                    .compute(
                        &surface,
                        book,
                        q * unit,
                        unit,
                        75.0,
                        1,
                        1,
                        QuoteReason::Market,
                        RiskState {
                            equity_usdc: equity,
                            ..RiskState::default()
                        },
                    )
                    .quotes
            };
            let a = decision(&paper_policy, 636, 297.88);
            let b = decision(&live_policy, 213, 100.0);
            assert_eq!(a.bid.map(|o| o.px), b.bid.map(|o| o.px));
            assert_eq!(a.ask.map(|o| o.px), b.ask.map(|o| o.px));
        }
        saved.variants[0].config_fingerprint = "missing frozen fit".to_owned();
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        assert!(live_strategy(&live).is_err());
    }

    #[test]
    fn every_variant_must_validate_before_resume() {
        let mut state = checkpoint();
        assert!(
            state.validate_variants().is_ok(),
            "nothing to refuse in an empty checkpoint"
        );
        state.variants.push(PersistedVariant {
            name: "wide8".to_owned(),
            config_fingerprint: "abc".to_owned(),
            config_changes: 0,
            inventory_unit: 10,
            last_bbo: None,
            account: mm_live::types::DryRunAccountState::default(),
            diagnostics: mm_live::execution::DryRunDiagnostics::default(),
            fills: 0,
            peak_equity_usdc: 1_000.0,
            max_drawdown_usdc: 0.0,
            failure: None,
            current_day: None,
            daily_realized_pnl_usdc: 0.0,
        });
        assert!(state.validate_variants().is_ok());
        state.variants[0].inventory_unit = 0;
        assert!(state.validate_variants().is_err());
        state.variants[0].inventory_unit = 10;
        state.variants[0].account.inventory_units = 1;
        assert!(state.validate_variants().is_err());
        state.variants[0].account.inventory_units = 0;
        state.variants.push(state.variants[0].clone());
        assert!(state.validate_variants().is_err());
    }

    #[test]
    fn missing_feed_health_never_becomes_a_clean_history() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("grid_state.json");
        let mut state = checkpoint();
        state.feed_health.event_loss = true;
        state.write_atomic(&path).unwrap();
        state.write_atomic(&path).unwrap();
        let mut incomplete = serde_json::to_value(&state).unwrap();
        incomplete["feed_health"]
            .as_object_mut()
            .unwrap()
            .remove("event_loss");
        std::fs::write(&path, serde_json::to_vec(&incomplete).unwrap()).unwrap();
        assert!(
            PersistedGridState::load(&path, |_| Ok(()))
                .unwrap()
                .feed_health
                .event_loss
        );
        std::fs::write(
            PersistedGridState::backup_path(&path),
            serde_json::to_vec(&incomplete).unwrap(),
        )
        .unwrap();
        assert!(PersistedGridState::load(&path, |_| Ok(())).is_err());
    }

    #[test]
    fn a_matching_checkpoint_is_resumable() {
        let state = checkpoint();
        assert!(state
            .rejection("CASHCAT", "execution=causal-v4;estimator=v5:direct")
            .is_none());
        // A key the checkpoint predates is not a change.
        assert!(state
            .rejection(
                "CASHCAT",
                "execution=causal-v4;estimator=v5:direct;starting_equity=250"
            )
            .is_none());
    }

    /// A different execution model or
    /// estimator schema makes the old accounting meaningless. Retuned
    /// parameters resume and are counted per row instead.
    #[test]
    fn a_changed_identity_is_not_resumable() {
        let reason = checkpoint()
            .rejection("CASHCAT", "execution=causal-v6;estimator=v5:direct")
            .expect("a new execution model must be refused");
        assert!(reason.contains("execution changed"), "{reason}");
    }

    #[test]
    fn only_the_known_ledger_compatible_execution_upgrade_is_allowed() {
        let state = checkpoint();
        assert!(state
            .rejection("CASHCAT", "execution=causal-v5;estimator=v5:direct")
            .is_none());
        assert!(state
            .rejection("CASHCAT", "execution=causal-v5;estimator=v6:direct")
            .is_some());
        let mut upgraded = state;
        upgraded.grid_fingerprint = "execution=causal-v5;starting_equity=297.88".to_owned();
        assert!(upgraded
            .rejection("CASHCAT", "execution=causal-v4")
            .is_some());
        assert!(upgraded
            .rejection("CASHCAT", "execution=causal-v5;starting_equity=1000")
            .is_some());
    }

    #[test]
    fn a_checkpoint_from_another_instrument_is_not_resumable() {
        let reason = checkpoint()
            .rejection("ETH", "execution=causal-v4;estimator=v5:direct")
            .expect("a different symbol must be refused");
        assert!(reason.contains("CASHCAT"), "{reason}");
    }

    #[test]
    fn a_checkpoint_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        checkpoint().write_atomic(&path).unwrap();
        let loaded = PersistedGridState::load(&path, |_| Ok(())).expect("must reload");
        assert_eq!(loaded.started_at_ms, 1_000);
        assert_eq!(loaded.feed_health.downtime_ms, 500);
        assert_eq!(loaded.trade_prints, 9_000);
    }

    #[test]
    fn event_loss_is_sticky_across_a_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        let mut state = checkpoint();
        state.feed_health.event_loss = true;
        state.write_atomic(&path).unwrap();
        assert!(
            PersistedGridState::load(&path, |_| Ok(()))
                .expect("checkpoint")
                .feed_health
                .event_loss
        );
    }

    #[test]
    fn output_directory_lock_refuses_a_second_grid() {
        let dir = tempfile::tempdir().unwrap();
        let first = GridRunLock::acquire(dir.path()).unwrap();
        assert!(GridRunLock::acquire(dir.path()).is_err());
        drop(first);
        assert!(GridRunLock::acquire(dir.path()).is_ok());
    }

    /// A half-written or stale-schema checkpoint must refuse recovery, never reset a
    /// long-running experiment.
    #[test]
    fn a_corrupt_checkpoint_refuses_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        std::fs::write(&path, b"{\"schema_version\":1,\"symbol\":").unwrap();
        assert!(PersistedGridState::load(&path, |_| Ok(())).is_err());
        assert!(PersistedGridState::load(&dir.path().join("absent.json"), |_| Ok(())).is_err());
    }

    /// The reason the checkpoint keeps one previous generation: a file torn by
    /// something outside this process — a full disk, power loss mid-write —
    /// should cost one stats interval, not the whole run.
    #[test]
    fn a_corrupt_checkpoint_falls_back_to_the_previous_generation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        let mut first = checkpoint();
        first.checkpoint_ms = 1_000;
        first.write_atomic(&path).unwrap();
        let mut second = checkpoint();
        second.checkpoint_ms = 2_000;
        second.write_atomic(&path).unwrap();
        // The live generation is now unreadable; the `.bak` still holds the one
        // before it.
        std::fs::write(&path, b"torn").unwrap();
        let loaded = PersistedGridState::load(&path, |_| Ok(())).expect("must fall back to .bak");
        assert_eq!(loaded.checkpoint_ms, 1_000);
    }

    /// Two fixed-size files, overwritten in place — the checkpoint is bounded,
    /// not something that accumulates with run length.
    #[test]
    fn checkpointing_repeatedly_leaves_exactly_two_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        for tick in 0..50_u64 {
            let mut state = checkpoint();
            state.checkpoint_ms = tick;
            state.write_atomic(&path).unwrap();
        }
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "grid_state.json".to_owned(),
                "grid_state.json.bak".to_owned()
            ]
        );
    }

    #[test]
    fn a_resumed_board_says_so_in_the_render() {
        let mut board = board_at(2_000, 1.0);
        board.resumes = 2;
        board.resumed_downtime_ms = 600_000;
        let text = board.render();
        assert!(text.contains("[RESUMED] 2 restart(s), 10.0 min"), "{text}");
        assert!(!board_at(2_000, 1.0).render().contains("[RESUMED]"));
    }
}
