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
    connected_ns: u64,
    ready_after_ns: u64,
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
/// nothing is the shipped configuration, which is what makes `baseline` a
/// meaningful control rather than a separately-maintained copy.
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
    /// Lot-age exit deadline before decision/acknowledgement latency.
    /// Zero disables this paper-only policy; see `docs/CAUSAL_EXECUTION_REVIEW.md`.
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
    pub scientifically_valid: bool,
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
    /// The feed's verdict on this run *so far*, recomputed on every write.
    ///
    /// It used to be evaluated only in the end-of-run teardown, and applied
    /// only to the per-variant `SessionReport`s. That made `leaderboard.json`
    /// -- the file that actually gets read and quoted into the write-ups --
    /// incapable of ever saying a run was disqualified. The 2026-08-27 run was
    /// killed by a host reboot before teardown and left 18 rows all claiming
    /// `scientifically_valid: true` after 42.5% downtime, with a longest gap
    /// 1,179x over the limit.
    ///
    /// A run can always die before its teardown, so the artifact that is
    /// rewritten continuously has to be the honest one.
    pub feed_health: FeedHealth,
    /// Why the feed disqualifies this run; empty means it does not.
    pub feed_failures: Vec<String>,
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
}

/// One variant's accounting, checkpointed so a restart can carry it forward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVariant {
    pub name: String,
    /// Guards against resuming a variant whose parameters were edited in the
    /// meantime, which would silently splice two different strategies into one
    /// P&L curve.
    pub config_fingerprint: String,
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
    /// Identity of the whole grid: every variant name and config fingerprint.
    /// A changed spec -- a variant added, removed or retuned -- starts fresh.
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

impl PersistedGridState {
    pub const SCHEMA_VERSION: u32 = 3;

    /// Read a checkpoint, falling back to the previous generation.
    ///
    /// A missing file is the ordinary first-run case, and a corrupt one is
    /// treated the same way rather than fatally: a bad checkpoint must cost a
    /// fresh run, not a grid that will not start.
    ///
    /// The `.bak` fallback is what makes that cost one stats interval instead
    /// of the whole run. `write_atomic` already rules out a torn file — the
    /// rename is atomic within a directory — but not a file torn by something
    /// outside this process: a full disk, a killed container mid-rename, a
    /// half-synced bind mount on a host that lost power. Those are exactly the
    /// circumstances this whole mechanism exists for.
    pub fn load(path: &Path) -> Option<Self> {
        Self::read_one(path).or_else(|| Self::read_one(&Self::backup_path(path)))
    }

    fn read_one(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let state: Self = match serde_json::from_slice(&bytes) {
            Ok(state) => state,
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "checkpoint is unreadable; trying backup or starting fresh");
                return None;
            }
        };
        if state.schema_version != Self::SCHEMA_VERSION {
            tracing::warn!(
                schema_version = state.schema_version,
                "checkpoint execution schema changed; starting fresh"
            );
            return None;
        }
        Some(state)
    }

    pub fn validate_variants(&self, expected: &[(&str, &str)]) -> Result<()> {
        if self.variants.len() != expected.len()
            || self
                .variants
                .iter()
                .map(|entry| &entry.name)
                .collect::<BTreeSet<_>>()
                .len()
                != expected.len()
            || self.started_at_ms > self.checkpoint_ms
            || self
                .run_id
                .strip_prefix("run-")
                .and_then(|suffix| suffix.parse::<u64>().ok())
                .is_none()
        {
            bail!("checkpoint has an incompatible variant set or run ID");
        }
        for (name, fingerprint) in expected {
            let entry = self
                .variants
                .iter()
                .find(|entry| entry.name == *name)
                .with_context(|| format!("checkpoint has no variant {name:?}"))?;
            let account = entry.account;
            if entry.config_fingerprint != *fingerprint
                || entry.inventory_unit <= 0
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
        if self.grid_fingerprint != grid_fingerprint {
            return Some(
                "grid spec or base config changed since the checkpoint; the variants are not \
                 the same strategies"
                    .to_owned(),
            );
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
    pub fn sort_by_promotion_pnl(&mut self) {
        self.rows.sort_by(
            |a, b| match (a.eligible_for_promotion, b.eligible_for_promotion) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b
                    .promotion_pnl_usdc
                    .partial_cmp(&a.promotion_pnl_usdc)
                    .unwrap_or(std::cmp::Ordering::Equal),
            },
        );
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
            "variant          flat P&L    net P&L   realized   fills     inv     maxDD  overrides\n",
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
                if row.eligible_for_promotion {
                    ""
                } else {
                    "  [INELIGIBLE]"
                },
            );
        }
        // Printed under the table rather than per row: the reason is the same
        // for every variant, and a reader who scrolls past a screenful of tags
        // still needs to be told what disqualified them.
        for reason in &self.feed_failures {
            let _ = writeln!(out, "\n  [FEED INVALID] {reason}");
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

    #[test]
    fn shipped_paper_candidates_match_saved_models_without_duplicate_rows() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let spec = GridSpec::load(&root.join("config/grid_cashcat.toml")).unwrap();
        let config = AppConfig::load(&root.join("config/cashcat_dryrun_realistic.toml")).unwrap();
        let sweep: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("../docs/cashcat_sweep.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(spec.variants.len(), 22);
        let fingerprints: BTreeSet<_> = spec
            .variants
            .iter()
            .map(|entry| spec.resolve_variant(entry, &config).unwrap().2)
            .collect();
        assert_eq!(fingerprints.len(), 22);
        let finalists = sweep["stage_c"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, entry)| (format!("sweep{}", index + 1), entry));
        let contenders = sweep["paper_contenders"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| (entry["name"].as_str().unwrap().to_owned(), entry));
        for (name, expected) in finalists.chain(contenders) {
            let entry = spec
                .variants
                .iter()
                .find(|entry| entry.name == name)
                .unwrap();
            let (applied, parameters, _) = spec.resolve_variant(entry, &config).unwrap();
            let expected_parameters = &expected["params"];
            let actual = parameters.unwrap();
            for (key, value) in [
                ("lambda+", actual.lambda_plus),
                ("lambda-", actual.lambda_minus),
                ("kappa+", actual.kappa_plus),
                ("kappa-", actual.kappa_minus),
                ("epsilon+", actual.epsilon_plus),
                ("epsilon-", actual.epsilon_minus),
            ] {
                let expected_value = expected_parameters[key].as_f64().unwrap();
                assert!(
                    (value - expected_value).abs() <= 2.0 * f64::EPSILON * expected_value.abs(),
                    "{name}: {key}"
                );
            }
            assert!(actual.sigma2_per_second.is_none());
            let surface = mm_live::hjb::solve_asymmetric(actual, &applied.model, 306.0, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(surface.max_final_residual <= applied.model.newton_tolerance);
            assert_eq!(
                applied.model.q_max,
                expected["risk"]["q_max"].as_i64().unwrap()
            );
            assert_eq!(
                applied.model.horizon_seconds,
                expected["risk"]["horizon_seconds"].as_f64().unwrap()
            );
            assert_eq!(
                applied.model.phi_kappa_t,
                expected["risk"]["phi_kappa_t"].as_f64().unwrap()
            );
            assert_eq!(
                applied.model.phi_kappa_t_max,
                expected["risk"]["phi_kappa_t_max"]
                    .as_f64()
                    .unwrap_or(applied.model.phi_kappa_t)
            );
            assert_eq!(
                applied.model.alpha_kappa,
                expected["risk"]["alpha_kappa"].as_f64().unwrap()
            );
            assert_eq!(
                applied.flow_guard.enabled,
                expected["risk"]["flow_guard"].as_bool().unwrap()
            );
            if let Some(spread) = expected["risk"]["min_half_spread_bps"].as_f64() {
                assert_eq!(applied.quoting.min_half_spread_bps, spread);
            }
            if let Some(deadline) = expected["risk"]["flatten_after_ms"].as_u64() {
                assert_eq!(entry.overrides.flatten_after_ms, Some(deadline));
            }
        }
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
            scientifically_valid: true,
            eligible_for_promotion: true,
        };
        let mut board = Leaderboard {
            generated_at_ms: 0,
            started_at_ms: 0,
            elapsed_seconds: 0,
            symbol: "CASHCAT".to_owned(),
            feed_health: healthy_feed(),
            feed_failures: Vec::new(),
            feed_down_for_ms: 0,
            quote_pause_reason: None,
            resumes: 0,
            resumed_downtime_ms: 0,
            rows: vec![row("a", -1.0), row("b", 2.0), row("c", 0.5)],
        };
        board.rows[1].eligible_for_promotion = false;
        board.sort_by_promotion_pnl();
        let order: Vec<&str> = board.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(order, vec!["c", "a", "b"]);
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
            feed_failures: Vec::new(),
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
                scientifically_valid: true,
                eligible_for_promotion: true,
            }],
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
            grid_fingerprint: "wide8=abc;wide16=def".to_owned(),
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
    fn every_variant_must_validate_before_resume() {
        let mut state = checkpoint();
        let expected = [("wide8", "abc")];
        assert!(state.validate_variants(&expected).is_err());
        state.variants.push(PersistedVariant {
            name: "wide8".to_owned(),
            config_fingerprint: "abc".to_owned(),
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
        assert!(state.validate_variants(&expected).is_ok());
        assert!(state.validate_variants(&[("wide8", "retuned")]).is_err());
        state.variants[0].inventory_unit = 0;
        assert!(state.validate_variants(&expected).is_err());
        state.variants[0].inventory_unit = 10;
        state.variants[0].account.inventory_units = 1;
        assert!(state.validate_variants(&expected).is_err());
        state.variants[0].account.inventory_units = 0;
        state.variants.push(state.variants[0].clone());
        assert!(state
            .validate_variants(&[("wide8", "abc"), ("wide8", "abc")])
            .is_err());
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
            PersistedGridState::load(&path)
                .unwrap()
                .feed_health
                .event_loss
        );
        std::fs::write(
            PersistedGridState::backup_path(&path),
            serde_json::to_vec(&incomplete).unwrap(),
        )
        .unwrap();
        assert!(PersistedGridState::load(&path).is_none());
    }

    #[test]
    fn a_matching_checkpoint_is_resumable() {
        assert!(checkpoint()
            .rejection("CASHCAT", "wide8=abc;wide16=def")
            .is_none());
    }

    /// The trap this guards: editing a variant and restarting would otherwise
    /// splice two different strategies into one P&L curve, with nothing in the
    /// output saying so.
    #[test]
    fn an_edited_grid_spec_is_not_resumable() {
        let reason = checkpoint()
            .rejection("CASHCAT", "wide8=abc;wide16=CHANGED")
            .expect("a retuned grid must be refused");
        assert!(reason.contains("not the same strategies"), "{reason}");
    }

    #[test]
    fn a_checkpoint_from_another_instrument_is_not_resumable() {
        let reason = checkpoint()
            .rejection("ETH", "wide8=abc;wide16=def")
            .expect("a different symbol must be refused");
        assert!(reason.contains("CASHCAT"), "{reason}");
    }

    #[test]
    fn a_checkpoint_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        checkpoint().write_atomic(&path).unwrap();
        let loaded = PersistedGridState::load(&path).expect("must reload");
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
            PersistedGridState::load(&path)
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

    /// A half-written or stale-schema checkpoint must cost a fresh run, never a
    /// grid that refuses to start.
    #[test]
    fn a_corrupt_checkpoint_reads_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid_state.json");
        std::fs::write(&path, b"{\"schema_version\":1,\"symbol\":").unwrap();
        assert!(PersistedGridState::load(&path).is_none());
        assert!(PersistedGridState::load(&dir.path().join("absent.json")).is_none());
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
        let loaded = PersistedGridState::load(&path).expect("must fall back to .bak");
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
