//! Live dry-run grid: several parameter sets quoting one shared market feed.
//!
//! Running N parameter sets as N `dry-run` processes would open N market
//! `WebSocket`s against a venue limit of ten simultaneous connections per IP —
//! a budget already shared with the three data-collector containers and with
//! any live session (which needs two). Exhausting it would take down data
//! collection, so the grid runs in one process behind one connection instead.
//!
//! What a variant may change is deliberately narrow: the four levers the
//! 161.95 h staged sweep implicated in the one burst window that produced the
//! entire loss (`docs/cashcat_sweep.md`). Everything else — latency, fees,
//! funding, capital — is held identical across variants so the comparison
//! stays like-for-like.

use anyhow::{bail, Context, Result};
use mm_live::config::AppConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Sparse overrides applied on top of the base config.
///
/// Every field is optional and `None` means "inherit". A variant that sets
/// nothing is the shipped configuration, which is what makes `baseline` a
/// meaningful control rather than a separately-maintained copy.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VariantOverrides {
    /// `model.q_max` — inventory cap. The sweep's clearest single effect:
    /// q=3 scored -279.11 against q=6's -585.61 on the same calibration.
    pub q_max: Option<i64>,
    /// `model.phi_kappa_t` — running inventory penalty. Higher pushes back to
    /// flat harder.
    pub phi_kappa_t: Option<f64>,
    /// `quoting.min_half_spread_bps` — floor on quoted depth. The losing window
    /// earned +683.89 of spread across 1,771 fills, about 0.39 per fill, which
    /// did not cover the adverse selection it took.
    pub min_half_spread_bps: Option<f64>,
    /// `quoting.min_order_lifetime_ms` — requote cadence. The only positive rung
    /// of the latency ladder was the 30 s-refresh one.
    pub min_order_lifetime_ms: Option<u64>,
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
        if let Some(value) = self.phi_kappa_t {
            config.model.phi_kappa_t = value;
        }
        if let Some(value) = self.min_half_spread_bps {
            config.quoting.min_half_spread_bps = value;
        }
        if let Some(value) = self.min_order_lifetime_ms {
            config.quoting.min_order_lifetime_ms = value;
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
        if let Some(value) = self.phi_kappa_t {
            parts.push(format!("phiKT={value}"));
        }
        if let Some(value) = self.min_half_spread_bps {
            parts.push(format!("minHalf={value}bps"));
        }
        if let Some(value) = self.min_order_lifetime_ms {
            parts.push(format!("lifetime={value}ms"));
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
    #[serde(rename = "variant")]
    pub variants: Vec<VariantSpec>,
}

impl GridSpec {
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
        let mut seen = BTreeSet::new();
        for variant in &self.variants {
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
/// Ordered by `net_pnl_usdc`. The tail columns are recorded but do not affect
/// the ordering — they are here because the staged sweep showed a single
/// six-hour window can be the entire result, so a variant leading on total
/// while carrying that shape should at least be visible.
#[derive(Debug, Clone, Serialize)]
pub struct LeaderboardRow {
    pub name: String,
    pub description: String,
    pub net_pnl_usdc: f64,
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
}

#[derive(Debug, Clone, Serialize)]
pub struct Leaderboard {
    pub generated_at_ms: u64,
    pub started_at_ms: u64,
    pub elapsed_seconds: u64,
    pub symbol: String,
    /// Ranked by net P&L, best first.
    pub rows: Vec<LeaderboardRow>,
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
    pub fn sort_by_net_pnl(&mut self) {
        self.rows.sort_by(|a, b| {
            b.net_pnl_usdc
                .partial_cmp(&a.net_pnl_usdc)
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
            "\n{} grid — {} variants, {}s elapsed (ranked by net P&L)\n",
            self.symbol,
            self.rows.len(),
            self.elapsed_seconds
        );
        out.push_str(
            "variant           net P&L   realized        mtm   fills     inv     maxDD  overrides\n",
        );
        for row in &self.rows {
            let _ = writeln!(
                out,
                "{:<12} {:>10.4} {:>10.4} {:>10.4} {:>7} {:>7} {:>9.4}  {}{}",
                row.name,
                row.net_pnl_usdc,
                row.realized_pnl_usdc,
                row.mark_to_market_pnl_usdc,
                row.fills,
                row.inventory_units,
                row.max_drawdown_usdc,
                row.description,
                if row.scientifically_valid {
                    ""
                } else {
                    "  [INVALID]"
                },
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
    fn sparse_overrides_land_on_the_right_fields_and_leave_the_rest_alone() {
        let config = base();
        let overrides = VariantOverrides {
            q_max: Some(2),
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
    fn leaderboard_ranks_by_net_pnl_best_first() {
        let row = |name: &str, pnl: f64| LeaderboardRow {
            name: name.to_owned(),
            description: String::new(),
            net_pnl_usdc: pnl,
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
        };
        let mut board = Leaderboard {
            generated_at_ms: 0,
            started_at_ms: 0,
            elapsed_seconds: 0,
            symbol: "CASHCAT".to_owned(),
            rows: vec![row("a", -1.0), row("b", 2.0), row("c", 0.5)],
        };
        board.sort_by_net_pnl();
        let order: Vec<&str> = board.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(order, vec!["b", "c", "a"]);
    }

    fn board_at(now_ms: u64, pnl: f64) -> Leaderboard {
        Leaderboard {
            generated_at_ms: now_ms,
            started_at_ms: 1_000,
            elapsed_seconds: (now_ms - 1_000) / 1_000,
            symbol: "CASHCAT".to_owned(),
            rows: vec![LeaderboardRow {
                name: "wide8".to_owned(),
                description: "minHalf=8bps".to_owned(),
                net_pnl_usdc: pnl,
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
}
