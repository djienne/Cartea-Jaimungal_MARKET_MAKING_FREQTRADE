//! Offline scoring: run the paper simulator over a Parquet window instead of a
//! live feed.
//!
//! Deliberately separate from the dry-run grid in `main.rs`. The grid is a
//! long-lived service driven by one shared WebSocket; this is a command that is
//! called from time to time and exits. What the two share is `PaperVariant` --
//! the pricing and accounting of a single parameter set -- so that a row scored
//! here and a row scored live are the same object, produced by the same code,
//! and can be compared field for field. That comparison is the point: it is the
//! only way to ask whether an effect seen in the grid survives a different
//! window, or a different latency assumption.
//!
//! What it cannot answer is whether historical venue fills would have matched
//! paper fills. See `docs/DRY_RUN_GRID.md` for the fidelity limits that apply
//! to every number this module writes.

use anyhow::{bail, Context, Result};
use mm_live::calibration::{CalibrationSnapshot, Calibrator};
use mm_live::config::{AppConfig, FeedHealth};
use mm_live::execution::{AccountStateProvider, DryRunBackend, ExecutionBackend, MarketDataSource};
use mm_live::flow_guard::{FlowGuard, MidWindow, VpinTracker};
use mm_live::hjb::solve_asymmetric;
use mm_live::latency::LatencySnapshot;
use mm_live::metrics::Metrics;
use mm_live::parquet_io::{load_market_window, MarketDataSet};
use mm_live::quote::CarteaJaimungalPolicy;
use mm_live::replay::ParquetReplaySource;
use mm_live::report::{
    JsonlEventLogger, LogBackpressure, LogFormat, LogRotation, ModelReport, ReplayInputs,
};
use mm_live::types::{unix_ms, Bbo, MarketEvent, QuoteReason};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tracing::{info, warn};

use crate::grid;
use crate::paper::{step_paper_variant, PaperVariant};
use crate::{event_ms, vpin_bucket_units, write_report};

/// Everything the `replay` subcommand was asked for, after argument parsing.
pub(crate) struct Request<'a> {
    /// Stem for the per-variant `SessionReport`s.
    pub report: Option<&'a Path>,
    /// Where the leaderboard-shaped summary goes.
    pub board: Option<&'a Path>,
    pub train_fraction: f64,
    pub grid_path: Option<&'a Path>,
    /// Explicit variant names. Empty with `all_variants` false means the base
    /// configuration, which is only valid without a grid.
    pub variants: Vec<String>,
    pub all_variants: bool,
    pub range: Option<(u64, u64)>,
    /// One rung each. Empty means the configuration's own latency.
    pub latencies: Vec<u64>,
    /// A live `leaderboard.json` to score against and compare with.
    pub against_live: Option<&'a Path>,
}

pub(crate) async fn run(
    base: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    request: Request<'_>,
) -> Result<()> {
    let started_at_ms = unix_ms();

    // The live board, when given, supplies three things the caller would
    // otherwise copy out by hand and get wrong: the window that run actually
    // covered, the set of variants it ran, and the numbers to print beside the
    // replay's own.
    let live = request
        .against_live
        .map(|path| -> Result<grid::Leaderboard> {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading live leaderboard {}", path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parsing live leaderboard {}", path.display()))
        })
        .transpose()?;
    if let Some(board) = &live {
        if board.replay.is_some() {
            bail!(
                "--against-live was handed a replay board, not a live one; \
                 comparing a replay against a replay measures nothing"
            );
        }
        if board.resumes > 0 {
            warn!(
                resumes = board.resumes,
                resumed_downtime_ms = board.resumed_downtime_ms,
                "the live run was stitched across restarts; the replay is one continuous pass over the same window"
            );
        }
    }
    let range = request.range.or_else(|| {
        live.as_ref()
            .map(|board| (board.started_at_ms, board.generated_at_ms))
    });

    let spec = request.grid_path.map(grid::GridSpec::load).transpose()?;
    let names = resolve_variants(&request, spec.as_ref(), live.as_ref())?;

    // Load the tape once and reuse it. `VariantOverrides` reaches only into
    // `[model]`, `[quoting]`, `[flow_guard]` and `dry_run.flatten_after_ms`, so
    // every variant would otherwise re-read and re-split byte-identical data --
    // which on a 24-day CASHCAT window is most of the cost of the run.
    let data = load_market_window(
        &base.storage.data_dir,
        &instrument.symbol,
        &base.calibration,
        range,
    )?;
    let (training, scoring) = data.split_for_replay(request.train_fraction)?;
    info!(
        variants = names.len(),
        rungs = request.latencies.len().max(1),
        training_start_ms = training.window_start_ms,
        scoring_start_ms = scoring.window_start_ms,
        scoring_end_ms = scoring.window_end_ms,
        "backtest window loaded"
    );

    // Same argument: `[calibration]` cannot be overridden, so the fit is shared
    // by every variant that does not carry a frozen `parameter_profile`.
    let needs_calibration = names.iter().any(|name| match (spec.as_ref(), name) {
        (Some(spec), Some(name)) => spec
            .variants
            .iter()
            .find(|entry| &entry.name == name)
            .is_none_or(|entry| entry.overrides.parameter_profile.is_none()),
        _ => true,
    });
    let shared_snapshot = if needs_calibration {
        let candidate =
            Calibrator::new(&instrument.symbol, base.calibration.clone()).calibrate(&training)?;
        if !candidate.is_quotable() {
            bail!(
                "replay training calibration failed closed: {:?}",
                candidate.status
            );
        }
        Some(candidate)
    } else {
        None
    };

    // Not overridable either, so it is one number for the whole run.
    let bucket_units =
        vpin_bucket_units(&training, &instrument, base.flow_guard.vpin_buckets_per_day);

    let rungs: Vec<Option<u64>> = if request.latencies.is_empty() {
        vec![None]
    } else {
        request.latencies.iter().copied().map(Some).collect()
    };
    let report_stem = request.report.map_or_else(
        || {
            base.storage
                .report_dir
                .join(format!("replay-{started_at_ms}"))
        },
        strip_json_suffix,
    );

    let mut boards = Vec::new();
    for latency in rungs {
        let mut rows = Vec::new();
        for name in &names {
            rows.push(
                score_one(
                    base,
                    &instrument,
                    spec.as_ref(),
                    name.as_deref(),
                    latency,
                    shared_snapshot.as_ref(),
                    &training,
                    &scoring,
                    bucket_units,
                    &report_stem,
                    started_at_ms,
                )
                .await?,
            );
        }
        let latency_ms = latency.unwrap_or(base.dry_run.decision_latency_ms);
        let scored_ms = (scoring.window_end_ms - scoring.window_start_ms).max(0.0) as u64;
        let mut board = grid::Leaderboard {
            generated_at_ms: unix_ms(),
            started_at_ms: scoring.window_start_ms as u64,
            elapsed_seconds: scored_ms / 1_000,
            symbol: instrument.symbol.clone(),
            // A replay reads a tape slice and cannot see a gap inside it, so
            // these are zeroed and mean "not measured". `replay` below is what
            // tells a reader that.
            feed_health: FeedHealth::new(0, 0, 0, scored_ms, false),
            feed_down_for_ms: 0,
            quote_pause_reason: None,
            resumes: 0,
            resumed_downtime_ms: 0,
            rows,
            replay: Some(grid::ReplayWindow {
                training_start_ms: training.window_start_ms,
                training_end_ms: training.window_end_ms,
                scoring_start_ms: scoring.window_start_ms,
                scoring_end_ms: scoring.window_end_ms,
                train_fraction: request.train_fraction,
                latency_ms,
            }),
        };
        board.sort_by_promotion_pnl();
        boards.push((latency_ms, board));
    }

    write_boards(&request, &report_stem, &boards)?;
    if let Some(live) = &live {
        print_comparison(live, &boards);
    }
    Ok(())
}

/// Which variants to score, in the order they will be reported.
fn resolve_variants(
    request: &Request<'_>,
    spec: Option<&grid::GridSpec>,
    live: Option<&grid::Leaderboard>,
) -> Result<Vec<Option<String>>> {
    let Some(spec) = spec else {
        if request.all_variants || !request.variants.is_empty() {
            bail!("--variant and --all-variants both require --grid");
        }
        return Ok(vec![None]);
    };
    let names: Vec<String> = if request.all_variants {
        spec.variants
            .iter()
            .map(|entry| entry.name.clone())
            .collect()
    } else if !request.variants.is_empty() {
        request.variants.clone()
    } else if let Some(board) = live {
        // Scoring against a live board with no variant list means "every row it
        // has", which is the comparison a reader actually wants.
        board.rows.iter().map(|row| row.name.clone()).collect()
    } else {
        bail!("a grid replay needs --variant, --all-variants or --against-live");
    };
    for name in &names {
        if !spec.variants.iter().any(|entry| &entry.name == name) {
            bail!("unknown replay variant {name:?}");
        }
    }
    Ok(names.into_iter().map(Some).collect())
}

/// Score one variant at one latency and return its leaderboard row.
#[allow(clippy::too_many_arguments)]
async fn score_one(
    base: &AppConfig,
    instrument: &mm_live::InstrumentSpec,
    spec: Option<&grid::GridSpec>,
    variant_name: Option<&str>,
    latency: Option<u64>,
    shared_snapshot: Option<&CalibrationSnapshot>,
    training: &MarketDataSet,
    scoring: &MarketDataSet,
    bucket_units: i64,
    report_stem: &Path,
    started_at_ms: u64,
) -> Result<grid::LeaderboardRow> {
    let (mut config, fixed_parameters, config_fingerprint, description) = match (spec, variant_name)
    {
        (Some(spec), Some(name)) => {
            let entry = spec
                .variants
                .iter()
                .find(|entry| entry.name == name)
                .with_context(|| format!("unknown replay variant {name:?}"))?;
            let (config, parameters, fingerprint) = spec.resolve_variant(entry, base)?;
            (config, parameters, fingerprint, entry.overrides.describe())
        }
        _ => (base.clone(), None, base.fingerprint()?, String::new()),
    };
    // Applied after variant resolution so a rung reaches every variant equally
    // rather than being overwritten by one variant's own settings.
    if let Some(ms) = latency {
        config.dry_run.decision_latency_ms = ms;
        config.dry_run.acknowledgement_latency_ms = ms;
        config.dry_run.cancel_latency_ms = ms;
    }

    let snapshot = if fixed_parameters.is_some() {
        None
    } else {
        shared_snapshot.cloned()
    };
    let parameters = fixed_parameters
        .or_else(|| snapshot.as_ref().map(|value| value.parameters))
        .context("replay has no usable parameters")?;
    let policy = CarteaJaimungalPolicy::new(
        instrument.clone(),
        config.quoting.clone(),
        config.risk.clone(),
    )?;
    let inventory_unit = policy.derive_inventory_unit(
        training.mids.last().context("no training mid")?.mid,
        config.model.q_max,
    )?;
    let surface = solve_asymmetric(
        parameters,
        &config.model,
        instrument.size_from_units(inventory_unit),
        1,
    )?;
    let mut replay = ReplayInputs {
        variant: variant_name.map(str::to_owned),
        time_source: training.time_source,
        scored_until_ms: None,
        training_start_ms: training.window_start_ms,
        training_end_ms: training.window_end_ms,
        scoring_start_ms: scoring.window_start_ms,
        scoring_end_ms: scoring.window_end_ms,
        parameters,
        vpin_bucket_units: bucket_units,
    };
    let slug = run_slug(variant_name, latency);
    let run_root = with_slug(report_stem, &slug);
    let guard_window_ms = config.flow_guard.fast_move_window_ms;
    let mid_capacity = guard_window_ms
        .saturating_mul(200)
        .div_ceil(1_000)
        .clamp(64, 8_192) as usize;
    let mut variant = PaperVariant {
        name: variant_name.unwrap_or("replay").to_owned(),
        description,
        config_fingerprint,
        config_changes: 0,
        fixed_parameters,
        backend: DryRunBackend::new(
            instrument.clone(),
            config.dry_run.clone(),
            config.quoting.clone(),
            config.risk.clone(),
        )?,
        logger: JsonlEventLogger::create_with_rotation(
            &run_root,
            "events",
            LogBackpressure::BlockWhenFull,
            LogFormat::Zstd,
            LogRotation {
                max_bytes: config.storage.live_log_max_mb * 1_024 * 1_024,
                keep: config.storage.live_log_keep,
            },
        )?,
        report_path: run_root.with_extension("json"),
        peak_equity_usdc: config.dry_run.starting_equity_usdc,
        guard: FlowGuard::new(config.flow_guard.clone()),
        mid_window: MidWindow::new(mid_capacity, guard_window_ms),
        config,
        policy,
        surface,
        inventory_unit,
        episode_start_ns: 0,
        quote_seq: 0,
        fills: 0,
        max_drawdown_usdc: 0.0,
        failure: None,
    };
    variant.logger.log("replay_inputs", None, &replay)?;
    let metrics = Arc::new(Metrics::default());
    let result = run_event_source(
        &mut variant,
        ParquetReplaySource::new(scoring, instrument)?,
        &metrics,
        replay.vpin_bucket_units,
    )
    .await;
    let (scored_until, last_bbo) = match &result {
        Ok((scored_until, bbo)) => (Some(*scored_until), *bbo),
        Err(error) => {
            variant.backend.invalidate(&format!("{error:#}"));
            (None, None)
        }
    };
    replay.scored_until_ms = scored_until;
    variant.logger.flush()?;
    write_report(
        &variant.config,
        Some(&variant.report_path),
        "replay",
        started_at_ms,
        instrument.clone(),
        snapshot,
        Some(ModelReport::from_surface(&variant.surface, inventory_unit)),
        LatencySnapshot::empty(
            &variant.config.instrument.symbol,
            started_at_ms,
            &variant.config.latency,
            false,
        ),
        &variant.backend,
        &metrics,
        variant
            .backend
            .diagnostics()
            .invalid_reason
            .iter()
            .cloned()
            .collect(),
        variant.logger.path(),
        0,
        Some(replay),
    )?;
    // Built from the same `PaperVariant` the live grid uses, which is what
    // makes a replay row and a leaderboard row comparable at all.
    Ok(variant.leaderboard_row(last_bbo))
}

/// Drive one variant over an offline event source until the tape runs out.
///
/// Returns the last decision time and the last book seen. The book is what
/// `leaderboard_row` needs to price carried inventory the way the live grid
/// does at the moment it writes its board.
pub(crate) async fn run_event_source<S: MarketDataSource>(
    variant: &mut PaperVariant,
    mut source: S,
    metrics: &Arc<Metrics>,
    vpin_bucket: i64,
) -> Result<(u64, Option<Bbo>)> {
    let mut latest_bbo = None;
    let mut decision_ms = 0;
    let mut vpin = VpinTracker::new(
        vpin_bucket,
        variant.config.flow_guard.vpin_window_buckets as usize,
    );
    let mut vpin_value = None;
    while let Some(event) = source.next_event().await? {
        metrics.market_messages.fetch_add(1, Ordering::Relaxed);
        match &event {
            MarketEvent::Bbo(_) => &metrics.bbo_updates,
            MarketEvent::Trade(_) => &metrics.trade_prints,
            MarketEvent::Book(_) => &metrics.book_updates,
        }
        .fetch_add(1, Ordering::Relaxed);
        decision_ms = decision_ms.max(event_ms(&event));
        if let MarketEvent::Trade(print) = &event {
            vpin_value = vpin.observe(print);
        }
        if let MarketEvent::Bbo(bbo) = &event {
            latest_bbo = Some(*bbo);
        }
        variant
            .logger
            .log("market_event", Some(event_ms(&event)), &event)?;
        if let Some(reason) =
            step_paper_variant(variant, &event, decision_ms, latest_bbo, vpin_value).await?
        {
            metrics.quote_decisions.fetch_add(1, Ordering::Relaxed);
            metrics.quote_publications.fetch_add(1, Ordering::Relaxed);
            if reason == QuoteReason::RiskLimit {
                metrics.risk_refusals.fetch_add(1, Ordering::Relaxed);
            }
        }
        metrics
            .fills
            .store(variant.backend.diagnostics().fills, Ordering::Relaxed);
        metrics.inventory_units.store(
            variant.backend.account_state().inventory_units,
            Ordering::Relaxed,
        );
        variant.observe_equity();
        if !variant.backend.scientifically_valid() {
            break;
        }
    }
    variant.backend.shutdown(decision_ms).await?;
    Ok((decision_ms, latest_bbo))
}

/// `sweep1_flat300` at 150 ms becomes `sweep1_flat300-lat150`.
fn run_slug(variant_name: Option<&str>, latency: Option<u64>) -> String {
    let name = variant_name.unwrap_or("base");
    match latency {
        Some(ms) => format!("{name}-lat{ms}"),
        None => name.to_owned(),
    }
}

fn with_slug(stem: &Path, slug: &str) -> PathBuf {
    let base = stem.file_name().map_or_else(
        || "replay".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    stem.with_file_name(format!("{base}-{slug}"))
}

fn strip_json_suffix(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|ext| ext == "json") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn write_boards(
    request: &Request<'_>,
    report_stem: &Path,
    boards: &[(u64, grid::Leaderboard)],
) -> Result<()> {
    for (latency_ms, board) in boards {
        let stem = request
            .board
            .map_or_else(|| report_stem.to_path_buf(), strip_json_suffix);
        let path = if boards.len() == 1 {
            stem.with_extension("json")
        } else {
            with_slug(&stem, &format!("lat{latency_ms}")).with_extension("json")
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(board)?)?;
        info!(path = %path.display(), latency_ms, rows = board.rows.len(), "backtest leaderboard written");
    }
    Ok(())
}

/// Print the live row beside the replay row for every variant in both.
///
/// Deliberately plain. The two numbers differ for reasons this table cannot
/// show -- a stitched run, a different window, an assumed latency -- so it
/// reports both and leaves the judgement to a reader who has read the fidelity
/// limits in `docs/DRY_RUN_GRID.md`.
fn print_comparison(live: &grid::Leaderboard, boards: &[(u64, grid::Leaderboard)]) {
    for (latency_ms, board) in boards {
        let hours = board.replay.as_ref().map_or(0.0, |window| {
            (window.scoring_end_ms - window.scoring_start_ms) / 3_600_000.0
        });
        println!(
            "\nlive vs replay | {} | latency {latency_ms} ms | replay scored {hours:.2} h",
            live.symbol
        );
        println!(
            "{:<20} {:>12} {:>12} {:>11} {:>12}",
            "variant", "live net", "replay net", "live fills", "replay fills"
        );
        for row in &board.rows {
            let Some(live_row) = live.rows.iter().find(|entry| entry.name == row.name) else {
                continue;
            };
            println!(
                "{:<20} {:>12.2} {:>12.2} {:>11} {:>12}",
                row.name, live_row.net_pnl_usdc, row.net_pnl_usdc, live_row.fills, row.fills
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_live::hjb::CjParameters;
    use mm_live::parquet_io::{MidRecord, ShardStats, TimeSource};
    use std::path::PathBuf;

    fn cashcat() -> mm_live::InstrumentSpec {
        mm_live::InstrumentSpec {
            symbol: "CASHCAT".to_owned(),
            dex: String::new(),
            asset_id: 231,
            sz_decimals: 0,
            max_price_decimals: 6,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 10.0,
            margin_table_id: 3,
            only_isolated: true,
            margin_mode: "strictIsolated".to_owned(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        }
    }

    /// A flatten variant, driven through the backtest path rather than the grid.
    fn flatten_variant(directory: &Path) -> PaperVariant {
        let mut config = AppConfig::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/cashcat_dryrun_realistic.toml"),
        )
        .unwrap();
        // The deadline is `flatten_after_ms` plus the decision and
        // acknowledgement legs, so zeroing the legs makes it exactly 1 ms and
        // the assertion below independent of the shipped latency.
        config.dry_run.flatten_after_ms = 1;
        config.dry_run.decision_latency_ms = 0;
        config.dry_run.acknowledgement_latency_ms = 0;
        config.dry_run.cancel_latency_ms = 0;
        let instrument = cashcat();
        let inventory_unit = 1_000;
        let policy = CarteaJaimungalPolicy::new(
            instrument.clone(),
            config.quoting.clone(),
            config.risk.clone(),
        )
        .unwrap();
        let surface = solve_asymmetric(
            CjParameters {
                lambda_plus: 0.1,
                lambda_minus: 0.1,
                kappa_plus: 1_000.0,
                kappa_minus: 1_000.0,
                epsilon_plus: 0.0,
                epsilon_minus: 0.0,
                sigma2_per_second: None,
            },
            &config.model,
            inventory_unit as f64,
            1,
        )
        .unwrap();
        let backend = DryRunBackend::new(
            instrument,
            config.dry_run.clone(),
            config.quoting.clone(),
            config.risk.clone(),
        )
        .unwrap();
        PaperVariant {
            name: "flatten300".to_owned(),
            description: "minHalf=60bps flatten=1ms".to_owned(),
            config_fingerprint: config.fingerprint().unwrap(),
            config_changes: 0,
            fixed_parameters: None,
            policy,
            surface,
            inventory_unit,
            backend,
            logger: JsonlEventLogger::create(directory, "test", 0).unwrap(),
            report_path: directory.join("test.json"),
            episode_start_ns: 0,
            quote_seq: 0,
            fills: 0,
            peak_equity_usdc: config.dry_run.starting_equity_usdc,
            max_drawdown_usdc: 0.0,
            guard: FlowGuard::new(config.flow_guard.clone()),
            mid_window: MidWindow::new(128, config.flow_guard.fast_move_window_ms),
            failure: None,
            config,
        }
    }

    fn two_book_tape() -> MarketDataSet {
        MarketDataSet {
            symbol: "CASHCAT".to_owned(),
            time_source: TimeSource::Exchange,
            mids: vec![
                MidRecord {
                    ts_ms: 1_000.0,
                    bid: 0.0999,
                    ask: 0.1001,
                    mid: 0.1,
                },
                MidRecord {
                    ts_ms: 60_000.0,
                    bid: 0.0999,
                    ask: 0.1001,
                    mid: 0.1,
                },
            ],
            trades: Vec::new(),
            books: Vec::new(),
            window_start_ms: 1_000.0,
            window_end_ms: 60_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        }
    }

    /// The check that breaks if the row plumbing regresses.
    ///
    /// A backtest exists to be compared against a live leaderboard row, which
    /// only works while a replay produces a *complete* row. Before this path
    /// existed the report carried no `net_pnl_usdc`, no `max_drawdown_usdc` and
    /// no promotion figure, and the Python wrapper that drove replays
    /// reconstructed the first by scraping the TOML for `starting_equity_usdc`.
    /// Losing any of those again would silently reduce the comparison to
    /// eyeballing equity.
    #[tokio::test]
    async fn a_flatten_replay_produces_a_complete_leaderboard_row() {
        let directory = tempfile::tempdir().unwrap();
        let mut variant = flatten_variant(directory.path());
        let starting_equity = variant.config.dry_run.starting_equity_usdc;

        // Seed a position the way a resumed run does, so the lot is already
        // past its deadline and the timed exit is what closes it. This tests
        // the flatten through the backtest path; `dry_run.rs` pins the deadline
        // arithmetic itself.
        let mut account = variant.backend.account_state();
        account.inventory_units = 500;
        account.average_entry_px = 0.1;
        account.cash_usdc = starting_equity - 50.0;
        variant
            .backend
            .restore_from_snapshot(
                account,
                mm_live::execution::DryRunDiagnostics::default(),
                1_000,
                None,
                0.0,
            )
            .unwrap();

        let data = two_book_tape();
        let metrics = Arc::new(Metrics::default());
        let (_, last_bbo) = run_event_source(
            &mut variant,
            ParquetReplaySource::new(&data, &cashcat()).unwrap(),
            &metrics,
            1,
        )
        .await
        .unwrap();

        let diagnostics = variant.backend.diagnostics();
        assert!(
            diagnostics.flatten_events > 0,
            "the timed exit never fired, so this asserts nothing about a flatten row"
        );
        assert!(last_bbo.is_some(), "the row needs a book to price against");

        let row = variant.leaderboard_row(last_bbo);
        let account = variant.backend.account_state();
        assert_eq!(row.net_pnl_usdc, account.equity_usdc - starting_equity);
        assert_eq!(row.fills, variant.fills);
        assert_eq!(row.inventory_units, account.inventory_units);
        assert_eq!(row.max_drawdown_usdc, variant.max_drawdown_usdc);
        assert!(row.max_drawdown_usdc >= 0.0);
        // A paper-only exit policy must never be offered to live promotion,
        // whichever simulator produced the row.
        assert!(!row.eligible_for_promotion);
    }
}
