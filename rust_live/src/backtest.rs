//! Offline scoring of all variants through the grid's paper execution path.
//! Collector receipt order and a virtual clock drive shared freshness and
//! restart actions. Reports separate interval results from restored accounting.
//! See `docs/DRY_RUN_GRID.md` for limits of historical state and feed reconstruction.

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
    JsonlEventLogger, LogBackpressure, LogFormat, LogRotation, ModelReport, ReplayDiagnostics,
    ReplayInputs, ReplayPause,
};
use mm_live::types::{unix_ms, Bbo, MarketEvent, QuoteReason};
use std::collections::BTreeMap;
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
    pub scoring_from: Option<u64>,
    pub inventory_units: Vec<String>,
    pub initial_state: Option<&'a Path>,
    pub max_carry_inventory_gap_seconds: u64,
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

    let units = parse_inventory_units(&request.inventory_units, &names)?;
    let initial = request
        .initial_state
        .map(grid::PersistedGridState::read_one)
        .transpose()?;
    if let Some(state) = &initial {
        state.validate_variants()?;
        let fingerprint = format!(
            "execution={};estimator=v{}:{};starting_equity={}",
            grid::EXECUTION_REVISION,
            mm_live::calibration::PARAMETER_SCHEMA_VERSION,
            mm_live::calibration::ESTIMATOR_SEMANTICS,
            base.dry_run.starting_equity_usdc
        );
        if let Some(reason) = state.rejection(&instrument.symbol, &fingerprint) {
            bail!("{reason}");
        }
    }
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
    let (training, scoring) = if let Some(start) = request.scoring_from {
        data.split_at(start as f64)?
    } else {
        data.split_for_replay(request.train_fraction)?
    };
    validate_initial_variants(
        initial.as_ref(),
        &names,
        &units,
        spec.as_ref(),
        base,
        scoring.window_start_ms as u64,
    )?;
    if initial.is_none() {
        warn!("no historical checkpoint: scoring starts flat with cold guards; grid state is not reconstructed");
    }
    if data.orderbook_shards.files_failed > 0 {
        bail!("replay cannot infer data gaps from unreadable L2 shards; repair or snapshot the data first");
    }
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
        let mut execution = BTreeMap::new();
        for name in &names {
            let (row, diagnostics) = score_one(
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
                units.get(name.as_deref().unwrap_or("base")).copied(),
                initial.as_ref(),
                &request,
            )
            .await?;
            execution.insert(row.name.clone(), diagnostics);
            rows.push(row);
        }
        let latency_ms = latency.unwrap_or(base.dry_run.decision_latency_ms);
        let scored_ms = (scoring.window_end_ms - scoring.window_start_ms).max(0.0) as u64;
        let mut board = grid::Leaderboard {
            generated_at_ms: unix_ms(),
            started_at_ms: scoring.window_start_ms as u64,
            elapsed_seconds: scored_ms / 1_000,
            symbol: instrument.symbol.clone(),
            // WebSocket health is not reconstructible; inferred tape pauses
            // are recorded separately in replay.execution.
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
                train_fraction: (training.window_end_ms - training.window_start_ms)
                    / (scoring.window_end_ms - training.window_start_ms),
                execution,
                latency_ms,
            }),
        };
        board.sort_by_promotion_pnl();
        boards.push((latency_ms, board));
    }

    write_boards(&request, &report_stem, &boards)?;
    if let Some(live) = &live {
        print_comparison(
            live,
            &boards,
            request.against_live.context("missing comparison path")?,
            &report_stem,
        )?;
    }
    Ok(())
}

/// Which variants to score, in the order they will be reported.
fn parse_inventory_units(
    values: &[String],
    names: &[Option<String>],
) -> Result<BTreeMap<String, i64>> {
    let mut units = BTreeMap::new();
    for value in values {
        let (name, amount) = value
            .split_once('=')
            .context("inventory unit must be variant=positive_units")?;
        let amount: i64 = amount
            .parse()
            .context("inventory unit must be an integer")?;
        if amount <= 0
            || !names
                .iter()
                .any(|entry| entry.as_deref().unwrap_or("base") == name)
        {
            bail!("invalid inventory unit or unselected variant: {value}");
        }
        if units.insert(name.to_owned(), amount).is_some() {
            bail!("duplicate inventory unit: {name}");
        }
    }
    Ok(units)
}

fn validate_initial_variants(
    state: Option<&grid::PersistedGridState>,
    names: &[Option<String>],
    units: &BTreeMap<String, i64>,
    spec: Option<&grid::GridSpec>,
    base: &AppConfig,
    scoring_start_ms: u64,
) -> Result<()> {
    let Some(state) = state else {
        return Ok(());
    };
    if state.checkpoint_ms > scoring_start_ms {
        bail!("initial checkpoint is from the future relative to scoring start");
    }
    for name in names {
        let key = name.as_deref().unwrap_or("base");
        let row = state
            .variants
            .iter()
            .find(|row| row.name == key)
            .with_context(|| format!("checkpoint is missing {key}"))?;
        if units
            .get(key)
            .is_some_and(|value| *value != row.inventory_unit)
        {
            bail!("inventory override conflicts with checkpoint for {key}");
        }
        let fingerprint = if let (Some(spec), Some(name)) = (spec, name) {
            let entry = spec
                .variants
                .iter()
                .find(|entry| &entry.name == name)
                .context("missing variant")?;
            spec.resolve_variant(entry, base)?.2
        } else {
            base.fingerprint()?
        };
        if row.config_fingerprint != fingerprint {
            bail!("initial checkpoint configuration differs for {key}; use its original configuration");
        }
    }
    Ok(())
}

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
    inventory_override: Option<i64>,
    initial: Option<&grid::PersistedGridState>,
    request: &Request<'_>,
) -> Result<(grid::LeaderboardRow, ReplayDiagnostics)> {
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
    let persisted = initial.and_then(|state| {
        state
            .variants
            .iter()
            .find(|row| row.name == variant_name.unwrap_or("base"))
    });
    let inventory_unit = if let Some(saved) = persisted {
        saved.inventory_unit
    } else if let Some(value) = inventory_override {
        value
    } else {
        policy.derive_inventory_unit(
            training.mids.last().context("no training mid")?.mid,
            config.model.q_max,
        )?
    };
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
        execution: ReplayDiagnostics::default(),
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
        )?
        .with_exit_settings(
            config.live.emergency_flatten_max_slippage_bps,
            config.runtime.market_stale_ms,
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
    if let Some(saved) = persisted {
        variant.restore(saved)?;
    }
    let initial_account = variant.backend.account_state();
    let initial_fills = variant.fills;
    // Drawdown is over the scored window, even when lifetime accounting is restored.
    variant.peak_equity_usdc = initial_account.equity_usdc;
    variant.max_drawdown_usdc = 0.0;
    let source = ParquetReplaySource::with_freshness(
        scoring,
        instrument,
        variant.config.runtime.market_stale_ms,
    )?;
    replay.execution = ReplayDiagnostics {
        revision: "receipt-flow-v1".into(),
        clock: "collector_receive_time; missing values fall back to exchange_time".into(),
        receive_time_fallbacks: source.receive_time_fallbacks,
        rejected_touches: source.rejected_touches,
        inventory_unit,
        initial_state: request.initial_state.map(|p| p.display().to_string()),
        initial_account,
        initial_fills,
        decision_latency_ms: variant.config.dry_run.decision_latency_ms,
        acknowledgement_latency_ms: variant.config.dry_run.acknowledgement_latency_ms,
        cancel_latency_ms: variant.config.dry_run.cancel_latency_ms,
        tail_latency_multiplier: variant.config.dry_run.tail_latency_multiplier,
        tail_latency_every: variant.config.dry_run.tail_latency_every,
        max_carry_inventory_gap_seconds: request.max_carry_inventory_gap_seconds,
        ..ReplayDiagnostics::default()
    };
    variant.logger.log("replay_inputs", None, &replay)?;
    if let Some(state) = initial {
        if (scoring.window_start_ms as u64).saturating_sub(state.checkpoint_ms)
            > request
                .max_carry_inventory_gap_seconds
                .saturating_mul(1_000)
        {
            if let Some(pnl) = variant.backend.flatten_carried_position()? {
                replay.execution.gap_closes += 1;
                replay.execution.gap_close_pnl_usdc += pnl;
                variant.logger.log(
                    "restart_gap_flattened",
                    Some(scoring.window_start_ms as u64),
                    &pnl,
                )?;
            }
        }
    }
    let metrics = Arc::new(Metrics::default());
    let result = run_timed_event_source(
        &mut variant,
        source,
        &metrics,
        replay.vpin_bucket_units,
        (scoring.window_start_ms as u64, scoring.window_end_ms as u64),
        &mut replay.execution,
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
    replay.execution.scored_until_ms = scored_until;
    let diagnostics = replay.execution.clone();
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
    let mut row = variant.leaderboard_row(last_bbo);
    row.net_pnl_usdc -= initial_account.mark_to_market_pnl_usdc;
    row.mark_to_market_pnl_usdc -= initial_account.mark_to_market_pnl_usdc;
    row.realized_pnl_usdc -= initial_account.realized_pnl_usdc;
    row.fees_usdc -= initial_account.fees_usdc;
    row.funding_usdc -= initial_account.funding_usdc;
    row.fills -= initial_fills;
    row.promotion_pnl_usdc = row
        .promotion_pnl_usdc
        .map(|value| value - initial_account.mark_to_market_pnl_usdc);
    Ok((row, diagnostics))
}

/// Drive one variant over an offline event source until the tape runs out.
///
/// Returns the last decision time and the last book seen. The book is what
/// `leaderboard_row` needs to price carried inventory the way the live grid
/// does at the moment it writes its board.
#[cfg(test)]
pub(crate) async fn run_event_source<S: MarketDataSource>(
    variant: &mut PaperVariant,
    source: S,
    metrics: &Arc<Metrics>,
    bucket: i64,
) -> Result<(u64, Option<Bbo>)> {
    run_timed_event_source(
        variant,
        source,
        metrics,
        bucket,
        (0, u64::MAX),
        &mut ReplayDiagnostics {
            max_carry_inventory_gap_seconds: 900,
            ..ReplayDiagnostics::default()
        },
    )
    .await
}

async fn run_timed_event_source<S: MarketDataSource>(
    variant: &mut PaperVariant,
    mut source: S,
    metrics: &Arc<Metrics>,
    vpin_bucket: i64,
    window: (u64, u64),
    diagnostics: &mut ReplayDiagnostics,
) -> Result<(u64, Option<Bbo>)> {
    let mut decision_ms = 0;
    let mut scored_until = window.0;
    let mut market = grid::PaperMarketState::default();
    let mut vpin = VpinTracker::new(
        vpin_bucket,
        variant.config.flow_guard.vpin_window_buckets as usize,
    );
    let mut vpin_value = None;
    let interval = variant.config.runtime.stats_interval_ms.max(1_000);
    let mut next_tick = window.0;
    if variant.failure.is_some() || !variant.backend.scientifically_valid() {
        return Ok((scored_until, variant.backend.checkpoint_bbo()));
    }
    while let Some(event) = source.next_event().await? {
        let now_ms = event.received_ns() / 1_000_000;
        if now_ms < window.0 {
            continue;
        }
        if now_ms > window.1 {
            break;
        }
        if window.0 == 0 && next_tick == 0 {
            next_tick = now_ms;
        }
        while next_tick <= now_ms {
            replay_observe(variant, &mut market, None, next_tick, diagnostics)?;
            next_tick = next_tick.saturating_add(interval);
        }
        replay_observe(variant, &mut market, Some(&event), now_ms, diagnostics)?;
        scored_until = now_ms;
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
        variant
            .logger
            .log("market_event", Some(event_ms(&event)), &event)?;
        if market.bbo.is_some() {
            if let Some(reason) =
                step_paper_variant(variant, &event, decision_ms, market.bbo, vpin_value).await?
            {
                metrics.quote_decisions.fetch_add(1, Ordering::Relaxed);
                metrics.quote_publications.fetch_add(1, Ordering::Relaxed);
                if reason == QuoteReason::RiskLimit {
                    metrics.risk_refusals.fetch_add(1, Ordering::Relaxed);
                }
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
    if window.1 != u64::MAX && variant.backend.scientifically_valid() {
        while next_tick <= window.1 {
            replay_observe(variant, &mut market, None, next_tick, diagnostics)?;
            next_tick = next_tick.saturating_add(interval);
        }
        scored_until = window.1;
    }
    if let Some(pause) = diagnostics
        .pauses
        .last_mut()
        .filter(|pause| pause.open_at_end)
    {
        pause.end_ms = scored_until;
    }
    variant.backend.shutdown(decision_ms).await?;
    Ok((scored_until, variant.backend.checkpoint_bbo()))
}

fn replay_observe(
    variant: &mut PaperVariant,
    market: &mut grid::PaperMarketState,
    event: Option<&MarketEvent>,
    now_ms: u64,
    diagnostics: &mut ReplayDiagnostics,
) -> Result<()> {
    let was_paused = market.pause_reason.is_some();
    let before = variant.backend.account_state();
    let max_age = variant.config.runtime.market_stale_ms;
    crate::paper::observe_paper_market(
        std::slice::from_mut(variant),
        market,
        event,
        now_ms.saturating_mul(1_000_000),
        now_ms,
        0,
        false,
        max_age,
        diagnostics
            .max_carry_inventory_gap_seconds
            .saturating_mul(1_000),
    )?;
    match (was_paused, market.pause_reason.is_some()) {
        (false, true) => diagnostics.pauses.push(ReplayPause {
            start_ms: now_ms,
            end_ms: now_ms,
            open_at_end: true,
        }),
        (true, false) => {
            if let Some(pause) = diagnostics.pauses.last_mut() {
                pause.end_ms = now_ms;
                pause.open_at_end = false;
            }
            variant
                .logger
                .log("market_data_resumed", Some(now_ms), &now_ms)?;
        }
        _ => {}
    }
    let after = variant.backend.account_state();
    if before.inventory_units != 0 && after.inventory_units == 0 {
        diagnostics.gap_closes += 1;
        diagnostics.gap_close_pnl_usdc += after.realized_pnl_usdc - before.realized_pnl_usdc;
    }
    variant.observe_equity();
    Ok(())
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
fn print_comparison(
    live: &grid::Leaderboard,
    boards: &[(u64, grid::Leaderboard)],
    live_path: &Path,
    report_stem: &Path,
) -> Result<()> {
    // The board is a pointer, not an interval ledger. Require its run history.
    let root = live_path.parent().context("leaderboard has no parent")?;
    let direct = root.join("equity_history.csv");
    let history = if direct.exists() {
        direct
    } else {
        root.join("runs")
            .join(format!("run-{}", live.started_at_ms))
            .join("equity_history.csv")
    };
    if !history.exists() {
        warn!(path = %history.display(), "no matching equity history; refusing cumulative-versus-window comparison");
        return Ok(());
    }
    let text = std::fs::read_to_string(&history)?;
    let mut lines = text.lines();
    let header: Vec<_> = lines
        .next()
        .context("empty equity history")?
        .split(',')
        .collect();
    let column = |key: &str| {
        header
            .iter()
            .position(|value| *value == key)
            .with_context(|| format!("missing {key}"))
    };
    let ti = column("ts_ms")?;
    let ni = column("variant")?;
    let pi = column("net_pnl_usdc")?;
    let fi = column("fills")?;
    let fee = column("fees_usdc")?;
    let ii = column("inventory_units")?;
    let run = column("run_started_ms")?;
    let records: Vec<Vec<&str>> = lines.map(|line| line.split(',').collect()).collect();
    let mut comparison = Vec::new();
    for (latency, board) in boards {
        let window = board.replay.as_ref().context("missing replay window")?;
        println!("Same-window comparison: {}..{}; boundary samples use last observation at/before each bound", window.scoring_start_ms, window.scoring_end_ms);
        for row in &board.rows {
            let Some(scoring_end) = window
                .execution
                .get(&row.name)
                .and_then(|run| run.scored_until_ms)
            else {
                println!("{}: no reliable scored end; no comparison", row.name);
                continue;
            };
            let mut boundaries = [None::<(u64, f64, u64, f64, i64)>; 2];
            for fields in &records {
                if fields.len() != header.len()
                    || fields[ni] != row.name
                    || fields[run].parse::<u64>().ok() != Some(live.started_at_ms)
                {
                    continue;
                }
                let t: u64 = fields[ti].parse()?;
                for (slot, bound) in boundaries
                    .iter_mut()
                    .zip([window.scoring_start_ms, scoring_end as f64])
                {
                    if t as f64 <= bound && slot.as_ref().is_none_or(|old| t > old.0) {
                        *slot = Some((
                            t,
                            fields[pi].parse()?,
                            fields[fi].parse()?,
                            fields[fee].parse()?,
                            fields[ii].parse()?,
                        ));
                    }
                }
            }
            if let [Some(start), Some(end)] = boundaries {
                println!("{} grid_delta={:.6} replay_delta={:.6} grid_fills={} replay_fills={} grid_fees={:.6} replay_fees={:.6} inventory={}..{} boundary_age_ms={:.0},{:.0}",
                    row.name, end.1-start.1, row.net_pnl_usdc, end.2.saturating_sub(start.2), row.fills,
                    end.3-start.3, row.fees_usdc, start.4, end.4, window.scoring_start_ms-start.0 as f64, scoring_end as f64-end.0 as f64);
                let mut replay_log = with_slug(report_stem, &row.name).join("events.jsonl.zst");
                if !replay_log.exists() {
                    replay_log = with_slug(report_stem, &format!("{}-lat{latency}", row.name))
                        .join("events.jsonl.zst");
                }
                let grid_log = history
                    .parent()
                    .context("history parent")?
                    .join(format!("grid-{}.jsonl.zst", row.name));
                let trace = compare_trace_heads(
                    &grid_log,
                    &replay_log,
                    window.scoring_start_ms as u64,
                    scoring_end,
                )
                .unwrap_or_else(|error| serde_json::json!({"unavailable": error.to_string()}));
                comparison.push(serde_json::json!({
                    "variant": row.name, "latency_ms": latency,
                    "scoring_start_ms": window.scoring_start_ms, "scoring_end_ms": scoring_end,
                    "grid_pnl_delta": end.1-start.1, "replay_pnl_delta": row.net_pnl_usdc,
                    "grid_fills_delta": end.2.saturating_sub(start.2), "replay_fills_delta": row.fills,
                    "grid_fees_delta": end.3-start.3, "replay_fees_delta": row.fees_usdc,
                    "grid_inventory_start": start.4, "grid_inventory_end": end.4,
                    "start_sample_ms": start.0, "end_sample_ms": end.0,
                    "trace": trace
                }));
            } else {
                println!("{}: missing boundary samples; no comparison", row.name);
            }
        }
    }
    std::fs::write(
        report_stem.with_extension("comparison.json"),
        serde_json::to_vec_pretty(&comparison)?,
    )?;
    Ok(())
}

/// Bounded prefix of retained decisions/fills, not a claim of complete history.
/// Enough to expose the first observable mismatch without a multi-GB in-memory join.
fn trace_head(
    path: &Path,
    from: u64,
    to: u64,
    errors: &mut Vec<String>,
) -> Result<Vec<(u64, serde_json::Value)>> {
    use std::io::BufRead;
    let mut output = Vec::new();
    let mut files = vec![(0, path.to_owned())];
    let prefix = format!(
        "{}.",
        path.file_name()
            .context("trace filename")?
            .to_string_lossy()
    );
    if let Ok(entries) = std::fs::read_dir(path.parent().context("trace parent")?) {
        for entry in entries.flatten() {
            if let Some(generation) = entry
                .file_name()
                .to_string_lossy()
                .strip_prefix(&prefix)
                .and_then(|s| s.parse::<usize>().ok())
            {
                files.push((generation, entry.path()));
            }
        }
    }
    files.sort_by_key(|(generation, _)| std::cmp::Reverse(*generation));
    for (_, path) in files {
        if !path.exists() {
            continue;
        }
        let reader = zstd::stream::read::Decoder::new(std::fs::File::open(&path)?)?;
        for line in std::io::BufReader::new(reader).lines() {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    errors.push(format!("{}: {error}", path.display()));
                    break;
                }
            };
            if !line.contains("quote_decision") && !line.contains("execution_event") {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(error) => {
                    errors.push(format!("{}: {error}", path.display()));
                    continue;
                }
            };
            let payload = &value["payload"];
            let (stamp, signature) = if value["event"] == "quote_decision" {
                (
                    payload["generated_ns"].as_u64().map(|n| n / 1_000_000),
                    serde_json::json!({"event":"quote", "bid":payload["bid"],"ask":payload["ask"],"reason":payload["reason"]}),
                )
            } else if payload["kind"] == "fill" {
                (
                    payload["exchange_ms"].as_u64(),
                    serde_json::json!({"event":"fill", "side":payload["side"],"px":payload["px"],"qty_units":payload["qty_units"],"maker":payload["maker"],"fee_usdc":payload["fee_usdc"]}),
                )
            } else {
                continue;
            };
            let Some(stamp) = stamp.filter(|t| *t >= from && *t <= to) else {
                continue;
            };
            output.push((stamp, signature));
            if output.len() == 512 {
                return Ok(output);
            }
        }
    }
    Ok(output)
}

fn compare_trace_heads(
    grid: &Path,
    replay: &Path,
    from: u64,
    to: u64,
) -> Result<serde_json::Value> {
    let mut errors = Vec::new();
    let grid_head = trace_head(grid, from, to, &mut errors)?;
    let replay_head = trace_head(replay, from, to, &mut errors)?;
    let start = grid_head
        .first()
        .zip(replay_head.first())
        .context("no overlapping retained decision/fill traces")?;
    let overlap = start.0 .0.max(start.1 .0);
    let grid_head = trace_head(grid, overlap, to, &mut errors)?;
    let replay_head = trace_head(replay, overlap, to, &mut errors)?;
    errors.sort();
    errors.dedup();
    let common = grid_head.len().min(replay_head.len());
    let mismatch = (0..common).find(|i| grid_head[*i] != replay_head[*i]);
    Ok(serde_json::json!({
        "scope": "first observable difference in retained trace prefix; not causal attribution or full parity",
        "overlap_start_ms": overlap, "requested_start_ms": from,
        "trace_read_errors": errors,
        "prefix_limit": 512, "grid_records": grid_head.len(), "replay_records": replay_head.len(),
        "first_difference": mismatch.map(|i| serde_json::json!({"index":i,"grid":grid_head[i],"replay":replay_head[i]})),
        "prefix_lengths_differ": grid_head.len() != replay_head.len()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_diagnostic_continues_after_a_truncated_old_generation() {
        let directory = tempfile::tempdir().unwrap();
        let grid = directory.path().join("grid.zst");
        let replay = directory.path().join("replay.zst");
        std::fs::write(grid.with_extension("zst.4"), b"incomplete").unwrap();
        for (path, bid) in [(&grid, 10), (&replay, 11)] {
            let line = serde_json::json!({"event":"quote_decision", "payload":{"generated_ns":1_000_000_000_u64,"bid":bid,"ask":12,"reason":"market"}}).to_string() + "\n";
            std::fs::write(path, zstd::stream::encode_all(line.as_bytes(), 1).unwrap()).unwrap();
        }
        let comparison = compare_trace_heads(&grid, &replay, 1_000, 2_000).unwrap();
        assert_eq!(comparison["first_difference"]["index"], 0);
        assert!(!comparison["trace_read_errors"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn comparison_uses_interval_deltas_and_actual_stop_boundary() {
        let directory = tempfile::tempdir().unwrap();
        let variant = flatten_variant(directory.path());
        let mut row = variant.leaderboard_row(None);
        row.net_pnl_usdc = 3.0;
        std::fs::write(directory.path().join("equity_history.csv"), format!(
            "ts_ms,run_started_ms,variant,net_pnl_usdc,fills,fees_usdc,inventory_units\n1000,1,{0},100,20,4,0\n2000,1,{0},103,22,5,0\n3000,1,{0},110,30,6,0\n", row.name)).unwrap();
        let board = grid::Leaderboard {
            generated_at_ms: 3_000,
            started_at_ms: 1,
            elapsed_seconds: 3,
            symbol: "CASHCAT".into(),
            feed_health: FeedHealth::new(0, 0, 0, 0, false),
            feed_down_for_ms: 0,
            quote_pause_reason: None,
            resumes: 0,
            resumed_downtime_ms: 0,
            rows: vec![row.clone()],
            replay: None,
        };
        let mut replay = board.clone();
        replay.replay = Some(grid::ReplayWindow {
            training_start_ms: 0.0,
            training_end_ms: 1_000.0,
            scoring_start_ms: 1_000.0,
            scoring_end_ms: 3_000.0,
            train_fraction: 1.0 / 3.0,
            latency_ms: 150,
            execution: BTreeMap::from([(
                row.name,
                ReplayDiagnostics {
                    scored_until_ms: Some(2_000),
                    ..ReplayDiagnostics::default()
                },
            )]),
        });
        let stem = directory.path().join("session");
        print_comparison(
            &board,
            &[(150, replay)],
            &directory.path().join("leaderboard.json"),
            &stem,
        )
        .unwrap();
        let comparison: serde_json::Value =
            serde_json::from_slice(&std::fs::read(stem.with_extension("comparison.json")).unwrap())
                .unwrap();
        assert_eq!(comparison[0]["grid_pnl_delta"], 3.0);
        assert_eq!(comparison[0]["grid_fills_delta"], 2);
        assert_eq!(comparison[0]["grid_fees_delta"], 1.0);
        assert_eq!(comparison[0]["scoring_end_ms"], 2_000);
    }

    #[tokio::test]
    async fn silent_suffix_and_trades_cannot_resume_a_stopped_or_blind_market() {
        for stopped in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut variant = flatten_variant(directory.path());
            let mut data = two_book_tape();
            data.mids.clear();
            data.books.clear();
            data.trades.push(mm_live::parquet_io::TradeRecord {
                ts_ms: 20_000.0,
                received_ms: Some(20_000.0),
                side: "buy".into(),
                price: 0.1,
                size: 500.0,
                trade_id: None,
            });
            if stopped {
                variant.backend.invalidate("terminal stop");
            }
            let before = variant.backend.account_state();
            let mut diagnostics = ReplayDiagnostics::default();
            let (end, _) = run_timed_event_source(
                &mut variant,
                ParquetReplaySource::new(&data, &cashcat()).unwrap(),
                &Arc::new(Metrics::default()),
                1,
                (1_000, 60_000),
                &mut diagnostics,
            )
            .await
            .unwrap();
            assert_eq!(before, variant.backend.account_state());
            assert_eq!(variant.quote_seq, 0);
            assert_eq!(variant.fills, 0);
            if stopped {
                assert_eq!(end, 1_000);
                assert!(diagnostics.pauses.is_empty());
            } else {
                assert_eq!(end, 60_000);
                assert_eq!(diagnostics.pauses[0].end_ms, end);
                assert!(diagnostics.pauses[0].open_at_end);
            }
        }
    }

    #[test]
    fn checkpoint_validation_and_restore_keep_lifetime_and_terminal_state() {
        let directory = tempfile::tempdir().unwrap();
        let mut variant = flatten_variant(directory.path());
        let saved = grid::PersistedVariant {
            name: "base".into(),
            config_fingerprint: variant.config_fingerprint.clone(),
            config_changes: 0,
            inventory_unit: 1_000,
            last_bbo: None,
            account: variant.backend.account_state(),
            diagnostics: variant.backend.diagnostics().clone(),
            fills: 17,
            peak_equity_usdc: 300.0,
            max_drawdown_usdc: 2.12,
            failure: Some("terminal".into()),
            current_day: Some(0),
            daily_realized_pnl_usdc: -0.25,
        };
        let mut state = grid::PersistedGridState {
            schema_version: grid::PersistedGridState::SCHEMA_VERSION,
            symbol: "CASHCAT".into(),
            grid_fingerprint: String::new(),
            run_id: "test".into(),
            started_at_ms: 1_000,
            checkpoint_ms: 2_000,
            resumes: 0,
            resumed_downtime_ms: 0,
            feed_health: FeedHealth::new(0, 0, 0, 0, false),
            trade_prints: 0,
            replayed_trades_ignored: 0,
            variants: vec![saved.clone()],
        };
        let names = vec![None];
        let mut units = BTreeMap::new();
        assert!(validate_initial_variants(
            Some(&state),
            &names,
            &units,
            None,
            &variant.config,
            1_999
        )
        .is_err());
        assert!(validate_initial_variants(
            Some(&state),
            &names,
            &units,
            None,
            &variant.config,
            2_000
        )
        .is_ok());
        units.insert("base".into(), 999);
        assert!(validate_initial_variants(
            Some(&state),
            &names,
            &units,
            None,
            &variant.config,
            2_000
        )
        .is_err());
        units.clear();
        state.variants[0].config_fingerprint = "incompatible".into();
        assert!(validate_initial_variants(
            Some(&state),
            &names,
            &units,
            None,
            &variant.config,
            2_000
        )
        .is_err());
        variant.restore(&saved).unwrap();
        assert_eq!(variant.fills, 17);
        assert_eq!(variant.failure, saved.failure);
        assert_eq!(variant.backend.account_state(), saved.account);
        assert_eq!(variant.backend.daily_risk_snapshot(), (Some(0), -0.25));
    }

    #[test]
    fn inventory_overrides_reject_ambiguity_and_invalid_quantities() {
        let names = vec![Some("one".into()), Some("two".into())];
        assert_eq!(
            parse_inventory_units(&["one=636".into(), "two=814".into()], &names).unwrap()["one"],
            636
        );
        for invalid in [
            vec!["one=0".into()],
            vec!["other=1".into()],
            vec!["one=1".into(), "one=2".into()],
            vec!["one".into()],
        ] {
            assert!(parse_inventory_units(&invalid, &names).is_err());
        }
    }

    #[tokio::test]
    async fn replay_clock_matches_grid_ticks_for_multiple_exit_configs() {
        for flatten_ms in [0, 1, 250] {
            let replay_dir = tempfile::tempdir().unwrap();
            let grid_dir = tempfile::tempdir().unwrap();
            let mut replay = flatten_variant(replay_dir.path());
            let mut live = flatten_variant(grid_dir.path());
            for variant in [&mut replay, &mut live] {
                variant.config.dry_run.flatten_after_ms = flatten_ms;
                variant.config.runtime.market_stale_ms = 10_000;
                variant.config.runtime.stats_interval_ms = 5_000;
                variant.backend = DryRunBackend::new(
                    cashcat(),
                    variant.config.dry_run.clone(),
                    variant.config.quoting.clone(),
                    variant.config.risk.clone(),
                )
                .unwrap();
                let mut account = variant.backend.account_state();
                account.inventory_units = 500;
                account.cash_usdc -= 50.0;
                account.average_entry_px = 0.1;
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
            }
            let data = two_book_tape();
            let metrics = Arc::new(Metrics::default());
            let mut diagnostics = ReplayDiagnostics {
                max_carry_inventory_gap_seconds: 900,
                ..Default::default()
            };
            run_timed_event_source(
                &mut replay,
                ParquetReplaySource::new(&data, &cashcat()).unwrap(),
                &metrics,
                1,
                (1_000, 60_000),
                &mut diagnostics,
            )
            .await
            .unwrap();

            // Independent grid driver: explicit timer ticks and book->touch messages,
            // rather than a second call to the replay or the backend fixture.
            let mut market = grid::PaperMarketState::default();
            let mut next_tick = 1_000;
            for at in [1_000, 2_000, 3_000, 60_000] {
                while next_tick <= at {
                    crate::paper::observe_paper_market(
                        std::slice::from_mut(&mut live),
                        &mut market,
                        None,
                        next_tick * 1_000_000,
                        next_tick,
                        0,
                        false,
                        10_000,
                        900_000,
                    )
                    .unwrap();
                    next_tick += 5_000;
                }
                let book = mm_live::types::BookSnapshot {
                    bids: vec![mm_live::types::BookLevel {
                        px: 99_900,
                        qty_units: 500,
                    }],
                    asks: vec![mm_live::types::BookLevel {
                        px: 100_100,
                        qty_units: 500,
                    }],
                    exchange_ms: at,
                    recv_ns: at * 1_000_000,
                };
                let touch = book.bbo().unwrap();
                let mut events = if at < 60_000 {
                    vec![MarketEvent::Book(book), MarketEvent::Bbo(touch)]
                } else {
                    vec![]
                };
                if at == 1_000 || at == 60_000 {
                    events.push(MarketEvent::Bbo(Bbo {
                        bid_sz: 0,
                        ask_sz: 0,
                        ..touch
                    }));
                }
                for event in events {
                    crate::paper::observe_paper_market(
                        std::slice::from_mut(&mut live),
                        &mut market,
                        Some(&event),
                        at * 1_000_000,
                        at,
                        0,
                        false,
                        10_000,
                        900_000,
                    )
                    .unwrap();
                    if market.bbo.is_some() {
                        step_paper_variant(&mut live, &event, at, market.bbo, None)
                            .await
                            .unwrap();
                    }
                    live.observe_equity();
                }
            }
            live.backend.shutdown(60_000).await.unwrap();
            assert_eq!(replay.backend.account_state(), live.backend.account_state());
            assert_eq!(replay.fills, live.fills);
            assert_eq!(replay.quote_seq, live.quote_seq);
            assert_eq!(
                replay.backend.diagnostics().flatten_events,
                live.backend.diagnostics().flatten_events
            );
            let decisions = |variant: &mut PaperVariant| {
                variant.logger.flush().unwrap();
                std::fs::read_to_string(variant.logger.path())
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                    .filter(|row| {
                        row["event"] == "quote_decision" || row["event"] == "execution_event"
                    })
                    .map(|row| row["payload"].clone())
                    .collect::<Vec<_>>()
            };
            assert_eq!(decisions(&mut replay), decisions(&mut live));
            assert!(diagnostics
                .pauses
                .iter()
                .any(|gap| gap.start_ms == 16_000 && gap.end_ms == 60_000));
        }
    }

    #[tokio::test]
    async fn virtual_silence_withdraws_and_long_gap_closes_before_new_mark() {
        for flatten in [0, 1, 250] {
            let directory = tempfile::tempdir().unwrap();
            let mut variant = flatten_variant(directory.path());
            variant.config.dry_run.flatten_after_ms = flatten;
            variant.backend = DryRunBackend::new(
                cashcat(),
                variant.config.dry_run.clone(),
                variant.config.quoting.clone(),
                variant.config.risk.clone(),
            )
            .unwrap();
            // Use the existing backend's restoration and the shared market transitions.
            variant.config.runtime.market_stale_ms = 10_000;
            let bbo = Bbo {
                bid_px: 99_900,
                ask_px: 100_100,
                bid_sz: 100,
                ask_sz: 100,
                exchange_ms: 1_000,
                recv_ns: 1_000_000_000,
            };
            let mut market = grid::PaperMarketState::default();
            let mut diagnostics = ReplayDiagnostics {
                max_carry_inventory_gap_seconds: 900,
                ..ReplayDiagnostics::default()
            };
            replay_observe(
                &mut variant,
                &mut market,
                Some(&MarketEvent::Bbo(bbo)),
                1_000,
                &mut diagnostics,
            )
            .unwrap();
            step_paper_variant(
                &mut variant,
                &MarketEvent::Bbo(bbo),
                1_000,
                market.bbo,
                None,
            )
            .await
            .unwrap();
            assert!(variant.backend.working_order_count() > 0);
            replay_observe(&mut variant, &mut market, None, 11_001, &mut diagnostics).unwrap();
            assert_eq!(variant.backend.working_order_count(), 0);
            assert!(market.bbo.is_none());
            let mut account = variant.backend.account_state();
            account.inventory_units = 10;
            account.average_entry_px = 0.1;
            account.cash_usdc -= 1.0;
            variant
                .backend
                .restore_from_snapshot(
                    account,
                    mm_live::execution::DryRunDiagnostics::default(),
                    1_000,
                    Some(0),
                    -0.25,
                )
                .unwrap();
            variant.backend.restore_checkpoint_bbo(Some(bbo));
            let resumed = Bbo {
                exchange_ms: 912_000,
                recv_ns: 912_000_000_000,
                bid_px: 199_900,
                ask_px: 200_100,
                ..bbo
            };
            replay_observe(
                &mut variant,
                &mut market,
                Some(&MarketEvent::Bbo(resumed)),
                912_000,
                &mut diagnostics,
            )
            .unwrap();
            assert_eq!(variant.backend.account_state().inventory_units, 0);
            assert_eq!(diagnostics.gap_closes, 1);
            assert!(
                diagnostics.gap_close_pnl_usdc < 0.0,
                "must close at old mark with costs, not profit from unseen jump"
            );
            assert_eq!(diagnostics.pauses.len(), 1);
            assert!(!diagnostics.pauses[0].open_at_end);
            assert!(variant.backend.daily_realized_pnl_usdc() < -0.25);
        }
    }
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
        // Zero transport delays isolate the age trigger and fresh-depth fill.
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
                price_drift_per_second: None,
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
                    received_ms: None,
                    bid_size: 0.0,
                    ask_size: 0.0,
                    ts_ms: 1_000.0,
                    bid: 0.0999,
                    ask: 0.1001,
                    mid: 0.1,
                },
                MidRecord {
                    received_ms: None,
                    bid_size: 0.0,
                    ask_size: 0.0,
                    ts_ms: 60_000.0,
                    bid: 0.0999,
                    ask: 0.1001,
                    mid: 0.1,
                },
            ],
            trades: Vec::new(),
            books: [1_000.0, 2_000.0, 3_000.0]
                .into_iter()
                .map(|ts_ms| mm_live::parquet_io::BookTopRecord {
                    received_ms: None,
                    ts_ms,
                    bid: 0.0999,
                    bid_size: 500.0,
                    ask: 0.1001,
                    ask_size: 500.0,
                    bid_levels: vec![(0.0999, 500.0)],
                    ask_levels: vec![(0.1001, 500.0)],
                })
                .collect(),
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
