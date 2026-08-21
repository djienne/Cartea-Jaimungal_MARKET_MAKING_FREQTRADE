use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use mm_live::calibration::{CalibrationSnapshot, Calibrator};
use mm_live::config::AppConfig;
use mm_live::execution::{AccountStateProvider, DryRunBackend, ExecutionBackend, MarketDataSource};
use mm_live::hjb::{solve_asymmetric, HjbSurface};
use mm_live::hot_path::{spawn_hot_path, AtomicRiskState, HotPathInputs, ModelBundle};
use mm_live::hyperliquid::market::{run_market_stream, MarketStreamArgs};
use mm_live::hyperliquid::meta::discover_instrument;
use mm_live::lockfree::{
    AsyncRing, AtomicBbo, HotPathSignal, SharedQuotes, HOT_SIGNAL_EXECUTION, HOT_SIGNAL_MODEL,
    HOT_SIGNAL_SHUTDOWN,
};
use mm_live::metrics::Metrics;
use mm_live::parquet_io::{
    ensure_no_external_writer, load_market_window, CollectorLock, MarketDataSet,
    ParquetEventRecorder,
};
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::replay::ParquetReplaySource;
use mm_live::report::{JsonlEventLogger, ModelReport, SessionReport};
use mm_live::types::{unix_ms, Bbo, MarketEvent, ProcessClock, QuoteReason};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "mm-live",
    version,
    about = "Cartea-Jaimungal market-making research engine"
)]
struct Cli {
    #[arg(long, default_value = "rust_live/config/cashcat.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate configuration and current Hyperliquid metadata.
    Validate,
    /// Run one all-Rust calibration and HJB solve over Parquet history.
    Calibrate,
    /// Replay the selected Parquet window deterministically.
    Replay {
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Connect to the public feed and simulate orders locally.
    DryRun {
        /// Optional bounded runtime for smoke tests. Zero runs until Ctrl-C.
        #[arg(long, default_value_t = 0)]
        duration_seconds: u64,
        /// Do not become a Parquet writer (useful while the reference collector runs).
        #[arg(long)]
        no_write_parquet: bool,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Reserved interface; real order submission is deliberately unavailable.
    Live,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("cannot install rustls ring crypto provider"))?;
    let cli = Cli::parse();
    let config = AppConfig::load(&cli.config)?;
    init_tracing(config.runtime.log_json);
    if matches!(cli.command, Command::Live) {
        bail!("live order submission is intentionally unavailable in this release");
    }
    let instrument = discover_instrument(config.runtime.network, &config.instrument).await?;
    if config.quoting.leverage > instrument.max_leverage {
        bail!(
            "configured leverage {} exceeds venue maximum {}",
            config.quoting.leverage,
            instrument.max_leverage
        );
    }
    match cli.command {
        Command::Validate => {
            println!("{}", serde_json::to_string_pretty(&instrument)?);
            Ok(())
        }
        Command::Calibrate => {
            let (_, snapshot, surface, inventory_unit) =
                calibrate_model(&config, &instrument, true)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "instrument": instrument,
                    "calibration": snapshot,
                    "hjb": {
                        "revision": surface.revision,
                        "q_min": surface.q_min,
                        "q_max": surface.q_max,
                        "n_steps": surface.n_steps,
                        "dt": surface.dt,
                        "phi_effective": surface.phi_effective,
                        "alpha_effective": surface.alpha_effective,
                    },
                    "inventory_unit_units": inventory_unit,
                }))?
            );
            Ok(())
        }
        Command::Replay { report } => {
            run_replay_command(&config, instrument, report.as_deref()).await
        }
        Command::DryRun {
            duration_seconds,
            no_write_parquet,
            report,
        } => {
            run_public_dry_run(
                &config,
                instrument,
                duration_seconds,
                no_write_parquet,
                report.as_deref(),
            )
            .await
        }
        Command::Live => unreachable!("live command returned before metadata discovery"),
    }
}

fn calibrate_model(
    config: &AppConfig,
    instrument: &mm_live::InstrumentSpec,
    require_current_data: bool,
) -> Result<(MarketDataSet, CalibrationSnapshot, HjbSurface, i64)> {
    let data = load_market_window(
        &config.storage.data_dir,
        &instrument.symbol,
        &config.calibration,
    )?;
    if require_current_data {
        let now_ms = unix_ms();
        let data_end_ms = data.window_end_ms.max(0.0) as u64;
        let future_limit = now_ms.saturating_add(
            config
                .calibration
                .max_future_skew_seconds
                .saturating_mul(1_000),
        );
        let maximum_age_ms = config
            .calibration
            .max_data_age_seconds
            .saturating_mul(1_000);
        if data_end_ms > future_limit || now_ms.saturating_sub(data_end_ms) > maximum_age_ms {
            bail!(
                "market calibration data is stale or future-dated: end_ms={data_end_ms}, now_ms={now_ms}"
            );
        }
    }
    let previous = match CalibrationSnapshot::load(&config.storage.calibration_path) {
        Ok(snapshot) => Some(snapshot),
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            None
        }
        Err(error) => {
            warn!(%error, path = %config.storage.calibration_path.display(), "cannot reuse prior calibration snapshot");
            None
        }
    };
    let candidate =
        Calibrator::new(&instrument.symbol, config.calibration.clone()).calibrate(&data)?;
    let snapshot = if candidate.is_quotable() {
        candidate
    } else if let Some(previous) = previous.filter(|snapshot| {
        snapshot.is_quotable()
            && snapshot.is_fresh(
                unix_ms(),
                config.calibration.max_age_seconds,
                config.calibration.max_future_skew_seconds,
            )
    }) {
        warn!(
            candidate_status = ?candidate.status,
            previous_revision = previous.revision,
            "new calibration rejected; retaining fresh last-good snapshot"
        );
        previous
    } else {
        bail!(
            "calibration failed closed with status {:?} and no fresh last-good snapshot",
            candidate.status
        );
    };
    let policy = CarteaJaimungalPolicy::new(
        instrument.clone(),
        config.quoting.clone(),
        config.risk.clone(),
    )?;
    let mid = data
        .mids
        .last()
        .context("calibration window has no final mid")?
        .mid;
    let inventory_unit = policy.derive_inventory_unit(mid, config.model.q_max)?;
    let surface = solve_asymmetric(
        snapshot.parameters,
        &config.model,
        instrument.size_from_units(inventory_unit),
        snapshot.revision,
    )?;
    if snapshot.revision == snapshot.generated_at_ms {
        snapshot.write_atomic(&config.storage.calibration_path)?;
    }
    Ok((data, snapshot, surface, inventory_unit))
}

async fn run_replay_command(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    report_path: Option<&Path>,
) -> Result<()> {
    let started_at_ms = unix_ms();
    let (data, snapshot, surface, inventory_unit) = calibrate_model(config, &instrument, false)?;
    let source = ParquetReplaySource::new(&data, &instrument)?;
    let metrics = Arc::new(Metrics::default());
    let mut backend = DryRunBackend::new(
        instrument.clone(),
        config.dry_run.clone(),
        config.quoting.clone(),
        config.risk.clone(),
    )?;
    let policy = CarteaJaimungalPolicy::new(
        instrument.clone(),
        config.quoting.clone(),
        config.risk.clone(),
    )?;
    let mut event_logger =
        JsonlEventLogger::create(&config.storage.report_dir, "replay", started_at_ms)?;
    run_event_source(
        config,
        &surface,
        inventory_unit,
        &policy,
        source,
        &mut backend,
        &metrics,
        &mut event_logger,
    )
    .await?;
    event_logger.flush()?;
    write_report(
        config,
        report_path,
        "replay",
        started_at_ms,
        instrument,
        Some(snapshot),
        Some(ModelReport::from_surface(&surface, inventory_unit)),
        &backend,
        &metrics,
        Vec::new(),
        event_logger.path(),
        0,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_event_source<S: MarketDataSource>(
    config: &AppConfig,
    surface: &HjbSurface,
    inventory_unit: i64,
    policy: &CarteaJaimungalPolicy,
    mut source: S,
    backend: &mut DryRunBackend,
    metrics: &Arc<Metrics>,
    event_logger: &mut JsonlEventLogger,
) -> Result<()> {
    let mut latest_bbo = None;
    let mut quote_seq = 0_u64;
    let mut episode_start_ms = None;
    while let Some(event) = source.next_event().await? {
        let event_ms = event_ms(&event);
        event_logger.log("market_event", Some(event_ms), &event)?;
        let execution_events = backend.on_market_event(&event).await?;
        for execution_event in &execution_events {
            event_logger.log("execution_event", Some(event_ms), execution_event)?;
        }
        let filled = !execution_events.is_empty();
        metrics
            .fills
            .fetch_add(execution_events.len() as u64, Ordering::Relaxed);
        if let MarketEvent::Bbo(bbo) = event {
            latest_bbo = Some(bbo);
        }
        let Some(bbo) = latest_bbo else { continue };
        let account = backend.account_state();
        let q_exact = account.inventory_units as f64 / inventory_unit as f64;
        let start = episode_start_ms.get_or_insert(event_ms);
        let mut elapsed = event_ms.saturating_sub(*start) as f64 / 1_000.0;
        let minimum_elapsed =
            config.model.horizon_seconds * config.model.episode_min_elapsed_fraction;
        if elapsed >= config.model.horizon_seconds
            || (config.model.episode_reset_on_flat
                && q_exact.round() == 0.0
                && elapsed >= minimum_elapsed)
        {
            *start = event_ms;
            elapsed = 0.0;
        }
        let tau = (config.model.horizon_seconds - elapsed).max(0.0);
        quote_seq = quote_seq.wrapping_add(1);
        let decision = policy.compute(
            surface,
            bbo,
            account.inventory_units,
            inventory_unit,
            tau,
            quote_seq,
            event_ms.saturating_mul(1_000_000),
            if filled {
                QuoteReason::Fill
            } else {
                QuoteReason::Market
            },
            RiskState {
                equity_usdc: account.equity_usdc,
                daily_realized_pnl_usdc: backend.daily_realized_pnl_usdc(),
                consecutive_losses: account.consecutive_losses,
            },
        );
        metrics.quote_decisions.fetch_add(1, Ordering::Relaxed);
        if decision.quotes.reason == QuoteReason::RiskLimit {
            metrics.risk_refusals.fetch_add(1, Ordering::Relaxed);
        }
        backend.reconcile(decision.quotes, event_ms).await?;
        event_logger.log("quote_decision", Some(event_ms), &decision.quotes)?;
        metrics.quote_publications.fetch_add(1, Ordering::Relaxed);
    }
    backend.shutdown(data_end_ms(latest_bbo)).await?;
    Ok(())
}

async fn run_public_dry_run(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    duration_seconds: u64,
    no_write_parquet: bool,
    report_path: Option<&Path>,
) -> Result<()> {
    let started_at_ms = unix_ms();
    let mut initial_model = match calibrate_model(config, &instrument, true) {
        Ok((_, snapshot, surface, inventory_unit)) => Some((snapshot, surface, inventory_unit)),
        Err(error) => {
            warn!(%error, "no valid startup calibration; collecting data with quotes disabled");
            None
        }
    };
    let write_parquet = config.storage.write_parquet && !no_write_parquet;
    let collector_lock = if write_parquet {
        ensure_no_external_writer(
            &config.storage.data_dir,
            &instrument.symbol,
            config.storage.flush_interval_seconds.saturating_mul(2),
        )
        .await?;
        Some(CollectorLock::acquire(
            &config.storage.writer_lock_path,
            config.storage.flush_interval_seconds.saturating_mul(3),
        )?)
    } else {
        None
    };
    let mut recorder = write_parquet.then(|| {
        ParquetEventRecorder::new(
            config.storage.data_dir.clone(),
            instrument.clone(),
            config.storage.flush_interval_seconds,
        )
    });
    let mut event_logger =
        JsonlEventLogger::create(&config.storage.report_dir, "dry_run", started_at_ms)?;
    let metrics = Arc::new(Metrics::default());
    let scientifically_valid = Arc::new(AtomicBool::new(true));
    let events = Arc::new(AsyncRing::new(config.runtime.market_event_capacity));
    let latest_bbo = Arc::new(AtomicBbo::default());
    let signal = Arc::new(HotPathSignal::default());
    let desired = Arc::new(SharedQuotes::default());
    let clock = Arc::new(ProcessClock::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let market_task = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: config.runtime.network.ws_url().to_owned(),
        instrument: instrument.clone(),
        latest_bbo: latest_bbo.clone(),
        events: events.clone(),
        signal: signal.clone(),
        clock: clock.clone(),
        metrics: metrics.clone(),
        scientifically_valid: scientifically_valid.clone(),
        shutdown: shutdown_rx.clone(),
        ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
        idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
    }));
    let mut backend = DryRunBackend::new(
        instrument.clone(),
        config.dry_run.clone(),
        config.quoting.clone(),
        config.risk.clone(),
    )?;
    let config_fingerprint = config.fingerprint()?;
    let _ = backend.restore_account_state(&config.storage.state_path, &config_fingerprint)?;
    if initial_model.is_none() {
        if let (Ok(persisted), Some(inventory_unit)) = (
            CalibrationSnapshot::load(&config.storage.calibration_path),
            backend.restored_inventory_unit(),
        ) {
            if persisted.is_quotable()
                && persisted.is_fresh(
                    unix_ms(),
                    config.calibration.max_age_seconds,
                    config.calibration.max_future_skew_seconds,
                )
            {
                let surface = solve_asymmetric(
                    persisted.parameters,
                    &config.model,
                    instrument.size_from_units(inventory_unit),
                    persisted.revision,
                )?;
                initial_model = Some((persisted, surface, inventory_unit));
                info!("restored fresh persisted calibration and inventory-unit identity");
            }
        }
    }
    let restored_account = backend.account_state();
    let inventory_units = Arc::new(AtomicI64::new(restored_account.inventory_units));
    let risk_state = Arc::new(AtomicRiskState::default());
    risk_state.store(RiskState {
        equity_usdc: restored_account.equity_usdc,
        daily_realized_pnl_usdc: backend.daily_realized_pnl_usdc(),
        consecutive_losses: restored_account.consecutive_losses,
    });
    let mut snapshot = None;
    let initial_bundle =
        if let Some((initial_snapshot, mut surface, mut inventory_unit)) = initial_model {
            if restored_account.inventory_units != 0 {
                let restored_unit = backend
                    .restored_inventory_unit()
                    .context("non-flat restored state has no inventory-unit identity")?;
                inventory_unit = restored_unit;
                surface = solve_asymmetric(
                    initial_snapshot.parameters,
                    &config.model,
                    instrument.size_from_units(inventory_unit),
                    initial_snapshot.revision,
                )?;
            }
            snapshot = Some(initial_snapshot.clone());
            Some(Arc::new(ModelBundle {
                surface,
                inventory_unit,
                generated_at_ms: initial_snapshot.generated_at_ms,
            }))
        } else {
            None
        };
    let model = Arc::new(arc_swap::ArcSwapOption::from(initial_bundle));
    let hot_thread = spawn_hot_path(HotPathInputs {
        latest_bbo: latest_bbo.clone(),
        signal: signal.clone(),
        desired: desired.clone(),
        model: model.clone(),
        instrument: instrument.clone(),
        quoting: config.quoting.clone(),
        risk: config.risk.clone(),
        model_config: config.model.clone(),
        inventory_units: inventory_units.clone(),
        risk_state: risk_state.clone(),
        scientifically_valid: scientifically_valid.clone(),
        market_stale_ms: config.runtime.market_stale_ms,
        calibration_max_age_seconds: config.calibration.max_age_seconds,
        calibration_max_future_skew_seconds: config.calibration.max_future_skew_seconds,
        clock,
        metrics: metrics.clone(),
        hot_path_cpu: config.runtime.hot_path_cpu,
    })?;
    let interval_duration = Duration::from_secs(config.calibration.interval_seconds.max(1));
    let mut calibration_interval = tokio::time::interval_at(
        tokio::time::Instant::now() + interval_duration,
        interval_duration,
    );
    calibration_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut observed_quote_seq = desired.load().quote_seq;
    let deadline = (duration_seconds > 0)
        .then(|| tokio::time::Instant::now() + Duration::from_secs(duration_seconds));
    let shutdown_signal = wait_for_shutdown_signal();
    tokio::pin!(shutdown_signal);
    loop {
        tokio::select! {
            biased;
            result = &mut shutdown_signal => {
                result?;
                break;
            }
            () = async {
                if let Some(deadline) = deadline {
                    tokio::time::sleep_until(deadline).await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => break,
            event = events.pop() => {
                let event_time = event_ms(&event);
                event_logger.log("market_event", Some(event_time), &event)?;
                if let Some(recorder) = recorder.as_mut() {
                    recorder.record(&event, unix_ms())?;
                }
                if !scientifically_valid.load(Ordering::Acquire) && backend.scientifically_valid() {
                    backend.invalidate("public market stream lost causal events or disconnected");
                    signal.notify(HOT_SIGNAL_EXECUTION);
                }
                let execution_events = backend.on_market_event(&event).await?;
                for execution_event in &execution_events {
                    event_logger.log("execution_event", Some(event_time), execution_event)?;
                }
                metrics.fills.fetch_add(execution_events.len() as u64, Ordering::Relaxed);
                let account = backend.account_state();
                inventory_units.store(account.inventory_units, Ordering::Relaxed);
                metrics.inventory_units.store(account.inventory_units, Ordering::Relaxed);
                risk_state.store(RiskState {
                    equity_usdc: account.equity_usdc,
                    daily_realized_pnl_usdc: backend.daily_realized_pnl_usdc(),
                    consecutive_losses: account.consecutive_losses,
                });
                if !execution_events.is_empty() {
                    signal.notify(HOT_SIGNAL_EXECUTION);
                }
            }
            next = desired.changed_after(observed_quote_seq) => {
                observed_quote_seq = next.quote_seq;
                backend.reconcile(next, unix_ms()).await?;
                event_logger.log("quote_decision", None, &next)?;
            }
            _ = calibration_interval.tick() => {
                match calibrate_model(config, &instrument, true) {
                    Ok((_, next, mut next_surface, mut next_inventory_unit)) => {
                        let account = backend.account_state();
                        if account.inventory_units != 0 {
                            let Some(retained_unit) = model
                                .load_full()
                                .map(|bundle| bundle.inventory_unit)
                                .or_else(|| backend.restored_inventory_unit())
                            else {
                                metrics.calibration_failures.fetch_add(1, Ordering::Relaxed);
                                warn!("cannot publish calibration while non-flat without a retained inventory unit");
                                continue;
                            };
                            next_inventory_unit = retained_unit;
                            next_surface = solve_asymmetric(
                                next.parameters,
                                &config.model,
                                instrument.size_from_units(next_inventory_unit),
                                next.revision,
                            )?;
                        }
                        model.store(Some(Arc::new(ModelBundle {
                            surface: next_surface,
                            inventory_unit: next_inventory_unit,
                            generated_at_ms: next.generated_at_ms,
                        })));
                        snapshot = Some(next.clone());
                        event_logger.log("calibration", None, &next)?;
                        metrics.calibration_runs.fetch_add(1, Ordering::Relaxed);
                        signal.notify(HOT_SIGNAL_MODEL);
                        if let Some(recorder) = recorder.as_mut() {
                            let now_ms = unix_ms();
                            let compacted = recorder.compact(
                                now_ms,
                                config.storage.compact_after_minutes,
                            )?;
                            let removed = recorder.prune(
                                now_ms,
                                config.storage.retention_minutes,
                            )?;
                            info!(compacted, removed, "Parquet maintenance complete");
                        }
                    }
                    Err(error) => {
                        metrics.calibration_failures.fetch_add(1, Ordering::Relaxed);
                        warn!(%error, "calibration refresh failed; retaining last good model until stale");
                    }
                }
            }
        }
    }
    signal.notify(HOT_SIGNAL_SHUTDOWN);
    let _ = shutdown_tx.send(true);
    backend.shutdown(unix_ms()).await?;
    if let (Some(snapshot), Some(bundle)) = (snapshot.as_ref(), model.load_full()) {
        backend.save_account_state(
            &config.storage.state_path,
            &config_fingerprint,
            &snapshot.fingerprint,
            bundle.inventory_unit,
        )?;
    }
    event_logger.flush()?;
    if let Some(recorder) = recorder.as_mut() {
        let now_ms = unix_ms();
        recorder.flush(now_ms)?;
        let compacted = recorder.compact(now_ms, config.storage.compact_after_minutes)?;
        let removed = recorder.prune(now_ms, config.storage.retention_minutes)?;
        info!(compacted, removed, "Parquet maintenance complete");
    }
    drop(collector_lock);
    let _ = tokio::time::timeout(Duration::from_secs(5), market_task).await;
    tokio::task::spawn_blocking(move || hot_thread.join())
        .await
        .context("cannot join hot-path task")?
        .map_err(|_| anyhow::anyhow!("hot-path thread panicked"))?;
    let mut invalid_reasons = Vec::new();
    if let Some(reason) = backend.diagnostics().invalid_reason.clone() {
        invalid_reasons.push(reason);
    }
    if snapshot.is_none() {
        invalid_reasons.push("no valid calibration was produced".to_owned());
    }
    write_report(
        config,
        report_path,
        "dry_run",
        started_at_ms,
        instrument,
        snapshot,
        model
            .load_full()
            .map(|bundle| ModelReport::from_surface(&bundle.surface, bundle.inventory_unit)),
        &backend,
        &metrics,
        invalid_reasons,
        event_logger.path(),
        events.high_water_mark(),
    )?;
    info!(
        scientifically_valid = backend.scientifically_valid(),
        "dry-run stopped"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_report(
    config: &AppConfig,
    report_path: Option<&Path>,
    mode: &str,
    started_at_ms: u64,
    instrument: mm_live::InstrumentSpec,
    calibration: Option<CalibrationSnapshot>,
    model: Option<ModelReport>,
    backend: &DryRunBackend,
    metrics: &Arc<Metrics>,
    invalid_reasons: Vec<String>,
    event_log_path: &Path,
    market_event_ring_high_water: usize,
) -> Result<()> {
    let finished_at_ms = unix_ms();
    let has_calibration = calibration.is_some();
    let report = SessionReport {
        schema_version: 1,
        session_id: format!("{}-{started_at_ms}", instrument.symbol),
        started_at_ms,
        finished_at_ms,
        mode: mode.to_owned(),
        config_fingerprint: config.fingerprint()?,
        instrument,
        calibration,
        model,
        account: backend.account_state(),
        execution: backend.diagnostics().clone(),
        metrics: metrics.snapshot(),
        scientifically_valid: backend.scientifically_valid()
            && invalid_reasons.is_empty()
            && has_calibration,
        invalid_reasons,
        event_log_path: event_log_path.display().to_string(),
        market_event_ring_high_water,
    };
    let path = report_path.map_or_else(
        || {
            config
                .storage
                .report_dir
                .join(format!("{mode}-{started_at_ms}.json"))
        },
        Path::to_owned,
    );
    report.write_atomic(&path)?;
    println!("report={}", path.display());
    Ok(())
}

fn event_ms(event: &MarketEvent) -> u64 {
    match event {
        MarketEvent::Bbo(value) => value.exchange_ms,
        MarketEvent::Trade(value) => value.exchange_ms,
        MarketEvent::Book(value) => value.exchange_ms,
    }
}

fn data_end_ms(bbo: Option<Bbo>) -> u64 {
    bbo.map_or_else(unix_ms, |value| value.exchange_ms)
}

fn init_tracing(json: bool) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("mm_live=info"));
    if json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn wait_for_shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}
