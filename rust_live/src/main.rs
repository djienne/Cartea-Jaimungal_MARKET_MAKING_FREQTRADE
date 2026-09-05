use anyhow::{bail, Context, Result};
#[cfg(feature = "live-acceptance")]
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use mm_live::calibration::{CalibrationSnapshot, Calibrator};
use mm_live::config::AppConfig;
use mm_live::execution::{
    AccountStateProvider, DryRunBackend, ExecutionBackend, HyperliquidLiveBackend, MarketDataSource,
};
use mm_live::flow_guard::{FlowGuard, MidWindow, VpinTracker};
use mm_live::hjb::{solve_asymmetric, CjParameters, HjbSurface};
use mm_live::hot_path::{
    flow_channel, risk_channel, spawn_hot_path, HotPathInputs, ModelBundle, RiskWriter,
};
use mm_live::hyperliquid::market::{run_market_stream, MarketStreamArgs};
use mm_live::hyperliquid::meta::discover_instrument;
#[cfg(feature = "live-acceptance")]
use mm_live::hyperliquid::signing::{make_cloid, parse_fixed, LiveOrderRequest, TimeInForce};
use mm_live::hyperliquid::{
    account::{run_account_stream, AccountStreamArgs, AccountStreamEvent, AccountStreamMetrics},
    auth::HyperliquidCredentials,
    exchange::HyperliquidExchangeClient,
};
use mm_live::latency::{LatencyKind, LatencyMonitor, LatencyObserver, LatencySnapshot};
use mm_live::lockfree::{
    bbo_channel, quote_channel, AsyncRing, HotPathSignal, HOT_SIGNAL_ACCOUNT, HOT_SIGNAL_FILL,
    HOT_SIGNAL_MODEL, HOT_SIGNAL_SHUTDOWN,
};
use mm_live::metrics::Metrics;
use mm_live::parquet_io::{
    ensure_no_external_writer, load_market_window, CollectorLock, MarketDataSet,
    ParquetEventRecorder, ParquetRecorderHandle,
};
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::replay::ParquetReplaySource;
use mm_live::report::{
    JsonlEventLogger, LiveSessionReport, LogBackpressure, LogFormat, LogRotation, ModelReport,
    ReplayInputs, SessionReport,
};
use mm_live::types::{
    unix_ms, Bbo, DesiredQuotes, ExecutionEvent, MarketEvent, ProcessClock, QuoteReason,
};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[cfg(feature = "live-acceptance")]
use hyperliquid_connector::acceptance as live_acceptance;

mod grid;

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
    /// Print embedded optimization, target, toolchain, and revision metadata.
    BuildInfo,
    /// Container healthcheck: is a running grid both alive and seeing the feed?
    ///
    /// Reads `leaderboard.json` only — no config, no runtime, no network — so it
    /// is cheap enough to run every minute and cannot perturb the run it checks.
    ///
    /// Reports quote readiness and scientific validity separately from liveness.
    GridHealth {
        #[arg(long, default_value = "reports/grid_live/leaderboard.json")]
        leaderboard: PathBuf,
        /// Staleness limit on `generated_at_ms`, which the grid rewrites every
        /// `stats_interval_ms`. Catches a hung or dead process.
        #[arg(long, default_value_t = 120)]
        max_age_seconds: u64,
        /// How long the public feed may be down before the container is called
        /// unhealthy, allowing the supervisor to recover a stuck connection.
        #[arg(long, default_value_t = 180)]
        max_feed_down_seconds: u64,
    },
    /// Validate configuration and current Hyperliquid metadata.
    Validate,
    /// Run one all-Rust calibration and HJB solve over Parquet history.
    Calibrate,
    /// Replay the selected Parquet window deterministically.
    Replay {
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 0.7)]
        train_fraction: f64,
        #[arg(long, requires = "variant")]
        grid: Option<PathBuf>,
        #[arg(long, requires = "grid")]
        variant: Option<String>,
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
    /// Run several parameter sets against one shared public feed and rank them.
    ///
    /// One WebSocket regardless of how many variants: the venue allows ten per
    /// IP and that budget is shared with the data collectors and any live
    /// session. Never writes Parquet and never touches credentials.
    DryRunGrid {
        /// Grid spec listing the variants and their sparse overrides.
        #[arg(long)]
        grid: PathBuf,
        /// Optional bounded runtime. Zero runs until Ctrl-C.
        #[arg(long, default_value_t = 0)]
        duration_seconds: u64,
        /// Directory for per-variant reports and the leaderboard.
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Seconds between equity-history samples. Zero disables the history.
        #[arg(long, default_value_t = 60)]
        history_seconds: u64,
        /// Maximum checkpoint age for resume, in seconds. Zero starts fresh.
        /// Process downtime is recorded separately from feed downtime.
        #[arg(long, default_value_t = 3600)]
        max_resume_gap_seconds: u64,
        /// Maximum gap for carrying inventory, in seconds. Longer resumable gaps
        /// close valid positions at checkpoint bid/ask with promotion exit costs.
        /// Must not exceed the resume window. Zero closes on every resume.
        #[arg(long, default_value_t = 900)]
        max_carry_inventory_gap_seconds: u64,
        /// Roll bounded event logs at this many MiB and before reopening.
        /// Zero disables rotation and permits unbounded append.
        #[arg(long, default_value_t = 64)]
        log_max_mb: u64,
        /// Rolled generations kept behind each live log. Worst-case disk is
        /// `(keep + 1) * log_max_mb` per variant.
        #[arg(long, default_value_t = 3)]
        log_keep: usize,
    },
    /// Select the highest valid, live-equivalent flatten-P&L row and atomically
    /// generate the micro-live configuration. Dry-run-only exit policies are
    /// excluded. This command performs no network activity.
    PromoteBest {
        #[arg(long)]
        grid: PathBuf,
        #[arg(long)]
        leaderboard: PathBuf,
        #[arg(long, default_value = "rust_live/run/cashcat-active-live.toml")]
        output: PathBuf,
        #[arg(long, default_value = "rust_live/run/cashcat-promotion.json")]
        manifest: PathBuf,
        #[arg(long, default_value_t = 43_200)]
        min_elapsed_seconds: u64,
    },
    /// Exercise credential parsing, account REST reads, and the account WebSocket without actions.
    ConnectorCheck {
        #[arg(long, default_value = "rust_live/hyperliquid.env")]
        credentials: PathBuf,
        #[arg(long, default_value_t = 65)]
        duration_seconds: u64,
    },
    #[cfg(feature = "live-acceptance")]
    /// Explicitly bounded real-money connector canary; never used by the strategy runtime.
    LiveCanary {
        #[arg(long, default_value = "rust_live/hyperliquid.env")]
        credentials: PathBuf,
        #[arg(long, value_enum, default_value_t = CanaryPhase::Passive)]
        phase: CanaryPhase,
        #[arg(long, default_value_t = 12.0)]
        max_notional_usdc: f64,
        #[arg(long)]
        confirm_real_money_risk: bool,
    },
    /// Run the continuous live backend when enabled in the central TOML configuration.
    Live {
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        duration_seconds: u64,
    },
    /// Cancel bot orders and reduce-only IOC the configured CASHCAT account to flat.
    LiveFlatten,
    #[cfg(feature = "live-acceptance")]
    /// Run one persisted phase of the bounded real-account acceptance campaign.
    LiveAcceptance {
        #[arg(long, value_enum)]
        phase: AcceptancePhase,
    },
    #[cfg(feature = "live-acceptance")]
    /// Bounded production-gate smoke; the short duration cannot satisfy network warm-up.
    LiveGateSmoke {
        #[arg(long, default_value_t = 15)]
        duration_seconds: u64,
    },
}

#[cfg(feature = "live-acceptance")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CanaryPhase {
    Passive,
    RoundTrip,
}

#[cfg(feature = "live-acceptance")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum AcceptancePhase {
    Verify,
    Leverage,
    TwoSided,
    CrossingAlo,
    UnknownOutcome,
    RestartOrderPrepare,
    RestartOrderRecover,
    Deadman,
    MakerFill,
    RestartPositionPrepare,
    RestartPositionRecover,
    Final,
}

fn main() -> Result<()> {
    mm_live::BuildInfo::current().ensure_optimized()?;
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("cannot install rustls ring crypto provider"))?;
    let cli = Cli::parse();
    if matches!(&cli.command, Command::BuildInfo) {
        println!(
            "{}",
            serde_json::to_string_pretty(&mm_live::BuildInfo::current())?
        );
        return Ok(());
    }
    // Before the config load: the healthcheck must work in a container that
    // mounts only the reports directory, and must not fail for reasons that
    // have nothing to do with the run it is judging.
    if let Command::GridHealth {
        leaderboard,
        max_age_seconds,
        max_feed_down_seconds,
    } = &cli.command
    {
        return match grid_health_verdict(leaderboard, *max_age_seconds, *max_feed_down_seconds) {
            Ok(summary) => {
                println!("{summary}");
                Ok(())
            }
            Err(reason) => {
                // stderr and a non-zero exit: Docker records the last line of
                // output against the failing probe, so the reason survives in
                // `docker inspect` rather than only in this process's ashes.
                eprintln!("{reason}");
                std::process::exit(1);
            }
        };
    }
    let config = AppConfig::load(&cli.config)?;
    if let Command::PromoteBest {
        grid,
        leaderboard,
        output,
        manifest,
        min_elapsed_seconds,
    } = &cli.command
    {
        return promote_best_config(
            &config,
            grid,
            leaderboard,
            output,
            manifest,
            *min_elapsed_seconds,
        );
    }
    init_tracing(config.runtime.log_json);
    // The runtime is built explicitly so hot-path isolation is real: the
    // default #[tokio::main] spawned one worker per logical CPU with no
    // affinity, leaving workers, the blocking pool (Parquet, HJB solves), and
    // the writer threads all schedulable on the hot-path core.
    let runtime = build_runtime(config.runtime.hot_path_cpu)?;
    runtime.block_on(run_command(cli, config))
}

/// Bounded, hot-core-avoiding tokio runtime.
///
/// Worker and blocking threads round-robin across every core except the one
/// reserved for the hot path. Rust-side affinity cannot keep *other
/// processes* off that core — pair it with OS-level isolation (Windows
/// process affinity / Linux `isolcpus`) for a quiet core.
fn build_runtime(hot_path_cpu: Option<usize>) -> Result<tokio::runtime::Runtime> {
    let cores = core_affinity::get_core_ids().unwrap_or_default();
    let available = cores.len().max(1);
    let workers = available
        .saturating_sub(usize::from(hot_path_cpu.is_some()))
        .clamp(2, 6);
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder
        .enable_all()
        .worker_threads(workers)
        .max_blocking_threads(16);
    if let Some(hot) = hot_path_cpu {
        let complement: Vec<core_affinity::CoreId> = cores
            .iter()
            .copied()
            .filter(|core| core.id != hot)
            .collect();
        if complement.is_empty() {
            bail!("hot_path_cpu {hot} leaves no cores for the runtime");
        }
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        builder.on_thread_start(move || {
            let index = next.fetch_add(1, Ordering::Relaxed) % complement.len();
            let _ = core_affinity::set_for_current(complement[index]);
        });
        info!(
            hot_path_cpu = hot,
            workers,
            runtime_cores = available - 1,
            "runtime configured; workers excluded from the hot-path core"
        );
    } else {
        info!(
            workers,
            "runtime configured without hot-path isolation (runtime.hot_path_cpu unset)"
        );
    }
    Ok(builder.build()?)
}

async fn run_command(cli: Cli, config: AppConfig) -> Result<()> {
    if command_requires_live(&cli.command) && !config.live.enabled {
        bail!("live.enabled=false; refusing before credentials or order transport");
    }
    #[cfg(feature = "live-acceptance")]
    if matches!(
        &cli.command,
        Command::LiveCanary {
            confirm_real_money_risk: false,
            ..
        }
    ) {
        bail!("live-canary requires --confirm-real-money-risk before network or credential access");
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
        Command::BuildInfo => unreachable!("build-info returned before configuration loading"),
        Command::GridHealth { .. } => {
            unreachable!("grid-health returned before configuration loading")
        }
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
        Command::Replay {
            report,
            train_fraction,
            grid,
            variant,
        } => {
            run_replay_command(
                &config,
                instrument,
                report.as_deref(),
                train_fraction,
                grid.as_deref(),
                variant.as_deref(),
            )
            .await
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
        Command::DryRunGrid {
            grid,
            duration_seconds,
            out_dir,
            history_seconds,
            max_resume_gap_seconds,
            max_carry_inventory_gap_seconds,
            log_max_mb,
            log_keep,
        } => {
            run_dry_run_grid(
                &config,
                instrument,
                &grid,
                duration_seconds,
                out_dir.as_deref(),
                history_seconds,
                max_resume_gap_seconds,
                max_carry_inventory_gap_seconds,
                mm_live::report::LogRotation {
                    max_bytes: log_max_mb.saturating_mul(1024 * 1024),
                    keep: log_keep,
                },
            )
            .await
        }
        Command::PromoteBest { .. } => unreachable!("promotion returned before runtime startup"),
        Command::ConnectorCheck {
            credentials,
            duration_seconds,
        } => run_connector_check(&config, instrument, &credentials, duration_seconds).await,
        #[cfg(feature = "live-acceptance")]
        Command::LiveCanary {
            credentials,
            phase,
            max_notional_usdc,
            confirm_real_money_risk,
        } => {
            run_live_canary(
                &config,
                instrument,
                &credentials,
                phase,
                max_notional_usdc,
                confirm_real_money_risk,
            )
            .await
        }
        Command::Live {
            report,
            duration_seconds,
        } => run_live(&config, instrument, report.as_deref(), duration_seconds).await,
        Command::LiveFlatten => run_live_flatten(&config, instrument).await,
        #[cfg(feature = "live-acceptance")]
        Command::LiveAcceptance { phase } => {
            live_acceptance::run(&config, instrument, acceptance_phase(phase)).await
        }
        #[cfg(feature = "live-acceptance")]
        Command::LiveGateSmoke { duration_seconds } => {
            let mut smoke = config.clone();
            smoke.live.mode = mm_live::config::LiveMode::Production;
            smoke.live.deadman_enabled = false;
            smoke.live.startup_warmup_seconds = 0;
            smoke.live.state_path = config
                .live
                .state_path
                .with_file_name("cashcat-live-gate-smoke.redb");
            run_live(&smoke, instrument, None, duration_seconds.min(30)).await
        }
    }
}

/// Market events drained per select visit once the quote branch outranks the
/// ring. Large enough to clear bursts quickly, small enough that a fresh quote
/// decision never waits behind more than one batch.
const MARKET_EVENT_DRAIN_BATCH: u32 = 32;

fn command_requires_live(command: &Command) -> bool {
    match command {
        Command::Live { .. } | Command::LiveFlatten => true,
        #[cfg(feature = "live-acceptance")]
        Command::LiveAcceptance { .. } | Command::LiveGateSmoke { .. } => true,
        _ => false,
    }
}

#[cfg(feature = "live-acceptance")]
const fn acceptance_phase(
    phase: AcceptancePhase,
) -> hyperliquid_connector::acceptance::AcceptancePhase {
    use hyperliquid_connector::acceptance::AcceptancePhase as Target;
    match phase {
        AcceptancePhase::Verify => Target::Verify,
        AcceptancePhase::Leverage => Target::Leverage,
        AcceptancePhase::TwoSided => Target::TwoSided,
        AcceptancePhase::CrossingAlo => Target::CrossingAlo,
        AcceptancePhase::UnknownOutcome => Target::UnknownOutcome,
        AcceptancePhase::RestartOrderPrepare => Target::RestartOrderPrepare,
        AcceptancePhase::RestartOrderRecover => Target::RestartOrderRecover,
        AcceptancePhase::Deadman => Target::Deadman,
        AcceptancePhase::MakerFill => Target::MakerFill,
        AcceptancePhase::RestartPositionPrepare => Target::RestartPositionPrepare,
        AcceptancePhase::RestartPositionRecover => Target::RestartPositionRecover,
        AcceptancePhase::Final => Target::Final,
    }
}

async fn run_connector_check(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    credentials_path: &Path,
    duration_seconds: u64,
) -> Result<()> {
    if duration_seconds == 0 {
        bail!("connector-check duration must be positive");
    }
    let started_at_ms = unix_ms();
    let credentials = Arc::new(HyperliquidCredentials::load(credentials_path)?);
    let clock = Arc::new(ProcessClock::default());
    let latency = Arc::new(LatencyMonitor::new(
        &instrument.symbol,
        started_at_ms,
        &config.latency,
        false,
    ));
    let latency_observer = LatencyObserver::spawn(
        latency.clone(),
        clock.clone(),
        instrument.symbol.clone(),
        started_at_ms,
        config.latency.clone(),
        false,
        Duration::from_millis(config.runtime.stats_interval_ms),
        config.storage.latency_path.clone(),
    )?;
    let client = HyperliquidExchangeClient::new(
        config.runtime.network,
        instrument.clone(),
        credentials.clone(),
        clock.clone(),
        Some(latency.clone()),
        Duration::from_secs(5),
        5_000,
        config.live.max_rest_weight_per_minute,
        config.live.safety_rest_weight_reserve,
    )?;
    let account_events = Arc::new(AsyncRing::new(config.runtime.execution_event_capacity));
    let account_metrics = Arc::new(AccountStreamMetrics::default());
    let account_healthy = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let account_task = tokio::spawn(run_account_stream(AccountStreamArgs {
        ws_url: config.runtime.network.ws_url().to_owned(),
        account: credentials.account().to_owned(),
        dex: instrument.dex.clone(),
        symbol: instrument.symbol.clone(),
        events: account_events.clone(),
        clock,
        latency: Some(latency.clone()),
        metrics: account_metrics.clone(),
        healthy: account_healthy.clone(),
        shutdown: shutdown_rx,
        ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
        idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
        connect_timeout: Duration::from_millis(config.runtime.ws_connect_timeout_ms),
    }));
    let (
        clearinghouse,
        open_orders,
        recent_fills,
        user_role,
        user_fees,
        active_asset_data,
        all_mids,
        user_rate_limit,
    ) = tokio::try_join!(
        client.clearinghouse_state(),
        client.open_orders(),
        client.recent_fills(),
        client.user_role(),
        client.user_fees(),
        client.active_asset_data(),
        client.all_mids(),
        client.user_rate_limit(),
    )?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(duration_seconds);
    let mut channel_events = BTreeMap::<String, u64>::new();
    let mut connection_events = 0_u64;
    let mut disconnect_events = 0_u64;
    loop {
        tokio::select! {
            () = tokio::time::sleep_until(deadline) => break,
            event = account_events.pop() => {
                match event {
                    AccountStreamEvent::Connected { .. } => connection_events += 1,
                    AccountStreamEvent::SubscriptionAcknowledged { .. } => {}
                    AccountStreamEvent::Data { channel, .. } => {
                        *channel_events.entry(channel).or_default() += 1;
                    }
                    AccountStreamEvent::Disconnected { .. } => disconnect_events += 1,
                }
            }
        }
    }
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(Duration::from_secs(5), account_task)
        .await
        .context("account WebSocket did not stop")?
        .context("account WebSocket task panicked")?;
    latency_observer.stop()?;
    let cashcat_position = clearinghouse
        .asset_positions
        .iter()
        .find(|position| position.position.coin == instrument.symbol)
        .map(|position| &position.position);
    let cashcat_open_orders = open_orders
        .iter()
        .filter(|order| order.coin == instrument.symbol)
        .count();
    let cashcat_recent_fills = recent_fills
        .iter()
        .filter(|fill| fill.coin == instrument.symbol)
        .count();
    let cashcat_fill_details: Vec<_> = recent_fills
        .iter()
        .filter(|fill| fill.coin == instrument.symbol)
        .take(10)
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "mode": "connector_check_read_only",
            "actions_sent": 0,
            "instrument": instrument,
            "identity": {
                "account": credentials.account(),
                "agent": credentials.agent_address(),
                "is_vault": credentials.is_vault(),
                "role": user_role,
            },
            "account": {
                "account_value": clearinghouse.margin_summary.account_value,
                "withdrawable": clearinghouse.withdrawable,
                "total_margin_used": clearinghouse.margin_summary.total_margin_used,
                "total_notional_position": clearinghouse.margin_summary.total_ntl_pos,
                "position_count": clearinghouse.asset_positions.len(),
                "cashcat_position": cashcat_position,
                "open_order_count": open_orders.len(),
                "cashcat_open_order_count": cashcat_open_orders,
                "recent_fill_count": recent_fills.len(),
                "cashcat_recent_fill_count": cashcat_recent_fills,
                "cashcat_recent_fills": cashcat_fill_details,
            },
            "fees": user_fees,
            "active_asset_data": active_asset_data,
            "user_rate_limit": user_rate_limit,
            "cashcat_mid": all_mids.get(&instrument.symbol),
            "account_websocket": {
                "healthy": account_healthy.load(Ordering::Acquire),
                "connection_events": connection_events,
                "disconnect_events": disconnect_events,
                "channel_events": channel_events,
                "metrics": account_metrics.snapshot(),
            },
            "latency": &*latency.snapshot(),
        }))?
    );
    Ok(())
}

#[cfg(feature = "live-acceptance")]
async fn run_live_canary(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    credentials_path: &Path,
    phase: CanaryPhase,
    max_notional_usdc: f64,
    confirm_real_money_risk: bool,
) -> Result<()> {
    if !confirm_real_money_risk {
        bail!("live-canary requires --confirm-real-money-risk before credentials are loaded");
    }
    if !max_notional_usdc.is_finite()
        || max_notional_usdc < instrument.minimum_notional
        || max_notional_usdc > 12.0
    {
        bail!(
            "live-canary max notional must be between venue minimum {} and hard cap 12 USDC",
            instrument.minimum_notional
        );
    }
    let started_at_ms = unix_ms();
    let credentials = Arc::new(HyperliquidCredentials::load(credentials_path)?);
    if !credentials.is_vault() {
        bail!("live-canary is restricted to a dedicated subaccount/vault");
    }
    let clock = Arc::new(ProcessClock::default());
    let latency = Arc::new(LatencyMonitor::new(
        &instrument.symbol,
        started_at_ms,
        &config.latency,
        false,
    ));
    let latency_observer = LatencyObserver::spawn(
        latency.clone(),
        clock.clone(),
        instrument.symbol.clone(),
        started_at_ms,
        config.latency.clone(),
        false,
        Duration::from_millis(config.runtime.stats_interval_ms),
        config.storage.latency_path.clone(),
    )?;
    let client = HyperliquidExchangeClient::new(
        config.runtime.network,
        instrument.clone(),
        credentials.clone(),
        clock.clone(),
        Some(latency.clone()),
        Duration::from_secs(5),
        5_000,
        config.live.max_rest_weight_per_minute,
        config.live.safety_rest_weight_reserve,
    )?;
    let initial_state = client.clearinghouse_state().await?;
    let initial_orders = client.open_orders().await?;
    if !initial_orders.is_empty() {
        bail!("live-canary requires zero account open orders and will not cancel foreign orders");
    }
    let nonzero_positions: Vec<_> = initial_state
        .asset_positions
        .iter()
        .filter_map(|position| {
            position
                .position
                .szi
                .parse::<f64>()
                .ok()
                .filter(|size| size.is_finite() && *size != 0.0)
                .map(|size| (&position.position, size))
        })
        .collect();
    if !nonzero_positions.is_empty() {
        if nonzero_positions.len() != 1 || nonzero_positions[0].0.coin != instrument.symbol {
            bail!("live-canary found an unrelated position and refuses to alter it");
        }
        let book = client.l2_book().await?;
        let (bid, ask) = best_book_prices(&book)?;
        let exposure = nonzero_positions[0].1.abs() * bid.max(ask);
        if exposure > max_notional_usdc * 1.05 {
            bail!("existing CASHCAT exposure {exposure:.6} exceeds canary recovery cap");
        }
        let mut recovery_cloids = Vec::new();
        cleanup_canary(
            &client,
            &instrument,
            started_at_ms,
            &mut recovery_cloids,
            &[],
        )
        .await?;
        let recovered = client.clearinghouse_state().await?;
        let recovered_orders = client.open_orders().await?;
        latency_observer.stop()?;
        let recovered_position = cashcat_position_units(&recovered, &instrument)?;
        if recovered_position != 0 || !recovered_orders.is_empty() {
            bail!("CRITICAL: emergency canary recovery did not leave the account flat and empty");
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "mode": "explicit_real_money_connector_canary_recovery",
                "recovered_position_units": nonzero_positions[0].1,
                "estimated_exposure_usdc": exposure,
                "final_position_units": recovered_position,
                "final_open_orders": recovered_orders.len(),
                "latency_gate_enforced": false,
                "latency": &*latency.snapshot(),
            }))?
        );
        return Ok(());
    }
    for position in &initial_state.asset_positions {
        let size = position
            .position
            .szi
            .parse::<f64>()
            .context("account position size is not numeric")?;
        if !size.is_finite() || size != 0.0 {
            bail!("live-canary requires a completely flat dedicated account");
        }
    }
    let role = client.user_role().await?;
    if role.get("role").and_then(serde_json::Value::as_str) != Some("subAccount") {
        bail!("live-canary requires userRole=subAccount");
    }
    let initial_equity = initial_state
        .margin_summary
        .account_value
        .parse::<f64>()
        .context("account value is not numeric")?;
    let account_events = Arc::new(AsyncRing::new(config.runtime.execution_event_capacity));
    let account_metrics = Arc::new(AccountStreamMetrics::default());
    let account_healthy = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let account_task = tokio::spawn(run_account_stream(AccountStreamArgs {
        ws_url: config.runtime.network.ws_url().to_owned(),
        account: credentials.account().to_owned(),
        dex: instrument.dex.clone(),
        symbol: instrument.symbol.clone(),
        events: account_events.clone(),
        clock,
        latency: Some(latency.clone()),
        metrics: account_metrics.clone(),
        healthy: account_healthy.clone(),
        shutdown: shutdown_rx,
        ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
        idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
        connect_timeout: Duration::from_millis(config.runtime.ws_connect_timeout_ms),
    }));
    tokio::time::timeout(Duration::from_secs(5), async {
        while account_metrics.snapshot().subscription_acks < 8 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("account WebSocket did not acknowledge all subscriptions")?;

    let mut owned_cloids = Vec::new();
    let mut owned_oids = Vec::new();
    let phase_result = match phase {
        CanaryPhase::Passive => {
            run_passive_canary(
                &client,
                &instrument,
                started_at_ms,
                max_notional_usdc,
                &mut owned_cloids,
                &mut owned_oids,
            )
            .await
        }
        CanaryPhase::RoundTrip => {
            run_round_trip_canary(
                &client,
                &instrument,
                started_at_ms,
                max_notional_usdc,
                &mut owned_cloids,
                &mut owned_oids,
            )
            .await
        }
    };
    let cleanup_result = cleanup_canary(
        &client,
        &instrument,
        started_at_ms,
        &mut owned_cloids,
        &owned_oids,
    )
    .await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let final_state = client.clearinghouse_state().await?;
    let final_orders = client.open_orders().await?;
    let final_equity = final_state
        .margin_summary
        .account_value
        .parse::<f64>()
        .context("final account value is not numeric")?;
    let recent_fills = client.recent_fills().await?;
    let canary_fills: Vec<_> = recent_fills
        .iter()
        .filter(|fill| fill.coin == instrument.symbol && fill.time >= started_at_ms)
        .collect();
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(Duration::from_secs(5), account_task)
        .await
        .context("account WebSocket did not stop")?
        .context("account WebSocket task panicked")?;
    latency_observer.stop()?;
    cleanup_result?;
    let final_position_units = cashcat_position_units(&final_state, &instrument)?;
    if final_position_units != 0 || !final_orders.is_empty() {
        bail!(
            "CRITICAL: canary cleanup incomplete: position_units={final_position_units}, open_orders={}",
            final_orders.len()
        );
    }
    let phase_evidence = phase_result?;
    let mut channel_events = BTreeMap::<String, u64>::new();
    while let Some(event) = account_events.try_pop() {
        if let AccountStreamEvent::Data { channel, .. } = event {
            *channel_events.entry(channel).or_default() += 1;
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "mode": "explicit_real_money_connector_canary",
            "phase": format!("{phase:?}").to_lowercase(),
            "instrument": instrument,
            "hard_max_notional_usdc": max_notional_usdc,
            "latency_gate_enforced": false,
            "identity": {
                "account": credentials.account(),
                "agent": credentials.agent_address(),
                "is_vault": credentials.is_vault(),
                "role": role,
            },
            "phase_evidence": phase_evidence,
            "canary_fills": canary_fills,
            "equity_before_usdc": initial_equity,
            "equity_after_usdc": final_equity,
            "equity_change_usdc": final_equity - initial_equity,
            "final_position_units": final_position_units,
            "final_open_orders": final_orders.len(),
            "account_websocket": {
                "healthy": account_healthy.load(Ordering::Acquire),
                "channel_events": channel_events,
                "metrics": account_metrics.snapshot(),
            },
            "latency": &*latency.snapshot(),
        }))?
    );
    Ok(())
}

#[cfg(feature = "live-acceptance")]
async fn run_passive_canary(
    client: &HyperliquidExchangeClient,
    instrument: &mm_live::InstrumentSpec,
    session_started_at_ms: u64,
    max_notional_usdc: f64,
    owned_cloids: &mut Vec<String>,
    owned_oids: &mut Vec<u64>,
) -> Result<serde_json::Value> {
    let book = client.l2_book().await?;
    let (bid, _) = best_book_prices(&book)?;
    let px_units = rounded_price_units(instrument, bid * 0.99, false)?;
    let qty_units = minimum_canary_quantity(instrument, px_units, max_notional_usdc)?;
    let cloid = make_cloid(session_started_at_ms, 1, mm_live::types::Side::Buy, 1);
    owned_cloids.push(cloid.clone());
    let request = LiveOrderRequest {
        side: mm_live::types::Side::Buy,
        px_units,
        qty_units,
        reduce_only: false,
        time_in_force: TimeInForce::Alo,
        cloid: cloid.clone(),
    };
    let place = client.place_orders(&[request]).await?;
    let place_body = place.require_known()?.clone();
    ensure_top_level_action_ok(&place_body)?;
    let status = first_action_status(&place_body)?;
    let oid = status
        .get("resting")
        .and_then(|resting| resting.get("oid"))
        .and_then(serde_json::Value::as_u64)
        .context("passive ALO did not return a resting OID")?;
    owned_oids.push(oid);
    let cancel = client.cancel_by_cloid(&[cloid]).await?;
    let cancel_body = cancel.require_known()?.clone();
    ensure_top_level_action_ok(&cancel_body)?;
    Ok(serde_json::json!({
        "price_units": px_units,
        "quantity_units": qty_units,
        "notional_usdc": order_notional(instrument, px_units, qty_units),
        "resting_oid": oid,
        "place_response": place_body,
        "cancel_response": cancel_body,
    }))
}

#[cfg(feature = "live-acceptance")]
async fn run_round_trip_canary(
    client: &HyperliquidExchangeClient,
    instrument: &mm_live::InstrumentSpec,
    session_started_at_ms: u64,
    max_notional_usdc: f64,
    owned_cloids: &mut Vec<String>,
    owned_oids: &mut Vec<u64>,
) -> Result<serde_json::Value> {
    let book = client.l2_book().await?;
    let (_, ask) = best_book_prices(&book)?;
    let px_units = rounded_price_units(instrument, ask * 1.005, true)?;
    let qty_units = minimum_canary_quantity(instrument, px_units, max_notional_usdc)?;
    let cloid = make_cloid(session_started_at_ms, 2, mm_live::types::Side::Buy, 1);
    owned_cloids.push(cloid.clone());
    let request = LiveOrderRequest {
        side: mm_live::types::Side::Buy,
        px_units,
        qty_units,
        reduce_only: false,
        time_in_force: TimeInForce::Ioc,
        cloid,
    };
    let outcome = client.place_orders(&[request]).await?;
    let body = outcome.require_known()?.clone();
    ensure_top_level_action_ok(&body)?;
    let status = first_action_status(&body)?;
    let filled = status
        .get("filled")
        .context("round-trip entry IOC did not fill")?;
    if let Some(oid) = filled.get("oid").and_then(serde_json::Value::as_u64) {
        owned_oids.push(oid);
    }
    Ok(serde_json::json!({
        "entry_price_limit_units": px_units,
        "entry_quantity_units": qty_units,
        "entry_max_notional_usdc": order_notional(instrument, px_units, qty_units),
        "entry_response": body,
    }))
}

#[cfg(feature = "live-acceptance")]
async fn cleanup_canary(
    client: &HyperliquidExchangeClient,
    instrument: &mm_live::InstrumentSpec,
    session_started_at_ms: u64,
    owned_cloids: &mut Vec<String>,
    owned_oids: &[u64],
) -> Result<()> {
    let mut flat_confirmations = 0_u8;
    for (attempt, slippage) in [0.005_f64, 0.005, 0.02, 0.05, 0.10, 0.10]
        .into_iter()
        .enumerate()
    {
        let open_orders = client.open_orders().await?;
        let owned_open_oids: Vec<u64> = open_orders
            .iter()
            .filter(|order| {
                order
                    .cloid
                    .as_ref()
                    .is_some_and(|cloid| owned_cloids.contains(cloid))
                    || owned_oids.contains(&order.oid)
            })
            .map(|order| order.oid)
            .collect();
        if !owned_open_oids.is_empty() {
            let _ = client.cancel_by_oid(&owned_open_oids).await;
        }
        let state = client.clearinghouse_state().await?;
        let position_units = cashcat_position_units(&state, instrument)?;
        if position_units == 0 && owned_open_oids.is_empty() {
            flat_confirmations += 1;
            if flat_confirmations >= 2 {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
            continue;
        }
        flat_confirmations = 0;
        if position_units != 0 {
            let book = client.l2_book().await?;
            let (bid, ask) = best_book_prices(&book)?;
            let (side, price, upward) = if position_units > 0 {
                (mm_live::types::Side::Sell, bid * (1.0 - slippage), false)
            } else {
                (mm_live::types::Side::Buy, ask * (1.0 + slippage), true)
            };
            let px_units = rounded_price_units(instrument, price, upward)?;
            let cloid = make_cloid(
                session_started_at_ms,
                10 + attempt as u64,
                side,
                attempt as u64,
            );
            owned_cloids.push(cloid.clone());
            let request = LiveOrderRequest {
                side,
                px_units,
                qty_units: position_units.abs(),
                reduce_only: true,
                time_in_force: TimeInForce::Ioc,
                cloid,
            };
            let _ = client.place_orders(&[request]).await;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    let state = client.clearinghouse_state().await?;
    let open_orders = client.open_orders().await?;
    bail!(
        "canary cleanup exhausted: CASHCAT position_units={}, account open_orders={}",
        cashcat_position_units(&state, instrument)?,
        open_orders.len()
    )
}

#[cfg(feature = "live-acceptance")]
fn best_book_prices(book: &serde_json::Value) -> Result<(f64, f64)> {
    let parse = |side: usize| -> Result<f64> {
        book.get("levels")
            .and_then(|levels| levels.get(side))
            .and_then(|levels| levels.get(0))
            .and_then(|level| level.get("px"))
            .and_then(serde_json::Value::as_str)
            .context("l2Book has no best price")?
            .parse::<f64>()
            .context("l2Book best price is invalid")
    };
    let bid = parse(0)?;
    let ask = parse(1)?;
    if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= bid {
        bail!("l2Book best prices are invalid");
    }
    Ok((bid, ask))
}

#[cfg(feature = "live-acceptance")]
fn rounded_price_units(
    instrument: &mm_live::InstrumentSpec,
    price: f64,
    upward: bool,
) -> Result<i64> {
    if !price.is_finite() || price <= 0.0 {
        bail!("canary price is invalid");
    }
    let raw = price * instrument.price_scale() as f64;
    let preliminary = if upward { raw.ceil() } else { raw.floor() } as i64;
    let quantum = instrument.price_quantum(preliminary);
    let rounded = if upward {
        preliminary.saturating_add(quantum - 1) / quantum * quantum
    } else {
        preliminary / quantum * quantum
    };
    Ok(rounded)
}

#[cfg(feature = "live-acceptance")]
fn minimum_canary_quantity(
    instrument: &mm_live::InstrumentSpec,
    px_units: i64,
    max_notional_usdc: f64,
) -> Result<i64> {
    let target = instrument.minimum_notional * 1.05;
    let raw =
        target * instrument.price_scale() as f64 * instrument.size_scale() as f64 / px_units as f64;
    let qty_units = raw.ceil() as i64;
    let notional = order_notional(instrument, px_units, qty_units);
    let micro_maximum = instrument.minimum_notional * 1.10;
    if qty_units <= 0 || notional > max_notional_usdc || notional > micro_maximum {
        bail!("minimum canary order would be {notional:.6} USDC, above cap {max_notional_usdc:.6}");
    }
    Ok(qty_units)
}

#[cfg(feature = "live-acceptance")]
fn order_notional(instrument: &mm_live::InstrumentSpec, px_units: i64, qty_units: i64) -> f64 {
    instrument.price_from_units(px_units) * instrument.size_from_units(qty_units)
}

#[cfg(feature = "live-acceptance")]
fn cashcat_position_units(
    state: &mm_live::hyperliquid::exchange::ClearinghouseState,
    instrument: &mm_live::InstrumentSpec,
) -> Result<i64> {
    state
        .asset_positions
        .iter()
        .find(|position| position.position.coin == instrument.symbol)
        .map_or(Ok(0), |position| {
            parse_fixed(&position.position.szi, instrument.sz_decimals)
        })
}

#[cfg(feature = "live-acceptance")]
fn ensure_top_level_action_ok(body: &serde_json::Value) -> Result<()> {
    if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        bail!("Hyperliquid action failed: {body}");
    }
    Ok(())
}

#[cfg(feature = "live-acceptance")]
fn first_action_status(body: &serde_json::Value) -> Result<&serde_json::Value> {
    body.pointer("/response/data/statuses/0")
        .context("Hyperliquid action response has no first status")
}

async fn run_live_flatten(config: &AppConfig, instrument: mm_live::InstrumentSpec) -> Result<()> {
    let started_at_ms = unix_ms();
    let clock = Arc::new(ProcessClock::default());
    let latency = Arc::new(LatencyMonitor::new(
        &instrument.symbol,
        started_at_ms,
        &config.latency,
        false,
    ));
    let observer = LatencyObserver::spawn(
        latency.clone(),
        clock.clone(),
        instrument.symbol.clone(),
        started_at_ms,
        config.latency.clone(),
        false,
        Duration::from_millis(config.runtime.stats_interval_ms),
        config.storage.latency_path.clone(),
    )?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let bootstrap = HyperliquidLiveBackend::bootstrap(
        config,
        instrument,
        clock,
        latency.clone(),
        shutdown_rx,
        true,
    )
    .await?;
    let mut backend = bootstrap.backend;
    let mut events = bootstrap.session_events;
    let task = bootstrap.session_task;
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let event = events
                .recv()
                .await
                .context("live flatten session stopped before readiness")?;
            backend.process_session_event(event)?;
            if backend.reconciliation_requested() {
                backend.reconcile_authoritative().await?;
            }
            if backend.operationally_healthy() {
                return Ok::<(), anyhow::Error>(());
            }
        }
    })
    .await
    .context("live flatten session readiness timed out")??;
    backend.market_close().await?;
    backend.shutdown(unix_ms()).await?;
    let final_account = backend.account_state();
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .context("live flatten session did not stop")?
        .context("live flatten session task panicked")?;
    observer.stop()?;
    if final_account.inventory_units != 0 {
        bail!(
            "live flatten failed: residual inventory {}",
            final_account.inventory_units
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "mode": "live_flatten",
            "final_account": final_account,
            "latency": &*latency.snapshot(),
        }))?
    );
    Ok(())
}

async fn run_live(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    report_path: Option<&Path>,
    duration_seconds: u64,
) -> Result<()> {
    let started_at_ms = unix_ms();
    let clock = Arc::new(ProcessClock::default());
    let gate_enforced = config.live.mode == mm_live::config::LiveMode::Production;
    let latency = Arc::new(LatencyMonitor::new(
        &instrument.symbol,
        started_at_ms,
        &config.latency,
        gate_enforced,
    ));
    let latency_observer = LatencyObserver::spawn(
        latency.clone(),
        clock.clone(),
        instrument.symbol.clone(),
        started_at_ms,
        config.latency.clone(),
        gate_enforced,
        Duration::from_millis(config.runtime.stats_interval_ms),
        config.storage.latency_path.clone(),
    )?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let bootstrap = HyperliquidLiveBackend::bootstrap(
        config,
        instrument.clone(),
        clock.clone(),
        latency.clone(),
        shutdown_rx.clone(),
        false,
    )
    .await?;
    let mut backend = bootstrap.backend;
    let mut session_events = bootstrap.session_events;
    let session_task = bootstrap.session_task;

    let mut effective_config = config.clone();
    effective_config.quoting = backend.effective_quoting_config()?;
    let (calibration_data, initial_snapshot, mut initial_surface, mut inventory_unit) =
        calibrate_model(&effective_config, &instrument, true)?;
    // Size the VPIN bucket from the calibration window's observed volume so the
    // threshold keeps its meaning as the instrument's activity changes.
    let vpin_bucket = vpin_bucket_units(
        &calibration_data,
        &instrument,
        effective_config.flow_guard.vpin_buckets_per_day,
    );
    drop(calibration_data);
    let account = backend.account_state();
    if account.inventory_units != 0 {
        inventory_unit = backend
            .persisted_inventory_unit()?
            .context("non-flat live account has no persisted inventory unit")?;
        initial_surface = solve_asymmetric(
            initial_snapshot.parameters,
            &effective_config.model,
            instrument.size_from_units(inventory_unit),
            initial_snapshot.revision,
        )?;
    }
    backend.persist_inventory_unit(inventory_unit)?;
    let model = Arc::new(arc_swap::ArcSwapOption::from(Some(Arc::new(
        prepare_model_bundle(
            initial_surface,
            inventory_unit,
            initial_snapshot.generated_at_ms,
            &effective_config,
            &clock,
        ),
    ))));
    let mut calibration_snapshot = Some(initial_snapshot);

    let metrics = Arc::new(Metrics::default());
    let quote_enabled = Arc::new(AtomicBool::new(false));
    let market_evidence_valid = Arc::new(AtomicBool::new(true));
    let events = Arc::new(AsyncRing::new(config.runtime.market_event_capacity));
    let (latest_bbo_writer, latest_bbo) = bbo_channel();
    // Order validation prices from the same shared snapshot the hot path
    // quotes from, not from the drained event ring.
    backend.attach_market_bbo(latest_bbo.clone());
    let signal = Arc::new(HotPathSignal::default());
    let (desired_writer, desired) = quote_channel();
    let market_task = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: config.runtime.network.ws_url().to_owned(),
        instrument: instrument.clone(),
        latest_bbo: latest_bbo_writer,
        events: events.clone(),
        signal: signal.clone(),
        clock: clock.clone(),
        metrics: metrics.clone(),
        latency: Some(latency.clone()),
        scientifically_valid: market_evidence_valid.clone(),
        shutdown: shutdown_rx,
        ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
        idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
        connect_timeout: Duration::from_millis(config.runtime.ws_connect_timeout_ms),
        max_trade_lag_ms: config.runtime.max_trade_lag_ms,
        max_bbo_lag_ms: config.runtime.market_stale_ms,
    }));
    let inventory_units = Arc::new(AtomicI64::new(account.inventory_units));
    let (risk_writer, risk_state) = risk_channel();
    let (flow_writer, flow_state) = flow_channel();
    // Bucket size is resized from observed volume as soon as a calibration
    // window is available; until then VPIN stays in warm-up and only the fast
    // breaker is armed.
    let mut vpin = VpinTracker::new(1, config.flow_guard.vpin_window_buckets as usize);
    vpin.resize_bucket(vpin_bucket);
    let initial_risk = backend.risk_scalars()?;
    risk_writer.store(RiskState {
        equity_usdc: account.equity_usdc,
        daily_realized_pnl_usdc: initial_risk.daily_realized_pnl_usdc,
        consecutive_losses: initial_risk.consecutive_losses,
    });
    let hot_thread = spawn_hot_path(HotPathInputs {
        latest_bbo: latest_bbo.clone(),
        signal: signal.clone(),
        desired: desired_writer,
        model: model.clone(),
        instrument: instrument.clone(),
        quoting: effective_config.quoting.clone(),
        risk: effective_config.risk.clone(),
        model_config: effective_config.model.clone(),
        inventory_units: inventory_units.clone(),
        risk_state: risk_state.clone(),
        flow_state: flow_state.clone(),
        flow_guard: config.flow_guard.clone(),
        scientifically_valid: quote_enabled.clone(),
        market_stale_ms: effective_config.runtime.market_stale_ms,
        clock: clock.clone(),
        metrics: metrics.clone(),
        latency: latency.clone(),
        latency_sample_every: effective_config.latency.hot_sample_every,
        hot_path_cpu: effective_config.runtime.hot_path_cpu,
    })?;
    let live_log_max_bytes = config.storage.live_log_max_mb.saturating_mul(1024 * 1024);
    let mut event_logger = JsonlEventLogger::create_with_rotation(
        &config.storage.report_dir,
        &format!("live-{started_at_ms}"),
        LogBackpressure::RefuseWhenFull,
        LogFormat::Zstd,
        LogRotation {
            max_bytes: live_log_max_bytes,
            keep: config.storage.live_log_keep,
        },
    )?;
    let live_heartbeat_path = config.storage.report_dir.join("live_heartbeat.json");
    let (heartbeat_tx, mut heartbeat_rx) = tokio::sync::mpsc::channel(1);
    let mut heartbeat_inflight = false;
    let mut last_live_heartbeat_ms = 0_u64;
    let warmup_deadline =
        tokio::time::Instant::now() + Duration::from_secs(config.live.startup_warmup_seconds);
    let mut armed = false;
    let mut observed_quote_seq = desired.load().quote_seq;
    let mut maintenance = tokio::time::interval(Duration::from_millis(100));
    maintenance.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let calibration_duration = Duration::from_secs(config.calibration.interval_seconds.max(1));
    let mut calibration_interval = tokio::time::interval_at(
        tokio::time::Instant::now() + calibration_duration,
        calibration_duration,
    );
    calibration_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let (calibration_tx, mut calibration_rx) =
        tokio::sync::mpsc::channel::<Result<(CalibrationSnapshot, HjbSurface, i64)>>(1);
    let mut calibration_inflight = false;
    let (reconcile_tx, mut reconcile_rx) = tokio::sync::mpsc::channel(1);
    let mut reconcile_inflight = false;
    let (rate_limit_tx, mut rate_limit_rx) = tokio::sync::mpsc::channel(1);
    let mut rate_limit_inflight = false;
    let mut rate_limit_refresh_failures = 0_u32;
    let mut rate_limit_refresh_backoff_until = tokio::time::Instant::now();
    // Backoff for failed authoritative reconciles: without it a venue blip
    // turned the 100ms maintenance tick into a retry storm that ran until the
    // REST weight limiter cut it off.
    let mut reconcile_failures = 0_u32;
    let mut reconcile_backoff_until = tokio::time::Instant::now();
    let mut feed_evidence_lost = false;
    let shutdown_signal = wait_for_shutdown_signal();
    tokio::pin!(shutdown_signal);
    let deadline = (duration_seconds > 0)
        .then(|| tokio::time::Instant::now() + Duration::from_secs(duration_seconds));

    // The loop lives inside an async block so a `?` in any branch body unwinds
    // only to here. Every exit — clean break, shutdown signal, or error — must
    // reach the teardown below, because that is what cancels resting orders.
    // Returning early would abandon live exposure to the venue dead-man
    // deadline, and an account under the venue's cumulative-volume threshold
    // cannot arm a dead-man at all.
    let loop_result: Result<()> = async {
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
                // Quote dispatch outranks the market drain: with the ring
                // ranked higher, a burst of market events deferred order
                // placement and cancellation exactly when requoting mattered
                // most. Shutdown and the deadline stay above everything.
                next = desired.changed_after(observed_quote_seq) => {
                    observed_quote_seq = next.quote_seq;
                    let backend_started_ns = clock.now_ns();
                    latency.record(
                        LatencyKind::DecisionToBackendStart,
                        backend_started_ns.saturating_sub(next.generated_ns),
                        backend_started_ns,
                    );
                    backend.reconcile(next, unix_ms()).await?;
                    let backend_done_ns = clock.now_ns();
                    latency.record(
                        LatencyKind::DecisionToBackendDone,
                        backend_done_ns.saturating_sub(next.generated_ns),
                        backend_done_ns,
                    );
                    event_logger.log("quote_decision", None, &next)?;
                }
                event = session_events.recv() => {
                    let Some(event) = event else {
                        backend.invalidate("live session event channel closed");
                        break;
                    };
                    event_logger.log_owned("live_session_event", None, event.clone())?;
                    let execution_events = backend.process_session_event(event)?;
                    apply_live_execution_events(
                        &execution_events,
                        &mut event_logger,
                        &backend,
                        &inventory_units,
                        &risk_writer,
                        &metrics,
                        &signal,
                    )?;
                }
                result = reconcile_rx.recv(), if reconcile_inflight => {
                    reconcile_inflight = false;
                    let Some(result) = result else { continue };
                    match result {
                        Ok(snapshot) => {
                            reconcile_failures = 0;
                            reconcile_backoff_until = tokio::time::Instant::now();
                            backend.apply_reconciliation(snapshot)?;
                            let execution_events = backend.maintenance(unix_ms()).await?;
                            apply_live_execution_events(
                                &execution_events,
                                &mut event_logger,
                                &backend,
                                &inventory_units,
                                &risk_writer,
                                &metrics,
                                &signal,
                            )?;
                        }
                        Err(error) => {
                            // Exponential backoff, capped at 30s: reconcile
                            // failures during venue blips must not become a
                            // 100ms retry storm. Health stays degraded (quotes
                            // paused) until a reconcile succeeds.
                            reconcile_failures = reconcile_failures.saturating_add(1);
                            let delay_ms = 2_000_u64
                                .saturating_mul(1_u64 << reconcile_failures.min(4))
                                .min(30_000);
                            reconcile_backoff_until = tokio::time::Instant::now()
                                + Duration::from_millis(delay_ms);
                            let reason = format!("live reconciliation failed: {error}");
                            backend.note_rate_limit_if_applicable(unix_ms(), &reason);
                            backend.invalidate(&reason);
                            warn!(reason, delay_ms, "live reconciliation failed; backing off");
                        }
                    }
                }
                result = rate_limit_rx.recv(), if rate_limit_inflight => {
                    rate_limit_inflight = false;
                    let Some(result) = result else { continue };
                    match result {
                        Ok(value) => {
                            rate_limit_refresh_failures = 0;
                            rate_limit_refresh_backoff_until = tokio::time::Instant::now();
                            backend.apply_user_rate_limit_refresh(&value, unix_ms())?;
                        }
                        Err(error) => {
                            rate_limit_refresh_failures =
                                rate_limit_refresh_failures.saturating_add(1);
                            let delay_ms = 2_000_u64
                                .saturating_mul(1_u64 << rate_limit_refresh_failures.min(4))
                                .min(30_000);
                            rate_limit_refresh_backoff_until = tokio::time::Instant::now()
                                + Duration::from_millis(delay_ms);
                            backend.note_user_rate_limit_refresh_failure(&error);
                        }
                    }
                }
                result = heartbeat_rx.recv(), if heartbeat_inflight => {
                    heartbeat_inflight = false;
                    let Some(result) = result else { continue };
                    result?;
                }
                event = events.pop() => {
                    // Ranked below quote dispatch, so drain a bounded batch
                    // per visit to keep the ring from growing while the quote
                    // branch wins the race.
                    let mut pending = Some(event);
                    let mut drained = 0_u32;
                    while let Some(event) = pending.take() {
                        // VPIN is a trade-flow statistic and the hot path only
                        // sees the BBO, so it is folded in here and published.
                        if let MarketEvent::Trade(print) = &event {
                            flow_writer.store(vpin.observe(print));
                        }
                        let dispatch_ns = clock.now_ns();
                        latency.record(
                            LatencyKind::MarketEventDispatch,
                            dispatch_ns.saturating_sub(event_recv_ns(&event)),
                            dispatch_ns,
                        );
                        event_logger.log("market_event", Some(event_ms(&event)), &event)?;
                        let execution_events = backend.on_market_event(&event).await?;
                        // In live mode a market event only refreshes the BBO;
                        // account state, risk publication, and the hot-path
                        // notification only need to move when execution
                        // events actually arrived.
                        if !execution_events.is_empty() {
                            apply_live_execution_events(
                                &execution_events,
                                &mut event_logger,
                                &backend,
                                &inventory_units,
                                &risk_writer,
                                &metrics,
                                &signal,
                            )?;
                        }
                        drained += 1;
                        if drained < MARKET_EVENT_DRAIN_BATCH {
                            pending = events.try_pop();
                        }
                    }
                }
                _ = maintenance.tick() => {
                    let market_valid = market_evidence_valid.load(Ordering::Acquire);
                    if !market_valid && !feed_evidence_lost {
                        // Continue-through-blips policy: taint the evidence
                        // (report-level), pause quoting until an authoritative
                        // reconcile confirms venue state, and keep the session.
                        // The market stream reconnects on its own and stale-BBO
                        // withdrawal protects quoting during the gap.
                        // A feed blip must not end a live session: the stream
                        // reconnects on its own, stale-BBO withdrawal protects
                        // quoting through the gap, and an authoritative
                        // reconcile re-establishes venue truth. Stopping would
                        // abandon an open position with no bot managing it.
                        // The report still records the evidence as invalid.
                        feed_evidence_lost = true;
                        backend.invalidate("public market feed lost continuity; session continues");
                        warn!("market feed continuity lost; reconciling and continuing");
                        if !reconcile_inflight {
                            reconcile_inflight = true;
                            backend.spawn_reconciliation(reconcile_tx.clone());
                        }
                    }
                    let can_quote = armed
                        && backend.operationally_healthy()
                        && latency.trading_allowed();
                    let was_enabled = quote_enabled.swap(can_quote, Ordering::AcqRel);
                    if was_enabled != can_quote {
                        signal.notify(HOT_SIGNAL_ACCOUNT);
                    }
                    if !armed
                        && tokio::time::Instant::now() >= warmup_deadline
                        && backend.operationally_healthy()
                        && latency.trading_allowed()
                        && latest_bbo.load().is_some()
                    {
                        backend.ensure_configured_leverage().await?;
                        if config.live.deadman_enabled {
                            backend.arm_or_refresh_deadman(unix_ms()).await?;
                        }
                        armed = true;
                        quote_enabled.store(true, Ordering::Release);
                        signal.notify(HOT_SIGNAL_ACCOUNT);
                        info!(mode = ?config.live.mode, "live backend armed after all startup gates");
                    }
                    let maintenance_now_ms = unix_ms();
                    let execution_events = backend.maintenance(maintenance_now_ms).await?;
                    apply_live_execution_events(
                        &execution_events,
                        &mut event_logger,
                        &backend,
                        &inventory_units,
                        &risk_writer,
                        &metrics,
                        &signal,
                    )?;
                    if backend.user_rate_limit_refresh_due(maintenance_now_ms)
                        && !rate_limit_inflight
                        && tokio::time::Instant::now() >= rate_limit_refresh_backoff_until
                    {
                        rate_limit_inflight = true;
                        backend.spawn_user_rate_limit_refresh(rate_limit_tx.clone());
                    }
                    if maintenance_now_ms.saturating_sub(last_live_heartbeat_ms) >= 5_000 {
                        last_live_heartbeat_ms = maintenance_now_ms;
                        let mut heartbeat = backend.health_snapshot();
                        heartbeat["market_evidence_valid"] = serde_json::json!(market_valid);
                        heartbeat["latency_trading_allowed"] =
                            serde_json::json!(latency.trading_allowed());
                        heartbeat["armed"] = serde_json::json!(armed);
                        if !heartbeat_inflight {
                            heartbeat_inflight = true;
                            let path = live_heartbeat_path.clone();
                            let bytes = serde_json::to_vec_pretty(&heartbeat)?;
                            let sender = heartbeat_tx.clone();
                            tokio::spawn(async move {
                                let result = tokio::task::spawn_blocking(move || {
                                    write_atomic_bytes(&path, &bytes)
                                })
                                .await
                                .map_err(|error| anyhow::anyhow!("heartbeat writer failed: {error}"))
                                .and_then(|result| result);
                                let _ = sender.send(result).await;
                            });
                        }
                    }
                    if backend.reconciliation_requested()
                        && !reconcile_inflight
                        && tokio::time::Instant::now() >= reconcile_backoff_until
                    {
                        reconcile_inflight = true;
                        backend.spawn_reconciliation(reconcile_tx.clone());
                    }
                }
                result = calibration_rx.recv() => {
                    calibration_inflight = false;
                    let Some(result) = result else { continue };
                    match result {
                        Ok((next, surface, next_unit)) => {
                            // The non-flat re-solve already happened on the
                            // worker against the unit captured at spawn time.
                            // Only the inventory-went-non-flat race between
                            // spawn and receipt remains; skipping keeps the
                            // last model until the next 30s interval, well
                            // inside its freshness window.
                            let account = backend.account_state();
                            let model_unit =
                                model.load_full().map(|bundle| bundle.inventory_unit);
                            if account.inventory_units != 0
                                && model_unit != Some(next_unit)
                            {
                                metrics.calibration_failures.fetch_add(1, Ordering::Relaxed);
                                warn!(
                                    next_unit,
                                    ?model_unit,
                                    "inventory went non-flat during calibration; result skipped"
                                );
                                continue;
                            }
                            backend.persist_inventory_unit(next_unit)?;
                            model.store(Some(Arc::new(prepare_model_bundle(
                                surface,
                                next_unit,
                                next.generated_at_ms,
                                &effective_config,
                                &clock,
                            ))));
                            calibration_snapshot = Some(next.clone());
                            metrics.calibration_runs.fetch_add(1, Ordering::Relaxed);
                            event_logger.log("calibration", None, &next)?;
                            signal.notify(HOT_SIGNAL_MODEL);
                        }
                        Err(error) => {
                            metrics.calibration_failures.fetch_add(1, Ordering::Relaxed);
                            warn!(%error, "live calibration refresh failed; last model remains until stale");
                        }
                    }
                }
                _ = calibration_interval.tick(), if !calibration_inflight => {
                    // While non-flat the inventory unit is pinned; hand it to
                    // the worker so the re-solve for that unit happens off the
                    // event loop instead of inline on receipt.
                    let forced_unit = (backend.account_state().inventory_units != 0)
                        .then(|| model.load_full().map(|bundle| bundle.inventory_unit))
                        .flatten();
                    calibration_inflight = true;
                    let job_config = effective_config.clone();
                    let job_instrument = instrument.clone();
                    let result_tx = calibration_tx.clone();
                    tokio::spawn(async move {
                        let result = tokio::task::spawn_blocking(move || {
                            calibrate_model_for_unit(
                                &job_config,
                                &job_instrument,
                                forced_unit,
                            )
                        })
                        .await
                        .map_err(|error| anyhow::anyhow!("live calibration worker failed: {error}"))
                        .and_then(|result| result);
                        let _ = result_tx.send(result).await;
                    });
                }
            }
        }
        Ok(())
    }
    .await;

    // Teardown. Nothing here may use `?` before the venue is flat of our orders,
    // and logging must not be fatal: if the loop died because the log writer
    // died, logging here fails too.
    if let Err(error) = &loop_result {
        backend.invalidate(&format!("live loop stopped: {error}"));
    }
    quote_enabled.store(false, Ordering::Release);
    signal.notify(HOT_SIGNAL_ACCOUNT);
    let shutdown_result = backend.shutdown(unix_ms()).await;
    let final_account = backend.account_state();
    inventory_units.store(final_account.inventory_units, Ordering::Release);
    metrics
        .inventory_units
        .store(final_account.inventory_units, Ordering::Release);
    signal.notify(HOT_SIGNAL_SHUTDOWN);
    let _ = shutdown_tx.send(true);
    let flush_result = event_logger.flush();
    let _ = tokio::time::timeout(Duration::from_secs(5), market_task).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), session_task).await;
    let hot_join_result = tokio::task::spawn_blocking(move || hot_thread.join())
        .await
        .context("cannot join live hot-path task")
        .and_then(|joined| joined.map_err(|_| anyhow::anyhow!("live hot-path thread panicked")));
    let observer_result = latency_observer.stop();
    let report = LiveSessionReport {
        schema_version: 4,
        build: mm_live::BuildInfo::current(),
        session_id: format!("{}-{started_at_ms}", instrument.symbol),
        started_at_ms,
        finished_at_ms: unix_ms(),
        config_fingerprint: config.fingerprint()?,
        instrument,
        calibration: calibration_snapshot,
        model: model
            .load_full()
            .map(|bundle| ModelReport::from_surface(&bundle.surface, bundle.inventory_unit)),
        account: backend.account_state(),
        execution: backend.diagnostics().clone(),
        metrics: metrics.snapshot(),
        latency: (*latency.snapshot()).clone(),
        scientifically_valid: backend.scientifically_valid()
            && market_evidence_valid.load(Ordering::Acquire),
        event_log_path: event_logger.path().display().to_string(),
        event_log_format: "jsonl.zst",
        event_log_max_bytes: live_log_max_bytes,
        event_log_keep: config.storage.live_log_keep,
        market_event_ring_high_water: events.high_water_mark(),
    };
    let path = report_path.map_or_else(
        || {
            config
                .storage
                .report_dir
                .join(format!("live-{started_at_ms}.json"))
        },
        Path::to_owned,
    );
    let write_result = report.write_atomic(&path);
    if write_result.is_ok() {
        println!("report={}", path.display());
    }

    // Surface the original cause first: the loop's error explains why the
    // session ended, while the teardown results only describe cleanup.
    loop_result?;
    shutdown_result?;
    flush_result?;
    hot_join_result?;
    observer_result?;
    write_result?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_live_execution_events(
    events: &[mm_live::types::ExecutionEvent],
    logger: &mut JsonlEventLogger,
    backend: &HyperliquidLiveBackend,
    inventory_units: &Arc<AtomicI64>,
    risk_state: &RiskWriter,
    metrics: &Arc<Metrics>,
    signal: &Arc<HotPathSignal>,
) -> Result<()> {
    for event in events {
        logger.log("execution_event", None, event)?;
    }
    let fill_count = events
        .iter()
        .filter(|event| matches!(event, mm_live::types::ExecutionEvent::Fill(_)))
        .count() as u64;
    if fill_count != 0 {
        metrics.fills.fetch_add(fill_count, Ordering::Relaxed);
    }
    let account = backend.account_state();
    inventory_units.store(account.inventory_units, Ordering::Relaxed);
    metrics
        .inventory_units
        .store(account.inventory_units, Ordering::Relaxed);
    // The venue does not report a loss streak, so both risk scalars come from
    // durable state rather than from `account_state()`.
    let risk = backend.risk_scalars()?;
    risk_state.store(RiskState {
        equity_usdc: account.equity_usdc,
        daily_realized_pnl_usdc: risk.daily_realized_pnl_usdc,
        consecutive_losses: risk.consecutive_losses,
    });
    if fill_count != 0 {
        signal.notify(HOT_SIGNAL_FILL);
    } else if !events.is_empty() {
        signal.notify(HOT_SIGNAL_ACCOUNT);
    }
    Ok(())
}

/// One parameter set inside the grid: its own configuration, model surface,
/// simulator and log. Nothing here is shared with a peer except the market feed.
struct PaperVariant {
    name: String,
    description: String,
    config_fingerprint: String,
    fixed_parameters: Option<CjParameters>,
    config: AppConfig,
    policy: CarteaJaimungalPolicy,
    surface: HjbSurface,
    inventory_unit: i64,
    backend: DryRunBackend,
    logger: JsonlEventLogger,
    report_path: PathBuf,
    episode_start_ns: u64,
    quote_seq: u64,
    fills: u64,
    peak_equity_usdc: f64,
    max_drawdown_usdc: f64,
    /// Per variant, because the thresholds are a lever. The VPIN statistic
    /// itself is a property of the market and is shared across variants; only
    /// the trip decision is per variant.
    guard: FlowGuard,
    mid_window: MidWindow,
    /// Set once this variant has failed. It then stops trading while the rest
    /// of the grid continues, and its report carries the reason.
    failure: Option<String>,
}

/// One variant's slice of a market event, in a form whose errors can be caught
/// per variant instead of aborting the whole grid.
#[allow(clippy::too_many_arguments)]
async fn step_paper_variant(
    variant: &mut PaperVariant,
    event: &MarketEvent,
    event_time: u64,
    bbo: Option<Bbo>,
    vpin_value: Option<f64>,
) -> Result<Option<QuoteReason>> {
    let execution_events = variant.backend.on_market_event(event).await?;
    for execution_event in &execution_events {
        variant
            .logger
            .log("execution_event", Some(event_time), execution_event)?;
    }
    // MAKER fills only. Counting every execution event made a flatten variant
    // show roughly twice its fills, since each maker entry is followed by a
    // taker exit -- and the leaderboard's fills column is what a reader uses to
    // judge whether a row has measured anything.
    variant.fills = variant.fills.saturating_add(
        execution_events
            .iter()
            .filter(|event| matches!(event, ExecutionEvent::Fill(fill) if fill.maker))
            .count() as u64,
    );
    if !variant.backend.scientifically_valid() {
        warn!(variant = %variant.name, reason = ?variant.backend.diagnostics().invalid_reason, "paper variant halted by execution risk; manual review required");
    }
    let reason = if execution_events.is_empty() {
        QuoteReason::Market
    } else {
        QuoteReason::Fill
    };
    if let Some(bbo) = bbo.filter(|_| variant.backend.scientifically_valid()) {
        return variant
            .step(bbo, event_time, reason, vpin_value)
            .await
            .map(Some);
    }
    Ok(None)
}

fn observe_grid_market(
    variants: &mut [PaperVariant],
    market: &mut grid::PaperMarketState,
    event: Option<&MarketEvent>,
    now_ns: u64,
    metrics: &Metrics,
    max_age_ms: u64,
) -> Result<()> {
    let previous = market.pause_reason;
    let withdraw = market.observe(
        event,
        now_ns,
        unix_ms(),
        metrics.feed_connected_at_ns.load(Ordering::Acquire),
        metrics.feed_disconnected_since_ms.load(Ordering::Relaxed) != 0,
        max_age_ms,
    );
    if withdraw {
        for variant in variants
            .iter_mut()
            .filter(|variant| variant.backend.scientifically_valid())
        {
            variant.backend.pause_market_data();
            variant.logger.log(
                "market_data_paused",
                None,
                &market.pause_reason.unwrap_or("connection changed"),
            )?;
        }
    }
    if previous != market.pause_reason {
        if let Some(reason) = market.pause_reason {
            warn!(reason, "paper quoting paused; recovery remains active");
        } else {
            info!("fresh BBO received; paper quoting resumed without resetting accounts");
        }
    }
    Ok(())
}

impl PaperVariant {
    /// Price this variant against the current book and hand the result to its
    /// own simulator. This is the same `policy.compute` the hot path calls; the
    /// grid deliberately does not spawn hot-path threads (see `src/grid.rs`).
    async fn step(
        &mut self,
        bbo: Bbo,
        decision_ms: u64,
        reason: QuoteReason,
        vpin: Option<f64>,
    ) -> Result<QuoteReason> {
        let account = self.backend.account_state();
        let q_exact = if self.inventory_unit == 0 {
            0.0
        } else {
            account.inventory_units as f64 / self.inventory_unit as f64
        };
        let model_now_ns = decision_ms.saturating_mul(1_000_000);
        if self.episode_start_ns == 0 {
            self.episode_start_ns = model_now_ns;
        }
        let elapsed = model_now_ns.saturating_sub(self.episode_start_ns) as f64 / 1_000_000_000.0;
        let horizon_seconds = self.config.model.horizon_seconds;
        let minimum_elapsed = horizon_seconds * self.config.model.episode_min_elapsed_fraction;
        let episode_rolled = elapsed >= horizon_seconds
            || (self.config.model.episode_reset_on_flat
                && q_exact.round() == 0.0
                && elapsed >= minimum_elapsed);
        let elapsed = if episode_rolled {
            self.episode_start_ns = model_now_ns;
            0.0
        } else {
            elapsed
        };
        let tau = (horizon_seconds - elapsed).max(0.0);
        let reason = if episode_rolled {
            QuoteReason::Episode
        } else {
            reason
        };
        let risk_state = RiskState {
            equity_usdc: account.equity_usdc,
            daily_realized_pnl_usdc: self.backend.daily_realized_pnl_usdc(),
            consecutive_losses: account.consecutive_losses,
        };
        self.quote_seq = self.quote_seq.wrapping_add(1);
        // Toxic-flow guard, mirroring the hot path's arm: empty quotes cancel
        // resting orders because a `None` target bypasses the requote hold.
        let move_bps = self.mid_window.observe(model_now_ns, bbo.mid_units());
        if self.guard.evaluate(decision_ms, move_bps, vpin) {
            let mut quotes =
                DesiredQuotes::empty(QuoteReason::ToxicFlow, self.quote_seq, model_now_ns);
            quotes.source_exchange_ms = bbo.exchange_ms;
            self.backend.reconcile(quotes, decision_ms).await?;
            self.logger.log("quote_decision", None, &quotes)?;
            return Ok(QuoteReason::ToxicFlow);
        }
        let quotes = self
            .policy
            .compute(
                &self.surface,
                bbo,
                account.inventory_units,
                self.inventory_unit,
                tau,
                self.quote_seq,
                model_now_ns,
                reason,
                risk_state,
            )
            .quotes;
        self.backend.reconcile(quotes, decision_ms).await?;
        self.logger.log("quote_decision", None, &quotes)?;
        Ok(quotes.reason)
    }

    fn observe_equity(&mut self) {
        let equity = self.backend.account_state().equity_usdc;
        if equity > self.peak_equity_usdc {
            self.peak_equity_usdc = equity;
        }
        let drawdown = self.peak_equity_usdc - equity;
        if drawdown > self.max_drawdown_usdc {
            self.max_drawdown_usdc = drawdown;
        }
    }

    fn leaderboard_row(&self, bbo: Option<Bbo>) -> grid::LeaderboardRow {
        let account = self.backend.account_state();
        let scientifically_valid = self.backend.scientifically_valid() && self.failure.is_none();
        let promotion_pnl_usdc = scientifically_valid
            .then(|| bbo.and_then(|value| self.backend.promotion_pnl_usdc(value)))
            .flatten();
        let has_live_equivalent =
            self.config.dry_run.flatten_after_ms == 0 && self.fixed_parameters.is_none();
        grid::LeaderboardRow {
            name: self.name.clone(),
            description: self.description.clone(),
            net_pnl_usdc: account.equity_usdc - self.config.dry_run.starting_equity_usdc,
            promotion_pnl_usdc,
            equity_usdc: account.equity_usdc,
            realized_pnl_usdc: account.realized_pnl_usdc,
            mark_to_market_pnl_usdc: account.mark_to_market_pnl_usdc,
            fees_usdc: account.fees_usdc,
            funding_usdc: account.funding_usdc,
            inventory_units: account.inventory_units,
            fills: self.fills,
            working_orders: self.backend.working_order_count(),
            max_drawdown_usdc: self.max_drawdown_usdc,
            scientifically_valid,
            eligible_for_promotion: scientifically_valid
                && promotion_pnl_usdc.is_some()
                && has_live_equivalent,
        }
    }
}

/// Run every variant in the grid against one shared public feed.
///
/// Deliberate differences from `run_public_dry_run`:
///
/// - **One WebSocket, N simulators.** The venue permits ten simultaneous
///   connections per IP and that budget is shared with the collector containers
///   and with any live session, so N processes is not an option.
/// - **No hot-path threads.** Each `HotPathSignal` can register exactly one
///   thread, so N variants would need N signals and N isolated cores. The grid
///   measures strategy economics, where the simulated latencies dominate real
///   compute by many orders of magnitude; `dry-run` remains the
///   latency-faithful path.
/// - **Never a Parquet writer.** Not a flag: the grid must not be able to
///   contend with the reference collector.
async fn run_dry_run_grid(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    grid_path: &Path,
    duration_seconds: u64,
    out_dir: Option<&Path>,
    history_seconds: u64,
    max_resume_gap_seconds: u64,
    max_carry_inventory_gap_seconds: u64,
    log_rotation: mm_live::report::LogRotation,
) -> Result<()> {
    if max_resume_gap_seconds > 0 && max_carry_inventory_gap_seconds > max_resume_gap_seconds {
        bail!(
            "--max-carry-inventory-gap-seconds ({max_carry_inventory_gap_seconds}) exceeds \
             --max-resume-gap-seconds ({max_resume_gap_seconds}): the carry window can \
             only narrow the resume window, never widen it"
        );
    }
    let launched_at_ms = unix_ms();
    // Overwritten by a resumed checkpoint below, so that elapsed time, the
    // downtime fraction and the equity curve all run from the original start
    // rather than restarting with the process.
    let mut started_at_ms = launched_at_ms;
    let spec = grid::GridSpec::load(grid_path)?;
    // Validate every variant before touching market data. An invalid spec must
    // fail immediately and by name -- not after a calibration it will never
    // use, and not with a data error on a machine that has no tape at all.
    for entry in &spec.variants {
        entry.overrides.apply(config).with_context(|| {
            format!("grid variant {:?} is not a valid configuration", entry.name)
        })?;
    }
    let out_dir = out_dir.map_or_else(
        || {
            config
                .storage
                .report_dir
                .join(format!("grid-{launched_at_ms}"))
        },
        Path::to_owned,
    );
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("cannot create grid output directory {}", out_dir.display()))?;
    let _grid_lock = grid::GridRunLock::acquire(&out_dir)?;

    let (grid_data, snapshot, _, _) = calibrate_model(config, &instrument, true)?;
    let mid = grid_data
        .mids
        .last()
        .context("calibration window has no final mid")?
        .mid;

    // The checkpoint is read before the variants are built, because it decides
    // one of their construction parameters: a resumed run keeps its ORIGINAL
    // inventory unit. That unit is derived from the calibration mid, so it
    // drifts with the market on every launch (345 -> 344 over two minutes), and
    // a restored position sized in one unit does not mean the same thing in
    // another. Keeping the checkpointed unit -- and re-solving the HJB surface
    // against it, as `dry-run` already does via `restored_inventory_unit()` --
    // is what makes the resumed run the same run rather than a similar one.
    let state_path = out_dir.join("grid_state.json");
    let mut variant_configs = Vec::with_capacity(spec.variants.len());
    for entry in &spec.variants {
        let (variant_config, parameters, fingerprint) =
            spec.resolve_variant(entry, config).with_context(|| {
                format!("grid variant {:?} is not a valid configuration", entry.name)
            })?;
        variant_configs.push((entry, variant_config, parameters, fingerprint));
    }
    // Two variants with one fingerprint are the same strategy under two names,
    // so one of twenty slots spends the whole run re-measuring its neighbour.
    // `guarded` did exactly that: `flow_guard_enabled = true` is already the
    // shipped default, so it was `baseline` with a label. Warned rather than
    // refused on purpose -- this grid's job is to come back unattended after a
    // reboot, and a startup check that can decline to start is a worse failure
    // than a wasted row.
    {
        let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
        for (entry, _, _, fingerprint) in &variant_configs {
            if let Some(twin) = seen.insert(fingerprint.as_str(), entry.name.as_str()) {
                warn!(
                    variant = %entry.name,
                    duplicate_of = %twin,
                    "two grid variants have identical configurations; one of them measures nothing"
                );
            }
        }
    }
    // The estimator semantics are part of the run's identity: a resume across
    // a parameter-schema change would splice two parameterisations into one
    // P&L curve and keep an inventory unit sized under the old one.
    let grid_fingerprint = std::iter::once(format!(
        "execution=causal-v4;estimator=v{}:{}",
        mm_live::calibration::PARAMETER_SCHEMA_VERSION,
        mm_live::calibration::ESTIMATOR_SEMANTICS
    ))
    .chain(
        variant_configs
            .iter()
            .map(|(entry, _, _, fingerprint)| format!("{}={fingerprint}", entry.name)),
    )
    .collect::<Vec<_>>()
    .join(";");
    let resume_from = load_resumable_checkpoint(
        &state_path,
        &instrument.symbol,
        &grid_fingerprint,
        launched_at_ms,
        max_resume_gap_seconds,
        &variant_configs
            .iter()
            .map(|(entry, _, _, fingerprint)| (entry.name.as_str(), fingerprint.as_str()))
            .collect::<Vec<_>>(),
    );
    let run_id = resume_from.as_ref().map_or_else(
        || format!("run-{launched_at_ms}"),
        |state| state.run_id.clone(),
    );
    let run_dir = out_dir.join("runs").join(&run_id);
    std::fs::create_dir_all(&run_dir)
        .with_context(|| format!("cannot create immutable grid run {}", run_dir.display()))?;

    let mut variants = Vec::with_capacity(spec.variants.len());
    for (entry, variant_config, fixed_parameters, config_fingerprint) in variant_configs {
        let policy = CarteaJaimungalPolicy::new(
            instrument.clone(),
            variant_config.quoting.clone(),
            variant_config.risk.clone(),
        )?;
        let inventory_unit = resume_from
            .as_ref()
            .and_then(|state| state.variants.iter().find(|v| v.name == entry.name))
            .map_or_else(
                || policy.derive_inventory_unit(mid, variant_config.model.q_max),
                |persisted| Ok(persisted.inventory_unit),
            )?;
        let surface = solve_asymmetric(
            fixed_parameters.unwrap_or(snapshot.parameters),
            &variant_config.model,
            instrument.size_from_units(inventory_unit),
            snapshot.revision,
        )
        .with_context(|| format!("grid variant {:?} HJB solve failed", entry.name))?;
        let backend = DryRunBackend::new(
            instrument.clone(),
            variant_config.dry_run.clone(),
            variant_config.quoting.clone(),
            variant_config.risk.clone(),
        )?;
        let logger = JsonlEventLogger::create_with_rotation(
            &run_dir,
            &format!("grid-{}", entry.name),
            LogBackpressure::RefuseWhenFull,
            LogFormat::Zstd,
            log_rotation,
        )?;
        logger.log(
            "run_started",
            None,
            &serde_json::json!({
                "started_at_ms": started_at_ms,
                "variant": entry.name,
                "overrides": entry.overrides.describe(),
                "parameters": fixed_parameters.unwrap_or(snapshot.parameters),
                "build": mm_live::BuildInfo::current(),
            }),
        )?;
        let guard_config = variant_config.flow_guard.clone();
        let guard_window_ms = guard_config.fast_move_window_ms;
        let mid_capacity = guard_window_ms
            .saturating_mul(200)
            .div_ceil(1_000)
            .clamp(64, 8_192) as usize;
        info!(
            variant = %entry.name,
            overrides = %entry.overrides.describe(),
            inventory_unit,
            flow_guard = guard_config.enabled,
            "grid variant armed"
        );
        variants.push(PaperVariant {
            name: entry.name.clone(),
            description: entry.overrides.describe(),
            config_fingerprint,
            fixed_parameters,
            report_path: run_dir.join(format!("{}.json", entry.name)),
            peak_equity_usdc: variant_config.dry_run.starting_equity_usdc,
            config: variant_config,
            policy,
            surface,
            inventory_unit,
            backend,
            logger,
            episode_start_ns: 0,
            quote_seq: 0,
            fills: 0,
            max_drawdown_usdc: 0.0,
            guard: FlowGuard::new(guard_config),
            mid_window: MidWindow::new(mid_capacity, guard_window_ms),
            failure: None,
        });
    }

    let mut resumes = 0_u32;
    let mut resumed_downtime_ms = 0_u64;
    let mut resumed_feed = (0_u64, 0_u64, 0_u64, 0_u64, 0_u64);
    let mut resumed_event_loss = false;
    if let Some(state) = resume_from {
        let gap_ms = launched_at_ms.saturating_sub(state.checkpoint_ms);
        match resume_grid(&mut variants, &state) {
            Ok(()) => {
                started_at_ms = state.started_at_ms;
                resumes = state.resumes.saturating_add(1);
                resumed_downtime_ms = state.resumed_downtime_ms.saturating_add(gap_ms);
                resumed_feed = (
                    state.feed_health.gaps,
                    state.feed_health.downtime_ms,
                    state.feed_health.longest_gap_ms,
                    state.trade_prints,
                    state.replayed_trades_ignored,
                );
                resumed_event_loss = state.feed_health.event_loss;
                let carry_limit_ms = max_carry_inventory_gap_seconds.saturating_mul(1_000);
                let (flattened_variants, flattened_pnl_usdc) =
                    if gap_ms > carry_limit_ms && !resumed_event_loss {
                        let mut count = 0_usize;
                        let mut total = 0.0_f64;
                        for variant in &mut variants {
                            if let Some(closed) = variant.backend.flatten_carried_position()? {
                                count += 1;
                                total += closed;
                            }
                        }
                        (count, total)
                    } else {
                        (0, 0.0)
                    };
                info!(
                    gap_ms,
                    resumes,
                    resumed_downtime_ms,
                    elapsed_seconds = launched_at_ms.saturating_sub(started_at_ms) / 1_000,
                    variants = variants.len(),
                    flattened_variants,
                    flattened_pnl_usdc,
                    "resumed the previous grid run; the interruption is process downtime, not feed downtime"
                );
                if flattened_variants > 0 {
                    warn!(
                        gap_ms,
                        carry_limit_ms,
                        flattened_variants,
                        "the gap exceeded the inventory carry window; every open position was \
                         closed at its last observed touch with promotion exit costs, so P&L after the gap is \
                         attributable to decisions taken after it"
                    );
                }
            }
            Err(error) => {
                return Err(error.context("validated checkpoint could not be restored"));
            }
        }
    }

    let metrics = Arc::new(Metrics::default());
    // Seed the counters so a resumed run reports cumulative totals rather than
    // this process's slice of them.
    metrics.feed_gaps.store(resumed_feed.0, Ordering::Relaxed);
    metrics
        .feed_downtime_ms
        .store(resumed_feed.1, Ordering::Relaxed);
    metrics
        .feed_longest_gap_ms
        .store(resumed_feed.2, Ordering::Relaxed);
    metrics
        .trade_prints
        .store(resumed_feed.3, Ordering::Relaxed);
    metrics
        .historical_trade_prints_ignored
        .store(resumed_feed.4, Ordering::Relaxed);
    let scientifically_valid = Arc::new(AtomicBool::new(!resumed_event_loss));
    let events = Arc::new(AsyncRing::new(config.runtime.market_event_capacity));
    let (latest_bbo_writer, latest_bbo) = bbo_channel();
    // Nothing registers against this signal: the grid has no hot-path thread.
    // The market stream still notifies it, which is a cheap atomic OR.
    let signal = Arc::new(HotPathSignal::default());
    let clock = Arc::new(ProcessClock::default());
    let latency = Arc::new(LatencyMonitor::new(
        &instrument.symbol,
        started_at_ms,
        &config.latency,
        false,
    ));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let mut market_task = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: config.runtime.network.ws_url().to_owned(),
        instrument: instrument.clone(),
        latest_bbo: latest_bbo_writer,
        events: events.clone(),
        signal: signal.clone(),
        clock: clock.clone(),
        metrics: metrics.clone(),
        latency: Some(latency.clone()),
        scientifically_valid: scientifically_valid.clone(),
        shutdown: shutdown_rx.clone(),
        ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
        idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
        connect_timeout: Duration::from_millis(config.runtime.ws_connect_timeout_ms),
        max_trade_lag_ms: config.runtime.max_trade_lag_ms,
        max_bbo_lag_ms: config.runtime.market_stale_ms,
    }));

    // VPIN is a property of the market, not of a parameter set, so one tracker
    // feeds every variant; only the trip threshold is per variant.
    let mut vpin = VpinTracker::new(
        vpin_bucket_units(
            &grid_data,
            &instrument,
            config.flow_guard.vpin_buckets_per_day,
        ),
        config.flow_guard.vpin_window_buckets as usize,
    );
    let mut vpin_value: Option<f64> = None;

    let leaderboard_path = out_dir.join("leaderboard.json");
    let run_leaderboard_path = run_dir.join("leaderboard.json");
    // Append-only, so a restart extends the same series rather than replacing
    // it -- the whole point is a curve that survives the run.
    let mut history = (history_seconds > 0)
        .then(|| {
            grid::EquityHistory::create(
                &run_dir.join("equity_history.csv"),
                history_seconds.saturating_mul(1_000),
            )
        })
        .transpose()?;
    let mut stats = tokio::time::interval(Duration::from_millis(
        config.runtime.stats_interval_ms.max(1_000),
    ));
    stats.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_feed_counters = (0_u64, 0_u64, 0_u64, false);
    let mut last_feed_log_ms = 0_u64;
    let deadline = (duration_seconds > 0)
        .then(|| tokio::time::Instant::now() + Duration::from_secs(duration_seconds));
    let shutdown_signal = wait_for_shutdown_signal();
    tokio::pin!(shutdown_signal);
    let mut last_event_exchange_ms = 0_u64;
    let mut market = grid::PaperMarketState::default();

    let loop_result: Result<()> = async {
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
                result = &mut market_task => {
                    bail!("public market task exited unexpectedly: {result:?}");
                }
                _ = stats.tick() => {
                    observe_grid_market(&mut variants, &mut market, None, clock.now_ns(), &metrics, config.runtime.market_stale_ms)?;
                    let feed = metrics.snapshot();
                    let now_ms = unix_ms();
                    let feed_down_for_ms = feed.feed_down_for_ms(now_ms);
                    let feed_is_down = feed_down_for_ms > 0;
                    let counters = (
                        feed.reconnects,
                        feed.feed_gaps,
                        feed.historical_trade_prints_ignored,
                        feed_is_down,
                    );
                    let floor_ms = if feed_is_down { 60_000 } else { 600_000 };
                    if counters != last_feed_counters
                        || now_ms.saturating_sub(last_feed_log_ms) >= floor_ms
                    {
                        last_feed_counters = counters;
                        last_feed_log_ms = now_ms;
                        if feed_is_down {
                            warn!(
                                feed_down_for_ms,
                                reconnects = feed.reconnects,
                                feed_gaps = feed.feed_gaps,
                                feed_downtime_ms = feed.feed_downtime_ms,
                                feed_longest_gap_ms = feed.feed_longest_gap_ms,
                                replayed_trades_ignored = feed.historical_trade_prints_ignored,
                                trade_prints = feed.trade_prints,
                                "grid feed health: public feed is DOWN"
                            );
                        } else {
                            info!(
                                feed_down_for_ms,
                                reconnects = feed.reconnects,
                                feed_gaps = feed.feed_gaps,
                                feed_downtime_ms = feed.feed_downtime_ms,
                                feed_longest_gap_ms = feed.feed_longest_gap_ms,
                                replayed_trades_ignored = feed.historical_trade_prints_ignored,
                                trade_prints = feed.trade_prints,
                                "grid feed health"
                            );
                        }
                    }
                    let board = write_grid_leaderboard(
                        &variants,
                        &run_leaderboard_path,
                        &instrument.symbol,
                        started_at_ms,
                        &metrics,
                        &config.runtime,
                        !scientifically_valid.load(Ordering::Acquire),
                        resumes,
                        resumed_downtime_ms,
                        latest_bbo.load(),
                        market.pause_reason,
                    )?;
                    board.write_atomic(&leaderboard_path)?;
                    // Checkpoint on the same tick as the leaderboard, so the
                    // two never disagree by more than one interval. A kill at
                    // any instant costs at most `stats_interval_ms` of the run.
                    checkpoint_grid(
                        &variants,
                        &state_path,
                        &instrument.symbol,
                        &grid_fingerprint,
                        started_at_ms,
                        resumes,
                        resumed_downtime_ms,
                        &metrics,
                        &run_id,
                        !scientifically_valid.load(Ordering::Acquire),
                    )?;
                    std::fs::copy(&state_path, run_dir.join("grid_state.json"))?;
                    if let Some(history) = history.as_mut() {
                        let mid = latest_bbo
                            .load()
                            .map(|bbo| instrument.price_from_units(bbo.mid_units()));
                        board.append_history(history, mid)?;
                    }
                }
                event = events.pop() => {
                    if !scientifically_valid.load(Ordering::Acquire) {
                        for variant in &mut variants {
                            if variant.backend.scientifically_valid() {
                                variant.backend.invalidate("causal market-event loss");
                            }
                        }
                        while events.try_pop().is_some() {}
                        continue;
                    }
                    let mut pending = Some(event);
                    let mut drained = 0_u32;
                    while let Some(event) = pending.take() {
                        observe_grid_market(&mut variants, &mut market, Some(&event), clock.now_ns(), &metrics, config.runtime.market_stale_ms)?;
                        if let MarketEvent::Trade(print) = &event {
                            vpin_value = vpin.observe(print);
                        }
                        let event_time = event_ms(&event);
                        if event_time != 0 {
                            last_event_exchange_ms = last_event_exchange_ms.max(event_time);
                        }
                        let bbo = market.bbo;
                        for variant in &mut variants {
                            if bbo.is_none() || variant.failure.is_some() || !variant.backend.scientifically_valid() {
                                continue;
                            }
                            let stepped = step_paper_variant(
                                variant,
                                &event,
                                last_event_exchange_ms,
                                bbo,
                                vpin_value,
                            )
                            .await;
                            if let Err(error) = stepped {
                                let reason = format!("{error:#}");
                                warn!(
                                    variant = %variant.name,
                                    error = %reason,
                                    "grid variant failed; it stops trading and the run continues"
                                );
                                variant.backend.invalidate(&reason);
                                variant.failure = Some(reason);
                                continue;
                            }
                            variant.observe_equity();
                        }
                        drained += 1;
                        if drained < MARKET_EVENT_DRAIN_BATCH {
                            pending = events.try_pop();
                        }
                    }
                }
            }
        }
        Ok(())
    }
    .await;

    // Teardown always runs, whatever ended the loop.
    let _ = shutdown_tx.send(true);
    let mut shutdown_errors = Vec::new();
    let finished_ms = last_event_exchange_ms.max(unix_ms());
    for variant in &mut variants {
        if let Err(error) = variant.backend.shutdown(finished_ms).await {
            shutdown_errors.push(format!("{}: {error}", variant.name));
        }
        if let Err(error) = variant.logger.flush() {
            shutdown_errors.push(format!("{} log flush: {error}", variant.name));
        }
    }
    let board = write_grid_leaderboard(
        &variants,
        &run_leaderboard_path,
        &instrument.symbol,
        started_at_ms,
        &metrics,
        &config.runtime,
        !scientifically_valid.load(Ordering::Acquire),
        resumes,
        resumed_downtime_ms,
        latest_bbo.load(),
        market.pause_reason,
    )?;
    board.write_atomic(&leaderboard_path)?;
    // Final checkpoint, so a deliberate stop-and-restart resumes from the run's
    // true end rather than from up to one interval earlier.
    checkpoint_grid(
        &variants,
        &state_path,
        &instrument.symbol,
        &grid_fingerprint,
        started_at_ms,
        resumes,
        resumed_downtime_ms,
        &metrics,
        &run_id,
        !scientifically_valid.load(Ordering::Acquire),
    )?;
    std::fs::copy(&state_path, run_dir.join("grid_state.json"))?;
    // Final sample regardless of the interval, so the curve ends where the run
    // ended instead of up to one interval short of it.
    if let Some(history) = history.as_mut() {
        let mid = latest_bbo
            .load()
            .map(|bbo| instrument.price_from_units(bbo.mid_units()));
        history.force_record(&board, mid)?;
    }
    // Feed gaps are measured and judged against a threshold rather than
    // latching the run invalid: a multi-day grid always sees a venue connection
    // recycle, so the old boolean was false for every long run and said nothing.
    //
    // The verdict comes from the board rather than being recomputed here, so
    // the session reports and the leaderboard cannot disagree about whether the
    // same run was valid.
    let feed_health = board.feed_health;
    let feed_failures = board.feed_failures.clone();
    let feed_valid = feed_failures.is_empty();
    if feed_valid {
        info!(
            gaps = feed_health.gaps,
            downtime_ms = feed_health.downtime_ms,
            longest_gap_ms = feed_health.longest_gap_ms,
            "public feed health within limits"
        );
    } else {
        for reason in &feed_failures {
            warn!(reason = %reason, "public feed health disqualifies this run");
        }
    }
    for variant in &variants {
        let report = SessionReport {
            schema_version: 2,
            build: mm_live::BuildInfo::current(),
            session_id: format!("{}-{}-{started_at_ms}", instrument.symbol, variant.name),
            started_at_ms,
            finished_at_ms: unix_ms(),
            mode: format!("dry_run_grid:{}", variant.name),
            config_fingerprint: variant.config_fingerprint.clone(),
            instrument: instrument.clone(),
            calibration: variant.fixed_parameters.is_none().then(|| snapshot.clone()),
            replay: None,
            model: Some(ModelReport::from_surface(
                &variant.surface,
                variant.inventory_unit,
            )),
            account: variant.backend.account_state(),
            execution: variant.backend.diagnostics().clone(),
            metrics: metrics.snapshot(),
            latency: (*latency.snapshot()).clone(),
            scientifically_valid: variant.backend.scientifically_valid() && feed_valid,
            invalid_reasons: variant
                .failure
                .iter()
                .cloned()
                .chain(feed_failures.iter().cloned())
                .collect(),
            event_log_path: run_dir.display().to_string(),
            market_event_ring_high_water: events.high_water_mark(),
        };
        report.write_atomic(&variant.report_path)?;
    }
    market_task.abort();
    println!("{}", board.render());
    println!("leaderboard={}", leaderboard_path.display());
    loop_result?;
    if !shutdown_errors.is_empty() {
        bail!("grid teardown errors: {}", shutdown_errors.join("; "));
    }
    Ok(())
}

/// The checkpoint to resume from, or `None` to start fresh -- with the reason
/// logged either way.
///
/// Every rejection here is a case where continuing would produce a number that
/// reads as one continuous measurement but is not one.
fn load_resumable_checkpoint(
    state_path: &Path,
    symbol: &str,
    grid_fingerprint: &str,
    launched_at_ms: u64,
    max_resume_gap_seconds: u64,
    expected: &[(&str, &str)],
) -> Option<grid::PersistedGridState> {
    let state = grid::PersistedGridState::load(state_path)?;
    if let Err(error) = state.validate_variants(expected) {
        warn!(%error, "checkpoint rejected before adopting run state or artifacts");
        return None;
    }
    if max_resume_gap_seconds == 0 {
        info!("resuming is disabled (--max-resume-gap-seconds 0); starting fresh");
        return None;
    }
    if let Some(reason) = state.rejection(symbol, grid_fingerprint) {
        warn!(reason = %reason, "not resuming the previous grid run; starting fresh");
        return None;
    }
    if state.checkpoint_ms > launched_at_ms {
        warn!("checkpoint is from the future; starting fresh");
        return None;
    }
    let gap_ms = launched_at_ms.saturating_sub(state.checkpoint_ms);
    let limit_ms = max_resume_gap_seconds.saturating_mul(1_000);
    if gap_ms > limit_ms {
        // Refusing is the conservative choice, not the cautious-looking one.
        // Resuming here would carry every variant's inventory across a price
        // path nobody observed and mark it at whatever the market had become --
        // precisely how the 46.4 h run came to report a 13.2% rally as profit.
        warn!(
            gap_ms,
            limit_ms,
            "previous grid run was interrupted for longer than the resume limit; starting fresh              rather than marking held inventory across an unobserved move"
        );
        return None;
    }
    Some(state)
}

/// Restore every variant's accounting from a checkpoint.
///
/// All-or-nothing: a checkpoint missing a variant, or carrying one whose
/// parameters have changed, is refused as a whole rather than applied in part.
/// A grid where some variants resumed and others started from zero equity would
/// produce a leaderboard whose rows are not comparable -- the one thing the
/// grid exists to make possible.
fn resume_grid(variants: &mut [PaperVariant], state: &grid::PersistedGridState) -> Result<()> {
    for variant in variants {
        let persisted = state
            .variants
            .iter()
            .find(|entry| entry.name == variant.name)
            .context("validated checkpoint lost a variant")?;
        variant.backend.restore_from_snapshot(
            persisted.account,
            persisted.diagnostics.clone(),
            persisted.inventory_unit,
            persisted.current_day,
            persisted.daily_realized_pnl_usdc,
        )?;
        variant.backend.restore_checkpoint_bbo(persisted.last_bbo);
        variant.fills = persisted.fills;
        variant.peak_equity_usdc = persisted.peak_equity_usdc;
        variant.max_drawdown_usdc = persisted.max_drawdown_usdc;
        variant.failure = persisted.failure.clone();
    }
    Ok(())
}

/// Write the checkpoint a later process can resume from.
#[allow(clippy::too_many_arguments)]
fn checkpoint_grid(
    variants: &[PaperVariant],
    path: &Path,
    symbol: &str,
    grid_fingerprint: &str,
    started_at_ms: u64,
    resumes: u32,
    resumed_downtime_ms: u64,
    metrics: &Metrics,
    run_id: &str,
    event_loss: bool,
) -> Result<()> {
    let feed = metrics.snapshot();
    let checkpoint_ms = unix_ms();
    let feed_down_for_ms = feed.feed_down_for_ms(checkpoint_ms);
    let mut persisted = Vec::with_capacity(variants.len());
    for variant in variants {
        let (current_day, daily_realized_pnl_usdc) = variant.backend.daily_risk_snapshot();
        persisted.push(grid::PersistedVariant {
            name: variant.name.clone(),
            config_fingerprint: variant.config_fingerprint.clone(),
            inventory_unit: variant.inventory_unit,
            last_bbo: variant.backend.checkpoint_bbo(),
            account: variant.backend.account_snapshot(),
            diagnostics: variant.backend.diagnostics().clone(),
            fills: variant.fills,
            peak_equity_usdc: variant.peak_equity_usdc,
            max_drawdown_usdc: variant.max_drawdown_usdc,
            failure: variant.failure.clone(),
            current_day,
            daily_realized_pnl_usdc,
        });
    }
    grid::PersistedGridState {
        schema_version: grid::PersistedGridState::SCHEMA_VERSION,
        symbol: symbol.to_owned(),
        grid_fingerprint: grid_fingerprint.to_owned(),
        run_id: run_id.to_owned(),
        started_at_ms,
        checkpoint_ms,
        resumes,
        resumed_downtime_ms,
        feed_health: mm_live::config::FeedHealth::new(
            feed.feed_gaps
                .saturating_add(u64::from(feed_down_for_ms > 0)),
            feed.feed_downtime_ms.saturating_add(feed_down_for_ms),
            feed.feed_longest_gap_ms.max(feed_down_for_ms),
            checkpoint_ms
                .saturating_sub(started_at_ms)
                .saturating_sub(resumed_downtime_ms),
            event_loss,
        ),
        trade_prints: feed.trade_prints,
        replayed_trades_ignored: feed.historical_trade_prints_ignored,
        variants: persisted,
    }
    .write_atomic(path)
}

/// Decide whether a running grid is healthy, from its `leaderboard.json` alone.
///
/// `Ok` carries a one-line summary for the probe log; `Err` carries the reason
/// it is unhealthy. Split out from the CLI branch so it can be tested without
/// spawning a process or exiting one.
fn grid_health_verdict(
    leaderboard: &Path,
    max_age_seconds: u64,
    max_feed_down_seconds: u64,
) -> std::result::Result<String, String> {
    let text = std::fs::read_to_string(leaderboard)
        .map_err(|error| format!("cannot read {}: {error}", leaderboard.display()))?;
    let board: grid::Leaderboard = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", leaderboard.display()))?;

    let now_ms = unix_ms();
    // A leaderboard stamped in the future means the clock moved, not that the
    // run is fresh; treat it as age zero rather than as a huge negative.
    let age_ms = now_ms.saturating_sub(board.generated_at_ms);
    if age_ms > max_age_seconds.saturating_mul(1_000) {
        return Err(format!(
            "leaderboard is {} s old (limit {max_age_seconds} s): the grid is hung or gone",
            age_ms / 1_000
        ));
    }
    if board.feed_down_for_ms > max_feed_down_seconds.saturating_mul(1_000) {
        return Err(format!(
            "public feed has been down {} s (limit {max_feed_down_seconds} s): the grid is \
             running blind",
            board.feed_down_for_ms / 1_000
        ));
    }
    let feed_status = if board.feed_down_for_ms > 0 {
        format!("feed down {} s; retrying", board.feed_down_for_ms / 1_000)
    } else {
        "feed connected".to_owned()
    };
    let quote_status = board
        .quote_pause_reason
        .as_deref()
        .unwrap_or("data ready; policy/risk gates apply");
    let valid_rows = board
        .rows
        .iter()
        .filter(|row| row.scientifically_valid)
        .count();
    let working_orders: usize = board.rows.iter().map(|row| row.working_orders).sum();
    let evidence = if board.feed_failures.is_empty() {
        String::new()
    } else {
        format!("; evidence INVALID: {}", board.feed_failures.join("; "))
    };
    Ok(format!(
        "grid responsive: leaderboard {} s old; {feed_status}; {quote_status}; {working_orders} working orders; {valid_rows}/{} scientifically valid; {} s elapsed{evidence}",
        age_ms / 1_000,
        board.rows.len(),
        board.elapsed_seconds
    ))
}

fn promote_best_config(
    base: &AppConfig,
    grid_path: &Path,
    leaderboard_path: &Path,
    output_path: &Path,
    manifest_path: &Path,
    min_elapsed_seconds: u64,
) -> Result<()> {
    let spec = grid::GridSpec::load(grid_path)?;
    let board: grid::Leaderboard = serde_json::from_slice(
        &std::fs::read(leaderboard_path)
            .with_context(|| format!("cannot read leaderboard {}", leaderboard_path.display()))?,
    )?;
    if board.elapsed_seconds < min_elapsed_seconds {
        bail!(
            "leaderboard has only {}s; {}s of corrected evidence are required",
            board.elapsed_seconds,
            min_elapsed_seconds
        );
    }
    if !board.feed_failures.is_empty() || board.feed_health.event_loss {
        bail!("leaderboard feed evidence is invalid; refusing live promotion");
    }
    let winner = board
        .rows
        .iter()
        .filter(|row| row.eligible_for_promotion)
        // Older leaderboards may predate the eligibility flag's dry-run-only
        // check. Re-read the spec so an experimental taker-flatten row can
        // never be turned into a live config whose backend has no such policy.
        .filter(|row| {
            spec.variants
                .iter()
                .find(|variant| variant.name == row.name)
                .is_some_and(|variant| {
                    variant.overrides.flatten_after_ms.unwrap_or(0) == 0
                        && variant.overrides.parameter_profile.is_none()
                })
        })
        .reduce(|best, row| {
            if row.promotion_pnl_usdc > best.promotion_pnl_usdc {
                row
            } else {
                best
            }
        })
        .context("leaderboard has no valid live-equivalent promotable row")?;
    if winner.promotion_pnl_usdc.is_none_or(|pnl| pnl <= 0.0) {
        bail!(
            "best valid promotion P&L is {:?}; live remains disabled until a variant is profitable",
            winner.promotion_pnl_usdc
        );
    }
    let variant = spec
        .variants
        .iter()
        .find(|variant| variant.name == winner.name)
        .with_context(|| format!("winner {:?} is absent from grid spec", winner.name))?;
    let mut selected = variant.overrides.apply(base)?;
    selected.live.enabled = true;
    selected.live.mode = mm_live::config::LiveMode::Production;
    selected.live.flatten_on_stop = true;
    selected.risk.max_daily_loss_usdc = selected
        .risk
        .max_daily_loss_usdc
        .min(selected.live.production_max_daily_realized_loss_usdc);
    selected.validate()?;
    // Keep the generated file portable between the Windows host and the live
    // container. AppConfig resolves these relative to the active config in
    // rust_live/run (or /opt/mm/run in the container).
    selected.instrument.evidence_path = PathBuf::from("../config/cashcat.validation.json");
    selected.storage.data_dir = PathBuf::from("../../scripts/HL_data");
    selected.storage.state_path = PathBuf::from("cashcat-dry-state.json");
    selected.storage.calibration_path = PathBuf::from("cashcat-calibration.json");
    selected.storage.latency_path = PathBuf::from("cashcat-live-latency.json");
    selected.storage.report_dir = PathBuf::from("../reports/live_active");
    selected.storage.writer_lock_path = PathBuf::from("cashcat-live-collector.lock");
    selected.live.credentials_path = PathBuf::from("../hyperliquid.env");
    selected.live.state_path = PathBuf::from("cashcat-live.redb");
    let selected_fingerprint = selected.fingerprint()?;
    let previous_fingerprint = std::fs::read(manifest_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("selected_config_fingerprint")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
        });
    let changed = previous_fingerprint.as_deref() != Some(selected_fingerprint.as_str());
    let config_text = toml::to_string_pretty(&selected)?;
    write_atomic_bytes(output_path, config_text.as_bytes())?;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "generated_at_ms": unix_ms(),
        "leaderboard_started_at_ms": board.started_at_ms,
        "leaderboard_generated_at_ms": board.generated_at_ms,
        "elapsed_seconds": board.elapsed_seconds,
        "symbol": board.symbol,
        "variant": winner.name,
        "promotion_pnl_usdc": winner.promotion_pnl_usdc,
        "net_pnl_usdc": winner.net_pnl_usdc,
        "base_config_fingerprint": base.fingerprint()?,
        "selected_config_fingerprint": selected_fingerprint,
        "changed": changed,
        "micro_live": {
            "min_order_notional_multiplier": selected.live.min_order_notional_multiplier,
            "max_order_notional_multiplier": selected.live.max_order_notional_multiplier,
            "max_directional_notional_multiplier": selected.live.max_directional_notional_multiplier,
            "max_working_gross_multiplier": selected.live.max_working_gross_multiplier,
            "max_daily_realized_loss_usdc": selected.live.production_max_daily_realized_loss_usdc,
            "address_action_reserve": selected.live.address_action_reserve,
        }
    });
    write_atomic_bytes(manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| anyhow::anyhow!(error.error))?;
    Ok(())
}

/// Write `leaderboard.json`, with the feed's verdict on the run so far folded
/// into every row.
///
/// The verdict is recomputed here, on every stats tick, rather than once in the
/// teardown. A grid can be killed at any moment -- a host reboot ended the
/// 2026-08-27 run mid-heartbeat -- and whatever it leaves behind is what gets
/// read months later. An artifact that can only tell the truth if the process
/// exits cleanly is an artifact that lies exactly when it matters.
fn write_grid_leaderboard(
    variants: &[PaperVariant],
    path: &Path,
    symbol: &str,
    started_at_ms: u64,
    metrics: &Metrics,
    runtime: &mm_live::config::RuntimeConfig,
    event_loss: bool,
    resumes: u32,
    resumed_downtime_ms: u64,
    latest_bbo: Option<Bbo>,
    quote_pause_reason: Option<&str>,
) -> Result<grid::Leaderboard> {
    let now = unix_ms();
    let feed = metrics.snapshot();
    let feed_down_for_ms = feed.feed_down_for_ms(now);
    let feed_health = mm_live::config::FeedHealth::new(
        feed.feed_gaps,
        feed.feed_downtime_ms.saturating_add(feed_down_for_ms),
        feed.feed_longest_gap_ms.max(feed_down_for_ms),
        now.saturating_sub(started_at_ms)
            .saturating_sub(resumed_downtime_ms),
        event_loss,
    );
    let feed_failures = feed_health.failures(runtime);
    let feed_valid = feed_failures.is_empty();
    let mut board = grid::Leaderboard {
        generated_at_ms: now,
        started_at_ms,
        elapsed_seconds: now.saturating_sub(started_at_ms) / 1_000,
        symbol: symbol.to_owned(),
        feed_health,
        feed_failures,
        feed_down_for_ms,
        quote_pause_reason: quote_pause_reason.map(str::to_owned),
        resumes,
        resumed_downtime_ms,
        rows: variants
            .iter()
            .map(|variant| {
                let mut row = variant.leaderboard_row(latest_bbo);
                row.scientifically_valid = row.scientifically_valid && feed_valid;
                row.eligible_for_promotion = row.eligible_for_promotion && feed_valid;
                row
            })
            .collect(),
    };
    board.sort_by_promotion_pnl();
    board.write_atomic(path)?;
    Ok(board)
}

/// Volume per VPIN bucket, derived from the calibration window rather than
/// pinned to a constant.
///
/// VPIN's denominator is `n · V`, so `V` sets the scale of the whole statistic.
/// Deriving it from observed volume (ADV / buckets-per-day, the reference
/// implementation's default of 50) means the threshold keeps its meaning as the
/// instrument's activity changes, instead of silently drifting toward 0 or 1.
fn vpin_bucket_units(
    data: &MarketDataSet,
    instrument: &mm_live::InstrumentSpec,
    buckets_per_day: u32,
) -> i64 {
    let span_ms = data.window_end_ms - data.window_start_ms;
    let span_days = (span_ms / 86_400_000.0).max(f64::MIN_POSITIVE);
    let total: f64 = data.trades.iter().map(|trade| trade.size.abs()).sum();
    if total <= 0.0 || !span_days.is_finite() {
        return 1;
    }
    let per_day = total / span_days;
    let bucket = per_day / f64::from(buckets_per_day.max(1));
    instrument.size_to_units(bucket).unwrap_or(1).max(1)
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

/// Calibration worker body: calibrate, then — when the caller is non-flat and
/// its inventory unit is therefore pinned — re-solve the surface for that unit
/// here, on the blocking pool. The receipt branch on the event loop is then
/// pure bookkeeping; the HJB solve (~1ms p99) no longer runs inline in the
/// select.
fn calibrate_model_for_unit(
    config: &AppConfig,
    instrument: &mm_live::InstrumentSpec,
    forced_unit: Option<i64>,
) -> Result<(CalibrationSnapshot, HjbSurface, i64)> {
    let (_, snapshot, mut surface, mut unit) = calibrate_model(config, instrument, true)?;
    if let Some(forced) = forced_unit {
        if forced != unit {
            surface = solve_asymmetric(
                snapshot.parameters,
                &config.model,
                instrument.size_from_units(forced),
                snapshot.revision,
            )?;
            unit = forced;
        }
    }
    Ok((snapshot, surface, unit))
}

fn prepare_model_bundle(
    surface: HjbSurface,
    inventory_unit: i64,
    generated_at_ms: u64,
    config: &AppConfig,
    clock: &ProcessClock,
) -> ModelBundle {
    ModelBundle::prepare(
        surface,
        inventory_unit,
        generated_at_ms,
        unix_ms(),
        clock.now_ns(),
        config.calibration.max_age_seconds,
        config.calibration.max_future_skew_seconds,
    )
}

async fn run_replay_command(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    report_path: Option<&Path>,
    train_fraction: f64,
    grid_path: Option<&Path>,
    variant_name: Option<&str>,
) -> Result<()> {
    let started_at_ms = unix_ms();
    let (config, fixed_parameters, config_fingerprint) = if let Some(path) = grid_path {
        let spec = grid::GridSpec::load(path)?;
        let name = variant_name.context("grid replay requires a variant name")?;
        let entry = spec
            .variants
            .iter()
            .find(|entry| entry.name == name)
            .with_context(|| format!("unknown replay variant {name:?}"))?;
        spec.resolve_variant(entry, config)?
    } else {
        (config.clone(), None, config.fingerprint()?)
    };
    let data = load_market_window(
        &config.storage.data_dir,
        &instrument.symbol,
        &config.calibration,
    )?;
    let (training, scoring) = data.split_for_replay(train_fraction)?;
    let snapshot = if fixed_parameters.is_none() {
        let candidate =
            Calibrator::new(&instrument.symbol, config.calibration.clone()).calibrate(&training)?;
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
        time_source: data.time_source,
        scored_until_ms: None,
        training_start_ms: training.window_start_ms,
        training_end_ms: training.window_end_ms,
        scoring_start_ms: scoring.window_start_ms,
        scoring_end_ms: scoring.window_end_ms,
        parameters,
        vpin_bucket_units: vpin_bucket_units(
            &training,
            &instrument,
            config.flow_guard.vpin_buckets_per_day,
        ),
    };
    let guard_window_ms = config.flow_guard.fast_move_window_ms;
    let mid_capacity = guard_window_ms
        .saturating_mul(200)
        .div_ceil(1_000)
        .clamp(64, 8_192) as usize;
    let mut variant = PaperVariant {
        name: variant_name.unwrap_or("replay").to_owned(),
        description: String::new(),
        config_fingerprint,
        fixed_parameters,
        backend: DryRunBackend::new(
            instrument.clone(),
            config.dry_run.clone(),
            config.quoting.clone(),
            config.risk.clone(),
        )?,
        logger: JsonlEventLogger::create_with_rotation(
            &config
                .storage
                .report_dir
                .join(format!("replay-{started_at_ms}")),
            "events",
            LogBackpressure::BlockWhenFull,
            LogFormat::Zstd,
            LogRotation {
                max_bytes: config.storage.live_log_max_mb * 1_024 * 1_024,
                keep: config.storage.live_log_keep,
            },
        )?,
        report_path: report_path.map_or_else(
            || {
                config
                    .storage
                    .report_dir
                    .join(format!("replay-{started_at_ms}.json"))
            },
            Path::to_owned,
        ),
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
        ParquetReplaySource::new(&scoring, &instrument)?,
        &metrics,
        replay.vpin_bucket_units,
    )
    .await;
    if let Err(error) = &result {
        variant.backend.invalidate(&format!("{error:#}"));
    }
    replay.scored_until_ms = result.as_ref().ok().copied();
    variant.logger.flush()?;
    write_report(
        &variant.config,
        Some(&variant.report_path),
        "replay",
        started_at_ms,
        instrument,
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
    result.map(|_| ())
}

async fn run_event_source<S: MarketDataSource>(
    variant: &mut PaperVariant,
    mut source: S,
    metrics: &Arc<Metrics>,
    vpin_bucket: i64,
) -> Result<u64> {
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
    Ok(decision_ms)
}

async fn run_public_dry_run(
    config: &AppConfig,
    instrument: mm_live::InstrumentSpec,
    duration_seconds: u64,
    no_write_parquet: bool,
    report_path: Option<&Path>,
) -> Result<()> {
    let started_at_ms = unix_ms();
    let mut vpin_bucket = 1_i64;
    let mut initial_model = match calibrate_model(config, &instrument, true) {
        Ok((data, snapshot, surface, inventory_unit)) => {
            vpin_bucket =
                vpin_bucket_units(&data, &instrument, config.flow_guard.vpin_buckets_per_day);
            Some((snapshot, surface, inventory_unit))
        }
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
    // All Parquet I/O (ZSTD shard writes, compaction, pruning) runs on the
    // recorder's own thread; this loop only queues events.
    let recorder = write_parquet
        .then(|| {
            ParquetRecorderHandle::spawn(ParquetEventRecorder::new(
                config.storage.data_dir.clone(),
                instrument.clone(),
                config.storage.flush_interval_seconds,
                config.storage.retention_minutes,
            ))
        })
        .transpose()?;
    // The other long-running dry-run accumulator: compressed for the same
    // reason as the grid. One file per run here, since a dry run is a session
    // rather than a series.
    let mut event_logger = JsonlEventLogger::create_with_format(
        &config.storage.report_dir,
        &format!("dry_run-{started_at_ms}"),
        LogBackpressure::RefuseWhenFull,
        LogFormat::Zstd,
    )?;
    let metrics = Arc::new(Metrics::default());
    let scientifically_valid = Arc::new(AtomicBool::new(true));
    let events = Arc::new(AsyncRing::new(config.runtime.market_event_capacity));
    let (latest_bbo_writer, latest_bbo) = bbo_channel();
    let signal = Arc::new(HotPathSignal::default());
    let (desired_writer, desired) = quote_channel();
    let clock = Arc::new(ProcessClock::default());
    let latency = Arc::new(LatencyMonitor::new(
        &instrument.symbol,
        started_at_ms,
        &config.latency,
        false,
    ));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let market_task = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: config.runtime.network.ws_url().to_owned(),
        instrument: instrument.clone(),
        latest_bbo: latest_bbo_writer,
        events: events.clone(),
        signal: signal.clone(),
        clock: clock.clone(),
        metrics: metrics.clone(),
        latency: Some(latency.clone()),
        scientifically_valid: scientifically_valid.clone(),
        shutdown: shutdown_rx.clone(),
        ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
        idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
        connect_timeout: Duration::from_millis(config.runtime.ws_connect_timeout_ms),
        max_trade_lag_ms: config.runtime.max_trade_lag_ms,
        max_bbo_lag_ms: config.runtime.market_stale_ms,
    }));
    let mut backend = DryRunBackend::new(
        instrument.clone(),
        config.dry_run.clone(),
        config.quoting.clone(),
        config.risk.clone(),
    )?;
    let config_fingerprint = config.fingerprint()?;
    let _ = backend.restore_account_state(
        &config.storage.state_path,
        &config_fingerprint,
        mm_live::calibration::PARAMETER_SCHEMA_VERSION,
    )?;
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
    let (risk_writer, risk_state) = risk_channel();
    let (flow_writer, flow_state) = flow_channel();
    // Bucket size is resized from observed volume as soon as a calibration
    // window is available; until then VPIN stays in warm-up and only the fast
    // breaker is armed.
    let mut vpin = VpinTracker::new(1, config.flow_guard.vpin_window_buckets as usize);
    vpin.resize_bucket(vpin_bucket);
    risk_writer.store(RiskState {
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
            Some(Arc::new(prepare_model_bundle(
                surface,
                inventory_unit,
                initial_snapshot.generated_at_ms,
                config,
                &clock,
            )))
        } else {
            None
        };
    let model = Arc::new(arc_swap::ArcSwapOption::from(initial_bundle));
    let latency_observer = LatencyObserver::spawn(
        latency.clone(),
        clock.clone(),
        instrument.symbol.clone(),
        started_at_ms,
        config.latency.clone(),
        false,
        Duration::from_millis(config.runtime.stats_interval_ms),
        config.storage.latency_path.clone(),
    )?;
    let hot_thread = spawn_hot_path(HotPathInputs {
        latest_bbo: latest_bbo.clone(),
        signal: signal.clone(),
        desired: desired_writer,
        model: model.clone(),
        instrument: instrument.clone(),
        quoting: config.quoting.clone(),
        risk: config.risk.clone(),
        model_config: config.model.clone(),
        inventory_units: inventory_units.clone(),
        risk_state: risk_state.clone(),
        flow_state: flow_state.clone(),
        flow_guard: config.flow_guard.clone(),
        scientifically_valid: scientifically_valid.clone(),
        market_stale_ms: config.runtime.market_stale_ms,
        clock: clock.clone(),
        metrics: metrics.clone(),
        latency: latency.clone(),
        latency_sample_every: config.latency.hot_sample_every,
        hot_path_cpu: config.runtime.hot_path_cpu,
    })?;
    let interval_duration = Duration::from_secs(config.calibration.interval_seconds.max(1));
    let mut calibration_interval = tokio::time::interval_at(
        tokio::time::Instant::now() + interval_duration,
        interval_duration,
    );
    calibration_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let (calibration_tx, mut calibration_rx) =
        tokio::sync::mpsc::channel::<Result<(CalibrationSnapshot, HjbSurface, i64)>>(1);
    let mut calibration_inflight = false;
    let mut observed_quote_seq = desired.load().quote_seq;
    let mut last_event_exchange_ms = 0_u64;
    let deadline = (duration_seconds > 0)
        .then(|| tokio::time::Instant::now() + Duration::from_secs(duration_seconds));
    let shutdown_signal = wait_for_shutdown_signal();
    tokio::pin!(shutdown_signal);
    let loop_result: Result<()> = async {
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
        // Quote dispatch outranks the market drain, mirroring the live loop.
        next = desired.changed_after(observed_quote_seq) => {
            observed_quote_seq = next.quote_seq;
            let backend_started_ns = clock.now_ns();
            latency.record(
                LatencyKind::DecisionToBackendStart,
                backend_started_ns.saturating_sub(next.generated_ns),
                backend_started_ns,
            );
            if next.source_recv_ns != 0 {
                latency.record(
                    LatencyKind::MarketDataToBackendStart,
                    backend_started_ns.saturating_sub(next.source_recv_ns),
                    backend_started_ns,
                );
            }
            // Exchange time, matching the simulator's matching engine. A BBO
            // can carry exchange_ms = 0 on a malformed frame, so fall back to
            // the newest event time seen.
            let simulated_now_ms = next.source_exchange_ms.max(last_event_exchange_ms);
            backend.reconcile(next, simulated_now_ms).await?;
            let backend_done_ns = clock.now_ns();
            latency.record(
                LatencyKind::BackendReconcile,
                backend_done_ns.saturating_sub(backend_started_ns),
                backend_done_ns,
            );
            latency.record(
                LatencyKind::DecisionToBackendDone,
                backend_done_ns.saturating_sub(next.generated_ns),
                backend_done_ns,
            );
            event_logger.log("quote_decision", None, &next)?;
        }
        event = events.pop() => {
            // Ranked below quote dispatch; drain a bounded batch per visit so
            // the ring cannot grow unboundedly while quotes win the race.
            let mut pending = Some(event);
            let mut drained = 0_u32;
            while let Some(event) = pending.take() {
                // VPIN is a trade-flow statistic, and the hot path only ever
                // sees the BBO -- so it is folded in here and published for the
                // hot path to read.
                if let MarketEvent::Trade(print) = &event {
                    flow_writer.store(vpin.observe(print));
                }
                let dispatch_ns = clock.now_ns();
                latency.record(
                    LatencyKind::MarketEventDispatch,
                    dispatch_ns.saturating_sub(event_recv_ns(&event)),
                    dispatch_ns,
                );
                let event_time = event_ms(&event);
                last_event_exchange_ms = last_event_exchange_ms.max(event_time);
                event_logger.log("market_event", Some(event_time), &event)?;
                if let Some(recorder) = recorder.as_ref() {
                    recorder.record(&event, unix_ms())?;
                }
                if !scientifically_valid.load(Ordering::Acquire) && backend.scientifically_valid() {
                    backend.invalidate("public market stream lost causal events or disconnected");
                    signal.notify(HOT_SIGNAL_ACCOUNT);
                }
                let execution_events = backend.on_market_event(&event).await?;
                for execution_event in &execution_events {
                    event_logger.log("execution_event", Some(event_time), execution_event)?;
                }
                metrics.fills.fetch_add(execution_events.len() as u64, Ordering::Relaxed);
                let account = backend.account_state();
                inventory_units.store(account.inventory_units, Ordering::Relaxed);
                metrics.inventory_units.store(account.inventory_units, Ordering::Relaxed);
                risk_writer.store(RiskState {
                    equity_usdc: account.equity_usdc,
                    daily_realized_pnl_usdc: backend.daily_realized_pnl_usdc(),
                    consecutive_losses: account.consecutive_losses,
                });
                if !execution_events.is_empty() {
                    signal.notify(HOT_SIGNAL_FILL);
                }
                drained += 1;
                if drained < MARKET_EVENT_DRAIN_BATCH {
                    pending = events.try_pop();
                }
            }
        }
        result = calibration_rx.recv() => {
            calibration_inflight = false;
            let Some(result) = result else { continue };
            match result {
                Ok((next, next_surface, next_inventory_unit)) => {
                // The non-flat re-solve already ran on the worker for the unit
                // captured at spawn time; only the went-non-flat race remains.
                let account = backend.account_state();
                if account.inventory_units != 0 {
                    let retained_unit = model
                        .load_full()
                        .map(|bundle| bundle.inventory_unit)
                        .or_else(|| backend.restored_inventory_unit());
                    if retained_unit != Some(next_inventory_unit) {
                        metrics.calibration_failures.fetch_add(1, Ordering::Relaxed);
                        warn!(
                            next_inventory_unit,
                            ?retained_unit,
                            "inventory went non-flat during calibration; result skipped"
                        );
                        continue;
                    }
                }
                model.store(Some(Arc::new(prepare_model_bundle(
                    next_surface,
                    next_inventory_unit,
                    next.generated_at_ms,
                    config,
                    &clock,
                ))));
                snapshot = Some(next.clone());
                event_logger.log("calibration", None, &next)?;
                metrics.calibration_runs.fetch_add(1, Ordering::Relaxed);
                signal.notify(HOT_SIGNAL_MODEL);
                if let Some(recorder) = recorder.as_ref() {
                    recorder.maintain(
                        unix_ms(),
                        config.storage.compact_after_minutes,
                        config.storage.retention_minutes,
                    )?;
                }
            }
            Err(error) => {
                metrics.calibration_failures.fetch_add(1, Ordering::Relaxed);
                warn!(%error, "calibration refresh failed; retaining last good model until stale");
            }
            }
        }
        _ = calibration_interval.tick(), if !calibration_inflight => {
            // While non-flat the unit is pinned; the worker re-solves for it
            // off the event loop, so the receipt branch never solves inline.
            let forced_unit = (backend.account_state().inventory_units != 0)
                .then(|| {
                    model
                        .load_full()
                        .map(|bundle| bundle.inventory_unit)
                        .or_else(|| backend.restored_inventory_unit())
                })
                .flatten();
            calibration_inflight = true;
            let job_config = config.clone();
            let job_instrument = instrument.clone();
            let result_tx = calibration_tx.clone();
            let job_latency = latency.clone();
            let job_clock = clock.clone();
            tokio::spawn(async move {
                let started_ns = job_clock.now_ns();
                let result = tokio::task::spawn_blocking(move || {
                    calibrate_model_for_unit(&job_config, &job_instrument, forced_unit)
                })
                .await
                .map_err(|error| anyhow::anyhow!("calibration worker failed: {error}"))
                .and_then(|result| result);
                let finished_ns = job_clock.now_ns();
                job_latency.record(
                    LatencyKind::CalibrationRefresh,
                    finished_ns.saturating_sub(started_ns),
                    finished_ns,
                );
                let _ = result_tx.send(result).await;
            });
            }
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = &loop_result {
        backend.invalidate(&format!("public dry-run loop stopped: {error}"));
    }
    signal.notify(HOT_SIGNAL_SHUTDOWN);
    let _ = shutdown_tx.send(true);
    let shutdown_result = backend.shutdown(unix_ms()).await;
    let state_save_result =
        if let (Some(snapshot), Some(bundle)) = (snapshot.as_ref(), model.load_full()) {
            backend.save_account_state(
                &config.storage.state_path,
                &config_fingerprint,
                &snapshot.fingerprint,
                mm_live::calibration::PARAMETER_SCHEMA_VERSION,
                bundle.inventory_unit,
            )
        } else {
            Ok(())
        };
    let log_flush_result = event_logger.flush();
    let recorder_result = if let Some(recorder) = recorder {
        // Final flush/compact/prune runs on the writer thread; this joins it.
        recorder
            .finish(
                unix_ms(),
                config.storage.compact_after_minutes,
                config.storage.retention_minutes,
            )
            .map(|(compacted, removed)| {
                info!(compacted, removed, "Parquet maintenance complete");
            })
    } else {
        Ok(())
    };
    drop(collector_lock);
    let market_task_result = tokio::time::timeout(Duration::from_secs(5), market_task)
        .await
        .context("market task did not stop within five seconds")
        .and_then(|joined| joined.context("market task panicked"));
    let hot_join_result = tokio::task::spawn_blocking(move || hot_thread.join())
        .await
        .context("cannot join hot-path task")
        .and_then(|joined| joined.map_err(|_| anyhow::anyhow!("hot-path thread panicked")));
    let observer_result = latency_observer.stop();
    let latency_snapshot = (*latency.snapshot()).clone();
    // A feed that died with no subsequent market event never reached the
    // in-loop invalidation, so the flag must be consulted directly here or the
    // report claims validity for a run whose evidence stream was broken.
    if !scientifically_valid.load(Ordering::Acquire) && backend.scientifically_valid() {
        backend.invalidate("public market stream lost causal events or disconnected");
    }
    let mut invalid_reasons = Vec::new();
    if let Some(reason) = backend.diagnostics().invalid_reason.clone() {
        invalid_reasons.push(reason);
    }
    if snapshot.is_none() {
        invalid_reasons.push("no valid calibration was produced".to_owned());
    }
    let write_result = write_report(
        config,
        report_path,
        "dry_run",
        started_at_ms,
        instrument,
        snapshot,
        model
            .load_full()
            .map(|bundle| ModelReport::from_surface(&bundle.surface, bundle.inventory_unit)),
        latency_snapshot,
        &backend,
        &metrics,
        invalid_reasons,
        event_logger.path(),
        events.high_water_mark(),
        None,
    );
    info!(
        scientifically_valid = backend.scientifically_valid(),
        "dry-run stopped"
    );
    loop_result?;
    shutdown_result?;
    state_save_result?;
    log_flush_result?;
    recorder_result?;
    market_task_result?;
    hot_join_result?;
    observer_result?;
    write_result?;
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
    latency: LatencySnapshot,
    backend: &DryRunBackend,
    metrics: &Arc<Metrics>,
    invalid_reasons: Vec<String>,
    event_log_path: &Path,
    market_event_ring_high_water: usize,
    replay: Option<ReplayInputs>,
) -> Result<()> {
    let finished_at_ms = unix_ms();
    let has_calibration = calibration.is_some() || replay.is_some();
    let mut config_fingerprint = config.fingerprint()?;
    if let Some(inputs) = &replay {
        config_fingerprint.push_str(";parameters=");
        config_fingerprint.push_str(&serde_json::to_string(&inputs.parameters)?);
    }
    let report = SessionReport {
        schema_version: 2,
        build: mm_live::BuildInfo::current(),
        session_id: format!("{}-{started_at_ms}", instrument.symbol),
        started_at_ms,
        finished_at_ms,
        mode: mode.to_owned(),
        config_fingerprint,
        instrument,
        calibration,
        replay,
        model,
        account: backend.account_state(),
        execution: backend.diagnostics().clone(),
        metrics: metrics.snapshot(),
        latency,
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

fn event_recv_ns(event: &MarketEvent) -> u64 {
    match event {
        MarketEvent::Bbo(value) => value.recv_ns,
        MarketEvent::Trade(value) => value.recv_ns,
        MarketEvent::Book(value) => value.recv_ns,
    }
}

fn init_tracing(json: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

/// Wait for any signal that means "stop", so teardown gets a chance to run.
///
/// On Windows this used to await `ctrl_c()` alone, which is `CTRL_C_EVENT` and
/// nothing else. A Start-menu shutdown sends `CTRL_SHUTDOWN_EVENT`, closing the
/// console window sends `CTRL_CLOSE_EVENT`, and logging off sends
/// `CTRL_LOGOFF_EVENT` -- so an ordinary reboot killed the process outright. On
/// 2026-08-27 that ended a 46 h grid mid-heartbeat with no teardown: no session
/// reports, no final equity sample, and no feed verdict on a run that was 42.5%
/// blind.
///
/// These handlers run against a hard OS deadline (a few seconds for a close,
/// the system shutdown timeout otherwise), so this widens the window, it does
/// not guarantee one. That is why `write_grid_leaderboard` records the feed
/// verdict on every tick instead of trusting teardown to happen.
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
    #[cfg(windows)]
    {
        use tokio::signal::windows::{ctrl_break, ctrl_close, ctrl_logoff, ctrl_shutdown};
        let mut shutdown = ctrl_shutdown()?;
        let mut close = ctrl_close()?;
        let mut logoff = ctrl_logoff()?;
        let mut brk = ctrl_break()?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result?,
            _ = shutdown.recv() => info!("received Windows shutdown signal"),
            _ = close.recv() => info!("received Windows console-close signal"),
            _ = logoff.recv() => info!("received Windows logoff signal"),
            _ = brk.recv() => info!("received Windows break signal"),
        }
    }
    #[cfg(not(any(unix, windows)))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

#[cfg(all(test, feature = "live-acceptance"))]
mod tests {
    use super::*;

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

    fn grid_variant(directory: &Path) -> PaperVariant {
        let mut config = AppConfig::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/cashcat_dryrun_realistic.toml"),
        )
        .unwrap();
        config.dry_run.decision_latency_ms = 100;
        config.dry_run.acknowledgement_latency_ms = 0;
        let instrument = cashcat();
        let inventory_unit = 1_000;
        let policy = CarteaJaimungalPolicy::new(
            instrument.clone(),
            config.quoting.clone(),
            config.risk.clone(),
        )
        .unwrap();
        let surface = solve_asymmetric(
            mm_live::hjb::CjParameters {
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
            name: "baseline".to_owned(),
            description: String::new(),
            config_fingerprint: config.fingerprint().unwrap(),
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
            peak_equity_usdc: 1_000.0,
            max_drawdown_usdc: 0.0,
            guard: FlowGuard::new(config.flow_guard.clone()),
            mid_window: MidWindow::new(128, config.flow_guard.fast_move_window_ms),
            failure: None,
            config,
        }
    }

    #[test]
    #[ignore = "numerical study: run with --release --all-features --ignored --nocapture"]
    fn paper_candidate_executable_quotes_converge_under_timestep_refinement() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let config = AppConfig::load(&root.join("config/cashcat_dryrun_realistic.toml")).unwrap();
        let spec = grid::GridSpec::load(&root.join("config/grid_cashcat.toml")).unwrap();
        let instrument = cashcat();
        let mut max_horizon = 0.0_f64;
        for name in ["sweep1", "sweep2", "sweep3", "contender_flat300"] {
            let entry = spec
                .variants
                .iter()
                .find(|entry| entry.name == name)
                .unwrap();
            let (applied, parameters, _) = spec.resolve_variant(entry, &config).unwrap();
            max_horizon = max_horizon.max(applied.model.horizon_seconds);
            let policy = CarteaJaimungalPolicy::new(
                instrument.clone(),
                applied.quoting.clone(),
                applied.risk.clone(),
            )
            .unwrap();
            let mut resolutions = Vec::new();
            for refinement in 0..=9 {
                let mut model = applied.model.clone();
                model.max_dt_seconds = 0.25 / 2.0_f64.powi(refinement);
                model.max_steps = (model.horizon_seconds / model.max_dt_seconds).ceil() as usize;
                let surface = solve_asymmetric(parameters.unwrap(), &model, 1.0, 1).unwrap();
                let mut quotes = Vec::new();
                for mid in [0.05, 0.10, 0.20, 0.25, 0.30] {
                    let unit = policy.derive_inventory_unit(mid, model.q_max).unwrap();
                    let book = Bbo {
                        bid_px: instrument.price_to_units(mid * 0.999).unwrap(),
                        ask_px: instrument.price_to_units(mid * 1.001).unwrap(),
                        bid_sz: unit,
                        ask_sz: unit,
                        exchange_ms: 1,
                        recv_ns: 1,
                    };
                    for half_inventory in -2 * model.q_max..=2 * model.q_max {
                        for remaining in (0..=160)
                            .map(|sample| f64::from(sample) / 32.0)
                            .chain([model.horizon_seconds / 2.0, model.horizon_seconds])
                        {
                            let decision = policy.compute(
                                &surface,
                                book,
                                half_inventory * unit / 2,
                                unit,
                                remaining,
                                1,
                                1,
                                QuoteReason::Market,
                                RiskState {
                                    equity_usdc: applied.dry_run.starting_equity_usdc,
                                    daily_realized_pnl_usdc: 0.0,
                                    consecutive_losses: 0,
                                },
                            );
                            for quote in [decision.quotes.bid, decision.quotes.ask] {
                                quotes.push(
                                    quote.map(|order| {
                                        (order.px, instrument.price_quantum(order.px))
                                    }),
                                );
                            }
                        }
                    }
                }
                resolutions.push((model.max_dt_seconds, quotes));
            }
            let mut configured_difference = None;
            for window in resolutions.windows(3) {
                let mut max_ticks = 0.0_f64;
                for finer in &window[1..] {
                    for (coarse, fine) in window[0].1.iter().zip(&finer.1) {
                        let difference = match (coarse, fine) {
                            (Some((coarse_px, quantum)), Some((fine_px, _))) => {
                                (*coarse_px - *fine_px).unsigned_abs() as f64 / *quantum as f64
                            }
                            (None, None) => 0.0,
                            _ => f64::INFINITY,
                        };
                        max_ticks = max_ticks.max(difference);
                    }
                }
                println!(
                    "{name}: dt={} max_quote_difference_ticks={max_ticks}",
                    window[0].0
                );
                if window[0].0 == config.model.max_dt_seconds {
                    configured_difference = Some(max_ticks);
                }
            }
            assert!(
                configured_difference.expect("configured timestep must be tested") <= 1.0,
                "{name}: configured timestep fails executable-quote convergence"
            );
        }
        assert!(
            config.model.max_steps >= (max_horizon / config.model.max_dt_seconds).ceil() as usize
        );
    }

    #[tokio::test]
    async fn paper_data_pause_withdraws_orders_and_resumes_without_resetting_risk() {
        let directory = tempfile::tempdir().unwrap();
        let mut variants = vec![grid_variant(directory.path())];
        let mut account = variants[0].backend.account_state();
        account.consecutive_losses = 2;
        variants[0]
            .backend
            .restore_from_snapshot(
                account,
                mm_live::execution::DryRunDiagnostics::default(),
                1_000,
                Some(unix_ms() / 86_400_000),
                -0.125,
            )
            .unwrap();
        let mut market = grid::PaperMarketState::default();
        let metrics = Metrics::default();
        let mut book = Bbo {
            bid_px: 99_900,
            ask_px: 100_100,
            bid_sz: 1,
            ask_sz: 1,
            exchange_ms: unix_ms(),
            recv_ns: 1_000_000,
        };
        observe_grid_market(
            &mut variants,
            &mut market,
            Some(&MarketEvent::Bbo(book)),
            book.recv_ns,
            &metrics,
            5_000,
        )
        .unwrap();
        step_paper_variant(
            &mut variants[0],
            &MarketEvent::Bbo(book),
            book.exchange_ms,
            market.bbo,
            None,
        )
        .await
        .unwrap();
        assert!(variants[0].backend.working_order_count() > 0);
        let before = variants[0].backend.account_state();
        metrics
            .feed_disconnected_since_ms
            .store(unix_ms(), Ordering::Relaxed);
        metrics.reconnects.store(1, Ordering::Relaxed);
        observe_grid_market(&mut variants, &mut market, None, 2_000_000, &metrics, 5_000).unwrap();
        assert_eq!(variants[0].backend.working_order_count(), 0);
        assert!(market.bbo.is_none());
        let paused = variants[0].backend.account_state();
        assert_eq!(paused.cash_usdc, before.cash_usdc);
        assert_eq!(paused.inventory_units, before.inventory_units);
        assert_eq!(paused.consecutive_losses, 2);
        assert_eq!(variants[0].backend.daily_realized_pnl_usdc(), -0.125);
        metrics
            .feed_disconnected_since_ms
            .store(0, Ordering::Relaxed);
        book.recv_ns = 3_000_000;
        metrics
            .feed_connected_at_ns
            .store(book.recv_ns, Ordering::Release);
        book.exchange_ms = unix_ms();
        observe_grid_market(
            &mut variants,
            &mut market,
            Some(&MarketEvent::Bbo(book)),
            book.recv_ns,
            &metrics,
            5_000,
        )
        .unwrap();
        step_paper_variant(
            &mut variants[0],
            &MarketEvent::Bbo(book),
            book.exchange_ms,
            market.bbo,
            None,
        )
        .await
        .unwrap();
        assert!(variants[0].backend.working_order_count() > 0);
        assert!(variants[0].backend.scientifically_valid());
        assert_eq!(variants[0].backend.daily_realized_pnl_usdc(), -0.125);
    }

    #[test]
    fn fixed_parameter_rows_are_not_live_promotable() {
        let directory = tempfile::tempdir().unwrap();
        let mut variant = grid_variant(directory.path());
        let book = Bbo {
            bid_px: 99_900,
            ask_px: 100_100,
            bid_sz: 1,
            ask_sz: 1,
            exchange_ms: 1_000,
            recv_ns: 1_000_000_000,
        };
        assert!(variant.leaderboard_row(Some(book)).eligible_for_promotion);
        variant.fixed_parameters = Some(CjParameters {
            lambda_plus: 1.0,
            lambda_minus: 1.0,
            kappa_plus: 1.0,
            kappa_minus: 1.0,
            epsilon_plus: 0.0,
            epsilon_minus: 0.0,
            sigma2_per_second: None,
        });
        assert!(!variant.leaderboard_row(Some(book)).eligible_for_promotion);
    }

    #[tokio::test]
    async fn grid_decisions_follow_consumed_events_not_the_source_book_clock() {
        let directory = tempfile::tempdir().unwrap();
        let mut variant = grid_variant(directory.path());
        let book = Bbo {
            bid_px: 99_900,
            ask_px: 100_100,
            bid_sz: 0,
            ask_sz: 0,
            exchange_ms: 1_000,
            recv_ns: 0,
        };
        variant.backend.restore_checkpoint_bbo(Some(book));
        let print = mm_live::types::TradePrint {
            aggressor: mm_live::types::AggressorSide::Sell,
            px: 1,
            qty_units: 100,
            exchange_ms: 2_000,
            recv_ns: 0,
            trade_id: 1,
        };
        step_paper_variant(
            &mut variant,
            &MarketEvent::Trade(print),
            2_000,
            Some(book),
            None,
        )
        .await
        .unwrap();
        assert_eq!(variant.quote_seq, 1);
        assert!(variant
            .backend
            .on_market_event(&MarketEvent::Trade(mm_live::types::TradePrint {
                exchange_ms: 2_050,
                trade_id: 2,
                ..print
            }))
            .await
            .unwrap()
            .is_empty());
        assert!(!variant
            .backend
            .on_market_event(&MarketEvent::Trade(mm_live::types::TradePrint {
                exchange_ms: 2_100,
                trade_id: 3,
                ..print
            }))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn grid_does_not_reconcile_after_an_event_invalidates_the_account() {
        let directory = tempfile::tempdir().unwrap();
        let mut variant = grid_variant(directory.path());
        let mut account = variant.backend.account_state();
        account.cash_usdc = -100.0;
        account.inventory_units = 1_000;
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
        let book = Bbo {
            bid_px: 990,
            ask_px: 1_010,
            bid_sz: 0,
            ask_sz: 0,
            exchange_ms: 2_000,
            recv_ns: 0,
        };
        step_paper_variant(
            &mut variant,
            &MarketEvent::Bbo(book),
            2_000,
            Some(book),
            None,
        )
        .await
        .unwrap();
        assert!(!variant.backend.scientifically_valid());
        assert_eq!(variant.quote_seq, 0);
    }

    #[tokio::test]
    async fn replay_stops_at_invalidation_and_writes_the_terminal_account() {
        use mm_live::parquet_io::{MidRecord, ShardStats, TimeSource};
        let directory = tempfile::tempdir().unwrap();
        let mut variant = grid_variant(directory.path());
        let mut account = variant.backend.account_state();
        account.cash_usdc = -100.0;
        account.inventory_units = 1_000;
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
        let data = MarketDataSet {
            symbol: "CASHCAT".to_owned(),
            time_source: TimeSource::Exchange,
            mids: vec![
                MidRecord {
                    ts_ms: 2_000.0,
                    bid: 0.099,
                    ask: 0.101,
                    mid: 0.1,
                },
                MidRecord {
                    ts_ms: 3_000.0,
                    bid: 0.299,
                    ask: 0.301,
                    mid: 0.3,
                },
            ],
            trades: Vec::new(),
            books: Vec::new(),
            window_start_ms: 2_000.0,
            window_end_ms: 3_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let metrics = Arc::new(Metrics::default());
        run_event_source(
            &mut variant,
            ParquetReplaySource::new(&data, &cashcat()).unwrap(),
            &metrics,
            1,
        )
        .await
        .unwrap();
        assert!(!variant.backend.scientifically_valid());
        assert_eq!(variant.quote_seq, 0);
        assert_eq!(variant.backend.account_state().equity_usdc, 0.0);
        write_report(
            &variant.config,
            Some(&variant.report_path),
            "replay",
            1,
            cashcat(),
            None,
            Some(ModelReport::from_surface(
                &variant.surface,
                variant.inventory_unit,
            )),
            LatencySnapshot::empty("CASHCAT", 1, &variant.config.latency, false),
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
            None,
        )
        .unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&variant.report_path).unwrap()).unwrap();
        assert_eq!(report["scientifically_valid"], false);
        assert_eq!(report["account"]["equity_usdc"], 0.0);
        assert_eq!(report["metrics"]["market_messages"], 1);
        assert_eq!(report["metrics"]["bbo_updates"], 1);
        assert_eq!(report["metrics"]["inventory_units"], 1_000);
        assert_eq!(report["invalid_reasons"][0], "liquidation buffer breached");
    }

    #[test]
    fn a_rejected_resume_leaves_previous_artifacts_untouched() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("grid_state.json");
        let state = grid::PersistedGridState {
            schema_version: grid::PersistedGridState::SCHEMA_VERSION,
            symbol: "CASHCAT".to_owned(),
            grid_fingerprint: "matching".to_owned(),
            run_id: "run-1000".to_owned(),
            started_at_ms: 1_000,
            checkpoint_ms: 2_000,
            resumes: 0,
            resumed_downtime_ms: 0,
            feed_health: mm_live::config::FeedHealth::new(0, 0, 0, 1_000, false),
            trade_prints: 0,
            replayed_trades_ignored: 0,
            variants: vec![],
        };
        state.write_atomic(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let report = directory.path().join("leaderboard.json");
        std::fs::write(&report, b"previous run").unwrap();
        assert!(load_resumable_checkpoint(
            &path,
            "CASHCAT",
            "matching",
            2_100,
            3_600,
            &[("missing", "config")]
        )
        .is_none());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(std::fs::read(&report).unwrap(), b"previous run");
    }

    #[test]
    fn canary_rounding_and_minimum_notional_stay_inside_hard_cap() {
        let instrument = cashcat();
        let bid = rounded_price_units(&instrument, 0.114_999, false).unwrap();
        let ask = rounded_price_units(&instrument, 0.114_999, true).unwrap();
        assert_eq!(bid, 114_990);
        assert_eq!(ask, 115_000);
        let quantity = minimum_canary_quantity(&instrument, ask, 12.0).unwrap();
        let notional = order_notional(&instrument, ask, quantity);
        assert!(notional >= 10.5);
        assert!(notional <= 11.0);
    }

    #[test]
    fn action_response_parser_distinguishes_resting_and_filled() {
        let resting = serde_json::json!({
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 7}}]}}
        });
        ensure_top_level_action_ok(&resting).unwrap();
        assert_eq!(first_action_status(&resting).unwrap()["resting"]["oid"], 7);
        let rejected = serde_json::json!({"status": "err", "response": "bad nonce"});
        assert!(ensure_top_level_action_ok(&rejected).is_err());
    }
}

#[cfg(test)]
mod grid_health_tests {
    use super::*;

    fn board(generated_at_ms: u64, feed_down_for_ms: u64) -> grid::Leaderboard {
        grid::Leaderboard {
            generated_at_ms,
            started_at_ms: generated_at_ms.saturating_sub(3_600_000),
            elapsed_seconds: 3_600,
            symbol: "CASHCAT".to_owned(),
            feed_health: mm_live::config::FeedHealth::new(
                0,
                feed_down_for_ms,
                feed_down_for_ms,
                3_600_000,
                false,
            ),
            feed_failures: Vec::new(),
            feed_down_for_ms,
            quote_pause_reason: None,
            resumes: 0,
            resumed_downtime_ms: 0,
            rows: Vec::new(),
        }
    }

    fn write(board: &grid::Leaderboard) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("leaderboard.json");
        std::fs::write(&path, serde_json::to_string(board).unwrap()).unwrap();
        (dir, path)
    }

    #[test]
    fn fresh_leaderboard_with_a_live_feed_is_healthy() {
        let (_dir, path) = write(&board(unix_ms(), 0));
        assert!(grid_health_verdict(&path, 120, 180).is_ok());
    }

    #[test]
    fn health_reports_recovery_and_invalid_evidence_without_a_restart_loop() {
        let mut snapshot = board(unix_ms(), 70_000);
        snapshot.quote_pause_reason = Some("public feed disconnected; retrying".to_owned());
        snapshot.feed_failures = vec!["gap exceeded the scientific limit".to_owned()];
        let (_directory, path) = write(&snapshot);
        let verdict = grid_health_verdict(&path, 120, 180).unwrap();
        assert!(verdict.contains("feed down 70 s; retrying"));
        assert!(verdict.contains("evidence INVALID"));
        assert!(!verdict.contains("feed up"));
        snapshot.feed_down_for_ms = 0;
        snapshot.quote_pause_reason = None;
        snapshot.write_atomic(&path).unwrap();
        let verdict = grid_health_verdict(&path, 120, 180).unwrap();
        assert!(verdict.contains("data ready"));
        assert!(verdict.contains("evidence INVALID"));
    }

    #[test]
    fn a_stale_leaderboard_means_the_grid_is_hung_or_gone() {
        let (_dir, path) = write(&board(unix_ms().saturating_sub(600_000), 0));
        let reason = grid_health_verdict(&path, 120, 180).unwrap_err();
        assert!(reason.contains("hung or gone"), "{reason}");
    }

    #[test]
    fn a_fresh_leaderboard_is_still_unhealthy_when_the_feed_is_down() {
        let (_dir, path) = write(&board(unix_ms(), 19 * 3_600 * 1_000));
        let reason = grid_health_verdict(&path, 120, 180).unwrap_err();
        assert!(reason.contains("running blind"), "{reason}");
    }

    #[test]
    fn a_missing_leaderboard_is_unhealthy_rather_than_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let reason = grid_health_verdict(&dir.path().join("absent.json"), 120, 180).unwrap_err();
        assert!(reason.contains("cannot read"), "{reason}");
    }

    #[test]
    fn promotion_selects_best_live_equivalent_and_skips_dry_run_only_rows() {
        let directory = tempfile::tempdir().unwrap();
        let grid_path = directory.path().join("grid.toml");
        std::fs::write(
            &grid_path,
            "[[variant]]\nname = \"baseline\"\n\n[[variant]]\nname = \"wide\"\nmin_half_spread_bps = 8.0\n\n[[variant]]\nname = \"flatten\"\nflatten_after_ms = 1\nmin_half_spread_bps = 60.0\n\n[parameter_profiles.fixed]\nlambda_plus = 1.0\nlambda_minus = 1.0\nkappa_plus = 1.0\nkappa_minus = 1.0\nepsilon_plus = 0.0\nepsilon_minus = 0.0\n\n[[variant]]\nname = \"fixed\"\nparameter_profile = \"fixed\"\n",
        )
        .unwrap();
        let mut leaderboard = board(unix_ms(), 0);
        leaderboard.started_at_ms = leaderboard.generated_at_ms.saturating_sub(43_200_000);
        leaderboard.elapsed_seconds = 43_200;
        let row = |name: &str, pnl: f64| grid::LeaderboardRow {
            name: name.to_owned(),
            description: String::new(),
            net_pnl_usdc: pnl,
            promotion_pnl_usdc: Some(pnl),
            equity_usdc: 297.88 + pnl,
            realized_pnl_usdc: pnl,
            mark_to_market_pnl_usdc: pnl,
            fees_usdc: 0.0,
            funding_usdc: 0.0,
            inventory_units: 0,
            fills: 1,
            working_orders: 0,
            max_drawdown_usdc: 0.0,
            scientifically_valid: true,
            eligible_for_promotion: true,
        };
        // Deliberately leave the stale-board eligibility bit true. Promotion
        // must still reject the higher-P&L dry-run-only exit policy.
        leaderboard.rows = vec![
            row("baseline", -2.0),
            row("wide", 1.0),
            row("flatten", 10.0),
            row("fixed", 20.0),
        ];
        let leaderboard_path = directory.path().join("leaderboard.json");
        leaderboard.write_atomic(&leaderboard_path).unwrap();
        let output = directory.path().join("cashcat-active-live.toml");
        let manifest = directory.path().join("promotion.json");
        let base = AppConfig::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat_dryrun_realistic.toml"),
        )
        .unwrap();
        promote_best_config(
            &base,
            &grid_path,
            &leaderboard_path,
            &output,
            &manifest,
            43_200,
        )
        .unwrap();
        let selected: AppConfig =
            toml::from_str(&std::fs::read_to_string(&output).unwrap()).unwrap();
        assert!(selected.live.enabled);
        assert_eq!(selected.quoting.min_half_spread_bps, 8.0);
        assert_eq!(selected.dry_run.flatten_after_ms, 0);
        assert_eq!(selected.risk.max_daily_loss_usdc, 1.0);
        let promotion: serde_json::Value =
            serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
        assert_eq!(promotion["variant"], "wide");
        assert_eq!(promotion["promotion_pnl_usdc"], 1.0);
        promote_best_config(
            &base,
            &grid_path,
            &leaderboard_path,
            &output,
            &directory.path().join("promotion.json"),
            43_200,
        )
        .unwrap();
        let second: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("promotion.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(second["changed"], false);
    }

    #[test]
    fn promotion_refuses_when_every_valid_variant_is_non_profitable() {
        let directory = tempfile::tempdir().unwrap();
        let grid_path = directory.path().join("grid.toml");
        std::fs::write(&grid_path, "[[variant]]\nname = \"baseline\"\n").unwrap();
        let mut leaderboard = board(unix_ms(), 0);
        leaderboard.started_at_ms = leaderboard.generated_at_ms.saturating_sub(43_200_000);
        leaderboard.elapsed_seconds = 43_200;
        leaderboard.rows = vec![grid::LeaderboardRow {
            name: "baseline".to_owned(),
            description: String::new(),
            net_pnl_usdc: 0.0,
            promotion_pnl_usdc: Some(0.0),
            equity_usdc: 297.88,
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
        }];
        let leaderboard_path = directory.path().join("leaderboard.json");
        leaderboard.write_atomic(&leaderboard_path).unwrap();
        let base = AppConfig::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat_dryrun_realistic.toml"),
        )
        .unwrap();
        let result = promote_best_config(
            &base,
            &grid_path,
            &leaderboard_path,
            &directory.path().join("active.toml"),
            &directory.path().join("promotion.json"),
            43_200,
        );
        assert!(result.is_err());
        assert!(!directory.path().join("active.toml").exists());
    }
}
