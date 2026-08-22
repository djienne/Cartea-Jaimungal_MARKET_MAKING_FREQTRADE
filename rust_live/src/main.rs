use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mm_live::calibration::{CalibrationSnapshot, Calibrator};
use mm_live::config::AppConfig;
use mm_live::execution::{AccountStateProvider, DryRunBackend, ExecutionBackend, MarketDataSource};
use mm_live::hjb::{solve_asymmetric, HjbSurface};
use mm_live::hot_path::{spawn_hot_path, AtomicRiskState, HotPathInputs, ModelBundle};
use mm_live::hyperliquid::market::{run_market_stream, MarketStreamArgs};
use mm_live::hyperliquid::meta::discover_instrument;
use mm_live::hyperliquid::{
    account::{run_account_stream, AccountStreamArgs, AccountStreamEvent, AccountStreamMetrics},
    auth::HyperliquidCredentials,
    exchange::HyperliquidExchangeClient,
    signing::{make_cloid, parse_fixed, LiveOrderRequest, TimeInForce},
};
use mm_live::latency::{LatencyKind, LatencyMonitor, LatencyObserver, LatencySnapshot};
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
use std::collections::BTreeMap;
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
    /// Exercise credential parsing, account REST reads, and the account WebSocket without actions.
    ConnectorCheck {
        #[arg(long, default_value = "rust_live/hyperliquid.env")]
        credentials: PathBuf,
        #[arg(long, default_value_t = 65)]
        duration_seconds: u64,
    },
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
    /// Reserved interface; real order submission is deliberately unavailable.
    Live,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CanaryPhase {
    Passive,
    RoundTrip,
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
        Command::ConnectorCheck {
            credentials,
            duration_seconds,
        } => run_connector_check(&config, instrument, &credentials, duration_seconds).await,
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
        Command::Live => unreachable!("live command returned before metadata discovery"),
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
    }));
    let (
        clearinghouse,
        open_orders,
        recent_fills,
        user_role,
        user_fees,
        active_asset_data,
        all_mids,
    ) = tokio::try_join!(
        client.clearinghouse_state(),
        client.open_orders(),
        client.recent_fills(),
        client.user_role(),
        client.user_fees(),
        client.active_asset_data(),
        client.all_mids(),
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

fn minimum_canary_quantity(
    instrument: &mm_live::InstrumentSpec,
    px_units: i64,
    max_notional_usdc: f64,
) -> Result<i64> {
    let target = instrument.minimum_notional * 1.01;
    let raw =
        target * instrument.price_scale() as f64 * instrument.size_scale() as f64 / px_units as f64;
    let qty_units = raw.ceil() as i64;
    let notional = order_notional(instrument, px_units, qty_units);
    if qty_units <= 0 || notional > max_notional_usdc {
        bail!("minimum canary order would be {notional:.6} USDC, above cap {max_notional_usdc:.6}");
    }
    Ok(qty_units)
}

fn order_notional(instrument: &mm_live::InstrumentSpec, px_units: i64, qty_units: i64) -> f64 {
    instrument.price_from_units(px_units) * instrument.size_from_units(qty_units)
}

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

fn ensure_top_level_action_ok(body: &serde_json::Value) -> Result<()> {
    if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        bail!("Hyperliquid action failed: {body}");
    }
    Ok(())
}

fn first_action_status(body: &serde_json::Value) -> Result<&serde_json::Value> {
    body.pointer("/response/data/statuses/0")
        .context("Hyperliquid action response has no first status")
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
    let latency_snapshot =
        LatencySnapshot::empty(&instrument.symbol, started_at_ms, &config.latency, false);
    write_report(
        config,
        report_path,
        "replay",
        started_at_ms,
        instrument,
        Some(snapshot),
        Some(ModelReport::from_surface(&surface, inventory_unit)),
        latency_snapshot,
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
        latest_bbo: latest_bbo.clone(),
        events: events.clone(),
        signal: signal.clone(),
        clock: clock.clone(),
        metrics: metrics.clone(),
        latency: Some(latency.clone()),
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
            let dispatch_ns = clock.now_ns();
            latency.record(
                LatencyKind::MarketEventDispatch,
                dispatch_ns.saturating_sub(event_recv_ns(&event)),
                dispatch_ns,
            );
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
            backend.reconcile(next, unix_ms()).await?;
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
        result = calibration_rx.recv() => {
            calibration_inflight = false;
            let Some(result) = result else { continue };
            match result {
                Ok((next, mut next_surface, mut next_inventory_unit)) => {
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
        _ = calibration_interval.tick(), if !calibration_inflight => {
            calibration_inflight = true;
            let job_config = config.clone();
            let job_instrument = instrument.clone();
            let result_tx = calibration_tx.clone();
            let job_latency = latency.clone();
            let job_clock = clock.clone();
            tokio::spawn(async move {
                let started_ns = job_clock.now_ns();
                let result = tokio::task::spawn_blocking(move || {
                    calibrate_model(&job_config, &job_instrument, true)
                        .map(|(_, snapshot, surface, inventory_unit)| {
                            (snapshot, surface, inventory_unit)
                        })
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
    latency_observer.stop()?;
    let latency_snapshot = (*latency.snapshot()).clone();
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
        latency_snapshot,
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
    latency: LatencySnapshot,
    backend: &DryRunBackend,
    metrics: &Arc<Metrics>,
    invalid_reasons: Vec<String>,
    event_log_path: &Path,
    market_event_ring_high_water: usize,
) -> Result<()> {
    let finished_at_ms = unix_ms();
    let has_calibration = calibration.is_some();
    let report = SessionReport {
        schema_version: 2,
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

#[cfg(test)]
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
        }
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
        assert!(notional >= 10.0);
        assert!(notional <= 12.0);
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
