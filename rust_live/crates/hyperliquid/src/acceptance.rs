use crate::config::{AppConfig, LiveMode};
use crate::exchange::ActionOutcome;
use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyMonitor, LatencyObserver};
use crate::live_state::LiveOrderStatus;
use crate::session::SessionEvent;
use crate::types::{unix_ms, ExecutionEvent, ProcessClock, Side};
use crate::HyperliquidLiveBackend;
use anyhow::{bail, Context, Result};
use mm_execution::{AccountStateProvider, ExecutionBackend};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptancePhase {
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

pub async fn run(
    config: &AppConfig,
    instrument: InstrumentSpec,
    phase: AcceptancePhase,
) -> Result<()> {
    if config.live.mode != LiveMode::AcceptanceTest {
        bail!("live-acceptance requires live.mode=acceptance_test");
    }
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
        false,
    )
    .await?;
    let mut backend = bootstrap.backend;
    let mut session_events = bootstrap.session_events;
    let session_task = bootstrap.session_task;
    wait_ready(&mut backend, &mut session_events).await?;
    backend.acceptance_budget_ok()?;

    let phase_result: Result<serde_json::Value> = {
        let phase_future = async {
            Ok(match phase {
                AcceptancePhase::Verify => verify(&mut backend).await?,
                AcceptancePhase::Leverage => {
                    backend.ensure_configured_leverage().await?;
                    serde_json::json!({"isolated_leverage_verified": 2})
                }
                AcceptancePhase::TwoSided => two_sided(&mut backend).await?,
                AcceptancePhase::CrossingAlo => crossing_alo(&mut backend).await?,
                AcceptancePhase::UnknownOutcome => {
                    unknown_outcome(&mut backend, &mut session_events).await?
                }
                AcceptancePhase::RestartOrderPrepare => {
                    let evidence = restart_order_prepare(&mut backend).await?;
                    println!("{}", serde_json::to_string_pretty(&evidence)?);
                    std::process::exit(99);
                }
                AcceptancePhase::RestartOrderRecover => {
                    let before = active_orders(&backend)?;
                    if before.len() != 1 {
                        bail!("restart-order recovery expected exactly one durable working order");
                    }
                    backend.cancel_all_bot_orders().await?;
                    serde_json::json!({
                        "recovered_cloid": before[0].cloid,
                        "recovered_oid": before[0].oid,
                    })
                }
                AcceptancePhase::Deadman => deadman(&mut backend, &mut session_events).await?,
                AcceptancePhase::MakerFill => maker_fill(&mut backend, &mut session_events).await?,
                AcceptancePhase::RestartPositionPrepare => {
                    let evidence = restart_position_prepare(&mut backend).await?;
                    println!("{}", serde_json::to_string_pretty(&evidence)?);
                    std::process::exit(99);
                }
                AcceptancePhase::RestartPositionRecover => {
                    let before = backend.account_state().inventory_units;
                    if before >= 0 {
                        bail!("restart-position recovery expected a short CASHCAT position");
                    }
                    backend.market_close().await?;
                    serde_json::json!({
                        "recovered_short_units": before,
                        "final_position_units": backend.account_state().inventory_units,
                    })
                }
                AcceptancePhase::Final => final_reconciliation(&mut backend).await?,
            })
        };
        tokio::pin!(phase_future);
        let result = tokio::select! {
            result = &mut phase_future => result,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                Err(anyhow::anyhow!("acceptance phase interrupted by operator"))
            }
        };
        result
    };

    let mut emergency_cleanup_result = Ok(());
    if phase_result.is_err() {
        emergency_cleanup_result = backend.cancel_all_bot_orders().await;
        if backend.account_state().inventory_units != 0 {
            emergency_cleanup_result = emergency_cleanup_result.and(backend.market_close().await);
        }
    }
    let budget_result = backend.acceptance_budget_ok();
    let shutdown_result = backend.shutdown(unix_ms()).await;
    let final_account = backend.account_state();
    let diagnostics = backend.diagnostics().clone();
    let durable = backend.durable_state()?;
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(Duration::from_secs(5), session_task)
        .await
        .context("acceptance session did not stop")?
        .context("acceptance session task panicked")?;
    observer.stop()?;
    emergency_cleanup_result?;
    shutdown_result?;
    budget_result?;
    let evidence = phase_result?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "mode": "live_acceptance",
            "phase": format!("{phase:?}").to_lowercase(),
            "evidence": evidence,
            "account": final_account,
            "diagnostics": diagnostics,
            "campaign": durable.campaign,
            "latency": &*latency.snapshot(),
        }))?
    );
    Ok(())
}

async fn wait_ready(
    backend: &mut HyperliquidLiveBackend,
    events: &mut mpsc::Receiver<SessionEvent>,
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let event = events
                .recv()
                .await
                .context("acceptance session stopped before readiness")?;
            let ready = matches!(event, SessionEvent::Ready { .. });
            backend.process_session_event(event)?;
            if ready || backend.reconciliation_requested() {
                backend.reconcile_authoritative().await?;
            }
            if ready && backend.operationally_healthy() {
                return Ok::<(), anyhow::Error>(());
            }
        }
    })
    .await
    .context("acceptance session readiness timed out")??;
    Ok(())
}

async fn verify(backend: &mut HyperliquidLiveBackend) -> Result<serde_json::Value> {
    backend.reconcile_authoritative().await?;
    let account = backend.account_state();
    let active = active_orders(backend)?;
    if account.inventory_units != 0 || !active.is_empty() {
        bail!("acceptance verify requires a flat account with no working orders");
    }
    Ok(serde_json::json!({
        "flat": true,
        "equity_usdc": account.equity_usdc,
        "active_orders": active.len(),
    }))
}

async fn two_sided(backend: &mut HyperliquidLiveBackend) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    let bbo = backend.current_bbo().await?;
    let bid_px = backend.passive_price(Side::Buy, bbo.bid_px, 100.0)?;
    let ask_px = backend.passive_price(Side::Sell, bbo.ask_px, 100.0)?;
    let bid_qty = backend.minimum_order_quantity(bid_px)?;
    let ask_qty = backend.minimum_order_quantity(ask_px)?;
    let outcome = backend
        .submit_acceptance_alo(
            1,
            &[(Side::Buy, bid_px, bid_qty), (Side::Sell, ask_px, ask_qty)],
        )
        .await?;
    let statuses = action_statuses(&outcome)?;
    if statuses.len() != 2
        || statuses
            .iter()
            .any(|status| status.get("resting").is_none())
    {
        backend.cancel_all_bot_orders().await?;
        bail!("both two-sided ALO legs must rest");
    }
    let mut active = active_orders(backend)?;
    active.sort_by_key(|order| order.side);
    if active.len() != 2 {
        bail!("two-sided action did not persist two active orders");
    }
    backend
        .cancel_bot_cloids(vec![active[0].cloid.clone()])
        .await?
        .require_known()?;
    let oid = active[1].oid.context("second two-sided order has no OID")?;
    backend.cancel_bot_oid(oid).await?.require_known()?;
    backend.reconcile_authoritative().await?;
    if backend.account_state().inventory_units != 0 {
        backend.market_close().await?;
        bail!("far two-sided acceptance ALO unexpectedly filled");
    }
    Ok(serde_json::json!({
        "bid_notional_usdc": backend.order_notional(bid_px, bid_qty),
        "ask_notional_usdc": backend.order_notional(ask_px, ask_qty),
        "cancel_by_cloid": active[0].cloid,
        "cancel_by_oid": oid,
    }))
}

async fn crossing_alo(backend: &mut HyperliquidLiveBackend) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    let bbo = backend.current_bbo().await?;
    let qty = backend.minimum_order_quantity(bbo.ask_px)?;
    let outcome = backend
        .submit_acceptance_alo(2, &[(Side::Buy, bbo.ask_px, qty)])
        .await?;
    let statuses = action_statuses(&outcome)?;
    let error = statuses
        .first()
        .and_then(|status| status.get("error"))
        .and_then(serde_json::Value::as_str)
        .context("crossing ALO was not rejected")?;
    backend.reconcile_authoritative().await?;
    if backend.account_state().inventory_units != 0 {
        backend.market_close().await?;
        bail!("crossing ALO produced an unexpected fill");
    }
    Ok(serde_json::json!({"rejection": error}))
}

async fn unknown_outcome(
    backend: &mut HyperliquidLiveBackend,
    events: &mut mpsc::Receiver<SessionEvent>,
) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    let bbo = backend.current_bbo().await?;
    let px = backend.passive_price(Side::Buy, bbo.bid_px, 150.0)?;
    let qty = backend.minimum_order_quantity(px)?;
    backend.inject_drop_next_action_response()?;
    let outcome = backend
        .submit_acceptance_alo(3, &[(Side::Buy, px, qty)])
        .await?;
    if !matches!(outcome, ActionOutcome::Unknown { .. }) {
        bail!("fault-injected action did not become UnknownOutcome");
    }
    wait_ready(backend, events).await?;
    backend.reconcile_authoritative().await?;
    let unresolved = backend
        .durable_state()?
        .orders
        .values()
        .filter(|order| order.status == LiveOrderStatus::UnknownOutcome)
        .count();
    backend.cancel_all_bot_orders().await?;
    if unresolved != 0 {
        bail!("unknown action was not resolved by CLOID reconciliation");
    }
    Ok(serde_json::json!({"unknown_outcome_resolved": true}))
}

async fn restart_order_prepare(backend: &mut HyperliquidLiveBackend) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    let bbo = backend.current_bbo().await?;
    let px = backend.passive_price(Side::Buy, bbo.bid_px, 200.0)?;
    let qty = backend.minimum_order_quantity(px)?;
    let outcome = backend
        .submit_acceptance_alo(4, &[(Side::Buy, px, qty)])
        .await?;
    let statuses = action_statuses(&outcome)?;
    if statuses
        .first()
        .and_then(|status| status.get("resting"))
        .is_none()
    {
        bail!("restart-order prepare did not rest");
    }
    let active = active_orders(backend)?;
    if active.len() != 1 {
        bail!("restart-order prepare did not persist one order");
    }
    Ok(serde_json::json!({
        "expected_process_exit": 99,
        "cloid": active[0].cloid,
        "oid": active[0].oid,
    }))
}

async fn deadman(
    backend: &mut HyperliquidLiveBackend,
    events: &mut mpsc::Receiver<SessionEvent>,
) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    let deadline = unix_ms().saturating_add(8_000);
    if let Err(error) = backend.schedule_deadman_at(Some(deadline)).await {
        return Ok(serde_json::json!({
            "available": false,
            "rejection": error.to_string(),
            "triggered": false,
        }));
    }
    let bbo = backend.current_bbo().await?;
    let px = backend.passive_price(Side::Buy, bbo.bid_px, 200.0)?;
    let qty = backend.minimum_order_quantity(px)?;
    let outcome = backend
        .submit_acceptance_alo(5, &[(Side::Buy, px, qty)])
        .await?;
    if action_statuses(&outcome)?
        .first()
        .and_then(|status| status.get("resting"))
        .is_none()
    {
        bail!("dead-man test order did not rest");
    }
    pump_events(backend, events, Duration::from_secs(12)).await?;
    backend.reconcile_authoritative().await?;
    if !active_orders(backend)?.is_empty() {
        bail!("dead-man trigger did not cancel the working order");
    }
    backend.record_deadman_trigger()?;
    backend.schedule_deadman_at(None).await?;
    Ok(serde_json::json!({"deadline_ms": deadline, "triggered": true}))
}

async fn maker_fill(
    backend: &mut HyperliquidLiveBackend,
    events: &mut mpsc::Receiver<SessionEvent>,
) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    if backend.account_state().inventory_units != 0 {
        backend.market_close().await?;
    }
    let initial_maker_fills = backend.diagnostics().maker_fills;
    let initial_taker_fills = backend.diagnostics().taker_fills;
    let mut quote_seq = 10_u64;
    loop {
        backend.acceptance_budget_ok()?;
        let bbo = backend.current_bbo().await?;
        let px = backend.improve_inside_spread(Side::Buy, bbo);
        let qty = backend.minimum_order_quantity(px)?;
        backend.persist_inventory_unit(qty)?;
        let outcome = backend
            .submit_acceptance_alo(quote_seq, &[(Side::Buy, px, qty)])
            .await?;
        outcome.require_known()?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            tokio::select! {
                () = tokio::time::sleep_until(deadline) => break,
                event = events.recv() => {
                    let event = event.context("maker-fill account stream stopped")?;
                    let execution_events = backend.process_session_event(event)?;
                    let maker_fill_seen = execution_events.iter().any(|event| {
                        matches!(event, ExecutionEvent::Fill(fill) if fill.maker)
                    });
                    let taker_fill_seen = execution_events.iter().any(|event| {
                        matches!(event, ExecutionEvent::Fill(fill) if !fill.maker)
                    });
                    if taker_fill_seen {
                        backend.enqueue_cancel_all_bot_orders()?;
                        backend.market_close().await?;
                        bail!("ALO acceptance order produced a taker fill");
                    }
                    if maker_fill_seen {
                        let filled_units = backend.account_state().inventory_units;
                        backend.enqueue_cancel_all_bot_orders()?;
                        backend.market_close().await?;
                        if backend.account_state().inventory_units != 0 {
                            bail!("maker-fill close did not finish flat");
                        }
                        return Ok(serde_json::json!({
                            "maker_fill_units": filled_units,
                            "maker_fill_count_delta": backend.diagnostics().maker_fills - initial_maker_fills,
                            "unexpected_taker_fill_delta": backend.diagnostics().taker_fills - initial_taker_fills,
                            "fast_reduce_only_close": true,
                        }));
                    }
                }
            }
        }
        cancel_active_fast(backend).await?;
        pump_events(backend, events, Duration::from_millis(250)).await?;
        if backend.account_state().inventory_units != 0 {
            if backend.diagnostics().maker_fills <= initial_maker_fills {
                backend.market_close().await?;
                bail!("inventory changed without an authoritative maker fill");
            }
            let filled_units = backend.account_state().inventory_units;
            backend.market_close().await?;
            return Ok(serde_json::json!({
                "maker_fill_units": filled_units,
                "maker_fill_detected_during_cancel_reconciliation": true,
            }));
        }
        quote_seq = quote_seq.saturating_add(1);
        if quote_seq.is_multiple_of(10) {
            backend.reconcile_authoritative().await?;
        }
    }
}

async fn restart_position_prepare(
    backend: &mut HyperliquidLiveBackend,
) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    if backend.account_state().inventory_units != 0 {
        backend.market_close().await?;
    }
    let bbo = backend.current_bbo().await?;
    let qty = backend.minimum_order_quantity(bbo.bid_px)?;
    backend.persist_inventory_unit(qty)?;
    let outcome = backend.market_open(Side::Sell, qty, 50.0).await?;
    outcome.require_known()?;
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        backend.reconcile_authoritative().await?;
        if backend.account_state().inventory_units < 0 {
            return Ok(serde_json::json!({
                "expected_process_exit": 99,
                "short_position_units": backend.account_state().inventory_units,
            }));
        }
    }
    bail!("restart-position IOC did not create a short position")
}

async fn final_reconciliation(backend: &mut HyperliquidLiveBackend) -> Result<serde_json::Value> {
    backend.cancel_all_bot_orders().await?;
    if backend.account_state().inventory_units != 0 {
        backend.market_close().await?;
    }
    if backend.deadman_armed() {
        backend.clear_deadman().await?;
    }
    backend.reconcile_authoritative().await?;
    let account = backend.account_state();
    let durable = backend.durable_state()?;
    let unresolved = durable
        .orders
        .values()
        .filter(|order| !order.status.terminal())
        .count();
    if account.inventory_units != 0 || unresolved != 0 {
        bail!("final acceptance reconciliation is not flat and terminal");
    }
    Ok(serde_json::json!({
        "flat": true,
        "unresolved_orders": unresolved,
        "equity_usdc": account.equity_usdc,
        "turnover_usdc": durable.campaign.turnover_usdc,
        "realized_pnl_usdc": durable.campaign.realized_pnl_usdc,
    }))
}

async fn pump_events(
    backend: &mut HyperliquidLiveBackend,
    events: &mut mpsc::Receiver<SessionEvent>,
    duration: Duration,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + duration;
    loop {
        tokio::select! {
            () = tokio::time::sleep_until(deadline) => return Ok(()),
            event = events.recv() => {
                backend.process_session_event(event.context("acceptance session stopped")?)?;
            }
        }
    }
}

async fn cancel_active_fast(backend: &mut HyperliquidLiveBackend) -> Result<()> {
    let cloids: Vec<String> = active_orders(backend)?
        .into_iter()
        .map(|order| order.cloid)
        .collect();
    if cloids.is_empty() {
        return Ok(());
    }
    backend.cancel_bot_cloids(cloids).await?.require_known()?;
    Ok(())
}

fn action_statuses(outcome: &ActionOutcome) -> Result<&Vec<serde_json::Value>> {
    let body = outcome.require_known()?;
    if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        bail!("Hyperliquid acceptance action failed: {body}");
    }
    body.pointer("/response/data/statuses")
        .and_then(serde_json::Value::as_array)
        .context("action response has no statuses")
}

fn active_orders(backend: &HyperliquidLiveBackend) -> Result<Vec<OpenOrderRecord>> {
    Ok(backend
        .durable_state()?
        .orders
        .into_values()
        .filter(|order| !order.status.terminal())
        .map(|order| OpenOrderRecord {
            cloid: order.cloid,
            oid: order.oid,
            side: order.side,
        })
        .collect())
}

struct OpenOrderRecord {
    cloid: String,
    oid: Option<u64>,
    side: Side,
}
