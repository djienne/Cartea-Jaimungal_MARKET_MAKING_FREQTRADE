use super::traits::{AccountStateProvider, ExecutionBackend};
use crate::config::{AppConfig, LiveMode, QuotingConfig, RiskConfig};
use crate::hyperliquid::account_types::{
    fill_key, funding_key, parse_clearinghouse_message, parse_open_orders_message,
    parse_order_updates, parse_user_fills, parse_user_fundings, LiveAccountSnapshot,
};
use crate::hyperliquid::auth::HyperliquidCredentials;
use crate::hyperliquid::exchange::{ActionOutcome, HyperliquidExchangeClient, OpenOrder, UserFill};
use crate::hyperliquid::live_state::{
    DurableNonceManager, LiveOrderStatus, LiveStateStore, PersistedLiveOrder,
};
use crate::hyperliquid::session::{
    spawn_session, HyperliquidSessionHandle, SessionEvent, SessionSpawnArgs,
};
use crate::hyperliquid::signing::{
    is_bot_cloid, make_cloid, parse_fixed, LiveOrderRequest, TimeInForce,
};
use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::types::{
    unix_ms, AccountState, Bbo, DesiredQuotes, ExecutionEvent, Fill, MarketEvent, ProcessClock,
    Side,
};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};

#[derive(Debug, Clone, Default, Serialize)]
pub struct LiveExecutionDiagnostics {
    pub operationally_healthy: bool,
    pub scientifically_valid: bool,
    pub invalid_reason: Option<String>,
    pub connection_generation: u64,
    pub reconciliations: u64,
    pub orders_submitted: u64,
    pub orders_rejected: u64,
    pub cancels_submitted: u64,
    pub fills: u64,
    pub partial_fills: u64,
    pub maker_fills: u64,
    pub taker_fills: u64,
    pub duplicate_fills: u64,
    pub funding_events: u64,
    pub unknown_outcomes: u64,
    pub deadman_refreshes: u64,
    pub actual_maker_fee_rate: f64,
    pub actual_taker_fee_rate: f64,
    pub campaign_turnover_usdc: f64,
    pub campaign_realized_pnl_usdc: f64,
    pub maximum_working_gross_usdc: f64,
    pub maximum_directional_notional_usdc: f64,
    pub address_requests_used: u64,
    pub address_requests_cap: u64,
    pub cumulative_volume_usdc: f64,
}

pub struct LiveBootstrap {
    pub backend: HyperliquidLiveBackend,
    pub session_events: mpsc::Receiver<SessionEvent>,
    pub session_task: tokio::task::JoinHandle<()>,
}

pub struct AuthoritativeSnapshot {
    clearinghouse: crate::hyperliquid::exchange::ClearinghouseState,
    open_orders: Vec<OpenOrder>,
    active_asset: serde_json::Value,
    fees: Option<serde_json::Value>,
    fills: Vec<UserFill>,
    order_statuses: BTreeMap<String, serde_json::Value>,
}

pub struct HyperliquidLiveBackend {
    instrument: InstrumentSpec,
    live: crate::config::LiveConfig,
    quoting: QuotingConfig,
    risk: RiskConfig,
    client: Arc<HyperliquidExchangeClient>,
    session: HyperliquidSessionHandle,
    state: Arc<LiveStateStore>,
    account: LiveAccountSnapshot,
    diagnostics: LiveExecutionDiagnostics,
    latest_bbo: Option<Bbo>,
    session_started_at_ms: u64,
    pending_events: Vec<ExecutionEvent>,
    last_reconcile_ms: u64,
    deadman_armed: bool,
    last_deadman_refresh_ms: u64,
    clock: Arc<ProcessClock>,
    market_stale_ms: u64,
    q_max: i64,
    latency: Arc<LatencyMonitor>,
    last_fill_received_ns: Option<u64>,
    allow_untracked_position: bool,
    reconcile_requested: bool,
    last_inventory_update_ms: u64,
}

impl HyperliquidLiveBackend {
    pub async fn bootstrap(
        config: &AppConfig,
        instrument: InstrumentSpec,
        clock: Arc<ProcessClock>,
        latency: Arc<LatencyMonitor>,
        shutdown: watch::Receiver<bool>,
        allow_untracked_position: bool,
    ) -> Result<LiveBootstrap> {
        if !config.live.enabled {
            bail!("live.enabled=false; refusing before credentials or order transport");
        }
        if instrument.is_delisted {
            bail!("instrument is delisted");
        }
        if !instrument.only_isolated || instrument.margin_mode != "strictIsolated" {
            bail!("validated CASHCAT live profile requires strict isolated margin");
        }
        let credentials = Arc::new(HyperliquidCredentials::load(&config.live.credentials_path)?);
        if !credentials.is_vault() {
            bail!("live CASHCAT requires a dedicated subaccount/vault");
        }
        let client = Arc::new(HyperliquidExchangeClient::new(
            config.runtime.network,
            instrument.clone(),
            credentials.clone(),
            clock.clone(),
            Some(latency.clone()),
            Duration::from_millis(config.live.action_timeout_ms),
            config.live.action_expiry_ms,
            if allow_untracked_position {
                1_200
            } else {
                config.live.max_rest_weight_per_minute
            },
        )?);
        let (role, clearinghouse, open_orders, active_asset, fees, user_rate_limit) = tokio::try_join!(
            client.user_role(),
            client.clearinghouse_state(),
            client.open_orders(),
            client.active_asset_data(),
            client.user_fees(),
            client.user_rate_limit(),
        )?;
        if role.get("role").and_then(serde_json::Value::as_str) != Some("subAccount") {
            bail!("live account must report userRole=subAccount");
        }
        let account = LiveAccountSnapshot::from_rest(
            &clearinghouse,
            &active_asset,
            &fees,
            open_orders,
            &instrument,
        )?;
        if !account.isolated {
            bail!("active CASHCAT leverage is not isolated");
        }
        if account.maker_fee_rate > config.live.max_maker_fee_rate {
            bail!(
                "actual maker fee {} exceeds configured maximum {}",
                account.maker_fee_rate,
                config.live.max_maker_fee_rate
            );
        }
        let cumulative_volume_usdc = user_rate_limit
            .get("cumVlm")
            .and_then(serde_json::Value::as_str)
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.0);
        if config.live.deadman_enabled && cumulative_volume_usdc < 1_000_000.0 {
            bail!(
                "dead-man is required but this account has only {cumulative_volume_usdc:.2} USDC cumulative volume; Hyperliquid requires 1000000"
            );
        }
        let state = Arc::new(LiveStateStore::open(
            &config.live.state_path,
            &instrument.symbol,
            credentials.account(),
            &credentials.agent_address(),
            &config.fingerprint()?,
            &instrument.metadata_fingerprint,
            account.inventory_units == 0 && account.open_orders.is_empty(),
        )?);
        let (event_tx, event_rx) = mpsc::channel(config.runtime.execution_event_capacity);
        let (session, session_task) = spawn_session(SessionSpawnArgs {
            network: config.runtime.network,
            ws_url: config.runtime.network.ws_url().to_owned(),
            instrument: instrument.clone(),
            credentials,
            state: state.clone(),
            config: config.live.clone(),
            clock: clock.clone(),
            latency: latency.clone(),
            events: event_tx,
            shutdown,
        });
        let maker_fee_rate = account.maker_fee_rate;
        let taker_fee_rate = account.taker_fee_rate;
        let mut backend = Self {
            instrument,
            live: config.live.clone(),
            quoting: config.quoting.clone(),
            risk: config.risk.clone(),
            client,
            session,
            state,
            account,
            diagnostics: LiveExecutionDiagnostics {
                scientifically_valid: true,
                actual_maker_fee_rate: maker_fee_rate,
                actual_taker_fee_rate: taker_fee_rate,
                address_requests_used: user_rate_limit
                    .get("nRequestsUsed")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                address_requests_cap: user_rate_limit
                    .get("nRequestsCap")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                cumulative_volume_usdc,
                ..LiveExecutionDiagnostics::default()
            },
            latest_bbo: None,
            session_started_at_ms: unix_ms(),
            pending_events: Vec::new(),
            last_reconcile_ms: 0,
            deadman_armed: false,
            last_deadman_refresh_ms: 0,
            clock,
            market_stale_ms: config.runtime.market_stale_ms,
            q_max: config.model.q_max,
            latency,
            last_fill_received_ns: None,
            allow_untracked_position,
            reconcile_requested: false,
            last_inventory_update_ms: unix_ms(),
        };
        backend.initialize_campaign()?;
        Ok(LiveBootstrap {
            backend,
            session_events: event_rx,
            session_task,
        })
    }

    pub const fn diagnostics(&self) -> &LiveExecutionDiagnostics {
        &self.diagnostics
    }

    pub const fn operationally_healthy(&self) -> bool {
        self.diagnostics.operationally_healthy
    }

    pub const fn deadman_armed(&self) -> bool {
        self.deadman_armed
    }

    pub fn daily_realized_pnl_usdc(&self) -> Result<f64> {
        let mut state = self.state.load_required()?;
        let day = unix_ms() / 86_400_000;
        if state.pnl_day != day {
            self.state.update(|state| {
                state.pnl_day = day;
                state.daily_realized_pnl_usdc = 0.0;
                Ok(())
            })?;
            state.daily_realized_pnl_usdc = 0.0;
        }
        Ok(state.daily_realized_pnl_usdc)
    }

    pub fn effective_quoting_config(&self) -> Result<QuotingConfig> {
        let effective_equity = self
            .quoting
            .available_capital_usdc
            .min(self.account.account_value_usdc);
        let usable = (effective_equity - self.risk.min_liquidation_buffer_usdc).max(0.0);
        if usable <= 0.0 {
            bail!("account equity does not exceed the liquidation reserve");
        }
        let mut quoting = self.quoting.clone();
        quoting.available_capital_usdc = if self.live.mode == LiveMode::AcceptanceTest {
            let minimum_sized_capital = self.instrument.minimum_notional
                * 1.02
                * self.quoting.leverage.recip()
                * self.quoting.target_capital_utilisation.recip()
                * self.q_max as f64;
            usable.min(minimum_sized_capital)
        } else {
            usable
        };
        quoting.maker_fee_rate = self.account.maker_fee_rate;
        Ok(quoting)
    }

    pub fn persisted_inventory_unit(&self) -> Result<Option<i64>> {
        Ok(self.state.load_required()?.inventory_unit)
    }

    pub fn durable_state(&self) -> Result<crate::hyperliquid::live_state::PersistedLiveState> {
        self.state.load_required()
    }

    pub fn inject_drop_next_action_response(&self) -> Result<()> {
        if self.live.mode != LiveMode::AcceptanceTest {
            bail!("action-response fault injection is acceptance-test only");
        }
        self.session.inject_drop_next_response()
    }

    pub async fn current_bbo(&mut self) -> Result<Bbo> {
        self.fresh_bbo().await
    }

    pub fn minimum_order_quantity(&self, px_units: i64) -> Result<i64> {
        if px_units <= 0 {
            bail!("minimum-order price must be positive");
        }
        let target = self.instrument.minimum_notional * 1.01;
        let raw =
            target * self.instrument.price_scale() as f64 * self.instrument.size_scale() as f64
                / px_units as f64;
        let quantity = raw.ceil() as i64;
        let notional =
            self.instrument.price_from_units(px_units) * self.instrument.size_from_units(quantity);
        if quantity <= 0
            || (self.live.mode == LiveMode::AcceptanceTest
                && notional > self.live.acceptance_max_order_notional_usdc)
        {
            bail!("venue-minimum order exceeds acceptance cap");
        }
        Ok(quantity)
    }

    pub fn order_notional(&self, px_units: i64, qty_units: i64) -> f64 {
        self.instrument.price_from_units(px_units) * self.instrument.size_from_units(qty_units)
    }

    pub fn passive_price(&self, side: Side, reference_px: i64, offset_bps: f64) -> Result<i64> {
        if reference_px <= 0 || !offset_bps.is_finite() || offset_bps < 0.0 {
            bail!("invalid passive-price inputs");
        }
        let multiplier = match side {
            Side::Buy => 1.0 - offset_bps / 10_000.0,
            Side::Sell => 1.0 + offset_bps / 10_000.0,
        };
        let preliminary = (reference_px as f64 * multiplier).round() as i64;
        let quantum = self.instrument.price_quantum(preliminary);
        Ok(match side {
            Side::Buy => preliminary / quantum * quantum,
            Side::Sell => preliminary.saturating_add(quantum - 1) / quantum * quantum,
        })
    }

    pub fn improve_inside_spread(&self, side: Side, bbo: Bbo) -> i64 {
        match side {
            Side::Buy => {
                let improved = bbo
                    .bid_px
                    .saturating_add(self.instrument.price_quantum(bbo.bid_px));
                if improved < bbo.ask_px {
                    improved
                } else {
                    bbo.bid_px
                }
            }
            Side::Sell => {
                let improved = bbo
                    .ask_px
                    .saturating_sub(self.instrument.price_quantum(bbo.ask_px));
                if improved > bbo.bid_px {
                    improved
                } else {
                    bbo.ask_px
                }
            }
        }
    }

    pub async fn submit_acceptance_alo(
        &mut self,
        quote_seq: u64,
        orders: &[(Side, i64, i64)],
    ) -> Result<ActionOutcome> {
        if self.live.mode != LiveMode::AcceptanceTest {
            bail!("acceptance ALO submission is acceptance-test only");
        }
        let mut requests = Vec::with_capacity(orders.len());
        for (side, px_units, qty_units) in orders {
            let sequence = self.state.next_cloid_sequence()?;
            requests.push(LiveOrderRequest {
                side: *side,
                px_units: *px_units,
                qty_units: *qty_units,
                reduce_only: false,
                time_in_force: TimeInForce::Alo,
                cloid: make_cloid(self.session_started_at_ms, quote_seq, *side, sequence),
            });
        }
        let bbo = self.fresh_bbo().await?;
        self.validate_new_orders(&requests, bbo)?;
        let outcome = self.session.place_orders(quote_seq, requests).await?;
        self.record_order_outcome(&outcome);
        Ok(outcome)
    }

    pub async fn cancel_all_bot_orders(&mut self) -> Result<()> {
        let state = self.state.load_required()?;
        let cloids: Vec<String> = state
            .orders
            .values()
            .filter(|order| !order.status.terminal())
            .map(|order| order.cloid.clone())
            .collect();
        if !cloids.is_empty() {
            let outcome = self.cancel_cloids_resilient(cloids).await?;
            self.diagnostics.cancels_submitted += 1;
            require_action_known(&outcome)?;
        }
        self.reconcile_authoritative().await?;
        if !self.account.open_orders.is_empty() {
            bail!("bot-order cancellation did not produce an empty venue order set");
        }
        Ok(())
    }

    pub async fn enqueue_cancel_all_bot_orders(&self) -> Result<()> {
        let cloids: Vec<String> = self
            .state
            .load_required()?
            .orders
            .values()
            .filter(|order| !order.status.terminal())
            .map(|order| order.cloid.clone())
            .collect();
        if cloids.is_empty() {
            return Ok(());
        }
        self.session.enqueue_cancel_cloids(cloids).await
    }

    pub async fn cancel_bot_oid(&mut self, oid: u64) -> Result<ActionOutcome> {
        let outcome = self.session.cancel_oids(vec![oid]).await?;
        self.diagnostics.cancels_submitted += 1;
        Ok(outcome)
    }

    pub async fn cancel_bot_cloids(&mut self, cloids: Vec<String>) -> Result<ActionOutcome> {
        let outcome = self.session.cancel_cloids(cloids).await?;
        self.diagnostics.cancels_submitted += 1;
        Ok(outcome)
    }

    pub async fn schedule_deadman_at(&mut self, time: Option<u64>) -> Result<ActionOutcome> {
        let outcome = self.session.schedule_cancel(time).await?;
        require_action_ok(&outcome)?;
        self.deadman_armed = time.is_some();
        self.last_deadman_refresh_ms = unix_ms();
        Ok(outcome)
    }

    pub fn acceptance_budget_ok(&self) -> Result<()> {
        if self.live.mode != LiveMode::AcceptanceTest {
            return Ok(());
        }
        let campaign = self.state.load_required()?.campaign;
        if campaign.turnover_usdc > self.live.acceptance_max_turnover_usdc
            || campaign.realized_pnl_usdc < -self.live.acceptance_max_realized_loss_usdc
        {
            bail!("acceptance campaign hard budget exceeded");
        }
        Ok(())
    }

    pub fn record_deadman_trigger(&mut self) -> Result<()> {
        if self.live.mode != LiveMode::AcceptanceTest {
            bail!("dead-man trigger accounting is acceptance-test only");
        }
        self.state.update(|state| {
            state.campaign.deadman_triggers = state.campaign.deadman_triggers.saturating_add(1);
            if state.campaign.deadman_triggers > 1 {
                bail!("acceptance campaign permits exactly one dead-man trigger");
            }
            Ok(())
        })
    }

    pub fn persist_inventory_unit(&self, inventory_unit: i64) -> Result<()> {
        if inventory_unit <= 0 {
            bail!("live inventory unit must be positive");
        }
        self.state.update(|state| {
            if state
                .inventory_unit
                .is_some_and(|existing| existing != inventory_unit)
                && self.account.inventory_units != 0
            {
                bail!("cannot change live inventory unit while non-flat");
            }
            state.inventory_unit = Some(inventory_unit);
            Ok(())
        })
    }

    pub fn process_session_event(&mut self, event: SessionEvent) -> Result<Vec<ExecutionEvent>> {
        match event {
            SessionEvent::Connected { generation, .. } => {
                self.diagnostics.connection_generation = generation;
                if generation > 1 {
                    self.diagnostics.scientifically_valid = false;
                    self.diagnostics.invalid_reason = Some("live WebSocket reconnected".to_owned());
                }
            }
            SessionEvent::Ready { generation, .. } => {
                self.diagnostics.connection_generation = generation;
                self.diagnostics.operationally_healthy = false;
                self.reconcile_requested = true;
            }
            SessionEvent::Disconnected { reason, .. } => {
                self.diagnostics.operationally_healthy = false;
                self.diagnostics.scientifically_valid = false;
                self.diagnostics.invalid_reason = Some(reason);
            }
            SessionEvent::AccountData {
                channel,
                data,
                received_ns,
                ..
            } => match channel.as_str() {
                "orderUpdates" => self.apply_order_updates(data)?,
                "userFills" => self.apply_fills(data, received_ns)?,
                "userFundings" => self.apply_fundings(data)?,
                "clearinghouseState" => {
                    let clearinghouse = parse_clearinghouse_message(data)?;
                    self.account
                        .apply_clearinghouse(&clearinghouse, &self.instrument)?;
                    self.pending_events.push(ExecutionEvent::AccountReconciled {
                        inventory_units: self.account.inventory_units,
                        equity_usdc: self.account.account_value_usdc,
                    });
                }
                "openOrders" => {
                    let orders = parse_open_orders_message(data)?;
                    for order in &orders {
                        if order
                            .cloid
                            .as_deref()
                            .is_none_or(|cloid| !is_bot_cloid(cloid))
                        {
                            bail!("foreign order appeared on dedicated account stream");
                        }
                    }
                    self.account.open_orders = orders;
                }
                "activeAssetData" => self
                    .account
                    .apply_active_asset_data(&data, &self.instrument)?,
                _ => {}
            },
            SessionEvent::ActionCompleted {
                purpose, outcome, ..
            } => {
                if matches!(outcome, ActionOutcome::Unknown { .. }) {
                    self.diagnostics.unknown_outcomes += 1;
                    self.diagnostics.operationally_healthy = false;
                    self.reconcile_requested = true;
                } else if let Some(body) = outcome.body() {
                    if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
                        self.diagnostics.operationally_healthy = false;
                        self.diagnostics.invalid_reason =
                            Some(format!("{purpose:?} action failed: {body}"));
                        if purpose == crate::hyperliquid::session::ActionPurpose::Deadman {
                            self.deadman_armed = false;
                        }
                    }
                }
            }
            SessionEvent::ActionRefused { reason, .. } => {
                self.diagnostics.operationally_healthy = false;
                self.diagnostics.invalid_reason = Some(reason);
            }
        }
        Ok(std::mem::take(&mut self.pending_events))
    }

    pub async fn maintenance(&mut self, now_ms: u64) -> Result<Vec<ExecutionEvent>> {
        if now_ms.saturating_sub(self.last_reconcile_ms) >= self.live.reconcile_interval_ms {
            self.reconcile_requested = true;
        }
        if self.deadman_armed
            && self.diagnostics.operationally_healthy
            && now_ms.saturating_sub(self.last_deadman_refresh_ms) >= self.live.deadman_refresh_ms
        {
            let deadline = now_ms.saturating_add(self.live.deadman_deadline_ms);
            self.session.enqueue_schedule_cancel(Some(deadline)).await?;
            self.last_deadman_refresh_ms = now_ms;
            self.diagnostics.deadman_refreshes += 1;
        }
        Ok(std::mem::take(&mut self.pending_events))
    }

    pub const fn reconciliation_requested(&self) -> bool {
        self.reconcile_requested
    }

    pub fn spawn_reconciliation(&mut self, sender: mpsc::Sender<Result<AuthoritativeSnapshot>>) {
        self.reconcile_requested = false;
        let client = self.client.clone();
        let state = self.state.clone();
        tokio::spawn(async move {
            let result = fetch_authoritative_snapshot(client, state).await;
            let _ = sender.send(result).await;
        });
    }

    pub fn apply_reconciliation(&mut self, snapshot: AuthoritativeSnapshot) -> Result<()> {
        self.apply_authoritative_snapshot(snapshot)
    }

    pub async fn ensure_configured_leverage(&mut self) -> Result<()> {
        let expected = self.quoting.leverage.round() as u32;
        if self.quoting.leverage != f64::from(expected)
            || expected == 0
            || f64::from(expected) > self.instrument.max_leverage
        {
            bail!("live leverage must be a supported positive integer");
        }
        if self.account.leverage == expected && self.account.isolated {
            return Ok(());
        }
        if self.account.inventory_units != 0 || !self.account.open_orders.is_empty() {
            bail!("cannot change leverage with position or open orders");
        }
        let outcome = self.session.update_leverage(expected, false).await?;
        require_action_ok(&outcome)?;
        for _ in 0..10 {
            tokio::time::sleep(Duration::from_millis(250)).await;
            self.reconcile_authoritative().await?;
            if self.account.leverage == expected && self.account.isolated {
                return Ok(());
            }
        }
        bail!("venue did not confirm isolated leverage {expected}")
    }

    pub async fn arm_or_refresh_deadman(&mut self, now_ms: u64) -> Result<()> {
        let deadline = now_ms.saturating_add(self.live.deadman_deadline_ms);
        let outcome = self.session.schedule_cancel(Some(deadline)).await?;
        require_action_ok(&outcome)?;
        self.deadman_armed = true;
        self.last_deadman_refresh_ms = now_ms;
        self.diagnostics.deadman_refreshes += 1;
        Ok(())
    }

    pub async fn clear_deadman(&mut self) -> Result<()> {
        if !self.deadman_armed {
            return Ok(());
        }
        let outcome = if self.session.healthy() {
            self.session.schedule_cancel(None).await?
        } else {
            let mut nonce = DurableNonceManager::new(self.state.clone())?;
            self.client
                .schedule_cancel_with_nonce(None, nonce.next_nonce()?)
                .await?
        };
        require_action_ok(&outcome)?;
        self.deadman_armed = false;
        Ok(())
    }

    pub async fn market_open(
        &mut self,
        side: Side,
        qty_units: i64,
        max_slippage_bps: f64,
    ) -> Result<ActionOutcome> {
        if qty_units <= 0 {
            bail!("market-open quantity must be positive");
        }
        let bbo = self.fresh_bbo().await?;
        let request = self.ioc_request(side, qty_units, false, max_slippage_bps, bbo)?;
        self.validate_new_orders(std::slice::from_ref(&request), bbo)?;
        let outcome = self.session.place_orders(0, vec![request]).await?;
        self.record_order_outcome(&outcome);
        Ok(outcome)
    }

    pub async fn market_close(&mut self) -> Result<()> {
        for (attempt, slippage) in [
            25.0_f64,
            100.0,
            self.live.emergency_flatten_max_slippage_bps,
        ]
        .into_iter()
        .enumerate()
        {
            if attempt != 0 || self.account.inventory_units == 0 {
                self.reconcile_authoritative().await?;
            }
            let inventory = self.account.inventory_units;
            if inventory == 0 {
                return Ok(());
            }
            let (side, quantity) = closing_side_and_quantity(inventory)
                .context("nonzero inventory has no closing intent")?;
            let bbo = self.fresh_bbo().await?;
            let request = self.ioc_request(side, quantity, true, slippage, bbo)?;
            let started_ns = self.clock.now_ns();
            if let Some(fill_ns) = self.last_fill_received_ns {
                self.latency.record(
                    LatencyKind::FillToCloseSend,
                    started_ns.saturating_sub(fill_ns),
                    started_ns,
                );
            }
            let outcome = self.session.place_orders(0, vec![request]).await?;
            self.record_order_outcome(&outcome);
            require_action_known(&outcome)?;
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(250)).await;
                self.reconcile_authoritative().await?;
                if self.account.inventory_units == 0 {
                    let done_ns = self.clock.now_ns();
                    self.latency.record(
                        LatencyKind::CloseSendToFill,
                        done_ns.saturating_sub(started_ns),
                        done_ns,
                    );
                    if let Some(fill_ns) = self.last_fill_received_ns {
                        self.latency.record(
                            LatencyKind::FillToFlat,
                            done_ns.saturating_sub(fill_ns),
                            done_ns,
                        );
                    }
                    return Ok(());
                }
            }
        }
        bail!(
            "reduce-only market close left residual inventory {}",
            self.account.inventory_units
        )
    }

    pub async fn reconcile_authoritative(&mut self) -> Result<()> {
        let snapshot =
            fetch_authoritative_snapshot(self.client.clone(), self.state.clone()).await?;
        self.apply_authoritative_snapshot(snapshot)
    }

    fn apply_authoritative_snapshot(&mut self, snapshot: AuthoritativeSnapshot) -> Result<()> {
        let AuthoritativeSnapshot {
            clearinghouse,
            open_orders,
            active_asset,
            fees,
            fills,
            order_statuses,
        } = snapshot;
        for order in &open_orders {
            let Some(cloid) = order.cloid.as_deref() else {
                bail!("foreign open order without CLOID on dedicated live account");
            };
            if !is_bot_cloid(cloid) {
                bail!("foreign open order {cloid} on dedicated live account");
            }
        }
        let persisted = self.state.load_required()?;
        if persisted.metadata_fingerprint != self.instrument.metadata_fingerprint
            && (self.account.inventory_units != 0
                || persisted
                    .orders
                    .values()
                    .any(|order| !order.status.terminal()))
        {
            bail!("live metadata changed while durable exposure exists");
        }
        let persisted_accounting = self.state.load_required()?;
        let effective_fees = fees.unwrap_or_else(|| {
            serde_json::json!({
                "userAddRate": self.account.maker_fee_rate.to_string(),
                "userCrossRate": self.account.taker_fee_rate.to_string(),
            })
        });
        self.account = LiveAccountSnapshot::from_rest(
            &clearinghouse,
            &active_asset,
            &effective_fees,
            open_orders.clone(),
            &self.instrument,
        )?;
        self.last_inventory_update_ms = unix_ms();
        self.account.realized_pnl_usdc = persisted_accounting.cumulative_realized_pnl_usdc;
        self.account.fees_usdc = persisted_accounting.cumulative_fees_usdc;
        self.account.funding_usdc = persisted_accounting.cumulative_funding_usdc;
        self.state.update(|state| {
            if state.metadata_fingerprint != self.instrument.metadata_fingerprint {
                state
                    .metadata_fingerprint
                    .clone_from(&self.instrument.metadata_fingerprint);
            }
            for open in &open_orders {
                let Some(cloid) = open.cloid.as_deref() else {
                    continue;
                };
                if state.orders.contains_key(cloid) {
                    continue;
                }
                let side = match open.side.as_str() {
                    "B" | "Buy" => Side::Buy,
                    "A" | "Sell" => Side::Sell,
                    value => bail!("unknown open-order side {value:?}"),
                };
                let qty_units = parse_fixed(&open.sz, self.instrument.sz_decimals)?;
                state.orders.insert(
                    cloid.to_owned(),
                    PersistedLiveOrder {
                        cloid: cloid.to_owned(),
                        quote_seq: 0,
                        side,
                        px_units: self.instrument.price_to_units(open.limit_px.parse()?)?,
                        original_qty_units: qty_units,
                        remaining_qty_units: qty_units,
                        reduce_only: false,
                        status: LiveOrderStatus::Resting,
                        nonce: None,
                        transport_id: None,
                        oid: Some(open.oid),
                        prepared_at_ms: open.timestamp,
                        last_update_ms: unix_ms(),
                        last_error: Some("recovered bot-prefixed venue order".to_owned()),
                    },
                );
            }
            Ok(())
        })?;
        if self.account.maker_fee_rate > self.live.max_maker_fee_rate {
            bail!("actual maker fee exceeds configured live maximum");
        }
        self.apply_fill_rows(&fills, false)?;
        self.reconcile_orders(&open_orders, &fills, &order_statuses)?;
        if self.account.inventory_units != 0
            && self.state.load_required()?.inventory_unit.is_none()
            && !self.allow_untracked_position
        {
            bail!("non-flat live account has no persisted inventory-unit identity");
        }
        self.last_reconcile_ms = unix_ms();
        self.diagnostics.reconciliations += 1;
        self.diagnostics.actual_maker_fee_rate = self.account.maker_fee_rate;
        self.diagnostics.actual_taker_fee_rate = self.account.taker_fee_rate;
        self.pending_events.push(ExecutionEvent::AccountReconciled {
            inventory_units: self.account.inventory_units,
            equity_usdc: self.account.account_value_usdc,
        });
        let unresolved = self
            .state
            .load_required()?
            .orders
            .values()
            .any(|order| order.status == LiveOrderStatus::UnknownOutcome);
        if self.session.healthy() && !unresolved {
            self.diagnostics.operationally_healthy = true;
        }
        Ok(())
    }

    fn reconcile_orders(
        &self,
        open_orders: &[OpenOrder],
        fills: &[UserFill],
        order_statuses: &BTreeMap<String, serde_json::Value>,
    ) -> Result<()> {
        let open_by_cloid: BTreeMap<&str, &OpenOrder> = open_orders
            .iter()
            .filter_map(|order| order.cloid.as_deref().map(|cloid| (cloid, order)))
            .collect();
        let filled_cloids: BTreeSet<&str> = fills
            .iter()
            .filter_map(|fill| fill.cloid.as_deref())
            .collect();
        let candidates: Vec<(String, u64)> = self
            .state
            .load_required()?
            .orders
            .values()
            .filter(|order| !order.status.terminal())
            .map(|order| (order.cloid.clone(), order.prepared_at_ms))
            .collect();
        for (cloid, prepared_at_ms) in candidates {
            if let Some(open) = open_by_cloid.get(cloid.as_str()) {
                let remaining = parse_fixed(&open.sz, self.instrument.sz_decimals)?;
                self.state.update(|state| {
                    if let Some(order) = state.orders.get_mut(&cloid) {
                        order.status = if remaining < order.original_qty_units {
                            LiveOrderStatus::PartiallyFilled
                        } else {
                            LiveOrderStatus::Resting
                        };
                        order.remaining_qty_units = remaining;
                        order.oid = Some(open.oid);
                        order.last_update_ms = unix_ms();
                    }
                    Ok(())
                })?;
            } else if filled_cloids.contains(cloid.as_str()) {
                self.transition_terminal(&cloid, LiveOrderStatus::Filled, None)?;
            } else {
                let status = order_statuses.get(&cloid).and_then(|status| {
                    status
                        .pointer("/order/status")
                        .and_then(serde_json::Value::as_str)
                        .or_else(|| status.get("status").and_then(serde_json::Value::as_str))
                });
                let terminal = if status == Some("unknownOid") {
                    let safe_after = prepared_at_ms
                        .saturating_add(self.live.action_expiry_ms)
                        .saturating_add(2_000);
                    if unix_ms() >= safe_after {
                        LiveOrderStatus::Rejected
                    } else {
                        LiveOrderStatus::UnknownOutcome
                    }
                } else {
                    status.map_or(LiveOrderStatus::UnknownOutcome, map_order_status)
                };
                self.transition_terminal(&cloid, terminal, None)?;
            }
        }
        Ok(())
    }

    fn initialize_campaign(&mut self) -> Result<()> {
        if self.live.mode != LiveMode::AcceptanceTest {
            return Ok(());
        }
        self.state.update(|state| {
            if state.campaign.started_at_ms == 0 {
                state.campaign.started_at_ms = unix_ms();
                state.campaign.starting_equity_usdc = self.account.account_value_usdc;
            }
            if state.cumulative_realized_pnl_usdc == 0.0 && state.campaign.realized_pnl_usdc != 0.0
            {
                state.cumulative_realized_pnl_usdc = state.campaign.realized_pnl_usdc;
                state.daily_realized_pnl_usdc = state.campaign.realized_pnl_usdc;
            }
            self.diagnostics.campaign_turnover_usdc = state.campaign.turnover_usdc;
            self.diagnostics.campaign_realized_pnl_usdc = state.campaign.realized_pnl_usdc;
            Ok(())
        })
    }

    fn apply_order_updates(&mut self, data: serde_json::Value) -> Result<()> {
        for update in parse_order_updates(data)? {
            if update.order.coin != self.instrument.symbol {
                continue;
            }
            let Some(cloid) = update.order.cloid.as_deref() else {
                continue;
            };
            if !is_bot_cloid(cloid) {
                self.diagnostics.operationally_healthy = false;
                bail!("foreign order update on dedicated account");
            }
            let status = map_order_status(&update.status);
            let remaining = parse_fixed(&update.order.sz, self.instrument.sz_decimals)?;
            self.state.update(|state| {
                if let Some(order) = state.orders.get_mut(cloid) {
                    order.status = status;
                    order.remaining_qty_units = remaining;
                    order.oid = Some(update.order.oid);
                    order.last_update_ms = update.status_timestamp;
                }
                Ok(())
            })?;
            match status {
                LiveOrderStatus::Resting => {
                    self.pending_events.push(ExecutionEvent::OrderAcknowledged {
                        cloid: cloid.to_owned(),
                        oid: Some(update.order.oid),
                    });
                }
                LiveOrderStatus::Canceled => {
                    self.pending_events.push(ExecutionEvent::OrderCanceled {
                        cloid: cloid.to_owned(),
                        oid: Some(update.order.oid),
                    });
                }
                LiveOrderStatus::Rejected => {
                    self.pending_events.push(ExecutionEvent::OrderRejected {
                        cloid: cloid.to_owned(),
                        reason: update.status,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn apply_fills(&mut self, data: serde_json::Value, received_ns: u64) -> Result<()> {
        let message = parse_user_fills(data)?;
        if !message.fills.is_empty() && !message.is_snapshot {
            self.last_fill_received_ns = Some(received_ns);
        }
        self.apply_fill_rows(&message.fills, true)
    }

    fn apply_fill_rows(&mut self, fills: &[UserFill], adjust_inventory: bool) -> Result<()> {
        for fill in fills {
            if fill.coin != self.instrument.symbol {
                continue;
            }
            let key = fill_key(fill);
            let fresh = self.state.update(|state| {
                if !state.processed_fill_keys.insert(key.clone()) {
                    return Ok(false);
                }
                if fill.time < state.event_checkpoint_ms {
                    return Ok(false);
                }
                let px: f64 = fill.px.parse()?;
                let size: f64 = fill.sz.parse()?;
                let fee: f64 = fill.fee.parse()?;
                let closed_pnl: f64 = fill.closed_pnl.parse()?;
                let day = fill.time / 86_400_000;
                if state.pnl_day != day {
                    state.pnl_day = day;
                    state.daily_realized_pnl_usdc = 0.0;
                }
                state.cumulative_realized_pnl_usdc += closed_pnl - fee;
                state.cumulative_fees_usdc += fee;
                state.daily_realized_pnl_usdc += closed_pnl - fee;
                state.campaign.turnover_usdc += px * size;
                state.campaign.realized_pnl_usdc += closed_pnl - fee;
                Ok(true)
            })?;
            if !fresh {
                self.diagnostics.duplicate_fills += 1;
                continue;
            }
            let qty_units = parse_fixed(&fill.sz, self.instrument.sz_decimals)?;
            let px_units = self.instrument.price_to_units(fill.px.parse()?)?;
            let fee: f64 = fill.fee.parse()?;
            let side = if fill.side == "B" {
                Side::Buy
            } else {
                Side::Sell
            };
            if adjust_inventory {
                let start_units = parse_fixed(&fill.start_position, self.instrument.sz_decimals)?;
                if fill.time >= self.last_inventory_update_ms {
                    self.account.inventory_units =
                        inventory_after_fill(start_units, side, qty_units);
                    self.last_inventory_update_ms = fill.time;
                }
            }
            let accounting = self.state.load_required()?;
            self.account.fees_usdc = accounting.cumulative_fees_usdc;
            self.account.realized_pnl_usdc = accounting.cumulative_realized_pnl_usdc;
            self.diagnostics.fills += 1;
            if fill.crossed {
                self.diagnostics.taker_fills += 1;
            } else {
                self.diagnostics.maker_fills += 1;
            }
            if let Some(cloid) = &fill.cloid {
                self.state.update(|state| {
                    if let Some(order) = state.orders.get_mut(cloid) {
                        order.remaining_qty_units =
                            order.remaining_qty_units.saturating_sub(qty_units);
                        order.status = if order.remaining_qty_units == 0 {
                            LiveOrderStatus::Filled
                        } else {
                            LiveOrderStatus::PartiallyFilled
                        };
                        order.oid = Some(fill.oid);
                        order.last_update_ms = fill.time;
                    }
                    Ok(())
                })?;
            }
            self.pending_events.push(ExecutionEvent::Fill(Fill {
                side,
                px: px_units,
                qty_units,
                fee_usdc: fee,
                exchange_ms: fill.time,
                virtual_order_id: fill.oid,
                maker: !fill.crossed,
            }));
        }
        let campaign = self.state.load_required()?.campaign;
        self.diagnostics.campaign_turnover_usdc = campaign.turnover_usdc;
        self.diagnostics.campaign_realized_pnl_usdc = campaign.realized_pnl_usdc;
        Ok(())
    }

    fn apply_fundings(&mut self, data: serde_json::Value) -> Result<()> {
        for funding in parse_user_fundings(data)?.fundings {
            if funding.coin != self.instrument.symbol {
                continue;
            }
            let key = funding_key(&funding);
            let fresh = self.state.update(|state| {
                if !state.processed_funding_keys.insert(key) {
                    return Ok(false);
                }
                let usdc: f64 = funding.usdc.parse()?;
                state.cumulative_funding_usdc += usdc;
                Ok(true)
            })?;
            if !fresh {
                continue;
            }
            let usdc: f64 = funding.usdc.parse()?;
            self.account.funding_usdc = self.state.load_required()?.cumulative_funding_usdc;
            self.diagnostics.funding_events += 1;
            self.pending_events.push(ExecutionEvent::Funding {
                coin: funding.coin,
                time_ms: funding.time,
                usdc,
            });
        }
        Ok(())
    }

    fn transition_terminal(
        &self,
        cloid: &str,
        status: LiveOrderStatus,
        error: Option<String>,
    ) -> Result<()> {
        self.state.update(|state| {
            if let Some(order) = state.orders.get_mut(cloid) {
                order.status = status;
                if status.terminal() {
                    order.remaining_qty_units = 0;
                }
                order.last_error = error;
                order.last_update_ms = unix_ms();
            }
            Ok(())
        })
    }

    async fn fresh_bbo(&mut self) -> Result<Bbo> {
        if let Some(bbo) = self.latest_bbo {
            if bbo.recv_ns != 0
                && self.clock.now_ns().saturating_sub(bbo.recv_ns)
                    <= self.market_stale_ms.saturating_mul(1_000_000)
            {
                return Ok(bbo);
            }
        }
        let book = self.client.l2_book().await?;
        let levels = book
            .get("levels")
            .and_then(serde_json::Value::as_array)
            .context("l2Book missing levels")?;
        let price = |side: usize| -> Result<(i64, i64)> {
            let level = levels
                .get(side)
                .and_then(|levels| levels.get(0))
                .context("l2Book side is empty")?;
            Ok((
                self.instrument.price_to_units(
                    level
                        .get("px")
                        .and_then(serde_json::Value::as_str)
                        .context("book px")?
                        .parse()?,
                )?,
                parse_fixed(
                    level
                        .get("sz")
                        .and_then(serde_json::Value::as_str)
                        .context("book sz")?,
                    self.instrument.sz_decimals,
                )?,
            ))
        };
        let (bid_px, bid_sz) = price(0)?;
        let (ask_px, ask_sz) = price(1)?;
        let bbo = Bbo {
            bid_px,
            bid_sz,
            ask_px,
            ask_sz,
            exchange_ms: book
                .get("time")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(unix_ms),
            recv_ns: self.clock.now_ns(),
        };
        self.latest_bbo = Some(bbo);
        Ok(bbo)
    }

    fn ioc_request(
        &self,
        side: Side,
        qty_units: i64,
        reduce_only: bool,
        slippage_bps: f64,
        bbo: Bbo,
    ) -> Result<LiveOrderRequest> {
        if !slippage_bps.is_finite()
            || slippage_bps < 0.0
            || slippage_bps > self.live.emergency_flatten_max_slippage_bps
        {
            bail!("IOC slippage exceeds configured limit");
        }
        let touch = match side {
            Side::Buy => bbo.ask_px,
            Side::Sell => bbo.bid_px,
        };
        let multiplier = match side {
            Side::Buy => 1.0 + slippage_bps / 10_000.0,
            Side::Sell => 1.0 - slippage_bps / 10_000.0,
        };
        let preliminary = (touch as f64 * multiplier).round() as i64;
        let quantum = self.instrument.price_quantum(preliminary);
        let px_units = match side {
            Side::Buy => preliminary.saturating_add(quantum - 1) / quantum * quantum,
            Side::Sell => preliminary / quantum * quantum,
        };
        let sequence = self.state.next_cloid_sequence()?;
        Ok(LiveOrderRequest {
            side,
            px_units,
            qty_units,
            reduce_only,
            time_in_force: TimeInForce::Ioc,
            cloid: make_cloid(self.session_started_at_ms, 0, side, sequence),
        })
    }

    fn validate_new_orders(&mut self, requests: &[LiveOrderRequest], bbo: Bbo) -> Result<()> {
        if self.diagnostics.address_requests_cap != 0
            && self
                .diagnostics
                .address_requests_cap
                .saturating_sub(self.diagnostics.address_requests_used)
                < 100
        {
            bail!("venue address-action reserve is below 100 requests");
        }
        let mid = self.instrument.price_from_units(bbo.mid_units());
        let persisted = self.state.load_required()?;
        let working: Vec<&PersistedLiveOrder> = persisted
            .orders
            .values()
            .filter(|order| !order.status.terminal())
            .collect();
        let existing_gross: f64 = working
            .iter()
            .map(|order| {
                self.instrument.price_from_units(order.px_units)
                    * self.instrument.size_from_units(order.remaining_qty_units)
            })
            .sum();
        let new_gross: f64 = requests
            .iter()
            .map(|order| {
                self.instrument.price_from_units(order.px_units)
                    * self.instrument.size_from_units(order.qty_units)
            })
            .sum();
        for request in requests {
            let notional = self.instrument.price_from_units(request.px_units)
                * self.instrument.size_from_units(request.qty_units);
            if notional < self.instrument.minimum_notional {
                bail!("live order is below venue minimum notional");
            }
            if self.live.mode == LiveMode::AcceptanceTest
                && notional > self.live.acceptance_max_order_notional_usdc
            {
                bail!("acceptance order exceeds 12-USDC hard cap");
            }
            let side_index = usize::from(request.side == Side::Sell);
            if !request.reduce_only
                && (request.qty_units > self.account.max_trade_units[side_index]
                    || notional > self.account.available_to_trade_usdc[side_index])
            {
                bail!("live order exceeds venue-reported available-to-trade limits");
            }
        }
        let mut buy_units = self.account.inventory_units;
        let mut sell_units = self.account.inventory_units;
        for order in working {
            match order.side {
                Side::Buy => buy_units = buy_units.saturating_add(order.remaining_qty_units),
                Side::Sell => sell_units = sell_units.saturating_sub(order.remaining_qty_units),
            }
        }
        for request in requests {
            match request.side {
                Side::Buy => buy_units = buy_units.saturating_add(request.qty_units),
                Side::Sell => sell_units = sell_units.saturating_sub(request.qty_units),
            }
        }
        let directional = self
            .instrument
            .size_from_units(buy_units.abs().max(sell_units.abs()))
            * mid;
        if self.live.mode == LiveMode::AcceptanceTest {
            if existing_gross + new_gross > self.live.acceptance_max_working_gross_usdc {
                bail!("acceptance working gross exceeds 24-USDC hard cap");
            }
            if directional > self.live.acceptance_max_directional_notional_usdc {
                bail!("acceptance directional exposure exceeds 12-USDC hard cap");
            }
            let campaign = persisted.campaign;
            if campaign.turnover_usdc >= self.live.acceptance_max_turnover_usdc - 12.0
                || campaign.realized_pnl_usdc
                    <= -(self.live.acceptance_max_realized_loss_usdc - 0.1)
            {
                bail!("acceptance campaign cleanup reserve reached");
            }
        } else {
            let usable_equity = self
                .quoting
                .available_capital_usdc
                .min(self.account.account_value_usdc)
                - self.risk.min_liquidation_buffer_usdc;
            let account_cap = usable_equity.max(0.0)
                * self.quoting.leverage
                * self.quoting.target_capital_utilisation;
            let configured_cap = if self.risk.max_notional_usdc > 0.0 {
                self.risk.max_notional_usdc
            } else {
                f64::INFINITY
            };
            if directional > account_cap.min(configured_cap) {
                bail!("prospective live directional exposure exceeds account-aware cap");
            }
        }
        self.diagnostics.maximum_working_gross_usdc = self
            .diagnostics
            .maximum_working_gross_usdc
            .max(existing_gross + new_gross);
        self.diagnostics.maximum_directional_notional_usdc = self
            .diagnostics
            .maximum_directional_notional_usdc
            .max(directional);
        Ok(())
    }

    fn active_orders_by_side(&self) -> Result<BTreeMap<Side, Vec<PersistedLiveOrder>>> {
        let mut by_side = BTreeMap::<Side, Vec<PersistedLiveOrder>>::new();
        for order in self.state.load_required()?.orders.into_values() {
            if !order.status.terminal() {
                by_side.entry(order.side).or_default().push(order);
            }
        }
        Ok(by_side)
    }

    async fn cancel_cloids_resilient(&mut self, cloids: Vec<String>) -> Result<ActionOutcome> {
        if self.session.healthy() {
            return self.session.cancel_cloids(cloids).await;
        }
        let mut nonce = DurableNonceManager::new(self.state.clone())?;
        self.client
            .cancel_by_cloid_with_nonce(&cloids, nonce.next_nonce()?)
            .await
    }

    fn record_order_outcome(&mut self, outcome: &ActionOutcome) {
        self.diagnostics.orders_submitted += 1;
        match outcome {
            ActionOutcome::Unknown { .. } => self.diagnostics.unknown_outcomes += 1,
            ActionOutcome::Response { body, .. } => {
                self.diagnostics.orders_rejected += body
                    .pointer("/response/data/statuses")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, |statuses| {
                        statuses
                            .iter()
                            .filter(|status| status.get("error").is_some())
                            .count() as u64
                    });
            }
        }
    }
}

#[async_trait]
impl ExecutionBackend for HyperliquidLiveBackend {
    async fn reconcile(&mut self, desired: DesiredQuotes, _now_ms: u64) -> Result<()> {
        let existing = self.active_orders_by_side()?;
        let mut cancel = Vec::new();
        let mut place = Vec::new();
        for side in [Side::Buy, Side::Sell] {
            let target = match side {
                Side::Buy => desired.bid,
                Side::Sell => desired.ask,
            };
            let side_orders = existing.get(&side).cloned().unwrap_or_default();
            let unchanged = target.is_some_and(|target| {
                side_orders.iter().any(|order| {
                    order.px_units == target.px
                        && order.remaining_qty_units == target.qty_units
                        && order.reduce_only == target.reduce_only
                        && matches!(
                            order.status,
                            LiveOrderStatus::Sent
                                | LiveOrderStatus::Resting
                                | LiveOrderStatus::PartiallyFilled
                        )
                })
            });
            for order in side_orders {
                if !unchanged {
                    cancel.push(order.cloid);
                }
            }
            if let Some(target) = target.filter(|_| !unchanged) {
                if !target.post_only {
                    bail!("normal live quotes must be post-only");
                }
                let sequence = self.state.next_cloid_sequence()?;
                place.push(LiveOrderRequest {
                    side,
                    px_units: target.px,
                    qty_units: target.qty_units,
                    reduce_only: target.reduce_only,
                    time_in_force: TimeInForce::Alo,
                    cloid: make_cloid(
                        self.session_started_at_ms,
                        desired.quote_seq,
                        side,
                        sequence,
                    ),
                });
            }
        }
        if !cancel.is_empty() {
            if self.session.healthy() {
                self.session.enqueue_cancel_cloids(cancel).await?;
            } else {
                let outcome = self.cancel_cloids_resilient(cancel).await?;
                require_action_known(&outcome)?;
            }
            self.diagnostics.cancels_submitted += 1;
        }
        if !place.is_empty() {
            let bbo = self
                .latest_bbo
                .context("cannot place live quote without BBO")?;
            self.validate_new_orders(&place, bbo)?;
            self.session
                .enqueue_orders(desired.quote_seq, place)
                .await?;
            self.diagnostics.orders_submitted += 1;
        }
        Ok(())
    }

    async fn on_market_event(&mut self, event: &MarketEvent) -> Result<Vec<ExecutionEvent>> {
        if let MarketEvent::Bbo(bbo) = event {
            self.latest_bbo = Some(*bbo);
        }
        Ok(std::mem::take(&mut self.pending_events))
    }

    async fn shutdown(&mut self, _now_ms: u64) -> Result<()> {
        let cloids: Vec<String> = self
            .state
            .load_required()?
            .orders
            .into_values()
            .filter(|order| !order.status.terminal())
            .map(|order| order.cloid)
            .collect();
        if !cloids.is_empty() {
            let outcome = self.cancel_cloids_resilient(cloids).await?;
            require_action_known(&outcome)?;
        }
        self.reconcile_authoritative().await?;
        if !self.account.open_orders.is_empty() {
            bail!("graceful live shutdown could not confirm empty open orders");
        }
        self.clear_deadman().await?;
        self.diagnostics.operationally_healthy = false;
        Ok(())
    }

    fn invalidate(&mut self, reason: &str) {
        self.diagnostics.operationally_healthy = false;
        self.diagnostics.scientifically_valid = false;
        self.diagnostics.invalid_reason = Some(reason.to_owned());
    }

    fn scientifically_valid(&self) -> bool {
        self.diagnostics.scientifically_valid
    }
}

impl AccountStateProvider for HyperliquidLiveBackend {
    fn account_state(&self) -> AccountState {
        let mark = self
            .latest_bbo
            .map_or(0.0, |bbo| self.instrument.price_from_units(bbo.mid_units()));
        self.account
            .account_state(mark, self.instrument.size_scale())
    }
}

async fn fetch_authoritative_snapshot(
    client: Arc<HyperliquidExchangeClient>,
    state: Arc<LiveStateStore>,
) -> Result<AuthoritativeSnapshot> {
    let needs_historical_orders = state
        .load_required()?
        .orders
        .values()
        .any(|order| !order.status.terminal());
    let (clearinghouse, open_orders, active_asset, fills) = tokio::try_join!(
        client.clearinghouse_state(),
        client.open_orders(),
        client.active_asset_data(),
        client.recent_fills(),
    )?;
    let historical_orders = if needs_historical_orders {
        client.historical_orders().await?
    } else {
        serde_json::Value::Array(Vec::new())
    };
    let open_cloids: BTreeSet<&str> = open_orders
        .iter()
        .filter_map(|order| order.cloid.as_deref())
        .collect();
    let fill_cloids: BTreeSet<&str> = fills
        .iter()
        .filter_map(|fill| fill.cloid.as_deref())
        .collect();
    let mut order_statuses = BTreeMap::new();
    if let Some(rows) = historical_orders.as_array() {
        for row in rows {
            let Some(cloid) = row
                .pointer("/order/cloid")
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            if let Some(status) = row.get("status").and_then(serde_json::Value::as_str) {
                order_statuses.insert(
                    cloid.to_owned(),
                    serde_json::json!({"order": {"status": status}}),
                );
            }
        }
    }
    let unresolved: Vec<String> = state
        .load_required()?
        .orders
        .values()
        .filter(|order| {
            !order.status.terminal()
                && !open_cloids.contains(order.cloid.as_str())
                && !fill_cloids.contains(order.cloid.as_str())
                && !order_statuses.contains_key(&order.cloid)
        })
        .map(|order| order.cloid.clone())
        .collect();
    if unresolved.len() > 16 {
        bail!(
            "{} durable orders remain unresolved after historicalOrders; refusing REST fan-out",
            unresolved.len()
        );
    }
    for cloid in unresolved {
        order_statuses.insert(
            cloid.clone(),
            client
                .order_status(serde_json::json!(cloid.clone()))
                .await?,
        );
    }
    Ok(AuthoritativeSnapshot {
        clearinghouse,
        open_orders,
        active_asset,
        fees: None,
        fills,
        order_statuses,
    })
}

fn map_order_status(status: &str) -> LiveOrderStatus {
    match status {
        "open" => LiveOrderStatus::Resting,
        "filled" => LiveOrderStatus::Filled,
        "canceled" | "cancelled" => LiveOrderStatus::Canceled,
        value if value.ends_with("Canceled") || value.ends_with("Cancelled") => {
            LiveOrderStatus::Canceled
        }
        value if value.ends_with("Rejected") => LiveOrderStatus::Rejected,
        _ => LiveOrderStatus::UnknownOutcome,
    }
}

fn closing_side_and_quantity(inventory_units: i64) -> Option<(Side, i64)> {
    match inventory_units.cmp(&0) {
        std::cmp::Ordering::Greater => Some((Side::Sell, inventory_units)),
        std::cmp::Ordering::Less => Some((Side::Buy, inventory_units.saturating_abs())),
        std::cmp::Ordering::Equal => None,
    }
}

fn inventory_after_fill(start_units: i64, side: Side, fill_units: i64) -> i64 {
    start_units.saturating_add(side.inventory_sign().saturating_mul(fill_units))
}

fn require_action_known(outcome: &ActionOutcome) -> Result<()> {
    if let ActionOutcome::Unknown { nonce, error } = outcome {
        bail!("action nonce {nonce} has unknown outcome: {error}");
    }
    Ok(())
}

fn require_action_ok(outcome: &ActionOutcome) -> Result<()> {
    let body = outcome.require_known()?;
    if body.get("status").and_then(serde_json::Value::as_str) == Some("ok")
        || body.get("type").and_then(serde_json::Value::as_str) == Some("default")
    {
        return Ok(());
    }
    bail!("Hyperliquid action was not successful: {body}")
}

pub fn live_state_path(config: &AppConfig) -> &Path {
    &config.live.state_path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_order_statuses_are_classified_without_collapsing_rejections() {
        assert_eq!(map_order_status("filled"), LiveOrderStatus::Filled);
        assert_eq!(
            map_order_status("marginCanceled"),
            LiveOrderStatus::Canceled
        );
        assert_eq!(
            map_order_status("badAloPxRejected"),
            LiveOrderStatus::Rejected
        );
        assert_eq!(map_order_status("mystery"), LiveOrderStatus::UnknownOutcome);
    }

    #[test]
    fn aggressive_ioc_rounding_is_marketability_preserving() {
        let instrument = InstrumentSpec {
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
            metadata_fingerprint: "meta".to_owned(),
        };
        let bbo = Bbo {
            bid_px: 114_500,
            bid_sz: 100,
            ask_px: 114_600,
            ask_sz: 100,
            exchange_ms: 1,
            recv_ns: 1,
        };
        let buy_preliminary = (bbo.ask_px as f64 * 1.025).round() as i64;
        let buy_quantum = instrument.price_quantum(buy_preliminary);
        let buy = buy_preliminary.saturating_add(buy_quantum - 1) / buy_quantum * buy_quantum;
        let sell_preliminary = (bbo.bid_px as f64 * 0.975).round() as i64;
        let sell_quantum = instrument.price_quantum(sell_preliminary);
        let sell = sell_preliminary / sell_quantum * sell_quantum;
        assert!(buy >= bbo.ask_px);
        assert!(sell <= bbo.bid_px);
    }

    #[test]
    fn reduce_only_close_direction_never_extends_or_flips_inventory() {
        assert_eq!(closing_side_and_quantity(88), Some((Side::Sell, 88)));
        assert_eq!(closing_side_and_quantity(-91), Some((Side::Buy, 91)));
        assert_eq!(closing_side_and_quantity(0), None);
        assert_eq!(
            closing_side_and_quantity(i64::MIN),
            Some((Side::Buy, i64::MAX))
        );
    }

    #[test]
    fn exact_fill_start_position_handles_partial_and_opposite_fills() {
        assert_eq!(inventory_after_fill(0, Side::Buy, 40), 40);
        assert_eq!(inventory_after_fill(40, Side::Buy, 20), 60);
        assert_eq!(inventory_after_fill(60, Side::Sell, 15), 45);
        assert_eq!(inventory_after_fill(-30, Side::Buy, 30), 0);
    }
}
