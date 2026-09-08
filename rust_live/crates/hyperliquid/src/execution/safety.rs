//! Owned safety I/O, shared by timed exits and explicit cleanup.
//! Only handles and an account snapshot move to the task; the live backend
//! remains the sole owner of fill accounting and quote admission.
use super::{
    closing_side_and_quantity, ensure_no_foreign_positions, make_cloid, parse_fixed,
    require_action_known, unix_ms, ActionOutcome, Bbo, BboReader, HyperliquidExchangeClient,
    HyperliquidLiveBackend, HyperliquidSessionHandle, InstrumentSpec, LatencyKind, LatencyMonitor,
    LiveAccountSnapshot, LiveOrderRequest, LiveStateStore, ProcessClock, Side, TimeInForce,
};
use crate::hyperliquid::exchange::ClearinghouseState;
use anyhow::{bail, Context, Result};
use std::{sync::Arc, time::Duration};

pub(super) struct SafetyExit {
    instrument: InstrumentSpec,
    max_slippage_bps: f64,
    client: Arc<HyperliquidExchangeClient>,
    session: HyperliquidSessionHandle,
    state: Arc<LiveStateStore>,
    account: LiveAccountSnapshot,
    clock: Arc<ProcessClock>,
    latency: Arc<LatencyMonitor>,
    market_bbo: Option<BboReader>,
    latest_bbo: Option<Bbo>,
    market_stale_ms: u64,
    session_started_at_ms: u64,
    last_fill_received_ns: Option<u64>,
    address_requests_before: u64,
    pub(super) confirmed: Option<ClearinghouseState>,
    pub(super) submitted_orders: u64,
    pub(super) canceled_orders: u64,
    pub(super) cancel_batches: u64,
}

impl SafetyExit {
    pub(super) fn new(backend: &HyperliquidLiveBackend) -> Self {
        Self {
            instrument: backend.instrument.clone(),
            max_slippage_bps: backend.live.emergency_flatten_max_slippage_bps,
            client: backend.client.clone(),
            session: backend.session.clone(),
            state: backend.state.clone(),
            account: backend.account.clone(),
            clock: backend.clock.clone(),
            latency: backend.latency.clone(),
            market_bbo: backend.market_bbo.clone(),
            latest_bbo: backend.latest_bbo,
            market_stale_ms: backend.market_stale_ms,
            session_started_at_ms: backend.session_started_at_ms,
            last_fill_received_ns: backend.last_fill_received_ns,
            address_requests_before: backend.diagnostics.address_requests_used,
            confirmed: None,
            submitted_orders: 0,
            canceled_orders: 0,
            cancel_batches: 0,
        }
    }

    pub(super) async fn run(mut self) -> (Self, Result<()>) {
        let result = async {
            self.cancel_all_bot_orders().await?;
            self.market_close().await
        }
        .await;
        (self, result)
    }

    pub(super) fn apply(self, backend: &mut HyperliquidLiveBackend) -> Result<()> {
        backend.diagnostics.orders_submitted += self.submitted_orders;
        // A quota refresh may already include these actions while I/O ran.
        backend.diagnostics.address_requests_used = backend.diagnostics.address_requests_used.max(
            self.address_requests_before
                .saturating_add(self.submitted_orders),
        );
        backend.diagnostics.cancels_submitted += self.cancel_batches;
        backend.count_cancel_actions(self.canceled_orders);
        if let Some(confirmed) = self.confirmed {
            if confirmed.time >= backend.last_inventory_update_exchange_ms {
                backend
                    .account
                    .apply_clearinghouse(&confirmed, &backend.instrument)?;
                backend.account.open_orders = self.account.open_orders;
                backend.last_inventory_update_exchange_ms = confirmed.time;
                backend.observe_inventory_for_timed_exit(unix_ms());
            }
        }
        Ok(())
    }

    pub(super) async fn cancel_all_bot_orders(&mut self) -> Result<()> {
        let cloids = self.state.with_state(|state| {
            state
                .orders
                .values()
                .filter(|order| !order.status.terminal())
                .map(|order| order.cloid.clone())
                .collect::<Vec<String>>()
        })?;
        if !cloids.is_empty() {
            let action_count = cloids.len() as u64;
            let outcome = self.cancel_cloids_resilient(cloids).await?;
            self.cancel_batches += 1;
            self.canceled_orders += action_count;
            require_action_known(&outcome)?;
        }
        self.reconcile_safety_position().await?;
        if !self.account.open_orders.is_empty() {
            bail!("bot-order cancellation did not produce an empty venue order set");
        }
        Ok(())
    }

    pub(super) async fn market_close(&mut self) -> Result<()> {
        for (attempt, slippage) in [25.0_f64, 100.0, self.max_slippage_bps]
            .into_iter()
            .enumerate()
        {
            if attempt != 0 || self.account.inventory_units == 0 {
                self.reconcile_safety_position().await?;
            }
            let inventory = self.account.inventory_units;
            if inventory == 0 {
                return Ok(());
            }
            let (side, quantity) = closing_side_and_quantity(inventory)
                .context("nonzero inventory has no closing intent")?;
            let bbo = self.fresh_bbo_with_priority(true).await?;
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
            self.submitted_orders += 1;
            require_action_known(&outcome)?;
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(250)).await;
                self.reconcile_safety_position().await?;
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

    pub(super) async fn reconcile_safety_position(&mut self) -> Result<()> {
        let (clearinghouse, open_orders) = tokio::try_join!(
            self.client.clearinghouse_state_safety(),
            self.client.open_orders_safety(),
        )?;
        ensure_no_foreign_positions(&clearinghouse, &self.instrument)?;
        self.account
            .apply_clearinghouse(&clearinghouse, &self.instrument)?;
        self.account.open_orders = open_orders;
        self.confirmed = Some(clearinghouse);
        Ok(())
    }

    pub(super) async fn fresh_bbo_with_priority(&mut self, safety_critical: bool) -> Result<Bbo> {
        if let Some(bbo) = self
            .market_bbo
            .as_ref()
            .and_then(BboReader::load)
            .or(self.latest_bbo)
        {
            if bbo.recv_ns != 0
                && self.clock.now_ns().saturating_sub(bbo.recv_ns)
                    <= self.market_stale_ms.saturating_mul(1_000_000)
            {
                return Ok(bbo);
            }
        }
        let book = if safety_critical {
            self.client.l2_book_safety().await?
        } else {
            self.client.l2_book().await?
        };
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

    pub(super) fn ioc_request(
        &self,
        side: Side,
        qty_units: i64,
        reduce_only: bool,
        slippage_bps: f64,
        bbo: Bbo,
    ) -> Result<LiveOrderRequest> {
        if !slippage_bps.is_finite() || slippage_bps < 0.0 || slippage_bps > self.max_slippage_bps {
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

    pub(super) async fn cancel_cloids_resilient(
        &mut self,
        cloids: Vec<String>,
    ) -> Result<ActionOutcome> {
        if self.session.healthy() {
            return self.session.cancel_cloids(cloids).await;
        }
        self.client
            .cancel_by_cloid_with_nonce(&cloids, self.state.emergency_nonce()?)
            .await
    }
}
