use super::traits::{AccountStateProvider, ExecutionBackend};
use crate::config::{DryRunConfig, QuotingConfig, RiskConfig};
use crate::instrument::InstrumentSpec;
use crate::types::{
    AggressorSide, Bbo, BookSnapshot, DesiredQuotes, DryRunAccountState, ExecutionEvent, Fill,
    MarketEvent, OrderIntent, Side,
};
use anyhow::{bail, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const DRY_RUN_STATE_MAX_AGE_MS: u64 = 15 * 60 * 1_000;
const DRY_RUN_STATE_MAX_FUTURE_SKEW_MS: u64 = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunDiagnostics {
    pub scientifically_valid: bool,
    pub invalid_reason: Option<String>,
    pub virtual_orders_created: u64,
    pub virtual_orders_canceled: u64,
    pub fills: u64,
    pub partial_fills: u64,
    pub queue_ahead_consumed_units: f64,
    pub queue_decay_units: f64,
    pub unknown_queue_activations: u64,
    pub max_working_orders: usize,
    pub liquidation_breach_events: u64,
    /// Post-only orders the venue would have rejected because the market moved
    /// through their price during the latency window. The live path sends ALO
    /// and handles `badAloPxRejected`; the dry run used to rest them anyway,
    /// letting them fill through the touch at a worse price than a real venue
    /// would ever have given.
    pub post_only_rejects: u64,
    /// Aggressive-flatten accounting, kept apart from maker economics so the
    /// entry and exit legs stay separable: this is what truncation cost.
    pub flatten_events: u64,
    pub flatten_units: i64,
    pub flatten_cost_usdc: f64,
    pub markout_usdc: BTreeMap<u64, f64>,
    pub markout_samples: BTreeMap<u64, u64>,
    pub fills_by_depth_bps: BTreeMap<String, u64>,
}

impl Default for DryRunDiagnostics {
    fn default() -> Self {
        Self {
            scientifically_valid: true,
            invalid_reason: None,
            virtual_orders_created: 0,
            virtual_orders_canceled: 0,
            fills: 0,
            partial_fills: 0,
            queue_ahead_consumed_units: 0.0,
            queue_decay_units: 0.0,
            unknown_queue_activations: 0,
            max_working_orders: 0,
            liquidation_breach_events: 0,
            post_only_rejects: 0,
            flatten_events: 0,
            flatten_units: 0,
            flatten_cost_usdc: 0.0,
            markout_usdc: BTreeMap::new(),
            markout_samples: BTreeMap::new(),
            fills_by_depth_bps: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct VirtualOrder {
    id: u64,
    intent: OrderIntent,
    active_ms: u64,
    cancel_effective_ms: Option<u64>,
    remaining_units: i64,
    queue_ahead_units: f64,
    initial_queue_units: f64,
    last_queue_update_ms: u64,
    activated: bool,
    queue_known: bool,
}

#[derive(Debug, Clone)]
struct PendingMarkout {
    due_ms: u64,
    side: Side,
    fill_px: f64,
    size_base: f64,
    horizon_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedDryRunState {
    schema_version: u32,
    symbol: String,
    config_fingerprint: String,
    model_fingerprint: String,
    /// Calibration schema the inventory unit and accounting were produced
    /// under. A state saved by an older estimator is not a safe basis for a
    /// new session even when the config fingerprint still matches.
    #[serde(default)]
    parameter_schema_version: u32,
    inventory_unit: i64,
    account: DryRunAccountState,
    saved_at_ms: u64,
    current_day: Option<u64>,
    daily_realized_pnl_usdc: f64,
}

#[derive(Debug)]
pub struct DryRunBackend {
    instrument: InstrumentSpec,
    config: DryRunConfig,
    quoting: QuotingConfig,
    risk: RiskConfig,
    starting_equity_usdc: f64,
    account: DryRunAccountState,
    diagnostics: DryRunDiagnostics,
    orders: Vec<VirtualOrder>,
    pending_markouts: Vec<PendingMarkout>,
    latest_bbo: Option<Bbo>,
    latest_book: Option<BookSnapshot>,
    next_order_id: u64,
    last_mark_ms: Option<u64>,
    current_day: Option<u64>,
    daily_realized_pnl_usdc: f64,
    restored_inventory_unit: Option<i64>,
    /// Exchange time of the last *submitted* quote action, for the requote
    /// cooldown. Only a real place/cancel advances it -- a held quote must not
    /// restart the window, or a busy market would defer forever.
    last_quote_action_ms: u64,
    inventory_at_last_quote_action: i64,
    /// A replacement that arrived inside the cooldown, applied on the next
    /// market event. Mirrors the live backend's `deferred_desired`.
    deferred_desired: Option<DesiredQuotes>,
    /// Open exposure in arrival order, `(exchange_ms, signed_units)`. Positive
    /// is long. Used only by the flatten policy, to age the OLDEST lot rather
    /// than the net position: a position that is repeatedly topped up would
    /// otherwise never look old enough to exit.
    open_lots: std::collections::VecDeque<(u64, i64)>,
}

impl DryRunBackend {
    pub fn new(
        instrument: InstrumentSpec,
        config: DryRunConfig,
        quoting: QuotingConfig,
        risk: RiskConfig,
    ) -> Result<Self> {
        instrument.validate()?;
        if config.queue_decay_per_second != 0.0 {
            bail!(
                "queue_decay_per_second must be zero until a queue-decay model is calibrated from venue evidence"
            );
        }
        if !config.flatten_fee_rate.is_finite()
            || config.flatten_fee_rate < 0.0
            || !config.flatten_slippage_bps.is_finite()
            || config.flatten_slippage_bps < 0.0
        {
            bail!("flatten fee and slippage must be finite and non-negative");
        }
        if !config.promotion_flatten_fee_rate.is_finite()
            || config.promotion_flatten_fee_rate < 0.0
            || !config.promotion_flatten_slippage_bps.is_finite()
            || config.promotion_flatten_slippage_bps < 0.0
        {
            bail!("promotion flatten fee and slippage must be finite and non-negative");
        }
        let starting_equity_usdc = config.starting_equity_usdc;
        Ok(Self {
            instrument,
            config,
            quoting,
            risk,
            starting_equity_usdc,
            account: DryRunAccountState {
                cash_usdc: starting_equity_usdc,
                equity_usdc: starting_equity_usdc,
                ..DryRunAccountState::default()
            },
            diagnostics: DryRunDiagnostics::default(),
            orders: Vec::new(),
            pending_markouts: Vec::new(),
            latest_bbo: None,
            latest_book: None,
            next_order_id: 1,
            last_mark_ms: None,
            current_day: None,
            daily_realized_pnl_usdc: 0.0,
            restored_inventory_unit: None,
            last_quote_action_ms: 0,
            inventory_at_last_quote_action: 0,
            deferred_desired: None,
            open_lots: std::collections::VecDeque::new(),
        })
    }

    pub const fn diagnostics(&self) -> &DryRunDiagnostics {
        &self.diagnostics
    }

    /// The account as it stands, for a caller that checkpoints it itself.
    ///
    /// `save_account_state` owns a whole file and one variant's identity, which
    /// suits `dry-run`. The grid checkpoints every configured variant plus
    /// run-level feed counters into one document, so it needs the parts rather
    /// than the file.
    pub const fn account_snapshot(&self) -> DryRunAccountState {
        self.account
    }

    /// Restore from a caller-held checkpoint, the counterpart of
    /// `account_snapshot`.
    ///
    /// Deliberately keeps `restore_account_state`'s reset of live state:
    /// resting orders, pending markouts and a deferred target all belonged to a
    /// book that no longer exists. Only the *accounting* -- cash, inventory,
    /// realized P&L, fees, and the diagnostics that summarise them -- carries
    /// across, which is exactly what makes a resumed run continuous without
    /// pretending the intervening market was observed.
    pub fn restore_from_snapshot(
        &mut self,
        account: DryRunAccountState,
        diagnostics: DryRunDiagnostics,
        inventory_unit: i64,
        current_day: Option<u64>,
        daily_realized_pnl_usdc: f64,
    ) -> Result<()> {
        if !account.cash_usdc.is_finite()
            || !account.equity_usdc.is_finite()
            || !account.average_entry_px.is_finite()
        {
            bail!("checkpointed dry-run state contains non-finite values");
        }
        if inventory_unit <= 0 {
            bail!("checkpointed dry-run state has a non-positive inventory unit");
        }
        self.account = account;
        self.diagnostics = diagnostics;
        self.current_day = current_day;
        self.daily_realized_pnl_usdc = daily_realized_pnl_usdc;
        self.restored_inventory_unit = Some(inventory_unit);
        self.orders.clear();
        self.pending_markouts.clear();
        self.deferred_desired = None;
        // Restored inventory has no lot behind it, and an empty deque would
        // make the flatten silently self-destruct: every reducing maker fill
        // would land as a NEW opposite-direction lot, get crossed back out, and
        // re-establish the position while paying taker each time -- the variant
        // could never get flat. Seed one lot for the whole position, aged from
        // now, so the deadline applies to it like any other.
        self.open_lots.clear();
        if self.account.inventory_units != 0 {
            self.open_lots
                // Timestamp 0: a position carried across a restart is older
                // than any deadline by definition, so it is crossed out on the
                // first event that offers a touch.
                .push_back((0, self.account.inventory_units));
        }
        self.last_quote_action_ms = 0;
        self.inventory_at_last_quote_action = self.account.inventory_units;
        Ok(())
    }

    /// Close a position carried across a long restart gap at the last price
    /// the run actually observed, returning the P&L that became realized.
    ///
    /// `restore_from_snapshot` deliberately carries inventory, which is right
    /// for a brief interruption: the position is marked a few seconds later at
    /// a price the run all but saw. Across a long gap it is wrong, and wrong in
    /// the direction that flatters. The first mark of the new session prices
    /// the carried position at a level whose whole path was unobserved, so the
    /// move lands in `mark_to_market_pnl_usdc` as though the strategy had
    /// earned it -- the mechanism that made the 46.4 h leaderboard of
    /// 2026-08-27 report a 13.2% rally as profit.
    ///
    /// Closing at the checkpoint's own mark is the honest resolution. Equity is
    /// unchanged across the call, because the position's market value simply
    /// moves into cash: the P&L curve stays continuous and no gain or loss is
    /// invented. What goes away is the exposure, so nothing that happens during
    /// the unobserved window can be attributed to a decision the strategy made
    /// before it.
    ///
    /// Returns `None` when the variant was already flat, so the caller can
    /// report how many positions this actually touched.
    pub fn flatten_carried_position(&mut self) -> Option<f64> {
        if self.account.inventory_units == 0 {
            return None;
        }
        // `mark_account` maintains `equity = cash + inventory_base * mid`, so
        // the difference is the position's value at the last observed mid --
        // recovered without needing that price to have been checkpointed.
        let market_value_usdc = self.account.equity_usdc - self.account.cash_usdc;
        let inventory_base = self
            .instrument
            .size_from_units(self.account.inventory_units);
        let closed_pnl_usdc = market_value_usdc - inventory_base * self.account.average_entry_px;

        self.account.realized_pnl_usdc += closed_pnl_usdc;
        self.account.cash_usdc = self.account.equity_usdc;
        self.account.inventory_units = 0;
        self.account.average_entry_px = 0.0;
        self.account.position_notional_usdc = 0.0;
        self.account.margin_used_usdc = 0.0;
        self.account.maintenance_margin_usdc = 0.0;
        self.account.liquidation_buffer_usdc = self.account.equity_usdc;
        // The seeded carry lot describes a position that no longer exists.
        self.open_lots.clear();
        self.inventory_at_last_quote_action = 0;
        Some(closed_pnl_usdc)
    }

    pub const fn daily_realized_pnl_usdc(&self) -> f64 {
        self.daily_realized_pnl_usdc
    }

    pub const fn daily_risk_snapshot(&self) -> (Option<u64>, f64) {
        (self.current_day, self.daily_realized_pnl_usdc)
    }

    /// P&L after immediately closing residual inventory at the executable side
    /// of the current book, including taker fee and configured adverse slippage.
    pub fn promotion_pnl_usdc(&self, bbo: Bbo) -> f64 {
        let inventory = self.account.inventory_units;
        if inventory == 0 {
            return self.account.cash_usdc - self.starting_equity_usdc;
        }
        let quantity = self.instrument.size_from_units(inventory.abs());
        let slippage = self.config.promotion_flatten_slippage_bps / 10_000.0;
        let execution_price = if inventory > 0 {
            self.instrument.price_from_units(bbo.bid_px) * (1.0 - slippage)
        } else {
            self.instrument.price_from_units(bbo.ask_px) * (1.0 + slippage)
        };
        let notional = quantity * execution_price;
        let fee = notional * self.config.promotion_flatten_fee_rate;
        let flattened_cash = if inventory > 0 {
            self.account.cash_usdc + notional - fee
        } else {
            self.account.cash_usdc - notional - fee
        };
        flattened_cash - self.starting_equity_usdc
    }

    pub fn working_order_count(&self) -> usize {
        self.orders.len()
    }

    pub fn save_account_state(
        &self,
        path: &Path,
        config_fingerprint: &str,
        model_fingerprint: &str,
        parameter_schema_version: u32,
        inventory_unit: i64,
    ) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary =
            tempfile::NamedTempFile::new_in(path.parent().unwrap_or_else(|| Path::new(".")))?;
        serde_json::to_writer_pretty(
            temporary.as_file(),
            &PersistedDryRunState {
                schema_version: 3,
                symbol: self.instrument.symbol.clone(),
                config_fingerprint: config_fingerprint.to_owned(),
                model_fingerprint: model_fingerprint.to_owned(),
                parameter_schema_version,
                inventory_unit,
                account: self.account,
                saved_at_ms: unix_ms(),
                current_day: self.current_day,
                daily_realized_pnl_usdc: self.daily_realized_pnl_usdc,
            },
        )?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| anyhow::anyhow!(error.error))?;
        Ok(())
    }

    pub fn restore_account_state(
        &mut self,
        path: &Path,
        expected_config_fingerprint: &str,
        expected_parameter_schema_version: u32,
    ) -> Result<bool> {
        let Ok(bytes) = std::fs::read(path) else {
            return Ok(false);
        };
        let Ok(persisted) = serde_json::from_slice::<PersistedDryRunState>(&bytes) else {
            // Pre-schema local state has no symbol/config identity and is not a
            // safe basis for a new scientific session.
            return Ok(false);
        };
        let now_ms = unix_ms();
        if persisted.schema_version != 3
            || persisted.symbol != self.instrument.symbol
            || persisted.config_fingerprint != expected_config_fingerprint
            || persisted.parameter_schema_version != expected_parameter_schema_version
            || persisted.inventory_unit <= 0
            || now_ms.saturating_sub(persisted.saved_at_ms) > DRY_RUN_STATE_MAX_AGE_MS
            || persisted.saved_at_ms > now_ms.saturating_add(DRY_RUN_STATE_MAX_FUTURE_SKEW_MS)
        {
            return Ok(false);
        }
        let state = persisted.account;
        if !state.cash_usdc.is_finite()
            || !state.equity_usdc.is_finite()
            || !state.average_entry_px.is_finite()
        {
            bail!("dry-run state contains non-finite values");
        }
        self.account = state;
        self.current_day = persisted.current_day;
        self.daily_realized_pnl_usdc = persisted.daily_realized_pnl_usdc;
        self.restored_inventory_unit = Some(persisted.inventory_unit);
        self.orders.clear();
        self.pending_markouts.clear();
        // A target deferred against the previous session's book is meaningless
        // once the book is gone, and the cooldown restarts with the session.
        // Restored inventory has no lot behind it, and an empty deque would
        // make the flatten silently self-destruct: every reducing maker fill
        // would land as a NEW opposite-direction lot, get crossed back out, and
        // re-establish the position while paying taker each time -- the variant
        // could never get flat. Seed one lot for the whole position, aged from
        // now, so the deadline applies to it like any other.
        self.open_lots.clear();
        if self.account.inventory_units != 0 {
            self.open_lots
                // Timestamp 0: a position carried across a restart is older
                // than any deadline by definition, so it is crossed out on the
                // first event that offers a touch.
                .push_back((0, self.account.inventory_units));
        }
        self.deferred_desired = None;
        self.last_quote_action_ms = 0;
        self.inventory_at_last_quote_action = self.account.inventory_units;
        Ok(true)
    }

    pub const fn restored_inventory_unit(&self) -> Option<i64> {
        self.restored_inventory_unit
    }

    /// A live (non-cancelling) virtual order is held when the new target sits
    /// inside the hold window at the same size and reduce-only flag.
    fn order_is_held(&self, order: &VirtualOrder, intent: OrderIntent) -> bool {
        order.cancel_effective_ms.is_none()
            && order.intent.side == intent.side
            && order.intent.qty_units == intent.qty_units
            && order.intent.reduce_only == intent.reduce_only
            && (order.intent.px - intent.px).abs()
                < self
                    .instrument
                    .requote_hold_window_units(&self.quoting, intent.px)
    }

    /// Apply a target now. Returns whether any order was actually placed or
    /// cancelled -- only a real action restarts the requote cooldown, matching
    /// the live backend where a held quote deliberately does not.
    fn reconcile_now(&mut self, desired: DesiredQuotes, now_ms: u64) -> bool {
        self.expire_cancels(now_ms);
        let mut acted = false;
        let cancel_latency_ms = self.simulated_latency_ms(self.config.cancel_latency_ms, now_ms);
        let decision_latency_ms =
            self.simulated_latency_ms(self.config.decision_latency_ms, now_ms);
        let acknowledgement_latency_ms =
            self.simulated_latency_ms(self.config.acknowledgement_latency_ms, now_ms);
        for side in [Side::Buy, Side::Sell] {
            let next = match side {
                Side::Buy => desired.bid,
                Side::Sell => desired.ask,
            };
            let unchanged = next.is_some_and(|intent| {
                self.orders
                    .iter()
                    .any(|order| self.order_is_held(order, intent))
            });
            let held: Vec<u64> = next
                .map(|intent| {
                    self.orders
                        .iter()
                        .filter(|order| self.order_is_held(order, intent))
                        .map(|order| order.id)
                        .collect()
                })
                .unwrap_or_default();
            let cancel_effective = now_ms.saturating_add(cancel_latency_ms);
            for order in self
                .orders
                .iter_mut()
                .filter(|order| order.intent.side == side && order.cancel_effective_ms.is_none())
            {
                if !held.contains(&order.id) {
                    order.cancel_effective_ms = Some(cancel_effective);
                    acted = true;
                }
            }
            if !unchanged {
                if let Some(intent) = next {
                    let id = self.next_order_id;
                    self.next_order_id = self.next_order_id.wrapping_add(1).max(1);
                    self.orders.push(VirtualOrder {
                        id,
                        intent,
                        active_ms: now_ms
                            .saturating_add(decision_latency_ms)
                            .saturating_add(acknowledgement_latency_ms),
                        cancel_effective_ms: None,
                        remaining_units: intent.qty_units,
                        queue_ahead_units: 0.0,
                        initial_queue_units: 0.0,
                        last_queue_update_ms: now_ms,
                        activated: false,
                        queue_known: false,
                    });
                    self.diagnostics.virtual_orders_created += 1;
                    acted = true;
                }
            }
        }
        self.diagnostics.max_working_orders =
            self.diagnostics.max_working_orders.max(self.orders.len());
        acted
    }

    fn simulated_latency_ms(&self, median_ms: u64, exchange_ms: u64) -> u64 {
        let tail = (exchange_ms / 1_000) % self.config.tail_latency_every
            == self.config.tail_latency_every - 1;
        if tail {
            (median_ms as f64 * self.config.tail_latency_multiplier).round() as u64
        } else {
            median_ms
        }
    }

    /// Apply a target, stamping the cooldown when it actually did something.
    fn apply_desired(&mut self, desired: DesiredQuotes, now_ms: u64) {
        if self.reconcile_now(desired, now_ms) {
            self.last_quote_action_ms = now_ms;
            self.inventory_at_last_quote_action = self.account.inventory_units;
        }
    }

    /// Apply a quote deferred by the cooldown, once the window has elapsed.
    ///
    /// This lives on the market-event path rather than at the top of the next
    /// `reconcile` because the public dry run only reconciles when the hot path
    /// publishes a new quote sequence: a deferred target could otherwise sit
    /// indefinitely while publication stalls.
    fn flush_deferred(&mut self, now_ms: u64) {
        if self.deferred_desired.is_none() {
            return;
        }
        if now_ms.saturating_sub(self.last_quote_action_ms) < self.quoting.min_order_lifetime_ms {
            return;
        }
        let Some(desired) = self.deferred_desired.take() else {
            return;
        };
        self.apply_desired(desired, now_ms);
    }

    fn process_event(&mut self, event: &MarketEvent) -> Vec<ExecutionEvent> {
        let now_ms = event_exchange_ms(event);
        self.flush_deferred(now_ms);
        self.roll_day(now_ms);
        self.expire_cancels(now_ms);
        self.activate_orders(now_ms);
        let mut output = Vec::new();
        match event {
            MarketEvent::Bbo(bbo) => {
                self.accrue_funding(*bbo);
                self.resolve_markouts(*bbo);
                self.latest_bbo = Some(*bbo);
                self.refresh_unknown_queues(now_ms);
                self.mark_account(bbo.mid_units());
            }
            MarketEvent::Book(book) => {
                self.latest_book = Some(book.clone());
                self.refresh_unknown_queues(now_ms);
            }
            MarketEvent::Trade(trade) => {
                self.refresh_unknown_queues(now_ms);
                for fill in self.match_trade(*trade) {
                    self.record_lot(
                        fill.exchange_ms,
                        fill.side.inventory_sign().saturating_mul(fill.qty_units),
                    );
                    self.apply_fill(fill);
                    output.push(ExecutionEvent::Fill(fill));
                }
            }
        }
        // After the event, not before: a lot opened by this very trade must be
        // allowed to age before it can be crossed out.
        output.extend(self.flatten_stale_lots(now_ms));
        output
    }

    /// FIFO-match a fill against opposing open lots; the remainder opens a new
    /// one. Mirrors `match_holding_time` in `scripts/replay_market_maker.py`, so
    /// the two simulators age a position the same way.
    fn record_lot(&mut self, ts_ms: u64, signed_units: i64) {
        let mut remaining = signed_units;
        while remaining != 0 {
            let Some(&(_, lot)) = self.open_lots.front() else {
                break;
            };
            if (lot > 0) == (remaining > 0) {
                break; // same direction: this adds to the position
            }
            let matched = lot.abs().min(remaining.abs());
            if matched <= 0 {
                break;
            }
            if lot.abs() > matched {
                let front = self.open_lots.front_mut().expect("checked above");
                front.1 -= matched * lot.signum();
            } else {
                self.open_lots.pop_front();
            }
            remaining -= matched * remaining.signum();
        }
        if remaining != 0 {
            self.open_lots.push_back((ts_ms, remaining));
        }
    }

    /// Cross out the oldest lot once it has been held past the deadline.
    ///
    /// Charges the same costs the replay does: the crossed half-spread (we take
    /// the far touch), `flatten_slippage_bps` for walking the book, and
    /// `flatten_fee_rate`. Without a touch there is nothing to cross into, so
    /// the lot waits rather than being priced from an invented market.
    fn flatten_stale_lots(&mut self, now_ms: u64) -> Vec<ExecutionEvent> {
        // The deadline carries the SAME round trip a quote pays: we notice the
        // position, decide, and the taker order reaches the venue one trip
        // later. Without it `flatten0` was a ~130 ms exit -- unreachable at a
        // p50 281 ms RTT -- so the grid would have tested a rung that does not
        // exist rather than the 450-500 ms cliff the replay predicts.
        let mut events = Vec::new();
        // Disabled is a property of the CONFIGURED deadline, not the computed
        // one: adding the round trip makes the latter non-zero even when the
        // policy is off.
        if self.config.flatten_after_ms == 0 {
            return events;
        }
        let deadline_ms = self
            .config
            .flatten_after_ms
            .saturating_add(self.config.decision_latency_ms)
            .saturating_add(self.config.acknowledgement_latency_ms);
        let Some(bbo) = self.latest_bbo else {
            return events;
        };
        while let Some(&(opened_ms, lot)) = self.open_lots.front() {
            if lot == 0 {
                self.open_lots.pop_front();
                continue;
            }
            if now_ms.saturating_sub(opened_ms) < deadline_ms {
                break;
            }
            let long = lot > 0;
            let touch_units = if long { bbo.bid_px } else { bbo.ask_px };
            if touch_units <= 0 {
                break;
            }
            let touch = self.instrument.price_from_units(touch_units);
            let slip = touch * (self.config.flatten_slippage_bps / 10_000.0);
            let exit_px = if long { touch - slip } else { touch + slip };
            if !exit_px.is_finite() || exit_px <= 0.0 {
                break;
            }
            // Round to the tick FIRST, then derive notional and fee from the
            // rounded price. The Fill carries integer units and `apply_fill`
            // reads the price back out of them, so a fee computed from the raw
            // price would not match the cash the same fill moves.
            let Ok(exit_units) = self.instrument.price_to_units(exit_px) else {
                break;
            };
            let exit_px = self.instrument.price_from_units(exit_units);
            let qty_units = lot.abs();
            let qty_base = self.instrument.size_from_units(qty_units);
            let notional = qty_base * exit_px;
            let fee = notional * self.config.flatten_fee_rate;
            let mid = self.instrument.price_from_units(bbo.mid_units());
            self.diagnostics.flatten_events += 1;
            self.diagnostics.flatten_units += qty_units;
            self.diagnostics.flatten_cost_usdc += (mid - exit_px).abs() * qty_base + fee;
            let fill = Fill {
                side: if long { Side::Sell } else { Side::Buy },
                px: exit_units,
                qty_units,
                fee_usdc: fee,
                exchange_ms: now_ms,
                virtual_order_id: 0,
                maker: false,
            };
            self.open_lots.pop_front();
            self.apply_fill(fill);
            events.push(ExecutionEvent::Fill(fill));
        }
        events
    }

    fn activate_orders(&mut self, now_ms: u64) {
        // ALO first: a post-only order that reaches the venue crossing the
        // opposite touch is REJECTED, not rested. The market moves during the
        // ~300 ms latency window, so this is not rare -- measured at ~8% of
        // baseline fills, and every one of them filled through the touch at a
        // price no venue would have given us.
        if let Some(bbo) = self.latest_bbo {
            let mut rejected = 0_u64;
            self.orders.retain(|order| {
                if order.activated || order.active_ms > now_ms || !order.intent.post_only {
                    return true;
                }
                let crosses = match order.intent.side {
                    Side::Buy => bbo.ask_px > 0 && order.intent.px >= bbo.ask_px,
                    Side::Sell => bbo.bid_px > 0 && order.intent.px <= bbo.bid_px,
                };
                if crosses {
                    rejected += 1;
                }
                !crosses
            });
            self.diagnostics.post_only_rejects += rejected;
        }
        for order in self
            .orders
            .iter_mut()
            .filter(|order| !order.activated && order.active_ms <= now_ms)
        {
            let visible = visible_queue(self.latest_bbo, self.latest_book.as_ref(), order.intent);
            if let Some(visible) = visible {
                order.queue_ahead_units = visible;
                order.initial_queue_units = visible;
                order.queue_known = true;
            } else {
                self.diagnostics.unknown_queue_activations =
                    self.diagnostics.unknown_queue_activations.saturating_add(1);
            }
            order.last_queue_update_ms = order.active_ms;
            order.activated = true;
        }
    }

    fn refresh_unknown_queues(&mut self, now_ms: u64) {
        for order in self
            .orders
            .iter_mut()
            .filter(|order| order.activated && !order.queue_known)
        {
            let Some(visible) =
                visible_queue(self.latest_bbo, self.latest_book.as_ref(), order.intent)
            else {
                continue;
            };
            order.queue_ahead_units = visible;
            order.initial_queue_units = visible;
            order.last_queue_update_ms = now_ms;
            order.queue_known = true;
        }
    }

    fn expire_cancels(&mut self, now_ms: u64) {
        let before = self.orders.len();
        self.orders.retain(|order| {
            order
                .cancel_effective_ms
                .is_none_or(|cancel_ms| cancel_ms > now_ms)
                && order.remaining_units > 0
        });
        self.diagnostics.virtual_orders_canceled += (before - self.orders.len()) as u64;
    }

    fn match_trade(&mut self, trade: crate::types::TradePrint) -> Vec<Fill> {
        let maker_side = match trade.aggressor {
            AggressorSide::Buy => Side::Sell,
            AggressorSide::Sell => Side::Buy,
        };
        let mut indices: Vec<usize> = self
            .orders
            .iter()
            .enumerate()
            .filter(|(_, order)| {
                order.activated
                    // A print STRICTLY BEYOND our price means the venue consumed
                    // everything resting at our price and kept going, so we were
                    // filled whatever our queue position was. Requiring
                    // `queue_known` for those excluded any quote resting deeper
                    // than the 20-level book from ever filling: the book reaches
                    // 60 bps on 0.2% of snapshots, so `wide60` took 4 fills in
                    // 15 h where the tape had 87 sweeps past it. Queue position
                    // only decides an AT-price print, which still needs it.
                    && (order.queue_known
                        || match maker_side {
                            Side::Buy => trade.px < order.intent.px,
                            Side::Sell => trade.px > order.intent.px,
                        })
                    && order.intent.side == maker_side
                    && order
                        .cancel_effective_ms
                        .is_none_or(|cancel| cancel > trade.exchange_ms)
                    && match maker_side {
                        Side::Buy => trade.px <= order.intent.px,
                        Side::Sell => trade.px >= order.intent.px,
                    }
            })
            .map(|(index, _)| index)
            .collect();
        indices.sort_by(|left, right| {
            let a = &self.orders[*left];
            let b = &self.orders[*right];
            let price_order = match maker_side {
                Side::Buy => b.intent.px.cmp(&a.intent.px),
                Side::Sell => a.intent.px.cmp(&b.intent.px),
            };
            price_order
                .then_with(|| a.active_ms.cmp(&b.active_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        let mut remaining_trade = trade.qty_units.max(0) as f64;
        let mut fills = Vec::new();
        let mut projected_inventory = self.account.inventory_units;
        for index in indices {
            if remaining_trade <= 0.0 {
                break;
            }
            let order = &mut self.orders[index];
            order.last_queue_update_ms = trade.exchange_ms;
            // A print strictly beyond our price consumed our whole level, so
            // there is no queue left to wait behind -- whether or not we could
            // see it. Charging it only in the unknown case made shallow
            // variants ~10% harder to fill than deep ones, a cross-variant bias
            // in exactly the comparison the grid exists to make.
            let beyond = match maker_side {
                Side::Buy => trade.px < order.intent.px,
                Side::Sell => trade.px > order.intent.px,
            };
            if beyond {
                order.queue_ahead_units = 0.0;
            }
            let queue_consumed = order.queue_ahead_units.min(remaining_trade);
            order.queue_ahead_units -= queue_consumed;
            remaining_trade -= queue_consumed;
            self.diagnostics.queue_ahead_consumed_units += queue_consumed;
            if order.queue_ahead_units > 0.0 || remaining_trade < 1.0 {
                continue;
            }
            let reduce_room = if order.intent.reduce_only {
                match order.intent.side {
                    Side::Buy if projected_inventory < 0 => {
                        projected_inventory.unsigned_abs() as i64
                    }
                    Side::Sell if projected_inventory > 0 => projected_inventory,
                    _ => 0,
                }
            } else {
                i64::MAX
            };
            let fill_units = (remaining_trade.floor() as i64)
                .min(order.remaining_units)
                .min(reduce_room)
                .max(0);
            if fill_units == 0 {
                continue;
            }
            order.remaining_units -= fill_units;
            remaining_trade -= fill_units as f64;
            projected_inventory = projected_inventory.saturating_add(
                order
                    .intent
                    .side
                    .inventory_sign()
                    .saturating_mul(fill_units),
            );
            self.diagnostics.fills += 1;
            if order.remaining_units > 0 {
                self.diagnostics.partial_fills += 1;
            }
            let price = self.instrument.price_from_units(order.intent.px);
            let base = self.instrument.size_from_units(fill_units);
            fills.push(Fill {
                side: order.intent.side,
                px: order.intent.px,
                qty_units: fill_units,
                fee_usdc: price * base * self.quoting.maker_fee_rate,
                exchange_ms: trade.exchange_ms,
                virtual_order_id: order.id,
                maker: true,
            });
        }
        self.orders.retain(|order| order.remaining_units > 0);
        fills
    }

    fn apply_fill(&mut self, fill: Fill) {
        let old_inventory = self.account.inventory_units;
        let signed_fill = fill.side.inventory_sign().saturating_mul(fill.qty_units);
        let new_inventory = old_inventory.saturating_add(signed_fill);
        let price = self.instrument.price_from_units(fill.px);
        let fill_base = self.instrument.size_from_units(fill.qty_units);
        let notional = fill_base * price;
        match fill.side {
            Side::Buy => self.account.cash_usdc -= notional + fill.fee_usdc,
            Side::Sell => self.account.cash_usdc += notional - fill.fee_usdc,
        }
        self.account.fees_usdc += fill.fee_usdc;
        if let Some(bbo) = self.latest_bbo {
            let mid = self.instrument.price_from_units(bbo.mid_units());
            if mid > 0.0 && fill.maker {
                let depth_bps = (price - mid).abs() / mid * 10_000.0;
                let bucket = (depth_bps * 10.0).floor() / 10.0;
                *self
                    .diagnostics
                    .fills_by_depth_bps
                    .entry(format!("{}:{bucket:.1}", fill.side.quote_side()))
                    .or_default() += 1;
            }
        }

        let same_direction = old_inventory == 0 || old_inventory.signum() == signed_fill.signum();
        let reduced_existing_position = !same_direction;
        let mut realized_delta = -fill.fee_usdc;
        if same_direction {
            let old_base = self
                .instrument
                .size_from_units(old_inventory.unsigned_abs() as i64);
            let total_base = old_base + fill_base;
            self.account.average_entry_px = if total_base > 0.0 {
                (old_base * self.account.average_entry_px + fill_base * price) / total_base
            } else {
                0.0
            };
        } else {
            let closing_units = old_inventory.unsigned_abs().min(fill.qty_units as u64) as i64;
            let closing_base = self.instrument.size_from_units(closing_units);
            realized_delta += if old_inventory > 0 {
                (price - self.account.average_entry_px) * closing_base
            } else {
                (self.account.average_entry_px - price) * closing_base
            };
            if new_inventory == 0 {
                self.account.average_entry_px = 0.0;
            } else if new_inventory.signum() != old_inventory.signum() {
                self.account.average_entry_px = price;
            }
        }
        self.account.inventory_units = new_inventory;
        self.account.realized_pnl_usdc += realized_delta;
        self.daily_realized_pnl_usdc += realized_delta;
        if reduced_existing_position {
            if realized_delta < 0.0 {
                self.account.consecutive_losses = self.account.consecutive_losses.saturating_add(1);
                // The breaker LATCHES rather than cooling off: `quote.rs` stops
                // quoting at the cap, and this counter only resets on a winning
                // closing fill -- which a variant that can no longer fill will
                // never get. So the variant is finished, and a row that keeps
                // reporting itself valid is a lie rather than a result. On
                // 2026-09-04 that hid a variant frozen for 8.7 h of a 9.6 h run.
                // Nothing else would have shown it: `Metrics::risk_refusals` is
                // hot-path-only and the grid spawns no hot-path threads.
                if self.account.consecutive_losses >= self.risk.max_consecutive_losses {
                    self.invalidate("consecutive-loss breaker latched");
                }
            } else if realized_delta > 0.0 {
                self.account.consecutive_losses = 0;
            }
        }
        // Markouts measure what our PASSIVE quotes are picked off by. A taker
        // exit is us choosing the price, so folding it in would contaminate the
        // one statistic that says whether the maker side is adversely selected.
        // Same for the depth histogram: an exit at the touch is not a quote
        // depth.
        for horizon in &self.config.markout_horizons_ms {
            if !fill.maker {
                break;
            }
            self.pending_markouts.push(PendingMarkout {
                due_ms: fill.exchange_ms.saturating_add(*horizon),
                side: fill.side,
                fill_px: price,
                size_base: fill_base,
                horizon_ms: *horizon,
            });
        }
        if let Some(bbo) = self.latest_bbo {
            self.mark_account(bbo.mid_units());
        }
    }

    fn mark_account(&mut self, mid_units: i64) {
        let mid = self.instrument.price_from_units(mid_units);
        let inventory_base = self
            .instrument
            .size_from_units(self.account.inventory_units);
        self.account.equity_usdc = self.account.cash_usdc + inventory_base * mid;
        self.account.mark_to_market_pnl_usdc = self.account.equity_usdc - self.starting_equity_usdc;
        self.account.position_notional_usdc = inventory_base.abs() * mid;
        self.account.margin_used_usdc = self.account.position_notional_usdc / self.quoting.leverage;
        self.account.maintenance_margin_usdc =
            self.account.position_notional_usdc * self.risk.maintenance_margin_rate;
        self.account.liquidation_buffer_usdc =
            self.account.equity_usdc - self.account.maintenance_margin_usdc;
        if self.account.liquidation_buffer_usdc < self.risk.min_liquidation_buffer_usdc {
            self.diagnostics.liquidation_breach_events += 1;
            self.invalidate("liquidation buffer breached");
        }
    }

    fn accrue_funding(&mut self, bbo: Bbo) {
        if let Some(previous_ms) = self.last_mark_ms {
            let hours = bbo.exchange_ms.saturating_sub(previous_ms) as f64 / 3_600_000.0;
            let inventory_base = self
                .instrument
                .size_from_units(self.account.inventory_units);
            let mid = self.instrument.price_from_units(bbo.mid_units());
            let funding = -inventory_base * mid * self.config.funding_rate_per_hour * hours;
            self.account.funding_usdc += funding;
            self.account.cash_usdc += funding;
        }
        self.last_mark_ms = Some(bbo.exchange_ms);
    }

    fn resolve_markouts(&mut self, bbo: Bbo) {
        let mid = self.instrument.price_from_units(bbo.mid_units());
        let mut pending = Vec::new();
        for sample in self.pending_markouts.drain(..) {
            if sample.due_ms > bbo.exchange_ms {
                pending.push(sample);
                continue;
            }
            let per_base = match sample.side {
                Side::Buy => mid - sample.fill_px,
                Side::Sell => sample.fill_px - mid,
            };
            *self
                .diagnostics
                .markout_usdc
                .entry(sample.horizon_ms)
                .or_default() += per_base * sample.size_base;
            *self
                .diagnostics
                .markout_samples
                .entry(sample.horizon_ms)
                .or_default() += 1;
        }
        self.pending_markouts = pending;
    }

    fn roll_day(&mut self, now_ms: u64) {
        let day = now_ms / 86_400_000;
        if self.current_day.is_some_and(|current| current != day) {
            self.daily_realized_pnl_usdc = 0.0;
        }
        self.current_day = Some(day);
    }
}

#[async_trait]
impl ExecutionBackend for DryRunBackend {
    async fn reconcile(&mut self, desired: DesiredQuotes, now_ms: u64) -> Result<()> {
        if !self.diagnostics.scientifically_valid
            && (desired.bid.is_some() || desired.ask.is_some())
        {
            bail!("cannot quote in an invalidated dry-run session");
        }
        // Requote cooldown, mirroring the live backend. Withdrawals and
        // inventory-moving fills are never deferred: an empty target is what
        // cancels resting orders, so the toxic-flow guard and the risk limits
        // depend on reaching the book immediately.
        let inventory_changed = self.account.inventory_units != self.inventory_at_last_quote_action;
        if desired.reason.replacement_may_wait(inventory_changed)
            && self.last_quote_action_ms != 0
            && now_ms.saturating_sub(self.last_quote_action_ms) < self.quoting.min_order_lifetime_ms
        {
            self.deferred_desired = Some(desired);
            return Ok(());
        }
        self.deferred_desired = None;
        self.apply_desired(desired, now_ms);
        Ok(())
    }

    async fn on_market_event(&mut self, event: &MarketEvent) -> Result<Vec<ExecutionEvent>> {
        Ok(self.process_event(event))
    }

    async fn shutdown(&mut self, now_ms: u64) -> Result<()> {
        for order in &mut self.orders {
            order.cancel_effective_ms = Some(now_ms);
        }
        self.expire_cancels(now_ms);
        Ok(())
    }

    fn invalidate(&mut self, reason: &str) {
        self.deferred_desired = None;
        self.diagnostics.scientifically_valid = false;
        self.diagnostics.invalid_reason = Some(reason.to_owned());
        self.orders.clear();
    }

    fn scientifically_valid(&self) -> bool {
        self.diagnostics.scientifically_valid
    }
}

impl AccountStateProvider for DryRunBackend {
    fn account_state(&self) -> DryRunAccountState {
        self.account
    }
}

fn event_exchange_ms(event: &MarketEvent) -> u64 {
    match event {
        MarketEvent::Bbo(value) => value.exchange_ms,
        MarketEvent::Trade(value) => value.exchange_ms,
        MarketEvent::Book(value) => value.exchange_ms,
    }
}

fn visible_queue(
    bbo: Option<Bbo>,
    book: Option<&BookSnapshot>,
    intent: OrderIntent,
) -> Option<f64> {
    if let Some(book) = book {
        let levels = match intent.side {
            Side::Buy => &book.bids,
            Side::Sell => &book.asks,
        };
        if let Some(level) = levels.iter().find(|level| level.px == intent.px) {
            return Some(level.qty_units.max(0) as f64);
        }
    }
    bbo.and_then(|bbo| match intent.side {
        Side::Buy if intent.px == bbo.bid_px => Some(bbo.bid_sz.max(0) as f64),
        Side::Sell if intent.px == bbo.ask_px => Some(bbo.ask_sz.max(0) as f64),
        _ => None,
    })
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{QuoteReason, TradePrint};

    fn instrument() -> InstrumentSpec {
        InstrumentSpec {
            symbol: "TEST".to_owned(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 0,
            max_price_decimals: 2,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        }
    }

    fn bid_quotes() -> DesiredQuotes {
        DesiredQuotes {
            bid: Some(OrderIntent {
                side: Side::Buy,
                px: 10_000,
                qty_units: 10,
                post_only: true,
                reduce_only: false,
            }),
            ask: None,
            quote_seq: 1,
            model_revision: 1,
            generated_ns: 0,
            source_recv_ns: 0,
            source_exchange_ms: 0,
            reason: QuoteReason::Market,
            mid: 100.01,
            q_exact: 0.0,
            q_rounded: 0,
            tau_remaining: 150.0,
        }
    }

    #[tokio::test]
    async fn latency_queue_and_partial_fill_are_causal() {
        let mut config = DryRunConfig::default();
        config.decision_latency_ms = 100;
        config.acknowledgement_latency_ms = 100;
        config.cancel_latency_ms = 50;
        config.queue_decay_per_second = 0.0;
        let mut backend = DryRunBackend::new(
            instrument(),
            config,
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 5,
            ask_px: 10_002,
            ask_sz: 5,
            exchange_ms: 0,
            recv_ns: 0,
        });
        backend.reconcile(bid_quotes(), 1_000).await.unwrap();
        let early = MarketEvent::Trade(TradePrint {
            aggressor: AggressorSide::Sell,
            px: 10_000,
            qty_units: 20,
            exchange_ms: 1_100,
            recv_ns: 0,
            trade_id: 1,
        });
        assert!(backend.on_market_event(&early).await.unwrap().is_empty());
        let first = MarketEvent::Trade(TradePrint {
            exchange_ms: 1_200,
            trade_id: 2,
            ..match early {
                MarketEvent::Trade(value) => value,
                _ => unreachable!(),
            }
        });
        let fills = backend.on_market_event(&first).await.unwrap();
        assert_eq!(fills.len(), 1);
        assert_eq!(backend.account.inventory_units, 10);
    }

    #[test]
    fn deterministic_latency_schedule_includes_the_measured_tail() {
        let backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        assert_eq!(backend.simulated_latency_ms(100, 18_000), 100);
        assert_eq!(backend.simulated_latency_ms(100, 19_000), 235);
        assert_eq!(backend.simulated_latency_ms(100, 20_000), 100);
    }

    fn flatten_backend(after_ms: u64) -> DryRunBackend {
        let mut config = DryRunConfig::default();
        config.flatten_after_ms = after_ms;
        config.promotion_flatten_slippage_bps = 0.0;
        config.promotion_flatten_fee_rate = 0.0;
        DryRunBackend::new(
            instrument(),
            config,
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn the_flatten_charges_its_own_knobs_not_the_promotion_ones() {
        // This is the regression that matters most. Reusing promotion_flatten_*
        // charged 25 bps a side instead of 2.5 and flipped the policy's SIGN --
        // measured at -5.1 USDC/day against +13.4 on the same fills. Every other
        // flatten test asserts counts, so without this the sign is unpinned.
        let mut config = DryRunConfig::default();
        config.flatten_after_ms = 1;
        config.flatten_slippage_bps = 2.5;
        config.flatten_fee_rate = 0.001;
        // Deliberately hostile: if the flatten ever reads these again, the
        // numbers below move by an order of magnitude and this test fails.
        config.promotion_flatten_slippage_bps = 250.0;
        config.promotion_flatten_fee_rate = 0.1;
        let mut backend = DryRunBackend::new(
            instrument(),
            config,
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 50,
            ask_px: 10_002,
            ask_sz: 50,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let cash_before = backend.account.cash_usdc;
        backend.record_lot(0, 10);
        assert_eq!(backend.flatten_stale_lots(1_000).len(), 1);

        // Sold 10 units into a bid of 10_000 units-of-price, minus 2.5 bps of
        // slippage, minus a 10 bps fee on the notional. The exit price is
        // tick-ROUNDED on the way into the Fill, so the cash reflects the
        // rounded price rather than the raw one -- assert against the same path
        // the code takes, or this pins the rounding instead of the cost.
        let touch = backend.instrument.price_from_units(10_000);
        let raw_exit = touch * (1.0 - 2.5 / 10_000.0);
        let exit = backend
            .instrument
            .price_from_units(backend.instrument.price_to_units(raw_exit).unwrap());
        assert!(exit < touch, "the slippage must move the exit against us");
        let qty = backend.instrument.size_from_units(10);
        let expected = cash_before + qty * exit - qty * exit * 0.001;
        assert!(
            (backend.account.cash_usdc - expected).abs() < 1e-9,
            "cash {} != expected {}",
            backend.account.cash_usdc,
            expected
        );
    }

    #[tokio::test]
    async fn a_nan_flatten_cost_is_refused_at_construction() {
        let mut config = DryRunConfig::default();
        config.flatten_slippage_bps = f64::NAN;
        assert!(DryRunBackend::new(
            instrument(),
            config,
            QuotingConfig::default(),
            RiskConfig::default()
        )
        .is_err());
    }

    #[tokio::test]
    async fn a_post_only_order_that_arrives_crossing_is_rejected_not_rested() {
        // The market moves during the latency window. The live path sends ALO
        // and the venue answers `badAloPxRejected`; resting it instead let the
        // order fill THROUGH the touch at a price no venue would have given.
        let mut backend = flatten_backend(0);
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 5,
            ask_px: 10_002,
            ask_sz: 5,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let mut quote = bid_quotes();
        // A bid at the ask: marketable, so ALO refuses it.
        quote.bid.as_mut().unwrap().px = 10_002;
        backend.reconcile(quote, 1_000).await.unwrap();
        backend.activate_orders(5_000);
        assert_eq!(backend.diagnostics.post_only_rejects, 1);
        assert!(backend.orders.is_empty());
    }

    #[tokio::test]
    async fn a_passive_post_only_order_still_rests() {
        let mut backend = flatten_backend(0);
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 5,
            ask_px: 10_002,
            ask_sz: 5,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let mut quote = bid_quotes();
        quote.bid.as_mut().unwrap().px = 9_999;
        backend.reconcile(quote, 1_000).await.unwrap();
        backend.activate_orders(5_000);
        assert_eq!(backend.diagnostics.post_only_rejects, 0);
        assert_eq!(backend.orders.len(), 1);
    }

    #[tokio::test]
    async fn the_flatten_deadline_includes_the_round_trip() {
        // Without this the deadline was "next event", a ~130 ms exit that is
        // unreachable at a p50 281 ms RTT -- the grid would test a rung that
        // does not exist.
        let mut backend = flatten_backend(250);
        backend.record_lot(1_000, 10);
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 50,
            ask_px: 10_002,
            ask_sz: 50,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let rtt = DryRunConfig::default().decision_latency_ms
            + DryRunConfig::default().acknowledgement_latency_ms;
        assert!(backend.flatten_stale_lots(1_000 + 250 + rtt - 1).is_empty());
        assert_eq!(backend.flatten_stale_lots(1_000 + 250 + rtt).len(), 1);
    }

    #[tokio::test]
    async fn restored_inventory_is_seeded_as_a_lot_so_it_can_be_flattened() {
        // An empty deque would make the policy self-destruct: every reducing
        // maker fill would open a NEW opposite lot, get crossed back out, and
        // re-establish the position while paying taker each time.
        let mut backend = flatten_backend(250);
        let mut account = backend.account;
        account.inventory_units = 40;
        backend
            .restore_from_snapshot(account, DryRunDiagnostics::default(), 10, None, 0.0)
            .unwrap();
        assert_eq!(backend.open_lots.len(), 1);
        assert_eq!(backend.open_lots.front().unwrap().1, 40);
        // Aged from 0, so it is past any deadline immediately.
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 50,
            ask_px: 10_002,
            ask_sz: 50,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        assert_eq!(backend.flatten_stale_lots(1_000).len(), 1);
        assert!(backend.open_lots.is_empty());
    }

    #[tokio::test]
    async fn a_gap_flatten_moves_the_position_into_cash_without_moving_equity() {
        // The whole point: closing at the checkpoint's own mark invents no P&L.
        // If equity moved here, the resume would be manufacturing exactly the
        // unearned gain the carry window exists to prevent.
        let mut backend = flatten_backend(250);
        let mut account = backend.account;
        account.inventory_units = 40;
        account.average_entry_px = 1.00;
        account.cash_usdc = 100.0;
        // 40 units marked above the entry: an unrealized gain of 20.
        account.equity_usdc = 160.0;
        account.realized_pnl_usdc = 5.0;
        backend
            .restore_from_snapshot(account, DryRunDiagnostics::default(), 10, None, 0.0)
            .unwrap();

        let equity_before = backend.account.equity_usdc;
        let position_value = backend.account.equity_usdc - backend.account.cash_usdc;
        let cost_basis = backend
            .instrument
            .size_from_units(backend.account.inventory_units)
            * backend.account.average_entry_px;

        let closed = backend.flatten_carried_position().unwrap();

        assert!((closed - (position_value - cost_basis)).abs() < 1e-9);
        assert!((backend.account.equity_usdc - equity_before).abs() < 1e-9);
        assert!((backend.account.cash_usdc - equity_before).abs() < 1e-9);
        assert!((backend.account.realized_pnl_usdc - (5.0 + closed)).abs() < 1e-9);
        assert_eq!(backend.account.inventory_units, 0);
        assert_eq!(backend.account.position_notional_usdc, 0.0);
        // The seeded carry lot describes a position that no longer exists;
        // leaving it would have the flatten policy cross out thin air.
        assert!(backend.open_lots.is_empty());
    }

    #[tokio::test]
    async fn a_gap_flatten_on_a_flat_variant_reports_nothing() {
        let mut backend = flatten_backend(250);
        assert_eq!(backend.account.inventory_units, 0);
        assert!(backend.flatten_carried_position().is_none());
    }

    #[tokio::test]
    async fn a_lot_held_past_the_deadline_is_crossed_out() {
        let mut backend = flatten_backend(250);
        backend.record_lot(1_000, 10);
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 50,
            ask_px: 10_002,
            ask_sz: 50,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let rtt = DryRunConfig::default().decision_latency_ms
            + DryRunConfig::default().acknowledgement_latency_ms;
        // Before the deadline nothing happens.
        assert!(backend.flatten_stale_lots(1_200 + rtt).is_empty());
        assert_eq!(backend.open_lots.len(), 1);
        // After it, the lot is sold into the bid and the book is flat.
        let events = backend.flatten_stale_lots(1_250 + rtt);
        assert_eq!(events.len(), 1);
        assert!(backend.open_lots.is_empty());
        assert_eq!(backend.account.inventory_units, -10);
        assert_eq!(backend.diagnostics.flatten_events, 1);
    }

    #[tokio::test]
    async fn the_deadline_ages_the_oldest_lot_not_the_net_position() {
        // A position topped up repeatedly must still exit on the FIRST lot's
        // clock, or a busy market defers the exit forever.
        let mut backend = flatten_backend(250);
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 50,
            ask_px: 10_002,
            ask_sz: 50,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        backend.record_lot(1_000, 5);
        backend.record_lot(1_200, 5);
        let rtt = DryRunConfig::default().decision_latency_ms
            + DryRunConfig::default().acknowledgement_latency_ms;
        let events = backend.flatten_stale_lots(1_250 + rtt);
        assert_eq!(events.len(), 1, "only the aged lot leaves");
        assert_eq!(backend.open_lots.len(), 1);
        assert_eq!(backend.open_lots.front().unwrap().0, 1_200);
    }

    #[tokio::test]
    async fn opposing_fills_close_lots_fifo_so_nothing_is_left_to_flatten() {
        let mut backend = flatten_backend(250);
        backend.record_lot(1_000, 10);
        backend.record_lot(1_100, -4);
        assert_eq!(backend.open_lots.len(), 1);
        assert_eq!(backend.open_lots.front().unwrap().1, 6);
        // A flip closes the remainder and opens the other way at the new time.
        backend.record_lot(1_200, -9);
        assert_eq!(backend.open_lots.len(), 1);
        assert_eq!(*backend.open_lots.front().unwrap(), (1_200, -3));
    }

    #[tokio::test]
    async fn a_zero_deadline_disables_the_flatten() {
        let mut backend = flatten_backend(0);
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 50,
            ask_px: 10_002,
            ask_sz: 50,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        backend.record_lot(1_000, 10);
        assert!(backend.flatten_stale_lots(9_999_999).is_empty());
        assert_eq!(backend.open_lots.len(), 1);
    }

    #[tokio::test]
    async fn without_a_touch_the_lot_waits_rather_than_inventing_a_price() {
        let mut backend = flatten_backend(250);
        backend.record_lot(1_000, 10);
        assert!(backend.flatten_stale_lots(5_000).is_empty());
        assert_eq!(backend.open_lots.len(), 1);
    }

    #[tokio::test]
    async fn a_print_beyond_an_off_book_quote_fills_it_despite_an_unknown_queue() {
        // The counterpart to the test below, and the one that was missing. A
        // print STRICTLY BEYOND our price means the venue cleared everything
        // resting at our price and carried on, so queue position cannot save us
        // -- we were filled. Excluding these cost `wide60` all but 4 of ~100
        // fills over 15 h of live tape, and made the whole dry-run leaderboard
        // a measurement of exact-level coincidence rather than fill economics.
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 5,
            ask_px: 10_002,
            ask_sz: 5,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let mut quote = bid_quotes();
        quote.bid.as_mut().unwrap().px = 9_990;
        backend.reconcile(quote, 1_000).await.unwrap();
        // Sweeps through 9_990 down to 9_985: everything at 9_990 is gone.
        let trade = TradePrint {
            aggressor: AggressorSide::Sell,
            px: 9_985,
            qty_units: 20,
            exchange_ms: 2_000,
            recv_ns: 0,
            trade_id: 1,
        };
        assert!(!backend
            .on_market_event(&MarketEvent::Trade(trade))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn an_off_book_quote_cannot_fill_until_its_queue_becomes_visible() {
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        backend.latest_bbo = Some(Bbo {
            bid_px: 10_000,
            bid_sz: 5,
            ask_px: 10_002,
            ask_sz: 5,
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        let mut quote = bid_quotes();
        quote.bid.as_mut().unwrap().px = 9_990;
        backend.reconcile(quote, 1_000).await.unwrap();
        let trade = TradePrint {
            aggressor: AggressorSide::Sell,
            px: 9_990,
            qty_units: 20,
            exchange_ms: 2_000,
            recv_ns: 0,
            trade_id: 1,
        };
        assert!(backend
            .on_market_event(&MarketEvent::Trade(trade))
            .await
            .unwrap()
            .is_empty());
        backend
            .on_market_event(&MarketEvent::Book(BookSnapshot {
                bids: vec![crate::types::BookLevel {
                    px: 9_990,
                    qty_units: 3,
                }],
                asks: vec![],
                exchange_ms: 2_001,
                recv_ns: 0,
            }))
            .await
            .unwrap();
        let fills = backend
            .on_market_event(&MarketEvent::Trade(TradePrint {
                exchange_ms: 2_002,
                qty_units: 4,
                trade_id: 2,
                ..trade
            }))
            .await
            .unwrap();
        assert_eq!(fills.len(), 1);
        let ExecutionEvent::Fill(fill) = fills[0] else {
            panic!("expected fill");
        };
        assert_eq!(fill.qty_units, 1);
        assert_eq!(backend.diagnostics.unknown_queue_activations, 1);
    }

    #[tokio::test]
    async fn overlapping_reduce_only_orders_share_one_inventory_balance() {
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        backend.account.inventory_units = 5;
        for id in 1..=2 {
            backend.orders.push(VirtualOrder {
                id,
                intent: OrderIntent {
                    side: Side::Sell,
                    px: 10_002,
                    qty_units: 5,
                    post_only: true,
                    reduce_only: true,
                },
                active_ms: 1_000,
                cancel_effective_ms: None,
                remaining_units: 5,
                queue_ahead_units: 0.0,
                initial_queue_units: 0.0,
                last_queue_update_ms: 1_000,
                activated: true,
                queue_known: true,
            });
        }
        let fills = backend
            .on_market_event(&MarketEvent::Trade(TradePrint {
                aggressor: AggressorSide::Buy,
                px: 10_002,
                qty_units: 20,
                exchange_ms: 1_001,
                recv_ns: 0,
                trade_id: 1,
            }))
            .await
            .unwrap();
        assert_eq!(fills.len(), 1);
        assert_eq!(backend.account.inventory_units, 0);
    }

    #[tokio::test]
    async fn invalidation_withdraws_orders_and_blocks_quotes() {
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        backend.invalidate("event loss");
        assert!(backend.reconcile(bid_quotes(), 0).await.is_err());
        assert!(!backend.scientifically_valid());
    }

    /// Drive `count` losing closing fills: buy a unit at 100, sell it back at
    /// 99, so each round trip reduces the position at a loss.
    fn take_losing_round_trips(backend: &mut DryRunBackend, count: u32) {
        for i in 0..count {
            backend.apply_fill(Fill {
                side: Side::Buy,
                px: 10_000,
                qty_units: 1,
                fee_usdc: 0.0,
                exchange_ms: u64::from(i),
                virtual_order_id: 0,
                maker: true,
            });
            backend.apply_fill(Fill {
                side: Side::Sell,
                px: 9_900,
                qty_units: 1,
                fee_usdc: 0.0,
                exchange_ms: u64::from(i),
                virtual_order_id: 0,
                maker: true,
            });
        }
    }

    #[test]
    fn the_latched_loss_breaker_marks_the_variant_invalid() {
        // The breaker never releases -- it resets only on a winning closing
        // fill, which a variant that has stopped quoting can never take. So a
        // row that reached the cap is finished, and must not keep reporting
        // itself as a result.
        let risk = RiskConfig {
            max_consecutive_losses: 3,
            ..RiskConfig::default()
        };
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            risk,
        )
        .unwrap();

        take_losing_round_trips(&mut backend, 2);
        assert_eq!(backend.account.consecutive_losses, 2);
        assert!(
            backend.scientifically_valid(),
            "one short of the cap still measures something"
        );

        take_losing_round_trips(&mut backend, 1);
        assert_eq!(backend.account.consecutive_losses, 3);
        assert!(!backend.scientifically_valid());
        assert_eq!(
            backend.diagnostics.invalid_reason.as_deref(),
            Some("consecutive-loss breaker latched")
        );
    }

    #[tokio::test]
    async fn a_liquidation_breach_invalidates_and_clears_the_deferred_target() {
        // Regression: this path used to hand-roll `invalidate` and forget
        // `deferred_desired`, leaving a queued target behind on the one code
        // path that means the account is in trouble.
        let risk = RiskConfig {
            min_liquidation_buffer_usdc: 1.0e9,
            ..RiskConfig::default()
        };
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            risk,
        )
        .unwrap();
        backend.deferred_desired = Some(bid_quotes());

        backend.mark_account(10_000);

        assert!(!backend.scientifically_valid());
        assert_eq!(backend.diagnostics.liquidation_breach_events, 1);
        assert_eq!(
            backend.diagnostics.invalid_reason.as_deref(),
            Some("liquidation buffer breached")
        );
        assert!(backend.deferred_desired.is_none());
        assert!(backend.reconcile(bid_quotes(), 0).await.is_err());
    }

    #[test]
    fn persisted_state_requires_matching_config_and_never_restores_orders() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let mut source = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        source.account.inventory_units = 7;
        source
            .save_account_state(&path, "config-a", "model-a", 5, 10)
            .unwrap();
        let mut restored = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        assert!(!restored.restore_account_state(&path, "wrong", 5).unwrap());
        assert!(
            !restored
                .restore_account_state(&path, "config-a", 4)
                .unwrap(),
            "a state from another estimator schema is not a safe basis"
        );
        assert!(restored
            .restore_account_state(&path, "config-a", 5)
            .unwrap());
        assert_eq!(restored.account.inventory_units, 7);
        assert_eq!(restored.restored_inventory_unit(), Some(10));
        assert_eq!(restored.working_order_count(), 0);
    }

    #[test]
    fn stale_persisted_state_is_not_restored() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let source = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        source
            .save_account_state(&path, "config-a", "model-a", 5, 10)
            .unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["saved_at_ms"] = serde_json::json!(1);
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let mut restored = DryRunBackend::new(
            instrument(),
            DryRunConfig::default(),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        assert!(!restored
            .restore_account_state(&path, "config-a", 5)
            .unwrap());
    }

    #[tokio::test]
    async fn partial_fill_cash_fee_and_mark_to_market_are_hand_checkable() {
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig {
                decision_latency_ms: 0,
                acknowledgement_latency_ms: 0,
                queue_decay_per_second: 0.0,
                ..DryRunConfig::default()
            },
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        let mut quote = bid_quotes();
        quote.bid.as_mut().unwrap().px = 9_999;
        backend.latest_book = Some(BookSnapshot {
            bids: vec![crate::types::BookLevel {
                px: 9_999,
                qty_units: 0,
            }],
            asks: vec![],
            exchange_ms: 1_000,
            recv_ns: 0,
        });
        backend.reconcile(quote, 1_000).await.unwrap();
        let event = MarketEvent::Trade(TradePrint {
            aggressor: AggressorSide::Sell,
            px: 9_999,
            qty_units: 4,
            exchange_ms: 1_000,
            recv_ns: 0,
            trade_id: 1,
        });
        let fills = backend.on_market_event(&event).await.unwrap();
        assert_eq!(fills.len(), 1);
        let ExecutionEvent::Fill(fill) = &fills[0] else {
            panic!("expected fill event");
        };
        assert_eq!(fill.qty_units, 4);
        assert_eq!(backend.account.inventory_units, 4);
        assert_eq!(backend.orders[0].remaining_units, 6);
        let fee = 4.0 * 99.99 * 0.00015;
        assert!((backend.account.cash_usdc - (1_000.0 - 4.0 * 99.99 - fee)).abs() < 1e-10);
        backend.process_event(&MarketEvent::Bbo(Bbo {
            bid_px: 9_999,
            bid_sz: 1,
            ask_px: 10_001,
            ask_sz: 1,
            exchange_ms: 1_001,
            recv_ns: 0,
        }));
        assert!((backend.account.equity_usdc - (1_000.0 + 0.04 - fee)).abs() < 1e-10);
        backend.process_event(&MarketEvent::Bbo(Bbo {
            bid_px: 10_009,
            bid_sz: 1,
            ask_px: 10_011,
            ask_sz: 1,
            exchange_ms: 1_100,
            recv_ns: 0,
        }));
        assert_eq!(backend.diagnostics.markout_samples.get(&100), Some(&1));
        assert!(backend.diagnostics.markout_usdc[&100] > 0.0);
    }

    #[tokio::test]
    async fn replacement_orders_overlap_only_for_cancel_latency() {
        let mut backend = DryRunBackend::new(
            instrument(),
            DryRunConfig {
                decision_latency_ms: 0,
                acknowledgement_latency_ms: 0,
                cancel_latency_ms: 100,
                ..DryRunConfig::default()
            },
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        let first = bid_quotes();
        backend.reconcile(first, 1_000).await.unwrap();
        let mut replacement = first;
        replacement.quote_seq = 2;
        replacement.bid.as_mut().unwrap().px -= 1;
        // Past the 100 ms requote cooldown: this test is about cancel-latency
        // overlap, not the cooldown, so the replacement must actually be sent.
        backend.reconcile(replacement, 1_101).await.unwrap();
        assert_eq!(backend.working_order_count(), 2);
        backend.process_event(&MarketEvent::Bbo(Bbo {
            bid_px: 9_999,
            bid_sz: 1,
            ask_px: 10_002,
            ask_sz: 1,
            // Past the replacement's 100 ms cancel latency.
            exchange_ms: 1_202,
            recv_ns: 0,
        }));
        assert_eq!(backend.working_order_count(), 1);
    }

    fn bbo_at(exchange_ms: u64) -> MarketEvent {
        MarketEvent::Bbo(Bbo {
            bid_px: 10_000,
            bid_sz: 1,
            ask_px: 10_002,
            ask_sz: 1,
            exchange_ms,
            recv_ns: 0,
        })
    }

    #[tokio::test]
    async fn a_routine_requote_inside_the_cooldown_is_deferred_then_applied() {
        let mut backend = backend_with_zero_latency();
        backend.reconcile(bid_quotes(), 1_000).await.unwrap();
        assert_eq!(backend.working_order_count(), 1);
        let mut replacement = bid_quotes();
        replacement.quote_seq = 2;
        replacement.bid.as_mut().unwrap().px -= 5;
        // 50 ms into the 100 ms window: held back, book unchanged.
        backend.reconcile(replacement, 1_050).await.unwrap();
        assert_eq!(backend.orders[0].intent.px, bid_quotes().bid.unwrap().px);
        // The deferred target is applied on the first event past the window,
        // not dropped.
        backend.process_event(&bbo_at(1_150));
        assert!(
            backend
                .orders
                .iter()
                .any(|order| order.intent.px == replacement.bid.unwrap().px),
            "the deferred replacement must be applied, not lost"
        );
    }

    #[tokio::test]
    async fn a_withdrawal_is_never_deferred() {
        // The toxic-flow guard and every risk limit withdraw by publishing an
        // empty target; if the cooldown could hold that back, the guard would
        // keep quoting into exactly the conditions it exists to avoid.
        let mut backend = backend_with_zero_latency();
        backend.reconcile(bid_quotes(), 1_000).await.unwrap();
        assert_eq!(backend.working_order_count(), 1);
        let withdraw = DesiredQuotes::empty(QuoteReason::ToxicFlow, 2, 0);
        backend.reconcile(withdraw, 1_010).await.unwrap();
        assert!(
            backend
                .orders
                .iter()
                .all(|o| o.cancel_effective_ms.is_some()),
            "an empty target must cancel immediately despite the cooldown"
        );
    }

    #[tokio::test]
    async fn a_fill_that_moved_inventory_bypasses_the_cooldown() {
        let mut backend = backend_with_zero_latency();
        backend.reconcile(bid_quotes(), 1_000).await.unwrap();
        // Pretend a fill moved inventory since the last quote action.
        backend.account.inventory_units += 1;
        let mut after_fill = bid_quotes();
        after_fill.quote_seq = 2;
        after_fill.reason = QuoteReason::Fill;
        after_fill.bid.as_mut().unwrap().px -= 5;
        backend.reconcile(after_fill, 1_010).await.unwrap();
        assert!(
            backend
                .orders
                .iter()
                .any(|order| order.intent.px == after_fill.bid.unwrap().px),
            "a fill that changed inventory must requote immediately"
        );
    }

    #[tokio::test]
    async fn a_held_quote_does_not_restart_the_cooldown() {
        // Only a submitted action restarts the window. If a held quote
        // restarted it, a busy market would defer forever.
        let mut backend = backend_with_zero_latency();
        backend.reconcile(bid_quotes(), 1_000).await.unwrap();
        // Same target inside the price hold window: nothing is submitted.
        let mut nudge = bid_quotes();
        nudge.quote_seq = 2;
        backend.reconcile(nudge, 1_200).await.unwrap();
        assert_eq!(
            backend.last_quote_action_ms, 1_000,
            "a held quote must not stamp a new action time"
        );
    }

    fn backend_with_zero_latency() -> DryRunBackend {
        DryRunBackend::new(
            instrument(),
            DryRunConfig {
                decision_latency_ms: 0,
                acknowledgement_latency_ms: 0,
                cancel_latency_ms: 0,
                ..DryRunConfig::default()
            },
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap()
    }
}
