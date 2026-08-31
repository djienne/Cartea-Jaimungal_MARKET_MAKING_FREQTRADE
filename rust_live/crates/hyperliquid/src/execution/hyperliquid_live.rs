use super::traits::{AccountStateProvider, ExecutionBackend};
use crate::config::{AppConfig, LiveMode, QuotingConfig, RiskConfig};
use crate::hyperliquid::account_types::{
    fill_key, funding_key, parse_clearinghouse_message, parse_open_orders_message,
    parse_order_updates, parse_user_fills, parse_user_fundings, LiveAccountSnapshot,
};
use crate::hyperliquid::auth::HyperliquidCredentials;
use crate::hyperliquid::exchange::{ActionOutcome, HyperliquidExchangeClient, OpenOrder, UserFill};
use crate::hyperliquid::live_state::{
    LiveOrderStatus, LiveStateStore, PersistedLiveOrder, RiskScalars, SessionIntent, TimedKey,
};
use crate::hyperliquid::session::{
    spawn_session, AccountChannel, HyperliquidSessionHandle, SessionEvent, SessionSpawnArgs,
};
use crate::hyperliquid::signing::{
    is_bot_cloid, make_cloid, parse_fixed, LiveOrderRequest, TimeInForce,
};
use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::lockfree::AtomicBbo;
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
use tracing::warn;

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
    pub rate_limit_cooldowns: u64,
    pub actual_maker_fee_rate: f64,
    pub actual_taker_fee_rate: f64,
    pub campaign_turnover_usdc: f64,
    pub campaign_realized_pnl_usdc: f64,
    pub maximum_working_gross_usdc: f64,
    pub maximum_directional_notional_usdc: f64,
    pub address_requests_used: u64,
    pub address_requests_cap: u64,
    pub cancel_requests_used: u64,
    pub cumulative_volume_usdc: f64,
    pub quota_refreshes: u64,
    pub placement_throttles: u64,
    pub consecutive_rejections: u32,
    pub next_placement_allowed_ms: u64,
}

pub struct LiveBootstrap {
    pub backend: HyperliquidLiveBackend,
    pub session_events: mpsc::Receiver<SessionEvent>,
    pub session_task: tokio::task::JoinHandle<()>,
}

/// After any rate-limit refusal, place no new orders for this long. The venue
/// and local limiters both work on 60s rolling windows, so a refusal means the
/// window is saturated; retrying immediately just re-triggers it. Cancels are
/// never suppressed — reducing exposure must always be possible.
const RATE_LIMIT_COOLDOWN_MS: u64 = 30_000;

/// Whether a refusal reason describes a saturated rate/quota budget rather
/// than an ordinary rejection. Matches the limiter messages raised by the
/// session actor, the REST client, and the address-action reserve check.
fn is_rate_limit_reason(reason: &str) -> bool {
    let reason = reason.to_ascii_lowercase();
    [
        "rate limit",
        "budget",
        "reserve is below",
        "in-flight posts",
    ]
    .iter()
    .any(|needle| reason.contains(needle))
}

fn parse_user_rate_limit(value: &serde_json::Value) -> Result<(f64, u64, u64)> {
    let parse_u64 = |name: &str| {
        value
            .get(name)
            .and_then(|field| {
                field
                    .as_u64()
                    .or_else(|| field.as_str().and_then(|text| text.parse().ok()))
            })
            .with_context(|| format!("userRateLimit.{name} is missing or invalid"))
    };
    let cumulative_volume_usdc = value
        .get("cumVlm")
        .and_then(|field| {
            field
                .as_f64()
                .or_else(|| field.as_str().and_then(|text| text.parse().ok()))
        })
        .context("userRateLimit.cumVlm is missing or invalid")?;
    let used = parse_u64("nRequestsUsed")?;
    let cap = parse_u64("nRequestsCap")?;
    if !cumulative_volume_usdc.is_finite() || cumulative_volume_usdc < 0.0 || cap == 0 {
        bail!("userRateLimit returned impossible cumulative volume or zero cap");
    }
    Ok((cumulative_volume_usdc, used, cap))
}

/// Replay horizon for fill/funding dedup keys, in exchange time. Far beyond any
/// WebSocket snapshot replay or REST `userFills` lookback, so a key older than
/// this can never be asked about again.
const EVENT_RETENTION_MS: u64 = 24 * 60 * 60 * 1_000;
/// Keep terminal orders at least this long so late acknowledgements and status
/// rows can still be attributed before the entry is dropped.
const MINIMUM_TERMINAL_ORDER_RETENTION_MS: u64 = 10 * 60 * 1_000;

/// Scalars an authoritative reconcile needs from the durable state, projected
/// under the lock so the growing collections are never cloned.
struct PersistedView {
    metadata_matches: bool,
    any_non_terminal: bool,
    cumulative_realized_pnl_usdc: f64,
    cumulative_fees_usdc: f64,
    cumulative_funding_usdc: f64,
}

/// Outcome of admitting one fill into durable accounting.
enum FillAdmission {
    /// Older than the replay horizon; its dedup key may already be pruned, so
    /// it must not be treated as new — and it is not a duplicate either.
    BeforeCheckpoint,
    Duplicate,
    Fresh {
        cumulative_fees_usdc: f64,
        cumulative_realized_pnl_usdc: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RejectionRecovery {
    Idle,
    AwaitingReconcile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupReconciliation {
    CleanSnapshot,
    Required,
    Complete,
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
    /// The hot path's shared BBO slot, when attached. Quote validation reads
    /// this so decision and validation see one snapshot; `latest_bbo` (fed by
    /// the drained event ring) is the fallback and can lag under backlog.
    market_bbo: Option<Arc<AtomicBbo>>,
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
    /// Exchange-time watermark up to which `account.inventory_units` already
    /// reflects trading. Compared against `fill.time`, so it must only ever be
    /// fed venue timestamps — a local clock here silently drops legitimate
    /// fills whenever it runs ahead of the exchange.
    last_inventory_update_exchange_ms: u64,
    last_quote_action_ms: u64,
    inventory_at_last_quote_action: i64,
    deferred_desired: Option<DesiredQuotes>,
    /// No new orders are placed before this instant. Set on any rate-limit
    /// refusal so the bot stays out of a saturated window instead of bouncing
    /// straight back into the limiter once health is restored.
    rate_limited_until_ms: u64,
    last_user_rate_limit_refresh_ms: u64,
    next_placement_allowed_ms: u64,
    consecutive_rejections: u32,
    rejection_recovery: RejectionRecovery,
    startup_reconciliation: StartupReconciliation,
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
        // The current `meta` response no longer publishes `onlyIsolated` or
        // `marginMode` for CASHCAT. Do not invent defaults as a live gate: the
        // account-specific activeAssetData below is authoritative and must
        // still confirm isolated leverage before any action transport is used.
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
            config.live.safety_rest_weight_reserve,
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
        ensure_no_foreign_positions(&clearinghouse, &instrument)?;
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
        let (cumulative_volume_usdc, requests_used, requests_cap) =
            parse_user_rate_limit(&user_rate_limit)?;
        let state = Arc::new(LiveStateStore::open(
            &config.live.state_path,
            &instrument.symbol,
            credentials.account(),
            &credentials.agent_address(),
            &config.fingerprint()?,
            &instrument.metadata_fingerprint,
            if allow_untracked_position {
                SessionIntent::ReduceOnly
            } else {
                SessionIntent::Quote {
                    venue_is_flat: account.inventory_units == 0 && account.open_orders.is_empty(),
                }
            },
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
            ping_interval: Duration::from_millis(config.runtime.ws_ping_interval_ms),
            idle_timeout: Duration::from_millis(config.runtime.ws_idle_timeout_ms),
            connect_timeout: Duration::from_millis(config.runtime.ws_connect_timeout_ms),
        });
        let maker_fee_rate = account.maker_fee_rate;
        let taker_fee_rate = account.taker_fee_rate;
        let initial_inventory_units = account.inventory_units;
        let startup_reconciliation = if account.inventory_units == 0
            && account.open_orders.is_empty()
            && state.with_state(|persisted| {
                persisted
                    .orders
                    .values()
                    .all(|order| order.status.terminal())
            })? {
            StartupReconciliation::CleanSnapshot
        } else {
            StartupReconciliation::Required
        };
        let initial_reconcile_ms = if startup_reconciliation == StartupReconciliation::CleanSnapshot
        {
            unix_ms()
        } else {
            0
        };
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
                address_requests_used: requests_used,
                address_requests_cap: requests_cap,
                cumulative_volume_usdc,
                ..LiveExecutionDiagnostics::default()
            },
            latest_bbo: None,
            market_bbo: None,
            session_started_at_ms: unix_ms(),
            pending_events: Vec::new(),
            last_reconcile_ms: initial_reconcile_ms,
            deadman_armed: false,
            last_deadman_refresh_ms: 0,
            clock,
            market_stale_ms: config.runtime.market_stale_ms,
            q_max: config.model.q_max,
            latency,
            last_fill_received_ns: None,
            allow_untracked_position,
            reconcile_requested: false,
            // Exchange time, never local: 0 accepts every admitted fill. The
            // checkpoint and dedup-key gates already keep stale replays from
            // reaching inventory adjustment at all.
            last_inventory_update_exchange_ms: 0,
            last_quote_action_ms: 0,
            inventory_at_last_quote_action: initial_inventory_units,
            deferred_desired: None,
            rate_limited_until_ms: 0,
            last_user_rate_limit_refresh_ms: unix_ms(),
            next_placement_allowed_ms: 0,
            consecutive_rejections: 0,
            rejection_recovery: RejectionRecovery::Idle,
            startup_reconciliation,
        };
        backend.initialize_campaign()?;
        Ok(LiveBootstrap {
            backend,
            session_events: event_rx,
            session_task,
        })
    }

    /// Attach the hot path's shared BBO slot so order validation prices from
    /// the same snapshot the quote decision used.
    pub fn attach_market_bbo(&mut self, slot: Arc<AtomicBbo>) {
        self.market_bbo = Some(slot);
    }

    /// Start the no-new-orders cooldown if `reason` describes a saturated
    /// budget. Returns whether a cooldown is now in effect.
    pub fn note_rate_limit_if_applicable(&mut self, now_ms: u64, reason: &str) -> bool {
        if !is_rate_limit_reason(reason) {
            return false;
        }
        let until = now_ms.saturating_add(RATE_LIMIT_COOLDOWN_MS);
        if until > self.rate_limited_until_ms {
            self.rate_limited_until_ms = until;
            self.diagnostics.rate_limit_cooldowns += 1;
            warn!(
                reason,
                cooldown_ms = RATE_LIMIT_COOLDOWN_MS,
                "rate limit hit; suspending new orders (cancels still allowed)"
            );
        }
        true
    }

    /// True while the post-rate-limit cooldown suppresses new orders.
    pub const fn rate_limited(&self, now_ms: u64) -> bool {
        now_ms < self.rate_limited_until_ms
    }

    fn safe_placement_allowance(&self) -> u64 {
        self.diagnostics
            .address_requests_cap
            .saturating_sub(self.diagnostics.address_requests_used)
            .saturating_sub(self.live.address_action_reserve)
            .saturating_sub(self.live.address_safety_action_reserve)
    }

    fn placement_quota_available(&mut self, now_ms: u64, action_count: u64) -> bool {
        let allowance = self.safe_placement_allowance();
        if action_count == 0 || allowance < action_count || now_ms < self.next_placement_allowed_ms
        {
            self.diagnostics.placement_throttles =
                self.diagnostics.placement_throttles.saturating_add(1);
            return false;
        }
        true
    }

    fn safety_placement_available(&self, action_count: u64) -> bool {
        action_count > 0
            && self
                .diagnostics
                .address_requests_cap
                .saturating_sub(self.diagnostics.address_requests_used)
                .saturating_sub(self.live.address_action_reserve)
                >= action_count
    }

    fn count_placement_actions(&mut self, now_ms: u64, action_count: u64) {
        let allowance = self.safe_placement_allowance().max(1);
        let pacing_ms = self
            .live
            .quota_horizon_seconds
            .saturating_mul(1_000)
            .checked_div(allowance.max(1))
            .unwrap_or(u64::MAX)
            .max(self.quoting.min_order_lifetime_ms);
        self.next_placement_allowed_ms =
            now_ms.saturating_add(pacing_ms.saturating_mul(action_count));
        self.diagnostics.next_placement_allowed_ms = self.next_placement_allowed_ms;
        self.count_address_actions(action_count);
    }

    fn count_address_actions(&mut self, count: u64) {
        self.diagnostics.address_requests_used =
            self.diagnostics.address_requests_used.saturating_add(count);
    }

    fn count_cancel_actions(&mut self, count: u64) {
        self.diagnostics.cancel_requests_used =
            self.diagnostics.cancel_requests_used.saturating_add(count);
    }

    async fn refresh_user_rate_limit(&mut self, now_ms: u64) -> Result<()> {
        let value = self.client.user_rate_limit().await?;
        let (volume, used, cap) = parse_user_rate_limit(&value)?;
        self.diagnostics.cumulative_volume_usdc = volume;
        // Never move the local estimate backwards between authoritative
        // refreshes: actions submitted concurrently with this info request may
        // not be reflected in its snapshot yet.
        self.diagnostics.address_requests_used = self.diagnostics.address_requests_used.max(used);
        self.diagnostics.address_requests_cap = cap;
        self.diagnostics.quota_refreshes = self.diagnostics.quota_refreshes.saturating_add(1);
        self.last_user_rate_limit_refresh_ms = now_ms;
        Ok(())
    }

    pub const fn diagnostics(&self) -> &LiveExecutionDiagnostics {
        &self.diagnostics
    }

    pub const fn operationally_healthy(&self) -> bool {
        self.diagnostics.operationally_healthy
    }

    pub fn health_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "generated_at_ms": unix_ms(),
            "operationally_healthy": self.diagnostics.operationally_healthy,
            "session_healthy": self.session.healthy(),
            "persistence_healthy": self.state.persistence_healthy(),
            "inventory_units": self.account.inventory_units,
            "open_orders": self.account.open_orders.len(),
            "address_requests_used": self.diagnostics.address_requests_used,
            "address_requests_cap": self.diagnostics.address_requests_cap,
            "cancel_requests_used": self.diagnostics.cancel_requests_used,
            "safe_placement_allowance": self.safe_placement_allowance(),
            "next_placement_allowed_ms": self.next_placement_allowed_ms,
            "consecutive_rejections": self.consecutive_rejections,
            "invalid_reason": self.diagnostics.invalid_reason,
        })
    }

    /// Pause trading without ending the session.
    ///
    /// Transient conditions — a full command queue, a refused action — must not
    /// unwind `run_live`, because that path abandons resting orders to the
    /// dead-man deadline. Quoting stops because `run_live` gates on
    /// `operationally_healthy`, and the next maintenance tick reconciles against
    /// the venue before anything resumes. This mirrors how a
    /// `SessionEvent::ActionRefused` is already handled.
    fn degrade(&mut self, context: &str, error: &anyhow::Error) {
        self.diagnostics.operationally_healthy = false;
        self.diagnostics.invalid_reason = Some(format!("{context}: {error}"));
        self.reconcile_requested = true;
        warn!(%error, context, "live action degraded; quotes paused until reconciled");
    }

    pub const fn deadman_armed(&self) -> bool {
        self.deadman_armed
    }

    /// Scalar risk inputs for the hot path, rolling the P&L day when the
    /// calendar has advanced.
    ///
    /// This runs on every execution event, so it deliberately avoids
    /// `load_required` — cloning the durable order and dedup collections here is
    /// what would make per-event cost grow with session length.
    pub fn risk_scalars(&self) -> Result<RiskScalars> {
        self.state.risk_scalars(unix_ms() / 86_400_000)
    }

    pub fn effective_quoting_config(&self) -> Result<QuotingConfig> {
        let usable = allocated_usable_equity(
            self.quoting.available_capital_usdc,
            self.account.account_value_usdc,
            self.risk.min_liquidation_buffer_usdc,
        );
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
        self.state.inventory_unit()
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
        let action_count = requests.len() as u64;
        let outcome = self.session.place_orders(quote_seq, requests).await?;
        self.count_address_actions(action_count);
        self.record_order_outcome(&outcome);
        Ok(outcome)
    }

    pub async fn cancel_all_bot_orders(&mut self) -> Result<()> {
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
            self.diagnostics.cancels_submitted += 1;
            self.count_cancel_actions(action_count);
            require_action_known(&outcome)?;
        }
        self.reconcile_safety_position().await?;
        if !self.account.open_orders.is_empty() {
            bail!("bot-order cancellation did not produce an empty venue order set");
        }
        Ok(())
    }

    pub fn enqueue_cancel_all_bot_orders(&mut self) -> Result<()> {
        let cloids = self.state.with_state(|state| {
            state
                .orders
                .values()
                .filter(|order| !order.status.terminal())
                .map(|order| order.cloid.clone())
                .collect::<Vec<String>>()
        })?;
        if cloids.is_empty() {
            return Ok(());
        }
        let action_count = cloids.len() as u64;
        self.session.enqueue_cancel_cloids(cloids)?;
        self.count_cancel_actions(action_count);
        Ok(())
    }

    pub async fn cancel_bot_oid(&mut self, oid: u64) -> Result<ActionOutcome> {
        let outcome = self.session.cancel_oids(vec![oid]).await?;
        self.diagnostics.cancels_submitted += 1;
        self.count_cancel_actions(1);
        Ok(outcome)
    }

    pub async fn cancel_bot_cloids(&mut self, cloids: Vec<String>) -> Result<ActionOutcome> {
        let action_count = cloids.len() as u64;
        let outcome = self.session.cancel_cloids(cloids).await?;
        self.diagnostics.cancels_submitted += 1;
        self.count_cancel_actions(action_count);
        Ok(outcome)
    }

    pub async fn schedule_deadman_at(&mut self, time: Option<u64>) -> Result<ActionOutcome> {
        let outcome = self.session.schedule_cancel(time).await?;
        self.count_address_actions(1);
        require_action_ok(&outcome)?;
        self.deadman_armed = time.is_some();
        self.last_deadman_refresh_ms = unix_ms();
        Ok(outcome)
    }

    pub fn acceptance_budget_ok(&self) -> Result<()> {
        if self.live.mode != LiveMode::AcceptanceTest {
            return Ok(());
        }
        let campaign = self.state.campaign()?;
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
                    self.startup_reconciliation = StartupReconciliation::Required;
                    self.diagnostics.scientifically_valid = false;
                    self.diagnostics.invalid_reason = Some("live WebSocket reconnected".to_owned());
                }
            }
            SessionEvent::Ready { generation, .. } => {
                self.diagnostics.connection_generation = generation;
                if generation == 1
                    && self.startup_reconciliation == StartupReconciliation::CleanSnapshot
                    && self.state.persistence_healthy()
                {
                    // Bootstrap already proved the dedicated account flat with
                    // no working or durable orders. The acknowledged account
                    // WebSocket is now the steady-state authority; avoid an
                    // immediate duplicate REST fan-out.
                    self.startup_reconciliation = StartupReconciliation::Complete;
                    self.last_reconcile_ms = unix_ms();
                    self.diagnostics.operationally_healthy = true;
                } else {
                    self.diagnostics.operationally_healthy = false;
                    self.reconcile_requested = true;
                }
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
            } => match channel {
                AccountChannel::OrderUpdates => self.apply_order_updates(data)?,
                AccountChannel::UserFills => self.apply_fills(data, received_ns)?,
                AccountChannel::UserFundings => self.apply_fundings(data)?,
                AccountChannel::ClearinghouseState => {
                    if self.last_inventory_update_exchange_ms == 0 {
                        let clearinghouse = parse_clearinghouse_message(data)?;
                        self.account
                            .apply_clearinghouse(&clearinghouse, &self.instrument)?;
                        self.pending_events.push(ExecutionEvent::AccountReconciled {
                            inventory_units: self.account.inventory_units,
                            equity_usdc: self.account.account_value_usdc,
                        });
                    } else {
                        // The stream shape has no exchange timestamp. Once a
                        // fill established an exchange-time watermark, applying
                        // an unwatermarked snapshot could roll inventory back.
                        self.reconcile_requested = true;
                    }
                }
                AccountChannel::OpenOrders => {
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
                AccountChannel::ActiveAssetData => self
                    .account
                    .apply_active_asset_data(&data, &self.instrument)?,
                AccountChannel::Notification => {
                    let text = data
                        .get("notification")
                        .and_then(serde_json::Value::as_str)
                        .or_else(|| data.as_str())
                        .unwrap_or_default();
                    let lower = text.to_ascii_lowercase();
                    if ["error", "reject", "invalid", "expired", "rate limit"]
                        .iter()
                        .any(|needle| lower.contains(needle))
                    {
                        self.diagnostics.operationally_healthy = false;
                        self.reconcile_requested = true;
                        self.diagnostics.invalid_reason =
                            Some(format!("venue notification: {text}"));
                    }
                }
                AccountChannel::Other(channel) => {
                    if channel.to_ascii_lowercase().contains("error") {
                        self.diagnostics.operationally_healthy = false;
                        self.reconcile_requested = true;
                        self.diagnostics.invalid_reason =
                            Some(format!("venue error channel {channel}: {data}"));
                    }
                }
                AccountChannel::LedgerUpdates => {}
            },
            SessionEvent::ActionCompleted {
                purpose, outcome, ..
            } => {
                if matches!(outcome, ActionOutcome::Unknown { .. }) {
                    self.diagnostics.unknown_outcomes += 1;
                    self.diagnostics.operationally_healthy = false;
                    self.reconcile_requested = true;
                } else if let Some(body) = outcome.body() {
                    let rejected = body
                        .pointer("/response/data/statuses")
                        .and_then(serde_json::Value::as_array)
                        .map_or(0_u64, |statuses| {
                            statuses
                                .iter()
                                .filter(|status| status.get("error").is_some())
                                .count() as u64
                        });
                    if rejected > 0
                        && matches!(
                            purpose,
                            crate::hyperliquid::session::ActionPurpose::Order
                                | crate::hyperliquid::session::ActionPurpose::ReduceOnly
                        )
                    {
                        self.diagnostics.orders_rejected =
                            self.diagnostics.orders_rejected.saturating_add(rejected);
                        self.consecutive_rejections = self.consecutive_rejections.saturating_add(1);
                        self.diagnostics.consecutive_rejections = self.consecutive_rejections;
                        let shift = self.consecutive_rejections.saturating_sub(1).min(5);
                        let backoff_ms = 1_000_u64.saturating_mul(1_u64 << shift).min(30_000);
                        self.rate_limited_until_ms = self
                            .rate_limited_until_ms
                            .max(unix_ms().saturating_add(backoff_ms));
                        self.diagnostics.operationally_healthy = false;
                        self.reconcile_requested = true;
                        self.diagnostics.invalid_reason = Some(format!(
                            "{rejected} order(s) rejected; backing off {backoff_ms}ms"
                        ));
                    } else if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
                        self.diagnostics.operationally_healthy = false;
                        self.diagnostics.invalid_reason =
                            Some(format!("{purpose:?} action failed: {body}"));
                        if purpose == crate::hyperliquid::session::ActionPurpose::Deadman {
                            self.deadman_armed = false;
                        }
                    } else if purpose == crate::hyperliquid::session::ActionPurpose::Order {
                        self.rejection_recovery = RejectionRecovery::AwaitingReconcile;
                    }
                }
            }
            SessionEvent::ActionRefused { reason, .. } => {
                self.diagnostics.operationally_healthy = false;
                self.note_rate_limit_if_applicable(unix_ms(), &reason);
                self.diagnostics.invalid_reason = Some(reason);
            }
        }
        Ok(std::mem::take(&mut self.pending_events))
    }

    pub async fn maintenance(&mut self, now_ms: u64) -> Result<Vec<ExecutionEvent>> {
        if !self.state.persistence_healthy() {
            self.diagnostics.operationally_healthy = false;
            self.diagnostics.invalid_reason =
                Some("live-state persistence is degraded; new exposure paused".to_owned());
            self.reconcile_requested = true;
        }
        if now_ms.saturating_sub(self.last_user_rate_limit_refresh_ms)
            >= self.live.user_rate_limit_refresh_ms
        {
            if let Err(error) = self.refresh_user_rate_limit(now_ms).await {
                self.degrade("userRateLimit refresh failed", &error);
            }
        }
        if self.deferred_desired.is_some()
            && now_ms.saturating_sub(self.last_quote_action_ms)
                >= self.quoting.min_order_lifetime_ms
        {
            let desired = self
                .deferred_desired
                .take()
                .expect("deferred desired quote was checked as present");
            self.reconcile(desired, now_ms).await?;
        }
        if now_ms.saturating_sub(self.last_reconcile_ms) >= self.live.reconcile_interval_ms {
            self.reconcile_requested = true;
        }
        if self.deadman_armed
            && self.diagnostics.operationally_healthy
            && now_ms.saturating_sub(self.last_deadman_refresh_ms) >= self.live.deadman_refresh_ms
        {
            let deadline = now_ms.saturating_add(self.live.deadman_deadline_ms);
            // Do not advance `last_deadman_refresh_ms` on failure: the refresh
            // interval is well inside the deadline, so leaving the timestamp
            // untouched retries on the next tick with time to spare.
            match self.session.enqueue_schedule_cancel(Some(deadline)) {
                Ok(()) => {
                    self.last_deadman_refresh_ms = now_ms;
                    self.diagnostics.deadman_refreshes += 1;
                    self.count_address_actions(1);
                }
                Err(error) => self.degrade("dead-man refresh enqueue refused", &error),
            }
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
        self.count_address_actions(1);
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
        self.count_address_actions(1);
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
            self.client
                .schedule_cancel_with_nonce(None, self.state.emergency_nonce()?)
                .await?
        };
        self.count_address_actions(1);
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
        self.count_address_actions(1);
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
            self.count_address_actions(1);
            self.record_order_outcome(&outcome);
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

    async fn reconcile_safety_position(&mut self) -> Result<()> {
        let (clearinghouse, open_orders) = tokio::try_join!(
            self.client.clearinghouse_state_safety(),
            self.client.open_orders_safety(),
        )?;
        ensure_no_foreign_positions(&clearinghouse, &self.instrument)?;
        let inventory = clearinghouse
            .asset_positions
            .iter()
            .find(|position| position.position.coin == self.instrument.symbol)
            .map_or(Ok(0), |position| {
                parse_fixed(&position.position.szi, self.instrument.sz_decimals)
            })?;
        self.account.inventory_units = inventory;
        self.account.open_orders = open_orders;
        self.last_reconcile_ms = unix_ms();
        self.startup_reconciliation = StartupReconciliation::Complete;
        self.diagnostics.reconciliations = self.diagnostics.reconciliations.saturating_add(1);
        Ok(())
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
        ensure_no_foreign_positions(&clearinghouse, &self.instrument)?;
        for order in &open_orders {
            let Some(cloid) = order.cloid.as_deref() else {
                bail!("foreign open order without CLOID on dedicated live account");
            };
            if !is_bot_cloid(cloid) {
                bail!("foreign open order {cloid} on dedicated live account");
            }
        }
        let view = self.state.with_state(|state| PersistedView {
            metadata_matches: state.metadata_fingerprint == self.instrument.metadata_fingerprint,
            any_non_terminal: state.orders.values().any(|order| !order.status.terminal()),
            cumulative_realized_pnl_usdc: state.cumulative_realized_pnl_usdc,
            cumulative_fees_usdc: state.cumulative_fees_usdc,
            cumulative_funding_usdc: state.cumulative_funding_usdc,
        })?;
        if !view.metadata_matches && (self.account.inventory_units != 0 || view.any_non_terminal) {
            bail!("live metadata changed while durable exposure exists");
        }
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
        // The REST position reflects trading up to the newest fill the venue
        // reports; that fill time is the only venue clock available here. With
        // no fills there is nothing newer to guard against, so the watermark
        // keeps its previous (exchange-time) value — never a local clock,
        // which would silently drop legitimate fills while it ran ahead.
        if let Some(latest_fill_ms) = fills.iter().map(|fill| fill.time).max() {
            self.last_inventory_update_exchange_ms =
                self.last_inventory_update_exchange_ms.max(latest_fill_ms);
        }
        self.account.realized_pnl_usdc = view.cumulative_realized_pnl_usdc;
        self.account.fees_usdc = view.cumulative_fees_usdc;
        self.account.funding_usdc = view.cumulative_funding_usdc;
        // Build recovered orders before entering `update`: it has no rollback,
        // and a parse failure mid-loop must not leave half the venue's orders
        // adopted.
        let mut recovered = Vec::new();
        for open in &open_orders {
            let Some(cloid) = open.cloid.as_deref() else {
                continue;
            };
            let side = match open.side.as_str() {
                "B" | "Buy" => Side::Buy,
                "A" | "Sell" => Side::Sell,
                value => bail!("unknown open-order side {value:?}"),
            };
            let qty_units = parse_fixed(&open.sz, self.instrument.sz_decimals)?;
            recovered.push(PersistedLiveOrder {
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
            });
        }
        self.state.update(|state| {
            if state.metadata_fingerprint != self.instrument.metadata_fingerprint {
                state
                    .metadata_fingerprint
                    .clone_from(&self.instrument.metadata_fingerprint);
            }
            for order in recovered {
                if !state.orders.contains_key(&order.cloid) {
                    state.orders.insert(order.cloid.clone(), order);
                }
            }
            Ok(())
        })?;
        if self.account.maker_fee_rate > self.live.max_maker_fee_rate {
            bail!("actual maker fee exceeds configured live maximum");
        }
        self.apply_fill_rows(&fills, false, false)?;
        self.reconcile_orders(&open_orders, &fills, &order_statuses)?;
        if self.account.inventory_units != 0
            && self.state.inventory_unit()?.is_none()
            && !self.allow_untracked_position
        {
            bail!("non-flat live account has no persisted inventory-unit identity");
        }
        self.prune_durable_history(&fills)?;
        self.last_reconcile_ms = unix_ms();
        self.diagnostics.reconciliations += 1;
        self.diagnostics.actual_maker_fee_rate = self.account.maker_fee_rate;
        self.diagnostics.actual_taker_fee_rate = self.account.taker_fee_rate;
        self.pending_events.push(ExecutionEvent::AccountReconciled {
            inventory_units: self.account.inventory_units,
            equity_usdc: self.account.account_value_usdc,
        });
        let unresolved = self.state.with_state(|state| {
            state
                .orders
                .values()
                .any(|order| order.status == LiveOrderStatus::UnknownOutcome)
        })?;
        if self.session.healthy() && !unresolved {
            self.diagnostics.operationally_healthy = true;
            if self.rejection_recovery == RejectionRecovery::AwaitingReconcile {
                self.rejection_recovery = RejectionRecovery::Idle;
                self.consecutive_rejections = 0;
                self.diagnostics.consecutive_rejections = 0;
            }
        }
        Ok(())
    }

    /// Bound the durable collections after each authoritative reconcile.
    ///
    /// Without this the dedup-key sets and terminal orders grew for the life
    /// of the store (surviving restarts), and every state clone and persist
    /// paid for that history. The venue is authoritative for anything still
    /// open once a reconcile has run, and the JSONL event log keeps forensics.
    fn prune_durable_history(&mut self, fills: &[UserFill]) -> Result<()> {
        // The replay horizon advances on exchange time only — the local clock
        // must not decide which venue events are re-admittable. The newest
        // fill time in the authoritative snapshot is a venue-time lower bound;
        // with no fills there is nothing new to prune against.
        if let Some(latest_fill_ms) = fills.iter().map(|fill| fill.time).max() {
            let candidate = latest_fill_ms.saturating_sub(EVENT_RETENTION_MS);
            self.state.advance_checkpoint_and_prune(candidate)?;
        }
        let terminal_grace_ms = self
            .live
            .action_expiry_ms
            .saturating_add(self.live.reconcile_interval_ms)
            .max(MINIMUM_TERMINAL_ORDER_RETENTION_MS);
        self.state
            .prune_terminal_orders(unix_ms().saturating_sub(terminal_grace_ms))?;
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
        let candidates: Vec<(String, u64)> = self.state.with_state(|state| {
            state
                .orders
                .values()
                .filter(|order| !order.status.terminal())
                .map(|order| (order.cloid.clone(), order.prepared_at_ms))
                .collect()
        })?;
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
            let applied = self.state.update(|state| {
                if let Some(order) = state.orders.get_mut(cloid) {
                    if update.status_timestamp < order.last_update_ms
                        || !order.status.can_transition_to(status)
                    {
                        return Ok(false);
                    }
                    order.status = status;
                    order.remaining_qty_units = order.remaining_qty_units.min(remaining);
                    order.oid = Some(update.order.oid);
                    order.last_update_ms = update.status_timestamp;
                    return Ok(true);
                }
                Ok(false)
            })?;
            if !applied {
                continue;
            }
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
        self.apply_fill_rows(&message.fills, true, !message.is_snapshot)
    }

    fn apply_fill_rows(
        &mut self,
        fills: &[UserFill],
        adjust_inventory: bool,
        reject_foreign: bool,
    ) -> Result<()> {
        let mut applied_any = false;
        for fill in fills {
            if fill.coin != self.instrument.symbol {
                continue;
            }
            if reject_foreign
                && fill
                    .cloid
                    .as_deref()
                    .is_some_and(|cloid| !is_bot_cloid(cloid))
            {
                self.diagnostics.operationally_healthy = false;
                bail!("foreign fill appeared on dedicated account stream");
            }
            // All fallible work happens before the state mutation: `update` has
            // no rollback, so a parse failure inside the closure would leave
            // the fill marked processed while its P&L — the daily-loss switch's
            // input — was silently dropped.
            let px: f64 = fill.px.parse()?;
            let size: f64 = fill.sz.parse()?;
            let fee: f64 = fill.fee.parse()?;
            let closed_pnl: f64 = fill.closed_pnl.parse()?;
            let qty_units = parse_fixed(&fill.sz, self.instrument.sz_decimals)?;
            let px_units = self.instrument.price_to_units(px)?;
            let start_units = parse_fixed(&fill.start_position, self.instrument.sz_decimals)?;
            let key = TimedKey(fill.time, fill_key(fill));
            let admission = self.state.update(|state| {
                // Checkpoint first: keys below it are pruned, and the time gate
                // is what keeps those pruned fills from being re-admitted.
                if fill.time < state.event_checkpoint_ms {
                    return Ok(FillAdmission::BeforeCheckpoint);
                }
                if !state.processed_fill_keys.insert(key.clone()) {
                    return Ok(FillAdmission::Duplicate);
                }
                let day = fill.time / 86_400_000;
                if state.pnl_day != day {
                    state.pnl_day = day;
                    state.daily_realized_pnl_usdc = 0.0;
                }
                let realized = closed_pnl - fee;
                state.cumulative_realized_pnl_usdc += realized;
                state.cumulative_fees_usdc += fee;
                state.daily_realized_pnl_usdc += realized;
                state.campaign.turnover_usdc += px * size;
                state.campaign.realized_pnl_usdc += realized;
                // Hyperliquid reports a non-zero closedPnl only on reducing
                // fills, so it is the live analogue of the dry-run backend's
                // `reduced_existing_position` gate. Judge the streak on the same
                // net figure the daily-loss switch uses, and — as in dry-run —
                // let a streak span days rather than resetting with the P&L day.
                if closed_pnl != 0.0 {
                    if realized < 0.0 {
                        state.consecutive_losses = state.consecutive_losses.saturating_add(1);
                    } else if realized > 0.0 {
                        state.consecutive_losses = 0;
                    }
                }
                Ok(FillAdmission::Fresh {
                    cumulative_fees_usdc: state.cumulative_fees_usdc,
                    cumulative_realized_pnl_usdc: state.cumulative_realized_pnl_usdc,
                })
            })?;
            match admission {
                FillAdmission::BeforeCheckpoint => continue,
                FillAdmission::Duplicate => {
                    self.diagnostics.duplicate_fills += 1;
                    continue;
                }
                FillAdmission::Fresh {
                    cumulative_fees_usdc,
                    cumulative_realized_pnl_usdc,
                } => {
                    self.account.fees_usdc = cumulative_fees_usdc;
                    self.account.realized_pnl_usdc = cumulative_realized_pnl_usdc;
                }
            }
            applied_any = true;
            let side = if fill.side == "B" {
                Side::Buy
            } else {
                Side::Sell
            };
            if adjust_inventory && fill.time >= self.last_inventory_update_exchange_ms {
                self.account.inventory_units = inventory_after_fill(start_units, side, qty_units);
                self.last_inventory_update_exchange_ms = fill.time;
            }
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
        if applied_any {
            let campaign = self.state.campaign()?;
            self.diagnostics.campaign_turnover_usdc = campaign.turnover_usdc;
            self.diagnostics.campaign_realized_pnl_usdc = campaign.realized_pnl_usdc;
        }
        Ok(())
    }

    fn apply_fundings(&mut self, data: serde_json::Value) -> Result<()> {
        for funding in parse_user_fundings(data)?.fundings {
            if funding.coin != self.instrument.symbol {
                continue;
            }
            // Parse before mutating: `update` has no rollback.
            let usdc: f64 = funding.usdc.parse()?;
            let key = TimedKey(funding.time, funding_key(&funding));
            let funding_total = self.state.update(|state| {
                // Same checkpoint gate as fills: pruned funding keys must not
                // be re-admittable through a replayed snapshot.
                if funding.time < state.event_checkpoint_ms
                    || !state.processed_funding_keys.insert(key.clone())
                {
                    return Ok(None);
                }
                state.cumulative_funding_usdc += usdc;
                Ok(Some(state.cumulative_funding_usdc))
            })?;
            let Some(funding_total) = funding_total else {
                continue;
            };
            self.account.funding_usdc = funding_total;
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
        self.fresh_bbo_with_priority(false).await
    }

    async fn fresh_bbo_with_priority(&mut self, safety_critical: bool) -> Result<Bbo> {
        if let Some(bbo) = self.latest_bbo {
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

    fn minimum_live_order_quantity(&self, px_units: i64) -> Result<i64> {
        let price = self.instrument.price_from_units(px_units);
        let one_unit_notional = price * self.instrument.size_from_units(1);
        if !one_unit_notional.is_finite() || one_unit_notional <= 0.0 {
            bail!("cannot derive minimum live order quantity from invalid price");
        }
        let target = self.instrument.minimum_notional * self.live.min_order_notional_multiplier;
        let quantity = (target / one_unit_notional).ceil() as i64;
        let notional = one_unit_notional * quantity as f64;
        let maximum = self.instrument.minimum_notional * self.live.max_order_notional_multiplier;
        if quantity <= 0 || notional > maximum {
            bail!("first valid lot is {notional:.6} USDC, above micro-live cap {maximum:.6} USDC");
        }
        Ok(quantity)
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
        // Project the working set under the lock: this runs per quote
        // replacement and must not clone the durable history.
        let (working, campaign) = self.state.with_state(|state| {
            let working: Vec<(Side, i64, i64)> = state
                .orders
                .values()
                .filter(|order| !order.status.terminal())
                .map(|order| (order.side, order.px_units, order.remaining_qty_units))
                .collect();
            (working, state.campaign)
        })?;
        let existing_gross: f64 = working
            .iter()
            .map(|(_, px_units, remaining_qty_units)| {
                self.instrument.price_from_units(*px_units)
                    * self.instrument.size_from_units(*remaining_qty_units)
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
            if self.live.mode == LiveMode::Production && !request.reduce_only {
                let minimum =
                    self.instrument.minimum_notional * self.live.min_order_notional_multiplier;
                let maximum =
                    self.instrument.minimum_notional * self.live.max_order_notional_multiplier;
                if notional < minimum || notional > maximum {
                    bail!(
                        "production order {notional:.6} USDC is outside micro-live range \
                         [{minimum:.6}, {maximum:.6}]"
                    );
                }
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
        for (side, _, remaining_qty_units) in working {
            match side {
                Side::Buy => buy_units = buy_units.saturating_add(remaining_qty_units),
                Side::Sell => sell_units = sell_units.saturating_sub(remaining_qty_units),
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
            if campaign.turnover_usdc >= self.live.acceptance_max_turnover_usdc - 12.0
                || campaign.realized_pnl_usdc
                    <= -(self.live.acceptance_max_realized_loss_usdc - 0.1)
            {
                bail!("acceptance campaign cleanup reserve reached");
            }
        } else {
            let directional_cap =
                self.instrument.minimum_notional * self.live.max_directional_notional_multiplier;
            let working_gross_cap =
                self.instrument.minimum_notional * self.live.max_working_gross_multiplier;
            if directional > directional_cap {
                bail!("prospective live directional exposure exceeds micro-live cap");
            }
            if existing_gross + new_gross > working_gross_cap {
                bail!("prospective live working gross exceeds micro-live cap");
            }
            let daily = self.state.risk_scalars(unix_ms() / 86_400_000)?;
            if daily.daily_realized_pnl_usdc <= -self.live.production_max_daily_realized_loss_usdc {
                bail!("production daily realized-loss stop is active");
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
        self.state.with_state(|state| {
            let mut by_side = BTreeMap::<Side, Vec<PersistedLiveOrder>>::new();
            for order in state.orders.values() {
                if !order.status.terminal() {
                    by_side.entry(order.side).or_default().push(order.clone());
                }
            }
            by_side
        })
    }

    fn mark_cancel_pending(&self, cloids: &[String], now_ms: u64) -> Result<()> {
        self.state.update(|state| {
            for cloid in cloids {
                if let Some(order) = state.orders.get_mut(cloid) {
                    if order
                        .status
                        .can_transition_to(LiveOrderStatus::CancelPending)
                    {
                        order.status = LiveOrderStatus::CancelPending;
                        order.last_update_ms = now_ms;
                    }
                }
            }
            Ok(())
        })
    }

    async fn cancel_cloids_resilient(&mut self, cloids: Vec<String>) -> Result<ActionOutcome> {
        if self.session.healthy() {
            return self.session.cancel_cloids(cloids).await;
        }
        self.client
            .cancel_by_cloid_with_nonce(&cloids, self.state.emergency_nonce()?)
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

fn allocated_usable_equity(
    allocated_capital_usdc: f64,
    account_equity_usdc: f64,
    liquidation_reserve_usdc: f64,
) -> f64 {
    let account_usable = (account_equity_usdc - liquidation_reserve_usdc).max(0.0);
    allocated_capital_usdc.max(0.0).min(account_usable)
}

#[cfg(test)]
fn live_directional_notional_cap(
    quoting: &QuotingConfig,
    risk: &RiskConfig,
    account_equity_usdc: f64,
) -> f64 {
    let leveraged_utilisation = quoting.leverage * quoting.target_capital_utilisation;
    let account_cap =
        (account_equity_usdc - risk.min_liquidation_buffer_usdc).max(0.0) * leveraged_utilisation;
    let allocation_cap = quoting
        .available_capital_usdc
        .min(account_equity_usdc.max(0.0))
        * leveraged_utilisation;
    let configured_cap = if risk.max_notional_usdc > 0.0 {
        risk.max_notional_usdc
    } else {
        f64::INFINITY
    };
    account_cap.min(allocation_cap).min(configured_cap)
}

#[async_trait]
impl ExecutionBackend for HyperliquidLiveBackend {
    async fn reconcile(&mut self, desired: DesiredQuotes, now_ms: u64) -> Result<()> {
        let inventory_changed = self.account.inventory_units != self.inventory_at_last_quote_action;
        let replacement_may_wait = desired.reason.replacement_may_wait(inventory_changed);
        if replacement_may_wait
            && self.last_quote_action_ms != 0
            && now_ms.saturating_sub(self.last_quote_action_ms) < self.quoting.min_order_lifetime_ms
        {
            self.deferred_desired = Some(desired);
            return Ok(());
        }
        self.deferred_desired = None;
        let mut existing = self.active_orders_by_side()?;
        let mut cancel = Vec::new();
        let mut place = Vec::new();
        for side in [Side::Buy, Side::Sell] {
            let target = match side {
                Side::Buy => desired.bid,
                Side::Sell => desired.ask,
            };
            let target = if let Some(mut target) = target {
                if self.live.mode == LiveMode::Production && !target.reduce_only {
                    target.qty_units = self.minimum_live_order_quantity(target.px)?;
                }
                Some(target)
            } else {
                None
            };
            let side_orders = existing.remove(&side).unwrap_or_default();
            // Hold a resting order whose price sits inside the requote hold
            // window: queue position is worth more than a sub-window price
            // improvement, and every avoided replacement saves two WebSocket
            // messages. Safety carve-outs, all load-bearing:
            // - `target.is_some_and` means a withdrawal (None target) bypasses
            //   hysteresis entirely — every fail-closed reason publishes empty
            //   quotes and still cancels immediately;
            // - any change in size or reduce-only forces a replacement;
            // - the comparison is against the *resting* price, never a rolling
            //   reference, so drift beyond the window always triggers.
            let unchanged = target.is_some_and(|target| {
                side_orders.iter().any(|order| {
                    (order.px_units - target.px).abs()
                        < self
                            .instrument
                            .requote_hold_window_units(&self.quoting, target.px)
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
                if !unchanged
                    && !cancel_already_in_flight(&order, now_ms, self.live.action_timeout_ms)
                {
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
        let mut action_submitted = false;
        if !cancel.is_empty() {
            let canceled_actions = cancel.len() as u64;
            self.mark_cancel_pending(&cancel, now_ms)?;
            if self.session.healthy() {
                // A momentarily full command queue is a transient condition, not
                // a reason to stop trading. Degrade and let the maintenance tick
                // retry; aborting here would tear down the session over
                // backpressure.
                if let Err(error) = self.session.enqueue_cancel_cloids(cancel) {
                    self.degrade("cancel enqueue refused", &error);
                    return Ok(());
                }
            } else {
                let outcome = self.cancel_cloids_resilient(cancel).await?;
                require_action_known(&outcome)?;
            }
            self.diagnostics.cancels_submitted += 1;
            self.count_cancel_actions(canceled_actions);
            action_submitted = true;
        }
        if !place.is_empty() && self.rate_limited(now_ms) {
            // Cancels above (if any) still went out; only new exposure waits.
            if action_submitted {
                self.last_quote_action_ms = now_ms;
                self.inventory_at_last_quote_action = self.account.inventory_units;
            }
            return Ok(());
        }
        if !place.is_empty() {
            let ordinary_count = place.iter().filter(|order| !order.reduce_only).count() as u64;
            if ordinary_count > 0 && !self.placement_quota_available(now_ms, ordinary_count) {
                // Keep the newest full target for a later quota window, but do
                // not let that throttle suppress risk-reducing orders now.
                self.deferred_desired = Some(desired);
                place.retain(|order| order.reduce_only);
            }
            let reducing_count = place.iter().filter(|order| order.reduce_only).count() as u64;
            if reducing_count > 0 && !self.safety_placement_available(reducing_count) {
                self.degrade(
                    "reduce-only placement reserve exhausted",
                    &anyhow::anyhow!("only the 100-action emergency reserve remains"),
                );
                return Ok(());
            }
            if place.is_empty() {
                if action_submitted {
                    self.last_quote_action_ms = now_ms;
                    self.inventory_at_last_quote_action = self.account.inventory_units;
                }
                return Ok(());
            }
        }
        if !place.is_empty() {
            let ordinary_actions = place.iter().filter(|order| !order.reduce_only).count() as u64;
            let reducing_actions = place.iter().filter(|order| order.reduce_only).count() as u64;
            // Prefer the hot path's shared slot: it is the exact snapshot the
            // quote decision priced from, while `latest_bbo` is fed by the
            // drained event ring and can lag under backlog.
            let bbo = self
                .market_bbo
                .as_ref()
                .and_then(|slot| slot.load())
                .or(self.latest_bbo)
                .context("cannot place live quote without BBO")?;
            // A refused validation means "do not place this order", never
            // "end the session": risk breaches and quota exhaustion both land
            // here, and the address-action reserve check is itself a rate
            // limit.
            if let Err(error) = self.validate_new_orders(&place, bbo) {
                let reason = error.to_string();
                self.note_rate_limit_if_applicable(now_ms, &reason);
                self.degrade("order validation refused", &error);
                if action_submitted {
                    self.last_quote_action_ms = now_ms;
                    self.inventory_at_last_quote_action = self.account.inventory_units;
                }
                return Ok(());
            }
            if let Err(error) = self.session.enqueue_orders(desired.quote_seq, place) {
                self.degrade("order enqueue refused", &error);
                // Cancels above may already be in flight; leaving the book
                // one-sided is safe, and the retry re-derives both sides.
                if action_submitted {
                    self.last_quote_action_ms = now_ms;
                    self.inventory_at_last_quote_action = self.account.inventory_units;
                }
                return Ok(());
            }
            self.diagnostics.orders_submitted += 1;
            if ordinary_actions > 0 {
                self.count_placement_actions(now_ms, ordinary_actions);
            }
            if reducing_actions > 0 {
                self.count_address_actions(reducing_actions);
            }
            action_submitted = true;
        }
        if action_submitted {
            self.last_quote_action_ms = now_ms;
            self.inventory_at_last_quote_action = self.account.inventory_units;
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
        // A known cancel response updates durable state inside the session
        // actor before its event reaches the strategy loop. Give that state a
        // bounded moment to settle; an immediate REST snapshot can lag the
        // successful WS cancel and provoke a duplicate.
        for _ in 0..15 {
            let only_cancel_pending = self.state.with_state(|state| {
                let mut working = state
                    .orders
                    .values()
                    .filter(|order| !order.status.terminal());
                let Some(first) = working.next() else {
                    return false;
                };
                first.status == LiveOrderStatus::CancelPending
                    && working.all(|order| order.status == LiveOrderStatus::CancelPending)
            })?;
            if !only_cancel_pending {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // One safety-priority REST confirmation catches a genuinely lost or
        // unknown cancel after the bounded WS wait.
        self.reconcile_safety_position().await?;
        let mut cloids = Vec::new();
        for order in &self.account.open_orders {
            let cloid = order
                .cloid
                .as_deref()
                .context("foreign open order without CLOID during shutdown")?;
            if !is_bot_cloid(cloid) {
                bail!("foreign open order {cloid} during shutdown");
            }
            cloids.push(cloid.to_owned());
        }
        if !cloids.is_empty() {
            let canceled_actions = cloids.len() as u64;
            let outcome = self.cancel_cloids_resilient(cloids).await?;
            self.diagnostics.cancels_submitted += 1;
            self.count_cancel_actions(canceled_actions);
            require_action_known(&outcome)?;
        }
        self.reconcile_safety_position().await?;
        if !self.account.open_orders.is_empty() {
            bail!("graceful live shutdown could not confirm empty open orders");
        }
        if self.live.flatten_on_stop && self.account.inventory_units != 0 {
            self.market_close().await?;
            self.reconcile_safety_position().await?;
            if self.account.inventory_units != 0 {
                bail!(
                    "flatten_on_stop left residual inventory {}",
                    self.account.inventory_units
                );
            }
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

fn ensure_no_foreign_positions(
    state: &crate::hyperliquid::exchange::ClearinghouseState,
    instrument: &InstrumentSpec,
) -> Result<()> {
    for position in &state.asset_positions {
        if position.position.coin == instrument.symbol {
            continue;
        }
        let size: f64 = position.position.szi.parse().with_context(|| {
            format!(
                "foreign {} position size is invalid",
                position.position.coin
            )
        })?;
        if !size.is_finite() || size != 0.0 {
            bail!(
                "foreign position {}={} appeared on dedicated live subaccount",
                position.position.coin,
                position.position.szi
            );
        }
    }
    Ok(())
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
    let needs_historical_orders =
        state.with_state(|state| state.orders.values().any(|order| !order.status.terminal()))?;
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
    let unresolved: Vec<String> = state.with_state(|state| {
        state
            .orders
            .values()
            .filter(|order| {
                !order.status.terminal()
                    && !open_cloids.contains(order.cloid.as_str())
                    && !fill_cloids.contains(order.cloid.as_str())
                    && !order_statuses.contains_key(&order.cloid)
            })
            .map(|order| order.cloid.clone())
            .collect()
    })?;
    // Resolve a bounded page per reconciliation. More than sixteen used to
    // fail every reconciliation forever; paging makes monotonic progress while
    // respecting the shared REST budget.
    for cloid in unresolved.into_iter().take(16) {
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

/// A cancel that is still in flight cannot be helped by sending it again: the
/// venue answers the duplicate with "Order was never placed, already canceled,
/// or filled", and every duplicate still spends an address action against the
/// rate budget. In a live run this was two thirds of all cancel traffic.
///
/// The bound is the session's own action timeout: until it elapses the first
/// cancel is genuinely outstanding, and after it the session marks the order
/// `UnknownOutcome` itself — at which point retrying is the safe direction and
/// this guard stops applying. The extra second only covers the timeout tick's
/// granularity, so a wedged session actor cannot strand a resting order.
fn cancel_already_in_flight(
    order: &PersistedLiveOrder,
    now_ms: u64,
    action_timeout_ms: u64,
) -> bool {
    order.status == LiveOrderStatus::CancelPending
        && now_ms.saturating_sub(order.last_update_ms) < action_timeout_ms.saturating_add(1_000)
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
    fn user_rate_limit_requires_a_nonzero_well_formed_cap() {
        assert_eq!(
            parse_user_rate_limit(&serde_json::json!({
                "cumVlm": "12.5", "nRequestsUsed": 7, "nRequestsCap": 10007
            }))
            .unwrap(),
            (12.5, 7, 10_007)
        );
        assert!(parse_user_rate_limit(&serde_json::json!({
            "cumVlm": "12.5", "nRequestsUsed": 7, "nRequestsCap": 0
        }))
        .is_err());
        assert!(parse_user_rate_limit(&serde_json::json!({})).is_err());
    }

    #[test]
    fn terminal_cancel_errors_are_not_counted_as_order_rejections() {
        let (_directory, mut backend, _) = lifecycle_backend();
        backend.diagnostics.operationally_healthy = true;
        backend
            .process_session_event(SessionEvent::ActionCompleted {
                purpose: crate::hyperliquid::session::ActionPurpose::Cancel,
                outcome: ActionOutcome::Response {
                    nonce: 1,
                    http_status: 200,
                    body: serde_json::json!({
                        "status":"ok",
                        "response":{"data":{"statuses":[
                            {"error":"Order was never placed, already canceled, or filled."}
                        ]}}
                    }),
                },
                received_ns: 1,
            })
            .unwrap();
        assert_eq!(backend.diagnostics.orders_rejected, 0);
        assert!(backend.operationally_healthy());
        assert!(!backend.reconciliation_requested());
    }

    #[test]
    fn clean_bootstrap_uses_ready_websocket_without_duplicate_rest_reconcile() {
        let (_directory, mut backend, _) = lifecycle_backend();
        backend.startup_reconciliation = StartupReconciliation::CleanSnapshot;
        backend.diagnostics.operationally_healthy = false;
        backend.reconcile_requested = false;
        backend
            .process_session_event(SessionEvent::Ready {
                generation: 1,
                received_ns: 1,
            })
            .unwrap();
        assert!(backend.operationally_healthy());
        assert!(!backend.reconciliation_requested());
        assert_eq!(
            backend.startup_reconciliation,
            StartupReconciliation::Complete
        );

        backend.startup_reconciliation = StartupReconciliation::Required;
        backend.diagnostics.operationally_healthy = false;
        backend.reconcile_requested = false;
        backend
            .process_session_event(SessionEvent::Ready {
                generation: 2,
                received_ns: 2,
            })
            .unwrap();
        assert!(!backend.operationally_healthy());
        assert!(backend.reconciliation_requested());
    }
    use crate::config::{LatencyConfig, LiveConfig, Network};
    use crate::types::QuoteReason;
    use proptest::prelude::*;

    fn lifecycle_backend() -> (tempfile::TempDir, HyperliquidLiveBackend, String) {
        let directory = tempfile::tempdir().unwrap();
        let credentials_path = directory.path().join("hyperliquid.env");
        std::fs::write(
            &credentials_path,
            "exchange=hyperliquid\nwallet_address=0x1111111111111111111111111111111111111111\nprivate_key=0x0000000000000000000000000000000000000000000000000000000000000001\nis_vault=true\n",
        )
        .unwrap();
        let credentials = Arc::new(HyperliquidCredentials::load(&credentials_path).unwrap());
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
        let state = Arc::new(
            LiveStateStore::open(
                &directory.path().join("live.redb"),
                "CASHCAT",
                credentials.account(),
                &credentials.agent_address(),
                "config",
                "meta",
                SessionIntent::Quote {
                    venue_is_flat: true,
                },
            )
            .unwrap(),
        );
        let cloid = make_cloid(unix_ms(), 1, Side::Buy, 1);
        let checkpoint = state.load_required().unwrap().event_checkpoint_ms;
        state
            .update(|persisted| {
                persisted.orders.insert(
                    cloid.clone(),
                    PersistedLiveOrder {
                        cloid: cloid.clone(),
                        quote_seq: 1,
                        side: Side::Buy,
                        px_units: 100_000,
                        original_qty_units: 10,
                        remaining_qty_units: 10,
                        reduce_only: false,
                        status: LiveOrderStatus::Sent,
                        nonce: Some(checkpoint),
                        transport_id: Some(1),
                        oid: None,
                        prepared_at_ms: checkpoint,
                        last_update_ms: checkpoint,
                        last_error: None,
                    },
                );
                Ok(())
            })
            .unwrap();
        let clock = Arc::new(ProcessClock::default());
        let latency_config = LatencyConfig {
            gate_enabled: false,
            ..LatencyConfig::default()
        };
        let latency = Arc::new(LatencyMonitor::new(
            "CASHCAT",
            checkpoint,
            &latency_config,
            false,
        ));
        let client = Arc::new(
            HyperliquidExchangeClient::new(
                Network::Mainnet,
                instrument.clone(),
                credentials,
                clock.clone(),
                Some(latency.clone()),
                Duration::from_secs(1),
                5_000,
                1_000,
                100,
            )
            .unwrap(),
        );
        let mut test_live = LiveConfig::default();
        test_live.mode = LiveMode::AcceptanceTest;
        test_live.acceptance_max_order_notional_usdc = 1_000.0;
        test_live.acceptance_max_directional_notional_usdc = 1_000.0;
        test_live.acceptance_max_working_gross_usdc = 2_000.0;
        test_live.acceptance_max_turnover_usdc = 10_000.0;
        test_live.acceptance_max_realized_loss_usdc = 100.0;
        let backend = HyperliquidLiveBackend {
            instrument,
            live: test_live,
            quoting: QuotingConfig::default(),
            risk: RiskConfig::default(),
            client,
            session: HyperliquidSessionHandle::test_stub(clock.clone()),
            state,
            account: LiveAccountSnapshot {
                account_value_usdc: 300.0,
                withdrawable_usdc: 300.0,
                available_to_trade_usdc: [300.0, 300.0],
                max_trade_units: [5_000, 5_000],
                isolated: true,
                leverage: 2,
                maker_fee_rate: 0.00015,
                taker_fee_rate: 0.00045,
                ..LiveAccountSnapshot::default()
            },
            diagnostics: LiveExecutionDiagnostics {
                operationally_healthy: true,
                scientifically_valid: true,
                address_requests_cap: 10_000,
                ..LiveExecutionDiagnostics::default()
            },
            latest_bbo: None,
            market_bbo: None,
            session_started_at_ms: checkpoint,
            pending_events: Vec::new(),
            last_reconcile_ms: 0,
            deadman_armed: false,
            last_deadman_refresh_ms: 0,
            clock,
            market_stale_ms: 5_000,
            q_max: 6,
            latency,
            last_fill_received_ns: None,
            allow_untracked_position: false,
            reconcile_requested: false,
            last_inventory_update_exchange_ms: 0,
            last_quote_action_ms: 0,
            inventory_at_last_quote_action: 0,
            deferred_desired: None,
            rate_limited_until_ms: 0,
            last_user_rate_limit_refresh_ms: 0,
            next_placement_allowed_ms: 0,
            consecutive_rejections: 0,
            rejection_recovery: RejectionRecovery::Idle,
            startup_reconciliation: StartupReconciliation::Complete,
        };
        (directory, backend, cloid)
    }

    /// Prepare a backend whose one resting bid can be re-quoted: the seeded
    /// order is resized to clear the venue minimum notional and a BBO is
    /// installed so the placement path can validate.
    fn requote_backend(
        px_units: i64,
        qty_units: i64,
    ) -> (tempfile::TempDir, HyperliquidLiveBackend) {
        let (directory, mut backend, cloid) = lifecycle_backend();
        backend
            .state
            .update(|state| {
                let order = state.orders.get_mut(&cloid).expect("seeded order");
                order.px_units = px_units;
                order.original_qty_units = qty_units;
                order.remaining_qty_units = qty_units;
                order.status = LiveOrderStatus::Resting;
                Ok(())
            })
            .unwrap();
        backend.latest_bbo = Some(Bbo {
            bid_px: 99_990,
            bid_sz: 1_000,
            ask_px: 100_100,
            ask_sz: 1_000,
            exchange_ms: 1,
            recv_ns: 1,
        });
        (directory, backend)
    }

    #[test]
    fn placement_allowance_keeps_emergency_and_safety_reserves() {
        let (_directory, mut backend, _) = lifecycle_backend();
        backend.diagnostics.address_requests_used = 0;
        backend.diagnostics.address_requests_cap = 120;
        assert_eq!(backend.safe_placement_allowance(), 10);
        assert!(backend.placement_quota_available(1_000, 2));
        backend.count_placement_actions(1_000, 2);
        assert_eq!(backend.safe_placement_allowance(), 8);
        assert!(!backend.placement_quota_available(1_001, 2));
        assert_eq!(backend.diagnostics.address_requests_used, 2);
    }

    #[test]
    fn reduce_only_can_use_safety_headroom_without_touching_emergency_reserve() {
        let (_directory, mut backend, _) = lifecycle_backend();
        backend.diagnostics.address_requests_used = 0;
        backend.diagnostics.address_requests_cap = 105;
        assert_eq!(backend.safe_placement_allowance(), 0);
        assert!(backend.safety_placement_available(1));
        assert!(!backend.safety_placement_available(6));
    }

    fn bid_target(px: i64, qty_units: i64) -> DesiredQuotes {
        DesiredQuotes {
            bid: Some(crate::types::OrderIntent {
                side: Side::Buy,
                px,
                qty_units,
                post_only: true,
                reduce_only: false,
            }),
            ..DesiredQuotes::empty(QuoteReason::Market, 9, 9)
        }
    }

    #[tokio::test]
    async fn sub_window_price_moves_hold_the_resting_order() {
        let (_directory, mut backend) = requote_backend(100_000, 200);
        // 15 units of drift < the 20-unit window: hold, no venue action.
        backend
            .reconcile(bid_target(100_015, 200), 1_000)
            .await
            .unwrap();
        assert_eq!(backend.diagnostics.cancels_submitted, 0);
        assert_eq!(backend.diagnostics.orders_submitted, 0);
        assert_eq!(
            backend.last_quote_action_ms, 0,
            "a held quote must not restart the requote cooldown"
        );
    }

    #[tokio::test]
    async fn moves_past_the_window_replace_the_order() {
        let (_directory, mut backend) = requote_backend(100_000, 200);
        backend
            .reconcile(bid_target(100_030, 200), 1_000)
            .await
            .unwrap();
        assert_eq!(backend.diagnostics.cancels_submitted, 1);
        assert_eq!(backend.diagnostics.orders_submitted, 1);
    }

    #[tokio::test]
    async fn size_changes_replace_even_inside_the_window() {
        let (_directory, mut backend) = requote_backend(100_000, 200);
        backend
            .reconcile(bid_target(100_000, 150), 1_000)
            .await
            .unwrap();
        assert_eq!(backend.diagnostics.cancels_submitted, 1);
        assert_eq!(backend.diagnostics.orders_submitted, 1);
    }

    /// Withdrawals bypass hysteresis: every fail-closed reason publishes empty
    /// quotes, and those must cancel immediately.
    #[tokio::test]
    async fn empty_quotes_cancel_immediately_despite_the_hold_window() {
        let (_directory, mut backend) = requote_backend(100_000, 200);
        backend
            .reconcile(DesiredQuotes::empty(QuoteReason::RiskLimit, 9, 9), 1_000)
            .await
            .unwrap();
        assert_eq!(backend.diagnostics.cancels_submitted, 1);
        assert_eq!(backend.diagnostics.orders_submitted, 0);
    }

    /// The inventory watermark is exchange time: fills gate on venue
    /// timestamps only, never on the local clock. A local clock running ahead
    /// used to silently drop legitimate fills' inventory adjustments until the
    /// next 30s reconcile, leaving the hot path quoting on wrong inventory.
    #[test]
    fn inventory_watermark_follows_venue_time_and_ignores_stale_replays() {
        let (_directory, mut backend, cloid) = lifecycle_backend();
        let checkpoint = backend.state.load_required().unwrap().event_checkpoint_ms;
        let feed = |backend: &mut HyperliquidLiveBackend,
                    tid: u64,
                    time: u64,
                    start_position: &str,
                    size: &str| {
            backend
                .process_session_event(SessionEvent::AccountData {
                    generation: 1,
                    received_ns: 100 + tid,
                    channel: AccountChannel::UserFills,
                    data: serde_json::json!({
                        "isSnapshot": false,
                        "fills": [{
                            "coin":"CASHCAT", "px":"0.1", "sz":size, "side":"B",
                            "time":time, "oid":7, "tid":tid, "cloid":cloid,
                            "startPosition":start_position, "crossed":false,
                            "fee":"0.001", "closedPnl":"0",
                            "hash":format!("0x{tid:x}")
                        }]
                    }),
                })
                .unwrap();
        };

        // A fill whose venue time is far behind the local wall clock (the
        // checkpoint is roughly "now"; this is only 10ms past it) must still
        // adjust inventory: the watermark starts at 0, not at unix_ms().
        feed(&mut backend, 1, checkpoint + 10, "0", "4");
        assert_eq!(backend.account.inventory_units, 4);
        assert_eq!(
            backend.last_inventory_update_exchange_ms,
            checkpoint + 10,
            "watermark must advance to the fill's venue time"
        );

        // An out-of-order (older venue time) fill must not regress inventory,
        // because inventory_after_fill is absolute from startPosition.
        feed(&mut backend, 2, checkpoint + 5, "0", "1");
        assert_eq!(
            backend.account.inventory_units, 4,
            "an older fill must not rewind inventory to its stale startPosition"
        );

        // A newer fill advances both inventory and the watermark.
        feed(&mut backend, 3, checkpoint + 20, "4", "2");
        assert_eq!(backend.account.inventory_units, 6);
        assert_eq!(backend.last_inventory_update_exchange_ms, checkpoint + 20);
    }

    #[test]
    fn rate_limit_reasons_are_recognised_and_ordinary_rejections_are_not() {
        for reason in [
            "Hyperliquid WebSocket message rate limit reached",
            "Hyperliquid regular WebSocket message budget reached",
            "configured Hyperliquid REST weight budget exhausted: used=1000",
            "venue address-action reserve is below 100 requests",
            "maximum live in-flight posts reached",
        ] {
            assert!(is_rate_limit_reason(reason), "{reason}");
        }
        for reason in [
            "live order is below venue minimum notional",
            "normal live quotes must be post-only",
            "cannot place live quote without BBO",
        ] {
            assert!(!is_rate_limit_reason(reason), "{reason}");
        }
    }

    /// After a rate-limit refusal the bot must stop adding exposure for the
    /// cooldown, even once health is restored — otherwise it bounces straight
    /// back into a saturated window. Cancels must keep working throughout.
    #[tokio::test]
    async fn rate_limit_cooldown_suspends_new_orders_but_never_cancels() {
        let (_directory, mut backend) = requote_backend(100_000, 200);
        backend
            .process_session_event(SessionEvent::ActionRefused {
                purpose: crate::hyperliquid::session::ActionPurpose::Order,
                reason: "Hyperliquid regular WebSocket message budget reached".to_owned(),
                received_ns: 1,
            })
            .unwrap();
        assert_eq!(backend.diagnostics.rate_limit_cooldowns, 1);
        let now = unix_ms();
        assert!(backend.rate_limited(now));

        // A quote that would add exposure places nothing while cooling down.
        backend
            .reconcile(bid_target(100_500, 200), now)
            .await
            .expect("cooldown must not end the session");
        assert_eq!(backend.diagnostics.orders_submitted, 0);

        // A withdrawal still cancels: reducing exposure is never suspended.
        backend
            .reconcile(DesiredQuotes::empty(QuoteReason::RiskLimit, 9, 9), now)
            .await
            .unwrap();
        assert_eq!(backend.diagnostics.cancels_submitted, 1);

        // Past the cooldown, placement resumes.
        assert!(!backend.rate_limited(now + RATE_LIMIT_COOLDOWN_MS + 1));
    }

    /// Marking the first cancel pending before enqueue prevents a second quote
    /// decision from filling the priority queue with the same cancel.
    #[tokio::test]
    async fn a_pending_cancel_is_not_enqueued_twice() {
        let (_directory, mut backend, _cloid) = lifecycle_backend();
        // `test_stub` backs both command queues with a capacity of one, so the
        // first cancel fills the priority queue and the second is coalesced.
        // `RiskLimit` withdraws both sides and bypasses the requote cooldown.
        backend
            .reconcile(DesiredQuotes::empty(QuoteReason::RiskLimit, 1, 0), 1_000)
            .await
            .expect("first cancel enqueues");
        assert!(backend.operationally_healthy());

        backend
            .reconcile(DesiredQuotes::empty(QuoteReason::RiskLimit, 2, 0), 2_000)
            .await
            .expect("duplicate cancel must be coalesced");
        assert!(backend.operationally_healthy());
        assert!(!backend.reconciliation_requested());
        assert_eq!(backend.diagnostics.cancels_submitted, 1);
    }

    /// The venue never reports a loss streak, so it is derived here from closing
    /// fills. Before this was tracked, `RiskState.consecutive_losses` was pinned
    /// at zero in live and `max_consecutive_losses` could never fire.
    #[test]
    fn consecutive_losses_track_closing_fills_and_reset_on_a_win() {
        let (_directory, mut backend, cloid) = lifecycle_backend();
        let checkpoint = backend.state.load_required().unwrap().event_checkpoint_ms;
        let mut tid = 0_u64;
        let mut feed = |backend: &mut HyperliquidLiveBackend, closed_pnl: &str| {
            tid += 1;
            backend
                .process_session_event(SessionEvent::AccountData {
                    generation: 1,
                    received_ns: 100 + tid,
                    channel: AccountChannel::UserFills,
                    data: serde_json::json!({
                        "isSnapshot": false,
                        "fills": [{
                            "coin":"CASHCAT", "px":"0.1", "sz":"1", "side":"B",
                            "time":checkpoint.saturating_add(10 * tid),
                            "oid":7, "tid":tid, "cloid":cloid,
                            "startPosition":"0", "crossed":false, "fee":"0.001",
                            "closedPnl":closed_pnl, "hash":format!("0x{tid:x}")
                        }]
                    }),
                })
                .unwrap();
        };

        // Opening fills carry no closedPnl and must not count as losses, even
        // though they do cost a fee.
        feed(&mut backend, "0");
        assert_eq!(backend.risk_scalars().unwrap().consecutive_losses, 0);

        for expected in 1..=3 {
            feed(&mut backend, "-1");
            assert_eq!(
                backend.risk_scalars().unwrap().consecutive_losses,
                expected,
                "a losing close must extend the streak"
            );
        }

        feed(&mut backend, "5");
        assert_eq!(
            backend.risk_scalars().unwrap().consecutive_losses,
            0,
            "a winning close must reset the streak"
        );

        // Daily realized P&L is net of fees and feeds the daily-loss switch.
        let expected_daily = -0.001 + 3.0 * (-1.0 - 0.001) + (5.0 - 0.001);
        let daily = backend.risk_scalars().unwrap().daily_realized_pnl_usdc;
        assert!(
            (daily - expected_daily).abs() < 1.0e-9,
            "daily realized pnl was {daily}, expected {expected_daily}"
        );
    }

    #[test]
    fn a_cancel_still_in_flight_is_not_sent_again() {
        let mut order = PersistedLiveOrder {
            cloid: "c-1".to_owned(),
            quote_seq: 1,
            side: Side::Buy,
            px_units: 100,
            original_qty_units: 10,
            remaining_qty_units: 10,
            reduce_only: false,
            status: LiveOrderStatus::CancelPending,
            nonce: None,
            transport_id: None,
            oid: None,
            prepared_at_ms: 10_000,
            last_update_ms: 10_000,
            last_error: None,
        };
        // Inside the action timeout the first cancel is still outstanding.
        assert!(cancel_already_in_flight(&order, 11_000, 2_000));
        // Past it the session has already marked the order unknown, so
        // retrying is the safe direction and the guard must stop applying.
        assert!(!cancel_already_in_flight(&order, 13_500, 2_000));
        // Only an in-flight cancel is held back; a resting order is always
        // cancellable, however recently it was touched.
        order.status = LiveOrderStatus::Resting;
        assert!(!cancel_already_in_flight(&order, 11_000, 2_000));
    }

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

    #[test]
    fn fill_ack_cancel_reordering_is_idempotent_and_never_resurrects_quantity() {
        let (_directory, mut backend, cloid) = lifecycle_backend();
        let checkpoint = backend.state.load_required().unwrap().event_checkpoint_ms;
        let fill_time = checkpoint.saturating_add(10);
        let fill = serde_json::json!({
            "isSnapshot": false,
            "fills": [{
                "coin":"CASHCAT", "px":"0.1", "sz":"4", "side":"B",
                "time":fill_time, "oid":7, "tid":9, "cloid":cloid,
                "startPosition":"0", "crossed":false, "fee":"0.001",
                "closedPnl":"0", "hash":"0xabc"
            }]
        });
        let events = backend
            .process_session_event(SessionEvent::AccountData {
                generation: 1,
                received_ns: 100,
                channel: AccountChannel::UserFills,
                data: fill.clone(),
            })
            .unwrap();
        assert!(matches!(events.as_slice(), [ExecutionEvent::Fill(_)]));
        assert_eq!(backend.account.inventory_units, 4);
        let partial = backend.state.load_required().unwrap().orders[&cloid].clone();
        assert_eq!(partial.status, LiveOrderStatus::PartiallyFilled);
        assert_eq!(partial.remaining_qty_units, 6);

        let stale_ack = serde_json::json!([{
            "order": {
                "coin":"CASHCAT", "side":"B", "limitPx":"0.1", "sz":"10",
                "oid":7, "timestamp":checkpoint, "cloid":cloid
            },
            "status":"open", "statusTimestamp":checkpoint.saturating_add(5)
        }]);
        assert!(backend
            .process_session_event(SessionEvent::AccountData {
                generation: 1,
                received_ns: 110,
                channel: AccountChannel::OrderUpdates,
                data: stale_ack,
            })
            .unwrap()
            .is_empty());
        let still_partial = &backend.state.load_required().unwrap().orders[&cloid];
        assert_eq!(still_partial.status, LiveOrderStatus::PartiallyFilled);
        assert_eq!(still_partial.remaining_qty_units, 6);

        assert!(backend
            .process_session_event(SessionEvent::AccountData {
                generation: 1,
                received_ns: 120,
                channel: AccountChannel::UserFills,
                data: fill,
            })
            .unwrap()
            .is_empty());
        assert_eq!(backend.diagnostics.duplicate_fills, 1);
        assert_eq!(backend.account.inventory_units, 4);

        let cancel = serde_json::json!([{
            "order": {
                "coin":"CASHCAT", "side":"B", "limitPx":"0.1", "sz":"6",
                "oid":7, "timestamp":checkpoint, "cloid":cloid
            },
            "status":"canceled", "statusTimestamp":fill_time.saturating_add(10)
        }]);
        let events = backend
            .process_session_event(SessionEvent::AccountData {
                generation: 1,
                received_ns: 130,
                channel: AccountChannel::OrderUpdates,
                data: cancel,
            })
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [ExecutionEvent::OrderCanceled { .. }]
        ));
        assert_eq!(
            backend.state.load_required().unwrap().orders[&cloid].status,
            LiveOrderStatus::Canceled
        );
    }

    #[test]
    fn foreign_fill_invalidates_dedicated_account_session() {
        let (_directory, mut backend, _) = lifecycle_backend();
        let time = backend
            .state
            .load_required()
            .unwrap()
            .event_checkpoint_ms
            .saturating_add(1);
        let result = backend.process_session_event(SessionEvent::AccountData {
            generation: 1,
            received_ns: 1,
            channel: AccountChannel::UserFills,
            data: serde_json::json!({
                "isSnapshot": false,
                "fills": [{
                    "coin":"CASHCAT", "px":"0.1", "sz":"1", "side":"B",
                    "time":time, "oid":7, "tid":10,
                    "cloid":"0x00000000000000000000000000000001",
                    "startPosition":"0", "crossed":false, "fee":"0",
                    "closedPnl":"0", "hash":"0xdef"
                }]
            }),
        });
        assert!(result.is_err());
        assert!(!backend.diagnostics.operationally_healthy);
    }

    #[test]
    fn historical_foreign_fill_snapshot_is_ignored() {
        let (_directory, mut backend, _) = lifecycle_backend();
        let checkpoint = backend.state.load_required().unwrap().event_checkpoint_ms;
        let result = backend.process_session_event(SessionEvent::AccountData {
            generation: 1,
            received_ns: 1,
            channel: AccountChannel::UserFills,
            data: serde_json::json!({
                "isSnapshot": true,
                "fills": [{
                    "coin":"CASHCAT", "px":"0.1", "sz":"1", "side":"B",
                    "time":checkpoint.saturating_sub(1), "oid":7, "tid":11,
                    "cloid":"0x00000000000000000000000000000001",
                    "startPosition":"0", "crossed":false, "fee":"0",
                    "closedPnl":"0", "hash":"0xsnapshot"
                }]
            }),
        });
        assert!(result.unwrap().is_empty());
        assert!(backend.diagnostics.operationally_healthy);
        assert_eq!(backend.account.inventory_units, 0);
    }

    #[test]
    fn action_refusal_and_unknown_outcome_fail_closed() {
        let (_directory, mut backend, _) = lifecycle_backend();
        assert!(backend
            .process_session_event(SessionEvent::ActionRefused {
                purpose: crate::hyperliquid::session::ActionPurpose::Order,
                reason: "queue saturated".to_owned(),
                received_ns: 1,
            })
            .unwrap()
            .is_empty());
        assert!(!backend.diagnostics.operationally_healthy);
        assert_eq!(
            backend.diagnostics.invalid_reason.as_deref(),
            Some("queue saturated")
        );
        backend
            .process_session_event(SessionEvent::ActionCompleted {
                purpose: crate::hyperliquid::session::ActionPurpose::Order,
                outcome: ActionOutcome::Unknown {
                    nonce: 1,
                    error: "ambiguous write".to_owned(),
                },
                received_ns: 2,
            })
            .unwrap();
        assert_eq!(backend.diagnostics.unknown_outcomes, 1);
        assert!(backend.reconcile_requested);
    }

    #[test]
    fn allocated_capital_and_account_buffer_are_independent_caps() {
        let quoting = QuotingConfig {
            available_capital_usdc: 67.56,
            target_capital_utilisation: 0.74,
            leverage: 2.0,
            ..QuotingConfig::default()
        };
        let risk = RiskConfig {
            max_notional_usdc: 100.0,
            min_liquidation_buffer_usdc: 100.0,
            ..RiskConfig::default()
        };
        let cap = live_directional_notional_cap(&quoting, &risk, 299.48);
        assert!((cap - 99.9888).abs() < 1.0e-9);
        let underfunded = live_directional_notional_cap(&quoting, &risk, 50.0);
        assert_eq!(underfunded, 0.0);
        assert!((allocated_usable_equity(67.56, 299.48, 100.0) - 67.56).abs() < 1.0e-9);
        assert_eq!(allocated_usable_equity(67.56, 50.0, 100.0), 0.0);
    }

    #[tokio::test]
    async fn market_replacements_coalesce_until_minimum_order_lifetime() {
        let (_directory, mut backend, _) = lifecycle_backend();
        backend
            .state
            .update(|state| {
                let mut ask = state
                    .orders
                    .values()
                    .next()
                    .context("test backend has no seed order")?
                    .clone();
                ask.cloid = make_cloid(unix_ms(), 1, Side::Sell, 2);
                ask.side = Side::Sell;
                ask.px_units = 103_000;
                state.orders.insert(ask.cloid.clone(), ask);
                Ok(())
            })
            .unwrap();
        backend.quoting.min_order_lifetime_ms = 2_000;
        backend.last_quote_action_ms = 1_000;
        backend.latest_bbo = Some(Bbo {
            bid_px: 100_000,
            bid_sz: 1_000,
            ask_px: 102_000,
            ask_sz: 1_000,
            exchange_ms: 1,
            recv_ns: 1,
        });
        let desired = DesiredQuotes {
            bid: Some(crate::types::OrderIntent {
                side: Side::Buy,
                px: 101_000,
                qty_units: 100,
                post_only: true,
                reduce_only: false,
            }),
            ask: Some(crate::types::OrderIntent {
                side: Side::Sell,
                px: 103_000,
                qty_units: 100,
                post_only: true,
                reduce_only: false,
            }),
            quote_seq: 2,
            model_revision: 1,
            reason: QuoteReason::Market,
            ..DesiredQuotes::default()
        };
        backend.reconcile(desired, 1_500).await.unwrap();
        assert_eq!(backend.deferred_desired, Some(desired));
        assert_eq!(backend.diagnostics.orders_submitted, 0);
        assert_eq!(backend.diagnostics.cancels_submitted, 0);

        backend.maintenance(3_000).await.unwrap();
        assert!(backend.deferred_desired.is_none());
        assert_eq!(backend.last_quote_action_ms, 3_000);
        assert_eq!(backend.diagnostics.orders_submitted, 1);
        assert_eq!(backend.diagnostics.cancels_submitted, 1);
        assert_eq!(backend.diagnostics.address_requests_used, 2);
        assert_eq!(backend.diagnostics.cancel_requests_used, 2);
    }

    #[test]
    fn only_real_inventory_changes_bypass_replacement_cooldown() {
        assert!(QuoteReason::replacement_may_wait(
            QuoteReason::Market,
            false
        ));
        assert!(QuoteReason::replacement_may_wait(QuoteReason::Fill, false));
        assert!(!QuoteReason::replacement_may_wait(QuoteReason::Fill, true));
        assert!(!QuoteReason::replacement_may_wait(
            QuoteReason::RiskLimit,
            false
        ));
        assert!(!QuoteReason::replacement_may_wait(
            QuoteReason::Shutdown,
            false
        ));
    }

    proptest! {
        #[test]
        fn bounded_reduce_only_fill_never_flips_inventory(
            inventory in -1_000_000_i64..=1_000_000_i64,
            fraction in 0_u32..=1_000,
        ) {
            if let Some((side, maximum)) = closing_side_and_quantity(inventory) {
                let fill = maximum.saturating_mul(i64::from(fraction)) / 1_000;
                let after = inventory_after_fill(inventory, side, fill);
                prop_assert!(after == 0 || after.signum() == inventory.signum());
                prop_assert!(after.unsigned_abs() <= inventory.unsigned_abs());
            } else {
                prop_assert_eq!(inventory, 0);
            }
        }
    }
}
