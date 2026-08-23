use super::auth::HyperliquidCredentials;
use super::exchange::ActionOutcome;
use super::live_state::{DurableNonceManager, LiveOrderStatus, LiveStateStore, PersistedLiveOrder};
use super::signing::{
    cancel_by_cloid_action, cancel_by_oid_action, order_action, schedule_cancel_action,
    sign_envelope, update_leverage_action, LiveOrderRequest,
};
use crate::config::{LiveConfig, Network};
use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::transport::{
    account_subscriptions, application_ping, subscription_request, ACCOUNT_SUBSCRIPTION_COUNT,
};
use crate::types::{unix_ms, ProcessClock};
use anyhow::{bail, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::warn;

#[derive(Debug, Clone)]
pub enum SessionEvent {
    Connected {
        generation: u64,
        received_ns: u64,
    },
    Ready {
        generation: u64,
        received_ns: u64,
    },
    AccountData {
        generation: u64,
        received_ns: u64,
        channel: AccountChannel,
        data: serde_json::Value,
    },
    ActionCompleted {
        purpose: ActionPurpose,
        outcome: ActionOutcome,
        received_ns: u64,
    },
    ActionRefused {
        purpose: ActionPurpose,
        reason: String,
        received_ns: u64,
    },
    Disconnected {
        generation: u64,
        received_ns: u64,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountChannel {
    OrderUpdates,
    UserFills,
    UserFundings,
    ClearinghouseState,
    OpenOrders,
    ActiveAssetData,
    LedgerUpdates,
    Notification,
    Other(String),
}

impl AccountChannel {
    fn from_wire(channel: &str) -> Self {
        match channel {
            "orderUpdates" => Self::OrderUpdates,
            "userFills" => Self::UserFills,
            "userFundings" => Self::UserFundings,
            "clearinghouseState" => Self::ClearinghouseState,
            "openOrders" => Self::OpenOrders,
            "activeAssetData" => Self::ActiveAssetData,
            "userNonFundingLedgerUpdates" => Self::LedgerUpdates,
            "notification" => Self::Notification,
            other => Self::Other(other.to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPurpose {
    Order,
    Cancel,
    Leverage,
    Deadman,
}

impl ActionPurpose {
    const fn latency_kind(self) -> LatencyKind {
        match self {
            Self::Cancel | Self::Deadman => LatencyKind::CancelToAck,
            Self::Order | Self::Leverage => LatencyKind::SubmitToAck,
        }
    }

    const fn safety_critical(self) -> bool {
        matches!(self, Self::Cancel | Self::Deadman)
    }
}

enum SessionAction {
    Orders {
        quote_seq: u64,
        requests: Vec<LiveOrderRequest>,
    },
    CancelCloids(Vec<String>),
    CancelOids(Vec<u64>),
    UpdateLeverage {
        leverage: u32,
        is_cross: bool,
    },
    ScheduleCancel(Option<u64>),
}

impl SessionAction {
    const fn purpose(&self) -> ActionPurpose {
        match self {
            Self::Orders { .. } => ActionPurpose::Order,
            Self::CancelCloids(_) | Self::CancelOids(_) => ActionPurpose::Cancel,
            Self::UpdateLeverage { .. } => ActionPurpose::Leverage,
            Self::ScheduleCancel(_) => ActionPurpose::Deadman,
        }
    }
}

struct SessionCommand {
    action: SessionAction,
    response: Option<oneshot::Sender<Result<ActionOutcome>>>,
    enqueued_ns: u64,
}

#[derive(Clone)]
pub struct HyperliquidSessionHandle {
    commands: mpsc::Sender<SessionCommand>,
    priority_commands: mpsc::Sender<SessionCommand>,
    clock: Arc<ProcessClock>,
    healthy: Arc<AtomicBool>,
    drop_next_response: Arc<AtomicBool>,
    fault_injection_allowed: bool,
}

impl HyperliquidSessionHandle {
    #[cfg(test)]
    pub(crate) fn test_stub(clock: Arc<ProcessClock>) -> Self {
        let (commands, command_rx) = mpsc::channel(1);
        let (priority_commands, priority_command_rx) = mpsc::channel(1);
        std::mem::forget(command_rx);
        std::mem::forget(priority_command_rx);
        Self {
            commands,
            priority_commands,
            clock,
            healthy: Arc::new(AtomicBool::new(true)),
            drop_next_response: Arc::new(AtomicBool::new(false)),
            fault_injection_allowed: false,
        }
    }

    pub fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    pub fn inject_drop_next_response(&self) -> Result<()> {
        if !self.fault_injection_allowed {
            bail!("live response fault injection is acceptance-test only");
        }
        self.drop_next_response.store(true, Ordering::Release);
        Ok(())
    }

    pub async fn place_orders(
        &self,
        quote_seq: u64,
        requests: Vec<LiveOrderRequest>,
    ) -> Result<ActionOutcome> {
        self.call(SessionAction::Orders {
            quote_seq,
            requests,
        })
        .await
    }

    pub fn enqueue_orders(&self, quote_seq: u64, requests: Vec<LiveOrderRequest>) -> Result<()> {
        self.enqueue(SessionAction::Orders {
            quote_seq,
            requests,
        })
    }

    pub async fn cancel_cloids(&self, cloids: Vec<String>) -> Result<ActionOutcome> {
        self.call(SessionAction::CancelCloids(cloids)).await
    }

    pub fn enqueue_cancel_cloids(&self, cloids: Vec<String>) -> Result<()> {
        self.enqueue(SessionAction::CancelCloids(cloids))
    }

    pub async fn cancel_oids(&self, oids: Vec<u64>) -> Result<ActionOutcome> {
        self.call(SessionAction::CancelOids(oids)).await
    }

    pub async fn update_leverage(&self, leverage: u32, is_cross: bool) -> Result<ActionOutcome> {
        self.call(SessionAction::UpdateLeverage { leverage, is_cross })
            .await
    }

    pub async fn schedule_cancel(&self, time: Option<u64>) -> Result<ActionOutcome> {
        self.call(SessionAction::ScheduleCancel(time)).await
    }

    pub fn enqueue_schedule_cancel(&self, time: Option<u64>) -> Result<()> {
        self.enqueue(SessionAction::ScheduleCancel(time))
    }

    async fn call(&self, action: SessionAction) -> Result<ActionOutcome> {
        let (response, received) = oneshot::channel();
        self.sender_for(&action)
            .send(SessionCommand {
                action,
                response: Some(response),
                enqueued_ns: self.clock.now_ns(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Hyperliquid session actor stopped"))?;
        received
            .await
            .map_err(|_| anyhow::anyhow!("Hyperliquid session dropped action response"))?
    }

    fn enqueue(&self, action: SessionAction) -> Result<()> {
        self.sender_for(&action)
            .try_send(SessionCommand {
                action,
                response: None,
                enqueued_ns: self.clock.now_ns(),
            })
            .map_err(|error| {
                anyhow::anyhow!("Hyperliquid session command queue refused action: {error}")
            })
    }

    fn sender_for(&self, action: &SessionAction) -> &mpsc::Sender<SessionCommand> {
        if action.purpose().safety_critical() {
            &self.priority_commands
        } else {
            &self.commands
        }
    }
}

pub struct SessionSpawnArgs {
    pub network: Network,
    pub ws_url: String,
    pub instrument: InstrumentSpec,
    pub credentials: Arc<HyperliquidCredentials>,
    pub state: Arc<LiveStateStore>,
    pub config: LiveConfig,
    pub clock: Arc<ProcessClock>,
    pub latency: Arc<LatencyMonitor>,
    pub events: mpsc::Sender<SessionEvent>,
    pub shutdown: watch::Receiver<bool>,
    pub ping_interval: Duration,
    pub idle_timeout: Duration,
}

pub fn spawn_session(
    args: SessionSpawnArgs,
) -> (HyperliquidSessionHandle, tokio::task::JoinHandle<()>) {
    let (commands, command_rx) = mpsc::channel(64);
    let (priority_commands, priority_command_rx) = mpsc::channel(64);
    let healthy = Arc::new(AtomicBool::new(false));
    let drop_next_response = Arc::new(AtomicBool::new(false));
    let handle = HyperliquidSessionHandle {
        commands,
        priority_commands,
        clock: args.clock.clone(),
        healthy: healthy.clone(),
        drop_next_response: drop_next_response.clone(),
        fault_injection_allowed: args.config.mode == crate::config::LiveMode::AcceptanceTest,
    };
    let task = tokio::spawn(run_session(
        args,
        command_rx,
        priority_command_rx,
        healthy,
        drop_next_response,
    ));
    (handle, task)
}

async fn run_session(
    mut args: SessionSpawnArgs,
    mut commands: mpsc::Receiver<SessionCommand>,
    mut priority_commands: mpsc::Receiver<SessionCommand>,
    healthy: Arc<AtomicBool>,
    drop_next_response: Arc<AtomicBool>,
) {
    let mut nonce = match DurableNonceManager::new(args.state.clone()) {
        Ok(nonce) => nonce,
        Err(error) => {
            warn!(%error, "cannot initialize durable Hyperliquid nonce manager");
            return;
        }
    };
    let mut backoff_ms = 250_u64;
    let mut generation = 0_u64;
    loop {
        if *args.shutdown.borrow() {
            return;
        }
        match connect_async(&args.ws_url).await {
            Ok((socket, _)) => {
                generation = generation.saturating_add(1);
                backoff_ms = 250;
                let result = run_connected(
                    &mut args,
                    &mut commands,
                    &mut priority_commands,
                    &mut nonce,
                    &healthy,
                    &drop_next_response,
                    socket,
                    generation,
                )
                .await;
                healthy.store(false, Ordering::Release);
                if let Err(error) = result {
                    let reason = error.to_string();
                    let _ = args.state.update(|state| {
                        for order in state.orders.values_mut() {
                            if order.status == LiveOrderStatus::Sent {
                                order.status = LiveOrderStatus::UnknownOutcome;
                                order.last_error = Some(reason.clone());
                                order.last_update_ms = unix_ms();
                            }
                        }
                        Ok(())
                    });
                    let _ = args
                        .events
                        .send(SessionEvent::Disconnected {
                            generation,
                            received_ns: args.clock.now_ns(),
                            reason: reason.clone(),
                        })
                        .await;
                    warn!(%error, generation, "Hyperliquid live session interrupted");
                } else {
                    return;
                }
            }
            Err(error) => warn!(%error, "cannot connect Hyperliquid live WebSocket"),
        }
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
            changed = args.shutdown.changed() => {
                if changed.is_err() || *args.shutdown.borrow() {
                    return;
                }
            }
        }
        backoff_ms = backoff_ms.saturating_mul(2).min(8_000);
    }
}

struct Inflight {
    nonce: u64,
    purpose: ActionPurpose,
    sent_ns: u64,
    response: Option<oneshot::Sender<Result<ActionOutcome>>>,
    /// Positionally aligned with the orders in the signed wire action: entry `i`
    /// describes wire order `i`, and is `None` where the venue was asked to act
    /// on a cloid we do not track locally. Venue status arrays are indexed by
    /// wire position, so this alignment is what keeps a response attributed to
    /// the order it actually describes.
    cloids: Vec<Option<String>>,
}

struct InflightBook {
    entries: HashMap<u64, Inflight>,
    state: Arc<LiveStateStore>,
}

impl InflightBook {
    fn new(state: Arc<LiveStateStore>) -> Self {
        Self {
            entries: HashMap::new(),
            state,
        }
    }
}

impl Drop for InflightBook {
    fn drop(&mut self) {
        for (_, entry) in self.entries.drain() {
            let reason = "live WebSocket ended before action response";
            let _ = mark_unknown(&self.state, &entry.cloids, reason);
            let outcome = ActionOutcome::Unknown {
                nonce: entry.nonce,
                error: reason.to_owned(),
            };
            if let Some(response) = entry.response {
                let _ = response.send(Ok(outcome));
            }
        }
    }
}

async fn run_connected<S>(
    args: &mut SessionSpawnArgs,
    commands: &mut mpsc::Receiver<SessionCommand>,
    priority_commands: &mut mpsc::Receiver<SessionCommand>,
    nonce: &mut DurableNonceManager,
    healthy: &AtomicBool,
    drop_next_response: &AtomicBool,
    socket: tokio_tungstenite::WebSocketStream<S>,
    generation: u64,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (mut write, mut read) = socket.split();
    let mut rate = WebSocketRateLimiter::new(
        args.config.max_ws_messages_per_minute,
        args.config.cancel_reserve_fraction,
    );
    for subscription in account_subscriptions(
        args.credentials.account(),
        &args.instrument.dex,
        &args.instrument.symbol,
    ) {
        rate.consume(false)?;
        write
            .send(Message::Text(subscription_request(&subscription).into()))
            .await?;
    }
    deliver_event(&args.events, healthy, SessionEvent::Connected {
            generation,
            received_ns: args.clock.now_ns(),
        })?;
    let mut subscription_acks = 0_usize;
    let mut request_id = 1_u64;
    let mut inflight = InflightBook::new(args.state.clone());
    let mut ping = tokio::time::interval(args.ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_inbound = tokio::time::Instant::now();
    let mut timeout_check = tokio::time::interval(Duration::from_millis(50));
    timeout_check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut pending_ping_ns = None;
    loop {
        tokio::select! {
            incoming = read.next() => {
                let Some(incoming) = incoming else { bail!("live WebSocket ended") };
                last_inbound = tokio::time::Instant::now();
                match incoming? {
                    Message::Text(text) => {
                        // One malformed frame is data corruption, not a
                        // transport failure: tearing the socket down here
                        // turned every in-flight order into an unknown outcome
                        // and permanently invalidated the run on reconnect.
                        // Matches the public market stream's tolerance.
                        let mut root: serde_json::Value = match serde_json::from_str(text.as_ref())
                        {
                            Ok(root) => root,
                            Err(error) => {
                                let prefix: String =
                                    text.as_str().chars().take(96).collect();
                                warn!(%error, prefix, "malformed account frame ignored");
                                continue;
                            }
                        };
                        // Owned so the account-data arm can move `data` out of
                        // the DOM instead of deep-cloning it.
                        let channel = root
                            .get("channel")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        let received_ns = args.clock.now_ns();
                        match channel.as_str() {
                            "pong" => {
                                if let Some(sent_ns) = pending_ping_ns.take() {
                                    args.latency.record(
                                        LatencyKind::AccountWsPingRtt,
                                        received_ns.saturating_sub(sent_ns),
                                        received_ns,
                                    );
                                }
                            }
                            "subscriptionResponse" => {
                                subscription_acks += 1;
                                if subscription_acks == ACCOUNT_SUBSCRIPTION_COUNT {
                                    healthy.store(true, Ordering::Release);
                                    deliver_event(&args.events, healthy, SessionEvent::Ready { generation, received_ns })?;
                                }
                            }
                            "post" => {
                                if drop_next_response.swap(false, Ordering::AcqRel) {
                                    bail!("acceptance fault: discarded post response after venue write");
                                }
                                let Some(id) = root.pointer("/data/id").and_then(serde_json::Value::as_u64) else {
                                    continue;
                                };
                                let Some(entry) = inflight.entries.remove(&id) else { continue };
                                let payload = root
                                    .pointer("/data/response/payload")
                                    .cloned()
                                    .unwrap_or_else(|| root.pointer("/data/response").cloned().unwrap_or_default());
                                args.latency.record(
                                    entry.purpose.latency_kind(),
                                    received_ns.saturating_sub(entry.sent_ns),
                                    received_ns,
                                );
                                apply_action_response(
                                    &args.state,
                                    &entry.cloids,
                                    entry.purpose,
                                    &payload,
                                    received_ns,
                                )?;
                                let outcome = ActionOutcome::Response {
                                    nonce: entry.nonce,
                                    http_status: 200,
                                    body: payload,
                                };
                                deliver_event(&args.events, healthy, SessionEvent::ActionCompleted {
                                        purpose: entry.purpose,
                                        outcome: outcome.clone(),
                                        received_ns,
                                    })?;
                                if let Some(response) = entry.response {
                                    let _ = response.send(Ok(outcome));
                                }
                            }
                            "" => {}
                            _ => {
                                deliver_event(&args.events, healthy, SessionEvent::AccountData {
                                    generation,
                                    received_ns,
                                    channel: AccountChannel::from_wire(&channel),
                                    // Move the payload out of the parsed frame
                                    // rather than deep-cloning the DOM again.
                                    data: root
                                        .get_mut("data")
                                        .map(serde_json::Value::take)
                                        .unwrap_or_default(),
                                })?;
                            }
                        }
                    }
                    Message::Ping(payload) => write.send(Message::Pong(payload)).await?,
                    Message::Close(frame) => bail!("server closed live WebSocket: {frame:?}"),
                    _ => {}
                }
            }
            command = next_session_command(priority_commands, commands) => {
                let Some(command) = command else {
                    bail!("all live-session command producers stopped");
                };
                let dequeued_ns = args.clock.now_ns();
                args.latency.record(
                    LatencyKind::ExecutionQueueWait,
                    dequeued_ns.saturating_sub(command.enqueued_ns),
                    dequeued_ns,
                );
                let purpose = command.action.purpose();
                if inflight.entries.len() >= args.config.max_inflight_posts {
                    let reason = "maximum live in-flight posts reached".to_owned();
                    if let Some(response) = command.response {
                        let _ = response.send(Err(anyhow::anyhow!(reason)));
                    } else {
                        deliver_event(&args.events, healthy, SessionEvent::ActionRefused {
                            purpose,
                            reason,
                            received_ns: args.clock.now_ns(),
                        })?;
                    }
                    continue;
                }
                if !healthy.load(Ordering::Acquire) && !purpose.safety_critical() {
                    let reason = "live session is not ready".to_owned();
                    if let Some(response) = command.response {
                        let _ = response.send(Err(anyhow::anyhow!(reason)));
                    } else {
                        deliver_event(&args.events, healthy, SessionEvent::ActionRefused {
                            purpose,
                            reason,
                            received_ns: args.clock.now_ns(),
                        })?;
                    }
                    continue;
                }
                if let Err(error) = rate.consume(purpose.safety_critical()) {
                    let reason = error.to_string();
                    if let Some(response) = command.response {
                        let _ = response.send(Err(error));
                    } else {
                        deliver_event(&args.events, healthy, SessionEvent::ActionRefused {
                            purpose,
                            reason,
                            received_ns: args.clock.now_ns(),
                        })?;
                    }
                    continue;
                }
                let id = request_id;
                request_id = request_id.saturating_add(1);
                let nonce_value = nonce.next_nonce()?;
                let preparation_started_ns = args.clock.now_ns();
                let cloids = prepare_action_state(&args.state, &command.action, nonce_value, id)?;
                let signing_started_ns = args.clock.now_ns();
                args.latency.record(
                    LatencyKind::ActionPreparation,
                    signing_started_ns.saturating_sub(preparation_started_ns),
                    signing_started_ns,
                );
                let envelope = signed_action_envelope(args, &command.action, nonce_value)?;
                let write_started_ns = args.clock.now_ns();
                args.latency.record(
                    LatencyKind::Signing,
                    write_started_ns.saturating_sub(signing_started_ns),
                    write_started_ns,
                );
                inflight.entries.insert(id, Inflight {
                    nonce: nonce_value,
                    purpose,
                    sent_ns: write_started_ns,
                    response: command.response,
                    cloids: cloids.clone(),
                });
                let request = json!({
                    "method": "post",
                    "id": id,
                    "request": {"type": "action", "payload": envelope},
                });
                if let Err(error) = write.send(Message::Text(request.to_string().into())).await {
                    if let Some(entry) = inflight.entries.remove(&id) {
                        mark_unknown(&args.state, &entry.cloids, &error.to_string())?;
                        let outcome = ActionOutcome::Unknown {
                            nonce: entry.nonce,
                            error: error.to_string(),
                        };
                        deliver_event(&args.events, healthy, SessionEvent::ActionCompleted {
                                purpose: entry.purpose,
                                outcome: outcome.clone(),
                                received_ns: args.clock.now_ns(),
                            })?;
                        if let Some(response) = entry.response {
                            let _ = response.send(Ok(outcome));
                        }
                    }
                    return Err(error.into());
                }
                let written_ns = args.clock.now_ns();
                args.latency.record(
                    LatencyKind::SocketWrite,
                    written_ns.saturating_sub(write_started_ns),
                    written_ns,
                );
                args.latency.record(
                    LatencyKind::DecisionToSocketWrite,
                    written_ns.saturating_sub(command.enqueued_ns),
                    written_ns,
                );
                if let Some(entry) = inflight.entries.get_mut(&id) {
                    entry.sent_ns = written_ns;
                }
            }
            _ = ping.tick() => {
                // An exhausted message budget must not cost us the socket:
                // tearing it down turns every in-flight order into an unknown
                // outcome and permanently invalidates the run on reconnect.
                // Skip the ping, degrade health, and let the idle timeout act
                // if the connection really is gone.
                match rate.consume(true) {
                    Ok(()) => {
                        pending_ping_ns = Some(args.clock.now_ns());
                        write.send(Message::Text(application_ping().into())).await?;
                    }
                    Err(error) => {
                        healthy.store(false, Ordering::Release);
                        warn!(%error, "skipping application ping: WebSocket message budget exhausted");
                    }
                }
            }
            () = tokio::time::sleep_until(last_inbound + args.idle_timeout) => {
                bail!("live WebSocket idle timeout");
            }
            _ = timeout_check.tick() => {
                let now_ns = args.clock.now_ns();
                let timeout_ns = args.config.action_timeout_ms.saturating_mul(1_000_000);
                let timed_out: Vec<u64> = inflight.entries.iter()
                    .filter(|(_, entry)| now_ns.saturating_sub(entry.sent_ns) >= timeout_ns)
                    .map(|(id, _)| *id)
                    .collect();
                for id in timed_out {
                    if let Some(entry) = inflight.entries.remove(&id) {
                        mark_unknown(&args.state, &entry.cloids, "WebSocket post timeout")?;
                        let outcome = ActionOutcome::Unknown {
                            nonce: entry.nonce,
                            error: "WebSocket post timeout".to_owned(),
                        };
                        deliver_event(&args.events, healthy, SessionEvent::ActionCompleted {
                                purpose: entry.purpose,
                                outcome: outcome.clone(),
                                received_ns: now_ns,
                            })?;
                        if let Some(response) = entry.response {
                            let _ = response.send(Ok(outcome));
                        }
                    }
                }
            }
            changed = args.shutdown.changed() => {
                if changed.is_err() || *args.shutdown.borrow() {
                    let _ = write.send(Message::Close(None)).await;
                    for (_, entry) in inflight.entries.drain() {
                        mark_unknown(&args.state, &entry.cloids, "session shutdown with action in flight")?;
                        let outcome = ActionOutcome::Unknown {
                            nonce: entry.nonce,
                            error: "session shutdown with action in flight".to_owned(),
                        };
                        if let Some(response) = entry.response {
                            let _ = response.send(Ok(outcome));
                        }
                    }
                    return Ok(());
                }
            }
        }
    }
}

const fn session_event_kind(event: &SessionEvent) -> &'static str {
    match event {
        SessionEvent::Connected { .. } => "connected",
        SessionEvent::Ready { .. } => "ready",
        SessionEvent::AccountData { .. } => "account_data",
        SessionEvent::ActionCompleted { .. } => "action_completed",
        SessionEvent::ActionRefused { .. } => "action_refused",
        SessionEvent::Disconnected { .. } => "disconnected",
    }
}

/// Non-blocking session-event delivery for use inside the dispatch loop.
///
/// The event channel is drained by the strategy loop. Awaiting a send here
/// would let a slow consumer stall order — and priority-cancel — dispatch,
/// and can deadlock outright: the strategy loop awaits our oneshot responses,
/// and we would be awaiting its channel capacity. On a full channel the event
/// is dropped and the session degrades to unhealthy; the next authoritative
/// reconcile restores exact durable state from the venue.
fn deliver_event(
    events: &mpsc::Sender<SessionEvent>,
    healthy: &AtomicBool,
    event: SessionEvent,
) -> Result<()> {
    match events.try_send(event) {
        Ok(()) => Ok(()),
        Err(mpsc::error::TrySendError::Full(event)) => {
            healthy.store(false, Ordering::Release);
            warn!(
                kind = session_event_kind(&event),
                "session event channel full; event dropped, session degraded"
            );
            Ok(())
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            bail!("session event consumer stopped")
        }
    }
}

async fn next_session_command(
    priority: &mut mpsc::Receiver<SessionCommand>,
    regular: &mut mpsc::Receiver<SessionCommand>,
) -> Option<SessionCommand> {
    if let Ok(command) = priority.try_recv() {
        return Some(command);
    }
    tokio::select! {
        biased;
        command = priority.recv() => command,
        command = regular.recv() => command,
    }
}

fn signed_action_envelope(
    args: &SessionSpawnArgs,
    action: &SessionAction,
    nonce: u64,
) -> Result<serde_json::Value> {
    let expires = match action {
        SessionAction::Orders { .. } => Some(order_expires_after(
            nonce,
            unix_ms(),
            args.config.action_expiry_ms,
        )),
        _ => None,
    };
    let body = match action {
        SessionAction::Orders { requests, .. } => {
            let wire = order_action(&args.instrument, requests)?;
            sign_envelope(&wire, &args.credentials, args.network, nonce, expires)?.body
        }
        SessionAction::CancelCloids(cloids) => {
            let wire = cancel_by_cloid_action(args.instrument.asset_id, cloids)?;
            sign_envelope(&wire, &args.credentials, args.network, nonce, None)?.body
        }
        SessionAction::CancelOids(oids) => {
            let wire = cancel_by_oid_action(args.instrument.asset_id, oids)?;
            sign_envelope(&wire, &args.credentials, args.network, nonce, None)?.body
        }
        SessionAction::UpdateLeverage { leverage, is_cross } => {
            let wire = update_leverage_action(args.instrument.asset_id, *leverage, *is_cross);
            sign_envelope(&wire, &args.credentials, args.network, nonce, None)?.body
        }
        SessionAction::ScheduleCancel(time) => {
            let wire = schedule_cancel_action(*time);
            sign_envelope(&wire, &args.credentials, args.network, nonce, None)?.body
        }
    };
    Ok(body)
}

const fn order_expires_after(nonce: u64, now_ms: u64, ttl_ms: u64) -> u64 {
    if now_ms > nonce {
        now_ms.saturating_add(ttl_ms)
    } else {
        nonce.saturating_add(ttl_ms)
    }
}

/// Record local intent for an action and return per-wire-order tracking slots.
///
/// The returned vector is positionally aligned with the orders the signed action
/// carries, so `result[i]` describes wire order `i`. Untracked entries are
/// `None` rather than omitted: the venue answers with one status per wire order,
/// and dropping entries here would shift every later status onto the wrong
/// order.
fn prepare_action_state(
    store: &LiveStateStore,
    action: &SessionAction,
    nonce: u64,
    transport_id: u64,
) -> Result<Vec<Option<String>>> {
    let now_ms = unix_ms();
    match action {
        SessionAction::Orders {
            quote_seq,
            requests,
        } => store.update(|state| {
            let mut cloids = Vec::with_capacity(requests.len());
            for request in requests {
                cloids.push(Some(request.cloid.clone()));
                state.orders.insert(
                    request.cloid.clone(),
                    PersistedLiveOrder {
                        cloid: request.cloid.clone(),
                        quote_seq: *quote_seq,
                        side: request.side,
                        px_units: request.px_units,
                        original_qty_units: request.qty_units,
                        remaining_qty_units: request.qty_units,
                        reduce_only: request.reduce_only,
                        status: LiveOrderStatus::Sent,
                        nonce: Some(nonce),
                        transport_id: Some(transport_id),
                        oid: None,
                        prepared_at_ms: now_ms,
                        last_update_ms: now_ms,
                        last_error: None,
                    },
                );
            }
            Ok(cloids)
        }),
        SessionAction::CancelCloids(cloids) => {
            let requested = cloids.clone();
            store.update(|state| {
                let mut tracked = Vec::with_capacity(requested.len());
                for cloid in &requested {
                    // The wire action cancels every requested cloid, including
                    // ones we cannot transition locally, so every request
                    // contributes exactly one slot here.
                    let claimed = state.orders.get_mut(cloid).is_some_and(|order| {
                        if !order
                            .status
                            .can_transition_to(LiveOrderStatus::CancelPending)
                        {
                            return false;
                        }
                        order.status = LiveOrderStatus::CancelPending;
                        order.nonce = Some(nonce);
                        order.transport_id = Some(transport_id);
                        order.last_update_ms = now_ms;
                        true
                    });
                    tracked.push(claimed.then(|| cloid.clone()));
                }
                Ok(tracked)
            })
        }
        _ => Ok(Vec::new()),
    }
}

fn apply_action_response(
    store: &LiveStateStore,
    cloids: &[Option<String>],
    purpose: ActionPurpose,
    body: &serde_json::Value,
    received_ns: u64,
) -> Result<()> {
    if cloids.iter().all(Option::is_none) {
        return Ok(());
    }
    let now_ms = unix_ms();
    let statuses = body
        .pointer("/response/data/statuses")
        .and_then(serde_json::Value::as_array);
    // Statuses are attributed by wire position, so a differently sized array
    // cannot be matched to orders at all. Fail closed rather than guess: an
    // unknown outcome blocks `operationally_healthy` until the next
    // authoritative reconcile resolves each order against the venue.
    if statuses.is_some_and(|values| values.len() != cloids.len()) {
        return mark_unknown(
            store,
            cloids,
            "venue status array length did not match the submitted action",
        );
    }
    store.update(|state| {
        for (index, cloid) in cloids.iter().enumerate() {
            let Some(cloid) = cloid.as_deref() else {
                continue;
            };
            let Some(order) = state.orders.get_mut(cloid) else {
                continue;
            };
            let status = statuses.and_then(|values| values.get(index));
            let (next_status, oid, last_error) =
                if let Some(resting) = status.and_then(|value| value.get("resting")) {
                    (
                        LiveOrderStatus::Resting,
                        resting.get("oid").and_then(serde_json::Value::as_u64),
                        None,
                    )
                } else if let Some(filled) = status.and_then(|value| value.get("filled")) {
                    (
                        LiveOrderStatus::Filled,
                        filled.get("oid").and_then(serde_json::Value::as_u64),
                        None,
                    )
                } else if let Some(error) = status.and_then(|value| value.get("error")) {
                    (
                        if purpose == ActionPurpose::Order {
                            LiveOrderStatus::Rejected
                        } else {
                            LiveOrderStatus::UnknownOutcome
                        },
                        None,
                        error.as_str().map(ToOwned::to_owned),
                    )
                } else if status.is_some_and(|value| value.as_str() == Some("success")) {
                    (
                        if purpose == ActionPurpose::Cancel {
                            LiveOrderStatus::Canceled
                        } else {
                            LiveOrderStatus::UnknownOutcome
                        },
                        None,
                        None,
                    )
                } else {
                    (
                        LiveOrderStatus::UnknownOutcome,
                        None,
                        Some(format!("unrecognized action response at {received_ns}")),
                    )
                };
            if !order.status.can_transition_to(next_status) {
                continue;
            }
            order.status = next_status;
            if next_status.terminal() {
                order.remaining_qty_units = 0;
            }
            if oid.is_some() {
                order.oid = oid;
            }
            order.last_error = last_error;
            order.last_update_ms = now_ms;
        }
        Ok(())
    })
}

fn mark_unknown(store: &LiveStateStore, cloids: &[Option<String>], reason: &str) -> Result<()> {
    let now_ms = unix_ms();
    store.update(|state| {
        for cloid in cloids.iter().flatten() {
            if let Some(order) = state.orders.get_mut(cloid) {
                if !order
                    .status
                    .can_transition_to(LiveOrderStatus::UnknownOutcome)
                {
                    continue;
                }
                order.status = LiveOrderStatus::UnknownOutcome;
                order.last_error = Some(reason.to_owned());
                order.last_update_ms = now_ms;
            }
        }
        Ok(())
    })
}

struct WebSocketRateLimiter {
    maximum_per_minute: u64,
    regular_limit: u64,
    sent: VecDeque<SentWebSocketMessage>,
    /// Non-safety-critical entries currently in `sent`, maintained on push and
    /// pop so `consume` never scans the window (it previously counted up to
    /// the full window on every regular message).
    regular_in_window: u64,
}

struct SentWebSocketMessage {
    at: tokio::time::Instant,
    safety_critical: bool,
}

impl WebSocketRateLimiter {
    fn new(maximum_per_minute: u64, reserve_fraction: f64) -> Self {
        Self {
            maximum_per_minute,
            regular_limit: ((maximum_per_minute as f64) * (1.0 - reserve_fraction)).floor() as u64,
            sent: VecDeque::new(),
            regular_in_window: 0,
        }
    }

    fn consume(&mut self, safety_critical: bool) -> Result<()> {
        let now = tokio::time::Instant::now();
        let cutoff = now - Duration::from_secs(60);
        while self.sent.front().is_some_and(|sent| sent.at < cutoff) {
            if let Some(expired) = self.sent.pop_front() {
                if !expired.safety_critical {
                    self.regular_in_window = self.regular_in_window.saturating_sub(1);
                }
            }
        }
        if self.sent.len() as u64 >= self.maximum_per_minute {
            bail!("Hyperliquid WebSocket message rate limit reached");
        }
        if !safety_critical && self.regular_in_window >= self.regular_limit {
            bail!("Hyperliquid regular WebSocket message budget reached");
        }
        if !safety_critical {
            self.regular_in_window += 1;
        }
        self.sent.push_back(SentWebSocketMessage {
            at: now,
            safety_critical,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_preserves_safety_reserve() {
        let mut limiter = WebSocketRateLimiter::new(4, 0.25);
        limiter.consume(false).unwrap();
        limiter.consume(false).unwrap();
        limiter.consume(false).unwrap();
        assert!(limiter.consume(false).is_err());
        limiter.consume(true).unwrap();
        assert!(limiter.consume(true).is_err());
    }

    #[test]
    fn safety_messages_do_not_consume_the_regular_budget() {
        let mut limiter = WebSocketRateLimiter::new(6, 0.5);
        limiter.consume(true).unwrap();
        limiter.consume(true).unwrap();
        limiter.consume(true).unwrap();
        limiter.consume(false).unwrap();
        limiter.consume(false).unwrap();
        limiter.consume(false).unwrap();
        assert!(limiter.consume(false).is_err());
        assert!(limiter.consume(true).is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn rate_limiter_expires_exact_rolling_window() {
        let mut limiter = WebSocketRateLimiter::new(2, 0.5);
        limiter.consume(false).unwrap();
        assert!(limiter.consume(false).is_err());
        tokio::time::advance(Duration::from_secs(61)).await;
        limiter.consume(false).unwrap();
    }

    #[tokio::test]
    async fn safety_commands_are_dequeued_before_regular_orders() {
        let (regular_tx, mut regular_rx) = mpsc::channel(2);
        let (priority_tx, mut priority_rx) = mpsc::channel(2);
        regular_tx
            .send(SessionCommand {
                action: SessionAction::Orders {
                    quote_seq: 1,
                    requests: Vec::new(),
                },
                response: None,
                enqueued_ns: 1,
            })
            .await
            .unwrap();
        priority_tx
            .send(SessionCommand {
                action: SessionAction::CancelCloids(Vec::new()),
                response: None,
                enqueued_ns: 2,
            })
            .await
            .unwrap();
        let command = next_session_command(&mut priority_rx, &mut regular_rx)
            .await
            .unwrap();
        assert_eq!(command.action.purpose(), ActionPurpose::Cancel);
    }

    #[test]
    fn order_expiry_uses_send_time_when_nonce_lease_is_old() {
        assert_eq!(order_expires_after(1_000, 50_000, 5_000), 55_000);
        assert_eq!(order_expires_after(50_000, 1_000, 5_000), 55_000);
    }

    #[test]
    fn timeout_and_late_responses_never_resurrect_terminal_orders() {
        let directory = tempfile::tempdir().unwrap();
        let store = LiveStateStore::open(
            &directory.path().join("state.redb"),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            false,
        )
        .unwrap();
        let cloid = "0x434a4d4d000000000000000000000001".to_owned();
        store
            .update(|state| {
                state.orders.insert(
                    cloid.clone(),
                    PersistedLiveOrder {
                        cloid: cloid.clone(),
                        quote_seq: 1,
                        side: crate::types::Side::Buy,
                        px_units: 100_000,
                        original_qty_units: 100,
                        remaining_qty_units: 0,
                        reduce_only: false,
                        status: LiveOrderStatus::Filled,
                        nonce: Some(1),
                        transport_id: Some(1),
                        oid: Some(7),
                        prepared_at_ms: 1,
                        last_update_ms: 2,
                        last_error: None,
                    },
                );
                Ok(())
            })
            .unwrap();

        let tracked = prepare_action_state(
            &store,
            &SessionAction::CancelCloids(vec![cloid.clone()]),
            2,
            2,
        )
        .unwrap();
        // The terminal order keeps its wire slot but is not claimed for cancel.
        assert_eq!(tracked, vec![None]);
        let slots = vec![Some(cloid.clone())];
        mark_unknown(&store, &slots, "late timeout").unwrap();
        apply_action_response(
            &store,
            &slots,
            ActionPurpose::Order,
            &serde_json::json!({
                "response": {"data": {"statuses": [{"resting": {"oid": 8}}]}}
            }),
            3,
        )
        .unwrap();
        let order = &store.load_required().unwrap().orders[&cloid];
        assert_eq!(order.status, LiveOrderStatus::Filled);
        assert_eq!(order.remaining_qty_units, 0);
        assert_eq!(order.oid, Some(7));
    }

    /// A cancel batch keeps one wire slot per requested cloid even when an order
    /// cannot be claimed locally, so venue statuses stay attributed by position.
    /// Without the `None` placeholder the second order would inherit the first
    /// order's status.
    #[test]
    fn cancel_statuses_stay_aligned_when_an_order_is_already_terminal() {
        let directory = tempfile::tempdir().unwrap();
        let store = LiveStateStore::open(
            &directory.path().join("state.redb"),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            false,
        )
        .unwrap();
        let filled = "0x434a4d4d000000000000000000000001".to_owned();
        let resting = "0x434a4d4d000000000000000000000002".to_owned();
        let template = |cloid: &str, status: LiveOrderStatus| PersistedLiveOrder {
            cloid: cloid.to_owned(),
            quote_seq: 1,
            side: crate::types::Side::Buy,
            px_units: 100_000,
            original_qty_units: 100,
            remaining_qty_units: if status.terminal() { 0 } else { 100 },
            reduce_only: false,
            status,
            nonce: Some(1),
            transport_id: Some(1),
            oid: Some(7),
            prepared_at_ms: 1,
            last_update_ms: 2,
            last_error: None,
        };
        store
            .update(|state| {
                state
                    .orders
                    .insert(filled.clone(), template(&filled, LiveOrderStatus::Filled));
                state
                    .orders
                    .insert(resting.clone(), template(&resting, LiveOrderStatus::Resting));
                Ok(())
            })
            .unwrap();

        let slots = prepare_action_state(
            &store,
            &SessionAction::CancelCloids(vec![filled.clone(), resting.clone()]),
            2,
            2,
        )
        .unwrap();
        assert_eq!(slots, vec![None, Some(resting.clone())]);

        // Status 0 belongs to the already-filled order and must not reach the
        // resting one; status 1 is the cancel that actually succeeded.
        apply_action_response(
            &store,
            &slots,
            ActionPurpose::Cancel,
            &serde_json::json!({
                "response": {"data": {"statuses": [
                    {"error": "order was already filled"},
                    "success"
                ]}}
            }),
            3,
        )
        .unwrap();

        let orders = store.load_required().unwrap().orders;
        assert_eq!(orders[&filled].status, LiveOrderStatus::Filled);
        assert_eq!(orders[&filled].last_error, None);
        assert_eq!(orders[&resting].status, LiveOrderStatus::Canceled);
    }

    /// A status array we cannot attribute by position must fail closed rather
    /// than align by guesswork.
    #[test]
    fn mismatched_status_length_marks_every_slot_unknown() {
        let directory = tempfile::tempdir().unwrap();
        let store = LiveStateStore::open(
            &directory.path().join("state.redb"),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            false,
        )
        .unwrap();
        let cloid = "0x434a4d4d000000000000000000000003".to_owned();
        store
            .update(|state| {
                state.orders.insert(
                    cloid.clone(),
                    PersistedLiveOrder {
                        cloid: cloid.clone(),
                        quote_seq: 1,
                        side: crate::types::Side::Buy,
                        px_units: 100_000,
                        original_qty_units: 100,
                        remaining_qty_units: 100,
                        reduce_only: false,
                        status: LiveOrderStatus::Sent,
                        nonce: Some(1),
                        transport_id: Some(1),
                        oid: None,
                        prepared_at_ms: 1,
                        last_update_ms: 2,
                        last_error: None,
                    },
                );
                Ok(())
            })
            .unwrap();

        apply_action_response(
            &store,
            &[Some(cloid.clone())],
            ActionPurpose::Order,
            &serde_json::json!({
                "response": {"data": {"statuses": [
                    {"resting": {"oid": 8}},
                    {"resting": {"oid": 9}}
                ]}}
            }),
            3,
        )
        .unwrap();

        let order = &store.load_required().unwrap().orders[&cloid];
        assert_eq!(order.status, LiveOrderStatus::UnknownOutcome);
        assert_eq!(order.oid, None);
    }
}
