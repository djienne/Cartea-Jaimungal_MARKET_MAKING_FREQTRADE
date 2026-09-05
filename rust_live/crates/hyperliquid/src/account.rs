use crate::latency::{LatencyKind, LatencyMonitor};
use crate::lockfree::AsyncRing;
use crate::transport::{account_subscriptions, application_ping, subscription_request};
use crate::types::ProcessClock;
use anyhow::{bail, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AccountStreamEvent {
    Connected {
        generation: u64,
        received_ns: u64,
    },
    SubscriptionAcknowledged {
        generation: u64,
        received_ns: u64,
        subscription: serde_json::Value,
    },
    Data {
        generation: u64,
        received_ns: u64,
        channel: String,
        data: serde_json::Value,
    },
    Disconnected {
        generation: u64,
        received_ns: u64,
        reason: String,
    },
}

#[derive(Debug, Default)]
pub struct AccountStreamMetrics {
    messages: AtomicU64,
    subscription_acks: AtomicU64,
    pings_sent: AtomicU64,
    pongs_received: AtomicU64,
    protocol_pings_received: AtomicU64,
    reconnects: AtomicU64,
    invalid_messages: AtomicU64,
    dropped_events: AtomicU64,
    idle_timeouts: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccountStreamMetricsSnapshot {
    pub messages: u64,
    pub subscription_acks: u64,
    pub pings_sent: u64,
    pub pongs_received: u64,
    pub protocol_pings_received: u64,
    pub reconnects: u64,
    pub invalid_messages: u64,
    pub dropped_events: u64,
    pub idle_timeouts: u64,
}

impl AccountStreamMetrics {
    pub fn snapshot(&self) -> AccountStreamMetricsSnapshot {
        AccountStreamMetricsSnapshot {
            messages: self.messages.load(Ordering::Relaxed),
            subscription_acks: self.subscription_acks.load(Ordering::Relaxed),
            pings_sent: self.pings_sent.load(Ordering::Relaxed),
            pongs_received: self.pongs_received.load(Ordering::Relaxed),
            protocol_pings_received: self.protocol_pings_received.load(Ordering::Relaxed),
            reconnects: self.reconnects.load(Ordering::Relaxed),
            invalid_messages: self.invalid_messages.load(Ordering::Relaxed),
            dropped_events: self.dropped_events.load(Ordering::Relaxed),
            idle_timeouts: self.idle_timeouts.load(Ordering::Relaxed),
        }
    }
}

pub struct AccountStreamArgs {
    pub ws_url: String,
    pub account: String,
    pub dex: String,
    pub symbol: String,
    pub events: Arc<AsyncRing<AccountStreamEvent>>,
    pub clock: Arc<ProcessClock>,
    pub latency: Option<Arc<LatencyMonitor>>,
    pub metrics: Arc<AccountStreamMetrics>,
    pub healthy: Arc<AtomicBool>,
    pub shutdown: watch::Receiver<bool>,
    pub ping_interval: Duration,
    pub idle_timeout: Duration,
    /// Deadline for connection establishment and subscription/heartbeat/close writes.
    pub connect_timeout: Duration,
}

pub async fn run_account_stream(mut args: AccountStreamArgs) {
    let mut backoff_ms = 250_u64;
    let mut generation = 0_u64;
    loop {
        if *args.shutdown.borrow() {
            return;
        }
        let attempt = tokio::time::timeout(args.connect_timeout, connect_async(&args.ws_url)).await;
        match attempt {
            Ok(Ok((socket, _))) => {
                generation = generation.saturating_add(1);
                if generation > 1 {
                    args.healthy.store(false, Ordering::Release);
                    args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                }
                backoff_ms = 250;
                info!(account = %args.account, generation, "account WebSocket connected");
                match run_connected(&mut args, socket, generation).await {
                    Ok(()) => return,
                    Err(error) => {
                        args.healthy.store(false, Ordering::Release);
                        let _ = push_event(
                            &args,
                            AccountStreamEvent::Disconnected {
                                generation,
                                received_ns: args.clock.now_ns(),
                                reason: error.to_string(),
                            },
                        );
                        warn!(%error, generation, "account WebSocket interrupted");
                    }
                }
            }
            Ok(Err(error)) => {
                args.healthy.store(false, Ordering::Release);
                args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                warn!(%error, "cannot connect account WebSocket");
            }
            Err(_elapsed) => {
                args.healthy.store(false, Ordering::Release);
                args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                warn!(
                    timeout_ms = args.connect_timeout.as_millis(),
                    "account WebSocket connect timed out"
                );
            }
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

async fn run_connected<S>(
    args: &mut AccountStreamArgs,
    socket: tokio_tungstenite::WebSocketStream<S>,
    generation: u64,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (mut write, mut read) = socket.split();
    for subscription in account_subscriptions(&args.account, &args.dex, &args.symbol) {
        tokio::time::timeout(
            args.connect_timeout,
            write.send(Message::Text(subscription_request(&subscription).into())),
        )
        .await
        .context("account WebSocket subscription write timed out")??;
    }
    push_event(
        args,
        AccountStreamEvent::Connected {
            generation,
            received_ns: args.clock.now_ns(),
        },
    )?;
    let mut ping = tokio::time::interval(args.ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_inbound = tokio::time::Instant::now();
    let mut pending_ping = None;
    loop {
        tokio::select! {
            incoming = read.next() => {
                let Some(incoming) = incoming else { bail!("account stream ended") };
                last_inbound = tokio::time::Instant::now();
                match incoming? {
                    Message::Text(text) => {
                        args.metrics.messages.fetch_add(1, Ordering::Relaxed);
                        let Ok(root) =
                            serde_json::from_str::<serde_json::Value>(text.as_ref())
                        else {
                            args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                            continue;
                        };
                        let channel = root
                            .get("channel")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default();
                        let received_ns = args.clock.now_ns();
                        if channel == "pong" {
                            if let Some(sent_ns) = pending_ping.take() {
                                if let Some(latency) = &args.latency {
                                    latency.record(
                                        LatencyKind::AccountWsPingRtt,
                                        received_ns.saturating_sub(sent_ns),
                                        received_ns,
                                    );
                                }
                            }
                            args.metrics.pongs_received.fetch_add(1, Ordering::Relaxed);
                        } else if channel == "subscriptionResponse" {
                            args.metrics.subscription_acks.fetch_add(1, Ordering::Relaxed);
                            push_event(
                                args,
                                AccountStreamEvent::SubscriptionAcknowledged {
                                    generation,
                                    received_ns,
                                    subscription: root.get("data").cloned().unwrap_or_default(),
                                },
                            )?;
                        } else if !channel.is_empty() {
                            push_event(
                                args,
                                AccountStreamEvent::Data {
                                    generation,
                                    received_ns,
                                    channel: channel.to_owned(),
                                    data: root.get("data").cloned().unwrap_or_default(),
                                },
                            )?;
                        } else {
                            args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Message::Ping(payload) => {
                        tokio::time::timeout(args.connect_timeout, write.send(Message::Pong(payload)))
                            .await
                            .context("account WebSocket pong write timed out")??;
                        args.metrics
                            .protocol_pings_received
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Message::Close(frame) => bail!("server closed account stream: {frame:?}"),
                    _ => {}
                }
            }
            _ = ping.tick() => {
                pending_ping = Some(args.clock.now_ns());
                tokio::time::timeout(args.connect_timeout, write.send(Message::Text(application_ping().into())))
                    .await
                    .context("account WebSocket ping write timed out")??;
                args.metrics.pings_sent.fetch_add(1, Ordering::Relaxed);
            }
            () = tokio::time::sleep_until(last_inbound + args.idle_timeout) => {
                args.metrics.idle_timeouts.fetch_add(1, Ordering::Relaxed);
                bail!("account WebSocket heartbeat or inbound data timed out");
            }
            changed = args.shutdown.changed() => {
                if changed.is_err() || *args.shutdown.borrow() {
                    let _ = tokio::time::timeout(args.connect_timeout, write.send(Message::Close(None))).await;
                    return Ok(());
                }
            }
        }
    }
}

fn push_event(args: &AccountStreamArgs, event: AccountStreamEvent) -> Result<()> {
    if args.events.try_push(event).is_err() {
        args.metrics.dropped_events.fetch_add(1, Ordering::Relaxed);
        args.healthy.store(false, Ordering::Release);
        bail!("account-event ring saturated");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::{tungstenite::protocol::Role, WebSocketStream};

    #[tokio::test]
    async fn stalled_account_subscription_write_times_out() {
        let (client, _server) = tokio::io::duplex(1);
        let socket = WebSocketStream::from_raw_socket(client, Role::Client, None).await;
        let (_shutdown, shutdown) = watch::channel(false);
        let mut args = AccountStreamArgs {
            ws_url: String::new(),
            account: "0x1111111111111111111111111111111111111111".to_owned(),
            dex: String::new(),
            symbol: "CASHCAT".to_owned(),
            events: Arc::new(AsyncRing::new(8)),
            clock: Arc::new(ProcessClock::default()),
            latency: None,
            metrics: Arc::new(AccountStreamMetrics::default()),
            healthy: Arc::new(AtomicBool::new(false)),
            shutdown,
            ping_interval: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_millis(50),
        };
        let error =
            tokio::time::timeout(Duration::from_secs(1), run_connected(&mut args, socket, 1))
                .await
                .expect("stalled subscription write prevented the deadline")
                .unwrap_err();
        assert!(error.to_string().contains("write timed out"), "{error:#}");
    }
}
