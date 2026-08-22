use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::lockfree::{AsyncRing, AtomicBbo, HotPathSignal, HOT_SIGNAL_MARKET};
use crate::metrics::Metrics;
use crate::types::{MarketEvent, ProcessClock};
use anyhow::{bail, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn};

pub struct MarketStreamArgs {
    pub ws_url: String,
    pub instrument: InstrumentSpec,
    pub latest_bbo: Arc<AtomicBbo>,
    pub events: Arc<AsyncRing<MarketEvent>>,
    pub signal: Arc<HotPathSignal>,
    pub clock: Arc<ProcessClock>,
    pub metrics: Arc<Metrics>,
    pub latency: Option<Arc<LatencyMonitor>>,
    pub scientifically_valid: Arc<AtomicBool>,
    pub shutdown: watch::Receiver<bool>,
    pub ping_interval: Duration,
    pub idle_timeout: Duration,
}

pub async fn run_market_stream(mut args: MarketStreamArgs) {
    let mut backoff_ms = 250_u64;
    let mut connected_once = false;
    loop {
        if *args.shutdown.borrow() {
            return;
        }
        match connect_async(&args.ws_url).await {
            Ok((socket, _)) => {
                if connected_once {
                    args.scientifically_valid.store(false, Ordering::Release);
                }
                connected_once = true;
                backoff_ms = 250;
                info!(symbol = %args.instrument.symbol, "public market WebSocket connected");
                match run_connected(&mut args, socket).await {
                    Ok(()) => return,
                    Err(error) => {
                        args.scientifically_valid.store(false, Ordering::Release);
                        args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                        warn!(%error, "public market stream interrupted; dry-run evidence invalidated");
                    }
                }
            }
            Err(error) => {
                args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                warn!(%error, "cannot connect public market WebSocket");
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
    args: &mut MarketStreamArgs,
    socket: tokio_tungstenite::WebSocketStream<S>,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let connected_at_ms = crate::types::unix_ms();
    let (mut write, mut read) = socket.split();
    for subscription_type in ["bbo", "trades", "l2Book"] {
        write
            .send(Message::Text(
                json!({
                    "method": "subscribe",
                    "subscription": {"type": subscription_type, "coin": args.instrument.symbol}
                })
                .to_string()
                .into(),
            ))
            .await?;
    }
    let mut ping = tokio::time::interval(args.ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_inbound = tokio::time::Instant::now();
    let mut trade_stream_live = false;
    let mut pending_application_ping_ns = None;
    loop {
        tokio::select! {
            incoming = read.next() => {
                let Some(incoming) = incoming else { bail!("market stream ended") };
                last_inbound = tokio::time::Instant::now();
                match incoming? {
                    Message::Text(text) => {
                        args.metrics.market_messages.fetch_add(1, Ordering::Relaxed);
                        let Ok(root) = serde_json::from_str::<serde_json::Value>(text.as_ref()) else {
                            args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                            continue;
                        };
                        if root.get("channel").and_then(serde_json::Value::as_str) == Some("pong") {
                            let pong_ns = args.clock.now_ns();
                            if let (Some(sent_ns), Some(latency)) =
                                (pending_application_ping_ns.take(), args.latency.as_ref())
                            {
                                latency.record(
                                    LatencyKind::PublicWsPingRtt,
                                    pong_ns.saturating_sub(sent_ns),
                                    pong_ns,
                                );
                            }
                            args.metrics
                                .application_pongs_received
                                .fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let recv_ns = args.clock.now_ns();
                        if let Some(bbo) = super::wire::parse_bbo(&root, &args.instrument, recv_ns) {
                            if bbo.is_valid() {
                                args.latest_bbo.store(bbo);
                                push_causal(args, MarketEvent::Bbo(bbo))?;
                                args.metrics.bbo_updates.fetch_add(1, Ordering::Relaxed);
                                args.signal.notify(HOT_SIGNAL_MARKET);
                            } else {
                                args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        let is_trade_frame = root
                            .get("channel")
                            .and_then(serde_json::Value::as_str)
                            == Some("trades");
                        let initial_trade_frame = is_trade_frame && !trade_stream_live;
                        for trade in super::wire::parse_trades(&root, &args.instrument, recv_ns) {
                            let startup_backlog =
                                trade.exchange_ms.saturating_add(2_000) < connected_at_ms;
                            if initial_trade_frame && startup_backlog {
                                args.metrics
                                    .historical_trade_prints_ignored
                                    .fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            if !initial_trade_frame
                                && crate::types::unix_ms().saturating_sub(trade.exchange_ms) > 2_000
                            {
                                args.scientifically_valid.store(false, Ordering::Release);
                                bail!("live trade arrived more than two seconds late");
                            }
                            push_causal(args, MarketEvent::Trade(trade))?;
                            args.metrics.trade_prints.fetch_add(1, Ordering::Relaxed);
                        }
                        if is_trade_frame {
                            trade_stream_live = true;
                        }
                        if let Some(book) = super::wire::parse_book(&root, &args.instrument, recv_ns) {
                            push_causal(args, MarketEvent::Book(book))?;
                            args.metrics.book_updates.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Message::Ping(payload) => {
                        write.send(Message::Pong(payload)).await?;
                        args.metrics
                            .protocol_pings_received
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Message::Close(frame) => bail!("server closed market stream: {frame:?}"),
                    _ => {}
                }
            }
            _ = ping.tick() => {
                pending_application_ping_ns = Some(args.clock.now_ns());
                write.send(Message::Text(json!({"method":"ping"}).to_string().into())).await?;
                args.metrics
                    .application_pings_sent
                    .fetch_add(1, Ordering::Relaxed);
            }
            () = tokio::time::sleep_until(last_inbound + args.idle_timeout) => {
                args.metrics
                    .ws_idle_timeouts
                    .fetch_add(1, Ordering::Relaxed);
                bail!("no inbound public WebSocket frame before idle timeout");
            }
            changed = args.shutdown.changed() => {
                if changed.is_err() || *args.shutdown.borrow() {
                    let _ = write.send(Message::Close(None)).await;
                    return Ok(());
                }
            }
        }
    }
}

fn push_causal(args: &MarketStreamArgs, event: MarketEvent) -> Result<()> {
    if args.events.try_push(event).is_err() {
        args.metrics
            .dropped_causal_events
            .fetch_add(1, Ordering::Relaxed);
        args.scientifically_valid.store(false, Ordering::Release);
        bail!("causal market-event ring saturated");
    }
    Ok(())
}
