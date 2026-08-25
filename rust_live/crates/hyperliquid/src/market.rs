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
    /// A genuinely new trade print older than this on arrival ends the stream
    /// (it reconnects). Replayed prints that predate the connection are
    /// ignored before this check and never trigger it.
    ///
    /// Has its own `runtime.max_trade_lag_ms` setting. It used to be fed from
    /// `runtime.market_stale_ms` on the theory that a separate knob "would only
    /// ever be set to the same value", but the two answer different questions:
    /// `market_stale_ms` is "is the top-of-book fresh enough to quote from",
    /// while this is "is a new trade so late the feed is broken". They have no
    /// reason to share a value.
    pub max_trade_lag_ms: u64,
}

pub async fn run_market_stream(mut args: MarketStreamArgs) {
    let mut backoff_ms = 250_u64;
    let mut connected_once = false;
    // When the stream went away, so the gap can be measured on the way back in.
    // A gap is missing data and must be recorded, but it is not the same thing
    // as event loss: a run that reconnected in three seconds is incomplete by a
    // knowable amount, whereas a saturated causal ring means the simulation
    // processed the wrong sequence and is simply wrong. Only the latter is
    // hard-invalid; gaps are counted and judged against a threshold at report
    // time (`AppConfig::feed_health_verdict`).
    let mut disconnected_at_ms: Option<u64> = None;
    loop {
        if *args.shutdown.borrow() {
            return;
        }
        match connect_async(&args.ws_url).await {
            Ok((socket, _)) => {
                if connected_once {
                    if let Some(since) = disconnected_at_ms.take() {
                        let gap_ms = crate::types::unix_ms().saturating_sub(since);
                        args.metrics.feed_gaps.fetch_add(1, Ordering::Relaxed);
                        args.metrics
                            .feed_downtime_ms
                            .fetch_add(gap_ms, Ordering::Relaxed);
                        args.metrics
                            .feed_longest_gap_ms
                            .fetch_max(gap_ms, Ordering::Relaxed);
                        info!(
                            gap_ms,
                            symbol = %args.instrument.symbol,
                            "public market feed gap closed"
                        );
                    }
                }
                connected_once = true;
                backoff_ms = 250;
                info!(symbol = %args.instrument.symbol, "public market WebSocket connected");
                match run_connected(&mut args, socket).await {
                    Ok(()) => return,
                    Err(error) => {
                        disconnected_at_ms.get_or_insert_with(crate::types::unix_ms);
                        args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                        warn!(%error, "public market stream interrupted; measuring the gap");
                    }
                }
            }
            Err(error) => {
                disconnected_at_ms.get_or_insert_with(crate::types::unix_ms);
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
    let mut pending_application_ping_ns = None;
    loop {
        tokio::select! {
            incoming = read.next() => {
                let Some(incoming) = incoming else { bail!("market stream ended") };
                last_inbound = tokio::time::Instant::now();
                match incoming? {
                    Message::Text(text) => {
                        args.metrics.market_messages.fetch_add(1, Ordering::Relaxed);
                        let recv_ns = args.clock.now_ns();
                        match super::wire::parse_public_frame(
                            text.as_str(),
                            &args.instrument,
                            recv_ns,
                        ) {
                            super::wire::PublicFrame::Invalid => {
                                args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                            }
                            super::wire::PublicFrame::Pong => {
                                if let (Some(sent_ns), Some(latency)) =
                                    (pending_application_ping_ns.take(), args.latency.as_ref())
                                {
                                    latency.record(
                                        LatencyKind::PublicWsPingRtt,
                                        recv_ns.saturating_sub(sent_ns),
                                        recv_ns,
                                    );
                                }
                                args.metrics
                                    .application_pongs_received
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            super::wire::PublicFrame::Bbo(bbo) => {
                                if bbo.is_valid() {
                                    args.latest_bbo.store(bbo);
                                    push_causal(args, MarketEvent::Bbo(bbo))?;
                                    args.metrics.bbo_updates.fetch_add(1, Ordering::Relaxed);
                                    args.signal.notify(HOT_SIGNAL_MARKET);
                                } else {
                                    args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            super::wire::PublicFrame::Trades(trades) => {
                                for trade in trades {
                                    // The venue replays recent trades after a
                                    // re-subscribe, stamped with their original
                                    // exchange time. Those predate this
                                    // connection, so they say nothing about how
                                    // fresh the feed is now and must never
                                    // justify tearing it down.
                                    //
                                    // This used to apply only to the FIRST
                                    // frame after connecting, but backfill
                                    // spans several frames: frame two onward
                                    // reached the lag check below and bailed,
                                    // which forced a reconnect, which replayed
                                    // more backfill. Measured over 183,344
                                    // CASHCAT trades, 94% of "late" prints
                                    // arrived within 2 s of another in 44
                                    // bursts of ~21 -- the loop, not a slow
                                    // feed. The body of the distribution is
                                    // fast (p50 378 ms, p99 2.4 s).
                                    let predates_connection =
                                        trade.exchange_ms.saturating_add(2_000) < connected_at_ms;
                                    if predates_connection {
                                        args.metrics
                                            .historical_trade_prints_ignored
                                            .fetch_add(1, Ordering::Relaxed);
                                        continue;
                                    }
                                    // The 2 s above is clock-skew tolerance for
                                    // the *ignore* decision, and it left a hole:
                                    // a replayed trade from up to 2 s before we
                                    // connected is fed through here, and if the
                                    // backfill burst takes more than ~3 s to
                                    // arrive it is already "5000ms late" and
                                    // bails -- reconnect, new backfill, another
                                    // near-boundary trade, bail again. Measured
                                    // live on 2026-08-25: 37 gaps in 15.9 h, the
                                    // last 21 of them every ~35 s, each
                                    // reconnect suppressing ~28 replays while
                                    // live trades kept flowing normally, and the
                                    // downtime fraction pinned at exactly the 5%
                                    // invalidation threshold.
                                    //
                                    // So the boundaries are separate on purpose:
                                    // whether a trade is FED is skew-tolerant,
                                    // whether it may KILL THE FEED is not. Any
                                    // trade born at or before the connection
                                    // instant is replay by definition and can
                                    // say nothing about current freshness.
                                    let born_before_connection =
                                        trade.exchange_ms <= connected_at_ms;
                                    // Only a genuinely new print can show the
                                    // feed has fallen behind.
                                    if !born_before_connection
                                        && crate::types::unix_ms().saturating_sub(trade.exchange_ms)
                                            > args.max_trade_lag_ms
                                    {
                                        // The reconnect that follows measures
                                        // this as a gap. Not event loss.
                                        bail!(
                                            "live trade arrived more than {}ms late",
                                            args.max_trade_lag_ms
                                        );
                                    }
                                    push_causal(args, MarketEvent::Trade(trade))?;
                                    args.metrics.trade_prints.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            super::wire::PublicFrame::Book(book) => {
                                push_causal(args, MarketEvent::Book(book))?;
                                args.metrics.book_updates.fetch_add(1, Ordering::Relaxed);
                            }
                            super::wire::PublicFrame::Other => {}
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
