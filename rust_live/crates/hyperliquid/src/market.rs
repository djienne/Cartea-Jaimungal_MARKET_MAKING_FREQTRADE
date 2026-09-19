use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::lockfree::{AsyncRing, BboWriter, HotPathSignal, HOT_SIGNAL_MARKET};
use crate::metrics::Metrics;
use crate::types::{Bbo, BookSnapshot, MarketEvent, ProcessClock};
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
    pub latest_bbo: BboWriter,
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
    pub max_bbo_lag_ms: u64,
    /// How long a single connect attempt may take before it counts as failed.
    ///
    /// Without this the reconnect loop below is only as reliable as the host's
    /// network stack: `connect_async` has no internal deadline, so a wedged
    /// stack makes it hang rather than return an error, and the loop never
    /// reaches its own retry. That is not hypothetical -- on 2026-08-26 one
    /// such call hung for 19.65 h. The backoff caps at 8 s, so a loop that was
    /// genuinely retrying would have logged thousands of failures; it logged
    /// none, which is how a silent hang is told apart from a noisy outage.
    pub connect_timeout: Duration,
}

/// Record that the public stream is down, both locally and where the stats loop
/// can read it. Idempotent: the *first* moment of an outage is the one that
/// matters, so a run of failed retries keeps the original timestamp.
fn mark_disconnected(args: &MarketStreamArgs, disconnected_at_ms: &mut Option<u64>) {
    let since = *disconnected_at_ms.get_or_insert_with(crate::types::unix_ms);
    args.metrics
        .feed_disconnected_since_ms
        .store(since, Ordering::Relaxed);
    args.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
}

pub async fn run_market_stream(mut args: MarketStreamArgs) {
    let mut backoff_ms = 250_u64;
    // When the stream went away, so the gap can be measured on the way back in.
    // A gap is missing data and must be recorded, but it is not the same thing
    // as event loss: a run that reconnected in three seconds is incomplete by a
    // knowable amount, whereas a saturated causal ring means the simulation
    // processed the wrong sequence and is simply wrong. Only the latter is
    // hard-invalid; gaps are counted and judged against a threshold at report
    // time (`FeedHealth::failures`, applied on every leaderboard write).
    //
    // `disconnected_at_ms` is mirrored into `metrics.feed_disconnected_since_ms`
    // so the stats loop can see an outage while it is still open. Keeping it
    // only here is what let a 19.65 h blackout look like a healthy heartbeat.
    let mut disconnected_at_ms: Option<u64> = None;
    loop {
        if *args.shutdown.borrow() {
            return;
        }
        let attempt = tokio::time::timeout(args.connect_timeout, connect_async(&args.ws_url)).await;
        match attempt {
            Ok(Ok((socket, _))) => {
                args.metrics
                    .feed_connected_at_ns
                    .store(args.clock.now_ns(), Ordering::Release);
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
                args.metrics
                    .feed_disconnected_since_ms
                    .store(0, Ordering::Relaxed);
                backoff_ms = 250;
                info!(symbol = %args.instrument.symbol, "public market WebSocket connected");
                match run_connected(&mut args, socket).await {
                    Ok(()) => return,
                    Err(error) => {
                        mark_disconnected(&args, &mut disconnected_at_ms);
                        warn!(%error, "public market stream interrupted; measuring the gap");
                    }
                }
            }
            Ok(Err(error)) => {
                mark_disconnected(&args, &mut disconnected_at_ms);
                warn!(%error, "cannot connect public market WebSocket");
            }
            Err(_elapsed) => {
                mark_disconnected(&args, &mut disconnected_at_ms);
                warn!(
                    timeout_ms = args.connect_timeout.as_millis(),
                    "public market WebSocket connect timed out"
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
    args: &mut MarketStreamArgs,
    socket: tokio_tungstenite::WebSocketStream<S>,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let connected_at_ms = crate::types::unix_ms();
    let (mut write, mut read) = socket.split();
    for subscription_type in ["bbo", "trades", "l2Book"] {
        tokio::time::timeout(
            args.connect_timeout,
            write.send(Message::Text(
                json!({
                    "method": "subscribe",
                    "subscription": {"type": subscription_type, "coin": args.instrument.symbol}
                })
                .to_string()
                .into(),
            )),
        )
        .await??;
    }
    let mut ping = tokio::time::interval(args.ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_bbo_update = tokio::time::Instant::now();
    let mut last_exchange_ms = 0;
    let mut pending_application_ping_ns = None;
    loop {
        tokio::select! {
            incoming = read.next() => {
                let Some(incoming) = incoming else { bail!("market stream ended") };
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
                                if publish_bbo(args, bbo, None, &mut last_exchange_ms)? {
                                    last_bbo_update = tokio::time::Instant::now();
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
                                    let lag_ms =
                                        crate::types::unix_ms().saturating_sub(trade.exchange_ms);
                                    if !born_before_connection && lag_ms > args.max_trade_lag_ms {
                                        // The reconnect that follows measures
                                        // this as a gap. Not event loss.
                                        //
                                        // The measured lag is in the message on
                                        // purpose: choosing this threshold needs
                                        // the distribution of what actually
                                        // trips it, and "more than 5000ms" tells
                                        // you nothing about whether the right
                                        // number is 6 s or 60 s.
                                        bail!(
                                            "live trade arrived {}ms late (limit {}ms, born {}ms after connect)",
                                            lag_ms,
                                            args.max_trade_lag_ms,
                                            trade.exchange_ms.saturating_sub(connected_at_ms)
                                        );
                                    }
                                    push_causal(args, MarketEvent::Trade(trade))?;
                                    args.metrics.trade_prints.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            super::wire::PublicFrame::Book(book) => {
                                // BBO is change-only; L2 snapshots also prove an
                                // unchanged touch is current. Preserve venue time.
                                let Some(bbo) = book.bbo() else {
                                    args.metrics.invalid_messages.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                };
                                if publish_bbo(args, bbo, Some(book), &mut last_exchange_ms)? {
                                    last_bbo_update = tokio::time::Instant::now();
                                }
                            }
                            super::wire::PublicFrame::Other => {}
                        }
                    }
                    Message::Ping(payload) => {
                        tokio::time::timeout(args.connect_timeout, write.send(Message::Pong(payload))).await??;
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
                tokio::time::timeout(args.connect_timeout, write.send(Message::Text(json!({"method":"ping"}).to_string().into()))).await??;
                args.metrics
                    .application_pings_sent
                    .fetch_add(1, Ordering::Relaxed);
            }
            () = tokio::time::sleep_until(last_bbo_update + args.idle_timeout) => {
                args.metrics
                    .ws_idle_timeouts
                    .fetch_add(1, Ordering::Relaxed);
                bail!("no public BBO before idle timeout");
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

fn publish_bbo(
    args: &MarketStreamArgs,
    bbo: Bbo,
    book: Option<BookSnapshot>,
    last_exchange_ms: &mut u64,
) -> Result<bool> {
    if !bbo.is_fresh(
        crate::types::unix_ms(),
        *last_exchange_ms,
        args.max_bbo_lag_ms,
    ) {
        args.metrics
            .invalid_messages
            .fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    }
    // Same ordering as replay: publish depth before the corresponding touch.
    if let Some(book) = book {
        push_causal(args, MarketEvent::Book(book))?;
        args.metrics.book_updates.fetch_add(1, Ordering::Relaxed);
    }
    push_causal(args, MarketEvent::Bbo(bbo))?;
    args.latest_bbo.store(bbo);
    *last_exchange_ms = bbo.exchange_ms;
    args.metrics.bbo_updates.fetch_add(1, Ordering::Relaxed);
    args.signal.notify(HOT_SIGNAL_MARKET);
    Ok(true)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lockfree::bbo_channel;
    use crate::types::BookLevel;
    use tokio_tungstenite::{tungstenite::protocol::Role, WebSocketStream};

    fn args(capacity: usize) -> MarketStreamArgs {
        MarketStreamArgs {
            ws_url: String::new(),
            instrument: serde_json::from_value(json!({
                "symbol": "CASHCAT", "asset_id": 231, "sz_decimals": 0,
                "max_price_decimals": 6, "max_significant_figures": 5,
                "max_leverage": 3.0, "minimum_notional": 10.0
            }))
            .unwrap(),
            latest_bbo: bbo_channel().0,
            events: Arc::new(AsyncRing::new(capacity)),
            signal: Arc::new(HotPathSignal::default()),
            clock: Arc::new(ProcessClock::default()),
            metrics: Arc::new(Metrics::default()),
            latency: None,
            scientifically_valid: Arc::new(AtomicBool::new(true)),
            shutdown: watch::channel(false).1,
            ping_interval: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(10),
            max_trade_lag_ms: 5_000,
            max_bbo_lag_ms: 5_000,
            connect_timeout: Duration::from_secs(1),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn l2_refreshes_an_unchanged_touch_but_pongs_and_trades_do_not() {
        let mut args = args(16);
        let (_shutdown, receiver) = watch::channel(false);
        args.shutdown = receiver;
        let events = args.events.clone();
        let (client, server) = tokio::io::duplex(16_384);
        let socket = WebSocketStream::from_raw_socket(client, Role::Client, None).await;
        let mut server = WebSocketStream::from_raw_socket(server, Role::Server, None).await;
        let task = tokio::spawn(async move { run_connected(&mut args, socket).await });
        for _ in 0..3 {
            server.next().await.unwrap().unwrap();
        }
        let start_ms = crate::types::unix_ms();
        for offset in 0..3 {
            server
                .send(Message::Text(
                    json!({
                        "channel":"l2Book", "data":{"coin":"CASHCAT", "time":start_ms + offset,
                        "levels":[[{"px":"0.18","sz":"10"}],[{"px":"0.19","sz":"20"}]]}
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            let book = tokio::time::timeout(Duration::from_secs(1), events.pop())
                .await
                .unwrap();
            let touch = tokio::time::timeout(Duration::from_secs(1), events.pop())
                .await
                .unwrap();
            let MarketEvent::Book(book) = book else {
                panic!("depth must be first")
            };
            let MarketEvent::Bbo(touch) = touch else {
                panic!("L2 must publish its touch")
            };
            assert_eq!(touch.exchange_ms, start_ms + offset);
            assert_eq!(touch.recv_ns, book.recv_ns);
            assert_eq!((touch.bid_px, touch.ask_px), (180_000, 190_000));
            tokio::time::advance(Duration::from_secs(6)).await;
            assert!(!task.is_finished());
        }
        server
            .send(Message::Text(r#"{"channel":"pong"}"#.into()))
            .await
            .unwrap();
        server
            .send(Message::Text(
                json!({"channel":"trades", "data":[{
                    "coin":"CASHCAT", "px":"0.18", "sz":"1", "side":"B", "time":start_ms, "tid":1
                }]})
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        assert!(matches!(events.pop().await, MarketEvent::Trade(_)));
        tokio::time::advance(Duration::from_secs(5)).await;
        assert!(task
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("no public BBO"));
    }

    #[test]
    fn quote_acceptance_rejects_bad_time_prices_and_preserves_event_loss() {
        let args = args(2);
        let now = crate::types::unix_ms();
        let touch = Bbo {
            bid_px: 10,
            ask_px: 11,
            bid_sz: 1,
            ask_sz: 2,
            exchange_ms: now,
            recv_ns: 123,
        };
        let mut last = now;
        for bad in [
            Bbo {
                exchange_ms: now - 1,
                ..touch
            },
            Bbo {
                exchange_ms: now - 6_000,
                ..touch
            },
            Bbo {
                exchange_ms: now + 6_000,
                ..touch
            },
            Bbo { ask_px: 9, ..touch },
            Bbo {
                bid_sz: -1,
                ..touch
            },
        ] {
            assert!(!publish_bbo(&args, bad, None, &mut last).unwrap());
        }
        assert_eq!(last, now);
        assert!(publish_bbo(&args, touch, None, &mut last).unwrap());
        let book = BookSnapshot {
            bids: vec![BookLevel {
                px: 10,
                qty_units: 1,
            }],
            asks: vec![BookLevel {
                px: 11,
                qty_units: 2,
            }],
            exchange_ms: now,
            recv_ns: 123,
        };
        assert!(publish_bbo(&args, touch, Some(book), &mut last).is_err());
        assert!(!args.scientifically_valid.load(Ordering::Acquire));
        assert_eq!(
            args.metrics.dropped_causal_events.load(Ordering::Relaxed),
            1
        );
    }
}
