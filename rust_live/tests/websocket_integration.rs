use futures_util::{SinkExt, StreamExt};
use mm_live::hyperliquid::market::{run_market_stream, MarketStreamArgs};
use mm_live::instrument::InstrumentSpec;
use mm_live::lockfree::{AsyncRing, AtomicBbo, HotPathSignal};
use mm_live::metrics::Metrics;
use mm_live::types::{MarketEvent, ProcessClock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[tokio::test]
async fn public_adapter_parses_mock_cashcat_stream_without_loss() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let now_ms = mm_live::types::unix_ms();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        for _ in 0..3 {
            let message = socket.next().await.unwrap().unwrap();
            assert!(matches!(message, Message::Text(_)));
        }
        let payloads = [
            format!(
                r#"{{"channel":"bbo","data":{{"coin":"CASHCAT","time":{now_ms},"bbo":[{{"px":"0.13197","sz":"1894"}},{{"px":"0.13220","sz":"2547"}}]}}}}"#
            ),
            format!(
                r#"{{"channel":"trades","data":[{{"coin":"CASHCAT","side":"B","px":"0.13220","sz":"12","time":{},"tid":42}}]}}"#,
                now_ms + 1
            ),
            format!(
                r#"{{"channel":"l2Book","data":{{"coin":"CASHCAT","time":{},"levels":[[{{"px":"0.13197","sz":"1894","n":1}}],[{{"px":"0.13220","sz":"2547","n":1}}]]}}}}"#,
                now_ms + 2
            ),
        ];
        for payload in payloads {
            socket.send(Message::Text(payload.into())).await.unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    });

    let instrument = InstrumentSpec {
        symbol: "CASHCAT".to_owned(),
        dex: String::new(),
        asset_id: 231,
        sz_decimals: 0,
        max_price_decimals: 6,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 10.0,
    };
    let events = Arc::new(AsyncRing::new(16));
    let latest_bbo = Arc::new(AtomicBbo::default());
    let metrics = Arc::new(Metrics::default());
    let valid = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let client = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: format!("ws://{address}"),
        instrument,
        latest_bbo: latest_bbo.clone(),
        events: events.clone(),
        signal: Arc::new(HotPathSignal::default()),
        clock: Arc::new(ProcessClock::default()),
        metrics: metrics.clone(),
        scientifically_valid: valid.clone(),
        shutdown: shutdown_rx,
        ping_interval: std::time::Duration::from_secs(30),
        idle_timeout: std::time::Duration::from_secs(45),
    }));
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), events.pop())
        .await
        .unwrap();
    let second = tokio::time::timeout(std::time::Duration::from_secs(2), events.pop())
        .await
        .unwrap();
    let third = tokio::time::timeout(std::time::Duration::from_secs(2), events.pop())
        .await
        .unwrap();
    assert!(matches!(first, MarketEvent::Bbo(_)));
    assert!(matches!(second, MarketEvent::Trade(_)));
    assert!(matches!(third, MarketEvent::Book(_)));
    assert_eq!(latest_bbo.load().unwrap().bid_px, 131_970);
    assert_eq!(metrics.snapshot().dropped_causal_events, 0);
    assert!(valid.load(Ordering::Acquire));
    let _ = shutdown_tx.send(true);
    server.await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), client)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn causal_ring_saturation_invalidates_the_session() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let now_ms = mm_live::types::unix_ms();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        for _ in 0..3 {
            socket.next().await.unwrap().unwrap();
        }
        let bbo = format!(
            r#"{{"channel":"bbo","data":{{"coin":"CASHCAT","time":{now_ms},"bbo":[{{"px":"0.13197","sz":"1"}},{{"px":"0.13220","sz":"1"}}]}}}}"#
        );
        let trade = format!(
            r#"{{"channel":"trades","data":[{{"coin":"CASHCAT","side":"B","px":"0.13220","sz":"1","time":{},"tid":99}}]}}"#,
            now_ms + 1
        );
        socket.send(Message::Text(bbo.into())).await.unwrap();
        socket.send(Message::Text(trade.into())).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    });
    let events = Arc::new(AsyncRing::new(1));
    let metrics = Arc::new(Metrics::default());
    let valid = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let client = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: format!("ws://{address}"),
        instrument: InstrumentSpec {
            symbol: "CASHCAT".to_owned(),
            dex: String::new(),
            asset_id: 231,
            sz_decimals: 0,
            max_price_decimals: 6,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 10.0,
        },
        latest_bbo: Arc::new(AtomicBbo::default()),
        events,
        signal: Arc::new(HotPathSignal::default()),
        clock: Arc::new(ProcessClock::default()),
        metrics: metrics.clone(),
        scientifically_valid: valid.clone(),
        shutdown: shutdown_rx,
        ping_interval: std::time::Duration::from_secs(30),
        idle_timeout: std::time::Duration::from_secs(45),
    }));
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while metrics.snapshot().dropped_causal_events == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!valid.load(Ordering::Acquire));
    let _ = shutdown_tx.send(true);
    server.await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), client)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn application_ping_and_protocol_pong_are_exercised() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (heartbeat_tx, heartbeat_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        let mut subscriptions = Vec::new();
        for _ in 0..3 {
            let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                panic!("subscription was not text");
            };
            let value: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
            subscriptions.push(value["subscription"]["type"].as_str().unwrap().to_owned());
        }
        subscriptions.sort();
        assert_eq!(subscriptions, ["bbo", "l2Book", "trades"]);

        let Message::Text(text) =
            tokio::time::timeout(std::time::Duration::from_millis(200), socket.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
        else {
            panic!("application heartbeat was not text");
        };
        let heartbeat: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
        assert_eq!(heartbeat["method"], "ping");
        socket
            .send(Message::Text(r#"{"channel":"pong"}"#.into()))
            .await
            .unwrap();
        socket.send(Message::Ping(vec![7, 8].into())).await.unwrap();

        tokio::time::timeout(std::time::Duration::from_millis(200), async {
            loop {
                if let Message::Pong(payload) = socket.next().await.unwrap().unwrap() {
                    assert_eq!(payload.as_ref(), [7, 8]);
                    break;
                }
            }
        })
        .await
        .unwrap();
        let _ = heartbeat_tx.send(());
        while let Some(message) = socket.next().await {
            if matches!(message, Ok(Message::Close(_))) {
                break;
            }
        }
    });

    let metrics = Arc::new(Metrics::default());
    let valid = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let client = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: format!("ws://{address}"),
        instrument: test_instrument(),
        latest_bbo: Arc::new(AtomicBbo::default()),
        events: Arc::new(AsyncRing::new(16)),
        signal: Arc::new(HotPathSignal::default()),
        clock: Arc::new(ProcessClock::default()),
        metrics: metrics.clone(),
        scientifically_valid: valid.clone(),
        shutdown: shutdown_rx,
        ping_interval: std::time::Duration::from_millis(20),
        idle_timeout: std::time::Duration::from_millis(200),
    }));
    tokio::time::timeout(std::time::Duration::from_millis(500), heartbeat_rx)
        .await
        .unwrap()
        .unwrap();
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(std::time::Duration::from_secs(2), client)
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
    assert!(metrics.snapshot().market_messages >= 1);
    assert!(metrics.snapshot().application_pings_sent >= 1);
    assert!(metrics.snapshot().application_pongs_received >= 1);
    assert_eq!(metrics.snapshot().protocol_pings_received, 1);
    assert_eq!(metrics.snapshot().reconnects, 0);
    assert!(valid.load(Ordering::Acquire));
}

#[tokio::test]
async fn idle_socket_reconnects_resubscribes_and_invalidates_evidence() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let now_ms = mm_live::types::unix_ms();
    let server = tokio::spawn(async move {
        let (first_stream, _) = listener.accept().await.unwrap();
        let mut first = accept_async(first_stream).await.unwrap();
        for _ in 0..3 {
            assert!(matches!(
                first.next().await.unwrap().unwrap(),
                Message::Text(_)
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        drop(first);

        let (second_stream, _) = listener.accept().await.unwrap();
        let mut second = accept_async(second_stream).await.unwrap();
        for _ in 0..3 {
            assert!(matches!(
                second.next().await.unwrap().unwrap(),
                Message::Text(_)
            ));
        }
        let bbo = format!(
            r#"{{"channel":"bbo","data":{{"coin":"CASHCAT","time":{now_ms},"bbo":[{{"px":"0.13197","sz":"1"}},{{"px":"0.13220","sz":"1"}}]}}}}"#
        );
        second.send(Message::Text(bbo.into())).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    });

    let events = Arc::new(AsyncRing::new(16));
    let metrics = Arc::new(Metrics::default());
    let valid = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let client = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: format!("ws://{address}"),
        instrument: test_instrument(),
        latest_bbo: Arc::new(AtomicBbo::default()),
        events: events.clone(),
        signal: Arc::new(HotPathSignal::default()),
        clock: Arc::new(ProcessClock::default()),
        metrics: metrics.clone(),
        scientifically_valid: valid.clone(),
        shutdown: shutdown_rx,
        ping_interval: std::time::Duration::from_millis(20),
        idle_timeout: std::time::Duration::from_millis(80),
    }));
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.pop())
        .await
        .unwrap();
    assert!(matches!(event, MarketEvent::Bbo(_)));
    assert!(metrics.snapshot().reconnects >= 1);
    assert!(metrics.snapshot().ws_idle_timeouts >= 1);
    assert!(!valid.load(Ordering::Acquire));
    let _ = shutdown_tx.send(true);
    server.await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), client)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn initial_trade_snapshot_ignores_old_rows_in_any_order() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let now_ms = mm_live::types::unix_ms();
    let (sent_tx, sent_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        for _ in 0..3 {
            assert!(matches!(
                socket.next().await.unwrap().unwrap(),
                Message::Text(_)
            ));
        }
        let mixed_snapshot = format!(
            r#"{{"channel":"trades","data":[
                {{"coin":"CASHCAT","side":"B","px":"0.13220","sz":"2","time":{},"tid":101}},
                {{"coin":"CASHCAT","side":"A","px":"0.13197","sz":"3","time":{},"tid":100}}
            ]}}"#,
            now_ms,
            now_ms.saturating_sub(8_000)
        );
        socket
            .send(Message::Text(mixed_snapshot.into()))
            .await
            .unwrap();
        let _ = sent_tx.send(());
        while let Some(message) = socket.next().await {
            if matches!(message, Ok(Message::Close(_))) {
                break;
            }
        }
    });

    let events = Arc::new(AsyncRing::new(16));
    let metrics = Arc::new(Metrics::default());
    let valid = Arc::new(AtomicBool::new(true));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let client = tokio::spawn(run_market_stream(MarketStreamArgs {
        ws_url: format!("ws://{address}"),
        instrument: test_instrument(),
        latest_bbo: Arc::new(AtomicBbo::default()),
        events: events.clone(),
        signal: Arc::new(HotPathSignal::default()),
        clock: Arc::new(ProcessClock::default()),
        metrics: metrics.clone(),
        scientifically_valid: valid.clone(),
        shutdown: shutdown_rx,
        ping_interval: std::time::Duration::from_secs(30),
        idle_timeout: std::time::Duration::from_secs(45),
    }));
    tokio::time::timeout(std::time::Duration::from_millis(500), sent_rx)
        .await
        .unwrap()
        .unwrap();
    let event = tokio::time::timeout(std::time::Duration::from_secs(1), events.pop())
        .await
        .unwrap();
    assert!(matches!(event, MarketEvent::Trade(_)));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(metrics.snapshot().trade_prints, 1);
    assert_eq!(metrics.snapshot().historical_trade_prints_ignored, 1);
    assert_eq!(metrics.snapshot().reconnects, 0);
    assert!(valid.load(Ordering::Acquire));
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(std::time::Duration::from_secs(2), client)
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
}

fn test_instrument() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "CASHCAT".to_owned(),
        dex: String::new(),
        asset_id: 231,
        sz_decimals: 0,
        max_price_decimals: 6,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 10.0,
    }
}
