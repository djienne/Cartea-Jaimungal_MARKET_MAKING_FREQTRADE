use futures_util::{SinkExt, StreamExt};
use mm_live::config::LatencyConfig;
use mm_live::hyperliquid::account::{
    run_account_stream, AccountStreamArgs, AccountStreamEvent, AccountStreamMetrics,
};
use mm_live::latency::{LatencyMonitor, LatencyObserver};
use mm_live::lockfree::AsyncRing;
use mm_live::types::ProcessClock;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[tokio::test]
async fn account_listener_subscribes_heartbeats_and_delivers_order_and_fill_events() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        let mut subscriptions = BTreeSet::new();
        for _ in 0..8 {
            let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                panic!("account subscription was not text");
            };
            let value: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
            let subscription = value["subscription"].clone();
            subscriptions.insert(subscription["type"].as_str().unwrap().to_owned());
            socket
                .send(Message::Text(
                    serde_json::json!({
                        "channel": "subscriptionResponse",
                        "data": {"subscription": subscription}
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
        }
        assert_eq!(
            subscriptions,
            BTreeSet::from([
                "activeAssetData".to_owned(),
                "clearinghouseState".to_owned(),
                "notification".to_owned(),
                "openOrders".to_owned(),
                "orderUpdates".to_owned(),
                "userFills".to_owned(),
                "userFundings".to_owned(),
                "userNonFundingLedgerUpdates".to_owned(),
            ])
        );
        loop {
            let message = socket.next().await.unwrap().unwrap();
            let Message::Text(text) = message else {
                continue;
            };
            let value: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
            if value["method"] == "ping" {
                break;
            }
        }
        socket
            .send(Message::Text(r#"{"channel":"pong"}"#.into()))
            .await
            .unwrap();
        socket
            .send(Message::Ping(vec![1, 2, 3].into()))
            .await
            .unwrap();
        socket
            .send(Message::Text(
                r#"{"channel":"orderUpdates","data":[{"status":"open"}]}"#.into(),
            ))
            .await
            .unwrap();
        socket
            .send(Message::Text(
                r#"{"channel":"userFills","data":{"isSnapshot":false,"fills":[{"tid":7}]}}"#.into(),
            ))
            .await
            .unwrap();
        while let Some(message) = socket.next().await {
            if matches!(message, Ok(Message::Close(_))) {
                break;
            }
        }
    });

    let events = Arc::new(AsyncRing::new(32));
    let metrics = Arc::new(AccountStreamMetrics::default());
    let healthy = Arc::new(AtomicBool::new(true));
    let clock = Arc::new(ProcessClock::default());
    let latency_config = LatencyConfig {
        gate_enabled: false,
        queue_capacity: 32,
        ..LatencyConfig::default()
    };
    let latency = Arc::new(LatencyMonitor::new("CASHCAT", 1, &latency_config, false));
    let latency_directory = tempfile::tempdir().unwrap();
    let latency_observer = LatencyObserver::spawn(
        latency.clone(),
        clock.clone(),
        "CASHCAT".to_owned(),
        1,
        latency_config,
        false,
        Duration::from_secs(60),
        latency_directory.path().join("latency.json"),
    )
    .unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let client = tokio::spawn(run_account_stream(AccountStreamArgs {
        ws_url: format!("ws://{address}"),
        account: "0x1111111111111111111111111111111111111111".to_owned(),
        dex: String::new(),
        symbol: "CASHCAT".to_owned(),
        events: events.clone(),
        clock,
        latency: Some(latency.clone()),
        metrics: metrics.clone(),
        healthy: healthy.clone(),
        shutdown: shutdown_rx,
        ping_interval: Duration::from_millis(20),
        idle_timeout: Duration::from_millis(200),
    }));

    let mut data_channels = BTreeSet::new();
    tokio::time::timeout(Duration::from_secs(2), async {
        while data_channels.len() < 2 {
            if let AccountStreamEvent::Data { channel, .. } = events.pop().await {
                data_channels.insert(channel);
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(
        data_channels,
        BTreeSet::from(["orderUpdates".to_owned(), "userFills".to_owned()])
    );
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), client)
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
    latency_observer.stop().unwrap();
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.subscription_acks, 8);
    assert!(snapshot.pings_sent >= 1);
    assert!(snapshot.pongs_received >= 1);
    assert_eq!(snapshot.protocol_pings_received, 1);
    assert_eq!(snapshot.reconnects, 0);
    assert_eq!(snapshot.dropped_events, 0);
    assert!(healthy.load(Ordering::Acquire));
    assert!(latency.snapshot().distributions["account_ws_ping_rtt"]
        .last_ns
        .is_some_and(|value| value > 0));
}
