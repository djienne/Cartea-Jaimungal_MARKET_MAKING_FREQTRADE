use futures_util::{SinkExt, StreamExt};
use mm_live::config::{LatencyConfig, LiveConfig, Network};
use mm_live::hyperliquid::auth::HyperliquidCredentials;
use mm_live::hyperliquid::exchange::ActionOutcome;
use mm_live::hyperliquid::live_state::{LiveOrderStatus, LiveStateStore};
use mm_live::hyperliquid::session::{spawn_session, SessionEvent, SessionSpawnArgs};
use mm_live::hyperliquid::signing::{make_cloid, LiveOrderRequest, TimeInForce};
use mm_live::instrument::InstrumentSpec;
use mm_live::latency::LatencyMonitor;
use mm_live::types::{ProcessClock, Side};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[tokio::test]
async fn websocket_action_response_is_correlated_and_persisted() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        acknowledge_subscriptions(&mut socket).await;
        loop {
            let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                continue;
            };
            let request: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
            if request["method"] == "ping" {
                socket
                    .send(Message::Text(r#"{"channel":"pong"}"#.into()))
                    .await
                    .unwrap();
                continue;
            }
            assert_eq!(request["method"], "post");
            assert_eq!(request["request"]["type"], "action");
            assert_eq!(request["request"]["payload"]["action"]["type"], "order");
            let id = request["id"].as_u64().unwrap();
            socket
                .send(Message::Text(
                    serde_json::json!({
                        "channel": "post",
                        "data": {
                            "id": id,
                            "response": {
                                "type": "action",
                                "payload": {
                                    "status": "ok",
                                    "response": {"type":"order","data":{"statuses":[{"resting":{"oid":77}}]}}
                                }
                            }
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            while let Some(message) = socket.next().await {
                if matches!(message, Ok(Message::Close(_))) {
                    return;
                }
            }
        }
    });
    let fixture = Fixture::new(format!("ws://{address}"));
    fixture.wait_ready().await;
    let cloid = make_cloid(1, 2, Side::Buy, 3);
    let outcome = fixture
        .handle
        .place_orders(
            2,
            vec![LiveOrderRequest {
                side: Side::Buy,
                px_units: 110_000,
                qty_units: 100,
                reduce_only: false,
                time_in_force: TimeInForce::Alo,
                cloid: cloid.clone(),
            }],
        )
        .await
        .unwrap();
    assert!(matches!(outcome, ActionOutcome::Response { .. }));
    let order = fixture.state.load_required().unwrap().orders[&cloid].clone();
    assert_eq!(order.status, LiveOrderStatus::Resting);
    assert_eq!(order.oid, Some(77));
    fixture.stop().await;
    server.await.unwrap();
}

#[tokio::test]
async fn socket_loss_after_write_returns_unknown_and_persists_it() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        acknowledge_subscriptions(&mut socket).await;
        loop {
            let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                continue;
            };
            let request: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
            if request["method"] == "post" {
                drop(socket);
                return;
            }
        }
    });
    let fixture = Fixture::new(format!("ws://{address}"));
    fixture.wait_ready().await;
    let cloid = make_cloid(4, 5, Side::Sell, 6);
    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        fixture.handle.place_orders(
            5,
            vec![LiveOrderRequest {
                side: Side::Sell,
                px_units: 120_000,
                qty_units: 100,
                reduce_only: false,
                time_in_force: TimeInForce::Alo,
                cloid: cloid.clone(),
            }],
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(matches!(outcome, ActionOutcome::Unknown { .. }));
    assert_eq!(
        fixture.state.load_required().unwrap().orders[&cloid].status,
        LiveOrderStatus::UnknownOutcome
    );
    fixture.stop().await;
    server.await.unwrap();
}

async fn acknowledge_subscriptions<S>(socket: &mut tokio_tungstenite::WebSocketStream<S>)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    for _ in 0..8 {
        let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
            panic!("subscription was not text");
        };
        let request: serde_json::Value = serde_json::from_str(text.as_ref()).unwrap();
        socket
            .send(Message::Text(
                serde_json::json!({
                    "channel": "subscriptionResponse",
                    "data": request["subscription"].clone()
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    state: Arc<LiveStateStore>,
    handle: mm_live::hyperliquid::session::HyperliquidSessionHandle,
    events: tokio::sync::Mutex<mpsc::Receiver<SessionEvent>>,
    shutdown: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl Fixture {
    fn new(ws_url: String) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let credential_path = directory.path().join("hyperliquid.env");
        std::fs::write(
            &credential_path,
            "exchange=hyperliquid\nwallet_address=0x1111111111111111111111111111111111111111\nprivate_key=0x0000000000000000000000000000000000000000000000000000000000000001\nis_vault=true\n",
        )
        .unwrap();
        let credentials = Arc::new(HyperliquidCredentials::load(&credential_path).unwrap());
        let state = Arc::new(
            LiveStateStore::open(
                &directory.path().join("state.redb"),
                "CASHCAT",
                credentials.account(),
                &credentials.agent_address(),
                "config",
                "meta",
                false,
            )
            .unwrap(),
        );
        let latency_config = LatencyConfig {
            gate_enabled: false,
            queue_capacity: 128,
            ..LatencyConfig::default()
        };
        let latency = Arc::new(LatencyMonitor::new("CASHCAT", 1, &latency_config, false));
        let (events_tx, events_rx) = mpsc::channel(64);
        let (shutdown, shutdown_rx) = watch::channel(false);
        let (handle, task) = spawn_session(SessionSpawnArgs {
            network: Network::Mainnet,
            ws_url,
            instrument: instrument(),
            credentials,
            state: state.clone(),
            config: LiveConfig {
                action_timeout_ms: 500,
                ..LiveConfig::default()
            },
            clock: Arc::new(ProcessClock::default()),
            latency,
            events: events_tx,
            shutdown: shutdown_rx,
            ping_interval: Duration::from_millis(50),
            idle_timeout: Duration::from_secs(2),
        });
        Self {
            _directory: directory,
            state,
            handle,
            events: tokio::sync::Mutex::new(events_rx),
            shutdown,
            task,
        }
    }

    async fn wait_ready(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if matches!(
                    self.events.lock().await.recv().await,
                    Some(SessionEvent::Ready { .. })
                ) {
                    return;
                }
            }
        })
        .await
        .unwrap();
    }

    async fn stop(self) {
        let _ = self.shutdown.send(true);
        tokio::time::timeout(Duration::from_secs(2), self.task)
            .await
            .unwrap()
            .unwrap();
    }
}

fn instrument() -> InstrumentSpec {
    InstrumentSpec {
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
    }
}
