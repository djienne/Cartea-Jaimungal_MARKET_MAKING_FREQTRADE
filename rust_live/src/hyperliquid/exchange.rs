use super::auth::HyperliquidCredentials;
use super::signing::{
    cancel_by_cloid_action, cancel_by_oid_action, order_action, schedule_cancel_action,
    sign_envelope, LiveOrderRequest,
};
use crate::config::Network;
use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::types::{unix_ms, ProcessClock};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct HyperliquidExchangeClient {
    http: reqwest::Client,
    network: Network,
    instrument: InstrumentSpec,
    credentials: Arc<HyperliquidCredentials>,
    nonce: MonotonicNonce,
    action_expiry_ms: u64,
    clock: Arc<ProcessClock>,
    latency: Option<Arc<LatencyMonitor>>,
    rate_limit: Mutex<RestRateLimiter>,
}

impl HyperliquidExchangeClient {
    pub fn new(
        network: Network,
        instrument: InstrumentSpec,
        credentials: Arc<HyperliquidCredentials>,
        clock: Arc<ProcessClock>,
        latency: Option<Arc<LatencyMonitor>>,
        action_timeout: Duration,
        action_expiry_ms: u64,
        max_rest_weight_per_minute: u64,
    ) -> Result<Self> {
        instrument.validate()?;
        if action_timeout.is_zero() {
            bail!("Hyperliquid action timeout must be positive");
        }
        if max_rest_weight_per_minute == 0 || max_rest_weight_per_minute > 1_200 {
            bail!("REST weight limit must be in 1..=1200 per minute");
        }
        let http = reqwest::Client::builder()
            .timeout(action_timeout)
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .pool_idle_timeout(Some(Duration::from_secs(120)))
            .pool_max_idle_per_host(4)
            .build()?;
        Ok(Self {
            http,
            network,
            instrument,
            credentials,
            nonce: MonotonicNonce::default(),
            action_expiry_ms,
            clock,
            latency,
            rate_limit: Mutex::new(RestRateLimiter::new(max_rest_weight_per_minute)),
        })
    }

    pub fn account(&self) -> &str {
        self.credentials.account()
    }

    pub fn agent_address(&self) -> String {
        self.credentials.agent_address()
    }

    pub fn instrument(&self) -> &InstrumentSpec {
        &self.instrument
    }

    pub async fn info(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let request_type = request
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        self.consume_rest_weight(info_weight(request_type))?;
        let started_ns = self.clock.now_ns();
        let response = self
            .http
            .post(self.network.info_url())
            .json(&request)
            .send()
            .await;
        let finished_ns = self.clock.now_ns();
        if let Some(latency) = &self.latency {
            latency.record(
                LatencyKind::InfoRequestRtt,
                finished_ns.saturating_sub(started_ns),
                finished_ns,
            );
        }
        response?
            .error_for_status()
            .context("Hyperliquid /info request failed")?
            .json()
            .await
            .context("cannot decode Hyperliquid /info response")
    }

    pub async fn clearinghouse_state(&self) -> Result<ClearinghouseState> {
        let value = self
            .info(json!({
                "type": "clearinghouseState",
                "user": self.account(),
                "dex": self.instrument.dex,
            }))
            .await?;
        serde_json::from_value(value).context("cannot decode clearinghouseState")
    }

    pub async fn open_orders(&self) -> Result<Vec<OpenOrder>> {
        let value = self
            .info(json!({
                "type": "openOrders",
                "user": self.account(),
                "dex": self.instrument.dex,
            }))
            .await?;
        serde_json::from_value(value).context("cannot decode openOrders")
    }

    pub async fn recent_fills(&self) -> Result<Vec<UserFill>> {
        let value = self
            .info(json!({"type": "userFills", "user": self.account()}))
            .await?;
        serde_json::from_value(value).context("cannot decode userFills")
    }

    pub async fn historical_orders(&self) -> Result<serde_json::Value> {
        self.info(json!({"type": "historicalOrders", "user": self.account()}))
            .await
    }

    pub async fn user_rate_limit(&self) -> Result<serde_json::Value> {
        self.info(json!({"type": "userRateLimit", "user": self.account()}))
            .await
    }

    pub async fn user_funding(&self, start_time: u64) -> Result<serde_json::Value> {
        self.info(json!({
            "type": "userFunding",
            "user": self.account(),
            "startTime": start_time,
        }))
        .await
    }

    pub async fn user_role(&self) -> Result<serde_json::Value> {
        self.info(json!({"type": "userRole", "user": self.account()}))
            .await
    }

    pub async fn user_fees(&self) -> Result<serde_json::Value> {
        self.info(json!({"type": "userFees", "user": self.account()}))
            .await
    }

    pub async fn active_asset_data(&self) -> Result<serde_json::Value> {
        self.info(json!({
            "type": "activeAssetData",
            "user": self.account(),
            "coin": self.instrument.symbol,
        }))
        .await
    }

    pub async fn all_mids(&self) -> Result<serde_json::Value> {
        self.info(json!({"type": "allMids", "dex": self.instrument.dex}))
            .await
    }

    pub async fn l2_book(&self) -> Result<serde_json::Value> {
        self.info(json!({"type": "l2Book", "coin": self.instrument.symbol}))
            .await
    }

    pub async fn order_status(&self, oid_or_cloid: serde_json::Value) -> Result<serde_json::Value> {
        self.info(json!({
            "type": "orderStatus",
            "user": self.account(),
            "oid": oid_or_cloid,
        }))
        .await
    }

    pub async fn place_orders(&self, requests: &[LiveOrderRequest]) -> Result<ActionOutcome> {
        let action = order_action(&self.instrument, requests)?;
        self.post_signed(&action, LatencyKind::SubmitToAck, true)
            .await
    }

    pub async fn cancel_by_cloid(&self, cloids: &[String]) -> Result<ActionOutcome> {
        let action = cancel_by_cloid_action(self.instrument.asset_id, cloids)?;
        self.post_signed(&action, LatencyKind::CancelToAck, false)
            .await
    }

    pub async fn cancel_by_cloid_with_nonce(
        &self,
        cloids: &[String],
        nonce: u64,
    ) -> Result<ActionOutcome> {
        let action = cancel_by_cloid_action(self.instrument.asset_id, cloids)?;
        self.post_signed_at(&action, LatencyKind::CancelToAck, None, nonce)
            .await
    }

    pub async fn cancel_by_oid(&self, oids: &[u64]) -> Result<ActionOutcome> {
        let action = cancel_by_oid_action(self.instrument.asset_id, oids)?;
        self.post_signed(&action, LatencyKind::CancelToAck, false)
            .await
    }

    pub async fn schedule_cancel_with_nonce(
        &self,
        time: Option<u64>,
        nonce: u64,
    ) -> Result<ActionOutcome> {
        let action = schedule_cancel_action(time);
        self.post_signed_at(&action, LatencyKind::CancelToAck, None, nonce)
            .await
    }

    async fn post_signed<A: Serialize>(
        &self,
        action: &A,
        latency_kind: LatencyKind,
        with_expiry: bool,
    ) -> Result<ActionOutcome> {
        let nonce = self.nonce.next();
        let expires_after = (with_expiry && self.action_expiry_ms != 0)
            .then(|| nonce.saturating_add(self.action_expiry_ms));
        self.post_signed_at(action, latency_kind, expires_after, nonce)
            .await
    }

    async fn post_signed_at<A: Serialize>(
        &self,
        action: &A,
        latency_kind: LatencyKind,
        expires_after: Option<u64>,
        nonce: u64,
    ) -> Result<ActionOutcome> {
        self.consume_rest_weight(1)?;
        let envelope = sign_envelope(
            action,
            &self.credentials,
            self.network,
            nonce,
            expires_after,
        )?;
        let started_ns = self.clock.now_ns();
        let response = self
            .http
            .post(format!("{}/exchange", self.network.api_url()))
            .json(&envelope.body)
            .send()
            .await;
        let finished_ns = self.clock.now_ns();
        if let Some(latency) = &self.latency {
            latency.record(
                latency_kind,
                finished_ns.saturating_sub(started_ns),
                finished_ns,
            );
        }
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                return Ok(ActionOutcome::Unknown {
                    nonce,
                    error: error.to_string(),
                });
            }
        };
        let http_status = response.status().as_u16();
        let text = response
            .text()
            .await
            .context("cannot read Hyperliquid action response")?;
        let body = serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text}));
        Ok(ActionOutcome::Response {
            nonce,
            http_status,
            body,
        })
    }

    fn consume_rest_weight(&self, weight: u64) -> Result<()> {
        self.rate_limit
            .lock()
            .map_err(|_| anyhow::anyhow!("REST rate limiter lock poisoned"))?
            .consume(weight)
    }
}

struct RestRateLimiter {
    maximum: u64,
    used: u64,
    entries: VecDeque<(Instant, u64)>,
}

impl RestRateLimiter {
    fn new(maximum: u64) -> Self {
        Self {
            maximum,
            used: 0,
            entries: VecDeque::new(),
        }
    }

    fn consume(&mut self, weight: u64) -> Result<()> {
        let cutoff = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .context("monotonic clock cannot represent REST rate window")?;
        while self.entries.front().is_some_and(|(at, _)| *at < cutoff) {
            if let Some((_, expired)) = self.entries.pop_front() {
                self.used = self.used.saturating_sub(expired);
            }
        }
        if self.used.saturating_add(weight) > self.maximum {
            bail!(
                "configured Hyperliquid REST weight budget exhausted: used={}, requested={}, maximum={}",
                self.used,
                weight,
                self.maximum
            );
        }
        self.entries.push_back((Instant::now(), weight));
        self.used = self.used.saturating_add(weight);
        Ok(())
    }
}

fn info_weight(request_type: &str) -> u64 {
    match request_type {
        "l2Book" | "allMids" | "clearinghouseState" | "orderStatus" | "exchangeStatus" => 2,
        "userRole" => 60,
        "userFills" | "userFillsByTime" | "historicalOrders" | "userFunding" => 40,
        _ => 20,
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ActionOutcome {
    Response {
        nonce: u64,
        http_status: u16,
        body: serde_json::Value,
    },
    Unknown {
        nonce: u64,
        error: String,
    },
}

impl ActionOutcome {
    pub fn body(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Response { body, .. } => Some(body),
            Self::Unknown { .. } => None,
        }
    }

    pub fn require_known(&self) -> Result<&serde_json::Value> {
        match self {
            Self::Response { body, .. } => Ok(body),
            Self::Unknown { nonce, error } => {
                bail!("action nonce {nonce} has unknown outcome: {error}")
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearinghouseState {
    pub margin_summary: MarginSummary,
    #[serde(default)]
    pub asset_positions: Vec<AssetPosition>,
    #[serde(default)]
    pub withdrawable: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MarginSummary {
    pub account_value: String,
    #[serde(default)]
    pub total_margin_used: String,
    #[serde(default)]
    pub total_ntl_pos: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AssetPosition {
    pub position: Position,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    pub coin: String,
    pub szi: String,
    #[serde(default)]
    pub entry_px: Option<String>,
    #[serde(default)]
    pub liquidation_px: Option<String>,
    #[serde(default)]
    pub margin_used: String,
    #[serde(default)]
    pub leverage: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenOrder {
    pub coin: String,
    pub oid: u64,
    #[serde(default)]
    pub cloid: Option<String>,
    #[serde(default)]
    pub side: String,
    #[serde(default)]
    pub limit_px: String,
    #[serde(default)]
    pub sz: String,
    #[serde(default)]
    pub timestamp: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UserFill {
    pub coin: String,
    pub px: String,
    pub sz: String,
    pub side: String,
    pub time: u64,
    pub oid: u64,
    pub tid: u64,
    #[serde(default)]
    pub cloid: Option<String>,
    #[serde(default)]
    pub start_position: String,
    #[serde(default)]
    pub direction: String,
    #[serde(default)]
    pub crossed: bool,
    #[serde(default)]
    pub fee: String,
    #[serde(default)]
    pub closed_pnl: String,
    #[serde(default)]
    pub hash: String,
    #[serde(default)]
    pub fee_token: String,
}

#[derive(Debug, Default)]
struct MonotonicNonce {
    last: AtomicU64,
}

impl MonotonicNonce {
    fn next(&self) -> u64 {
        let wall = unix_ms();
        loop {
            let previous = self.last.load(Ordering::Acquire);
            let candidate = wall.max(previous.saturating_add(1));
            if self
                .last
                .compare_exchange(previous, candidate, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return candidate;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_is_strictly_increasing_inside_one_signer_process() {
        let nonce = MonotonicNonce::default();
        let first = nonce.next();
        let second = nonce.next();
        assert!(second > first);
    }

    #[test]
    fn known_and_unknown_outcomes_are_not_conflated() {
        let known = ActionOutcome::Response {
            nonce: 1,
            http_status: 200,
            body: json!({"status": "ok"}),
        };
        assert_eq!(known.require_known().unwrap()["status"], "ok");
        let unknown = ActionOutcome::Unknown {
            nonce: 2,
            error: "timeout after write".to_owned(),
        };
        assert!(unknown.require_known().is_err());
    }

    #[test]
    fn rest_rate_limiter_uses_documented_weights() {
        assert_eq!(info_weight("clearinghouseState"), 2);
        assert_eq!(info_weight("userRole"), 60);
        assert_eq!(info_weight("userFees"), 20);
        let mut limiter = RestRateLimiter::new(3);
        limiter.consume(2).unwrap();
        assert!(limiter.consume(2).is_err());
        limiter.consume(1).unwrap();
    }
}
