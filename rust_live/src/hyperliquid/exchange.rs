use super::auth::HyperliquidCredentials;
use super::signing::{
    cancel_by_cloid_action, cancel_by_oid_action, order_action, sign_envelope, LiveOrderRequest,
};
use crate::config::Network;
use crate::instrument::InstrumentSpec;
use crate::latency::{LatencyKind, LatencyMonitor};
use crate::types::{unix_ms, ProcessClock};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct HyperliquidExchangeClient {
    http: reqwest::Client,
    network: Network,
    instrument: InstrumentSpec,
    credentials: Arc<HyperliquidCredentials>,
    nonce: MonotonicNonce,
    action_expiry_ms: u64,
    clock: Arc<ProcessClock>,
    latency: Option<Arc<LatencyMonitor>>,
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
    ) -> Result<Self> {
        instrument.validate()?;
        if action_timeout.is_zero() {
            bail!("Hyperliquid action timeout must be positive");
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

    pub async fn cancel_by_oid(&self, oids: &[u64]) -> Result<ActionOutcome> {
        let action = cancel_by_oid_action(self.instrument.asset_id, oids)?;
        self.post_signed(&action, LatencyKind::CancelToAck, false)
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
    pub crossed: bool,
    #[serde(default)]
    pub fee: String,
    #[serde(default)]
    pub closed_pnl: String,
    #[serde(default)]
    pub hash: String,
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
}
