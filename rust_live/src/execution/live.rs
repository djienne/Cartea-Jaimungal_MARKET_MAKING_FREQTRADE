use super::traits::{AccountStateProvider, ExecutionBackend};
use crate::types::{DesiredQuotes, DryRunAccountState, ExecutionEvent, MarketEvent};
use anyhow::{bail, Result};
use async_trait::async_trait;

/// Deliberately non-trading adapter. The public interfaces and event contracts
/// are ready for a separately validated venue implementation, but no credential
/// or order-submission path exists in this release.
#[derive(Debug, Default)]
pub struct LiveExecutionUnavailable;

#[async_trait]
impl ExecutionBackend for LiveExecutionUnavailable {
    async fn reconcile(&mut self, _desired: DesiredQuotes, _now_ms: u64) -> Result<()> {
        bail!("live order submission is intentionally unavailable")
    }

    async fn on_market_event(&mut self, _event: &MarketEvent) -> Result<Vec<ExecutionEvent>> {
        bail!("live order submission is intentionally unavailable")
    }

    async fn shutdown(&mut self, _now_ms: u64) -> Result<()> {
        Ok(())
    }

    fn invalidate(&mut self, _reason: &str) {}

    fn scientifically_valid(&self) -> bool {
        false
    }
}

impl AccountStateProvider for LiveExecutionUnavailable {
    fn account_state(&self) -> DryRunAccountState {
        DryRunAccountState::default()
    }
}
