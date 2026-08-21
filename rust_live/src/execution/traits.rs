use crate::types::{DesiredQuotes, DryRunAccountState, ExecutionEvent, MarketEvent};
use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait MarketDataSource: Send {
    async fn next_event(&mut self) -> Result<Option<MarketEvent>>;
}

#[async_trait]
pub trait ExecutionBackend: Send {
    async fn reconcile(&mut self, desired: DesiredQuotes, now_ms: u64) -> Result<()>;
    async fn on_market_event(&mut self, event: &MarketEvent) -> Result<Vec<ExecutionEvent>>;
    async fn shutdown(&mut self, now_ms: u64) -> Result<()>;
    fn invalidate(&mut self, reason: &str);
    fn scientifically_valid(&self) -> bool;
}

pub trait AccountStateProvider {
    fn account_state(&self) -> DryRunAccountState;
}
