use serde::Serialize;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct Metrics {
    pub market_messages: AtomicU64,
    pub bbo_updates: AtomicU64,
    pub trade_prints: AtomicU64,
    pub historical_trade_prints_ignored: AtomicU64,
    pub book_updates: AtomicU64,
    pub invalid_messages: AtomicU64,
    pub dropped_causal_events: AtomicU64,
    pub reconnects: AtomicU64,
    pub application_pings_sent: AtomicU64,
    pub application_pongs_received: AtomicU64,
    pub protocol_pings_received: AtomicU64,
    pub ws_idle_timeouts: AtomicU64,
    pub quote_decisions: AtomicU64,
    pub quote_publications: AtomicU64,
    pub fills: AtomicU64,
    pub risk_refusals: AtomicU64,
    pub calibration_runs: AtomicU64,
    pub calibration_failures: AtomicU64,
    pub inventory_units: AtomicI64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub market_messages: u64,
    pub bbo_updates: u64,
    pub trade_prints: u64,
    pub historical_trade_prints_ignored: u64,
    pub book_updates: u64,
    pub invalid_messages: u64,
    pub dropped_causal_events: u64,
    pub reconnects: u64,
    pub application_pings_sent: u64,
    pub application_pongs_received: u64,
    pub protocol_pings_received: u64,
    pub ws_idle_timeouts: u64,
    pub quote_decisions: u64,
    pub quote_publications: u64,
    pub fills: u64,
    pub risk_refusals: u64,
    pub calibration_runs: u64,
    pub calibration_failures: u64,
    pub inventory_units: i64,
}

impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            market_messages: self.market_messages.load(Ordering::Relaxed),
            bbo_updates: self.bbo_updates.load(Ordering::Relaxed),
            trade_prints: self.trade_prints.load(Ordering::Relaxed),
            historical_trade_prints_ignored: self
                .historical_trade_prints_ignored
                .load(Ordering::Relaxed),
            book_updates: self.book_updates.load(Ordering::Relaxed),
            invalid_messages: self.invalid_messages.load(Ordering::Relaxed),
            dropped_causal_events: self.dropped_causal_events.load(Ordering::Relaxed),
            reconnects: self.reconnects.load(Ordering::Relaxed),
            application_pings_sent: self.application_pings_sent.load(Ordering::Relaxed),
            application_pongs_received: self.application_pongs_received.load(Ordering::Relaxed),
            protocol_pings_received: self.protocol_pings_received.load(Ordering::Relaxed),
            ws_idle_timeouts: self.ws_idle_timeouts.load(Ordering::Relaxed),
            quote_decisions: self.quote_decisions.load(Ordering::Relaxed),
            quote_publications: self.quote_publications.load(Ordering::Relaxed),
            fills: self.fills.load(Ordering::Relaxed),
            risk_refusals: self.risk_refusals.load(Ordering::Relaxed),
            calibration_runs: self.calibration_runs.load(Ordering::Relaxed),
            calibration_failures: self.calibration_failures.load(Ordering::Relaxed),
            inventory_units: self.inventory_units.load(Ordering::Relaxed),
        }
    }
}
