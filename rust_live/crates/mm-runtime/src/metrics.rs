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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reads_every_counter_without_side_effects() {
        let metrics = Metrics::default();
        metrics.market_messages.store(1, Ordering::Relaxed);
        metrics.bbo_updates.store(2, Ordering::Relaxed);
        metrics.trade_prints.store(3, Ordering::Relaxed);
        metrics
            .historical_trade_prints_ignored
            .store(4, Ordering::Relaxed);
        metrics.book_updates.store(5, Ordering::Relaxed);
        metrics.invalid_messages.store(6, Ordering::Relaxed);
        metrics.dropped_causal_events.store(7, Ordering::Relaxed);
        metrics.reconnects.store(8, Ordering::Relaxed);
        metrics.application_pings_sent.store(9, Ordering::Relaxed);
        metrics
            .application_pongs_received
            .store(10, Ordering::Relaxed);
        metrics.protocol_pings_received.store(11, Ordering::Relaxed);
        metrics.ws_idle_timeouts.store(12, Ordering::Relaxed);
        metrics.quote_decisions.store(13, Ordering::Relaxed);
        metrics.quote_publications.store(14, Ordering::Relaxed);
        metrics.fills.store(15, Ordering::Relaxed);
        metrics.risk_refusals.store(16, Ordering::Relaxed);
        metrics.calibration_runs.store(17, Ordering::Relaxed);
        metrics.calibration_failures.store(18, Ordering::Relaxed);
        metrics.inventory_units.store(-19, Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.market_messages, 1);
        assert_eq!(snapshot.protocol_pings_received, 11);
        assert_eq!(snapshot.calibration_failures, 18);
        assert_eq!(snapshot.inventory_units, -19);
    }
}
