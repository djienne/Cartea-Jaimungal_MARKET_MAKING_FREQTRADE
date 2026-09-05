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
    pub feed_connected_at_ns: AtomicU64,
    pub application_pings_sent: AtomicU64,
    pub application_pongs_received: AtomicU64,
    pub protocol_pings_received: AtomicU64,
    pub ws_idle_timeouts: AtomicU64,
    /// Feed gaps: how many times the public stream went away and came back.
    ///
    /// Separate from `reconnects`, which cannot serve as a gap count: it misses
    /// the reconnect-success case and double-counts a stale-trade bail. These
    /// three exist so a finished run can be judged on how much data it actually
    /// missed instead of on a boolean that any single blip latches false --
    /// which, past about three hours, is every run.
    pub feed_gaps: AtomicU64,
    pub feed_downtime_ms: AtomicU64,
    pub feed_longest_gap_ms: AtomicU64,
    /// Unix ms at which the public stream went away, or 0 while it is up.
    ///
    /// The three counters above are only written when a gap *closes*, so while
    /// one is open they say nothing at all. On 2026-08-26 the feed was down for
    /// 19.65 h and the grid's health line printed 117 times with byte-identical
    /// values -- `reconnects=28 feed_gaps=27 feed_downtime_ms=282835` -- because
    /// the only state that knew was a local variable inside the reconnect loop.
    /// This is that state, published, so an in-progress outage is visible while
    /// it is still happening rather than only in the post-mortem.
    pub feed_disconnected_since_ms: AtomicU64,
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
    pub feed_gaps: u64,
    pub feed_downtime_ms: u64,
    pub feed_longest_gap_ms: u64,
    pub feed_disconnected_since_ms: u64,
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
            feed_gaps: self.feed_gaps.load(Ordering::Relaxed),
            feed_downtime_ms: self.feed_downtime_ms.load(Ordering::Relaxed),
            feed_longest_gap_ms: self.feed_longest_gap_ms.load(Ordering::Relaxed),
            feed_disconnected_since_ms: self.feed_disconnected_since_ms.load(Ordering::Relaxed),
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

impl MetricsSnapshot {
    /// How long the public feed has been down *right now*, in ms; 0 when up.
    ///
    /// `feed_downtime_ms` deliberately stays the closed-gap total -- every
    /// recorded run's numbers are quoted against that meaning -- so the open
    /// gap is reported beside it rather than folded into it.
    #[must_use]
    pub fn feed_down_for_ms(&self, now_ms: u64) -> u64 {
        if self.feed_disconnected_since_ms == 0 {
            0
        } else {
            now_ms.saturating_sub(self.feed_disconnected_since_ms)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_down_for_ms_is_zero_while_connected_and_grows_while_not() {
        let metrics = Metrics::default();
        assert_eq!(metrics.snapshot().feed_down_for_ms(10_000), 0);
        metrics
            .feed_disconnected_since_ms
            .store(4_000, Ordering::Relaxed);
        assert_eq!(metrics.snapshot().feed_down_for_ms(10_000), 6_000);
        // A clock that steps backwards must not report a negative-turned-huge gap.
        assert_eq!(metrics.snapshot().feed_down_for_ms(3_000), 0);
        metrics
            .feed_disconnected_since_ms
            .store(0, Ordering::Relaxed);
        assert_eq!(metrics.snapshot().feed_down_for_ms(10_000), 0);
    }

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
