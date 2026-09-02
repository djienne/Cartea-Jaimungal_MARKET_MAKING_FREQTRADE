//! Toxic-flow guard: withdraw quoting when order flow turns against us.
//!
//! # What this is for
//!
//! The 161.95 h CASHCAT tape had exactly one losing six-hour window. 26 of the
//! other 27 sum to +35.28 USDC; that one is −241.17 on 1,771 fills. It was a
//! liquidation cascade, not a slow toxic drift:
//!
//! ```text
//! 05:11:00  mid 0.12305     —
//! 05:11:20  mid 0.09952  −31%    ten seconds
//! 05:12:00  mid 0.03900  −70%    sixty seconds
//! 05:16:00  mid ~0.11           recovered
//! ```
//!
//! This guard does not predict that event. Its narrower goal is to stop adding
//! exposure once the move is observable and stay out of the aftermath.
//!
//! # Why two tiers
//!
//! Re-verified over 165.11 h, both thresholds had **zero** breaches outside the
//! cascade:
//!
//! | tier | first trip | mid had already moved |
//! |---|---|---|
//! | fast mid-move breaker (≥8% in 5 s) | 05:11:15 | −14% |
//! | VPIN ≥ 0.40 | 05:11:30 | −45% |
//!
//! The breaker is 16 seconds and 31 percentage points earlier, so it bounds the
//! damage. VPIN is slower but identifies the *regime* — 86% of that window's
//! volume arrived after it fired — so it is what prevents re-entering the
//! aftermath. Neither alone is enough, which is why both ship.
//!
//! The existing `calibration.max_toxicity` gate (κ·ε, threshold 1.5) is a
//! different thing and did not fire: it read 0.254/0.235 through this cascade.
//! It is a slowly-varying property of the calibration fit, not a flow alarm.
//!
//! # What it is not
//!
//! It cannot prevent the first fills — at −14% resting bids have already been
//! hit. And the thresholds are fitted on a single event (n=1) in one tape, so
//! they are a starting point to re-check as the tape grows, not a constant of
//! nature.

use cj_core::types::{AggressorSide, TradePrint};
use mm_settings::FlowGuardConfig;

const RECENT_TRADE_ID_CAPACITY: usize = 65_536;

/// Trailing mid samples for the fast breaker.
///
/// A fixed-capacity ring owned by the hot-path loop: no allocation, no locks,
/// nothing shared. It answers one question — how far has the mid moved against
/// the oldest sample still inside the window.
#[derive(Debug)]
pub struct MidWindow {
    samples: Box<[(u64, i64)]>,
    head: usize,
    tail: usize,
    len: usize,
    window_ns: u64,
}

impl MidWindow {
    /// `capacity` bounds how many samples the window can hold. When it
    /// overflows, the oldest is dropped and the comparison is made against a
    /// *younger* reference, which understates the move — the conservative
    /// direction for a guard (it delays a trip, never invents one).
    pub fn new(capacity: usize, window_ms: u64) -> Self {
        let capacity = capacity.max(2);
        Self {
            samples: vec![(0_u64, 0_i64); capacity].into_boxed_slice(),
            head: 0,
            tail: 0,
            len: 0,
            window_ns: window_ms.saturating_mul(1_000_000),
        }
    }

    /// Record a mid and return the absolute move against the oldest sample
    /// inside the window, in basis points.
    pub fn observe(&mut self, now_ns: u64, mid_units: i64) -> f64 {
        let written = self.head;
        self.samples[self.head] = (now_ns, mid_units);
        self.head = (self.head + 1) % self.samples.len();
        if self.len == self.samples.len() {
            self.tail = (self.tail + 1) % self.samples.len();
        } else {
            if self.len == 0 {
                self.tail = written;
            }
            self.len += 1;
        }

        self.advance(now_ns, mid_units)
    }

    /// Age the trailing window and measure against its oldest retained sample
    /// without inserting another observation. Hot-path control/model wakes can
    /// call this freely without shortening the physical BBO window.
    pub fn advance(&mut self, now_ns: u64, mid_units: i64) -> f64 {
        let cutoff = now_ns.saturating_sub(self.window_ns);
        while self.len > 1 && self.samples[self.tail].0 < cutoff {
            self.tail = (self.tail + 1) % self.samples.len();
            self.len -= 1;
        }
        if self.len == 0 {
            return 0.0;
        }
        let reference = self.samples[self.tail].1;
        if reference <= 0 || mid_units <= 0 {
            return 0.0;
        }
        let move_ratio = (mid_units - reference) as f64 / reference as f64;
        move_ratio.abs() * 10_000.0
    }

    pub const fn window_ns(&self) -> u64 {
        self.window_ns
    }

    #[cfg(test)]
    const fn len(&self) -> usize {
        self.len
    }
}

/// Volume-synchronised probability of informed trading.
///
/// `VPIN = Σ|V_buy − V_sell| / (n · V)` over a rolling window of `n` volume
/// buckets of fixed size `V`, matching `VPIN_PYTHON_CRYPTO/vpin_calculator.py`.
///
/// One deliberate difference from that reference: it infers buy volume from
/// Binance's taker-buy aggregate on 1-minute klines, whereas every
/// [`TradePrint`] here carries the aggressor side directly, so the buy/sell
/// split is exact rather than reconstructed.
///
/// The reference's rolling-CDF percentile is **not** used. On this tape it
/// reaches 1.00 in four benign windows at a 2-day lookback — the tape is far
/// shorter than the 90 days that percentile assumes — while the raw statistic
/// separated the cascade on the frozen tape (peak 0.655 against a 0.367 maximum
/// elsewhere). An absolute threshold on raw VPIN is the honest instrument here.
#[derive(Debug)]
pub struct VpinTracker {
    bucket_volume_units: i64,
    window_buckets: usize,
    buy_units: i64,
    sell_units: i64,
    imbalances: std::collections::VecDeque<i64>,
    imbalance_sum: i64,
    recent_trade_ids: std::collections::HashSet<u64>,
    recent_trade_order: std::collections::VecDeque<u64>,
}

impl VpinTracker {
    pub fn new(bucket_volume_units: i64, window_buckets: usize) -> Self {
        Self {
            bucket_volume_units: bucket_volume_units.max(1),
            window_buckets: window_buckets.max(1),
            buy_units: 0,
            sell_units: 0,
            imbalances: std::collections::VecDeque::with_capacity(window_buckets.max(1)),
            imbalance_sum: 0,
            recent_trade_ids: std::collections::HashSet::with_capacity(RECENT_TRADE_ID_CAPACITY),
            recent_trade_order: std::collections::VecDeque::with_capacity(RECENT_TRADE_ID_CAPACITY),
        }
    }

    /// Bucket size is derived from observed volume rather than pinned, so the
    /// statistic tracks the instrument as its activity changes.
    pub fn resize_bucket(&mut self, bucket_volume_units: i64) {
        let next = bucket_volume_units.max(1);
        if next != self.bucket_volume_units {
            self.bucket_volume_units = next;
            // Both partial and completed buckets were measured against the old
            // denominator. Retaining either would mix scales and make the
            // reported VPIN dimensionally inconsistent, so warm up again.
            self.buy_units = 0;
            self.sell_units = 0;
            self.imbalances.clear();
            self.imbalance_sum = 0;
        }
    }

    /// Feed one trade print. Returns the current VPIN once a full window of
    /// buckets exists, `None` while still warming up.
    pub fn observe(&mut self, trade: &TradePrint) -> Option<f64> {
        // The feed can replay several prints across reconnects, not merely the
        // immediately preceding print. Zero means "no usable id" and must not
        // collapse unrelated anonymous trades into one.
        if trade.trade_id != 0 && !self.remember_trade_id(trade.trade_id) {
            return self.value();
        }

        let mut remaining = trade.qty_units.max(0);
        if remaining == 0 {
            return self.value();
        }

        // First finish the current partial bucket in strict arrival order.
        let current = self.buy_units + self.sell_units;
        if current != 0 {
            let taken = remaining.min(self.bucket_volume_units - current);
            self.add_volume(trade.aggressor, taken);
            remaining -= taken;
            if self.buy_units + self.sell_units == self.bucket_volume_units {
                self.finish_bucket();
            }
        }

        // Every complete bucket from the remainder is wholly on this trade's
        // side. Collapse pathological huge prints to at most the retained
        // window instead of looping qty / V times.
        let full_buckets = remaining / self.bucket_volume_units;
        if full_buckets != 0 {
            self.push_one_sided_buckets(full_buckets);
            remaining %= self.bucket_volume_units;
        }
        self.add_volume(trade.aggressor, remaining);
        self.value()
    }

    fn remember_trade_id(&mut self, trade_id: u64) -> bool {
        if !self.recent_trade_ids.insert(trade_id) {
            return false;
        }
        self.recent_trade_order.push_back(trade_id);
        while self.recent_trade_order.len() > RECENT_TRADE_ID_CAPACITY {
            if let Some(expired) = self.recent_trade_order.pop_front() {
                self.recent_trade_ids.remove(&expired);
            }
        }
        true
    }

    fn add_volume(&mut self, side: AggressorSide, qty: i64) {
        match side {
            AggressorSide::Buy => self.buy_units += qty,
            AggressorSide::Sell => self.sell_units += qty,
        }
    }

    fn finish_bucket(&mut self) {
        debug_assert_eq!(self.buy_units + self.sell_units, self.bucket_volume_units);
        let imbalance = (self.buy_units - self.sell_units).abs();
        debug_assert!(imbalance <= self.bucket_volume_units);
        self.push_bucket(imbalance);
        self.buy_units = 0;
        self.sell_units = 0;
    }

    fn push_one_sided_buckets(&mut self, count: i64) {
        if count >= self.window_buckets as i64 {
            self.imbalances.clear();
            self.imbalances
                .resize(self.window_buckets, self.bucket_volume_units);
            self.imbalance_sum = self
                .bucket_volume_units
                .saturating_mul(self.window_buckets as i64);
            return;
        }
        for _ in 0..count {
            self.push_bucket(self.bucket_volume_units);
        }
    }

    #[cfg(test)]
    fn partial_volume(&self) -> i64 {
        self.buy_units + self.sell_units
    }

    fn push_bucket(&mut self, imbalance: i64) {
        self.imbalances.push_back(imbalance);
        self.imbalance_sum = self.imbalance_sum.saturating_add(imbalance);
        while self.imbalances.len() > self.window_buckets {
            if let Some(old) = self.imbalances.pop_front() {
                self.imbalance_sum = self.imbalance_sum.saturating_sub(old);
            }
        }
    }

    pub fn value(&self) -> Option<f64> {
        if self.imbalances.len() < self.window_buckets {
            return None;
        }
        let denominator = self.window_buckets as f64 * self.bucket_volume_units as f64;
        if denominator <= 0.0 {
            return None;
        }
        let value = self.imbalance_sum as f64 / denominator;
        debug_assert!((0.0..=1.0).contains(&value));
        Some(value)
    }

    pub fn buckets_seen(&self) -> usize {
        self.imbalances.len()
    }
}

/// Why the guard is currently withholding quotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripCause {
    FastMove,
    Vpin,
}

/// Trip-and-re-entry state machine.
///
/// Re-entry requires **both** conditions, deliberately: VPIN back under
/// threshold *and* the cooldown elapsed. VPIN alone can dip mid-cascade as a
/// bucket completes, and a timer alone would re-enter a live cascade. On the
/// 2026-08-22 event VPIN cleared ~16 minutes after the trip, about when price
/// stabilised, so the two conditions agreed rather than fighting.
#[derive(Debug)]
pub struct FlowGuard {
    config: FlowGuardConfig,
    tripped_at_ms: Option<u64>,
    cause: Option<TripCause>,
    trips: u64,
}

impl FlowGuard {
    pub fn new(config: FlowGuardConfig) -> Self {
        Self {
            config,
            tripped_at_ms: None,
            cause: None,
            trips: 0,
        }
    }

    pub const fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Fold in the latest observations and return whether quoting is withheld.
    ///
    /// `vpin` is `None` while the tracker warms up, which must not be treated
    /// as safe *or* as toxic — during warm-up only the fast breaker is armed.
    pub fn evaluate(&mut self, now_ms: u64, move_bps: f64, vpin: Option<f64>) -> bool {
        if !self.config.enabled {
            return false;
        }
        if move_bps >= self.config.fast_move_threshold_bps {
            self.trip(now_ms, TripCause::FastMove);
        } else if vpin.is_some_and(|value| value >= self.config.vpin_threshold) {
            self.trip(now_ms, TripCause::Vpin);
        }
        let Some(tripped_at_ms) = self.tripped_at_ms else {
            return false;
        };
        let elapsed_ms = now_ms.saturating_sub(tripped_at_ms);
        let reopen = match vpin {
            Some(value) => {
                elapsed_ms >= self.config.cooldown_ms && value < self.config.vpin_threshold
            }
            // VPIN is warming up. Re-entering on a statistic we do not have
            // would defeat the cooldown, but staying closed for the whole
            // warm-up (~14 h after a restart) froze inventory for hours. The
            // warm-up rule is not a bare timer: the trailing mid move — the
            // same window the breaker reads — must be calm at a stricter line
            // than the trip line, over a longer cooldown, and `trip` above
            // has already re-armed the clock on any fresh breach.
            None => {
                elapsed_ms >= self.config.warmup_reentry_cooldown_ms
                    && move_bps
                        < self.config.fast_move_threshold_bps
                            * self.config.warmup_reentry_calm_fraction
            }
        };
        if reopen {
            self.tripped_at_ms = None;
            self.cause = None;
            return false;
        }
        true
    }

    fn trip(&mut self, now_ms: u64, cause: TripCause) {
        if self.tripped_at_ms.is_none() {
            self.trips = self.trips.saturating_add(1);
        }
        // Re-arm the clock on every fresh breach so a cascade cannot expire its
        // own cooldown while still running.
        self.tripped_at_ms = Some(now_ms);
        self.cause = Some(cause);
    }

    pub const fn is_tripped(&self) -> bool {
        self.tripped_at_ms.is_some()
    }

    pub const fn cause(&self) -> Option<TripCause> {
        self.cause
    }

    pub const fn trips(&self) -> u64 {
        self.trips
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn trade(id: u64, side: AggressorSide, qty: i64) -> TradePrint {
        TradePrint {
            aggressor: side,
            px: 100,
            qty_units: qty,
            exchange_ms: id,
            recv_ns: id * 1_000_000,
            trade_id: id,
        }
    }

    fn config() -> FlowGuardConfig {
        FlowGuardConfig {
            enabled: true,
            fast_move_window_ms: 5_000,
            fast_move_threshold_bps: 800.0,
            vpin_buckets_per_day: 50,
            vpin_window_buckets: 2,
            vpin_threshold: 0.40,
            cooldown_ms: 60_000,
            warmup_reentry_cooldown_ms: 1_800_000,
            warmup_reentry_calm_fraction: 0.5,
            reduce_only_while_tripped: false,
        }
    }

    #[test]
    fn mid_window_measures_the_move_against_the_oldest_in_window_sample() {
        let mut window = MidWindow::new(16, 5_000);
        let second = 1_000_000_000_u64;
        assert_eq!(window.observe(second, 10_000), 0.0);
        // 10% down inside the window.
        let moved = window.observe(second + 2 * second, 9_000);
        assert!(
            (moved - 1_000.0).abs() < 1.0,
            "expected ~1000 bps, got {moved}"
        );
    }

    #[test]
    fn mid_window_forgets_samples_older_than_the_window() {
        let mut window = MidWindow::new(16, 5_000);
        let second = 1_000_000_000_u64;
        window.observe(second, 10_000);
        // Ten seconds later the old sample is outside a five second window, so
        // the only reference left is this observation itself: no move.
        let moved = window.observe(second + 10 * second, 5_000);
        assert_eq!(
            moved, 0.0,
            "a sample outside the window must not be a reference"
        );
    }

    #[test]
    fn mid_window_survives_wraparound() {
        let mut window = MidWindow::new(4, 60_000);
        let second = 1_000_000_000_u64;
        for i in 0..12 {
            window.observe(second * (i + 1), 10_000);
        }
        // Every retained sample is 10_000, so a 5% move must read as 500 bps
        // even though the ring wrapped three times.
        let moved = window.observe(second * 13, 10_500);
        assert!(
            (moved - 500.0).abs() < 1.0,
            "expected ~500 bps, got {moved}"
        );
    }

    #[test]
    fn control_signal_storm_does_not_shorten_the_bbo_window() {
        let mut window = MidWindow::new(4, 5_000);
        let second = 1_000_000_000_u64;
        window.observe(second, 10_000);
        window.observe(2 * second, 9_000);
        for _ in 0..10_000 {
            let moved = window.advance(4 * second, 9_000);
            assert!((moved - 1_000.0).abs() < 1.0);
        }
        assert_eq!(window.len(), 2, "non-market wakes must not insert samples");
    }

    #[test]
    fn vpin_is_one_when_flow_is_entirely_one_sided() {
        let mut tracker = VpinTracker::new(100, 2);
        // Two full buckets of pure buying: |buy - sell| == bucket size each.
        for id in 0..4 {
            tracker.observe(&trade(id, AggressorSide::Buy, 50));
        }
        let value = tracker.value().expect("two buckets should be complete");
        assert!(
            (value - 1.0).abs() < 1e-9,
            "perfectly toxic flow is VPIN 1, got {value}"
        );
    }

    #[test]
    fn vpin_is_zero_when_flow_is_perfectly_balanced() {
        let mut tracker = VpinTracker::new(100, 2);
        for id in 0..4 {
            let side = if id % 2 == 0 {
                AggressorSide::Buy
            } else {
                AggressorSide::Sell
            };
            tracker.observe(&trade(id, side, 50));
        }
        let value = tracker.value().expect("two buckets should be complete");
        assert!(value.abs() < 1e-9, "balanced flow is VPIN 0, got {value}");
    }

    #[test]
    fn vpin_ignores_a_repeated_trade_id() {
        let mut tracker = VpinTracker::new(100, 1);
        tracker.observe(&trade(7, AggressorSide::Buy, 60));
        // The same print arriving twice across a reconnect must not count twice.
        tracker.observe(&trade(7, AggressorSide::Buy, 60));
        assert_eq!(
            tracker.buckets_seen(),
            0,
            "a duplicate must not complete a bucket"
        );
        tracker.observe(&trade(8, AggressorSide::Buy, 60));
        assert_eq!(tracker.buckets_seen(), 1);
    }

    #[test]
    fn oversized_one_sided_trade_cannot_produce_vpin_above_one() {
        let mut tracker = VpinTracker::new(100, 2);
        let value = tracker
            .observe(&trade(1, AggressorSide::Buy, 250))
            .expect("one print completes two retained buckets");
        assert_eq!(value, 1.0);
        assert_eq!(tracker.partial_volume(), 50);
    }

    #[test]
    fn mixed_side_overshoot_is_sliced_in_arrival_order() {
        let mut tracker = VpinTracker::new(100, 2);
        tracker.observe(&trade(1, AggressorSide::Buy, 80));
        tracker.observe(&trade(2, AggressorSide::Sell, 70));
        let value = tracker
            .observe(&trade(3, AggressorSide::Buy, 50))
            .expect("two exact buckets should be complete");
        // Bucket one is 80B/20S (imbalance 60), bucket two 50B/50S.
        assert_eq!(value, 0.30);
        assert_eq!(tracker.partial_volume(), 0);
    }

    #[test]
    fn resizing_bucket_clears_every_old_scale_bucket() {
        let mut tracker = VpinTracker::new(100, 2);
        tracker.observe(&trade(1, AggressorSide::Buy, 200));
        assert!(tracker.value().is_some());
        tracker.resize_bucket(200);
        assert!(tracker.value().is_none());
        assert_eq!(tracker.buckets_seen(), 0);
        assert_eq!(tracker.partial_volume(), 0);
    }

    #[test]
    fn nonconsecutive_duplicate_trade_ids_are_ignored() {
        let mut tracker = VpinTracker::new(100, 1);
        tracker.observe(&trade(7, AggressorSide::Buy, 60));
        tracker.observe(&trade(8, AggressorSide::Sell, 10));
        tracker.observe(&trade(7, AggressorSide::Buy, 60));
        assert!(tracker.value().is_none());
        assert_eq!(tracker.partial_volume(), 70);
    }

    proptest! {
        #[test]
        fn vpin_is_bounded_for_arbitrary_trade_sequences(
            bucket in 1_i64..10_000,
            window in 1_usize..32,
            rows in prop::collection::vec((any::<bool>(), 0_i64..100_000), 0..256),
        ) {
            let mut tracker = VpinTracker::new(bucket, window);
            for (index, (buy, qty)) in rows.into_iter().enumerate() {
                let value = tracker.observe(&trade(
                    index as u64 + 1,
                    if buy { AggressorSide::Buy } else { AggressorSide::Sell },
                    qty,
                ));
                if let Some(value) = value {
                    prop_assert!((0.0..=1.0).contains(&value));
                }
                prop_assert!(tracker.partial_volume() < bucket);
            }
        }
    }

    #[test]
    fn the_fast_breaker_trips_and_the_cooldown_alone_does_not_clear_it() {
        let mut guard = FlowGuard::new(config());
        assert!(!guard.evaluate(0, 100.0, Some(0.1)));
        // 9% move: above the 8% threshold.
        assert!(guard.evaluate(1_000, 900.0, Some(0.1)));
        assert_eq!(guard.cause(), Some(TripCause::FastMove));
        // Cooldown elapsed but flow still toxic: stays closed.
        assert!(guard.evaluate(1_000 + 60_000, 10.0, Some(0.9)));
        // Both conditions met: reopens.
        assert!(!guard.evaluate(1_000 + 120_000, 10.0, Some(0.1)));
        assert_eq!(guard.trips(), 1);
    }

    #[test]
    fn vpin_alone_does_not_clear_the_guard_before_the_cooldown() {
        let mut guard = FlowGuard::new(config());
        assert!(guard.evaluate(0, 10.0, Some(0.9)));
        assert_eq!(guard.cause(), Some(TripCause::Vpin));
        // Flow calm immediately, but the cooldown has not elapsed.
        assert!(guard.evaluate(1_000, 10.0, Some(0.05)));
        assert!(!guard.evaluate(60_000, 10.0, Some(0.05)));
    }

    #[test]
    fn a_fresh_breach_rearms_the_cooldown() {
        let mut guard = FlowGuard::new(config());
        assert!(guard.evaluate(0, 900.0, Some(0.1)));
        // A second breach most of the way through the cooldown restarts it, so
        // a cascade cannot expire its own timer while still running.
        assert!(guard.evaluate(50_000, 900.0, Some(0.1)));
        assert!(guard.evaluate(70_000, 10.0, Some(0.1)));
        assert!(!guard.evaluate(110_001, 10.0, Some(0.1)));
    }

    /// Inside the warm-up cooldown (1 800 s here) an unknown VPIN keeps the
    /// guard closed: the ordinary 60 s cooldown does not apply without the
    /// statistic. This is *not* "never re-opens"; see the warm-up tests below.
    #[test]
    fn unknown_vpin_keeps_a_tripped_guard_closed() {
        let mut guard = FlowGuard::new(config());
        assert!(guard.evaluate(0, 900.0, None));
        // Warming up: no VPIN available. Re-entering on a statistic we do not
        // have would defeat the cooldown.
        assert!(guard.evaluate(600_000, 10.0, None));
    }

    #[test]
    fn warmup_reentry_reopens_after_the_longer_cooldown_when_the_mid_is_calm() {
        let mut guard = FlowGuard::new(config());
        assert!(guard.evaluate(0, 900.0, None));
        // The ordinary cooldown is not enough without VPIN.
        assert!(guard.evaluate(60_000, 10.0, None));
        // The longer cooldown alone is not enough either: 500 bps is above the
        // 0.5 x 800 calm line.
        assert!(guard.evaluate(1_800_000, 500.0, None));
        // Longer cooldown and a calm window: re-open.
        assert!(!guard.evaluate(1_800_000, 10.0, None));
        assert!(!guard.is_tripped());
    }

    #[test]
    fn a_live_cascade_cannot_expire_its_own_warmup_cooldown() {
        let mut guard = FlowGuard::new(config());
        assert!(guard.evaluate(0, 900.0, None));
        let mut now_ms = 0;
        for _ in 0..5 {
            now_ms += 1_800_000 - 1;
            // A fresh breach re-arms the clock every time.
            assert!(guard.evaluate(now_ms, 900.0, None));
            assert!(
                guard.evaluate(now_ms + 1, 10.0, None),
                "re-armed by the breach"
            );
        }
        assert!(guard.is_tripped());
    }

    #[test]
    fn a_disabled_guard_never_withholds_quotes() {
        let mut disabled = config();
        disabled.enabled = false;
        let mut guard = FlowGuard::new(disabled);
        assert!(!guard.evaluate(0, 10_000.0, Some(0.99)));
        assert!(!guard.is_tripped());
    }
}
