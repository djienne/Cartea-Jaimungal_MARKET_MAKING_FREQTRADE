use crate::types::{Bbo, DesiredQuotes, OrderIntent, QuoteReason, Side};
use crossbeam_queue::ArrayQueue;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::thread::{self, Thread};
use std::time::Duration;
use tokio::sync::{watch, Notify};

pub const HOT_SIGNAL_MARKET: usize = 1 << 0;
pub const HOT_SIGNAL_MODEL: usize = 1 << 1;
pub const HOT_SIGNAL_EXECUTION: usize = 1 << 2;
pub const HOT_SIGNAL_EPISODE: usize = 1 << 3;
pub const HOT_SIGNAL_SHUTDOWN: usize = 1 << 4;

#[derive(Debug, Default)]
#[repr(align(64))]
pub struct HotPathSignal {
    pending: AtomicUsize,
    hot_thread: OnceLock<Thread>,
}

impl HotPathSignal {
    pub fn register_current_thread(&self) {
        let current = thread::current();
        let _ = self.hot_thread.set(current.clone());
        if self.pending.load(Ordering::Acquire) != 0 {
            current.unpark();
        }
    }

    #[inline]
    pub fn notify(&self, bits: usize) -> bool {
        debug_assert_ne!(bits, 0);
        let previous = self.pending.fetch_or(bits, Ordering::Release);
        if previous == 0 {
            if let Some(thread) = self.hot_thread.get() {
                thread.unpark();
            }
        }
        previous & bits != bits
    }

    #[inline]
    pub fn take_pending(&self) -> usize {
        self.pending.swap(0, Ordering::AcqRel)
    }

    #[inline]
    pub fn park_timeout(&self, timeout: Duration) {
        thread::park_timeout(timeout);
    }
}

/// Single-writer coherent latest-value publication for the BBO.
#[derive(Debug, Default)]
#[repr(align(64))]
pub struct AtomicBbo {
    seq: AtomicU64,
    valid: AtomicBool,
    bid_px: AtomicI64,
    bid_sz: AtomicI64,
    ask_px: AtomicI64,
    ask_sz: AtomicI64,
    exchange_ms: AtomicU64,
    recv_ns: AtomicU64,
}

impl AtomicBbo {
    #[inline]
    pub fn store(&self, bbo: Bbo) {
        self.seq.fetch_add(1, Ordering::AcqRel);
        self.bid_px.store(bbo.bid_px, Ordering::Relaxed);
        self.bid_sz.store(bbo.bid_sz, Ordering::Relaxed);
        self.ask_px.store(bbo.ask_px, Ordering::Relaxed);
        self.ask_sz.store(bbo.ask_sz, Ordering::Relaxed);
        self.exchange_ms.store(bbo.exchange_ms, Ordering::Relaxed);
        self.recv_ns.store(bbo.recv_ns, Ordering::Relaxed);
        self.valid.store(bbo.is_valid(), Ordering::Relaxed);
        self.seq.fetch_add(1, Ordering::Release);
    }

    #[inline]
    pub fn load(&self) -> Option<Bbo> {
        loop {
            let before = self.seq.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let valid = self.valid.load(Ordering::Relaxed);
            let value = Bbo {
                bid_px: self.bid_px.load(Ordering::Relaxed),
                bid_sz: self.bid_sz.load(Ordering::Relaxed),
                ask_px: self.ask_px.load(Ordering::Relaxed),
                ask_sz: self.ask_sz.load(Ordering::Relaxed),
                exchange_ms: self.exchange_ms.load(Ordering::Relaxed),
                recv_ns: self.recv_ns.load(Ordering::Relaxed),
            };
            let after = self.seq.load(Ordering::Acquire);
            if before == after && before & 1 == 0 {
                return valid.then_some(value);
            }
            std::hint::spin_loop();
        }
    }
}

#[derive(Debug)]
struct AtomicIntent {
    valid: AtomicBool,
    side: AtomicU8,
    px: AtomicI64,
    qty_units: AtomicI64,
    post_only: AtomicBool,
    reduce_only: AtomicBool,
}

impl AtomicIntent {
    const fn bid() -> Self {
        Self {
            valid: AtomicBool::new(false),
            side: AtomicU8::new(0),
            px: AtomicI64::new(0),
            qty_units: AtomicI64::new(0),
            post_only: AtomicBool::new(true),
            reduce_only: AtomicBool::new(false),
        }
    }

    const fn ask() -> Self {
        Self {
            valid: AtomicBool::new(false),
            side: AtomicU8::new(1),
            px: AtomicI64::new(0),
            qty_units: AtomicI64::new(0),
            post_only: AtomicBool::new(true),
            reduce_only: AtomicBool::new(false),
        }
    }

    fn store(&self, value: Option<OrderIntent>) {
        if let Some(value) = value {
            self.side.store(side_to_u8(value.side), Ordering::Relaxed);
            self.px.store(value.px, Ordering::Relaxed);
            self.qty_units.store(value.qty_units, Ordering::Relaxed);
            self.post_only.store(value.post_only, Ordering::Relaxed);
            self.reduce_only.store(value.reduce_only, Ordering::Relaxed);
            self.valid.store(true, Ordering::Relaxed);
        } else {
            self.valid.store(false, Ordering::Relaxed);
        }
    }

    fn load(&self) -> Option<OrderIntent> {
        self.valid.load(Ordering::Relaxed).then(|| OrderIntent {
            side: side_from_u8(self.side.load(Ordering::Relaxed)),
            px: self.px.load(Ordering::Relaxed),
            qty_units: self.qty_units.load(Ordering::Relaxed),
            post_only: self.post_only.load(Ordering::Relaxed),
            reduce_only: self.reduce_only.load(Ordering::Relaxed),
        })
    }
}

#[derive(Debug)]
#[repr(align(64))]
struct AtomicDesiredQuotes {
    seq: AtomicU64,
    bid: AtomicIntent,
    ask: AtomicIntent,
    quote_seq: AtomicU64,
    model_revision: AtomicU64,
    generated_ns: AtomicU64,
    reason: AtomicU8,
    mid: AtomicU64,
    q_exact: AtomicU64,
    q_rounded: AtomicI64,
    tau_remaining: AtomicU64,
}

impl Default for AtomicDesiredQuotes {
    fn default() -> Self {
        Self {
            seq: AtomicU64::new(0),
            bid: AtomicIntent::bid(),
            ask: AtomicIntent::ask(),
            quote_seq: AtomicU64::new(0),
            model_revision: AtomicU64::new(0),
            generated_ns: AtomicU64::new(0),
            reason: AtomicU8::new(reason_to_u8(QuoteReason::Startup)),
            mid: AtomicU64::new(0.0_f64.to_bits()),
            q_exact: AtomicU64::new(0.0_f64.to_bits()),
            q_rounded: AtomicI64::new(0),
            tau_remaining: AtomicU64::new(0.0_f64.to_bits()),
        }
    }
}

impl AtomicDesiredQuotes {
    fn store(&self, value: DesiredQuotes) {
        self.seq.fetch_add(1, Ordering::AcqRel);
        self.bid.store(value.bid);
        self.ask.store(value.ask);
        self.quote_seq.store(value.quote_seq, Ordering::Relaxed);
        self.model_revision
            .store(value.model_revision, Ordering::Relaxed);
        self.generated_ns
            .store(value.generated_ns, Ordering::Relaxed);
        self.reason
            .store(reason_to_u8(value.reason), Ordering::Relaxed);
        self.mid.store(value.mid.to_bits(), Ordering::Relaxed);
        self.q_exact
            .store(value.q_exact.to_bits(), Ordering::Relaxed);
        self.q_rounded.store(value.q_rounded, Ordering::Relaxed);
        self.tau_remaining
            .store(value.tau_remaining.to_bits(), Ordering::Relaxed);
        self.seq.fetch_add(1, Ordering::Release);
    }

    fn load(&self) -> DesiredQuotes {
        loop {
            let before = self.seq.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let value = DesiredQuotes {
                bid: self.bid.load(),
                ask: self.ask.load(),
                quote_seq: self.quote_seq.load(Ordering::Relaxed),
                model_revision: self.model_revision.load(Ordering::Relaxed),
                generated_ns: self.generated_ns.load(Ordering::Relaxed),
                reason: reason_from_u8(self.reason.load(Ordering::Relaxed)),
                mid: f64::from_bits(self.mid.load(Ordering::Relaxed)),
                q_exact: f64::from_bits(self.q_exact.load(Ordering::Relaxed)),
                q_rounded: self.q_rounded.load(Ordering::Relaxed),
                tau_remaining: f64::from_bits(self.tau_remaining.load(Ordering::Relaxed)),
            };
            let after = self.seq.load(Ordering::Acquire);
            if before == after && before & 1 == 0 {
                return value;
            }
            std::hint::spin_loop();
        }
    }
}

pub struct SharedQuotes {
    value: AtomicDesiredQuotes,
    waiters: AtomicUsize,
    changed: Notify,
}

impl Default for SharedQuotes {
    fn default() -> Self {
        Self {
            value: AtomicDesiredQuotes::default(),
            waiters: AtomicUsize::new(0),
            changed: Notify::new(),
        }
    }
}

impl SharedQuotes {
    #[inline]
    pub fn publish(&self, quotes: DesiredQuotes) {
        self.value.store(quotes);
        if self.waiters.load(Ordering::Acquire) != 0 {
            self.changed.notify_one();
        }
    }

    #[inline]
    pub fn load(&self) -> DesiredQuotes {
        self.value.load()
    }

    pub async fn changed_after(&self, observed_quote_seq: u64) -> DesiredQuotes {
        loop {
            let current = self.load();
            if current.quote_seq != observed_quote_seq {
                return current;
            }
            let notified = self.changed.notified();
            tokio::pin!(notified);
            self.waiters.fetch_add(1, Ordering::AcqRel);
            notified.as_mut().enable();
            let current = self.load();
            if current.quote_seq != observed_quote_seq {
                self.waiters.fetch_sub(1, Ordering::AcqRel);
                return current;
            }
            notified.as_mut().await;
            self.waiters.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Bounded MPMC queue. Successful operations are lock-free; asynchronous
/// notification is used only when a producer or consumer must wait.
pub struct AsyncRing<T> {
    queue: ArrayQueue<T>,
    high_water: AtomicUsize,
    waiting_producers: AtomicUsize,
    waiting_consumers: AtomicUsize,
    not_empty: Notify,
    not_full: Notify,
}

impl<T> AsyncRing<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring capacity must be positive");
        Self {
            queue: ArrayQueue::new(capacity),
            high_water: AtomicUsize::new(0),
            waiting_producers: AtomicUsize::new(0),
            waiting_consumers: AtomicUsize::new(0),
            not_empty: Notify::new(),
            not_full: Notify::new(),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    #[inline]
    pub fn high_water_mark(&self) -> usize {
        self.high_water.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn try_push(&self, value: T) -> Result<(), T> {
        self.queue.push(value).map(|()| {
            self.observe_high_water();
            if self.waiting_consumers.load(Ordering::Acquire) != 0 {
                self.not_empty.notify_one();
            }
        })
    }

    pub async fn push_or_shutdown(
        &self,
        mut value: T,
        shutdown: &mut watch::Receiver<bool>,
    ) -> Result<bool, T> {
        let mut backpressured = false;
        loop {
            if *shutdown.borrow() {
                return Err(value);
            }
            match self.try_push(value) {
                Ok(()) => return Ok(backpressured),
                Err(returned) => {
                    value = returned;
                    backpressured = true;
                }
            }
            let notified = self.not_full.notified();
            tokio::pin!(notified);
            self.waiting_producers.fetch_add(1, Ordering::AcqRel);
            notified.as_mut().enable();
            match self.try_push(value) {
                Ok(()) => {
                    self.waiting_producers.fetch_sub(1, Ordering::AcqRel);
                    return Ok(backpressured);
                }
                Err(returned) => value = returned,
            }
            tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    self.waiting_producers.fetch_sub(1, Ordering::AcqRel);
                    if changed.is_err() || *shutdown.borrow() {
                        return Err(value);
                    }
                }
                () = notified.as_mut() => {
                    self.waiting_producers.fetch_sub(1, Ordering::AcqRel);
                }
            }
        }
    }

    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        let value = self.queue.pop();
        if value.is_some() && self.waiting_producers.load(Ordering::Acquire) != 0 {
            self.not_full.notify_one();
        }
        value
    }

    pub async fn pop(&self) -> T {
        loop {
            if let Some(value) = self.try_pop() {
                return value;
            }
            let notified = self.not_empty.notified();
            tokio::pin!(notified);
            self.waiting_consumers.fetch_add(1, Ordering::AcqRel);
            notified.as_mut().enable();
            if let Some(value) = self.try_pop() {
                self.waiting_consumers.fetch_sub(1, Ordering::AcqRel);
                return value;
            }
            notified.as_mut().await;
            self.waiting_consumers.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn observe_high_water(&self) {
        let depth = self.queue.len();
        self.high_water.fetch_max(depth, Ordering::Relaxed);
    }
}

const fn side_to_u8(side: Side) -> u8 {
    match side {
        Side::Buy => 0,
        Side::Sell => 1,
    }
}

const fn side_from_u8(value: u8) -> Side {
    if value == 1 {
        Side::Sell
    } else {
        Side::Buy
    }
}

const fn reason_to_u8(reason: QuoteReason) -> u8 {
    match reason {
        QuoteReason::Startup => 0,
        QuoteReason::Market => 1,
        QuoteReason::Fill => 2,
        QuoteReason::Calibration => 3,
        QuoteReason::Episode => 4,
        QuoteReason::StaleMarket => 5,
        QuoteReason::StaleCalibration => 6,
        QuoteReason::RiskLimit => 7,
        QuoteReason::InvalidRun => 8,
        QuoteReason::Shutdown => 9,
    }
}

const fn reason_from_u8(value: u8) -> QuoteReason {
    match value {
        1 => QuoteReason::Market,
        2 => QuoteReason::Fill,
        3 => QuoteReason::Calibration,
        4 => QuoteReason::Episode,
        5 => QuoteReason::StaleMarket,
        6 => QuoteReason::StaleCalibration,
        7 => QuoteReason::RiskLimit,
        8 => QuoteReason::InvalidRun,
        9 => QuoteReason::Shutdown,
        _ => QuoteReason::Startup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_bbo_round_trip() {
        let slot = AtomicBbo::default();
        let bbo = Bbo {
            bid_px: 100,
            bid_sz: 4,
            ask_px: 102,
            ask_sz: 5,
            exchange_ms: 7,
            recv_ns: 9,
        };
        slot.store(bbo);
        assert_eq!(slot.load(), Some(bbo));
    }

    #[tokio::test]
    async fn shared_quotes_coalesces() {
        let shared = SharedQuotes::default();
        shared.publish(DesiredQuotes {
            quote_seq: 2,
            ..DesiredQuotes::default()
        });
        assert_eq!(shared.changed_after(0).await.quote_seq, 2);
    }

    #[tokio::test]
    async fn ring_is_bounded_and_fifo() {
        let ring = AsyncRing::new(2);
        assert!(ring.try_push(1).is_ok());
        assert!(ring.try_push(2).is_ok());
        assert_eq!(ring.try_push(3), Err(3));
        assert_eq!(ring.pop().await, 1);
        assert_eq!(ring.pop().await, 2);
    }
}
