use crate::types::{Bbo, DesiredQuotes, OrderIntent, QuoteReason, Side};
use crossbeam_queue::ArrayQueue;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::thread::{self, Thread};
use std::time::Duration;
use tokio::sync::Notify;

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
        self.bid_px.store(bbo.bid_px, Ordering::Release);
        self.bid_sz.store(bbo.bid_sz, Ordering::Release);
        self.ask_px.store(bbo.ask_px, Ordering::Release);
        self.ask_sz.store(bbo.ask_sz, Ordering::Release);
        self.exchange_ms.store(bbo.exchange_ms, Ordering::Release);
        self.recv_ns.store(bbo.recv_ns, Ordering::Release);
        self.valid.store(bbo.is_valid(), Ordering::Release);
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
            let valid = self.valid.load(Ordering::Acquire);
            let value = Bbo {
                bid_px: self.bid_px.load(Ordering::Acquire),
                bid_sz: self.bid_sz.load(Ordering::Acquire),
                ask_px: self.ask_px.load(Ordering::Acquire),
                ask_sz: self.ask_sz.load(Ordering::Acquire),
                exchange_ms: self.exchange_ms.load(Ordering::Acquire),
                recv_ns: self.recv_ns.load(Ordering::Acquire),
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
            self.side.store(side_to_u8(value.side), Ordering::Release);
            self.px.store(value.px, Ordering::Release);
            self.qty_units.store(value.qty_units, Ordering::Release);
            self.post_only.store(value.post_only, Ordering::Release);
            self.reduce_only.store(value.reduce_only, Ordering::Release);
            self.valid.store(true, Ordering::Release);
        } else {
            self.valid.store(false, Ordering::Release);
        }
    }

    fn load(&self) -> Option<OrderIntent> {
        self.valid.load(Ordering::Acquire).then(|| OrderIntent {
            side: side_from_u8(self.side.load(Ordering::Acquire)),
            px: self.px.load(Ordering::Acquire),
            qty_units: self.qty_units.load(Ordering::Acquire),
            post_only: self.post_only.load(Ordering::Acquire),
            reduce_only: self.reduce_only.load(Ordering::Acquire),
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
    source_recv_ns: AtomicU64,
    source_exchange_ms: AtomicU64,
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
            source_recv_ns: AtomicU64::new(0),
            source_exchange_ms: AtomicU64::new(0),
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
        self.quote_seq.store(value.quote_seq, Ordering::Release);
        self.model_revision
            .store(value.model_revision, Ordering::Release);
        self.generated_ns
            .store(value.generated_ns, Ordering::Release);
        self.source_recv_ns
            .store(value.source_recv_ns, Ordering::Release);
        self.source_exchange_ms
            .store(value.source_exchange_ms, Ordering::Release);
        self.reason
            .store(reason_to_u8(value.reason), Ordering::Release);
        self.mid.store(value.mid.to_bits(), Ordering::Release);
        self.q_exact
            .store(value.q_exact.to_bits(), Ordering::Release);
        self.q_rounded.store(value.q_rounded, Ordering::Release);
        self.tau_remaining
            .store(value.tau_remaining.to_bits(), Ordering::Release);
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
                quote_seq: self.quote_seq.load(Ordering::Acquire),
                model_revision: self.model_revision.load(Ordering::Acquire),
                generated_ns: self.generated_ns.load(Ordering::Acquire),
                source_recv_ns: self.source_recv_ns.load(Ordering::Acquire),
                source_exchange_ms: self.source_exchange_ms.load(Ordering::Acquire),
                reason: reason_from_u8(self.reason.load(Ordering::Acquire)),
                mid: f64::from_bits(self.mid.load(Ordering::Acquire)),
                q_exact: f64::from_bits(self.q_exact.load(Ordering::Acquire)),
                q_rounded: self.q_rounded.load(Ordering::Acquire),
                tau_remaining: f64::from_bits(self.tau_remaining.load(Ordering::Acquire)),
            };
            let after = self.seq.load(Ordering::Acquire);
            if before == after && before & 1 == 0 {
                return value;
            }
            std::hint::spin_loop();
        }
    }
}

/// Increments a waiter counter for exactly as long as the waiting future is
/// alive. `tokio::select!` drops losing futures mid-await; without the guard
/// the decrement on the completion path never ran, the counter leaked upward
/// forever, and the notify-elision it exists for was permanently defeated.
struct WaiterGuard<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> WaiterGuard<'a> {
    fn register(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
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
            // Guard, not manual decrement: this future is routinely dropped
            // mid-await by select loops.
            let _waiter = WaiterGuard::register(&self.waiters);
            notified.as_mut().enable();
            let current = self.load();
            if current.quote_seq != observed_quote_seq {
                return current;
            }
            notified.as_mut().await;
        }
    }

    #[cfg(test)]
    pub(crate) fn waiter_count(&self) -> usize {
        self.waiters.load(Ordering::Acquire)
    }
}

/// Bounded MPMC queue. Successful operations are lock-free; asynchronous
/// notification is used only when a consumer must wait. Producers never wait:
/// `try_push` fails on a full ring, and both feeds treat that as evidence
/// loss rather than something to block on.
pub struct AsyncRing<T> {
    queue: ArrayQueue<T>,
    high_water: AtomicUsize,
    waiting_consumers: AtomicUsize,
    not_empty: Notify,
}

impl<T> AsyncRing<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring capacity must be positive");
        Self {
            queue: ArrayQueue::new(capacity),
            high_water: AtomicUsize::new(0),
            waiting_consumers: AtomicUsize::new(0),
            not_empty: Notify::new(),
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

    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        self.queue.pop()
    }

    pub async fn pop(&self) -> T {
        loop {
            if let Some(value) = self.try_pop() {
                return value;
            }
            let notified = self.not_empty.notified();
            tokio::pin!(notified);
            // Guard, not manual decrement: select loops drop this future
            // mid-await on every iteration another branch wins.
            let _waiter = WaiterGuard::register(&self.waiting_consumers);
            notified.as_mut().enable();
            if let Some(value) = self.try_pop() {
                return value;
            }
            notified.as_mut().await;
        }
    }

    #[cfg(test)]
    pub(crate) fn waiter_count(&self) -> usize {
        self.waiting_consumers.load(Ordering::Acquire)
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
        QuoteReason::LatencyLimit => 10,
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
        10 => QuoteReason::LatencyLimit,
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

    #[test]
    fn atomic_bbo_remains_coherent_under_contention() {
        let slot = std::sync::Arc::new(AtomicBbo::default());
        slot.store(Bbo {
            bid_px: 1,
            bid_sz: 10,
            ask_px: 2,
            ask_sz: 20,
            exchange_ms: 1,
            recv_ns: 100,
        });
        let writer_slot = slot.clone();
        let writer = std::thread::spawn(move || {
            for value in 1..=100_000_i64 {
                writer_slot.store(Bbo {
                    bid_px: value,
                    bid_sz: value * 10,
                    ask_px: value + 1,
                    ask_sz: value * 20,
                    exchange_ms: value as u64,
                    recv_ns: value as u64 * 100,
                });
            }
        });
        let mut readers = Vec::new();
        for _ in 0..4 {
            let slot = slot.clone();
            readers.push(std::thread::spawn(move || {
                for _ in 0..100_000 {
                    let bbo = slot.load().unwrap();
                    assert_eq!(bbo.ask_px, bbo.bid_px + 1);
                    assert_eq!(bbo.bid_sz, bbo.bid_px * 10);
                    assert_eq!(bbo.ask_sz, bbo.bid_px * 20);
                    assert_eq!(bbo.recv_ns, bbo.exchange_ms * 100);
                }
            }));
        }
        writer.join().unwrap();
        for reader in readers {
            reader.join().unwrap();
        }
    }

    #[tokio::test]
    async fn shared_quotes_coalesces() {
        let shared = SharedQuotes::default();
        shared.publish(DesiredQuotes {
            quote_seq: 2,
            generated_ns: 11,
            source_recv_ns: 7,
            ..DesiredQuotes::default()
        });
        let observed = shared.changed_after(0).await;
        assert_eq!(observed.quote_seq, 2);
        assert_eq!(observed.generated_ns, 11);
        assert_eq!(observed.source_recv_ns, 7);
    }

    #[test]
    fn latency_limit_reason_survives_atomic_publication() {
        let shared = SharedQuotes::default();
        shared.publish(DesiredQuotes {
            quote_seq: 1,
            reason: QuoteReason::LatencyLimit,
            ..DesiredQuotes::default()
        });
        assert_eq!(shared.load().reason, QuoteReason::LatencyLimit);
    }

    /// Dropping a waiting future mid-await — what `tokio::select!` does on
    /// every iteration another branch wins — must release its waiter slot.
    /// Before the drop guard, each such drop leaked the counter upward and
    /// permanently defeated the notify-elision.
    #[tokio::test]
    async fn dropped_waiters_release_their_counter_slots() {
        let shared = SharedQuotes::default();
        for _ in 0..8 {
            let waiting = shared.changed_after(0);
            tokio::pin!(waiting);
            // Poll once so the future registers as a waiter, then drop it.
            let poll = futures_util::poll!(waiting.as_mut());
            assert!(poll.is_pending());
        }
        assert_eq!(shared.waiter_count(), 0);

        let ring: AsyncRing<u64> = AsyncRing::new(2);
        for _ in 0..8 {
            let waiting = ring.pop();
            tokio::pin!(waiting);
            let poll = futures_util::poll!(waiting.as_mut());
            assert!(poll.is_pending());
        }
        assert_eq!(ring.waiter_count(), 0);

        // Wakeups still work after churn.
        shared.publish(DesiredQuotes {
            quote_seq: 5,
            ..DesiredQuotes::default()
        });
        assert_eq!(shared.changed_after(0).await.quote_seq, 5);
        assert!(ring.try_push(9).is_ok());
        assert_eq!(ring.pop().await, 9);
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
