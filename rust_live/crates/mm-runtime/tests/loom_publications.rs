use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

struct PairSlot {
    sequence: AtomicUsize,
    left: AtomicUsize,
    right: AtomicUsize,
}

#[test]
fn seqlock_never_publishes_a_torn_pair() {
    loom::model(|| {
        let slot = Arc::new(PairSlot {
            sequence: AtomicUsize::new(0),
            left: AtomicUsize::new(0),
            right: AtomicUsize::new(0),
        });
        let writer = {
            let slot = slot.clone();
            thread::spawn(move || {
                slot.sequence.fetch_add(1, Ordering::AcqRel);
                slot.left.store(1, Ordering::Release);
                slot.right.store(2, Ordering::Release);
                slot.sequence.fetch_add(1, Ordering::Release);
            })
        };
        let reader = {
            let slot = slot.clone();
            thread::spawn(move || loop {
                let before = slot.sequence.load(Ordering::Acquire);
                if before & 1 == 1 {
                    thread::yield_now();
                    continue;
                }
                let left = slot.left.load(Ordering::Acquire);
                let right = slot.right.load(Ordering::Acquire);
                let after = slot.sequence.load(Ordering::Acquire);
                if before == after && after & 1 == 0 {
                    assert!(
                        matches!((left, right), (0, 0) | (1, 2)),
                        "torn pair: before={before}, after={after}, left={left}, right={right}"
                    );
                    break;
                }
            })
        };
        writer.join().unwrap();
        reader.join().unwrap();
    });
}

#[test]
fn registration_and_notification_cannot_lose_pending_work() {
    loom::model(|| {
        let pending = Arc::new(AtomicUsize::new(0));
        let registered = Arc::new(AtomicBool::new(false));
        let wakeups = Arc::new(AtomicUsize::new(0));
        let producer = {
            let pending = pending.clone();
            let registered = registered.clone();
            let wakeups = wakeups.clone();
            thread::spawn(move || {
                let previous = pending.fetch_or(1, Ordering::Release);
                if previous == 0 && registered.load(Ordering::Acquire) {
                    wakeups.fetch_add(1, Ordering::Relaxed);
                }
            })
        };
        let consumer = {
            let pending = pending.clone();
            let registered = registered.clone();
            let wakeups = wakeups.clone();
            thread::spawn(move || {
                registered.store(true, Ordering::Release);
                if pending.load(Ordering::Acquire) != 0 {
                    wakeups.fetch_add(1, Ordering::Relaxed);
                }
                pending.swap(0, Ordering::AcqRel)
            })
        };
        producer.join().unwrap();
        let consumed = consumer.join().unwrap();
        let still_pending = pending.swap(0, Ordering::AcqRel);
        assert_eq!(consumed | still_pending, 1);
        if consumed == 0 && wakeups.load(Ordering::Relaxed) == 0 {
            assert_eq!(still_pending, 1);
        }
    });
}
