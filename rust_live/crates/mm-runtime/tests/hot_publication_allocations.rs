use mm_runtime::hot_path::AtomicRiskState;
use mm_runtime::lockfree::{AtomicBbo, HotPathSignal, SharedQuotes, HOT_SIGNAL_MARKET};
use mm_runtime::quote::RiskState;
use mm_runtime::types::{Bbo, DesiredQuotes, QuoteReason};

#[test]
fn hot_publication_primitives_are_allocation_free_after_construction() {
    let bbo = AtomicBbo::default();
    let quotes = SharedQuotes::default();
    let risk = AtomicRiskState::default();
    let signal = HotPathSignal::default();
    let allocations = allocation_counter::measure(|| {
        for sequence in 1..=1_000_000_u64 {
            bbo.store(Bbo {
                bid_px: sequence as i64,
                bid_sz: 10,
                ask_px: sequence as i64 + 1,
                ask_sz: 20,
                exchange_ms: sequence,
                recv_ns: sequence * 100,
            });
            std::hint::black_box(bbo.load());
            risk.store(RiskState {
                equity_usdc: 1_000.0,
                daily_realized_pnl_usdc: 0.0,
                consecutive_losses: 0,
            });
            std::hint::black_box(risk.load());
            quotes.publish(DesiredQuotes::empty(
                QuoteReason::Market,
                sequence,
                sequence,
            ));
            std::hint::black_box(quotes.load());
            signal.notify(HOT_SIGNAL_MARKET);
            std::hint::black_box(signal.take_pending());
        }
    });
    assert_eq!(allocations.count_total, 0, "{allocations:?}");
    assert_eq!(allocations.bytes_total, 0, "{allocations:?}");
}
