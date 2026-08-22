use cj_core::config::{ModelConfig, QuotingConfig, RiskConfig};
use cj_core::hjb::{solve_asymmetric, CjParameters};
use cj_core::instrument::InstrumentSpec;
use cj_core::quote::{CarteaJaimungalPolicy, RiskState};
use cj_core::types::{Bbo, QuoteReason};

fn instrument() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SYN".to_owned(),
        dex: String::new(),
        asset_id: 0,
        sz_decimals: 0,
        max_price_decimals: 6,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 1.0,
        margin_table_id: 0,
        only_isolated: false,
        margin_mode: String::new(),
        is_delisted: false,
        metadata_fingerprint: String::new(),
    }
}

fn parameters() -> CjParameters {
    CjParameters {
        lambda_plus: 0.117,
        lambda_minus: 0.103,
        kappa_plus: 10_538.0,
        kappa_minus: 9_161.0,
        epsilon_plus: 2.38e-5,
        epsilon_minus: 3.42e-5,
        sigma2_per_second: Some(3.8e-9),
    }
}

#[test]
fn prepared_surface_and_policy_compute_are_allocation_free() {
    let model = ModelConfig::default();
    let surface = solve_asymmetric(parameters(), &model, 2_430.0, 1).unwrap();
    let policy = CarteaJaimungalPolicy::new(
        instrument(),
        QuotingConfig::default(),
        RiskConfig::default(),
    )
    .unwrap();
    let bbo = Bbo {
        bid_px: 100_000,
        bid_sz: 10_000,
        ask_px: 100_010,
        ask_sz: 10_000,
        exchange_ms: 1,
        recv_ns: 1,
    };
    let risk = RiskState {
        equity_usdc: 1_000.0,
        daily_realized_pnl_usdc: 0.0,
        consecutive_losses: 0,
    };

    // Warm all code paths before enabling thread-local allocation counting.
    std::hint::black_box(policy.compute(
        &surface,
        bbo,
        0,
        2_430,
        75.0,
        0,
        1,
        QuoteReason::Market,
        risk,
    ));

    let allocations = allocation_counter::measure(|| {
        for sequence in 1..=1_000_000 {
            let q = (sequence % 11) as i64 - 5;
            let decision = policy.compute(
                &surface,
                bbo,
                q * 2_430 + 1_215,
                2_430,
                75.25,
                sequence,
                sequence,
                QuoteReason::Market,
                risk,
            );
            std::hint::black_box(decision);
        }
    });

    assert_eq!(allocations.count_total, 0, "{allocations:?}");
    assert_eq!(allocations.count_current, 0, "{allocations:?}");
    assert_eq!(allocations.bytes_total, 0, "{allocations:?}");
}

#[test]
fn depth_pair_is_allocation_free_at_boundaries_and_fractional_inventory() {
    let surface = solve_asymmetric(parameters(), &ModelConfig::default(), 2_430.0, 1).unwrap();
    let allocations = allocation_counter::measure(|| {
        for index in 0..1_000_000_u64 {
            let q = -6.0 + (index % 25) as f64 * 0.5;
            std::hint::black_box(surface.depth_pair(q, (index % 151) as f64));
        }
    });
    assert_eq!(allocations.count_total, 0, "{allocations:?}");
    assert_eq!(allocations.bytes_total, 0, "{allocations:?}");
}
