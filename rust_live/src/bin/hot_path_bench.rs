use mm_live::config::{ModelConfig, QuotingConfig, RiskConfig};
use mm_live::hjb::{solve_asymmetric, CjParameters};
use mm_live::instrument::InstrumentSpec;
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::types::{Bbo, QuoteReason};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let parameters = CjParameters {
        lambda_plus: 0.135,
        lambda_minus: 0.174,
        kappa_plus: 11_265.0,
        kappa_minus: 11_044.0,
        epsilon_plus: 3.26e-5,
        epsilon_minus: 1.92e-5,
        sigma2_per_second: Some(3.8e-9),
    };
    let model_config = ModelConfig::default();
    let instrument = InstrumentSpec {
        symbol: "BENCH".to_owned(),
        dex: String::new(),
        asset_id: 0,
        sz_decimals: 0,
        max_price_decimals: 6,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 1.0,
    };
    let surface =
        solve_asymmetric(parameters, &model_config, 1_868.0, 1).expect("benchmark HJB must solve");
    let policy =
        CarteaJaimungalPolicy::new(instrument, QuotingConfig::default(), RiskConfig::default())
            .expect("benchmark policy must validate");
    let bbo = Bbo {
        bid_px: 131_970,
        bid_sz: 2_000,
        ask_px: 132_200,
        ask_sz: 3_000,
        exchange_ms: 0,
        recv_ns: 0,
    };
    let iterations = 5_000_000_u64;
    let started = Instant::now();
    let mut guard = 0_i64;
    for index in 0..iterations {
        let q = (index as i64 % 11 - 5) * 1_868;
        let decision = policy.compute(
            black_box(&surface),
            black_box(bbo),
            black_box(q),
            1_868,
            black_box(75.0),
            index,
            index,
            QuoteReason::Market,
            RiskState {
                equity_usdc: 1_000.0,
                ..RiskState::default()
            },
        );
        guard ^= decision.quotes.bid.map_or(0, |order| order.px);
    }
    let total_ns = started.elapsed().as_nanos();
    println!(
        "benchmark=cj_quote_compute iterations={iterations} total_ns={total_ns} ns_per_iteration={:.2} guard={guard}",
        total_ns as f64 / iterations as f64
    );

    let solve_iterations = 100_u64;
    let started = Instant::now();
    let mut solve_guard = 0_usize;
    for revision in 0..solve_iterations {
        let solved = solve_asymmetric(
            black_box(parameters),
            black_box(&model_config),
            black_box(1_868.0),
            revision,
        )
        .expect("benchmark HJB must solve");
        solve_guard ^= solved.n_steps;
    }
    let total_ns = started.elapsed().as_nanos();
    println!(
        "benchmark=cj_hjb_solve iterations={solve_iterations} total_ns={total_ns} ms_per_solve={:.3} guard={solve_guard}",
        total_ns as f64 / solve_iterations as f64 / 1_000_000.0
    );
}
