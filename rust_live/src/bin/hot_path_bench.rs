use mm_live::config::{LatencyConfig, ModelConfig, QuotingConfig, RiskConfig};
use mm_live::hjb::{solve_asymmetric, CjParameters, HjbSurface};
use mm_live::instrument::InstrumentSpec;
use mm_live::latency::{HotLatencySampler, LatencyKind, LatencyMonitor};
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::types::{Bbo, ProcessClock, QuoteReason};
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
    let iterations = 1_000_000_u64;
    let repetitions = 9;
    let sample_every = 16_u64;
    let _ = run_quote_loop(&policy, &surface, bbo, 100_000, false, sample_every);
    let _ = run_quote_loop(&policy, &surface, bbo, 100_000, true, sample_every);
    let mut baseline = Vec::with_capacity(repetitions);
    let mut monitored = Vec::with_capacity(repetitions);
    let mut guard = 0_i64;
    for repetition in 0..repetitions {
        if repetition % 2 == 0 {
            let (value, run_guard) =
                run_quote_loop(&policy, &surface, bbo, iterations, false, sample_every);
            baseline.push(value);
            guard ^= run_guard;
            let (value, run_guard) =
                run_quote_loop(&policy, &surface, bbo, iterations, true, sample_every);
            monitored.push(value);
            guard ^= run_guard;
        } else {
            let (value, run_guard) =
                run_quote_loop(&policy, &surface, bbo, iterations, true, sample_every);
            monitored.push(value);
            guard ^= run_guard;
            let (value, run_guard) =
                run_quote_loop(&policy, &surface, bbo, iterations, false, sample_every);
            baseline.push(value);
            guard ^= run_guard;
        }
    }
    let baseline_median = median(&baseline);
    let monitored_median = median(&monitored);
    let overhead_ns = monitored_median - baseline_median;
    let overhead_percent = overhead_ns / baseline_median * 100.0;
    println!(
        "benchmark=cj_quote_compute_paired iterations_per_run={iterations} repetitions={repetitions} baseline_median_ns={baseline_median:.2} baseline_min_ns={:.2} baseline_max_ns={:.2} guard={guard}",
        baseline.iter().copied().fold(f64::INFINITY, f64::min),
        baseline.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    println!(
        "benchmark=latency_monitor_producer iterations_per_run={iterations} repetitions={repetitions} hot_sample_every={sample_every} monitored_median_ns={monitored_median:.2} monitored_min_ns={:.2} monitored_max_ns={:.2} incremental_median_ns={overhead_ns:.2} incremental_median_percent={overhead_percent:.2}",
        monitored.iter().copied().fold(f64::INFINITY, f64::min),
        monitored.iter().copied().fold(f64::NEG_INFINITY, f64::max),
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

fn run_quote_loop(
    policy: &CarteaJaimungalPolicy,
    surface: &HjbSurface,
    bbo: Bbo,
    iterations: u64,
    monitored: bool,
    sample_every: u64,
) -> (f64, i64) {
    let clock = ProcessClock::default();
    let started = Instant::now();
    let mut guard = 0_i64;
    if monitored {
        let latency_config = LatencyConfig {
            gate_enabled: false,
            hot_sample_every: sample_every,
            queue_capacity: iterations.div_ceil(sample_every) as usize + 2,
            ..LatencyConfig::default()
        };
        let latency = LatencyMonitor::new("BENCH", 1, &latency_config, false);
        let sample_mask = sample_every - 1;
        let mut latency_counter = 0_u64;
        let mut latency_sampler = HotLatencySampler::default();
        for index in 0..iterations {
            let q = (index as i64 % 11 - 5) * 1_868;
            let decision = policy.compute(
                black_box(surface),
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
            latency_counter = latency_counter.wrapping_add(1);
            let sample_hot = latency_counter & sample_mask == 0;
            if !sample_hot {
                continue;
            }
            let observed_ns = clock.now_ns();
            latency_sampler.record(
                &latency,
                LatencyKind::HotDecision,
                observed_ns.saturating_sub(index),
                observed_ns,
            );
        }
        latency_sampler.flush(&latency);
    } else {
        for index in 0..iterations {
            let q = (index as i64 % 11 - 5) * 1_868;
            let decision = policy.compute(
                black_box(surface),
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
    }
    (
        started.elapsed().as_nanos() as f64 / iterations as f64,
        guard,
    )
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
