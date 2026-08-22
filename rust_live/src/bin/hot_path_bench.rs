use mm_live::config::{LatencyConfig, ModelConfig, QuotingConfig, RiskConfig};
use mm_live::hjb::{solve_asymmetric, CjParameters, HjbSurface};
use mm_live::instrument::InstrumentSpec;
use mm_live::latency::{HotLatencySampler, LatencyKind, LatencyMonitor};
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::types::{Bbo, ProcessClock, QuoteReason};
use serde::Serialize;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct Distribution {
    p50: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    build: mm_live::BuildInfo,
    pinned_cpu: Option<usize>,
    iterations_per_run: u64,
    repetitions: usize,
    quote_batch_ns_per_decision: Distribution,
    baseline_run_ns_per_decision: Distribution,
    monitored_run_ns_per_decision: Distribution,
    monitoring_overhead_percent: f64,
    hjb_solve_ms: Distribution,
    guard: i64,
}

fn main() {
    mm_live::BuildInfo::current()
        .ensure_optimized()
        .expect("benchmark must be compiled at opt-level=3");
    let pinned_cpu = std::env::var("MM_BENCH_CPU").ok().map(|value| {
        value
            .parse::<usize>()
            .expect("MM_BENCH_CPU must be an integer")
    });
    if let Some(cpu) = pinned_cpu {
        let cores = core_affinity::get_core_ids().expect("cannot enumerate benchmark CPUs");
        let core = cores
            .into_iter()
            .find(|core| core.id == cpu)
            .expect("MM_BENCH_CPU is not available");
        assert!(
            core_affinity::set_for_current(core),
            "cannot pin benchmark thread"
        );
    }
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
        margin_table_id: 0,
        only_isolated: false,
        margin_mode: String::new(),
        is_delisted: false,
        metadata_fingerprint: String::new(),
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
    let quote_batches = quote_batch_distribution(&policy, &surface, bbo, 20_000, 64);

    let solve_iterations = 100_u64;
    let mut solve_samples = Vec::with_capacity(solve_iterations as usize);
    let mut solve_guard = 0_usize;
    for revision in 0..solve_iterations {
        let started = Instant::now();
        let solved = solve_asymmetric(
            black_box(parameters),
            black_box(&model_config),
            black_box(1_868.0),
            revision,
        )
        .expect("benchmark HJB must solve");
        solve_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        solve_guard ^= solved.n_steps;
    }
    guard ^= solve_guard as i64;
    let report = BenchmarkReport {
        schema_version: 2,
        build: mm_live::BuildInfo::current(),
        pinned_cpu,
        iterations_per_run: iterations,
        repetitions,
        quote_batch_ns_per_decision: distribution(&quote_batches),
        baseline_run_ns_per_decision: distribution(&baseline),
        monitored_run_ns_per_decision: distribution(&monitored),
        monitoring_overhead_percent: overhead_percent,
        hjb_solve_ms: distribution(&solve_samples),
        guard,
    };
    let encoded = serde_json::to_string_pretty(&report).expect("benchmark report must serialize");
    println!("{encoded}");
    if let Ok(path) = std::env::var("MM_BENCH_OUTPUT") {
        std::fs::write(Path::new(&path), encoded).expect("cannot write MM_BENCH_OUTPUT");
    }
}

fn quote_batch_distribution(
    policy: &CarteaJaimungalPolicy,
    surface: &HjbSurface,
    bbo: Bbo,
    batches: u64,
    batch_size: u64,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(batches as usize);
    let mut sequence = 0_u64;
    for _ in 0..batches {
        let started = Instant::now();
        for _ in 0..batch_size {
            let q = (sequence as i64 % 11 - 5) * 1_868;
            let decision = policy.compute(
                black_box(surface),
                black_box(bbo),
                black_box(q),
                1_868,
                black_box(75.0),
                sequence,
                sequence,
                QuoteReason::Market,
                RiskState {
                    equity_usdc: 1_000.0,
                    ..RiskState::default()
                },
            );
            black_box(decision);
            sequence = sequence.wrapping_add(1);
        }
        samples.push(started.elapsed().as_nanos() as f64 / batch_size as f64);
    }
    samples
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

fn distribution(values: &[f64]) -> Distribution {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Distribution {
        p50: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
        min: sorted.first().copied().unwrap_or_default(),
        max: sorted.last().copied().unwrap_or_default(),
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}
