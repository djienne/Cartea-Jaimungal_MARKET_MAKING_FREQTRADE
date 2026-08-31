use arc_swap::ArcSwapOption;
use mm_runtime::config::{FlowGuardConfig, LatencyConfig, ModelConfig, QuotingConfig, RiskConfig};
use mm_runtime::hjb::{solve_asymmetric, CjParameters};
use mm_runtime::hot_path::{flow_channel, risk_channel, HotPathEngine, HotPathInputs, ModelBundle};
use mm_runtime::instrument::InstrumentSpec;
use mm_runtime::latency::LatencyMonitor;
use mm_runtime::lockfree::{bbo_channel, quote_channel, HotPathSignal, HOT_SIGNAL_MARKET};
use mm_runtime::metrics::Metrics;
use mm_runtime::quote::RiskState;
use mm_runtime::types::{Bbo, DesiredQuotes, ProcessClock, QuoteReason};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

#[test]
fn hot_publication_primitives_are_allocation_free_after_construction() {
    let (bbo_writer, bbo) = bbo_channel();
    let (quote_writer, quotes) = quote_channel();
    let (risk_writer, risk) = risk_channel();
    let signal = HotPathSignal::default();
    let allocations = allocation_counter::measure(|| {
        for sequence in 1..=1_000_000_u64 {
            bbo_writer.store(Bbo {
                bid_px: sequence as i64,
                bid_sz: 10,
                ask_px: sequence as i64 + 1,
                ask_sz: 20,
                exchange_ms: sequence,
                recv_ns: sequence * 100,
            });
            std::hint::black_box(bbo.load());
            risk_writer.store(RiskState {
                equity_usdc: 1_000.0,
                daily_realized_pnl_usdc: 0.0,
                consecutive_losses: 0,
            });
            std::hint::black_box(risk.load());
            quote_writer.publish(DesiredQuotes::empty(
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

#[test]
fn complete_production_hot_step_is_allocation_free_after_construction() {
    let model_config = ModelConfig::default();
    let surface = solve_asymmetric(
        CjParameters {
            lambda_plus: 0.117,
            lambda_minus: 0.103,
            kappa_plus: 10_538.0,
            kappa_minus: 9_161.0,
            epsilon_plus: 2.38e-5,
            epsilon_minus: 3.42e-5,
            sigma2_per_second: Some(3.8e-9),
        },
        &model_config,
        2_430.0,
        1,
    )
    .unwrap();
    let instrument = InstrumentSpec {
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
    };
    let (bbo_writer, bbo_reader) = bbo_channel();
    bbo_writer.store(Bbo {
        bid_px: 100_000,
        bid_sz: 10_000,
        ask_px: 100_010,
        ask_sz: 10_000,
        exchange_ms: 1,
        recv_ns: 1,
    });
    let (quote_writer, _quote_reader) = quote_channel();
    let (risk_writer, risk_reader) = risk_channel();
    risk_writer.store(RiskState {
        equity_usdc: 1_000.0,
        ..RiskState::default()
    });
    let (_flow_writer, flow_reader) = flow_channel();
    let inventory = Arc::new(AtomicI64::new(0));
    let latency_config = LatencyConfig {
        gate_enabled: false,
        hot_sample_every: 64,
        queue_capacity: 65_536,
        ..LatencyConfig::default()
    };
    let inputs = HotPathInputs {
        latest_bbo: bbo_reader,
        signal: Arc::new(HotPathSignal::default()),
        desired: quote_writer,
        model: Arc::new(ArcSwapOption::from(Some(Arc::new(ModelBundle {
            surface,
            inventory_unit: 2_430,
            generated_at_ms: 1,
            valid_until_ns: u64::MAX,
        })))),
        instrument,
        quoting: QuotingConfig::default(),
        risk: RiskConfig::default(),
        model_config,
        inventory_units: inventory.clone(),
        risk_state: risk_reader,
        flow_state: flow_reader,
        flow_guard: FlowGuardConfig {
            enabled: false,
            ..FlowGuardConfig::default()
        },
        scientifically_valid: Arc::new(AtomicBool::new(true)),
        market_stale_ms: u64::MAX,
        clock: Arc::new(ProcessClock::default()),
        metrics: Arc::new(Metrics::default()),
        latency: Arc::new(LatencyMonitor::new("SYN", 1, &latency_config, false)),
        latency_sample_every: 64,
        hot_path_cpu: None,
    };
    let mut engine = HotPathEngine::new(&inputs).unwrap();
    engine.step(&inputs, HOT_SIGNAL_MARKET, 2);
    let allocations = allocation_counter::measure(|| {
        for sequence in 1..=1_000_000_u64 {
            inventory.store((sequence as i64 % 11 - 5) * 2_430, Ordering::Relaxed);
            engine.step(&inputs, HOT_SIGNAL_MARKET, sequence.saturating_add(2));
        }
    });
    assert_eq!(allocations.count_total, 0, "{allocations:?}");
    assert_eq!(allocations.bytes_total, 0, "{allocations:?}");
}
