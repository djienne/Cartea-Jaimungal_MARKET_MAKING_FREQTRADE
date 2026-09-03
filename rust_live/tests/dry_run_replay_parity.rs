//! The public dry run and the deterministic replay must agree on simulated
//! fills, which requires one time base: exchange time. Before this suite the
//! dry run scheduled order activation and cancellation from the local wall
//! clock while matching trades on exchange time, so scheduler jitter and I/O
//! stalls changed simulated fills — replay was immune, and the two disagreed.

use mm_live::config::{DryRunConfig, QuotingConfig, RiskConfig};
use mm_live::execution::{AccountStateProvider, DryRunBackend, ExecutionBackend};
use mm_live::instrument::InstrumentSpec;
use mm_live::types::{
    AggressorSide, Bbo, DesiredQuotes, ExecutionEvent, MarketEvent, OrderIntent, QuoteReason, Side,
    TradePrint,
};

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

fn backend() -> DryRunBackend {
    DryRunBackend::new(
        instrument(),
        DryRunConfig {
            starting_equity_usdc: 1_000.0,
            decision_latency_ms: 100,
            acknowledgement_latency_ms: 100,
            cancel_latency_ms: 100,
            tail_latency_multiplier: 2.35,
            tail_latency_every: 20,
            queue_decay_per_second: 0.0,
            promotion_flatten_fee_rate: 0.00035,
            promotion_flatten_slippage_bps: 25.0,
            funding_rate_per_hour: 0.0,
            // Zero: this fixture pins Rust/Python replay parity, and the
            // flatten policy exists only on the dry-run side. Turning it on
            // here would compare two different strategies.
            flatten_after_ms: 0,
            flatten_slippage_bps: 2.5,
            flatten_fee_rate: 0.00045,
            markout_horizons_ms: vec![],
        },
        QuotingConfig::default(),
        RiskConfig::default(),
    )
    .unwrap()
}

fn bbo(exchange_ms: u64) -> Bbo {
    Bbo {
        bid_px: 100_000,
        bid_sz: 1,
        ask_px: 102_000,
        ask_sz: 1,
        exchange_ms,
        recv_ns: exchange_ms * 1_000_000,
    }
}

/// The decision the hot path would publish from the t=1000 BBO: join the best
/// bid for 20 units, stamped with the source BBO's exchange time.
fn decision(source: Bbo) -> DesiredQuotes {
    DesiredQuotes {
        bid: Some(OrderIntent {
            side: Side::Buy,
            px: source.bid_px,
            qty_units: 20,
            post_only: true,
            reduce_only: false,
        }),
        ask: None,
        quote_seq: 1,
        model_revision: 1,
        generated_ns: source.recv_ns,
        source_recv_ns: source.recv_ns,
        source_exchange_ms: source.exchange_ms,
        reason: QuoteReason::Market,
        mid: 0.101,
        q_exact: 0.0,
        q_rounded: 0,
        tau_remaining: 150.0,
    }
}

/// Feed the fixture stream: quote from the first BBO, activation on the second,
/// then an aggressor-sell print that should consume the 1-unit queue ahead and
/// fill 4 of our 20 units at the bid.
async fn run_fixture(reconcile_now_ms: u64) -> (Vec<ExecutionEvent>, f64) {
    let mut backend = backend();
    let first = bbo(1_000);
    let mut fills = Vec::new();
    fills.extend(
        backend
            .on_market_event(&MarketEvent::Bbo(first))
            .await
            .unwrap(),
    );
    backend
        .reconcile(decision(first), reconcile_now_ms)
        .await
        .unwrap();
    fills.extend(
        backend
            .on_market_event(&MarketEvent::Bbo(bbo(1_500)))
            .await
            .unwrap(),
    );
    fills.extend(
        backend
            .on_market_event(&MarketEvent::Trade(TradePrint {
                aggressor: AggressorSide::Sell,
                px: 100_000,
                qty_units: 5,
                exchange_ms: 2_000,
                recv_ns: 2_000_000_000,
                trade_id: 7,
            }))
            .await
            .unwrap(),
    );
    fills.extend(
        backend
            .on_market_event(&MarketEvent::Bbo(bbo(3_000)))
            .await
            .unwrap(),
    );
    let equity = backend.account_state().equity_usdc;
    (fills, equity)
}

fn fill_events(events: &[ExecutionEvent]) -> Vec<(i64, i64, u64, bool)> {
    events
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::Fill(fill) => {
                Some((fill.px, fill.qty_units, fill.exchange_ms, fill.maker))
            }
            _ => None,
        })
        .collect()
}

/// Hand-check the exchange-time fixture directly. This deliberately does not
/// claim top-level replay/public-path parity: both modes share this backend,
/// and running the same helper twice would be tautological evidence.
#[tokio::test]
async fn exchange_time_fixture_produces_the_expected_fill() {
    let result = run_fixture(1_000).await;
    let expected = vec![(100_000, 4, 2_000, true)];
    assert_eq!(fill_events(&result.0), expected);
    assert!(result.1.is_finite());
}

/// The old dry-run bug in one assertion: reconciling on a clock that runs
/// ahead of exchange time (as `unix_ms()` does whenever the host clock skews
/// or the loop stalls) delays simulated activation past the trade and the
/// fill vanishes. This is exactly the distortion that contaminated recorded
/// dry-run evidence.
#[tokio::test]
async fn wall_clock_skew_changes_simulated_fills() {
    let exchange_time = run_fixture(1_000).await;
    let skewed_clock = run_fixture(1_000 + 5_000).await;
    assert_eq!(fill_events(&exchange_time.0).len(), 1);
    assert!(
        fill_events(&skewed_clock.0).is_empty(),
        "a skewed reconcile clock must visibly distort fills; if this ever \
         fills, the simulator's activation scheduling changed"
    );
}

/// The hot path stamps every decision with the exchange time of the BBO it
/// quoted from; the dry-run loop depends on that field being populated.
#[test]
fn policy_decisions_carry_the_source_bbo_exchange_time() {
    use mm_live::hjb::{solve_asymmetric, CjParameters};
    use mm_live::quote::{CarteaJaimungalPolicy, RiskState};

    let surface = solve_asymmetric(
        CjParameters {
            lambda_plus: 0.12,
            lambda_minus: 0.11,
            kappa_plus: 10_000.0,
            kappa_minus: 9_000.0,
            epsilon_plus: 2.0e-5,
            epsilon_minus: 3.0e-5,
            sigma2_per_second: Some(3.0e-9),
        },
        &mm_live::config::ModelConfig::default(),
        2_000.0,
        1,
    )
    .unwrap();
    let policy = CarteaJaimungalPolicy::new(
        instrument(),
        QuotingConfig::default(),
        RiskConfig::default(),
    )
    .unwrap();
    let source = Bbo {
        exchange_ms: 777,
        ..bbo(1_000)
    };
    let decision = policy.compute(
        &surface,
        source,
        0,
        2_000,
        150.0,
        1,
        1,
        QuoteReason::Market,
        RiskState {
            equity_usdc: 1_000.0,
            ..RiskState::default()
        },
    );
    assert_eq!(decision.quotes.source_exchange_ms, 777);
}
