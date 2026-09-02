//! Golden values are produced by the untouched Python reference modules:
//! `estimator_common.py`, `get_epsilon.py`, `hjb.py`, and `mm_core.py`.
//! The fixture is deterministic and contains no venue/network dependency.
//!
//! Regenerate the literals with `python scripts/parity_fixture.py` (repository
//! root); it rebuilds the same window in Python and prints every value pinned
//! here. `--unscaled` reproduces the schema-v4 goldens as a self-check.

use approx::assert_relative_eq;
use mm_live::calibration::{CalibrationStatus, Calibrator};
use mm_live::config::{CalibrationConfig, ModelConfig, QuotingConfig, RiskConfig};
use mm_live::hjb::{solve_asymmetric, CjParameters};
use mm_live::instrument::InstrumentSpec;
use mm_live::parquet_io::{MarketDataSet, MidRecord, ShardStats, TimeSource, TradeRecord};
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::types::{Bbo, QuoteReason, Side};

const RELATIVE_TOLERANCE: f64 = 1.0e-8;

fn deterministic_python_fixture() -> MarketDataSet {
    let mids: Vec<MidRecord> = (0..=2_000)
        .map(|index| {
            let ts_ms = index as f64 * 100.0;
            let mid = 0.132 + 0.000_05 * (index as f64 / 37.0).sin() + 0.000_000_02 * index as f64;
            MidRecord {
                ts_ms,
                bid: mid - 0.000_01,
                ask: mid + 0.000_01,
                mid,
            }
        })
        .collect();
    let trades = (1..400)
        .map(|index| {
            let ts_ms = index as f64 * 500.0 + 50.0;
            let pre_index = ((ts_ms - 1.0) / 100.0).floor() as usize;
            let pre_mid = mids[pre_index].mid;
            let buy = index % 2 == 0;
            let u = ((index * 73) % 397) as f64 / 398.0 + 0.5 / 398.0;
            let depth = -(1.0 - u).max(1.0e-12).ln() / 9_000.0;
            TradeRecord {
                ts_ms,
                side: if buy { "buy" } else { "sell" }.to_owned(),
                price: if buy {
                    pre_mid + depth
                } else {
                    pre_mid - depth
                },
                size: f64::from(10 + index % 7),
                trade_id: Some(index.to_string()),
            }
        })
        .collect();
    MarketDataSet {
        symbol: "ORACLE".to_owned(),
        time_source: TimeSource::Exchange,
        mids,
        trades,
        books: Vec::new(),
        window_start_ms: 0.0,
        window_end_ms: 200_000.0,
        duplicate_trade_ids_dropped: 0,
        price_shards: ShardStats::default(),
        trade_shards: ShardStats::default(),
        orderbook_shards: ShardStats::default(),
    }
}

#[test]
fn schema_v5_parameters_match_python_reference() {
    let mut config = CalibrationConfig::default();
    config.max_toxicity = 10.0;
    let snapshot = Calibrator::new("ORACLE", config)
        .calibrate(&deterministic_python_fixture())
        .unwrap();
    assert_eq!(snapshot.status, CalibrationStatus::Ok);
    let parameters = snapshot.parameters;
    assert_relative_eq!(
        parameters.kappa_plus,
        8_993.138_741_402_856,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        parameters.kappa_minus,
        9_068.665_345_705_7,
        max_relative = RELATIVE_TOLERANCE
    );
    // Schema v5: lambda is the raw rate times the survival-fit intercept.
    assert_relative_eq!(
        parameters.lambda_plus,
        0.998_544_300_340_291_4,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        parameters.lambda_minus,
        1.000_449_358_777_403_3,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        snapshot.diagnostics.plus.lambda_raw.unwrap(),
        0.995,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        snapshot.diagnostics.minus.lambda_raw.unwrap(),
        1.0,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        snapshot.diagnostics.plus.survival_intercept.unwrap(),
        1.003_562_110_894_765_2,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        snapshot.diagnostics.minus.survival_intercept.unwrap(),
        1.000_449_358_777_403_3,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        parameters.epsilon_plus,
        6.000_260_615_141_559e-9,
        epsilon = 1.0e-14
    );
    assert_relative_eq!(parameters.epsilon_minus, 0.0, epsilon = 1.0e-14);
    assert_relative_eq!(
        parameters.sigma2_per_second.unwrap(),
        9.154_778_008_016_623e-11,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_eq!(snapshot.diagnostics.plus.market_orders, 199);
    assert_eq!(snapshot.diagnostics.minus.market_orders, 200);
    assert_eq!(snapshot.diagnostics.plus.n_points, 197);
    assert_eq!(snapshot.diagnostics.minus.n_points, 198);
    assert_eq!(snapshot.diagnostics.plus.epsilon_events, 199);
    assert_eq!(snapshot.diagnostics.minus.epsilon_events, 200);
    assert_relative_eq!(
        snapshot.diagnostics.plus.r_squared.unwrap(),
        0.999_818_754_710_105_3,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        snapshot.diagnostics.minus.r_squared.unwrap(),
        0.999_831_159_214_848_2,
        max_relative = RELATIVE_TOLERANCE
    );
}

#[test]
fn hjb_surface_and_final_spreads_match_python_reference() {
    let parameters = CjParameters {
        kappa_plus: 8_993.138_741_402_856,
        kappa_minus: 9_068.665_345_705_7,
        lambda_plus: 0.998_544_300_340_291_4,
        lambda_minus: 1.000_449_358_777_403_3,
        epsilon_plus: 6.000_260_615_141_559e-9,
        epsilon_minus: 0.0,
        sigma2_per_second: Some(9.154_778_008_016_623e-11),
    };
    let surface = solve_asymmetric(parameters, &ModelConfig::default(), 1_868.0, 1).unwrap();
    assert_eq!(surface.n_steps, 600);
    assert_relative_eq!(surface.dt, 0.25, epsilon = 0.0);
    assert_relative_eq!(
        surface.phi_effective,
        0.000_147_649_763_688_759_99,
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        surface.alpha_effective,
        5.536_545_492_228_768e-6,
        max_relative = RELATIVE_TOLERANCE
    );
    let points = [
        (
            150.0,
            -5.0,
            -0.000_387_674_299_159_536_84,
            0.000_649_182_034_339_312_7,
        ),
        (
            150.0,
            0.0,
            0.000_265_568_793_491_248_55,
            0.000_265_839_169_909_092_8,
        ),
        (
            75.0,
            1.5,
            0.000_454_955_356_588_174_5,
            -0.000_116_719_437_232_775_29,
        ),
        (
            1.0,
            5.0,
            0.000_642_246_269_056_885_8,
            -0.000_377_529_416_505_007_93,
        ),
    ];
    for (tau, q, expected_bid, expected_ask) in points {
        assert_relative_eq!(
            surface.depth(Side::Buy, q, tau).unwrap(),
            expected_bid,
            // The fail-closed Newton solve converges the governing residual
            // before normalization; its cross-language depth delta is below
            // 1e-10 absolute and does not change a venue price unit.
            max_relative = 1.0e-7,
            epsilon = 1.0e-10
        );
        assert_relative_eq!(
            surface.depth(Side::Sell, q, tau).unwrap(),
            expected_ask,
            max_relative = 1.0e-7,
            epsilon = 1.0e-10
        );
    }

    let instrument = InstrumentSpec {
        symbol: "ORACLE".to_owned(),
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
    let policy =
        CarteaJaimungalPolicy::new(instrument, QuotingConfig::default(), RiskConfig::default())
            .unwrap();
    let bbo = Bbo {
        bid_px: 132_080,
        bid_sz: 10_000,
        ask_px: 132_090,
        ask_sz: 10_000,
        exchange_ms: 1,
        recv_ns: 1,
    };
    let flat = policy.compute(
        &surface,
        bbo,
        0,
        1_868,
        150.0,
        1,
        1,
        QuoteReason::Market,
        RiskState {
            equity_usdc: 1_000.0,
            ..RiskState::default()
        },
    );
    assert_eq!(flat.quotes.bid.unwrap().px, 131_790);
    assert_eq!(flat.quotes.ask.unwrap().px, 132_380);

    let fractional = policy.compute(
        &surface,
        bbo,
        2_802,
        1_868,
        75.0,
        2,
        2,
        QuoteReason::Market,
        RiskState {
            equity_usdc: 1_000.0,
            ..RiskState::default()
        },
    );
    assert_eq!(fractional.quotes.bid.unwrap().px, 131_610);
    assert_eq!(fractional.quotes.ask.unwrap().px, 132_110);
}
