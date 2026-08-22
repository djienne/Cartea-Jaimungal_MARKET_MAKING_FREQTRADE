use cj_core::instrument::InstrumentSpec;
use proptest::prelude::*;

fn instrument(price_decimals: u32, size_decimals: u32) -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SYN".to_owned(),
        dex: String::new(),
        asset_id: 0,
        sz_decimals: size_decimals,
        max_price_decimals: price_decimals,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 0.0,
        margin_table_id: 0,
        only_isolated: false,
        margin_mode: String::new(),
        is_delisted: false,
        metadata_fingerprint: String::new(),
    }
}

proptest! {
    #[test]
    fn unit_conversions_are_monotone_and_round_trip(
        price_decimals in 0_u32..=8,
        size_decimals in 0_u32..=8,
        price_units in 1_i64..1_000_000_000,
        size_units in 0_i64..1_000_000_000,
    ) {
        let instrument = instrument(price_decimals, size_decimals);
        let price = instrument.price_from_units(price_units);
        let size = instrument.size_from_units(size_units);
        prop_assert_eq!(instrument.price_to_units(price).unwrap(), price_units);
        prop_assert_eq!(instrument.size_to_units(size).unwrap(), size_units);
        let quantum = instrument.price_quantum(price_units);
        prop_assert!(quantum >= 1 && quantum <= instrument.price_scale());
        prop_assert_eq!(instrument.price_scale() % quantum, 0);
    }

    #[test]
    fn price_quantum_is_a_power_of_ten(
        price_decimals in 0_u32..=8,
        price_units in 1_i64..1_000_000_000,
    ) {
        let quantum = instrument(price_decimals, 0).price_quantum(price_units);
        prop_assert!(quantum > 0);
        let mut remaining = quantum;
        while remaining > 1 && remaining % 10 == 0 {
            remaining /= 10;
        }
        prop_assert_eq!(remaining, 1);
    }
}
