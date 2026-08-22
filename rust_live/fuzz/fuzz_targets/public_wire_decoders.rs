#![no_main]

use hyperliquid_connector::instrument::InstrumentSpec;
use hyperliquid_connector::wire::{parse_bbo, parse_book, parse_trades};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };
    let instrument = InstrumentSpec {
        symbol: "CASHCAT".to_owned(),
        dex: String::new(),
        asset_id: 0,
        sz_decimals: 0,
        max_price_decimals: 6,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 10.0,
        margin_table_id: 0,
        only_isolated: false,
        margin_mode: String::new(),
        is_delisted: false,
        metadata_fingerprint: String::new(),
    };
    let _ = parse_bbo(&value, &instrument, 1);
    let _ = parse_trades(&value, &instrument, 1);
    let _ = parse_book(&value, &instrument, 1);
});
