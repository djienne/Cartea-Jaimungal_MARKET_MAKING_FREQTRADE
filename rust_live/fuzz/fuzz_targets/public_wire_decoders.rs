#![no_main]

use hyperliquid_connector::instrument::InstrumentSpec;
use hyperliquid_connector::wire::parse_public_frame;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let instrument = InstrumentSpec {
        symbol: "CASHCAT".to_owned(),
        dex: String::new(),
        asset_id: 231,
        sz_decimals: 0,
        max_price_decimals: 6,
        max_significant_figures: 5,
        max_leverage: 3.0,
        minimum_notional: 10.0,
        margin_table_id: 3,
        only_isolated: true,
        margin_mode: "strictIsolated".to_owned(),
        is_delisted: false,
        metadata_fingerprint: String::new(),
    };
    // The decoder must never panic on arbitrary input; it exercises the real
    // feed path (envelope pass plus typed per-channel payload decode).
    let _ = parse_public_frame(text, &instrument, 1);
});
