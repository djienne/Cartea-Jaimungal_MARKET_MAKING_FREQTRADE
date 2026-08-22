#![no_main]

use hyperliquid_connector::account_types::{
    parse_clearinghouse_message, parse_open_orders_message, parse_order_updates, parse_user_fills,
    parse_user_fundings,
};
use hyperliquid_connector::live_state::PersistedLiveState;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };
    let _ = parse_order_updates(value.clone());
    let _ = parse_user_fills(value.clone());
    let _ = parse_user_fundings(value.clone());
    let _ = parse_clearinghouse_message(value.clone());
    let _ = parse_open_orders_message(value);
    let _ = serde_json::from_slice::<PersistedLiveState>(data);
});
