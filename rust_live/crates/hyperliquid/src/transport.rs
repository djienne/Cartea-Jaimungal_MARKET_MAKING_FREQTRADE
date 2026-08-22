use serde_json::{json, Value};

pub const ACCOUNT_SUBSCRIPTION_COUNT: usize = 8;

pub fn account_subscriptions(account: &str, dex: &str, symbol: &str) -> Vec<Value> {
    vec![
        json!({"type": "orderUpdates", "user": account}),
        json!({"type": "userFills", "user": account, "aggregateByTime": false}),
        json!({"type": "userFundings", "user": account}),
        json!({"type": "clearinghouseState", "user": account, "dex": dex}),
        json!({"type": "openOrders", "user": account, "dex": dex}),
        json!({"type": "activeAssetData", "user": account, "coin": symbol}),
        json!({"type": "userNonFundingLedgerUpdates", "user": account}),
        json!({"type": "notification", "user": account}),
    ]
}

pub fn subscription_request(subscription: &Value) -> String {
    json!({"method": "subscribe", "subscription": subscription}).to_string()
}

pub fn application_ping() -> String {
    json!({"method": "ping"}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn account_subscription_set_is_complete_and_stable() {
        let subscriptions = account_subscriptions("0xabc", "dex", "CASHCAT");
        assert_eq!(subscriptions.len(), ACCOUNT_SUBSCRIPTION_COUNT);
        assert_eq!(subscriptions[0]["type"], "orderUpdates");
        assert_eq!(subscriptions[5]["coin"], "CASHCAT");
        assert_eq!(application_ping(), r#"{"method":"ping"}"#);
    }
}
