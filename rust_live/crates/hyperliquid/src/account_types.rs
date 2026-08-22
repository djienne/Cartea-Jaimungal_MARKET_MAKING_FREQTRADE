use super::exchange::{ClearinghouseState, OpenOrder, UserFill};
use super::signing::parse_fixed;
use crate::instrument::InstrumentSpec;
use crate::types::{AccountState, Side};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderUpdate {
    pub order: OrderUpdateOrder,
    pub status: String,
    pub status_timestamp: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderUpdateOrder {
    pub coin: String,
    pub side: String,
    pub limit_px: String,
    pub sz: String,
    pub oid: u64,
    pub timestamp: u64,
    #[serde(default)]
    pub reduce_only: bool,
    #[serde(default)]
    pub orig_sz: String,
    #[serde(default)]
    pub tif: String,
    #[serde(default)]
    pub cloid: Option<String>,
}

impl OrderUpdateOrder {
    pub fn parsed_side(&self) -> Result<Side> {
        match self.side.as_str() {
            "B" | "Buy" => Ok(Side::Buy),
            "A" | "Sell" => Ok(Side::Sell),
            value => bail!("unknown Hyperliquid order side {value:?}"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UserFillsMessage {
    #[serde(default)]
    pub is_snapshot: bool,
    #[serde(default)]
    pub user: String,
    #[serde(default)]
    pub fills: Vec<UserFill>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FundingPayment {
    pub time: u64,
    pub coin: String,
    pub usdc: String,
    pub szi: String,
    pub funding_rate: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UserFundingsMessage {
    #[serde(default)]
    pub is_snapshot: bool,
    #[serde(default)]
    pub user: String,
    #[serde(default)]
    pub fundings: Vec<FundingPayment>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LiveAccountSnapshot {
    pub account_value_usdc: f64,
    pub withdrawable_usdc: f64,
    pub total_margin_used_usdc: f64,
    pub total_notional_position_usdc: f64,
    pub inventory_units: i64,
    pub entry_px: f64,
    pub liquidation_px: Option<f64>,
    pub margin_used_usdc: f64,
    pub available_to_trade_usdc: [f64; 2],
    pub max_trade_units: [i64; 2],
    pub leverage: u32,
    pub isolated: bool,
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
    pub open_orders: Vec<OpenOrder>,
    pub realized_pnl_usdc: f64,
    pub fees_usdc: f64,
    pub funding_usdc: f64,
}

impl LiveAccountSnapshot {
    pub fn from_rest(
        clearinghouse: &ClearinghouseState,
        active_asset_data: &serde_json::Value,
        user_fees: &serde_json::Value,
        open_orders: Vec<OpenOrder>,
        instrument: &InstrumentSpec,
    ) -> Result<Self> {
        let position = clearinghouse
            .asset_positions
            .iter()
            .find(|position| position.position.coin == instrument.symbol)
            .map(|position| &position.position);
        let inventory_units = position.map_or(Ok(0), |position| {
            parse_fixed(&position.szi, instrument.sz_decimals)
        })?;
        let pair_values = |name: &str| -> Result<&Vec<serde_json::Value>> {
            let values = active_asset_data
                .get(name)
                .and_then(serde_json::Value::as_array)
                .with_context(|| format!("activeAssetData missing {name}"))?;
            if values.len() != 2 {
                bail!("activeAssetData {name} must contain buy/sell values");
            }
            Ok(values)
        };
        let max_values = pair_values("maxTradeSzs")?;
        let max_trade_units = [
            parse_fixed(
                max_values[0]
                    .as_str()
                    .context("active asset size is not text")?,
                instrument.sz_decimals,
            )?,
            parse_fixed(
                max_values[1]
                    .as_str()
                    .context("active asset size is not text")?,
                instrument.sz_decimals,
            )?,
        ];
        let available_values = pair_values("availableToTrade")?;
        let available_to_trade_usdc = [
            parse_number(
                available_values[0]
                    .as_str()
                    .context("available-to-trade value is not text")?,
            )?,
            parse_number(
                available_values[1]
                    .as_str()
                    .context("available-to-trade value is not text")?,
            )?,
        ];
        let leverage = active_asset_data
            .pointer("/leverage/value")
            .and_then(serde_json::Value::as_u64)
            .context("activeAssetData missing leverage")? as u32;
        let isolated = active_asset_data
            .pointer("/leverage/type")
            .and_then(serde_json::Value::as_str)
            == Some("isolated");
        Ok(Self {
            account_value_usdc: parse_number(&clearinghouse.margin_summary.account_value)?,
            withdrawable_usdc: parse_number(&clearinghouse.withdrawable)?,
            total_margin_used_usdc: parse_number(&clearinghouse.margin_summary.total_margin_used)?,
            total_notional_position_usdc: parse_number(
                &clearinghouse.margin_summary.total_ntl_pos,
            )?,
            inventory_units,
            entry_px: position
                .and_then(|position| position.entry_px.as_deref())
                .map_or(Ok(0.0), parse_number)?,
            liquidation_px: position
                .and_then(|position| position.liquidation_px.as_deref())
                .map(parse_number)
                .transpose()?,
            margin_used_usdc: position
                .map_or(Ok(0.0), |position| parse_number(&position.margin_used))?,
            available_to_trade_usdc,
            max_trade_units,
            leverage,
            isolated,
            maker_fee_rate: user_fees
                .get("userAddRate")
                .and_then(serde_json::Value::as_str)
                .context("userFees missing userAddRate")?
                .parse()?,
            taker_fee_rate: user_fees
                .get("userCrossRate")
                .and_then(serde_json::Value::as_str)
                .context("userFees missing userCrossRate")?
                .parse()?,
            open_orders,
            ..Self::default()
        })
    }

    pub fn account_state(&self, mark_px: f64, size_scale: i64) -> AccountState {
        let position_base = self.inventory_units as f64 / size_scale as f64;
        let mark_to_market = if self.entry_px > 0.0 {
            (mark_px - self.entry_px) * position_base
        } else {
            0.0
        };
        AccountState {
            cash_usdc: self.withdrawable_usdc,
            inventory_units: self.inventory_units,
            average_entry_px: self.entry_px,
            realized_pnl_usdc: self.realized_pnl_usdc,
            fees_usdc: self.fees_usdc,
            funding_usdc: self.funding_usdc,
            mark_to_market_pnl_usdc: mark_to_market,
            equity_usdc: self.account_value_usdc,
            position_notional_usdc: self.total_notional_position_usdc,
            margin_used_usdc: self.total_margin_used_usdc,
            maintenance_margin_usdc: 0.0,
            liquidation_buffer_usdc: self.withdrawable_usdc,
            // The venue does not report a loss streak. The risk-bearing value is
            // tracked durably and read via `HyperliquidLiveBackend::risk_scalars`;
            // do not wire this field into `RiskState`.
            consecutive_losses: 0,
        }
    }

    pub fn apply_clearinghouse(
        &mut self,
        clearinghouse: &ClearinghouseState,
        instrument: &InstrumentSpec,
    ) -> Result<()> {
        let position = clearinghouse
            .asset_positions
            .iter()
            .find(|position| position.position.coin == instrument.symbol)
            .map(|position| &position.position);
        self.account_value_usdc = parse_number(&clearinghouse.margin_summary.account_value)?;
        self.withdrawable_usdc = parse_number(&clearinghouse.withdrawable)?;
        self.total_margin_used_usdc =
            parse_number(&clearinghouse.margin_summary.total_margin_used)?;
        self.total_notional_position_usdc =
            parse_number(&clearinghouse.margin_summary.total_ntl_pos)?;
        self.inventory_units = position.map_or(Ok(0), |position| {
            parse_fixed(&position.szi, instrument.sz_decimals)
        })?;
        self.entry_px = position
            .and_then(|position| position.entry_px.as_deref())
            .map_or(Ok(0.0), parse_number)?;
        self.liquidation_px = position
            .and_then(|position| position.liquidation_px.as_deref())
            .map(parse_number)
            .transpose()?;
        self.margin_used_usdc =
            position.map_or(Ok(0.0), |position| parse_number(&position.margin_used))?;
        Ok(())
    }

    pub fn apply_active_asset_data(
        &mut self,
        active: &serde_json::Value,
        instrument: &InstrumentSpec,
    ) -> Result<()> {
        let pair = |name: &str| -> Result<&Vec<serde_json::Value>> {
            let values = active
                .get(name)
                .and_then(serde_json::Value::as_array)
                .with_context(|| format!("activeAssetData missing {name}"))?;
            if values.len() != 2 {
                bail!("activeAssetData {name} must have two sides");
            }
            Ok(values)
        };
        let available = pair("availableToTrade")?;
        self.available_to_trade_usdc = [
            parse_number(available[0].as_str().context("availableToTrade buy")?)?,
            parse_number(available[1].as_str().context("availableToTrade sell")?)?,
        ];
        let maximum = pair("maxTradeSzs")?;
        self.max_trade_units = [
            parse_fixed(
                maximum[0].as_str().context("maxTradeSzs buy")?,
                instrument.sz_decimals,
            )?,
            parse_fixed(
                maximum[1].as_str().context("maxTradeSzs sell")?,
                instrument.sz_decimals,
            )?,
        ];
        self.leverage = active
            .pointer("/leverage/value")
            .and_then(serde_json::Value::as_u64)
            .context("activeAssetData missing leverage")? as u32;
        self.isolated = active
            .pointer("/leverage/type")
            .and_then(serde_json::Value::as_str)
            == Some("isolated");
        Ok(())
    }
}

pub fn parse_order_updates(data: serde_json::Value) -> Result<Vec<OrderUpdate>> {
    serde_json::from_value(data).context("cannot decode orderUpdates")
}

pub fn parse_user_fills(data: serde_json::Value) -> Result<UserFillsMessage> {
    serde_json::from_value(data).context("cannot decode userFills")
}

pub fn parse_user_fundings(data: serde_json::Value) -> Result<UserFundingsMessage> {
    serde_json::from_value(data).context("cannot decode userFundings")
}

pub fn parse_clearinghouse_message(data: serde_json::Value) -> Result<ClearinghouseState> {
    let value = data.get("clearinghouseState").cloned().unwrap_or(data);
    serde_json::from_value(value).context("cannot decode clearinghouseState stream")
}

pub fn parse_open_orders_message(data: serde_json::Value) -> Result<Vec<OpenOrder>> {
    let value = data.get("orders").cloned().unwrap_or(data);
    serde_json::from_value(value).context("cannot decode openOrders stream")
}

pub fn fill_key(fill: &UserFill) -> String {
    format!("{}:{}:{}", fill.coin, fill.tid, fill.hash)
}

pub fn funding_key(funding: &FundingPayment) -> String {
    format!(
        "{}:{}:{}:{}",
        funding.coin, funding.time, funding.usdc, funding.szi
    )
}

fn parse_number(value: &str) -> Result<f64> {
    let value: f64 = value.parse()?;
    if !value.is_finite() {
        bail!("Hyperliquid numeric field is non-finite");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperliquid::exchange::{AssetPosition, MarginSummary};

    #[test]
    fn order_fill_and_funding_messages_decode_exact_wire_shapes() {
        let orders = parse_order_updates(serde_json::json!([{
            "order": {
                "coin": "CASHCAT", "side": "B", "limitPx": "0.1", "sz": "5.0",
                "oid": 7, "timestamp": 8, "reduceOnly": false, "origSz": "10.0",
                "tif": "Alo", "cloid": "0x434a4d4d000000000000000000000001"
            },
            "status": "open", "statusTimestamp": 9
        }]))
        .unwrap();
        assert_eq!(orders[0].order.parsed_side().unwrap(), Side::Buy);
        let fills = parse_user_fills(serde_json::json!({
            "isSnapshot": false,
            "fills": [{
                "coin":"CASHCAT","px":"0.1","sz":"2.0","side":"B","time":1,
                "oid":7,"tid":9,"crossed":false,"fee":"0.001","closedPnl":"0",
                "hash":"0xabc"
            }]
        }))
        .unwrap();
        assert_eq!(fill_key(&fills.fills[0]), "CASHCAT:9:0xabc");
        let funding = parse_user_fundings(serde_json::json!({
            "isSnapshot": false,
            "fundings": [{"time":1,"coin":"CASHCAT","usdc":"-0.1","szi":"2.0","fundingRate":"0.0001"}]
        }))
        .unwrap();
        assert_eq!(funding_key(&funding.fundings[0]), "CASHCAT:1:-0.1:2.0");
    }

    #[test]
    fn cashcat_account_fixture_distinguishes_usdc_availability_from_base_size() {
        let clearinghouse = ClearinghouseState {
            margin_summary: MarginSummary {
                account_value: "299.782699".to_owned(),
                total_margin_used: "0.0".to_owned(),
                total_ntl_pos: "0.0".to_owned(),
            },
            asset_positions: Vec::<AssetPosition>::new(),
            withdrawable: "299.782699".to_owned(),
        };
        let active = serde_json::json!({
            "availableToTrade": ["299.782699", "299.782699"],
            "maxTradeSzs": ["7744.0", "7744.0"],
            "leverage": {"type": "isolated", "value": 2}
        });
        let fees = serde_json::json!({
            "userAddRate": "0.00015",
            "userCrossRate": "0.00045"
        });
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
            metadata_fingerprint: "meta".to_owned(),
        };
        let snapshot =
            LiveAccountSnapshot::from_rest(&clearinghouse, &active, &fees, Vec::new(), &instrument)
                .unwrap();
        assert_eq!(snapshot.available_to_trade_usdc, [299.782_699; 2]);
        assert_eq!(snapshot.max_trade_units, [7_744; 2]);
        assert_eq!(snapshot.leverage, 2);
    }

    #[test]
    fn account_snapshot_wrappers_decode() {
        let clearing = parse_clearinghouse_message(serde_json::json!({
            "user": "0x1",
            "clearinghouseState": {
                "marginSummary": {
                    "accountValue": "10.0",
                    "totalMarginUsed": "0.0",
                    "totalNtlPos": "0.0"
                },
                "assetPositions": [],
                "withdrawable": "10.0"
            }
        }))
        .unwrap();
        assert_eq!(clearing.margin_summary.account_value, "10.0");
        let orders = parse_open_orders_message(serde_json::json!({
            "user": "0x1",
            "orders": [{
                "coin":"CASHCAT","oid":7,"cloid":"0x434a4d4d000000000000000000000001",
                "side":"B","limitPx":"0.1","sz":"100","timestamp":1
            }]
        }))
        .unwrap();
        assert_eq!(orders[0].oid, 7);
    }
}
