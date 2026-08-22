use crate::instrument::InstrumentSpec;
use crate::types::{AggressorSide, Bbo, BookLevel, BookSnapshot, TradePrint};
use anyhow::{bail, Result};
use serde_json::Value;

pub fn parse_decimal_scaled(input: &str, decimals: u32) -> Result<i64> {
    let input = input.trim();
    if input.is_empty() {
        bail!("empty decimal");
    }
    let (negative, body) = match input.as_bytes()[0] {
        b'-' => (true, &input[1..]),
        b'+' => (false, &input[1..]),
        _ => (false, input),
    };
    if body.is_empty() || body.contains(['e', 'E']) {
        bail!("unsupported decimal {input}");
    }
    let mut parts = body.split('.');
    let whole = parts.next().unwrap_or("0");
    let fractional = parts.next().unwrap_or("");
    if parts.next().is_some()
        || !whole.bytes().all(|value| value.is_ascii_digit())
        || !fractional.bytes().all(|value| value.is_ascii_digit())
    {
        bail!("invalid decimal {input}");
    }
    let scale = 10_i128.pow(decimals);
    let whole_value = if whole.is_empty() {
        0
    } else {
        whole.parse::<i128>()?
    };
    let keep = decimals as usize;
    let mut fractional_value = 0_i128;
    for (index, byte) in fractional.bytes().take(keep).enumerate() {
        fractional_value += i128::from(byte - b'0') * 10_i128.pow((keep - index - 1) as u32);
    }
    let mut value = whole_value
        .checked_mul(scale)
        .and_then(|scaled| scaled.checked_add(fractional_value))
        .ok_or_else(|| anyhow::anyhow!("decimal overflow"))?;
    if fractional.len() > keep && fractional.as_bytes()[keep] >= b'5' {
        value = value.checked_add(1).context("decimal overflow")?;
    }
    if negative {
        value = -value;
    }
    Ok(i64::try_from(value)?)
}

pub fn parse_bbo(root: &Value, instrument: &InstrumentSpec, recv_ns: u64) -> Option<Bbo> {
    if root.get("channel")?.as_str()? != "bbo" {
        return None;
    }
    let data = root.get("data")?;
    if data.get("coin")?.as_str()? != instrument.symbol {
        return None;
    }
    let levels = data.get("bbo")?.as_array()?;
    let bid = levels.first()?;
    let ask = levels.get(1)?;
    if bid.is_null() || ask.is_null() {
        return None;
    }
    Some(Bbo {
        bid_px: parse_decimal_scaled(bid.get("px")?.as_str()?, instrument.max_price_decimals)
            .ok()?,
        bid_sz: parse_decimal_scaled(bid.get("sz")?.as_str()?, instrument.sz_decimals).ok()?,
        ask_px: parse_decimal_scaled(ask.get("px")?.as_str()?, instrument.max_price_decimals)
            .ok()?,
        ask_sz: parse_decimal_scaled(ask.get("sz")?.as_str()?, instrument.sz_decimals).ok()?,
        exchange_ms: u64_field(data, "time").unwrap_or_default(),
        recv_ns,
    })
}

pub fn parse_trades(root: &Value, instrument: &InstrumentSpec, recv_ns: u64) -> Vec<TradePrint> {
    if root.get("channel").and_then(Value::as_str) != Some("trades") {
        return Vec::new();
    }
    let Some(items) = root.get("data").and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter(|item| item.get("coin").and_then(Value::as_str) == Some(&instrument.symbol))
        .filter_map(|item| {
            let aggressor = match item.get("side")?.as_str()? {
                "B" => AggressorSide::Buy,
                "A" => AggressorSide::Sell,
                _ => return None,
            };
            Some(TradePrint {
                aggressor,
                px: parse_decimal_scaled(item.get("px")?.as_str()?, instrument.max_price_decimals)
                    .ok()?,
                qty_units: parse_decimal_scaled(item.get("sz")?.as_str()?, instrument.sz_decimals)
                    .ok()?,
                exchange_ms: u64_field(item, "time").unwrap_or_default(),
                recv_ns,
                trade_id: u64_field(item, "tid").unwrap_or_default(),
            })
        })
        .collect()
}

pub fn parse_book(root: &Value, instrument: &InstrumentSpec, recv_ns: u64) -> Option<BookSnapshot> {
    if root.get("channel")?.as_str()? != "l2Book" {
        return None;
    }
    let data = root.get("data")?;
    if data.get("coin")?.as_str()? != instrument.symbol {
        return None;
    }
    let sides = data.get("levels")?.as_array()?;
    let parse_levels = |value: &Value| -> Option<Vec<BookLevel>> {
        Some(
            value
                .as_array()?
                .iter()
                .take(20)
                .filter_map(|level| {
                    Some(BookLevel {
                        px: parse_decimal_scaled(
                            level.get("px")?.as_str()?,
                            instrument.max_price_decimals,
                        )
                        .ok()?,
                        qty_units: parse_decimal_scaled(
                            level.get("sz")?.as_str()?,
                            instrument.sz_decimals,
                        )
                        .ok()?,
                    })
                })
                .collect(),
        )
    };
    Some(BookSnapshot {
        bids: parse_levels(sides.first()?)?,
        asks: parse_levels(sides.get(1)?)?,
        exchange_ms: u64_field(data, "time").unwrap_or_default(),
        recv_ns,
    })
}

fn u64_field(value: &Value, key: &str) -> Option<u64> {
    value
        .get(key)?
        .as_u64()
        .or_else(|| value.get(key)?.as_str()?.parse().ok())
}

use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_parser_rounds_like_python_collector_precision() {
        assert_eq!(parse_decimal_scaled("0.131975", 5).unwrap(), 13_198);
        assert_eq!(parse_decimal_scaled("-2.5", 0).unwrap(), -3);
    }
}
