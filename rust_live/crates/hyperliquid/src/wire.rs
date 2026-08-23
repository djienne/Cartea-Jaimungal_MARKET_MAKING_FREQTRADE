use crate::instrument::InstrumentSpec;
use crate::types::{AggressorSide, Bbo, BookLevel, BookSnapshot, TradePrint};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::borrow::Cow;

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

/// One decoded public WebSocket frame.
///
/// The feed previously built a full `serde_json::Value` DOM per frame and ran
/// three DOM-walking parsers over it. This decode is two-stage: a cheap
/// envelope pass borrows the channel name and the raw payload slice, then only
/// the matching channel's payload is deserialized — typed and borrowed, no
/// DOM. This sits upstream of the quote kernel on the same critical path.
#[derive(Debug, PartialEq, Eq)]
pub enum PublicFrame {
    Pong,
    Bbo(Bbo),
    /// A trades frame for this instrument's channel; individual prints from
    /// other coins or with malformed fields are dropped, matching the DOM
    /// parser's per-item tolerance.
    Trades(Vec<TradePrint>),
    Book(BookSnapshot),
    /// Valid JSON that is not a market payload for this instrument.
    Other,
    /// Not valid JSON at all.
    Invalid,
}

#[derive(Deserialize)]
struct Envelope<'a> {
    #[serde(default, borrow)]
    channel: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    data: Option<&'a serde_json::value::RawValue>,
}

/// Exchange timestamps arrive as integers but are tolerated as strings,
/// matching the old `u64_field` behavior.
#[derive(Deserialize)]
#[serde(untagged)]
enum WireU64<'a> {
    Int(u64),
    #[serde(borrow)]
    Text(Cow<'a, str>),
}

impl WireU64<'_> {
    fn value(&self) -> u64 {
        match self {
            Self::Int(value) => *value,
            Self::Text(text) => text.parse().unwrap_or_default(),
        }
    }
}

fn wire_u64(field: Option<&WireU64<'_>>) -> u64 {
    field.map(WireU64::value).unwrap_or_default()
}

#[derive(Deserialize)]
struct WireLevel<'a> {
    #[serde(borrow)]
    px: Cow<'a, str>,
    #[serde(borrow)]
    sz: Cow<'a, str>,
}

#[derive(Deserialize)]
struct WireBbo<'a> {
    #[serde(borrow)]
    coin: Cow<'a, str>,
    bbo: Vec<Option<WireLevel<'a>>>,
    #[serde(default, borrow)]
    time: Option<WireU64<'a>>,
}

#[derive(Deserialize)]
struct WireTrade<'a> {
    #[serde(borrow)]
    coin: Cow<'a, str>,
    #[serde(borrow)]
    px: Cow<'a, str>,
    #[serde(borrow)]
    sz: Cow<'a, str>,
    #[serde(borrow)]
    side: Cow<'a, str>,
    #[serde(default, borrow)]
    time: Option<WireU64<'a>>,
    #[serde(default, borrow)]
    tid: Option<WireU64<'a>>,
}

#[derive(Deserialize)]
struct WireBook<'a> {
    #[serde(borrow)]
    coin: Cow<'a, str>,
    #[serde(borrow)]
    levels: Vec<Vec<WireLevel<'a>>>,
    #[serde(default, borrow)]
    time: Option<WireU64<'a>>,
}

pub fn parse_public_frame(text: &str, instrument: &InstrumentSpec, recv_ns: u64) -> PublicFrame {
    let Ok(envelope) = serde_json::from_str::<Envelope<'_>>(text) else {
        return PublicFrame::Invalid;
    };
    let channel = envelope.channel.as_deref().unwrap_or_default();
    match channel {
        "pong" => PublicFrame::Pong,
        "bbo" => envelope
            .data
            .and_then(|data| decode_bbo(data.get(), instrument, recv_ns))
            .map_or(PublicFrame::Other, PublicFrame::Bbo),
        "trades" => PublicFrame::Trades(
            envelope
                .data
                .map(|data| decode_trades(data.get(), instrument, recv_ns))
                .unwrap_or_default(),
        ),
        "l2Book" => envelope
            .data
            .and_then(|data| decode_book(data.get(), instrument, recv_ns))
            .map_or(PublicFrame::Other, PublicFrame::Book),
        _ => PublicFrame::Other,
    }
}

fn decode_bbo(payload: &str, instrument: &InstrumentSpec, recv_ns: u64) -> Option<Bbo> {
    let data: WireBbo<'_> = serde_json::from_str(payload).ok()?;
    if data.coin != instrument.symbol {
        return None;
    }
    let mut levels = data.bbo.into_iter();
    let bid = levels.next()??;
    let ask = levels.next()??;
    Some(Bbo {
        bid_px: parse_decimal_scaled(&bid.px, instrument.max_price_decimals).ok()?,
        bid_sz: parse_decimal_scaled(&bid.sz, instrument.sz_decimals).ok()?,
        ask_px: parse_decimal_scaled(&ask.px, instrument.max_price_decimals).ok()?,
        ask_sz: parse_decimal_scaled(&ask.sz, instrument.sz_decimals).ok()?,
        exchange_ms: wire_u64(data.time.as_ref()),
        recv_ns,
    })
}

fn decode_trades(payload: &str, instrument: &InstrumentSpec, recv_ns: u64) -> Vec<TradePrint> {
    let Ok(items) = serde_json::from_str::<Vec<WireTrade<'_>>>(payload) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter(|item| item.coin == instrument.symbol)
        .filter_map(|item| {
            let aggressor = match item.side.as_ref() {
                "B" => AggressorSide::Buy,
                "A" => AggressorSide::Sell,
                _ => return None,
            };
            Some(TradePrint {
                aggressor,
                px: parse_decimal_scaled(&item.px, instrument.max_price_decimals).ok()?,
                qty_units: parse_decimal_scaled(&item.sz, instrument.sz_decimals).ok()?,
                exchange_ms: wire_u64(item.time.as_ref()),
                recv_ns,
                trade_id: wire_u64(item.tid.as_ref()),
            })
        })
        .collect()
}

fn decode_book(payload: &str, instrument: &InstrumentSpec, recv_ns: u64) -> Option<BookSnapshot> {
    let data: WireBook<'_> = serde_json::from_str(payload).ok()?;
    if data.coin != instrument.symbol {
        return None;
    }
    let mut sides = data.levels.into_iter();
    let decode_side = |levels: Vec<WireLevel<'_>>| -> Vec<BookLevel> {
        levels
            .into_iter()
            .take(20)
            .filter_map(|level| {
                Some(BookLevel {
                    px: parse_decimal_scaled(&level.px, instrument.max_price_decimals).ok()?,
                    qty_units: parse_decimal_scaled(&level.sz, instrument.sz_decimals).ok()?,
                })
            })
            .collect()
    };
    Some(BookSnapshot {
        bids: decode_side(sides.next()?),
        asks: decode_side(sides.next()?),
        exchange_ms: wire_u64(data.time.as_ref()),
        recv_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_parser_rounds_like_python_collector_precision() {
        assert_eq!(parse_decimal_scaled("0.131975", 5).unwrap(), 13_198);
        assert_eq!(parse_decimal_scaled("-2.5", 0).unwrap(), -3);
    }
}
