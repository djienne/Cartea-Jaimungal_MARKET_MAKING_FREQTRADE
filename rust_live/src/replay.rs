use crate::execution::MarketDataSource;
use crate::instrument::InstrumentSpec;
use crate::parquet_io::MarketDataSet;
use crate::types::{AggressorSide, Bbo, BookLevel, BookSnapshot, MarketEvent, TradePrint};
use anyhow::Result;
use async_trait::async_trait;
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub struct ParquetReplaySource {
    events: std::vec::IntoIter<MarketEvent>,
}

impl ParquetReplaySource {
    pub fn new(data: &MarketDataSet, instrument: &InstrumentSpec) -> Result<Self> {
        let mut events = Vec::new();
        for mid in &data.mids {
            events.push((
                mid.ts_ms,
                2_u8,
                MarketEvent::Bbo(Bbo {
                    bid_px: instrument.price_to_units(mid.bid)?,
                    bid_sz: 0,
                    ask_px: instrument.price_to_units(mid.ask)?,
                    ask_sz: 0,
                    exchange_ms: mid.ts_ms.max(0.0) as u64,
                    recv_ns: (mid.ts_ms.max(0.0) * 1_000_000.0) as u64,
                }),
            ));
        }
        for trade in &data.trades {
            let trade_id = trade.trade_id.as_deref().map_or(0, stable_trade_id);
            events.push((
                trade.ts_ms,
                0_u8,
                MarketEvent::Trade(TradePrint {
                    aggressor: if trade.side == "buy" {
                        AggressorSide::Buy
                    } else {
                        AggressorSide::Sell
                    },
                    px: instrument.price_to_units(trade.price)?,
                    qty_units: instrument.size_to_units(trade.size)?,
                    exchange_ms: trade.ts_ms.max(0.0) as u64,
                    recv_ns: (trade.ts_ms.max(0.0) * 1_000_000.0) as u64,
                    trade_id,
                }),
            ));
        }
        for book in &data.books {
            events.push((
                book.ts_ms,
                1_u8,
                MarketEvent::Book(BookSnapshot {
                    bids: vec![BookLevel {
                        px: instrument.price_to_units(book.bid)?,
                        qty_units: instrument.size_to_units(book.bid_size)?,
                    }],
                    asks: vec![BookLevel {
                        px: instrument.price_to_units(book.ask)?,
                        qty_units: instrument.size_to_units(book.ask_size)?,
                    }],
                    exchange_ms: book.ts_ms.max(0.0) as u64,
                    recv_ns: (book.ts_ms.max(0.0) * 1_000_000.0) as u64,
                }),
            ));
        }
        // Trades precede book/BBO updates at an identical exchange timestamp so
        // a quote cannot see a post-trade book while simulating that trade.
        events.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        Ok(Self {
            events: events
                .into_iter()
                .map(|(_, _, event)| event)
                .collect::<Vec<_>>()
                .into_iter(),
        })
    }
}

#[async_trait]
impl MarketDataSource for ParquetReplaySource {
    async fn next_event(&mut self) -> Result<Option<MarketEvent>> {
        Ok(self.events.next())
    }
}

fn stable_trade_id(value: &str) -> u64 {
    if let Ok(parsed) = value.parse() {
        return parsed;
    }
    let digest = Sha256::digest(value.as_bytes());
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_io::{MidRecord, ShardStats, TimeSource, TradeRecord};

    #[tokio::test]
    async fn identical_dataset_produces_identical_event_sequence() {
        let data = MarketDataSet {
            symbol: "SYN".to_owned(),
            time_source: TimeSource::Exchange,
            mids: vec![MidRecord {
                ts_ms: 1_000.0,
                bid: 99.0,
                ask: 101.0,
                mid: 100.0,
            }],
            trades: vec![TradeRecord {
                ts_ms: 1_000.0,
                side: "buy".to_owned(),
                price: 101.0,
                size: 2.0,
                trade_id: Some("7".to_owned()),
            }],
            books: Vec::new(),
            window_start_ms: 0.0,
            window_end_ms: 1_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let instrument = InstrumentSpec {
            symbol: "SYN".to_owned(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 0,
            max_price_decimals: 0,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
        };
        let mut first = ParquetReplaySource::new(&data, &instrument).unwrap();
        let mut second = ParquetReplaySource::new(&data, &instrument).unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        while let Some(event) = first.next_event().await.unwrap() {
            a.push(event);
        }
        while let Some(event) = second.next_event().await.unwrap() {
            b.push(event);
        }
        assert_eq!(a, b);
        assert!(matches!(a[0], MarketEvent::Trade(_)));
    }
}
