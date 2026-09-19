use crate::execution::MarketDataSource;
use crate::instrument::InstrumentSpec;
use crate::parquet_io::MarketDataSet;
use crate::types::{AggressorSide, Bbo, BookLevel, BookSnapshot, MarketEvent, TradePrint};
use anyhow::Result;
use async_trait::async_trait;
use sha2::{Digest, Sha256};

/// Convert one side's recorded levels to venue units, falling back to the
/// top-of-book fields for shards that carried only level 0.
fn book_levels(
    instrument: &InstrumentSpec,
    levels: &[(f64, f64)],
    top_px: f64,
    top_size: f64,
) -> Result<Vec<BookLevel>> {
    if levels.is_empty() {
        return Ok(vec![BookLevel {
            px: instrument.price_to_units(top_px)?,
            qty_units: instrument.size_to_units(top_size)?,
        }]);
    }
    levels
        .iter()
        .map(|(px, size)| {
            Ok(BookLevel {
                px: instrument.price_to_units(*px)?,
                qty_units: instrument.size_to_units(*size)?,
            })
        })
        .collect()
}

#[derive(Debug)]
pub struct ParquetReplaySource {
    events: std::vec::IntoIter<MarketEvent>,
    pub receive_time_fallbacks: usize,
    pub rejected_touches: usize,
}

impl ParquetReplaySource {
    pub fn new(data: &MarketDataSet, instrument: &InstrumentSpec) -> Result<Self> {
        Self::with_freshness(data, instrument, u64::MAX)
    }

    pub fn with_freshness(
        data: &MarketDataSet,
        instrument: &InstrumentSpec,
        max_age_ms: u64,
    ) -> Result<Self> {
        let mut events = Vec::new();
        let mut receive_time_fallbacks = 0;
        let mut received = |value: Option<f64>, exchange: f64| {
            value
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or_else(|| {
                    receive_time_fallbacks += 1;
                    exchange
                })
        };
        for mid in &data.mids {
            let arrival = received(mid.received_ms, mid.ts_ms);
            events.push((
                arrival,
                2_u8,
                MarketEvent::Bbo(Bbo {
                    bid_px: instrument.price_to_units(mid.bid)?,
                    bid_sz: instrument.size_to_units(mid.bid_size)?,
                    ask_px: instrument.price_to_units(mid.ask)?,
                    ask_sz: instrument.size_to_units(mid.ask_size)?,
                    exchange_ms: mid.ts_ms.max(0.0) as u64,
                    recv_ns: (arrival.max(0.0) * 1_000_000.0).round() as u64,
                }),
            ));
        }
        for trade in &data.trades {
            let arrival = received(trade.received_ms, trade.ts_ms);
            let trade_id = trade.trade_id.as_deref().map_or(0, stable_trade_id);
            events.push((
                arrival,
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
                    recv_ns: (arrival.max(0.0) * 1_000_000.0).round() as u64,
                    trade_id,
                }),
            ));
        }
        for book in &data.books {
            let arrival = received(book.received_ms, book.ts_ms);
            // Every recorded level, not just the top: the simulator fills a
            // virtual order only once it can see the queue at that price,
            // and maker quotes rest inside the book, not at the touch.
            events.push((
                arrival,
                1_u8,
                MarketEvent::Book(BookSnapshot {
                    bids: book_levels(instrument, &book.bid_levels, book.bid, book.bid_size)?,
                    asks: book_levels(instrument, &book.ask_levels, book.ask, book.ask_size)?,
                    exchange_ms: book.ts_ms.max(0.0) as u64,
                    recv_ns: (arrival.max(0.0) * 1_000_000.0).round() as u64,
                }),
            ));
        }
        // Recorded arrival order; ties are trades, books, direct BBOs, then
        // stable file/row order. A book's derived BBO stays adjacent to it.
        events.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let mut normalized = Vec::with_capacity(events.len());
        let mut last_exchange_ms = 0;
        let mut rejected_touches = 0;
        for (arrival, _, event) in events {
            let touch = match &event {
                MarketEvent::Book(book) => book.bbo(),
                MarketEvent::Bbo(bbo) => Some(*bbo),
                MarketEvent::Trade(_) => {
                    normalized.push(event);
                    continue;
                }
            };
            let Some(bbo) =
                touch.filter(|bbo| bbo.is_fresh(arrival as u64, last_exchange_ms, max_age_ms))
            else {
                rejected_touches += 1;
                continue;
            };
            last_exchange_ms = bbo.exchange_ms;
            if matches!(event, MarketEvent::Book(_)) {
                normalized.push(event);
            }
            normalized.push(MarketEvent::Bbo(bbo));
        }
        Ok(Self {
            events: normalized.into_iter(),
            receive_time_fallbacks,
            rejected_touches,
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
    async fn receipt_order_book_touch_and_fallback_are_explicit() {
        let data = MarketDataSet {
            symbol: "SYN".into(),
            time_source: TimeSource::Exchange,
            mids: vec![MidRecord {
                received_ms: Some(1_030.0),
                ts_ms: 1_000.0,
                bid: 99.0,
                ask: 101.0,
                mid: 100.0,
                bid_size: 3.0,
                ask_size: 4.0,
            }],
            trades: vec![TradeRecord {
                received_ms: Some(1_020.0),
                ts_ms: 1_010.0,
                side: "buy".into(),
                price: 101.0,
                size: 1.0,
                trade_id: Some("1".into()),
            }],
            books: vec![crate::parquet_io::BookTopRecord {
                received_ms: Some(1_040.0),
                ts_ms: 1_015.0,
                bid: 100.0,
                ask: 102.0,
                bid_size: 7.0,
                ask_size: 8.0,
                bid_levels: vec![],
                ask_levels: vec![],
            }],
            window_start_ms: 900.0,
            window_end_ms: 2_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let instrument = InstrumentSpec {
            symbol: "SYN".into(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 0,
            max_price_decimals: 0,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        };
        let mut source = ParquetReplaySource::with_freshness(&data, &instrument, 100).unwrap();
        assert_eq!(source.receive_time_fallbacks, 0);
        assert!(matches!(
            source.next_event().await.unwrap(),
            Some(MarketEvent::Trade(_))
        ));
        let Some(MarketEvent::Bbo(first)) = source.next_event().await.unwrap() else {
            panic!("missing BBO")
        };
        assert_eq!(first.bid_sz, 3);
        let Some(MarketEvent::Book(book)) = source.next_event().await.unwrap() else {
            panic!("missing depth")
        };
        let Some(MarketEvent::Bbo(touch)) = source.next_event().await.unwrap() else {
            panic!("missing derived touch")
        };
        assert_eq!(Some(touch), book.bbo());
        assert_eq!(touch.bid_sz, 7);
        assert!(source.next_event().await.unwrap().is_none());
        let mut legacy = data.clone();
        legacy.mids[0].received_ms = None;
        assert_eq!(
            ParquetReplaySource::new(&legacy, &instrument)
                .unwrap()
                .receive_time_fallbacks,
            1
        );
        assert_eq!(
            ParquetReplaySource::with_freshness(&data, &instrument, 5)
                .unwrap()
                .rejected_touches,
            2
        );
    }

    #[test]
    fn replay_partition_excludes_cutoff_and_future_observations_from_training() {
        let mut data = MarketDataSet {
            symbol: "SYN".to_owned(),
            time_source: TimeSource::Exchange,
            mids: [0.0, 499.0, 500.0, 1_000.0]
                .into_iter()
                .map(|ts_ms| MidRecord {
                    received_ms: None,
                    bid_size: 0.0,
                    ask_size: 0.0,
                    ts_ms,
                    bid: 99.0,
                    ask: 101.0,
                    mid: 100.0,
                })
                .collect(),
            trades: [0.0, 499.0, 500.0, 1_000.0]
                .into_iter()
                .map(|ts_ms| TradeRecord {
                    received_ms: None,
                    ts_ms,
                    side: "buy".to_owned(),
                    price: 101.0,
                    size: 2.0,
                    trade_id: None,
                })
                .collect(),
            books: Vec::new(),
            window_start_ms: 0.0,
            window_end_ms: 1_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let (training, scoring) = data.split_for_replay(0.5).unwrap();
        assert_eq!(training.mids.len(), 2);
        assert_eq!(scoring.mids[0].ts_ms, 500.0);
        for row in &mut data.mids[2..] {
            row.mid *= 100.0;
        }
        for row in &mut data.trades[2..] {
            row.size *= 1_000.0;
        }
        let (unchanged, _) = data.split_for_replay(0.5).unwrap();
        assert_eq!(training.mids, unchanged.mids);
        assert_eq!(training.trades, unchanged.trades);
        let mut silent = data.clone();
        silent.mids.retain(|row| row.ts_ms < 500.0);
        silent.trades.retain(|row| row.ts_ms < 500.0);
        let (_, scored_silence) = silent.split_at(500.0).unwrap();
        assert!(scored_silence.mids.is_empty() && scored_silence.trades.is_empty());
        for fraction in [0.0, 1.0, -1.0, f64::NAN] {
            assert!(data.split_for_replay(fraction).is_err());
        }
        data.trades.retain(|row| row.ts_ms >= 500.0);
        assert!(data.split_for_replay(0.5).is_err());
    }

    #[tokio::test]
    async fn identical_dataset_produces_identical_event_sequence() {
        let data = MarketDataSet {
            symbol: "SYN".to_owned(),
            time_source: TimeSource::Exchange,
            mids: vec![MidRecord {
                received_ms: None,
                bid_size: 0.0,
                ask_size: 0.0,
                ts_ms: 1_000.0,
                bid: 99.0,
                ask: 101.0,
                mid: 100.0,
            }],
            trades: vec![TradeRecord {
                received_ms: None,
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
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
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
