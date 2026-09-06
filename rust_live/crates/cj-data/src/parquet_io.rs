use crate::config::CalibrationConfig;
use crate::instrument::InstrumentSpec;
use crate::types::MarketEvent;
use anyhow::{bail, Context, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray,
};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
use fs2::FileExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeSource {
    Exchange,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MidRecord {
    pub ts_ms: f64,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeRecord {
    pub ts_ms: f64,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub trade_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BookTopRecord {
    pub ts_ms: f64,
    pub bid: f64,
    pub bid_size: f64,
    pub ask: f64,
    pub ask_size: f64,
    /// Every recorded level, best first, as `(price, size)`; level 0 repeats
    /// the top fields. The replay simulator can only fill a virtual order
    /// whose queue it can see, and a maker quote normally rests well inside
    /// the book rather than at the touch, so with only the top level (and
    /// zero queue decay) replay fills were structurally impossible.
    #[serde(default)]
    pub bid_levels: Vec<(f64, f64)>,
    #[serde(default)]
    pub ask_levels: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardStats {
    pub files_present: usize,
    pub files_considered: usize,
    pub files_read: usize,
    pub files_failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataSet {
    pub symbol: String,
    pub time_source: TimeSource,
    pub mids: Vec<MidRecord>,
    pub trades: Vec<TradeRecord>,
    pub books: Vec<BookTopRecord>,
    pub window_start_ms: f64,
    pub window_end_ms: f64,
    pub duplicate_trade_ids_dropped: usize,
    pub price_shards: ShardStats,
    pub trade_shards: ShardStats,
    pub orderbook_shards: ShardStats,
}

impl MarketDataSet {
    pub fn split_for_replay(&self, train_fraction: f64) -> Result<(Self, Self)> {
        if !train_fraction.is_finite()
            || !(0.0..1.0).contains(&train_fraction)
            || train_fraction == 0.0
        {
            bail!("replay train fraction must be strictly between zero and one");
        }
        let cutoff =
            self.window_start_ms + train_fraction * (self.window_end_ms - self.window_start_ms);
        let mut training = self.clone();
        let mut scoring = self.clone();
        training.mids.retain(|row| row.ts_ms < cutoff);
        training.trades.retain(|row| row.ts_ms < cutoff);
        training.books.retain(|row| row.ts_ms < cutoff);
        training.window_end_ms = cutoff;
        scoring.mids.retain(|row| row.ts_ms >= cutoff);
        scoring.trades.retain(|row| row.ts_ms >= cutoff);
        scoring.books.retain(|row| row.ts_ms >= cutoff);
        scoring.window_start_ms = cutoff;
        if training.mids.is_empty()
            || training.trades.is_empty()
            || scoring.mids.is_empty()
            || scoring.trades.is_empty()
        {
            bail!("replay requires prices and trades in both training and scoring partitions");
        }
        Ok((training, scoring))
    }
}

#[derive(Debug, Clone)]
struct RawPrice {
    local_ms: f64,
    exchange_ms: Option<i64>,
    price: f64,
    size: f64,
    side: String,
}

#[derive(Debug, Clone)]
struct RawTrade {
    local_ms: f64,
    exchange_ms: Option<i64>,
    price: f64,
    size: f64,
    side: String,
    trade_id: Option<String>,
}

#[derive(Debug, Clone)]
struct RawBookTop {
    local_ms: f64,
    exchange_ms: Option<i64>,
    bid: f64,
    bid_size: f64,
    ask: f64,
    ask_size: f64,
    bid_levels: Vec<(f64, f64)>,
    ask_levels: Vec<(f64, f64)>,
}

/// `range` is an explicit `(start_ms, end_ms)` window over the collected tape.
/// Without it the window is `config.window_minutes` ending at the newest shard.
pub fn load_market_window(
    data_root: &Path,
    symbol: &str,
    config: &CalibrationConfig,
    range: Option<(u64, u64)>,
) -> Result<MarketDataSet> {
    let base = data_root.join(symbol);
    let (window_minutes, anchor_ms) = match range {
        Some((start, end)) if end > start => ((end - start) / 60_000 + 1, Some(end)),
        Some(_) => bail!("replay range end must be after its start"),
        None => (config.window_minutes, None),
    };
    let margin = config.shard_window_margin_seconds;
    let (mut prices, price_shards) =
        read_prices(&base.join("prices"), window_minutes, margin, anchor_ms)?;
    let (mut trades, trade_shards) =
        read_trades(&base.join("trades"), window_minutes, margin, anchor_ms)?;
    if prices.is_empty() {
        bail!("no usable price data under {}", base.display());
    }
    validate_shard_failures(&price_shards, config.max_shard_failure_rate, "prices")?;
    validate_shard_failures(&trade_shards, config.max_shard_failure_rate, "trades")?;

    let use_exchange = exchange_coverage_prices(&prices) >= 0.99
        && (trades.is_empty() || exchange_coverage_trades(&trades) >= 0.99);
    let time_source = if use_exchange {
        TimeSource::Exchange
    } else {
        TimeSource::Local
    };
    let timestamp_price = |row: &RawPrice| match time_source {
        TimeSource::Exchange => row.exchange_ms.map_or(row.local_ms, |value| value as f64),
        TimeSource::Local => row.local_ms,
    };
    let timestamp_trade = |row: &RawTrade| match time_source {
        TimeSource::Exchange => row.exchange_ms.map_or(row.local_ms, |value| value as f64),
        TimeSource::Local => row.local_ms,
    };
    prices.sort_by(|a, b| timestamp_price(a).total_cmp(&timestamp_price(b)));
    trades.sort_by(|a, b| timestamp_trade(a).total_cmp(&timestamp_trade(b)));

    let mids = build_mids(&prices, time_source);
    if mids.is_empty() {
        bail!("price shards contain no complete, valid BBO");
    }
    let (normalized_trades, duplicate_trade_ids_dropped) = normalize_trades(trades, time_source);
    let mid_end = mids.last().map_or(0.0, |row| row.ts_ms);
    let data_end_ms = normalized_trades
        .last()
        .map_or(mid_end, |row| mid_end.min(row.ts_ms));
    let (window_start_ms, window_end_ms) = match range {
        Some((start, end)) => (start as f64, (end as f64).min(data_end_ms)),
        None => (data_end_ms - window_minutes as f64 * 60_000.0, data_end_ms),
    };
    let mids: Vec<_> = mids
        .into_iter()
        .filter(|row| row.ts_ms >= window_start_ms && row.ts_ms <= window_end_ms)
        .collect();
    let trades: Vec<_> = normalized_trades
        .into_iter()
        .filter(|row| row.ts_ms >= window_start_ms && row.ts_ms <= window_end_ms)
        .collect();

    let (raw_books, orderbook_shards) =
        read_book_tops(&base.join("orderbooks"), window_minutes, margin, anchor_ms)?;
    // Historical orderbooks improve replay queue initialization but do not
    // enter kappa/lambda/epsilon estimation. The legacy collector can atomically
    // compact/delete these shards while Rust is reading; retain the failure
    // counts as diagnostics without blocking a scientifically complete
    // price+trade calibration.
    let mut books: Vec<_> = raw_books
        .into_iter()
        .map(|row| BookTopRecord {
            ts_ms: match time_source {
                TimeSource::Exchange => row.exchange_ms.map_or(row.local_ms, |value| value as f64),
                TimeSource::Local => row.local_ms,
            },
            bid: row.bid,
            bid_size: row.bid_size,
            ask: row.ask,
            ask_size: row.ask_size,
            bid_levels: row.bid_levels,
            ask_levels: row.ask_levels,
        })
        .filter(|row| row.ts_ms >= window_start_ms && row.ts_ms <= window_end_ms)
        .collect();
    books.sort_by(|a, b| a.ts_ms.total_cmp(&b.ts_ms));

    Ok(MarketDataSet {
        symbol: symbol.to_owned(),
        time_source,
        mids,
        trades,
        books,
        window_start_ms,
        window_end_ms,
        duplicate_trade_ids_dropped,
        price_shards,
        trade_shards,
        orderbook_shards,
    })
}

fn read_prices(
    directory: &Path,
    window_minutes: u64,
    margin_seconds: u64,
    anchor_ms: Option<u64>,
) -> Result<(Vec<RawPrice>, ShardStats)> {
    let files = selected_shards(directory, window_minutes, margin_seconds, anchor_ms)?;
    let mut rows = Vec::new();
    let mut stats = ShardStats {
        files_present: count_parquet_files(directory)?,
        files_considered: files.len(),
        ..ShardStats::default()
    };
    for path in files {
        match read_batches(&path).and_then(|batches| parse_price_batches(&batches)) {
            Ok(mut batch_rows) => {
                stats.files_read += 1;
                rows.append(&mut batch_rows);
            }
            Err(_) => stats.files_failed += 1,
        }
    }
    Ok((rows, stats))
}

fn read_trades(
    directory: &Path,
    window_minutes: u64,
    margin_seconds: u64,
    anchor_ms: Option<u64>,
) -> Result<(Vec<RawTrade>, ShardStats)> {
    let files = selected_shards(directory, window_minutes, margin_seconds, anchor_ms)?;
    let mut rows = Vec::new();
    let mut stats = ShardStats {
        files_present: count_parquet_files(directory)?,
        files_considered: files.len(),
        ..ShardStats::default()
    };
    for path in files {
        match read_batches(&path).and_then(|batches| parse_trade_batches(&batches)) {
            Ok(mut batch_rows) => {
                stats.files_read += 1;
                rows.append(&mut batch_rows);
            }
            Err(_) => stats.files_failed += 1,
        }
    }
    Ok((rows, stats))
}

fn read_book_tops(
    directory: &Path,
    window_minutes: u64,
    margin_seconds: u64,
    anchor_ms: Option<u64>,
) -> Result<(Vec<RawBookTop>, ShardStats)> {
    if !directory.exists() {
        return Ok((Vec::new(), ShardStats::default()));
    }
    let files = selected_shards(directory, window_minutes, margin_seconds, anchor_ms)?;
    let mut rows = Vec::new();
    let mut stats = ShardStats {
        files_present: count_parquet_files(directory)?,
        files_considered: files.len(),
        ..ShardStats::default()
    };
    for path in files {
        match read_batches(&path).and_then(|batches| parse_book_batches(&batches)) {
            Ok(mut batch_rows) => {
                stats.files_read += 1;
                rows.append(&mut batch_rows);
            }
            Err(_) => stats.files_failed += 1,
        }
    }
    Ok((rows, stats))
}

/// Shards stamped within `window_minutes` (plus margin) of `anchor_ms`, or of
/// the newest shard when no anchor is given.
fn selected_shards(
    directory: &Path,
    window_minutes: u64,
    margin_seconds: u64,
    anchor_ms: Option<u64>,
) -> Result<Vec<PathBuf>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut stamped = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("parquet") {
            continue;
        }
        if let Some(timestamp) = shard_timestamp(&path) {
            stamped.push((timestamp, path));
        }
    }
    stamped.sort_by_key(|(timestamp, _)| *timestamp);
    let Some((newest, _)) = stamped.last() else {
        return Ok(Vec::new());
    };
    let latest = anchor_ms.unwrap_or(*newest);
    let lookback = (window_minutes * 60 + margin_seconds) * 1_000;
    let cutoff = latest.saturating_sub(lookback);
    let ceiling = latest.saturating_add(margin_seconds * 1_000);
    Ok(stamped
        .into_iter()
        .filter_map(|(timestamp, path)| {
            (timestamp >= cutoff && timestamp <= ceiling).then_some(path)
        })
        .collect())
}

/// Parse a UTC instant given as RFC 3339 (`2026-09-02T00:00:00Z`) or epoch ms.
pub fn parse_utc_ms(text: &str) -> Result<u64> {
    if let Ok(ms) = text.parse::<u64>() {
        return Ok(ms);
    }
    let parsed = chrono::DateTime::parse_from_rfc3339(text)
        .with_context(|| format!("{text:?} is neither epoch ms nor RFC 3339"))?;
    u64::try_from(parsed.timestamp_millis()).context("timestamp before 1970")
}

fn shard_timestamp(path: &Path) -> Option<u64> {
    path.file_stem()?.to_str()?.rsplit('_').next()?.parse().ok()
}

fn count_parquet_files(directory: &Path) -> Result<usize> {
    if !directory.exists() {
        return Ok(0);
    }
    Ok(std::fs::read_dir(directory)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| {
            entry.path().extension().and_then(|value| value.to_str()) == Some("parquet")
        })
        .count())
}

fn read_batches(path: &Path) -> Result<Vec<RecordBatch>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8_192)
        .build()?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn parse_price_batches(batches: &[RecordBatch]) -> Result<Vec<RawPrice>> {
    let mut rows = Vec::new();
    for batch in batches {
        let timestamp = numeric_f64(batch, "timestamp")?;
        let price = numeric_f64(batch, "price")?;
        let size = numeric_f64(batch, "size")?;
        let side = string_values(batch, "side")?;
        let exchange = optional_i64(batch, "exchange_timestamp")?;
        for index in 0..batch.num_rows() {
            rows.push(RawPrice {
                local_ms: timestamp[index] * 1_000.0,
                exchange_ms: exchange[index],
                price: price[index],
                size: size[index],
                side: side[index].clone(),
            });
        }
    }
    Ok(rows)
}

fn parse_trade_batches(batches: &[RecordBatch]) -> Result<Vec<RawTrade>> {
    let mut rows = Vec::new();
    for batch in batches {
        let timestamp = numeric_f64(batch, "timestamp")?;
        let price = numeric_f64(batch, "price")?;
        let size = numeric_f64(batch, "size")?;
        let side = string_values(batch, "side")?;
        let trade_id = optional_strings(batch, "trade_id")?;
        let exchange = optional_i64(batch, "exchange_timestamp")?;
        for index in 0..batch.num_rows() {
            rows.push(RawTrade {
                local_ms: timestamp[index] * 1_000.0,
                exchange_ms: exchange[index],
                price: price[index],
                size: size[index],
                side: side[index].clone(),
                trade_id: trade_id[index].clone(),
            });
        }
    }
    Ok(rows)
}

/// Deepest level the collector writes (`bid_price_0..19`).
const MAX_BOOK_LEVELS: usize = 20;

fn usable_level(px: f64, size: f64) -> bool {
    px.is_finite() && px > 0.0 && size.is_finite() && size >= 0.0
}

fn parse_book_batches(batches: &[RecordBatch]) -> Result<Vec<RawBookTop>> {
    let mut rows = Vec::new();
    for batch in batches {
        let timestamp = numeric_f64(batch, "timestamp")?;
        let exchange = optional_i64(batch, "exchange_timestamp")?;
        // Level 0 is mandatory; deeper levels are read while the shard has
        // them, so an older or narrower shard still loads as top-of-book.
        let mut levels: Vec<[Vec<f64>; 4]> = Vec::with_capacity(MAX_BOOK_LEVELS);
        for level in 0..MAX_BOOK_LEVELS {
            let names = [
                format!("bid_price_{level}"),
                format!("bid_size_{level}"),
                format!("ask_price_{level}"),
                format!("ask_size_{level}"),
            ];
            if level > 0
                && names
                    .iter()
                    .any(|name| batch.schema().index_of(name).is_err())
            {
                break;
            }
            levels.push([
                numeric_f64(batch, &names[0])?,
                numeric_f64(batch, &names[1])?,
                numeric_f64(batch, &names[2])?,
                numeric_f64(batch, &names[3])?,
            ]);
        }
        for index in 0..batch.num_rows() {
            let mut bid_levels = Vec::with_capacity(levels.len());
            let mut ask_levels = Vec::with_capacity(levels.len());
            for [bid_px, bid_sz, ask_px, ask_sz] in &levels {
                if usable_level(bid_px[index], bid_sz[index]) {
                    bid_levels.push((bid_px[index], bid_sz[index]));
                }
                if usable_level(ask_px[index], ask_sz[index]) {
                    ask_levels.push((ask_px[index], ask_sz[index]));
                }
            }
            rows.push(RawBookTop {
                local_ms: timestamp[index] * 1_000.0,
                exchange_ms: exchange[index],
                bid: levels[0][0][index],
                bid_size: levels[0][1][index],
                ask: levels[0][2][index],
                ask_size: levels[0][3][index],
                bid_levels,
                ask_levels,
            });
        }
    }
    Ok(rows)
}

fn numeric_f64(batch: &RecordBatch, name: &str) -> Result<Vec<f64>> {
    let index = batch.schema().index_of(name)?;
    let array = batch.column(index);
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..values.len()).map(|row| values.value(row)).collect());
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok((0..values.len())
            .map(|row| f64::from(values.value(row)))
            .collect());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..values.len())
            .map(|row| values.value(row) as f64)
            .collect());
    }
    bail!(
        "column {name} has unsupported numeric type {:?}",
        array.data_type()
    )
}

fn optional_i64(batch: &RecordBatch, name: &str) -> Result<Vec<Option<i64>>> {
    let Ok(index) = batch.schema().index_of(name) else {
        return Ok(vec![None; batch.num_rows()]);
    };
    let array = batch.column(index);
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..values.len())
            .map(|row| (!values.is_null(row)).then(|| values.value(row)))
            .collect());
    }
    bail!(
        "column {name} has unsupported timestamp type {:?}",
        array.data_type()
    )
}

fn string_values(batch: &RecordBatch, name: &str) -> Result<Vec<String>> {
    optional_strings(batch, name)?
        .into_iter()
        .map(|value| value.ok_or_else(|| anyhow::anyhow!("column {name} contains a null value")))
        .collect()
}

fn optional_strings(batch: &RecordBatch, name: &str) -> Result<Vec<Option<String>>> {
    let Ok(index) = batch.schema().index_of(name) else {
        return Ok(vec![None; batch.num_rows()]);
    };
    let array = batch.column(index);
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok((0..values.len())
            .map(|row| (!values.is_null(row)).then(|| values.value(row).to_owned()))
            .collect());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok((0..values.len())
            .map(|row| (!values.is_null(row)).then(|| values.value(row).to_owned()))
            .collect());
    }
    let converted = cast(array, &DataType::Utf8)?;
    let values = converted
        .as_any()
        .downcast_ref::<StringArray>()
        .context("dictionary string cast did not produce Utf8")?;
    Ok((0..values.len())
        .map(|row| (!values.is_null(row)).then(|| values.value(row).to_owned()))
        .collect())
}

fn exchange_coverage_prices(rows: &[RawPrice]) -> f64 {
    rows.iter()
        .filter(|row| row.exchange_ms.is_some_and(|value| value > 0))
        .count() as f64
        / rows.len().max(1) as f64
}

fn exchange_coverage_trades(rows: &[RawTrade]) -> f64 {
    rows.iter()
        .filter(|row| row.exchange_ms.is_some_and(|value| value > 0))
        .count() as f64
        / rows.len().max(1) as f64
}

fn build_mids(rows: &[RawPrice], source: TimeSource) -> Vec<MidRecord> {
    let timestamp = |row: &RawPrice| match source {
        TimeSource::Exchange => row.exchange_ms.map_or(row.local_ms, |value| value as f64),
        TimeSource::Local => row.local_ms,
    };
    let mut output = Vec::new();
    let mut bid = None;
    let mut ask = None;
    let mut index = 0;
    while index < rows.len() {
        let ts = timestamp(&rows[index]);
        let mut next = index;
        while next < rows.len() && timestamp(&rows[next]) == ts {
            match rows[next].side.to_ascii_lowercase().as_str() {
                "bid" => bid = Some((rows[next].price, rows[next].size)),
                "ask" => ask = Some((rows[next].price, rows[next].size)),
                _ => {}
            }
            next += 1;
        }
        if let (Some((bid_px, _)), Some((ask_px, _))) = (bid, ask) {
            if bid_px > 0.0 && ask_px > bid_px {
                output.push(MidRecord {
                    ts_ms: ts,
                    bid: bid_px,
                    ask: ask_px,
                    mid: 0.5 * (bid_px + ask_px),
                });
            }
        }
        index = next;
    }
    output
}

fn normalize_trades(rows: Vec<RawTrade>, source: TimeSource) -> (Vec<TradeRecord>, usize) {
    let missing = ["", "None", "nan", "NaT", "<NA>", "none", "null"];
    let mut seen = HashSet::new();
    let mut dropped = 0;
    let mut output = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(id) = row.trade_id.as_deref() {
            if !missing.contains(&id) && !seen.insert(id.to_owned()) {
                dropped += 1;
                continue;
            }
        }
        let side = row.side.to_ascii_lowercase();
        if side != "buy" && side != "sell" {
            continue;
        }
        output.push(TradeRecord {
            ts_ms: match source {
                TimeSource::Exchange => row.exchange_ms.map_or(row.local_ms, |value| value as f64),
                TimeSource::Local => row.local_ms,
            },
            side,
            price: row.price,
            size: row.size,
            trade_id: row.trade_id,
        });
    }
    output.sort_by(|a, b| a.ts_ms.total_cmp(&b.ts_ms));
    (output, dropped)
}

fn validate_shard_failures(stats: &ShardStats, maximum: f64, stream: &str) -> Result<()> {
    if stats.files_considered == 0 {
        return Ok(());
    }
    let rate = stats.files_failed as f64 / stats.files_considered as f64;
    if rate > maximum {
        bail!("{stream} shard read failure rate {rate:.3} exceeds {maximum:.3}");
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct PriceRow {
    pub timestamp_seconds: f64,
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub exchange_timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct TradeRow {
    pub timestamp_seconds: f64,
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub trade_id: String,
    pub exchange_timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct BookRow {
    pub timestamp_seconds: f64,
    pub sequence: i64,
    pub exchange_timestamp: i64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

pub fn write_price_shard(path: &Path, rows: &[PriceRow]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Float64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("size", DataType::Float64, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("exchange_timestamp", DataType::Int64, false),
    ]));
    let sides: Vec<&str> = rows.iter().map(|row| row.side.as_str()).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.timestamp_seconds),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.price),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.size),
        )),
        Arc::new(StringArray::from(sides)),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.exchange_timestamp),
        )),
    ];
    write_batch_atomic(
        path,
        schema.clone(),
        &RecordBatch::try_new(schema, columns)?,
    )
}

pub fn write_trade_shard(path: &Path, rows: &[TradeRow]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Float64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("size", DataType::Float64, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("trade_id", DataType::Utf8, false),
        Field::new("exchange_timestamp", DataType::Int64, false),
    ]));
    let sides: Vec<&str> = rows.iter().map(|row| row.side.as_str()).collect();
    let ids: Vec<&str> = rows.iter().map(|row| row.trade_id.as_str()).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.timestamp_seconds),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.price),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.size),
        )),
        Arc::new(StringArray::from(sides)),
        Arc::new(StringArray::from(ids)),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.exchange_timestamp),
        )),
    ];
    write_batch_atomic(
        path,
        schema.clone(),
        &RecordBatch::try_new(schema, columns)?,
    )
}

pub fn write_book_shard(path: &Path, rows: &[BookRow], depth: usize) -> Result<()> {
    let mut fields = vec![
        Field::new("timestamp", DataType::Float64, false),
        Field::new("sequence", DataType::Int64, false),
        Field::new("exchange_timestamp", DataType::Int64, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.timestamp_seconds),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.sequence),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.exchange_timestamp),
        )),
    ];
    for level in 0..depth {
        for (side, is_bid) in [("bid", true), ("ask", false)] {
            fields.push(Field::new(
                format!("{side}_price_{level}"),
                DataType::Float64,
                true,
            ));
            fields.push(Field::new(
                format!("{side}_size_{level}"),
                DataType::Float64,
                true,
            ));
            let values = rows
                .iter()
                .map(|row| {
                    let levels = if is_bid { &row.bids } else { &row.asks };
                    levels.get(level).copied()
                })
                .collect::<Vec<_>>();
            columns.push(Arc::new(Float64Array::from(
                values
                    .iter()
                    .map(|value| value.map(|entry| entry.0))
                    .collect::<Vec<_>>(),
            )));
            columns.push(Arc::new(Float64Array::from(
                values
                    .iter()
                    .map(|value| value.map(|entry| entry.1))
                    .collect::<Vec<_>>(),
            )));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    write_batch_atomic(
        path,
        schema.clone(),
        &RecordBatch::try_new(schema, columns)?,
    )
}

fn write_batch_atomic(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) -> Result<()> {
    let parent = path.parent().context("Parquet output path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    let output = temporary.reopen()?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer = ArrowWriter::try_new(output, schema, Some(properties))?;
    writer.write(batch)?;
    writer.close()?;
    temporary
        .persist(path)
        .map_err(|error| anyhow::anyhow!(error.error))?;
    Ok(())
}

pub struct CollectorLock {
    path: PathBuf,
    file: File,
}

impl CollectorLock {
    pub fn acquire(path: &Path, _stale_after_seconds: u64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("cannot open collector lock: {}", path.display()))?;
        file.try_lock_exclusive()
            .with_context(|| format!("collector lock already held: {}", path.display()))?;
        file.set_len(0)?;
        writeln!(
            file,
            "pid={} created_ms={}",
            std::process::id(),
            crate::types::unix_ms()
        )?;
        Ok(Self {
            path: path.to_owned(),
            file,
        })
    }
}

impl Drop for CollectorLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
        let _ = std::fs::remove_file(&self.path);
    }
}

pub struct ParquetEventRecorder {
    root: PathBuf,
    symbol: String,
    instrument: InstrumentSpec,
    flush_interval_ms: u64,
    last_flush_ms: u64,
    prices: Vec<PriceRow>,
    trades: Vec<TradeRow>,
    books: Vec<BookRow>,
    seen_trade_ids: HashSet<u64>,
    seen_trade_order: VecDeque<(u64, u64)>,
    trade_time_high_water_ms: u64,
    trade_dedup_horizon_ms: u64,
    created_price_shards: Vec<PathBuf>,
    created_trade_shards: Vec<PathBuf>,
    created_book_shards: Vec<PathBuf>,
}

impl ParquetEventRecorder {
    pub fn new(
        root: PathBuf,
        instrument: InstrumentSpec,
        flush_interval_seconds: u64,
        retention_minutes: u64,
    ) -> Self {
        Self {
            symbol: instrument.symbol.clone(),
            root,
            instrument,
            flush_interval_ms: flush_interval_seconds.max(1).saturating_mul(1_000),
            last_flush_ms: crate::types::unix_ms(),
            prices: Vec::new(),
            trades: Vec::new(),
            books: Vec::new(),
            seen_trade_ids: HashSet::new(),
            seen_trade_order: VecDeque::new(),
            trade_time_high_water_ms: 0,
            trade_dedup_horizon_ms: retention_minutes.saturating_mul(60_000),
            created_price_shards: Vec::new(),
            created_trade_shards: Vec::new(),
            created_book_shards: Vec::new(),
        }
    }

    pub fn record(&mut self, event: &MarketEvent, local_ms: u64) -> Result<()> {
        let timestamp_seconds = local_ms as f64 / 1_000.0;
        match event {
            MarketEvent::Bbo(bbo) => {
                self.prices.push(PriceRow {
                    timestamp_seconds,
                    price: self.instrument.price_from_units(bbo.bid_px),
                    size: self.instrument.size_from_units(bbo.bid_sz),
                    side: "bid".to_owned(),
                    exchange_timestamp: bbo.exchange_ms as i64,
                });
                self.prices.push(PriceRow {
                    timestamp_seconds,
                    price: self.instrument.price_from_units(bbo.ask_px),
                    size: self.instrument.size_from_units(bbo.ask_sz),
                    side: "ask".to_owned(),
                    exchange_timestamp: bbo.exchange_ms as i64,
                });
            }
            MarketEvent::Trade(trade) => {
                self.trade_time_high_water_ms =
                    self.trade_time_high_water_ms.max(trade.exchange_ms);
                let cutoff = self
                    .trade_time_high_water_ms
                    .saturating_sub(self.trade_dedup_horizon_ms);
                while self
                    .seen_trade_order
                    .front()
                    .is_some_and(|(exchange_ms, _)| *exchange_ms < cutoff)
                {
                    if let Some((_, expired_id)) = self.seen_trade_order.pop_front() {
                        self.seen_trade_ids.remove(&expired_id);
                    }
                }
                if trade.trade_id != 0 && !self.seen_trade_ids.insert(trade.trade_id) {
                    return Ok(());
                }
                if trade.trade_id != 0 {
                    self.seen_trade_order
                        .push_back((trade.exchange_ms, trade.trade_id));
                }
                self.trades.push(TradeRow {
                    timestamp_seconds,
                    price: self.instrument.price_from_units(trade.px),
                    size: self.instrument.size_from_units(trade.qty_units),
                    side: match trade.aggressor {
                        crate::types::AggressorSide::Buy => "buy",
                        crate::types::AggressorSide::Sell => "sell",
                    }
                    .to_owned(),
                    trade_id: trade.trade_id.to_string(),
                    exchange_timestamp: trade.exchange_ms as i64,
                });
            }
            MarketEvent::Book(book) => {
                self.books.push(BookRow {
                    timestamp_seconds,
                    sequence: book.exchange_ms as i64,
                    exchange_timestamp: book.exchange_ms as i64,
                    bids: book
                        .bids
                        .iter()
                        .map(|level| {
                            (
                                self.instrument.price_from_units(level.px),
                                self.instrument.size_from_units(level.qty_units),
                            )
                        })
                        .collect(),
                    asks: book
                        .asks
                        .iter()
                        .map(|level| {
                            (
                                self.instrument.price_from_units(level.px),
                                self.instrument.size_from_units(level.qty_units),
                            )
                        })
                        .collect(),
                });
            }
        }
        if local_ms.saturating_sub(self.last_flush_ms) >= self.flush_interval_ms {
            self.flush(local_ms)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn seen_trade_count(&self) -> usize {
        self.seen_trade_ids.len()
    }

    pub fn flush(&mut self, now_ms: u64) -> Result<()> {
        let base = self.root.join(&self.symbol);
        if !self.prices.is_empty() {
            let path = base.join("prices").join(format!("prices_{now_ms}.parquet"));
            write_price_shard(&path, &self.prices)?;
            self.created_price_shards.push(path);
            self.prices.clear();
        }
        if !self.trades.is_empty() {
            let path = base.join("trades").join(format!("trades_{now_ms}.parquet"));
            write_trade_shard(&path, &self.trades)?;
            self.created_trade_shards.push(path);
            self.trades.clear();
        }
        if !self.books.is_empty() {
            let path = base
                .join("orderbooks")
                .join(format!("orderbooks_{now_ms}.parquet"));
            write_book_shard(&path, &self.books, 20)?;
            self.created_book_shards.push(path);
            self.books.clear();
        }
        self.last_flush_ms = now_ms;
        Ok(())
    }

    pub fn prune(&self, now_ms: u64, retention_minutes: u64) -> Result<usize> {
        let cutoff = now_ms.saturating_sub(retention_minutes.saturating_mul(60_000));
        let mut removed = 0;
        for stream in ["prices", "trades", "orderbooks"] {
            let directory = self.root.join(&self.symbol).join(stream);
            if !directory.exists() {
                continue;
            }
            for entry in std::fs::read_dir(&directory)? {
                let path = entry?.path();
                if path.extension().and_then(|value| value.to_str()) != Some("parquet") {
                    continue;
                }
                if shard_timestamp(&path).is_some_and(|timestamp| timestamp < cutoff) {
                    std::fs::remove_file(&path)?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }

    /// Compact only shards created by this process. Legacy Python shards may
    /// use different Arrow encodings and remain the legacy collector's remit.
    pub fn compact(&mut self, now_ms: u64, compact_after_minutes: u64) -> Result<usize> {
        let cutoff = now_ms.saturating_sub(compact_after_minutes.saturating_mul(60_000));
        let base = self.root.join(&self.symbol);
        let mut compacted = 0;
        compacted += compact_created_shards(
            &mut self.created_price_shards,
            &base.join("prices"),
            "prices",
            cutoff,
        )?;
        compacted += compact_created_shards(
            &mut self.created_trade_shards,
            &base.join("trades"),
            "trades",
            cutoff,
        )?;
        compacted += compact_created_shards(
            &mut self.created_book_shards,
            &base.join("orderbooks"),
            "orderbooks",
            cutoff,
        )?;
        Ok(compacted)
    }
}

const RECORDER_QUEUE_CAPACITY: usize = 16_384;

enum RecorderCommand {
    Record(MarketEvent, u64),
    Maintain {
        now_ms: u64,
        compact_after_minutes: u64,
        retention_minutes: u64,
    },
    Finish {
        now_ms: u64,
        compact_after_minutes: u64,
        retention_minutes: u64,
        reply: SyncSender<std::result::Result<(usize, usize), String>>,
    },
    Shutdown {
        now_ms: u64,
    },
}

/// [`ParquetEventRecorder`] behind a dedicated writer thread.
///
/// `ParquetEventRecorder::record` performs a synchronous ZSTD Parquet write of
/// up to a whole flush interval of rows whenever the flush cadence fires, and
/// compaction rewrites entire shards — none of which belongs on the runtime
/// thread that drains the market-event ring. The strategy loop only sends;
/// buffering, flushing, compaction, and pruning all run here.
pub struct ParquetRecorderHandle {
    sender: SyncSender<RecorderCommand>,
    error: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl ParquetRecorderHandle {
    pub fn spawn(mut recorder: ParquetEventRecorder) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(RECORDER_QUEUE_CAPACITY);
        let error = Arc::new(Mutex::new(None::<String>));
        let thread_error = error.clone();
        let thread = thread::Builder::new()
            .name("parquet-recorder".to_owned())
            .spawn(move || {
                while let Ok(command) = receiver.recv() {
                    let result = match command {
                        RecorderCommand::Record(event, local_ms) => {
                            recorder.record(&event, local_ms)
                        }
                        RecorderCommand::Maintain {
                            now_ms,
                            compact_after_minutes,
                            retention_minutes,
                        } => {
                            recorder
                                .compact(now_ms, compact_after_minutes)
                                .and_then(|compacted| {
                                    let removed = recorder.prune(now_ms, retention_minutes)?;
                                    tracing::info!(
                                        compacted,
                                        removed,
                                        "Parquet maintenance complete"
                                    );
                                    Ok(())
                                })
                        }
                        RecorderCommand::Finish {
                            now_ms,
                            compact_after_minutes,
                            retention_minutes,
                            reply,
                        } => {
                            let result = recorder.flush(now_ms).and_then(|()| {
                                let compacted = recorder.compact(now_ms, compact_after_minutes)?;
                                let removed = recorder.prune(now_ms, retention_minutes)?;
                                Ok((compacted, removed))
                            });
                            let _ = reply.send(result.map_err(|error| error.to_string()));
                            return;
                        }
                        RecorderCommand::Shutdown { now_ms } => {
                            let _ = recorder.flush(now_ms);
                            return;
                        }
                    };
                    if let Err(error) = result {
                        if let Ok(mut destination) = thread_error.lock() {
                            *destination = Some(error.to_string());
                        }
                        return;
                    }
                }
            })?;
        Ok(Self {
            sender,
            error,
            thread: Some(thread),
        })
    }

    fn check_error(&self) -> Result<()> {
        let error = self
            .error
            .lock()
            .map_err(|_| anyhow::anyhow!("parquet recorder error lock poisoned"))?;
        if let Some(error) = error.as_deref() {
            bail!("parquet recorder failed: {error}");
        }
        Ok(())
    }

    /// Queue one market event for recording. A saturated queue is refused for
    /// the same reason a saturated event log is: silently dropping collector
    /// rows would corrupt the calibration evidence this data feeds.
    pub fn record(&self, event: &MarketEvent, local_ms: u64) -> Result<()> {
        self.check_error()?;
        match self
            .sender
            .try_send(RecorderCommand::Record(event.clone(), local_ms))
        {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                bail!("parquet recorder queue saturated; refusing an incomplete scientific run")
            }
            Err(TrySendError::Disconnected(_)) => {
                self.check_error()?;
                bail!("parquet recorder thread stopped unexpectedly")
            }
        }
    }

    /// Queue compaction and pruning. Skipped silently when the writer is busy;
    /// the next calibration interval retries.
    pub fn maintain(
        &self,
        now_ms: u64,
        compact_after_minutes: u64,
        retention_minutes: u64,
    ) -> Result<()> {
        self.check_error()?;
        match self.sender.try_send(RecorderCommand::Maintain {
            now_ms,
            compact_after_minutes,
            retention_minutes,
        }) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                self.check_error()?;
                bail!("parquet recorder thread stopped unexpectedly")
            }
        }
    }

    /// Flush, compact, and prune one final time, then join the writer.
    pub fn finish(
        mut self,
        now_ms: u64,
        compact_after_minutes: u64,
        retention_minutes: u64,
    ) -> Result<(usize, usize)> {
        self.check_error()?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.sender
            .send(RecorderCommand::Finish {
                now_ms,
                compact_after_minutes,
                retention_minutes,
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("parquet recorder stopped before final flush"))?;
        let result = reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("parquet recorder dropped its final acknowledgement"))?
            .map_err(anyhow::Error::msg);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        result
    }
}

impl Drop for ParquetRecorderHandle {
    fn drop(&mut self) {
        // Error paths that skip `finish` still get a best-effort final flush.
        let _ = self.sender.try_send(RecorderCommand::Shutdown {
            now_ms: crate::types::unix_ms(),
        });
        // Replace the sender so the channel disconnects: if the queue was full
        // above, the writer would otherwise drain it and then wait on `recv`
        // forever while we wait on `join`.
        let (disconnected, _) = mpsc::sync_channel(1);
        drop(std::mem::replace(&mut self.sender, disconnected));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn compact_created_shards(
    tracked: &mut Vec<PathBuf>,
    directory: &Path,
    prefix: &str,
    cutoff: u64,
) -> Result<usize> {
    let candidates: Vec<PathBuf> = tracked
        .iter()
        .filter(|path| shard_timestamp(path).is_some_and(|timestamp| timestamp < cutoff))
        .cloned()
        .collect();
    if candidates.len() < 2 {
        return Ok(0);
    }
    let newest = candidates
        .iter()
        .filter_map(|path| shard_timestamp(path))
        .max()
        .context("compact candidates have no timestamps")?;
    let mut batches = Vec::new();
    for path in &candidates {
        batches.extend(read_batches(path)?);
    }
    let schema = batches
        .first()
        .context("compact candidates contain no record batches")?
        .schema();
    if batches.iter().any(|batch| batch.schema() != schema) {
        bail!("cannot compact {prefix} shards with differing Arrow schemas");
    }
    let output = directory.join(format!("{prefix}_compact_{newest}.parquet"));
    write_batches_atomic(&output, schema, &batches)?;
    for path in &candidates {
        std::fs::remove_file(path)?;
    }
    tracked.retain(|path| !candidates.contains(path));
    tracked.push(output);
    Ok(candidates.len())
}

fn write_batches_atomic(path: &Path, schema: Arc<Schema>, batches: &[RecordBatch]) -> Result<()> {
    let parent = path.parent().context("Parquet output path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    let output = temporary.reopen()?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer = ArrowWriter::try_new(output, schema, Some(properties))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    temporary
        .persist(path)
        .map_err(|error| anyhow::anyhow!(error.error))?;
    Ok(())
}

/// Refuse to become a second writer when another collector is actively adding
/// shards but does not know about the Rust lock file (the legacy Python case).
pub async fn ensure_no_external_writer(
    data_root: &Path,
    symbol: &str,
    observation_seconds: u64,
) -> Result<()> {
    let before = newest_shard_timestamp(data_root, symbol)?;
    tokio::time::sleep(std::time::Duration::from_secs(observation_seconds.max(1))).await;
    let after = newest_shard_timestamp(data_root, symbol)?;
    if before.is_some() && after > before {
        bail!("another collector wrote {symbol} Parquet shards during writer preflight");
    }
    Ok(())
}

fn newest_shard_timestamp(data_root: &Path, symbol: &str) -> Result<Option<u64>> {
    let mut newest = None;
    for stream in ["prices", "trades", "orderbooks"] {
        let directory = data_root.join(symbol).join(stream);
        if !directory.exists() {
            continue;
        }
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path();
            if let Some(timestamp) = shard_timestamp(&path) {
                newest = Some(newest.map_or(timestamp, |current: u64| current.max(timestamp)));
            }
        }
    }
    Ok(newest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_round_trips_python_compatible_price_schema() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prices_1000.parquet");
        write_price_shard(
            &path,
            &[
                PriceRow {
                    timestamp_seconds: 1.0,
                    price: 100.0,
                    size: 3.0,
                    side: "bid".to_owned(),
                    exchange_timestamp: 1_000,
                },
                PriceRow {
                    timestamp_seconds: 1.0,
                    price: 101.0,
                    size: 4.0,
                    side: "ask".to_owned(),
                    exchange_timestamp: 1_000,
                },
            ],
        )
        .unwrap();
        let batches = read_batches(&path).unwrap();
        let rows = parse_price_batches(&batches).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].side, "bid");
    }

    #[test]
    fn collector_lock_is_exclusive() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("collector.lock");
        let _first = CollectorLock::acquire(&path, 60).unwrap();
        assert!(CollectorLock::acquire(&path, 60).is_err());
    }

    #[test]
    fn recorder_compacts_its_own_cold_shards() {
        let directory = tempfile::tempdir().unwrap();
        let instrument = InstrumentSpec {
            symbol: "SYN".to_owned(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 0,
            max_price_decimals: 2,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        };
        let mut recorder =
            ParquetEventRecorder::new(directory.path().to_owned(), instrument, 10, 180);
        for (timestamp, bid) in [(1_000, 10_000), (2_000, 10_001)] {
            recorder
                .record(
                    &MarketEvent::Bbo(crate::types::Bbo {
                        bid_px: bid,
                        bid_sz: 5,
                        ask_px: bid + 2,
                        ask_sz: 6,
                        exchange_ms: timestamp,
                        recv_ns: 0,
                    }),
                    timestamp,
                )
                .unwrap();
            recorder.flush(timestamp).unwrap();
        }
        assert_eq!(recorder.compact(1_000_000, 0).unwrap(), 2);
        let files = std::fs::read_dir(directory.path().join("SYN/prices"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 1);
        assert!(files[0].file_name().to_string_lossy().contains("compact"));
    }

    #[test]
    fn retention_prunes_only_expired_parquet_shards() {
        let directory = tempfile::tempdir().unwrap();
        let instrument = InstrumentSpec {
            symbol: "SYN".to_owned(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 0,
            max_price_decimals: 2,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        };
        let recorder = ParquetEventRecorder::new(directory.path().to_owned(), instrument, 10, 180);
        let stream = directory.path().join("SYN/prices");
        std::fs::create_dir_all(&stream).unwrap();
        std::fs::write(stream.join("prices_1000.parquet"), b"old").unwrap();
        std::fs::write(stream.join("prices_999000.parquet"), b"fresh").unwrap();
        std::fs::write(stream.join("notes.txt"), b"keep").unwrap();
        assert_eq!(recorder.prune(1_000_000, 1).unwrap(), 1);
        assert!(!stream.join("prices_1000.parquet").exists());
        assert!(stream.join("prices_999000.parquet").exists());
        assert!(stream.join("notes.txt").exists());
    }

    #[test]
    fn recorder_trade_dedup_is_bounded_by_retention_horizon() {
        let directory = tempfile::tempdir().unwrap();
        let instrument = InstrumentSpec {
            symbol: "SYN".to_owned(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 0,
            max_price_decimals: 2,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        };
        let mut recorder =
            ParquetEventRecorder::new(directory.path().to_owned(), instrument, 10_000, 1);
        let event = |trade_id, exchange_ms| {
            MarketEvent::Trade(crate::types::TradePrint {
                aggressor: crate::types::AggressorSide::Buy,
                px: 10_000,
                qty_units: 1,
                exchange_ms,
                recv_ns: exchange_ms * 1_000_000,
                trade_id,
            })
        };
        recorder.record(&event(1, 1_000), 1_000).unwrap();
        recorder.record(&event(2, 2_000), 2_000).unwrap();
        assert_eq!(recorder.seen_trade_count(), 2);
        recorder.record(&event(3, 70_000), 70_000).unwrap();
        assert_eq!(recorder.seen_trade_count(), 1);
        // ID 1's shard is now outside retention, so admitting it again cannot
        // duplicate any data that the recorder still promises to retain.
        recorder.record(&event(1, 71_000), 71_000).unwrap();
        recorder.record(&event(3, 72_000), 72_000).unwrap();
        assert_eq!(recorder.seen_trade_count(), 2);
    }
}
