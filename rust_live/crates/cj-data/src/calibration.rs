use crate::config::CalibrationConfig;
use crate::hjb::CjParameters;
use crate::parquet_io::{MarketDataSet, MidRecord, TimeSource, TradeRecord};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Schema v5: `lambda_±` is the per-side market-order rate scaled by that
/// side's survival-fit intercept `A` (see `ESTIMATOR_SEMANTICS`). v4 fed the
/// raw rate to the HJB while the fitted fill probability was `A·exp(-κδ)`, so
/// fill intensity was off by `A`; v4 snapshots are refused by `is_quotable`.
pub const PARAMETER_SCHEMA_VERSION: u32 = 5;
/// Direct-window semantics, revision 2: no cross-window smoothing, and the
/// arrival rate the HJB sees is `lambda_raw · survival_intercept`, so that
/// `lambda · exp(-κδ)` is exactly the measured fill intensity at every depth
/// inside the fit support. The raw rate is kept as `lambda_raw` per side.
pub const ESTIMATOR_SEMANTICS: &str = "direct_window_v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    Ok,
    InsufficientData,
    PoorFit,
    Toxic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SideFitDiagnostics {
    pub r_squared: Option<f64>,
    pub n_points: usize,
    pub depth_p95: Option<f64>,
    pub depth_min_fitted: Option<f64>,
    pub depth_max_fitted: Option<f64>,
    pub support_depth_lower: Option<f64>,
    pub support_depth_upper: Option<f64>,
    pub survival_intercept: Option<f64>,
    /// Market orders per covered second on this side before the intercept
    /// scaling; `parameters.lambda_±` is this times `survival_intercept`.
    #[serde(default)]
    pub lambda_raw: Option<f64>,
    pub negative_depths_truncated: usize,
    pub market_orders: usize,
    pub epsilon_events: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDiagnostics {
    pub time_source: TimeSource,
    pub window_start_ms: f64,
    pub window_end_ms: f64,
    pub observed_seconds: f64,
    pub outage_seconds: f64,
    pub outage_count: usize,
    pub mid_updates: usize,
    pub trade_prints: usize,
    pub duplicate_trade_ids_dropped: usize,
    pub skipped_without_pre_mid: usize,
    pub plus: SideFitDiagnostics,
    pub minus: SideFitDiagnostics,
    pub toxicity_plus: Option<f64>,
    pub toxicity_minus: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSnapshot {
    pub schema_version: u32,
    pub symbol: String,
    pub revision: u64,
    pub generated_at_ms: u64,
    pub generated_at: String,
    pub status: CalibrationStatus,
    pub lambda_source: String,
    pub estimator_semantics: String,
    pub parameters: CjParameters,
    pub diagnostics: CalibrationDiagnostics,
    pub fingerprint: String,
}

impl CalibrationSnapshot {
    pub fn is_quotable(&self) -> bool {
        self.schema_version == PARAMETER_SCHEMA_VERSION
            && self.status == CalibrationStatus::Ok
            && self.lambda_source == "mo_survival_fit"
            && self.estimator_semantics == ESTIMATOR_SEMANTICS
            && self.parameters.validate().is_ok()
    }

    pub fn is_fresh(
        &self,
        now_ms: u64,
        max_age_seconds: u64,
        max_future_skew_seconds: u64,
    ) -> bool {
        let future_limit = now_ms.saturating_add(max_future_skew_seconds.saturating_mul(1_000));
        self.generated_at_ms <= future_limit
            && now_ms.saturating_sub(self.generated_at_ms) <= max_age_seconds.saturating_mul(1_000)
    }

    pub fn write_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary =
            tempfile::NamedTempFile::new_in(path.parent().unwrap_or_else(|| Path::new(".")))?;
        let payload = self.canonical_payload()?;
        let fingerprint = hex::encode(Sha256::digest(&payload));
        if fingerprint != self.fingerprint {
            bail!("calibration fingerprint is stale before persistence");
        }
        let raw = serde_json::value::RawValue::from_string(String::from_utf8(payload)?)?;
        serde_json::to_writer_pretty(
            temporary.as_file(),
            &CalibrationEnvelope {
                fingerprint: &fingerprint,
                snapshot: &raw,
            },
        )?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| anyhow::anyhow!(error.error))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("cannot read calibration snapshot {}", path.display()))?;
        let envelope: CalibrationEnvelopeOwned = serde_json::from_slice(&bytes)?;
        let expected = envelope.fingerprint;
        let computed = hex::encode(Sha256::digest(envelope.snapshot.get().as_bytes()));
        if expected != computed {
            bail!("calibration snapshot fingerprint mismatch: expected {expected}, computed {computed}");
        }
        let mut snapshot: Self = serde_json::from_str(envelope.snapshot.get())?;
        snapshot.fingerprint = expected;
        Ok(snapshot)
    }

    fn compute_fingerprint(&self) -> Result<String> {
        Ok(hex::encode(Sha256::digest(self.canonical_payload()?)))
    }

    fn canonical_payload(&self) -> Result<Vec<u8>> {
        let mut copy = self.clone();
        copy.fingerprint.clear();
        Ok(serde_json::to_vec_pretty(&copy)?)
    }
}

#[derive(Serialize)]
struct CalibrationEnvelope<'a> {
    fingerprint: &'a str,
    snapshot: &'a serde_json::value::RawValue,
}

#[derive(Deserialize)]
struct CalibrationEnvelopeOwned {
    fingerprint: String,
    snapshot: Box<serde_json::value::RawValue>,
}

#[derive(Debug, Clone)]
pub struct Calibrator {
    symbol: String,
    config: CalibrationConfig,
}

impl Calibrator {
    pub fn new(symbol: impl Into<String>, config: CalibrationConfig) -> Self {
        Self {
            symbol: symbol.into(),
            config,
        }
    }

    pub fn calibrate(&self, data: &MarketDataSet) -> Result<CalibrationSnapshot> {
        if data.symbol != self.symbol {
            bail!(
                "dataset symbol {} does not match {}",
                data.symbol,
                self.symbol
            );
        }
        if data.mids.is_empty() {
            bail!("calibration requires mid data");
        }

        let market_orders = aggregate_market_orders(&data.trades);
        let attached = attach_pre_mid(&market_orders, &data.mids);
        let mut buy_depths = Vec::new();
        let mut sell_depths = Vec::new();
        let mut skipped_without_pre_mid = 0;
        let mut negative_plus = 0;
        let mut negative_minus = 0;
        for order in &attached {
            let Some(pre_mid) = order.pre_mid else {
                skipped_without_pre_mid += 1;
                continue;
            };
            if order.side == MarketOrderSide::Buy {
                let raw = order.price_extreme - pre_mid;
                negative_plus += usize::from(raw < 0.0);
                buy_depths.push(raw.max(0.0));
            } else {
                let raw = pre_mid - order.price_extreme;
                negative_minus += usize::from(raw < 0.0);
                sell_depths.push(raw.max(0.0));
            }
        }

        let plus_fit = fit_kappa_survival(
            &buy_depths,
            self.config.support_quantile_lower_plus,
            self.config.support_quantile_upper_plus,
        )?;
        let minus_fit = fit_kappa_survival(
            &sell_depths,
            self.config.support_quantile_lower_minus,
            self.config.support_quantile_upper_minus,
        )?;
        let (outage_count, outage_seconds) = outage_seconds(
            &data.mids,
            &data.trades,
            data.window_start_ms,
            data.window_end_ms,
            self.config.outage_threshold_seconds,
        );
        let first_mid = data
            .mids
            .first()
            .map_or(data.window_start_ms, |row| row.ts_ms);
        let covered_start = data.window_start_ms.max(first_mid);
        let wall_seconds = ((data.window_end_ms - covered_start) / 1_000.0).max(1.0e-6);
        let covered_seconds = (wall_seconds - outage_seconds).max(1.0e-6);
        let plus_count = attached
            .iter()
            .filter(|order| order.side == MarketOrderSide::Buy)
            .count();
        let minus_count = attached.len() - plus_count;
        let lambda_plus_raw = plus_count as f64 / covered_seconds;
        let lambda_minus_raw = minus_count as f64 / covered_seconds;
        // The fit says P(depth >= δ) = A·exp(-κδ) over its support; the HJB
        // models fill intensity as λ·exp(-κδ). Hand it λ_raw·A so the two
        // agree at every depth, instead of being off by A everywhere.
        let lambda_plus = lambda_plus_raw * plus_fit.survival_intercept.unwrap_or(1.0);
        let lambda_minus = lambda_minus_raw * minus_fit.survival_intercept.unwrap_or(1.0);

        let (epsilon_plus, plus_epsilon_events) = estimate_epsilon(
            &attached,
            &data.mids,
            data.window_end_ms,
            self.config.epsilon_horizon_ms_plus,
            MarketOrderSide::Buy,
        );
        let (epsilon_minus, minus_epsilon_events) = estimate_epsilon(
            &attached,
            &data.mids,
            data.window_end_ms,
            self.config.epsilon_horizon_ms_minus,
            MarketOrderSide::Sell,
        );
        let sigma2 = realized_sigma2_per_second(&data.mids, 1.0, 5.0, 60);
        let parameters = CjParameters {
            lambda_plus,
            lambda_minus,
            kappa_plus: plus_fit.kappa.unwrap_or(0.0),
            kappa_minus: minus_fit.kappa.unwrap_or(0.0),
            epsilon_plus: epsilon_plus.unwrap_or(0.0),
            epsilon_minus: epsilon_minus.unwrap_or(0.0),
            sigma2_per_second: sigma2,
        };

        let insufficient = parameters.validate().is_err()
            || plus_count == 0
            || minus_count == 0
            || plus_fit.n_points < self.config.min_kappa_fit_points
            || minus_fit.n_points < self.config.min_kappa_fit_points
            || plus_epsilon_events < self.config.min_epsilon_events
            || minus_epsilon_events < self.config.min_epsilon_events;
        let poor_fit = !insufficient
            && (plus_fit
                .r_squared
                .is_none_or(|value| value < self.config.min_kappa_r2)
                || minus_fit
                    .r_squared
                    .is_none_or(|value| value < self.config.min_kappa_r2));
        let toxicity_plus = plus_fit.kappa.zip(epsilon_plus).map(|(k, e)| k * e);
        let toxicity_minus = minus_fit.kappa.zip(epsilon_minus).map(|(k, e)| k * e);
        let toxic = !insufficient
            && !poor_fit
            && toxicity_plus
                .into_iter()
                .chain(toxicity_minus)
                .any(|value| value > self.config.max_toxicity);
        let status = if insufficient {
            CalibrationStatus::InsufficientData
        } else if poor_fit {
            CalibrationStatus::PoorFit
        } else if toxic {
            CalibrationStatus::Toxic
        } else {
            CalibrationStatus::Ok
        };

        let generated_at_ms = crate::types::unix_ms();
        let generated_at = DateTime::<Utc>::from_timestamp_millis(generated_at_ms as i64)
            .unwrap_or_else(Utc::now)
            .to_rfc3339_opts(SecondsFormat::Secs, true);
        let diagnostics = CalibrationDiagnostics {
            time_source: data.time_source,
            window_start_ms: data.window_start_ms,
            window_end_ms: data.window_end_ms,
            observed_seconds: covered_seconds,
            outage_seconds,
            outage_count,
            mid_updates: data.mids.len(),
            trade_prints: data.trades.len(),
            duplicate_trade_ids_dropped: data.duplicate_trade_ids_dropped,
            skipped_without_pre_mid,
            plus: SideFitDiagnostics {
                r_squared: plus_fit.r_squared,
                n_points: plus_fit.n_points,
                depth_p95: plus_fit.depth_p95,
                depth_min_fitted: plus_fit.depth_min_fitted,
                depth_max_fitted: plus_fit.depth_max_fitted,
                support_depth_lower: plus_fit.support_depth_lower,
                support_depth_upper: plus_fit.support_depth_upper,
                survival_intercept: plus_fit.survival_intercept,
                lambda_raw: Some(lambda_plus_raw),
                negative_depths_truncated: negative_plus,
                market_orders: plus_count,
                epsilon_events: plus_epsilon_events,
            },
            minus: SideFitDiagnostics {
                r_squared: minus_fit.r_squared,
                n_points: minus_fit.n_points,
                depth_p95: minus_fit.depth_p95,
                depth_min_fitted: minus_fit.depth_min_fitted,
                depth_max_fitted: minus_fit.depth_max_fitted,
                support_depth_lower: minus_fit.support_depth_lower,
                support_depth_upper: minus_fit.support_depth_upper,
                survival_intercept: minus_fit.survival_intercept,
                lambda_raw: Some(lambda_minus_raw),
                negative_depths_truncated: negative_minus,
                market_orders: minus_count,
                epsilon_events: minus_epsilon_events,
            },
            toxicity_plus,
            toxicity_minus,
        };
        let mut snapshot = CalibrationSnapshot {
            schema_version: PARAMETER_SCHEMA_VERSION,
            symbol: self.symbol.clone(),
            revision: generated_at_ms,
            generated_at_ms,
            generated_at,
            status,
            lambda_source: "mo_survival_fit".to_owned(),
            estimator_semantics: ESTIMATOR_SEMANTICS.to_owned(),
            parameters,
            diagnostics,
            fingerprint: String::new(),
        };
        snapshot.fingerprint = snapshot.compute_fingerprint()?;
        Ok(snapshot)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketOrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
struct MarketOrder {
    ts_ms: f64,
    side: MarketOrderSide,
    price_extreme: f64,
    pre_mid: Option<f64>,
}

fn aggregate_market_orders(trades: &[TradeRecord]) -> Vec<MarketOrder> {
    let mut rows = trades.to_vec();
    rows.sort_by(|a, b| {
        a.ts_ms
            .total_cmp(&b.ts_ms)
            .then_with(|| a.side.cmp(&b.side))
    });
    let mut output = Vec::new();
    let mut index = 0;
    while index < rows.len() {
        let ts_ms = rows[index].ts_ms;
        let side_text = rows[index].side.as_str();
        let side = if side_text == "buy" {
            MarketOrderSide::Buy
        } else {
            MarketOrderSide::Sell
        };
        let mut extreme = rows[index].price;
        let mut next = index;
        while next < rows.len() && rows[next].ts_ms == ts_ms && rows[next].side == side_text {
            extreme = match side {
                MarketOrderSide::Buy => extreme.max(rows[next].price),
                MarketOrderSide::Sell => extreme.min(rows[next].price),
            };
            next += 1;
        }
        output.push(MarketOrder {
            ts_ms,
            side,
            price_extreme: extreme,
            pre_mid: None,
        });
        index = next;
    }
    output.sort_by(|a, b| a.ts_ms.total_cmp(&b.ts_ms));
    output
}

fn attach_pre_mid(orders: &[MarketOrder], mids: &[MidRecord]) -> Vec<MarketOrder> {
    orders
        .iter()
        .cloned()
        .map(|mut order| {
            let position = mids.partition_point(|mid| mid.ts_ms < order.ts_ms);
            order.pre_mid = position.checked_sub(1).map(|index| mids[index].mid);
            order
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
struct FitResult {
    kappa: Option<f64>,
    r_squared: Option<f64>,
    n_points: usize,
    depth_p95: Option<f64>,
    depth_min_fitted: Option<f64>,
    depth_max_fitted: Option<f64>,
    support_depth_lower: Option<f64>,
    support_depth_upper: Option<f64>,
    survival_intercept: Option<f64>,
}

fn fit_kappa_survival(depths: &[f64], lower_q: f64, upper_q: f64) -> Result<FitResult> {
    if !(0.0 <= lower_q && lower_q < upper_q && upper_q <= 1.0) {
        bail!("invalid kappa support [{lower_q}, {upper_q}]");
    }
    let mut sorted: Vec<f64> = depths
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() < 10 {
        return Ok(FitResult::default());
    }
    let depth_p95 = percentile(&sorted, 0.95);
    let support_upper = percentile(&sorted, upper_q);
    let support_lower = (lower_q > 0.0).then(|| percentile(&sorted, lower_q));
    let mut grid = Vec::new();
    for depth in &sorted {
        if *depth > support_upper || support_lower.is_some_and(|floor| *depth < floor) {
            continue;
        }
        if grid.last().is_none_or(|previous| *previous != *depth) {
            grid.push(*depth);
        }
    }
    if grid.len() < 3 {
        return Ok(FitResult {
            depth_p95: Some(depth_p95),
            support_depth_lower: support_lower,
            support_depth_upper: Some(support_upper),
            ..FitResult::default()
        });
    }
    let mut points = Vec::new();
    for depth in grid {
        let tail = sorted.len() - sorted.partition_point(|candidate| *candidate < depth);
        if tail >= 2 {
            let survival = tail as f64 / sorted.len() as f64;
            points.push((depth, survival.ln(), tail as f64));
        }
    }
    if points.len() < 3 {
        return Ok(FitResult::default());
    }

    // Weighted least squares for y = intercept - kappa*depth, with objective
    // weights equal to tail count (the Python design uses sqrt(tail_count)).
    let sw: f64 = points.iter().map(|point| point.2).sum();
    let sx: f64 = points.iter().map(|point| point.2 * -point.0).sum();
    let sy: f64 = points.iter().map(|point| point.2 * point.1).sum();
    let sxx: f64 = points.iter().map(|point| point.2 * point.0.powi(2)).sum();
    let sxy: f64 = points
        .iter()
        .map(|point| point.2 * -point.0 * point.1)
        .sum();
    let denominator = sw * sxx - sx * sx;
    if denominator.abs() < 1.0e-300 {
        return Ok(FitResult::default());
    }
    let intercept = (sy * sxx - sx * sxy) / denominator;
    let kappa = ((sw * sxy - sx * sy) / denominator).max(0.0);
    let y_mean = sy / sw;
    let ss_res: f64 = points
        .iter()
        .map(|point| {
            let predicted = intercept - kappa * point.0;
            point.2 * (point.1 - predicted).powi(2)
        })
        .sum();
    let ss_tot: f64 = points
        .iter()
        .map(|point| point.2 * (point.1 - y_mean).powi(2))
        .sum();
    let r_squared = (ss_tot > 0.0).then(|| 1.0 - ss_res / ss_tot);
    Ok(FitResult {
        kappa: Some(kappa),
        r_squared,
        n_points: points.len(),
        depth_p95: Some(depth_p95),
        depth_min_fitted: points.first().map(|point| point.0),
        depth_max_fitted: points.last().map(|point| point.0),
        support_depth_lower: support_lower,
        support_depth_upper: Some(support_upper),
        survival_intercept: Some(intercept.exp()),
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    if low == high {
        sorted[low]
    } else {
        sorted[low] + (rank - low as f64) * (sorted[high] - sorted[low])
    }
}

fn estimate_epsilon(
    orders: &[MarketOrder],
    mids: &[MidRecord],
    window_end_ms: f64,
    horizon_ms: u64,
    side: MarketOrderSide,
) -> (Option<f64>, usize) {
    let mut impacts = Vec::new();
    for order in orders.iter().filter(|order| order.side == side) {
        let Some(pre_mid) = order.pre_mid else {
            continue;
        };
        let target = order.ts_ms + horizon_ms as f64;
        if target > window_end_ms {
            continue;
        }
        let position = mids.partition_point(|mid| mid.ts_ms <= target);
        let Some(index) = position.checked_sub(1) else {
            continue;
        };
        let post_mid = mids[index].mid;
        let impact = match side {
            MarketOrderSide::Buy => post_mid - pre_mid,
            MarketOrderSide::Sell => pre_mid - post_mid,
        };
        if impact.is_finite() {
            impacts.push(impact);
        }
    }
    if impacts.is_empty() {
        return (None, 0);
    }
    let mean = impacts.iter().sum::<f64>() / impacts.len() as f64;
    let variance = impacts
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / impacts.len() as f64;
    let standard_deviation = variance.sqrt();
    let clean: Vec<f64> = if standard_deviation > 0.0 {
        impacts
            .into_iter()
            .filter(|value| (*value - mean).abs() < 3.0 * standard_deviation)
            .collect()
    } else {
        impacts
    };
    if clean.is_empty() {
        return (None, 0);
    }
    let epsilon = (clean.iter().sum::<f64>() / clean.len() as f64).max(0.0);
    (Some(epsilon), clean.len())
}

fn outage_seconds(
    mids: &[MidRecord],
    trades: &[TradeRecord],
    start_ms: f64,
    end_ms: f64,
    threshold_seconds: f64,
) -> (usize, f64) {
    let mut timestamps: Vec<f64> = mids
        .iter()
        .map(|row| row.ts_ms)
        .chain(trades.iter().map(|row| row.ts_ms))
        .filter(|value| *value >= start_ms && *value <= end_ms)
        .collect();
    timestamps.sort_by(f64::total_cmp);
    let mut count = 0;
    let mut total = 0.0;
    for pair in timestamps.windows(2) {
        let seconds = (pair[1] - pair[0]) / 1_000.0;
        if seconds > threshold_seconds {
            count += 1;
            total += seconds;
        }
    }
    (count, total)
}

fn realized_sigma2_per_second(
    mids: &[MidRecord],
    sample_seconds: f64,
    max_gap_seconds: f64,
    min_samples: usize,
) -> Option<f64> {
    if mids.is_empty() {
        return None;
    }
    let first = mids.first()?.ts_ms / 1_000.0;
    let last = mids.last()?.ts_ms / 1_000.0;
    let step = sample_seconds.max(1.0e-3);
    let mut grid = (first / step).ceil() * step;
    let mut samples = Vec::new();
    while grid <= last + 1.0e-9 {
        let target_ms = grid * 1_000.0;
        let position = mids.partition_point(|mid| mid.ts_ms <= target_ms);
        if let Some(index) = position.checked_sub(1) {
            let staleness = grid - mids[index].ts_ms / 1_000.0;
            samples.push((staleness <= max_gap_seconds).then_some(mids[index].mid));
        } else {
            samples.push(None);
        }
        grid += step;
    }
    let diffs: Vec<f64> = samples
        .windows(2)
        .filter_map(|pair| Some(pair[1]? - pair[0]?))
        .collect();
    if diffs.len() < min_samples {
        return None;
    }
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let variance = diffs
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / diffs.len() as f64;
    (variance.is_finite() && variance >= 0.0).then_some(variance / step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_io::{ShardStats, TimeSource};
    use approx::assert_relative_eq;

    #[test]
    fn weighted_survival_recovers_exponential_slope() {
        let depths: Vec<f64> = (1..=1_000)
            .map(|index| -(1.0 - (index as f64 - 0.5) / 1_000.0).ln() / 10.0)
            .collect();
        let fit = fit_kappa_survival(&depths, 0.0, 0.99).unwrap();
        assert_relative_eq!(fit.kappa.unwrap(), 10.0, epsilon = 0.2);
        assert!(fit.r_squared.unwrap() > 0.99);
    }

    #[test]
    fn synthetic_two_sided_window_calibrates() {
        let mut mids = Vec::new();
        let mut trades = Vec::new();
        for index in 0..1_000 {
            let ts = index as f64 * 1_000.0;
            let mid = 100.0 + (index as f64 / 100.0).sin() * 0.01;
            mids.push(MidRecord {
                ts_ms: ts,
                bid: mid - 0.01,
                ask: mid + 0.01,
                mid,
            });
            if index > 0 {
                let buy = index % 2 == 0;
                let depth = 0.001 * (1 + index % 20) as f64;
                trades.push(TradeRecord {
                    ts_ms: ts + 100.0,
                    side: if buy { "buy" } else { "sell" }.to_owned(),
                    price: if buy { mid + depth } else { mid - depth },
                    size: 1.0,
                    trade_id: Some(index.to_string()),
                });
            }
        }
        let data = MarketDataSet {
            symbol: "SYN".to_owned(),
            time_source: TimeSource::Exchange,
            mids,
            trades,
            books: Vec::new(),
            window_start_ms: 0.0,
            window_end_ms: 999_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let mut config = CalibrationConfig::default();
        config.min_kappa_r2 = -1.0;
        config.max_toxicity = 100.0;
        let snapshot = Calibrator::new("SYN", config).calibrate(&data).unwrap();
        assert_eq!(snapshot.status, CalibrationStatus::Ok);
        assert!(snapshot.parameters.validate().is_ok());
    }

    #[test]
    fn insufficient_window_is_fail_closed_and_json_serializable() {
        let data = MarketDataSet {
            symbol: "SYN".to_owned(),
            time_source: TimeSource::Exchange,
            mids: vec![MidRecord {
                ts_ms: 0.0,
                bid: 99.0,
                ask: 101.0,
                mid: 100.0,
            }],
            trades: Vec::new(),
            books: Vec::new(),
            window_start_ms: 0.0,
            window_end_ms: 1_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let snapshot = Calibrator::new("SYN", CalibrationConfig::default())
            .calibrate(&data)
            .unwrap();
        assert_eq!(snapshot.status, CalibrationStatus::InsufficientData);
        assert!(!snapshot.is_quotable());
        assert!(serde_json::to_vec(&snapshot).is_ok());
    }

    #[test]
    fn snapshot_fingerprint_survives_json_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("snapshot.json");
        let mut config = CalibrationConfig::default();
        config.min_kappa_r2 = -1.0;
        config.max_toxicity = 100.0;
        let mut mids = Vec::new();
        let mut trades = Vec::new();
        for index in 0..1_000 {
            let ts_ms = f64::from(index) * 1_000.0;
            let mid = 100.0 + (f64::from(index) / 100.0).sin() * 0.01;
            mids.push(MidRecord {
                ts_ms,
                bid: mid - 0.01,
                ask: mid + 0.01,
                mid,
            });
            if index > 0 {
                let buy = index % 2 == 0;
                let depth = 0.001 * f64::from(1 + index % 20);
                trades.push(TradeRecord {
                    ts_ms: ts_ms + 100.0,
                    side: if buy { "buy" } else { "sell" }.to_owned(),
                    price: if buy { mid + depth } else { mid - depth },
                    size: 1.0,
                    trade_id: Some(index.to_string()),
                });
            }
        }
        let data = MarketDataSet {
            symbol: "SYN".to_owned(),
            time_source: TimeSource::Exchange,
            mids,
            trades,
            books: Vec::new(),
            window_start_ms: 0.0,
            window_end_ms: 999_000.0,
            duplicate_trade_ids_dropped: 0,
            price_shards: ShardStats::default(),
            trade_shards: ShardStats::default(),
            orderbook_shards: ShardStats::default(),
        };
        let snapshot = Calibrator::new("SYN", config).calibrate(&data).unwrap();
        snapshot.write_atomic(&path).unwrap();
        let restored = CalibrationSnapshot::load(&path).unwrap();
        assert_eq!(restored.fingerprint, snapshot.fingerprint);
    }
}
