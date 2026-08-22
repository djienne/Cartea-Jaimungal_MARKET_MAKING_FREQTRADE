use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Venue metadata needed by the model and execution adapters.
///
/// The engine is generic over this description. Scientific validation is a
/// separate property carried by the selected profile, not by these mechanics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentSpec {
    pub symbol: String,
    #[serde(default)]
    pub dex: String,
    pub asset_id: u32,
    pub sz_decimals: u32,
    pub max_price_decimals: u32,
    pub max_significant_figures: u32,
    pub max_leverage: f64,
    pub minimum_notional: f64,
    #[serde(default)]
    pub margin_table_id: u32,
    #[serde(default)]
    pub only_isolated: bool,
    #[serde(default)]
    pub margin_mode: String,
    #[serde(default)]
    pub is_delisted: bool,
    #[serde(default)]
    pub metadata_fingerprint: String,
}

impl InstrumentSpec {
    pub fn validate(&self) -> Result<()> {
        if self.symbol.trim().is_empty() {
            bail!("instrument symbol must not be empty");
        }
        if self.sz_decimals > 12 || self.max_price_decimals > 12 {
            bail!("instrument decimal counts must be <= 12");
        }
        if self.max_significant_figures == 0 || self.max_significant_figures > 10 {
            bail!("max_significant_figures must be in 1..=10");
        }
        if !self.max_leverage.is_finite() || self.max_leverage <= 0.0 {
            bail!("max_leverage must be finite and positive");
        }
        if !self.minimum_notional.is_finite() || self.minimum_notional < 0.0 {
            bail!("minimum_notional must be finite and non-negative");
        }
        if self.only_isolated && self.margin_mode.is_empty() {
            bail!("isolated instruments must report a margin mode");
        }
        Ok(())
    }

    #[inline]
    pub const fn price_scale(&self) -> i64 {
        10_i64.pow(self.max_price_decimals)
    }

    #[inline]
    pub const fn size_scale(&self) -> i64 {
        10_i64.pow(self.sz_decimals)
    }

    /// Hyperliquid accepts at most `max_significant_figures`, while integer
    /// prices remain valid irrespective of their digit count.
    #[inline]
    pub fn price_quantum(&self, px_units: i64) -> i64 {
        if px_units == 0 {
            return 1;
        }
        let px = px_units.unsigned_abs() as f64 / self.price_scale() as f64;
        let decimal_places = (self.max_significant_figures as i32 - 1 - px.log10().floor() as i32)
            .clamp(0, self.max_price_decimals as i32) as u32;
        10_i64.pow(self.max_price_decimals - decimal_places)
    }

    #[inline]
    pub fn price_from_units(&self, value: i64) -> f64 {
        value as f64 / self.price_scale() as f64
    }

    #[inline]
    pub fn size_from_units(&self, value: i64) -> f64 {
        value as f64 / self.size_scale() as f64
    }

    pub fn price_to_units(&self, value: f64) -> Result<i64> {
        if !value.is_finite() || value <= 0.0 {
            bail!("price must be finite and positive");
        }
        Ok((value * self.price_scale() as f64).round() as i64)
    }

    pub fn size_to_units(&self, value: f64) -> Result<i64> {
        if !value.is_finite() || value < 0.0 {
            bail!("size must be finite and non-negative");
        }
        Ok((value * self.size_scale() as f64).round() as i64)
    }
}
