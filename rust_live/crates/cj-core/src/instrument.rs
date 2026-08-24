use crate::config::QuotingConfig;
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
    ///
    /// The price decade is computed in integer arithmetic:
    /// `floor(log10(units * 10^-d)) = ilog10(units) - d` exactly, with no
    /// float `log10` on the quote path and no rounding surprises at decade
    /// boundaries.
    #[inline]
    pub fn price_quantum(&self, px_units: i64) -> i64 {
        if px_units == 0 {
            return 1;
        }
        let decade = px_units.unsigned_abs().ilog10() as i32 - self.max_price_decimals as i32;
        let decimal_places = (self.max_significant_figures as i32 - 1 - decade)
            .clamp(0, self.max_price_decimals as i32) as u32;
        10_i64.pow(self.max_price_decimals - decimal_places)
    }

    /// How far a target price may drift before a resting order is replaced,
    /// in price units: the larger of the tick threshold and the price-relative
    /// bps threshold.
    ///
    /// Shared by the live backend and the simulator deliberately. It lived as
    /// two byte-identical copies, one in each, with a comment on both saying
    /// they had to match -- the same arrangement that let
    /// `min_order_lifetime_ms` be honored live and silently ignored in every
    /// replay. One implementation cannot drift from itself.
    ///
    /// The zero point is `ticks = 1, bps = 0`, which reproduces exact-price
    /// requoting.
    #[inline]
    pub fn requote_hold_window_units(&self, quoting: &QuotingConfig, target_px_units: i64) -> i64 {
        let ticks = quoting
            .replace_threshold_ticks
            .max(0)
            .saturating_mul(self.price_quantum(target_px_units));
        let bps =
            (quoting.replace_threshold_bps.max(0.0) / 10_000.0 * target_px_units as f64) as i64;
        ticks.max(bps)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Moved here from the live backend when the helper was deduped: one
    /// implementation, one test.
    #[test]
    fn requote_hold_window_is_max_of_ticks_and_bps() {
        let instrument = valid();
        let mut quoting = QuotingConfig::default();
        quoting.replace_threshold_ticks = 1;
        quoting.replace_threshold_bps = 2.0;
        // 2 bps of 100_015 is 20 units; one tick is 10. The bps term wins.
        assert_eq!(instrument.requote_hold_window_units(&quoting, 100_015), 20);
        quoting.replace_threshold_bps = 0.0;
        assert_eq!(instrument.requote_hold_window_units(&quoting, 100_015), 10);
        quoting.replace_threshold_ticks = 0;
        assert_eq!(instrument.requote_hold_window_units(&quoting, 100_015), 0);
    }

    fn valid() -> InstrumentSpec {
        InstrumentSpec {
            symbol: "SYN".to_owned(),
            dex: String::new(),
            asset_id: 0,
            sz_decimals: 2,
            max_price_decimals: 6,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 1.0,
            margin_table_id: 0,
            only_isolated: false,
            margin_mode: String::new(),
            is_delisted: false,
            metadata_fingerprint: String::new(),
        }
    }

    #[test]
    fn invalid_metadata_and_numeric_inputs_fail_closed() {
        let mut instrument = valid();
        assert!(instrument.validate().is_ok());
        instrument.symbol.clear();
        assert!(instrument.validate().is_err());
        instrument = valid();
        instrument.max_significant_figures = 0;
        assert!(instrument.validate().is_err());
        instrument = valid();
        instrument.only_isolated = true;
        assert!(instrument.validate().is_err());
        let instrument = valid();
        assert!(instrument.price_to_units(f64::NAN).is_err());
        assert!(instrument.price_to_units(0.0).is_err());
        assert!(instrument.size_to_units(-1.0).is_err());
        assert_eq!(instrument.price_quantum(0), 1);
    }
}
