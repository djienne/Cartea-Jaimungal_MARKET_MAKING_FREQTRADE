use crate::config::{QuotingConfig, RiskConfig};
use crate::hjb::HjbSurface;
use crate::instrument::InstrumentSpec;
use crate::types::{Bbo, DesiredQuotes, OrderIntent, QuoteReason, Side};
use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RiskState {
    pub equity_usdc: f64,
    pub daily_realized_pnl_usdc: f64,
    pub consecutive_losses: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpreadDiagnostic {
    pub model_depth: f64,
    pub fee_cushion: f64,
    pub extra_cushion: f64,
    pub before_clamp: f64,
    pub final_depth: f64,
    pub clamped: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteDecision {
    pub quotes: DesiredQuotes,
    pub bid_spread: Option<SpreadDiagnostic>,
    pub ask_spread: Option<SpreadDiagnostic>,
}

#[derive(Debug, Clone)]
pub struct CarteaJaimungalPolicy {
    instrument: InstrumentSpec,
    quoting: QuotingConfig,
    risk: RiskConfig,
}

impl CarteaJaimungalPolicy {
    pub fn new(
        instrument: InstrumentSpec,
        quoting: QuotingConfig,
        risk: RiskConfig,
    ) -> Result<Self> {
        instrument.validate()?;
        if !quoting.spread_multiplier.is_finite() || quoting.spread_multiplier <= 0.0 {
            bail!("spread_multiplier must be finite and positive");
        }
        if !quoting.maker_fee_rate.is_finite() || quoting.maker_fee_rate < 0.0 {
            bail!("maker_fee_rate must be finite and non-negative");
        }
        if quoting.min_half_spread_bps < 0.0
            || quoting.min_half_spread_bps >= quoting.max_half_spread_bps
        {
            bail!("invalid half-spread bounds");
        }
        Ok(Self {
            instrument,
            quoting,
            risk,
        })
    }

    pub const fn instrument(&self) -> &InstrumentSpec {
        &self.instrument
    }

    /// Derive one inventory jump from capital and round down to the venue size
    /// quantum. This is called only when physical inventory is flat.
    pub fn derive_inventory_unit(&self, mid: f64, q_max: i64) -> Result<i64> {
        if !mid.is_finite() || mid <= 0.0 || q_max <= 0 {
            bail!("cannot size inventory unit from invalid mid/q_max");
        }
        let base = self.quoting.available_capital_usdc
            * self.quoting.target_capital_utilisation
            * self.quoting.leverage
            / (q_max as f64 * mid);
        let units = (base * self.instrument.size_scale() as f64).floor() as i64;
        if units <= 0 {
            bail!("capital-derived inventory unit rounds to zero");
        }
        if self.instrument.size_from_units(units) * mid < self.instrument.minimum_notional {
            bail!("capital-derived inventory unit is below venue minimum notional");
        }
        Ok(units)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &self,
        surface: &HjbSurface,
        bbo: Bbo,
        inventory_units: i64,
        inventory_unit: i64,
        tau_remaining: f64,
        quote_seq: u64,
        now_ns: u64,
        reason: QuoteReason,
        risk_state: RiskState,
    ) -> QuoteDecision {
        let empty = |reason| {
            let mut quotes = DesiredQuotes::empty(reason, quote_seq, now_ns);
            quotes.source_recv_ns = bbo.recv_ns;
            quotes.source_exchange_ms = bbo.exchange_ms;
            QuoteDecision {
                quotes,
                bid_spread: None,
                ask_spread: None,
            }
        };
        if self.risk.kill_switch || !bbo.is_valid() || inventory_unit <= 0 {
            return empty(QuoteReason::RiskLimit);
        }
        if risk_state.daily_realized_pnl_usdc <= -self.risk.max_daily_loss_usdc
            || risk_state.consecutive_losses >= self.risk.max_consecutive_losses
        {
            return empty(QuoteReason::RiskLimit);
        }

        let mid = self.instrument.price_from_units(bbo.mid_units());
        let market_spread_bps =
            self.instrument.price_from_units(bbo.ask_px - bbo.bid_px) / mid * 10_000.0;
        if !mid.is_finite() || mid <= 0.0 || market_spread_bps > self.risk.max_market_spread_bps {
            return empty(QuoteReason::RiskLimit);
        }
        let q_exact_unclamped = inventory_units as f64 / inventory_unit as f64;
        let q_exact = q_exact_unclamped.clamp(surface.q_min as f64, surface.q_max as f64);
        let q_rounded = q_exact.round() as i64;

        let depths = surface.depth_pair(q_exact, tau_remaining);
        let mut bid_spread = depths
            .bid
            .and_then(|depth| self.assemble_spread(depth, mid));
        let mut ask_spread = depths
            .ask
            .and_then(|depth| self.assemble_spread(depth, mid));

        let threshold = self.quoting.reduce_only_threshold_q.max(0.0);
        if q_exact >= threshold && q_exact > 0.0 {
            bid_spread = None;
        }
        if q_exact <= -threshold && q_exact < 0.0 {
            ask_spread = None;
        }

        let bid = bid_spread.and_then(|spread| {
            let px = self.round_bid(mid - spread.final_depth);
            self.build_intent(
                Side::Buy,
                px,
                inventory_units,
                inventory_unit,
                mid,
                q_exact <= -threshold && q_exact < 0.0,
                risk_state,
            )
        });
        let ask = ask_spread.and_then(|spread| {
            let px = self.round_ask(mid + spread.final_depth);
            self.build_intent(
                Side::Sell,
                px,
                inventory_units,
                inventory_unit,
                mid,
                q_exact >= threshold && q_exact > 0.0,
                risk_state,
            )
        });

        let bid = bid.filter(|order| order.px > 0 && order.px < bbo.ask_px);
        let ask = ask.filter(|order| order.px > bbo.bid_px);
        if bid.is_some_and(|order| ask.is_some_and(|other| order.px >= other.px)) {
            return empty(QuoteReason::RiskLimit);
        }

        QuoteDecision {
            quotes: DesiredQuotes {
                bid,
                ask,
                quote_seq,
                model_revision: surface.revision,
                generated_ns: now_ns,
                source_recv_ns: bbo.recv_ns,
                source_exchange_ms: bbo.exchange_ms,
                reason,
                mid,
                q_exact,
                q_rounded,
                tau_remaining,
            },
            bid_spread,
            ask_spread,
        }
    }

    fn assemble_spread(&self, model_depth: f64, mid: f64) -> Option<SpreadDiagnostic> {
        if !model_depth.is_finite() || !mid.is_finite() || mid <= 0.0 {
            return None;
        }
        let fee_cushion = self.quoting.maker_fee_rate * mid;
        let extra_cushion = self.quoting.extra_cushion_bps / 10_000.0 * mid;
        let before_clamp =
            model_depth * self.quoting.spread_multiplier + fee_cushion + extra_cushion;
        let floor = self.quoting.min_half_spread_bps / 10_000.0 * mid;
        let cap = self.quoting.max_half_spread_bps / 10_000.0 * mid;
        let final_depth = before_clamp.clamp(floor, cap);
        let clamped = if before_clamp < floor {
            Some("floor")
        } else if before_clamp > cap {
            Some("cap")
        } else {
            None
        };
        Some(SpreadDiagnostic {
            model_depth,
            fee_cushion,
            extra_cushion,
            before_clamp,
            final_depth,
            clamped,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_intent(
        &self,
        side: Side,
        px: i64,
        inventory_units: i64,
        qty_units: i64,
        mid: f64,
        reduce_only: bool,
        risk_state: RiskState,
    ) -> Option<OrderIntent> {
        let next_inventory =
            inventory_units.saturating_add(side.inventory_sign().saturating_mul(qty_units));
        if reduce_only
            && (next_inventory.abs() >= inventory_units.abs()
                || inventory_units.signum() == side.inventory_sign())
        {
            return None;
        }
        let next_base = self.instrument.size_from_units(next_inventory.abs());
        let next_notional = next_base * mid;
        if self.risk.max_notional_usdc > 0.0 && next_notional > self.risk.max_notional_usdc {
            return None;
        }
        let margin = next_notional / self.quoting.leverage;
        if self.risk.max_margin_usdc > 0.0 && margin > self.risk.max_margin_usdc {
            return None;
        }
        let maintenance = next_notional * self.risk.maintenance_margin_rate;
        if risk_state.equity_usdc - maintenance < self.risk.min_liquidation_buffer_usdc {
            return None;
        }
        let order_notional = self.instrument.size_from_units(qty_units) * mid;
        if order_notional < self.instrument.minimum_notional {
            return None;
        }
        Some(OrderIntent {
            side,
            px,
            qty_units,
            post_only: true,
            reduce_only,
        })
    }

    fn round_bid(&self, price: f64) -> i64 {
        let raw = (price * self.instrument.price_scale() as f64).floor() as i64;
        floor_to_quantum(raw, self.instrument.price_quantum(raw))
    }

    fn round_ask(&self, price: f64) -> i64 {
        let raw = (price * self.instrument.price_scale() as f64).ceil() as i64;
        ceil_to_quantum(raw, self.instrument.price_quantum(raw))
    }
}

#[inline]
fn floor_to_quantum(value: i64, quantum: i64) -> i64 {
    value.div_euclid(quantum) * quantum
}

#[inline]
fn ceil_to_quantum(value: i64, quantum: i64) -> i64 {
    let quotient = value.div_euclid(quantum);
    if value.rem_euclid(quantum) == 0 {
        quotient * quantum
    } else {
        (quotient + 1) * quantum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::hjb::{solve_asymmetric, CjParameters};

    fn instrument(symbol: &str, price_decimals: u32, size_decimals: u32) -> InstrumentSpec {
        InstrumentSpec {
            symbol: symbol.to_owned(),
            dex: String::new(),
            asset_id: 1,
            sz_decimals: size_decimals,
            max_price_decimals: price_decimals,
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

    fn surface() -> HjbSurface {
        solve_asymmetric(
            CjParameters {
                lambda_plus: 0.12,
                lambda_minus: 0.11,
                kappa_plus: 10_000.0,
                kappa_minus: 9_000.0,
                epsilon_plus: 2.0e-5,
                epsilon_minus: 3.0e-5,
                sigma2_per_second: Some(3.0e-9),
            },
            &ModelConfig::default(),
            2_000.0,
            9,
        )
        .unwrap()
    }

    #[test]
    fn cashcat_quotes_are_post_only_and_two_sided() {
        let policy = CarteaJaimungalPolicy::new(
            instrument("CASHCAT", 6, 0),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        let decision = policy.compute(
            &surface(),
            Bbo {
                bid_px: 131_970,
                bid_sz: 2_000,
                ask_px: 132_200,
                ask_sz: 3_000,
                exchange_ms: 1,
                recv_ns: 1,
            },
            0,
            2_000,
            150.0,
            1,
            1,
            QuoteReason::Market,
            RiskState {
                equity_usdc: 1_000.0,
                ..RiskState::default()
            },
        );
        assert!(decision.quotes.bid.is_some());
        assert!(decision.quotes.ask.is_some());
        assert!(decision.quotes.bid.unwrap().px < 132_200);
        assert!(decision.quotes.ask.unwrap().px > 131_970);
    }

    #[test]
    fn reducing_side_only_is_reduce_only_at_threshold() {
        let policy = CarteaJaimungalPolicy::new(
            instrument("CASHCAT", 6, 0),
            QuotingConfig::default(),
            RiskConfig::default(),
        )
        .unwrap();
        let decision = policy.compute(
            &surface(),
            Bbo {
                bid_px: 131_970,
                bid_sz: 2_000,
                ask_px: 132_200,
                ask_sz: 3_000,
                exchange_ms: 1,
                recv_ns: 1,
            },
            10_000,
            2_000,
            75.0,
            1,
            1,
            QuoteReason::Market,
            RiskState {
                equity_usdc: 1_000.0,
                ..RiskState::default()
            },
        );
        assert!(decision.quotes.bid.is_none());
        assert!(decision.quotes.ask.unwrap().reduce_only);
    }

    #[test]
    fn synthetic_decimals_use_same_policy_code() {
        let mut quoting = QuotingConfig::default();
        quoting.available_capital_usdc = 10_000.0;
        let policy = CarteaJaimungalPolicy::new(
            instrument("SYNTHETIC", 2, 3),
            quoting,
            RiskConfig {
                max_notional_usdc: 100_000.0,
                max_margin_usdc: 100_000.0,
                ..RiskConfig::default()
            },
        )
        .unwrap();
        assert!(policy.derive_inventory_unit(100.0, 6).unwrap() > 0);
    }

    #[test]
    fn prospective_notional_limit_removes_only_the_increasing_side() {
        let policy = CarteaJaimungalPolicy::new(
            instrument("CASHCAT", 6, 0),
            QuotingConfig::default(),
            RiskConfig {
                max_notional_usdc: 300.0,
                max_margin_usdc: 10_000.0,
                ..RiskConfig::default()
            },
        )
        .unwrap();
        let decision = policy.compute(
            &surface(),
            Bbo {
                bid_px: 131_970,
                bid_sz: 2_000,
                ask_px: 132_200,
                ask_sz: 3_000,
                exchange_ms: 1,
                recv_ns: 1,
            },
            2_000,
            2_000,
            75.0,
            1,
            1,
            QuoteReason::Market,
            RiskState {
                equity_usdc: 1_000.0,
                ..RiskState::default()
            },
        );
        assert!(decision.quotes.bid.is_none());
        assert!(decision.quotes.ask.is_some());
    }
}
