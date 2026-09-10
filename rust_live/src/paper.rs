//! The unit both simulators score: one parameter set, its own account and log.
//!
//! Split out of `main.rs` so the offline backtest (`backtest.rs`) and the live
//! dry-run grid share the pricing and accounting path without either owning it.
//! That sharing is the whole basis for comparing a replay row against a
//! leaderboard row: both come from `PaperVariant::leaderboard_row`.

use anyhow::Result;
use mm_live::config::AppConfig;
use mm_live::execution::{AccountStateProvider, DryRunBackend, ExecutionBackend};
use mm_live::flow_guard::{FlowGuard, MidWindow};
use mm_live::hjb::{CjParameters, HjbSurface};
use mm_live::quote::{CarteaJaimungalPolicy, RiskState};
use mm_live::report::JsonlEventLogger;
use mm_live::types::{Bbo, DesiredQuotes, ExecutionEvent, MarketEvent, QuoteReason};
use std::path::PathBuf;
use tracing::warn;

use crate::grid;

/// One parameter set being simulated: its own configuration, model surface,
/// simulator and log. Nothing here is shared with a peer except the market feed.
///
/// Shared deliberately by the two things that score parameters -- the live
/// dry-run grid in `main.rs` and the offline backtest in `backtest.rs`. They
/// differ in where events come from, not in how a variant is priced or
/// accounted, so a backtest row and a leaderboard row are the same object.
pub(crate) struct PaperVariant {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) config_fingerprint: String,
    pub(crate) config_changes: u32,
    pub(crate) fixed_parameters: Option<CjParameters>,
    pub(crate) config: AppConfig,
    pub(crate) policy: CarteaJaimungalPolicy,
    pub(crate) surface: HjbSurface,
    pub(crate) inventory_unit: i64,
    pub(crate) backend: DryRunBackend,
    pub(crate) logger: JsonlEventLogger,
    pub(crate) report_path: PathBuf,
    pub(crate) episode_start_ns: u64,
    pub(crate) quote_seq: u64,
    pub(crate) fills: u64,
    pub(crate) peak_equity_usdc: f64,
    pub(crate) max_drawdown_usdc: f64,
    /// Per variant, because the thresholds are a lever. The VPIN statistic
    /// itself is a property of the market and is shared across variants; only
    /// the trip decision is per variant.
    pub(crate) guard: FlowGuard,
    pub(crate) mid_window: MidWindow,
    /// Set once this variant has failed. It then stops trading while the rest
    /// of the grid continues, and its report carries the reason.
    pub(crate) failure: Option<String>,
}

/// One variant's slice of a market event, in a form whose errors can be caught
/// per variant instead of aborting the whole grid.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn step_paper_variant(
    variant: &mut PaperVariant,
    event: &MarketEvent,
    event_time: u64,
    bbo: Option<Bbo>,
    vpin_value: Option<f64>,
) -> Result<Option<QuoteReason>> {
    let execution_events = variant.backend.on_market_event(event).await?;
    for execution_event in &execution_events {
        variant
            .logger
            .log("execution_event", Some(event_time), execution_event)?;
    }
    // MAKER fills only. Counting every execution event made a flatten variant
    // show roughly twice its fills, since each maker entry is followed by a
    // taker exit -- and the leaderboard's fills column is what a reader uses to
    // judge whether a row has measured anything.
    variant.fills = variant.fills.saturating_add(
        execution_events
            .iter()
            .filter(|event| matches!(event, ExecutionEvent::Fill(fill) if fill.maker))
            .count() as u64,
    );
    if !variant.backend.scientifically_valid() {
        warn!(variant = %variant.name, reason = ?variant.backend.diagnostics().invalid_reason, "paper variant halted by execution risk; manual review required");
    }
    let reason = if execution_events.is_empty() {
        QuoteReason::Market
    } else {
        QuoteReason::Fill
    };
    if let Some(bbo) = bbo.filter(|_| variant.backend.scientifically_valid()) {
        return variant
            .step(bbo, event_time, reason, vpin_value)
            .await
            .map(Some);
    }
    Ok(None)
}

impl PaperVariant {
    /// Price this variant against the current book and hand the result to its
    /// own simulator. This is the same `policy.compute` the hot path calls; the
    /// grid deliberately does not spawn hot-path threads (see `src/grid.rs`).
    pub(crate) async fn step(
        &mut self,
        bbo: Bbo,
        decision_ms: u64,
        reason: QuoteReason,
        vpin: Option<f64>,
    ) -> Result<QuoteReason> {
        let account = self.backend.account_state();
        let q_exact = if self.inventory_unit == 0 {
            0.0
        } else {
            account.inventory_units as f64 / self.inventory_unit as f64
        };
        let model_now_ns = decision_ms.saturating_mul(1_000_000);
        if self.episode_start_ns == 0 {
            self.episode_start_ns = model_now_ns;
        }
        let elapsed = model_now_ns.saturating_sub(self.episode_start_ns) as f64 / 1_000_000_000.0;
        let horizon_seconds = self.config.model.horizon_seconds;
        let minimum_elapsed = horizon_seconds * self.config.model.episode_min_elapsed_fraction;
        let episode_rolled = elapsed >= horizon_seconds
            || (self.config.model.episode_reset_on_flat
                && q_exact.round() == 0.0
                && elapsed >= minimum_elapsed);
        let elapsed = if episode_rolled {
            self.episode_start_ns = model_now_ns;
            0.0
        } else {
            elapsed
        };
        let tau = (horizon_seconds - elapsed).max(0.0);
        let reason = if episode_rolled {
            QuoteReason::Episode
        } else {
            reason
        };
        let risk_state = RiskState {
            equity_usdc: account.equity_usdc,
            daily_realized_pnl_usdc: self.backend.daily_realized_pnl_usdc(),
            consecutive_losses: account.consecutive_losses,
        };
        self.quote_seq = self.quote_seq.wrapping_add(1);
        // Toxic-flow guard, mirroring the hot path's arm: empty quotes cancel
        // resting orders because a `None` target bypasses the requote hold.
        let move_bps = self.mid_window.observe(model_now_ns, bbo.mid_units());
        if self.guard.evaluate(decision_ms, move_bps, vpin) {
            let mut quotes =
                DesiredQuotes::empty(QuoteReason::ToxicFlow, self.quote_seq, model_now_ns);
            quotes.source_exchange_ms = bbo.exchange_ms;
            self.backend.reconcile(quotes, decision_ms).await?;
            self.logger.log("quote_decision", None, &quotes)?;
            return Ok(QuoteReason::ToxicFlow);
        }
        let quotes = self
            .policy
            .compute(
                &self.surface,
                bbo,
                account.inventory_units,
                self.inventory_unit,
                tau,
                self.quote_seq,
                model_now_ns,
                reason,
                risk_state,
            )
            .quotes;
        self.backend.reconcile(quotes, decision_ms).await?;
        self.logger.log("quote_decision", None, &quotes)?;
        Ok(quotes.reason)
    }

    pub(crate) fn observe_equity(&mut self) {
        let equity = self.backend.account_state().equity_usdc;
        if equity > self.peak_equity_usdc {
            self.peak_equity_usdc = equity;
        }
        let drawdown = self.peak_equity_usdc - equity;
        if drawdown > self.max_drawdown_usdc {
            self.max_drawdown_usdc = drawdown;
        }
    }

    pub(crate) fn leaderboard_row(&self, bbo: Option<Bbo>) -> grid::LeaderboardRow {
        let account = self.backend.account_state();
        let scientifically_valid = self.backend.scientifically_valid() && self.failure.is_none();
        let promotion_pnl_usdc = scientifically_valid
            .then(|| bbo.and_then(|value| self.backend.promotion_pnl_usdc(value)))
            .flatten();
        let has_live_equivalent =
            self.config.dry_run.flatten_after_ms == 0 && self.fixed_parameters.is_none();
        grid::LeaderboardRow {
            name: self.name.clone(),
            description: self.description.clone(),
            net_pnl_usdc: account.equity_usdc - self.config.dry_run.starting_equity_usdc,
            promotion_pnl_usdc,
            equity_usdc: account.equity_usdc,
            realized_pnl_usdc: account.realized_pnl_usdc,
            mark_to_market_pnl_usdc: account.mark_to_market_pnl_usdc,
            fees_usdc: account.fees_usdc,
            funding_usdc: account.funding_usdc,
            inventory_units: account.inventory_units,
            fills: self.fills,
            working_orders: self.backend.working_order_count(),
            max_drawdown_usdc: self.max_drawdown_usdc,
            config_changes: self.config_changes,
            scientifically_valid,
            eligible_for_promotion: scientifically_valid
                && promotion_pnl_usdc.is_some()
                && has_live_equivalent,
        }
    }
}
