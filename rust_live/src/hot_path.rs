use crate::config::{ModelConfig, QuotingConfig, RiskConfig};
use crate::hjb::HjbSurface;
use crate::instrument::InstrumentSpec;
use crate::latency::{HotLatencySampler, LatencyKind, LatencyMonitor};
use crate::lockfree::{
    AtomicBbo, HotPathSignal, SharedQuotes, HOT_SIGNAL_EXECUTION, HOT_SIGNAL_MODEL,
    HOT_SIGNAL_SHUTDOWN,
};
use crate::metrics::Metrics;
use crate::quote::{CarteaJaimungalPolicy, RiskState};
use crate::types::{DesiredQuotes, ProcessClock, QuoteReason};
use anyhow::Result;
use arc_swap::ArcSwapOption;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ModelBundle {
    pub surface: HjbSurface,
    pub inventory_unit: i64,
    pub generated_at_ms: u64,
}

#[derive(Debug, Default)]
pub struct AtomicRiskState {
    equity_usdc: AtomicU64,
    daily_realized_pnl_usdc: AtomicU64,
    consecutive_losses: AtomicU32,
}

impl AtomicRiskState {
    pub fn store(&self, value: RiskState) {
        self.equity_usdc
            .store(value.equity_usdc.to_bits(), Ordering::Relaxed);
        self.daily_realized_pnl_usdc
            .store(value.daily_realized_pnl_usdc.to_bits(), Ordering::Relaxed);
        self.consecutive_losses
            .store(value.consecutive_losses, Ordering::Relaxed);
    }

    pub fn load(&self) -> RiskState {
        RiskState {
            equity_usdc: f64::from_bits(self.equity_usdc.load(Ordering::Relaxed)),
            daily_realized_pnl_usdc: f64::from_bits(
                self.daily_realized_pnl_usdc.load(Ordering::Relaxed),
            ),
            consecutive_losses: self.consecutive_losses.load(Ordering::Relaxed),
        }
    }
}

pub struct HotPathInputs {
    pub latest_bbo: Arc<AtomicBbo>,
    pub signal: Arc<HotPathSignal>,
    pub desired: Arc<SharedQuotes>,
    pub model: Arc<ArcSwapOption<ModelBundle>>,
    pub instrument: InstrumentSpec,
    pub quoting: QuotingConfig,
    pub risk: RiskConfig,
    pub model_config: ModelConfig,
    pub inventory_units: Arc<AtomicI64>,
    pub risk_state: Arc<AtomicRiskState>,
    pub scientifically_valid: Arc<AtomicBool>,
    pub market_stale_ms: u64,
    pub calibration_max_age_seconds: u64,
    pub calibration_max_future_skew_seconds: u64,
    pub clock: Arc<ProcessClock>,
    pub metrics: Arc<Metrics>,
    pub latency: Arc<LatencyMonitor>,
    pub latency_sample_every: u64,
    pub hot_path_cpu: Option<usize>,
}

pub fn spawn_hot_path(inputs: HotPathInputs) -> Result<JoinHandle<()>> {
    Ok(thread::Builder::new()
        .name("cj-hot-path".to_owned())
        .spawn(move || run_hot_path(&inputs))?)
}

fn run_hot_path(inputs: &HotPathInputs) {
    pin_current_thread(inputs.hot_path_cpu);
    inputs.signal.register_current_thread();
    let Ok(policy) = CarteaJaimungalPolicy::new(
        inputs.instrument.clone(),
        inputs.quoting.clone(),
        inputs.risk.clone(),
    ) else {
        inputs.scientifically_valid.store(false, Ordering::Release);
        return;
    };
    let market_stale_ns = inputs.market_stale_ms.saturating_mul(1_000_000);
    let idle_poll = Duration::from_millis(100);
    let mut quote_seq = 0_u64;
    let mut last_quotes = DesiredQuotes::default();
    let mut episode_start_ns = inputs.clock.now_ns();
    let latency_sample_mask = inputs.latency_sample_every.saturating_sub(1);
    let mut latency_counter = 0_u64;
    let mut latency_sampler = HotLatencySampler::default();
    let mut unreported_quote_decisions = 0_u64;
    loop {
        let mut pending = inputs.signal.take_pending();
        if pending == 0 {
            inputs.signal.park_timeout(idle_poll);
            pending = inputs.signal.take_pending();
        }
        if pending & HOT_SIGNAL_SHUTDOWN != 0 {
            latency_sampler.flush(&inputs.latency);
            if unreported_quote_decisions != 0 {
                inputs
                    .metrics
                    .quote_decisions
                    .fetch_add(unreported_quote_decisions, Ordering::Relaxed);
            }
            quote_seq = quote_seq.wrapping_add(1);
            inputs.desired.publish(DesiredQuotes::empty(
                QuoteReason::Shutdown,
                quote_seq,
                inputs.clock.now_ns(),
            ));
            return;
        }
        let now_ns = inputs.clock.now_ns();
        let started_ns = now_ns;
        let Some(bbo) = inputs.latest_bbo.load() else {
            continue;
        };
        quote_seq = quote_seq.wrapping_add(1);
        let Some(bundle) = inputs.model.load_full() else {
            let mut next = DesiredQuotes::empty(QuoteReason::StaleCalibration, quote_seq, now_ns);
            next.source_recv_ns = bbo.recv_ns;
            if quote_changed(last_quotes, next) {
                inputs.desired.publish(next);
                inputs
                    .metrics
                    .quote_publications
                    .fetch_add(1, Ordering::Relaxed);
                last_quotes = next;
            }
            continue;
        };
        let inventory = inputs.inventory_units.load(Ordering::Relaxed);
        let q_exact = inventory as f64 / bundle.inventory_unit as f64;
        let elapsed = now_ns.saturating_sub(episode_start_ns) as f64 / 1_000_000_000.0;
        let minimum_elapsed =
            inputs.model_config.horizon_seconds * inputs.model_config.episode_min_elapsed_fraction;
        let episode_rolled = elapsed >= inputs.model_config.horizon_seconds
            || (inputs.model_config.episode_reset_on_flat
                && q_exact.round() == 0.0
                && elapsed >= minimum_elapsed);
        let elapsed = if episode_rolled {
            episode_start_ns = now_ns;
            0.0
        } else {
            elapsed
        };
        let tau = (inputs.model_config.horizon_seconds - elapsed).max(0.0);
        let wall_now = crate::types::unix_ms();
        let future_limit = wall_now.saturating_add(
            inputs
                .calibration_max_future_skew_seconds
                .saturating_mul(1_000),
        );
        let calibration_fresh = bundle.generated_at_ms <= future_limit
            && wall_now.saturating_sub(bundle.generated_at_ms)
                <= inputs.calibration_max_age_seconds.saturating_mul(1_000);
        let market_age_ns = now_ns.saturating_sub(bbo.recv_ns);
        let reason = if pending & HOT_SIGNAL_EXECUTION != 0 {
            QuoteReason::Fill
        } else if pending & HOT_SIGNAL_MODEL != 0 {
            QuoteReason::Calibration
        } else if episode_rolled {
            QuoteReason::Episode
        } else {
            QuoteReason::Market
        };
        let mut next = if !inputs.scientifically_valid.load(Ordering::Acquire) {
            DesiredQuotes::empty(QuoteReason::InvalidRun, quote_seq, now_ns)
        } else if !calibration_fresh {
            DesiredQuotes::empty(QuoteReason::StaleCalibration, quote_seq, now_ns)
        } else if market_age_ns > market_stale_ns {
            DesiredQuotes::empty(QuoteReason::StaleMarket, quote_seq, now_ns)
        } else if !inputs.latency.trading_allowed() {
            DesiredQuotes::empty(QuoteReason::LatencyLimit, quote_seq, now_ns)
        } else {
            policy
                .compute(
                    &bundle.surface,
                    bbo,
                    inventory,
                    bundle.inventory_unit,
                    tau,
                    quote_seq,
                    now_ns,
                    reason,
                    inputs.risk_state.load(),
                )
                .quotes
        };
        if next.source_recv_ns == 0 {
            next.source_recv_ns = bbo.recv_ns;
        }
        if next.reason == QuoteReason::RiskLimit {
            inputs.metrics.risk_refusals.fetch_add(1, Ordering::Relaxed);
        }
        unreported_quote_decisions += 1;
        if unreported_quote_decisions == 64 {
            inputs
                .metrics
                .quote_decisions
                .fetch_add(unreported_quote_decisions, Ordering::Relaxed);
            unreported_quote_decisions = 0;
        }
        if quote_changed(last_quotes, next) {
            inputs.desired.publish(next);
            inputs
                .metrics
                .quote_publications
                .fetch_add(1, Ordering::Relaxed);
            last_quotes = next;
        }
        latency_counter = latency_counter.wrapping_add(1);
        if latency_counter & latency_sample_mask == 0 {
            let finished_ns = inputs.clock.now_ns();
            latency_sampler.record(
                &inputs.latency,
                LatencyKind::HotDecision,
                finished_ns.saturating_sub(started_ns),
                finished_ns,
            );
        }
    }
}

fn quote_changed(previous: DesiredQuotes, next: DesiredQuotes) -> bool {
    previous.bid != next.bid
        || previous.ask != next.ask
        || previous.model_revision != next.model_revision
        || previous.reason != next.reason
        || previous.q_rounded != next.q_rounded
}

fn pin_current_thread(cpu: Option<usize>) {
    let Some(cpu) = cpu else { return };
    let Some(cores) = core_affinity::get_core_ids() else {
        return;
    };
    if let Some(core) = cores.into_iter().find(|core| core.id == cpu) {
        let _ = core_affinity::set_for_current(core);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LatencyConfig, ModelConfig, QuotingConfig, RiskConfig};
    use crate::instrument::InstrumentSpec;
    use crate::lockfree::HOT_SIGNAL_MARKET;

    #[test]
    fn atomic_risk_state_round_trips() {
        let value = RiskState {
            equity_usdc: 999.5,
            daily_realized_pnl_usdc: -2.0,
            consecutive_losses: 3,
        };
        let atomic = AtomicRiskState::default();
        atomic.store(value);
        assert_eq!(atomic.load(), value);
    }

    #[test]
    fn absent_calibration_publishes_no_orders() {
        let bbo = Arc::new(AtomicBbo::default());
        bbo.store(crate::types::Bbo {
            bid_px: 100,
            bid_sz: 10,
            ask_px: 102,
            ask_sz: 10,
            exchange_ms: crate::types::unix_ms(),
            recv_ns: 0,
        });
        let signal = Arc::new(HotPathSignal::default());
        let desired = Arc::new(SharedQuotes::default());
        let risk_state = Arc::new(AtomicRiskState::default());
        risk_state.store(RiskState {
            equity_usdc: 1_000.0,
            ..RiskState::default()
        });
        let handle = spawn_hot_path(HotPathInputs {
            latest_bbo: bbo,
            signal: signal.clone(),
            desired: desired.clone(),
            model: Arc::new(ArcSwapOption::empty()),
            instrument: InstrumentSpec {
                symbol: "SYN".to_owned(),
                dex: String::new(),
                asset_id: 0,
                sz_decimals: 0,
                max_price_decimals: 0,
                max_significant_figures: 5,
                max_leverage: 3.0,
                minimum_notional: 1.0,
            },
            quoting: QuotingConfig::default(),
            risk: RiskConfig::default(),
            model_config: ModelConfig::default(),
            inventory_units: Arc::new(AtomicI64::new(0)),
            risk_state,
            scientifically_valid: Arc::new(AtomicBool::new(true)),
            market_stale_ms: 5_000,
            calibration_max_age_seconds: 120,
            calibration_max_future_skew_seconds: 10,
            clock: Arc::new(ProcessClock::default()),
            metrics: Arc::new(Metrics::default()),
            latency: Arc::new(LatencyMonitor::new(
                "SYN",
                1,
                &LatencyConfig {
                    gate_enabled: false,
                    hot_sample_every: 1,
                    ..LatencyConfig::default()
                },
                false,
            )),
            latency_sample_every: 1,
            hot_path_cpu: None,
        })
        .unwrap();
        signal.notify(HOT_SIGNAL_MARKET);
        for _ in 0..100 {
            let quote = desired.load();
            if quote.quote_seq > 0 {
                assert!(quote.bid.is_none());
                assert!(quote.ask.is_none());
                assert_eq!(quote.reason, QuoteReason::StaleCalibration);
                signal.notify(HOT_SIGNAL_SHUTDOWN);
                handle.join().unwrap();
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("hot path did not publish fail-closed quote state");
    }
}
