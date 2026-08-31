use crate::config::{FlowGuardConfig, ModelConfig, QuotingConfig, RiskConfig};
use crate::flow_guard::{FlowGuard, MidWindow};
use crate::hjb::HjbSurface;
use crate::instrument::InstrumentSpec;
use crate::latency::{HotLatencySampler, LatencyKind, LatencyMonitor};
use crate::lockfree::{
    BboReader, HotPathSignal, QuoteWriter, HOT_SIGNAL_ACCOUNT, HOT_SIGNAL_FILL, HOT_SIGNAL_MODEL,
    HOT_SIGNAL_SHUTDOWN,
};
use crate::metrics::Metrics;
use crate::quote::{CarteaJaimungalPolicy, RiskState};
use crate::types::{DesiredQuotes, ProcessClock, QuoteReason};
use anyhow::{bail, Context, Result};
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
    pub valid_until_ns: u64,
}

impl ModelBundle {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        surface: HjbSurface,
        inventory_unit: i64,
        generated_at_ms: u64,
        installed_at_ms: u64,
        installed_at_ns: u64,
        max_age_seconds: u64,
        max_future_skew_seconds: u64,
    ) -> Self {
        let maximum_age_ms = max_age_seconds.saturating_mul(1_000);
        let maximum_future_ms = max_future_skew_seconds.saturating_mul(1_000);
        let future_is_valid = generated_at_ms <= installed_at_ms.saturating_add(maximum_future_ms);
        let age_ms = installed_at_ms.saturating_sub(generated_at_ms);
        let remaining_ms = maximum_age_ms.saturating_sub(age_ms);
        let valid_until_ns = if future_is_valid && age_ms <= maximum_age_ms {
            installed_at_ns.saturating_add(remaining_ms.saturating_mul(1_000_000))
        } else {
            0
        };
        Self {
            surface,
            inventory_unit,
            generated_at_ms,
            valid_until_ns,
        }
    }
}

/// VPIN published from the event loop, where trade prints are seen, and read
/// by the hot path, which only ever sees the BBO.
///
/// `NOT_READY` distinguishes "the tracker is still warming up" from "flow is
/// calm": the guard must not re-open on a statistic it does not yet have. A
/// plain f64 sentinel would collide with a real value, so warm-up is encoded as
/// a separate flag rather than a magic number.
#[derive(Debug, Default)]
#[repr(align(64))]
struct AtomicFlowState {
    vpin_bits: AtomicU64,
    ready: AtomicBool,
}

impl AtomicFlowState {
    fn store(&self, vpin: Option<f64>) {
        match vpin {
            Some(value) => {
                self.vpin_bits.store(value.to_bits(), Ordering::Release);
                self.ready.store(true, Ordering::Release);
            }
            None => self.ready.store(false, Ordering::Release),
        }
    }

    fn load(&self) -> Option<f64> {
        if !self.ready.load(Ordering::Acquire) {
            return None;
        }
        Some(f64::from_bits(self.vpin_bits.load(Ordering::Acquire)))
    }
}

pub struct FlowWriter(Arc<AtomicFlowState>);

#[derive(Clone)]
pub struct FlowReader(Arc<AtomicFlowState>);

pub fn flow_channel() -> (FlowWriter, FlowReader) {
    let inner = Arc::new(AtomicFlowState::default());
    (FlowWriter(inner.clone()), FlowReader(inner))
}

impl FlowWriter {
    pub fn store(&self, vpin: Option<f64>) {
        self.0.store(vpin);
    }
}

impl FlowReader {
    pub fn load(&self) -> Option<f64> {
        self.0.load()
    }
}

#[derive(Debug, Default)]
#[repr(align(64))]
struct AtomicRiskState {
    seq: AtomicU64,
    equity_usdc: AtomicU64,
    daily_realized_pnl_usdc: AtomicU64,
    consecutive_losses: AtomicU32,
}

impl AtomicRiskState {
    fn store(&self, value: RiskState) {
        self.seq.fetch_add(1, Ordering::AcqRel);
        self.equity_usdc
            .store(value.equity_usdc.to_bits(), Ordering::Release);
        self.daily_realized_pnl_usdc
            .store(value.daily_realized_pnl_usdc.to_bits(), Ordering::Release);
        self.consecutive_losses
            .store(value.consecutive_losses, Ordering::Release);
        self.seq.fetch_add(1, Ordering::Release);
    }

    fn load(&self) -> RiskState {
        loop {
            let before = self.seq.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let value = RiskState {
                equity_usdc: f64::from_bits(self.equity_usdc.load(Ordering::Acquire)),
                daily_realized_pnl_usdc: f64::from_bits(
                    self.daily_realized_pnl_usdc.load(Ordering::Acquire),
                ),
                consecutive_losses: self.consecutive_losses.load(Ordering::Acquire),
            };
            let after = self.seq.load(Ordering::Acquire);
            if before == after && before & 1 == 0 {
                return value;
            }
            std::hint::spin_loop();
        }
    }
}

pub struct RiskWriter(Arc<AtomicRiskState>);

#[derive(Clone)]
pub struct RiskReader(Arc<AtomicRiskState>);

pub fn risk_channel() -> (RiskWriter, RiskReader) {
    let inner = Arc::new(AtomicRiskState::default());
    (RiskWriter(inner.clone()), RiskReader(inner))
}

impl RiskWriter {
    pub fn store(&self, value: RiskState) {
        self.0.store(value);
    }
}

impl RiskReader {
    pub fn load(&self) -> RiskState {
        self.0.load()
    }
}

pub struct HotPathInputs {
    pub latest_bbo: BboReader,
    pub signal: Arc<HotPathSignal>,
    pub desired: QuoteWriter,
    pub model: Arc<ArcSwapOption<ModelBundle>>,
    pub instrument: InstrumentSpec,
    pub quoting: QuotingConfig,
    pub risk: RiskConfig,
    pub model_config: ModelConfig,
    pub inventory_units: Arc<AtomicI64>,
    pub risk_state: RiskReader,
    pub flow_state: FlowReader,
    pub flow_guard: FlowGuardConfig,
    pub scientifically_valid: Arc<AtomicBool>,
    pub market_stale_ms: u64,
    pub clock: Arc<ProcessClock>,
    pub metrics: Arc<Metrics>,
    pub latency: Arc<LatencyMonitor>,
    pub latency_sample_every: u64,
    pub hot_path_cpu: Option<usize>,
}

pub fn spawn_hot_path(inputs: HotPathInputs) -> Result<JoinHandle<()>> {
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let handle = thread::Builder::new()
        .name("cj-hot-path".to_owned())
        .spawn(move || match pin_current_thread(inputs.hot_path_cpu) {
            Ok(()) => {
                let _ = ready_tx.send(Ok(()));
                run_hot_path(&inputs);
            }
            Err(error) => {
                inputs.scientifically_valid.store(false, Ordering::Release);
                let _ = ready_tx.send(Err(error));
            }
        })?;
    match ready_rx
        .recv()
        .context("hot-path thread exited before affinity validation")?
    {
        Ok(()) => Ok(handle),
        Err(error) => {
            let _ = handle.join();
            Err(error)
        }
    }
}

pub struct HotPathEngine {
    policy: CarteaJaimungalPolicy,
    market_stale_ns: u64,
    mid_window: MidWindow,
    flow_guard: FlowGuard,
    quote_seq: u64,
    last_quotes: DesiredQuotes,
    episode_start_ns: u64,
    latency_sample_mask: u64,
    latency_counter: u64,
    latency_sampler: HotLatencySampler,
    unreported_quote_decisions: u64,
    last_flow_bbo_recv_ns: u64,
}

impl HotPathEngine {
    pub fn new(inputs: &HotPathInputs) -> Result<Self> {
        let policy = CarteaJaimungalPolicy::new(
            inputs.instrument.clone(),
            inputs.quoting.clone(),
            inputs.risk.clone(),
        )?;
        let mid_capacity = inputs
            .flow_guard
            .fast_move_window_ms
            .saturating_mul(200)
            .div_ceil(1_000)
            .clamp(64, 8_192) as usize;
        Ok(Self {
            policy,
            market_stale_ns: inputs.market_stale_ms.saturating_mul(1_000_000),
            mid_window: MidWindow::new(mid_capacity, inputs.flow_guard.fast_move_window_ms),
            flow_guard: FlowGuard::new(inputs.flow_guard.clone()),
            quote_seq: 0,
            last_quotes: DesiredQuotes::default(),
            episode_start_ns: inputs.clock.now_ns(),
            latency_sample_mask: inputs.latency_sample_every.saturating_sub(1),
            latency_counter: 0,
            latency_sampler: HotLatencySampler::default(),
            unreported_quote_decisions: 0,
            last_flow_bbo_recv_ns: 0,
        })
    }
}

fn run_hot_path(inputs: &HotPathInputs) {
    inputs.signal.register_current_thread();
    let Ok(mut engine) = HotPathEngine::new(inputs) else {
        inputs.scientifically_valid.store(false, Ordering::Release);
        return;
    };
    let idle_poll = Duration::from_millis(100);
    loop {
        let mut pending = inputs.signal.take_pending();
        if pending == 0 {
            inputs.signal.park_timeout(idle_poll);
            pending = inputs.signal.take_pending();
        }
        if pending & HOT_SIGNAL_SHUTDOWN != 0 {
            engine.latency_sampler.flush(&inputs.latency);
            if engine.unreported_quote_decisions != 0 {
                inputs
                    .metrics
                    .quote_decisions
                    .fetch_add(engine.unreported_quote_decisions, Ordering::Relaxed);
            }
            engine.quote_seq = engine.quote_seq.wrapping_add(1);
            inputs.desired.publish(DesiredQuotes::empty(
                QuoteReason::Shutdown,
                engine.quote_seq,
                inputs.clock.now_ns(),
            ));
            return;
        }
        engine.step(inputs, pending, inputs.clock.now_ns());
    }
}

impl HotPathEngine {
    #[inline]
    pub fn step(&mut self, inputs: &HotPathInputs, pending: usize, now_ns: u64) {
        let started_ns = now_ns;
        let Some(bbo) = inputs.latest_bbo.load() else {
            return;
        };
        self.quote_seq = self.quote_seq.wrapping_add(1);
        let bundle_guard = inputs.model.load();
        let Some(bundle) = bundle_guard.as_ref() else {
            let mut next =
                DesiredQuotes::empty(QuoteReason::StaleCalibration, self.quote_seq, now_ns);
            next.source_recv_ns = bbo.recv_ns;
            next.source_exchange_ms = bbo.exchange_ms;
            if quote_changed(self.last_quotes, next) {
                inputs.desired.publish(next);
                inputs
                    .metrics
                    .quote_publications
                    .fetch_add(1, Ordering::Relaxed);
                self.last_quotes = next;
            }
            return;
        };
        let inventory = inputs.inventory_units.load(Ordering::Relaxed);
        let q_exact = inventory as f64 / bundle.inventory_unit as f64;
        let elapsed = now_ns.saturating_sub(self.episode_start_ns) as f64 / 1_000_000_000.0;
        let minimum_elapsed =
            inputs.model_config.horizon_seconds * inputs.model_config.episode_min_elapsed_fraction;
        let episode_rolled = elapsed >= inputs.model_config.horizon_seconds
            || (inputs.model_config.episode_reset_on_flat
                && q_exact.round() == 0.0
                && elapsed >= minimum_elapsed);
        let elapsed = if episode_rolled {
            self.episode_start_ns = now_ns;
            0.0
        } else {
            elapsed
        };
        let tau = (inputs.model_config.horizon_seconds - elapsed).max(0.0);
        let calibration_fresh = now_ns <= bundle.valid_until_ns;
        let market_age_ns = now_ns.saturating_sub(bbo.recv_ns);
        if bbo.recv_ns != self.last_flow_bbo_recv_ns {
            self.mid_window.observe(bbo.recv_ns, bbo.mid_units());
            self.last_flow_bbo_recv_ns = bbo.recv_ns;
        }
        let move_bps = self.mid_window.advance(now_ns, bbo.mid_units());
        let flow_tripped =
            self.flow_guard
                .evaluate(now_ns / 1_000_000, move_bps, inputs.flow_state.load());
        let reason = if pending & HOT_SIGNAL_FILL != 0 {
            QuoteReason::Fill
        } else if pending & HOT_SIGNAL_ACCOUNT != 0 {
            QuoteReason::AccountUpdate
        } else if pending & HOT_SIGNAL_MODEL != 0 {
            QuoteReason::Calibration
        } else if episode_rolled {
            QuoteReason::Episode
        } else {
            QuoteReason::Market
        };
        let mut next = if !inputs.scientifically_valid.load(Ordering::Acquire) {
            DesiredQuotes::empty(QuoteReason::InvalidRun, self.quote_seq, now_ns)
        } else if !calibration_fresh {
            DesiredQuotes::empty(QuoteReason::StaleCalibration, self.quote_seq, now_ns)
        } else if market_age_ns > self.market_stale_ns {
            DesiredQuotes::empty(QuoteReason::StaleMarket, self.quote_seq, now_ns)
        } else if !inputs.latency.trading_allowed() {
            DesiredQuotes::empty(QuoteReason::LatencyLimit, self.quote_seq, now_ns)
        } else if flow_tripped {
            DesiredQuotes::empty(QuoteReason::ToxicFlow, self.quote_seq, now_ns)
        } else {
            self.policy
                .compute(
                    &bundle.surface,
                    bbo,
                    inventory,
                    bundle.inventory_unit,
                    tau,
                    self.quote_seq,
                    now_ns,
                    reason,
                    inputs.risk_state.load(),
                )
                .quotes
        };
        if next.source_recv_ns == 0 {
            next.source_recv_ns = bbo.recv_ns;
        }
        if next.source_exchange_ms == 0 {
            next.source_exchange_ms = bbo.exchange_ms;
        }
        if next.reason == QuoteReason::RiskLimit {
            inputs.metrics.risk_refusals.fetch_add(1, Ordering::Relaxed);
        }
        self.unreported_quote_decisions += 1;
        if self.unreported_quote_decisions == 64 {
            inputs
                .metrics
                .quote_decisions
                .fetch_add(self.unreported_quote_decisions, Ordering::Relaxed);
            self.unreported_quote_decisions = 0;
        }
        if quote_changed(self.last_quotes, next) {
            inputs.desired.publish(next);
            inputs
                .metrics
                .quote_publications
                .fetch_add(1, Ordering::Relaxed);
            self.last_quotes = next;
        }
        self.latency_counter = self.latency_counter.wrapping_add(1);
        if self.latency_counter & self.latency_sample_mask == 0 {
            let finished_ns = inputs.clock.now_ns();
            self.latency_sampler.record(
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

fn pin_current_thread(cpu: Option<usize>) -> Result<()> {
    let Some(cpu) = cpu else { return Ok(()) };
    let Some(cores) = core_affinity::get_core_ids() else {
        bail!("CPU affinity was requested for core {cpu}, but available cores cannot be queried");
    };
    let Some(core) = cores.into_iter().find(|core| core.id == cpu) else {
        bail!("requested hot-path CPU {cpu} is not available");
    };
    if !core_affinity::set_for_current(core) {
        bail!("failed to pin hot-path thread to CPU {cpu}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LatencyConfig, ModelConfig, QuotingConfig, RiskConfig};
    use crate::instrument::InstrumentSpec;
    use crate::lockfree::{bbo_channel, quote_channel, HOT_SIGNAL_MARKET};

    #[test]
    fn atomic_risk_state_round_trips() {
        let value = RiskState {
            equity_usdc: 999.5,
            daily_realized_pnl_usdc: -2.0,
            consecutive_losses: 3,
        };
        let (writer, reader) = risk_channel();
        writer.store(value);
        assert_eq!(reader.load(), value);
    }

    #[test]
    fn model_freshness_is_converted_to_a_monotonic_deadline_once() {
        let surface = crate::hjb::solve_asymmetric(
            crate::hjb::CjParameters {
                lambda_plus: 0.1,
                lambda_minus: 0.1,
                kappa_plus: 100.0,
                kappa_minus: 100.0,
                epsilon_plus: 0.0,
                epsilon_minus: 0.0,
                sigma2_per_second: None,
            },
            &ModelConfig::default(),
            1.0,
            1,
        )
        .unwrap();
        let fresh = ModelBundle::prepare(surface.clone(), 1, 9_950, 10_000, 7_000, 2, 1);
        assert_eq!(fresh.valid_until_ns, 1_950_007_000);
        let stale = ModelBundle::prepare(surface.clone(), 1, 1, 10_000, 7_000, 2, 1);
        assert_eq!(stale.valid_until_ns, 0);
        let future = ModelBundle::prepare(surface, 1, 12_000, 10_000, 7_000, 2, 1);
        assert_eq!(future.valid_until_ns, 0);
    }

    #[test]
    fn unavailable_requested_cpu_fails_closed() {
        assert!(pin_current_thread(Some(usize::MAX)).is_err());
    }

    #[test]
    fn concurrent_risk_publication_is_coherent() {
        let (writer_state, state) = risk_channel();
        writer_state.store(RiskState {
            equity_usdc: 1.0,
            daily_realized_pnl_usdc: 10.0,
            consecutive_losses: 100,
        });
        let writer = std::thread::spawn(move || {
            for index in 0..100_000 {
                let base = if index & 1 == 0 { 1.0 } else { 2.0 };
                writer_state.store(RiskState {
                    equity_usdc: base,
                    daily_realized_pnl_usdc: base * 10.0,
                    consecutive_losses: base as u32 * 100,
                });
            }
        });
        for _ in 0..100_000 {
            let value = state.load();
            assert_eq!(value.daily_realized_pnl_usdc, value.equity_usdc * 10.0);
            assert_eq!(value.consecutive_losses, value.equity_usdc as u32 * 100);
        }
        writer.join().unwrap();
    }

    #[test]
    fn absent_calibration_publishes_no_orders() {
        let (bbo_writer, bbo) = bbo_channel();
        bbo_writer.store(crate::types::Bbo {
            bid_px: 100,
            bid_sz: 10,
            ask_px: 102,
            ask_sz: 10,
            exchange_ms: crate::types::unix_ms(),
            recv_ns: 0,
        });
        let signal = Arc::new(HotPathSignal::default());
        let (desired_writer, desired) = quote_channel();
        let (risk_writer, risk_state) = risk_channel();
        risk_writer.store(RiskState {
            equity_usdc: 1_000.0,
            ..RiskState::default()
        });
        let handle = spawn_hot_path(HotPathInputs {
            flow_state: flow_channel().1,
            // The guard is off in this test: it exercises the publication path,
            // and an armed breaker would withdraw quotes on the synthetic jumps.
            flow_guard: FlowGuardConfig {
                enabled: false,
                ..FlowGuardConfig::default()
            },
            latest_bbo: bbo,
            signal: signal.clone(),
            desired: desired_writer,
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
                margin_table_id: 0,
                only_isolated: false,
                margin_mode: String::new(),
                is_delisted: false,
                metadata_fingerprint: String::new(),
            },
            quoting: QuotingConfig::default(),
            risk: RiskConfig::default(),
            model_config: ModelConfig::default(),
            inventory_units: Arc::new(AtomicI64::new(0)),
            risk_state,
            scientifically_valid: Arc::new(AtomicBool::new(true)),
            market_stale_ms: 5_000,
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
