use crate::config::LatencyConfig;
use crate::types::{unix_ms, ProcessClock};
use anyhow::Result;
use arc_swap::ArcSwap;
use crossbeam_queue::ArrayQueue;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyKind {
    MarketEventDispatch,
    PublicWsPingRtt,
    AccountWsPingRtt,
    InfoRequestRtt,
    HotDecision,
    MarketDataToBackendStart,
    DecisionToBackendStart,
    BackendReconcile,
    DecisionToBackendDone,
    CalibrationRefresh,
    SubmitToAck,
    CancelToAck,
    AckToFill,
    FillToCloseSend,
    CloseSendToFill,
    FillToFlat,
}

impl LatencyKind {
    const ALL: [Self; 16] = [
        Self::MarketEventDispatch,
        Self::PublicWsPingRtt,
        Self::AccountWsPingRtt,
        Self::InfoRequestRtt,
        Self::HotDecision,
        Self::MarketDataToBackendStart,
        Self::DecisionToBackendStart,
        Self::BackendReconcile,
        Self::DecisionToBackendDone,
        Self::CalibrationRefresh,
        Self::SubmitToAck,
        Self::CancelToAck,
        Self::AckToFill,
        Self::FillToCloseSend,
        Self::CloseSendToFill,
        Self::FillToFlat,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::MarketEventDispatch => "market_event_dispatch",
            Self::PublicWsPingRtt => "public_ws_ping_rtt",
            Self::AccountWsPingRtt => "account_ws_ping_rtt",
            Self::InfoRequestRtt => "info_request_rtt",
            Self::HotDecision => "hot_decision",
            Self::MarketDataToBackendStart => "market_data_to_backend_start",
            Self::DecisionToBackendStart => "decision_to_backend_start",
            Self::BackendReconcile => "backend_reconcile",
            Self::DecisionToBackendDone => "decision_to_backend_done",
            Self::CalibrationRefresh => "calibration_refresh",
            Self::SubmitToAck => "submit_to_ack",
            Self::CancelToAck => "cancel_to_ack",
            Self::AckToFill => "ack_to_fill",
            Self::FillToCloseSend => "fill_to_close_send",
            Self::CloseSendToFill => "close_send_to_fill",
            Self::FillToFlat => "fill_to_flat",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LatencySample {
    kind: LatencyKind,
    value_ns: u64,
    observed_ns: u64,
}

impl LatencySample {
    const EMPTY: Self = Self {
        kind: LatencyKind::HotDecision,
        value_ns: 0,
        observed_ns: 0,
    };
}

const HOT_BATCH_CAPACITY: usize = 4;

#[derive(Debug, Clone, Copy)]
struct LatencyBatch {
    samples: [LatencySample; HOT_BATCH_CAPACITY],
    len: u8,
}

impl LatencyBatch {
    const fn single(sample: LatencySample) -> Self {
        let mut samples = [LatencySample::EMPTY; HOT_BATCH_CAPACITY];
        samples[0] = sample;
        Self { samples, len: 1 }
    }
}

pub struct HotLatencySampler {
    samples: [LatencySample; HOT_BATCH_CAPACITY],
    len: usize,
}

impl Default for HotLatencySampler {
    fn default() -> Self {
        Self {
            samples: [LatencySample::EMPTY; HOT_BATCH_CAPACITY],
            len: 0,
        }
    }
}

impl HotLatencySampler {
    #[inline]
    pub fn record(
        &mut self,
        monitor: &LatencyMonitor,
        kind: LatencyKind,
        value_ns: u64,
        observed_ns: u64,
    ) {
        self.samples[self.len] = LatencySample {
            kind,
            value_ns,
            observed_ns,
        };
        self.len += 1;
        if self.len == HOT_BATCH_CAPACITY {
            self.flush(monitor);
        }
    }

    #[inline]
    pub fn flush(&mut self, monitor: &LatencyMonitor) {
        if self.len == 0 {
            return;
        }
        let batch = LatencyBatch {
            samples: self.samples,
            len: self.len as u8,
        };
        monitor.push_batch(batch);
        self.len = 0;
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LatencyDistribution {
    pub total_samples: u64,
    pub window_samples: u64,
    pub last_ns: Option<u64>,
    pub min_ns: Option<u64>,
    pub p50_ns: Option<u64>,
    pub p95_ns: Option<u64>,
    pub p99_ns: Option<u64>,
    pub p999_ns: Option<u64>,
    pub max_ns: Option<u64>,
    pub sample_age_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyGateSnapshot {
    pub enabled: bool,
    pub enforced: bool,
    pub trading_allowed: bool,
    pub reason: String,
    pub max_acceptable_p99_ms: f64,
    pub minimum_samples: u64,
    pub minimum_network_samples: u64,
    pub blocking_kind: Option<String>,
    pub observed_p99_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencySnapshot {
    pub schema_version: u32,
    pub symbol: String,
    pub session_started_at_ms: u64,
    pub generated_at_ms: u64,
    pub window_seconds: u64,
    pub hot_sample_every: u64,
    pub dropped_samples: u64,
    pub observer_errors: u64,
    pub gate: LatencyGateSnapshot,
    pub distributions: BTreeMap<String, LatencyDistribution>,
}

impl LatencySnapshot {
    pub fn empty(
        symbol: &str,
        session_started_at_ms: u64,
        config: &LatencyConfig,
        gate_enforced: bool,
    ) -> Self {
        let distributions = LatencyKind::ALL
            .into_iter()
            .map(|kind| (kind.name().to_owned(), LatencyDistribution::default()))
            .collect();
        Self {
            schema_version: 1,
            symbol: symbol.to_owned(),
            session_started_at_ms,
            generated_at_ms: unix_ms(),
            window_seconds: config.window_seconds,
            hot_sample_every: config.hot_sample_every,
            dropped_samples: 0,
            observer_errors: 0,
            gate: initial_gate(config, gate_enforced),
            distributions,
        }
    }

    fn write_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary =
            tempfile::NamedTempFile::new_in(path.parent().unwrap_or_else(|| Path::new(".")))?;
        serde_json::to_writer_pretty(temporary.as_file(), self)?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| anyhow::anyhow!(error.error))?;
        Ok(())
    }
}

pub struct LatencyMonitor {
    queue: ArrayQueue<LatencyBatch>,
    dropped_samples: AtomicU64,
    observer_errors: AtomicU64,
    trading_allowed: AtomicBool,
    current: ArcSwap<LatencySnapshot>,
}

impl LatencyMonitor {
    pub fn new(
        symbol: &str,
        session_started_at_ms: u64,
        config: &LatencyConfig,
        gate_enforced: bool,
    ) -> Self {
        let enforce = config.gate_enabled && gate_enforced;
        Self {
            queue: ArrayQueue::new(config.queue_capacity),
            dropped_samples: AtomicU64::new(0),
            observer_errors: AtomicU64::new(0),
            trading_allowed: AtomicBool::new(!enforce),
            current: ArcSwap::from_pointee(LatencySnapshot::empty(
                symbol,
                session_started_at_ms,
                config,
                gate_enforced,
            )),
        }
    }

    #[inline]
    pub fn record(&self, kind: LatencyKind, value_ns: u64, observed_ns: u64) {
        self.push_batch(LatencyBatch::single(LatencySample {
            kind,
            value_ns,
            observed_ns,
        }));
    }

    #[inline]
    fn push_batch(&self, batch: LatencyBatch) {
        if let Err(batch) = self.queue.push(batch) {
            self.dropped_samples
                .fetch_add(u64::from(batch.len), Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> Arc<LatencySnapshot> {
        self.current.load_full()
    }

    #[inline]
    pub fn trading_allowed(&self) -> bool {
        self.trading_allowed.load(Ordering::Acquire)
    }
}

pub struct LatencyObserver {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl LatencyObserver {
    pub fn spawn(
        monitor: Arc<LatencyMonitor>,
        clock: Arc<ProcessClock>,
        symbol: String,
        session_started_at_ms: u64,
        config: LatencyConfig,
        gate_enforced: bool,
        publish_interval: Duration,
        output_path: PathBuf,
    ) -> Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let handle = thread::Builder::new()
            .name("latency-observer".to_owned())
            .spawn(move || {
                let mut aggregator =
                    LatencyAggregator::new(&symbol, session_started_at_ms, config, gate_enforced);
                loop {
                    thread::park_timeout(publish_interval);
                    let mut snapshot = aggregator.drain(&monitor, clock.now_ns());
                    monitor
                        .trading_allowed
                        .store(snapshot.gate.trading_allowed, Ordering::Release);
                    if snapshot.write_atomic(&output_path).is_err() {
                        snapshot.observer_errors = monitor
                            .observer_errors
                            .fetch_add(1, Ordering::Relaxed)
                            .saturating_add(1);
                    }
                    monitor.current.store(Arc::new(snapshot));
                    if thread_shutdown.load(Ordering::Acquire) {
                        break;
                    }
                }
            })?;
        Ok(Self {
            shutdown,
            handle: Some(handle),
        })
    }

    pub fn stop(mut self) -> Result<()> {
        self.stop_inner()
    }

    fn stop_inner(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("latency observer thread panicked"))?;
        }
        Ok(())
    }
}

impl Drop for LatencyObserver {
    fn drop(&mut self) {
        let _ = self.stop_inner();
    }
}

struct LatencyAggregator {
    symbol: String,
    session_started_at_ms: u64,
    window_ns: u64,
    config: LatencyConfig,
    gate_enforced: bool,
    windows: BTreeMap<LatencyKind, VecDeque<LatencySample>>,
    totals: BTreeMap<LatencyKind, u64>,
}

impl LatencyAggregator {
    fn new(
        symbol: &str,
        session_started_at_ms: u64,
        config: LatencyConfig,
        gate_enforced: bool,
    ) -> Self {
        Self {
            symbol: symbol.to_owned(),
            session_started_at_ms,
            window_ns: config.window_seconds.saturating_mul(1_000_000_000),
            config,
            gate_enforced,
            windows: LatencyKind::ALL
                .into_iter()
                .map(|kind| (kind, VecDeque::new()))
                .collect(),
            totals: LatencyKind::ALL.into_iter().map(|kind| (kind, 0)).collect(),
        }
    }

    fn drain(&mut self, monitor: &LatencyMonitor, now_ns: u64) -> LatencySnapshot {
        while let Some(batch) = monitor.queue.pop() {
            for sample in batch.samples.into_iter().take(usize::from(batch.len)) {
                self.windows
                    .entry(sample.kind)
                    .or_default()
                    .push_back(sample);
                *self.totals.entry(sample.kind).or_default() += 1;
            }
        }
        let cutoff = now_ns.saturating_sub(self.window_ns);
        let mut distributions = BTreeMap::new();
        for kind in LatencyKind::ALL {
            let samples = self.windows.entry(kind).or_default();
            samples.retain(|sample| sample.observed_ns >= cutoff);
            distributions.insert(
                kind.name().to_owned(),
                distribution(
                    samples,
                    self.totals.get(&kind).copied().unwrap_or(0),
                    now_ns,
                ),
            );
        }
        let mut snapshot = LatencySnapshot {
            schema_version: 1,
            symbol: self.symbol.clone(),
            session_started_at_ms: self.session_started_at_ms,
            generated_at_ms: unix_ms(),
            window_seconds: self.config.window_seconds,
            hot_sample_every: self.config.hot_sample_every,
            dropped_samples: monitor.dropped_samples.load(Ordering::Relaxed),
            observer_errors: monitor.observer_errors.load(Ordering::Relaxed),
            gate: initial_gate(&self.config, self.gate_enforced),
            distributions,
        };
        snapshot.gate = evaluate_gate(
            &snapshot.distributions,
            &self.config,
            snapshot.dropped_samples,
            snapshot.observer_errors,
            self.gate_enforced,
        );
        snapshot
    }
}

fn initial_gate(config: &LatencyConfig, gate_enforced: bool) -> LatencyGateSnapshot {
    let enforce = config.gate_enabled && gate_enforced;
    LatencyGateSnapshot {
        enabled: config.gate_enabled,
        enforced: enforce,
        trading_allowed: !enforce,
        reason: if !config.gate_enabled {
            "disabled".to_owned()
        } else if !gate_enforced {
            "not_enforced_in_mode".to_owned()
        } else {
            "warming_up".to_owned()
        },
        max_acceptable_p99_ms: config.max_acceptable_p99_ms,
        minimum_samples: config.minimum_samples,
        minimum_network_samples: config.minimum_network_samples,
        blocking_kind: enforce.then(|| LatencyKind::MarketEventDispatch.name().to_owned()),
        observed_p99_ms: None,
    }
}

fn evaluate_gate(
    distributions: &BTreeMap<String, LatencyDistribution>,
    config: &LatencyConfig,
    dropped_samples: u64,
    observer_errors: u64,
    gate_enforced: bool,
) -> LatencyGateSnapshot {
    if !config.gate_enabled || !gate_enforced {
        return initial_gate(config, gate_enforced);
    }
    if observer_errors != 0 {
        return blocked_gate(config, "observer_error", None, None);
    }
    if dropped_samples != 0 {
        return blocked_gate(config, "samples_dropped", None, None);
    }
    for (mandatory_kind, minimum_samples) in [
        (LatencyKind::MarketEventDispatch, config.minimum_samples),
        (LatencyKind::PublicWsPingRtt, config.minimum_network_samples),
        (
            LatencyKind::AccountWsPingRtt,
            config.minimum_network_samples,
        ),
    ] {
        let mandatory = &distributions[mandatory_kind.name()];
        if mandatory.window_samples < minimum_samples {
            return blocked_gate(config, "warming_up", Some(mandatory_kind), mandatory.p99_ns);
        }
        if mandatory
            .sample_age_ms
            .is_none_or(|age| age > config.max_sample_age_ms)
        {
            return blocked_gate(
                config,
                "samples_stale",
                Some(mandatory_kind),
                mandatory.p99_ns,
            );
        }
    }
    let gated_kinds = [
        LatencyKind::MarketEventDispatch,
        LatencyKind::PublicWsPingRtt,
        LatencyKind::AccountWsPingRtt,
        LatencyKind::InfoRequestRtt,
        LatencyKind::HotDecision,
        LatencyKind::DecisionToBackendDone,
        LatencyKind::SubmitToAck,
        LatencyKind::CancelToAck,
        LatencyKind::AckToFill,
        LatencyKind::FillToCloseSend,
    ];
    let threshold_ns = config.max_acceptable_p99_ms * 1_000_000.0;
    for kind in gated_kinds {
        let distribution = &distributions[kind.name()];
        if distribution.window_samples == 0 {
            continue;
        }
        if distribution
            .p99_ns
            .is_some_and(|value| value as f64 > threshold_ns)
        {
            return blocked_gate(config, "p99_exceeded", Some(kind), distribution.p99_ns);
        }
    }
    LatencyGateSnapshot {
        enabled: true,
        enforced: true,
        trading_allowed: true,
        reason: "ok".to_owned(),
        max_acceptable_p99_ms: config.max_acceptable_p99_ms,
        minimum_samples: config.minimum_samples,
        minimum_network_samples: config.minimum_network_samples,
        blocking_kind: None,
        observed_p99_ms: None,
    }
}

fn blocked_gate(
    config: &LatencyConfig,
    reason: &str,
    kind: Option<LatencyKind>,
    observed_p99_ns: Option<u64>,
) -> LatencyGateSnapshot {
    LatencyGateSnapshot {
        enabled: true,
        enforced: true,
        trading_allowed: false,
        reason: reason.to_owned(),
        max_acceptable_p99_ms: config.max_acceptable_p99_ms,
        minimum_samples: config.minimum_samples,
        minimum_network_samples: config.minimum_network_samples,
        blocking_kind: kind.map(|value| value.name().to_owned()),
        observed_p99_ms: observed_p99_ns.map(|value| value as f64 / 1_000_000.0),
    }
}

fn distribution(
    samples: &VecDeque<LatencySample>,
    total_samples: u64,
    now_ns: u64,
) -> LatencyDistribution {
    if samples.is_empty() {
        return LatencyDistribution {
            total_samples,
            ..LatencyDistribution::default()
        };
    }
    let mut values: Vec<u64> = samples.iter().map(|sample| sample.value_ns).collect();
    values.sort_unstable();
    let latest = samples
        .iter()
        .max_by_key(|sample| sample.observed_ns)
        .expect("samples is non-empty");
    LatencyDistribution {
        total_samples,
        window_samples: values.len() as u64,
        last_ns: Some(latest.value_ns),
        min_ns: values.first().copied(),
        p50_ns: percentile(&values, 500),
        p95_ns: percentile(&values, 950),
        p99_ns: percentile(&values, 990),
        p999_ns: percentile(&values, 999),
        max_ns: values.last().copied(),
        sample_age_ms: Some(now_ns.saturating_sub(latest.observed_ns) / 1_000_000),
    }
}

fn percentile(sorted: &[u64], permille: usize) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = sorted.len().saturating_mul(permille).div_ceil(1_000);
    sorted.get(rank.saturating_sub(1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(capacity: usize, window_seconds: u64) -> LatencyConfig {
        LatencyConfig {
            gate_enabled: false,
            queue_capacity: capacity,
            window_seconds,
            hot_sample_every: 16,
            ..LatencyConfig::default()
        }
    }

    #[test]
    fn rolling_distribution_and_percentiles_are_exact() {
        let config = test_config(16, 5);
        let monitor = LatencyMonitor::new("SYN", 1, &config, false);
        for (index, value) in [10, 20, 30, 40, 50].into_iter().enumerate() {
            monitor.record(
                LatencyKind::HotDecision,
                value,
                9_000_000_000 + index as u64,
            );
        }
        let mut aggregator = LatencyAggregator::new("SYN", 1, config, false);
        let snapshot = aggregator.drain(&monitor, 10_000_000_000);
        let hot = &snapshot.distributions["hot_decision"];
        assert_eq!(hot.total_samples, 5);
        assert_eq!(hot.window_samples, 5);
        assert_eq!(hot.min_ns, Some(10));
        assert_eq!(hot.p50_ns, Some(30));
        assert_eq!(hot.p95_ns, Some(50));
        assert_eq!(hot.p99_ns, Some(50));
        assert_eq!(hot.p999_ns, Some(50));
        assert_eq!(hot.max_ns, Some(50));
    }

    #[test]
    fn old_samples_leave_the_rolling_window() {
        let config = test_config(4, 1);
        let monitor = LatencyMonitor::new("SYN", 1, &config, false);
        monitor.record(LatencyKind::HotDecision, 10, 1);
        monitor.record(LatencyKind::HotDecision, 20, 2_000_000_000);
        let mut aggregator = LatencyAggregator::new("SYN", 1, config, false);
        let snapshot = aggregator.drain(&monitor, 2_500_000_000);
        let hot = &snapshot.distributions["hot_decision"];
        assert_eq!(hot.total_samples, 2);
        assert_eq!(hot.window_samples, 1);
        assert_eq!(hot.last_ns, Some(20));
    }

    #[test]
    fn full_queue_counts_dropped_samples() {
        let config = test_config(1, 1);
        let monitor = LatencyMonitor::new("SYN", 1, &config, false);
        monitor.record(LatencyKind::HotDecision, 1, 1);
        monitor.record(LatencyKind::HotDecision, 2, 2);
        assert_eq!(monitor.dropped_samples.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn observer_publishes_atomic_current_snapshot_and_stops() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("latency.json");
        let clock = Arc::new(ProcessClock::default());
        let config = test_config(16, 60);
        let monitor = Arc::new(LatencyMonitor::new("SYN", 1, &config, false));
        let observer = LatencyObserver::spawn(
            monitor.clone(),
            clock.clone(),
            "SYN".to_owned(),
            1,
            config,
            false,
            Duration::from_millis(10),
            path.clone(),
        )
        .unwrap();
        let observed_ns = clock.now_ns();
        monitor.record(LatencyKind::HotDecision, 42, observed_ns);
        std::thread::sleep(Duration::from_millis(30));
        observer.stop().unwrap();
        let persisted: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(persisted["distributions"]["hot_decision"]["last_ns"], 42);
        assert_eq!(
            monitor.snapshot().distributions["hot_decision"].last_ns,
            Some(42)
        );
    }

    #[test]
    fn gate_warms_up_opens_and_blocks_on_bad_ack_latency() {
        let config = LatencyConfig {
            gate_enabled: true,
            max_acceptable_p99_ms: 1.0,
            minimum_samples: 3,
            minimum_network_samples: 1,
            max_sample_age_ms: 10_000,
            queue_capacity: 16,
            window_seconds: 60,
            ..LatencyConfig::default()
        };
        let monitor = LatencyMonitor::new("SYN", 1, &config, true);
        let mut aggregator = LatencyAggregator::new("SYN", 1, config, true);
        for observed_ns in [1_000, 2_000, 3_000] {
            monitor.record(LatencyKind::MarketEventDispatch, 100_000, observed_ns);
        }
        monitor.record(LatencyKind::PublicWsPingRtt, 500_000, 3_500);
        monitor.record(LatencyKind::AccountWsPingRtt, 500_000, 3_600);
        let open = aggregator.drain(&monitor, 4_000);
        assert!(open.gate.trading_allowed);
        assert_eq!(open.gate.reason, "ok");

        monitor.record(LatencyKind::SubmitToAck, 2_000_000, 5_000);
        let blocked = aggregator.drain(&monitor, 6_000);
        assert!(!blocked.gate.trading_allowed);
        assert_eq!(blocked.gate.reason, "p99_exceeded");
        assert_eq!(blocked.gate.blocking_kind.as_deref(), Some("submit_to_ack"));
    }

    #[test]
    fn configured_gate_is_observed_but_not_enforced_outside_production_live() {
        let config = LatencyConfig {
            gate_enabled: true,
            max_acceptable_p99_ms: 0.001,
            minimum_samples: 1,
            queue_capacity: 4,
            ..LatencyConfig::default()
        };
        let monitor = LatencyMonitor::new("SYN", 1, &config, false);
        monitor.record(LatencyKind::MarketEventDispatch, 10_000_000, 1_000);
        let mut aggregator = LatencyAggregator::new("SYN", 1, config, false);
        let snapshot = aggregator.drain(&monitor, 2_000);
        assert!(snapshot.gate.enabled);
        assert!(!snapshot.gate.enforced);
        assert!(snapshot.gate.trading_allowed);
        assert!(monitor.trading_allowed());
        assert_eq!(snapshot.gate.reason, "not_enforced_in_mode");
    }

    #[test]
    fn production_gate_blocks_on_slow_websocket_rtt() {
        let config = LatencyConfig {
            gate_enabled: true,
            max_acceptable_p99_ms: 150.0,
            minimum_samples: 2,
            minimum_network_samples: 1,
            max_sample_age_ms: 10_000,
            queue_capacity: 8,
            window_seconds: 60,
            ..LatencyConfig::default()
        };
        let monitor = LatencyMonitor::new("SYN", 1, &config, true);
        monitor.record(LatencyKind::MarketEventDispatch, 10_000, 1_000);
        monitor.record(LatencyKind::MarketEventDispatch, 20_000, 2_000);
        monitor.record(LatencyKind::PublicWsPingRtt, 500_000_000, 3_000);
        monitor.record(LatencyKind::AccountWsPingRtt, 10_000_000, 3_500);
        let mut aggregator = LatencyAggregator::new("SYN", 1, config, true);
        let snapshot = aggregator.drain(&monitor, 4_000);
        assert!(!snapshot.gate.trading_allowed);
        assert_eq!(snapshot.gate.reason, "p99_exceeded");
        assert_eq!(
            snapshot.gate.blocking_kind.as_deref(),
            Some("public_ws_ping_rtt")
        );
        assert_eq!(snapshot.gate.observed_p99_ms, Some(500.0));
    }
}
