use crate::types::{unix_ms, Side};
use anyhow::{bail, Context, Result};
use fs2::FileExt;
use redb::{Database, Durability, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

const LEGACY_STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("live_state");
const META_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("live_meta_v2");
const ORDER_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("live_orders_v2");
const LEGACY_FILL_TABLE_V2: TableDefinition<&str, u8> = TableDefinition::new("live_fill_keys_v2");
const LEGACY_FUNDING_TABLE_V2: TableDefinition<&str, u8> =
    TableDefinition::new("live_funding_keys_v2");
const FILL_TABLE: TableDefinition<&str, u8> = TableDefinition::new("live_fill_keys_v3");
const FUNDING_TABLE: TableDefinition<&str, u8> = TableDefinition::new("live_funding_keys_v3");
const STATE_KEY: &str = "state";
const NONCE_RESERVATION_SIZE: u64 = 1_024;
/// Prefetch the next nonce range once this many nonces remain in the current
/// one, so the fsync that makes a range durable happens off the dispatch path.
const NONCE_PREFETCH_HEADROOM: u64 = NONCE_RESERVATION_SIZE / 4;
const STATE_SCHEMA_VERSION: u32 = 3;

/// A dedup key stamped with the exchange time of the event it deduplicates.
///
/// Ordering by time first makes retention pruning an `O(log n)` range split
/// instead of a scan. Legacy (schema ≤ 2) keys carried no time and deserialize
/// with time `0`, so they fall to the first checkpoint advance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct TimedKey(pub u64, pub String);

impl<'de> Deserialize<'de> for TimedKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Compat {
            Timed(u64, String),
            Legacy(String),
        }
        Ok(match Compat::deserialize(deserializer)? {
            Compat::Timed(time, key) => Self(time, key),
            Compat::Legacy(key) => Self(0, key),
        })
    }
}

impl TimedKey {
    /// Fixed-width time prefix keeps redb's lexicographic key order equal to
    /// the `(time, key)` order used in memory.
    fn encode(&self) -> String {
        format!("{:020}|{}", self.0, self.1)
    }

    fn decode(raw: &str) -> Self {
        match raw.split_once('|') {
            Some((time, key)) => Self(time.parse().unwrap_or(0), key.to_owned()),
            None => Self(0, raw.to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveOrderStatus {
    Prepared,
    Sent,
    Resting,
    PartiallyFilled,
    CancelPending,
    Filled,
    Rejected,
    Canceled,
    UnknownOutcome,
}

impl LiveOrderStatus {
    pub const fn terminal(self) -> bool {
        matches!(self, Self::Filled | Self::Rejected | Self::Canceled)
    }

    pub fn can_transition_to(self, next: Self) -> bool {
        if self == next {
            return true;
        }
        match self {
            Self::Prepared => matches!(next, Self::Sent | Self::Rejected | Self::UnknownOutcome),
            Self::Sent => matches!(
                next,
                Self::Resting
                    | Self::PartiallyFilled
                    | Self::CancelPending
                    | Self::Filled
                    | Self::Rejected
                    | Self::Canceled
                    | Self::UnknownOutcome
            ),
            Self::Resting => matches!(
                next,
                Self::PartiallyFilled
                    | Self::CancelPending
                    | Self::Filled
                    | Self::Canceled
                    | Self::UnknownOutcome
            ),
            Self::PartiallyFilled => matches!(
                next,
                Self::CancelPending | Self::Filled | Self::Canceled | Self::UnknownOutcome
            ),
            Self::CancelPending => matches!(
                next,
                Self::Resting
                    | Self::PartiallyFilled
                    | Self::Filled
                    | Self::Canceled
                    | Self::UnknownOutcome
            ),
            Self::UnknownOutcome => !matches!(next, Self::Prepared | Self::Sent),
            Self::Filled | Self::Rejected | Self::Canceled => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersistedLiveOrder {
    pub cloid: String,
    pub quote_seq: u64,
    pub side: Side,
    pub px_units: i64,
    pub original_qty_units: i64,
    pub remaining_qty_units: i64,
    pub reduce_only: bool,
    pub status: LiveOrderStatus,
    pub nonce: Option<u64>,
    pub transport_id: Option<u64>,
    pub oid: Option<u64>,
    pub prepared_at_ms: u64,
    /// Local wall-clock time of the last mutation. Drives the in-flight
    /// cancel guard and terminal-order pruning; never compared with venue
    /// timestamps.
    pub last_update_ms: u64,
    /// Exchange time of the newest venue status or fill applied to this order.
    /// The only field that may order WebSocket `orderUpdates` — the local and
    /// venue clocks differ by an unknown offset, so comparing across them
    /// silently dropped legitimate acknowledgements and cancels.
    #[serde(default)]
    pub last_venue_status_ms: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceBudgetState {
    pub started_at_ms: u64,
    pub starting_equity_usdc: f64,
    pub turnover_usdc: f64,
    pub realized_pnl_usdc: f64,
    pub deadman_triggers: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedLiveState {
    pub schema_version: u32,
    pub symbol: String,
    pub account: String,
    pub agent: String,
    pub config_fingerprint: String,
    pub metadata_fingerprint: String,
    pub event_checkpoint_ms: u64,
    pub nonce_reserved_through: u64,
    pub next_cloid_sequence: u64,
    pub inventory_unit: Option<i64>,
    pub orders: BTreeMap<String, PersistedLiveOrder>,
    pub processed_fill_keys: BTreeSet<TimedKey>,
    pub processed_funding_keys: BTreeSet<TimedKey>,
    #[serde(default)]
    pub cumulative_realized_pnl_usdc: f64,
    #[serde(default)]
    pub cumulative_fees_usdc: f64,
    #[serde(default)]
    pub cumulative_funding_usdc: f64,
    #[serde(default)]
    pub pnl_day: u64,
    #[serde(default)]
    pub daily_realized_pnl_usdc: f64,
    /// Consecutive losing closes. Streaks span days, matching `DryRunBackend`,
    /// so the daily P&L roll deliberately leaves this alone.
    #[serde(default)]
    pub consecutive_losses: u32,
    pub campaign: AcceptanceBudgetState,
}

/// Scalar risk inputs the hot path needs every time execution state moves.
///
/// Read through [`LiveStateStore::risk_scalars`] rather than `load_required`:
/// this is on a per-event path, and cloning the order and dedup collections
/// there is what makes per-event cost grow with session length.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RiskScalars {
    pub daily_realized_pnl_usdc: f64,
    pub consecutive_losses: u32,
}

impl PersistedLiveState {
    fn new(
        symbol: &str,
        account: &str,
        agent: &str,
        config_fingerprint: &str,
        metadata_fingerprint: &str,
    ) -> Self {
        Self {
            schema_version: STATE_SCHEMA_VERSION,
            symbol: symbol.to_owned(),
            account: account.to_owned(),
            agent: agent.to_owned(),
            config_fingerprint: config_fingerprint.to_owned(),
            metadata_fingerprint: metadata_fingerprint.to_owned(),
            event_checkpoint_ms: unix_ms(),
            nonce_reserved_through: 0,
            next_cloid_sequence: 1,
            inventory_unit: None,
            orders: BTreeMap::new(),
            processed_fill_keys: BTreeSet::new(),
            processed_funding_keys: BTreeSet::new(),
            cumulative_realized_pnl_usdc: 0.0,
            cumulative_fees_usdc: 0.0,
            cumulative_funding_usdc: 0.0,
            pnl_day: unix_ms() / 86_400_000,
            daily_realized_pnl_usdc: 0.0,
            consecutive_losses: 0,
            campaign: AcceptanceBudgetState::default(),
        }
    }
}

pub struct LiveStateStore {
    _process_lock: File,
    _signer_lock: File,
    path: PathBuf,
    state: Arc<Mutex<PersistedLiveState>>,
    persistence_tx: SyncSender<PersistCommand>,
    persistence_error: Arc<Mutex<Option<String>>>,
    persistence_thread: Mutex<Option<JoinHandle<()>>>,
}

enum PersistCommand {
    Wake,
    Flush(SyncSender<std::result::Result<(), String>>),
    Shutdown,
}

/// How a process intends to use the durable store, which decides what
/// configuration drift it is allowed to open across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionIntent {
    /// Normal quoting. It may adopt a new configuration only when there is no
    /// exposure to inherit anywhere — neither on the venue nor in the store —
    /// because the sizes, thresholds and risk limits it would reconcile
    /// against are not the ones the position was opened with.
    Quote { venue_is_flat: bool },
    /// `live-flatten`. It only cancels and reduces, never opens exposure, so it
    /// must be able to open exactly the drifted-with-exposure store a quoting
    /// session refuses — that state is the one it exists to clear.
    ReduceOnly,
}

impl LiveStateStore {
    pub fn open(
        path: &Path,
        symbol: &str,
        account: &str,
        agent: &str,
        config_fingerprint: &str,
        metadata_fingerprint: &str,
        intent: SessionIntent,
    ) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let lock_path = path.with_extension("lock");
        let process_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("cannot open live-state lock {}", lock_path.display()))?;
        process_lock.try_lock_exclusive().with_context(|| {
            format!(
                "another live process owns the state lock {}",
                lock_path.display()
            )
        })?;
        let signer_lock_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("signer-{}.lock", agent.to_ascii_lowercase()));
        let signer_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&signer_lock_path)
            .with_context(|| {
                format!(
                    "cannot open signer nonce lock {}",
                    signer_lock_path.display()
                )
            })?;
        signer_lock.try_lock_exclusive().with_context(|| {
            format!(
                "another live process is using signer {} via {}",
                agent,
                signer_lock_path.display()
            )
        })?;
        let database = Database::create(path)
            .with_context(|| format!("cannot open live state {}", path.display()))?;
        let current = load_database(&database)?;
        let mut current = if let Some(mut current) = current {
            if !matches!(current.schema_version, 1 | 2 | STATE_SCHEMA_VERSION)
                || current.symbol != symbol
                || !current.account.eq_ignore_ascii_case(account)
                || !current.agent.eq_ignore_ascii_case(agent)
            {
                bail!("live-state identity/configuration mismatch; refusing unsafe reuse");
            }
            current.schema_version = STATE_SCHEMA_VERSION;
            if current.config_fingerprint != config_fingerprint {
                let durable_exposure = current
                    .orders
                    .values()
                    .any(|order| !order.status.terminal());
                match intent {
                    SessionIntent::Quote { venue_is_flat } => {
                        if !venue_is_flat || durable_exposure {
                            bail!("live-state configuration changed while durable exposure exists");
                        }
                    }
                    // A half-finished flatten leaves exposure behind, so the
                    // fingerprint is not adopted below and a quoting session is
                    // still refused on the next start.
                    SessionIntent::ReduceOnly => {}
                }
                if !durable_exposure {
                    config_fingerprint.clone_into(&mut current.config_fingerprint);
                    metadata_fingerprint.clone_into(&mut current.metadata_fingerprint);
                }
            }
            current
        } else {
            PersistedLiveState::new(
                symbol,
                account,
                agent,
                config_fingerprint,
                metadata_fingerprint,
            )
        };
        current.schema_version = STATE_SCHEMA_VERSION;
        persist_snapshot(&database, &current, Durability::Immediate)?;

        let initial_image = current.clone();
        let state = Arc::new(Mutex::new(current));
        let persistence_error = Arc::new(Mutex::new(None));
        let (persistence_tx, persistence_rx) = mpsc::sync_channel(1);
        let writer_state = state.clone();
        let writer_error = persistence_error.clone();
        let persistence_thread = thread::Builder::new()
            .name("live-state-writer".to_owned())
            .spawn(move || {
                run_persistence_writer(
                    &database,
                    &writer_state,
                    &writer_error,
                    &persistence_rx,
                    initial_image,
                );
            })?;
        Ok(Self {
            _process_lock: process_lock,
            _signer_lock: signer_lock,
            path: path.to_owned(),
            state,
            persistence_tx,
            persistence_error,
            persistence_thread: Mutex::new(Some(persistence_thread)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Deep-clone the whole persisted state, growing collections included.
    ///
    /// Reserve this for genuinely whole-state consumers (reconciliation,
    /// acceptance reporting, tests). Anything on a per-event or per-order path
    /// must use [`Self::with_state`] or a scalar accessor instead — this clone
    /// is what made per-event cost grow with session length.
    pub fn load_required(&self) -> Result<PersistedLiveState> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state memory lock poisoned"))
            .map(|state| state.clone())
    }

    /// Read under the lock and project out only what is needed, cloning
    /// nothing by default. The closure must be short and must never block or
    /// await: it holds the same mutex the persistence writer snapshots under.
    pub fn with_state<T>(&self, read: impl FnOnce(&PersistedLiveState) -> T) -> Result<T> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state memory lock poisoned"))?;
        Ok(read(&state))
    }

    pub fn inventory_unit(&self) -> Result<Option<i64>> {
        self.with_state(|state| state.inventory_unit)
    }

    pub fn campaign(&self) -> Result<AcceptanceBudgetState> {
        self.with_state(|state| state.campaign)
    }

    /// Apply a mutation and wake the persistence writer.
    ///
    /// Contract: the closure must not fail after mutating. There is no
    /// rollback — an `Err` returned mid-mutation leaves the partial mutation
    /// in place (and it will still be persisted by the next successful wake).
    /// Do all fallible work (parsing, validation) *before* calling `update`,
    /// and keep the closure itself infallible or fail-first.
    pub fn update<T>(
        &self,
        update: impl FnOnce(&mut PersistedLiveState) -> Result<T>,
    ) -> Result<T> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state memory lock poisoned"))?;
        let result = update(&mut state)?;
        drop(state);
        self.signal_persistence()?;
        Ok(result)
    }

    /// Read the scalar risk inputs, rolling the P&L day when the calendar has
    /// advanced. Takes the lock once and clones nothing; the writer is woken
    /// only when the day actually rolled.
    pub fn risk_scalars(&self, day: u64) -> Result<RiskScalars> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state memory lock poisoned"))?;
        let rolled = state.pnl_day != day;
        if rolled {
            state.pnl_day = day;
            state.daily_realized_pnl_usdc = 0.0;
        }
        let scalars = RiskScalars {
            daily_realized_pnl_usdc: state.daily_realized_pnl_usdc,
            consecutive_losses: state.consecutive_losses,
        };
        drop(state);
        if rolled {
            self.signal_persistence()?;
        }
        Ok(scalars)
    }

    /// Coalescing wake for the persistence writer. A full channel already means
    /// a snapshot is pending, so dropping the signal is correct.
    fn signal_persistence(&self) -> Result<()> {
        match self.persistence_tx.try_send(PersistCommand::Wake) {
            Ok(()) | Err(TrySendError::Full(PersistCommand::Wake)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                bail!("live-state persistence writer stopped unexpectedly")
            }
            Err(TrySendError::Full(_)) => unreachable!("only Wake uses try_send"),
        }
    }

    pub fn flush(&self) -> Result<()> {
        self.check_persistence_error()?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.persistence_tx
            .send(PersistCommand::Flush(reply_tx))
            .map_err(|_| anyhow::anyhow!("live-state persistence writer stopped before flush"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("live-state persistence flush acknowledgement lost"))?
            .map_err(anyhow::Error::msg)?;
        self.check_persistence_error()
    }

    fn check_persistence_error(&self) -> Result<()> {
        let error = self
            .persistence_error
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state persistence error lock poisoned"))?;
        if let Some(error) = error.as_deref() {
            bail!("live-state persistence failed: {error}");
        }
        Ok(())
    }

    pub fn persistence_healthy(&self) -> bool {
        self.persistence_error
            .lock()
            .map(|error| error.is_none())
            .unwrap_or(false)
    }

    /// Last-resort nonce for cancel/flatten when the persistence device is
    /// degraded. It advances beyond the entire durable reservation, so it
    /// cannot collide with the session actor's in-memory lease. This is used
    /// only for risk reduction; normal quoting still requires durable state.
    pub fn emergency_nonce(&self) -> Result<u64> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state memory lock poisoned"))?;
        let nonce = unix_ms().max(state.nonce_reserved_through.saturating_add(1));
        state.nonce_reserved_through = nonce;
        Ok(nonce)
    }

    pub fn next_cloid_sequence(&self) -> Result<u64> {
        self.update(|state| {
            let sequence = state.next_cloid_sequence;
            state.next_cloid_sequence = state.next_cloid_sequence.saturating_add(1);
            Ok(sequence)
        })
    }

    pub fn reserve_nonce_range(&self) -> Result<(u64, u64)> {
        let range = self.update(|state| {
            let start = unix_ms().max(state.nonce_reserved_through.saturating_add(1));
            let end = start.saturating_add(NONCE_RESERVATION_SIZE - 1);
            state.nonce_reserved_through = end;
            Ok((start, end))
        })?;
        // A nonce range is the only state that must be durable before use.
        self.flush()?;
        Ok(range)
    }

    /// Durably advance the fill/funding replay horizon, then prune dedup keys
    /// strictly below it. Returns the number of keys removed.
    ///
    /// The checkpoint only ever moves forward — moving it back would re-admit
    /// fills that predate the session. Ordering is load-bearing: the advanced
    /// checkpoint is fsynced *before* any key is removed, because pruning
    /// first would let a crash re-process (and double-count) old fills whose
    /// keys are gone while the old checkpoint still admits them.
    pub fn advance_checkpoint_and_prune(&self, candidate_ms: u64) -> Result<usize> {
        let advanced = self.update(|state| {
            if candidate_ms > state.event_checkpoint_ms {
                state.event_checkpoint_ms = candidate_ms;
                return Ok(true);
            }
            Ok(false)
        })?;
        if advanced {
            self.flush()?;
        }
        self.update(|state| {
            let cutoff = state.event_checkpoint_ms;
            let mut removed = 0_usize;
            for keys in [
                &mut state.processed_fill_keys,
                &mut state.processed_funding_keys,
            ] {
                let before = keys.len();
                // Keys at exactly the cutoff stay: events at `checkpoint` are
                // still admitted by the `time < checkpoint` gates.
                *keys = keys.split_off(&TimedKey(cutoff, String::new()));
                removed += before - keys.len();
            }
            Ok(removed)
        })
    }

    /// Drop terminal orders whose last update is older than `older_than_ms`.
    ///
    /// After an authoritative reconcile the venue is the source of truth for
    /// anything still open, and the JSONL event log keeps full forensics, so
    /// aged terminal entries only cost clone and persist time.
    pub fn prune_terminal_orders(&self, older_than_ms: u64) -> Result<usize> {
        self.update(|state| {
            let before = state.orders.len();
            state.orders.retain(|_, order| {
                !order.status.terminal() || order.last_update_ms >= older_than_ms
            });
            Ok(before - state.orders.len())
        })
    }
}

impl Drop for LiveStateStore {
    fn drop(&mut self) {
        let _ = self.flush();
        let _ = self.persistence_tx.send(PersistCommand::Shutdown);
        if let Ok(mut handle) = self.persistence_thread.lock() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn load_database(database: &Database) -> Result<Option<PersistedLiveState>> {
    let read = database.begin_read()?;
    if let Ok(table) = read.open_table(META_TABLE) {
        if let Some(value) = table.get(STATE_KEY)? {
            let mut state: PersistedLiveState = serde_json::from_slice(value.value())?;
            if let Ok(orders) = read.open_table(ORDER_TABLE) {
                for entry in orders.iter()? {
                    let (key, value) = entry?;
                    let order: PersistedLiveOrder = serde_json::from_slice(value.value())?;
                    state.orders.insert(key.value().to_owned(), order);
                }
            }
            if let Ok(fills) = read.open_table(FILL_TABLE) {
                for entry in fills.iter()? {
                    let (key, _) = entry?;
                    state
                        .processed_fill_keys
                        .insert(TimedKey::decode(key.value()));
                }
            }
            if let Ok(funding) = read.open_table(FUNDING_TABLE) {
                for entry in funding.iter()? {
                    let (key, _) = entry?;
                    state
                        .processed_funding_keys
                        .insert(TimedKey::decode(key.value()));
                }
            }
            // Schema-2 stores carried untimed keys in the v2 tables; adopt them
            // at time 0 so they fall to the first checkpoint advance.
            if let Ok(fills) = read.open_table(LEGACY_FILL_TABLE_V2) {
                for entry in fills.iter()? {
                    let (key, _) = entry?;
                    state
                        .processed_fill_keys
                        .insert(TimedKey(0, key.value().to_owned()));
                }
            }
            if let Ok(funding) = read.open_table(LEGACY_FUNDING_TABLE_V2) {
                for entry in funding.iter()? {
                    let (key, _) = entry?;
                    state
                        .processed_funding_keys
                        .insert(TimedKey(0, key.value().to_owned()));
                }
            }
            return Ok(Some(state));
        }
    }

    let legacy = match read.open_table(LEGACY_STATE_TABLE) {
        Ok(table) => table,
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let Some(value) = legacy.get(STATE_KEY)? else {
        return Ok(None);
    };
    Ok(Some(serde_json::from_slice(value.value())?))
}

fn metadata_blob(state: &PersistedLiveState) -> Result<Vec<u8>> {
    let mut metadata = state.clone();
    metadata.schema_version = STATE_SCHEMA_VERSION;
    metadata.orders.clear();
    metadata.processed_fill_keys.clear();
    metadata.processed_funding_keys.clear();
    Ok(serde_json::to_vec(&metadata)?)
}

/// Full rewrite of every table. Used once at `open()` as the consistency point
/// (it also migrates away legacy tables); steady-state persists go through
/// [`persist_delta`].
fn persist_snapshot(
    database: &Database,
    state: &PersistedLiveState,
    durability: Durability,
) -> Result<()> {
    let metadata_bytes = metadata_blob(state)?;
    let order_bytes = state
        .orders
        .iter()
        .map(|(key, order)| Ok((key, serde_json::to_vec(order)?)))
        .collect::<Result<Vec<_>>>()?;

    let mut write = database.begin_write()?;
    write.set_durability(durability)?;
    {
        let mut table = write.open_table(META_TABLE)?;
        table.insert(STATE_KEY, metadata_bytes.as_slice())?;
    }
    {
        let mut table = write.open_table(ORDER_TABLE)?;
        table.retain(|_, _| false)?;
        for (key, value) in order_bytes {
            table.insert(key.as_str(), value.as_slice())?;
        }
    }
    {
        let mut table = write.open_table(FILL_TABLE)?;
        table.retain(|_, _| false)?;
        for key in &state.processed_fill_keys {
            table.insert(key.encode().as_str(), 1)?;
        }
    }
    {
        let mut table = write.open_table(FUNDING_TABLE)?;
        table.retain(|_, _| false)?;
        for key in &state.processed_funding_keys {
            table.insert(key.encode().as_str(), 1)?;
        }
    }
    if let Ok(mut legacy) = write.open_table(LEGACY_STATE_TABLE) {
        legacy.remove(STATE_KEY)?;
    }
    // Schema-2 key tables were adopted into the v3 tables at load; drop them so
    // they are not re-adopted (and re-timestamped at 0) on the next open.
    for stale in [LEGACY_FILL_TABLE_V2, LEGACY_FUNDING_TABLE_V2] {
        match write.delete_table(stale) {
            Ok(_) | Err(redb::TableError::TableDoesNotExist(_)) => {}
            Err(error) => return Err(error.into()),
        }
    }
    write.commit()?;
    Ok(())
}

struct PersistDelta {
    metadata: PersistedLiveState,
    order_upserts: Vec<(String, PersistedLiveOrder)>,
    order_removals: Vec<String>,
    fill_additions: Vec<TimedKey>,
    fill_removals: Vec<TimedKey>,
    funding_additions: Vec<TimedKey>,
    funding_removals: Vec<TimedKey>,
}

fn metadata_only(state: &PersistedLiveState) -> PersistedLiveState {
    PersistedLiveState {
        schema_version: state.schema_version,
        symbol: state.symbol.clone(),
        account: state.account.clone(),
        agent: state.agent.clone(),
        config_fingerprint: state.config_fingerprint.clone(),
        metadata_fingerprint: state.metadata_fingerprint.clone(),
        event_checkpoint_ms: state.event_checkpoint_ms,
        nonce_reserved_through: state.nonce_reserved_through,
        next_cloid_sequence: state.next_cloid_sequence,
        inventory_unit: state.inventory_unit,
        orders: BTreeMap::new(),
        processed_fill_keys: BTreeSet::new(),
        processed_funding_keys: BTreeSet::new(),
        cumulative_realized_pnl_usdc: state.cumulative_realized_pnl_usdc,
        cumulative_fees_usdc: state.cumulative_fees_usdc,
        cumulative_funding_usdc: state.cumulative_funding_usdc,
        pnl_day: state.pnl_day,
        daily_realized_pnl_usdc: state.daily_realized_pnl_usdc,
        consecutive_losses: state.consecutive_losses,
        campaign: state.campaign,
    }
}

fn build_persist_delta(
    previous: &PersistedLiveState,
    current: &PersistedLiveState,
) -> PersistDelta {
    PersistDelta {
        metadata: metadata_only(current),
        order_upserts: current
            .orders
            .iter()
            .filter(|(cloid, order)| previous.orders.get(*cloid) != Some(*order))
            .map(|(cloid, order)| (cloid.clone(), order.clone()))
            .collect(),
        order_removals: previous
            .orders
            .keys()
            .filter(|cloid| !current.orders.contains_key(*cloid))
            .cloned()
            .collect(),
        fill_additions: current
            .processed_fill_keys
            .difference(&previous.processed_fill_keys)
            .cloned()
            .collect(),
        fill_removals: previous
            .processed_fill_keys
            .difference(&current.processed_fill_keys)
            .cloned()
            .collect(),
        funding_additions: current
            .processed_funding_keys
            .difference(&previous.processed_funding_keys)
            .cloned()
            .collect(),
        funding_removals: previous
            .processed_funding_keys
            .difference(&current.processed_funding_keys)
            .cloned()
            .collect(),
    }
}

/// Persist only the small typed delta captured under the state mutex. Redb I/O
/// and serialization happen after the mutex is released; a growing history can
/// no longer turn one writer wake into a whole-state deep clone.
fn persist_delta(database: &Database, delta: &PersistDelta, durability: Durability) -> Result<()> {
    let metadata_bytes = metadata_blob(&delta.metadata)?;
    let mut write = database.begin_write()?;
    write.set_durability(durability)?;
    {
        let mut table = write.open_table(META_TABLE)?;
        table.insert(STATE_KEY, metadata_bytes.as_slice())?;
    }
    {
        let mut table = write.open_table(ORDER_TABLE)?;
        for cloid in &delta.order_removals {
            table.remove(cloid.as_str())?;
        }
        for (cloid, order) in &delta.order_upserts {
            table.insert(cloid.as_str(), serde_json::to_vec(order)?.as_slice())?;
        }
    }
    for (removed_keys, added_keys, definition) in [
        (&delta.fill_removals, &delta.fill_additions, FILL_TABLE),
        (
            &delta.funding_removals,
            &delta.funding_additions,
            FUNDING_TABLE,
        ),
    ] {
        let mut table = write.open_table(definition)?;
        for removed in removed_keys {
            table.remove(removed.encode().as_str())?;
        }
        for added in added_keys {
            table.insert(added.encode().as_str(), 1)?;
        }
    }
    write.commit()?;
    Ok(())
}

fn apply_delta(state: &mut PersistedLiveState, delta: &PersistDelta) {
    let orders = std::mem::take(&mut state.orders);
    let fill_keys = std::mem::take(&mut state.processed_fill_keys);
    let funding_keys = std::mem::take(&mut state.processed_funding_keys);
    *state = delta.metadata.clone();
    state.orders = orders;
    state.processed_fill_keys = fill_keys;
    state.processed_funding_keys = funding_keys;
    for cloid in &delta.order_removals {
        state.orders.remove(cloid);
    }
    for (cloid, order) in &delta.order_upserts {
        state.orders.insert(cloid.clone(), order.clone());
    }
    for key in &delta.fill_removals {
        state.processed_fill_keys.remove(key);
    }
    state
        .processed_fill_keys
        .extend(delta.fill_additions.iter().cloned());
    for key in &delta.funding_removals {
        state.processed_funding_keys.remove(key);
    }
    state
        .processed_funding_keys
        .extend(delta.funding_additions.iter().cloned());
}

fn run_persistence_writer(
    database: &Database,
    state: &Arc<Mutex<PersistedLiveState>>,
    persistence_error: &Arc<Mutex<Option<String>>>,
    receiver: &mpsc::Receiver<PersistCommand>,
    // The image `open()` wrote with the initial full snapshot; deltas start
    // from here.
    mut last_persisted: PersistedLiveState,
) {
    while let Ok(command) = receiver.recv() {
        let durability = match &command {
            PersistCommand::Wake => Durability::None,
            PersistCommand::Flush(_) | PersistCommand::Shutdown => Durability::Immediate,
        };
        let shutting_down = matches!(command, PersistCommand::Shutdown);
        let Ok(state_guard) = state.lock() else {
            store_writer_error(persistence_error, "live-state memory lock poisoned");
            break;
        };
        let delta = build_persist_delta(&last_persisted, &state_guard);
        drop(state_guard);
        let mut result = persist_delta(database, &delta, durability);
        for retry in 0_u32..5 {
            if result.is_ok() {
                break;
            }
            store_writer_error(
                persistence_error,
                &result.as_ref().expect_err("checked as error").to_string(),
            );
            std::thread::sleep(std::time::Duration::from_millis(
                50_u64.saturating_mul(1_u64 << retry).min(1_000),
            ));
            result = persist_delta(database, &delta, durability);
        }
        if result.is_ok() {
            apply_delta(&mut last_persisted, &delta);
            clear_writer_error(persistence_error);
        }
        if let PersistCommand::Flush(reply) = command {
            let acknowledgement = match &result {
                Ok(()) => Ok(()),
                Err(error) => Err(error.to_string()),
            };
            let _ = reply.send(acknowledgement);
        }
        if let Err(error) = result {
            store_writer_error(persistence_error, &error.to_string());
            if shutting_down {
                break;
            }
            // Stay alive. A later coalesced wake retries the newest complete
            // in-memory image after a transient disk/full/locking fault clears.
            continue;
        }
        if shutting_down {
            break;
        }
    }
}

fn store_writer_error(destination: &Mutex<Option<String>>, error: &str) {
    if let Ok(mut destination) = destination.lock() {
        *destination = Some(error.to_owned());
    }
}

fn clear_writer_error(destination: &Mutex<Option<String>>) {
    if let Ok(mut destination) = destination.lock() {
        *destination = None;
    }
}

pub struct DurableNonceManager {
    store: Arc<LiveStateStore>,
    next: u64,
    reserved_through: u64,
    prefetch: Option<mpsc::Receiver<Result<(u64, u64)>>>,
}

impl DurableNonceManager {
    pub fn new(store: Arc<LiveStateStore>) -> Result<Self> {
        let (next, reserved_through) = store.reserve_nonce_range()?;
        Ok(Self {
            store,
            next,
            reserved_through,
            prefetch: None,
        })
    }

    /// Hand out the next durable nonce.
    ///
    /// Reserving a range requires an fsync before any nonce in it is used, and
    /// this runs on the session actor's dispatch path — so the next range is
    /// reserved on a background thread while plenty of the current range
    /// remains. Only if a burst outruns the prefetch does this block on the
    /// in-flight reservation; the durability invariant is never relaxed.
    pub fn next_nonce(&mut self) -> Result<u64> {
        let remaining = self.reserved_through.saturating_sub(self.next);
        if remaining <= NONCE_PREFETCH_HEADROOM && self.prefetch.is_none() {
            let (result_tx, result_rx) = mpsc::channel();
            let store = self.store.clone();
            thread::Builder::new()
                .name("nonce-range-prefetch".to_owned())
                .spawn(move || {
                    let _ = result_tx.send(store.reserve_nonce_range());
                })?;
            self.prefetch = Some(result_rx);
        }
        if self.next > self.reserved_through {
            (self.next, self.reserved_through) = match self.prefetch.take() {
                Some(pending) => pending
                    .recv()
                    .map_err(|_| anyhow::anyhow!("nonce prefetch thread dropped its result"))??,
                None => self.store.reserve_nonce_range()?,
            };
        }
        let nonce = self.next;
        self.next = self.next.saturating_add(1);
        Ok(nonce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(directory: &tempfile::TempDir) -> LiveStateStore {
        LiveStateStore::open(
            &directory.path().join("live.redb"),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            SessionIntent::Quote {
                venue_is_flat: true,
            },
        )
        .unwrap()
    }

    fn order_template(
        cloid: &str,
        status: LiveOrderStatus,
        last_update_ms: u64,
    ) -> PersistedLiveOrder {
        PersistedLiveOrder {
            cloid: cloid.to_owned(),
            quote_seq: 1,
            side: Side::Buy,
            px_units: 100,
            original_qty_units: 10,
            remaining_qty_units: if status.terminal() { 0 } else { 10 },
            reduce_only: false,
            status,
            nonce: None,
            transport_id: None,
            oid: None,
            prepared_at_ms: 1,
            last_update_ms,
            last_venue_status_ms: 0,
            last_error: None,
        }
    }

    #[test]
    fn degraded_persistence_never_blocks_safety_reads_or_emergency_nonce() {
        let directory = tempfile::tempdir().unwrap();
        let store = open(&directory);
        *store.persistence_error.lock().unwrap() = Some("injected disk fault".to_owned());
        assert!(store.load_required().is_ok());
        assert!(store.with_state(|state| state.orders.len()).is_ok());
        let before = store
            .with_state(|state| state.nonce_reserved_through)
            .unwrap();
        let emergency = store.emergency_nonce().unwrap();
        assert!(emergency > before);
    }

    #[test]
    fn signer_lock_spans_different_state_files() {
        let directory = tempfile::tempdir().unwrap();
        let first = open(&directory);
        let second = LiveStateStore::open(
            &directory.path().join("other.redb"),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            SessionIntent::Quote {
                venue_is_flat: true,
            },
        );
        assert!(second.is_err());
        drop(first);
        assert!(LiveStateStore::open(
            &directory.path().join("other.redb"),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            SessionIntent::Quote {
                venue_is_flat: true,
            },
        )
        .is_ok());
    }

    /// The replay horizon only moves forward, and keys strictly below it are
    /// pruned while keys at or above it survive — including across a restart,
    /// which exercises the delta persistence of removals.
    #[test]
    fn checkpoint_advance_prunes_dedup_keys_and_survives_restart() {
        let directory = tempfile::tempdir().unwrap();
        {
            let store = open(&directory);
            let base = store.load_required().unwrap().event_checkpoint_ms;
            store
                .update(|state| {
                    for offset in 0..10_u64 {
                        state
                            .processed_fill_keys
                            .insert(TimedKey(base + offset, format!("fill-{offset}")));
                        state
                            .processed_funding_keys
                            .insert(TimedKey(base + offset, format!("funding-{offset}")));
                    }
                    Ok(())
                })
                .unwrap();
            let removed = store.advance_checkpoint_and_prune(base + 5).unwrap();
            assert_eq!(
                removed, 10,
                "five fills and five fundings fall below the horizon"
            );
            // A lower candidate must never move the checkpoint back.
            let removed = store.advance_checkpoint_and_prune(base).unwrap();
            assert_eq!(removed, 0);
            let state = store.load_required().unwrap();
            assert_eq!(state.event_checkpoint_ms, base + 5);
            assert_eq!(state.processed_fill_keys.len(), 5);
            assert!(state
                .processed_fill_keys
                .contains(&TimedKey(base + 5, "fill-5".to_owned())));
        }
        let reopened = open(&directory);
        let state = reopened.load_required().unwrap();
        assert_eq!(state.processed_fill_keys.len(), 5);
        assert_eq!(state.processed_funding_keys.len(), 5);
        assert!(!state.processed_fill_keys.contains(&TimedKey(
            state.event_checkpoint_ms - 1,
            "fill-4".to_owned()
        )));
    }

    #[test]
    fn config_drift_blocks_a_quoting_session_but_never_the_flatten_that_clears_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("live.redb");
        let quoting = SessionIntent::Quote {
            venue_is_flat: true,
        };
        let reopen = |fingerprint: &str, intent: SessionIntent| {
            LiveStateStore::open(
                &path,
                "CASHCAT",
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
                fingerprint,
                "meta",
                intent,
            )
        };
        let store = reopen("config-a", quoting).unwrap();
        store
            .update(|state| {
                state.orders.insert(
                    "resting".to_owned(),
                    order_template("resting", LiveOrderStatus::Resting, 1_000),
                );
                Ok(())
            })
            .unwrap();
        drop(store);

        // A quoting session must not adopt exposure opened under another config.
        // Even with the venue reporting flat, durable exposure blocks it.
        assert!(reopen("config-b", quoting).is_err());

        // `live-flatten` is exactly the tool for this state and must open.
        let store = reopen("config-b", SessionIntent::ReduceOnly).unwrap();
        // The fingerprint is not adopted while exposure remains, so a quoting
        // session is still refused if the flatten does not finish.
        assert_eq!(
            store.load_required().unwrap().config_fingerprint,
            "config-a"
        );
        store
            .update(|state| {
                state.orders.get_mut("resting").unwrap().status = LiveOrderStatus::Canceled;
                Ok(())
            })
            .unwrap();
        drop(store);

        // Once flat, the store migrates to the new configuration as before.
        let store = reopen("config-b", SessionIntent::ReduceOnly).unwrap();
        assert_eq!(
            store.load_required().unwrap().config_fingerprint,
            "config-b"
        );
    }

    #[test]
    fn terminal_orders_prune_by_age_but_working_orders_never_do() {
        let directory = tempfile::tempdir().unwrap();
        let store = open(&directory);
        store
            .update(|state| {
                state.orders.insert(
                    "old-filled".to_owned(),
                    order_template("old-filled", LiveOrderStatus::Filled, 1_000),
                );
                state.orders.insert(
                    "new-filled".to_owned(),
                    order_template("new-filled", LiveOrderStatus::Filled, 9_000),
                );
                state.orders.insert(
                    "old-resting".to_owned(),
                    order_template("old-resting", LiveOrderStatus::Resting, 1_000),
                );
                Ok(())
            })
            .unwrap();
        let removed = store.prune_terminal_orders(5_000).unwrap();
        assert_eq!(removed, 1);
        let orders = store.load_required().unwrap().orders;
        assert!(!orders.contains_key("old-filled"));
        assert!(orders.contains_key("new-filled"));
        assert!(
            orders.contains_key("old-resting"),
            "a non-terminal order is never pruned, whatever its age"
        );
    }

    /// The dedup-key sets and order map must not grow with session history
    /// once events pass beyond the retention horizon. Growth here was
    /// unbounded before schema 3: nothing ever advanced the checkpoint, so
    /// nothing could ever be pruned.
    #[test]
    fn durable_collections_stay_bounded_across_simulated_history() {
        let directory = tempfile::tempdir().unwrap();
        let store = open(&directory);
        let base = store.load_required().unwrap().event_checkpoint_ms;
        let retention = 1_000_u64;
        for batch in 0..20_u64 {
            let now = base + batch * 500;
            store
                .update(|state| {
                    for row in 0..50_u64 {
                        state
                            .processed_fill_keys
                            .insert(TimedKey(now, format!("fill-{batch}-{row}")));
                    }
                    Ok(())
                })
                .unwrap();
            store
                .advance_checkpoint_and_prune(now.saturating_sub(retention))
                .unwrap();
        }
        let state = store.load_required().unwrap();
        assert!(
            state.processed_fill_keys.len() <= 3 * 50,
            "bounded by retention window, found {}",
            state.processed_fill_keys.len()
        );
    }

    #[test]
    fn state_and_orders_survive_restart() {
        let directory = tempfile::tempdir().unwrap();
        {
            let store = open(&directory);
            store
                .update(|state| {
                    state.inventory_unit = Some(88);
                    state.orders.insert(
                        "0x00000000000000000000000000000001".to_owned(),
                        PersistedLiveOrder {
                            cloid: "0x00000000000000000000000000000001".to_owned(),
                            quote_seq: 1,
                            side: Side::Buy,
                            px_units: 100,
                            original_qty_units: 10,
                            remaining_qty_units: 10,
                            reduce_only: false,
                            status: LiveOrderStatus::Prepared,
                            nonce: None,
                            transport_id: None,
                            oid: None,
                            prepared_at_ms: 1,
                            last_update_ms: 1,
                            last_venue_status_ms: 0,
                            last_error: None,
                        },
                    );
                    Ok(())
                })
                .unwrap();
        }
        let reopened = open(&directory);
        let state = reopened.load_required().unwrap();
        assert_eq!(state.inventory_unit, Some(88));
        assert_eq!(state.orders.len(), 1);
    }

    #[test]
    fn restart_skips_the_entire_reserved_nonce_range() {
        let directory = tempfile::tempdir().unwrap();
        let first_reserved;
        {
            let store = Arc::new(open(&directory));
            let mut nonce = DurableNonceManager::new(store.clone()).unwrap();
            let first = nonce.next_nonce().unwrap();
            first_reserved = store.load_required().unwrap().nonce_reserved_through;
            assert!(first <= first_reserved);
        }
        let store = Arc::new(open(&directory));
        let mut restarted = DurableNonceManager::new(store).unwrap();
        assert!(restarted.next_nonce().unwrap() > first_reserved);
    }

    #[test]
    fn process_lock_and_identity_mismatch_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let first = open(&directory);
        assert!(LiveStateStore::open(
            first.path(),
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            SessionIntent::Quote {
                venue_is_flat: true
            },
        )
        .is_err());
        drop(first);
        assert!(LiveStateStore::open(
            &directory.path().join("live.redb"),
            "OTHER",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            SessionIntent::Quote {
                venue_is_flat: true
            },
        )
        .is_err());
    }

    #[test]
    fn concurrent_read_modify_write_updates_do_not_lose_state() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(open(&directory));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let store = store.clone();
            workers.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    store
                        .update(|state| {
                            state.next_cloid_sequence = state.next_cloid_sequence.saturating_add(1);
                            Ok(())
                        })
                        .unwrap();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(store.load_required().unwrap().next_cloid_sequence, 201);
    }

    #[test]
    fn legacy_blob_is_migrated_to_normalized_schema() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("live.redb");
        let mut legacy = PersistedLiveState::new(
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
        );
        legacy.schema_version = 1;
        legacy.orders.insert(
            "0x00000000000000000000000000000001".to_owned(),
            PersistedLiveOrder {
                cloid: "0x00000000000000000000000000000001".to_owned(),
                quote_seq: 1,
                side: Side::Buy,
                px_units: 100,
                original_qty_units: 10,
                remaining_qty_units: 10,
                reduce_only: false,
                status: LiveOrderStatus::Resting,
                nonce: Some(1),
                transport_id: Some(1),
                oid: Some(7),
                prepared_at_ms: 1,
                last_update_ms: 1,
                last_venue_status_ms: 0,
                last_error: None,
            },
        );
        {
            let database = Database::create(&path).unwrap();
            let mut write = database.begin_write().unwrap();
            write.set_durability(Durability::Immediate).unwrap();
            {
                let mut table = write.open_table(LEGACY_STATE_TABLE).unwrap();
                // Schema-1 blobs carried untimed dedup keys as plain strings;
                // patch the JSON so the wire shape matches that era exactly.
                let mut blob = serde_json::to_value(&legacy).unwrap();
                blob["processed_fill_keys"] = serde_json::json!(["fill-1"]);
                let bytes = serde_json::to_vec(&blob).unwrap();
                table.insert(STATE_KEY, bytes.as_slice()).unwrap();
            }
            write.commit().unwrap();
        }

        let store = open(&directory);
        let migrated = store.load_required().unwrap();
        assert_eq!(migrated.schema_version, STATE_SCHEMA_VERSION);
        assert_eq!(migrated.orders.len(), 1);
        assert!(migrated
            .processed_fill_keys
            .contains(&TimedKey(0, "fill-1".to_owned())));
        store.flush().unwrap();
        drop(store);

        let database = Database::create(&path).unwrap();
        let normalized = load_database(&database).unwrap().unwrap();
        assert_eq!(normalized.schema_version, STATE_SCHEMA_VERSION);
        assert_eq!(normalized.orders.len(), 1);
    }

    #[test]
    fn order_transition_table_rejects_terminal_resurrection() {
        assert!(LiveOrderStatus::Sent.can_transition_to(LiveOrderStatus::Resting));
        assert!(LiveOrderStatus::Resting.can_transition_to(LiveOrderStatus::CancelPending));
        assert!(LiveOrderStatus::CancelPending.can_transition_to(LiveOrderStatus::Canceled));
        for terminal in [
            LiveOrderStatus::Filled,
            LiveOrderStatus::Rejected,
            LiveOrderStatus::Canceled,
        ] {
            assert!(!terminal.can_transition_to(LiveOrderStatus::Resting));
            assert!(!terminal.can_transition_to(LiveOrderStatus::Prepared));
        }
    }

    #[test]
    fn unknown_newer_state_schema_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("live.redb");
        let mut future = PersistedLiveState::new(
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
        );
        future.schema_version = STATE_SCHEMA_VERSION + 1;
        {
            let database = Database::create(&path).unwrap();
            let write = database.begin_write().unwrap();
            {
                let mut table = write.open_table(META_TABLE).unwrap();
                let bytes = serde_json::to_vec(&future).unwrap();
                table.insert(STATE_KEY, bytes.as_slice()).unwrap();
            }
            write.commit().unwrap();
        }
        assert!(LiveStateStore::open(
            &path,
            "CASHCAT",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "config",
            "meta",
            SessionIntent::Quote {
                venue_is_flat: true
            },
        )
        .is_err());
    }
}
