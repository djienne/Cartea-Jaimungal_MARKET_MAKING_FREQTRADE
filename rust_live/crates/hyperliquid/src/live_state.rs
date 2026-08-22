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
const FILL_TABLE: TableDefinition<&str, u8> = TableDefinition::new("live_fill_keys_v2");
const FUNDING_TABLE: TableDefinition<&str, u8> = TableDefinition::new("live_funding_keys_v2");
const STATE_KEY: &str = "state";
const NONCE_RESERVATION_SIZE: u64 = 1_024;
const STATE_SCHEMA_VERSION: u32 = 2;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub last_update_ms: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    pub processed_fill_keys: BTreeSet<String>,
    pub processed_funding_keys: BTreeSet<String>,
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

impl LiveStateStore {
    pub fn open(
        path: &Path,
        symbol: &str,
        account: &str,
        agent: &str,
        config_fingerprint: &str,
        metadata_fingerprint: &str,
        allow_flat_config_migration: bool,
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
        let database = Database::create(path)
            .with_context(|| format!("cannot open live state {}", path.display()))?;
        let current = load_database(&database)?;
        let mut current = if let Some(mut current) = current {
            if !matches!(current.schema_version, 1 | STATE_SCHEMA_VERSION)
                || current.symbol != symbol
                || !current.account.eq_ignore_ascii_case(account)
                || !current.agent.eq_ignore_ascii_case(agent)
            {
                bail!("live-state identity/configuration mismatch; refusing unsafe reuse");
            }
            current.schema_version = STATE_SCHEMA_VERSION;
            if current.config_fingerprint != config_fingerprint {
                if !allow_flat_config_migration
                    || current
                        .orders
                        .values()
                        .any(|order| !order.status.terminal())
                {
                    bail!("live-state configuration changed while durable exposure exists");
                }
                config_fingerprint.clone_into(&mut current.config_fingerprint);
                metadata_fingerprint.clone_into(&mut current.metadata_fingerprint);
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

        let state = Arc::new(Mutex::new(current));
        let persistence_error = Arc::new(Mutex::new(None));
        let (persistence_tx, persistence_rx) = mpsc::sync_channel(1);
        let writer_state = state.clone();
        let writer_error = persistence_error.clone();
        let persistence_thread = thread::Builder::new()
            .name("live-state-writer".to_owned())
            .spawn(move || {
                run_persistence_writer(&database, &writer_state, &writer_error, &persistence_rx);
            })?;
        Ok(Self {
            _process_lock: process_lock,
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

    pub fn load_required(&self) -> Result<PersistedLiveState> {
        self.check_persistence_error()?;
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state memory lock poisoned"))
            .map(|state| state.clone())
    }

    pub fn update<T>(
        &self,
        update: impl FnOnce(&mut PersistedLiveState) -> Result<T>,
    ) -> Result<T> {
        self.check_persistence_error()?;
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
        self.check_persistence_error()?;
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
                    state.processed_fill_keys.insert(key.value().to_owned());
                }
            }
            if let Ok(funding) = read.open_table(FUNDING_TABLE) {
                for entry in funding.iter()? {
                    let (key, _) = entry?;
                    state.processed_funding_keys.insert(key.value().to_owned());
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

fn persist_snapshot(
    database: &Database,
    state: &PersistedLiveState,
    durability: Durability,
) -> Result<()> {
    let mut metadata = state.clone();
    metadata.schema_version = STATE_SCHEMA_VERSION;
    metadata.orders.clear();
    metadata.processed_fill_keys.clear();
    metadata.processed_funding_keys.clear();
    let metadata_bytes = serde_json::to_vec(&metadata)?;
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
            table.insert(key.as_str(), 1)?;
        }
    }
    {
        let mut table = write.open_table(FUNDING_TABLE)?;
        table.retain(|_, _| false)?;
        for key in &state.processed_funding_keys {
            table.insert(key.as_str(), 1)?;
        }
    }
    if let Ok(mut legacy) = write.open_table(LEGACY_STATE_TABLE) {
        legacy.remove(STATE_KEY)?;
    }
    write.commit()?;
    Ok(())
}

fn run_persistence_writer(
    database: &Database,
    state: &Arc<Mutex<PersistedLiveState>>,
    persistence_error: &Arc<Mutex<Option<String>>>,
    receiver: &mpsc::Receiver<PersistCommand>,
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
        let snapshot = state_guard.clone();
        drop(state_guard);
        let result = persist_snapshot(database, &snapshot, durability);
        if let PersistCommand::Flush(reply) = command {
            let acknowledgement = match &result {
                Ok(()) => Ok(()),
                Err(error) => Err(error.to_string()),
            };
            let _ = reply.send(acknowledgement);
        }
        if let Err(error) = result {
            store_writer_error(persistence_error, &error.to_string());
            break;
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

pub struct DurableNonceManager {
    store: Arc<LiveStateStore>,
    next: u64,
    reserved_through: u64,
}

impl DurableNonceManager {
    pub fn new(store: Arc<LiveStateStore>) -> Result<Self> {
        let (next, reserved_through) = store.reserve_nonce_range()?;
        Ok(Self {
            store,
            next,
            reserved_through,
        })
    }

    pub fn next_nonce(&mut self) -> Result<u64> {
        if self.next > self.reserved_through {
            (self.next, self.reserved_through) = self.store.reserve_nonce_range()?;
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
            false,
        )
        .unwrap()
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
            false,
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
            false,
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
        legacy.processed_fill_keys.insert("fill-1".to_owned());
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
                last_error: None,
            },
        );
        {
            let database = Database::create(&path).unwrap();
            let mut write = database.begin_write().unwrap();
            write.set_durability(Durability::Immediate).unwrap();
            {
                let mut table = write.open_table(LEGACY_STATE_TABLE).unwrap();
                let bytes = serde_json::to_vec(&legacy).unwrap();
                table.insert(STATE_KEY, bytes.as_slice()).unwrap();
            }
            write.commit().unwrap();
        }

        let store = open(&directory);
        let migrated = store.load_required().unwrap();
        assert_eq!(migrated.schema_version, STATE_SCHEMA_VERSION);
        assert_eq!(migrated.orders.len(), 1);
        assert!(migrated.processed_fill_keys.contains("fill-1"));
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
            false,
        )
        .is_err());
    }
}
