use crate::types::{unix_ms, Side};
use anyhow::{bail, Context, Result};
use fs2::FileExt;
use redb::{Database, Durability, ReadableDatabase, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("live_state");
const STATE_KEY: &str = "state";
const NONCE_RESERVATION_SIZE: u64 = 1_024;

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
    pub campaign: AcceptanceBudgetState,
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
            schema_version: 1,
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
            campaign: AcceptanceBudgetState::default(),
        }
    }
}

pub struct LiveStateStore {
    database: Database,
    _process_lock: File,
    path: PathBuf,
    update_lock: Mutex<()>,
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
        let store = Self {
            database,
            _process_lock: process_lock,
            path: path.to_owned(),
            update_lock: Mutex::new(()),
        };
        let current = store.load()?;
        if let Some(mut current) = current {
            if current.schema_version != 1
                || current.symbol != symbol
                || !current.account.eq_ignore_ascii_case(account)
                || !current.agent.eq_ignore_ascii_case(agent)
            {
                bail!("live-state identity/configuration mismatch; refusing unsafe reuse");
            }
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
                store.save(&current)?;
            }
        } else {
            store.save(&PersistedLiveState::new(
                symbol,
                account,
                agent,
                config_fingerprint,
                metadata_fingerprint,
            ))?;
        }
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load_required(&self) -> Result<PersistedLiveState> {
        self.load()?.context("live state is unexpectedly absent")
    }

    fn load(&self) -> Result<Option<PersistedLiveState>> {
        let read = self.database.begin_read()?;
        let table = match read.open_table(STATE_TABLE) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let Some(value) = table.get(STATE_KEY)? else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_slice(value.value())?))
    }

    pub fn save(&self, state: &PersistedLiveState) -> Result<()> {
        let bytes = serde_json::to_vec(state)?;
        let mut write = self.database.begin_write()?;
        write.set_durability(Durability::Immediate)?;
        {
            let mut table = write.open_table(STATE_TABLE)?;
            table.insert(STATE_KEY, bytes.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }

    pub fn update<T>(
        &self,
        update: impl FnOnce(&mut PersistedLiveState) -> Result<T>,
    ) -> Result<T> {
        let _guard = self
            .update_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("live-state update lock poisoned"))?;
        let mut state = self.load_required()?;
        let result = update(&mut state)?;
        self.save(&state)?;
        Ok(result)
    }

    pub fn next_cloid_sequence(&self) -> Result<u64> {
        self.update(|state| {
            let sequence = state.next_cloid_sequence;
            state.next_cloid_sequence = state.next_cloid_sequence.saturating_add(1);
            Ok(sequence)
        })
    }

    pub fn reserve_nonce_range(&self) -> Result<(u64, u64)> {
        self.update(|state| {
            let start = unix_ms().max(state.nonce_reserved_through.saturating_add(1));
            let end = start.saturating_add(NONCE_RESERVATION_SIZE - 1);
            state.nonce_reserved_through = end;
            Ok((start, end))
        })
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
}
