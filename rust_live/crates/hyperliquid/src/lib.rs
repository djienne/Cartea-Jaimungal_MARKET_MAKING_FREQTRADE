pub mod account;
pub mod account_types;
pub mod auth;
pub mod exchange;
pub mod live_state;
pub mod market;
pub mod meta;
pub mod session;
pub mod signing;
pub mod transport;
pub mod wire;

#[cfg(feature = "acceptance")]
pub mod acceptance;

pub mod execution;

pub use cj_core::{instrument, types};
pub use mm_config as config;
pub use mm_runtime::{latency, lockfree, metrics};

// Transitional internal namespace retained while the transport modules are
// decomposed. External users access the typed modules directly.
pub mod hyperliquid {
    pub use crate::{
        account, account_types, auth, exchange, live_state, market, meta, session, signing,
        transport, wire,
    };
}

pub use execution::{HyperliquidLiveBackend, LiveBootstrap, LiveExecutionDiagnostics};
