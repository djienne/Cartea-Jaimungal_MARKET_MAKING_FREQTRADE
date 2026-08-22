use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    #[inline]
    pub const fn inventory_sign(self) -> i64 {
        match self {
            Self::Buy => 1,
            Self::Sell => -1,
        }
    }

    #[inline]
    pub const fn quote_side(self) -> &'static str {
        match self {
            Self::Buy => "bid",
            Self::Sell => "ask",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggressorSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bbo {
    pub bid_px: i64,
    pub bid_sz: i64,
    pub ask_px: i64,
    pub ask_sz: i64,
    pub exchange_ms: u64,
    pub recv_ns: u64,
}

impl Bbo {
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.bid_px > 0 && self.ask_px > self.bid_px && self.bid_sz >= 0 && self.ask_sz >= 0
    }

    #[inline]
    pub const fn mid_units(self) -> i64 {
        self.bid_px + (self.ask_px - self.bid_px) / 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TradePrint {
    pub aggressor: AggressorSide,
    pub px: i64,
    pub qty_units: i64,
    pub exchange_ms: u64,
    pub recv_ns: u64,
    pub trade_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookLevel {
    pub px: i64,
    pub qty_units: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookSnapshot {
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub exchange_ms: u64,
    pub recv_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarketEvent {
    Bbo(Bbo),
    Trade(TradePrint),
    Book(BookSnapshot),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderIntent {
    pub side: Side,
    pub px: i64,
    pub qty_units: i64,
    pub post_only: bool,
    pub reduce_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteReason {
    Startup,
    Market,
    Fill,
    Calibration,
    Episode,
    StaleMarket,
    StaleCalibration,
    RiskLimit,
    LatencyLimit,
    InvalidRun,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DesiredQuotes {
    pub bid: Option<OrderIntent>,
    pub ask: Option<OrderIntent>,
    pub quote_seq: u64,
    pub model_revision: u64,
    pub generated_ns: u64,
    #[serde(default)]
    pub source_recv_ns: u64,
    pub reason: QuoteReason,
    pub mid: f64,
    pub q_exact: f64,
    pub q_rounded: i64,
    pub tau_remaining: f64,
}

impl DesiredQuotes {
    pub const fn empty(reason: QuoteReason, quote_seq: u64, generated_ns: u64) -> Self {
        Self {
            bid: None,
            ask: None,
            quote_seq,
            model_revision: 0,
            generated_ns,
            source_recv_ns: 0,
            reason,
            mid: 0.0,
            q_exact: 0.0,
            q_rounded: 0,
            tau_remaining: 0.0,
        }
    }
}

impl Default for DesiredQuotes {
    fn default() -> Self {
        Self::empty(QuoteReason::Startup, 0, 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    pub side: Side,
    pub px: i64,
    pub qty_units: i64,
    pub fee_usdc: f64,
    pub exchange_ms: u64,
    pub virtual_order_id: u64,
    pub maker: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionEvent {
    Fill(Fill),
    OrderAcknowledged {
        cloid: String,
        oid: Option<u64>,
    },
    OrderRejected {
        cloid: String,
        reason: String,
    },
    OrderCanceled {
        cloid: String,
        oid: Option<u64>,
    },
    Funding {
        coin: String,
        time_ms: u64,
        usdc: f64,
    },
    AccountReconciled {
        inventory_units: i64,
        equity_usdc: f64,
    },
    UnknownOutcome {
        cloid: Option<String>,
        reason: String,
    },
    Connected,
    Disconnected {
        reason: String,
    },
    Invalidated {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DryRunAccountState {
    pub cash_usdc: f64,
    pub inventory_units: i64,
    pub average_entry_px: f64,
    pub realized_pnl_usdc: f64,
    pub fees_usdc: f64,
    pub funding_usdc: f64,
    pub mark_to_market_pnl_usdc: f64,
    pub equity_usdc: f64,
    pub position_notional_usdc: f64,
    pub margin_used_usdc: f64,
    pub maintenance_margin_usdc: f64,
    pub liquidation_buffer_usdc: f64,
    pub consecutive_losses: u32,
}

pub type AccountState = DryRunAccountState;

#[derive(Debug, Clone)]
pub struct ProcessClock {
    origin: Instant,
}

impl Default for ProcessClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl ProcessClock {
    #[inline]
    pub fn now_ns(&self) -> u64 {
        self.origin.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
    }
}

pub fn unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}
