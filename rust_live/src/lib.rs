#![forbid(unsafe_code)]

pub mod calibration;
pub mod config;
pub mod execution;
pub mod hjb;
pub mod hot_path;
pub mod hyperliquid;
pub mod instrument;
pub mod latency;
pub mod lockfree;
pub mod metrics;
pub mod parquet_io;
pub mod quote;
pub mod replay;
pub mod report;
pub mod types;

pub use calibration::{CalibrationSnapshot, Calibrator};
pub use config::{AppConfig, Network};
pub use hjb::HjbSurface;
pub use hot_path::ModelBundle;
pub use instrument::InstrumentSpec;
pub use quote::{CarteaJaimungalPolicy, QuoteDecision};
