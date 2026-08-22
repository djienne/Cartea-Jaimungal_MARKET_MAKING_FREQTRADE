#![forbid(unsafe_code)]

pub mod build_info;
pub use mm_config as config;
pub mod execution;
pub use cj_data::{calibration, parquet_io, replay};
pub use hyperliquid_connector as hyperliquid;
pub use mm_runtime::{hot_path, latency, lockfree, metrics};
pub mod report;
pub use cj_core::{hjb, instrument, quote, types};

pub use build_info::BuildInfo;
pub use cj_data::{CalibrationSnapshot, Calibrator};
pub use config::{AppConfig, Network};
pub use hjb::HjbSurface;
pub use hot_path::ModelBundle;
pub use instrument::InstrumentSpec;
pub use quote::{CarteaJaimungalPolicy, QuoteDecision};
