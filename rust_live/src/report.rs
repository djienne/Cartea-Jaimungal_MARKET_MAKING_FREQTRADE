use crate::calibration::CalibrationSnapshot;
use crate::execution::DryRunDiagnostics;
use crate::execution::LiveExecutionDiagnostics;
use crate::instrument::InstrumentSpec;
use crate::latency::LatencySnapshot;
use crate::metrics::MetricsSnapshot;
use crate::types::DryRunAccountState;
use anyhow::Result;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize)]
pub struct SessionReport {
    pub schema_version: u32,
    pub session_id: String,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    pub mode: String,
    pub config_fingerprint: String,
    pub instrument: InstrumentSpec,
    pub calibration: Option<CalibrationSnapshot>,
    pub model: Option<ModelReport>,
    pub account: DryRunAccountState,
    pub execution: DryRunDiagnostics,
    pub metrics: MetricsSnapshot,
    pub latency: LatencySnapshot,
    pub scientifically_valid: bool,
    pub invalid_reasons: Vec<String>,
    pub event_log_path: String,
    pub market_event_ring_high_water: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveSessionReport {
    pub schema_version: u32,
    pub session_id: String,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    pub config_fingerprint: String,
    pub instrument: InstrumentSpec,
    pub calibration: Option<CalibrationSnapshot>,
    pub model: Option<ModelReport>,
    pub account: DryRunAccountState,
    pub execution: LiveExecutionDiagnostics,
    pub metrics: MetricsSnapshot,
    pub latency: LatencySnapshot,
    pub scientifically_valid: bool,
    pub event_log_path: String,
    pub market_event_ring_high_water: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelReport {
    pub revision: u64,
    pub inventory_unit: i64,
    pub q_min: i64,
    pub q_max: i64,
    pub horizon_seconds: f64,
    pub n_steps: usize,
    pub dt: f64,
    pub phi_effective: f64,
    pub alpha_effective: f64,
    pub kappa_average: f64,
}

impl ModelReport {
    pub fn from_surface(surface: &crate::hjb::HjbSurface, inventory_unit: i64) -> Self {
        Self {
            revision: surface.revision,
            inventory_unit,
            q_min: surface.q_min,
            q_max: surface.q_max,
            horizon_seconds: surface.horizon_seconds,
            n_steps: surface.n_steps,
            dt: surface.dt,
            phi_effective: surface.phi_effective,
            alpha_effective: surface.alpha_effective,
            kappa_average: surface.kappa_average,
        }
    }
}

pub struct JsonlEventLogger {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl JsonlEventLogger {
    pub fn create(directory: &Path, mode: &str, started_at_ms: u64) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        let path = directory.join(format!("{mode}-{started_at_ms}.jsonl"));
        let writer = BufWriter::new(File::create(&path)?);
        Ok(Self { path, writer })
    }

    pub fn log<T: Serialize>(
        &mut self,
        event_type: &str,
        exchange_ms: Option<u64>,
        payload: &T,
    ) -> Result<()> {
        let envelope = serde_json::json!({
            "event": event_type,
            "logged_at_ms": crate::types::unix_ms(),
            "exchange_ms": exchange_ms,
            "payload": serde_json::to_value(payload)?,
        });
        serde_json::to_writer(&mut self.writer, &envelope)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl SessionReport {
    pub fn write_atomic(&self, path: &Path) -> Result<()> {
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

impl LiveSessionReport {
    pub fn write_atomic(&self, path: &Path) -> Result<()> {
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
