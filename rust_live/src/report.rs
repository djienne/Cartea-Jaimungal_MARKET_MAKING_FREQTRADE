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
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};

const EVENT_LOG_QUEUE_CAPACITY: usize = 8_192;
type EventWrite = Box<dyn FnOnce(&mut BufWriter<File>) -> Result<()> + Send>;

enum WriterMessage {
    Event(EventWrite),
    Flush(SyncSender<std::result::Result<(), String>>),
    Shutdown,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionReport {
    pub schema_version: u32,
    pub build: crate::BuildInfo,
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
    pub build: crate::BuildInfo,
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
    sender: SyncSender<WriterMessage>,
    writer_thread: Option<JoinHandle<()>>,
}

impl JsonlEventLogger {
    pub fn create(directory: &Path, mode: &str, started_at_ms: u64) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        let path = directory.join(format!("{mode}-{started_at_ms}.jsonl"));
        let writer = BufWriter::new(File::create(&path)?);
        let (sender, receiver) = mpsc::sync_channel(EVENT_LOG_QUEUE_CAPACITY);
        let writer_thread = thread::Builder::new()
            .name("jsonl-event-writer".to_owned())
            .spawn(move || run_event_writer(writer, &receiver))?;
        Ok(Self {
            path,
            sender,
            writer_thread: Some(writer_thread),
        })
    }

    pub fn log<T>(&self, event_type: &str, exchange_ms: Option<u64>, payload: &T) -> Result<()>
    where
        T: Serialize + Clone + Send + 'static,
    {
        let event_type = event_type.to_owned();
        let payload = payload.clone();
        let logged_at_ms = crate::types::unix_ms();
        let job: EventWrite = Box::new(move |writer| {
            let envelope = serde_json::json!({
                "event": event_type,
                "logged_at_ms": logged_at_ms,
                "exchange_ms": exchange_ms,
                "payload": payload,
            });
            serde_json::to_writer(&mut *writer, &envelope)?;
            writer.write_all(b"\n")?;
            Ok(())
        });
        self.sender
            .try_send(WriterMessage::Event(job))
            .map_err(|error| {
                anyhow::anyhow!(match error {
                    TrySendError::Full(_) => {
                        "event-log queue saturated; refusing an incomplete scientific run"
                    }
                    TrySendError::Disconnected(_) => "event-log writer stopped unexpectedly",
                })
            })
    }

    pub fn flush(&mut self) -> Result<()> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.sender
            .send(WriterMessage::Flush(reply_tx))
            .map_err(|_| anyhow::anyhow!("event-log writer stopped before flush"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("event-log writer dropped flush acknowledgement"))?
            .map_err(anyhow::Error::msg)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for JsonlEventLogger {
    fn drop(&mut self) {
        let _ = self.sender.send(WriterMessage::Shutdown);
        if let Some(handle) = self.writer_thread.take() {
            let _ = handle.join();
        }
    }
}

fn run_event_writer(mut writer: BufWriter<File>, receiver: &mpsc::Receiver<WriterMessage>) {
    while let Ok(message) = receiver.recv() {
        match message {
            WriterMessage::Event(write) => {
                if write(&mut writer).is_err() {
                    break;
                }
            }
            WriterMessage::Flush(reply) => {
                let result = writer.flush().map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
            WriterMessage::Shutdown => {
                let _ = writer.flush();
                break;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asynchronous_event_writer_preserves_order_and_flushes() {
        let directory = tempfile::tempdir().unwrap();
        let path;
        {
            let mut logger = JsonlEventLogger::create(directory.path(), "test", 1).unwrap();
            path = logger.path().to_owned();
            for sequence in 0..1_000_u64 {
                logger.log("sequence", Some(sequence), &sequence).unwrap();
            }
            logger.flush().unwrap();
        }
        let rows = std::fs::read_to_string(path).unwrap();
        let parsed = rows
            .lines()
            .map(|row| serde_json::from_str::<serde_json::Value>(row).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(parsed.len(), 1_000);
        for (sequence, row) in parsed.into_iter().enumerate() {
            assert_eq!(row["payload"], sequence as u64);
            assert_eq!(row["exchange_ms"], sequence as u64);
        }
    }
}
