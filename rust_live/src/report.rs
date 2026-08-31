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
use tracing::{info, warn};

const EVENT_LOG_QUEUE_CAPACITY: usize = 8_192;
/// The sink is `dyn Write` rather than a concrete `BufWriter<File>` so the same
/// serialization closure works for a plain file and a zstd stream.
type EventWrite = Box<dyn FnOnce(&mut dyn Write) -> Result<()> + Send>;

/// Compressed logs are flushed after this many events even if nobody calls
/// `flush()`.
///
/// This exists because zstd is a *framed* format. Without a periodic flush the
/// file stays undecodable until the run ends, which would break reading a log
/// while the grid is still writing it — and a run measured in days would risk
/// its whole tail on a kill. Flushing ends a block and makes everything written
/// so far decodable, at a small cost in ratio; doing it per line instead would
/// collapse the ratio toward 1.
const COMPRESSED_FLUSH_EVERY_EVENTS: u64 = 2_000;

enum WriterMessage {
    Event(EventWrite),
    Flush(SyncSender<std::result::Result<(), String>>),
    Shutdown,
}

/// Size-based rotation for an append-mode event log.
///
/// The grid's logs append to a stable stem so a restart extends the same
/// stream, and since the grid resumes across restarts that stream now has no
/// natural end. Measured on the 46.4 h run: 0.31 MB/h per variant compressed,
/// which is 3.9 GB/month across eighteen. Without a bound the only thing that
/// ever stopped it was losing the run.
///
/// `grid-wide8.jsonl.zst` rolls to `grid-wide8.1.jsonl.zst`, that to `.2.`, and
/// the oldest past `keep` is deleted -- so the worst case is `(keep + 1) *
/// max_bytes` per log, and total disk is bounded by construction rather than by
/// how long the run happens to last.
#[derive(Debug, Clone, Copy)]
pub struct LogRotation {
    /// Roll once the live file is at least this large. Zero disables rotation.
    pub max_bytes: u64,
    /// How many rolled generations to keep behind the live file.
    pub keep: usize,
}

impl LogRotation {
    /// Off — the log grows without limit.
    ///
    /// Correct for the bounded callers: `replay` and `live` embed a start
    /// timestamp in the filename and never reopen the same path, so their logs
    /// are already one-per-run.
    pub const DISABLED: Self = Self {
        max_bytes: 0,
        keep: 0,
    };

    const fn enabled(self) -> bool {
        self.max_bytes > 0
    }
}

/// Roll `path` to `path.1`, `path.1` to `path.2`, and so on, deleting the
/// generation that falls off the end.
///
/// Renames from the oldest backwards so no generation is ever overwritten while
/// something still points at it. A failure to roll is reported but not fatal:
/// losing the ability to rotate is better than losing the run, and the caller
/// carries on writing to the file it already holds.
fn rotate_log_files(path: &Path, keep: usize) -> std::io::Result<()> {
    let generation = |index: usize| -> PathBuf {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(format!(".{index}"));
        path.with_file_name(name)
    };
    let oldest = generation(keep);
    if oldest.exists() {
        std::fs::remove_file(&oldest)?;
    }
    for index in (1..keep).rev() {
        let from = generation(index);
        if from.exists() {
            std::fs::rename(&from, generation(index + 1))?;
        }
    }
    if keep > 0 {
        std::fs::rename(path, generation(1))?;
    } else {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

/// Where an event log's bytes end up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Plain `.jsonl`, greppable without tooling. Used for the `live` audit
    /// trail and for bounded `replay` runs.
    Plain,
    /// zstd-compressed `.jsonl.zst`, ~16x smaller (measured on real grid logs).
    ///
    /// Used for the long-running dry-run accumulators, where the plain format
    /// costs 158 MB per variant per 20 h. zstd frames concatenate, so a stopped
    /// and restarted run appends to the same file and readers see one
    /// continuous stream.
    Zstd,
}

impl LogFormat {
    const fn extension(self) -> &'static str {
        match self {
            Self::Plain => "jsonl",
            Self::Zstd => "jsonl.zst",
        }
    }
}

/// The writer thread's sink, owning whichever encoder the format needs.
enum EventSink {
    Plain(BufWriter<File>),
    /// `zstd::Encoder` must be told to `finish()` so the final frame is
    /// terminated; dropping it would leave the tail unreadable. The `Option` is
    /// what lets `finish()` consume it from behind `&mut self` at shutdown.
    Zstd(Option<zstd::Encoder<'static, BufWriter<File>>>),
}

impl EventSink {
    fn writer(&mut self) -> Option<&mut dyn Write> {
        match self {
            Self::Plain(writer) => Some(writer),
            Self::Zstd(encoder) => encoder.as_mut().map(|value| value as &mut dyn Write),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(writer) => writer.flush(),
            Self::Zstd(Some(encoder)) => encoder.flush(),
            Self::Zstd(None) => Ok(()),
        }
    }

    /// Terminate the stream. For zstd this closes the frame and flushes the
    /// underlying file; a plain writer only needs the flush.
    fn finish(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(writer) => writer.flush(),
            Self::Zstd(slot) => match slot.take() {
                Some(encoder) => encoder.finish()?.flush(),
                None => Ok(()),
            },
        }
    }
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
    pub event_log_format: &'static str,
    pub event_log_max_bytes: u64,
    pub event_log_keep: usize,
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
    pub max_newton_iterations_used: usize,
    pub max_final_residual: f64,
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
            max_newton_iterations_used: surface.max_newton_iterations_used,
            max_final_residual: surface.max_final_residual,
        }
    }
}

/// What a full event-log queue means for the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogBackpressure {
    /// Real-time sessions: a saturated queue is evidence loss, refuse the run.
    RefuseWhenFull,
    /// Offline replay: the producer runs at replay speed and nothing is
    /// real-time, so completeness is preserved by blocking on the writer.
    BlockWhenFull,
}

pub struct JsonlEventLogger {
    path: PathBuf,
    sender: SyncSender<WriterMessage>,
    backpressure: LogBackpressure,
    writer_thread: Option<JoinHandle<()>>,
}

impl JsonlEventLogger {
    pub fn create(directory: &Path, mode: &str, started_at_ms: u64) -> Result<Self> {
        Self::create_with_backpressure(
            directory,
            mode,
            started_at_ms,
            LogBackpressure::RefuseWhenFull,
        )
    }

    pub fn create_with_backpressure(
        directory: &Path,
        mode: &str,
        started_at_ms: u64,
        backpressure: LogBackpressure,
    ) -> Result<Self> {
        Self::create_with_format(
            directory,
            &format!("{mode}-{started_at_ms}"),
            backpressure,
            LogFormat::Plain,
        )
    }

    /// Open an event log, choosing the on-disk format.
    ///
    /// `stem` is the filename without extension. Callers that want a run to
    /// *append* across restarts pass a stable stem (no timestamp); callers that
    /// want one file per run include the start time, which is what
    /// [`Self::create_with_backpressure`] does.
    ///
    /// Compressed logs are opened for append: zstd frames concatenate, so a
    /// restart extends the same stream rather than truncating it. Plain logs
    /// keep the historical create-truncate behaviour, since their callers embed
    /// the start time in the name and never reopen the same path.
    pub fn create_with_format(
        directory: &Path,
        stem: &str,
        backpressure: LogBackpressure,
        format: LogFormat,
    ) -> Result<Self> {
        Self::create_with_rotation(directory, stem, backpressure, format, LogRotation::DISABLED)
    }

    /// As [`Self::create_with_format`], with a size bound on the log.
    ///
    /// Rotation is only meaningful for a log that appends across restarts, so
    /// it is off by default and requested explicitly by the grid.
    pub fn create_with_rotation(
        directory: &Path,
        stem: &str,
        backpressure: LogBackpressure,
        format: LogFormat,
        rotation: LogRotation,
    ) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        let path = directory.join(format!("{stem}.{}", format.extension()));
        let sink = open_sink(&path, format)?;
        let (sender, receiver) = mpsc::sync_channel(EVENT_LOG_QUEUE_CAPACITY);
        let flush_every = match format {
            LogFormat::Plain => 0,
            LogFormat::Zstd => COMPRESSED_FLUSH_EVERY_EVENTS,
        };
        let writer_path = path.clone();
        let writer_thread = thread::Builder::new()
            .name("jsonl-event-writer".to_owned())
            .spawn(move || {
                run_event_writer(sink, &receiver, flush_every, &writer_path, format, rotation);
            })?;
        Ok(Self {
            path,
            sender,
            backpressure,
            writer_thread: Some(writer_thread),
        })
    }

    pub fn log<T>(&self, event_type: &str, exchange_ms: Option<u64>, payload: &T) -> Result<()>
    where
        T: Serialize + Clone + Send + 'static,
    {
        self.log_owned(event_type, exchange_ms, payload.clone())
    }

    /// Queue an owned payload so serialization stays on the writer thread and
    /// the caller does not pay an additional clone.
    pub fn log_owned<T>(&self, event_type: &str, exchange_ms: Option<u64>, payload: T) -> Result<()>
    where
        T: Serialize + Send + 'static,
    {
        let event_type = event_type.to_owned();
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
        match self.backpressure {
            LogBackpressure::BlockWhenFull => self
                .sender
                .send(WriterMessage::Event(job))
                .map_err(|_| anyhow::anyhow!("event-log writer stopped unexpectedly")),
            LogBackpressure::RefuseWhenFull => self
                .sender
                .try_send(WriterMessage::Event(job))
                .map_err(|error| {
                    anyhow::anyhow!(match error {
                        TrySendError::Full(_) => {
                            "event-log queue saturated; refusing an incomplete scientific run"
                        }
                        TrySendError::Disconnected(_) => "event-log writer stopped unexpectedly",
                    })
                }),
        }
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

/// Open the sink for `path`, appending for zstd and truncating for plain.
fn open_sink(path: &Path, format: LogFormat) -> Result<EventSink> {
    Ok(match format {
        LogFormat::Plain => EventSink::Plain(BufWriter::new(File::create(path)?)),
        LogFormat::Zstd => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            // Level 3: measured 15.8x on real grid logs at 736 MB/s, some
            // 350x the actual write rate, so the ratio is free here.
            EventSink::Zstd(Some(zstd::Encoder::new(BufWriter::new(file), 3)?))
        }
    })
}

/// Roll the log if it has outgrown its bound, returning the fresh sink.
///
/// Called only at a flush boundary, where the frame is already terminated, so
/// the size on disk is the real size and the rolled file is complete and
/// readable rather than a truncated frame.
fn rotate_if_oversized(
    sink: EventSink,
    path: &Path,
    format: LogFormat,
    rotation: LogRotation,
) -> EventSink {
    if !rotation.enabled() {
        return sink;
    }
    let Ok(metadata) = std::fs::metadata(path) else {
        return sink;
    };
    if metadata.len() < rotation.max_bytes {
        return sink;
    }
    // finish(), not flush(): the current frame has to be terminated before the
    // file is renamed, or the rolled generation's tail is unreadable.
    let mut sink = sink;
    let _ = sink.finish();
    if let Err(error) = rotate_log_files(path, rotation.keep) {
        // Reopen what we just closed and carry on unbounded rather than lose
        // the run over a filesystem hiccup.
        warn!(
            path = %path.display(),
            %error,
            "cannot rotate the event log; continuing on the existing file"
        );
        return open_sink(path, format).unwrap_or(sink);
    }
    // Say what was dropped. A rotation that silently deletes the oldest
    // generation reads, later, as a run that simply has no history that far
    // back.
    info!(
        path = %path.display(),
        rolled_bytes = metadata.len(),
        keep = rotation.keep,
        "event log rotated; the oldest generation beyond `keep` was deleted"
    );
    match open_sink(path, format) {
        Ok(fresh) => fresh,
        Err(error) => {
            warn!(path = %path.display(), %error, "cannot reopen the event log after rotating");
            sink
        }
    }
}

fn run_event_writer(
    mut sink: EventSink,
    receiver: &mpsc::Receiver<WriterMessage>,
    flush_every: u64,
    path: &Path,
    format: LogFormat,
    rotation: LogRotation,
) {
    let mut since_flush = 0_u64;
    while let Ok(message) = receiver.recv() {
        match message {
            WriterMessage::Event(write) => {
                let Some(writer) = sink.writer() else {
                    break; // stream already finished; nothing can be written
                };
                if write(writer).is_err() {
                    break;
                }
                since_flush = since_flush.saturating_add(1);
                if flush_every > 0 && since_flush >= flush_every {
                    since_flush = 0;
                    // A failed periodic flush is not fatal: the next explicit
                    // flush or the shutdown finish() will surface it, and
                    // dropping events because a flush hiccuped would lose more
                    // than it protects.
                    let _ = sink.flush();
                    sink = rotate_if_oversized(sink, path, format, rotation);
                }
            }
            WriterMessage::Flush(reply) => {
                since_flush = 0;
                let result = sink.flush().map_err(|error| error.to_string());
                let _ = reply.send(result);
                sink = rotate_if_oversized(sink, path, format, rotation);
            }
            WriterMessage::Shutdown => {
                // finish(), not flush(): a zstd frame left unterminated makes
                // the tail unreadable.
                let _ = sink.finish();
                break;
            }
        }
    }
    // Any other exit path (a write error, or the sender being dropped) must
    // still terminate the frame.
    let _ = sink.finish();
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

    /// Decode a zstd log, keeping whatever decoded before any truncated tail.
    ///
    /// Reading a file the writer still has open means the final frame is
    /// mid-write, and the decoder reports an error when it runs into it. The
    /// bytes before that point are still valid, so this returns them rather
    /// than discarding the whole read -- the same tolerance the Python readers
    /// rely on.
    fn decode_zstd(path: &Path) -> String {
        use std::io::Read as _;
        let Ok(file) = File::open(path) else {
            return String::new();
        };
        let Ok(mut decoder) = zstd::Decoder::new(file) else {
            return String::new();
        };
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 8_192];
        while let Ok(read) = decoder.read(&mut chunk) {
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..read]);
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // A torn tail can also cut a line in half; drop any trailing partial.
        match text.rfind('\n') {
            Some(end) => text[..=end].to_owned(),
            None => String::new(),
        }
    }

    #[test]
    fn compressed_event_log_round_trips_every_line() {
        let directory = tempfile::tempdir().unwrap();
        let path;
        {
            let mut logger = JsonlEventLogger::create_with_format(
                directory.path(),
                "grid-wide8",
                LogBackpressure::RefuseWhenFull,
                LogFormat::Zstd,
            )
            .unwrap();
            path = logger.path().to_owned();
            assert_eq!(path.file_name().unwrap(), "grid-wide8.jsonl.zst");
            for sequence in 0..5_000_u64 {
                logger.log("sequence", Some(sequence), &sequence).unwrap();
            }
            logger.flush().unwrap();
        }
        let rows: Vec<serde_json::Value> = decode_zstd(&path)
            .lines()
            .map(|row| serde_json::from_str(row).unwrap())
            .collect();
        assert_eq!(rows.len(), 5_000);
        for (sequence, row) in rows.into_iter().enumerate() {
            assert_eq!(row["payload"], sequence as u64);
        }
    }

    #[test]
    fn a_compressed_log_is_readable_while_it_is_still_being_written() {
        // The periodic flush exists so a run measured in days can be inspected
        // mid-flight. Without it the file would be undecodable until shutdown.
        let directory = tempfile::tempdir().unwrap();
        let logger = JsonlEventLogger::create_with_format(
            directory.path(),
            "grid-live",
            LogBackpressure::RefuseWhenFull,
            LogFormat::Zstd,
        )
        .unwrap();
        let events = COMPRESSED_FLUSH_EVERY_EVENTS * 2;
        for sequence in 0..events {
            logger.log("sequence", Some(sequence), &sequence).unwrap();
        }
        // Deliberately NOT flushing and NOT dropping: the logger is still live.
        // Wait for the writer thread to drain and cross a periodic flush.
        let path = logger.path().to_owned();
        let mut decoded = 0;
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            decoded = decode_zstd(&path).lines().count();
            if decoded > 0 {
                break;
            }
        }
        assert!(
            decoded > 0,
            "a periodic flush must leave the stream decodable mid-run"
        );
        assert!(
            decoded <= events as usize,
            "decoded {decoded} lines but only {events} were written"
        );
    }

    #[test]
    fn a_restart_appends_to_the_same_compressed_stream() {
        // zstd frames concatenate, so stopping and restarting must extend the
        // file rather than truncate it -- this is what makes a multi-day grid
        // survivable across restarts.
        let directory = tempfile::tempdir().unwrap();
        let path;
        {
            let mut first = JsonlEventLogger::create_with_format(
                directory.path(),
                "grid-wide8",
                LogBackpressure::RefuseWhenFull,
                LogFormat::Zstd,
            )
            .unwrap();
            path = first.path().to_owned();
            for sequence in 0..100_u64 {
                first.log("first_run", Some(sequence), &sequence).unwrap();
            }
            first.flush().unwrap();
        }
        {
            let mut second = JsonlEventLogger::create_with_format(
                directory.path(),
                "grid-wide8",
                LogBackpressure::RefuseWhenFull,
                LogFormat::Zstd,
            )
            .unwrap();
            assert_eq!(second.path(), path, "a restart must reuse the same file");
            for sequence in 100..200_u64 {
                second.log("second_run", Some(sequence), &sequence).unwrap();
            }
            second.flush().unwrap();
        }
        let text = decode_zstd(&path);
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|row| serde_json::from_str(row).unwrap())
            .collect();
        assert_eq!(rows.len(), 200, "both runs must survive in one stream");
        assert_eq!(rows[0]["event"], "first_run");
        assert_eq!(rows[199]["event"], "second_run");
        for (sequence, row) in rows.into_iter().enumerate() {
            assert_eq!(
                row["payload"], sequence as u64,
                "order preserved across the restart"
            );
        }
    }

    #[test]
    fn a_plain_log_keeps_its_extension_and_is_not_compressed() {
        let directory = tempfile::tempdir().unwrap();
        let logger = JsonlEventLogger::create(directory.path(), "live", 7).unwrap();
        assert_eq!(logger.path().file_name().unwrap(), "live-7.jsonl");
    }

    fn generations(directory: &Path, stem: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(directory)
            .unwrap()
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_string_lossy().into_owned();
                name.starts_with(stem).then_some(name)
            })
            .collect();
        names.sort();
        names
    }

    #[test]
    fn rotate_log_files_shifts_generations_and_drops_the_oldest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("grid-wide8.jsonl.zst");
        // Three rolls with keep=2: the third must evict the first.
        for generation in ["first", "second", "third"] {
            std::fs::write(&path, generation).unwrap();
            rotate_log_files(&path, 2).unwrap();
        }
        assert_eq!(
            generations(directory.path(), "grid-wide8"),
            vec![
                "grid-wide8.jsonl.zst.1".to_owned(),
                "grid-wide8.jsonl.zst.2".to_owned()
            ]
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("grid-wide8.jsonl.zst.1")).unwrap(),
            "third"
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("grid-wide8.jsonl.zst.2")).unwrap(),
            "second"
        );
    }

    /// `keep: 0` means "bound the log, retain no history" — the live file is
    /// discarded rather than renamed, so disk stays bounded by `max_bytes`.
    #[test]
    fn rotate_log_files_with_no_retention_discards_the_live_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("grid-wide8.jsonl.zst");
        std::fs::write(&path, "only").unwrap();
        rotate_log_files(&path, 0).unwrap();
        assert!(generations(directory.path(), "grid-wide8").is_empty());
    }

    #[test]
    fn an_oversized_log_rotates_and_keeps_writing() {
        let directory = tempfile::tempdir().unwrap();
        let rotation = LogRotation {
            max_bytes: 1_024,
            keep: 2,
        };
        {
            let mut logger = JsonlEventLogger::create_with_rotation(
                directory.path(),
                "grid-wide8",
                LogBackpressure::BlockWhenFull,
                LogFormat::Zstd,
                rotation,
            )
            .unwrap();
            // Flush is a rotation boundary, so drive several of them.
            for round in 0..12_u64 {
                for sequence in 0..500_u64 {
                    logger
                        .log("sequence", Some(sequence), &(round, sequence))
                        .unwrap();
                }
                logger.flush().unwrap();
            }
        }
        let names = generations(directory.path(), "grid-wide8");
        assert!(
            names.contains(&"grid-wide8.jsonl.zst".to_owned()),
            "the live log must exist after rotating: {names:?}"
        );
        // keep=2 bounds it at the live file plus two generations, however long
        // the run goes on.
        assert!(
            names.len() <= 3,
            "rotation must bound the file count: {names:?}"
        );
        assert!(
            names.len() > 1,
            "the log should have rotated at least once: {names:?}"
        );
    }

    #[test]
    fn rotation_is_off_by_default_so_bounded_callers_are_unaffected() {
        assert!(!LogRotation::DISABLED.enabled());
        let directory = tempfile::tempdir().unwrap();
        {
            let mut logger = JsonlEventLogger::create_with_format(
                directory.path(),
                "grid-x",
                LogBackpressure::BlockWhenFull,
                LogFormat::Zstd,
            )
            .unwrap();
            for round in 0..12_u64 {
                for sequence in 0..500_u64 {
                    logger
                        .log("sequence", Some(sequence), &(round, sequence))
                        .unwrap();
                }
                logger.flush().unwrap();
            }
        }
        assert_eq!(
            generations(directory.path(), "grid-x"),
            vec!["grid-x.jsonl.zst".to_owned()]
        );
    }
}
