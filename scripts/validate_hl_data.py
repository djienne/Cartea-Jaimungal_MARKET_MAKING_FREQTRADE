#!/usr/bin/env python3
"""Validate collected Hyperliquid parquet shards before replay or trading."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "prices": {"timestamp"},
    "trades": {"timestamp", "price"},
    "orderbooks": {"timestamp"},
}


@dataclass
class FileValidation:
    path: str
    ok: bool
    rows: int = 0
    columns: list[str] | None = None
    latest_timestamp: str | None = None
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def timestamp_to_utc(value) -> datetime | None:
    try:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        numeric = float(value)
        if numeric > 1_000_000_000_000:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except Exception:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None


def latest_parquet_timestamp(path: Path) -> datetime | None:
    try:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["timestamp"])
        if frame.empty or "timestamp" not in frame:
            return None
        series = frame["timestamp"].dropna()
        if series.empty:
            return None
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return None
            return timestamp_to_utc(float(numeric.max()))
        parsed = pd.to_datetime(series, utc=True, errors="coerce").dropna()
        if parsed.empty:
            return None
        return timestamp_to_utc(parsed.max().to_pydatetime())
    except Exception:
        return None


def timestamp_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_parquet_file(path: Path, required: set[str]) -> FileValidation:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        columns = list(pf.schema_arrow.names)
        missing = sorted(required - set(columns))
        if missing:
            return FileValidation(
                path=str(path),
                ok=False,
                rows=pf.metadata.num_rows,
                columns=columns,
                error=f"missing_columns:{','.join(missing)}",
            )
        latest_ts = latest_parquet_timestamp(path) if "timestamp" in required else None
        return FileValidation(
            path=str(path),
            ok=True,
            rows=pf.metadata.num_rows,
            columns=columns,
            latest_timestamp=timestamp_iso(latest_ts),
        )
    except Exception as exc:
        return FileValidation(path=str(path), ok=False, error=str(exc))


def iter_parquet_files(
    symbol_dir: Path,
    streams: Iterable[str],
    *,
    newest_per_stream: int | None = None,
) -> Iterable[tuple[str, Path]]:
    for stream in streams:
        stream_dir = symbol_dir / stream
        if not stream_dir.is_dir():
            continue
        files = list(stream_dir.glob("*.parquet"))
        if newest_per_stream is not None:
            files = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:newest_per_stream]
        else:
            files = sorted(files)
        for path in files:
            yield stream, path


def validate_symbol(
    data_dir: Path,
    symbol: str,
    *,
    max_files: int | None = None,
    newest_per_stream: int | None = None,
    max_age_seconds: int | None = None,
) -> dict:
    symbol_dir = data_dir / symbol
    results: dict[str, list[FileValidation]] = {stream: [] for stream in REQUIRED_COLUMNS}
    newest_stream_timestamps: dict[str, datetime | None] = {stream: None for stream in REQUIRED_COLUMNS}
    newest_mtime: float | None = None
    checked = 0

    for stream, path in iter_parquet_files(symbol_dir, REQUIRED_COLUMNS.keys(), newest_per_stream=newest_per_stream):
        if max_files is not None and checked >= max_files:
            break
        checked += 1
        try:
            mtime = path.stat().st_mtime
            newest_mtime = mtime if newest_mtime is None else max(newest_mtime, mtime)
        except Exception:
            pass
        validation = validate_parquet_file(path, REQUIRED_COLUMNS[stream])
        results[stream].append(validation)
        latest_ts = timestamp_to_utc(validation.latest_timestamp)
        if latest_ts is not None:
            current = newest_stream_timestamps.get(stream)
            if current is None or latest_ts > current:
                newest_stream_timestamps[stream] = latest_ts

    stream_payload = {}
    total_files = 0
    total_rows = 0
    corrupt_files = []
    missing_streams = []
    for stream, validations in results.items():
        if not validations:
            missing_streams.append(stream)
        total_files += len(validations)
        rows = sum(item.rows for item in validations if item.ok)
        total_rows += rows
        bad = [item for item in validations if not item.ok]
        corrupt_files.extend(bad)
        latest_ts = newest_stream_timestamps.get(stream)
        stream_payload[stream] = {
            "files": len(validations),
            "ok_files": len(validations) - len(bad),
            "bad_files": len(bad),
            "rows": rows,
            "latest_timestamp": timestamp_iso(latest_ts),
        }

    present_stream_timestamps = [ts for ts in newest_stream_timestamps.values() if ts is not None]
    newest_data_timestamp = max(present_stream_timestamps, default=None)
    oldest_required_timestamp = (
        min(present_stream_timestamps) if len(present_stream_timestamps) == len(REQUIRED_COLUMNS) else None
    )
    now = datetime.now(timezone.utc)
    latest_data_age_seconds = (
        (now - newest_data_timestamp).total_seconds() if newest_data_timestamp is not None else None
    )
    oldest_required_data_age_seconds = (
        (now - oldest_required_timestamp).total_seconds() if oldest_required_timestamp is not None else None
    )
    latest_mtime = (
        datetime.fromtimestamp(newest_mtime, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if newest_mtime is not None
        else None
    )
    mtime_age_seconds = (now.timestamp() - newest_mtime) if newest_mtime is not None else None
    fresh = True if max_age_seconds is None else (
        oldest_required_data_age_seconds is not None
        and oldest_required_data_age_seconds <= float(max_age_seconds)
    )
    ok = total_files > 0 and not corrupt_files and not missing_streams and fresh
    return {
        "symbol": symbol,
        "data_dir": str(data_dir),
        "generated_at": utc_now_iso(),
        "ok": ok,
        "checked_files": total_files,
        "max_files": max_files,
        "newest_per_stream": newest_per_stream,
        "total_rows": total_rows,
        "streams": stream_payload,
        "missing_streams": missing_streams,
        "bad_files": [asdict(item) for item in corrupt_files[:100]],
        "bad_file_count": len(corrupt_files),
        "latest_data_timestamp": timestamp_iso(newest_data_timestamp),
        "latest_data_age_seconds": latest_data_age_seconds,
        "oldest_required_data_timestamp": timestamp_iso(oldest_required_timestamp),
        "oldest_required_data_age_seconds": oldest_required_data_age_seconds,
        "freshness_age_seconds": oldest_required_data_age_seconds,
        "latest_file_mtime": latest_mtime,
        "latest_file_age_seconds": mtime_age_seconds,
        "max_age_seconds": max_age_seconds,
        "fresh": fresh,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HL_data parquet shards for replay readiness.")
    parser.add_argument("--symbol", default="CASHCAT")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "HL_data")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--newest-per-stream", type=int, default=None)
    parser.add_argument("--max-age-seconds", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-bad-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_symbol(
        args.data_dir,
        args.symbol,
        max_files=args.max_files,
        newest_per_stream=args.newest_per_stream,
        max_age_seconds=args.max_age_seconds,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 1 if args.fail_on_bad_data and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
