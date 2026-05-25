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
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
        return FileValidation(path=str(path), ok=True, rows=pf.metadata.num_rows, columns=columns)
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
        results[stream].append(validate_parquet_file(path, REQUIRED_COLUMNS[stream]))

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
        stream_payload[stream] = {
            "files": len(validations),
            "ok_files": len(validations) - len(bad),
            "bad_files": len(bad),
            "rows": rows,
        }

    latest_mtime = (
        datetime.fromtimestamp(newest_mtime, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if newest_mtime is not None
        else None
    )
    age_seconds = (datetime.now(timezone.utc).timestamp() - newest_mtime) if newest_mtime is not None else None
    fresh = True if max_age_seconds is None else (
        age_seconds is not None and age_seconds <= float(max_age_seconds)
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
        "latest_file_mtime": latest_mtime,
        "latest_file_age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "fresh": fresh,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HL_data parquet shards for replay readiness.")
    parser.add_argument("--symbol", default="ETH")
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
