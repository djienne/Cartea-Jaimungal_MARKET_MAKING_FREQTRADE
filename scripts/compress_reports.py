"""Compress historical dry-run event logs in rust_live/reports/ to .jsonl.zst.

The grid now writes compressed logs directly, but earlier runs left plain
`.jsonl` behind -- 1.5 GB of it in one grid directory alone. This converts them
with the same codec and level the Rust writer uses (zstd-3, measured ~16x on
this data).

Safety rules, in order of importance:

* **Never touch a file a writer still holds.** A log whose mtime is recent is
  assumed live and skipped; pass --min-age-minutes 0 only when you know every
  grid is stopped.
* **Verify before deleting.** The original is removed only after the compressed
  copy is read back and its line count and byte length match exactly.
* **Verify with stream_reader, never decompress().** The one-shot API stops at
  the first zstd frame and reports success, so it would "verify" a truncated
  file as good.
* Only `rust_live/reports/` is touched. The Parquet tape and the collector are
  out of scope and are never opened.

Usage:
    python scripts/compress_reports.py --dry-run     # show what would happen
    python scripts/compress_reports.py               # convert
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zstandard

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "rust_live" / "reports"
LEVEL = 3
CHUNK = 8 << 20


def measure(path: Path) -> tuple[int, int]:
    """(lines, bytes) of a plain file."""
    lines = 0
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            lines += chunk.count(b"\n")
            size += len(chunk)
    return lines, size


def measure_zst(path: Path) -> tuple[int, int]:
    """(lines, bytes) of a compressed file, read across every frame."""
    lines = 0
    size = 0
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        while chunk := reader.read(CHUNK):
            lines += chunk.count(b"\n")
            size += len(chunk)
    return lines, size


def compress(path: Path, keep: bool) -> tuple[int, int] | None:
    target = path.with_suffix(path.suffix + ".zst")
    if target.exists():
        print(f"  skip {path.name}: {target.name} already exists")
        return None
    before_lines, before_bytes = measure(path)
    temporary = target.with_suffix(".zst.tmp")
    compressor = zstandard.ZstdCompressor(level=LEVEL)
    with path.open("rb") as src, temporary.open("wb") as dst:
        compressor.copy_stream(src, dst)
    after_lines, after_bytes = measure_zst(temporary)
    if (after_lines, after_bytes) != (before_lines, before_bytes):
        temporary.unlink(missing_ok=True)
        print(
            f"  FAIL {path.name}: round trip differs "
            f"({before_lines} lines/{before_bytes} B -> {after_lines}/{after_bytes}); original kept"
        )
        return None
    temporary.replace(target)
    compressed = target.stat().st_size
    if not keep:
        path.unlink()
    ratio = before_bytes / compressed if compressed else 0.0
    print(
        f"  {path.name}: {before_bytes / 1e6:8.1f} MB -> {compressed / 1e6:7.2f} MB  "
        f"({ratio:5.1f}x, {before_lines:,} lines verified)"
    )
    return before_bytes, compressed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(REPORTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep originals after verifying")
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=10.0,
        help="skip files modified more recently than this (a live writer may hold them)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    cutoff = time.time() - args.min_age_minutes * 60.0

    candidates, skipped_live = [], []
    for path in sorted(root.rglob("*.jsonl")):
        (skipped_live if path.stat().st_mtime > cutoff else candidates).append(path)

    total = sum(p.stat().st_size for p in candidates)
    print(f"{len(candidates)} file(s), {total / 1e9:.2f} GB eligible under {root}")
    for path in skipped_live:
        age = (time.time() - path.stat().st_mtime) / 60.0
        print(f"  skip {path.name}: modified {age:.1f} min ago, a writer may hold it")
    if args.dry_run:
        for path in candidates:
            print(f"  would compress {path.relative_to(root)} ({path.stat().st_size / 1e6:.1f} MB)")
        return
    if not candidates:
        return

    before_total = after_total = 0
    for path in candidates:
        result = compress(path, args.keep)
        if result:
            before_total += result[0]
            after_total += result[1]
    if after_total:
        print(
            f"\n{before_total / 1e9:.2f} GB -> {after_total / 1e9:.3f} GB "
            f"({before_total / after_total:.1f}x, {(before_total - after_total) / 1e9:.2f} GB reclaimed)"
        )


if __name__ == "__main__":
    sys.exit(main())
