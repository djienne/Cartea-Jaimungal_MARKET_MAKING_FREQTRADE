#!/usr/bin/env python3
"""Archive one retention window before the collector deletes it.

WHY THIS EXISTS. `hl-cashcat-collector` keeps 30 days
(`CASHCAT_RETENTION_MINUTES: 43200`) and deletes everything older. Two
irreplaceable things ride that clock:

1. **Replay.** A sweep can only score a window while its Parquet shards exist.
   Once a period rolls off, no replay can ever be run against it again -- not
   re-run more cheaply, not re-run at all.
2. **The dry-run grid.** Its event logs rotate at ~34 days
   (`--log-max-mb 64 --log-keep 3`), so grid history expires too.

Every `--cadence-days` this attempts to write a period directory holding a fresh
sweep plus the grid's P&L curve, small enough to commit. A 21-day cadence against
30-day retention leaves 9 days to retry an interrupted or failed attempt; the
failure is harmless only if a successful `--force` rerun lands before that
margin expires.

TWO WINDOW CONVENTIONS, deliberately different, both recorded in the period
README:

- the **sweep** scores the whole tape currently on disk (up to 30 days), so
  consecutive archives overlap -- that overlap is the safety margin;
- the **grid curve** is sliced to the period since the last archive, so the
  archives concatenate into one non-overlapping timeline.

Usage:
    python scripts/archive_period.py --dry-run     # what would be written
    python scripts/archive_period.py               # archive if due
    python scripts/archive_period.py --force       # archive regardless
    python scripts/archive_period.py --loop        # daemon: check hourly
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

SHARD_RE = re.compile(r"_(\d{13})\.parquet$")

# A symbol qualifies for archiving if its tape spans more than this. The 30-day
# collector sits far above the line and the 3-day ones (ETH, ACE, CHIP, PENGU,
# NIL at RETENTION_MINUTES 4320) far below, so the split is unambiguous without
# this repo having to read another project's compose file. A long-retention coin
# added later is picked up automatically once its tape grows past the line.
DEFAULT_MIN_TAPE_DAYS = 7.0

# The sweep's non-symbol defaults -- mid fallback, inventory unit base, tick
# size -- are CASHCAT's. Running them against a different instrument would
# silently produce confident numbers for the wrong asset, so an unknown symbol
# is refused rather than archived wrongly.
KNOWN_INSTRUMENTS = {"CASHCAT"}


def log(message):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(stamp + " " + message, flush=True)


def utc(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# Tape inspection
# --------------------------------------------------------------------------


def tape_span(data_dir, symbol):
    """(oldest_ms, newest_ms, shard_count) from shard filenames, or None.

    Filenames, not file contents: the collector stamps the flush time into the
    name and both readers in this project already select shards that way, so the
    span costs one directory listing rather than opening 4,000 Parquet files.
    """
    stamps = []
    for stream in ("trades", "prices", "orderbooks"):
        for path in glob.glob(str(Path(data_dir) / symbol / stream / "*.parquet")):
            match = SHARD_RE.search(path)
            if match:
                stamps.append(int(match.group(1)))
    if not stamps:
        return None
    return min(stamps), max(stamps), len(stamps)


def qualifying_symbols(data_dir, min_days):
    """Symbols whose tape is long enough to be worth archiving."""
    found = []
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return found
    for entry in sorted(os.listdir(data_dir)):
        if not (data_dir / entry).is_dir():
            continue
        span = tape_span(data_dir, entry)
        if span is None:
            continue
        days = (span[1] - span[0]) / 86_400_000
        if days > min_days:
            found.append(entry)
        else:
            log("  skip %s: %.1f d tape, under the %.0f d line" % (entry, days, min_days))
    return found


# --------------------------------------------------------------------------
# Due-ness, taken from disk rather than from a timer
# --------------------------------------------------------------------------


def last_period(out_dir, symbol):
    """The newest existing period for this symbol, by the date in its name."""
    newest = None
    for path in sorted(Path(out_dir).glob("*_" + symbol)):
        try:
            stamp = dt.datetime.strptime(path.name.split("_")[0], "%Y-%m-%d")
        except ValueError:
            continue
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
        if newest is None or stamp > newest[1]:
            newest = (path, stamp)
    return newest


def is_due(out_dir, symbol, cadence_days, now):
    """Whether `symbol` is due, and the reason either way.

    Deliberately derived from what is on disk, never from a sleep timer. A timer
    restarts its countdown on every reboot, and on a machine that reboots for
    Windows updates it can plausibly never fire -- the same class of failure
    that cost this project a 46 h grid run and a 19.65 h feed blackout.
    """
    previous = last_period(out_dir, symbol)
    if previous is None:
        return True, "no previous archive; establishing the baseline"
    age = (now - previous[1]).total_seconds() / 86_400
    if age >= cadence_days:
        return True, "last archive %s is %.1f d old" % (previous[0].name, age)
    return False, "last archive %s is %.1f d old, due at %.0f d" % (
        previous[0].name, age, cadence_days,
    )


# --------------------------------------------------------------------------
# The grid's curve
# --------------------------------------------------------------------------


def slice_and_downsample(history, target, since_ms, minutes):
    """Write the period's slice of the equity history, thinned to `minutes`.

    Full resolution is 60 s per variant, about 95 MB/month -- far too much to
    commit every three weeks. Thinning to 15 min and compressing keeps a
    month-scale retrospective legible at roughly 1/200th the size. The
    full-resolution equity file stays on local disk and is intentionally not
    rotated; only the per-variant event logs have a rotation ceiling.
    """
    import pandas as pd

    frame = pd.read_csv(history)
    total = len(frame)
    frame = frame[frame["ts_ms"] >= since_ms]
    if frame.empty:
        return {"rows": 0, "rows_in_file": total}
    bucket = int(minutes * 60_000)
    frame = frame.assign(_bucket=frame["ts_ms"] // bucket)
    thinned = frame.groupby(["variant", "_bucket"], as_index=False).last()
    thinned = thinned.drop(columns="_bucket").sort_values("ts_ms")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    thinned.to_csv(target, index=False)
    return {
        "rows": int(len(thinned)),
        "rows_before_thinning": int(len(frame)),
        "rows_in_file": total,
        "resolution_minutes": minutes,
        "first_ts_ms": int(thinned["ts_ms"].iloc[0]),
        "last_ts_ms": int(thinned["ts_ms"].iloc[-1]),
    }


def compress(path):
    """zstd the file, replacing it, and return the compressed path."""
    import zstandard

    path = Path(path)
    target = path.with_suffix(path.suffix + ".zst")
    compressor = zstandard.ZstdCompressor(level=10)
    with path.open("rb") as src, target.open("wb") as dst:
        compressor.copy_stream(src, dst)
    path.unlink()
    return target


# --------------------------------------------------------------------------
# Subprocess
# --------------------------------------------------------------------------


def run(command, label):
    printable = " ".join(str(c) for c in command)
    log("  $ " + printable)
    done = subprocess.run(
        [str(c) for c in command], capture_output=True, text=True, cwd=str(ROOT)
    )
    tail = (done.stdout or "")[-4000:] + (done.stderr or "")[-4000:]
    if done.returncode != 0:
        log("  %s exited %d" % (label, done.returncode))
    return done.returncode, tail


# --------------------------------------------------------------------------
# The period README
# --------------------------------------------------------------------------


def sweep_headline(sweep_json):
    """A few lines a reader can act on without opening the 280 KB payload."""
    try:
        payload = json.loads(Path(sweep_json).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - a bad artifact must not kill the archive
        return ["- sweep payload unreadable: %r" % (error,)]
    lines = [
        "- status: `%s`" % payload.get("status"),
        "- search scenario: `%s`" % payload.get("search_scenario"),
        "- train/held-out split at `%s`" % payload.get("split_at"),
    ]
    stage_c = payload.get("stage_c") or []
    if stage_c:
        lines += ["", "| configuration | train P&L | held-out P&L | fills |", "| --- | ---: | ---: | ---: |"]
        for row in stage_c[:3]:
            lines.append(
                "| `%s` | %+.2f | **%+.2f** | %s |"
                % (
                    row.get("label", "?"),
                    row.get("train_pnl", 0.0),
                    row.get("holdout_pnl", 0.0),
                    row.get("fills", 0),
                )
            )
    ladder = payload.get("latency_ladder") or []
    if ladder:
        lines += ["", "| latency scenario | P&L | fills |", "| --- | ---: | ---: |"]
        for row in ladder:
            lines.append(
                "| %s | %+.2f | %s |"
                % (row.get("scenario", "?"), row.get("pnl", 0.0), row.get("fills", 0))
            )
    return lines


def leaderboard_headline(path):
    try:
        board = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return ["- leaderboard unreadable: %r" % (error,)]
    health = board.get("feed_health") or {}
    failures = board.get("feed_failures") or []
    lines = [
        "- elapsed: %.1f h, %d resume(s)"
        % (board.get("elapsed_seconds", 0) / 3600.0, board.get("resumes", 0)),
        "- feed: %.2f%% down, verdict %s"
        % (health.get("downtime_fraction", 0.0) * 100.0, "INVALID" if failures else "VALID"),
    ]
    for reason in failures:
        lines.append("  - " + reason)
    rows = board.get("rows") or []
    if rows:
        lines += ["", "| variant | net P&L | fills | inventory |", "| --- | ---: | ---: | ---: |"]
        shown = rows[:3] + ([None] if len(rows) > 6 else []) + rows[-3:] if len(rows) > 6 else rows
        for row in shown:
            if row is None:
                lines.append("| ... | | | |")
                continue
            lines.append(
                "| %s | %+.2f | %s | %s |"
                % (row["name"], row["net_pnl_usdc"], row["fills"], row["inventory_units"])
            )
    return lines


def write_readme(period_dir, symbol, facts):
    span = facts["tape_span"]
    body = [
        "# %s - period archived %s" % (symbol, facts["archived_at"]),
        "",
        "Written by `scripts/archive_period.py` because the collector keeps only",
        "%s and this window would otherwise be deleted. A replay can only score a"
        % facts["retention_note"],
        "window while its Parquet shards exist, so once this period rolls off the tape",
        "these files are the only remaining record of it.",
        "",
        "## Provenance",
        "",
        "- tape scored: **%.1f days** (%s -> %s UTC), %d shards"
        % (span["days"], utc(span["oldest_ms"]), utc(span["newest_ms"]), span["shards"]),
        "- grid curve sliced from: %s UTC%s"
        % (
            utc(facts["grid_since_ms"]),
            " (all history - first archive)" if facts["first_archive"] else "",
        ),
        "- build: `%s`" % facts.get("git_revision", "unknown"),
        "",
        "The two windows differ on purpose. The sweep scores the whole tape on disk, so",
        "consecutive archives overlap and a skipped cycle still loses nothing; the grid",
        "curve covers only the period since the last archive, so the archives concatenate",
        "into one continuous timeline.",
        "",
        "## Replay sweep",
        "",
    ]
    body += facts["sweep_lines"]
    body += ["", "## Dry-run grid", ""]
    body += facts["grid_lines"]
    body += [
        "",
        "## Files",
        "",
        "| file | what it is |",
        "| --- | --- |",
        "| `sweep.md` / `sweep.json` | the full staged sweep on the tape above |",
        "| `grid_leaderboard.json` | the grid's ranking at the moment of archiving |",
        "| `grid_equity_curve.csv.zst` | the period's P&L curve, thinned and compressed |",
        "| `grid_pnl_curve.png` | that curve, rendered |",
        "",
    ]
    (Path(period_dir) / "README.md").write_text("\n".join(body) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# One archive
# --------------------------------------------------------------------------


def archive(symbol, args, now):
    span = tape_span(args.data_dir, symbol)
    if span is None:
        log("  %s: no tape, nothing to archive" % symbol)
        return False
    oldest_ms, newest_ms, shards = span
    days = (newest_ms - oldest_ms) / 86_400_000

    final_dir = Path(args.out) / ("%s_%s" % (now.strftime("%Y-%m-%d"), symbol))
    previous = last_period(args.out, symbol)
    grid_since_ms = int(previous[1].timestamp() * 1000) if previous is not None else 0

    log("  %s: tape %.1f d, %d shards -> %s" % (symbol, days, shards, final_dir.name))
    if args.dry_run:
        log("  --dry-run: stopping before the sweep")
        return False

    # Build in `.partial` and rename at the end, so a period directory only ever
    # exists complete. The sweep takes tens of minutes and the artifacts land one
    # at a time; without this, a `git add docs/history` mid-run would commit a
    # half-written period, and `last_period` would count it as done and skip the
    # next cycle. `.partial` is gitignored for the same reason.
    period_dir = final_dir.with_name(final_dir.name + ".partial")
    if period_dir.exists():
        shutil.rmtree(period_dir)
    period_dir.mkdir(parents=True, exist_ok=True)

    # 1. The sweep. --search-max-price-events is left at its default 0 (the full
    #    train slice): a truncated search is exactly the defect the archived
    #    artifact exists to replace.
    _, tail = run(
        [
            sys.executable,
            SCRIPTS / "sweep_replay.py",
            "--symbol", symbol,
            "--data-dir", args.data_dir,
            "--output", period_dir / "sweep.json",
            "--markdown-output", period_dir / "sweep.md",
            "--workers", str(args.workers),
        ],
        "sweep_replay.py",
    )
    if not (period_dir / "sweep.json").exists():
        (period_dir / "sweep_FAILED.log").write_text(tail, encoding="utf-8")
        log("  sweep produced no artifact; keeping the period with its failure log")

    # 2. The grid: leaderboard, the period's slice of the curve, and the render.
    grid_lines = ["- no grid run found"]
    leaderboard = Path(args.grid_dir) / "leaderboard.json"
    if leaderboard.exists():
        shutil.copy2(leaderboard, period_dir / "grid_leaderboard.json")
        grid_lines = leaderboard_headline(period_dir / "grid_leaderboard.json")

    history = Path(args.grid_dir) / "equity_history.csv"
    if history.exists():
        sliced = period_dir / "grid_equity_curve.csv"
        stats = slice_and_downsample(history, sliced, grid_since_ms, args.resolution_minutes)
        if stats["rows"]:
            run(
                [
                    sys.executable,
                    SCRIPTS / "grid_pnl_curve.py",
                    "--history", sliced,
                    "--out", period_dir / "grid_pnl_curve.png",
                ],
                "grid_pnl_curve.py",
            )
            packed = compress(sliced)
            log("  curve: %d rows -> %.0f KB" % (stats["rows"], packed.stat().st_size / 1024))
        else:
            sliced.unlink(missing_ok=True)
            log("  curve: no grid samples inside this period")

    write_readme(
        period_dir,
        symbol,
        {
            "archived_at": now.strftime("%Y-%m-%d"),
            "retention_note": "30 days of CASHCAT tape",
            "tape_span": {
                "oldest_ms": oldest_ms,
                "newest_ms": newest_ms,
                "days": days,
                "shards": shards,
            },
            "grid_since_ms": grid_since_ms or oldest_ms,
            "first_archive": previous is None,
            "git_revision": os.environ.get("MM_GIT_REVISION", "unknown"),
            "sweep_lines": (
                sweep_headline(period_dir / "sweep.json")
                if (period_dir / "sweep.json").exists()
                else ["- **the sweep failed**; see `sweep_FAILED.log`"]
            ),
            "grid_lines": grid_lines,
        },
    )
    size = sum(f.stat().st_size for f in period_dir.rglob("*") if f.is_file())
    # The rename is what publishes the period: until now nothing outside
    # `.partial` existed, so there was never a half-written archive to commit or
    # to mistake for a finished one.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    period_dir.rename(final_dir)
    log("  wrote %s: %.0f KB" % (final_dir.name, size / 1024))
    return True


def uncommitted(out_dir):
    """How many top-level history entries Git reports uncommitted, or -1.

    The container writes but never commits -- that would mean mounting an SSH
    key into it, a poor trade for the convenience. So it says so instead, on
    every wake, and `git status` shows the same thing.
    """
    try:
        # --no-optional-locks so this works against a read-only .git: the
        # container mounts it that way deliberately, since it only ever needs to
        # read. Without the flag git tries to refresh the index and fails.
        result = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain", "--", str(out_dir)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=30,
        )
    except Exception:  # noqa: BLE001
        return -1
    if result.returncode != 0:
        return -1
    periods = set()
    for line in result.stdout.splitlines():
        parts = line[3:].strip().strip('"').split("/")
        if len(parts) > 2:
            periods.add(parts[2])
    return len(periods)


def cycle(args):
    now = dt.datetime.now(dt.timezone.utc)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    symbols = qualifying_symbols(args.data_dir, args.min_tape_days)
    if not symbols:
        log("no symbol has more than %.0f d of tape yet" % args.min_tape_days)
        return
    for symbol in symbols:
        if symbol not in KNOWN_INSTRUMENTS:
            # The sweep's instrument defaults are CASHCAT's; running them against
            # another asset would produce confident numbers for the wrong one.
            log("  %s: qualifies on tape length but has no instrument profile; skipping" % symbol)
            continue
        due, why = is_due(args.out, symbol, args.cadence_days, now)
        log("  %s: %s - %s" % (symbol, "DUE" if (due or args.force) else "not due", why))
        if due or args.force:
            archive(symbol, args, now)
    pending = uncommitted(args.out)
    if pending > 0:
        log("%d history path(s) uncommitted - inspect `git status` before committing" % pending)
    elif pending == 0:
        log("all periods committed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=SCRIPTS / "HL_data")
    parser.add_argument(
        "--grid-dir", type=Path, default=ROOT / "rust_live" / "reports" / "grid_live"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "history")
    parser.add_argument(
        "--cadence-days", type=float, default=21.0,
        help="9 days of slack against the 30-day retention",
    )
    parser.add_argument(
        "--min-tape-days", type=float, default=DEFAULT_MIN_TAPE_DAYS,
        help="a symbol qualifies above this; separates the 30-day collector from the 3-day ones",
    )
    parser.add_argument(
        "--resolution-minutes", type=float, default=15.0,
        help="equity curve thinning; 60 s full resolution is ~95 MB/month",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="sweep workers (capped at 4 inside the sweep)"
    )
    parser.add_argument("--check-interval-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would happen, write nothing"
    )
    parser.add_argument("--force", action="store_true", help="archive even when not due")
    parser.add_argument("--loop", action="store_true", help="stay up and check on the interval")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log("archiver: cadence %.0f d, out %s" % (args.cadence_days, args.out))
    if not args.loop:
        cycle(args)
        return 0
    while True:
        try:
            cycle(args)
        except Exception as error:  # noqa: BLE001 - a bad cycle must not end the daemon
            log("cycle failed: %r" % (error,))
        # --force is a one-shot intent; honouring it every hour would rewrite the
        # same period forever.
        args.force = False
        time.sleep(args.check_interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
