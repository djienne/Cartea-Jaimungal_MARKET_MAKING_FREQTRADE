#!/usr/bin/env python3
"""Archive one retention window before the collector deletes it.

WHY THIS EXISTS. `hl-cashcat-collector` keeps a bounded tape
(`CASHCAT_RETENTION_MINUTES` in HYPERLIQUID_DATA/docker-compose.yml -- the one
place that value is set) and deletes everything older. Two
irreplaceable things ride that clock:

1. **Replay.** A replay can only score a window while its Parquet shards exist.
   Once a period rolls off, no replay can ever be run against it again -- not
   re-run more cheaply, not re-run at all.
2. **The dry-run grid.** Its event logs rotate at ~34 days
   (`--log-max-mb 64 --log-keep 3`), so grid history expires too.

Every `--cadence-days` this attempts to write a period directory holding a fresh
replay plus the grid's P&L curve, small enough to commit. The margin for retrying
an interrupted or failed attempt is retention minus cadence; the failure is
harmless only if a successful `--force` rerun lands before that margin expires.

TWO WINDOW CONVENTIONS, deliberately different, both recorded in the period
README:

- the **replay** scores the whole tape currently on disk (up to the retention
  window), so
  consecutive archives overlap -- that overlap is the safety margin;
- the **grid curve** is sliced to the period since the last archive, preserving
  `run_started_ms` boundaries so independent runs are never spliced.

Usage:
    python scripts/archive_period.py --dry-run     # what would be written
    python scripts/archive_period.py               # archive if due
    python scripts/archive_period.py --force       # archive regardless

The cadence is the Windows task "MM CASHCAT period archive", daily at 04:30;
this script decides for itself whether a period is due, so most runs no-op.
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
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

SHARD_RE = re.compile(r"_(\d{13})\.parquet$")

# A symbol qualifies for archiving if its tape spans more than this. The
# long-retention collector sits far above the line and the 3-day controls far
# below, so the split is unambiguous without
# this repo having to read another project's compose file. A long-retention coin
# added later is picked up automatically once its tape grows past the line.
DEFAULT_MIN_TAPE_DAYS = 7.0

# A period holds a leaderboard, ~22 session reports and a thinned P&L curve.
# The 2026-08-30 archive was 0.2 MB; 50 MB is generous headroom and still far
# below anything that hurts a clone.
SIZE_BUDGET_BYTES = 50 * 1024 * 1024

# A replay is only meaningful for a symbol that has both a dry-run config and a
# grid spec: those carry the instrument's tick size, lot base and variant rows.
# A symbol with a long tape but no profile is refused rather than archived with
# another asset's numbers.
CONFIGS = ROOT / "rust_live" / "config"


def instrument_profile(symbol):
    """The (config, grid spec) pair for a symbol, or None when it has none."""
    config = CONFIGS / ("%s_dryrun_realistic.toml" % symbol.lower())
    spec = CONFIGS / ("grid_%s.toml" % symbol.lower())
    return (config, spec) if config.exists() and spec.exists() else None


def mm_live_binary():
    """The natively built trader binary.

    Replay is a command run from time to time, not a service, so it is a host
    binary rather than a container: only the collectors and the dry-run grid
    need to run continuously. `MM_LIVE_BIN` overrides for an unusual layout.
    Needs a binary built with the default `backtest` feature; a
    `--no-default-features` build has no `replay` subcommand, so the archive
    step fails with a usage error rather than a missing binary.
    """
    override = os.environ.get("MM_LIVE_BIN")
    if override:
        return Path(override)
    suffix = ".exe" if os.name == "nt" else ""
    for profile in ("release", "debug"):
        candidate = ROOT / "rust_live" / "target" / profile / ("mm-live" + suffix)
        if candidate.exists():
            return candidate
    return None


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


def grid_history_paths(grid_dir):
    """Legacy root history plus every immutable run history, oldest first."""
    grid_dir = Path(grid_dir)
    paths = []
    legacy = grid_dir / "equity_history.csv"
    if legacy.is_file():
        paths.append(legacy)
    paths.extend(sorted((grid_dir / "runs").glob("*/equity_history.csv")))
    return paths


def slice_and_downsample(histories, target, since_ms, minutes):
    """Write the period's equity histories, thinned to `minutes`.

    Current runs live under ``runs/<run-id>``; the legacy root file is included
    when present. Independent ``run_started_ms`` values remain separate so a
    reset is never spliced into a continuous-looking curve.
    """
    import pandas as pd

    if isinstance(histories, (str, os.PathLike, Path)):
        histories = [histories]
    frames = [pd.read_csv(path) for path in histories]
    if not frames:
        return {"rows": 0, "rows_in_file": 0}
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    total = len(frame)
    frame = frame[frame["ts_ms"] >= since_ms]
    if frame.empty:
        return {"rows": 0, "rows_in_file": total}
    bucket = int(minutes * 60_000)
    frame = frame.assign(_bucket=frame["ts_ms"] // bucket)
    groups = ["variant", "_bucket"]
    if "run_started_ms" in frame.columns:
        groups.insert(0, "run_started_ms")
    thinned = frame.groupby(groups, as_index=False).last()
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


def replay_headline(board_json):
    """The replay board, rendered the same way the live one is.

    Both boards are the same schema on purpose -- a replay row and a live row
    come from the same `PaperVariant::leaderboard_row` -- so this reuses
    `leaderboard_headline` and only adds what is specific to a replay: the
    window it scored and the latency it assumed.
    """
    try:
        board = json.loads(Path(board_json).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - a bad artifact must not kill the archive
        return ["- replay board unreadable: %r" % (error,)]
    window = board.get("replay")
    if not window:
        return ["- **this is not a replay board**; refusing to present it as one"]
    return [
        "- scored: %s -> %s UTC at %s ms assumed latency"
        % (
            utc(window["scoring_start_ms"]),
            utc(window["scoring_end_ms"]),
            window.get("latency_ms", "?"),
        ),
        "- prefix: first %.0f%% sizes orders and fits recalibrated rows; frozen profiles bypass fitting"
        % (100.0 * float(window.get("train_fraction", 0.0)),),
        "- Frozen-profile results before their original fitting/selection dates are in-sample; overlapping archives are not independent holdouts.",
    ] + leaderboard_headline(board_json)[2:]


def leaderboard_headline(path):
    try:
        board = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return ["- leaderboard unreadable: %r" % (error,)]
    health = board.get("feed_health") or {}
    lines = [
        "- elapsed: %.1f h, %d resume(s)"
        % (board.get("elapsed_seconds", 0) / 3600.0, board.get("resumes", 0)),
        "- feed: %.2f%% down, event loss %s"
        % (health.get("downtime_fraction", 0.0) * 100.0, "YES" if health.get("event_loss") else "no"),
    ]
    # Rank here instead of trusting file order. Boards written before
    # 2026-09-10 lead with the best *promotable* row rather than the best row,
    # and promotability is false for every flatten variant, so `rows[:3]` on
    # one of those headlines the period with `baseline` at -193.50 while
    # `sweep1_flat300` sits at +953.05 further down. Nothing selects on
    # promotability any more, and these archives outlive the tape they score.
    rows = sorted(
        board.get("rows") or [],
        key=lambda row: (
            row.get("promotion_pnl_usdc") is None,
            -(row.get("promotion_pnl_usdc") or 0.0),
            str(row.get("name", "")).lower(),
        ),
    )
    if rows:
        lines += ["", "Exit values are hypothetical closes. Invalid rows retain stopped marks, not executed liquidation proceeds.", "", "| variant | net P&L | fills | inventory |", "| --- | ---: | ---: | ---: |"]
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
        "The two windows differ on purpose. The replay scores the whole tape on disk, so",
        "consecutive archives overlap and a skipped cycle still loses nothing; the grid",
        "curve covers only the period since the last archive. Run boundaries remain explicit",
        "through `run_started_ms`; archive periods do not overlap.",
        "",
        "## Replay",
        "",
        "The same variants the grid ran, scored offline by the same simulator over the",
        "window above. Rows are directly comparable to the grid table below, but they are",
        "not the same measurement: the replay is one continuous pass at an assumed",
        "latency, while a grid row may be stitched across restarts. See",
        "`docs/DRY_RUN_GRID.md` for the fidelity limits.",
        "",
    ]
    body += facts["replay_lines"]
    body += ["", "## Dry-run grid", ""]
    body += facts["grid_lines"]
    body += [
        "",
        "## Files",
        "",
        "| file | what it is |",
        "| --- | --- |",
        "| `replay_leaderboard.json` | every grid variant, replayed over the tape above |",
        "| `replay-<variant>.json` | that variant's full session report |",
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
        log("  --dry-run: stopping before the replay")
        return False

    # Build in `.partial` and rename at the end, so a period directory only ever
    # exists complete. The replay takes many minutes and the artifacts land one
    # at a time; without this, a `git add docs/history` mid-run would commit a
    # half-written period, and `last_period` would count it as done and skip the
    # next cycle. `.partial` is gitignored for the same reason.
    period_dir = final_dir.with_name(final_dir.name + ".partial")
    if period_dir.exists():
        shutil.rmtree(period_dir)
    period_dir.mkdir(parents=True, exist_ok=True)

    # 1. The replay: every variant of the grid spec, over the whole tape on
    #    disk. The train fraction is small because this scores a window the
    #    grid itself ran; a large prefix would throw most of the comparison
    #    away to refit parameters the grid already had.
    binary = mm_live_binary()
    config, spec = instrument_profile(symbol)
    if binary is None:
        (period_dir / "replay_FAILED.log").write_text(
            "no mm-live binary; build it with `cargo build --release` in rust_live/",
            encoding="utf-8",
        )
        log("  no mm-live binary found; keeping the period with its failure log")
    else:
        # The session reports are written to scratch, then only their JSON is
        # copied in. A replay also writes a per-variant event log beside each
        # report, and over a 25-day tape those are ~60 MB PER VARIANT -- 1.6 GB
        # for the spec, into a directory whose whole premise is being small
        # enough to commit. The logs are a debugging artifact of one run, not
        # evidence about the window, so they are not archived at all.
        with tempfile.TemporaryDirectory(prefix="mm-archive-") as scratch:
            _, tail = run(
                [
                    binary,
                    "--config", config,
                    "replay",
                    "--grid", spec,
                    "--all-variants",
                    # The whole tape, explicitly. Without a range `mm-live
                    # replay` falls back to the config's
                    # `calibration.window_minutes` (two hours) ending at the
                    # newest shard -- a calibration window, not an archive, and
                    # it fails closed on InsufficientData long before it
                    # produces anything worth keeping.
                    "--from", str(oldest_ms),
                    "--to", str(newest_ms),
                    "--train-fraction", str(args.train_fraction),
                    "--board", period_dir / "replay_leaderboard.json",
                    "--report", Path(scratch) / "replay",
                ],
                "mm-live replay",
            )
            for report in sorted(Path(scratch).glob("replay-*.json")):
                shutil.copy2(report, period_dir / report.name)
        if not (period_dir / "replay_leaderboard.json").exists():
            (period_dir / "replay_FAILED.log").write_text(tail, encoding="utf-8")
            log("  replay produced no artifact; keeping the period with its failure log")

    # 2. The grid: leaderboard, the period's slice of the curve, and the render.
    grid_lines = ["- no grid run found"]
    leaderboard = Path(args.grid_dir) / "leaderboard.json"
    if leaderboard.exists():
        shutil.copy2(leaderboard, period_dir / "grid_leaderboard.json")
        grid_lines = leaderboard_headline(period_dir / "grid_leaderboard.json")

    histories = grid_history_paths(args.grid_dir)
    if histories:
        sliced = period_dir / "grid_equity_curve.csv"
        stats = slice_and_downsample(histories, sliced, grid_since_ms, args.resolution_minutes)
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
            "retention_note": "a retention-limited window of CASHCAT tape",
            "tape_span": {
                "oldest_ms": oldest_ms,
                "newest_ms": newest_ms,
                "days": days,
                "shards": shards,
            },
            "grid_since_ms": grid_since_ms or oldest_ms,
            "first_archive": previous is None,
            "git_revision": os.environ.get("MM_GIT_REVISION", "unknown"),
            "replay_lines": (
                replay_headline(period_dir / "replay_leaderboard.json")
                if (period_dir / "replay_leaderboard.json").exists()
                else ["- **the replay failed**; see `replay_FAILED.log`"]
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
    # These are committed, so a period that quietly grows by three orders of
    # magnitude is a defect in whatever wrote it, not something to discover in
    # `git push`. The archive is kept either way -- evidence beats tidiness --
    # but it says so loudly enough that nobody commits 1.6 GB by accident.
    if size > SIZE_BUDGET_BYTES:
        biggest = sorted(
            ((sum(f.stat().st_size for f in e.rglob("*")) if e.is_dir() else e.stat().st_size, e.name)
             for e in final_dir.iterdir()),
            reverse=True,
        )[:3]
        log("  WARNING %s is %.0f MB, over the %.0f MB budget for a committed period"
            % (final_dir.name, size / 1e6, SIZE_BUDGET_BYTES / 1e6))
        for entry_size, name in biggest:
            log("    %-32s %.0f MB" % (name, entry_size / 1e6))
        log("    do not commit this until it is understood")
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
        if instrument_profile(symbol) is None:
            # Without a config and a grid spec there is no instrument to replay
            # against, and borrowing another asset's would produce confident
            # numbers for the wrong one.
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
        help="slack for a failed attempt is retention minus cadence",
    )
    parser.add_argument(
        "--min-tape-days", type=float, default=DEFAULT_MIN_TAPE_DAYS,
        help="a symbol qualifies above this; separates the long-retention collector from the 3-day ones",
    )
    parser.add_argument(
        "--resolution-minutes", type=float, default=15.0,
        help="equity curve thinning; 60 s full resolution is ~95 MB/month",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.05,
        help="prefix used only to fit and size; the rest is scored",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would happen, write nothing"
    )
    parser.add_argument("--force", action="store_true", help="archive even when not due")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log("archiver: cadence %.0f d, out %s" % (args.cadence_days, args.out))
    cycle(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
