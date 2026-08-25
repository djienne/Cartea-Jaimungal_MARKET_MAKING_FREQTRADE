#!/usr/bin/env python3
"""Staged parameter sweep for the market maker, on one pinned tape.

WHAT THIS IS FOR. Every replay number this project has published was a single
configuration scored once. That is enough to say "this setting loses money" and
not enough to say "no setting makes money", which is the question actually being
asked of CASHCAT. This driver searches the model's own knobs systematically and
reports the answer either way.

FOUR RULES IT ENFORCES, each because breaking it has already cost this project a
result:

1. ONE TAPE, LOADED ONCE. The collector keeps writing shards, so two runs minutes
   apart score different data. That invalidated a phi sweep where one setting came
   back at both -3.98 and -19.39 USDC. Everything here scores bytes from a single
   ``load_tape``.

2. CALIBRATION IS FITTED ON TRAIN ONLY. kappa/epsilon/lambda are estimated from the
   train slice and then applied unchanged to the held-out slice. Fitting on the
   whole tape and reporting the held-out score is the oldest way to manufacture an
   edge that is not there.

3. THE SEARCH TOUCHES THE BOOK'S KNOBS AND THE CALIBRATION'S FREE CHOICES, NOTHING
   ELSE. phi*kappa*T, alpha*kappa, T and the inventory grid are the model's own risk
   preference. The epsilon horizon and the kappa fit support are choices the
   estimator has to make and cannot read off the book. What is NOT swept:
   extra_cushion_bps and the half-spread clamps, which are additive widening bolted
   on top of delta* -- tuning those would produce "delta* plus a fudge" and the
   result would not be the book's.

4. EPSILON HORIZONS STAY INSIDE WHAT EPSILON IS. eq. 10.22 defines epsilon as the
   midprice jump AT the market order's arrival. A long markout is a different
   estimand: it absorbs every other MO in the window, and the HJB already carries
   that net-flow drift in the q(lambda+ eps+ - lambda- eps-) term of eq. 10.26, so a
   long horizon double-counts the trend. The data says the same thing -- measured on
   CASHCAT, eps+ runs 2.62 -> 6.12 bps from 200 ms to 5 s and then FALLS BACK to
   1.86 bps at 30 s, and a permanent impact does not decay. So the candidate
   horizons stop at 1 s. 5 s and 30 s are computed as diagnostics and are never
   allowed into a shipped calibration.

WHAT IT DOES NOT DO. It does not decide whether the winner is deployable. That is
the acceptance rule in Phase 4: net positive out-of-sample with enough fills,
positive on most held-out days, and still positive under the fee and latency
stresses. This driver produces the evidence; a human reads it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from replay_market_maker import (  # noqa: E402
    ReplayConfig,
    ReplayTape,
    load_tape,
    run_replay,
)

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

# Epsilon post-trade horizons, per side, in ms. Capped at 1 s on purpose -- see
# rule 4 in the module docstring. 200 ms is the shipped default; 500 ms and 1 s
# allow for an illiquid coin's BBO taking longer than 200 ms to settle after an
# arrival, which is the only honest reason to look past the default.
EPSILON_HORIZONS_MS = (200, 500, 1000)

# Lower quantile of the market-order depth distribution the kappa survival fit is
# regressed over. 0.0 is the shipped behaviour (fit everything). Raising it drops
# the shallow orders that dominate the sample, which is worth testing because
# 1/kappa is meant to describe the depth our quotes actually rest at, not the
# depth of the median tiny market order.
KAPPA_SUPPORT_LOWER = (0.0, 0.5, 0.75)

# The book's running inventory penalty, as the dimensionless product the model
# actually responds to (eq. 10.28 puts phi in the transition matrix as
# -phi*kappa*q^2, so the raw value is meaningless across symbols). The shipped
# setting is 10.0 and is explicitly NOT a measured optimum.
#
# The upper end is set by REACHABILITY, measured rather than guessed. At the live
# CASHCAT calibration the flat-inventory half-spread runs:
#
#   phi*kappa*T     1      3     10     30    100    300   1000
#   bid q=0 bps  14.8   15.8   18.2   23.2   32.4   41.9   52.5
#   ask q=0 bps  11.1   12.3   14.9   19.8   28.4   37.3   47.2
#
# and the 23.1 h profit curve puts the empirical optimum at 45 bps on the bid and
# 60 bps on the ask. A grid stopping at 100 therefore could not reach the region
# it is supposed to be searching -- it would have reported "no setting is
# profitable" about a range that never contained the candidate.
PHI_KAPPA_T = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

# The terminal penalty, same normalisation. This one has never been swept: it was
# set to 0.05 while the agent read only the t=0 slice, which left alpha
# structurally inert. Making the control episodic made it live, so the shipped
# value carries no evidence at all.
ALPHA_KAPPA = (0.05, 0.5, 5.0)

# Episode length in seconds. 150 was best in docs/time_mode_ab.md, but that A/B
# ran at 500 ms latency with a symmetric long-only solve, so it is a prior rather
# than a settled answer.
HORIZONS_SECONDS = (150.0, 300.0, 600.0)

# Inventory grid radius. The book's own sizing rule is q_max ~ 20% of the expected
# trade count over T.
Q_MAX = (3, 6)


@dataclass(frozen=True)
class Calibration:
    """One estimator setting. These are the choices the book does not make for us."""

    epsilon_ms_plus: int = 200
    epsilon_ms_minus: int = 200
    kappa_support_lower_plus: float = 0.0
    kappa_support_lower_minus: float = 0.0

    def key(self) -> str:
        return (
            f"eps{self.epsilon_ms_plus}/{self.epsilon_ms_minus}"
            f"_klo{self.kappa_support_lower_plus:g}/{self.kappa_support_lower_minus:g}"
        )

    def is_shipped_default(self) -> bool:
        return self == Calibration()


@dataclass(frozen=True)
class RiskSetting:
    """One point of the book's own risk preference."""

    phi_kappa_t: float = 10.0
    alpha_kappa: float = 0.05
    horizon_seconds: float = 150.0
    q_max: int = 6

    def key(self) -> str:
        return (
            f"phiKT{self.phi_kappa_t:g}_alphaK{self.alpha_kappa:g}"
            f"_T{self.horizon_seconds:g}_q{self.q_max}"
        )


@dataclass(frozen=True)
class Scenario:
    """A coherent machine: order latency and requote cadence move together.

    Sweeping latency alone at a fixed refresh interval measures almost nothing,
    because resting exposure is max(0, refresh - latency) + cancel -- cutting
    latency at a fixed refresh actually LENGTHENS the time a quote sits. Each row
    here is a plausible piece of infrastructure instead.
    """

    name: str
    latency_ms: int
    refresh_ms: int


SCENARIOS = (
    Scenario("colocated", 50, 100),
    Scenario("good", 100, 250),
    Scenario("mid", 200, 500),
    Scenario("this_stack", 500, 1000),
    Scenario("reality", 500, 30000),
)

# The scenario every stage of the search is scored at. Searching at "reality"
# would tune the model to a 30 s requote interval, which is an infrastructure
# defect rather than a property of the market; searching at "colocated" would tune
# to a rung the tape cannot even resolve. "good" is the fastest rung that sits
# above the tape's own cadence.
SEARCH_SCENARIO = "good"


# ---------------------------------------------------------------------------
# Tape handling
# ---------------------------------------------------------------------------


def iso_bound(value: pd.Timestamp) -> str:
    """A window bound the estimator can actually parse.

    ``pd.Timestamp.isoformat()`` emits NANOSECOND precision when the timestamp has
    it -- '2026-08-17T15:17:11.508999999+00:00' -- and ``datetime.fromisoformat``,
    which is what estimator_common.parse_window_bound_ms falls back on, rejects
    nine fractional digits. Every calibration in the first full-tape sweep failed
    on exactly this, and the failure looked like a broken estimator rather than a
    malformed argument.

    Flooring to milliseconds is lossless for this purpose: the collector stamps
    shards in milliseconds and the estimator selects them the same way, so there is
    no sub-millisecond boundary to preserve.
    """
    return value.floor("ms").isoformat()


def tape_bounds(tape: ReplayTape) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last event on the tape, across every stream that has rows."""
    stamps: list[pd.Timestamp] = []
    for frame in (tape.prices, tape.trades, tape.orderbooks):
        if frame is not None and len(frame) and "timestamp" in frame.columns:
            column = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
            if len(column):
                stamps.append(column.min())
                stamps.append(column.max())
    if not stamps:
        raise SystemExit("tape carries no timestamped rows")
    return min(stamps), max(stamps)


def slice_tape(tape: ReplayTape, start: pd.Timestamp, end: pd.Timestamp) -> ReplayTape:
    """A half-open [start, end) view of the tape.

    Slicing rather than reloading is the whole point: a train slice and a held-out
    slice cut from ONE load are guaranteed to be the same bytes the full-tape run
    saw, so a difference between them is the split and not the collector.
    """

    def cut(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or not len(frame) or "timestamp" not in frame.columns:
            return frame
        stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        mask = (stamps >= start) & (stamps < end)
        return frame.loc[mask].reset_index(drop=True)

    return ReplayTape(
        prices=cut(tape.prices),
        trades=cut(tape.trades),
        orderbooks=cut(tape.orderbooks),
        input_files=dict(tape.input_files),
    )


def freeze_tape(tape: ReplayTape, directory: Path) -> Path:
    """Write a tape to disk so worker processes can share the exact same bytes.

    A grid of any useful size runs in parallel (a replay is ~26-28 us per price
    event, so a full-train search is minutes, not hours -- see truncate_tape).
    Windows has no fork, which means a pool would otherwise pickle the whole tape
    to every worker for every task. Freezing it once and having each worker load
    it once is both faster and STRICTER: the
    workers are provably scoring one immutable file rather than three DataFrames
    that were serialised separately per task.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("prices", tape.prices), ("trades", tape.trades), ("orderbooks", tape.orderbooks)):
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    (directory / "input_files.json").write_text(json.dumps(tape.input_files), encoding="utf-8")
    return directory


def thaw_tape(directory: Path) -> ReplayTape:
    """Load a frozen tape. The inverse of :func:`freeze_tape`."""
    return ReplayTape(
        prices=pd.read_parquet(directory / "prices.parquet"),
        trades=pd.read_parquet(directory / "trades.parquet"),
        orderbooks=pd.read_parquet(directory / "orderbooks.parquet"),
        input_files=json.loads((directory / "input_files.json").read_text(encoding="utf-8")),
    )


# Per-worker state. A pool worker loads its tape once in the initializer and then
# scores many grid points against it; re-reading per task would dominate runtime.
_WORKER_TAPES: dict[str, ReplayTape] = {}


def _worker_init(tape_dirs: dict[str, str]) -> None:
    for label, directory in tape_dirs.items():
        _WORKER_TAPES[label] = thaw_tape(Path(directory))


def _worker_score(job: dict[str, Any]) -> dict[str, Any]:
    """Score one grid point inside a pool worker."""
    tape = _WORKER_TAPES[job["tape"]]
    config = ReplayConfig(**job["config"])
    return {"index": job["index"], **score(config, job["params"], tape, min_fills=job["min_fills"])}


def _config_kwargs(config: ReplayConfig) -> dict[str, Any]:
    """ReplayConfig as plain kwargs, so a job crosses the process boundary cheaply."""
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def run_jobs(
    jobs: list[dict[str, Any]], *, workers: int, progress: str
) -> list[dict[str, Any]]:
    """Score every job, in a pool when asked and inline when not."""
    if workers <= 1 or len(jobs) <= 1:
        results = []
        for position, job in enumerate(jobs, start=1):
            results.append({"index": job["index"], **score(
                ReplayConfig(**job["config"]), job["params"], _WORKER_TAPES[job["tape"]],
                min_fills=job["min_fills"],
            )})
            print(f"  [{progress}] {position}/{len(jobs)}", flush=True)
        return results

    import multiprocessing as mp

    tape_dirs = {label: str(path) for label, path in _FROZEN_TAPES.items()}
    results = []
    with mp.Pool(processes=workers, initializer=_worker_init, initargs=(tape_dirs,)) as pool:
        for position, result in enumerate(pool.imap_unordered(_worker_score, jobs), start=1):
            results.append(result)
            if position % max(1, len(jobs) // 20) == 0 or position == len(jobs):
                print(f"  [{progress}] {position}/{len(jobs)}", flush=True)
    return sorted(results, key=lambda row: row["index"])


# Frozen tapes for this process, by label. Populated once in run_sweep.
_FROZEN_TAPES: dict[str, Path] = {}


def truncate_tape(tape: ReplayTape, max_price_rows: int | None) -> ReplayTape:
    """A contiguous prefix of a tape, bounded by price-event count.

    A random subsample would break the event ordering the replay depends on, so
    when this is used it takes a contiguous stretch.

    HISTORY, because the reason this existed was wrong for most of its life.
    The original docstring said "one run over ~100k price rows takes about three
    minutes" -- 1.8 ms per price event. That was the pre-optimisation replay.
    Commit 5afa6f6 ("51x faster") and then b10fbc5 (numba) cut it to ~26-28 us
    per event, and this module was written in the commit immediately AFTER
    5afa6f6, so the figure was already stale the day it was recorded. It then
    justified two deferrals of the fix.

    Measured now: searching the full train split costs ~30 min at four workers,
    not the ~32 h the stale figure implied. So truncation is no longer a cost
    necessity, and `--search-max-price-events` now defaults to 0 (off). It is
    kept as a knob for quick exploratory runs, where a deliberately partial
    search is fine as long as the artifact says so -- which it now does.
    """
    if not max_price_rows or len(tape.prices) <= int(max_price_rows):
        return tape
    stamps = pd.to_datetime(tape.prices["timestamp"], utc=True, errors="coerce")
    cutoff = stamps.iloc[int(max_price_rows) - 1]
    start = stamps.min()
    return slice_tape(tape, start, cutoff)


def tape_identity(tape: ReplayTape) -> dict[str, Any]:
    """What data this was, so a result can never be read against the wrong tape."""
    start, end = tape_bounds(tape)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "hours": round((end - start).total_seconds() / 3600.0, 4),
        "price_rows": int(len(tape.prices)),
        "trade_rows": int(len(tape.trades)),
        "orderbook_rows": int(len(tape.orderbooks)),
        "input_files": dict(tape.input_files),
    }


def window_slices(
    tape: ReplayTape, start: pd.Timestamp, end: pd.Timestamp, *, hours: float
) -> list[tuple[str, ReplayTape]]:
    """Consecutive equal-length slices of a span.

    WHY EVERY CANDIDATE IS SCORED PER SUB-WINDOW AND NOT ONLY IN TOTAL. The
    per-side edge on this instrument is not stable. Measured on an 11 h CASHCAT
    window the resting ask earned and the resting bid lost at every depth; measured
    on the 23.1 h window that contains it, the bid is the better side by an order
    of magnitude (+13.24 bps edge at 45 bps depth against the ask's +1.75). Which
    side pays is a property of the regime the window landed in, not of the market.

    A total hides exactly that. One configuration that wins in every stretch and
    one that loses in all but a single lucky run can produce the same aggregate,
    and only the first is a strategy. So a candidate reports where it made its
    money, and a reader can see whether it is one lucky window wearing a total.
    """
    out: list[tuple[str, ReplayTape]] = []
    span = timedelta(hours=float(hours))
    cursor = start
    while cursor < end:
        stop = min(cursor + span, end)
        piece = slice_tape(tape, cursor, stop)
        if len(piece.prices):
            out.append((cursor.strftime("%m-%d %H:%M"), piece))
        cursor = stop
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def shard_index(data_dir: Path, symbol: str) -> dict[str, int] | None:
    """Every input shard the estimator could read, as {stream/name: bytes}.

    Sizes rather than mtimes: a shard is written once and its mtime moves when
    nothing about its contents has, which would make any cache built on it
    useless without making it safer. None means the directory could not be read,
    which disables caching rather than guessing.
    """
    index: dict[str, int] = {}
    for stream in ("prices", "trades", "orderbooks"):
        directory = Path(data_dir) / symbol / stream
        if not directory.is_dir():
            continue
        try:
            for entry in os.scandir(directory):
                if entry.is_file() and entry.name.endswith(".parquet"):
                    index[f"{stream}/{entry.name}"] = entry.stat().st_size
        except OSError:
            return None
    return index


def data_fingerprint(data_dir: Path, symbol: str) -> str:
    """:func:`shard_index` as one hash, for reporting and for tests."""
    index = shard_index(data_dir, symbol)
    if index is None:
        return "unreadable"
    digest = hashlib.sha256()
    for name in sorted(index):
        digest.update(f"{name}:{index[name]};".encode())
    return digest.hexdigest()


def shard_index_still_holds(recorded: dict[str, int], current: dict[str, int] | None) -> bool:
    """Is every shard the fit READ still present, unchanged?

    Not equality, and the difference is the whole design. The collector is
    appending shards the entire time a sweep runs, so requiring the directory to
    be untouched means the cache never fires on the only host it matters on --
    measured: the shard set changed during every single estimator pass. What has
    to hold is narrower: the bytes the fit was computed FROM are still there and
    still say the same thing.

    The assumption that makes this sound is that the collector only ever appends
    -- a new shard carries events later than every shard before it. That is what
    the collector does, and it is the same assumption the tape pinning already
    rests on. It matters because the estimator fits over a WINDOW: a sweep's
    calibration window ends at the train/held-out split, deep inside a tape that
    was already on disk, so a shard flushed afterwards cannot contain a row
    inside it. If that ever stops being true -- a backfill, an importer writing
    history -- this cache would hand back a fit made from less data than a fresh
    one would use, and --no-calibration-cache is the way out.
    """
    if current is None:
        return False
    return all(current.get(name) == size for name, size in recorded.items())


def calibration_cache_key(
    calibration: Calibration,
    *,
    symbol: str,
    data_dir: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict[str, Any]:
    """Everything that decides what the estimator returns, except the data."""
    return {
        "schema": 1,
        "symbol": str(symbol),
        "data_dir": str(Path(data_dir).resolve()),
        "window_start": pd.Timestamp(window_start).isoformat(),
        "window_end": pd.Timestamp(window_end).isoformat(),
        "calibration": dataclasses.asdict(calibration),
    }


def calibrate(
    calibration: Calibration,
    *,
    symbol: str,
    data_dir: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    scratch: Path,
    reuse: bool = True,
) -> dict[str, Any]:
    """Estimate kappa/epsilon/lambda on one window at one calibration setting.

    Runs estimate_all.py in --emit-params-json mode, which computes exactly what
    the live estimator computes but writes ONLY to the path given. Kept that way
    after the freqtrade legs retired: nothing reads the live snapshots today, but
    a sweep that writes outside its own output path is one live consumer away
    from changing the thing it is measuring.

    ``reuse`` lets a re-run skip the subprocess when the scratch directory
    already holds a fit made from the same arguments over shards that have not
    changed since -- one estimator pass costs about six seconds on a 16 h tape
    and grows with the data, and the ``full`` calibration grid is 81 of them, so
    a re-run otherwise spends most of its wall clock recomputing numbers it
    already has. What "have not changed" means, and why it is not simply "the
    directory is identical", is in :func:`shard_index_still_holds`.
    """
    cached_at = scratch / f"cache_{calibration.key().replace('/', '-')}.json"
    cache_key = calibration_cache_key(
        calibration,
        symbol=symbol,
        data_dir=data_dir,
        window_start=window_start,
        window_end=window_end,
    )
    index_before = shard_index(data_dir, symbol)
    if reuse and cached_at.exists():
        try:
            cached = json.loads(cached_at.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("key") == cache_key
            and isinstance(cached.get("shards"), dict)
            and isinstance(cached.get("result"), dict)
            and shard_index_still_holds(cached["shards"], index_before)
        ):
            return {**cached["result"], "cached": True}

    out = scratch / f"params_{calibration.key().replace('/', '-')}.json"
    command = [
        sys.executable,
        str(SCRIPTS / "estimate_all.py"),
        "--crypto", symbol,
        "--data-dir", str(data_dir),
        "--window-start", iso_bound(window_start),
        "--window-end", iso_bound(window_end),
        "--post-horizon-ms-plus", str(calibration.epsilon_ms_plus),
        "--post-horizon-ms-minus", str(calibration.epsilon_ms_minus),
        "--support-quantile-lower-plus", str(calibration.kappa_support_lower_plus),
        "--support-quantile-lower-minus", str(calibration.kappa_support_lower_minus),
        "--emit-params-json", str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(SCRIPTS))
    if result.returncode != 0 or not out.exists():
        return {"ok": False, "reason": f"estimator_failed:{result.returncode}", "stderr": result.stderr[-400:]}

    payload = json.loads(out.read_text(encoding="utf-8"))
    kappa = payload.get("kappa") or {}
    epsilon = payload.get("epsilon") or {}
    lambda_ = payload.get("lambda") or {}
    for block, name in ((kappa, "kappa"), (epsilon, "epsilon"), (lambda_, "lambda")):
        if str(block.get("status", "")).lower() != "ok":
            return {"ok": False, "reason": f"{name}_status_{block.get('status')}"}

    params = {
        "kappa+": kappa.get("kappa+"),
        "kappa-": kappa.get("kappa-"),
        "lambda+": lambda_.get("lambda+"),
        "lambda-": lambda_.get("lambda-"),
        "epsilon+": epsilon.get("epsilon+"),
        "epsilon-": epsilon.get("epsilon-"),
    }
    if any(value is None for value in params.values()):
        return {"ok": False, "reason": "incomplete_params"}
    fitted = {
        "ok": True,
        "params": {name: float(value) for name, value in params.items()},
        "diagnostics": {
            "r2_plus": kappa.get("r2_plus"),
            "r2_minus": kappa.get("r2_minus"),
            "n_points_plus": kappa.get("n_points_plus"),
            "n_points_minus": kappa.get("n_points_minus"),
            "n_buy_events": epsilon.get("n_buy_events"),
            "n_sell_events": epsilon.get("n_sell_events"),
            "toxicity_plus": epsilon.get("toxicity_plus"),
            "toxicity_minus": epsilon.get("toxicity_minus"),
            "window": payload.get("window"),
        },
    }
    # Record what the estimator READ, which is the shard set as it was before it
    # started -- and only if every one of those shards survived the run
    # unchanged. A shard rewritten or truncated under the reader would make the
    # fit describe bytes the index does not, and a stale entry wearing a
    # confident key is worse than no cache at all. Shards that APPEARED during
    # the run are fine and are simply not recorded; see shard_index_still_holds.
    if index_before is not None and shard_index_still_holds(
        index_before, shard_index(data_dir, symbol)
    ):
        cached_at.write_text(
            json.dumps(
                {"key": cache_key, "shards": index_before, "result": fitted},
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return fitted


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def build_config(
    base: ReplayConfig, risk: RiskSetting, scenario: Scenario
) -> ReplayConfig:
    """One replay configuration. dataclasses.replace, never field-by-field.

    Rebuilding a config attribute by attribute is how run_replay_report silently
    reset every knob it did not list, so each acceptance variant ran the whole HJB
    block at library defaults.
    """
    latency_half = int(scenario.latency_ms) // 2
    return dataclasses.replace(
        base,
        hjb_phi_kappa_t=risk.phi_kappa_t,
        # The ceiling must be lifted above whatever the grid is asking for, or the
        # sweep silently measures the ceiling instead of the setting. It defaults to
        # 50, so every grid point above that would have collapsed onto one identical
        # configuration -- the same class of degeneracy that made params_soft and
        # params_hard return byte-identical metrics. The ceiling exists to stop the
        # sigma2 volatility channel re-pinning quotes to the clamps, which is a
        # runtime safety concern and not a reason to refuse to measure a value here.
        hjb_phi_kappa_t_max=max(float(base.hjb_phi_kappa_t_max), float(risk.phi_kappa_t)),
        hjb_alpha_kappa=risk.alpha_kappa,
        hjb_horizon_seconds=risk.horizon_seconds,
        q_max=risk.q_max,
        q_min=-risk.q_max,
        decision_latency_ms=latency_half,
        order_ack_latency_ms=scenario.latency_ms - latency_half,
        quote_refresh_interval_ms=scenario.refresh_ms,
    )


def score(
    config: ReplayConfig,
    params: dict[str, float],
    tape: ReplayTape,
    *,
    min_fills: int,
) -> dict[str, Any]:
    """Run one configuration and reduce it to the numbers a sweep compares.

    ``usable`` is separate from P&L on purpose. A configuration that quotes so wide
    it takes three fills can post a lovely number and has measured nothing; the
    honest way to express that is to keep the P&L and refuse to rank it.
    """
    try:
        metrics = run_replay(config, params, tape=tape).to_dict()
    except Exception as exc:  # a bad grid point must not end the sweep
        return {"usable": False, "reason": f"replay_failed:{type(exc).__name__}:{exc}"[:200]}

    fills = int(metrics.get("maker_fills", 0))
    by_side = metrics.get("fills_by_side") or {}
    pnl_by_side = metrics.get("pnl_by_side") or {}
    depth_by_side = metrics.get("mean_fill_depth_bps_by_side") or {}
    return {
        "usable": fills >= int(min_fills),
        "reason": None if fills >= int(min_fills) else f"insufficient_fills:{fills}<{min_fills}",
        "pnl_usdc": float(metrics.get("equity_usdc", 0.0)) - float(metrics.get("starting_equity_usdc", 0.0)),
        "net_realized_spread_usdc": float(metrics.get("realized_spread_usdc", 0.0)) - float(metrics.get("fees_usdc", 0.0)),
        "maker_fills": fills,
        "taker_fills": int(metrics.get("taker_fills", 0)),
        "fills_bid": int(by_side.get("bid", 0)),
        "fills_ask": int(by_side.get("ask", 0)),
        "pnl_bid": float(pnl_by_side.get("bid", 0.0)),
        "pnl_ask": float(pnl_by_side.get("ask", 0.0)),
        "depth_bid_bps": depth_by_side.get("bid"),
        "depth_ask_bps": depth_by_side.get("ask"),
        "quote_attempts": int(metrics.get("quote_attempts", 0)),
        "post_only_reject_ratio": float(metrics.get("post_only_reject_ratio", 0.0)),
        "final_inventory_base": float(metrics.get("inventory_base", 0.0)),
        # Spread capture is the part of P&L that market making earns. The rest is
        # a directional bet on whatever the mid did. A row where they diverge is a
        # position, not a strategy -- a +38.15 USDC run on this instrument turned
        # out to be an accumulated short in a -3.7% window with half a cent of
        # spread capture behind it.
        "directional_usdc": (
            float(metrics.get("equity_usdc", 0.0)) - float(metrics.get("starting_equity_usdc", 0.0))
        ) - (float(metrics.get("realized_spread_usdc", 0.0)) - float(metrics.get("fees_usdc", 0.0))),
    }


def rank_key(row: dict[str, Any]) -> tuple:
    """Rank usable rows by P&L, and push unusable ones to the bottom regardless."""
    return (1 if row.get("usable") else 0, float(row.get("pnl_usdc", 0.0)))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def calibration_grid(mode: str) -> list[Calibration]:
    """``matched`` moves both sides together; ``full`` crosses them.

    Matched is the cheap first pass. Full is what actually answers the CASHCAT
    question, because the whole diagnosis is that the two sides need different
    treatment -- a matched-only sweep cannot express that.
    """
    if mode == "matched":
        return [
            Calibration(eps, eps, lo, lo)
            for eps in EPSILON_HORIZONS_MS
            for lo in KAPPA_SUPPORT_LOWER
        ]
    return [
        Calibration(eps_p, eps_m, lo_p, lo_m)
        for eps_p, eps_m, lo_p, lo_m in itertools.product(
            EPSILON_HORIZONS_MS, EPSILON_HORIZONS_MS, KAPPA_SUPPORT_LOWER, KAPPA_SUPPORT_LOWER
        )
    ]


def parse_float_list(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    """Comma-separated override for one grid axis, or the shipped default."""
    if not text:
        return tuple(float(value) for value in default)
    return tuple(float(part) for part in str(text).split(",") if part.strip())


def risk_grid(
    phi_values: Iterable[float] | None = None,
    alpha_values: Iterable[float] | None = None,
) -> list[RiskSetting]:
    """The risk grid, with the two axes worth overriding exposed.

    phi is overridable because the first full sweep put its winner at the TOP of
    the shipped range (phi*kappa*T = 1000), and a winner on a boundary is not a
    winner -- it is a grid that stopped too early. Extending it answers the
    question that decides whether that result means anything: does P&L keep
    rising as fills collapse toward zero (the signature of negative expected
    value per fill, where "best" just means "trade least"), or does it peak and
    turn over (a genuine interior optimum)?
    """
    phis = tuple(phi_values) if phi_values is not None else PHI_KAPPA_T
    alphas = tuple(alpha_values) if alpha_values is not None else ALPHA_KAPPA
    return [
        RiskSetting(phi, alpha, horizon, q_max)
        for phi, alpha, horizon, q_max in itertools.product(
            phis, alphas, HORIZONS_SECONDS, Q_MAX
        )
    ]


def scenario_by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise SystemExit(f"unknown scenario {name!r}")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    base = ReplayConfig(
        symbol=args.symbol,
        data_dir=Path(args.data_dir),
        mid_fallback=args.mid,
        inventory_unit_base=args.inventory_unit_base,
        price_tick_size=args.price_tick_size,
        amount_step_size=args.amount_step_size,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        starting_equity_usdc=args.starting_equity_usdc,
        leverage=args.leverage,
        newest_per_stream=args.newest_per_stream,
        max_price_events=args.max_price_events,
        hjb_time_mode="episodic",
        event_clock="exchange",
    )

    print(f"[tape] loading {args.symbol} from {args.data_dir} ...", flush=True)
    tape = load_tape(base)
    identity = tape_identity(tape)
    start, end = tape_bounds(tape)
    print(
        f"[tape] {identity['hours']} h, {identity['price_rows']} price rows, "
        f"{identity['trade_rows']} trades",
        flush=True,
    )

    # Floored to milliseconds so the tape slice and the calibration window agree on
    # the same instant: the replay slices on this Timestamp while the estimator gets
    # it as a millisecond bound, and an unfloored split would put them a few hundred
    # nanoseconds apart for no gain.
    split_at = (start + (end - start) * float(args.train_fraction)).floor("ms")
    train = slice_tape(tape, start, split_at)
    held_out = slice_tape(tape, split_at, end + timedelta(seconds=1))
    print(
        f"[split] train {tape_identity(train)['hours']} h -> held-out "
        f"{tape_identity(held_out)['hours']} h at {split_at.isoformat()}",
        flush=True,
    )

    search_tape = truncate_tape(train, args.search_max_price_events)
    search_identity = tape_identity(search_tape)
    train_identity = tape_identity(train)
    search_is_partial = len(search_tape.prices) < len(train.prices)
    print(
        f"[search] scoring the grid on {search_identity['hours']} h / "
        f"{len(search_tape.prices)} price rows of the train slice"
        + (
            f"  ** PARTIAL: {train_identity['hours']} h available **"
            if search_is_partial
            else ""
        ),
        flush=True,
    )

    _FROZEN_TAPES["search"] = freeze_tape(search_tape, scratch / "tape_search")
    _FROZEN_TAPES["held_out"] = freeze_tape(held_out, scratch / "tape_held_out")
    _WORKER_TAPES["search"] = search_tape
    _WORKER_TAPES["held_out"] = held_out

    search = scenario_by_name(args.search_scenario)
    calibrations = calibration_grid(args.calibration_mode)
    print(
        f"[stage A] {len(calibrations)} calibrations at risk reference "
        f"{RiskSetting().key()}, scenario {search.name}, {args.workers} workers",
        flush=True,
    )

    # Stage A -- calibration, with the risk knobs held at what ships today.
    #
    # The estimator runs as a subprocess, so these are launched from a thread
    # pool: the work is entirely in the child processes and the parent only
    # waits on them. Concurrency is the SAME --workers budget the replay pool
    # uses, for the same reason -- the Rust dry-run grid and several
    # collectors share this host, and starving them would corrupt the data
    # being measured. Results are collected back in grid order, so the
    # report does not depend on which one finished first.
    reference_risk = RiskSetting()
    stage_a: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    def fit(calibration: Calibration) -> dict[str, Any]:
        return calibrate(
            calibration,
            symbol=args.symbol,
            data_dir=Path(args.data_dir),
            window_start=start,
            window_end=split_at,
            scratch=scratch,
            reuse=not args.no_calibration_cache,
        )

    if args.workers > 1 and len(calibrations) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fits = list(pool.map(fit, calibrations))
    else:
        fits = [fit(calibration) for calibration in calibrations]

    reused = sum(1 for fitted in fits if fitted.get("cached"))
    if reused:
        print(f"  {reused}/{len(fits)} calibrations reused from {scratch}", flush=True)
    for calibration, fitted in zip(calibrations, fits):
        row = {
            "calibration": dataclasses.asdict(calibration),
            "key": calibration.key(),
        }
        if not fitted.get("ok"):
            # Keep the subprocess's stderr on the row. Without it a failing
            # calibration reports only "estimator_failed:1", which reads as a broken
            # estimator when the actual cause was a malformed argument this driver
            # constructed -- an entire 81-point sweep was lost to exactly that.
            row.update({
                "usable": False,
                "reason": fitted.get("reason"),
                "stderr": fitted.get("stderr"),
            })
            stage_a.append(row)
            detail = (fitted.get("stderr") or "").strip().splitlines()
            print(
                f"  {calibration.key():<34} SKIPPED [{fitted.get('reason')}]"
                + (f" {detail[-1][:160]}" if detail else ""),
                flush=True,
            )
            continue
        row.update({"params": fitted["params"], "diagnostics": fitted["diagnostics"]})
        stage_a.append(row)
        jobs.append({
            "index": len(stage_a) - 1,
            "tape": "search",
            "config": _config_kwargs(build_config(base, reference_risk, search)),
            "params": fitted["params"],
            "min_fills": args.min_fills,
        })

    for result in run_jobs(jobs, workers=args.workers, progress="stage A"):
        index = result.pop("index")
        stage_a[index].update(result)
    for row in stage_a:
        if "pnl_usdc" in row:
            print(
                f"  {row['key']:<34} pnl={row['pnl_usdc']:+8.2f} "
                f"fills={row.get('maker_fills', 0):>5} "
                f"({row.get('fills_bid', 0)}b/{row.get('fills_ask', 0)}a) "
                f"spread={row.get('net_realized_spread_usdc', 0.0):+8.2f}"
                + ("" if row.get("usable") else f"  [{row.get('reason')}]"),
                flush=True,
            )

    survivors = [row for row in sorted(stage_a, key=rank_key, reverse=True) if row.get("usable")]
    survivors = survivors[: args.keep]
    if not survivors:
        return {
            "status": "no_usable_calibration",
            "tape": identity,
            "split_at": split_at.isoformat(),
            "search_scenario": dataclasses.asdict(search),
        # What the grid was actually SELECTED on. Recorded because it was not:
        # published artifacts said "tape: 161.951 h" while selection had run on
        # ~3.2 h of it, and a reader had no way to tell.
        "search_tape": {
            "hours": search_identity["hours"],
            "price_rows": int(len(search_tape.prices)),
            "max_price_events": int(args.search_max_price_events),
            "is_partial": bool(search_is_partial),
            "train_hours": train_identity["hours"],
        },
            "stage_a": stage_a,
        }

    # Stage B -- the book's risk preference, on the calibrations that survived.
    risks = risk_grid(args.phi_kappa_t_grid, args.alpha_kappa_grid)
    print(
        f"[stage B] {len(risks)} risk settings x {len(survivors)} calibrations "
        f"= {len(risks) * len(survivors)} runs",
        flush=True,
    )
    stage_b: list[dict[str, Any]] = []
    jobs = []
    for survivor in survivors:
        for risk in risks:
            stage_b.append({
                "calibration_key": survivor["key"],
                "calibration": survivor["calibration"],
                "risk": dataclasses.asdict(risk),
                "key": f"{survivor['key']}|{risk.key()}",
                "params": survivor["params"],
            })
            jobs.append({
                "index": len(stage_b) - 1,
                "tape": "search",
                "config": _config_kwargs(build_config(base, risk, search)),
                "params": survivor["params"],
                "min_fills": args.min_fills,
            })
    for result in run_jobs(jobs, workers=args.workers, progress="stage B"):
        index = result.pop("index")
        stage_b[index].update(result)

    finalists = [row for row in sorted(stage_b, key=rank_key, reverse=True) if row.get("usable")]
    finalists = finalists[: args.keep]

    # Stage C -- held-out, per day, and the latency ladder for the winner.
    print(f"[stage C] held-out evaluation of {len(finalists)} finalists", flush=True)
    stage_c: list[dict[str, Any]] = []
    for row in finalists:
        risk = RiskSetting(**row["risk"])
        config = build_config(base, risk, search)
        out_of_sample = score(config, row["params"], held_out, min_fills=args.min_fills)
        # Sub-windows across the WHOLE tape, train included. The point is not the
        # train/held-out split -- it is whether the candidate survives a change of
        # regime, and the train stretch is regime evidence like any other.
        windows = [
            {"window": label, **score(config, row["params"], piece, min_fills=1)}
            for label, piece in window_slices(
                tape, start, end + timedelta(seconds=1), hours=args.window_hours
            )
        ]
        scored = [entry for entry in windows if entry.get("maker_fills", 0) > 0]
        positive = sum(1 for entry in scored if entry.get("pnl_usdc", 0.0) > 0)
        stage_c.append({
            "key": row["key"],
            "calibration": row["calibration"],
            "risk": row["risk"],
            "params": row["params"],
            "train": {key: row[key] for key in ("pnl_usdc", "maker_fills", "net_realized_spread_usdc")},
            "held_out": out_of_sample,
            "windows": windows,
            "window_hours": float(args.window_hours),
            "windows_positive": positive,
            "windows_scored": len(scored),
            # The worst sub-window is the number that says whether a total is being
            # carried by one stretch. A candidate whose aggregate is positive and
            # whose worst window is catastrophic has not shown anything repeatable.
            "worst_window_pnl": min((entry.get("pnl_usdc", 0.0) for entry in scored), default=None),
            "best_window_pnl": max((entry.get("pnl_usdc", 0.0) for entry in scored), default=None),
        })
        print(
            f"  {row['key']:<60} held-out pnl={out_of_sample.get('pnl_usdc', 0.0):+8.2f} "
            f"fills={out_of_sample.get('maker_fills', 0)} "
            f"windows+={positive}/{len(scored)} "
            f"worst={stage_c[-1]['worst_window_pnl']}",
            flush=True,
        )

    ladder: list[dict[str, Any]] = []
    if stage_c:
        winner = stage_c[0]
        risk = RiskSetting(**winner["risk"])
        print("[stage C] latency ladder for the winner", flush=True)
        for scenario in SCENARIOS:
            result = score(
                build_config(base, risk, scenario),
                winner["params"],
                held_out,
                min_fills=1,
            )
            ladder.append({"scenario": dataclasses.asdict(scenario), **result})
            print(
                f"  {scenario.name:<12} lat={scenario.latency_ms:>4}ms refresh={scenario.refresh_ms:>6}ms "
                f"pnl={result.get('pnl_usdc', 0.0):+8.2f} fills={result.get('maker_fills', 0):>4}",
                flush=True,
            )

    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "symbol": args.symbol,
        "tape": identity,
        "split_at": split_at.isoformat(),
        "train_fraction": float(args.train_fraction),
        "search_scenario": dataclasses.asdict(search),
        "min_fills": int(args.min_fills),
        "calibration_mode": args.calibration_mode,
        "grids": {
            "epsilon_horizons_ms": list(EPSILON_HORIZONS_MS),
            "kappa_support_lower": list(KAPPA_SUPPORT_LOWER),
            "phi_kappa_t": list(PHI_KAPPA_T),
            "alpha_kappa": list(ALPHA_KAPPA),
            "horizons_seconds": list(HORIZONS_SECONDS),
            "q_max": list(Q_MAX),
        },
        "stage_a": stage_a,
        "stage_b_top": sorted(stage_b, key=rank_key, reverse=True)[:40],
        "stage_c": stage_c,
        "latency_ladder": ladder,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _row(values: Iterable[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def to_markdown(payload: dict[str, Any]) -> str:
    tape = payload.get("tape", {})
    search_tape = payload.get("search_tape", {})
    lines = [
        f"# Parameter sweep — {payload.get('symbol', '?')}",
        "",
        f"- generated: `{payload.get('generated_at')}`",
        f"- tape: **{tape.get('hours')} h**, {tape.get('price_rows')} price rows, "
        f"{tape.get('trade_rows')} trades (`{tape.get('start')}` → `{tape.get('end')}`)",
        (
            f"- **selection ran on {search_tape.get('hours')} h of the "
            f"{search_tape.get('train_hours')} h train slice "
            f"({search_tape.get('price_rows')} price rows, "
            f"`--search-max-price-events {search_tape.get('max_price_events')}`). "
            "Stage A/B rankings below are a SHORTLIST from a partial tape, not a "
            "ranking of the full one.**"
            if search_tape.get("is_partial")
            else f"- selection ran on the full {search_tape.get('hours')} h train slice"
        ),
        f"- train/held-out split at `{payload.get('split_at')}` "
        f"({payload.get('train_fraction')} train)",
        f"- searched at scenario `{(payload.get('search_scenario') or {}).get('name')}` "
        f"(latency {(payload.get('search_scenario') or {}).get('latency_ms')} ms, "
        f"refresh {(payload.get('search_scenario') or {}).get('refresh_ms')} ms)",
        f"- a row is ranked only if it took ≥ {payload.get('min_fills')} maker fills",
        "",
        "Calibration is fitted on the train slice only and applied unchanged to the",
        "held-out slice. Epsilon horizons stop at 1 s because epsilon is the arrival",
        "jump (eq. 10.22), not a markout.",
        "",
    ]

    if payload.get("status") != "ok":
        lines += ["", f"**status: {payload.get('status')}**", ""]

    stage_a = [row for row in payload.get("stage_a", []) if row.get("usable")]
    if stage_a:
        lines += [
            "## Stage A — calibration (risk knobs at shipped values)",
            "",
            _row(["calibration", "P&L", "net spread", "fills (bid/ask)", "depth bid/ask bps", "P&L bid/ask"]),
            _row(["---", "---:", "---:", "---:", "---:", "---:"]),
        ]
        for row in sorted(stage_a, key=rank_key, reverse=True)[:15]:
            depth_bid = row.get("depth_bid_bps")
            depth_ask = row.get("depth_ask_bps")
            lines.append(_row([
                f"`{row.get('key')}`",
                f"{row.get('pnl_usdc', 0.0):+.2f}",
                f"{row.get('net_realized_spread_usdc', 0.0):+.2f}",
                f"{row.get('maker_fills', 0)} ({row.get('fills_bid', 0)}/{row.get('fills_ask', 0)})",
                f"{depth_bid:.1f} / {depth_ask:.1f}" if depth_bid and depth_ask else "—",
                f"{row.get('pnl_bid', 0.0):+.2f} / {row.get('pnl_ask', 0.0):+.2f}",
            ]))
        lines.append("")

    stage_b = payload.get("stage_b_top", [])
    if stage_b:
        lines += [
            "## Stage B — the book's risk preference (top 15 on train)",
            "",
            _row(["calibration", "risk", "P&L", "net spread", "directional", "fills (bid/ask)"]),
            _row(["---", "---", "---:", "---:", "---:", "---:"]),
        ]
        for row in stage_b[:15]:
            risk = row.get("risk", {})
            lines.append(_row([
                f"`{row.get('calibration_key')}`",
                f"φκT={risk.get('phi_kappa_t')} ακ={risk.get('alpha_kappa')} "
                f"T={risk.get('horizon_seconds')} q={risk.get('q_max')}",
                f"{row.get('pnl_usdc', 0.0):+.2f}",
                f"{row.get('net_realized_spread_usdc', 0.0):+.2f}",
                f"{row.get('directional_usdc', 0.0):+.2f}",
                f"{row.get('maker_fills', 0)} ({row.get('fills_bid', 0)}/{row.get('fills_ask', 0)})",
            ]))
        lines.append("")

    stage_c = payload.get("stage_c", [])
    if stage_c:
        lines += [
            "## Stage C — held-out",
            "",
            _row(["configuration", "train P&L", "held-out P&L", "net spread", "directional", "fills", "windows +", "worst"]),
            _row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
        for row in stage_c:
            held = row.get("held_out", {})
            worst = row.get("worst_window_pnl")
            lines.append(_row([
                f"`{row.get('key')}`",
                f"{(row.get('train') or {}).get('pnl_usdc', 0.0):+.2f}",
                f"**{held.get('pnl_usdc', 0.0):+.2f}**",
                f"{held.get('net_realized_spread_usdc', 0.0):+.2f}",
                f"{held.get('directional_usdc', 0.0):+.2f}",
                held.get("maker_fills", 0),
                f"{row.get('windows_positive')}/{row.get('windows_scored')}",
                f"{worst:+.2f}" if worst is not None else "—",
            ]))
        lines.append("")

        # The per-window detail for the leader, because "which regime paid" is the
        # question the aggregate cannot answer.
        leader = stage_c[0]
        if leader.get("windows"):
            lines += [
                f"### Leader per {leader.get('window_hours')} h window — `{leader.get('key')}`",
                "",
                _row(["window", "P&L", "net spread", "directional", "fills (bid/ask)"]),
                _row(["---", "---:", "---:", "---:", "---:"]),
            ]
            for entry in leader["windows"]:
                lines.append(_row([
                    entry.get("window"),
                    f"{entry.get('pnl_usdc', 0.0):+.2f}",
                    f"{entry.get('net_realized_spread_usdc', 0.0):+.2f}",
                    f"{entry.get('directional_usdc', 0.0):+.2f}",
                    f"{entry.get('maker_fills', 0)} ({entry.get('fills_bid', 0)}/{entry.get('fills_ask', 0)})",
                ]))
            lines.append("")

    ladder = payload.get("latency_ladder", [])
    if ladder:
        lines += [
            "## Latency ladder — winner, held-out slice",
            "",
            "Latency and requote cadence move together: resting exposure is",
            "`max(0, refresh − latency) + cancel`, so cutting latency at a fixed refresh",
            "lengthens the time a quote sits. Each row is a plausible machine.",
            "",
            _row(["scenario", "latency", "refresh", "P&L", "net spread", "directional", "fills"]),
            _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
        for row in ladder:
            scenario = row.get("scenario", {})
            lines.append(_row([
                scenario.get("name"),
                f"{scenario.get('latency_ms')} ms",
                f"{scenario.get('refresh_ms')} ms",
                f"{row.get('pnl_usdc', 0.0):+.2f}",
                f"{row.get('net_realized_spread_usdc', 0.0):+.2f}",
                f"{row.get('directional_usdc', 0.0):+.2f}",
                row.get("maker_fills", 0),
            ]))
        lines.append("")

    lines += [
        "## How to read the columns",
        "",
        "**net spread** is what market making earns: realized spread minus fees.",
        "**directional** is P&L minus that — a bet on where the mid went, which the",
        "tape decides and the model does not. A row whose P&L is mostly directional has",
        "not demonstrated market making, however good the total looks.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="CASHCAT")
    parser.add_argument("--data-dir", type=Path, default=SCRIPTS / "HL_data")
    parser.add_argument("--mid", type=float, default=0.1015)
    parser.add_argument("--price-tick-size", type=float, default=1e-6)
    parser.add_argument("--amount-step-size", type=float, default=0.0)
    parser.add_argument("--inventory-unit-base", type=float, default=2092.0)
    parser.add_argument("--maker-fee", type=float, default=0.00015)
    parser.add_argument("--taker-fee", type=float, default=0.00045)
    parser.add_argument("--starting-equity-usdc", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--newest-per-stream", type=int, default=None)
    parser.add_argument("--max-price-events", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--calibration-mode",
        choices=("matched", "full"),
        default="matched",
        help="matched moves both sides together (9 points); full crosses them (81).",
    )
    parser.add_argument("--search-scenario", default=SEARCH_SCENARIO,
                        choices=[scenario.name for scenario in SCENARIOS])
    parser.add_argument("--keep", type=int, default=3, help="survivors carried between stages")
    parser.add_argument(
        "--phi-kappa-t-grid",
        type=lambda text: parse_float_list(text, PHI_KAPPA_T),
        default=None,
        help="Comma-separated override for the phi*kappa*T axis (default: the shipped grid).",
    )
    parser.add_argument(
        "--alpha-kappa-grid",
        type=lambda text: parse_float_list(text, ALPHA_KAPPA),
        default=None,
        help=(
            "Comma-separated override for the alpha*kappa axis. Swept 2026-08-18 and "
            "found INERT at high phi -- all values returned bit-identical metrics -- so "
            "a single value is usually the right choice."
        ),
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=6.0,
        help=(
            "Length of the sub-windows each finalist is scored over, across the whole "
            "tape. The per-side edge on CASHCAT reversed between an 11 h and a 23 h "
            "window, so a candidate's aggregate says little without this breakdown."
        ),
    )
    parser.add_argument(
        "--min-fills",
        type=int,
        default=30,
        help="a configuration below this took too few fills to have measured anything",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Parallel replay processes, capped at 4. This host runs the Rust dry-run "
            "grid and several collectors; a sweep that starves them of CPU would corrupt "
            "the very data it is measuring."
        ),
    )
    parser.add_argument(
        "--search-max-price-events",
        type=int,
        default=0,
        help=(
            "Price rows the SEARCH stages score, as a contiguous prefix of the train "
            "slice. 0 (the default) scores the whole train slice. This used to default "
            "to 25000 (~3.2 h of a ~113 h split) on a cost estimate that was stale by "
            "~40x; the full search is ~30 min at four workers. Set it non-zero only for "
            "quick exploratory runs -- the value is recorded in the artifact so a "
            "partial search is never mistaken for a full one."
        ),
    )
    parser.add_argument(
        "--no-calibration-cache",
        action="store_true",
        help=(
            "Always re-run the estimator, even when the scratch directory holds a fit "
            "made from identical arguments and identical shards. The cache keys on a "
            "hash of every input shard's name and size, so it cannot survive the "
            "collector writing; this flag exists for when you want the belt as well."
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "cashcat_sweep.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "docs" / "cashcat_sweep.md")
    parser.add_argument("--scratch", type=Path,
                        default=Path.home() / ".cache" / "mm_sweep")
    return parser.parse_args()


MAX_WORKERS = 4


def main() -> int:
    args = parse_args()
    if args.workers > MAX_WORKERS:
        print(f"[workers] capping {args.workers} -> {MAX_WORKERS}", flush=True)
    args.workers = max(1, min(int(args.workers), MAX_WORKERS))
    payload = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    args.markdown_output.write_text(to_markdown(payload), encoding="utf-8")
    print(f"[done] {args.output}  {args.markdown_output}", flush=True)
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
