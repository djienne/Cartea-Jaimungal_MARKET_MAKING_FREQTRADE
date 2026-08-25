---
name: dry-run-operation
description: How to run the Rust market-making dry run and grid, and the runtime gotchas that bite
metadata:
  type: project
---

**Rewritten 2026-08-25.** This file described the freqtrade stack — two legs on
separate sub-accounts, a param-estimator publishing to Redis, and a docker
compose project. That trader is retired (tag `freqtrade-trader-final`); its
containers are stopped and removed. What follows is the stack that actually
runs. The freqtrade-era gotchas that died with it — the `param_update.lock`
abandoned-lock recovery, the ccxt 4.5.22 pin and its fee invariant, the
gate-battery profiles, `docker compose up -d mm-long mm-short` — are gone with
the code; look them up at the tag if a question about the old bot ever comes up.

## What runs

The trader is `rust_live/`, a plain process, not a compose service. Three modes
matter:

```
mm-live --config config/cashcat_dryrun_realistic.toml dry-run
mm-live --config config/cashcat_dryrun_realistic.toml dry-run-grid \
        --grid config/grid_cashcat.toml --duration-seconds 0 --out-dir reports/grid_live
mm-live --config config/cashcat.toml live          # real money, explicit, gated
```

The grid opens **one** WebSocket no matter how many variants it runs. That is
deliberate: Hyperliquid allows ten per IP and the budget is shared with the
collectors and any live session. The grid never writes Parquet and never reads
credentials.

## Collectors — separate, and must stay that way

Not operated from this repo. `hl-cashcat-collector` (CASHCAT alone, 30-day
retention) and `hl-collector` (ETH, ACE, CHIP, PENGU, NIL, 3 days) live in
`HYPERLIQUID_DATA/docker-compose.yml` and write to the shared `./data/eth_mm`
tree, reachable here as the `scripts/HL_data` junction.

**Their `SYMBOLS` lists must stay disjoint** or every trade lands on disk twice:
`estimator_common._load_parquet_dir()` concatenates every `*.parquet` and
`normalize_trades()` does not de-duplicate, so `n_trades` and λ± inflate ~2x
(measured 2104 rows / 1055 unique trade_ids on 2026-08-16).

**`scripts/` is their Docker build context.** `HYPERLIQUID_DATA/docker-compose.yml`
builds both with `context: ../Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/scripts`,
`dockerfile: Dockerfile`, and the image copies exactly `hyperliquid_data_collector.py`
and `run_collector.py`. Do not move or rename those three paths. The failure is
silent — running containers keep going and only break at the next rebuild — so
after touching `scripts/`, run
`docker compose -f ../HYPERLIQUID_DATA/docker-compose.yml build hl-cashcat-collector`.

Collector websocket gotcha (fixed 2026-08-19, 945648e): Hyperliquid expires a
websocket session about every 3 hours and sends a close frame; the SDK logs it,
its manager thread exits, and nothing in the SDK reconnects. Recovery used to be
the time-based `_watchdog_inactivity`, so every routine expiry cost a full
`INACTIVITY_TIMEOUT_SEC` (180 s) of missing data — over 60.3 h of CASHCAT that
was 20 gaps of 3.1–3.5 min on a clockwork ~3 h cadence, 71% of all missing data.
`_websocket_is_down()` now reads the SDK's own state and the watchdog acts within
~10 s.

## Gotchas that bite

**Rebuilding while the grid runs fails.** `cargo build --release --target-dir
target/measure` returns `Access is denied. (os error 5)` because the running
`mm-live.exe` holds the file. Stop the grid, build, relaunch — and expect to
lose the run's elapsed clock, which matters if you are mid-measurement.

**Redirect the log into the run directory.** `... --out-dir reports/grid_live >
reports/grid_live/run.log` — anything else and the feed-health lines land where
the monitor does not read them.

**Feed health is logged on change, not on a timer.** A `grid feed health` line
appears when `reconnects`, `feed_gaps` or `replayed_trades_ignored` moves, plus
a floor of one line per ten minutes. So `trade_prints` in that line is stale by
design; it is a snapshot from the last time a health counter moved, not a live
count.

**Replayed trades are expected, not a fault.** The venue replays history on
every subscribe — measured 27 prints on connect against 13 live ones, then a
further ~30 on each reconnect. `replayed_trades_ignored` rising while
`feed_gaps` stays flat is the system working.

**Disk.** ~17 MB/h at 18 variants, so ~12 GB/month, with zstd holding 16×.
`quote_decision` events are **99.1%** of the bytes (157k events / 79.7 MB against
2.9k execution events / 0.7 MB), so sampling them is the only lever that matters
if that ever needs to come down.

**`equity_history.csv` is append-only across restarts** and stamps
`run_started_ms`, so a P&L curve survives a relaunch — split on that column
rather than assuming one run per file.

## Calibration

All-Rust now: `mm-live calibrate` solves κ/λ/ε and the HJB surface over Parquet
history. The Python estimators (`estimate_all.py`, `get_{kappa,lambda,epsilon}.py`)
are kept for replay and analysis, not for feeding a live bot — nothing consumes
their snapshots any more. See [[cartea-jaimungal-phi-kappa-trap]] before
touching φ, and [[hjb-alpha-untuned-after-episodic]] for α.
