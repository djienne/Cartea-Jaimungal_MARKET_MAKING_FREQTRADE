# Market data collection — who owns it

Continuous collection is fundamental: backtests, parameter sweeps, the
estimators and the replay acceptance gate all read this tape. This page records
who produces it, why nothing in this repo can disturb it, and how to check it.

## It runs in Docker, and it is not the bot

| | |
|---|---|
| container | `hl-cashcat-collector` (CASHCAT, 30-day retention) |
| sibling | `hl-collector` (ETH, ACE, CHIP, PENGU, NIL — 3-day retention) |
| compose project | `hyperliquid_data` — `HYPERLIQUID_DATA/docker-compose.yml` |
| entrypoint | `python run_collector.py` |
| build context | this repo's `scripts/` (the collector code is `hyperliquid_data_collector.py`) |
| restart policy | `unless-stopped` |
| caps | `cpus: 0.1`, `mem_limit: 1g`, json-file logs rotated at 10 MB × 3 |

The two collectors **must never share a symbol** — both write into the same
directory, and a symbol listed twice silently doubles `n_trades` and the
lambda estimates. CASHCAT lives only in `hl-cashcat-collector`.

## The data is on the host, not inside the container

The mount is a **bind mount**, so the tape is ordinary Windows files:

```
type=bind  rw=true
C:\Users\david\Desktop\freqtrade\HYPERLIQUID_DATA\data\eth_mm  ->  /app/HL_data
```

`rust_live/config/*.toml` sets `storage.data_dir = "../../scripts/HL_data"`,
and that path is an **NTFS junction** onto the same directory:

```
Cartea-Jaimungal_MARKET_MAKING_FREQTRADE\scripts\HL_data
  --> C:\Users\david\Desktop\freqtrade\HYPERLIQUID_DATA\data\eth_mm
```

Verified identical: the newest shards listed through the junction and through
the bind-mount source are the same files with the same timestamps. Streams
collected are `orderbooks/`, `prices/`, `trades/`, and — since 2026-08-23 —
`asset_ctx/` (oracle/mark/mid price, open interest, funding, premium from the
`activeAssetCtx` channel; multiplexed on the same WebSocket, so it costs no
extra connection). `asset_ctx` exists for the oracle-dislocation question
deferred in `FLOW_GUARD_CANDIDATES.md`; rows are kept on change plus a 60 s
heartbeat, carry only local receive `timestamp` (the venue does not stamp the
channel), and the stream is deliberately outside both the container
healthcheck and `validate_hl_data.py` while it is young — a failure in it must
not restart the proven collectors.

## Independent of everything else

- Its own compose project (`hyperliquid_data`) and its own network
  (`hyperliquid_data_default`).
- **No `depends_on`**, no Redis, no link to any trading stack. The compose
  project that once sat beside it (`cartea-jaimungal_market_making_freqtrade`,
  two freqtrade legs plus a param estimator and Redis) is retired; the trader
  is now `rust_live/`, which runs as a plain process and owns no compose
  services here. Nothing anyone does to the trader can touch the collector.
- Nothing in `rust_live` writes here — see the next section.

## A live bot cannot disturb collection, by code

`storage.write_parquet` is read in exactly one place: `run_public_dry_run` in
`rust_live/src/main.rs`. The `live` command never constructs a
`ParquetRecorderHandle` and never takes the writer lock, so **starting or
stopping a live session has no effect on collection at all**.

The only command that can contend is `dry-run`, guarded twice:

- `storage.writer_lock_path` — exclusive lock file; and
- a shard preflight that refuses to start when another writer has recently
  written shards: *"another collector wrote CASHCAT Parquet shards during
  writer preflight"*.

Both were observed firing correctly against the running container. To run a dry
run alongside the collector, decline the writer role:

```
mm-live --config rust_live/config/cashcat.toml dry-run --no-write-parquet
```

**Common misreading:** `write_parquet = true` also appears in the live configs.
It is inert there. Do not infer that a live session records market data, and do
not expect collection to stop when a live session stops.

## Self-healing

Stalls are handled inside the collector rather than by an external watchdog:

- `INACTIVITY_TIMEOUT_SEC=180` — no data for 3 minutes counts as a stall;
- `MAX_RECONNECT_ATTEMPTS=3` with `RECONNECT_BACKOFF_SEC=5`;
- on exhaustion the process calls `os._exit(1)`
  (`hyperliquid_data_collector.py`), and `restart: unless-stopped` starts it
  again.

That loop is not theoretical — the container reports `RestartCount=1`.

## Checking it

```powershell
docker ps --filter name=hl-cashcat-collector
docker logs --tail 20 hl-cashcat-collector
python HYPERLIQUID_DATA/inventory.py     # what is collected, and is it fresh
```

Healthy logs report a rolling BBO count and periodic
`Flushed buffers for 3 data types across symbols`.

## Watchdog

Two layers, because they fail differently.

**Inside the collector** — covers a dead or quiet WebSocket:

- `INACTIVITY_TIMEOUT_SEC=180` — no data for 3 minutes is a stall;
- `MAX_RECONNECT_ATTEMPTS=3` with `RECONNECT_BACKOFF_SEC=5`;
- on exhaustion the process calls `os._exit(1)`
  (`hyperliquid_data_collector.py`), and `restart: unless-stopped` restarts it.

That loop is not theoretical — the container had already logged
`RestartCount=1` before any of this was added.

**Docker `HEALTHCHECK` + autoheal** (added 2026-08-23) — covers what the above
cannot: a process wedged while still holding the socket, or writes failing
silently. In those states the container stays "Up" forever and nothing acts.

The check asserts *data is landing*, not that a process exists. It takes the
flush timestamp from the newest shard **filename** rather than calling `stat()`
per file, so its cost tracks directory size and not retention — the same reason
the readers were changed on 2026-08-17. Measured: 6,219 files in 29 ms, run
once a minute against a `cpus: 0.1` cap.

It is scoped to each container's own `SYMBOLS`. Both collectors share the mount
and can see each other's directories, so an unscoped check would let a wedged
CASHCAT collector look healthy off the other collector's writes.

The threshold is 600 s (`HEALTH_MAX_AGE_SEC`), deliberately well above the
app's own recovery window (180 s + 3 reconnects), so it only fires once that
path has already failed to.

Docker never acts on a failing healthcheck outside Swarm — it only marks the
container. The `autoheal` service (`willfarrell/autoheal`) does the restarting.
It is scoped **by label** (`autoheal=true`), not `AUTOHEAL_CONTAINER_LABEL=all`,
so it can only ever touch the two collectors and never the trading stack or any
other project sharing this daemon.

### Verified, not assumed

- The check passes on healthy collectors (`newest_age_s` 4.3 and 1.3).
- It fails when data is stale (`HEALTH_MAX_AGE_SEC=0` → exit 1) and when the
  symbol has no data (→ exit 1). A check that cannot fail is worthless.
- The autoheal loop was proven with a disposable canary container labelled
  `autoheal=true` and a permanently failing healthcheck, rather than by
  breaking a real collector.

### Restarts are graceful (2026-08-23)

The watchdog restarts a collector; that restart must not itself cost data. It
used to. The entrypoint runs as **PID 1**, which gets no default signal
handlers, and the collector only caught `KeyboardInterrupt` (SIGINT). So
`docker stop`, `docker restart`, `docker compose down` and autoheal all waited
out the full stop timeout and then SIGKILLed — skipping `stop_collection()` and
therefore the final `_flush_buffers()`, losing everything still in memory.

`run_collector.py` now installs a SIGTERM handler that raises
`KeyboardInterrupt`, reusing the already-tested shutdown path rather than
adding a second one. Measured before and after on `hl-cashcat-collector`:

| | before | after |
|---|---|---|
| restart duration | ~30 s (timeout, then SIGKILL) | **3 s** |
| final flush | skipped | `Shutting down...` → `Flushed buffers` |

`AUTOHEAL_DEFAULT_STOP_TIMEOUT` was lowered 30 s → 15 s to match: it is now
headroom for a graceful flush rather than a wait for the inevitable kill.

Attaching the healthchecks cost a **5–6 second** gap in the tape, taken one
collector at a time.
