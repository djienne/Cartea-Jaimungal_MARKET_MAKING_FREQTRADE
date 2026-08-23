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
collected are `orderbooks/`, `prices/`, `trades/`.

## Independent of everything else

- Its own compose project (`hyperliquid_data`) and its own network
  (`hyperliquid_data_default`).
- **No `depends_on`**, no Redis, no link to the trading stack, which is a
  separate project (`cartea-jaimungal_market_making_freqtrade`, whose services
  do depend on `redis`). `docker compose down` on the trading stack cannot
  touch the collector.
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

## Known gap

There is **no Docker `HEALTHCHECK`** on either collector (`mm-redis` and
`polymarket-1d-paper` have one; these do not). The `os._exit(1)` path covers a
dead or quiet WebSocket, but not a process that is wedged while still holding
the socket, nor writes that fail silently — in those states the container stays
"Up" and nothing restarts it.

A freshness-based healthcheck plus an autoheal sidecar would close it. It is not
applied here because adding a healthcheck forces container recreation, which
costs a gap in the tape; do it at a deliberate moment, not mid-session.
