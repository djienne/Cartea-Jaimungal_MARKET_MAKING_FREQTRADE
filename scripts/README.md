# Hyperliquid Market Making Suite

A comprehensive Python suite for collecting real-time tick data from Hyperliquid and estimating market making parameters. Includes data collection via WebSocket API, advanced parameter estimation, and a Dockerized setup for easy deployment.

---

## Features

### Data Collection (`hyperliquid_data_collector.py`)
- **Real-time data collection** via WebSocket connections
- **Multiple data types**:
  - Best Bid/Offer (BBO) prices with timestamps
  - Trade executions with side, price, and volume
  - Order book snapshots (configurable depth, default 20 levels)
- **Parquet output** (compressed) organized by symbol and type
- **Live statistics** showing collection rates and summaries
- **Asynchronous data writing** to minimize performance impact
- **Graceful shutdown** with data preservation

### Parameter Estimation (aligned with Cartea-Jaimungal model, snapshot schema v4)
- **Market-order aggregation**: trade prints sharing side + exchange timestamp are ONE market order; depths and impacts are measured per MO, not per print
- **Mid-relative coordinates**: depths are measured from the prevailing mid (last BBO update strictly before the MO, exchange-timestamp aligned via merge_asof) — the same coordinate the strategy quotes in; negative depths are truncated to 0, not dropped
- **κ± (Kappa)**: survival-function fit — weighted log-linear regression of P(depth ≥ δ); saved to `kappa.json` with `depth_p95`/`depth_max_fitted` calibration diagnostics
- **λ± (Lambda)**: raw per-side MO arrival rate — the survival-consistent fill rate the HJB needs (`lambda_source: mo_survival_fit`); the old binned-density intercept was bin-width dependent and ~3× too small (kept as `lambda0_intercept_±` diagnostic). Since 2026-08-19 the denominator is *observed* seconds, not wall clock: time when the collector was down (no print in the union of the price and trade streams) is subtracted, because dividing by wall clock understated λ by exactly the missing fraction. Both halves are published as `lambda_covered_seconds` / `lambda_outage_seconds_excluded` so the denominator is auditable. Measured effect on real windows: +1.2% to +6.0%; inside the 1 h price gap the replay now tolerates it would have been 50%
- **ε± (Epsilon)**: per-MO mid impact at a 5 s primary horizon (permanent impact), with 200 ms and 1 s trimmed means recorded as diagnostics (`epsilon_200ms_±`, `epsilon_1s_±`); floor at 0 (C-J defines ε ≥ 0); saved to `epsilon.json`
- **σ² (`sigma2_per_sec`)**: realized mid variance (USDC²/s from 1 s increments, gap-tolerant), feeding the strategy's volatility-aware inventory penalty
- **Direct model parameters**: primary κ/ε/λ values are the validated estimates from the selected market-data window; no temporal smoothing is applied
- **λ_trades± (Lambda trades)**: unconditional trade-print rates from raw counts; saved to `lambda_trades.json` (monitoring only)
- **Status gating**: snapshots ship `status: ok` only when fit points ≥ 6, R² ≥ 0.30 and ε events ≥ 50 per side (mirrored by the strategy's validation floors)
- **Market toxicity assessment** based on ε×κ product

---

## Installation (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start (local)

### 1. Collect Data

```bash
# Start collecting data (writes Parquet shards into HL_data/<SYMBOL>/<dtype>/)
python hyperliquid_data_collector.py
```

### 2. Estimate Parameters

```bash
# Survival-fit κ + per-side MO arrival rate λ + σ², saves kappa.json & lambda.json
python get_kappa.py --crypto CASHCAT --minutes 30

# Event-level ε per MO at the 5s permanent-impact horizon, saves epsilon.json
python get_epsilon.py --crypto CASHCAT --minutes 30 [--post-horizon-ms 5000]

# Optional raw trades/sec sanity check (writes lambda_trades.json)
python get_lambda.py --crypto CASHCAT --minutes 30

# Inspect spreads across inventory (refreshes κ/ε/λ, then shows bid/ask and bps by q;
# pass --spread-multiplier 3.0 to mirror the production config)
# --mid defaults to the freshly collected BBO mid via mid_price.json when omitted
python compute_spreads.py --crypto CASHCAT --qmax 3 --spread-multiplier 3.0
```

---

## Hyperliquid Data Collector (Docker)

**There is no compose file in this folder any more.** Both collectors that build
from here are defined in `HYPERLIQUID_DATA/docker-compose.yml`, alongside the
other three Hyperliquid collectors, and are operated from there:

- **`hl-cashcat-collector`** — `SYMBOLS=CASHCAT`, `RETENTION_MINUTES=43200`
  (30 days). CASHCAT is the symbol the strategy quotes, and the sweeps, replays
  and the acceptance gate all read the traded symbol, so it needs a far longer
  tape than the rest.
- **`hl-collector`** — `SYMBOLS=ETH,ACE,CHIP,PENGU,NIL`, `RETENTION_MINUTES=4320`
  (3 days), as controls and candidates.

**The two `SYMBOLS` lists must never overlap.** Both write into the same
`./data/eth_mm` directory. On 2026-08-16 two collectors were run over the same
symbol into one directory: each wrote its own shards under its own names, nothing
collided or errored, and every trade simply landed on disk twice — the estimators
read the directory, not the writer, so `n_trades` and λ± silently doubled. If you
add a symbol to one list, remove it from the other in the same edit.

### Quick start

```bash
# from HYPERLIQUID_DATA/, not from here
docker compose up -d --build hl-collector hl-cashcat-collector
docker compose logs -f hl-cashcat-collector
python inventory.py            # what is being collected, and is it fresh
```

### Configuration

Environment variables, read by `run_collector.py` / `hyperliquid_data_collector.py`:

| Variable                  | Default   | Description                                                        |
| ------------------------- | --------- | ------------------------------------------------------------------ |
| `SYMBOLS`                 | `ETH`     | Comma-separated list of symbols to collect                         |
| `OUTPUT_DIR`              | `HL_data` | Directory where Parquet files are written                          |
| `ORDERBOOK_DEPTH`         | `20`      | Orderbook depth to record                                          |
| `FLUSH_INTERVAL_SEC`      | `10`      | Buffer flush cadence; must stay well under the strategy's 30 s freshness window |
| `COMPACT_AFTER_MINUTES`   | `15`      | Merge an hour's shards into one file once they are this old        |
| `RETENTION_MINUTES`       | `60`      | Prune shards older than this (the compose services override it)    |
| `INACTIVITY_TIMEOUT_SEC`  | `180`     | Reconnect after this long with no data at all                      |
| `WS_HEALTH_GRACE_SEC`     | `20`      | Ignore "socket is down" readings for this long after any connect   |
| `TZ`                      | `UTC`     | Timezone inside the container                                      |

### Websocket expiry (fixed 2026-08-19)

Hyperliquid expires a websocket session about every 3 hours and sends a close
frame; the SDK logs it, its manager thread exits, and nothing in the SDK
reconnects. Recovery used to come only from the time-based inactivity watchdog,
so every routine expiry cost a full `INACTIVITY_TIMEOUT_SEC` of missing data —
measured over 60.3 h of CASHCAT as 20 gaps of 3.1-3.5 min on a clockwork ~3 h
cadence, **71% of all missing data and 2.5% of the span**. The watchdog now also
reads the SDK's own socket state and acts within ~10 s, guarded against a
reconnect loop by requiring two consecutive down readings and by
`WS_HEALTH_GRACE_SEC` after any connect (the SDK starts its thread before the
handshake completes). The inactivity path is unchanged and still covers a socket
that looks alive but delivers nothing.

### Data persistence

Collected Parquet files are stored on the host in `HYPERLIQUID_DATA/data/eth_mm`.
The market-making project reaches them through the `scripts/HL_data` junction, so
the parameter estimation scripts (`get_kappa.py`, `get_epsilon.py`,
`get_lambda.py`) still find their data where they always did.

### Logs

* Run `docker compose logs -f <service>` from `HYPERLIQUID_DATA/` to watch live output.
* Press **CTRL-C** to stop watching (collector continues running in the background).
* To detach from a non-detached `docker compose up`, press **CTRL-P + CTRL-Q**.

### Updating

If you change code in this folder, rebuild from `HYPERLIQUID_DATA/` — both
collector services use this folder as their build context, so **both must be
rebuilt together** or they run different code against the same output directory:

```bash
docker compose up -d --build hl-collector hl-cashcat-collector
```

---

## Output Files

For each symbol, the collector writes **Parquet shards** into per-type subdirectories. Flush cadence is controlled by `FLUSH_INTERVAL_SEC` (default 10s — it must stay well below the strategy's `max_collector_age_seconds=30` freshness window or quotes get rejected as `stale_collector_data`).

> **Retention warning:** `RETENTION_MINUTES` (code default 60; the compose services set 4320 and 43200) prunes shards older than the window. Sixty minutes is plenty for the live estimators, but it silently deletes the history the replay/calibration tooling (`replay_market_maker.py`, `sweep_replay.py`, `calibrate_replay_from_logs.py`) consumes. When collecting a dataset for replays, **raise** `RETENTION_MINUTES` to cover the full capture — do not set it to 0. Reads select shards by the flush timestamp in the filename (2026-08-17), so read cost tracks the window you ask for rather than everything on disk; before that change a long retention would have stalled the estimator outright.

* `HL_data/<SYMBOL>/prices/prices_<epoch_ms>.parquet` (BBO updates)
* `HL_data/<SYMBOL>/trades/trades_<epoch_ms>.parquet` (trade executions)
* `HL_data/<SYMBOL>/orderbooks/orderbooks_<epoch_ms>.parquet` (order book snapshots)

### Example (symbols: BTC, ETH, SOL)

```
HL_data/
  BTC/
    prices/
      prices_1765401094883.parquet
    trades/
      trades_1765401094883.parquet
    orderbooks/
      orderbooks_1765401094883.parquet
  ETH/
    prices/
      prices_1765401094883.parquet
    trades/
      trades_1765401094883.parquet
    orderbooks/
      orderbooks_1765401094883.parquet
```

### File Format

* **Prices**:

  * `timestamp`: Local receive time
  * `exchange_timestamp`: Exchange-provided time
  * `price`, `size`, `side`

* **Trades**:

  * `timestamp`, `exchange_timestamp`
  * `price`, `size`, `side` (`buy`/`sell`)
  * `trade_id`

* **Orderbooks**:

  * `timestamp`, `exchange_timestamp`, `sequence`
  * `bid_price_0` … `bid_price_N`, `bid_size_0` … `bid_size_N`
  * `ask_price_0` … `ask_price_N`, `ask_size_0` … `ask_size_N`

### Benefits

* **Columnar + compressed**: smaller files and faster reads for analytics.
* **Append-friendly**: sharded files avoid constantly rewriting a single giant file.
* **Simple partitioning**: data is separated by symbol and type.

---

## Parameter Estimation

### Parameters Estimated

| Parameter         | Symbol | Description                                  | Estimation Method                                      |
| ----------------- | ------ | -------------------------------------------- | ------------------------------------------------------ |
| **Lambda Plus**   | λ+     | Buy MO arrival rate (MOs/sec)                | Raw per-side MO count / covered window seconds         |
| **Lambda Minus**  | λ-     | Sell MO arrival rate (MOs/sec)               | Raw per-side MO count / covered window seconds         |
| **Epsilon Plus**  | ε+     | Permanent mid jump from buy MOs (USDC)       | Per-MO mid change at 5 s horizon (trimmed mean, ≥0)    |
| **Epsilon Minus** | ε-     | Permanent mid jump from sell MOs (USDC)      | Per-MO mid change at 5 s horizon (trimmed mean, ≥0)    |
| **Kappa Plus**    | κ+     | Ask side depth sensitivity (1/USDC)          | Survival fit: log P(depth ≥ δ) vs δ, mid-relative      |
| **Kappa Minus**   | κ-     | Bid side depth sensitivity (1/USDC)          | Survival fit: log P(depth ≥ δ) vs δ, mid-relative      |
| **Sigma²**        | σ²     | Realized mid variance (USDC²/s)              | Variance of 1 s mid increments (gap-tolerant)          |

### Market Assessment

The tool automatically assesses market making viability using the **ε×κ product**:

* **ε×κ < 1.0**: ✅ Favorable
* **1.0 ≤ ε×κ < 1.5**: 🟡 Moderate
* **ε×κ ≥ 1.5**: ❌ Toxic

---

## Real-time Statistics

During collection, statistics are printed every 30s, including rates and buffer sizes:

```
============================================================
DATA COLLECTION SUMMARY - 14:23:45
============================================================
Runtime: 0h 5m 23s
Data collected:
  bbo_updates: 1,234 (234.5/min)
  trades: 567 (107.2/min)
  orderbook_updates: 891 (168.9/min)

Buffer sizes by symbol:
  BTC: 45 (32 prices, 8 trades, 5 orderbooks)
  ETH: 23 (18 prices, 3 trades, 2 orderbooks)
  SOL: 12 (8 prices, 2 trades, 2 orderbooks)
============================================================
```

---

## Performance Features

* **Buffered writing**: Data is collected in memory and flushed to disk periodically (default: every 10 seconds, `FLUSH_INTERVAL_SEC`).
* **Threaded I/O**: Parquet writing happens in background threads to avoid blocking.
* **Configurable buffer sizes**: Prevent memory issues during bursts
* **Efficient data structures**: Uses deques for O(1) appends

---

## Stopping the Collector

* **Local**: Press `Ctrl+C` → graceful shutdown (flushes data and closes connections)
* **Docker**: `docker compose stop hl-cashcat-collector` from `HYPERLIQUID_DATA/`.
  A bare `docker compose down` there takes all five Hyperliquid collectors with it,
  not just this one.

---

## Dependencies

* `hyperliquid-python-sdk`: Official Hyperliquid SDK
* `websockets`: WebSocket client library
* `pandas`, `numpy`, `pyarrow` (Parquet read/write)
* `docker` / `docker compose` (for containerized mode)

---

## Troubleshooting

### WebSocket Connection Issues

* Check internet connection
* Verify Hyperliquid API is accessible
* Reduce number of symbols

### High Memory Usage

* Reduce buffer sizes
* Decrease flush interval
* Monitor number of active symbols

### Missing Data

* Check console logs
* Verify symbol names
* Ensure disk space available
