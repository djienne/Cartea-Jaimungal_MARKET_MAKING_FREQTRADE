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

### Parameter Estimation (aligned with Cartea-Jaimungal model, snapshot schema v3)
- **Market-order aggregation**: trade prints sharing side + exchange timestamp are ONE market order; depths and impacts are measured per MO, not per print
- **Mid-relative coordinates**: depths are measured from the prevailing mid (last BBO update strictly before the MO, exchange-timestamp aligned via merge_asof) — the same coordinate the strategy quotes in; negative depths are truncated to 0, not dropped
- **κ± (Kappa)**: survival-function fit — weighted log-linear regression of P(depth ≥ δ); saved to `kappa.json` with `depth_p95`/`depth_max_fitted` calibration diagnostics
- **λ± (Lambda)**: raw per-side MO arrival rate (count / covered window seconds) — the survival-consistent fill rate the HJB needs (`lambda_source: mo_survival_fit`); the old binned-density intercept was bin-width dependent and ~3× too small (kept as `lambda0_intercept_±` diagnostic)
- **ε± (Epsilon)**: per-MO mid impact at a 5 s primary horizon (permanent impact), with 200 ms and 1 s trimmed means recorded as diagnostics (`epsilon_200ms_±`, `epsilon_1s_±`); floor at 0 (C-J defines ε ≥ 0); saved to `epsilon.json`
- **σ² (`sigma2_per_sec`)**: realized mid variance (USDC²/s from 1 s increments, gap-tolerant), feeding the strategy's volatility-aware inventory penalty
- **EMA smoothing**: primary κ/ε/λ values are time-aware EMA-smoothed across cycles (`--ema-tau` / `PARAM_EMA_TAU_SECONDS`, default 300 s; 0 disables); per-window raw values live in the `*_raw` keys; the EMA never blends across schema versions or non-ok snapshots
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
python get_epsilon.py --crypto CASHCAT --minutes 30 [--post-horizon-ms 5000] [--ema-tau 300]

# Optional raw trades/sec sanity check (writes lambda_trades.json)
python get_lambda.py --crypto CASHCAT --minutes 30

# Inspect spreads across inventory (refreshes κ/ε/λ, then shows bid/ask and bps by q;
# pass --spread-multiplier 3.0 to mirror the production config)
# --mid defaults to the freshly collected BBO mid via mid_price.json when omitted
python compute_spreads.py --crypto CASHCAT --qmax 3 --spread-multiplier 3.0
```

---

## Hyperliquid Data Collector (Docker)

This project also includes a **Dockerized setup** for the data collector.

### Quick start

```bash
# Build and start the collector in background
docker compose up -d

# Tail the logs (press CTRL-C to detach, container keeps running)
docker compose logs -f

# Stop the collector
docker compose down
```

### Configuration

The service is defined in [`docker-compose.yml`](./docker-compose.yml).
You can configure it via environment variables, either inline or using a `.env` file in the same folder:

| Variable          | Default            | Description                                |
| ----------------- | ------------------ | ------------------------------------------ |
| `SYMBOLS`         | `ETH` | Comma-separated list of symbols to collect |
| `OUTPUT_DIR`      | `HL_data` | Directory where Parquet files are written           |
| `ORDERBOOK_DEPTH` | `20`               | Orderbook depth to record                  |
| `TZ`              | `UTC`              | Timezone inside the container              |

Example `.env`:

```env
SYMBOLS=BTC,ETH
OUTPUT_DIR=HL_data
ORDERBOOK_DEPTH=20
TZ=UTC
```

### Data persistence

Collected Parquet files are stored on the host in `./HL_data` (mounted into the container).
This makes them directly usable by the parameter estimation scripts (`get_kappa.py`, `get_epsilon.py`, `get_lambda.py`).

### Logs

* Run `docker compose logs -f` to watch live output.
* Press **CTRL-C** to stop watching (collector continues running in the background).
* To detach from a non-detached `docker compose up`, press **CTRL-P + CTRL-Q**.

### Updating

If you change code or dependencies, rebuild with:

```bash
docker compose build
docker compose up -d
```

---

## Output Files

For each symbol, the collector writes **Parquet shards** into per-type subdirectories. Flush cadence is controlled by `FLUSH_INTERVAL_SEC` (default 10s — it must stay well below the strategy's `max_collector_age_seconds=30` freshness window or quotes get rejected as `stale_collector_data`).

> **Retention warning:** `RETENTION_MINUTES` (default 60, also set in `docker-compose.yml`) prunes shards older than the window. That is plenty for the live estimators (~30-min lookback), but it silently deletes the history the replay/calibration tooling (`replay_market_maker.py`, `calibrate_replay_from_logs.py`) consumes. When collecting a dataset for replays, set `RETENTION_MINUTES=0` (disable) or a value covering the full capture.

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
* **Docker**: Run `docker compose down`

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
