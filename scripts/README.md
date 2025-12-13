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

### Parameter Estimation (aligned with Cartea-Jaimungal model)
- **κ± (Kappa)**: Order book depth sensitivity estimated from λ(δ)=λ₀·exp(−κδ); saved to `kappa.json`
- **λ₀± (Lambda)**: Base arrival intensity at δ=0 (trades/sec) from the κ regression; saved to `lambda.json`
- **λ_trades± (Lambda trades)**: Unconditional trade arrival rates (trades/sec) from raw counts; saved to `lambda_trades.json` (sanity check)
- **ε± (Epsilon)**: Event-level permanent impact per trade from immediate mid jumps (~200 ms); saved to `epsilon.json`
- **Automatic data loading** with configurable time ranges
- **Market toxicity assessment** based on ε×κ product
- **HJB-ready outputs** for the strategy (λ, κ, ε feed optimal δ* computation)

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
# Joint κ/λ fit (λ(δ)=λ0·exp(-κδ)), saves kappa.json & lambda.json (trades/sec)
python get_kappa.py --crypto ETH --minutes 30

# Event-level ε from immediate post-trade jumps, saves epsilon.json
python get_epsilon.py --crypto ETH --minutes 30

# Optional raw trades/sec sanity check (writes lambda_trades.json)
python get_lambda.py --crypto ETH --minutes 30

# Inspect spreads across inventory (refreshes κ/ε/λ, then shows bid/ask and bps by q)
python compute_spreads.py --crypto ETH --mid 4322.05 --qmax 3
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

For each symbol, the collector writes **Parquet shards** into per-type subdirectories (flushed every ~60 seconds by default).

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

| Parameter         | Symbol | Description                               | Estimation Method                                |
| ----------------- | ------ | ----------------------------------------- | ------------------------------------------------ |
| **Lambda Plus**   | λ+     | Buy order arrival intensity (trades/sec)  | Base λ₀ from λ(δ)=λ₀·exp(−κδ) fit                |
| **Lambda Minus**  | λ-     | Sell order arrival intensity (trades/sec) | Base λ₀ from λ(δ)=λ₀·exp(−κδ) fit                |
| **Epsilon Plus**  | ε+     | Instant permanent jump from buy MOs       | Immediate mid change after trade (~200 ms)       |
| **Epsilon Minus** | ε-     | Instant permanent jump from sell MOs      | Immediate mid change after trade (~200 ms)       |
| **Kappa Plus**    | κ+     | Ask side order book depth sensitivity     | λ(δ) exponential decay regression                |
| **Kappa Minus**   | κ-     | Bid side order book depth sensitivity     | λ(δ) exponential decay regression                |

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

* **Buffered writing**: Data is collected in memory and flushed to disk periodically (default: every 60 seconds).
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
