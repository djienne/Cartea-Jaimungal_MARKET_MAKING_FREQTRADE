# Hyperliquid Market Making Suite

A comprehensive Python suite for collecting real-time tick data from Hyperliquid and estimating market making parameters. Includes data collection via WebSocket API, advanced parameter estimation, and a Dockerized setup for easy deployment.

---

## Features

### 📊 Data Collection (`hyperliquid_data_collector.py`)
- **Real-time data collection** via WebSocket connections
- **Multiple data types**:
  - Best Bid/Offer (BBO) prices with timestamps
  - Trade executions with side, price, and volume
  - Order book snapshots (configurable depth, default 20 levels)
- **One CSV per symbol and data type** (rolling append mode)
- **CSV output** for easy analysis and storage
- **Live statistics** showing collection rates and summaries
- **Asynchronous data writing** to minimize performance impact
- **Graceful shutdown** with data preservation

### 🧮 Parameter Estimation TO BE DONE
- **λ (Lambda)**: Market order arrival intensities for buy/sell orders
- **ε (Epsilon)**: Permanent price impact from informed trading
- **κ (Kappa)**: Order book depth sensitivity (liquidity measure)
- **Automatic data loading** with configurable time ranges
- **Quality metrics** and confidence intervals for estimates
- **Market toxicity assessment** based on ε×κ product
- **Comprehensive validation** and warning system

---

## Installation (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

---

## Quick Start (local)

### 1. Collect Data

```bash
# Start collecting data (creates one CSV per symbol & type)
python hyperliquid_data_collector.py
```

### 2. Estimate Parameters

```bash
# Estimate parameters for BTC using last 10 minutes of data
python market_making_parameters_estimator.py --symbol BTC --minutes 10

# Or use the example script
python example_parameter_estimation.py
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
| `SYMBOLS`         | `BTC,ETH,SOL,WLFI` | Comma-separated list of symbols to collect |
| `OUTPUT_DIR`      | `hyperliquid_data` | Directory where CSVs are written           |
| `ORDERBOOK_DEPTH` | `20`               | Orderbook depth to record                  |
| `TZ`              | `UTC`              | Timezone inside the container              |

Example `.env`:

```env
SYMBOLS=BTC,ETH
OUTPUT_DIR=hyperliquid_data
ORDERBOOK_DEPTH=20
TZ=UTC
```

### Data persistence

Collected CSVs are stored on the host in `./hyperliquid_data` (mounted into the container).
This makes them directly usable by the parameter estimation script (`test.py`).

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

For each symbol, the collector now maintains **one CSV file per data type** inside the output directory. Data is continuously appended while the collector runs.

* **`prices_SYMBOL.csv`** → BBO updates (bid/ask quotes and mid)
* **`trades_SYMBOL.csv`** → trade executions
* **`orderbooks_SYMBOL.csv`** → order book snapshots (configurable depth)

### Example (symbols: BTC, ETH, SOL)

```
hyperliquid_data/
├── prices_BTC.csv
├── trades_BTC.csv
├── orderbooks_BTC.csv
├── prices_ETH.csv
├── trades_ETH.csv
├── orderbooks_ETH.csv
├── prices_SOL.csv
├── trades_SOL.csv
└── orderbooks_SOL.csv
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

* **Continuous appends**: all new data is added to the same file
* **Simpler analysis**: no need to manage multiple timestamped files
* **Persistence**: files grow as long as the collector runs (rotate/compress periodically if needed)

---

## Parameter Estimation

### Parameters Estimated

| Parameter         | Symbol | Description                               | Estimation Method                |
| ----------------- | ------ | ----------------------------------------- | -------------------------------- |
| **Lambda Plus**   | λ+     | Buy order arrival intensity (orders/min)  | Trade frequency analysis         |
| **Lambda Minus**  | λ-     | Sell order arrival intensity (orders/min) | Trade frequency analysis         |
| **Epsilon Plus**  | ε+     | Permanent price impact from buy orders    | Price impact before/after trades |
| **Epsilon Minus** | ε-     | Permanent price impact from sell orders   | Price impact before/after trades |
| **Kappa Plus**    | κ+     | Ask side order book depth sensitivity     | Order book decay fitting         |
| **Kappa Minus**   | κ-     | Bid side order book depth sensitivity     | Order book decay fitting         |

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

* **Buffered writing**: Data is collected in memory and flushed to disk every few seconds
* **Threaded I/O**: CSV writing happens in background threads to avoid blocking
* **Configurable buffer sizes**: Prevent memory issues during bursts
* **Efficient data structures**: Uses deques for O(1) appends

---

## Stopping the Collector

* **Local**: Press `Ctrl+C` → graceful shutdown (flushes data and closes connections)
* **Docker**: Run `docker compose down`

---

## Dependencies

* `hyperliquid-python-sdk`: Official Hyperliquid SDK
* `websocket-client`: WebSocket client library
* `pandas`, `numpy`
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