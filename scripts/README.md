# Hyperliquid Market Making Suite

Collects Hyperliquid tick data and estimates the Cartea-Jaimungal parameters
the Rust trader calibrates against. The collectors run from `HYPERLIQUID_DATA/`,
not from here.

---

## Features

### Data Collection (`hyperliquid_data_collector.py`)
- **Real-time data collection** via WebSocket connections
- **Multiple data types**:
  - Best Bid/Offer (BBO) prices with timestamps
  - Trade executions with side, price, and volume
  - Order book snapshots (configurable depth, default 20 levels)
  - Active-asset context (oracle/mark/mid, open interest, funding, premium)
- **Parquet output** (compressed) organized by symbol and type
- **Live statistics** showing collection rates and summaries
- **Asynchronous data writing** to minimize performance impact
- **Graceful shutdown** with data preservation

### Parameter Estimation (aligned with Cartea-Jaimungal model, snapshot schema v5)
- **Market-order aggregation**: trade prints sharing side + exchange timestamp are ONE market order; depths and impacts are measured per MO, not per print
- **Mid-relative coordinates**: depths are measured from the prevailing mid (last BBO update strictly before the MO, exchange-timestamp aligned via merge_asof) — the same coordinate the strategy quotes in; negative depths are truncated to 0, not dropped
- **κ± (Kappa)**: survival-function fit — weighted log-linear regression of P(depth ≥ δ); saved to `kappa.json` with `depth_p95`/`depth_max_fitted` calibration diagnostics
- **λ± (Lambda)**: per-side MO arrival rate scaled by the survival fit's intercept A (schema v5, 2026-09-02: the fit says P(depth ≥ δ) = A·e^(−κδ), the HJB models λ·e^(−κδ), so the raw rate was off by A; `lambda±_raw` keeps the unscaled rate) — the survival-consistent fill rate the HJB needs (`lambda_source: mo_survival_fit`); the old binned-density intercept was bin-width dependent and ~3× too small (kept as `lambda0_intercept_±` diagnostic). Since 2026-08-19 the denominator is *observed* seconds, not wall clock: time when the collector was down (no print in the union of the price and trade streams) is subtracted, because dividing by wall clock understated λ by exactly the missing fraction. Both halves are published as `lambda_covered_seconds` / `lambda_outage_seconds_excluded` so the denominator is auditable. Measured effect on real windows: +1.2% to +6.0%; inside a 1 h price gap it would have been 50%
- **ε± (Epsilon)**: per-MO mean mid jump at the 200 ms primary horizon, after 3σ bad-tick clipping; 1 s and 5 s means are diagnostics (`epsilon_1s_±`, `epsilon_5s_±`), not model inputs; floor at 0 because C-J defines ε ≥ 0; saved to `epsilon.json`
- **σ² (`sigma2_per_sec`)**: realized mid variance (USDC²/s from 1 s increments, gap-tolerant), feeding the strategy's volatility-aware inventory penalty
- **Direct model parameters**: primary κ/ε/λ values are the validated estimates from the selected market-data window; no temporal smoothing is applied
- **λ_trades± (Lambda trades)**: unconditional trade-print rates from raw counts; saved to `lambda_trades.json` (monitoring only)
- **Status gating**: snapshots ship `status: ok` only when fit points ≥ 6, R² ≥ 0.30 and ε events ≥ 50 per side (the Rust calibrator enforces the same floors)
- **Relative adverse-selection diagnostic** based on ε×κ; this is not a profitability test because it omits fees, queue position, and latency

---

## Installation (local)

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

# Event-level ε per MO at the 200 ms arrival-jump horizon, saves epsilon.json
python get_epsilon.py --crypto CASHCAT --minutes 30

# Optional raw trades/sec sanity check (writes lambda_trades.json)
python get_lambda.py --crypto CASHCAT --minutes 30

# Inspect spreads across inventory (refreshes κ/ε/λ, then shows bid/ask and bps by q)
# --mid defaults to the freshly collected BBO mid via mid_price.json when omitted
python compute_spreads.py --crypto CASHCAT --qmax 6 --spread-multiplier 1.0
```

---

## Hyperliquid Data Collector (Docker)

**There is no compose file in this folder any more.** Both collectors that build
from here are defined in `HYPERLIQUID_DATA/docker-compose.yml`, alongside the
other three Hyperliquid collectors, and are operated from there:

- **`hl-cashcat-collector`** — `SYMBOLS=CASHCAT`, long retention
  (`CASHCAT_RETENTION_MINUTES`). The traded
  symbol needs a far longer tape than the rest: replay and the period archive
  can only score a window while its shards exist.
- **`hl-collector`** — `SYMBOLS=ETH,ACE,CHIP,PENGU,NIL`, 3 days, as controls.

**The two `SYMBOLS` lists must never overlap.** Both write into the same
directory and the estimators read the directory, not the writer, so an overlap
lands every trade twice. Retention values, the 2026-08-16 measurement and the
dedup that now protects calibration: `docs/DATA_COLLECTION.md`.

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
| `FLUSH_INTERVAL_SEC`      | `10`      | Buffer flush cadence; kept well below the Rust calibrator's 120 s maximum data age |
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

For each symbol, the collector writes **Parquet shards** into per-type
subdirectories. Flush cadence is controlled by `FLUSH_INTERVAL_SEC` (default
10 s). The Rust profiles reject calibration data older than 120 s, so 10 s
leaves margin for scheduling, file visibility, and a 30 s calibration cadence.

> **Retention warning:** `RETENTION_MINUTES` (collector-code default 60; the
> compose services override it — 3 days for `hl-collector`,
> `CASHCAT_RETENTION_MINUTES` for the traded symbol) prunes old shards. The code default is
> shorter than the Rust profiles' 120-minute calibration window plus required
> margin and is intended only for standalone collection tests. Replay
> datasets need retention covering the full capture; raise it explicitly, but do
> not use 0 because pruning interprets that literally. Shard-name selection keeps
> read cost tied to the requested window rather than total retention.

* `HL_data/<SYMBOL>/prices/prices_<epoch_ms>.parquet` (BBO updates)
* `HL_data/<SYMBOL>/trades/trades_<epoch_ms>.parquet` (trade executions)
* `HL_data/<SYMBOL>/orderbooks/orderbooks_<epoch_ms>.parquet` (order book snapshots)
* `HL_data/<SYMBOL>/asset_ctx/asset_ctx_<epoch_ms>.parquet` (active-asset context)

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

* **Asset context**:

  * `timestamp` (local receive time; this channel has no exchange timestamp)
  * `oracle_px`, `mark_px`, `mid_px`, `open_interest`, `funding`, `premium`
  * `impact_bid_px`, `impact_ask_px`, `day_ntl_vlm`

## Parameter Estimation

Each estimator and how it fails closed is described under Features above;
`docs/UNITS.md` carries the units.

### Relative adverse-selection diagnostic

The estimator reports **ε×κ** using the configured operating bands:

* **ε×κ < 1.0**: below the caution band
* **1.0 ≤ ε×κ < 1.5**: caution band
* **ε×κ ≥ 1.5**: rejected by the shipped calibration gate

These labels describe adverse selection relative to the model's natural depth
`1/κ`; they do not establish profit or loss. Run `verify_market_viability.py`
and replay representative windows before drawing an economic conclusion.

---

## Stopping the collector

Ctrl+C flushes buffered rows before exit. `docker compose down` from
`HYPERLIQUID_DATA/` does the same for the containers; killing them does not.

