# Advanced Market Making with Freqtrade and Hyperliquid

A sophisticated market making system built on Freqtrade, implementing dynamic spread optimization using Cartea-Jaimungal models and real-time parameter calculation for optimal bid-ask pricing.
**Works ONLY for Hyperliquid**.

## Overview

This project implements an advanced market making strategy that:

- **Dynamically calculates optimal bid-ask spreads** using Cartea-Jaimungal market making models
- **Continuously adapts to market conditions** through real-time parameter estimation (kappa, epsilon)
- **Integrates with Hyperliquid exchange** for high-frequency trading
- **Uses mathematical optimization** to minimize inventory risk while maximizing profits

**💰 Support this project**:
- **Hyperliquid**: Sign up with [this referral link](https://app.hyperliquid.xyz/join/FREQTRADE) for 10% fee reduction

## Key Features

### 🎯 Dynamic Spread Calculation
- **Kappa parameters** (`kappa+`, `kappa-`): Control order book depth and steepness
- **Epsilon parameters** (`epsilon+`, `epsilon-`): Adjust for market volatility and adverse selection
- **Real-time recalibration** by the `param-estimator` sidecar (default every ~30s, over a 30-minute window); the strategy reloads the snapshot every 20s

### 📊 Market Data Integration
- **Order book analysis** for bid-ask spread calculation
- **Trade flow analysis** for lambda (arrival rate) estimation
- **Mid-price tracking** for relative spread calculation

### 🔄 Automated Parameter Optimization
- **Continuous parameter estimation** using recent market data (30-minutes window by default)
- **Exponential decay models** for lambda estimation
- **Statistical analysis** of trade patterns and volatility

### 🏗️ Modular Architecture (four Docker Compose services)
- **`freqtrade` (MM_ADV)**: `Market_Making.py` - Main Freqtrade strategy (consumes parameter snapshots, quotes)
- **`hl-collector2`**: `hyperliquid_data_collector.py` - Streams live Hyperliquid order book / trade / price data to Parquet shards
- **`param-estimator`**: `periodic_test_runner.py --loop` - Sidecar that runs `get_kappa.py` / `get_epsilon.py` / `get_lambda.py` each cycle, validates the snapshot, copies it for the strategy, and publishes it to Redis
- **`redis` (mm-redis)**: Atomic transport for the κ/ε/λ snapshot (`scripts/param_store.py`). The strategy prefers the single Redis blob (no torn multi-file reads, no estimator-lock stalls) and falls back to the JSON files when Redis is unavailable.

## Project Structure

```
Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/
├── user_data/
│   ├── config.json
│   ├── logs/
│   │   └── mm_debug.jsonl                 # Strategy debug JSONL (quotes/HJB/params)
│   └── strategies/
│       ├── Market_Making.py               # Main market making strategy
│       ├── periodic_test_runner.py        # Estimator sidecar (per-cycle lock, Redis publish)
│       ├── kappa.json
│       ├── epsilon.json
│       ├── lambda.json
│       └── lambda_trades.json
├── scripts/
│   ├── docker-compose.yml                 # Hyperliquid data collector (standalone)
│   ├── Dockerfile                         # Collector image (includes pyarrow)
│   ├── hyperliquid_data_collector.py      # Writes Parquet shards to HL_data/<SYMBOL>/<dtype>/
│   ├── run_collector.py
│   ├── param_store.py                     # Redis transport for the κ/ε/λ snapshot
│   ├── get_kappa.py
│   ├── get_epsilon.py
│   ├── get_lambda.py
│   ├── hjb.py
│   ├── compute_spreads.py
│   ├── mid_price.py
│   └── HL_data/
├── tests/                                 # pytest suite (strategy guards, runner, gates, replay)
├── docker-compose.yml                     # Full stack: freqtrade + collector + estimator + redis
├── Dockerfile.technical                   # Freqtrade image (adds deps + pinned ccxt)
├── analyze_trades.py
└── README.md
```

## Mathematical Foundation

### Cartea-Jaimungal Model with Adverse Selection

The strategy implements the Cartea-Jaimungal market making model, combining inventory risk and adverse selection:

**Core Stochastic Elements:**

| Element | Formula | Interpretation |
|---------|---------|----------------|
| **Mid-price dynamics** | `dS_t = σ dW_t + ε⁺ dM_t⁺ - ε⁻ dM_t⁻` | Brownian noise + permanent jumps from informed orders |
| **Market order arrivals** | `M_t± ~ Poisson(λ± t)` | Separate arrival rates for buy/sell market orders |
| **Quote depths** | Ask = `S_t + δ_t⁺`, Bid = `S_t - δ_t⁻` | Optimal spreads around mid-price |
| **Fill probability** | `P_hit = exp(-κ± δ±)` | Exponential decay with distance from mid |
| **Inventory** | `Q_t`: +1 when bid hit, -1 when ask hit | Running position from market making |

**Optimal Pricing Strategy:**
```
δ⁺* = 1/κ⁺ + ε⁺ - [h(t,q-1) - h(t,q)]    (Ask depth)
δ⁻* = 1/κ⁻ + ε⁻ - [h(t,q+1) - h(t,q)]    (Bid depth)
```

**Three-Component Decomposition:**
```
Half-Spread = 1/κ        + ε          + skew(Q)
             (friction)   (insurance)   (inventory)
```

Where:
- `κ±`: Order book depth sensitivity (higher = thinner book)  
- `ε±`: Permanent price impact from informed trading
- `h(t,q)`: Value function encoding inventory risk preference
- `fees`: Exchange maker fees (0.015% for Hyperliquid)

### Objective Function and Solution Method

**Market Maker's Optimization Problem:**
```
max E[X_T + Q_T S_T - α Q_T² - φ ∫₀ᵀ Q_u² du]
```
Where:
- `X_T + Q_T S_T`: Final P&L (cash + mark-to-market inventory)
- `α`: Terminal inventory penalty (end-of-day risk)
- `φ`: Running inventory penalty (intraday risk aversion)

**Solution Method - Hamilton-Jacobi-Bellman:**
1. **Ansatz**: `H(t,x,S,q) = x + qS + h(t,q)` (value function decomposition)
2. **Matrix method**: For symmetric κ, solve `∂_t ω + A ω = 0` where `h = log(ω)/κ`
3. **Backward Euler**: For asymmetric κ (κ+ ≠ κ-), solve the nonlinear HJB on a (t,q) grid via implicit backward-Euler.
4. **Boundary condition**: `h(T,q) = -α q²` (terminal penalty)

### Parameter Estimation and Calibration

**Lambda (λ±) - Order Arrival Intensity:**
- Estimated from trade frequency: `λ(δ) = λ₀ exp(-κδ)`
- Separate calibration for buy (`λ⁺`) and sell (`λ⁻`) sides
- Uses sliding window of recent market data

**Kappa (κ±) - Order Book Sensitivity:**
- Estimated from fill probability: `P(fill) = exp(-κδ)`
- Measures order book depth and liquidity
- Critical parameter: controls base spread width

**Epsilon (ε±) - Adverse Selection Cost:**
- Estimated from permanent price impact distribution
- Often follows Pareto distribution: `ε ~ Pareto(α, scale)`
- **Key insight**: if `κ × ε ≥ 1.5`, market becomes unprofitable due to toxicity

### Market Regimes and Profitability

**Profitable Conditions:**
- **High λ**: Many market orders → frequent spread capture
- **Low κ**: Deep order book → ability to charge wider spreads  
- **Low ε**: Limited informed trading → minimal adverse selection

**Toxicity Thresholds:**
- `κ × ε < 1`: Low toxicity, potentially profitable with good latency
- `1 ≤ κ × ε ≤ 2`: Competitive but manageable with superior models
- `κ × ε ≥ 2`: Highly toxic market, avoid unless exceptional edge

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Hyperliquid API credentials

### Quick Start

1. **Clone and configure:**
   ```bash
   # Configure exchange credentials in user_data/config.json
   # Set your Hyperliquid API keys
   ```

2. **Run the full stack** (collector + parameter estimator + redis + freqtrade):
   ```bash
   # from the root directory of this project
   docker compose up -d
   ```
   - `hl-collector2` writes order book / price / trade Parquet shards to `scripts/HL_data` (flushed every `FLUSH_INTERVAL_SEC`, pruned after `RETENTION_MINUTES` — disable pruning when capturing replay datasets).
   - `param-estimator` recomputes κ/ε/λ each cycle and publishes the snapshot to Redis (plus the JSON files as fallback).
   - `freqtrade` (MM_ADV) quotes only when params, collector data, and order book are all fresh.

   Only use in dry-run (paper trading).
   Monitor from the Freqtrade web client, or set up the Telegram interface.
   **WARNING: let the data collector run ~30 minutes before expecting valid parameters** (the estimators need a full window).

## Configuration

### Main Configuration (`user_data/config.json`)
Uses ETH by default now.
```json
{
    "max_open_trades": 1,
    "stake_currency": "USDC",
    "trading_mode": "futures",
    "exchange": {
        "name": "hyperliquid",
        "pair_whitelist": ["ETH/USDC:USDC"]
    },
    "unfilledtimeout": {
        "entry": 15,
        "exit": 15
    }
}
```

### Parameter Files

The system maintains dynamic parameters in JSON files:

- `kappa.json`: Order book depth sensitivity (`kappa+`, `kappa-`)
- `epsilon.json`: Adverse selection / permanent impact (`epsilon+`, `epsilon-`)
- `lambda.json`: Baseline trade arrival intensity (`lambda+`, `lambda-`)
- `lambda_trades.json`: Raw trades/sec monitor (sanity check; optional)

These are regenerated by the `param-estimator` sidecar each cycle (default every ~30s) from recent collector data, published to Redis as one atomic blob, and copied to `user_data/strategies/` as the file fallback. The strategy reloads its snapshot every `market_making.param_snapshot_reload_interval_seconds` (default 20s) and rejects anything older than `max_param_age_seconds`.

### Market-making knobs (`market_making` block in `config.json`)

The final half-spread per side is assembled as:

```text
delta_total = clamp(HJB_depth * spread_multiplier + maker_fee * mid,
                    min_half_spread_bps, max_half_spread_bps)
```

The fee cushion is **one** maker fee per side (a round trip costs two fees and
collects the cushion twice), and the multiplier scales only the model term, so
widening quotes defensively never inflates the fee compensation.

- `trading_enabled`: master switch for quoting (safe with `dry_run: true`; live use additionally requires post-only evidence and stage gates — fail-closed).
- `maker_fee_rate`: the maker fee the quotes assume. Must match what ccxt reports for the exchange or the fee gate fail-closes (pinned ccxt 4.5.22 reports Hyperliquid's documented 0.00015).
- `spread_multiplier`: scales the HJB model depth (1.0 = model optimum; production default 3.0).
- `min_half_spread_bps` / `max_half_spread_bps` (defaults 3 / 80): hard clamps on the final half-spread including fees. The floor guarantees every round trip collects at least ~2x the round-trip fee; the cap bounds how far quotes can drift from mid. Clamp activity is logged per quote (`clamped: "floor"|"cap"`).
- `gamma_inventory_risk` (default 0.05): volatility-aware inventory penalty. The HJB running penalty becomes `phi_effective = hjb_phi + gamma * sigma2_per_sec * inventory_unit_base` using the realized mid variance published by the estimator; missing sigma2 falls back to the static `hjb_phi`.
- `min_kappa_fit_points` / `min_kappa_r2` / `min_epsilon_events` (defaults 6 / 0.30 / 50): validation floors on the estimator fit diagnostics; snapshots below them are rejected (fail-closed).
- `PARAM_EMA_TAU_SECONDS` (environment, estimator-side, default 300): time constant of the EMA smoothing applied to the primary κ/ε/λ values across estimator cycles (0 disables; raw per-window values are kept in the `*_raw` keys).
- `REDIS_URL` (environment, set in `docker-compose.yml`): enables the atomic Redis snapshot transport; without it the strategy uses the JSON files guarded by `param_update.lock`.

## Usage Examples

### Manual Parameter calibration from data in `scripts/HL_data`

```bash
# Test kappa calculation
python scripts/get_kappa.py --crypto ETH

# Test epsilon calculation  
python scripts/get_epsilon.py --crypto ETH

# Quick spread check (refreshes κ/ε/λ first, then prints table of spreads vs inventory)
python scripts/compute_spreads.py --crypto ETH --spread-multiplier 3.0
```

### Safety gates

```bash
# fast profile (~2.5 min): static gates + full pytest + evidence evaluators
python scripts/run_safety_gates.py --markdown-output docs/LAST_SAFETY_GATES.md

# full battery (~17 min): adds docker probes + both dry-run smokes; restores MM_ADV afterwards
python scripts/run_safety_gates.py --include-runtime --json-output docs/last_safety_gates.json --markdown-output docs/LAST_SAFETY_GATES.md

# rerun: everything cheap re-runs live, smoke results reused from the last battery (<=6h old), ~4 min
python scripts/run_safety_gates.py --include-runtime --reuse-smoke-artifacts --json-output docs/last_safety_gates.json --markdown-output docs/LAST_SAFETY_GATES.md
```

See `docs/DEPLOYMENT_GATES.md` for the full gate list, battery profiles, and promotion requirements.

## Key Components

### Market_Making.py

The main strategy implementing:
- **Dynamic spread calculation** based on current parameters
- **Order book analysis** for mid-price determination  
- **Custom entry/exit bid-ask spread pricing** using Cartea-Jaimungal formulas
- **Real-time parameter loading** from JSON configuration files
- **Inventory skew adjustment**: Implemented via HJB grid (uses asymmetric κ+/κ- and ε+/ε- by default).
- **Debugging**: Writes `user_data/logs/mm_debug.jsonl` with per-quote spreads (bps from mid) and the parameters/HJB surface used.

### Parameter Calculation Scripts

- **get_kappa.py**: κ± from a survival-function fit of market-order depths measured from the mid (BBO stream, exchange timestamps, prints aggregated per MO); λ± is the raw per-side MO arrival rate. Also publishes `sigma2_per_sec` (realized mid variance) and `depth_p95` calibration diagnostics.
- **get_epsilon.py**: Event-level ε± per market order from the BBO mid stream; primary horizon 5 s (permanent impact), with 200 ms / 1 s diagnostics showing the decay profile.
- **get_lambda.py**: Trades/sec sanity check from raw trade counts (per-symbol); writes `lambda_trades.json`
- All primary κ/ε/λ values are EMA-smoothed across estimator cycles (`PARAM_EMA_TAU_SECONDS`, default 300 s); unsmoothed values live in the `*_raw` keys. Snapshot schema version: 3.
- **compute_spreads.py**: Refreshes κ/ε/λ then prints bid/ask prices and spreads (bps) across inventory levels
- **periodic_test_runner.py**: Orchestrates continuous parameter updates

## Risk Management

### Built-in Protections

- **Maximum drawdown protection**: Optional (currently commented out in `user_data/strategies/Market_Making.py`)
- **Position limits**: Single position with unlimited stake
- **Order timeouts**: 15-second unfilled order cancellation
- **Inventory risk control**: Dynamic spread adjustment based on position

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in dry-run mode before deploying with real capital. Past performance does not guarantee future results.
ONLY USE IN DRY-RUN

## License


This project implements academic market making models and is intended for research and educational use.




