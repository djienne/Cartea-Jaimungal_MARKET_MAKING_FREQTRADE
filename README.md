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
- **Real-time recalibration** every 15 seconds during trading, over a 30 minute Window by default

### 📊 Market Data Integration
- **Order book analysis** for bid-ask spread calculation
- **Trade flow analysis** for lambda (arrival rate) estimation
- **Mid-price tracking** for relative spread calculation

### 🔄 Automated Parameter Optimization
- **Continuous parameter estimation** using recent market data (30-minutes window by default)
- **Exponential decay models** for lambda estimation
- **Statistical analysis** of trade patterns and volatility

### 🏗️ Modular Architecture
- **Core strategy**: `Market_Making.py` - Main Freqtrade strategy
- **Parameter calculation**: `get_kappa.py`, `get_epsilon.py` - Dynamic parameter estimation
- **Data collection**: `hyperliquid_data_collector.py` - Market data gathering
- `periodic_test_runner.py` - Automated parameter updates to be used by Freqtrade

## Project Structure

```
Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/
├── user_data/
│   ├── config.json
│   ├── logs/
│   │   └── mm_debug.jsonl                 # Strategy debug JSONL (quotes/HJB/params)
│   └── strategies/
│       ├── Market_Making.py               # Main market making strategy
│       ├── periodic_test_runner.py        # Updates kappa/epsilon/lambda JSONs
│       ├── kappa.json
│       ├── epsilon.json
│       ├── lambda.json
│       └── lambda_trades.json
├── scripts/
│   ├── docker-compose.yml                 # Hyperliquid data collector
│   ├── Dockerfile                         # Collector image (includes pyarrow)
│   ├── hyperliquid_data_collector.py      # Writes Parquet shards to HL_data/<SYMBOL>/<dtype>/
│   ├── run_collector.py
│   ├── get_kappa.py
│   ├── get_epsilon.py
│   ├── get_lambda.py
│   ├── hjb.py
│   ├── compute_spreads.py
│   ├── mid_price.py
│   └── HL_data/
├── docker-compose.yml                     # Freqtrade compose
├── Dockerfile.technical                   # Freqtrade image (adds deps)
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

2. **Start data collection:**
   ```bash
   cd scripts
   docker-compose up -d
   ```
   Will write orderbook, price and orders data flow to files in directory `scripts/HL_data`

3. **Run the strategy:**
   ```bash
   # cd to root directory of this project
   docker compose up -d
   ```
   Only use in dry-run (paper trading)
   Monitor from Freqtrade web client, or set-up Telegram interface.
   **WARNING: run the data collector for a while before launching live trading**

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

These are automatically updated each bot loop (throttled by `internals.process_throttle_secs`, default 15s) based on recent market data.

## Usage Examples

### Manual Parameter calibration from data in `scripts/HL_data`

```bash
# Test kappa calculation
python scripts/get_kappa.py --crypto ETH

# Test epsilon calculation  
python scripts/get_epsilon.py --crypto ETH

# Quick spread check (refreshes κ/ε/λ first, then prints table of spreads vs inventory)
python scripts/compute_spreads.py --crypto ETH --mid 4322.05
```

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

- **get_kappa.py**: Jointly estimates κ± and λ₀± via λ(δ)=λ₀·exp(−κδ) (trades/sec, price-unit depths)
- **get_epsilon.py**: Event-level ε± from immediate post-trade mid jumps (∼200ms window)
- **get_lambda.py**: Trades/sec sanity check from raw trade counts (per-symbol); writes `lambda_trades.json`
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




