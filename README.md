# Advanced Market Making with Freqtrade

A sophisticated market making system built on Freqtrade, implementing dynamic spread optimization using Cartea-Jaimungal models and real-time parameter calculation for optimal bid-ask pricing.
**Works ONLY for Hyperliquid**.

## Overview

This project implements an advanced market making strategy that:

- **Dynamically calculates optimal bid-ask spreads** using Cartea-Jaimungal market making models
- **Continuously adapts to market conditions** through real-time parameter estimation (kappa, epsilon)
- **Integrates with Hyperliquid exchange** for high-frequency trading
- **Uses mathematical optimization** to minimize inventory risk while maximizing profits

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
- **Parameter calculation**: `test_kappa.py`, `test_epsilon.py` - Dynamic parameter estimation
- **Data collection**: `hyperliquid_data_collector.py` - Market data gathering
- `periodic_test_runner.py` - Automated parameter updates to be used by Freqtrade

## Project Structure

```
ADVANCED_MM/
├── user_data/
├── docker-compose.yml                # Docker compose to run Freqtrade (trading bot), WARNING: run the data collector for a while before launching live trading
│   ├── strategies/
│       ├── Market_Making.py          # Main market making strategy
│       ├── periodic_test_runner.py   # Parameter update orchestrator (update values in kappa.json/epsilon.json and copy files here so Market_Making.py can use)
│       ├── kappa.json                # Current kappa parameters
│       └── epsilon.json              # Current epsilon parameters
│   ├── config.json                   # Freqtrade configuration
├── scripts/
│   ├── docker-compose.yml           # Docker compose to run the data collector (collects inside `HL_data` full order book over +-20 levels, trades, bid-ask prices)
│   ├── HL_data                      # contains Hyperliquid gathered data with full order book over +-20 levels, trades, bid-ask prices
│   ├── test_kappa.py                # Kappa parameter calculation
│   ├── test_epsilon.py              # Epsilon parameter calculation
│   └── hyperliquid_data_collector.py # Market data collection
└── commands.txt                     # Common Freqtrade commands
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
3. **Boundary condition**: `h(T,q) = -α q²` (terminal penalty)

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
   Will write orderbook, price and orders data flow to files in directory `script/HL_data`

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
Uses WLFI by default now.
```json
{
    "max_open_trades": 1,
    "stake_currency": "USDC",
    "trading_mode": "futures",
    "exchange": {
        "name": "hyperliquid",
        "pair_whitelist": ["WLFI/USDC:USDC"]
    },
    "unfilledtimeout": {
        "entry": 15,
        "exit": 15
    }
}
```

### Parameter Files

The system maintains dynamic parameters in JSON files:

- `kappa.json`: order book depth parameter
- `epsilon.json`: Market impact adjustments

These are automatically updated every 15 seconds based on market conditions.

## Usage Examples

### Manual Parameter calibration from data in `script/HL_data`

```bash
# Test kappa calculation
python scripts/test_kappa.py --crypto WLFI

# Test epsilon calculation  
python scripts/test_epsilon.py --crypto WLFI
```

## Key Components

### Market_Making.py

The main strategy implementing:
- **Dynamic spread calculation** based on current parameters
- **Order book analysis** for mid-price determination  
- **Custom entry/exit bid-ask spread pricing** using Cartea-Jaimungal formulas
- **Real-time parameter loading** from JSON configuration files
- **Inventory skew adjustment**: *[Not yet implemented - planned enhancement to adjust optimal bid-ask spreads asymmetrically based on inventory position to remain market neutral]*

### Parameter Calculation Scripts

- **test_kappa.py**: Estimates parameter for order book depth
- **test_epsilon.py**: Calculates market impact adjustments
- **periodic_test_runner.py**: Orchestrates continuous parameter updates

## Risk Management

### Built-in Protections

- **Maximum drawdown protection**: Stops trading at 5% drawdown
- **Position limits**: Single position with unlimited stake
- **Order timeouts**: 15-second unfilled order cancellation
- **Inventory risk control**: Dynamic spread adjustment based on position

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in dry-run mode before deploying with real capital. Past performance does not guarantee future results.
ONLY USE IN DRY-RUN

## License


This project implements academic market making models and is intended for research and educational use.








