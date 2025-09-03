# Advanced Market Making with Freqtrade

A sophisticated market making system built on Freqtrade, implementing dynamic spread optimization using Cartea-Jaimungal models and real-time parameter calculation for optimal bid-ask pricing.

## Overview

This project implements an advanced market making strategy that:

- **Dynamically calculates optimal bid-ask spreads** using Cartea-Jaimungal market making models
- **Continuously adapts to market conditions** through real-time parameter estimation (kappa, epsilon)
- **Integrates with Hyperliquid exchange** for high-frequency trading
- **Uses mathematical optimization** to minimize inventory risk while maximizing profits

## Key Features

### 🎯 Dynamic Spread Calculation
- **Kappa parameters** (`kappa+`, `kappa-`): Control spread sensitivity to inventory levels
- **Epsilon parameters** (`epsilon+`, `epsilon-`): Adjust for market volatility and adverse selection
- **Real-time recalibration** every 15 seconds during trading, over a 30 minute Window by default

### 📊 Market Data Integration
- **Order book analysis** for bid-ask spread calculation
- **Trade flow analysis** for lambda (arrival rate) estimation
- **Mid-price tracking** for relative spread calculation

### 🔄 Automated Parameter Optimization
- **Continuous parameter estimation** using recent market data
- **Exponential decay models** for lambda estimation
- **Statistical analysis** of trade patterns and volatility

### 🏗️ Modular Architecture
- **Core strategy**: `Market_Making.py` - Main Freqtrade strategy
- **Parameter calculation**: `test_kappa.py`, `test_epsilon.py` - Dynamic parameter estimation
- **Data collection**: `hyperliquid_data_collector.py` - Market data gathering
- `periodic_test_runner.py` - Automated parameter updates

## Project Structure

```
ADVANCED_MM/
├── user_data/
│   ├── strategies/
│   │   └── Market_Making.py          # Main market making strategy
│   ├── config.json                   # Freqtrade configuration
│   ├── kappa.json                    # Current kappa parameters
│   └── epsilon.json                  # Current epsilon parameters
├── scripts/
│   ├── test_kappa.py                # Kappa parameter calculation
│   ├── test_epsilon.py              # Epsilon parameter calculation
│   ├── periodic_test_runner.py      # Parameter update orchestrator
│   ├── hyperliquid_data_collector.py # Market data collection
│   └── market_making_introduction.py # Mathematical models
├── docker-compose.yml               # Container orchestration
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
   docker-compose up -d
   ```

3. **Run the strategy:**
   ```bash
   # Dry run (recommended first)
   freqtrade trade --config ./user_data/config.json --strategy Market_Making --dry-run
   
   # Live trading (after testing)
   freqtrade trade --config ./user_data/config.json --strategy Market_Making
   ```

### Docker Deployment

```bash
# Build and start the market making bot
docker-compose up -d

# View logs
docker logs MM_ADV -f

# Stop the bot
docker-compose down
```

## Configuration

### Main Configuration (`user_data/config.json`)

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

- `kappa.json`: Risk aversion parameters
- `epsilon.json`: Market impact adjustments

These are automatically updated every 15 seconds based on market conditions.

## Usage Examples

### Backtesting

```bash
# Download historical data
freqtrade download-data --timeframe 1m 5m 15m --days 190 --config ./user_data/config.json

# Run backtest
freqtrade backtesting --strategy Market_Making --config ./user_data/config.json --timeframe 1m --timerange 20240701-
```

### Parameter Optimization

```bash
# Hyperopt for parameter optimization
freqtrade hyperopt --strategy Market_Making --timeframe 1m --min-trades 25 \
  --config ./user_data/config.json --hyperopt-loss MultiMetricHyperOptLoss \
  --timerange 20240701- --spaces buy -j 8 -e 1000

# View results
freqtrade hyperopt-show --best -n -1
```

### Manual Parameter Testing

```bash
# Test kappa calculation
python scripts/test_kappa.py --crypto WLFI --time-range 60

# Test epsilon calculation  
python scripts/test_epsilon.py --crypto WLFI --time-range 60
```

## Key Components

### Market_Making.py

The main strategy implementing:
- **Dynamic spread calculation** based on current parameters
- **Order book analysis** for mid-price determination  
- **Custom entry/exit pricing** using Cartea-Jaimungal formulas
- **Real-time parameter loading** from JSON configuration files
- **HJB solution integration**: Uses pre-computed value function `h(t,q)`
- **Inventory skew adjustment**: *[Not yet implemented - planned enhancement to adjust optimal bid-ask spreads asymmetrically based on inventory position to remain market neutral]*

### Parameter Calculation Scripts

- **test_kappa.py**: Estimates optimal inventory risk aversion
- **test_epsilon.py**: Calculates market impact adjustments
- **periodic_test_runner.py**: Orchestrates continuous parameter updates

### Data Collection and Analysis

- **hyperliquid_data_collector.py**: Real-time market data collection
- **Order book analysis**: Captures bid-ask spreads and depth at multiple levels
- **Trade flow decomposition**: Separates market orders by direction and size
- **Statistical calibration**: Fits exponential models to fill probabilities
- **Jump detection**: Identifies permanent price impacts from informed trading

## Risk Management

### Built-in Protections

- **Maximum drawdown protection**: Stops trading at 5% drawdown
- **Position limits**: Single position with unlimited stake
- **Order timeouts**: 15-second unfilled order cancellation
- **Inventory risk control**: Dynamic spread adjustment based on position

### Monitoring

- **Real-time logging**: Comprehensive trade and parameter logging
- **Parameter tracking**: Historical parameter evolution
- **Performance metrics**: P&L, spread efficiency, inventory turnover

## Performance Optimization

### Strategy Tuning

1. **Adjust risk parameters**: Modify `phi` (inventory penalty) in parameter calculation
2. **Optimize time ranges**: Tune lookback periods for parameter estimation
3. **Fee optimization**: Account for exchange-specific fee structures
4. **Spread bounds**: Set minimum/maximum spread limits

### System Performance

- **Fast execution**: 15-second order timeout for rapid market response
- **Efficient data usage**: Optimized parameter calculation algorithms
- **Memory management**: Periodic cleanup of historical data

## Troubleshooting

### Common Issues

1. **Parameter files not found**: Ensure `kappa.json` and `epsilon.json` exist
2. **No market data**: Check Hyperliquid data collector is running
3. **Orders not filling**: Verify spread calculation and market conditions
4. **High inventory**: Review risk parameters and position limits

### Debugging

```bash
# Check parameter calculation
python scripts/test_kappa.py --verbose

# Verify market data
ls -la scripts/HL_data/

# Review strategy logs
tail -f user_data/logs/freqtrade.log
```

## Contributing

This is a sophisticated market making implementation suitable for:
- **Quantitative researchers** developing market microstructure models
- **High-frequency traders** implementing systematic strategies  
- **Academic researchers** studying market making dynamics
- **Crypto market makers** on low-latency exchanges like Hyperliquid

The system implements state-of-the-art research from Cartea & Jaimungal on optimal market making with adverse selection, providing a practical framework for systematic liquidity provision.

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in dry-run mode before deploying with real capital. Past performance does not guarantee future results.

## License


This project implements academic market making models and is intended for research and educational use.

