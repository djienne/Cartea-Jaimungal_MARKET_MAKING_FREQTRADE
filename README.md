# Advanced Market Making with Freqtrade and Hyperliquid

A market making system built on Freqtrade, implementing the Cartea–Jaimungal–Penalva
model (Chapter 10 of *Algorithmic and High-Frequency Trading*, 2015) with real-time
parameter estimation. **Works ONLY for Hyperliquid**.

---

## ⚠️ Read this before running it with real money

**The goal of this project is a faithful implementation of the Cartea–Jaimungal
method, not a competitive trading strategy. It is not expected to be profitable
as deployed, and measurement says it is not.**

**Latency is the binding constraint, and this stack does not have it.** A market
maker earns the spread and pays for being slow: every millisecond your quote sits
stale is time an informed trader can pick it off. The model assumes you requote
*continuously*. This stack does not come close:

| | measured |
|---|---|
| median requote interval | **~30 s** |
| p90 | ~50 s |
| worst observed | ~100 s |

That is Freqtrade's loop cadence plus Python plus ccxt plus your home
connection — three to five orders of magnitude slower than the venue moves.

What that costs, measured rather than assumed: adverse selection roughly
**doubles** between a 200 ms and a 5 s markout (ε went 1.98→3.98 bps on the ask
and 3.53→6.14 bps on the bid). Your quotes are exposed for *thirty seconds*.

A live dry run on CASHCAT lost **64 USDC over 16 trades** while the price ran
+10%, and a replay sweep over the inventory-penalty parameter found **every
setting loses money** — the best of them −23 USDC over 10 hours. Shrinking the
order size 7.5× only cut the loss 2.6×, which is the signature of negative
expected value rather than a mis-set risk limit.

**If you intend to trade this seriously you need a co-located VPS** — an AWS
instance in the region nearest the exchange (Tokyo is the usual choice for
Hyperliquid), on a low-latency link, with the quoting loop rewritten to requote
in milliseconds rather than on a candle cadence. **And even that may not be
enough.** Competitive market making on a 20 bps spread is a latency race against
firms with dedicated infrastructure; nothing here has been shown to win it.

Use this to understand and verify the model. Treat any profit as unproven.

---

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
- **Real-time recalibration** by the `param-estimator` sidecar (default every ~30s, over a **120-minute** window via `PARAM_ESTIMATOR_WINDOW_MINUTES`); the strategy reloads the snapshot every 20s. The window was widened from 30 min because a short window often contains too few distinct market-order depths for the kappa survival fit to clear its gate, so cycles passed or failed on luck

### 📊 Market Data Integration
- **Order book analysis** for bid-ask spread calculation
- **Trade flow analysis** for lambda (arrival rate) estimation
- **Mid-price tracking** for relative spread calculation

### 🔄 Automated Parameter Optimization
- **Continuous parameter estimation** using recent market data (120-minute window by default)
- **Exponential decay models** for lambda estimation
- **Statistical analysis** of trade patterns and volatility

### 🏗️ Modular Architecture (four services here, plus a shared collector)
- **`mm-long` (MM_ADV_LONG) and `mm-short` (MM_ADV_SHORT)**: two instances of
  `Market_Making.py`, on separate Hyperliquid sub-accounts, that together present a
  two-sided quote. Freqtrade rests at most one order per pair and suppresses an exit
  signal whenever an entry is on the same candle, so one instance cannot quote both
  sides; Hyperliquid perps also net one-way per coin, so the two legs need separate
  sub-accounts to hold an independent long and short. Roles come from
  `config.long.json` / `config.short.json`. Net inventory `q = q_long + q_short` is
  shared over Redis and is what prices both legs; `mm_core.route_sides` decides which
  leg owns which side each cycle, preferring the assignment that unwinds gross
  inventory. Each leg fails closed if its peer's heartbeat goes stale.
- **`hl-collector`** *(not in this compose file)*: `hyperliquid_data_collector.py` - Streams live Hyperliquid order book / trade / price data to Parquet shards. It is operated from `HYPERLIQUID_DATA/docker-compose.yml` alongside the other three Hyperliquid collectors; `scripts/HL_data` is a junction into `HYPERLIQUID_DATA/data/eth_mm`, so every reader here still finds its data where it always did. **Do not add a collector to this project's compose** — two collectors sharing one output directory write every tick twice under different shard names, which silently doubled `n_trades` and `λ±` on 2026-08-16.
- **`param-estimator`**: `periodic_test_runner.py --loop` - Sidecar that runs `estimate_all.py` each cycle, validates the snapshot, copies it for the strategy, and publishes it to Redis. `estimate_all.py` loads the market window **once** and drives κ, ε and λ from it; running the three scripts separately made each re-scan the same parquet shards, which is how a cycle could outrun its own interval and stall parameter updates for 87 minutes on 2026-08-16. Cycle duration is recorded in `param_update_status.json`, and `--cycle-timeout-seconds` bounds a wedged cycle so it cannot hold the lock and starve every later one.
- **`redis` (mm-redis)**: Atomic transport for the κ/ε/λ snapshot (`scripts/param_store.py`). The strategy prefers the single Redis blob (no torn multi-file reads, no estimator-lock stalls) and falls back to the JSON files when Redis is unavailable.

## Project Structure

```
Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/
├── user_data/
│   ├── config.json
│   ├── logs/
│   │   └── mm_debug.jsonl                 # Debug JSONL, shared by both legs; each
│                                      # record carries a "role" field
│   └── strategies/
│       ├── Market_Making.py               # Main market making strategy
│       ├── periodic_test_runner.py        # Estimator sidecar (per-cycle lock, Redis publish)
│       ├── kappa.json
│       ├── epsilon.json
│       ├── lambda.json
│       └── lambda_trades.json
├── scripts/
│   ├── Dockerfile                         # Collector image (built from HYPERLIQUID_DATA's compose)
│   ├── hyperliquid_data_collector.py      # Writes Parquet shards to HL_data/<SYMBOL>/<dtype>/
│   ├── run_collector.py
│   ├── mm_core.py                         # Shared quoting core (engine + replay import this)
│   ├── verify_market_viability.py         # Can a passive maker profit here at all?
│   ├── estimate_all.py                    # One market-window load per estimator cycle
│   ├── param_store.py                     # Redis transport for the κ/ε/λ snapshot
│   ├── get_kappa.py
│   ├── get_epsilon.py
│   ├── get_lambda.py
│   ├── hjb.py
│   ├── replay_market_maker.py             # Event replay harness (queue, latency, markouts)
│   ├── hyperliquid_alo_executor.py        # Guarded post-only (Alo) order primitives
│   ├── hyperliquid_risk_executor.py       # Reduce-only IOC flatten
│   ├── compute_spreads.py
│   ├── mid_price.py
│   └── HL_data/                           # junction -> HYPERLIQUID_DATA/data/eth_mm
├── tests/                                 # pytest suite (strategy guards, runner, gates, replay)
├── docker-compose.yml                     # freqtrade + estimator + redis (collector lives elsewhere)
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

**Reading the control back out.** The solution is `δ*(t,q)`, a *surface*, and
both coordinates are read as such:

- **Time.** The solver keeps every backward step, and each quote uses the slice
  at the episode's actual time-to-go (`hjb_time_mode = "episodic"`). A perpetual
  instrument has no terminal time, so episodes run for `T` and restart — at the
  horizon, or once inventory is genuinely flat. Setting `"stationary"` reverts to
  reading only the `t=0` slice, which is what this project did until now and
  which leaves `α` inert.
  - **Departure:** the book liquidates the residual at `T` at market and pays
    `α q²`. Every quote here is post-only by construction, so the terminal
    condition acts through the depths alone — it cannot force a taker unwind.
  - At our calibration the terminal effect runs *opposite* to the textbook
    picture: `φκT = 10` against `ακ = 0.05`, so the running penalty — which is
    what remains to be paid over the time left, hence largest at `t=0` and gone
    at `T` — dominates, and the agent unwinds hardest at the *start* of an
    episode. See `docs/UNITS.md`.
- **Inventory.** Eq. 10.2 makes `q` a unit-jump count, so `h` exists only at
  integer `q` — but partial fills land in between. Depths are blended linearly
  between the bracketing integers (exact at every integer), and the leftover
  `q_residual` is logged on every quote and fill so unmodelled risk is visible
  rather than rounded away.

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

**⚠️ The fee comes first. `κ × ε` says nothing about whether you can pay it.**

The toxicity thresholds below are a *relative* measure of adverse selection.
They contain no fee term, so a market can score beautifully on them and still be
guaranteed to lose money. That is not hypothetical — it is what this project did
for months on ETH:

| | |
|---|---|
| ETH perp quoted spread | **0.53 bps** (exactly one tick) |
| Round trip at the touch earns | 0.53 bps |
| Two maker fees cost | **3.00 bps** |
| Net per round trip | **−2.47 bps** |
| `κ × ε` verdict | 0.02 — "low toxicity, potentially profitable" |

Every quote was floor-clamped at `min_half_spread_bps` and sat ~11× past the
depth 95% of market orders ever reach, and the replay recorded **no maker fills
in any variant**. The model was working correctly; it was being asked a question
that cannot detect an unpayable fee.

**Check viability before calibrating anything:**

```bash
python scripts/verify_market_viability.py --crypto ALL
```

It answers the prior question — *can a passive maker profit here at all?* — from
an empirical profit curve rather than the fitted model, so a degenerate κ cannot
hide the answer:

```
edge(δ)   = δ − maker_fee·mid − E[markout | depth ≥ δ]
volume(δ) = traded size of market orders reaching δ, per hour
pnl(δ)    = volume(δ) · edge(δ)
```

maximised over every observed depth, on both sides, summing the losing side too.
A screen of all 60 Hyperliquid perps above $0.5M daily volume found **28 pinned
at exactly one tick like ETH** — a maker cannot even improve the quote there —
and only 9 that clear the 3 bps round-trip fee *and* are wider than one tick.

**Necessary condition, in plain terms:** the quoted spread must exceed
`2 × maker_fee + adverse selection`. On Hyperliquid's 1.5 bps base maker tier
that means a spread wider than ~3 bps before any edge exists at all.

**Then, and only then, the toxicity thresholds apply:**
- `κ × ε < 1`: Low toxicity, potentially profitable with good latency
- `1 ≤ κ × ε ≤ 2`: Competitive but manageable with superior models
- `κ × ε ≥ 2`: Highly toxic market, avoid unless exceptional edge

**Other conditions:**
- **High λ**: Many market orders → frequent spread capture
- **Low κ**: Deep order book → ability to charge wider spreads
- **Low ε**: Limited informed trading → minimal adverse selection

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
   - `hl-collector` (started separately from `HYPERLIQUID_DATA`) writes order book / price / trade Parquet shards to `scripts/HL_data`, flushed every `FLUSH_INTERVAL_SEC`, merged into one file per hour after `COMPACT_AFTER_MINUTES`, and pruned after `RETENTION_MINUTES` (default 3 days).
     Compaction is what makes a long retention affordable: a 10 s flush writes ~360 shards/hour/stream, and for the 83-column orderbook schema parquet spends 27,343 bytes per row on 664 bytes of data. Merging an hour of ETH orderbooks takes 18.4 MB → 0.27 MB and the estimator's read of that directory from 2199 ms → 6 ms.
     **Do not disable pruning to capture a replay dataset** — raise `RETENTION_MINUTES` instead. Reads select shards by the timestamp in the filename, so cost tracks the window you ask for rather than everything on disk.
   - `param-estimator` recomputes κ/ε/λ each cycle and publishes the snapshot to Redis (plus the JSON files as fallback).
   - `freqtrade` (MM_ADV) quotes only when params, collector data, and order book are all fresh.

   Only use in dry-run (paper trading).
   Monitor from the Freqtrade web client, or set up the Telegram interface.
   **WARNING: let the data collector run at least as long as the estimation window (120 minutes by default) before expecting valid parameters** — the estimators need a full window.

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
        "pair_whitelist": ["CASHCAT/USDC:USDC"]
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
delta_total = clamp(HJB_depth * spread_multiplier + maker_fee * mid + extra_cushion_bps/1e4 * mid,
                    min_half_spread_bps, max_half_spread_bps)
```

The fee cushion is **one** maker fee per side (a round trip costs two fees and
collects the cushion twice), and the multiplier scales only the model term, so
widening quotes defensively never inflates the fee compensation.

- `trading_enabled`: master switch for quoting (safe with `dry_run: true`; live use additionally requires post-only evidence and stage gates — fail-closed).
- `maker_fee_rate`: the maker fee the quotes assume. Must match what ccxt reports for the exchange or the fee gate fail-closes (pinned ccxt 4.5.22 reports Hyperliquid's documented 0.00015).
- `spread_multiplier`: scales the HJB model depth. **Kept at 1.0**: nothing in the book supports scaling delta*, least of all the 1/kappa term which *is* the optimum, and a multiplier scales the inventory skew along with it. Use the additive `extra_cushion_bps` (default 0) for defensive widening — additive shifts both sides equally and leaves the skew intact.
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

### Market viability (run this first)

```bash
# Can a passive maker profit on this instrument at all, before any calibration?
python scripts/verify_market_viability.py --crypto ALL --minutes 4320
```

Writes `docs/market_viability_report.json`. Exits non-zero when no symbol clears
the bar. It refuses to issue a verdict on less than 6 hours of collector data —
a short window describes whichever regime it landed in, not the instrument
(CASHCAT was measured running 19.3× its own daily average volume for 15 minutes,
and over that burst the profit curve read +$4,117/h).

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




