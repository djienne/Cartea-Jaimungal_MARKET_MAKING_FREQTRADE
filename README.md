# Advanced Market Making with Freqtrade and Hyperliquid

A market making system built on Freqtrade, implementing the Cartea–Jaimungal–Penalva
model (Chapter 10 of *Algorithmic and High-Frequency Trading*, 2015) with real-time
parameter estimation. **Works ONLY for Hyperliquid**.

<p align="center">
  <a href="docs/spread_calculation.pdf">
    <img src="docs/img/spread_calculation_cover.png" width="290"
         alt="How the Spread Is Computed — cover of the PDF walkthrough">
  </a>
</p>

<p align="center">
  <b><a href="docs/spread_calculation.pdf">📄 How the Spread Is Computed</a></b><br>
  <sub>A beginner-friendly walkthrough: what the book says, the Python that
  implements it, and a fully worked example from L2 tick data to a resting order.<br>
  Every listing is checked against the source and every number recomputed by
  <a href="scripts/verify_spread_doc.py"><code>verify_spread_doc.py</code></a>.</sub>
</p>

---

## ⚠️ Read this before running it with real money

**The goal of this project is a faithful implementation of the Cartea–Jaimungal
method, not a competitive trading strategy. It is not expected to be profitable
as deployed, and measurement says it is not.**

**Latency is the binding constraint, and this stack does not have it.** A market
maker earns the spread and pays for being slow: every millisecond your quote sits
stale is time an informed trader can pick it off. The model assumes you requote
*continuously*. This stack does not come close:

| | measured, between order placements |
|---|---|
| p10 | 1.0 s |
| median requote interval | **5.5 s** |
| p90 | 10.9 s |

Measured 2026-08-18 at `unfilledtimeout` 3 s and `internals.process_throttle_secs`
2; at the previous 15 s / 15 s the same measurement read p50 30.3 s, p90 ~60 s.
Since 2026-08-19 a *third* requote path is also live — Freqtrade's position-adjust
replace, which cancels and re-places the resting order rather than stacking a
second one, and unlike `replace_order` is not candle-gated — so the figures above
are now an upper bound on the interval. It is still Freqtrade's loop cadence plus
Python plus ccxt plus your home connection: three to five orders of magnitude
slower than the venue moves.

What that costs, measured rather than assumed: adverse selection roughly
**doubles** between a 200 ms and a 5 s markout (ε went 1.98→3.98 bps on the ask
and 3.53→6.14 bps on the bid). Your quotes are exposed for *seconds*.

**And cutting the cadence only makes it pay in the colocated scenario.** The
current evidence is a staged parameter sweep on a pinned **95.23 h** CASHCAT
tape (546,818 price rows, 243,653 trades, 0.7 train/held-out split) —
`docs/cashcat_sweep.md`:

- **Every finalist still loses out of sample**: −10.24 USDC, only 5 of 16
  six-hour windows positive, worst window −13.55.
- **Latency is economically decisive in this tape.** The complete infrastructure
  ladder reads: colocated (50 ms latency / 100 ms refresh) **+23.07**, good
  (100/250 ms) −10.24, mid (200/500 ms) −16.49, this stack (500/1000 ms)
  −49.19, and the slow-refresh reality case (500 ms / 30 s) −148.60. Latency and
  requote cadence move together in this ladder, so it compares plausible
  machines rather than claiming a pure one-variable latency experiment.
- **The spread is earned; the direction gives it back.** At 100 ms the net
  realized spread after fees is +273.91 USDC, but the directional/adverse term
  is −284.15, leaving −10.24. At 50 ms, +301.16 of net spread barely outruns
  −278.10 of directional loss.
- **Performance is unstable across time.** Only 5/16 held-out windows are
  positive. The same selected configuration ranges from +7.63 to −13.55 USDC
  over six-hour windows despite 1,135 held-out maker fills in aggregate.
- Stage A, which holds the risk knobs at the strategy default
  `hjb_phi_kappa_t = 10`, loses on **all 81 calibrations**, on every tape measured.

`docs/replay_acceptance_report.*` is an older 24-minute fail-closed gate smoke
with `ok=false`; it is retained as evidence but must not be mistaken for the
95-hour staged sweep above.

**Real money has since been risked, twice.** `docs/live_canary_20260823.md`
records two mainnet CASHCAT sessions at minimum size: what they cost, the three
production defects they surfaced, and why the venue's **address-action budget**
— a lifetime account allowance of `10,000 + 1 per USDC traded` that never
resets — is the constraint that actually binds this strategy, ahead of the
WebSocket message rate the configuration validates against.

`docs/latency_hysteresis_sweep.md` crosses simulated latency (50/100/200/500 ms)
with the requote hold window on one frozen two-hour window. It resolves the
**address-action cost** cleanly — `bps = 4` buys ~2.2x the runtime per unit of
venue allowance that the shipped `bps = 2` does — but it is explicit that a
two-hour window at ~50 fills **cannot** resolve latency economics, and that the
95-hour ladder above remains the evidence for those.

Market data is produced by a Docker container, not by this repo; see
`docs/DATA_COLLECTION.md` for who owns the tape, why no live session can
disturb it, and how to check that it is still running.

Shorter tapes read positive out of sample — +1.18 on 24.8 h, +1.56 on 31.23 h
(`docs/cashcat_sweep_phitail.md`), +1.37 on 44.97 h — while both the 60.32 h and
95.23 h tapes read negative. The winning calibration also moved between tapes,
so nothing here supports a claim of calibration stability. The later Rust
direct-window replay independently lost 17.43 USDC over 112 fills and showed
positive 100 ms markout turning sharply negative by 1–30 seconds.

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

### 🏗️ Modular Architecture (four services here, plus two shared collectors)
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
- **`hl-collector` and `hl-cashcat-collector`** *(neither in this compose file)*: two instances of `hyperliquid_data_collector.py` - Stream live Hyperliquid order book / trade / price data to Parquet shards. They are operated from `HYPERLIQUID_DATA/docker-compose.yml` alongside the other three Hyperliquid collectors; `scripts/HL_data` is a junction into `HYPERLIQUID_DATA/data/eth_mm`, so every reader here still finds its data where it always did. `hl-cashcat-collector` carries **CASHCAT alone at 30-day retention** because the traded symbol is what the sweeps and the replay acceptance gate read; `hl-collector` carries `ETH,ACE,CHIP,PENGU,NIL` at 3 days as controls and candidates. **Their `SYMBOLS` lists must stay disjoint, and do not add a collector to this project's compose** — two collectors sharing one output directory write every tick twice under different shard names, which silently doubled `n_trades` and `λ±` on 2026-08-16. A watchdog fix on 2026-08-19 made a collector reconnect on a websocket *close* within ~10 s instead of waiting out `INACTIVITY_TIMEOUT_SEC=180`; before it, Hyperliquid's ~3-hourly session expiry cost 20 gaps of 3.1-3.5 min over 60 h of CASHCAT — 71% of all missing data, 2.5% of the span.
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
│   ├── verify_strategy_attributes.py      # Static check: no self.X the strategy never defines
│   ├── sweep_replay.py                    # Staged train/held-out parameter sweep
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
├── rust_live/                             # Pure-Rust CASHCAT runtime workspace
│   ├── crates/cj-core/                    # HJB, quote policy, instrument math
│   ├── crates/cj-data/                    # Parquet calibration and replay
│   ├── crates/mm-execution/               # Execution traits and dry-run simulator
│   ├── crates/mm-runtime/                 # Hot thread, atomics, latency observer
│   └── crates/hyperliquid/                # Signing, transport, state, live backend
├── docker-compose.yml                     # mm-long + mm-short + estimator + redis (collectors live elsewhere)
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
    picture: the shipped `φκT = 200` against `ακ = 0.05`, so the running penalty
    — which is what remains to be paid over the time left, hence largest at `t=0`
    and gone at `T` — dominates, and the agent unwinds hardest at the *start* of
    an episode. See `docs/UNITS.md`.
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

2. **Run the stack** (two quoting legs + parameter estimator + redis; the
   collectors are started separately, from `HYPERLIQUID_DATA`):
   ```bash
   # from the root directory of this project
   docker compose up -d
   ```
   - `hl-collector` and `hl-cashcat-collector` (started separately from `HYPERLIQUID_DATA`) write order book / price / trade Parquet shards to `scripts/HL_data`, flushed every `FLUSH_INTERVAL_SEC`, merged into one file per hour after `COMPACT_AFTER_MINUTES`, and pruned after `RETENTION_MINUTES` (3 days for the five control symbols, 30 days for CASHCAT; the collector's own default is 60 minutes).
     Compaction is what makes a long retention affordable: a 10 s flush writes ~360 shards/hour/stream, and for the 83-column orderbook schema parquet spends 27,343 bytes per row on 664 bytes of data. Merging an hour of ETH orderbooks takes 18.4 MB → 0.27 MB and the estimator's read of that directory from 2199 ms → 6 ms.
     **Do not disable pruning to capture a replay dataset** — raise `RETENTION_MINUTES` instead. Reads select shards by the timestamp in the filename, so cost tracks the window you ask for rather than everything on disk.
   - `param-estimator` recomputes κ/ε/λ each cycle and publishes the snapshot to Redis (plus the JSON files as fallback).
   - `mm-long` (MM_ADV_LONG) and `mm-short` (MM_ADV_SHORT) quote only when params, collector data, and order book are all fresh.

   Only use in dry-run (paper trading).
   Monitor from the Freqtrade web client, or set up the Telegram interface.
   **WARNING: let the data collector run at least as long as the estimation window (120 minutes by default) before expecting valid parameters** — the estimators need a full window.

## Configuration

### Main Configuration (`user_data/config.json`)
Uses CASHCAT by default now — it is the only symbol that clears the viability gate.
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
        "entry": 3,
        "exit": 3,
        "unit": "seconds"
    },
    "internals": {
        "process_throttle_secs": 2
    }
}
```

`unfilledtimeout` and `internals.process_throttle_secs` are the requote cadence:
a resting order is cancelled on timeout and re-placed some number of bot
iterations later. They do not compose linearly — 15 s / 15 s measured a median of
30.3 s, 3 s / 2 s measured 5.5 s, and tightening to 2 s / 1 s measured no faster
while doubling API load.

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
- `min_half_spread_bps` / `max_half_spread_bps` (defaults 1.5 / 80): hard clamps on the final half-spread including fees. The floor is anchored to the maker fee — a half-spread below your own fee loses on the round trip regardless of anything else. It was 3.0 until 2026-08-17, where on live ETH it clamped all 24 quote sides and flattened the HJB's inventory skew into a constant; at 1.5 it binds on 1 of 24 sides, and below ~1.4 it never binds, because the fee cushion already guarantees that much. Defensive widening belongs in `extra_cushion_bps`, which is additive and leaves the skew intact. The cap bounds how far quotes can drift from mid. Clamp activity is logged per quote (`clamped: "floor"|"cap"`).
- `hjb_phi_kappa_t` / `hjb_phi_kappa_t_max` (shipped 200 / 300; strategy defaults 10 / 50): the dimensionless running inventory penalty `φκT` and its ceiling. `φ` is not κ-invariant (eq. 10.28), so the raw value is meaningless across symbols and the solver derives `φ` from live κ at every refresh. Until 2026-08-18 the ceiling was never passed through from config, so any `hjb_phi_kappa_t` above `QuoteConfig`'s default of 50 was silently clamped back to 50. At 200 the flat-inventory half-spread is ~39 bps rather than 15-18; the 23.1 h empirical profit curve put the optimum at 45 bps on the bid and 60 on the ask, so the previous setting was quoting inside the region where measured edge is negative.
- `gamma_inventory_risk` (default 0.05): volatility-aware inventory penalty. The HJB running penalty becomes `phi_effective = hjb_phi + gamma * sigma2_per_sec * inventory_unit_base` using the realized mid variance published by the estimator; missing sigma2 falls back to the static `hjb_phi`.
- `min_kappa_fit_points` / `min_kappa_r2` / `min_epsilon_events` (defaults 6 / 0.30 / 50): validation floors on the estimator fit diagnostics; snapshots below them are rejected (fail-closed).
- Parameter snapshots publish the direct validated estimates from each configured market-data window; no temporal smoothing is applied.
- `REDIS_URL` (environment, set in `docker-compose.yml`): enables the atomic Redis snapshot transport; without it the strategy uses the JSON files guarded by `param_update.lock`.

## Usage Examples

### Manual Parameter calibration from data in `scripts/HL_data`

```bash
# Test kappa calculation
python scripts/get_kappa.py --crypto CASHCAT

# Test epsilon calculation  
python scripts/get_epsilon.py --crypto CASHCAT

# Quick spread check (refreshes κ/ε/λ first, then prints table of spreads vs inventory)
python scripts/compute_spreads.py --crypto CASHCAT --spread-multiplier 3.0
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

# full battery (~17 min): adds docker probes + both dry-run smokes
# full battery incl. the docker gates and both dry-run smokes, ~17 min. Repaired
# 2026-08-19; it runs under MM_GATE_SMOKE and does not touch the live legs.
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
- **Requote cadence**: there is no refresh gate to configure. Three Freqtrade paths re-place a resting order: the `unfilledtimeout` cancel (dominant), the candle-gated `replace_order` → `adjust_entry_price`/`adjust_exit_price` (rare — with a 1 m timeframe and a 3 s timeout an order lives across a candle boundary about 3/60 of the time), and `process_open_trade_positions` → `adjust_trade_position`, live since 2026-08-19 and not candle-gated. The `adjust_trade_position` docstring documents all three against freqtrade 2025.10.
- **Debugging**: Writes `user_data/logs/mm_debug.jsonl` with per-quote spreads (bps from mid) and the parameters/HJB surface used.

### Parameter Calculation Scripts

- **get_kappa.py**: κ± from a survival-function fit of market-order depths measured from the mid (BBO stream, exchange timestamps, prints aggregated per MO); λ± is the raw per-side MO arrival rate. Also publishes `sigma2_per_sec` (realized mid variance) and `depth_p95` calibration diagnostics.
- **get_epsilon.py**: Event-level ε± per market order from the BBO mid stream; primary horizon 200 ms (the arrival jump in eq. 10.22), with 1 s / 5 s diagnostics showing the decay profile.
- **get_lambda.py**: Trades/sec sanity check from raw trade counts (per-symbol); writes `lambda_trades.json`
- All primary κ/ε/λ values are direct validated rolling-window estimates without temporal smoothing. Snapshot schema version: 4.
- **compute_spreads.py**: Refreshes κ/ε/λ then prints bid/ask prices and spreads (bps) across inventory levels
- **periodic_test_runner.py**: Orchestrates continuous parameter updates

## Risk Management

### Built-in Protections

- **Maximum drawdown protection**: Freqtrade's `protections()` block is optional and currently commented out in `user_data/strategies/Market_Making.py`; the strategy's own `max_daily_loss_usdc` (200) and `max_consecutive_losses` (25) kill switches are live.
- **Position limits**: `max_open_trades: 1` and `stake_amount: 500` USDC against `available_capital: 1000`. Inventory is carried as the *size* of that one trade, not as a count of trades: `custom_stake_amount` caps a new entry at one inventory unit and `adjust_trade_position` moves it one unit at a time, bounded by `hjb_q_max` (6) on both directions.
- **Order timeouts**: 3-second unfilled order cancellation (`unfilledtimeout`), with `internals.process_throttle_secs: 2`.
- **Inventory risk control**: Dynamic spread adjustment based on position
- **Inventory adjustment was dead until 2026-08-19.** `adjust_trade_position` read `self._kill_switch_active`, an attribute assigned nowhere, so it raised `AttributeError` on every call from commit d20b27d onward — 520 raises in 12 h on the long leg and 410 on the short. Freqtrade's `strategy_wrapper` catches that and returns `None`, which is indistinguishable from the callback declining, so the path was dead for weeks while the bot looked healthy. `scripts/verify_strategy_attributes.py` now catches this class of bug statically and runs as a safety gate. With the crash fixed, the short leg turned out to be inverted in every case — freqtrade reads the stake sign as "grow or shrink the position" while the code derived it from the direction `q` moves, and a bid on the short leg buys back and *shrinks* it — and only the positive branch was gated, so the short leg's adds ignored `q_max` entirely. Both are fixed, with 18 tests in `tests/test_strategy_guards.py`.

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in dry-run mode before deploying with real capital. Past performance does not guarantee future results.
ONLY USE IN DRY-RUN

## License


This project implements academic market making models and is intended for research and educational use.

## Standalone Rust Cartea–Jaimungal runtime

The Freqtrade/Python implementation in this repository remains the reference
strategy and numerical oracle. A separate Rust dry-run and replay engine lives in
[`rust_live/`](rust_live/README.md). It is validated for CASHCAT, calibrates the
same asymmetric Cartea–Jaimungal model directly from the existing Parquet data,
and now contains a stateful pure-Rust Hyperliquid live backend. The tracked
profile keeps `live.enabled=false`; production live additionally enforces the
rolling p95 latency gate. Real-account acceptance code is feature-gated into a
separate binary and is absent from the production image. See
[`rust_live/VALIDATION.md`](rust_live/VALIDATION.md) for the bounded connector
evidence and [`rust_live/PERFORMANCE.md`](rust_live/PERFORMANCE.md) for measured
hot-path and network latency.
