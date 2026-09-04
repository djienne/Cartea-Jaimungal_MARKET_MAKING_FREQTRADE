# Cartea–Jaimungal Market Making on Hyperliquid

A market maker implementing the Cartea–Jaimungal–Penalva model (Chapter 10 of
*Algorithmic and High-Frequency Trading*, 2015) with real-time parameter
estimation, as a **standalone Rust runtime** plus a Python replay and
calibration toolchain. **Works ONLY for Hyperliquid.**

> **The former Freqtrade trader is retired.** It was removed on 2026-08-25 and
> remains available at tag `freqtrade-trader-final`. The current trader is
> `rust_live/`; Python is retained for collection, estimation, replay, and
> independent numerical comparison.

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

**The cadence constraint is gone; the economics did not follow.** The retired
Python loop requoted at a measured p50 of 5.5 s — three to five orders of
magnitude slower than the venue moves — and that was the headline objection to
this stack for most of its life. `rust_live/` removed it: the approved local
benchmark measures a **134 ns batch-mean p99** for the complete hot step (not an
individual-decision tail), against 150 ms of *simulated* one-way latency. Compute
is negligible beside the network. What remains is
network distance and the venue's own limits, and neither is what the model was
losing to. Adverse selection was, and still is: measured, it roughly **doubles**
between a 200 ms and a 5 s markout (ε went 1.98→3.98 bps on the ask, 3.53→6.14
bps on the bid).

**What the live grid has established.** The current grid spec runs against
one shared feed (`docs/DRY_RUN_GRID.md`). Two earlier day-length runs suggested
a broad advantage for wider spread floors, but they predate the corrected queue,
feed-validity, and post-only simulation and are not promotion evidence. Their
40/48/60 bps ordering also reversed between runs. Treat the current leaderboard
as a mutable experiment, not a ranking to copy into a live configuration.

**The latest model-parameter sweep still finds no held-out edge.** The schema-v5
sweep uses a pinned **395.69 h** CASHCAT tape (3,342,794 price rows, 1,648,021 trades)
and selects on the full 276.98-hour train slice (`docs/cashcat_sweep.md`). All
nine Stage-A calibrations lose. The three train-selected finalists earn
+340.99 to +400.20 in sample and lose **13.55 to 17.80 USDC** held out: roughly
+1,100 of net spread is offset by slightly larger directional/adverse loss.
The guard fires on the August 22 cascade in train and nowhere in the held-out
slice, so this tape cannot score its out-of-sample benefit.

A later policy study asks a different question (`docs/cashcat_flatten_fast.md`):
a 60 bps floor is positive in both slices, while an aggressive exit earns less
but ends flat instead of carrying a large residual position. The queue at those
deep quotes is not recorded, so the live grid is testing both hypotheses and
neither is established for live use.

The coupled latency/cadence ladder also does not isolate latency: its 50 ms /
100 ms scenario is +29.52 held out while all four slower/longer-hold scenarios
lose, but both variables change together. Earlier 95-hour and 162-hour ladders
ranked these scenarios differently. The dated predecessors remain as
`docs/cashcat_sweep_20260820_95h.*` and
`docs/cashcat_sweep_20260823_162h_v4.*`; they are comparison points, not current
defaults.

`docs/cashcat_sweep.md` also records a reporting defect: its serialized guard
counters are zero even though the guard ran; P&L and rankings are unaffected.
The period archive (`docs/history/`) attempts a full-search sweep every 21 days
before the 30-day tape deletes the window; see
[`docs/DRY_RUN_GRID.md`](docs/DRY_RUN_GRID.md#the-period-archive--what-outlives-the-tape).

**The cascade is now guarded.** `docs/TOXIC_FLOW_GUARD.md` covers the
toxic-flow guard built against it — a fast mid-move breaker plus VPIN — which on
a frozen replay of the cascade cut the loss 74% (−87.95 → −23.13, re-baselined
in `docs/FLOW_GUARD_CANDIDATES.md`) and bounded ending inventory, while being
bit-identical on a calm control window. It does not predict the crash; it
bounds how much of one gets ridden down, and both A/B variants still lose money. Four
candidate improvements to the guard were then studied on 165 h of frozen tape
(`docs/FLOW_GUARD_CANDIDATES.md`): all four rejected or deferred — including
the counter-intuitive result that a spread gate firing 46 minutes *earlier*
made the crash 3.8x *worse*, because withdrawing quotes freezes inventory
instead of de-risking it.

The practical reading: a short tape can invert this conclusion, so every sweep
must run on the maximum tape available (`docs/DATA_COLLECTION.md` covers how the
tape is produced and how far back it goes).

`docs/replay_acceptance_report.*` is an older 24-minute fail-closed gate smoke
with `ok=false`; it is retained as evidence but must not be mistaken for the
multi-day staged sweeps.

**On 2026-08-23, two minimum-size mainnet sessions were run.**
`docs/live_canary_20260823.md` records what they cost, the three defects they
surfaced, and the venue-reported **address-action budget**—a cumulative lifetime
allowance observed as `10,000 + 1 per USDC traded`. It bound before the
configured WebSocket message-rate limit in that campaign.

`docs/latency_hysteresis_sweep.md` crosses simulated latency (50/100/200/500 ms)
with the requote hold window on one frozen two-hour window. It resolves the
**address-action cost** cleanly — `bps = 4` used about 2.2x fewer actions per fill
than `bps = 2` — but cannot resolve latency economics at roughly 50 fills. The
longer sweeps reordered the machine scenarios while coupling latency to refresh
cadence, so no isolated causal latency ranking is established.

Market data is produced by Docker containers operated from a separate compose
project; their source lives under `scripts/`. See `docs/DATA_COLLECTION.md` for
who owns the tape, why live/grid sessions cannot write it, and how to check it.

**Serious live use requires a host whose measured network latency passes the
configured production gate.** This development machine does not. The Rust quote
loop is already event-driven and its compute cost is negligible beside the
network, but the replay scenarios change latency and refresh together, so they
do not establish that latency alone creates an edge. Nothing here has been
shown to trade profitably in production.

Use this to understand and verify the model. Treat any profit as unproven.

---

## Overview

Three pieces, deliberately separate:

| | what it is | where |
|---|---|---|
| **Trader** | Pure-Rust runtime: calibration, HJB solve, quoting, dry-run simulator, multi-variant grid, and a stateful Hyperliquid live backend | [`rust_live/`](rust_live/README.md) |
| **Measurement** | Event replay harness, staged train/held-out sweeps, κ/ε/λ estimators, market-viability screen | `scripts/` |
| **Data** | Two collectors writing Parquet shards, operated from a *separate* compose project so no trading session can disturb the tape | `docs/DATA_COLLECTION.md` |

The trader and the replay quote from the same arithmetic on purpose. `mm_core.py`
is the single Python implementation the replay imports, and
`rust_live/crates/cj-core` carries the same model in Rust for the live path — so
a backtest simulates the shape of what actually quotes rather than a second guess
at it.

**💰 Support this project**: sign up on
[Hyperliquid with this referral link](https://app.hyperliquid.xyz/join/FREQTRADE)
for a 10% fee reduction.

## The Rust runtime

```bash
cd rust_live

cargo run --locked --release -- --config config/cashcat.toml validate     # config + venue metadata
cargo run --locked --release -- --config config/cashcat.toml calibrate    # κ/λ/ε + HJB surface from Parquet
cargo run --locked --release -- --config config/cashcat.toml replay       # deterministic Parquet replay

# Live public feed, simulated orders. Never reads credentials.
cargo run --locked --release -- --config config/cashcat_dryrun_realistic.toml dry-run

# N parameter sets against ONE shared WebSocket, ranked live.
# Normally run as a container instead: `docker compose up -d` from the repo root.
cargo run --locked --release -- --config config/cashcat_dryrun_realistic.toml dry-run-grid \
    --grid config/grid_cashcat.toml --duration-seconds 0 --out-dir reports/grid_live
```

The grid opens **one** socket regardless of variant count — the venue allows ten
per IP and that budget is shared with the collectors. It never writes Parquet and
never touches credentials. It checkpoints every stats tick and **resumes** on
restart, so a reboot costs a gap rather than the run; past
`--max-carry-inventory-gap-seconds` (900) every position is closed at its last
observed mark rather than marked across a price move nobody saw, and past
`--max-resume-gap-seconds` (3600) the grid starts fresh. Real money is a single explicit config
(`config/cashcat.toml` with `live.enabled = true`), never a grid;
`rust_live/tests/cli_safety.rs` asserts grid mode cannot reach the live backend
even when handed a live-enabled config.

See [`rust_live/VALIDATION.md`](rust_live/VALIDATION.md) for connector evidence
and [`rust_live/PERFORMANCE.md`](rust_live/PERFORMANCE.md) for measured hot-path
and network latency.

## Project Structure

```
Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/
├── rust_live/                             # the trader
│   ├── src/main.rs                        # CLI: validate/calibrate/replay/dry-run/grid/live
│   ├── src/grid.rs                        # variant ranking, equity history
│   ├── config/cashcat.toml                # live config (live.enabled gated)
│   ├── config/grid_cashcat.toml           # dry-run variant specification
│   └── crates/
│       ├── cj-core/                       # HJB, quote policy, instrument math
│       ├── cj-data/                       # Parquet calibration and replay
│       ├── mm-execution/                  # execution traits, dry-run simulator
│       ├── mm-runtime/                    # hot thread, atomics, latency observer
│       └── hyperliquid/                   # signing, transport, state, live backend
├── scripts/                               # measurement + data collection
│   ├── mm_core.py                         # single Python quoting implementation
│   ├── hjb.py                             # symmetric + asymmetric HJB solvers
│   ├── replay_market_maker.py             # event replay (queue, latency, markouts)
│   ├── sweep_replay.py                    # staged train/held-out parameter sweep
│   ├── benchmark_replay.py                # replay throughput, minimum-of-repeats
│   ├── grid_pnl_curve.py                  # P&L curves from a grid run
│   ├── compress_reports.py                # zstd migration for report logs
│   ├── verify_market_viability.py         # can a passive maker profit here at all?
│   ├── estimate_all.py                    # one market-window load per estimator cycle
│   ├── get_{kappa,epsilon,lambda}.py      # κ/λ, ε, raw trade-rate estimators
│   ├── guard_study/                       # frozen-tape studies of guard candidates
│   ├── Dockerfile                         # collector image (built from HYPERLIQUID_DATA)
│   ├── hyperliquid_data_collector.py      # writes Parquet shards
│   ├── run_collector.py
│   └── HL_data/                           # junction -> HYPERLIQUID_DATA/data/eth_mm
├── tests/                                 # pytest: replay, estimators, quoting core
├── docs/                                  # evidence: sweeps, guard, canary, grid, units
└── memory/dry-run-operation.md            # how to actually run it, and what bites
```

**`scripts/` is a live Docker build context.** `HYPERLIQUID_DATA/docker-compose.yml`
builds both collectors from it, copying `hyperliquid_data_collector.py` and
`run_collector.py`. Do not move or rename those, or `scripts/Dockerfile`; the
breakage is silent until the next rebuild.

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
- `κ±`: fill-depth decay rate (higher means fill probability falls faster with
  distance; interpreting it as literal book thickness is only a heuristic)
- `ε±`: adverse-selection jump in the model; estimated from the 200 ms arrival
  markout
- `h(t,q)`: Value function encoding inventory risk preference
- `fees`: maker fee loaded from the live account; replay defaults to the
  0.015% rate measured for the CASHCAT account on 2026-08-23

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

- **Time.** The Rust solver keeps every backward step, and each quote uses the
  episode's actual time-to-go. A perpetual instrument has no terminal time, so
  episodes run for `T` and restart at the horizon or once inventory is genuinely
  flat. Python names this `hjb_time_mode = "episodic"`; its `"stationary"` option
  is retained only for comparisons and reads `t=0`, making `α` nearly inert.
  - **Departure:** the book liquidates the residual at `T` at market and pays
    `α q²`. Every quote here is post-only by construction, so the terminal
    condition acts through the depths alone — it cannot force a taker unwind.
  - At our calibration the terminal effect runs opposite to the naive
    "flatten harder near T" picture: the shipped `φκT = 300` against `ακ = 0.05`, so the running penalty
    — which is what remains to be paid over the time left, hence largest at `t=0`
    and gone at `T` — dominates, and the agent unwinds hardest at the *start* of
    an episode. This matches the book's running-penalty-dominated example; see
    `docs/UNITS.md`.
- **Inventory.** Eq. 10.2 makes `q` a unit-jump count, so `h` exists only at
  integer `q` — but partial fills land in between. Depths are blended linearly
  between the bracketing integers (exact at every integer). Quote diagnostics
  record `q_exact` and `q_rounded`; Python replay also reports the residual
  summary so fractional risk is visible rather than rounded away.

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
- Estimated as the mean arrival jump at a 200 ms horizon, after bad-tick clipping
- Floored at zero because the model assumes `ε ≥ 0`; 1 s and 5 s values are
  diagnostics rather than model inputs
- `κ × ε` is a dimensionless calibration diagnostic. The configured 1.5 ceiling
  is a fail-closed operating rule, not a profitability theorem.

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
| `κ × ε` diagnostic | 0.02 — low relative arrival impact, but no fee information |

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
The dated 2026-08-17 screen recorded in `docs/market_viability_report.json` found
**28 of 60 screened Hyperliquid perps above $0.5M daily volume pinned
at exactly one tick like ETH** — a maker cannot even improve the quote there —
and only 9 that clear the 3 bps round-trip fee *and* are wider than one tick.

**Necessary condition, in plain terms:** the quoted spread must exceed
`2 × maker_fee + adverse selection`. At the 1.5 bps per-side fee measured for
this account, that means a spread wider than ~3 bps before adverse selection is
even considered.

**For calibration monitoring, the operating bands are:**
- `κ × ε < 1`: below the caution band
- `1 ≤ κ × ε < 1.5`: caution band; inspect fit stability and empirical markouts
- `κ × ε ≥ 1.5`: rejected by the shipped fail-closed calibration rule

These bands are not economic verdicts; the empirical viability curve and replay
carry that burden.

**Other conditions:**
- **High λ**: Many market orders → frequent spread capture
- **Low κ**: heavier tail of market-order walk depths → fills remain possible
  farther from the mid
- **Low ε**: Limited informed trading → minimal adverse selection

## Setup

**Prerequisites:** the Rust 1.92 toolchain pinned by
`rust_live/rust-toolchain.toml`, Python 3.10+, Docker for the dry-run grid and
collectors, and Hyperliquid API credentials only for live/account commands.

```bash
pip install -r scripts/requirements.txt
cd rust_live && cargo build --release
```

Market data is produced by containers in a separate project, not by this repo.
Start there first — `docs/DATA_COLLECTION.md` covers who owns the tape and how to
check it is still running.

## Usage

### Market viability — run this first

```bash
python scripts/verify_market_viability.py --crypto ALL --minutes 4320
```

Writes `docs/market_viability_report.json`, exits non-zero when no symbol clears
the bar, and refuses a verdict on less than 6 hours of data — a short window
describes whichever regime it landed in, not the instrument. (CASHCAT was
measured running 19.3× its own daily average volume for 15 minutes; over that
burst the profit curve read +$4,117/h.)

### Calibration and spreads

```bash
python scripts/get_kappa.py            # κ± survival fit, λ± arrival rates
python scripts/get_epsilon.py          # ε± arrival jump, 200 ms primary
python scripts/compute_spreads.py      # refresh κ/ε/λ, print spreads vs inventory
```

### Replay and sweeps

```bash
# Deterministic replay of a Parquet window
python scripts/replay_market_maker.py --data-dir scripts/HL_data --symbol CASHCAT

# Staged train/held-out parameter sweep. Selection runs on the whole train slice
# by default; --search-max-price-events truncates it, and every artifact records
# which tape selection actually ran on.
python scripts/sweep_replay.py --data-dir scripts/HL_data --symbol CASHCAT

# Replay throughput (minimum over repeats, since this host is shared)
python scripts/benchmark_replay.py --data-dir scripts/HL_data --symbol CASHCAT
```

### Reading a grid run

```bash
python scripts/grid_pnl_curve.py --report-dir rust_live/reports/grid_live
```

Prefers `equity_history.csv`, which is append-only across restarts and stamps
`run_started_ms`, so a curve survives a relaunch. Per-variant event logs are
zstd-compressed (~16×); `scripts/compress_reports.py` migrates older plain ones.

### Tests

```bash
python -m pytest -q
cd rust_live && cargo test --workspace
```

## Risk management

Enforced in the Rust runtime, not in a strategy config:

- **Inventory cap.** `q_max` bounds signed inventory in both directions in the
  current runtime. The historical 185-hour spread replay did not enforce the
  prospective cap and ended with a directional position worth 130% of equity;
  that result is therefore not evidence of a deployable market-making book.
- **Liquidation buffer.** A run that breaches it aborts rather than quoting on.
- **Toxic-flow guard.** A fast adverse mid-move breaker plus VPIN withdraws
  quoting. `docs/TOXIC_FLOW_GUARD.md` for what it does;
  `docs/FLOW_GUARD_CANDIDATES.md` for four candidate improvements, all rejected
  or deferred — including the finding that withdrawing *earlier* made a crash
  3.8× worse, because withdrawal freezes inventory instead of de-risking it.
- **Feed validity.** Gaps, downtime fraction and trade lag are measured; a run
  exceeding the thresholds is marked scientifically invalid rather than silently
  trusted.
- **Address-action budget.** The 2026-08-23 mainnet canary observed the venue's
  cumulative address allowance as `10,000 + 1 per USDC traded`; it bound well
  before the configured message-rate limit. Recheck the venue documentation
  before live use; see `docs/live_canary_20260823.md` for the dated evidence.

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in dry-run mode before deploying with real capital. Past performance does not guarantee future results.
ONLY USE IN DRY-RUN

## License


This project implements academic market making models and is intended for research and educational use.
