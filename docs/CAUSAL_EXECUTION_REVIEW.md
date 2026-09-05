# CASHCAT execution validation

## Conclusion

Neither the historical search nor the paper-matched replay establishes a tradable
edge. All three train-selected finalists lose on the Python search's scored
suffix. On a common six-hour Rust replay, exit-adjusted P&L ranges from -1.21 to
+0.49 USDC with only 22-30 fills per account. These are different experiments,
not interchangeable return estimates. No strategy is promoted.

## Historical search (Python causal-v2)

### Method

The frozen tape contains 457.4267 hours, 4,083,323 price rows and 2,127,998 trades
from 2026-08-16 21:57:55.999 to 2026-09-04 23:23:32.030 UTC. Its 20,891 Parquet
files were copied at 2026-09-04 23:23:42 UTC. The chronological 70/30 split is
2026-08-30 06:09:51.220 UTC: 320.1983 training hours and 137.2273 scored hours.
This is reused historical research data, **not an untouched holdout**.

Calibration and VPIN volume scale use training data only. Matching processes
trades, books and decisions chronologically, with activation-time post-only
checks, queue depletion and partial fills without reusing trade volume. The
primary queue model uses visible same-price-level size with zero time decay.
Finite book depth, missing order-level queue information and incomplete venue/
initial-margin constraints remain model limitations.

The staged search fits 81 calibrations, tests 324 risk combinations across three
survivors on the full training slice, and scores three finalists, five execution
scenarios and six-hour windows. It is a staged search, not an exhaustive joint
Cartesian search. Up to four workers were used; all stages completed successfully.
All 81 default-risk Stage-A paths breach maintenance, so their continued replay
rankings are diagnostic shortlists, not executable strategy returns.

### Results

All P&L is marked in USDC from 1,000 USDC initial equity. Finalists are ordered
by training P&L, not selected again on the scored suffix.

| Calibration epsilon + / - (ms) | phi*kappa*T | T (s) | q max | Train P&L | Scored P&L | Maker fills |
|---|---:|---:|---:|---:|---:|---:|
| 500/1000 | 3000 | 150 | 3 | -179.95 | -583.11 | 9,209 |
| 500/1000 | 3000 | 300 | 3 | -203.68 | -753.39 | 10,596 |
| 1000/500 | 3000 | 150 | 6 | -204.10 | -520.97 | 9,372 |

All three use lower fit quantiles 0.75/0.75, alpha*kappa=0.05 and the flow guard.
None breaches maintenance in the scored suffix. The leader has 30 positive
windows among 75 with fills; these reset-account windows span the whole tape,
including training, and are retrospective regime checks rather than independent
holdouts. Its worst window is -75.63 USDC.

For that same leader, the paired latency/refresh scenarios produce -485.51
(50/100 ms), -583.11 (100/250), -627.75 (200/500), -848.95 (500/1000) and
-1,683.57 USDC (500/30000). The last scenario breaches maintenance; its return
is not executable. Changing both timing variables does not isolate latency.

### Fixed controls

Controls use 2,092 base units/order, phi*kappa*T=300 (ceiling 450), alpha*kappa=0.05,
T=150 s, q in [-6,6], 300 ms activation, 100 ms refresh and 150 ms cancellation.
Maker/taker fees are 1.5/4.5 bps, funding is 0.0000125/hour, and aggressive-exit
slippage is 2.5 bps. These fixed controls differ from adaptive Rust grid sizing.

| Control | Scored P&L | Maker fills | Taker fills | Below maintenance |
|---|---:|---:|---:|---|
| baseline / unguarded | -2,649.65 | 21,900 | 0 | yes |
| wide40 | -669.36 | 3,765 | 0 | yes |
| wide60 | +1,008.38 | 1,401 | 0 | no |
| flatten300 | +191.93 | 1,987 | 1,971 | no |
| flatten550 | -267.11 | 1,987 | 1,943 | no |

The guard never triggers on the scored suffix, so the identical A/B result says
nothing about protection during a cascade. Wide60 ends with 8,722 base units:
its gain is not liquidated cash. Moving the flatten target from 301 to 550 ms reverses
the P&L sign. Replay continues accounting after maintenance breaches; fill-count
eligibility is not solvency or promotion eligibility. Reported `net spread` is
quote distance from the decision-time mid, weighted by filled quantity and net
of fees, not post-fill realized spread. Its P&L residual includes adverse
selection, inventory revaluation and funding.

## Paper-execution comparison (Rust causal-v3)

The three saved models score the same frozen tape from September 4, 2026,
17:23:32.030 to 23:23:32.030 UTC. The preceding two hours set order sizes and
VPIN volume scale; model parameters remain the saved training fits. Rust replay
and grid use the same paper step, 297.88 USDC initial equity and common
latency/fees/risk settings. Scoring starts flat with cold guards. This window
is reused research data, not an independent holdout or a rerun of the full search.

| Row | Marked P&L | Exit-adjusted P&L | Maker fills | End inventory (base units) |
|---|---:|---:|---:|---:|
| sweep1 | -1.1980 | -1.2112 | 25 | -15 |
| sweep2 | +0.0301 | -0.1103 | 30 | 160 |
| sweep3 | +0.5000 | +0.4878 | 22 | 14 |

All accounts remain valid through all 62,417 scored events. Exit adjustment uses
the final executable side, 25 bps slippage and 3.5 bps fee; it is a valuation,
not an executed liquidation. Reconstruction from the 77 logged fills recovers
exact inventory/fees and cash within 4.10e-12 USDC, including recorded funding.
Missing queue size is unknown, not zero: at-price fills await visible depth,
while strictly-through prints may fill. Incomplete L2/order-level data remain
limitations; Python and Rust also retain different cadence and latency-tail
assumptions, so matching their P&L is not a validation target.

The release-mode HJB study compares executable, venue-rounded quotes against
both successive timestep refinements. For the three saved fits, five prices
from 0.05 to 0.30, half-unit inventory states and sampled times concentrated near
expiry, 1/512 second is the coarsest common timestep within one price increment
of both finer checks. The 300-second profile needs 153,600 steps. This is a
sampled numerical-convergence result, not an error bound for arbitrary fits.
It changes neither quote cadence nor simulated latency.

## Validation

- 392 Python tests and 281 Rust workspace/all-target/all-feature tests pass,
  plus the separately invoked release-mode numerical study. Clippy with warnings
  denied, formatting and source whitespace checks pass.
- Causality/accounting regressions cover activation, partial volume, funding,
  terminal invalidation, checkpoint rejection, costed gap exits and damaged logs.
  HJB/calibration checks remain; historical P&L is not a reference answer.
- The restart/accounting experiment (`run-1788566042942`) retains 20 valid
  accounts with no event loss across a graceful restart and fresh-book checks.
  All 44 prior-run artifacts remain unchanged; 5,112 logged fills reconstruct
  inventory exactly and cash within 6.99e-12 USDC. This supports execution
  accounting, not strategy profitability.
- The 20-row paper roster includes all three saved models and four targeted
  combinations; operational assumptions are defined in `DRY_RUN_GRID.md`.
  No real-money service or collector is changed.

Detailed search tables and machine-readable scores are in
`cashcat_sweep_causal_20260904.md` and `cashcat_sweep_causal_20260904.json`.
Operational behavior is documented once in `DRY_RUN_GRID.md`. Frozen inputs,
control settings/results and rollout samples are local research artifacts under
`rust_live/reports/causal-study-20260904/`. The six-hour replay, numerical study
and accounting checks are under `rust_live/reports/replay-parity-20260905/`.
These are local research artifacts, not application source.
