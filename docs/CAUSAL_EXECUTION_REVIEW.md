# CASHCAT execution validation

## Conclusion

The causal-v2 replay does not establish a tradable edge. All three train-selected
finalists lose on the scored suffix, and the selected configuration loses in all
five latency/cadence scenarios. Positive wide-quote and fast-flatten controls are
sensitive to inventory exposure and exit timing. No strategy is promoted.

## Method

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

## Results

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

## Validation

- 390 Python tests and 278 Rust workspace/all-target/all-feature tests pass;
  Clippy with warnings denied, formatting and whitespace checks pass.
- Causality/accounting regressions cover activation, partial volume, funding,
  terminal invalidation, checkpoint rejection, costed gap exits and damaged logs.
  HJB/calibration checks remain; historical P&L is not a reference answer.
- The execution-validation grid `run-1788566042942` completes 30 minutes before and after a
  graceful restart, retaining all 20 valid accounts with no event loss. All 44
  prior-run artifacts remain unchanged. Reconstructing 5,112 logged fills gives
  exact inventory and a maximum cash residual of 6.99e-12 USDC across all 20 rows.
- An additional 30-minute observation verifies the fresh-book guard with all
  20 accounts valid and no event loss. The current paper roster includes the
  three saved sweep models and targeted combinations under common dry-run
  conditions, as described in `DRY_RUN_GRID.md`. No real-money service or
  collector is changed.

Detailed search tables and machine-readable scores are in
`cashcat_sweep_causal_20260904.md` and `cashcat_sweep_causal_20260904.json`.
Operational behavior is documented once in `DRY_RUN_GRID.md`. Frozen inputs,
control settings/results and rollout samples are local research artifacts under
`rust_live/reports/causal-study-20260904/`, not application source.
