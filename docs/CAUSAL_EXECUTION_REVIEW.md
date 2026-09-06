# CASHCAT execution validation

## Conclusion

The full staged search finds no profitable fill-qualified training candidate.
The three selected fits lose on the scored Python suffix and in the matched
Rust paper model. Positive simulated returns come from short lot-age exits,
not from the unmodified finalists. These are hypotheses for prospective paper
observation, not a demonstrated tradable edge or permission for live trading.

## Dataset and search

The frozen tape spans August 16, 2026, 21:57:55.999 through September 5, 2026,
11:09:36.518 UTC: 469.1946 hours, 4,196,666 price rows and 2,179,254 deduplicated
trades. All 17,615 copied Parquet files pass validation. The split is August 30,
2026, 14:24:06.362 UTC: 328.4362 training hours and 140.7581 scored hours. This is
reused research data, not an untouched holdout. Collection continues separately.

The full calibration grid crosses 200/500/1000 ms arrival-jump horizons and
0/0.5/0.75 lower depth quantiles independently on both sides: 81 fitted models.
The three surviving calibrations each score 108 risk settings, for 324 Stage B
runs: nine phi*kappa*T values, three horizons, two inventory bounds and the
on/off flow guard. Alpha*kappa stays at 0.05. No price-event cap or shortened
training tape is used. This is the complete configured staged sweep, not an
exhaustive crossing of every calibration with every risk setting.

Python causal-v3 uses 1,000 USDC initial equity, 2,092 base units/order, leverage
one, 1.5/4.5 bps maker/taker fees and the search scenario's 100 ms activation and
250 ms refresh. It is a shortlisting experiment, not the paper execution model.

| Finalist | Arrival horizons (+/- ms) | q max | Training P&L | Scored P&L | Positive six-hour windows |
|---|---|---:|---:|---:|---:|
| `sweep1` | 1000/1000 | 3 | -186.44 | -505.81 | 29/77 |
| `sweep2` | 1000/500 | 6 | -187.28 | -516.26 | 32/77 |
| `sweep3` | 500/1000 | 6 | -192.09 | -583.68 | 30/77 |

All three use phi*kappa*T=3000, alpha*kappa=0.05, T=150 s and the flow guard.
The regime check scores 79 chronological windows across the whole tape,
including training; 77 contain fills. These are not independent replications.

All 81 Stage A reference accounts breach maintenance, so their continued
accounting paths are not executable returns. The selected Stage B rows and
three scored suffixes do not breach maintenance. Fill-count eligibility alone
is not solvency. The first finalist loses in all five latency/refresh scenarios;
the 500/30000 ms scenario loses 1,711.68 USDC and breaches maintenance. Changing
both timing variables does not isolate latency.

## Paper-model comparison

Rust causal-v4 scores August 30, 2026, 14:24:06.362 through September 5, 2026,
11:09:28.590 UTC, the last common complete BBO/trade endpoint. The preceding
training prefix sets capital-derived order sizes and VPIN scale; model fits
remain fixed. Accounts start flat with 297.88 USDC and cold guards. Replay and
grid share the paper step, latency tail, fees, funding and terminal risk gates.
Their common settings are defined once in `DRY_RUN_GRID.md`.

The unmodified finalists lose 147.78, 81.91 and 79.82 USDC respectively over the
full suffix. The positive, fully scored, flat-ending variants are:

| Row | Marked and exit-adjusted P&L | Maker fills | Lot-age exits |
|---|---:|---:|---:|
| `sweep1_flat300` | +190.70 | 718 | 718 |
| `sweep1_flat550` | +117.60 | 718 | 718 |
| `contender_flat300` | +96.61 | 942 | 942 |
| `contender_flat550` | +44.99 | 946 | 946 |

The contenders freeze the full-support 200/200 ms training fit. Their recent-fit
counterparts remain separate controls. Both pairs lose substantial P&L when the
exit target moves from 301 to 550 ms. The choice of these controls follows
inspection of reused data; prospective observation is required. Guarded and
unguarded first-finalist replays are identical on this suffix, not evidence
that protection during a cascade is unnecessary.

Widening alone is not a winner: the fixed-control wide60 replay stops at 69.49%
of the suffix and the first-finalist wide60 combination at 51.25%, both on the
liquidation-buffer gate. Their terminal losses are not full-period returns.
Python's fixed-control wide60 gain of 870.04 USDC includes 12,493 open base units
and does not transfer to the paper model. Python's flatten300/550 controls return
+226.62/-256.49 USDC under different sizing and execution assumptions.

Queue position is only inferred from finite visible depth; venue order priority,
market impact and aggressive exit liquidity are not known. Quote distance from
the decision-time mid, net of fees, is not realized post-fill spread. Its P&L
residual includes adverse selection, inventory revaluation and funding. Positive
paper-model returns are therefore neither executable guarantees nor validation
against real fills.

## Numerical and accounting checks

The shipped HJB timestep is 1/512 s and Newton residual tolerance is 1e-10.
For the three finalists and shared contender quote model, executable quotes at
five prices (0.05--0.30), half-unit inventory states and sampled times including
the last five seconds differ by at most one venue increment against both further
halvings. This is sampled convergence, not an error bound for arbitrary fits.
The check tests the configured timestep directly: at tolerance 1e-8, the
contender differed by four increments despite passing a coarser comparison.
Tighter Newton solves remove that non-monotone discrepancy without a solver
rewrite. All paper results here are rerun at the shipped tolerance.

All terminal accounts are finite. Cash plus signed inventory valued at the last
logged BBO reproduces reported equity within 5.69e-14 USDC. Exit-adjusted P&L
uses the executable side, 25 bps slippage and 3.5 bps fees; this is a valuation,
not an executed liquidation. Bounded logs do not retain every historical fill.
A native/container duplicate replay matches account equity, inventory and fill
count; final replays use the native release build on the same frozen inputs.

285 Rust tests, the separately run release-mode convergence study, 392 Python
tests, Clippy, formatting and whitespace checks pass. The TeX check confirms
12 source snippets and 55 recomputed numbers; no mathematical exposition change
is needed. These checks support numerical/accounting consistency, not an edge.

## Artifacts and paper roster

`cashcat_sweep.json` and `cashcat_sweep.md` are the canonical complete search
and paper-comparison records. The 22-row paper roster contains the three exact
finalist fits, four targeted first-finalist combinations, two fixed-fit flatten
contenders and 13 recent-fit controls. Fixed-fit and lot-age-exit rows remain
ineligible for live promotion. Roster changes start a separate paper run rather
than rewriting prior accounts; operation and log bounds stay in `DRY_RUN_GRID.md`.

Frozen inputs, invocation, control comparisons, numerical studies and bounded
replay logs are local research artifacts under
`rust_live/reports/full-sweep-20260905/`, not application source. No collector
or real-money service is changed.
