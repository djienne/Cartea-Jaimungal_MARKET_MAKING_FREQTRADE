# A/B: stationary vs episodic control

> **Historical experiment (2026-08-17).** The Redis snapshot and configuration
> language below describe the system at the time. The current Rust runtime reads
> an episodic surface directly from its in-process calibration; the Python
> `stationary` switch remains only for controlled replay comparisons.

**What this measures:** whether reading the book's `δ*(t,q)` at the episode's real
time-to-go, instead of always reading the `t=0` slice, changes anything.

**This is a fidelity change, not a profitability change.** The result is recorded
whichever way it went, and a worse number would not have justified reverting to
the stationary approximation.

## Method

One pinned CASHCAT tape — **17.46 h, 106 573 price rows, 42 009 trades** — loaded
**once**, with `load_symbol_data` monkeypatched to hand every variant the same
bytes. This matters: the collector keeps writing shards, and letting `run_replay`
load per-variant scored different data each time. That invalidated an earlier φ
sweep, where one setting came back at both −3.98 and −19.39.

Parameters are the then-live snapshot from `mm:params:CASHCAT` at 2026-08-17T15:25Z
(κ⁺ 10538, κ⁻ 9161, λ⁺ 0.117, λ⁻ 0.103, ε⁺ 2.38e-5, ε⁻ 3.42e-5), with the shipped
sizing (`q_max=6`, `inventory_unit_base=2430`, leverage 2, tick 1e-5).

φ is re-derived at every `T` from the dimensionless `φκT = 10`, exactly as
`mm_core.solve_hjb` does in replay — φ is **not** κ-invariant (eq. 10.28), and holding
it fixed across a horizon sweep would sweep two things at once. `ακ = 0.05`.

(`φκT = 10` was the deployed value when this A/B was run. The current Rust
profile uses 300 with a ceiling of 450. The numbers below are the A/B as run and
have not been re-measured at 300.)

Reproduce with `scripts/replay_market_maker.py --hjb-time-mode {stationary,episodic}`.

## Result

| T | mode | PnL (USDC) | maker fills | mean \|q_residual\| | episodes |
|---:|---|---:|---:|---:|---:|
| 150 s | stationary | −42.38 | 1171 | 0.217 | — |
| 150 s | **episodic** | −45.98 | 1186 | 0.233 | 1274 |
| 300 s | stationary | −69.97 | 1344 | 0.250 | — |
| 300 s | **episodic** | −65.16 | 1366 | 0.239 | 580 |
| 600 s | stationary | −73.54 | 1476 | 0.256 | — |
| 600 s | **episodic** | −70.67 | 1493 | 0.250 | 290 |
| 1800 s | stationary | −100.00 | 1638 | 0.237 | — |
| 1800 s | **episodic** | −103.75 | 1646 | 0.242 | 90 |

Quote attempts were identical (56 356) in all eight runs, as expected — the time
mode changes *where* quotes sit, not whether the agent quotes.

## Reading

**1. Episodic vs stationary is noise here: ±4 USDC on a −42 to −104 base, and the
sign flips twice.** That is not a disappointment, it is the prediction. At this
calibration `φκT = 10` against `ακ = 0.05` — the running penalty is **200× the
terminal one**. The running penalty is what remains to be paid over the time
left, so it dominates the whole surface and the terminal condition barely bends
it. Moving along the time axis therefore moves the depths very little.

(Note: an earlier version of this file said this makes depths tighten at the
start of an episode "the opposite of the book's figures". That was wrong — the
book's Fig. 10.8 runs `phi*kappa*T = 60` against `alpha*kappa = 0.01`, so it is
in the same running-penalty-dominated regime and shows the same shape. Corrected
2026-08-17 against the book PDF.)

The corollary matters more than the number: **`alpha_kappa` remains untuned.** A
later 2026-09-02 check found that at the current `φκT=300` its influence is
confined to roughly the last 8.75 seconds of a 150-second episode, and Stage B
returned identical P&L at `alpha_kappa=0.05`, `0.5`, and `5.0`. Revisit it only
with a lower running penalty where the terminal layer is actually observable.

**2. Every configuration loses money, and longer horizons lose more**
(−42 → −100 as T goes 150 s → 1800 s, with fills rising 1171 → 1638). Consistent
with the earlier finding that the loss is inventory risk rather than mispricing:
a longer horizon tolerates inventory for longer, takes more fills, and pays for
them. `T = 150 s` remains the best of these in both modes.

**3. Mean `|q_residual|` sits at 0.22–0.26 in every run.** The average position is
about a quarter of an inventory unit off the integer grid — real risk that the
model was not pricing and that nothing in the codebase could previously see,
because `q` alone looks perfectly healthy. This is a diagnostic, not a current
acceptance threshold, but it is far enough from zero to confirm the partial-fill
gap was worth closing.
