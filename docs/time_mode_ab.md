# A/B: stationary vs episodic control

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

Parameters are the live snapshot from `mm:params:CASHCAT` at 2026-08-17T15:25Z
(κ⁺ 10538, κ⁻ 9161, λ⁺ 0.117, λ⁻ 0.103, ε⁺ 2.38e-5, ε⁻ 3.42e-5), with the shipped
sizing (`q_max=6`, `inventory_unit_base=2430`, leverage 2, tick 1e-5).

φ is re-derived at every `T` from the dimensionless `φκT = 10`, exactly as
`mm_core.solve_hjb` does live — φ is **not** κ-invariant (eq. 10.28), and holding
it fixed across a horizon sweep would sweep two things at once. `ακ = 0.05`.

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

The corollary matters more than the number: **`hjb_alpha_kappa` is now a live
knob and it is untuned.** It was set to 0.05 while α was structurally inert, so
that value carries no evidence. Raising it is what would make the time axis do
visible work, and that is the sweep worth running next.

**2. Every configuration loses money, and longer horizons lose more**
(−42 → −100 as T goes 150 s → 1800 s, with fills rising 1171 → 1638). Consistent
with the earlier finding that the loss is inventory risk rather than mispricing:
a longer horizon tolerates inventory for longer, takes more fills, and pays for
them. `T = 150 s` remains the best of these in both modes.

**3. Mean `|q_residual|` sits at 0.22–0.26 in every run.** The average position is
about a quarter of an inventory unit off the integer grid — real risk that the
model was not pricing and that nothing in the codebase could previously see,
because `q` alone looks perfectly healthy. Below the 0.35 gate threshold, but far
enough from zero to confirm the partial-fill gap was worth closing.
