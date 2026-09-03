# Truncating the holding period — CASHCAT

The first positive held-out P&L in this project, and the conditions it needs.

## The idea

`eps(d)` — the fill-conditional adverse selection — grows steeply with the markout
horizon: 12.84 bps at 200 ms against 29.77 bps at 6.6 s (`cashcat_epsilon_conditional.md`).
The bot holds inventory for a **6.5 s median** because it waits for an offsetting MAKER
fill, so it eats the entire accrual. That, not the quoting model, is why it loses.

Truncate the hold and you truncate the adverse selection. The cost of truncating is the
half-spread crossed plus the taker fee. At the depths this bot quotes, the captured
spread is large enough to cover that — if you can act quickly enough.

## Result (held-out slice, 118.7 h, eps500/klo0.75, phi*kappa*T = 300, q_max = 6)

| scenario | latency | hold (shipped) | flatten@200ms | @500ms | @1000ms |
| :--- | ---: | ---: | ---: | ---: | ---: |
| colocated | 50 ms | -101.59 | **+371.99** | +49.94 | -139.95 |
| good | 100 ms | -137.14 | **+222.40** | -66.73 | -224.99 |
| this_stack | 500 ms | -384.65 | **-276.60** | -329.02 | -348.93 |

The mechanism is visible in the decomposition: at `good`/200 ms the directional leg
improves from -1751.7 to -1050.1 while spread capture barely moves (1614.5 -> 1272.5).
Median hold falls from 6.48 s to 0.35 s. That is the markout being cut off, exactly as
predicted.

## What it costs, and why both charges matter

**Book walking is charged.** Our lot is 2092 base units against a median touch of 733
(bid) / 854 (ask) — 2.5-2.9x — and exceeds the touch on 81-88% of events, so a flatten
does not get the touch price. Measured from the 20-level snapshots, the volume-weighted
fill for our lot sits **1.6-2.1 bps (median), 2.3-2.8 (mean)** past the touch;
`flatten_slippage_bps` defaults to 2.5 and is charged on every flatten.

**Latency is charged on the exit too.** The flatten deadline carries the same round trip
the quoting path pays: we notice the position, decide, and the taker order arrives one
latency later. An earlier version executed at the deadline price and returned +372 under
`good`; charging the round trip cuts that to +222. Executing at a price you could not
have traded on is the easiest way to manufacture an edge that does not exist.

## The conclusion, which is a latency conclusion

**The effect is real and large — and it is entirely a latency game.** Every scenario
improves on holding, but only sub-100 ms round trips make it positive. At the latency
this stack actually runs at (`docs/DRY_RUN_GRID.md` records venue RTT p50 281 ms, p95
661 ms), the policy still beats holding by ~108 but remains firmly negative.

So the open question is no longer "which model or parameter" — phi, epsilon and L1 OBI
are all closed — but **"can this stack reach sub-100 ms round trips to Hyperliquid"**.
That is an infrastructure question, and it is answerable before any more modelling work.

## Caveats

- One tape, one instrument, and the same held-out slice many analyses have now touched.
  Needs a fresh window before anyone trades on it.
- The flatten always crosses. A smarter policy would first try to exit passively and only
  cross on a deadline, which should dominate this and is untested.
- Crossing doubles actions per round trip, and the Hyperliquid action budget is a
  near-exhausted lifetime allowance, so this is replay/dry-run only for now.
