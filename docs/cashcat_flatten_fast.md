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

## At 200 ms — the realistic operating point

`mid` scenario (200 ms latency, 500 ms refresh), held-out slice. `flatten@0ms` means
flatten as soon as possible; the exit still lands 200 ms later because the deadline
carries the round trip.

| phi*kappa*T | hold | **flatten@0ms** | @100ms | @200ms | @500ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | -294.87 | **+218.34** | +152.65 | +89.35 | -236.80 |
| 1000 | -182.60 | **+256.58** | +205.73 | +139.35 | -96.30 |
| 3000 | -111.74 | **+227.45** | +191.27 | +135.44 | -34.38 |

Positive at every phi, and the ordering barely depends on phi -- which is itself a
result, since phi was the axis that could never find an optimum. Flattening sooner is
always better, so the policy is "exit as fast as the venue allows", not a tuned deadline.

## Why the per-fill number is so large, and the caveat that limits it

| | mean fill depth | maker fills | per maker fill |
| :--- | ---: | ---: | ---: |
| hold | 31.2 / 23.4 bps | 1978 | -4.35 bps |
| flatten@0 | **60.8 / 57.4 bps** | 683 | **+17.69 bps** |

Flattening keeps inventory near zero, so `q ~ 0` and the HJB quotes near its 80 bps cap
instead of tightening one side to unwind. The economics are consistent at that depth --
capture ~60 bps against a ~24 bps markout at 200 ms -- but it means **the whole result
rests on getting filled at ~60 bps**.

**And that is where the replay is most optimistic.** `queue_ahead` is set to 0 for any
quote that does not join the best (`replay_market_maker.py:2124-2130`), and a 60 bps
quote never joins the best against a 5.6 bps median half-spread. So the simulator fills
us on the first print that reaches our price, with no time priority. Measured from the
20-level snapshots, the size resting at-or-better at 60 bps out is a median of **51,124
units (ask) / 46,061 (bid) -- 22-24x our 2092-unit lot**.

Two consequences, and the second is the one that matters:

1. **The fill COUNT is an upper bound.** 683 maker fills against ~916 sweeps that reached
   60 bps in the same window means we are capturing most deep sweeps, which only happens
   with no queue ahead of us.
2. **The per-fill economics would degrade too, not just the count.** With real time
   priority we are filled only by sweeps large enough to clear the queue -- and those are
   the sweeps with the worst markout. It is the same selection effect that makes
   `eps(d)` conditional in the first place, one level deeper.

So this is not yet a number to trade on. A queue-position model at depth is the single
modelling gap between this result and a trustworthy one, and it is the obvious next step.

## The latency threshold

The question this whole line of work exists to answer: **how fast does the round trip
have to be for this to make money?** Held-out slice, `phi*kappa*T = 1000`, flatten as
soon as possible, run under BOTH queue models so the answer is bracketed rather than
asserted.

| round trip | `touch_only` P&L | fills | `book_depth` P&L | fills |
| ---: | ---: | ---: | ---: | ---: |
| 50 ms | +277.49 | 781 | +294.76 | 433 |
| 100 ms | +289.81 | 712 | +282.39 | 388 |
| 150 ms | +236.54 | 631 | +280.08 | 343 |
| 200 ms | +254.10 | 632 | +281.92 | 349 |
| **250 ms** | **+203.14** | 638 | **+242.54** | 324 |
| 300 ms | +170.33 | 591 | +174.39 | 278 |
| 400 ms | +80.54 | 628 | +113.74 | 299 |
| 450 ms | +36.35 | 601 | +49.84 | 287 |
| **500 ms** | **-3.83** | 615 | **+20.61** | 284 |
| 550 ms | -35.06 | 613 | -24.76 | 282 |
| 700 ms | -135.37 | 657 | -124.31 | 274 |

**Breakeven is ~500 ms** under both models. At 250 ms the result is +203 to +243, so a
sub-250 ms target carries roughly a **2x margin** over breakeven rather than sitting on
the edge of it.

### The queue model barely moves the answer, which is the reassuring part

`book_depth` makes fills much harder -- it counts the ~51k units resting at or better
than a 60 bps quote, cutting fill counts by ~45% -- and the P&L is **unchanged or
slightly better** at every latency. I expected the opposite: that real time priority
would leave us filled only by the largest, most toxic sweeps and degrade the per-fill
economics. It does not. The fills the queue removes are worth less than average, so the
result survives the assumption it most depended on.

That is the difference between an upper bound and a finding. The number is not an
artefact of an empty queue.

### What can still be said, and what cannot

**Can be said:** on 118.7 h of held-out CASHCAT, a Cartea-Jaimungal maker that flattens
inventory immediately rather than waiting for an offsetting maker fill is profitable in
replay at round trips under ~500 ms, with a comfortable margin at 250 ms, and the result
holds under a conservative queue model, a measured book-walking cost and latency charged
on both the quoting and the exit path.

**Cannot be said:** that it is profitable live. This is one instrument, one tape, and one
held-out slice that many analyses have now touched -- the multiple-comparison risk is
real and a fresh window is the first thing to check. The policy always crosses, where a
passive-first exit with a crossing deadline should dominate and is untested. And crossing
doubles actions per round trip against a near-exhausted lifetime budget.

The honest next step is the dry-run grid, which costs nothing and answers the
reproducibility question directly.

## Correction: the queue model had the wrong pairing

The first `book_depth` model charged the **cumulative** size at our level or better
(~51k units at 60 bps out). But `scan_for_matching_trade` decrements the queue only on
prints at our price **or beyond** (`candidate_price >= price` for an ask) -- the volume
that clears the *better* levels arrives as prints BELOW our price and never decrements
anything. So the queue was charged against a stream that can only ever be a fraction of
it: an **~19x over-penalty** (51,124 cumulative against 2,652 at our own level).

`queue_model = "book_level"` is the correct pairing: a sweep that reaches us has already
cleared the better levels by definition, so what stands between us and the fill is the
size queued at our own price -- median 2,652 units against our 2,092-unit lot, so still
a real queue. `book_depth` is retained deliberately as a stress bound.

| round trip | `book_level` (the model) | fills | `book_depth` (stress bound) | fills |
| ---: | ---: | ---: | ---: | ---: |
| 100 ms | +379.38 | 670 | +282.39 | 388 |
| 200 ms | +315.58 | 598 | +281.92 | 349 |
| **250 ms** | **+253.92** | 581 | **+242.54** | 324 |
| 300 ms | +201.49 | 519 | +174.39 | 278 |
| 400 ms | +93.37 | 562 | +113.74 | 299 |
| 500 ms | -36.63 | 542 | +20.61 | 284 |
| 600 ms | -110.58 | 558 | -49.74 | 291 |

**Breakeven is ~450 ms (correct model) to ~520 ms (stress bound), and 250 ms returns
+243 to +254 either way.** The conclusion did not move -- the earlier table was right
for the wrong reason, having survived a penalty 19x harsher than reality.

The three models now bracket the answer from both sides: `touch_only` assumes no queue
at all, `book_level` is the honest one, `book_depth` is ~19x too harsh. All three are
positive at 250 ms, which is a stronger statement than any one of them alone.
