# Fast inventory exit on CASHCAT — what survived review

> **Current conclusion.** A 60 bps half-spread floor, not aggressive flattening,
> is what makes the replay positive. Crossing out inventory quickly earns less
> but removes the large residual position and long holding time. Both results
> remain replay hypotheses because the queue at a 60 bps quote is not observed.

## Final comparison

The experiment uses the 395.69-hour schema-v5 tape split into the 118.7-hour
held-out slice first studied here and the previously untouched 277-hour train slice.
The compared policies use `q_max=6` and the same calibration/model settings:

- **hold, floor 60:** `min_half_spread_bps=60`, passive offsetting fills;
- **flat0:** the same wide quote, then cross out an open lot as soon as the
  simulated round trip permits, charging a 4.5 bps taker fee and measured
  2.5 bps walking cost.

| tape | policy | P&L | net spread | directional | final inventory | median hold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| held-out | **hold, floor 60** | **+417.79** | +957.3 | -539.5 | **9,735** | **42 min** |
| held-out | flat0 | +258.04 | +1049.3 | -791.3 | 0 | 0.3 s |
| train | **hold, floor 60** | **+1337.58** | +2715.1 | -1377.5 | **6,380** | 9 min |
| train | flat0 | +438.90 | +2361.1 | -1922.2 | 0 | 0.3 s |

The wide hold wins on P&L in both slices, and its directional component is
negative, so the gain is not a favorable ride on the tape. It is also highly
leveraged: held-out inventory finishes near the `q_max` ceiling on a 1,000 USDC
account. The flatten gives up P&L to end flat with a 0.3-second median hold.
These are different risk profiles; neither table alone establishes live edge.

## What the flatten tests

Fill-conditional adverse selection grows from 12.84 bps at 200 ms to 29.77 bps
at the model-parameter sweep's 6.5-second median hold
(`cashcat_epsilon_conditional.md`). A fast taker exit truncates that exposure but
pays the crossed half-spread, book walking, and taker fee. At a 250 ms simulated
round trip, three queue thresholds spanning an order of magnitude produced:

| queue threshold | held-out P&L | train P&L |
| --- | ---: | ---: |
| no queue (`touch_only`) | +203 | +298 |
| deepest visible level (`book_level`) | +254 | +425 |
| cumulative visible depth (`book_depth`) | +243 | +207 |

This is useful sensitivity, not queue validation. The 20-level snapshots reach
60 bps on only 0.9% of observations, so neither book-based threshold measures
the queue at the quote. Deep sweeps print a median 31,000 units at or beyond 60
bps, making the true time-priority penalty unobservable from this tape.

Across the held-out latency ladder, breakeven is roughly 450--520 ms depending
on the queue threshold; all three thresholds are positive at 250 ms. The exit
uses the first BBO at or after its deadline and charges latency on the exit path,
so it has no price lookahead. A passive-first variant was worse on both slices.

## Corrections incorporated

- A stale-`q` bug priced a new quote from inventory held before the flatten.
  Fixing it moved held-out 250 ms P&L from +253.92 to +258.04; the bug was a drag,
  not the source of the sign.
- The first queue interpretation called `book_level` the queue at our price. It
  is only the deepest *visible* level, usually about 37 bps out. The table above
  is therefore a threshold sensitivity check, not three physical queue models.
- The original explanation attributed the result to flattening keeping `q≈0`.
  The hold control is also flat on 98--99% of events. The binding change is the
  60 bps floor on the inventory-reducing branch; flattening changes the risk and
  holding period, not the existence of the wide-spread replay edge.

## What can and cannot be claimed

The replay supports two statements on this tape: a 60 bps floor is positive in
both slices under the simulator, and a fast taker exit trades some of that P&L
for much lower inventory exposure. It does **not** show that either policy is
profitable live. The queue at the quote is missing, the same held-out slice has
been examined repeatedly, and aggressive exits roughly double address actions
on an account with little remaining allowance.

The live grid therefore compares `wide60` with `flatten300` and `flatten550`.
Those names are effective exit ages: the configured 1/250 ms deadlines also pay
150 ms decision plus 150 ms acknowledgement latency. The flatten rows are
dry-run-only and ineligible for live promotion because the live backend has no
equivalent exit policy.
