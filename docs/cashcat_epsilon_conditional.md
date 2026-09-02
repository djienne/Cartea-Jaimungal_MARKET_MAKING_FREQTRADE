# Fill-conditional adverse selection — CASHCAT

`E[mid jump | the sweep reached depth d]`, the quantity the fill term of the
Cartea-Jaimungal DPE actually needs. The shipped estimator computes the `d = 0`
row of this table and uses it at every depth.

- window: `2026-08-28T10:56:50Z` → `2026-09-02T09:39:00Z` (118.7 h)
- market orders: 223850
- maker fee assumed: 1.5 bps

Per-fill maker edge at depth `d` is `(1 - b) * d - a - fee`, so `b` decides
whether ANY depth is profitable. Read the CI on `b`, not the point estimate:
at `b = 1` the optimal depth is degenerate.

## Markout horizon 200 ms

| reach d (bps) | E[jump given reach >= d] | N | usable |
| ---: | ---: | ---: | :--- |
| 5 | 2.93 | 101850 | True |
| 10 | 5.29 | 43864 | True |
| 15 | 7.99 | 21570 | True |
| 20 | 10.55 | 12298 | True |
| 25 | 12.84 | 7608 | True |
| 30 | 14.94 | 5053 | True |
| 35 | 16.77 | 3512 | True |
| 40 | 18.28 | 2569 | True |
| 45 | 19.76 | 1929 | True |
| 50 | 21.22 | 1474 | True |
| 55 | 22.28 | 1157 | True |
| 60 | 23.87 | 916 | True |

| side | unconditional | a (bps) | b | 95% CI on b | makeable | edge@26bps |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| `epsilon+` | 1.89 | +2.32 | 0.354 | [0.319, 0.390] | True | +12.98 |
| `epsilon-` | 2.43 | +2.77 | 0.394 | [0.348, 0.435] | True | +11.49 |
| `pooled` | 2.13 | +2.45 | 0.378 | [0.353, 0.406] | True | +12.23 |

## Markout horizon 1000 ms

| reach d (bps) | E[jump given reach >= d] | N | usable |
| ---: | ---: | ---: | :--- |
| 5 | 4.78 | 101850 | True |
| 10 | 8.36 | 43864 | True |
| 15 | 13.05 | 21570 | True |
| 20 | 18.26 | 12298 | True |
| 25 | 23.39 | 7608 | True |
| 30 | 28.17 | 5053 | True |
| 35 | 32.73 | 3512 | True |
| 40 | 37.00 | 2569 | True |
| 45 | 40.76 | 1929 | True |
| 50 | 44.65 | 1474 | True |
| 55 | 47.85 | 1157 | True |
| 60 | 51.52 | 916 | True |

| side | unconditional | a (bps) | b | 95% CI on b | makeable | edge@26bps |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| `epsilon+` | 3.13 | +0.27 | 0.829 | [0.779, 0.884] | True | +2.68 |
| `epsilon-` | 3.91 | +1.75 | 0.903 | [0.843, 0.964] | True | -0.73 |
| `pooled` | 3.48 | +0.83 | 0.873 | [0.835, 0.912] | True | +0.96 |

## Markout horizon 5000 ms

| reach d (bps) | E[jump given reach >= d] | N | usable |
| ---: | ---: | ---: | :--- |
| 5 | 5.74 | 101843 | True |
| 10 | 10.22 | 43859 | True |
| 15 | 15.92 | 21567 | True |
| 20 | 22.32 | 12297 | True |
| 25 | 28.46 | 7607 | True |
| 30 | 34.33 | 5053 | True |
| 35 | 40.22 | 3512 | True |
| 40 | 44.98 | 2569 | True |
| 45 | 49.37 | 1929 | True |
| 50 | 54.17 | 1474 | True |
| 55 | 57.55 | 1157 | True |
| 60 | 61.94 | 916 | True |

| side | unconditional | a (bps) | b | 95% CI on b | makeable | edge@26bps |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| `epsilon+` | 3.94 | +0.86 | 1.031 | [0.953, 1.108] | False | -3.17 |
| `epsilon-` | 4.51 | +1.86 | 1.064 | [0.964, 1.149] | False | -5.03 |
| `pooled` | 4.19 | +1.28 | 1.051 | [0.997, 1.104] | False | -4.10 |

## Markout horizon 6600 ms

| reach d (bps) | E[jump given reach >= d] | N | usable |
| ---: | ---: | ---: | :--- |
| 5 | 5.78 | 101841 | True |
| 10 | 10.37 | 43857 | True |
| 15 | 15.96 | 21567 | True |
| 20 | 22.25 | 12297 | True |
| 25 | 28.35 | 7607 | True |
| 30 | 34.50 | 5053 | True |
| 35 | 40.26 | 3512 | True |
| 40 | 45.09 | 2569 | True |
| 45 | 49.51 | 1929 | True |
| 50 | 54.45 | 1474 | True |
| 55 | 57.42 | 1157 | True |
| 60 | 61.80 | 916 | True |

| side | unconditional | a (bps) | b | 95% CI on b | makeable | edge@26bps |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| `epsilon+` | 4.06 | +1.09 | 1.042 | [0.965, 1.116] | False | -3.68 |
| `epsilon-` | 4.42 | +1.67 | 1.056 | [0.953, 1.144] | False | -4.63 |
| `pooled` | 4.22 | +1.36 | 1.050 | [0.987, 1.124] | False | -4.15 |

## Markout horizon 30000 ms

| reach d (bps) | E[jump given reach >= d] | N | usable |
| ---: | ---: | ---: | :--- |
| 5 | 6.17 | 101826 | True |
| 10 | 10.83 | 43849 | True |
| 15 | 15.96 | 21563 | True |
| 20 | 22.00 | 12294 | True |
| 25 | 27.73 | 7605 | True |
| 30 | 33.94 | 5051 | True |
| 35 | 40.33 | 3511 | True |
| 40 | 43.80 | 2568 | True |
| 45 | 47.99 | 1928 | True |
| 50 | 52.82 | 1473 | True |
| 55 | 54.89 | 1156 | True |
| 60 | 58.32 | 916 | True |

| side | unconditional | a (bps) | b | 95% CI on b | makeable | edge@26bps |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| `epsilon+` | 4.99 | +3.51 | 0.995 | [0.882, 1.113] | False | -4.88 |
| `epsilon-` | 4.03 | +1.03 | 0.993 | [0.861, 1.132] | False | -2.36 |
| `pooled` | 4.56 | +2.46 | 0.988 | [0.905, 1.094] | False | -3.64 |

## Stage 1: the horizon, measured rather than assumed

`phi_kappa_t` taught us not to trust a number whose units nobody checked, so the markout
horizon here is derived from how long the bot actually holds inventory, not picked.
`ReplayMetrics` now FIFO-matches each fill against its offsetting fill
(`match_holding_time`), so a holding time is the life of a real exposure rather than the
gap between consecutive fills. Held-out slice, `eps500/klo0.75`, `q_max=6`, `T=150`:

| phi*kappa*T | fills | mean hold | **median hold** | p90 hold | still open at end |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 3907 | 112.0 s | **6.60 s** | 220.2 s | 50 |
| 300 (shipped) | 3379 | 121.1 s | **6.48 s** | 229.2 s | 546 |
| 1000 | 2023 | 192.9 s | **5.33 s** | 296.2 s | 717 |

The distribution is heavily skewed: half of all inventory is round-tripped inside ~6.5 s,
while a long tail sits for minutes. The mean is therefore the wrong summary -- it is
dragged by the stuck tail -- and ~6.5 s is the horizon the median exposure implies.

**At that horizon the edge is negative, and it is not a knife-edge choice.** `b` plateaus
at ~1.0 from 5 s all the way out to 30 s, so the answer is stable across every horizon in
the plausible range:

| horizon | 200 ms | 1 s | 5 s | **6.6 s (measured)** | 30 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| b | 0.378 | 0.873 | 1.051 | **1.050** | 0.988 |
| edge @26 bps | +12.23 | +0.96 | -4.10 | **-4.15** | -3.64 |

**The shipped 200 ms horizon is the only one that makes this strategy look profitable,
and it is roughly 30x shorter than the median holding period.** That is the whole error in
one line: the model books the adverse selection of the first 200 ms and then holds the
position for seconds to minutes.

### A feedback loop the phi ladder never modelled

Holding time *grows* with phi -- mean 112 s at 200, 193 s at 1000 -- because wider quotes
fill less often on the offsetting side, so inventory sits longer. Widening therefore
raises the markout horizon that applies to each fill, which raises adverse selection per
fill. Quoting wider is not a free reduction in toxicity, and the phi ladder scored it as
though it were.
