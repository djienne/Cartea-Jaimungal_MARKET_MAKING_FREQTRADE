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
| `epsilon+` | 1.89 | +2.32 | 0.354 | [0.322, 0.386] | True | +12.98 |
| `epsilon-` | 2.43 | +2.77 | 0.394 | [0.361, 0.430] | True | +11.49 |
| `pooled` | 2.13 | +2.45 | 0.378 | [0.353, 0.401] | True | +12.23 |

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
| `epsilon+` | 3.13 | +0.27 | 0.829 | [0.775, 0.883] | True | +2.68 |
| `epsilon-` | 3.91 | +1.75 | 0.903 | [0.849, 0.975] | True | -0.73 |
| `pooled` | 3.48 | +0.83 | 0.873 | [0.837, 0.912] | True | +0.96 |

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
| `epsilon+` | 3.94 | +0.86 | 1.031 | [0.952, 1.092] | False | -3.17 |
| `epsilon-` | 4.51 | +1.86 | 1.064 | [1.005, 1.165] | False | -5.03 |
| `pooled` | 4.19 | +1.28 | 1.051 | [0.994, 1.115] | False | -4.10 |

