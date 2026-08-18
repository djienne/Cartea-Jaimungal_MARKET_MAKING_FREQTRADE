# Parameter sweep — CASHCAT

- generated: `2026-08-18T05:15:26Z`
- tape: **31.2327 h**, 176931 price rows, 76499 trades (`2026-08-16T21:57:55.999000+00:00` → `2026-08-18T05:11:53.750000+00:00`)
- train/held-out split at `2026-08-17T19:49:42.424000+00:00` (0.7 train)
- searched at scenario `good` (latency 100 ms, refresh 250 ms)
- a row is ranked only if it took ≥ 30 maker fills

Calibration is fitted on the train slice only and applied unchanged to the
held-out slice. Epsilon horizons stop at 1 s because epsilon is the arrival
jump (eq. 10.22), not a markout.

## Stage A — calibration (risk knobs at shipped values)

| calibration | P&L | net spread | fills (bid/ask) | depth bid/ask bps | P&L bid/ask |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.75/0.75` | -62.60 | +410.43 | 1944 (984/960) | 20.3 / 17.4 | +223.15 / +187.28 |
| `eps1000/1000_klo0.5/0.5` | -73.14 | +435.53 | 2323 (1167/1156) | 18.7 / 15.5 | +243.01 / +192.52 |
| `eps500/500_klo0.75/0.75` | -75.14 | +416.70 | 2092 (1062/1030) | 19.5 / 16.1 | +231.82 / +184.88 |
| `eps200/200_klo0.75/0.75` | -76.26 | +425.62 | 2250 (1136/1114) | 18.9 / 15.7 | +237.07 / +188.55 |
| `eps500/500_klo0.5/0.5` | -82.31 | +447.84 | 2533 (1268/1265) | 17.9 / 14.4 | +253.26 / +194.58 |
| `eps500/500_klo0/0` | -84.78 | +454.67 | 2722 (1373/1349) | 17.2 / 13.9 | +255.94 / +198.73 |
| `eps1000/1000_klo0/0` | -88.59 | +448.75 | 2504 (1260/1244) | 18.1 / 14.6 | +254.27 / +194.48 |
| `eps200/200_klo0.5/0.5` | -92.21 | +447.30 | 2719 (1384/1335) | 17.0 / 14.0 | +248.38 / +198.92 |
| `eps200/200_klo0/0` | -92.59 | +465.87 | 2965 (1476/1489) | 16.9 / 12.9 | +267.64 / +198.24 |

## Stage B — the book's risk preference (top 15 on train)

| calibration | risk | P&L | net spread | directional | fills (bid/ask) |
| --- | --- | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.5/0.5` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | +23.20 | +156.04 | -132.85 | 385 (202/183) |
| `eps1000/1000_klo0.5/0.5` | φκT=3000.0 ακ=0.05 T=300.0 q=6 | +23.20 | +156.04 | -132.85 | 385 (202/183) |
| `eps1000/1000_klo0.5/0.5` | φκT=10000.0 ακ=0.05 T=150.0 q=3 | +19.76 | +125.61 | -105.84 | 293 (149/144) |
| `eps1000/1000_klo0.5/0.5` | φκT=10000.0 ακ=0.05 T=150.0 q=6 | +19.76 | +125.61 | -105.84 | 293 (149/144) |
| `eps1000/1000_klo0.5/0.5` | φκT=30000.0 ακ=0.05 T=300.0 q=3 | +17.39 | +129.15 | -111.75 | 287 (151/136) |
| `eps1000/1000_klo0.5/0.5` | φκT=30000.0 ακ=0.05 T=300.0 q=6 | +17.39 | +129.15 | -111.75 | 287 (151/136) |
| `eps1000/1000_klo0.5/0.5` | φκT=10000.0 ακ=0.05 T=300.0 q=3 | +15.30 | +129.49 | -114.19 | 311 (160/151) |
| `eps1000/1000_klo0.5/0.5` | φκT=10000.0 ακ=0.05 T=300.0 q=6 | +15.30 | +129.49 | -114.19 | 311 (160/151) |
| `eps1000/1000_klo0.5/0.5` | φκT=30000.0 ακ=0.05 T=600.0 q=3 | +12.87 | +124.18 | -111.32 | 285 (141/144) |
| `eps1000/1000_klo0.5/0.5` | φκT=30000.0 ακ=0.05 T=600.0 q=6 | +12.87 | +124.18 | -111.32 | 285 (141/144) |
| `eps1000/1000_klo0.75/0.75` | φκT=1000.0 ακ=0.05 T=300.0 q=3 | +12.84 | +149.97 | -137.13 | 396 (196/200) |
| `eps1000/1000_klo0.75/0.75` | φκT=1000.0 ακ=0.05 T=300.0 q=6 | +12.84 | +149.97 | -137.13 | 396 (196/200) |
| `eps1000/1000_klo0.5/0.5` | φκT=1000.0 ακ=0.05 T=150.0 q=3 | +12.23 | +164.94 | -152.71 | 441 (226/215) |
| `eps1000/1000_klo0.5/0.5` | φκT=1000.0 ακ=0.05 T=150.0 q=6 | +12.23 | +164.94 | -152.71 | 441 (226/215) |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=600.0 q=3 | +11.11 | +146.45 | -135.34 | 358 (172/186) |

## Stage C — held-out

| configuration | train P&L | held-out P&L | net spread | directional | fills | windows + | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.5/0.5|phiKT3000_alphaK0.05_T300_q3` | +23.20 | **+1.56** | +10.43 | -8.87 | 29 | 4/5 | -2.88 |
| `eps1000/1000_klo0.5/0.5|phiKT3000_alphaK0.05_T300_q6` | +23.20 | **+1.56** | +10.43 | -8.87 | 29 | 4/5 | -2.88 |

### Leader per 6.0 h window — `eps1000/1000_klo0.5/0.5|phiKT3000_alphaK0.05_T300_q3`

| window | P&L | net spread | directional | fills (bid/ask) |
| --- | ---: | ---: | ---: | ---: |
| 08-16 21:57 | +7.77 | +54.12 | -46.35 | 132 (66/66) |
| 08-17 03:57 | +0.56 | +7.28 | -6.72 | 21 (11/10) |
| 08-17 09:57 | -2.88 | +16.25 | -19.14 | 39 (20/19) |
| 08-17 15:57 | +14.57 | +85.01 | -70.45 | 218 (114/104) |
| 08-17 21:57 | +0.00 | +0.00 | +0.00 | 0 (0/0) |
| 08-18 03:57 | +0.59 | +1.78 | -1.19 | 3 (1/2) |

## Latency ladder — winner, held-out slice

Latency and requote cadence move together: resting exposure is
`max(0, refresh − latency) + cancel`, so cutting latency at a fixed refresh
lengthens the time a quote sits. Each row is a plausible machine.

| scenario | latency | refresh | P&L | net spread | directional | fills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| colocated | 50 ms | 100 ms | +0.27 | +14.79 | -14.52 | 39 |
| good | 100 ms | 250 ms | +1.56 | +10.43 | -8.87 | 29 |
| mid | 200 ms | 500 ms | -1.17 | +16.54 | -17.71 | 45 |
| this_stack | 500 ms | 1000 ms | +1.51 | +12.50 | -11.00 | 34 |
| reality | 500 ms | 30000 ms | -7.24 | +30.84 | -38.07 | 82 |

## How to read the columns

**net spread** is what market making earns: realized spread minus fees.
**directional** is P&L minus that — a bet on where the mid went, which the
tape decides and the model does not. A row whose P&L is mostly directional has
not demonstrated market making, however good the total looks.
