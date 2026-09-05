# Parameter sweep — CASHCAT

- generated: `2026-09-05T14:45:01Z`
- execution model: `causal-v3`; queue decay per second: `0.0`
- tape: **469.1946 h**, 4196666 price rows, 2179254 trades (`2026-08-16T21:57:55.999000+00:00` → `2026-09-05T11:09:36.518000+00:00`)
- selection ran on the full 328.4362 h train slice
- train/held-out split at `2026-08-30T14:24:06.362000+00:00` (0.7 train)
- searched at scenario `good` (latency 100 ms, refresh 250 ms)
- a row is ranked only if it took ≥ 30 maker fills and ≥ 3.0/day
- the toxic-flow guard is an axis: `guard` on rows is the shipped guard, `off` disables that guard without changing the execution model

Calibration is fitted on the train slice only and applied unchanged to the
held-out slice. Epsilon horizons stop at 1 s because epsilon is the arrival
jump (eq. 10.22), not a markout.

## Stage A — calibration (risk knobs at shipped values)

| calibration | P&L | net spread | fills (bid/ask) | depth bid/ask bps | P&L bid/ask |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.75/0.75` | -4462.47 | +15717.95 | 80414 (39971/40443) | 17.0 / 15.6 | +8281.76 / +7436.19 |
| `eps500/1000_klo0.75/0.75` | -4558.31 | +15790.54 | 82142 (40841/41301) | 16.8 / 15.5 | +8321.14 / +7469.40 |
| `eps1000/500_klo0.75/0.75` | -4593.79 | +15751.54 | 82360 (40824/41536) | 16.8 / 15.3 | +8356.21 / +7395.33 |
| `eps500/500_klo0.75/0.75` | -4694.74 | +15812.36 | 83961 (41707/42254) | 16.6 / 15.2 | +8369.43 / +7442.94 |
| `eps200/1000_klo0.75/0.75` | -4703.83 | +15878.88 | 84312 (41916/42396) | 16.7 / 15.3 | +8385.49 / +7493.39 |
| `eps1000/200_klo0.75/0.75` | -4781.94 | +15862.83 | 84875 (42118/42757) | 16.6 / 15.0 | +8396.26 / +7466.57 |
| `eps500/200_klo0.75/0.75` | -4793.94 | +15953.46 | 86797 (43064/43733) | 16.3 / 14.9 | +8451.51 / +7501.96 |
| `eps200/500_klo0.75/0.75` | -4822.29 | +15963.69 | 86416 (42902/43514) | 16.4 / 15.0 | +8433.30 / +7530.39 |
| `eps1000/1000_klo0.75/0.5` | -5062.57 | +16350.53 | 88662 (44173/44489) | 16.2 / 15.3 | +8542.71 / +7807.82 |
| `eps200/200_klo0.75/0.75` | -5066.08 | +16075.54 | 89152 (44278/44874) | 16.1 / 14.7 | +8509.11 / +7566.43 |
| `eps1000/1000_klo0.5/0.75` | -5090.24 | +16391.42 | 89606 (44323/45283) | 16.6 / 14.7 | +8766.89 / +7624.53 |
| `eps500/1000_klo0.5/0.75` | -5154.23 | +16470.94 | 91537 (45286/46251) | 16.4 / 14.6 | +8808.29 / +7662.65 |
| `eps1000/500_klo0.75/0.5` | -5161.55 | +16373.25 | 90934 (45325/45609) | 15.9 / 15.0 | +8566.63 / +7806.62 |
| `eps500/1000_klo0.75/0.5` | -5166.76 | +16441.75 | 90848 (45333/45515) | 16.0 / 15.1 | +8577.49 / +7864.27 |
| `eps1000/500_klo0.5/0.75` | -5203.02 | +16546.67 | 92423 (45724/46699) | 16.4 / 14.5 | +8856.26 / +7690.41 |

## Stage B — risk settings (top 15 on train)

| calibration | risk | guard | P&L | net spread | directional | fills (bid/ask) | fills/day |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -186.44 | +4978.67 | -5165.12 | 13073 (6608/6465) | 955.3 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -187.28 | +5166.63 | -5353.91 | 13510 (6779/6731) | 987.3 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -192.09 | +5103.51 | -5295.60 | 13437 (6737/6700) | 981.9 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -212.05 | +5004.36 | -5216.40 | 13192 (6599/6593) | 964.0 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -261.66 | +5101.87 | -5363.52 | 13464 (6773/6691) | 983.9 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -267.83 | +5007.13 | -5274.96 | 13316 (6615/6701) | 973.1 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | on (1 trip) | -277.60 | +5339.46 | -5617.06 | 14516 (7348/7168) | 1060.8 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | on (1 trip) | -292.05 | +5512.81 | -5804.86 | 15294 (7683/7611) | 1117.6 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=6 | on (1 trip) | -298.43 | +5453.30 | -5751.73 | 14866 (7438/7428) | 1086.4 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | on (1 trip) | -322.79 | +5435.62 | -5758.41 | 15000 (7520/7480) | 1096.2 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=6 | on (1 trip) | -375.08 | +5583.36 | -5958.43 | 15505 (7835/7670) | 1133.1 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=6 | on (1 trip) | -397.40 | +5466.12 | -5863.52 | 14948 (7504/7444) | 1092.4 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=600.0 q=3 | on (1 trip) | -404.95 | +6257.43 | -6662.38 | 18561 (9269/9292) | 1356.4 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=600.0 q=6 | on (1 trip) | -425.44 | +6345.72 | -6771.16 | 18664 (9299/9365) | 1363.9 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=600.0 q=3 | on (1 trip) | -427.38 | +6178.02 | -6605.40 | 18223 (9147/9076) | 1331.7 |

## Stage C — held-out

Window counts span the whole tape, including training. Each window resets
the account; these are retrospective regime checks, not independent holdouts.

| configuration | train P&L | held-out P&L | net spread | directional | fills | windows + | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.75/0.75\|phiKT3000_alphaK0.05_T150_q3_guard` | -186.44 | **-505.81** | +3530.61 | -4036.42 | 9067 | 29/77 | -96.51 |
| `eps1000/500_klo0.75/0.75\|phiKT3000_alphaK0.05_T150_q6_guard` | -187.28 | **-516.26** | +3568.80 | -4085.07 | 9430 | 32/77 | -89.72 |
| `eps500/1000_klo0.75/0.75\|phiKT3000_alphaK0.05_T150_q6_guard` | -192.09 | **-583.68** | +3581.46 | -4165.14 | 9562 | 30/77 | -98.00 |

Per-window scores are retained in the companion JSON.

## Latency ladder — winner, held-out slice

Latency and requote cadence change together. Actual resting exposure depends
on the event schedule, activation, cancellation and partial fills; these are
assumed scenarios, not measured host execution capabilities.

| scenario | latency | refresh | P&L | net spread | directional | fills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| colocated | 50 ms | 100 ms | -524.74 | +3852.07 | -4376.81 | 10987 |
| good | 100 ms | 250 ms | -505.81 | +3530.61 | -4036.42 | 9067 |
| mid | 200 ms | 500 ms | -677.50 | +3349.41 | -4026.91 | 8476 |
| this_stack | 500 ms | 1000 ms | -914.38 | +2920.11 | -3834.49 | 7755 |
| reality | 500 ms | 30000 ms | -1711.68 | +6143.20 | -7854.89 | 19470 |

## How to read the columns

**net spread** is filled quantity times distance from the decision-time mid,
minus fees. Despite the legacy JSON name, it is quoted spread capture, not
post-fill realized spread or proof of profit. **directional** is the residual
P&L, including adverse selection, inventory revaluation and funding.
Ranking eligibility tests fill counts only, not solvency or promotion readiness.
Replay keeps accounting after maintenance breaches; inspect the JSON breach
counts and minimum liquidation buffers before interpreting a return.

## Paper-only contenders

These additional controls use the saved 200/200 ms, full-support training fit.
Their separate Rust causal-v4 replay uses 297.88 USDC, capital-derived sizing,
150 ms decision/acknowledgement/cancellation delays, the configured latency tail,
fees, funding and terminal risk gates. These are not the Python search returns.

| Paper row | Marked P&L (USDC) | Maker fills | Lot-age exits | Final inventory |
|---|---:|---:|---:|---:|
| `contender_flat300` | 96.61 | 942 | 942 | 0 |
| `contender_flat550` | 44.99 | 946 | 946 | 0 |

Both finish the full scored suffix without a risk stop. Their selection follows
inspection of reused research data, so the gains require prospective paper
validation. Neither fixed-fit nor lot-age-exit rows can be promoted live.
The paper roster and execution interpretation are maintained in
`DRY_RUN_GRID.md` and `CAUSAL_EXECUTION_REVIEW.md`.

## Converged paper-model comparison

All rows use the common full scored window and the paper assumptions above,
with HJB timestep 1/512 s and Newton residual tolerance 1e-10. The fixed-fit
`control_wide60` is an offline rejection diagnostic; the paper roster retains
its separate recent-fit `wide60` control. Returns below are in USDC.

| Row | Marked P&L | Exit-adjusted P&L | Maker fills | Lot exits | Scored fraction | Valid |
|---|---:|---:|---:|---:|---:|---|
| `sweep1` | -147.78 | -147.79 | 3281 | 0 | 100.00% | yes |
| `sweep2` | -81.91 | -81.92 | 2498 | 0 | 100.00% | yes |
| `sweep3` | -79.82 | -79.87 | 2463 | 0 | 100.00% | yes |
| `sweep1_unguarded` | -147.78 | -147.79 | 3281 | 0 | 100.00% | yes |
| `sweep1_wide60` | -178.40 | -179.58 | 268 | 0 | 51.25% | risk stop |
| `sweep1_flat300` | 190.70 | 190.70 | 718 | 718 | 100.00% | yes |
| `sweep1_flat550` | 117.60 | 117.60 | 718 | 718 | 100.00% | yes |
| `contender_flat300` | 96.61 | 96.61 | 942 | 942 | 100.00% | yes |
| `contender_flat550` | 44.99 | 44.99 | 946 | 946 | 100.00% | yes |
| `control_wide60` | -168.21 | -170.47 | 470 | 0 | 69.49% | risk stop |

Risk-stopped rows do not cover the whole suffix and cannot be ranked as
full-period returns. Exit adjustment values remaining inventory at the executable
side with configured exit costs; it is not an executed liquidation.
