# Parameter sweep — CASHCAT

- generated: `2026-09-05T02:47:37Z`
- execution model: `causal-v2`; queue decay per second: `0.0`
- tape: **457.4267 h**, 4083323 price rows, 2127998 trades (`2026-08-16T21:57:55.999000+00:00` → `2026-09-04T23:23:32.030000+00:00`)
- selection ran on the full 320.1983 h train slice
- train/held-out split at `2026-08-30T06:09:51.220000+00:00` (0.7 train)
- searched at scenario `good` (latency 100 ms, refresh 250 ms)
- a row is ranked only if it took ≥ 30 maker fills and ≥ 3.0/day
- the toxic-flow guard is an axis: `guard` on rows is the shipped guard, `off` disables that guard without changing the execution model

Calibration is fitted on the train slice only and applied unchanged to the
held-out slice. Epsilon horizons stop at 1 s because epsilon is the arrival
jump (eq. 10.22), not a markout.

## Stage A — calibration (risk knobs at shipped values)

| calibration | P&L | net spread | fills (bid/ask) | depth bid/ask bps | P&L bid/ask |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eps1000/1000_klo0.75/0.75` | -4322.09 | +15437.96 | 78677 (39158/39519) | 17.2 / 15.7 | +8178.28 / +7259.68 |
| `eps1000/500_klo0.75/0.75` | -4363.37 | +15563.28 | 80955 (40260/40695) | 17.1 / 15.4 | +8265.18 / +7298.10 |
| `eps500/1000_klo0.75/0.75` | -4487.57 | +15535.93 | 80372 (39957/40415) | 17.1 / 15.5 | +8234.19 / +7301.74 |
| `eps500/500_klo0.75/0.75` | -4544.57 | +15597.78 | 82555 (40994/41561) | 16.8 / 15.2 | +8272.35 / +7325.43 |
| `eps200/1000_klo0.75/0.75` | -4546.52 | +15700.67 | 82959 (41265/41694) | 16.9 / 15.3 | +8297.53 / +7403.15 |
| `eps1000/200_klo0.75/0.75` | -4565.85 | +15661.75 | 83449 (41494/41955) | 16.8 / 15.1 | +8308.59 / +7353.16 |
| `eps200/500_klo0.75/0.75` | -4651.05 | +15696.89 | 84788 (42146/42642) | 16.5 / 15.1 | +8290.15 / +7406.74 |
| `eps500/200_klo0.75/0.75` | -4672.29 | +15725.98 | 84998 (42185/42813) | 16.6 / 14.9 | +8361.20 / +7364.78 |
| `eps200/200_klo0.75/0.75` | -4848.15 | +15857.56 | 87444 (43439/44005) | 16.3 / 14.8 | +8404.06 / +7453.50 |
| `eps1000/1000_klo0.75/0.5` | -4873.46 | +16096.57 | 87154 (43444/43710) | 16.4 / 15.4 | +8404.99 / +7691.58 |
| `eps1000/1000_klo0.5/0.75` | -4916.80 | +16206.43 | 88270 (43717/44553) | 16.8 / 14.8 | +8687.87 / +7518.56 |
| `eps500/1000_klo0.75/0.5` | -4991.80 | +16208.93 | 89104 (44529/44575) | 16.2 / 15.2 | +8472.87 / +7736.06 |
| `eps1000/500_klo0.75/0.5` | -4999.74 | +16202.61 | 89568 (44663/44905) | 16.2 / 15.0 | +8497.70 / +7704.91 |
| `eps500/1000_klo0.5/0.75` | -5052.76 | +16216.83 | 89685 (44445/45240) | 16.7 / 14.6 | +8693.71 / +7523.11 |
| `eps1000/500_klo0.5/0.75` | -5063.12 | +16313.51 | 90804 (45007/45797) | 16.5 / 14.6 | +8721.12 / +7592.39 |

## Stage B — risk settings (top 15 on train)

| calibration | risk | guard | P&L | net spread | directional | fills (bid/ask) | fills/day |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -179.95 | +4956.78 | -5136.73 | 13102 (6617/6485) | 982.1 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | on (1 trip) | -203.68 | +5474.28 | -5677.96 | 15053 (7619/7434) | 1128.3 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -204.10 | +5080.82 | -5284.92 | 13324 (6713/6611) | 998.7 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -221.89 | +5027.54 | -5249.43 | 13127 (6626/6501) | 984.0 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | on (1 trip) | -251.80 | +5335.87 | -5587.67 | 14622 (7480/7142) | 1096.0 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -254.09 | +4830.11 | -5084.20 | 12816 (6556/6260) | 960.7 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -256.07 | +4987.55 | -5243.62 | 13037 (6543/6494) | 977.2 |
| `eps1000/500_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=3 | on (1 trip) | -266.42 | +5514.93 | -5781.34 | 15132 (7623/7509) | 1134.3 |
| `eps1000/1000_klo0.75/0.75` | φκT=1000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -294.32 | +5893.56 | -6187.88 | 16516 (8359/8157) | 1238.0 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=6 | on (1 trip) | -294.48 | +5444.10 | -5738.59 | 15004 (7559/7445) | 1124.7 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=150.0 q=6 | on (1 trip) | -311.25 | +5052.10 | -5363.36 | 13394 (6783/6611) | 1004.0 |
| `eps500/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=600.0 q=3 | on (1 trip) | -342.95 | +6145.50 | -6488.45 | 17859 (8995/8864) | 1338.7 |
| `eps1000/1000_klo0.75/0.75` | φκT=3000.0 ακ=0.05 T=300.0 q=6 | on (1 trip) | -346.68 | +5440.52 | -5787.20 | 14892 (7520/7372) | 1116.3 |
| `eps1000/1000_klo0.75/0.75` | φκT=1000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -372.93 | +5813.53 | -6186.47 | 16466 (8311/8155) | 1234.3 |
| `eps500/1000_klo0.75/0.75` | φκT=1000.0 ακ=0.05 T=150.0 q=3 | on (1 trip) | -373.05 | +5841.29 | -6214.35 | 16589 (8356/8233) | 1243.5 |

## Stage C — held-out

Window counts span the whole tape, including training. Each window resets
the account; these are retrospective regime checks, not independent holdouts.

| configuration | train P&L | held-out P&L | net spread | directional | fills | windows + | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eps500/1000_klo0.75/0.75\|phiKT3000_alphaK0.05_T150_q3_guard` | -179.95 | **-583.11** | +3431.19 | -4014.29 | 9209 | 30/75 | -75.63 |
| `eps500/1000_klo0.75/0.75\|phiKT3000_alphaK0.05_T300_q3_guard` | -203.68 | **-753.39** | +3727.82 | -4481.21 | 10596 | 23/75 | -113.43 |
| `eps1000/500_klo0.75/0.75\|phiKT3000_alphaK0.05_T150_q6_guard` | -204.10 | **-520.97** | +3554.23 | -4075.20 | 9372 | 29/75 | -99.20 |

Per-window scores are retained in the companion JSON.

## Latency ladder — winner, held-out slice

Latency and requote cadence change together. Actual resting exposure depends
on the event schedule, activation, cancellation and partial fills; these are
assumed scenarios, not measured host execution capabilities.

| scenario | latency | refresh | P&L | net spread | directional | fills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| colocated | 50 ms | 100 ms | -485.51 | +3738.61 | -4224.11 | 10516 |
| good | 100 ms | 250 ms | -583.11 | +3431.19 | -4014.29 | 9209 |
| mid | 200 ms | 500 ms | -627.75 | +3176.58 | -3804.34 | 8086 |
| this_stack | 500 ms | 1000 ms | -848.95 | +2808.81 | -3657.76 | 7507 |
| reality | 500 ms | 30000 ms | -1683.57 | +5905.45 | -7589.02 | 19148 |

## How to read the columns

**net spread** is filled quantity times distance from the decision-time mid,
minus fees. Despite the legacy JSON name, it is quoted spread capture, not
post-fill realized spread or proof of profit. **directional** is the residual
P&L, including adverse selection, inventory revaluation and funding.
Ranking eligibility tests fill counts only, not solvency or promotion readiness.
Replay keeps accounting after maintenance breaches; inspect the JSON breach
counts and minimum liquidation buffers before interpreting a return.
