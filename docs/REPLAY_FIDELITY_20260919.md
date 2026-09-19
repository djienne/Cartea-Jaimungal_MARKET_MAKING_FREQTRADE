# Rust replay receipt-flow-v1: validation, 2026-09-19 UTC

The replay and grid share quote, execution, restart and pause/resume logic.
This revision reconstructs recorded-data availability; it cannot reconstruct
every grid process/network interruption. The historical execution boundary
remains **2026-09-12 20:26:31 UTC**, when `causal-v5` was deployed.

## Conditions and checks

- Docker: 318 Rust tests passed; the optional HJB timestep-refinement study
  stayed ignored. Clippy passed with warnings denied. No Python parity gate or new dependency.
- Shared-event checks compare grid and replay decisions, execution events, fills
  and accounts across three exit settings, including open inventory, silent
  intervals, terminal stops, short resumes and interruptions over 900 seconds.
- A future checkpoint was also rejected through the actual replay command.
- LaTeX: 12 source snippets and 55 worked-example numbers checked; PDF rebuilt.
  These Python document checks are separate from Rust execution validation.
- Separate image `mm-live:replay-fidelity-20260919`; old results retained under
  `rust_live/reports/`. The running grid and real-money settings were not changed.

Short check: **2026-09-17 07:00–09:00 UTC**, six-hour training prefix,
`sweep1_flat300=636`. A pause from **07:14:20.000 to 07:54:54.666 UTC**
was reconstructed (2,434.666 seconds). Grid and replay both recorded no fills
and zero interval P&L. This checks availability recovery, not fill fidelity.
A separate **07:15–07:45 UTC** run, wholly inside the outage, completed with
one open 1,800-second pause and zero fills; an empty scoring tape is accepted.

Full check: **2026-09-17 00:00–2026-09-18 00:00 UTC**, training from
**2026-09-16 18:00 UTC**. All 22 variants used the same engine and the grid's
persisted sizes: 318 by default; `q3` and all `sweep1*` rows 636; `q9` 212;
`q12` 159. Latencies stayed **150/150/150 ms**, tail **2.35**, one slow second
in twenty; carry limit **900 s**. No compatible checkpoint at the starting
boundary was available: replay starts flat with cold guards and fresh loss
budgets. A current checkpoint was used only to read sizes, never to seed past accounts.

## Same-window results

P&L is in USDC. Grid CSV samples precede the bounds by 1.669 and 19.316 seconds.
“Active” is the grid's validity flag at start/end; replay completed all 22 rows.
The comparison is between interval changes, never lifetime profits.

| Variant | Grid P&L | Replay P&L | Maker fills grid/replay | Grid starting inventory | Grid active start/end |
|---|---:|---:|---:|---:|---|
| wide60 | 6.551664 | 12.511393 | 75 / 71 | -966 | 1 / 1 |
| wide40 | 2.688882 | 10.190827 | 160 / 165 | -1382 | 1 / 1 |
| sweep1_wide60 | 8.683770 | 3.958458 | 51 / 53 | 526 | 1 / 1 |
| flatten550 | -3.049364 | -3.049364 | 75 / 75 | 0 | 1 / 1 |
| contender_flat550 | -3.049365 | -3.049364 | 75 / 75 | 0 | 1 / 1 |
| flatten300 | -3.294704 | -3.294704 | 74 / 74 | 0 | 1 / 1 |
| contender_flat300 | -3.294704 | -3.294704 | 74 / 74 | 0 | 1 / 1 |
| sweep3 | -2.651954 | -3.362421 | 114 / 86 | 39 | 1 / 1 |
| sweep2 | -2.755695 | -3.381501 | 117 / 87 | 39 | 1 / 1 |
| sweep1_flat300 | -4.679231 | -4.679231 | 25 / 25 | 0 | 1 / 1 |
| sweep1_flat550 | -5.383128 | -5.383128 | 25 / 25 | 0 | 1 / 1 |
| flatten300w40 | -12.591775 | -12.483607 | 177 / 177 | 0 | 1 / 1 |
| sweep1 | -4.575365 | -15.658014 | 139 / 92 | -40 | 1 / 1 |
| sweep1_unguarded | -3.410471 | -15.658014 | 139 / 92 | 137 | 1 / 1 |
| q12 | -16.172856 | -18.076316 | 1499 / 1267 | -24 | 1 / 1 |
| q9 | -23.586306 | -23.166162 | 1441 / 1291 | -3 | 1 / 1 |
| phi1000 | -26.623227 | -23.773251 | 1005 / 899 | 57 | 1 / 1 |
| baseline | -0.549836 | -33.520149 | 36 / 1354 | -66 | 1 / 0 |
| unguarded | 0.614265 | -33.520149 | 0 / 1354 | 28 | 1 / 1 |
| q3 | 0.000000 | -62.147668 | 0 / 1517 | 714 | 0 / 0 |
| slow5s | 0.000000 | -68.646400 | 0 / 3212 | -695 | 0 / 0 |
| slow15s | 0.000000 | -90.206191 | 0 / 3797 | -270 | 0 / 0 |

Six flatten rows have the same maker-fill count and P&L within **0.000001 USDC**.
For `sweep1_flat300`: **25 fills**, grid **−4.679231 USDC**, replay
**−4.67923139320203 USDC**; fees **1.290261** versus **1.290261087 USDC**.
This is agreement on one window, not proof of venue-fill realism.

All rows saw 14 inferred pauses totaling **3,100.574 seconds**, 44 rejected
touches and no missing receipt timestamps. Fourteen variants closed carried
inventory after the long interruption. Final cash plus signed position value
equaled equity for all 22 accounts (maximum measured residual: zero).
Repeated runs produced identical leaderboard rows, execution diagnostics and
final accounts for every variant.

Remaining differences have identifiable initial-state confounders: `q3`,
`slow5s` and `slow15s` were already stopped in the grid; `baseline` stopped
during the window. Several grid rows started with inventory, and rolling-fit
controls and flow guards do not start with the same calibration/history as a
fresh replay. These facts do not quantify every P&L difference or assign it to latency.

For `sweep1_flat300`, the first comparable retained quote prefix starts at
**11:32:41.140 UTC**: grid bid/ask **0.199700 / 0.202930**; the first replay
quote at/after it is **11:32:41.341**, **0.199670 / 0.202910**, both size 636.
Earlier complete quote history is unavailable in these retained logs. The
201 ms difference is an observed trace offset, **not a measured transport
latency**. Aggregate fills can agree while individual quote traces differ.

Detailed local evidence: `rust_live/reports/replay_receipt_flow_20260919/`,
including arguments, unit snapshot, future-state rejection, per-variant
reports, comparison traces and Docker checks. Operational options and tie
ordering are documented in [DRY_RUN_GRID.md](DRY_RUN_GRID.md).
