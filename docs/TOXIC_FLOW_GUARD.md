# Toxic-flow guard

## The problem it addresses

The 161.95 h staged sweep put the entire loss in one six-hour window. 26 of the
other 27 sum to **+35.28 USDC**; `08-22 03:57` alone is **−241.17** on 1,771
fills. Looking at what that window actually was changes the problem, because it
is not a slow toxic drift — it is a liquidation cascade:

```
05:11:00  mid 0.12305     —
05:11:15  mid 0.11453   −8%
05:11:20  mid 0.09952  −31%     ← ten seconds
05:11:30  mid 0.08414  −45%
05:12:00  mid 0.03900  −70%     ← sixty seconds
05:16:00  mid ~0.11            recovered
```

Nothing predicts that. The goal is narrower: stop riding it down, and stay out
of the aftermath.

## Why the existing toxicity gate did not help

There is already a toxicity metric — `toxicity = κ·ε` per side, gated by
`calibration.max_toxicity = 1.5` (`crates/cj-data/src/calibration.rs`). Through
this cascade the sweep's own diagnostics record **0.254 / 0.235**, six times
below the threshold. It is a slowly-varying property of the calibration fit, not
a flow alarm, and it never fires here. It is left alone; this guard is separate.

## Two tiers, measured

Both thresholds were chosen on the full 6.8-day tape, and both have **zero false
positives outside the cascade**:

| tier | fires | outside crash | first trip | mid had moved |
|---|---:|---:|---|---:|
| mid move ≥ 8% in 5 s | 72 | **0** | 05:11:15 | **−14%** |
| VPIN ≥ 0.40 | 64 buckets | **0** | 05:11:31 | −45% |

They are kept together because they do different jobs. The breaker is 16 seconds
and 31 percentage points earlier, so it bounds the damage. VPIN is slower but
identifies the *regime* — 86% of the losing window's volume arrived after it
fired — so it is what prevents re-entering the aftermath.

At 8% the breaker was the tightest threshold with no false positives; 5%
produced four. The whole-tape maximum VPIN outside the cascade was 0.362, and the
cascade peaked at 0.663.

### Why not VPIN's own CDF

The reference implementation (`VPIN_PYTHON_CRYPTO`) ranks VPIN as a rolling
percentile and alerts above 0.99. That was tried and rejected: on a 2-day
lookback the CDF reaches 1.00 in **four benign windows**, because the tape is far
shorter than the 90 days that percentile assumes. Raw VPIN with an absolute
threshold separates cleanly here, so that is what ships.

One input is better than the reference's: it infers buy volume from Binance's
taker-buy aggregate on 1-minute klines, whereas every `TradePrint` here carries
the aggressor side, so the buy/sell split is exact rather than reconstructed.

## Behaviour

- **Trip** on either tier. Quotes go empty, which cancels resting orders
  immediately — a `None` target bypasses the requote hold window by design.
- **Re-entry needs both** VPIN back under threshold *and* `cooldown_ms` elapsed.
  VPIN alone can dip mid-cascade as a bucket completes; a timer alone would
  re-enter a live one. A fresh breach re-arms the clock, so a cascade cannot
  expire its own cooldown. On this event VPIN cleared ~16 minutes after the
  trip, about when price stabilised — the two conditions agreed.
- While VPIN is still warming up it reads `None`, which is treated as neither
  safe nor toxic: only the fast breaker is armed, and a tripped guard stays
  closed rather than re-opening on a statistic it does not have.
- It is a *quoting* gate, not a kill switch: inventory is left to the existing
  reduce-only machinery.

Withdrawals are attributed to `QuoteReason::ToxicFlow`, so every report and JSONL
line says why, and trip counts are visible in the dry-run logs.

## The A/B

Replayed over frozen shard sets (the collector keeps writing, so an unfrozen
tape would score different data on each leg). Same binary, same config, one
field changed.

| 16 h window | guard off | guard on | effect |
|---|---:|---:|---:|
| **crash** 08-21 20:00 → 08-22 12:00 | −87.95 | **−23.13** | **+64.82** |
| **calm** 08-19 08:00 → 08-20 00:00 | −80.90 | −80.90 | **0.00** |

*(Re-baselined by the guard-candidate study, `FLOW_GUARD_CANDIDATES.md`. The
original A/B recorded −33.88 for the guard-on leg; the guard-off leg and the
trip anatomy — first trip 05:11:15, quotes withheld to ~06:40 — reproduce
exactly, and the guard-on delta is re-entry-timing sensitivity: VPIN bucket
volume derives from the loaded calibration window, so small window-derivation
differences move the re-entry minute. Direction and mechanism unchanged.)*

In the crash window the loss falls by 61%, and the mechanism is visible in the
detail rather than only the total:

| | off | on |
|---|---:|---:|
| ending inventory | 2,091 | **74** |
| 5 s markout | −166.48 | **−29.40** |
| `risk_limit` withdrawals | 83,792 | 311 |
| fills | 389 | 274 |

It stopped the cascade accumulation, which is what the −241 was made of. The
`risk_limit` collapse is the same fact seen from the other side: the guard
prevented the losses that were tripping the daily-loss limit.

In the calm control it is **bit-identical** — same 1,208 fills, same ending
inventory, same P&L. It never fired in 16 hours of normal trading.

### A first attempt that measured nothing

The first A/B ran over the whole 165 h tape and returned *identical* P&L with the
guard on and off, despite 31,499 trips. The reason is worth recording: the
reason counts were `risk_limit 432,753` with the guard off, and
`risk_limit 401,254 + toxic_flow 31,499` with it on — the same total. Quoting was
already withdrawn by a risk limit that had been latched since 22:03 on day one,
so the guard only relabelled the reason. Scoping the window so the strategy
starts unlatched before the cascade is what made the test informative.

## Limits, stated plainly

- **n = 1.** One cascade in 6.8 days. The thresholds were chosen on the single
  event they are then tested against, so the zero-false-positive claim is only as
  good as that window. Re-check as the tape grows toward its 30-day retention.
  *Re-checked on 165.11 h (2026-08-23, `FLOW_GUARD_CANDIDATES.md`): still zero
  false positives for both tiers; max 5s move outside the cascade 427 bps
  (threshold 800), max VPIN outside 0.371 (threshold 0.40 — was 0.362, so the
  headroom is narrowing slowly).*
- **It cannot prevent the first fills.** At −14% the resting bids have already
  been hit. This bounds the damage; it does not avoid it.
- **A guard is not an edge.** Both A/B legs still lose money. If CASHCAT's
  economics only work once a 70% cascade is excluded, that is a finding about the
  instrument, not a fixed strategy.

## Configuration

`[flow_guard]` in the config, and a lever in the dry-run grid
(`flow_guard_enabled`, `vpin_threshold`, `fast_move_threshold_bps`) so guarded
and unguarded variants can quote the same live feed side by side — the honest
out-of-sample test that the replay A/B above cannot be.
