# Requote hysteresis sweep and adverse-selection re-derivation

Companion to `requote_hysteresis_sweep.json`. Generated 2026-08-23 from a
deterministic replay of one frozen 120-minute CASHCAT window at four values of
`quoting.replace_threshold_bps` (with `replace_threshold_ticks = 1` held
fixed). The zero point (`bps = 0`) reproduces the pre-hysteresis exact-price
requote behavior bit for bit.

Both replay and the public dry run now simulate on **exchange time**
(`source_exchange_ms`), so these are the first figures free of the wall-clock
contamination that affected earlier dry-run evidence: previously, local
scheduler stalls delayed *simulated cancels*, leaving stale quotes resting
longer and inflating simulated pick-off.

## Results (120-minute window, 1000 USDC starting equity)

| bps | replaces → WS msgs/min | fills | realized P&L | fees | final equity | markout 100ms | 1s | 5s | 30s |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 0 (old behavior) | 212 | 41 | −5.46 | 0.53 | 994.45 | +1.09 | −1.29 | −4.24 | −3.37 |
| 1 | 143 | 45 | −3.35 | 0.60 | 996.54 | +1.90 | −1.19 | −5.01 | −5.03 |
| **2 (default)** | **99** | **54** | **−2.26** | **0.81** | **997.38** | **+3.31** | **+0.33** | **−2.66** | **−4.99** |
| 4 | 51 | 51 | −4.42 | 0.75 | 995.56 | +1.72 | −1.35 | −4.58 | −5.93 |

Reading:

- **bps = 2 wins this window on both economics and budget.** Best equity,
  best realized P&L, more (and better-priced) fills from preserved queue
  position, and less than half the zero point's message traffic — 99/min
  against the 1600/min WebSocket budget, versus 212/min at the zero point.
- **bps = 4 over-holds.** The 30s markout is the worst of the sweep: quotes
  left up to 4 bps stale get picked off in moves. The hold window should stay
  below `min_half_spread_bps` territory in spirit, and 4 bps is past the
  useful range for this instrument.
- The shipped default stays `replace_threshold_bps = 2.0`, now with evidence
  behind it rather than only the remediation plan's argument.

## Address-action cost — the constraint that actually binds

Added 2026-08-23, after a live run showed the venue's **address-action budget**
is a lifetime account allowance (`10,000 + 1 per USDC of cumulative volume`)
that never resets, not a rolling window. The sweep above was scored on
WebSocket messages/minute against a 1600/min budget; that budget was never the
binding one. Each order and each cancel costs one address action, so the true
cost is `virtual_orders_created + virtual_orders_canceled`, derived here from
the same `requote_hysteresis_sweep.json` runs:

| bps | orders | cancels | address actions | fills | actions/fill |
|----:|----:|----:|----:|----:|----:|
| 0 | 12,755 | 12,745 | **25,500** | 41 | **622** |
| 1 | 8,599 | 8,587 | **17,186** | 45 | **382** |
| 2 | 5,959 | 5,943 | **11,902** | 54 | **220** |
| 4 | 3,080 | 3,068 | **6,148** | 51 | **121** |

Reading this against the economics table above:

- **The shipped default is not the action-efficient choice.** `bps = 4` costs
  1.9x fewer actions than `bps = 2` (6,148 vs 11,902) for almost the same fill
  count (51 vs 54) — 120 actions per fill against 220.
- **No setting is sustainable on its own.** Allowance breaks even only when a
  fill generates more than 1 USDC of volume per action spent — roughly 220 USDC
  of notional per fill at `bps = 2`, or 120 at `bps = 4`. Live CASHCAT fills
  were ~11 USDC. The knob changes the multiple, not the sign.
- One 120-minute window at `bps = 2` costs 11,902 actions, so a **fresh
  account's 10,000-request buffer does not cover a single two-hour session.**

The default stays at `2.0`: it wins on equity, realized P&L and every markout
horizon, and `bps = 4` has the worst 30s markout in the sweep. But the choice
is now an explicit trade of allowance for adverse selection, not a free win. If
runtime per unit of allowance becomes the objective — a long unattended session
on a fixed budget — `bps = 4` is the setting to re-examine, and the decision
should be re-scored on actions-per-fill rather than messages/minute.

**Caveat:** one two-hour window on one day. This supports the default; it does
not close the question. Re-run this sweep (four replay invocations over a
frozen `HL_data` snapshot) before promoting any different value.

## Adverse-selection conclusion, restated on the corrected time base

The recorded finding — *"spread capture is positive everywhere, markout is
slightly bigger, and the loss grows with fill count"* — was derived from
dry-run evidence produced on the contaminated wall-clock base. Replay was
always event-time-clean, so the zero-point row above is the corrected
baseline for the same claim:

- Spread capture remains positive at 100ms (+1.09 USDC over 41 fills).
- Markout turns negative by 1s (−1.29) and deepens by 5s (−4.24).
- Realized P&L over the window is −5.46 USDC.

**The direction of the recorded conclusion stands on clean data**: CASHCAT
fills earn the spread and then lose more than the spread to adverse selection
within seconds. The wall-clock bug inflated the *magnitude* of older dry-run
figures (stale quotes rested longer than they would have), but it did not
manufacture the effect.

What changes the picture is the hysteresis: at `bps = 2` the same window's
1-second markout is positive (+0.33) and the realized loss shrinks by more
than half (−2.26 vs −5.46). Preserved queue position converts some previously
picked-off requotes into earlier, better-queued fills. The strategy remains
unprofitable net of adverse selection in this window — the finding is
mitigated, not overturned.

Older artifacts derived from pre-fix dry-run sessions
(`dry_run_quality_report.json`, markout figures in session reports predating
2026-08-23) should be treated as **stale** for quantitative use.
