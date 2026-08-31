# Spread ladder over 185 h — the first positive replay rungs

> **RE-BASED 2026-08-24.** Every number below was first measured with a
> simulator in which `min_order_lifetime_ms` was inert
> (`DRY_RUN_GRID.md`). Fixing that changed the whole ladder by −58.73 to
> +29.84 per rung, so the original figures are not comparable with anything
> measured afterwards. The table now carries both. **The conclusion survives
> in direction but is weaker and noisier than first reported**: 40–60 bps is
> still the only profitable region, but at roughly half the magnitude, and the
> middle of the ladder is no longer cleanly ordered.

> **Subsequent live evidence.** Two independent day-length grid runs with a
> binding `q_max` kept 24/40/48/60 bps positive, but reversed the ordering of
> 40/48/60 and ended with material directional inventory. The replay table below
> is evidence for a promising width region, not a deployable winner; see
> `DRY_RUN_GRID.md`.

| half-spread | pre-fix | **post-fix** | delta | fills | 5 s markout | end inv |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5 (shipped) | −109.20 | **−118.61** | −9.41 | 5,424 | −82.17 | 179 |
| 4 | −88.91 | **−59.06** | +29.84 | 3,266 | −45.15 | 241 |
| 8 | −61.53 | **−95.90** | −34.36 | 5,239 | −74.99 | −58 |
| 16 | −62.29 | **−69.46** | −7.17 | 4,206 | −62.50 | −44 |
| 24 | −62.50 | **−79.49** | −16.99 | 3,472 | −57.51 | −35 |
| **40** | +40.78 | **+21.02** | −19.76 | 1,781 | −37.37 | −413 |
| **60** | +91.84 | **+33.11** | −58.73 | 731 | −20.76 | −2,440 |

What changed in the reading:

- **Only the wide end is still clearly positive** — 40 and 60 bps, at +21.02
  and +33.11 rather than +40.78 and +91.84.
- **The clean monotone shape is gone.** Post-fix, 4 bps (−59.06) beats 8
  (−95.90), 16 (−69.46) and 24 (−79.49). The middle of the ladder is noise;
  only the extremes carry signal. The pre-fix monotonicity was partly an
  artefact of the broken gate.
- **The markout mechanism survives intact** — 5 s markout still improves
  monotonically with width, −82.17 → −20.76. Widening reduces measured adverse
  selection on this tape; it does not by itself establish positive expected P&L.
- spread60's ending short grew to 2,440 units, deepening the leverage caveat
  below rather than relieving it.

## Spread × cadence, post-fix

Now that the cadence lever actually functions, the two can be crossed. They do
**not** compose monotonically:

| half-spread | 100 ms | 30 s | effect of slowing |
|---:|---:|---:|---:|
| 8 | −95.90 | **aborted** — liquidation-buffer breach | ruinous |
| 24 | −79.49 | **−65.81** | +13.68 |
| 40 | **+21.02** | −151.74 | −172.76 |

Slowing helps only in the middle. At 8 bps a 30 s quote is run over hard enough
to breach the liquidation buffer and abort the run; at 40 bps it destroys the
only profitable rung. This kills the `wide8slow30s` hypothesis — the corner
that looked most promising when the lever was inert — and it is a result that
was simply unobtainable before the fix.

## The original measurement

### Original (pre-fix) table

| half-spread | net P&L | gross (ex-fees) | fills | fills/h | per fill | 5 s markout | end inventory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 (shipped) | −109.20 | −77.22 | 5,392 | 29.1 | −0.0203 | −74.59 | −46 |
| 4 | −88.91 | −57.53 | 5,219 | 28.2 | −0.0170 | −65.48 | −43 |
| 8 | −61.53 | −31.59 | 4,956 | 26.8 | −0.0124 | −49.99 | −42 |
| 16 | −62.29 | −36.36 | 4,100 | 22.2 | −0.0152 | −55.36 | −379 |
| 24 | −62.50 | −41.79 | 3,168 | 17.1 | −0.0197 | −50.73 | −29 |
| **40** | **+40.78** | +51.88 | 1,686 | 9.1 | +0.0242 | −43.88 | −354 |
| **60** | **+91.84** | +96.21 | 685 | 3.7 | +0.1341 | −23.54 | **−2,180** |

The P&L is **realized, not paper**: spread40 is +40.17 realized of +40.78 net;
spread60 is +91.70 of +91.84. Ending inventory is a residual, not the source.

The mechanism is visible in the markout column — 5 s markout improves
monotonically from −74.59 to −23.54. Wider quotes are picked off less. On this
tape, moving away from the touch reduces pick-off enough for the widest rungs to
become positive.

## Per-window: the number that matters

Sixteen 12-hour windows, P&L per window:

| rung | windows positive | best | worst | total | total excluding the last window |
|---|---:|---:|---:|---:|---:|
| 8 | **0/16** | −1.3 | −7.7 | −61.6 | −53.9 |
| 40 | 10/16 | +32.8 | −10.7 | +40.2 | **+7.4** |
| 60 | 10/16 | +30.3 | −21.7 | +95.1 | **+70.1** |

**At 8 bps every one of these 16 windows loses.** The consistent sign is stronger
than an aggregate total, but it remains evidence from one tape rather than a
regime-independent law.

**spread40's total is 82% one window** (+32.8 of +40.2). Strip the final window
and it is +7.4 over 15 windows: marginal, not established.

**spread60 survives the strip** (+70.1 over 15 windows) with gains spread across
windows 3, 9, 11, 12, 13, 15 and 16. It is not dominated by the final window in
this sample, but its window-to-window range is −21.7 to +30.3, far wider than
anything narrower.

## Honest limits

- **spread60 ends short 2,180 units — 386 USDC notional against 297.88 equity,
  130%.** That is a leveraged directional position, not a market-making book.
  It accumulated slowly enough that no risk limit caught it. Any real use of
  this rung needs an inventory cap that actually binds, and the fact that
  `q_max` did not bind here is itself a finding.
- **685 fills over 185 h is 3.7/h.** Thin. The per-window sign test (10/16) is
  the strongest claim the sample supports; the point estimate is not precise.
- **60 bps half-spread means quoting 120 bps wide** against a median book
  spread of 12.9 bps — roughly ten times outside the touch. Fills come from
  sweeps, so this is closer to liquidity provision of last resort than to
  making a market.
- **One tape, one instrument, one regime.** The 185 h contains a −70% cascade
  and a +50% grind. A different regime may rank the ladder differently.
- Each simulated decision/ack/cancel delay is held at 150 ms. The post-fix matrix
  above shows that cadence and width interact non-monotonically; there is no
  general evidence that slower refresh is beneficial.

## What this changes

The current 18-variant grid carries 40/48/60 bps rungs and a limited
spread-by-cadence matrix. Its completed runs support **a broad 24–60 bps region**
over the 1.5–8 bps controls, but do not rank the wide rungs and do not eliminate
directional inventory risk. The remaining question is whether that region stays
positive across substantially more independent regimes with inventory and
feed-validity constraints active.
