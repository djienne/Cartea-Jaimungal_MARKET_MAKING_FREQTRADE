# Spread ladder over 185 h — the first profitable configuration

Replay of `min_half_spread_bps` from 1.5 to 60 over a frozen 185.1 h CASHCAT
tape (08-16 21:59 → 08-24 15:07, 2,401 shards per stream). Everything else is
the shipped `cashcat_dryrun_realistic.toml`: same latency (150/150/150 ms), same
fees, same funding, flow guard on. One lever moves.

Run with `scripts/guard_study/make_spread_configs.py` + `mm-live replay`.

## Result

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
monotonically from −74.59 to −23.54. Wider quotes are picked off less. That is
the entire story: CASHCAT's adverse selection is severe enough that the only
way to survive it is to stop competing for the touch.

## Per-window: the number that matters

Sixteen 12-hour windows, P&L per window:

| rung | windows positive | best | worst | total | total excluding the last window |
|---|---:|---:|---:|---:|---:|
| 8 | **0/16** | −1.3 | −7.7 | −61.6 | −53.9 |
| 40 | 10/16 | +32.8 | −10.7 | +40.2 | **+7.4** |
| 60 | 10/16 | +30.3 | −21.7 | +95.1 | **+70.1** |

**At 8 bps every single window loses.** Not variance — structure.

**spread40's total is 82% one window** (+32.8 of +40.2). Strip the final window
and it is +7.4 over 15 windows: marginal, not established.

**spread60 survives the strip** (+70.1 over 15 windows) with gains spread across
windows 3, 9, 11, 12, 13, 15 and 16. It is the only rung whose profit is not an
artefact of one move — but its window-to-window range is −21.7 to +30.3, far
wider than anything narrower.

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
- Latency is held at 150 ms throughout. The replay's separate latency ladder
  showed cadence is worth up to 407 USDC on its own, and the two levers have
  not been crossed here.

## What this changes

The live 20-variant grid already carries `wide40` and the `wide*slow*` matrix,
so the out-of-sample check is running. The replay says the interesting region
is **40–60 bps**, well beyond where the grid's ladder used to stop (8), and
that the shipped 1.5 bps floor is the worst rung tested by a wide margin.

Next question, not answered here: whether cadence composes with width, and
whether an inventory cap that actually binds preserves spread60's edge or
removes it.
