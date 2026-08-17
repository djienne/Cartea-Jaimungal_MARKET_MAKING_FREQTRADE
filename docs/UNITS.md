# Model Units

The strategy and replay code must use these units consistently.

| Quantity | Unit |
| --- | --- |
| Price | USDC per base asset |
| Depth / delta | USDC |
| Epsilon | USDC |
| Kappa | 1 / USDC |
| Lambda | market orders / second (per side; prints sharing side + exchange timestamp are one MO) |
| HJB horizon `T` | seconds |
| Inventory `q` | `inventory_unit_base` units of base asset |
| `inventory_unit_base` | base asset amount represented by one inventory step. **Auto-derived** from `available_capital * target_capital_utilisation * target_leverage / (q_max * mid)`, recomputed only while flat, so it carries across symbols (~2430 CASHCAT ~ 247 USDC at the shipped settings). Must track the actual order amount = min(stake, unit×mid)/mid, and must stay above the exchange's 10-USDC minimum notional — the strategy emits `inventory_unit_mismatch` when the rounded order amount drifts >25% from the unit) |
| `sigma2_per_sec` | USDC² / second (variance of 1 s mid increments, per base asset) |
| `min_half_spread_bps` / `max_half_spread_bps` | basis points of mid (1e-4), clamps on the final half-spread including the fee cushion |

## Quote assembly

```text
delta_total = clamp(delta_model * spread_multiplier + maker_fee * mid + extra_cushion_bps/1e4 * mid,
                    min_half_spread_bps/1e4 * mid, max_half_spread_bps/1e4 * mid)
```

`delta_model` is the HJB depth in USDC; the fee cushion (`maker_fee * mid`) is
one maker fee per side, so a filled round trip collects the cushion twice and
pays the fee twice.

## Volatility-aware inventory penalty

```text
phi_effective = hjb_phi + gamma_inventory_risk * sigma2_per_sec * inventory_unit_base
```

Derivation: the HJB value function `h` is expressed per inventory unit (one
unit = `inventory_unit_base` base asset). A mid-variance penalty on the real
USDC exposure `q * inventory_unit_base` scales as `sigma2 * unit^2 * q^2` in
USDC; dividing by one factor of `unit` to match `h`'s per-unit normalization
leaves `gamma * sigma2 * unit * q^2`. `gamma_inventory_risk` is the
dimensionless risk-aversion knob (default 0.05, conservative; tune from replay
markout evidence). When `sigma2_per_sec` is missing or invalid the strategy
falls back to the static `hjb_phi` — the volatility channel must never stop
quoting.

For a long-only leg:

```text
q_exact = clip(max(0, signed_base_position) / inventory_unit_base, 0, hjb_q_max)
q       = round(q_exact)
```

A two-sided leg (`can_short=true`) uses the signed value and clips to
`[-hjb_q_max, +hjb_q_max]`. If the Freqtrade strategy sees a position pointing
the wrong way for its role, it must fail closed.

## Time and inventory: how `delta*(t,q)` is actually read

The book's control is `delta*(t,q)` on `[0,T]` with terminal condition
`h(T,q) = -alpha*q^2` (eq. 10.26). Two coordinates, two conventions:

| | how it is read | why |
| --- | --- | --- |
| `t` | `hjb_time_mode="episodic"`: the solver returns the whole surface and the quote uses the slice at the episode's real time-to-go, interpolated between time nodes. `"stationary"` reads only the `t=0` slice. | Reading `t=0` forever means the agent never approaches `T`, `alpha` never bites, and the time axis of eq. 10.26 is decorative. |
| `q` | Depths are blended linearly between the bracketing integers. Exact at every integer `q`. | Eq. 10.2 is a unit-jump process, so `h` exists only at integer `q`; partial fills land in between. Rounding hides up to half a unit of live risk — 0.49 units reads as flat. Interpolating **depths** not `h`: `delta` reads a *difference* of `h`, so piecewise-linear `h` gives piecewise-*constant* depths, no better than rounding. |

`q_residual = q_exact - q` is logged on every quote decision and fill, and
`verify_dry_run_quality.py` fails a run whose mean `|q_residual|` exceeds 0.35.

### Named departures from the book

1. **No forced liquidation at `T`.** The book liquidates the residual at market
   and pays `alpha*q^2`. Every quote here is post-only by construction, so
   taking liquidity to flatten would contradict the rest of the design. The
   terminal condition acts through the depths alone and the clock restarts.
2. **Episodes restart on a real clock**, at `T` or once flat past
   `hjb_episode_min_elapsed_fraction * T`. A perpetual instrument has no natural
   terminal time; the book's agent starts flat at `t=0`, so reaching flat is the
   natural place to restart.
3. **The outermost inventory interval does not quote the adding side.** Any
   non-finite bracketing node disables the blend, so with `q_max=6` a bid stops
   above `q=5.0` rather than `q=5.5`. Conservative: a bid at `q=5.9` would permit
   a jump to 6.9, past the boundary.
4. **A fee cushion, bps clamps, leverage and margin** are all outside the model
   entirely. See "Quote assembly" above and `leverage()`.

### What the terminal condition actually does at our calibration

Not the textbook "flatten harder as `t -> T`" picture, because the *running*
penalty dwarfs the terminal one: `phi*kappa*T = 10` against `alpha*kappa = 0.05`,
a factor of 200. The running penalty is what remains to be **paid** over the
time left, so it is largest at `t=0` and vanishes at `T`. Measured at `q=+3` on
live-scale CASHCAT parameters, the ask depth runs `-6.8e-6 -> +1.0e-4` as `tau`
goes `150s -> 0`: the agent unwinds hardest at the **start** of an episode and
relaxes into the terminal. That is the correct reading of eq. 10.26 here.

**This matches the book rather than contradicting it.** An earlier version of
this note claimed the book's figures show the opposite; they do not. Fig. 10.8
(p. 265) plots depths for `q = -2..3` fanned wide at `t=0` and converging into
`T` — the same shape — and its stated parameters are `phi=0.02, kappa=100, T=30,
alpha=0.0001`, i.e. `phi*kappa*T = 60` against `alpha*kappa = 0.01`, a ratio of
6000:1. The book's own illustration is even more running-penalty-dominated than
ours, so the agreement is expected. Checked against the book PDF 2026-08-17.

Consequence: `hjb_alpha_kappa` was tuned while `alpha` was effectively inert and
now is not. Treat it as an untuned knob.

### Solver resolution

`n_steps` scales with `T` to hold `dt <= hjb_max_dt_seconds` (0.25 s), bounded by
`hjb_n_steps_min`/`_max`. Backward Euler is first order and its error at a fixed
time-to-go `tau` grows as `tau` shrinks — at `T=150s, kappa=100, alpha=0.01` the
depth error is 1e-9 at `tau=75s`, 2e-4 at `tau=10s`, 9e-3 at `tau=0.75s` with
`n_steps=200`. Treat the final `dt` of an episode as unresolved.
