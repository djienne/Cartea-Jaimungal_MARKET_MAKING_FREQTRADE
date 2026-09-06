# Model Units

The Rust runtime and Python replay/reference code must use these units
consistently.

| Quantity | Unit |
| --- | --- |
| Price | USDC per base asset |
| Depth / delta | USDC |
| Epsilon | USDC |
| Kappa | 1 / USDC |
| Lambda | market orders / second per side, multiplied by that side's survival-fit intercept `A` (schema v5), so `lambda * exp(-kappa * depth)` is the measured fill intensity; the unscaled rate is `lambda_raw`. Prints sharing side + exchange timestamp are one MO. |
| HJB horizon `T` | seconds |
| Inventory `q` | physical base position divided by the base amount represented by one inventory unit |
| inventory unit | Rust stores `inventory_unit` in venue size quanta and converts it to base units for the HJB; Python calls the base amount `inventory_unit_base`. It is derived while flat as `available_capital_usdc * target_capital_utilisation * leverage / (q_max * mid)`, rounded down to the venue size quantum. The runtime refuses a derived unit below the venue minimum notional and preserves the existing unit while inventory is non-zero. In `cashcat.toml`, 1000 USDC capital gives about 247 USDC notional per unit; `cashcat_dryrun_realistic.toml` intentionally uses less capital and therefore a smaller unit. |
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
phi_base      = phi_kappa_t / (kappa_average * T)       # when target > 0
phi_effective = phi_base + volatility_risk_coefficient * sigma2_per_sec * inventory_unit_base
phi_effective = min(phi_effective, phi_kappa_t_max / (kappa_average * T))
```

Derivation: the HJB value function `h` is expressed per inventory unit (one
unit = `inventory_unit_base` base asset). A mid-variance penalty on the real
USDC exposure `q * inventory_unit_base` scales as `sigma2 * unit^2 * q^2` in
USDC; dividing by one factor of `unit` to match `h`'s per-unit normalization
leaves `gamma * sigma2 * unit * q^2`. Rust names the coefficient
`volatility_risk_coefficient`; Python names the parity field
`gamma_inventory_risk`. Its default is 0.05, but that is a model choice rather
than a measured universal constant. When `sigma2_per_sec` is missing or invalid,
the volatility increment is zero and the configured base penalty remains.

The Python reference can still reproduce a legacy long-only inventory domain:

```text
q_exact = clip(max(0, signed_base_position) / inventory_unit_base, 0, hjb_q_max)
q       = round(q_exact)
```

With `allow_short=true`, Python uses the signed value and clips to
`[-q_max, +q_max]`. `rust_live/` always quotes both sides from one signed
inventory. The long-only option and `mm_core.route_sides` remain only to
reproduce historical replay cases; they are not the current trader architecture.

## Time and inventory: how `delta*(t,q)` is actually read

The book's control is `delta*(t,q)` on `[0,T]` with terminal condition
`h(T,q) = -alpha*q^2` (eq. 10.26). Two coordinates, two conventions:

| | how it is read | why |
| --- | --- | --- |
| `t` | Rust always returns the whole surface and reads the episode's real time-to-go, interpolated between nodes. Python calls this `hjb_time_mode="episodic"`; its `"stationary"` option is retained for controlled comparisons and reads only `t=0`. | Reading `t=0` forever means the agent never approaches `T`, `alpha` barely affects the control, and the time axis of eq. 10.26 is discarded. |
| `q` | Depths are blended linearly between the bracketing integers. Exact at every integer `q`. | Eq. 10.2 is a unit-jump process, so `h` exists only at integer `q`; partial fills land in between. Rounding hides up to half a unit of live risk — 0.49 units reads as flat. Interpolating **depths** not `h`: `delta` reads a *difference* of `h`, so piecewise-linear `h` gives piecewise-*constant* depths, no better than rounding. |

`q_exact` and `q_rounded` are recorded in quote diagnostics; Python replay also
reports mean `|q_exact - round(q_exact)|`. This is an interpretability diagnostic,
not a current acceptance threshold.

### Named departures from the book

1. **No HJB-forced liquidation at `T`.** The book liquidates the residual at
   market and pays `alpha*q^2`; the implemented terminal condition acts through
   quote depths and the episode clock restarts. Ordinary quotes remain post-only.
   Several dry-run-grid rows (`flatten_after_ms`) and the optional
   `live.flatten_after_ms` test a separately accounted timed taker exit, but that
   policy is outside the HJB.
2. **Episodes restart on a real clock**, at `T` or once flat past
   `episode_min_elapsed_fraction * T`. A perpetual instrument has no natural
   terminal time; the book's agent starts flat at `t=0`, so reaching flat is the
   natural place to restart.
3. **The outermost inventory interval does not quote the adding side.** Any
   non-finite bracketing node disables the blend, so with `q_max=6` a bid stops
   above `q=5.0` rather than `q=5.5`. Conservative: a bid at `q=5.9` would permit
   a jump to 6.9, past the boundary.
4. **A fee cushion, bps clamps, leverage and margin** are all outside the model
   entirely. See "Quote assembly" above and `cj-core/src/quote.rs`.

### What the terminal condition actually does at our calibration

Not the textbook "flatten harder as `t -> T`" picture, because the *running*
penalty dwarfs the terminal one. The shipped config runs
`phi*kappa*T = 300` (ceiling `phi_kappa_t_max = 450`) against
`alpha*kappa = 0.05`, a factor of 6000. Historical Python sweeps used 10, a
factor of 200; those artifacts are dated and should not be read as the current
Rust profile. In either case the running penalty is what remains to be **paid**
over the time left, so it is largest at `t=0` and vanishes at `T`. Measured at
`q=+3` on live-scale CASHCAT parameters at `phi*kappa*T = 10`, the ask depth
runs `-6.8e-6 -> +1.0e-4` as `tau` goes `150s -> 0`: the agent unwinds hardest
at the **start** of an episode and relaxes into the terminal. That is the
correct reading of eq. 10.26 here, and raising `phi` only deepens it.

**This matches the book rather than contradicting it.** An earlier version of
this note claimed the book's figures show the opposite; they do not. Fig. 10.8
(p. 265) plots depths for `q = -2..3` fanned wide at `t=0` and converging into
`T` — the same shape — and its stated parameters are `phi=0.02, kappa=100, T=30,
alpha=0.0001`, i.e. `phi*kappa*T = 60` against `alpha*kappa = 0.01`, a ratio of
6000:1. The book's own illustration is even more running-penalty-dominated than
ours, so the agreement is expected. Checked against the book PDF 2026-08-17.

Consequence: Python's `hjb_alpha_kappa` / Rust's `alpha_kappa` was chosen while
`alpha` was effectively inert. Episodic control makes it mathematically active,
but at the current `phi*kappa*T=300` its influence is confined to about the
final 12 seconds (3.5 s at phi=1000), measured on the shipped config. A later
sweep returned bit-identical P&L at 0.05, 0.5, and 5.0, so this
regime cannot tune it; revisit only with a lower running penalty.

### Solver resolution

`n_steps` scales with `T` to hold `dt <= max_dt_seconds` (0.25 s), bounded by
`min_steps`/`max_steps` (the Python aliases retain an `hjb_` prefix). Backward
Euler is first order and its error at a fixed
time-to-go `tau` grows as `tau` shrinks — at `T=150s, kappa=100, alpha=0.01` the
depth error is 1e-9 at `tau=75s`, 2e-4 at `tau=10s`, 9e-3 at `tau=0.75s` with
`n_steps=200`. Treat the final `dt` of an episode as unresolved.
