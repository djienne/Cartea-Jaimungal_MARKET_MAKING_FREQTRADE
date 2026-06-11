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
| `inventory_unit_base` | base asset amount represented by one inventory step (0.01 ETH; must track the actual order amount = min(stake, unit×mid)/mid, and must stay above the exchange's 10-USDC minimum notional — the strategy emits `inventory_unit_mismatch` when the rounded order amount drifts >25% from the unit) |
| `sigma2_per_sec` | USDC² / second (variance of 1 s mid increments, per base asset) |
| `min_half_spread_bps` / `max_half_spread_bps` | basis points of mid (1e-4), clamps on the final half-spread including the fee cushion |

## Quote assembly

```text
delta_total = clamp(delta_model * spread_multiplier + maker_fee * mid,
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

For the current long-only Freqtrade implementation:

```text
q = round(max(0, signed_base_position) / inventory_unit_base)
q = clip(q, 0, hjb_q_max)
```

Negative inventory is only valid for the model solver and future signed market
making. If the Freqtrade strategy sees a short position while `can_short` is
false, it must fail closed.
