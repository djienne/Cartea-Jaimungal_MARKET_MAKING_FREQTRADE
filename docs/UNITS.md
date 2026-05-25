# Model Units

The strategy and replay code must use these units consistently.

| Quantity | Unit |
| --- | --- |
| Price | USDC per base asset |
| Depth / delta | USDC |
| Epsilon | USDC |
| Kappa | 1 / USDC |
| Lambda | events / second |
| HJB horizon `T` | seconds |
| Inventory `q` | `inventory_unit_base` units of base asset |
| `inventory_unit_base` | base asset amount represented by one inventory step |

For the current long-only Freqtrade implementation:

```text
q = round(max(0, signed_base_position) / inventory_unit_base)
q = clip(q, 0, hjb_q_max)
```

Negative inventory is only valid for the model solver and future signed market
making. If the Freqtrade strategy sees a short position while `can_short` is
false, it must fail closed.
