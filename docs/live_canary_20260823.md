# Live canary, 2026-08-23 — CASHCAT real money

Two real-money sessions on mainnet CASHCAT, minimum order size, 100 USDC
maximum open position. Latency was expected to be poor: this is a Windows
desktop, not an AWS Tokyo VPS. The purpose was to surface defects the test
suite cannot, and it did.

## Session A — the defect run (~40 min, killed)

Ran on a binary predating that morning's rate-limit and resilience commits.

| | |
|---|---|
| cancel cloids | 3,815 |
| of which the venue rejected as already gone | **2,440 (64%)** |
| average cancel batch | 2.5 climbing to 5.8 |
| address actions | 129.8/min |
| net | −0.174 USDC, 0 fills |

Every rejection read `Order was never placed, already canceled, or filled.
asset=231`. Individually benign; the volume was the signal. Two compounding
defects, both fixed in `9769354`:

1. `active_orders_by_side` returns everything non-terminal and `CancelPending`
   is not terminal, so an order with a cancel already in flight was cancelled
   again on the next quote cycle.
2. The venue's rejection was recorded as `UnknownOutcome`, also not terminal,
   so the cloid never left the cancel set. The batch grew without bound.

Defect 2 was heading somewhere worse than waste: past 16 unresolved orders,
`fetch_authoritative_snapshot` refuses its bounded REST fan-out, which would
have ended reconciliation for the remainder of the session.

Clearing the exposure exposed a third defect: `live-flatten` could not open a
store whose configuration had drifted while orders were still working — the
exact state it exists to clear. `LiveStateStore::open` now takes a
`SessionIntent`; a reduce-only session opens where a quoting session is
refused, and the fingerprint is adopted only once the store is flat.

## Session B — the fixed binary (~40 min, stopped on budget)

| | Session A | Session B |
|---|---|---|
| stale cancel share | 64% | **11%** |
| average cancel batch | 2.5 → 5.8 (climbing) | 2.34 → 1.98 (**falling**) |
| orders resting / errors | — | 1,016 / 1 |
| refused actions | 0 | 0 |
| fills | 0 | 3 |
| net | −0.174 USDC | **+0.005 USDC** |

The residual 11% is the legitimate kind: a cancel losing the race to a fill, or
to an order the venue dropped itself.

Two degradation paths were exercised for real and both behaved:

- **Network.** One reconcile failure against `api.hyperliquid.xyz/info`, 4 s
  backoff, recovered, no repeat. The session never stopped.
- **Latency gate.** Tripped at `public_ws_ping_rtt` p95 = 4,666 ms against a
  2,000 ms limit, withdrew quotes, left a −95 unit short unhedged (≈11 USDC),
  then cleared three healthy windows and resumed on its own. Inventory closed
  back to flat.

## Why it stopped: the address-action budget

Not a bot fault and not a per-run limit. The venue allowance is cumulative for
the life of the account and never resets:

```
cap        = 10,000 + 1 per USDC of cumulative volume = 13,060
used       = 12,941      (lifetime)
remaining  =    119
```

The backend's own reserve guard fired below 100 remaining — suspend new orders,
keep cancels allowed, 30 s cooldown — and **held** the reserve at ~119 rather
than draining it, leaving room to flatten. Correct behaviour.

Ending state, verified directly against the venue: 0 open orders, no positions,
297.880787 USDC.

## What this changes

The binding constraint on this strategy is the address-action budget, not the
WebSocket message rate the configuration validates against. At ~2 actions per
requote and ~11 USDC notional per fill, allowance breaks even at roughly one
fill per 5.5 requotes; this run managed one per 350. See the action-cost
section added to `requote_hysteresis_sweep.md` — the shipped
`replace_threshold_bps = 2.0` costs 220 actions per fill, and `4.0` costs 120.

Further live validation on this account is blocked: 115 requests remain and the
only way to earn more is volume, which itself costs requests. A fresh
subaccount or testnet is required.
