# CASHCAT corrected-grid promotion and micro-live operation

The pre-2026-08-31 grid artifacts are historical only. They used zero queue
ahead outside the visible book, time-erased queue after twenty seconds, and
could lose feed-invalid state across a restart. They must not be promoted.

The corrected grid:

- leaves queue position unknown until a quote price is visible;
- uses no uncalibrated time-based queue decay;
- drives simulation from exchange time and includes deterministic measured
  latency tails;
- persists open gaps, event loss and daily loss across restarts;
- ranks valid rows by executable-side, fee-adjusted flatten P&L;
- stores each scientific run below `rust_live/reports/grid_live/runs/` while the
  root `leaderboard.json` remains the healthcheck's latest pointer.

After at least 43,200 seconds, generate the live configuration with:

```powershell
scripts\Manage-CashcatLive.ps1 -Action Promote
```

The selector takes the valid, live-equivalent row with the highest
`promotion_pnl_usdc` and requires that value to be positive; otherwise live
remains disabled. Rows with `flatten_after_ms > 0` or a fixed
`parameter_profile` are excluded: promote-best does not translate a paper
lot-age exit into `live.flatten_after_ms`, and a frozen fit is not a live
calibration. A successful selection writes
`rust_live/run/cashcat-active-live.toml` and `cashcat-promotion.json` atomically.

Live orders are the first valid lot between 1.05 and 1.10 times the current
CASHCAT minimum notional. Directional exposure is one such order, working gross
is two, and the daily realised-loss stop is 1 USDC. Quote calculations continue
normally, but the executor coalesces intermediate targets and paces placements
from the venue-reported address allowance while preserving 100 placement
actions plus ten scheduled safety actions. The venue dead-man runs an 8 h
deadline refreshed every 6 h for budget reasons
(`rust_live/HYPERLIQUID_LIVE_CONNECTOR.md` §7.5); the one-minute host watchdog
is the primary fast recovery path. Cancels have separate accounting and are never blocked by the
ordinary placement throttle.

`Canary` runs the selected production pathway for 7,200 seconds and always runs
`live-flatten` afterwards. It writes the pass evidence only when the full
duration, at least one fill, zero unknown/rejected actions, scientific validity,
and final flatness all hold. `Arm` refuses without that evidence. Once armed,
the Windows supervisor checks health every minute and promotion every twelve
hours. A changed winner is applied only after stop, cancel, flatten and flat
verification; failures leave live stopped.

```powershell
scripts\Manage-CashcatLive.ps1 -Action Canary
scripts\Manage-CashcatLive.ps1 -Action Arm
scripts\Manage-CashcatLive.ps1 -Action Status
scripts\Manage-CashcatLive.ps1 -Action Disarm
```

## Hyperliquid transport choices

- Typed `orderUpdates`, `userFills`, `userFundings`, `clearinghouseState`,
  `openOrders`, `activeAssetData`, ledger, and notification subscriptions are
  the steady-state account data plane. Their snapshot markers make reconnect
  recovery explicit. `webData3` was rejected because it is a larger frontend
  aggregate and its documented type explicitly warns that undocumented fields
  will be removed.
- Signed order/cancel/dead-man actions already use WebSocket `post`. The API also
  permits info requests through WebSocket `post`, but steady state does not need
  polling equivalents of subscribed streams. REST remains only for the initial
  identity/fee/quota/flatness snapshot and degraded-state drift recovery;
  `userRole`, `userFees`, and `userRateLimit` have no equivalent typed stream.
- Batching reduces IP request weight but not the per-address action count, so it
  is used for paired quotes without pretending it creates address quota.
- Nonce invalidation (`noop`) is useful for pending transactions, not as a
  replacement for confirmed `cancelByCloid` of resting orders. Safety cancels
  therefore keep their documented separate allowance.
- A local non-validating node/order-book server requires roughly 32 logical
  cores and high disk throughput. It would improve latency and depth but is
  disproportionate for this minimum-notional, quota-constrained test account;
  the existing public feed plus causal-loss gates remains the justified path.
