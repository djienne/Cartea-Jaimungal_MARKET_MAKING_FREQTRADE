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

The selector takes the valid row with the highest `promotion_pnl_usdc`, even
when every row is negative. It writes `rust_live/run/cashcat-active-live.toml`
and `cashcat-promotion.json` atomically.

Live orders are the first valid lot between 1.05 and 1.10 times the current
CASHCAT minimum notional. Directional exposure is one such order, working gross
is two, and the daily realised-loss stop is 1 USDC. Quote calculations continue
normally, but the executor coalesces intermediate targets and paces placements
from the venue-reported address allowance while preserving 100 placement
actions plus ten scheduled safety actions. Because this account currently has
very little placement allowance, the venue dead-man uses an eight-hour deadline
refreshed every six hours; the one-minute host watchdog is the primary fast
recovery path. Cancels have separate accounting and are never blocked by the
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
