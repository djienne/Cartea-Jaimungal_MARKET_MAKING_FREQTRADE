# CASHCAT live strategy

**There is no automatic promotion.** The live configuration is
`rust_live/config/cashcat.toml`. Its `live.paper_strategy` explicitly selects
`sweep1_flat300` from the existing grid definition and checkpoint. The shared
resolver supplies the model, frozen fit, quote rules, flow guard and exit age;
the live file holds the allocation and venue safeguards. Live remains disabled.

The allocation ceiling is 100 USDC at 2x, with a 200-USDC directional notional
ceiling (including pending cancellations), 100-USDC margin ceiling and the
stricter 1-USDC daily realized-loss stop. The inventory unit is
`floor(paper_unit * live_allocation / paper_allocation)`: 636 becomes 213 units
at 100/297.88 capital, about 36 USDC per entry near a 0.17 price. Actual account
equity can reduce the allocation; venue available-to-trade limits still apply.
An allocation below the minimum viable order fails rather than silently sizing up.
Paper balances, fills and P&L are never copied into the live account.

Frozen parameters are not periodically refitted or expired as rolling estimates.
Recent collector data still supplies the initial VPIN volume scale. Changes to
the referenced fit or sizing unit participate in live state identity. Read-only
`validate` prints the resolved strategy. Emergency `live-flatten` deliberately
does not require the paper files, so missing research data cannot block cleanup.

The configured age is 1 ms after inventory observation, not an artificial
301-ms or six-second delay. Live safety exits reconcile and close the actual
remaining position; paper ages lots FIFO and waits for recorded depth. Quotas,
latency admission, real fees, acknowledgements and fill outcomes remain live
constraints. Sharing a strategy does not imply identical execution or P&L.

Automatic promotion existed until 2026-09-10: `promote-best` picked the highest
`promotion_pnl_usdc` row and generated a derived `cashcat-active-live.toml`. It
was removed as unnecessary and confusing -- a generated config nobody edited,
selected by a rule whose exclusions had to be explained every time, and a
supervisor that could swap the running strategy every twelve hours. Choosing
what trades real money is a decision to make deliberately, not one to automate
off a leaderboard.

`promotion_pnl_usdc` survives as the leaderboard's **ranking metric**, and it is
the number to read: it marks residual inventory out at a pessimistic 25 bps, so
a row whose P&L is really an open directional position ranks below one that
actually took the money. `eligible_for_promotion` likewise still marks rows with
no live equivalent -- a paper lot-age exit or a frozen `parameter_profile` --
which is a fact about the row, not a promotion verdict.

The grid ranks valid rows by that metric, leaves queue position unknown until a
quote price is visible, uses no uncalibrated time-based queue decay, drives
simulation from exchange time with measured latency tails, and persists open
gaps, event loss and daily loss across restarts. Each scientific run lives below
`rust_live/reports/grid_live/runs/`; the root `leaderboard.json` is the
healthcheck's latest pointer. Pre-2026-08-31 grid artifacts predate all of that
and must not be used.

Profiles without a paper reference retain the micro-live order-size limits.
The linked profile preserves its proportional model quantity. Quote calculations continue
normally, but the executor coalesces intermediate targets and paces placements
from the venue-reported address allowance while preserving 100 placement
actions plus ten scheduled safety actions. The venue dead-man runs an 8 h
deadline refreshed every 6 h for budget reasons
(`rust_live/HYPERLIQUID_LIVE_CONNECTOR.md` §7.5); the one-minute host watchdog
is the primary fast recovery path. Cancels have separate accounting and are never blocked by the
ordinary placement throttle.

`Canary` runs `config/cashcat.toml` for 7,200 seconds and always runs
`live-flatten` afterwards. It writes the pass evidence only when the full
duration, at least one fill, zero unknown/rejected actions, operational validity,
successful shutdown and final flatness all hold. A recovered private socket
reconnect retains its scientific discontinuity flag but can pass the operational
verdict after reconciliation; event loss or an unresolved fault cannot.
`Arm` refuses without that evidence. Once armed, the Windows supervisor checks
health every minute and does exactly one thing: it restarts an exited or
unhealthy container, flattening first. It never changes the configuration --
that is yours to edit, and a change takes effect on the next start -- and it
makes no economic judgement.

It used to make one, stopping live unless the best `eligible_for_promotion` row
had positive `promotion_pnl_usdc`. That rule only meant something while
`promote-best` generated the live config from that same row. Once the config
became hand-edited, the shipped `sweep1_flat300` was permanently ineligible and
the gate was reading an unrelated row to decide its fate, so it was removed
rather than re-pointed at a row name the config would have to keep claiming
truthfully. What stops a losing live session is measured on the live account:
`production_max_daily_realized_loss_usdc` (1 USDC) pauses new placements off
realised P&L, and `Disarm` stops it deliberately.

That switch was exercised on real money on 2026-08-31: while flat, the durable
state accepted a config-fingerprint change to `wide4` and the new two-sided ALO
batch rested and cancelled cleanly, with one placement batch, one cancel batch,
no duplicate terminal cancel and no REST reconciliation error. Maximum working
gross was 21.1977 USDC, maximum directional exposure 10.59777 USDC, and a
54-unit IOC round trip (0.19679 in, 0.19662 out) cost 0.018738 USDC. The run's
one defect was a duplicate startup REST reconciliation, now replaced by the
fully acknowledged eight-channel account WebSocket snapshot; REST is a
five-minute drift audit, not a parallel polling loop.

```powershell
scripts\Manage-CashcatLive.ps1 -Action Canary
scripts\Manage-CashcatLive.ps1 -Action Arm
scripts\Manage-CashcatLive.ps1 -Action Status
scripts\Manage-CashcatLive.ps1 -Action Disarm
```

## Bounded operational profile

`rust_live/config/cashcat_canary.toml` is the reusable, default-disabled profile
for a supervised four-hour execution experiment. It uses isolated 1x, about
11 USDC directional exposure, a 0.25 USDC loss stop and timed reduce-only exits.
Its comments specify the differences from the paper winner; it does not test
the frozen `sweep_a` fit or paper execution timing. After explicitly enabling
it, run `mm-live --config rust_live/config/cashcat_canary.toml live
--duration-seconds 14400` and then the same config with `live-flatten`.
The paper grid and collectors keep their own state and processes.

Live reports include resolved settings, `stop_reason`, `shutdown_succeeded` and
`operationally_valid`. `execution.operationally_healthy` is current state and is
false after orderly shutdown. Historical `invalid_reason` remains visible;
heartbeat `active_fault_reason` is null after recovery. Latency `session_summary`
retains counts, extrema and means with constant memory; rolling percentiles
still control admission. `ack_to_fill` measures the first observed fill after
an acknowledgement in this process and includes normal resting time, so it is
reported but never used as a transport gate. Snapshot/replayed fills have no
invented acknowledgement time.

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
