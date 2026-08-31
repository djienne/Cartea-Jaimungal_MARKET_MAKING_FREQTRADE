# CASHCAT real-money execution and config-switch test — 2026-08-31

Purpose: exercise the important live pathways independently of strategy
profitability. Continuous live remained blocked because the corrected dry-run
grid had no strictly profitable row.

## Evidence

- The initial baseline attempt exposed duplicate startup REST reconciliation;
  no order was sent and the account remained flat.
- Clean, flat startup now uses the fully acknowledged eight-channel account
  WebSocket snapshot. REST is a five-minute drift audit or degraded recovery
  path, not a parallel polling loop.
- Baseline placed a real two-sided ALO batch at minimum-live size. Both orders
  rested and were canceled successfully.
- While flat, the same durable state accepted a config-fingerprint change to
  `wide4`. A new real two-sided ALO batch rested and canceled successfully.
- The switch run finished scientifically valid with no unknown outcomes, event
  loss, or residual inventory. Maximum working gross was 21.1977 USDC and
  maximum directional exposure was 10.59777 USDC.
- A controlled 54-unit IOC buy filled at 0.19679 and the reduce-only IOC close
  filled 54 units at 0.19662. Combined equity change was -0.018738 USDC.
- Final account checks, separated in time, both reported zero CASHCAT position,
  zero open orders, and a healthy account WebSocket. No live container or arm
  marker remained.
- The final-image `wide4` rerun produced exactly one successful placement batch
  and one successful cancel batch, with no duplicate terminal cancel and no
  REST reconciliation error. Address allowance finished at 114 requests above
  usage. The filled turnover replenished most of the bounded test spend.

## Transport decision

Typed account subscriptions remain the steady-state authority. `webData3` is a
larger frontend aggregate whose documented shape warns that undocumented fields
will be removed. Signed actions use WebSocket `post`; REST remains only for
identity, fees, `userRateLimit`, initial flatness, and recovery because those
queries have no equivalent stable typed stream. WebSocket info `post` is a
valid future recovery optimization, but adding a second correlation path is not
justified now that normal startup and steady state require no reconciliation
fan-out.

The local-node/order-book-server option was also rejected for this account: the
official guidance calls for approximately 32 logical cores and high disk
throughput, disproportionate to minimum-notional, quota-constrained operation.
