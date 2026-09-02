# Hyperliquid live execution connector reference

Research and local-code audit date: **2026-08-21**. Target runtime:
`rust_live`, initially validated only for **CASHCAT**.

This is the implementation reference for the pure-Rust real-money Hyperliquid
connector. The backend is selected only when the tracked default-off TOML flag
is explicitly enabled. Hyperliquid's API changes over time, so release work must
recheck the linked official documentation and pinned protocol fixtures.

## Remediation changes (2026-08-23)

A full-review remediation series (commits `43c4d6b..4dc3db4`) changed connector
behavior in ways that supersede older statements in this document:

- **Teardown always runs.** `run_live`'s loop errors no longer unwind past the
  shutdown sequence; cancel-resting-orders, reconcile, and dead-man clearing
  execute on every exit path, and the release profile unwinds on panic. Event
  log saturation, enqueue refusals, and refused pings degrade (pause quoting,
  request reconcile) instead of ending the session.
- **Cancel responses are attributed positionally.** Action state keeps one
  slot per wire order (`None` where untracked), so a venue status array can no
  longer be applied to the wrong order when a cancel raced a fill; mismatched
  status lengths fail closed to `UnknownOutcome`.
- **A cancel is sent once per outcome.** An order with a cancel still in
  flight is skipped rather than re-cancelled, and a venue rejection that proves
  the order left the book (`Order was never placed, already canceled, or
  filled`) is recorded as `Canceled`, not `UnknownOutcome`. Without both, each
  order was cancelled roughly three times: a live run spent 64% of its cancel
  traffic (2440 of 3815 cloids) on duplicates the venue could only reject, and
  the unresolved set grew until it would have tripped the authoritative
  snapshot's 16-order REST fan-out limit. Marking the order terminal cannot
  lose a fill — fills are booked from `userFills` keyed by fill id, never from
  a cancel response.
- **Durable state is bounded (schema 3).** Fill/funding dedup keys carry
  exchange time; the replay checkpoint advances after each authoritative
  reconcile (24h retention, fsynced before pruning) and aged terminal orders
  are pruned. The persistence writer writes deltas instead of rewriting every
  table per wake; nonce-range fsyncs are prefetched off the dispatch path.
- **Risk inputs are durable and live.** `consecutive_losses` is tracked from
  closing fills (it was previously pinned to zero in live, making
  `max_consecutive_losses` dead); daily P&L rolls are read through a scalar
  accessor with no full-state clone per event.
- **Clocks are venue clocks.** The inventory watermark compares fill times
  against exchange time only, and the dry-run simulator schedules activation
  and cancellation from `source_exchange_ms`, matching replay.
- **The session actor never blocks on the strategy loop.** Event delivery is
  non-blocking (drop-and-degrade on a full channel), removing a mutual-wait
  with awaited oneshot responses; one malformed account frame is skipped, not
  a reason to tear down the socket.
- **Requote hysteresis.** A resting order within
  `max(replace_threshold_ticks × quantum, replace_threshold_bps)` of the new
  target (same size and reduce-only) is held; withdrawals bypass the window.
  Evidence: `docs/requote_hysteresis_sweep.md`. `min_order_lifetime_ms` rose
  to 100 and config validation rejects WebSocket budgets the worst-case
  requote rate plus pings and dead-man refreshes cannot fit.
- **Latency gate un-latched.** Dropped-sample/observer-error blocks are
  per-window (deltas), not per-session (lifetime counters).

## Implementation status (updated 2026-08-23)

The persistent `live` command now owns a stateful execution backend and remains
fail-closed while `[live].enabled=false`. Implemented and exercised pieces are:

- zeroizing, non-debug four-key dotenv loading;
- fixed-point price/size formatting and strict CLOIDs;
- vault-aware MessagePack action hashing including the expiry separator;
- phantom-agent EIP-712/secp256k1 signing with independent golden vectors;
- crash-safe persisted nonce ranges and explicit transport-unknown outcomes;
- account/open-order/fill/fee/role/active-asset/book `/info` reads;
- WebSocket-post ALO/IOC batches, CLOID/OID cancel, isolated leverage, and REST
  emergency cancel;
- all eight account subscriptions, application heartbeat with measured RTT,
  protocol pong, bounded delivery, reconnect, and health metrics;
- normalized transactional order/fill/funding tables with coalesced background
  persistence, typed partial-fill lifecycle, account-aware sizing, rate
  reserves, restart reconciliation, and explicit market close; nonce ranges are
  fsynced before use while lifecycle telemetry never blocks the socket task;
- production-only rolling-p95 latency enforcement with probe, dry-run, replay,
  and feature-gated acceptance bypass;
- an explicitly guarded 12-directional/24-gross/60-turnover/0.5-loss campaign
  compiled into `mm-live-acceptance`, not the production binary.

Real evidence covers two-sided ALO, both cancel identifiers, post-only refusal,
response loss, hard restarts with order and position, a genuine maker fill, and
long/short reduce-only market close. The account ended flat and empty. The
tested subaccount could not use `scheduleCancel` on 2026-08-22 because the venue
required one million USDC cumulative volume; production with dead-man required
therefore refused on that account.

**Runtime invariant:** the live and dry-run traders are pure Rust. They must not
import, launch, or depend on Python, the Hyperliquid Python SDK, Passivbot, or
CCXT. Python/JavaScript implementations may generate offline test fixtures only.

## 1. Decision summary

The implemented design is:

1. `cj-core` owns the Cartea–Jaimungal HJB and quote policy without Tokio,
   Arrow/Parquet, HTTP, redb, or JSON dependencies. `mm-runtime` owns the
   dedicated hot thread, coherent atomics, and latency sampling.
2. `hyperliquid-connector` is the cold-path live boundary. It owns signing,
   WebSocket `post` requests, account subscriptions, order state,
   reconciliation, and authoritative account projection behind the generic
   execution traits.
3. Use a dedicated Hyperliquid API/agent wallet for one dedicated account or
   subaccount. Never put the master wallet key in the bot.
4. Use WebSocket action posts for normal order/cancel flow, but retain `/info`
   REST queries as the authoritative recovery and reconciliation path.
5. Give every order a unique 128-bit `cloid`. A timeout or disconnect after a
   send is an **unknown outcome**, never permission to submit another order
   blindly.
6. The runtime uses its pinned pure-Rust typed signer and golden wire fixtures.
   The official Python SDK is an independent offline oracle only and is absent
   from runtime; adopting another SDK later requires the same fixtures to pass.
7. Useful older local implementations remain comparison/test material, not
   runtime dependencies.

The general API index currently points to the official Python SDK and a community
Rust SDK; the Hyperliquid GitHub organization also maintains a Rust SDK. See the
[API index](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api),
[official Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk),
[Hyperliquid Rust SDK](https://github.com/hyperliquid-dex/hyperliquid-rust-sdk),
and [Infinite Field hypersdk](https://github.com/infinitefield/hypersdk).

## 2. Network and transport endpoints

| Purpose | Mainnet | Testnet |
| --- | --- | --- |
| Info REST | `https://api.hyperliquid.xyz/info` | `https://api.hyperliquid-testnet.xyz/info` |
| Exchange REST | `https://api.hyperliquid.xyz/exchange` | `https://api.hyperliquid-testnet.xyz/exchange` |
| WebSocket | `wss://api.hyperliquid.xyz/ws` | `wss://api.hyperliquid-testnet.xyz/ws` |

The same WebSocket supports subscriptions, unsigned info posts, and signed action
posts. Explorer requests are not supported over WebSocket. See
[WebSocket post requests](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/post-requests).

Use three logical tasks even if two share a physical WebSocket initially:

- public market-data task: `bbo`, `l2Book`, and `trades` for CASHCAT;
- private/account task: order, fill, funding, position, and open-order streams;
- execution task: signed action posts and response correlation.

Separating them prevents a slow signer or account parser from delaying the
allocation-free quote thread. A private socket should serve one actual account
address because some order/user messages do not carry an account identifier.

## 3. Account identities and credentials

Hyperliquid has three identities that must never be conflated:

| Identity | Purpose | Used where |
| --- | --- | --- |
| master account | owns funds and approves agents/subaccounts | offline/UI setup only |
| API/agent wallet | private key used by the bot to sign | signer and nonce namespace |
| traded account/subaccount | positions, orders, fills, fees, margin | `/info` queries and user subscriptions |

An API wallet is a signer, not the address whose positions should be queried.
Querying the agent address commonly returns an empty account. API-wallet nonces
are tracked by signer address, so one agent reused across processes or
subaccounts creates a shared nonce namespace. The official guidance is one API
wallet per trading process, preferably one per subaccount. See
[Nonces and API wallets](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/nonces-and-api-wallets).

For a master account, `vaultAddress` is absent. For a subaccount or vault, the
signed hash and the outer request both include that subaccount/vault address.
Subaccounts and vaults do not have independent private keys. See
[Exchange endpoint: subaccounts and vaults](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint).

### Required credential configuration

Keep the secret file compatible with the four entries in Passivbot's
`api-keys.json`, expressed in dotenv syntax:

```dotenv
exchange=hyperliquid
wallet_address=0x...
private_key=0x...
is_vault=true
```

Only the field names are shared with Passivbot. The Rust connector parses this
file itself and does not call Passivbot or CCXT.

The tracked [`hyperliquid.env.example`](hyperliquid.env.example) contains these
exact keys with deliberately invalid dummy values. Copy it to
`rust_live/hyperliquid.env`; the real file is ignored by Git.

When `is_vault=true`, `wallet_address` is also folded into the signed action hash
and sent as `vaultAddress`. When false, `vaultAddress` is absent. In both cases
`private_key` is the API/agent signer, not the master wallet key. Network,
symbol, DEX, risk, and live-enable values remain in validated TOML rather than
the secret file.

Startup must derive the agent address from `private_key` and verify that it is an
approved API/agent wallet for the account (resolving the master through
`userRole` when `wallet_address` is a subaccount). It must also query metadata,
account state, and user fees before enabling orders. A mismatch is fatal; no
extra credential field is required.

### Security requirements

- Never accept a master private key in the trading process.
- Never accept a key on a command line; process listings and shell history can
  expose it.
- Do not log keys, signed payloads containing unexpected sensitive fields, or
  unredacted environment dumps.
- Put key material in a zeroizing type and keep it out of crash reports.
- Compile transfer, withdrawal, staking, agent-approval, and builder-fee approval
  actions out of the trading adapter. It needs order/cancel/modify/leverage only.
- Use a dedicated subaccount so the dead-man switch cannot cancel another bot's
  orders and a strategy fault cannot access unrelated positions.
- Keep dry-run and live binaries/features distinct. `live` must continue to fail
  before loading secrets until all acceptance gates in section 16 pass.

API wallets can be deregistered, expire, or be pruned when the owning account has
no funds. The documentation warns against reusing a deregistered agent address
because its stored nonce history may be pruned, allowing old signed actions to
be replayed. Generate a new API wallet after deregistration.

## 4. CASHCAT instrument facts and startup validation

A public mainnet `meta` query on 2026-08-21T21:44:02Z returned:

| Field | Observed value |
| --- | ---: |
| perp DEX | first/default DEX, `""` |
| universe index / current asset ID | `231` |
| `szDecimals` | `0` |
| maximum leverage | `3` |
| `marginTableId` | `3` |
| `onlyIsolated` | `true` |

These are dated observations, not constants. The connector discovers and
validates them at every startup. The implemented `InstrumentSpec` carries
`only_isolated`, `margin_mode`, `margin_table_id`, delisting state, and a metadata
fingerprint; an isolated-only instrument makes `updateLeverage` use
`isCross=false`.

Perp asset IDs are the positions in the requested `meta.universe`. Spot and
builder-deployed perp IDs use different formulas; do not generalize the CASHCAT
index. See [Asset IDs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/asset-ids).

### Price and size encoding

- Prices: at most five significant figures and at most
  `6 - szDecimals` decimal places for perps. Integer prices are allowed even if
  they contain more than five digits.
- Sizes: at most `szDecimals` decimal places.
- Wire numbers are decimal strings with trailing zeroes removed and no exponent
  notation.
- CASHCAT therefore currently uses integer base sizes and at most six price
  decimals.
- Perp orders below 10 USDC notional are rejected.

See [Tick and lot size](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size)
and [Error responses](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/error-responses).

Use fixed-point integers or `rust_decimal` at the adapter boundary. Do not
re-round the policy's already instrument-aware integer price and size through
`f64`. For post-only orders, outward rounding remains mandatory: bids down, asks
up.

## 5. Exact L1 action signing

Normal orders, cancels, modifies, leverage changes, `scheduleCancel`, and `noop`
are L1 actions. They use the phantom-agent EIP-712 scheme. Human-readable
transfers and agent approval use a different user-signed scheme and are outside
this connector.

### 5.1 Action hash

For an action object, nonce, optional vault/subaccount, and optional expiry:

```text
packed = msgpack(action)                         # map/named encoding
bytes  = packed
       || nonce.to_be_bytes_u64()
       || (0x00                                  # no vault
           OR 0x01 || raw_20_byte_vault_address)
       || (nothing                               # no expiresAfter
           OR 0x00 || expires_after.to_be_bytes_u64())
connection_id = keccak256(bytes)
```

The vault address bytes are essential. Appending only the `0x01` marker is
wrong. The `expiresAfter` separator byte is also essential. The current official
Python implementation is the oracle; see
[`action_hash` and `sign_l1_action`](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/utils/signing.py).

MessagePack map order affects the hash. Use typed Rust structs with declaration
order matching the wire schema and `rmp_serde::to_vec_named`; do not construct
actions from an unordered map. Golden tests must pin the exact MessagePack bytes.

### 5.2 Phantom agent EIP-712 message

```text
domain:
  name              = "Exchange"
  version           = "1"
  chainId           = 1337
  verifyingContract = 0x0000000000000000000000000000000000000000

primary type: Agent
fields:
  source       : string   # "a" mainnet, "b" testnet
  connectionId : bytes32  # action hash above
```

Sign the EIP-712 digest with secp256k1 and serialize `{r, s, v}`, with `r` and
`s` as 32-byte `0x` hex strings and `v` in the form accepted by the SDK/API.

### 5.3 Outer action envelope

```json
{
  "action": { "type": "..." },
  "nonce": 1700000000000,
  "signature": { "r": "0x...", "s": "0x...", "v": 27 },
  "vaultAddress": "0x... optional ...",
  "expiresAfter": 1700000003000
}
```

`vaultAddress` and `expiresAfter` must be identical to the values folded into
the action hash. `expiresAfter` is an absolute Unix-millisecond deadline. An
action rejected because it arrived after that deadline consumes five times the
normal address-based rate-limit budget, so expiry is a safety device rather than
a substitute for reconciliation.

## 6. Nonce manager

Hyperliquid stores the highest 100 nonces per signer. A new nonce must be unique,
larger than the smallest retained nonce, and within `(block_time - 2 days,
block_time + 1 day)`. Transactions can arrive out of order.

Implement one atomic nonce allocator per API wallet:

```text
next = max(current_unix_ms, previous + 1)
```

Persist the last issued nonce atomically so a local clock rollback plus restart
cannot reuse one. Never allow two processes to share an agent key. Batch orders
and cancels roughly every 100 ms when useful, and keep ALO-only batches separate
from IOC/GTC batches because validators prioritize ALO-only batches.

The nonce belongs to the signer, not the traded subaccount. Sending actions for
two subaccounts through one agent shares the same 100-nonce window and is not an
approved architecture for this bot.

## 7. Trading actions required by the maker

Only the following exchange actions are needed initially. Full payload details
remain authoritative in the
[Exchange endpoint documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint).

### 7.1 Place batch of post-only orders

```text
action.type     = "order"
action.orders[] = {
  a: asset_id,
  b: is_buy,
  p: decimal_price_string,
  s: decimal_size_string,
  r: reduce_only,
  t: { limit: { tif: "Alo" } },
  c: optional_cloid
}
action.grouping = "na"
```

ALO is the only normal maker TIF. If it would match immediately, Hyperliquid
cancels/rejects it rather than taking liquidity. A `BadAloPx` result is an
expected post-only refusal and must not be interpreted as a fill.

Use one batch for the bid and ask generated by the same quote revision when both
sides are present. Parse every element in the returned status vector. A batch
can also fail with one pre-validation error applying to the whole batch.

### 7.2 Cancel by CLOID

Preferred cancellation form:

```text
action.type      = "cancelByCloid"
action.cancels[] = { asset: asset_id, cloid: "0x + 32 hex digits" }
```

The OID form is also required for foreign/recovered orders:

```text
action.type      = "cancel"
action.cancels[] = { a: asset_id, o: oid }
```

A missing order can mean it already filled or was already canceled. Resolve it
from fills/order status; do not assume cancel success.

There is no ordinary unsigned `cancelAll` shortcut. For controlled shutdown,
fetch `openOrders`, select the bot/account scope, and send batched OID/CLOID
cancels. `scheduleCancel` is the separate emergency dead-man mechanism.

### 7.3 Modify

Hyperliquid supports `modify` and `batchModify` by OID or CLOID. The initial
implementation should retain the dry-run model's explicit cancel-and-replace
semantics with a new CLOID and overlapping in-flight orders. That makes partial
fills and uncertainty visible. Native modify can be added later only after its
queue and partial-fill behavior is measured.

### 7.4 Leverage

```text
action = {
  type: "updateLeverage",
  asset: asset_id,
  isCross: false,
  leverage: 2
}
```

For current CASHCAT metadata, `isCross=true` is invalid because
`onlyIsolated=true`. Confirm the resulting position/account state before quoting.

### 7.5 Dead-man switch

```text
action = { type: "scheduleCancel", time: future_unix_ms }
```

Omitting `time` clears the schedule. The deadline must be at least five seconds
ahead. A triggered schedule cancels all open orders for the account, increments
a daily counter, and can trigger at most ten times per UTC day. A dedicated
account is therefore mandatory.

Recommended policy: refresh a deadline around 30 seconds ahead every 10 seconds.
On a graceful stop, explicitly cancel all bot orders, reconcile an empty open
order set, then clear the schedule. On a process/network failure, let it fire.

**What the shipped profile actually does (2026-09-02).** `config/cashcat.toml`
sets an 8 h deadline refreshed every 6 h. That is a budget decision: each
refresh is one address action against the venue's lifetime budget
(10k + 1 per USDC of volume, ~115 left on the test account), and the
recommended 30 s / 10 s policy would cost ~8.6k actions a day. The consequence
is that after a hard crash resting orders can sit for up to 8 h; the venue's
own documented constraints are only a 5 s minimum offset and 10 triggers per
UTC day. The safety net that actually protects a session is the teardown that
runs on every exit path — cancel resting orders, reconcile, clear the schedule
— together with `live-flatten`. Tighten the deadline once the account has
earned budget. Note also that on 2026-08-22 the venue refused `scheduleCancel`
on this subaccount for lacking 1M USDC cumulative volume, so on such an account
the dead-man is not available at all and `deadman_enabled = true` refuses to
arm.

### 7.6 No-op nonce invalidation

`{type:"noop"}` can consume a pending nonce and is documented as an alternative
to cancel spam for invalidating an in-flight action. Do not use it in the first
live version; first establish a correct CLOID/status reconciliation model.

## 8. WebSocket protocol

### 8.1 Application heartbeat

The server closes a connection if it has sent no message to the client for 60
seconds. Send the documented application message periodically:

```json
{ "method": "ping" }
```

and expect `{ "channel": "pong" }`. Protocol-level WebSocket ping frames are
not a substitute for this documented heartbeat. See
[Timeouts and heartbeats](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/timeouts-and-heartbeats).

Track last inbound frame, last pong, connection generation, and subscription
acknowledgements. Force reconnect on staleness even if writes still succeed.

### 8.2 Subscriptions needed

Subscription wrapper:

```json
{
  "method": "subscribe",
  "subscription": { "type": "..." }
}
```

Required public subscriptions for CASHCAT:

```text
{ type: "bbo",    coin: "CASHCAT" }
{ type: "l2Book", coin: "CASHCAT" }
{ type: "trades", coin: "CASHCAT" }
```

Required account subscriptions, using the **actual account/subaccount address**:

```text
{ type: "orderUpdates",       user: account }
{ type: "userFills",          user: account, aggregateByTime: false }
{ type: "userFundings",       user: account }
{ type: "clearinghouseState", user: account, dex: "" }
{ type: "openOrders",         user: account, dex: "" }
```

There is no private-session authentication handshake for these subscriptions;
the address selects the onchain account data. Authentication is required for
actions through their signatures, not for reading an address's streams.

Recommended additional subscriptions:

```text
{ type: "activeAssetData", user: account, coin: "CASHCAT" }
{ type: "userNonFundingLedgerUpdates", user: account }
{ type: "notification", user: account }
```

The complete current list and message types are in
[WebSocket subscriptions](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions).

Do not subscribe to both generic `userEvents` fills and dedicated `userFills`
unless all events are deduplicated. Dedicated streams are clearer and include
snapshot markers.

### 8.3 Essential incoming fields

Order update:

```text
order: coin, side, limitPx, sz_remaining, oid, timestamp, origSz, cloid?
status: open | filled | canceled | triggered | rejected | ...Canceled | ...Rejected
statusTimestamp
```

Fill:

```text
coin, px, sz, side, time, startPosition, closedPnl, oid, tid,
crossed, fee, feeToken, builderFee?, hash
```

Funding:

```text
time, coin, usdc, szi, fundingRate
```

For `userFills` and `userFundings`, the first message can be a snapshot with
`isSnapshot=true`; later messages use `false`. Store a durable fill/funding key
and process the initial snapshot with an overlap checkpoint rather than simply
discarding it. Public trade `tid` is only globally unique together with block
time and coin; use a composite key where appropriate.

### 8.4 Posting actions over WebSocket

```json
{
  "method": "post",
  "id": 12345,
  "request": {
    "type": "action",
    "payload": {
      "action": { "type": "order", "orders": [], "grouping": "na" },
      "nonce": 1700000000000,
      "signature": { "r": "0x...", "s": "0x...", "v": 27 },
      "vaultAddress": null,
      "expiresAfter": 1700000003000
    }
  }
}
```

Every post gets a unique transport `id`, separate from the order CLOID. Maintain
a bounded in-flight map keyed by `id`. Response wrapper:

```text
channel = "post"
data.id = request id
data.response.type = "action" | "info" | "error"
data.response.payload = normal HTTP-equivalent response
```

A `post` response confirms what the API returned, not that the local account
state is fully reconciled. The private order/fill streams and `/info` recovery
remain authoritative.

## 9. Authoritative `/info` calls

Use unsigned `/info` POSTs for startup and recovery. The minimum required set is:

| Request type | Purpose |
| --- | --- |
| `meta` | universe, `szDecimals`, leverage, margin mode/table, delisting |
| `metaAndAssetCtxs` | mark/oracle/mid, current funding, open interest |
| `clearinghouseState` | signed positions, entry/liquidation price, margin/equity |
| `openOrders` | authoritative currently working orders |
| `frontendOpenOrders` | richer diagnostics when needed |
| `orderStatus` | resolve OID/CLOID after uncertainty |
| `userFills` / `userFillsByTime` | fill recovery and exact fees/PnL |
| `userFunding` | funding recovery by time range |
| `userFees` | actual maker/taker rate for the account |
| `userRateLimit` | action budget monitoring |
| `activeAssetData` | available-to-trade and maximum sizes |
| `userRole` / `extraAgents` | startup identity validation |

Use the actual account/subaccount address, never the agent address. Time-range
responses paginate; the current general rule is at most 500 elements/distinct
blocks per page, while fill endpoints additionally have their documented 2,000
item response and recent-history limits. See the
[Info endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
and [perpetual-specific info calls](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals).

Maker fee must come from `userFees` and each fill's `fee`, not from the current
0.00015 profile assumption. Fee tiers, staking discounts, maker rebates, aligned
collateral, and HIP-3 deployer settings can change the effective rate. See
[Fees](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees).

Funding is paid hourly. The connector records the authoritative `usdc` funding
events rather than recomputing them. See
[Funding](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding).

## 10. Order lifecycle and reconciliation

### 10.1 CLOID

A Hyperliquid CLOID is exactly 128 bits encoded as `0x` plus 32 hexadecimal
digits. Generate it deterministically from stable identity fields such as:

```text
hash128(config_fingerprint, session_id, quote_generation, side, order_sequence)
```

Persist the mapping from CLOID to quote revision, side, intended price/size,
reduce-only flag, send nonce, transport request ID, and local timestamps. Never
reuse a CLOID for a different economic intent.

### 10.2 State machine

At minimum, track:

```text
Prepared
  -> Sent
  -> Resting | Filled | Rejected | UnknownOutcome
Resting
  -> CancelPending | PartiallyFilled | Filled | VenueCanceled
CancelPending
  -> Canceled | PartiallyFilled | Filled | UnknownOutcome
UnknownOutcome
  -> Resting | PartiallyFilled | Filled | Rejected | Canceled
```

`PartiallyFilled` is derived from unique fills and remaining size; never infer a
fill merely because an order disappeared. The one venue answer that *is* proof
of disappearance is the cancel rejection `Order was never placed, already
canceled, or filled`: it is definitive that the order is not on the book, so it
resolves to `Canceled` rather than leaving the order unresolved forever. Order updates and fill events can be
duplicated or reordered relative to action responses.

### 10.3 Unknown outcomes

If a socket closes, a timeout fires, JSON parsing fails, or the response is lost
after the request may have been written:

1. mark the order/CLOID unknown;
2. suppress new inventory-increasing orders on the affected side;
3. query `orderStatus` by CLOID;
4. query `openOrders` and overlapping `userFillsByTime`;
5. only then decide whether no order exists and a replacement is safe.

Never retry with a new CLOID or nonce solely because an action response timed
out. This is the primary double-order/double-position prevention invariant.

### 10.4 Reconnect procedure

On any private or execution connection loss:

1. mark execution unhealthy and withdraw desired quotes;
2. stop all new order actions, but allow bounded cancel/recovery actions;
3. reconnect with capped exponential backoff and resubscribe;
4. require subscription acknowledgements and a fresh heartbeat;
5. fetch metadata fingerprint, `clearinghouseState`, `openOrders`, user fills
   with a checkpoint overlap, funding, and relevant unknown CLOID statuses;
6. rebuild local order/account state and deduplicate all events;
7. cancel stale or foreign bot orders according to the dedicated-account policy;
8. require consistency across position, fills, and open orders before quoting.

No virtual/live order or queue priority is restored from disk after restart.
Only durable identifiers, account checkpoints, and unresolved actions are
restored, then reconciled against the venue.

## 11. Account and risk semantics

Hyperliquid perps use one-way signed positions. In account state, positive `szi`
is long and negative is short. The connector publishes physical base inventory
to the existing one-engine signed-inventory policy.

Before accepting an order intent, combine model risk with venue state:

- current signed position and working-order worst-case fills;
- `availableToTrade`/`maxTradeSzs` where available;
- actual account equity, margin used, withdrawable balance, and liquidation
  price;
- current asset margin mode/table and maximum leverage;
- actual fee schedule and funding events;
- prospective notional and margin if every overlapping order fills;
- order minimum notional and instrument precision;
- account abstraction mode. Fail closed on an unvalidated portfolio/unified
  mode.

CASHCAT's current `onlyIsolated=true` means the existing generic dry-run margin
assumptions are not enough to authorize live orders. Venue account state is the
source of truth; local formulas are conservative prechecks.

The fill field `crossed` verifies maker/taker status. A normal ALO fill should be
maker (`crossed=false`). A crossed fill, unknown liquidity flag, or effective fee
inconsistent with the expected maker schedule must disable quoting and trigger
reconciliation.

## 12. Responses and errors

Successful order action elements normally contain one of:

- `resting { oid }`;
- `filled { totalSz, avgPx, oid }`;
- `error: "..."`.

Batch errors are usually parallel to the input array, but some payload-level
pre-validation errors return one error for the whole batch. The parser must
support both shapes.

Important order errors include invalid tick, minimum notional, insufficient
margin, reduce-only violation, post-only crossing (`BadAloPx`), IOC no-fill,
invalid trigger, no liquidity, oracle deviation, open-interest limits, maximum
position, and insufficient spot balance. Cancel can return missing order. Treat
each as a typed result and retain the raw string for diagnostics.

`orderStatus` exposes terminal causes including user cancel, margin cancel,
self-trade cancel, reduce-only cancel, liquidation, delisting, scheduled cancel,
and specific rejection variants. Do not collapse them all into `Canceled` when
evaluating risk or scientific execution evidence.

## 13. Rate limits and batching

Limits last checked against the official documentation on 2026-08-21 included:

- 1,200 aggregate REST weight per minute;
- 10 simultaneous WebSocket connections;
- 30 new WebSocket connections per minute;
- 1,000 subscriptions and 10 unique users across user-specific subscriptions;
- 2,000 client-to-server WebSocket messages per minute;
- 100 simultaneous in-flight WebSocket post messages.

Exchange actions cost `1 + floor(batch_length / 40)` IP weight. Common light
info calls such as `l2Book`, `allMids`, `clearinghouseState`, and `orderStatus`
cost weight 2; most other info calls cost 20, with additional response-size
weights on history endpoints.

Address limits are separate, and they are the constraint that actually binds
this strategy. An address begins with a 10,000-request buffer and earns roughly
one action request per cumulative USDC traded. Cancels receive a larger
allowance. A batch counts once for IP weight but each contained action counts
toward the address limit. During congestion, block share also depends on maker
share. See
[Rate limits and user limits](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits).

**The address counter is cumulative for the life of the account. It does not
reset.** Query it with `{"type":"userRateLimit","user":<addr>}` and read
`nRequestsUsed` / `nRequestsCap`; never assume a per-run or rolling-window
budget. Measured on the CASHCAT account on 2026-08-23:

```
cumVlm        = 3,060.56 USDC
nRequestsCap  = 13,060      (= 10,000 + cumulative volume)
nRequestsUsed = 12,941      (lifetime)
remaining     =    119
```

Two consequences follow, and both were observed live:

- A wasted action is gone permanently. The duplicate-cancel defect fixed in
  `9769354` burned 2,440 actions in a single 40-minute session — about 19% of
  the entire lifetime cap.
- Replay of one 120-minute CASHCAT window at the shipped
  `replace_threshold_bps = 2.0` costs 11,902 address actions
  (`docs/requote_hysteresis_sweep.md`). A *fresh* account's 10,000-request
  buffer does not cover one two-hour session. Sustained operation requires the
  volume traded per action to exceed 1 USDC; that window managed roughly 1
  fill per 220 actions.

When the remaining reserve falls below 100 requests the backend suspends new
orders while still permitting cancels, logs `rate limit hit; suspending new
orders`, and enters a 30-second cooldown before retrying. That path was
exercised live on 2026-08-23: it held the reserve at ~119 rather than draining
it to zero, leaving room to flatten.

Implementation consequences:

- rate-limit before serializing/signing where possible;
- coalesce superseded quote revisions;
- batch bid/ask ALO orders and batch cancels;
- do not repeatedly cancel an order after a confirmed result;
- reserve capacity for emergency cancels and reconciliation;
- record REST weight, action count, WebSocket messages, in-flight posts, and
  address budget from `userRateLimit`.

## 14. Implemented architecture in `rust_live`

The live backend now has these boundaries:

1. `InstrumentSpec` carries margin mode/`only_isolated`, margin table ID,
   delisting status, and a metadata revision/fingerprint discovered from the
   venue.
2. The shared account state preserves the dry-run accounting fields while the
   live projection is rebuilt from authoritative account/order/fill messages.
3. `ExecutionEvent` distinguishes acknowledgements, rejections, cancels, fills,
   funding, account reconciliation, unknown outcomes, connection state, and
   invalidation.
4. Authenticated events use bounded delivery independently of public market
   events; fills trigger execution processing without waiting for another BBO.
5. Quoting requires fresh public data, calibration, latency eligibility, account
   state, and reconciliation. Queue saturation or causal event loss fails
   closed.
6. Signing, nonce leases, fixed-point actions, WebSocket post correlation,
   subscriptions, rate limits, order state, and reconciliation are separate
   connector modules or clearly isolated responsibilities.
7. Signing/HTTP/JSON and persistence remain off the hot thread. Desired quotes
   stay integer, venue-neutral `OrderIntent`s.
8. `[live].enabled=false` remains tracked; enabling it selects the live backend,
   while production mode always enforces the latency gate.
9. `LatencyMonitor` records queue wait, action preparation, signing, socket
   write, submit/cancel acknowledgement, and fill/close stages using monotonic
   timestamps. The socket and hot tasks enqueue raw samples only; the observer
   thread owns rolling percentiles and I/O.

Current workspace/module layout:

```text
rust_live/
  crates/cj-core/                    # HJB, policy, instrument and shared types
  crates/cj-data/                    # calibration, Parquet and replay
  crates/mm-execution/               # generic traits and dry-run simulator
  crates/mm-runtime/                 # hot thread, atomics, metrics and latency
  crates/mm-config/                  # validated central TOML schema
  crates/mm-settings/                # dependency-light setting value types
  crates/hyperliquid/src/
    execution/hyperliquid_live.rs    # live backend and reconciliation policy
    session.rs                       # action/account WebSocket actor
    transport.rs                     # shared subscriptions and heartbeat wire
    signing.rs                       # fixed-point actions, CLOIDs and signer
    live_state.rs                    # normalized redb state and nonce leases
    exchange.rs                      # authoritative REST client
    account_types.rs                 # typed account/order/fill payloads
    market.rs / wire.rs / meta.rs    # public feed and metadata
  src/main.rs                        # production CLI/orchestration only
  src/bin/mm_live_acceptance.rs      # feature-gated real-account acceptance
```

## 15. Audit of existing projects under `C:\Users\david\Desktop\freqtrade`

The audit searched Rust, Python, JavaScript, and TypeScript sources while
excluding build/dependency directories. Read-only collectors, delegated CCXT
paths, SDK documentation copies, and duplicate project
copies were separated from actual connector implementations.

### 15.1 Rust implementations

| Local implementation | What is useful | Why it is not a drop-in live maker connector |
| --- | --- | --- |
| `XEMM_CROSS_EXCHANGE_MARKET_MAKING_PACIFICA_HYPERLIQUID\src\connector\hyperliquid\` (line-identical duplicate under `XEMM\XEMM_CROSS_...`) | EIP-712 domain/digest, monotonic atomic nonce, metadata cache, public L2 WS, REST IOC submission with CLOID, order-status/fill/account parsing, timeout-aware unknown outcome comments | Only builds IOC market orders; no maker ALO lifecycle, cancel/modify/dead-man switch, or private WS. `construct_connection_id` appends a vault marker but **not the 20 vault address bytes** and has no expiry encoding. Its tests cover only `vault=None`. Uses `f64`, stores key as `String`, and uses the large deprecated-style `ethers` stack. Do not copy its signer. |
| `OLD\XEMM_dry_run_evaluator\src\livebot\exec\{hyperliquid,crypto,sign,creds}.rs` plus `src\connectors\hyperliquid.rs` | Best local low-level reference: typed MessagePack wire structs, correct vault marker + address bytes, correct expiry separator, main/test phantom agent, secp256k1 signatures, monotonic nonce, golden vectors, ALO/IOC construction, OID cancel, leverage action, open orders/account reads, robust public `bbo`/book/trade WS heartbeat and reconnect | Retired project, not a maintained Git repository. Assumes a subaccount/vault on every live path, has no master-account/no-vault mode, cancel-by-CLOID, modify/batch, `scheduleCancel`, private/account WS, durable fill deduplication, or complete reconciliation. Key bytes are retained without a zeroizing secret type. Port fixtures/ideas only. |
| Current `rust_live` | Generic instrument, lock-free CJ hot path, zeroizing signer, exact fixed-point actions, persisted nonce/order/fill/funding state, WebSocket posts/account streams, REST recovery, rate reserves, account-aware risk, explicit market close, and guarded acceptance tooling | Continuous live is implemented but tracked off. Current development latency fails the production gate, and the present low-volume account is ineligible for the venue dead-man feature. |

Conclusion: the older Rust signer contains valuable byte-level work, but none of
the pre-existing local projects was complete enough to enable real money safely.
The current `rust_live` connector fills that lifecycle gap, remains tracked off,
and still refuses production trading on this development machine's latency.

### 15.2 JavaScript implementations

| Local implementation | Reuse assessment |
| --- | --- |
| `DELTA_NEUTRAL\DELTA_NEUTRAL_HYPERLIQUID_PERP_SPOT\hyperliquid.js` and `tests\unit\hyperliquid-conformance.test.js` | Valuable independent fixtures: correct vault bytes and expiry separator, strict CLOID validation, monotonic nonce, application ping, WS post correlation, timeouts classified as unknown outcomes, REST rate limiting, stale-book refusal. It mainly implements aggressive market orders and lacks maker cancel/order/private-stream reconciliation. Use its conformance tests as behavioral references. |
| `XEMM\standalone-utils\connectors\hyperliquid.js` and older copies | Public WS and WS action-post examples, but the signer omits vault address bytes and the expiry separator. Do not reuse signing code. |

### 15.3 Python implementations

| Local implementation | Reuse assessment |
| --- | --- |
| `XEMM\hyperliquid-python-sdk-master` | A local official-SDK copy and the best language-independent signing/API oracle, but it is version 0.19.0. The current published line observed during this audit is 0.24.0, so refresh/pin it before generating fixtures. Do not import Python at Rust runtime. |
| Current `scripts\hyperliquid_alo_executor.py` and `hyperliquid_risk_executor.py` | Good ALO/IOC intent, outward rounding, CLOID, response classification, cancel-after-probe, and explicit real-order guard references. They are guarded command tools using the official SDK, not persistent account WebSocket connectors. Also avoid their optional command-line private-key input in production. |
| `passivbot_real_run\src\exchanges\hyperliquid.py` and related Passivbot copies | Operational lifecycle evidence through CCXT Pro: `watch_orders`, REST open-order/position recovery, ALO parameters, vault handling, error retries, and minimum-notional adaptation. Useful for behavior, but CCXT hides signing/wire details and its state model should not be transplanted into the Rust engine. |
| `DELTA_NEUTRAL\CROSS_EXCHANGE_DELTA_NEUTRAL_HL_PAC\hyperliquid_connector.py` | Small official-Python-SDK example for market IOC, leverage, position, balance, and funding. Not a maker or private-WS connector. |
| Older Python/CCXT diagnostics and data collectors | Mostly public/read-only, delegated, or application-specific. They do not add a lower-level authenticated connector. |

### 15.4 Recommended reuse order

1. Current official documentation and current official SDK sources.
2. Official Python SDK 0.24.x as the independent signing/response oracle.
3. `hl_sdk` pinned and audited for typed actions/signing only.
4. Golden vectors and correct vault/expiry encoding from the retired Rust stack.
5. Unknown-outcome, heartbeat, and conformance cases from the newer JavaScript
   connector.
6. CLOID/ALO/risk response semantics from this project's Python tools.
7. Passivbot only as an operational comparison.

## 16. Test and release gates

### 16.1 Offline deterministic tests

Generate fixtures with a known throwaway key using the current official Python
SDK. Rust must match exactly for both networks and for:

- order action with no vault/no expiry;
- order action with subaccount/vault;
- order action with vault and expiry;
- ALO bid/ask, IOC reduce-only, cancel OID, cancel CLOID, batch order/cancel,
  leverage, and schedule-cancel actions;
- exact MessagePack bytes, action hash, EIP-712 digest, recovered signer, and
  `{r,s,v}`;
- CASHCAT integer size and outward price formatting;
- whole-batch and per-element response errors;
- all order-status variants and duplicated/reordered private events.

Use at least two independent implementations for cryptographic parity: current
official Python and pinned Rust SDK/manual implementation. A local old fixture is
supporting evidence, not ground truth.

### 16.2 Mock transport and fault injection

- action response before/after order update;
- fill before/after resting acknowledgement;
- partial fill during cancel;
- duplicate fill/order/funding snapshots;
- socket loss before write, during write, and after write;
- action timeout followed by CLOID reconciliation;
- REST response loss and malformed JSON;
- clock rollback and multiple actions in one millisecond;
- bounded-ring saturation;
- reconnect/resubscribe with snapshot overlap;
- rate-limit exhaustion with emergency cancel reserve;
- dead-man refresh failure;
- stale metadata or unexpected CASHCAT margin-mode change;
- process restart with unresolved actions and working venue orders.

### 16.3 Read-only network validation

On testnet, then mainnet public/read-only:

- discover metadata and compare fingerprints;
- verify API wallet role/approval and actual account address separation;
- stream all required subscriptions for hours with reconnect injection;
- reconcile WebSocket snapshots against `/info` positions/open orders/fills;
- query actual fees, funding, and rate budget;
- submit no action.

### 16.4 Testnet trading validation

Use a dedicated testnet API wallet/account. CASHCAT need not exist on testnet;
connector mechanics can be validated on another test instrument without claiming
CASHCAT strategy validation.

1. Place one far passive ALO above minimum notional; prove it rests and is maker.
2. Cancel by CLOID; prove empty open orders and terminal status.
3. Submit a deliberately crossing ALO; prove `BadAloPx`/cancel and no fill.
4. Exercise partial fills, cancel races, unknown outcomes, and restart recovery.
5. Exercise isolated leverage and reduce-only IOC flattening.
6. Exercise and clear the dead-man switch without exceeding its daily trigger
   allowance.
7. Run a long canary with forced socket/process/network failures.

### 16.5 Mainnet canary prerequisites

The explicitly authorized mainnet campaign used the dedicated low-balance
CASHCAT subaccount and enforced its directional/gross/turnover/loss caps. It
finished flat and empty. This does not automatically enable the tracked profile
or waive the production latency gate. Dead-man triggering remains unvalidated
because the venue rejected `scheduleCancel` below its one-million-USDC
cumulative-volume requirement.

## 17. Implementation checklist

- [x] Recheck the official exchange, subscription, nonce, heartbeat, and rate-limit documentation.
- [x] Pin and audit the manual `k256`/MessagePack implementation instead of adding a second SDK.
- [ ] Refresh the Python oracle to a pinned current release.
- [x] Extend `InstrumentSpec` for isolated/margin metadata and fingerprinting.
- [x] Persist crash-safe monotonic nonce reservations across restarts.
- [x] Add exact signing golden fixtures for vault and expiry.
- [x] Add order/CLOID-cancel/OID-cancel, isolated leverage, and schedule-cancel action encoding.
- [x] Add WebSocket post correlation and unknown-outcome handling.
- [x] Add durable fill/funding deduplication and account subscriptions.
- [x] Add startup/reconnect/restart reconciliation.
- [x] Add actual fee/funding/account-state ingestion.
- [x] Add REST/WS rate budgets and emergency cancel reserve.
- [x] Keep tracked live disabled and fail before credentials until enabled.
- [x] Perform the authorized bounded mainnet campaign and finish flat/empty.
- [ ] Trigger the dead-man on an eligible account; the 2026-08-22 test account was venue-ineligible below $1M cumulative volume.

## 18. Primary references

- [Hyperliquid API index](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api)
- [Exchange endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint)
- [Info endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
- [Perpetual info endpoints](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)
- [Asset IDs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/asset-ids)
- [Tick and lot size](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size)
- [Nonces and API wallets](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/nonces-and-api-wallets)
- [Error responses](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/error-responses)
- [Rate and user limits](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits)
- [WebSocket subscriptions](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
- [WebSocket post requests](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/post-requests)
- [WebSocket heartbeat](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/timeouts-and-heartbeats)
- [Fees](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
- [Funding](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding)
- [Contract specifications](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/contract-specifications)
- [Official Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
- [Official Python signing source](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/utils/signing.py)
- [Official Python exchange source](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/exchange.py)
- [Official Python WebSocket source](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/websocket_manager.py)
- [Hyperliquid Rust SDK](https://github.com/hyperliquid-dex/hyperliquid-rust-sdk)
- [`hl_sdk` API documentation](https://docs.rs/hl_sdk/latest/hl_sdk/)
- [Community hypersdk](https://github.com/infinitefield/hypersdk)
