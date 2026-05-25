# Fail-Closed Market Making Hardening Plan

This plan turns the current project into a safer research and dry-run market-making
system before any real capital is used.

The goal is not to make the current bot trade. The goal is:

> Make the implementation fail-closed, mathematically consistent with the
> Cartea-Jaimungal model, maker-safe on Hyperliquid, observable, and testable in
> replay before any real capital is used.

## Consistency Check Against Original Roadmap

This document keeps the original roadmap structure and preserves all original
critical fixes. The following implementation decisions are now locked so future
work does not need to choose between alternatives:

- Scope: implement the full roadmap, including replay and deployment gates.
- First execution target: long-only Freqtrade research mode.
- Live behavior: `trading_enabled` defaults to `False` and must remain false
  until the deployment gates pass.
- Market-making path: Path A from the original plan is selected first:
  HJB-priced passive long entry plus passive unwind. This is not production
  two-sided market making.
- Post-only policy: add local maker guards now, but do not assume Freqtrade
  `PO` maps to Hyperliquid `Alo` until verified. If that mapping cannot be
  proven, live market making must move to a direct Hyperliquid execution layer.
- Existing local user changes must be preserved unless they are explicitly part
  of a safety/config change.

The only intentional refinement versus the original is the post-only handling:
the original listed `order_time_in_force: PO` as a desired config target, with a
caveat that support must be verified. This plan resolves that caveat by making
verification a hard gate. The strategy must stay disabled if post-only support is
unknown.

## Current Blockers

| Severity | Area | Current issue | Why it matters | Fix phase |
| ---: | --- | --- | --- | --- |
| Critical | HJB math | Boundary quotes are clamped instead of disabled | The model can quote the forbidden side at inventory bounds | Phase 1 |
| Critical | Execution | `custom_entry_price()` returning `None` can fall back to `proposed_rate` | Invalid quotes can still become orders | Phase 3 |
| Critical | Inventory | Inventory is `len(open_trades)`, not signed base exposure | HJB state does not match controlled position | Phase 2 |
| Critical | Maker safety | GTC limit orders are used; post-only is not guaranteed | Crossing quotes can become taker fills | Phase 4 |
| High | Lifecycle | `minimal_roi = {"0": -1}` forces immediate time-based exit logic | Brittle substitute for ask-side quoting | Phase 3 |
| High | Risk | `hjb_alpha = 0`, `hjb_phi = 0` | Inventory risk is mostly disabled | Phase 2 |
| High | Params | The bot starts estimator threads and immediately reads JSON files | Can load stale, partial, or inconsistent params | Phase 5 |
| High | Fees | Strategy fee is 1.5 bps, config fee is 10 bps | Model, dry-run, and backtest disagree | Phase 4 |
| High | Backtesting | Candle high/low fills are not maker-fill simulation | Backtests can be misleading | Phase 6 |
| Medium | Freshness | Data check only verifies that a Parquet file exists | Stale data can enable trading | Phase 5 |
| Medium | Symbols | Some paths are hardcoded to `ETH` | Multi-symbol or renamed pair handling breaks | Phase 5 |
| Medium | Observability | Logs lack kill-switch and accounting fields | Cannot prove maker safety or stale-data behavior | Phase 7 |

## Phase 0 - Freeze Trading And Fail Closed

Do this before changing model behavior.

### Strategy Safety Lock

Add strategy-level state to `Market_Making.py`:

```python
trading_enabled = False
fail_closed_reason = "initial_safety_lock"
```

`populate_entry_trend()` must start by clearing entries:

```python
dataframe.loc[:, "enter_long"] = 0
if not self.trading_enabled:
    return dataframe
```

The strategy may load params, compute HJB, write health logs, and run replay
support while locked, but it must place zero orders.

### Config Safety Defaults

Update `user_data/config.json` for validation mode:

```json
{
  "dry_run": true,
  "force_entry_enable": false,
  "stake_amount": 25,
  "tradable_balance_ratio": 0.10,
  "fee": 0.00015,
  "custom_price_max_distance_ratio": 0.05
}
```

Also set nested API force-entry fields off where present:

```json
{
  "api_server": {
    "forcebuy_enable": false,
    "force_entry_enable": false
  }
}
```

Do not enable live trading from config. The strategy kill switch remains the
final authority.

### Acceptance Criteria

- Bot starts with `dry_run: true`.
- Params and HJB can refresh.
- Debug and health logs are written.
- Zero orders are placed while `trading_enabled = False`.
- Force-entry commands are disabled by config.

## Phase 1 - Fix HJB Solver Math

The HJB inventory grid is:

```text
q_grid = [-q_max, ..., 0, ..., +q_max]
```

Use this convention everywhere:

```text
Ask / sell quote:
  fill moves q -> q - 1
  allowed only if q > q_min
  disabled at q = q_min

Bid / buy quote:
  fill moves q -> q + 1
  allowed only if q < q_max
  disabled at q = q_max
```

So:

```python
can_ask = i > 0
can_bid = i < d - 1
```

### `scripts/hjb.py`

Patch `compute_h_symmetric()` so final deltas start disabled:

```python
delta_plus = np.full(d, np.inf, dtype=float)
delta_minus = np.full(d, np.inf, dtype=float)

for i, q in enumerate(q_grid):
    h_q = h[i]

    if i > 0:
        h_qm1 = h[i - 1]
        delta_plus[i] = (1.0 / kappa) + eps_p - (h_qm1 - h_q)

    if i < d - 1:
        h_qp1 = h[i + 1]
        delta_minus[i] = (1.0 / kappa) + eps_m - (h_qp1 - h_q)
```

Patch `compute_h_asymmetric()` so `_compute_g()` excludes forbidden boundary
terms. The boundary side contributes no HJB value.

```python
def _compute_g(h_vec: np.ndarray) -> np.ndarray:
    g_vec = np.zeros_like(h_vec)

    for i, q in enumerate(q_grid):
        h_q = h_vec[i]
        value_total = -float(phi) * float(q * q)

        if i > 0:
            _, val_p = _optimal_delta_and_value(
                lam_p,
                kappa_p,
                eps_p,
                h_vec[i - 1] - h_q,
                clip_at_zero=clip_deltas,
            )
            value_total += val_p

        if i < d - 1:
            _, val_m = _optimal_delta_and_value(
                lam_m,
                kappa_m,
                eps_m,
                h_vec[i + 1] - h_q,
                clip_at_zero=clip_deltas,
            )
            value_total += val_m

        g_vec[i] = value_total

    return g_vec
```

Then compute final asymmetric deltas with the same `np.inf` boundary rule.

Add solver validation:

- `q_max >= 1`
- `T_seconds > 0`
- `kappa_plus > 0`, `kappa_minus > 0`
- `lambda_plus >= 0`, `lambda_minus >= 0`
- `epsilon_plus` and `epsilon_minus` finite
- `alpha >= 0`, `phi >= 0`

Return metadata:

```python
{
  "method": "matrix_exponential" or "backward_euler",
  "boundary_policy": "disabled_side_is_inf",
  "q_min": -q_max,
  "q_max": q_max
}
```

### `scripts/compute_spreads.py`

Render disabled sides explicitly:

```python
if not np.isfinite(delta_bid):
    bid_px = None
    bid_bps = None
    bid_label = "DISABLED"
else:
    delta_bid_total = delta_bid + fee_cushion
```

Do not add fee cushion to `np.inf`.

Print boundary verification:

```text
q = -q_max => ask DISABLED
q = +q_max => bid DISABLED
```

### HJB Tests

Create `tests/test_hjb.py` with:

- Boundary depths are infinite:
  - `delta_plus[0]` is `inf`
  - `delta_minus[-1]` is `inf`
  - all other side depths are finite
- Symmetric and asymmetric solvers approximately match when kappas are equal.
- With `alpha = phi = epsilon = 0`, center deltas are close to `1 / kappa`.
- With symmetric params and risk enabled:
  - positive inventory has ask closer than bid
  - negative inventory has bid closer than ask
- Invalid solver inputs raise `ValueError`.

### Acceptance Criteria

```bash
pytest tests/test_hjb.py
python scripts/compute_spreads.py --crypto ETH --mid 4322.05 --qmax 3 --asym-kappa
```

The output must show disabled forbidden sides at both inventory boundaries.

## Phase 2 - Fix Inventory, Units, And Risk

The model inventory `Q_t` is an inventory unit count, not an open-trade count.

### First Implementation: Long-Only Freqtrade

Use Path A:

```text
bid quote = passive long entry
ask quote = passive exit / reduction
no short inventory
operational q in [0, q_max]
```

Keep:

```python
can_short = False
```

The full HJB solver may still support negative `q` for model validation and
future signed market making, but the Freqtrade strategy must not use negative
inventory states in this first implementation.

### Strategy Units

Add strategy fields:

```python
inventory_unit_base = 0.01
max_inventory_units = 3
hjb_alpha = 0.001
hjb_phi = 0.0001
```

For long-only mode:

```python
q_raw = round(max(0.0, signed_base_position) / inventory_unit_base)
q = min(hjb_q_max, max(0, q_raw))
```

If a short position is detected while `can_short = False`, reject new orders,
log `unexpected_short_position`, and keep `trading_enabled = False`.

### Position Source

Implement `_signed_base_position(pair)` with this priority:

1. Exchange/futures position endpoint if available through Freqtrade or CCXT.
2. Open Freqtrade `Trade` objects, using amount and direction.
3. Wallet/base balance only as a fallback, and only in dry-run/replay contexts.

For futures-compatible code:

```python
if position_side == "long":
    signed_base = +abs(size)
elif position_side == "short":
    signed_base = -abs(size)
else:
    signed_base = 0.0
```

### Units Manifest

Create `docs/UNITS.md` with:

```text
price_unit: USDC per base asset
depth_unit: USDC
epsilon_unit: USDC
kappa_unit: 1 / USDC
lambda_unit: events / second
T_unit: seconds
q_unit: inventory_unit_base units of base asset
inventory_unit_base: base asset amount represented by one q step
```

### Acceptance Criteria

Every quote decision log includes:

```json
{
  "signed_base_position": 0.02,
  "inventory_unit_base": 0.01,
  "q": 2
}
```

`q` must match signed base exposure, not number of open trades.

## Phase 3 - Fix Freqtrade Wiring And Fail-Closed Order Logic

Freqtrade custom prices can fall back to `proposed_rate` when returning `None`
or invalid values. Therefore:

```text
custom_entry_price/custom_exit_price compute prices.
confirm_trade_entry/confirm_trade_exit decide whether orders are allowed.
```

### Callback Signatures

Patch `custom_entry_price()`:

```python
def custom_entry_price(
    self,
    pair: str,
    trade: Trade | None,
    current_time: datetime,
    proposed_rate: float,
    entry_tag: str | None,
    side: str,
    **kwargs,
) -> float:
```

Keep `custom_exit_price()` compatible with the installed Freqtrade version and
add tests to catch signature mismatch.

### Quote State Validation

Add:

```python
def _quote_state_valid(
    self,
    pair: str,
    side: str,
    rate: float,
    current_time: datetime,
) -> tuple[bool, str]:
    ...
```

It must reject:

- no HJB cache
- stale HJB
- invalid or nonfinite rate
- missing or invalid params
- stale params
- stale orderbook
- stale collector data
- boundary side disabled
- inventory limit reached
- post-only not verified when trading would be live

### Confirm Gates

Add `confirm_trade_entry()`:

```python
ok, reason = self._quote_state_valid(pair, side, rate, current_time)
if not ok:
    log entry_rejected
    return False

ok, reason = self._maker_safe(pair, "bid", rate)
if not ok:
    log entry_rejected
    return False

return True
```

Add `confirm_trade_exit()`:

- Always allow `stop_loss`, `stoploss_on_exchange`, `liquidation`, and emergency
  exits.
- For normal passive asks, require valid quote state and maker safety.

### Entry And Exit Signals

Replace `minimal_roi = {"0": -1}` with:

```python
minimal_roi = {"0": 10}
use_exit_signal = True
```

`populate_entry_trend()` must:

- clear all entries by default
- return immediately if `trading_enabled` is false
- require `_model_ready(pair)`
- require `_inventory_allows_bid(pair)`
- set only the latest candle:

```python
dataframe.loc[dataframe.index[-1], "enter_long"] = 1
dataframe.loc[dataframe.index[-1], "enter_tag"] = "mm_bid"
```

`populate_exit_trend()` or `custom_exit()` must only request passive asks when
the model and inventory state allow unwind. Use the path that reliably triggers
`custom_exit_price()` in the installed Freqtrade version.

### Repricing Policy

Freqtrade 1m candles are not high-frequency quote management. Treat repricing as
low-frequency passive quoting only:

- `adjust_entry_price()` may update once per candle.
- Add a symmetric exit repricing path only if supported by this Freqtrade
  version.
- Do not describe this as production two-sided market making.

### Acceptance Criteria

Logs must prove:

- no HJB delta rejects the order instead of falling back to `proposed_rate`
- stale params reject
- boundary `inf` delta rejects
- crossing quote rejects
- stop-loss exits are never blocked

## Phase 4 - Enforce Maker-Safe Hyperliquid Execution

The model assumes maker economics. GTC does not guarantee maker behavior.
Hyperliquid native post-only is `Alo`.

### Post-Only Policy

Target live behavior:

```text
entry bid TIF = Alo
exit ask TIF = Alo
```

Implementation policy:

- Add local maker guards immediately.
- Keep `trading_enabled = False` until Freqtrade/CCXT `PO` mapping to
  Hyperliquid `Alo` is verified with logs.
- If `PO -> Alo` cannot be verified, do not use Freqtrade for live maker
  execution.

### Local Maker Guard

Add:

```python
def _maker_safe(self, pair: str, quote_side: str, rate: float) -> tuple[bool, str]:
    ob = self.dp.orderbook(pair, maximum=1)
    if not ob or not ob.get("bids") or not ob.get("asks"):
        return False, "empty_orderbook"

    best_bid = float(ob["bids"][0][0])
    best_ask = float(ob["asks"][0][0])

    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        return False, "crossed_or_invalid_book"

    if quote_side == "bid" and rate >= best_ask:
        return False, "bid_crosses_ask"

    if quote_side == "ask" and rate <= best_bid:
        return False, "ask_crosses_bid"

    return True, "ok"
```

### Rounding

Order:

```text
raw model price
-> round to exchange tick
-> maker-safety check
-> submit
```

Rounding policy:

- bid: round down
- ask: round up
- amount: round down to lot/min amount

After rounding, re-check:

```text
bid < best_ask
ask > best_bid
```

### Fees

Update strategy comment:

```python
fees_maker_HL = 0.0150 / 100.0  # 0.015% = 1.5 bps base perp maker fee
```

Config fee must match:

```json
"fee": 0.00015
```

Longer term, load and log the actual account fee tier.

### Fill Accounting And Kill Switch

Every fill log should include:

```json
{
  "liquidity": "maker",
  "expected_fee_rate": 0.00015,
  "actual_fee_paid": "...",
  "order_type": "limit",
  "tif": "Alo",
  "quote_side": "bid"
}
```

If a taker fill occurs during market-making mode:

```text
trigger kill switch
cancel open orders
set trading_enabled = False
write kill_switch event
```

### Acceptance Criteria

- Accepted test orders rest as maker orders or reject as ALO/post-only.
- Taker fills are zero.
- Every order is tick-safe and lot-safe.
- Strategy fee, config fee, and logged exchange fee agree.

## Phase 5 - Fix Parameter And Data Pipeline

The strategy currently risks reading stale or partially written parameter files.

### Atomic Writes

Add a shared helper, preferably in a small importable utility:

```python
def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=4, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
```

Use it for:

- `kappa.json`
- `lambda.json`
- `epsilon.json`
- `lambda_trades.json`
- parameter runner status file

### Schema Version 2

`kappa.json` and `lambda.json` must preserve the distinction:

```text
lambda0_fit: model intensity from kappa/depth fit, used by HJB
lambda_raw: raw trade intensity from get_lambda.py, monitoring only
```

Example symbol payload:

```json
{
  "ETH": {
    "schema_version": 2,
    "kappa+": 2.88,
    "kappa-": 1.08,
    "lambda+": 0.136,
    "lambda-": 0.095,
    "lambda_source": "lambda0_fit",
    "unit": {
      "kappa": "1/USDC",
      "lambda": "events_per_second"
    },
    "window_start": "2026-05-25T10:00:00Z",
    "window_end": "2026-05-25T10:30:00Z",
    "generated_at": "2026-05-25T10:30:01Z",
    "n_quotes": 12345,
    "n_trades": 456,
    "r2_plus": 0.73,
    "r2_minus": 0.69
  }
}
```

Example epsilon payload:

```json
{
  "ETH": {
    "schema_version": 2,
    "epsilon+": 0.0123,
    "epsilon-": 0.0107,
    "unit": "USDC",
    "estimator": "trimmed_mean",
    "window_ms": 200,
    "window_start": "2026-05-25T10:00:00Z",
    "window_end": "2026-05-25T10:30:00Z",
    "generated_at": "2026-05-25T10:30:01Z",
    "n_buy_events": 120,
    "n_sell_events": 104,
    "toxicity_plus": 0.035,
    "toxicity_minus": 0.012
  }
}
```

Backwards-compatible readers may parse v1 files for diagnostics, but trading
readiness must require schema version 2.

### Parameter Validation

Add:

```python
def _validate_params(self, symbol: str, params: dict, now: datetime) -> tuple[bool, str]:
    ...
```

Reject:

- missing required keys
- schema version not current
- nonfinite values
- nonpositive kappa
- negative lambda
- negative epsilon
- toxicity above `max_toxicity`
- `generated_at` older than `max_param_age_seconds`
- insufficient kappa fit diagnostics
- insufficient epsilon sample count
- active writer/status failure

### Collector Freshness

Replace "some Parquet exists" with:

```python
def _market_data_fresh(self, symbol: str, max_age_seconds: int = 30) -> tuple[bool, str]:
    latest_ts = self._latest_collector_timestamp(symbol)
    if latest_ts is None:
        return False, "no_collector_data"

    age = (datetime.now(timezone.utc) - latest_ts).total_seconds()
    if age > max_age_seconds:
        return False, f"collector_data_stale_{age:.1f}s"

    return True, "ok"
```

Use symbol derived from pair, not hardcoded `ETH`.

### Estimator Locking

Prevent overlapping estimator runs:

```python
_param_update_lock = threading.Lock()
_param_update_running = False
_last_param_update: datetime | None = None
```

`bot_loop_start()` must:

- skip if an update is already running
- throttle updates
- run one update at a time
- load params only after the update completes
- accept new params only after validation
- keep last known good params if validation fails
- log failure reason

Long-term preferred architecture: run the estimator as a separate service and
let the strategy consume atomic snapshots only.

### Acceptance Criteria

The strategy refuses to trade unless:

- parameter schema is current
- all params are finite and in allowed ranges
- param age is within threshold
- collector data age is within threshold
- kappa/lambda fit diagnostics pass
- epsilon sample diagnostics pass
- no estimator process is currently writing files

## Phase 6 - Replace Candle Backtesting With Event Replay

Freqtrade candle backtests are not queue-aware maker-fill simulations. Build an
event replay before serious dry-run conclusions.

### Replay Input

Read:

```text
HL_data/<SYMBOL>/prices/*.parquet
HL_data/<SYMBOL>/trades/*.parquet
HL_data/<SYMBOL>/orderbooks/*.parquet
```

### Replay Loop

For each event timestamp:

```text
update order book
update trades
refresh model params on schedule
compute q from simulated inventory
compute HJB bid/ask
round price/amount
apply post-only check
place/cancel simulated orders
simulate fills with queue model
update cash, inventory, PnL, fees
write metrics
```

### Queue Models

Version 1:

- If quote is strictly worse than best, fill only if the book moves through it.
- If quote joins best, fill only after traded volume at that price exceeds
  estimated queue ahead.
- If quote crosses, post-only rejects.

Version 2:

- Use orderbook size at level as `queue_ahead`.
- Subtract marketable volume.
- Include conservative queue decay for cancellations.

Version 3:

- Calibrate effective fill probability from dry-run/testnet logs.

### Latency

Defaults:

```python
decision_latency_ms = 250
order_ack_latency_ms = 250
cancel_latency_ms = 250
```

Quote active time:

```text
t + decision_latency_ms + order_ack_latency_ms
```

Cancel effective time:

```text
t + cancel_latency_ms
```

### Fees And Funding

Replay must accept a fee schedule input and track:

- maker fees
- taker fees
- funding payments
- margin/futures effects where available

### Replay Metrics

Minimum metrics:

- realized spread
- markout after 100 ms, 1 s, 5 s, and 30 s
- maker ratio
- post-only reject ratio
- fill ratio by depth
- PnL by side
- inventory histogram
- time at q boundary
- stale quote cancels
- kappa/lambda/epsilon time series
- toxicity time series

### Acceptance Criteria

- Realized spread after fees is positive or explainable.
- Maker ratio is greater than 99%.
- Taker fills are zero by construction.
- Inventory stays within q bounds.
- PnL is not dominated by one directional price move.
- Markouts after fills are not strongly negative.
- Bad params and stale data cause refusal to quote.

## Phase 7 - Observability And Kill Switches

Expand debug logging into an audit trail.

### Quote Decision Log

Every quote decision should log:

```json
{
  "event": "quote_decision",
  "ts": "...",
  "pair": "ETH/USDC:USDC",
  "side": "bid",
  "q": 1,
  "signed_base_position": 0.01,
  "inventory_unit_base": 0.01,
  "mid": 4322.05,
  "best_bid": 4322.00,
  "best_ask": 4322.10,
  "raw_price": 4321.72,
  "rounded_price": 4321.70,
  "delta_model": 0.20,
  "fee_cushion": 1.30,
  "delta_total": 1.50,
  "kappa_plus": 2.88,
  "kappa_minus": 1.08,
  "lambda_plus": 0.136,
  "lambda_minus": 0.095,
  "epsilon_plus": 0.0,
  "epsilon_minus": 0.0,
  "hjb_generation": 42,
  "param_age_seconds": 12.4,
  "book_age_ms": 180,
  "post_only_verified": false,
  "decision": "reject",
  "reason": "post_only_not_verified"
}
```

### Standard Rejection Reasons

Use stable reason strings:

```text
no_hjb_cache
stale_hjb
stale_params
stale_orderbook
stale_collector_data
invalid_rate
invalid_kappa
invalid_lambda
invalid_epsilon
toxicity_too_high
boundary_side_disabled
quote_crosses_spread
not_post_only_supported
post_only_not_verified
position_limit_reached
unexpected_short_position
drawdown_limit_reached
unexpected_taker_fill
estimator_running
param_schema_unsupported
```

### Kill Switches

Implement:

```python
kill_on_taker_fill = True
kill_on_unknown_liquidity_fill = True
max_param_age_seconds = 90
max_book_age_seconds = 5
max_toxicity = 1.5
max_abs_inventory_units = 3
max_daily_loss_usdc = 20
max_consecutive_losses = 10
max_post_only_reject_rate = 0.80
```

When triggered:

```text
set trading_enabled = False
set fail_closed_reason
cancel open orders if possible
write kill_switch event
```

### Health Log

Every minute log:

```json
{
  "event": "health",
  "trading_enabled": false,
  "fail_closed_reason": "initial_safety_lock",
  "collector_fresh": true,
  "params_fresh": true,
  "hjb_fresh": true,
  "open_orders": 0,
  "position": 0.01,
  "q": 1,
  "maker_fills": 23,
  "taker_fills": 0,
  "post_only_rejects": 5,
  "realized_pnl": 1.23,
  "unrealized_pnl": -0.21
}
```

### Acceptance Criteria

Logs alone can answer:

- Why was this order accepted or rejected?
- Was it maker-safe at decision time?
- Which HJB generation produced it?
- Which params produced the HJB?
- Was market data fresh?
- Was post-only verified?
- What fee was expected and paid?
- What was markout after fill?
- Which kill switch fired, if any?

## Phase 8 - Deployment Gates

Do not jump from code changes to live trading.

### Gate 1 - Static Checks

```bash
python -m compileall scripts user_data/strategies
pytest tests/test_hjb.py
pytest tests/test_strategy_guards.py
pytest tests/test_params.py
```

Pass criteria:

- all tests pass
- no unexpected NaN or inf
- only intentional disabled boundary deltas are inf
- no callback signature mismatch

### Gate 2 - Deterministic Dry-Run With Trading Disabled

Settings:

```text
trading_enabled = False
dry_run = True
```

Pass criteria:

- bot starts
- collector writes fresh data
- params update atomically
- HJB refreshes
- health logs are produced
- zero orders placed

### Gate 3 - Dry-Run With Order Creation Enabled

Settings:

```text
trading_enabled = True
dry_run = True
stake_amount = 25
post_only_verified = false
```

Pass criteria:

- model-valid orders are attempted only in dry-run
- invalid HJB/params/book cause explicit rejection
- no `proposed_rate` fallback behavior
- boundary states disable the correct side
- live mode remains unavailable while post-only is unverified

### Gate 4 - Hyperliquid Testnet Or Tiny Integration

Use smallest order size satisfying exchange minimums.

Pass criteria:

- passive ALO orders rest
- crossing ALO orders reject instead of filling
- actual order TIF is confirmed as ALO
- liquidity flag and fee are logged
- unexpected taker fill triggers kill switch

### Gate 5 - Event Replay Over Multiple Days

Pass criteria:

- PnL survives fee sensitivity
- PnL survives latency sensitivity
- PnL survives kappa/lambda/epsilon perturbation
- PnL is not only directional drift
- inventory boundaries are respected
- markouts are acceptable

### Gate 6 - Live Canary

Only if all previous gates pass.

Settings:

```text
tiny fixed stake
one symbol
hard daily loss limit
post-only required
kill-on-taker-fill required
manual monitoring required
```

Pass criteria:

- several sessions with zero taker fills
- strategy logs match exchange fills
- no stale-data orders
- no parameter read/write errors
- kill switch tested and observable

## Concrete File Changes

### `scripts/hjb.py`

- Fix boundary deltas.
- Fix asymmetric boundary HJB contribution.
- Return disabled side as `np.inf`.
- Add method and boundary metadata.
- Add validation for `q_max`, kappa, lambda, epsilon, alpha, phi, horizon.

### `scripts/compute_spreads.py`

- Render `np.inf` as `DISABLED`.
- Do not add fee cushion to `np.inf`.
- Show boundary states visually.
- Include parameter metadata and age when schema v2 is available.

### `user_data/strategies/Market_Making.py`

- Add `trading_enabled` and `fail_closed_reason`.
- Fix `custom_entry_price()` signature.
- Add `confirm_trade_entry()`.
- Add `confirm_trade_exit()`.
- Remove reliance on `None` returns to skip orders.
- Replace open-trade-count inventory.
- Remove `minimal_roi = {"0": -1}`.
- Add maker-safety checks.
- Add tick and lot rounding.
- Reject stale params, stale HJB, and stale orderbook.
- Stop hardcoded `ETH` data path.
- Log signed inventory, quote decisions, health, kill switches, and rejection reasons.

### `user_data/config.json`

- Keep `dry_run: true`.
- Set force-entry fields to false.
- Set `stake_amount: 25`.
- Set `tradable_balance_ratio: 0.10`.
- Set `fee: 0.00015`.
- Add `custom_price_max_distance_ratio: 0.05`.
- Do not treat `PO` as live-safe until `PO -> Alo` is verified.

### `user_data/strategies/periodic_test_runner.py`

- Add single-run locking/status.
- Avoid overlapping runs.
- Write status atomically.
- Pass symbol from config/CLI, not hardcoded `ETH`.
- Copy or publish params only after successful validation.

### `scripts/get_kappa.py`

- Use atomic JSON writes.
- Write schema version 2.
- Write diagnostics, timestamps, units, and sample counts.
- Reject poor fit or too few points.
- Label HJB lambda as `lambda0_fit`.

### `scripts/get_epsilon.py`

- Use atomic JSON writes.
- Include sample counts, estimator type, window length, timestamps, units, and toxicity metrics.
- Do not silently set missing epsilon to zero without a status field.

### `scripts/get_lambda.py`

- Keep output as raw trade arrival monitoring.
- Label output as `lambda_raw`.
- Do not feed raw lambda into HJB unless explicitly intended and documented.

### New Files

- `tests/test_hjb.py`
- `tests/test_strategy_guards.py`
- `tests/test_params.py`
- `docs/UNITS.md`
- Replay simulator module/script, preferably under `scripts/replay_market_maker.py` or `scripts/replay/`

## Suggested Implementation Order

### Day 1 - Safety And HJB Correctness

1. Add `trading_enabled = False`.
2. Add HJB boundary tests.
3. Fix `scripts/hjb.py`.
4. Fix `scripts/compute_spreads.py`.
5. Verify:

```text
q = -q_max => ask disabled
q = +q_max => bid disabled
```

### Day 2 - Strategy Fail-Closed Logic

1. Fix callback signatures.
2. Add `confirm_trade_entry()`.
3. Add `confirm_trade_exit()`.
4. Remove `minimal_roi = {"0": -1}`.
5. Add maker-safety checks.
6. Add stale-model checks.
7. Test that custom price `None` cases do not place fallback orders.

### Day 3 - Inventory And Units

1. Define `inventory_unit_base`.
2. Replace `_inventory_level()`.
3. Add signed-position logging.
4. Lock long-only Freqtrade behavior.
5. Turn on small nonzero `alpha` and `phi`.
6. Add inventory skew tests.

### Day 4 - Parameter Pipeline

1. Add atomic writes.
2. Add schema metadata.
3. Add validation.
4. Add freshness checks.
5. Remove hardcoded `ETH`.
6. Prevent concurrent estimator runs.

### Days 5 To 7 - Replay Simulator

1. Build basic event replay.
2. Add post-only simulation.
3. Add fee accounting.
4. Add latency.
5. Compare replay quotes with strategy logs.

### After That - Testnet Only

1. Verify ALO/post-only behavior.
2. Verify maker/taker classification.
3. Verify kill-on-taker-fill.
4. Verify fee tier.
5. Run tiny canary only after logs prove correct behavior.

## Minimum Viable Fixed Version

The smallest version safe for serious dry-run has:

- HJB boundary bug fixed.
- `custom_entry_price()` signature fixed.
- `confirm_trade_entry()` and `confirm_trade_exit()` added.
- Invalid custom prices cannot fall back into live orders.
- `minimal_roi = {"0": -1}` removed.
- Inventory q based on signed exposure.
- Nonzero `alpha` or `phi`.
- Params loaded atomically with timestamps.
- Stale params and stale data reject orders.
- Local maker-safety guard added.
- Post-only/Alo verified or trading disabled.
- Fee mismatch fixed.
- Event replay exists.

Until those are done, this remains an educational model wired into Freqtrade,
not a production-ready market maker.

## References

- [Freqtrade configuration](https://www.freqtrade.io/en/stable/configuration/)
- [Freqtrade strategy callbacks](https://www.freqtrade.io/en/stable/strategy-callbacks/)
- [Hyperliquid exchange endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint)
- [Hyperliquid order types](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/order-types)
- [Hyperliquid fees](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
