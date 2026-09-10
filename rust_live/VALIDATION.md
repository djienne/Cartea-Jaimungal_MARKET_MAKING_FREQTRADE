# Validation status

> **Dated evidence, not a status report.** Current operational guidance is
> `README.md` and `../docs/DRY_RUN_GRID.md`; latency distributions are
> `PERFORMANCE.md`; protocol behaviour and release gates are
> `HYPERLIQUID_LIVE_CONNECTOR.md`. Results below predate estimator schema v5
> (2026-09-02), which rescaled `lambda_±` by the survival fit's intercept `A`
> (1.04 / 0.99 on CASHCAT). Re-score with `mm-live replay` rather than reading
> P&L off this file.

The stateful continuous `live` backend is implemented; the tracked CASHCAT
profile ships `live.enabled=false`. An explicitly authorized acceptance campaign
exercised minimum-notional real actions on a dedicated CASHCAT subaccount and
ended flat with zero open orders.

## Coverage

`cj-core` HJB and quote policy are above 92% line coverage; calibration above
91%, dry-run execution above 88%, latency aggregation above 92%, live state
above 87%, signing above 89%, public wire decoding above 90%.

The all-feature workspace total is about 52%, reported rather than hidden:
real-account acceptance, CLI orchestration and much of the network-backed live
backend are intentionally not executed by normal CI. CI hard-gates the pure
core, dry-run execution and the runtime crates, and publishes the full connector
report for review.

Current suite sizes are 302 Rust tests (`cargo test --workspace --all-features`)
and 255 Python (`pytest tests/`). The Python count fell by 137 on 2026-09-10
when the Python replay was deleted.

## Defects this campaign found

These are the reason the campaign was worth running, and each is pinned by a
regression test.

- **A mixed-age `trades` snapshot caused a false reconnect.** The initial
  subscription snapshot's 30 rows spanned about eight seconds and were not
  ordered the way the row-level transition assumed. Fixed with a frame-level
  startup transition; the following bounded canary recorded three application
  pings and pongs with no idle timeout, reconnect, invalid message or causal
  drop.
- **The venue reports integer sizes with a trailing zero.** CASHCAT size came
  back as `88.0`, which the account-decimal parser rejected. After the fix a
  reduce-only recovery sold all 88 at 0.11523, fees 0.009119 USDC, closed PnL
  +0.01408 USDC, and the account was independently confirmed flat. A regression
  test pins redundant trailing-zero parsing.
- **`scheduleCancel` is refused below 1,000,000 USDC cumulative volume.** It was
  signed correctly and refused before any test order, on a subaccount with about
  1,930 USDC of volume. Production with `deadman_enabled=true` therefore fails
  closed on this account. The option can be disabled explicitly, but no dead-man
  trigger is claimed as validated.

## What the real-money campaign established

Connector correctness, not strategy profitability. A passive ALO was placed,
rested and cancelled by its own CLOID; an IOC round trip bought 88 at 0.11451
and closed reduce-only at 0.11436 about 1.82 s later, losing 0.022262 USDC
including fees, with order and fill stream events received and the account flat
afterwards. Most of the campaign's loss came from an adverse move during it and
is retained as real evidence rather than hidden.

A bounded continuous-production smoke ran the actual Cartea-Jaimungal hot loop,
public and account sockets, background calibration and the live backend with the
latency gate enforced, submitting zero orders, cancels or dead-man actions.

## Remaining boundary

Continuous live operation is gated on a strictly profitable, promotable dry-run
row; `../docs/CASHCAT_LIVE_PROMOTION.md` holds the selection rule and the
canary/arm sequence.
