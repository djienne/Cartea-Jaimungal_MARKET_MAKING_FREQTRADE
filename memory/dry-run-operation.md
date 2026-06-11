---
name: dry-run-operation
description: How to run the market-making dry run and the runtime gotchas that block quoting
metadata:
  type: project
---

The Cartea-Jaimungal MM dry run only quotes when all three docker-compose services run together: `hl-collector2` (live HL data), `param-estimator` (writes status=ok kappa/epsilon/lambda into user_data/strategies), and `freqtrade` (reads snapshots, quotes). `market_making.trading_enabled` must be `true` (safe under `dry_run: true`; live-only gates are bypassed when dry-run).

Runtime gotchas discovered and fixed (2026-05-28), all previously prevented any quoting:
- Collector flushed every 60s but the strategy rejects collector data >30s old → made flush interval configurable (`FLUSH_INTERVAL_SEC`, default 10) + added retention pruning (`RETENTION_MINUTES`).
- `collector_timestamp_cache_seconds` defaulted to 90 (> the 30s freshness window) so the cached timestamp aged past it → lowered default to 5.
- `periodic_test_runner.py --loop` held `param_update.lock` / status `running` for the whole process lifetime; the strategy treats a present lock as "params invalid", so it NEVER quoted. Fixed: lock is now acquired/released per cycle (held only during compute+copy).
- freqtrade does NOT set `self.exchange` on strategies (only on FreqtradeBot); the strategy's tick/lot rounding was dead at runtime → wired `self.exchange = self.dp._exchange` in `bot_start`.
- Strategy assumed Hyperliquid maker fee 0.00015, but ccxt 4.4.77 reported 0.0001 → fee gate fail-closed. Made it a config knob `market_making.maker_fee_rate`. UPDATE 2026-06-11: ccxt is now pinned to 4.5.22 (see below), which reports the documented 0.00015 — config `fee`/`maker_fee_rate` are back to 0.00015. The invariant: these two must equal whatever the pinned ccxt's hyperliquid `swap` fee table says, or quoting fail-closes with `exchange_fee_mismatch`.

Exit-side bugs (only surfaced once a bid actually filled; all three blocked the ask/exit side so the bot could enter but never unwind):
- `_signed_base_position` called `Trade.get_trades(is_open=True, pair=pair)` — real freqtrade `Trade.get_trades` takes a SQLAlchemy filter list, not kwargs, so it raised TypeError, was swallowed by `except: pass`, and position read as 0 (q=0). Fixed: use `Trade.get_trades_proxy(...)` (the kwargs-safe, live/dry-run/backtest accessor) at all 3 sites. Tests missed it because DummyTrade stubbed `get_trades(**kwargs)`.
- get_epsilon wrote slightly negative ε (short-window noise) with status=ok, but the strategy rejects ε<0 (invalid_epsilon) → kept stale params. Fixed: floor ε at 0 in get_epsilon (C-J model treats ε≥0).
- freqtrade suppresses an exit signal when an entry signal is on the same candle (interface.py `exit_ and not enter`). The strategy set enter_long=1 at any q<3, cancelling exit_long at q=1. Fixed: only emit the bid when flat (q==0).

Verified two-sided quoting: bids below mid + asks above mid, on the 0.1 tick, maker-safe, ~1.7-4 bps half-spread, all gates green; full loop = place passive quote → 15s unfilled cancel → re-quote; fills flip q 0↔1 and the side switches. `exit_rejected: estimator_running` is the expected transient during a param-update lock window (re-quotes next loop). See [[mm-fee-assumption]].

Outage 2026-06-10/11 (~28h, RestartCount=6920): ccxt 4.4.77's `hyperliquid.fetch_spot_markets` crashed on new exchange spot metadata (`None` base token → TypeError) — `load_markets` died at startup, freqtrade crash-looped. Even ccxt 4.5.22 crashes on the same metadata; the durable fix (applied) is ccxt==4.5.22 pinned in Dockerfile.technical PLUS `exchange.ccxt_config.options.fetchMarkets.types=["swap"]` in config.json so the spot parser is never invoked (futures-only bot). Watch for: any freqtrade restart loop with `TypeError ... 'NoneType' and 'str'` in fetch_spot_markets.

Abandoned-lock gotcha: docker stops containers with SIGTERM; python's default SIGTERM action skips `finally`, so an estimator killed mid-cycle left `param_update.lock` behind — the next estimator then skips every cycle until the stale window (1h) and quoting stalls (params age out). Fixed: runner now traps SIGTERM → KeyboardInterrupt (cleanup path releases the lock), and lock release is ownership-token-checked so no runner can delete another's active lock. If it ever recurs: verify the estimator is idle (CPU ~0) and remove `user_data/strategies/param_update.lock` manually.

Gate-battery gotcha: `run_safety_gates.py --include-runtime` (the dry-run smoke gates) stops the main `MM_ADV` freqtrade container to run its own instance and leaves it STOPPED afterwards — quoting silently halts. After any `--include-runtime` run: `docker rm MM_ADV` if a stale exited container holds the name, then `docker compose up -d freqtrade`, and confirm `Parameter source: redis` + `Trading symbol: ETH` in the logs.
