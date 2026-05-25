# Deployment Gates

This project must stay fail-closed until every automated and manual gate below
has evidence. The local checks can be run with:

```bash
python scripts/run_safety_gates.py --markdown-output docs/LAST_SAFETY_GATES.md
```

When Docker is available, include non-trading runtime load checks:

```bash
python scripts/run_safety_gates.py --include-runtime --markdown-output docs/LAST_SAFETY_GATES.md
```

## Automated Local Gates

These gates do not require a live exchange connection:

- `compileall`: Python syntax/import bytecode checks for `scripts` and
  `user_data/strategies`.
- `pytest_core`: HJB math, strategy guard, parameter writer, and replay smoke
  tests.
- `compute_spreads_boundary_smoke`: verifies disabled HJB boundary sides render
  as disabled instead of finite quotes.
- `replay_smoke`: verifies the replay CLI runs without candle-fill assumptions.
- `post_only_probe_plan`: documents the Hyperliquid `Alo` evidence required
  without placing orders.
- `post_only_evidence_report`: writes
  `docs/post_only_evidence_report.json`. Without submit artifacts this command
  is expected to return nonzero and the gate runner treats that as an artifact
  check, not as exchange proof. The report must become `ok=true` before live
  trading can be considered.
- `direct_alo_adapter_plan`: writes `docs/direct_alo_adapter_plan.json`,
  documenting the no-network direct Hyperliquid SDK fallback path that submits
  `order_type={"limit": {"tif": "Alo"}}` only after local BBO maker-safety.

Optional Docker runtime gates:

- `docker_compose_config`: validates Compose configuration.
- `freqtrade_runtime_load`: runs `freqtrade list-strategies` inside the
  configured container and requires `Market_Making` to load with status `OK`.
- `dry_run_disabled_smoke`: starts the configured bot briefly in dry-run with
  the default safety lock, stops it, stores logs, and fails if any order creation
  line appears.
- `dry_run_enabled_smoke`: starts the public collector, writes temporary
  schema-v2 OK parameter snapshots under `user_data/logs`, starts Freqtrade with
  `dry_run=true` and `trading_enabled=true` through a temporary config, and
  requires fresh collector data, fresh params, fresh HJB, an accepted quote
  decision, and dry-run order evidence. Checked-in config and checked-in params
  remain fail-closed. Strategy fill callbacks also schedule delayed
  `fill_markout` audit events at 100 ms, 1 s, 5 s, and 30 s after fills when
  fill price, size, and side are available. Quote, health, and fill logs include
  fee snapshots that compare the strategy maker fee with config, exchange
  metadata, and actual fill fee rates when available. Trading enablement and
  quote confirmation reject config fee mismatches; quote confirmation also
  rejects exchange maker-fee mismatches when exchange metadata is available.
  When post-only support is marked verified, final entry/exit confirmation also
  rejects non-post-only time-in-force values before submission, and any fill
  reported with a non-post-only or missing TIF triggers a kill switch.
- `replay_latest_data_smoke`: after runtime collector gates have produced fresh
  shards, replays the newest local data window and writes
  `docs/replay_latest_smoke.json` with input coverage, maker/taker counts,
  post-only rejects, stale cancels, inventory, PnL, and markout samples where
  fills occur.
- `replay_acceptance_report_artifact`: runs baseline, fee, latency, and
  parameter-perturbation replay variants and writes
  `docs/replay_acceptance_report.json` plus a markdown summary. The gate command
  uses `--allow-incomplete` so the artifact is always produced; the report's own
  `ok` field remains false until the multi-day acceptance criteria are met. The
  report includes a directional-drift ratio so mark-to-market PnL that is not
  explained by net realized spread cannot masquerade as maker edge. Replay
  quotes and fill accounting both use the configured maker fee, so fee
  sensitivity changes quote distances as well as realized fees. The replay
  queue model includes queue-ahead traded volume plus conservative cancellation
  decay, reported as `queue_decay_base`. The report also includes refusal checks
  for bad parameters and stale collector data, proving those scenarios reject
  quoting instead of silently producing orders. Replay metrics also track
  starting equity, leverage, notional exposure, margin used, maintenance margin,
  equity, liquidation buffer, and maintenance-margin breach counts; acceptance
  fails any variant with a maintenance-margin breach.

Data freshness/replay-readiness report:

```bash
python scripts/validate_hl_data.py --symbol ETH --newest-per-stream 25 --max-age-seconds 30 --output docs/hl_data_validation.json
```

The freshness report is intentionally separate from the default pass/fail gate:
historical replay can use older data, but live/dry-run quoting must refuse stale
collector data.

Passing these gates means the code is locally coherent. It does not mean the bot
is safe to trade.

## Manual Runtime Gates

These gates require the real Freqtrade/Hyperliquid runtime:

- `deterministic_dry_run_trading_disabled`: covered by
  `dry_run_disabled_smoke` when Docker is available. It proves the bot starts,
  sees fresh collector data, writes health logs, and creates zero orders while
  the default lock is active.
- `dry_run_order_creation`: covered by `dry_run_enabled_smoke` when Docker is
  available. It proves a temporary unlocked dry-run can accept a valid passive
  quote and create a dry-run order, while checked-in config/params remain
  locked.
- `hyperliquid_post_only_mapping`: Freqtrade `PO` is currently not verified for
  Hyperliquid. In local runtime evidence, Freqtrade 2025.4 rejected Hyperliquid
  `PO` as unsupported, so the dry-run harness uses `GTC` and live trading remains
  blocked. See `docs/POST_ONLY_VERIFICATION.md`.
  Prove a native `Alo` path before live use; intentionally crossing `Alo` orders
  must reject or cancel without filling, and passive `Alo` orders must rest,
  cancel, or fill maker-only.
  Start with the no-network probe plan:

```bash
python scripts/verify_post_only_mapping.py --mode plan
```

  Any submit-mode use must be run only on testnet or tiny canary size with
  explicit acknowledgement flags and retained exchange logs. The canonical
  checker is:

```bash
python scripts/verify_post_only_mapping.py --mode evaluate-evidence --crossing-result docs/post_only_crossing_result.json --passive-result docs/post_only_passive_result.json --output docs/post_only_evidence_report.json
```

  If Freqtrade remains unable to submit native `Alo`, the direct SDK fallback
  scaffold is:

```bash
python scripts/hyperliquid_alo_executor.py --mode plan
```

  Direct SDK submit artifacts from `hyperliquid_alo_executor.py` can be fed to
  the same `verify_post_only_mapping.py --mode evaluate-evidence` checker. The
  checker normalizes SDK `order_type={"limit": {"tif": "Alo"}}` and evaluates
  the adapter's resting/rejected/fill classification. Use
  `hyperliquid_alo_executor.py --mode submit-crossing-alo` for a direct SDK
  rejection probe and `--mode submit-passive-alo` for passive resting/maker
  evidence with automatic cancellation of any resting order ids returned by the
  SDK. Plain `--mode submit-alo` is reserved for the future direct execution
  layer and may leave a passive order working.
- `multi_day_event_replay`: the automated latest-data smoke proves the replay
  parser and conservative fill loop work on real shards, and the acceptance
  report records exactly which criteria are still failing. The remaining manual
  gate is a several-day replay with acceptable realized spread, markouts, maker
  ratio, inventory, latency, fee sensitivity, parameter perturbation, and
  directional-drift attribution.
- `live_canary`: only after all previous gates pass, with tiny fixed stake,
  one symbol, hard loss limits, post-only required, and kill-on-taker-fill
  enabled.

## Current Safety Posture

- The strategy default is `trading_enabled = False`.
- Checked-in parameter snapshots are schema v2 but have
  `status = "seeded_unverified"`, so the strategy rejects them.
- A real estimator run must atomically replace the snapshots with
  `status = "ok"`, current timestamps, and sufficient diagnostics before any
  quote can pass the guards.
- Live trading is still blocked unless post-only support is verified outside
  dry-run.
