# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 5

- PASS `compileall` (1.003s)
- PASS `pytest_core` (13.827s)
- PASS `config_safety_report` (0.066s)
- PASS `strategy_safety_report` (0.102s)
- PASS `strategy_attribute_report` (0.125s)
- PASS `compute_spreads_boundary_smoke` (0.744s)
- PASS `replay_smoke` (0.97s)
- PASS `adapter_plans` (0.52s)
- PASS `post_only_evidence_report` (0.062s)
- PASS `live_canary_evidence_report` (0.095s)
- PASS `promotion_evidence_manifest` (0.084s)

Manual/external gate evidence:
- PASS `deterministic_dry_run_trading_disabled`: Requires running the bot loop and confirming zero orders plus health logs in Freqtrade logs.
- WAIT `freqtrade_runtime_load`: Requires a Freqtrade environment with exchange/config plugins installed.
  - reason: `no_machine_check`
- WAIT `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely.
  - reason: `ok_not_true`
- WAIT `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
  - reason: `ok_not_true`
- WAIT `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
  - reason: `ok_not_true`
- WAIT `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.
  - reason: `ok_not_true`

Manual gates still required:
- `freqtrade_runtime_load`: Requires a Freqtrade environment with exchange/config plugins installed.
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.