# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.055s)
- PASS `pytest_core` (24.778s)
- PASS `config_safety_report` (0.062s)
- PASS `strategy_safety_report` (0.085s)
- PASS `compute_spreads_boundary_smoke` (1.328s)
- PASS `replay_smoke` (2.015s)
- PASS `post_only_probe_plan` (0.06s)
- PASS `post_only_evidence_report` (0.063s)
- PASS `direct_alo_adapter_plan` (0.071s)
- PASS `hyperliquid_fee_capture_plan` (0.068s)
- PASS `docker_compose_config` (0.154s)
- PASS `freqtrade_runtime_load` (7.343s)
- PASS `dry_run_disabled_smoke` (224.475s)
- PASS `dry_run_enabled_smoke` (313.699s)
- PASS `replay_log_calibration_artifact` (0.736s)
- PASS `fee_evidence_report` (0.083s)
- PASS `hl_data_validation_report` (1.175s)
- PASS `replay_latest_data_smoke` (3.34s)
- PASS `replay_acceptance_report_artifact` (16.149s)
- PASS `live_canary_evidence_report` (0.185s)

Post-run audits:
- PASS `plan_status_audit` (0.067s)

Manual/external gate evidence:
- WAIT `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely.
  - reason: `ok_not_true`
- WAIT `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
  - reason: `ok_not_true`
- WAIT `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
  - reason: `ok_not_true`
- WAIT `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.
  - reason: `ok_not_true`

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.