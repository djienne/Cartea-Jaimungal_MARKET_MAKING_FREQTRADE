# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.034s)
- PASS `pytest_core` (25.595s)
- PASS `config_safety_report` (0.059s)
- PASS `strategy_safety_report` (0.099s)
- PASS `compute_spreads_boundary_smoke` (1.504s)
- PASS `replay_smoke` (1.805s)
- PASS `post_only_probe_plan` (0.103s)
- PASS `post_only_evidence_report` (0.085s)
- PASS `direct_alo_adapter_plan` (0.117s)
- PASS `hyperliquid_fee_capture_plan` (0.07s)
- PASS `docker_compose_config` (0.156s)
- PASS `freqtrade_runtime_load` (7.908s)
- PASS `dry_run_disabled_smoke` (224.788s)
- PASS `dry_run_enabled_smoke` (313.822s)
- PASS `replay_log_calibration_artifact` (0.779s)
- PASS `fee_evidence_report` (0.081s)
- PASS `hl_data_validation_report` (1.926s)
- PASS `replay_latest_data_smoke` (3.043s)
- PASS `replay_acceptance_report_artifact` (15.936s)
- PASS `live_canary_evidence_report` (0.148s)

Post-run audits:
- PASS `plan_status_audit` (0.065s)

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