# Safety Gate Results

- PASS `compileall` (1.089s)
- PASS `pytest_core` (22.893s)
- PASS `compute_spreads_boundary_smoke` (1.43s)
- PASS `replay_smoke` (2.121s)
- PASS `post_only_probe_plan` (0.116s)
- PASS `post_only_evidence_report` (0.103s)
- PASS `direct_alo_adapter_plan` (0.171s)
- PASS `docker_compose_config` (0.279s)
- PASS `freqtrade_runtime_load` (8.537s)
- PASS `dry_run_disabled_smoke` (224.942s)
- PASS `dry_run_enabled_smoke` (314.086s)
- PASS `replay_log_calibration_artifact` (0.57s)
- PASS `fee_evidence_report` (0.084s)
- PASS `hl_data_validation_report` (1.702s)
- PASS `replay_latest_data_smoke` (3.454s)
- PASS `replay_acceptance_report_artifact` (17.305s)
- PASS `live_canary_evidence_report` (0.166s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.