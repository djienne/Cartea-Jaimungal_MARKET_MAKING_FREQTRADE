# Safety Gate Results

- PASS `compileall` (1.076s)
- PASS `pytest_core` (22.528s)
- PASS `compute_spreads_boundary_smoke` (1.302s)
- PASS `replay_smoke` (2.09s)
- PASS `post_only_probe_plan` (0.104s)
- PASS `post_only_evidence_report` (0.088s)
- PASS `direct_alo_adapter_plan` (0.11s)
- PASS `docker_compose_config` (0.273s)
- PASS `freqtrade_runtime_load` (7.047s)
- PASS `dry_run_disabled_smoke` (224.865s)
- PASS `dry_run_enabled_smoke` (319.453s)
- PASS `replay_log_calibration_artifact` (1.097s)
- PASS `fee_evidence_report` (0.08s)
- PASS `hl_data_validation_report` (2.176s)
- PASS `replay_latest_data_smoke` (4.112s)
- PASS `replay_acceptance_report_artifact` (17.894s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.