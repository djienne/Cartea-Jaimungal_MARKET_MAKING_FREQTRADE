# Safety Gate Results

- PASS `compileall` (1.198s)
- PASS `pytest_core` (24.695s)
- PASS `compute_spreads_boundary_smoke` (1.447s)
- PASS `replay_smoke` (1.836s)
- PASS `post_only_probe_plan` (0.071s)
- PASS `post_only_evidence_report` (0.103s)
- PASS `direct_alo_adapter_plan` (0.117s)
- PASS `docker_compose_config` (0.349s)
- PASS `freqtrade_runtime_load` (7.984s)
- PASS `dry_run_disabled_smoke` (224.715s)
- PASS `dry_run_enabled_smoke` (313.84s)
- PASS `replay_log_calibration_artifact` (0.906s)
- PASS `fee_evidence_report` (0.086s)
- PASS `hl_data_validation_report` (2.098s)
- PASS `replay_latest_data_smoke` (3.748s)
- PASS `replay_acceptance_report_artifact` (17.065s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.