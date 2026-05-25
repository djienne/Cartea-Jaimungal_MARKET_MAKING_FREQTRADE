# Safety Gate Results

- PASS `compileall` (1.062s)
- PASS `pytest_core` (24.483s)
- PASS `compute_spreads_boundary_smoke` (1.795s)
- PASS `replay_smoke` (1.961s)
- PASS `post_only_probe_plan` (0.106s)
- PASS `post_only_evidence_report` (0.087s)
- PASS `direct_alo_adapter_plan` (0.121s)
- PASS `docker_compose_config` (0.285s)
- PASS `freqtrade_runtime_load` (8.87s)
- PASS `dry_run_disabled_smoke` (224.88s)
- PASS `dry_run_enabled_smoke` (254.224s)
- PASS `replay_latest_data_smoke` (3.326s)
- PASS `replay_acceptance_report_artifact` (17.122s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.