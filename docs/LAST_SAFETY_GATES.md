# Safety Gate Results

- PASS `compileall` (1.225s)
- PASS `pytest_core` (25.513s)
- PASS `compute_spreads_boundary_smoke` (1.552s)
- PASS `replay_smoke` (2.339s)
- PASS `post_only_probe_plan` (0.106s)
- PASS `post_only_evidence_report` (0.101s)
- PASS `direct_alo_adapter_plan` (0.136s)
- PASS `docker_compose_config` (0.268s)
- PASS `freqtrade_runtime_load` (10.14s)
- PASS `dry_run_disabled_smoke` (225.003s)
- PASS `dry_run_enabled_smoke` (314.12s)
- PASS `replay_latest_data_smoke` (3.807s)
- PASS `replay_acceptance_report_artifact` (17.4s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.