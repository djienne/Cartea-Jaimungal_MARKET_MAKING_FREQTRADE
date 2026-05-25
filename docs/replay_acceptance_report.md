# Replay Acceptance Report

- status: FAIL
- symbol: `ETH`
- generated_at: `2026-05-25T20:18:48Z`

## Variant Summary

| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Net spread | Directional ratio | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | FAIL | 0.010605 | 3821 | 1 | 0 | 0.00631415 | 1.684398 | insufficient_coverage_days:0.010605<min_3.000000, directional_drift_dominates_pnl:1.684398>max_0.750000 |
| fee_2x | FAIL | 0.010605 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.010605<min_3.000000, no_maker_fills |
| latency_2x | FAIL | 0.010605 | 3821 | 1 | 0 | 0.00631415 | 1.684398 | insufficient_coverage_days:0.010605<min_3.000000, directional_drift_dominates_pnl:1.684398>max_0.750000 |
| params_soft | FAIL | 0.010605 | 3821 | 1 | 0 | 0.00990811 | 1.893275 | insufficient_coverage_days:0.010605<min_3.000000, directional_drift_dominates_pnl:1.893275>max_0.750000 |
| params_hard | FAIL | 0.010605 | 3821 | 1 | 0 | 0.00581741 | 1.598340 | insufficient_coverage_days:0.010605<min_3.000000, directional_drift_dominates_pnl:1.598340>max_0.750000 |

## Refusal Checks

| Check | Status | Expected | Decision | Reason |
| --- | --- | --- | --- | --- |
| bad_params_nonpositive_kappa | PASS | reject | reject | invalid_kappa |
| bad_params_toxicity | PASS | reject | reject | toxicity_too_high |
| stale_collector_data | PASS | reject | reject | stale_collector_data |

## Blocking Reasons

- `baseline:insufficient_coverage_days:0.010605<min_3.000000`
- `baseline:directional_drift_dominates_pnl:1.684398>max_0.750000`
- `fee_2x:insufficient_coverage_days:0.010605<min_3.000000`
- `fee_2x:no_maker_fills`
- `latency_2x:insufficient_coverage_days:0.010605<min_3.000000`
- `latency_2x:directional_drift_dominates_pnl:1.684398>max_0.750000`
- `params_soft:insufficient_coverage_days:0.010605<min_3.000000`
- `params_soft:directional_drift_dominates_pnl:1.893275>max_0.750000`
- `params_hard:insufficient_coverage_days:0.010605<min_3.000000`
- `params_hard:directional_drift_dominates_pnl:1.598340>max_0.750000`