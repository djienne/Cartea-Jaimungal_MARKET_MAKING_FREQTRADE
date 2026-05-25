# Replay Acceptance Report

- status: FAIL
- symbol: `ETH`
- generated_at: `2026-05-25T18:25:22Z`

## Variant Summary

| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Net spread | Directional ratio | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | FAIL | 0.009834 | 3942 | 2 | 0 | 0.01797501 | 1.449095 | insufficient_coverage_days:0.009834<min_3.000000, directional_drift_dominates_pnl:1.449095>max_0.750000 |
| fee_2x | FAIL | 0.009834 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.009834<min_3.000000, no_maker_fills |
| latency_2x | FAIL | 0.009834 | 3942 | 2 | 0 | 0.01797501 | 1.449095 | insufficient_coverage_days:0.009834<min_3.000000, directional_drift_dominates_pnl:1.449095>max_0.750000 |
| params_soft | FAIL | 0.009834 | 3942 | 2 | 0 | 0.02085292 | 1.561361 | insufficient_coverage_days:0.009834<min_3.000000, directional_drift_dominates_pnl:1.561361>max_0.750000 |
| params_hard | FAIL | 0.009834 | 3942 | 2 | 0 | 0.01651541 | 1.398110 | insufficient_coverage_days:0.009834<min_3.000000, directional_drift_dominates_pnl:1.398110>max_0.750000 |

## Refusal Checks

| Check | Status | Expected | Decision | Reason |
| --- | --- | --- | --- | --- |
| bad_params_nonpositive_kappa | PASS | reject | reject | invalid_kappa |
| bad_params_toxicity | PASS | reject | reject | toxicity_too_high |
| stale_collector_data | PASS | reject | reject | stale_collector_data |

## Blocking Reasons

- `baseline:insufficient_coverage_days:0.009834<min_3.000000`
- `baseline:directional_drift_dominates_pnl:1.449095>max_0.750000`
- `fee_2x:insufficient_coverage_days:0.009834<min_3.000000`
- `fee_2x:no_maker_fills`
- `latency_2x:insufficient_coverage_days:0.009834<min_3.000000`
- `latency_2x:directional_drift_dominates_pnl:1.449095>max_0.750000`
- `params_soft:insufficient_coverage_days:0.009834<min_3.000000`
- `params_soft:directional_drift_dominates_pnl:1.561361>max_0.750000`
- `params_hard:insufficient_coverage_days:0.009834<min_3.000000`
- `params_hard:directional_drift_dominates_pnl:1.398110>max_0.750000`