# Replay Acceptance Report

- status: FAIL
- symbol: `ETH`
- generated_at: `2026-05-26T09:04:05Z`

## Variant Summary

| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Post-only reject % | Stale cancel % | Net spread | Directional ratio | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | FAIL | 0.005291 | 2000 | 0 | 0 | 0.00% | 100.00% | 0.00000000 | 0.000000 | stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000, insufficient_coverage_days:0.005291<min_3.000000, no_maker_fills |
| fee_2x | FAIL | 0.005291 | 2000 | 0 | 0 | 0.00% | 100.00% | 0.00000000 | 0.000000 | stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000, insufficient_coverage_days:0.005291<min_3.000000, no_maker_fills |
| latency_2x | FAIL | 0.005291 | 2000 | 0 | 0 | 0.00% | 100.00% | 0.00000000 | 0.000000 | stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000, insufficient_coverage_days:0.005291<min_3.000000, no_maker_fills |
| params_soft | FAIL | 0.005291 | 2000 | 0 | 0 | 0.00% | 100.00% | 0.00000000 | 0.000000 | stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000, insufficient_coverage_days:0.005291<min_3.000000, no_maker_fills |
| params_hard | FAIL | 0.005291 | 3465 | 1 | 0 | 0.00% | 99.97% | 0.00782476 | 1.769000 | stale_quote_cancel_ratio_above_threshold:0.999711>max_0.990000, insufficient_coverage_days:0.005291<min_3.000000, directional_drift_dominates_pnl:1.769000>max_0.750000 |

## Refusal Checks

| Check | Status | Expected | Decision | Reason |
| --- | --- | --- | --- | --- |
| bad_params_nonpositive_kappa | PASS | reject | reject | invalid_kappa |
| bad_params_toxicity | PASS | reject | reject | toxicity_too_high |
| stale_collector_data | PASS | reject | reject | stale_collector_data |

## Blocking Reasons

- `baseline:stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000`
- `baseline:insufficient_coverage_days:0.005291<min_3.000000`
- `baseline:no_maker_fills`
- `fee_2x:stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000`
- `fee_2x:insufficient_coverage_days:0.005291<min_3.000000`
- `fee_2x:no_maker_fills`
- `latency_2x:stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000`
- `latency_2x:insufficient_coverage_days:0.005291<min_3.000000`
- `latency_2x:no_maker_fills`
- `params_soft:stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000`
- `params_soft:insufficient_coverage_days:0.005291<min_3.000000`
- `params_soft:no_maker_fills`
- `params_hard:stale_quote_cancel_ratio_above_threshold:0.999711>max_0.990000`
- `params_hard:insufficient_coverage_days:0.005291<min_3.000000`
- `params_hard:directional_drift_dominates_pnl:1.769000>max_0.750000`