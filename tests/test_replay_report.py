from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_replay_report import (  # noqa: E402
    REQUIRED_MARKOUT_HORIZONS_MS,
    build_refusal_checks,
    coverage_days,
    directional_drift_ratio,
    directional_pnl_proxy_usdc,
    evaluate_metrics,
    markout_summary,
    render_markdown,
    replay_param_guard,
)


def minimal_metrics(**overrides):
    payload = {
        "data_start": "2026-05-25T00:00:00Z",
        "data_end": "2026-05-28T00:00:00Z",
        "quote_attempts": 1000,
        "maker_fills": 10,
        "taker_fills": 0,
        "maker_ratio": 1.0,
        "realized_spread_usdc": 1.0,
        "fees_usdc": 0.1,
        "mark_to_market_pnl_usdc": 0.9,
        "pnl_by_side": {"bid": 0.8, "ask": 0.1},
        "inventory_histogram": {"0": 900, "1": 100},
        "quote_attempts_by_depth": {"bid:5.00bps": 900, "ask:5.00bps": 100},
        "fills_by_depth": {"bid:5.00bps": 9, "ask:5.00bps": 1},
        "fill_ratio_by_depth": {"bid:5.00bps": 0.01, "ask:5.00bps": 0.01},
        "markout_samples": [
            {"horizon_ms": horizon_ms, "markout_usdc": 0.02}
            for horizon_ms in REQUIRED_MARKOUT_HORIZONS_MS
        ],
        "parameter_series": [
            {
                "schema_version": 1,
                "ts": "2026-05-25T00:00:00Z",
                "source": "static_replay_params",
                "symbol": "ETH",
                "kappa_plus": 2.0,
                "kappa_minus": 2.0,
                "lambda_plus": 0.1,
                "lambda_minus": 0.1,
                "epsilon_plus": 0.0,
                "epsilon_minus": 0.0,
            }
        ],
        "toxicity_series": [
            {
                "schema_version": 1,
                "ts": "2026-05-25T00:00:00Z",
                "source": "static_replay_params",
                "symbol": "ETH",
                "toxicity_plus": 0.0,
                "toxicity_minus": 0.0,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_coverage_days_from_replay_metrics():
    assert coverage_days(minimal_metrics()) == 3.0


def test_markout_summary_groups_horizons():
    summary = markout_summary(
        [
            {"horizon_ms": 100, "markout_usdc": 0.01},
            {"horizon_ms": 100, "markout_usdc": 0.03},
            {"horizon_ms": 1000, "markout_usdc": -0.01},
        ]
    )

    assert summary["100"]["count"] == 2.0
    assert summary["100"]["mean_usdc"] == 0.02
    assert summary["1000"]["min_usdc"] == -0.01


def test_evaluate_metrics_passes_when_acceptance_is_met():
    ok, reasons = evaluate_metrics(
        minimal_metrics(),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert ok
    assert reasons == []


def test_evaluate_metrics_requires_parameter_and_toxicity_series():
    ok, reasons = evaluate_metrics(
        minimal_metrics(parameter_series=[], toxicity_series=[]),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "missing_parameter_series" in reasons
    assert "missing_toxicity_series" in reasons


def test_evaluate_metrics_requires_all_plan_markout_horizons():
    ok, reasons = evaluate_metrics(
        minimal_metrics(markout_samples=[{"horizon_ms": 100, "markout_usdc": 0.02}]),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "missing_markout_horizon:1000ms" in reasons
    assert "missing_markout_horizon:5000ms" in reasons
    assert "missing_markout_horizon:30000ms" in reasons


def test_evaluate_metrics_rejects_inconsistent_fill_ratio_by_depth():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            quote_attempts_by_depth={"bid:5.00bps": 10},
            fills_by_depth={"bid:5.00bps": 2},
            fill_ratio_by_depth={"bid:5.00bps": 0.1},
        ),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "fills_by_depth_total_mismatch:2!=10" in reasons
    assert "fill_ratio_by_depth_mismatch:bid:5.00bps:0.100000000000!=0.200000000000" in reasons


def test_evaluate_metrics_rejects_inconsistent_pnl_by_side():
    ok, reasons = evaluate_metrics(
        minimal_metrics(pnl_by_side={"bid": 0.1, "ask": 0.1}),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "pnl_by_side_total_mismatch:0.200000000000!=0.900000000000" in reasons


def test_evaluate_metrics_rejects_corrupt_parameter_and_toxicity_series():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            parameter_series=[
                {
                    "schema_version": 1,
                    "ts": "not-a-date",
                    "source": "static_replay_params",
                    "symbol": "ETH",
                    "kappa_plus": "bad",
                    "kappa_minus": 2.0,
                    "lambda_plus": 0.1,
                    "lambda_minus": 0.1,
                    "epsilon_plus": 0.0,
                    "epsilon_minus": 0.0,
                }
            ],
            toxicity_series=[
                {
                    "schema_version": 1,
                    "ts": "",
                    "source": "static_replay_params",
                    "symbol": "ETH",
                    "toxicity_plus": float("nan"),
                    "toxicity_minus": 0.0,
                }
            ],
        ),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "parameter_series_missing_or_invalid_ts:0" in reasons
    assert "parameter_series_nonfinite:0:kappa_plus" in reasons
    assert "toxicity_series_missing_or_invalid_ts:0" in reasons
    assert "toxicity_series_nonfinite:0:toxicity_plus" in reasons


def test_evaluate_metrics_reports_insufficient_coverage_and_no_fills():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            data_end="2026-05-25T00:10:00Z",
            maker_fills=0,
            maker_ratio=0.0,
            markout_samples=[],
        ),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert any(reason.startswith("insufficient_coverage_days") for reason in reasons)
    assert "no_maker_fills" in reasons


def test_evaluate_metrics_rejects_taker_and_out_of_bounds_inventory():
    ok, reasons = evaluate_metrics(
        minimal_metrics(taker_fills=1, inventory_histogram={"4": 1}),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "taker_fills_nonzero:1" in reasons
    assert "inventory_out_of_bounds:4" in reasons


def test_evaluate_metrics_rejects_maintenance_margin_breach():
    ok, reasons = evaluate_metrics(
        minimal_metrics(liquidation_breach_events=2),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert "maintenance_margin_breached:2" in reasons


def test_directional_drift_proxy_detects_unexplained_pnl():
    metrics = minimal_metrics(mark_to_market_pnl_usdc=10.0)

    assert round(directional_pnl_proxy_usdc(metrics), 6) == 9.1
    assert round(directional_drift_ratio(metrics), 6) == 0.91


def test_evaluate_metrics_rejects_directional_drift_dominated_pnl():
    ok, reasons = evaluate_metrics(
        minimal_metrics(mark_to_market_pnl_usdc=10.0),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
    )

    assert not ok
    assert any(reason.startswith("directional_drift_dominates_pnl") for reason in reasons)


def test_replay_param_guard_rejects_toxicity():
    ok, reason = replay_param_guard(
        {
            "kappa+": 2.0,
            "kappa-": 2.0,
            "lambda+": 0.1,
            "lambda-": 0.1,
            "epsilon+": 1.0,
            "epsilon-": 0.0,
        },
        max_toxicity=1.5,
    )

    assert not ok
    assert reason == "toxicity_too_high"


def test_refusal_checks_require_bad_params_and_stale_data_to_reject():
    checks = build_refusal_checks(
        params={
            "kappa+": 2.0,
            "kappa-": 2.0,
            "lambda+": 0.1,
            "lambda-": 0.1,
            "epsilon+": 0.0,
            "epsilon-": 0.0,
        },
        baseline_metrics=minimal_metrics(),
        max_toxicity=1.5,
        max_data_age_seconds=30,
    )

    assert {check["name"] for check in checks} == {
        "bad_params_nonpositive_kappa",
        "bad_params_toxicity",
        "stale_collector_data",
    }
    assert all(check["ok"] for check in checks)
    assert all(check["decision"] == "reject" for check in checks)


def test_render_markdown_includes_reasons():
    report = {
        "ok": False,
        "symbol": "ETH",
        "generated_at": "2026-05-25T00:00:00Z",
        "reasons": ["baseline:no_maker_fills"],
        "refusal_checks": [
            {
                "name": "bad_params_nonpositive_kappa",
                "ok": True,
                "expected_decision": "reject",
                "decision": "reject",
                "reason": "invalid_kappa",
            }
        ],
        "variants": [
            {
                "variant": {"name": "baseline"},
                "ok": False,
                "reasons": ["no_maker_fills"],
                "coverage_days": 0.1,
                "net_realized_spread_usdc": 0.0,
                "metrics": {"quote_attempts": 10, "maker_fills": 0, "taker_fills": 0},
            }
        ],
    }

    markdown = render_markdown(report)

    assert "Replay Acceptance Report" in markdown
    assert "Refusal Checks" in markdown
    assert "bad_params_nonpositive_kappa" in markdown
    assert "`baseline:no_maker_fills`" in markdown
