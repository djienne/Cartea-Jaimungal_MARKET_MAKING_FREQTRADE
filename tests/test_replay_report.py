from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mm_core  # noqa: E402
from run_replay_report import (  # noqa: E402
    DEFAULT_VARIANTS,
    REQUIRED_MARKOUT_HORIZONS_MS,
    ReplayVariant,
    build_refusal_checks,
    build_report,
    coverage_days,
    directional_drift_ratio,
    directional_pnl_proxy_usdc,
    evaluate_metrics,
    markout_summary,
    replay_required_stream_guard,
    render_markdown,
    variant_config,
    variant_params,
    replay_param_guard,
)
from replay_market_maker import (  # noqa: E402
    ReplayConfig,
    ReplayMetrics,
    hjb_quote_config,
    solve_replay_hjb,
)


def minimal_metrics(**overrides):
    payload = {
        "data_start": "2026-05-25T00:00:00Z",
        "data_end": "2026-05-28T00:00:00Z",
        "input_rows": {"prices": 3000, "trades": 300, "orderbooks": 300},
        "data_span_seconds": 259200.0,
        "price_event_count": 3000,
        "price_events_per_day": 1000.0,
        "max_price_gap_seconds": 60.0,
        "p95_price_gap_seconds": 60.0,
        "quote_attempts": 1000,
        "post_only_rejects": 0,
        "post_only_reject_ratio": 0.0,
        "maker_fills": 10,
        "taker_fills": 0,
        "maker_ratio": 1.0,
        "stale_quote_cancels": 0,
        "stale_quote_cancel_ratio": 0.0,
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


def test_evaluate_metrics_requires_fill_calibration_when_requested():
    ok, reasons = evaluate_metrics(
        minimal_metrics(),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
        require_fill_calibration=True,
    )

    assert not ok
    assert "missing_fill_calibration" in reasons


def test_evaluate_metrics_accepts_usable_fill_calibration_when_requested():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            fill_calibration={
                "provided": True,
                "usable": True,
                "applied": True,
                "fill_probability_by_depth": {"bid:5.00-6.00bps": 0.01},
            }
        ),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
        require_fill_calibration=True,
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


def test_evaluate_metrics_rejects_bad_quote_quality_ratios():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            post_only_rejects=810,
            post_only_reject_ratio=0.81,
            stale_quote_cancels=1000,
            stale_quote_cancel_ratio=1.0,
        ),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
        max_post_only_reject_ratio=0.80,
        max_stale_quote_cancel_ratio=0.99,
    )

    assert not ok
    assert "post_only_reject_ratio_above_threshold:0.810000>max_0.800000" in reasons
    assert "stale_quote_cancel_ratio_above_threshold:1.000000>max_0.990000" in reasons


def test_evaluate_metrics_rejects_sparse_or_gappy_price_coverage():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            input_rows={"prices": 30, "trades": 300, "orderbooks": 300},
            price_event_count=30,
            price_events_per_day=10.0,
            max_price_gap_seconds=600.0,
        ),
        min_days=3.0,
        q_max=3,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
        min_price_events_per_day=1000.0,
        max_price_gap_seconds=300.0,
    )

    assert not ok
    assert "insufficient_price_events_per_day:10.000000<min_1000.000000" in reasons
    assert "max_price_gap_seconds_above_threshold:600.000000>max_300.000000" in reasons


def test_evaluate_metrics_rejects_inconsistent_quote_quality_ratios():
    ok, reasons = evaluate_metrics(
        minimal_metrics(
            post_only_rejects=10,
            post_only_reject_ratio=0.02,
            stale_quote_cancels=10,
            stale_quote_cancel_ratio=0.02,
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
    assert "post_only_reject_ratio_mismatch:0.020000000000!=0.010000000000" in reasons
    assert "stale_quote_cancel_ratio_mismatch:0.020000000000!=0.010000000000" in reasons


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
        "missing_trade_stream",
        "stale_collector_data",
    }
    assert all(check["ok"] for check in checks)
    assert all(check["decision"] == "reject" for check in checks)


def test_replay_required_stream_guard_rejects_missing_trade_stream():
    ok, reason, missing = replay_required_stream_guard(
        minimal_metrics(input_rows={"prices": 10, "trades": 0, "orderbooks": 10})
    )

    assert ok is False
    assert reason == "missing_collector_streams:trades"
    assert missing == ["trades"]


def test_replay_required_stream_guard_rejects_empty_stream_even_when_file_exists():
    ok, reason, missing = replay_required_stream_guard(
        minimal_metrics(
            input_files={"prices": 1, "trades": 1, "orderbooks": 1},
            input_rows={"prices": 10, "trades": 0, "orderbooks": 10},
        )
    )

    assert ok is False
    assert reason == "missing_collector_streams:trades"
    assert missing == ["trades"]


# ---------------------------------------------------------------------------
# Parameter-stress variants on a sub-dollar, high-kappa instrument
# ---------------------------------------------------------------------------

# The live CASHCAT snapshot the sweep is being set up against. What makes it the
# right test fixture is the scale: mid 0.1179 and kappa in the thousands, where
# an absolute epsilon offset sized for ETH is hundreds of bps of mid.
CASHCAT_PARAMS = {
    "kappa+": 6200.0,
    "kappa-": 8500.0,
    "lambda+": 0.23,
    "lambda-": 0.49,
    "epsilon+": 2.97e-5,
    "epsilon-": 1.19e-5,
}
CASHCAT_MID = 0.1179
MAX_TOXICITY = 1.5


def _cashcat_config() -> ReplayConfig:
    return ReplayConfig(
        symbol="CASHCAT",
        data_dir=Path("unused"),
        mid_fallback=CASHCAT_MID,
        inventory_unit_base=2092.0,
        q_max=6,
        q_min=-6,
        price_tick_size=1e-6,
        hjb_phi_kappa_t=10.0,
        hjb_alpha_kappa=0.05,
        hjb_horizon_seconds=150.0,
    )


def _toxicity(params: dict[str, float]) -> float:
    return max(params["kappa+"] * params["epsilon+"], params["kappa-"] * params["epsilon-"])


def _quote_bps(params: dict[str, float]) -> dict[tuple[int, str], tuple[float, str | None]]:
    """Assembled half-spread (bps of mid, clamp flag) at every inventory the
    agent can hold -- the quotes a variant would actually rest."""
    config = _cashcat_config()
    quote_config = hjb_quote_config(config)
    hjb = solve_replay_hjb(config, params)
    quotes: dict[tuple[int, str], tuple[float, str | None]] = {}
    for q in range(-config.q_max, config.q_max + 1):
        for side in ("bid", "ask"):
            spread = mm_core.assemble_half_spread(
                mm_core.select_delta(hjb, float(q), side), CASHCAT_MID, quote_config
            )
            if spread is not None:
                quotes[(q, side)] = (float(spread.bps), spread.clamped)
    return quotes


def _named_variant(name: str) -> ReplayVariant:
    return next(variant for variant in DEFAULT_VARIANTS if variant.name == name)


def test_parameter_stress_variants_stay_inside_the_toxicity_gate_on_cashcat():
    """The regression. With the old absolute epsilon_add these toxicities were
    49.7 and 149.0 against a 1.5 ceiling -- the stressed variants described a
    market the live validator would have refused to quote at all."""
    soft = variant_params(CASHCAT_PARAMS, _named_variant("params_soft"))
    hard = variant_params(CASHCAT_PARAMS, _named_variant("params_hard"))

    assert _toxicity(CASHCAT_PARAMS) < _toxicity(soft) < _toxicity(hard) < MAX_TOXICITY
    # The stress is a multiple of what was measured, so it tracks the instrument.
    assert soft["epsilon+"] == pytest.approx(2.0 * CASHCAT_PARAMS["epsilon+"])
    assert hard["epsilon+"] == pytest.approx(4.0 * CASHCAT_PARAMS["epsilon+"])


def test_parameter_stress_variants_produce_different_uncapped_quotes_on_cashcat():
    """The test that would have caught it: soft and hard must disagree.

    Under the old offsets every quote pinned to the 80 bps cap or the 1.5 bps
    floor, so the two variants returned identical depths, identical fills and
    identical P&L, and the acceptance report read that as two independent passes.
    """
    soft = _quote_bps(variant_params(CASHCAT_PARAMS, _named_variant("params_soft")))
    hard = _quote_bps(variant_params(CASHCAT_PARAMS, _named_variant("params_hard")))

    shared = sorted(set(soft) & set(hard))
    assert shared, "the two stresses must quote at overlapping inventories"

    # Nothing may ride the cap: a capped quote carries no information about the
    # parameters that produced it, which is precisely how the bug hid.
    assert [key for key in soft if soft[key][1] == "cap"] == []
    assert [key for key in hard if hard[key][1] == "cap"] == []

    # And they must differ by an amount an operator could act on, not by noise.
    biggest_gap = max(abs(soft[key][0] - hard[key][0]) for key in shared)
    assert biggest_gap > 1.0, f"params_soft and params_hard quote within {biggest_gap:.3f} bps"


def test_absolute_epsilon_offset_is_still_reachable_for_old_artifacts():
    """The ETH-scale behaviour stays reproducible -- explicitly, never by
    default. This is also a live demonstration of why it cannot be the default:
    at CASHCAT scale the same offset is a toxicity of 50."""
    legacy = ReplayVariant("legacy_params_soft", params_multiplier=0.8, epsilon_add=0.01)

    params = variant_params(CASHCAT_PARAMS, legacy)

    assert params["epsilon+"] == pytest.approx(CASHCAT_PARAMS["epsilon+"] + 0.01)
    assert _toxicity(params) > MAX_TOXICITY
    assert all(variant.epsilon_add == 0.0 for variant in DEFAULT_VARIANTS)


def test_default_replay_variants_include_widened_tick_stress():
    names = [variant.name for variant in DEFAULT_VARIANTS]
    widened = next(variant for variant in DEFAULT_VARIANTS if variant.name == "widened_tick")
    config = ReplayConfig(symbol="ETH", data_dir=Path("unused"), mid_fallback=2000.0)

    stressed = variant_config(config, widened)

    assert "widened_tick" in names
    assert stressed.price_tick_size == 1.0


def test_build_report_runs_widened_tick_variant(monkeypatch):
    def fake_run_replay(config: ReplayConfig, params: dict[str, float]) -> ReplayMetrics:
        metrics = ReplayMetrics()
        metrics.input_files = {"prices": 1, "trades": 1, "orderbooks": 1}
        metrics.input_rows = {"prices": 3000, "trades": 300, "orderbooks": 300}
        metrics.data_start = "2026-05-25T00:00:00Z"
        metrics.data_end = "2026-05-28T00:00:00Z"
        metrics.data_span_seconds = 259200.0
        metrics.price_event_count = 3000
        metrics.price_events_per_day = 1000.0
        metrics.max_price_gap_seconds = 60.0
        metrics.quote_attempts = 1000
        metrics.maker_fills = 10
        metrics.maker_ratio = 1.0
        metrics.realized_spread_usdc = 1.0
        metrics.fees_usdc = 0.1
        metrics.mark_to_market_pnl_usdc = 0.9
        metrics.inventory_histogram = {0: 900, 1: 100}
        metrics.quote_attempts_by_depth = {"bid:5.00bps": 900, "ask:5.00bps": 100}
        metrics.fills_by_depth = {"bid:5.00bps": 9, "ask:5.00bps": 1}
        metrics.pnl_by_side = {"bid": 0.8, "ask": 0.1}
        metrics.markout_samples = [
            {"horizon_ms": horizon_ms, "markout_usdc": 0.02}
            for horizon_ms in REQUIRED_MARKOUT_HORIZONS_MS
        ]
        metrics.parameter_series = minimal_metrics()["parameter_series"]
        metrics.toxicity_series = minimal_metrics()["toxicity_series"]
        metrics.price_tick_size = config.price_tick_size
        return metrics

    monkeypatch.setattr("run_replay_report.run_replay", fake_run_replay)
    report = build_report(
        config=ReplayConfig(symbol="ETH", data_dir=Path("unused"), mid_fallback=2000.0),
        params={
            "kappa+": 2.0,
            "kappa-": 2.0,
            "lambda+": 0.1,
            "lambda-": 0.1,
            "epsilon+": 0.0,
            "epsilon-": 0.0,
        },
        min_days=3.0,
        min_quote_attempts=1000,
        min_maker_ratio=0.99,
        min_net_realized_spread=0.0,
        min_mean_markout_usdc=-0.01,
        max_directional_drift_ratio=0.75,
        max_post_only_reject_ratio=0.8,
        max_stale_quote_cancel_ratio=0.99,
        min_price_events_per_day=1000.0,
        max_price_gap_seconds=300.0,
        require_fill_calibration=False,
    )

    widened = next(item for item in report["variants"] if item["variant"]["name"] == "widened_tick")
    assert widened["metrics"]["price_tick_size"] == 1.0
    assert any(check["name"] == "missing_trade_stream" for check in report["refusal_checks"])


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
