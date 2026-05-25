from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_replay_report import (  # noqa: E402
    coverage_days,
    directional_drift_ratio,
    directional_pnl_proxy_usdc,
    evaluate_metrics,
    markout_summary,
    render_markdown,
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
        "inventory_histogram": {"0": 900, "1": 100},
        "markout_samples": [{"horizon_ms": 100, "markout_usdc": 0.02}],
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


def test_render_markdown_includes_reasons():
    report = {
        "ok": False,
        "symbol": "ETH",
        "generated_at": "2026-05-25T00:00:00Z",
        "reasons": ["baseline:no_maker_fills"],
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
    assert "`baseline:no_maker_fills`" in markdown
