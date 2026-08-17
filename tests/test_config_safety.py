from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_config_safety import build_config_safety_report  # noqa: E402


def safe_config() -> dict:
    return {
        "dry_run": True,
        "force_entry_enable": False,
        "max_open_trades": 1,
        "stake_amount": 25,
        "tradable_balance_ratio": 0.10,
        "fee": 0.00015,
        "custom_price_max_distance_ratio": 0.05,
        "order_types": {"entry": "limit", "exit": "limit"},
        "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
        "api_server": {
            "force_entry_enable": False,
            "forcebuy_enable": False,
            "Force_entry": False,
        },
        "market_making": {
            "trading_enabled": False,
            "post_only_verified": False,
            "deployment_stage": "research",
            "run_estimators_in_strategy": False,
        },
    }


def report_for(config: dict):
    return build_config_safety_report(config, config_path=Path("user_data/config.json"))


def test_checked_in_safety_config_shape_passes():
    report = report_for(safe_config())

    assert report["ok"] is True
    assert report["reasons"] == []


def test_config_safety_rejects_force_entry_unlimited_stake_and_bad_fee():
    config = safe_config()
    config["force_entry_enable"] = True
    config["stake_amount"] = "unlimited"
    config["fee"] = 0.001
    config["api_server"]["forcebuy_enable"] = True

    report = report_for(config)

    assert report["ok"] is False
    assert "force_entry_enable_not_false" in report["reasons"]
    assert "api_server_forcebuy_enable_not_false" in report["reasons"]
    assert "stake_amount_unlimited" in report["reasons"]
    assert "fee_mismatch:0.001!=expected_0.00015" in report["reasons"]


def test_config_safety_rejects_checked_in_live_enablement():
    config = safe_config()
    config["dry_run"] = False
    config["market_making"]["trading_enabled"] = True
    config["market_making"]["post_only_verified"] = True
    config["market_making"]["deployment_stage"] = "canary"
    config["market_making"]["run_estimators_in_strategy"] = True

    report = report_for(config)

    assert report["ok"] is False
    assert "checked_in_trading_enabled_without_dry_run" in report["reasons"]
    assert "checked_in_post_only_verified_true" in report["reasons"]
    assert "checked_in_deployment_stage_not_research" in report["reasons"]
    assert "checked_in_internal_param_estimator_true" in report["reasons"]


def test_config_safety_allows_trading_enabled_under_dry_run():
    # trading_enabled=true with dry_run=true is the dry-run quoting mode; only
    # the live combination (dry_run not true) counts as checked-in enablement.
    config = safe_config()
    config["market_making"]["trading_enabled"] = True

    report = report_for(config)

    assert report["ok"] is True
    assert report["reasons"] == []


def test_config_safety_fee_follows_maker_fee_rate_override():
    # With market_making.maker_fee_rate present, the accounting fee must match
    # it (mirrors the strategy's runtime config_fee_mismatch gate).
    config = safe_config()
    config["fee"] = 0.0001
    config["market_making"]["maker_fee_rate"] = 0.0001
    assert report_for(config)["ok"] is True

    config["fee"] = 0.00015
    report = report_for(config)
    assert report["ok"] is False
    assert "fee_mismatch:0.00015!=expected_0.0001" in report["reasons"]

    # Absurd overrides are rejected outright rather than trusted.
    config["fee"] = 0.0
    config["market_making"]["maker_fee_rate"] = 0.0
    report = report_for(config)
    assert report["ok"] is False
    assert "maker_fee_rate_invalid" in report["reasons"]


def test_config_safety_rejects_oversized_testing_exposure():
    config = safe_config()
    config["dry_run"] = False
    # The stake ceiling rose to 600 because stake_amount must now EXCEED one
    # inventory unit (0.15 ETH ~ 250-450 USDC): if the proposed-stake term binds
    # first, a "one unit" fill is quietly smaller than one unit of q and the
    # HJB's unit-jump mapping drifts. Use a value above the new ceiling.
    config["stake_amount"] = 1000
    config["tradable_balance_ratio"] = 0.99
    config["custom_price_max_distance_ratio"] = 0.10
    config["max_open_trades"] = 5

    report = report_for(config)

    assert report["ok"] is False
    assert "dry_run_not_true" in report["reasons"]
    assert "stake_amount_above_limit:1000>max_600" in report["reasons"]
    assert "tradable_balance_ratio_above_limit:0.99>max_0.1" in report["reasons"]
    assert "custom_price_max_distance_ratio_above_limit:0.1>max_0.05" in report["reasons"]
    assert "max_open_trades_above_one" in report["reasons"]
