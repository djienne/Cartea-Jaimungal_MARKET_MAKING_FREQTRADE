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
    config["market_making"]["trading_enabled"] = True
    config["market_making"]["post_only_verified"] = True
    config["market_making"]["deployment_stage"] = "canary"

    report = report_for(config)

    assert report["ok"] is False
    assert "checked_in_trading_enabled_true" in report["reasons"]
    assert "checked_in_post_only_verified_true" in report["reasons"]
    assert "checked_in_deployment_stage_not_research" in report["reasons"]


def test_config_safety_rejects_oversized_testing_exposure():
    config = safe_config()
    config["dry_run"] = False
    config["stake_amount"] = 100
    config["tradable_balance_ratio"] = 0.99
    config["custom_price_max_distance_ratio"] = 0.10
    config["max_open_trades"] = 5

    report = report_for(config)

    assert report["ok"] is False
    assert "dry_run_not_true" in report["reasons"]
    assert "stake_amount_above_limit:100>max_25" in report["reasons"]
    assert "tradable_balance_ratio_above_limit:0.99>max_0.1" in report["reasons"]
    assert "custom_price_max_distance_ratio_above_limit:0.1>max_0.05" in report["reasons"]
    assert "max_open_trades_above_one" in report["reasons"]
