from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_strategy_safety import build_strategy_safety_report  # noqa: E402


SAFE_SOURCE = """
class Market_Making:
    trading_enabled = False
    fail_closed_reason = "initial_safety_lock"
    post_only_verified = False
    can_short = False
    use_exit_signal = True
    position_adjustment_enable = False
    minimal_roi = {"0": 10}
    order_types = {"entry": "limit", "exit": "limit"}
    hjb_alpha = 0.001
    hjb_phi = 0.0001
    hjb_q_max = 3
    inventory_unit_base = 0.01
    max_abs_inventory_units = 3
    max_daily_loss_usdc = 20
    kill_on_taker_fill = True
    kill_on_time_in_force_mismatch = True
    kill_on_unknown_liquidity_fill = True

    def custom_entry_price(self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs):
        pass

    def custom_exit_price(self, pair, trade, current_time, proposed_rate, current_profit, exit_tag, **kwargs):
        pass

    def confirm_trade_entry(self):
        pass

    def confirm_trade_exit(self):
        pass

    def adjust_entry_price(self):
        pass

    def adjust_exit_price(self):
        pass
"""


def report_for(source: str):
    return build_strategy_safety_report(source, strategy_path=Path("user_data/strategies/Market_Making.py"))


def test_strategy_safety_shape_passes():
    report = report_for(SAFE_SOURCE)

    assert report["ok"] is True
    assert report["reasons"] == []


def test_strategy_safety_rejects_live_enabled_defaults_and_roi_force_exit():
    source = SAFE_SOURCE.replace("trading_enabled = False", "trading_enabled = True").replace(
        'minimal_roi = {"0": 10}',
        'minimal_roi = {"0": -1}',
    )

    report = report_for(source)

    assert report["ok"] is False
    assert "trading_enabled_default_not_false" in report["reasons"]
    assert "minimal_roi_force_exit_minus_one" in report["reasons"]


def test_strategy_safety_rejects_missing_trade_callback_argument():
    source = SAFE_SOURCE.replace(
        "def custom_entry_price(self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs):",
        "def custom_entry_price(self, pair, current_time, proposed_rate, entry_tag, side, **kwargs):",
    )

    report = report_for(source)

    assert report["ok"] is False
    assert "custom_entry_price_signature_mismatch" in report["reasons"]


def test_strategy_safety_rejects_disabled_risk_and_kill_switches():
    source = (
        SAFE_SOURCE.replace("hjb_alpha = 0.001", "hjb_alpha = 0.0")
        .replace("hjb_phi = 0.0001", "hjb_phi = 0.0")
        .replace("kill_on_taker_fill = True", "kill_on_taker_fill = False")
    )

    report = report_for(source)

    assert report["ok"] is False
    assert "hjb_alpha_not_positive" in report["reasons"]
    assert "hjb_phi_not_positive" in report["reasons"]
    assert "kill_on_taker_fill_not_true" in report["reasons"]
