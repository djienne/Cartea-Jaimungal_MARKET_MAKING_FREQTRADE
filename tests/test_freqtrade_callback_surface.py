from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_freqtrade_callback_surface import build_callback_surface_report  # noqa: E402


class GoodStrategy:
    def custom_entry_price(self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs):
        pass

    def custom_exit_price(self, pair, trade, current_time, proposed_rate, current_profit, exit_tag, **kwargs):
        pass

    def custom_stake_amount(
        self,
        pair,
        current_time,
        current_rate,
        proposed_stake,
        min_stake,
        max_stake,
        leverage,
        entry_tag,
        side,
        **kwargs,
    ):
        pass

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        pass

    def confirm_trade_entry(
        self,
        pair,
        order_type,
        amount,
        rate,
        time_in_force,
        current_time,
        entry_tag,
        side,
        **kwargs,
    ):
        pass

    def confirm_trade_exit(
        self,
        pair,
        trade,
        order_type,
        amount,
        rate,
        time_in_force,
        exit_reason,
        current_time,
        **kwargs,
    ):
        pass

    def adjust_entry_price(
        self,
        trade,
        order,
        pair,
        current_time,
        proposed_rate,
        current_order_rate,
        entry_tag,
        side,
        **kwargs,
    ):
        pass

    def adjust_exit_price(
        self,
        trade,
        order,
        pair,
        current_time,
        proposed_rate,
        current_order_rate,
        exit_tag,
        side,
        **kwargs,
    ):
        pass

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        pass


def report_for(strategy_class: type) -> dict:
    return build_callback_surface_report(strategy_class, strategy_path=Path("strategy.py"))


def test_callback_surface_report_accepts_expected_freqtrade_callbacks():
    report = report_for(GoodStrategy)

    assert report["ok"] is True
    assert report["reasons"] == []
    assert report["callbacks"]["custom_stake_amount"]["parameters"][:9] == [
        "self",
        "pair",
        "current_time",
        "current_rate",
        "proposed_stake",
        "min_stake",
        "max_stake",
        "leverage",
        "entry_tag",
    ]
    assert report["callbacks"]["order_filled"]["has_var_keyword"] is True


def test_callback_surface_report_rejects_missing_trade_argument():
    class BadStrategy(GoodStrategy):
        def custom_entry_price(self, pair, current_time, proposed_rate, entry_tag, side, **kwargs):
            pass

    report = report_for(BadStrategy)

    assert report["ok"] is False
    assert "custom_entry_price_signature_mismatch" in report["reasons"]


def test_callback_surface_report_rejects_missing_kwargs():
    class BadStrategy(GoodStrategy):
        def order_filled(self, pair, trade, order, current_time):
            pass

    report = report_for(BadStrategy)

    assert report["ok"] is False
    assert "order_filled_missing_kwargs" in report["reasons"]
