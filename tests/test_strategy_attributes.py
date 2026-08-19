from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_strategy_attributes import build_strategy_attribute_report  # noqa: E402


STRATEGY = ROOT / "user_data" / "strategies" / "Market_Making.py"


def test_live_strategy_defines_every_attribute_it_reads():
    report = build_strategy_attribute_report(STRATEGY.read_text(encoding="utf-8"), strategy_path=STRATEGY)
    assert report["ok"], report["undefined_attributes"]
    assert report["attributes_read"] > 100


def test_phantom_attribute_is_flagged():
    source = """
class Market_Making:
    trading_enabled = False

    def adjust_trade_position(self):
        if not self.trading_enabled or self._kill_switch_active:
            return None
        return 1
"""
    report = build_strategy_attribute_report(source)
    assert not report["ok"]
    assert [item["attribute"] for item in report["undefined_attributes"]] == ["_kill_switch_active"]


def test_inherited_and_assigned_attributes_are_allowed():
    source = """
class Market_Making:
    declared = 1

    def start(self):
        self._runtime = 2
        setattr(self, "_dynamic", 3)

    def use(self):
        return self.declared, self._runtime, self._dynamic, self.config, self.dp
"""
    report = build_strategy_attribute_report(source)
    assert report["ok"], report["undefined_attributes"]


def test_getattr_with_default_is_not_flagged():
    source = """
class Market_Making:
    def use(self):
        return getattr(self, "_optional_state", 0)
"""
    report = build_strategy_attribute_report(source)
    assert report["ok"], report["undefined_attributes"]


def test_regression_the_shipped_bug_would_have_been_caught():
    """The pre-fix HEAD of the strategy must fail this check.

    Guards against the verifier being weakened until it no longer catches the
    AttributeError that killed adjust_trade_position from d20b27d to 2026-08-19.
    """
    result = subprocess.run(
        ["git", "show", "d20b27d:user_data/strategies/Market_Making.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        import pytest

        pytest.skip("git history unavailable")
    report = build_strategy_attribute_report(result.stdout)
    assert not report["ok"]
    assert "_kill_switch_active" in {item["attribute"] for item in report["undefined_attributes"]}
