from __future__ import annotations

import sys
import types
import inspect
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "user_data" / "strategies"
SCRIPTS = ROOT / "scripts"
for path in (STRATEGIES, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class DummyTrade:
    _open_trades = []
    _open_order_trades = []

    def __init__(
        self,
        amount=0.0,
        is_short=False,
        trade_id=1,
        open_rate=100.0,
        has_open_orders=None,
        orders=None,
        pair="ETH/USDC:USDC",
    ):
        self.amount = amount
        self.is_short = is_short
        self.id = trade_id
        self.open_rate = open_rate
        self.pair = pair
        if has_open_orders is not None:
            self.has_open_orders = has_open_orders
        if orders is not None:
            self.orders = orders

    @classmethod
    def get_trades(cls, **kwargs):
        return list(cls._open_trades)

    @classmethod
    def get_trades_proxy(cls, **kwargs):
        # Mirror freqtrade's Trade.get_trades_proxy (the strategy-safe accessor
        # that works in live/dry-run/backtest, unlike get_trades).
        return list(cls._open_trades)

    @classmethod
    def get_open_order_trades(cls):
        return list(cls._open_order_trades)


def install_freqtrade_stubs():
    freqtrade = types.ModuleType("freqtrade")
    strategy = types.ModuleType("freqtrade.strategy")
    exchange = types.ModuleType("freqtrade.exchange")
    persistence = types.ModuleType("freqtrade.persistence")
    vendor = types.ModuleType("freqtrade.vendor")
    qtpylib = types.ModuleType("freqtrade.vendor.qtpylib")
    indicators = types.ModuleType("freqtrade.vendor.qtpylib.indicators")

    class IStrategy:
        config = {"dry_run": True}

    def identity_decorator(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper

    strategy.IStrategy = IStrategy
    strategy.BooleanParameter = object
    strategy.CategoricalParameter = object
    strategy.DecimalParameter = object
    strategy.IntParameter = object
    strategy.stoploss_from_absolute = lambda *args, **kwargs: None
    strategy.informative = identity_decorator
    exchange.timeframe_to_prev_date = lambda *args, **kwargs: None
    persistence.Trade = DummyTrade
    persistence.Order = object

    sys.modules["freqtrade"] = freqtrade
    sys.modules["freqtrade.strategy"] = strategy
    sys.modules["freqtrade.exchange"] = exchange
    sys.modules["freqtrade.persistence"] = persistence
    sys.modules["freqtrade.vendor"] = vendor
    sys.modules["freqtrade.vendor.qtpylib"] = qtpylib
    sys.modules["freqtrade.vendor.qtpylib.indicators"] = indicators
    sys.modules["talib"] = types.ModuleType("talib")
    sys.modules["talib.abstract"] = types.ModuleType("talib.abstract")
    sys.modules["pandas_ta"] = types.ModuleType("pandas_ta")


install_freqtrade_stubs()
from Market_Making import Market_Making  # noqa: E402


class DummyDP:
    def __init__(self, best_bid=99.0, best_ask=101.0, timestamp=None):
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.timestamp = timestamp

    def current_whitelist(self):
        return ["ETH/USDC:USDC"]

    def orderbook(self, pair, maximum=1):
        payload = {"bids": [[self.best_bid, 1.0]], "asks": [[self.best_ask, 1.0]]}
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload


class EmptyWhitelistDP(DummyDP):
    def current_whitelist(self):
        return []


# Methods the strategy probes on its exchange handle that freqtrade's Exchange
# does NOT define. Verified by introspecting freqtrade 2025.10 (Python 3.13.8)
# inside the ft-*:2025.10 image:
#
#     cancel_all_orders   False        cancel_order     True
#     fetch_open_orders   False        fetch_positions  True
#     get_open_orders     False        fetch_trading_fee False
#
# A fixture that defines any of these makes the suite assert against an API that
# does not exist. That is exactly how the kill switch shipped believing it
# cancelled resting orders: _cancel_open_orders_for_kill_switch probes
# cancel_all_orders, then fetch_open_orders, then get_open_orders, and against
# real freqtrade every probe misses and it returns no_open_order_source having
# cancelled nothing.
ABSENT_FROM_REAL_FREQTRADE_EXCHANGE = (
    "cancel_all_orders",
    "fetch_open_orders",
    "get_open_orders",
    "fetch_trading_fee",
)

# Trade.get_open_order_trades does not exist on 2025.10 either; the strategy's
# getattr() probe for it is dead but harmless.
ABSENT_FROM_REAL_FREQTRADE_TRADE = ("get_open_order_trades",)


class DummyExchange:
    def __init__(self):
        self.markets = {
            "ETH/USDC:USDC": {
                "precision": {"amount": 3, "price": 2},
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 1.0}},
                "maker": 0.00015,
                "taker": 0.00045,
            }
        }
        self.positions = []
        self.cancel_calls = []

    def amount_to_precision(self, pair, amount):
        return f"{float(amount):.3f}"

    def price_to_precision(self, pair, price):
        return f"{float(price):.2f}"

    def fetch_positions(self, pairs=None):
        return list(self.positions)

    def cancel_order(self, order_id, pair=None):
        # Real freqtrade DOES provide this one, so the fixture must too -- without
        # it the cancel path bails a step earlier than production and reports
        # "unavailable" instead of the truthful "no_open_order_source".
        self.cancel_calls.append((order_id, pair))

    # NOTE: no cancel_all_orders / fetch_open_orders / get_open_orders /
    # open_orders here, on purpose. See ABSENT_FROM_REAL_FREQTRADE_EXCHANGE --
    # this fixture used to invent cancel_all_orders, which made the kill switch
    # look like it cancelled resting orders when against real freqtrade it
    # cancels nothing.


def write_param_snapshot_files(directory: Path) -> Path:
    """Write a minimal kappa/epsilon/lambda snapshot trio into ``directory``.

    These files are produced by the estimator every ~30s and are no longer
    tracked (they turned every commit into calibration churn), so a test that
    needs them must create them. Relying on the repo shipping a generated file
    made the suite pass or fail depending on whether an estimator had ever run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kappa.json").write_text(
        json.dumps({"ETH": {"schema_version": 3, "status": "ok", "kappa+": 2.0, "kappa-": 2.0}}),
        encoding="utf-8",
    )
    (directory / "epsilon.json").write_text(
        json.dumps({"ETH": {"schema_version": 3, "status": "ok", "epsilon+": 0.0, "epsilon-": 0.0}}),
        encoding="utf-8",
    )
    (directory / "lambda.json").write_text(
        json.dumps(
            {
                "ETH": {
                    "schema_version": 3,
                    "status": "ok",
                    "lambda+": 0.2,
                    "lambda-": 0.2,
                    "lambda_source": "mo_survival_fit",
                }
            }
        ),
        encoding="utf-8",
    )
    return directory


def make_bot() -> Market_Making:
    DummyTrade._open_trades = []
    DummyTrade._open_order_trades = []
    bot = Market_Making()
    bot.dp = DummyDP()
    bot.exchange = DummyExchange()
    bot.config = {"dry_run": True, "fee": 0.00015}
    bot.debug_json_log = False
    bot.trading_enabled = False
    bot.post_only_verified = False
    bot._quote_id_sequence = 0
    bot._accepted_quote_decisions = []
    bot._accepted_order_attempt_links = []
    bot.param_update_status_path = ""
    # These tests exercise guard MECHANICS, not the shipped sizing. Pin the
    # values they were written against so a deliberate change to the production
    # defaults cannot masquerade as a behavioural regression -- the defaults
    # themselves are asserted directly in test_production_sizing_defaults.
    bot.auto_size_inventory_unit = False
    bot.inventory_unit_base = 0.01
    bot.hjb_q_max = 3
    bot.max_abs_inventory_units = 3
    bot.max_notional_exposure_usdc = 150.0
    bot.max_margin_used_usdc = 150.0
    bot.hjb_cache = {
        "q_grid": np.array([-1, 0, 1]),
        "delta_plus": np.array([np.inf, 0.5, 0.4]),
        "delta_minus": np.array([0.4, 0.5, np.inf]),
    }
    bot._hjb_last_refresh_dt = datetime.now(timezone.utc)
    bot.kappas = {
        "ETH": {
            "schema_version": 3,
            "status": "ok",
            "kappa+": 2.0,
            "kappa-": 2.0,
            "kappa+_raw": 2.0,
            "kappa-_raw": 2.0,
            "lambda+_raw": 0.1,
            "lambda-_raw": 0.1,
            "n_points_plus": 8,
            "n_points_minus": 8,
            "r2_plus": 0.5,
            "r2_minus": 0.5,
            "depth_p95_plus": 0.9,
            "depth_p95_minus": 0.9,
            "depth_max_fitted_plus": 1.2,
            "depth_max_fitted_minus": 1.2,
            "sigma2_per_sec": 0.02,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    }
    bot.epsilons = {
        "ETH": {
            "schema_version": 3,
            "status": "ok",
            "epsilon+": 0.0,
            "epsilon-": 0.0,
            "epsilon+_raw": 0.0,
            "epsilon-_raw": 0.0,
            "n_buy_events": 60,
            "n_sell_events": 60,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    }
    bot.lambdas = {
        "ETH": {
            "schema_version": 3,
            "status": "ok",
            "lambda+": 0.1,
            "lambda-": 0.1,
            "lambda+_raw": 0.1,
            "lambda-_raw": 0.1,
            "lambda_source": "mo_survival_fit",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    }
    bot._market_data_fresh = lambda symbol, max_age_seconds=None: (True, "ok")
    bot._collector_age_seconds = lambda symbol, now=None: 0.0
    return bot


def write_gate_report(
    tmp_path: Path,
    name: str,
    *,
    ok: bool = True,
    generated_at: str | None = "now",
) -> str:
    path = tmp_path / f"{name}.json"
    payload = {"ok": ok, "reasons": [] if ok else [f"{name}_failed"]}
    if generated_at == "now":
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if generated_at is not None:
        payload["generated_at"] = generated_at
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def live_gate_config(tmp_path: Path, *, stage: str = "canary", include_live_canary: bool = False) -> dict:
    config = {
        "deployment_stage": stage,
        "manual_monitoring_ack": True,
        "post_only_evidence_report_path": write_gate_report(tmp_path, "post_only"),
        "fee_evidence_report_path": write_gate_report(tmp_path, "fee"),
        "replay_acceptance_report_path": write_gate_report(tmp_path, "replay"),
    }
    if include_live_canary:
        config["live_canary_report_path"] = write_gate_report(tmp_path, "live_canary")
    return config


def test_trading_disabled_clears_entry_signal():
    bot = make_bot()
    df = pd.DataFrame({"close": [100.0, 101.0]})

    out = bot.populate_entry_trend(df, {"pair": "ETH/USDC:USDC"})

    assert out["enter_long"].sum() == 0


def _entry_quote_decision(bot, pair="ETH/USDC:USDC", proposed_rate=99.5):
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.custom_entry_price(pair, None, datetime.now(timezone.utc), proposed_rate=proposed_rate, entry_tag="mm_bid", side="long")
    decisions = [p for e, p in events if e == "quote_decision" and p.get("delta_total") is not None]
    assert decisions, "expected a quote_decision with delta_total"
    return decisions[-1]


def test_spread_multiplier_scales_model_term_only():
    # delta_total = delta_model * multiplier + fee_cushion (affine, NOT linear):
    # the fee cushion must not be inflated by the defensive multiplier.
    def delta_total_for(mult):
        bot = make_bot()
        bot.trading_enabled = True
        bot.spread_multiplier = mult
        bot.max_half_spread_bps = 500.0  # keep the cap out of the linearity check
        return _entry_quote_decision(bot)

    base = delta_total_for(1.0)
    doubled = delta_total_for(2.0)
    delta_model = 0.5  # bid delta at q=0 from the stub HJB grid
    fee_cushion = 0.00015 * 100.0  # one maker fee per side at mid=100
    assert abs(float(base["delta_total"]) - (delta_model + fee_cushion)) < 1e-9
    assert abs(float(base["fee_cushion"]) - fee_cushion) < 1e-12
    assert abs(float(doubled["delta_total"]) - float(base["delta_total"]) - delta_model) < 1e-9


def test_half_spread_floor_clamp_applies_and_is_logged():
    bot = make_bot()
    bot.trading_enabled = True
    # Explicit floor: the shipped default is anchored to the maker fee, so a
    # small positive model depth lands at or above it and the clamp never fires.
    # This test asserts the clamp MECHANISM, not the shipped constant.
    bot.min_half_spread_bps = 3.0
    # Tiny model delta: pre-clamp ~= 0.01*1 + 0.015 = 0.025 -> 2.5 bps < 3 bps floor.
    bot.hjb_cache = {
        "q_grid": np.array([-1, 0, 1]),
        "delta_plus": np.array([np.inf, 0.02, 0.01]),
        "delta_minus": np.array([0.01, 0.01, np.inf]),
    }
    decision = _entry_quote_decision(bot)
    assert decision["clamped"] == "floor"
    assert abs(float(decision["bps"]) - 3.0) < 1e-9
    assert abs(float(decision["delta_total"]) - 0.03) < 1e-12  # 3 bps of mid 100
    assert float(decision["delta_pre_clamp"]) < 0.03


def test_half_spread_cap_clamp_applies_and_is_logged():
    bot = make_bot()
    bot.trading_enabled = True
    # Huge model delta: pre-clamp ~= 2.0 + 0.015 -> ~201 bps > 80 bps cap.
    bot.hjb_cache = {
        "q_grid": np.array([-1, 0, 1]),
        "delta_plus": np.array([np.inf, 2.0, 2.0]),
        "delta_minus": np.array([2.0, 2.0, np.inf]),
    }
    decision = _entry_quote_decision(bot, proposed_rate=99.0)
    assert decision["clamped"] == "cap"
    assert abs(float(decision["bps"]) - 80.0) < 1e-9
    assert abs(float(decision["delta_total"]) - 0.8) < 1e-12


def test_unclamped_quote_logs_clamped_none_and_calibration_flag():
    bot = make_bot()
    bot.trading_enabled = True
    decision = _entry_quote_decision(bot)
    assert decision["clamped"] is None
    # delta_total 0.515 <= depth_p95_minus 0.9 from the stub kappa snapshot.
    assert decision["quote_outside_calibrated_range"] is False
    assert decision["depth_p95"] == 0.9

    bot2 = make_bot()
    bot2.trading_enabled = True
    bot2.kappas["ETH"]["depth_p95_minus"] = 0.2  # fit only covers 0.2 USDC of depth
    decision2 = _entry_quote_decision(bot2)
    assert decision2["quote_outside_calibrated_range"] is True

    bot3 = make_bot()
    bot3.trading_enabled = True
    del bot3.kappas["ETH"]["depth_p95_minus"]
    bot3.kappas["ETH"]["depth_p95_plus"] = None
    decision3 = _entry_quote_decision(bot3)
    assert decision3["quote_outside_calibrated_range"] is None
    assert decision3["depth_p95"] is None


def test_half_spread_bounds_config_override_and_pair_validation():
    bot = make_bot()
    bot.config = {
        "dry_run": True,
        "fee": 0.00015,
        "market_making": {"min_half_spread_bps": 5.0, "max_half_spread_bps": 60.0},
    }
    bot._apply_runtime_safety_config()
    assert bot.min_half_spread_bps == 5.0
    assert bot.max_half_spread_bps == 60.0

    # min >= max is rejected as a pair; prior values are kept.
    bot.config = {
        "dry_run": True,
        "fee": 0.00015,
        "market_making": {"min_half_spread_bps": 90.0, "max_half_spread_bps": 60.0},
    }
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._apply_runtime_safety_config()
    assert bot.min_half_spread_bps == 5.0
    assert bot.max_half_spread_bps == 60.0
    assert any(p.get("key") == "half_spread_bps_bounds" for e, p in events if e == "runtime_config_rejected")

    # Non-positive values are rejected individually.
    bot.config = {"dry_run": True, "fee": 0.00015, "market_making": {"min_half_spread_bps": -1.0}}
    bot._apply_runtime_safety_config()
    assert bot.min_half_spread_bps == 5.0


def test_diagnostic_floors_config_overridable():
    bot = make_bot()
    bot.config = {
        "dry_run": True,
        "fee": 0.00015,
        "market_making": {"min_kappa_fit_points": 10, "min_kappa_r2": 0.5, "min_epsilon_events": 100},
    }
    bot._apply_runtime_safety_config()
    assert bot.min_kappa_fit_points == 10
    assert bot.min_kappa_r2 == 0.5
    assert bot.min_epsilon_events == 100

    # The stub snapshot (8 points, r2 0.5, 60 events) now fails the raised floors.
    ok, reason = bot._params_are_valid("ETH/USDC:USDC")
    assert not ok
    assert reason == "insufficient_kappa_diagnostics"


def test_hjb_refresh_uses_sigma2_volatility_channel():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._refresh_hjb("ETH/USDC:USDC")
    refreshes = [p for e, p in events if e == "hjb_refresh"]
    assert refreshes, "expected an hjb_refresh event"
    inputs = refreshes[-1]["inputs"]
    expected_phi = 0.0001 + bot.gamma_inventory_risk * 0.02 * bot.inventory_unit_base
    assert inputs["phi_source"] == "sigma2_channel"
    assert abs(inputs["phi_effective"] - expected_phi) < 1e-15
    assert inputs["sigma2_per_sec"] == 0.02


def test_hjb_refresh_falls_back_to_base_phi_without_sigma2():
    bot = make_bot()
    bot.kappas["ETH"]["sigma2_per_sec"] = None
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._refresh_hjb("ETH/USDC:USDC")
    refreshes = [p for e, p in events if e == "hjb_refresh"]
    assert refreshes, "expected an hjb_refresh event (missing sigma2 must not block)"
    inputs = refreshes[-1]["inputs"]
    assert inputs["phi_source"] == "phi_base_fallback"
    assert inputs["phi_effective"] == 0.0001


def test_spread_multiplier_config_override_rejects_invalid():
    bot = make_bot()
    bot.config = {"dry_run": True, "fee": 0.00015, "market_making": {"spread_multiplier": 3.0}}
    bot._apply_runtime_safety_config()
    assert bot.spread_multiplier == 3.0

    # Non-positive / non-finite values are rejected and the prior value kept.
    bot.config = {"dry_run": True, "fee": 0.00015, "market_making": {"spread_multiplier": 0}}
    bot._apply_runtime_safety_config()
    assert bot.spread_multiplier == 3.0


def test_custom_entry_price_signature_includes_trade_argument():
    params = list(inspect.signature(Market_Making.custom_entry_price).parameters)

    assert params[:7] == ["self", "pair", "trade", "current_time", "proposed_rate", "entry_tag", "side"]


def test_custom_stake_amount_signature_matches_freqtrade_stable():
    params = list(inspect.signature(Market_Making.custom_stake_amount).parameters)

    assert params[:9] == [
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


def test_strategy_default_emergency_exit_is_market():
    assert Market_Making.order_types["emergency_exit"] == "market"


def test_strategy_time_in_force_uses_runtime_supported_research_mode():
    assert Market_Making.order_time_in_force == {"entry": "GTC", "exit": "GTC"}
    assert Market_Making.post_only_verified is False


def test_fee_snapshot_reports_config_and_exchange_agreement():
    bot = make_bot()

    snapshot = bot._fee_snapshot("ETH/USDC:USDC")

    assert snapshot["strategy_maker_fee_rate"] == bot.fees_maker_HL
    assert snapshot["config_fee_rate"] == 0.00015
    assert snapshot["config_fee_matches_strategy"] is True
    assert snapshot["exchange_fee_source"] == "exchange_markets"
    assert snapshot["exchange_maker_fee_rate"] == 0.00015
    assert snapshot["exchange_taker_fee_rate"] == 0.00045
    assert snapshot["exchange_maker_fee_matches_strategy"] is True
    assert snapshot["fee_agreement_ok"] is True


def test_fee_snapshot_reports_config_mismatch():
    bot = make_bot()
    bot.config["fee"] = 0.001

    snapshot = bot._fee_snapshot("ETH/USDC:USDC")

    assert snapshot["config_fee_matches_strategy"] is False
    assert snapshot["exchange_maker_fee_matches_strategy"] is True
    assert snapshot["fee_agreement_ok"] is False


def test_dry_run_config_can_enable_research_mode_without_post_only():
    bot = make_bot()
    bot.config = {
        "dry_run": True,
        "fee": 0.00015,
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": False,
            "max_param_age_seconds": 300,
            "max_collector_age_seconds": 180,
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is True
    assert bot.fail_closed_reason == "none"
    assert bot.max_param_age_seconds == 300
    assert bot.max_collector_age_seconds == 180


def test_runtime_enable_rejects_fee_mismatch():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.config = {
        "dry_run": True,
        "fee": 0.001,
        "market_making": {"trading_enabled": True, "post_only_verified": False},
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "config_fee_mismatch"
    assert events == [
        (
            "trading_enable_rejected",
            {"reason": "config_fee_mismatch", "dry_run": True},
        )
    ]


def test_live_config_cannot_enable_without_post_only_verification():
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "market_making": {"trading_enabled": True, "post_only_verified": False},
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "post_only_not_verified"


def test_live_quote_state_rechecks_deployment_gate_reports():
    bot = make_bot()
    now = datetime.now(timezone.utc)
    bot.trading_enabled = True
    bot.post_only_verified = True
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {"trading_enabled": True, "post_only_verified": True},
    }

    ok, reason = bot._quote_state_valid("ETH/USDC:USDC", "bid", 99.5, now)

    assert not ok
    assert reason == "deployment_stage_not_set"


def test_live_config_cannot_enable_with_verified_post_only_but_gtc_tif():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
        "market_making": {"trading_enabled": True, "post_only_verified": True},
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "time_in_force_not_post_only"
    assert events == [
        (
            "trading_enable_rejected",
            {
                "reason": "time_in_force_not_post_only",
                "dry_run": False,
                "entry_time_in_force": "GTC",
                "exit_time_in_force": "GTC",
                "entry_time_in_force_canonical": "gtc",
                "exit_time_in_force_canonical": "gtc",
            },
        )
    ]


def test_live_config_cannot_enable_without_deployment_stage():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {"trading_enabled": True, "post_only_verified": True},
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "deployment_stage_not_set"
    assert events[0][0] == "trading_enable_rejected"
    assert events[0][1]["reason"] == "deployment_stage_not_set"
    assert events[0][1]["deployment_stage"] == "research"


def test_live_canary_config_requires_manual_monitoring_ack(tmp_path):
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path),
            "manual_monitoring_ack": False,
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "manual_monitoring_not_acknowledged"


def test_live_canary_config_requires_gate_reports(tmp_path):
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path),
            "fee_evidence_report_path": write_gate_report(tmp_path, "fee", ok=False),
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "fee_gate_not_passed"


def test_live_canary_config_rejects_missing_gate_report_timestamp(tmp_path):
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path),
            "replay_acceptance_report_path": write_gate_report(tmp_path, "replay", generated_at=None),
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "replay_gate_not_passed"


def test_live_canary_config_rejects_stale_gate_report(tmp_path):
    bot = make_bot()
    now = datetime.now(timezone.utc)
    bot._now_utc = lambda: now
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path),
            "replay_acceptance_report_path": write_gate_report(
                tmp_path,
                "replay",
                generated_at=(now - timedelta(seconds=86_460)).isoformat().replace("+00:00", "Z"),
            ),
            "max_deployment_report_age_seconds": 86_400,
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "replay_gate_not_passed"
    replay_status = events[0][1]["gate_reports"]["replay"]
    assert replay_status["reason"] == "stale_report"
    assert replay_status["age_seconds"] == 86_460.0
    assert replay_status["max_age_seconds"] == 86_400.0


def test_live_canary_config_rejects_future_gate_report(tmp_path):
    bot = make_bot()
    now = datetime.now(timezone.utc)
    bot._now_utc = lambda: now
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path),
            "fee_evidence_report_path": write_gate_report(
                tmp_path,
                "fee",
                generated_at=(now + timedelta(seconds=60)).isoformat().replace("+00:00", "Z"),
            ),
            "max_deployment_report_age_seconds": 86_400,
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "fee_gate_not_passed"
    fee_status = events[0][1]["gate_reports"]["fee"]
    assert fee_status["reason"] == "future_generated_at"
    assert fee_status["age_seconds"] == -60.0


def test_live_config_can_enable_after_post_only_tif_verification_and_gate_reports(tmp_path):
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path),
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is True
    assert bot.fail_closed_reason == "none"
    assert bot._expected_time_in_force("bid") == "Alo"
    assert bot._configured_post_only_tif_valid() == (
        True,
        "ok",
        {
            "entry_time_in_force": "Alo",
            "exit_time_in_force": "Alo",
            "entry_time_in_force_canonical": "post_only",
            "exit_time_in_force_canonical": "post_only",
        },
    )


def test_production_config_requires_live_canary_report(tmp_path):
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path, stage="production", include_live_canary=False),
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "live_canary_gate_not_passed"


def test_production_config_requires_manual_monitoring_ack(tmp_path):
    bot = make_bot()
    config = live_gate_config(tmp_path, stage="production", include_live_canary=True)
    config["manual_monitoring_ack"] = False
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **config,
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "manual_monitoring_not_acknowledged"


def test_production_config_can_enable_after_live_canary_report(tmp_path):
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "fee": 0.00015,
        "order_time_in_force": {"entry": "Alo", "exit": "Alo"},
        "market_making": {
            "trading_enabled": True,
            "post_only_verified": True,
            **live_gate_config(tmp_path, stage="production", include_live_canary=True),
        },
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is True
    assert bot.fail_closed_reason == "none"


def test_custom_stake_amount_caps_to_one_inventory_unit():
    bot = make_bot()

    assert bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        4000.0,
        proposed_stake=100.0,
        min_stake=None,
        max_stake=1000.0,
        entry_tag="mm_bid",
        side="long",
    ) == 40.0  # one unit = 0.01 * 4000
    assert bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        4000.0,
        proposed_stake=25.0,
        min_stake=None,
        max_stake=1000.0,
        entry_tag="mm_bid",
        side="long",
    ) == 24.0  # 25/4000 = 0.00625 floors to the 0.001 lot step -> 0.006 * 4000


def test_custom_stake_amount_returns_zero_when_min_stake_exceeds_inventory_unit():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    stake = bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_stake=25.0,
        min_stake=5.0,
        max_stake=25.0,
        leverage=1.0,
        entry_tag="mm_bid",
        side="long",
    )

    assert stake == 0.0
    assert events[0][0] == "stake_rejected"
    assert events[0][1]["reason"] == "min_stake_exceeds_inventory_unit"
    assert events[0][1]["risk_cap"] == 1.0  # 0.01 units * rate 100


def test_strategy_leverage_is_the_configured_target_capped_by_the_venue():
    """Leverage is a fixed financing choice, not a solver output.

    It does not appear in the Cartea-Jaimungal model at all: it changes how much
    margin backs a notional, not inventory risk, adverse selection or funding,
    which are all set by q * inventory_unit_base * mid.
    """
    bot = make_bot()
    bot.target_leverage = 2.0

    assert bot.leverage(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_leverage=5.0,
        max_leverage=20.0,
        entry_tag="mm_bid",
        side="long",
    ) == 2.0

    # Never exceed what the venue permits, whatever the target says.
    assert bot.leverage(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_leverage=5.0,
        max_leverage=1.5,
        entry_tag="mm_bid",
        side="long",
    ) == 1.5

    # A nonsense ceiling must not silently disable leverage.
    assert bot.leverage(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_leverage=5.0,
        max_leverage=float("nan"),
        entry_tag="mm_bid",
        side="long",
    ) == 2.0


def test_production_sizing_defaults_match_the_capital_plan():
    """The shipped defaults, asserted directly rather than via make_bot (which
    pins the legacy values so guard tests stay about mechanics).

    unit = available_capital * utilisation * leverage / (q_max * mid)
         = 1000 * 0.74 * 2 / (6 * 1650) = 0.1495 -> 0.15 ETH
    giving 0.9 ETH at q_max, ~1485 USDC notional, ~742 USDC margin at 2x.
    """
    bot = Market_Making()

    assert bot.hjb_q_max == 6
    assert bot.max_abs_inventory_units == bot.hjb_q_max  # must not drift apart
    assert bot.inventory_unit_base == 0.15
    assert bot.target_leverage == 2.0
    assert bot.spread_multiplier == 1.0  # book optimum, no scaling
    assert bot.extra_cushion_bps == 0.0  # widening is opt-in and additive

    derived = (1000.0 * bot.target_capital_utilisation * bot.target_leverage) / (
        bot.hjb_q_max * 1650.0
    )
    assert abs(derived - bot.inventory_unit_base) / bot.inventory_unit_base < 0.01

    notional = bot.inventory_unit_base * bot.hjb_q_max * 1650.0
    assert notional < bot.max_notional_exposure_usdc
    assert notional / bot.target_leverage < bot.max_margin_used_usdc


def test_custom_stake_amount_rounds_base_amount_down_to_lot_step():
    bot = make_bot()
    bot.inventory_unit_base = 0.0194
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    stake = bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        100.0,
        proposed_stake=10.0,
        min_stake=None,
        max_stake=10.0,
        entry_tag="mm_bid",
        side="long",
    )

    assert stake == 1.9
    assert events[0][0] == "stake_sized"
    assert round(events[0][1]["raw_amount"], 6) == 0.0194
    assert events[0][1]["rounded_amount"] == 0.019
    assert events[0][1]["amount_rounding_applied"] is True


def test_custom_stake_amount_emits_inventory_unit_mismatch_when_amount_drifts():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    # rate 4000: stake cap 25 -> amount 0.006 after lot rounding, 40% below the
    # 0.01 unit -> diagnostic fires (no hard fail).
    stake = bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        4000.0,
        proposed_stake=25.0,
        min_stake=None,
        max_stake=1000.0,
        entry_tag="mm_bid",
        side="long",
    )
    assert stake == 24.0
    mismatches = [p for e, p in events if e == "inventory_unit_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0]["rounded_amount"] == 0.006
    assert mismatches[0]["deviation"] > 0.25

    # rate 100: one unit = 1.0 USDC, amount == unit -> no event.
    events.clear()
    bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        100.0,
        proposed_stake=25.0,
        min_stake=None,
        max_stake=1000.0,
        entry_tag="mm_bid",
        side="long",
    )
    assert not [p for e, p in events if e == "inventory_unit_mismatch"]


def test_confirm_entry_rejects_missing_hjb_cache():
    bot = make_bot()
    bot.trading_enabled = True
    bot.hjb_cache = None

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_confirm_entry_rejects_when_trading_disabled_even_with_valid_quote():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "initial_safety_lock"


def test_confirm_entry_rejects_short_side_in_long_only_mode():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_short",
        "short",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "short_entries_disabled"


def test_confirm_entry_rejects_non_limit_order_type():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "market",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "non_limit_order_type"
    assert events[0][1]["order_type"] == "market"


def test_confirm_entry_rejects_price_not_on_exchange_tick():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.505,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "price_not_tick_safe"
    assert events[0][1]["rounded_price"] == 99.5


def test_post_only_verified_rejects_gtc_entry_time_in_force():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "time_in_force_not_post_only"
    assert events[0][1]["time_in_force"] == "GTC"
    assert events[0][1]["expected_tif"] == "Alo"


def test_repeated_post_only_tif_rejects_trigger_reject_rate_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True
    bot.min_post_only_reject_samples = 2
    bot.max_post_only_reject_rate = 0.5
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    for _ in range(2):
        assert not bot.confirm_trade_entry(
            "ETH/USDC:USDC",
            "limit",
            0.01,
            99.5,
            "GTC",
            datetime.now(timezone.utc),
            "mm_bid",
            "long",
        )

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "post_only_reject_rate_exceeded"
    assert bot._quote_decisions_count == 2
    assert bot._post_only_rejects == 2
    assert [event for event, _ in events] == ["entry_rejected", "kill_switch", "entry_rejected"]
    assert events[1][1]["reason"] == "post_only_reject_rate_exceeded"
    assert events[1][1]["post_only_rejects"] == 2


def test_post_only_verified_accepts_alo_entry_time_in_force():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True

    assert bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.02,
        99.5,
        "Alo",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_accepted_order_attempt_logs_matching_quote_id():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    now = datetime.now(timezone.utc)

    rate = bot.custom_entry_price(
        "ETH/USDC:USDC",
        None,
        now,
        proposed_rate=99.5,
        entry_tag="mm_bid",
        side="long",
    )

    assert bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.02,
        rate,
        "GTC",
        now + timedelta(seconds=1),
        "mm_bid",
        "long",
    )

    quote_payload = events[0][1]
    accepted_payload = events[1][1]
    assert events[0][0] == "quote_decision"
    assert events[1][0] == "order_attempt_accepted"
    assert accepted_payload["quote_id"] == quote_payload["quote_id"]
    assert accepted_payload["quote_id_source"] == "quote_decision_cache"
    assert accepted_payload["trading_enabled"] is True
    assert accepted_payload["dry_run"] is True
    assert accepted_payload["post_only"] is False
    assert accepted_payload["expected_tif"] == "GTC"


def test_post_only_verified_rejects_gtc_exit_time_in_force():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_exit(
        "ETH/USDC:USDC",
        DummyTrade(amount=0.01),
        "limit",
        0.01,
        100.5,
        "GTC",
        "exit_signal",
        datetime.now(timezone.utc),
    )

    assert events[0][0] == "exit_rejected"
    assert events[0][1]["reason"] == "time_in_force_not_post_only"
    assert events[0][1]["time_in_force"] == "GTC"
    assert events[0][1]["expected_tif"] == "Alo"


def test_param_snapshot_status_must_be_ok():
    bot = make_bot()
    bot.kappas["ETH"]["status"] = "seeded_unverified"

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "param_status_not_ok")


def test_lambda_snapshot_must_be_mo_survival_fit():
    bot = make_bot()
    bot.lambdas["ETH"]["lambda_source"] = "lambda_raw"

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "invalid_lambda_source")

    # The legacy v2 source value (binned-density intercept) is rejected too:
    # its lambda is bin-width dependent and ~3x too small.
    bot.lambdas["ETH"]["lambda_source"] = "lambda0_fit"
    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "invalid_lambda_source")


def test_param_update_lock_file_rejects_params(tmp_path):
    bot = make_bot()
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    bot._now_utc = lambda: now
    status_path = tmp_path / "param_update_status.json"
    bot.param_update_status_path = str(status_path)
    status_path.with_name("param_update.lock").write_text(
        json.dumps({"started_at": now.isoformat().replace("+00:00", "Z")}),
        encoding="utf-8",
    )

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "estimator_running")


def test_stale_param_update_lock_fails_closed_with_stale_reason(tmp_path):
    bot = make_bot()
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    bot._now_utc = lambda: now
    bot.param_update_lock_stale_seconds = 60
    status_path = tmp_path / "param_update_status.json"
    bot.param_update_status_path = str(status_path)
    status_path.with_name("param_update.lock").write_text(
        json.dumps({"started_at": (now - timedelta(seconds=61)).isoformat().replace("+00:00", "Z")}),
        encoding="utf-8",
    )

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "estimator_lock_stale")


def test_future_param_update_lock_fails_closed(tmp_path):
    bot = make_bot()
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    bot._now_utc = lambda: now
    status_path = tmp_path / "param_update_status.json"
    bot.param_update_status_path = str(status_path)
    status_path.with_name("param_update.lock").write_text(
        json.dumps({"started_at": (now + timedelta(seconds=30)).isoformat().replace("+00:00", "Z")}),
        encoding="utf-8",
    )

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "estimator_lock_timestamp_future")


def test_future_param_timestamp_rejected():
    bot = make_bot()
    bot._now_utc = lambda: datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    future = "2026-05-25T12:01:00Z"
    bot.kappas["ETH"]["generated_at"] = future

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "param_timestamp_future")


def test_stale_params_reject_entry():
    bot = make_bot()
    bot.trading_enabled = True
    stale = "2026-01-01T00:00:00Z"
    bot.kappas["ETH"]["generated_at"] = stale

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_custom_entry_does_not_log_accept_when_params_stale():
    bot = make_bot()
    bot.trading_enabled = True
    stale = "2026-01-01T00:00:00Z"
    bot.kappas["ETH"]["generated_at"] = stale
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    returned = bot.custom_entry_price(
        "ETH/USDC:USDC",
        None,
        datetime.now(timezone.utc),
        proposed_rate=99.5,
        entry_tag="mm_bid",
        side="long",
    )

    assert returned == 99.5
    assert events[0][0] == "quote_decision"
    assert events[0][1]["decision"] == "reject"
    assert events[0][1]["reason"] == "stale_params"


def test_confirm_entry_rejects_config_fee_mismatch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.config["fee"] = 0.001
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "config_fee_mismatch"


def test_confirm_entry_rejects_exchange_fee_mismatch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.exchange.markets["ETH/USDC:USDC"]["maker"] = 0.001
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "exchange_fee_mismatch"


def test_live_entry_requires_exchange_fee_evidence_after_post_only_verification():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True
    bot.config = {"dry_run": False, "fee": 0.00015}
    bot.exchange.markets["ETH/USDC:USDC"].pop("maker")
    bot.exchange.markets["ETH/USDC:USDC"].pop("taker")
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "Alo",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert events[0][0] == "entry_rejected"
    assert events[0][1]["reason"] == "exchange_fee_unavailable"


def test_live_mode_requires_post_only_verification():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = False
    bot.config = {"dry_run": False}

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_confirm_entry_rejects_boundary_inf_delta():
    bot = make_bot()
    bot.trading_enabled = True
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_custom_entry_price_fallback_still_rejects_boundary_order():
    bot = make_bot()
    bot.trading_enabled = True
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    returned = bot.custom_entry_price(
        "ETH/USDC:USDC",
        None,
        datetime.now(timezone.utc),
        proposed_rate=99.5,
        entry_tag="mm_bid",
        side="long",
    )
    allowed = bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        returned,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert returned == 99.5
    assert not allowed
    assert events[0][0] == "quote_decision"
    assert events[0][1]["reason"] == "boundary_side_disabled"
    assert events[1][0] == "entry_rejected"
    assert events[1][1]["reason"] == "boundary_side_disabled"


def test_custom_entry_price_distance_rejection_blocks_confirm_fallback():
    bot = make_bot()
    bot.trading_enabled = True
    bot.config["custom_price_max_distance_ratio"] = 0.001
    bot.hjb_cache["delta_minus"][1] = 5.0
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    now = datetime.now(timezone.utc)

    returned = bot.custom_entry_price(
        "ETH/USDC:USDC",
        None,
        now,
        proposed_rate=100.0,
        entry_tag="mm_bid",
        side="long",
    )
    allowed = bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        returned,
        "GTC",
        now + timedelta(seconds=1),
        "mm_bid",
        "long",
    )

    assert returned == 100.0
    assert not allowed
    assert events[0][0] == "quote_decision"
    assert events[0][1]["decision"] == "reject"
    assert events[0][1]["reason"] == "custom_price_too_far"
    assert events[1][0] == "entry_rejected"
    assert events[1][1]["reason"] == "custom_price_too_far"


def test_confirm_exit_rejects_boundary_inf_delta():
    bot = make_bot()
    bot.trading_enabled = True
    bot.hjb_cache["delta_plus"][1] = np.inf
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_exit(
        "ETH/USDC:USDC",
        DummyTrade(amount=0.0),
        "limit",
        0.01,
        101.0,
        "GTC",
        "mm_ask",
        datetime.now(timezone.utc),
    )

    assert events[0][0] == "exit_rejected"
    assert events[0][1]["reason"] == "boundary_side_disabled"


def test_custom_exit_price_fallback_still_rejects_boundary_order():
    bot = make_bot()
    bot.trading_enabled = True
    bot.hjb_cache["delta_plus"][1] = np.inf
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    trade = DummyTrade(amount=0.0)

    returned = bot.custom_exit_price(
        "ETH/USDC:USDC",
        trade,
        datetime.now(timezone.utc),
        proposed_rate=101.0,
        current_profit=0.0,
        exit_tag="mm_ask",
    )
    allowed = bot.confirm_trade_exit(
        "ETH/USDC:USDC",
        trade,
        "limit",
        0.01,
        returned,
        "GTC",
        "mm_ask",
        datetime.now(timezone.utc),
    )

    assert returned == 101.0
    assert not allowed
    assert events[0][0] == "quote_decision"
    assert events[0][1]["reason"] == "boundary_side_disabled"
    assert events[1][0] == "exit_rejected"
    assert events[1][1]["reason"] == "boundary_side_disabled"


def test_custom_exit_price_rejects_when_inventory_disallows_ask():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    returned = bot.custom_exit_price(
        "ETH/USDC:USDC",
        DummyTrade(amount=0.0),
        datetime.now(timezone.utc),
        proposed_rate=101.0,
        current_profit=0.0,
        exit_tag="mm_ask",
    )

    assert returned == 101.0
    assert events[0][0] == "quote_decision"
    assert events[0][1]["decision"] == "reject"
    assert events[0][1]["reason"] == "position_limit_reached"


def test_custom_exit_price_accept_path_prices_ask_above_mid():
    # Exercises the ACCEPT path of custom_exit_price, which logs the quote
    # decision with the exit_tag. A NameError there (e.g. referencing entry_tag
    # instead of exit_tag) only surfaces on this path, not the reject paths.
    bot = make_bot()
    bot.trading_enabled = True
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]  # q=1 -> inventory allows ask
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    returned = bot.custom_exit_price(
        "ETH/USDC:USDC",
        DummyTrade(amount=0.01),
        datetime.now(timezone.utc),
        proposed_rate=100.0,
        current_profit=0.0,
        exit_tag="mm_ask",
    )

    assert returned > 100.0  # ask sits above mid (DummyDP mid=100)
    qd = [p for e, p in events if e == "quote_decision"]
    assert qd, "expected a quote_decision on the accept path"
    assert qd[-1]["action"] == "exit"
    assert qd[-1]["decision"] == "accept"
    assert qd[-1]["exit_tag"] == "mm_ask"


def test_confirm_entry_rejects_crossing_bid():
    bot = make_bot()
    bot.trading_enabled = True

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        101.0,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_stop_loss_exit_is_not_blocked():
    bot = make_bot()
    bot.hjb_cache = None

    for exit_reason in (
        "stop_loss",
        "stoploss",
        "stoploss_on_exchange",
        "liquidation",
        "emergency_exit",
        "force_exit",
        "force_sell",
    ):
        assert bot.confirm_trade_exit(
            "ETH/USDC:USDC",
            DummyTrade(amount=0.01),
            "market",
            0.01,
            90.0,
            "GTC",
            exit_reason,
            datetime.now(timezone.utc),
        )


def test_confirm_exit_rejects_non_limit_passive_order_type():
    bot = make_bot()
    bot.trading_enabled = True
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_exit(
        "ETH/USDC:USDC",
        DummyTrade(amount=0.01),
        "market",
        0.01,
        101.5,
        "GTC",
        "mm_ask",
        datetime.now(timezone.utc),
    )

    assert events[0][0] == "exit_rejected"
    assert events[0][1]["reason"] == "non_limit_order_type"
    assert events[0][1]["order_type"] == "market"


def test_taker_fill_triggers_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="taker-1",
        liquidity="taker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="buy",
        price=100.0,
        amount=0.01,
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "unexpected_taker_fill"
    assert [event for event, _ in events] == ["fill", "kill_switch"]
    assert events[0][1]["liquidity"] == "taker"
    assert events[0][1]["quote_side"] == "bid"
    assert events[0][1]["is_taker_fill"] is True
    assert events[1][1]["reason"] == "unexpected_taker_fill"


def test_fill_log_normalizes_quote_side_fee_and_tif_fields():
    bot = make_bot()
    bot.post_only_verified = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="maker-1",
        quote_id="quote-000000000123",
        liquidity="maker",
        order_type="limit",
        time_in_force="Alo",
        ft_order_side="buy",
        price=100.0,
        amount=1.0,
        fee={"cost": 0.015, "rate": 0.00015},
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert len(events) == 1
    event, payload = events[0]
    assert event == "fill"
    assert payload["quote_id"] == "quote-000000000123"
    assert payload["quote_id_source"] == "order_attribute"
    assert payload["liquidity"] == "maker"
    assert payload["liquidity_normalized"] == "maker"
    assert payload["is_maker_fill"] is True
    assert payload["is_taker_fill"] is False
    assert payload["raw_order_side"] == "buy"
    assert payload["quote_side"] == "bid"
    assert payload["actual_fee_paid"] == 0.015
    assert payload["actual_fee_rate"] == 0.00015
    assert payload["expected_fee_rate"] == bot.fees_maker_HL
    assert payload["fee_snapshot"]["actual_fee_matches_strategy"] is True
    assert payload["fee_snapshot"]["config_fee_matches_strategy"] is True
    assert payload["fee_snapshot"]["exchange_maker_fee_matches_strategy"] is True
    assert payload["fee_snapshot"]["fee_agreement_ok"] is True
    assert payload["expected_tif"] == "Alo"
    assert payload["tif_canonical"] == "post_only"
    assert payload["expected_tif_canonical"] == "post_only"
    assert payload["tif_matches_expected"] is True
    assert bot._maker_fill_count == 1


def test_fill_log_infers_quote_id_from_accepted_order_attempt():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    now = datetime.now(timezone.utc)

    rate = bot.custom_entry_price(
        "ETH/USDC:USDC",
        None,
        now,
        proposed_rate=99.5,
        entry_tag="mm_bid",
        side="long",
    )
    assert bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.02,
        rate,
        "GTC",
        now + timedelta(seconds=1),
        "mm_bid",
        "long",
    )
    quote_id = events[0][1]["quote_id"]
    events.clear()
    order = types.SimpleNamespace(
        id="maker-no-quote-id",
        liquidity="maker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="buy",
        price=rate,
        amount=0.02,
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.02), order, now + timedelta(seconds=10))

    assert events[0][0] == "fill"
    assert events[0][1]["quote_id"] == quote_id
    assert events[0][1]["quote_id_source"] == "accepted_order_attempt"


def test_post_only_fill_time_in_force_mismatch_triggers_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="maker-gtc-1",
        liquidity="maker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="buy",
        price=100.0,
        amount=1.0,
        fee={"cost": 0.015, "rate": 0.00015},
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "unexpected_time_in_force"
    assert [event for event, _ in events] == ["fill", "kill_switch"]
    fill_payload = events[0][1]
    assert fill_payload["tif_canonical"] == "gtc"
    assert fill_payload["expected_tif_canonical"] == "post_only"
    assert fill_payload["tif_matches_expected"] is False
    assert events[1][1]["reason"] == "unexpected_time_in_force"


def test_post_only_unknown_fill_liquidity_triggers_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="unknown-liquidity-1",
        liquidity="unknown",
        order_type="limit",
        time_in_force="Alo",
        ft_order_side="buy",
        price=100.0,
        amount=1.0,
        fee={"cost": 0.015, "rate": 0.00015},
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "unknown_fill_liquidity"
    assert [event for event, _ in events] == ["fill", "kill_switch"]
    fill_payload = events[0][1]
    assert fill_payload["liquidity"] == "unknown"
    assert fill_payload["tif_canonical"] == "post_only"
    assert fill_payload["tif_matches_expected"] is True
    assert events[1][1]["reason"] == "unknown_fill_liquidity"


def test_unknown_fill_liquidity_does_not_kill_before_post_only_verified():
    bot = make_bot()
    bot.trading_enabled = True
    bot.post_only_verified = False
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="unknown-liquidity-dryrun-1",
        liquidity="unknown",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="buy",
        price=100.0,
        amount=1.0,
        fee={"cost": 0.015, "rate": 0.00015},
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert bot.trading_enabled
    assert [event for event, _ in events] == ["fill"]
    assert events[0][1]["liquidity"] == "unknown"


def test_fill_markout_events_are_logged_after_horizon():
    bot = make_bot()
    bot.fill_markout_horizons_ms = (100, 1_000)
    bot.dp = DummyDP(best_bid=99.0, best_ask=101.0)
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    fill_ts = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    order = types.SimpleNamespace(
        id="maker-2",
        liquidity="maker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="buy",
        price=99.0,
        amount=0.5,
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.5), order, fill_ts)

    assert [event for event, _ in events] == ["fill"]
    bot._process_pending_fill_markouts("ETH/USDC:USDC", fill_ts + timedelta(milliseconds=100))

    assert [event for event, _ in events] == ["fill", "fill_markout"]
    markout = events[1][1]
    assert markout["quote_side"] == "bid"
    assert markout["horizon_ms"] == 100
    assert markout["fill_price"] == 99.0
    assert markout["future_mid"] == 100.0
    assert markout["markout_usdc_per_base"] == 1.0
    assert markout["markout_usdc"] == 0.5
    assert len(bot._pending_fill_markouts) == 1


def test_ask_fill_markout_uses_ask_sign_convention():
    bot = make_bot()
    bot.fill_markout_horizons_ms = (100,)
    bot.dp = DummyDP(best_bid=99.0, best_ask=101.0)
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    fill_ts = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    order = types.SimpleNamespace(
        id="maker-ask-1",
        liquidity="maker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="sell",
        price=101.0,
        amount=0.5,
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.5), order, fill_ts)
    bot._process_pending_fill_markouts("ETH/USDC:USDC", fill_ts + timedelta(milliseconds=100))

    assert [event for event, _ in events] == ["fill", "fill_markout"]
    markout = events[1][1]
    assert markout["quote_side"] == "ask"
    assert markout["markout_usdc_per_base"] == 1.0
    assert markout["markout_usdc"] == 0.5
    assert bot._pending_fill_markouts == []


def test_amount_below_minimum_is_rejected():
    bot = make_bot()
    bot.trading_enabled = True

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.001,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )


def test_confirm_exit_rejects_price_not_on_exchange_tick():
    bot = make_bot()
    bot.trading_enabled = True
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]
    trade = DummyTrade(amount=0.01)
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    assert not bot.confirm_trade_exit(
        "ETH/USDC:USDC",
        trade,
        "limit",
        0.01,
        100.005,
        "GTC",
        "exit_signal",
        datetime.now(timezone.utc),
    )

    assert events[0][0] == "exit_rejected"
    assert events[0][1]["reason"] == "price_not_tick_safe"
    assert events[0][1]["rounded_price"] == 100.01


def test_param_refresh_skips_without_hardcoded_symbol_fallback(monkeypatch):
    bot = make_bot()
    bot.dp = EmptyWhitelistDP()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot.bot_loop_start(datetime.now(timezone.utc))

    assert not hasattr(sys.modules["Market_Making"], "schedule_tests")
    assert events == [("param_update_skipped", {"reason": "no_pair"})]


def test_bot_loop_reloads_parameter_snapshots_without_running_estimators(monkeypatch):
    module = sys.modules["Market_Making"]
    bot = make_bot()
    now = datetime.now(timezone.utc)
    new_kappas = json.loads(json.dumps(bot.kappas))
    new_epsilons = json.loads(json.dumps(bot.epsilons))
    new_lambdas = json.loads(json.dumps(bot.lambdas))
    new_kappas["ETH"]["kappa+"] = 3.0
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._refresh_hjb = lambda pair: events.append(("hjb_refreshed", {"pair": pair}))
    monkeypatch.setattr(module, "load_configs", lambda start_dir=None: (new_kappas, new_epsilons, new_lambdas))

    bot.bot_loop_start(now)

    assert not hasattr(module, "schedule_tests")
    assert bot.kappas["ETH"]["kappa+"] == 3.0
    assert ("hjb_refreshed", {"pair": "ETH/USDC:USDC"}) in events


def test_bot_loop_rejects_internal_estimator_config_and_consumes_snapshots(monkeypatch):
    module = sys.modules["Market_Making"]
    bot = make_bot()
    bot.config = {"dry_run": True, "fee": 0.00015, "market_making": {"run_estimators_in_strategy": True}}
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._refresh_hjb = lambda pair: None
    monkeypatch.setattr(module, "load_configs", lambda start_dir=None: (bot.kappas, bot.epsilons, bot.lambdas))

    bot.bot_loop_start(datetime.now(timezone.utc))

    assert ("param_update_skipped", {"reason": "internal_estimator_disabled_use_sidecar"}) in events


class _FakeParamStore:
    def __init__(self, blob):
        self._blob = blob
        self.calls = []

    def fetch_params(self, redis_url, crypto):
        self.calls.append((redis_url, crypto))
        return self._blob


def test_load_params_falls_back_to_files_without_redis(tmp_path):
    bot = make_bot()
    # No redis_url anywhere -> file path. The snapshot files are generated at
    # runtime and untracked, so the test supplies its own and points the
    # strategy at them via market_making.param_dir.
    param_dir = write_param_snapshot_files(tmp_path / "params")
    bot.config["market_making"] = {"param_dir": str(param_dir)}

    kappas, epsilons, lambdas = bot._load_params()

    assert bot._param_source == "file"
    assert kappas["ETH"]["kappa+"] == 2.0
    assert epsilons["ETH"]["epsilon+"] == 0.0
    assert lambdas["ETH"]["lambda_source"] == "mo_survival_fit"


def test_load_params_uses_redis_blob_when_available():
    bot = make_bot()
    bot.config["market_making"] = {"redis_url": "redis://fake:6379/0"}
    blob = {
        "kappa": {"kappa+": 3.0, "status": "ok"},
        "epsilon": {"epsilon+": 0.0, "status": "ok"},
        "lambda": {"lambda+": 0.2, "lambda_source": "mo_survival_fit"},
    }
    bot._param_store = _FakeParamStore(blob)
    bot._param_store_loaded = True

    kappas, epsilons, lambdas = bot._load_params()

    assert bot._param_source == "redis"
    assert kappas == {"ETH": blob["kappa"]}
    assert epsilons == {"ETH": blob["epsilon"]}
    assert lambdas == {"ETH": blob["lambda"]}


def test_redis_source_skips_file_lock_in_params_valid(tmp_path):
    # A present lock file would reject the file path with estimator_running, but
    # the Redis (atomic blob) path must not be blocked by it.
    bot = make_bot()
    status_path = tmp_path / "param_update_status.json"
    bot.param_update_status_path = str(status_path)
    status_path.with_name("param_update.lock").write_text(
        json.dumps({"started_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )

    bot._param_source = "file"
    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "estimator_running")

    bot._param_source = "redis"
    ok, reason = bot._params_are_valid("ETH/USDC:USDC")
    assert reason != "estimator_running"
    assert (ok, reason) == (True, "ok")


def test_file_fallback_gate_blocks_reload_when_redis_down_and_estimator_running(tmp_path):
    # Redis configured but unavailable -> _load_params falls back to the JSON
    # files. On the periodic reload path (enforce_file_gate=True) a present
    # estimator lock must block the load (torn-read protection), and the
    # provenance marker must keep describing the params still in memory.
    from Market_Making import ParamReloadBlocked

    bot = make_bot()
    param_dir = write_param_snapshot_files(tmp_path / "params")
    bot.config["market_making"] = {
        "redis_url": "redis://fake:6379/0",
        "param_dir": str(param_dir),
    }
    bot._param_store = _FakeParamStore(None)  # configured but serving nothing
    bot._param_store_loaded = True
    status_path = tmp_path / "param_update_status.json"
    bot.param_update_status_path = str(status_path)
    status_path.with_name("param_update.lock").write_text(
        json.dumps({"started_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    bot._param_source = "redis"

    try:
        bot._load_params(enforce_file_gate=True)
    except ParamReloadBlocked as blocked:
        assert blocked.reason == "estimator_running"
    else:
        raise AssertionError("expected ParamReloadBlocked while estimator lock is present")
    assert bot._param_source == "redis"  # marker untouched on a blocked reload

    # Without the gate (bot_start initial load) the file load proceeds.
    kappas, _, _ = bot._load_params()
    assert bot._param_source == "file"
    assert isinstance(kappas, dict)

    # Once the estimator lock is gone, the gated reload also proceeds.
    status_path.with_name("param_update.lock").unlink()
    kappas, _, _ = bot._load_params(enforce_file_gate=True)
    assert bot._param_source == "file"
    assert isinstance(kappas, dict)


def test_real_strategy_callback_surface_matches_freqtrade():
    # Validate the ACTUAL strategy's callback signatures against freqtrade's
    # expected surface. Previously only a hand-written reference class was
    # checked, so real signature drift (e.g. adjust_exit_price using exit_tag
    # instead of freqtrade's entry_tag) slipped through and only blew up at
    # runtime.
    from verify_freqtrade_callback_surface import build_callback_surface_report

    report = build_callback_surface_report(
        Market_Making, strategy_path=STRATEGIES / "Market_Making.py"
    )
    assert report["ok"] is True, report["reasons"]


def test_strategy_uses_live_safe_trade_accessor():
    # freqtrade's Trade.get_trades() takes a SQLAlchemy filter list, not
    # is_open=/pair= kwargs, and raises TypeError at runtime when called that
    # way. That error is swallowed (except: pass) and silently reads the
    # position as zero, so the bot can enter but never sees its inventory and
    # never quotes exits. The strategy must use the get_trades_proxy accessor,
    # which works in live/dry-run/backtest.
    src = (STRATEGIES / "Market_Making.py").read_text(encoding="utf-8")
    assert "Trade.get_trades(" not in src, "use Trade.get_trades_proxy (kwargs-safe), not Trade.get_trades"
    assert "Trade.get_trades_proxy(" in src


def test_exchange_position_takes_priority_over_open_trade_count():
    bot = make_bot()
    bot.exchange.positions = [{"symbol": "ETH/USDC:USDC", "side": "long", "contracts": "0.02"}]
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]

    assert bot._inventory_level("ETH/USDC:USDC") == 2  # 0.02 / unit 0.01, not the open trade's 0.01


def test_unexpected_short_position_rejects_entry_and_kills_strategy():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.exchange.positions = [{"symbol": "ETH/USDC:USDC", "side": "short", "contracts": "0.02"}]

    assert not bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        "limit",
        0.01,
        99.5,
        "GTC",
        datetime.now(timezone.utc),
        "mm_bid",
        "long",
    )

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "unexpected_short_position"
    assert [event for event, _ in events] == ["kill_switch", "risk_flatten_requested", "entry_rejected"]
    assert events[0][1] == {
        "reason": "unexpected_short_position",
        "pair": "ETH/USDC:USDC",
        "signed_base_position": -0.02,
        # Which leg of the pair died: with two instances running, the reason
        # alone no longer identifies the instance.
        "role": "long",
        # no_open_order_source is what real freqtrade produces: none of the three
        # methods the cancel path probes exist on its Exchange. The fixture used
        # to define cancel_all_orders and hide that.
        "cancel_open_orders_requested": 0,
        "cancel_method": "no_open_order_source",
        "risk_flatten_request_emitted": True,
        "risk_action_id": "risk-000000000001",
    }
    assert events[1][1]["risk_action_id"] == "risk-000000000001"
    assert events[1][1]["reason"] == "unexpected_short_position"
    assert events[1][1]["flatten_side"] == "buy"
    assert events[1][1]["size_base"] == 0.02
    assert events[1][1]["reference_price"] == 100.0
    assert events[1][1]["order_type"] == "limit"
    assert events[1][1]["time_in_force"] == "Ioc"
    assert events[1][1]["reduce_only"] is True
    assert events[1][1]["client_order_id"].startswith("risk|sess=freqtrade|rid=risk-000000000001|mode=flatten")
    assert events[1][1]["cloid"].startswith("0x")
    assert events[2][1]["reason"] == "unexpected_short_position"
    assert events[2][1]["signed_base_position"] == -0.02
    assert events[2][1]["q"] == 0


def test_kill_switch_cancels_open_orders_with_cancel_order_fallback():
    """Exercises the cancel_order fallback for an exchange that DOES expose open
    orders. Real freqtrade does not (see ABSENT_FROM_REAL_FREQTRADE_EXCHANGE), so
    this covers the code path, not the shipped configuration.
    """
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    class CancelOrderExchange:
        def __init__(self):
            self.open_orders = [
                {"symbol": "ETH/USDC:USDC", "id": "eth-open", "status": "open", "remaining": 0.01},
                {"symbol": "ETH/USDC:USDC", "id": "eth-closed", "status": "closed", "remaining": 0.0},
                {"symbol": "BTC/USDC:USDC", "id": "btc-open", "status": "open", "remaining": 0.01},
            ]
            self.cancel_calls = []

        def cancel_order(self, order_id, pair):
            self.cancel_calls.append((order_id, pair))

    exchange = CancelOrderExchange()
    bot.exchange = exchange

    bot._trigger_kill_switch("manual_test", {"pair": "ETH/USDC:USDC"})

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "manual_test"
    assert exchange.cancel_calls == [("eth-open", "ETH/USDC:USDC")]
    assert bot._cancel_open_order_requests == 1
    assert events == [
        (
            "kill_switch",
            {
                "reason": "manual_test",
                "pair": "ETH/USDC:USDC",
                "cancel_open_orders_requested": 1,
                "cancel_method": "cancel_order",
                "cancel_source": "exchange_open_orders",
                "cancelled_order_ids": ["eth-open"],
                "cancel_errors": [],
                "risk_flatten_request_emitted": False,
            },
        )
    ]


def test_kill_switch_emits_reduce_only_flatten_request_for_remaining_long_position():
    bot = make_bot()
    bot.trading_enabled = True
    bot.exchange.positions = [{"symbol": "ETH/USDC:USDC", "side": "long", "contracts": "0.02"}]
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot._trigger_kill_switch("position_limit_reached", {"pair": "ETH/USDC:USDC", "price": 125.0})

    assert [event for event, _ in events] == ["kill_switch", "risk_flatten_requested"]
    assert events[0][1]["risk_flatten_request_emitted"] is True
    flatten = events[1][1]
    assert flatten["risk_action_id"] == "risk-000000000001"
    assert flatten["reason"] == "position_limit_reached"
    assert flatten["signed_base_position"] == 0.02
    assert flatten["flatten_side"] == "sell"
    assert flatten["size_base"] == 0.02
    assert flatten["reference_price"] == 125.0
    assert flatten["notional_usdc"] == 2.5
    assert flatten["executor"] == "hyperliquid_risk_executor.py"
    assert flatten["executor_mode"] == "submit-flatten"
    assert flatten["requires_env"] == "HYPERLIQUID_RISK_FLATTEN_ALLOW=1"
    assert flatten["requires_acknowledgement"] == "--acknowledge-risk-reducing-taker"


def test_daily_loss_triggers_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.max_daily_loss_usdc = 20
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="loss-1",
        liquidity="maker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="sell",
        price=100.0,
        amount=0.01,
        realized_pnl=-21.0,
    )

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "drawdown_limit_reached"
    assert [event for event, _ in events] == ["fill", "risk_update", "kill_switch"]
    assert events[1][1]["daily_realized_pnl"] == -21.0
    assert events[1][1]["max_daily_loss_usdc"] == 20.0
    assert events[2][1]["daily_realized_pnl"] == -21.0


def test_consecutive_losses_trigger_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.max_daily_loss_usdc = 1_000
    bot.max_consecutive_losses = 2
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    for idx in range(2):
        order = types.SimpleNamespace(
            id=f"loss-{idx}",
            liquidity="maker",
            order_type="limit",
            time_in_force="GTC",
            ft_order_side="sell",
            price=100.0,
            amount=0.01,
            realized_pnl=-1.0,
        )
        bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, datetime.now(timezone.utc))

    assert not bot.trading_enabled
    assert bot.fail_closed_reason == "consecutive_losses_limit_reached"
    risk_updates = [payload for event, payload in events if event == "risk_update"]
    assert [payload["consecutive_losses"] for payload in risk_updates] == [1, 2]
    assert risk_updates[-1]["max_consecutive_losses"] == 2
    kill_payload = events[-1][1]
    assert kill_payload["reason"] == "consecutive_losses_limit_reached"
    assert kill_payload["consecutive_losses"] == 2


def test_duplicate_fill_does_not_double_count_realized_pnl():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    order = types.SimpleNamespace(
        id="same-fill",
        liquidity="maker",
        order_type="limit",
        time_in_force="GTC",
        ft_order_side="sell",
        price=100.0,
        amount=0.01,
        realized_pnl=-2.0,
    )
    now = datetime.now(timezone.utc)

    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, now)
    bot.order_filled("ETH/USDC:USDC", DummyTrade(amount=0.01), order, now)

    assert bot._daily_realized_pnl_usdc == -2.0
    assert bot._consecutive_losses == 1
    assert [event for event, _ in events].count("risk_update") == 1


def test_adjust_exit_price_reprices_passive_ask():
    bot = make_bot()
    bot.trading_enabled = True
    bot.config["custom_price_max_distance_ratio"] = 0.05
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]
    trade = DummyTrade(amount=0.01)
    order = types.SimpleNamespace()

    adjusted = bot.adjust_exit_price(
        trade,
        order,
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        proposed_rate=100.0,
        current_order_rate=100.0,
        entry_tag="mm_ask",
        side="long",
    )

    assert adjusted > 100.0
    assert events[0][0] == "quote_decision"
    assert events[0][1]["action"] == "adjust_exit"
    assert events[0][1]["decision"] == "accept"
    assert events[0][1]["custom_price_distance_ratio"] is not None
    assert events[0][1]["custom_price_max_distance_ratio"] == 0.05


def test_adjust_entry_price_cancels_open_order_when_params_are_stale():
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot.kappas["ETH"]["generated_at"] = "2026-01-01T00:00:00Z"
    trade = DummyTrade(amount=0.0)
    order = types.SimpleNamespace()

    adjusted = bot.adjust_entry_price(
        trade,
        order,
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        proposed_rate=99.5,
        current_order_rate=99.5,
        entry_tag="mm_bid",
        side="long",
    )

    assert adjusted is None
    assert events[0][0] == "quote_decision"
    assert events[0][1]["action"] == "adjust_entry"
    assert events[0][1]["decision"] == "reject"
    assert events[0][1]["reason"] == "stale_params"
    assert events[0][1]["cancel_open_order"] is True


def test_param_age_seconds_uses_oldest_snapshot():
    bot = make_bot()
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    bot.kappas["ETH"]["generated_at"] = (now - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    bot.epsilons["ETH"]["generated_at"] = (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    bot.lambdas["ETH"]["generated_at"] = (now - timedelta(seconds=20)).isoformat().replace("+00:00", "Z")

    assert bot._param_age_seconds("ETH", now) == 30.0


def test_book_freshness_rejects_stale_timestamp():
    bot = make_bot()
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    bot.dp = DummyDP(timestamp=now - timedelta(seconds=6))
    bot.max_book_age_seconds = 5

    assert bot._book_is_fresh("ETH/USDC:USDC", now) == (False, "stale_orderbook")


def test_market_data_fresh_uses_thirty_second_default():
    bot = make_bot()
    delattr(bot, "_market_data_fresh")
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    bot._now_utc = lambda: now
    bot._collector_stream_timestamps = lambda symbol: {
        stream: now - timedelta(seconds=31)
        for stream in bot.collector_required_streams
    }

    ok, reason = bot._market_data_fresh("ETH")

    assert not ok
    assert reason == "collector_data_stale_orderbooks_31.0s"


def test_market_data_fresh_uses_parquet_timestamps_not_file_mtime(tmp_path):
    bot = make_bot()
    delattr(bot, "_market_data_fresh")
    delattr(bot, "_collector_age_seconds")
    now = datetime.now(timezone.utc)
    stale_ts = now - timedelta(seconds=120)
    symbol_dir = tmp_path / "ETH"
    for stream in bot.collector_required_streams:
        stream_dir = symbol_dir / stream
        stream_dir.mkdir(parents=True)
        shard = stream_dir / f"{stream}.parquet"
        pd.DataFrame({"timestamp": [stale_ts.timestamp()]}).to_parquet(shard, index=False)
        os.utime(shard, None)

    bot._collector_symbol_dir = lambda symbol: symbol_dir
    bot.collector_timestamp_cache_seconds = 0

    ok, reason = bot._market_data_fresh("ETH", max_age_seconds=30)

    assert not ok
    assert reason.startswith("collector_data_stale_")
    assert bot._collector_age_seconds("ETH", now) >= 100


def test_market_data_fresh_rejects_missing_required_stream(tmp_path):
    bot = make_bot()
    delattr(bot, "_market_data_fresh")
    symbol_dir = tmp_path / "ETH"
    stream_dir = symbol_dir / "prices"
    stream_dir.mkdir(parents=True)
    pd.DataFrame({"timestamp": [datetime.now(timezone.utc).timestamp()]}).to_parquet(
        stream_dir / "prices.parquet",
        index=False,
    )
    bot._collector_symbol_dir = lambda symbol: symbol_dir
    bot.collector_timestamp_cache_seconds = 0

    ok, reason = bot._market_data_fresh("ETH", max_age_seconds=30)

    assert not ok
    assert reason == "missing_collector_streams:orderbooks,trades"


def test_quote_state_maps_missing_collector_stream_to_stale_collector_data():
    bot = make_bot()
    delattr(bot, "_market_data_fresh")
    bot.trading_enabled = True
    now = datetime.now(timezone.utc)
    bot._now_utc = lambda: now
    bot._collector_stream_timestamps = lambda symbol: {
        "prices": now,
        "orderbooks": None,
        "trades": None,
    }

    ok, reason = bot._quote_state_valid(
        "ETH/USDC:USDC",
        "long",
        99.5,
        now,
    )

    assert not ok
    assert reason == "stale_collector_data"


def test_quote_state_rejects_projected_notional_exposure_above_cap():
    bot = make_bot()
    bot.trading_enabled = True
    bot.max_notional_exposure_usdc = 0.5
    now = datetime.now(timezone.utc)

    ok, reason = bot._quote_state_valid("ETH/USDC:USDC", "bid", 99.5, now)

    assert not ok
    assert reason == "notional_exposure_limit_reached"


def test_position_risk_allows_risk_reducing_ask_when_current_position_is_over_cap():
    bot = make_bot()
    bot.max_notional_exposure_usdc = 0.5
    bot._signed_base_position = lambda pair: 0.02

    ok, reason, snapshot = bot._position_risk_valid("ETH/USDC:USDC", "ask", 100.0, amount=0.01)

    assert ok is True
    assert reason == "ok"
    assert snapshot["risk_reducing"] is True
    assert snapshot["notional_exposure_usdc"] == 1.0


def test_live_position_risk_requires_liquidation_buffer_evidence():
    bot = make_bot()
    bot.config = {"dry_run": False, "fee": 0.00015}

    ok, reason, snapshot = bot._position_risk_valid("ETH/USDC:USDC", "bid", 100.0, amount=0.01)

    assert ok is False
    assert reason == "liquidation_buffer_unknown"
    assert snapshot["liquidation_buffer_usdc"] is None


def test_live_position_risk_rejects_low_liquidation_buffer():
    bot = make_bot()
    bot.config = {"dry_run": False, "fee": 0.00015, "available_capital": 100.0}

    ok, reason, snapshot = bot._position_risk_valid("ETH/USDC:USDC", "bid", 100.0, amount=0.01)

    assert ok is False
    assert reason == "liquidation_buffer_too_low"
    assert snapshot["liquidation_buffer_usdc"] == 99.95


def test_confirm_trade_entry_rechecks_exact_order_notional_exposure():
    bot = make_bot()
    bot.trading_enabled = True
    bot.max_notional_exposure_usdc = 10.0
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    now = datetime.now(timezone.utc)

    ok = bot.confirm_trade_entry(
        "ETH/USDC:USDC",
        order_type="limit",
        amount=1.0,
        rate=99.5,
        time_in_force="GTC",
        current_time=now,
        entry_tag="mm_bid",
        side="long",
    )

    assert ok is False
    assert events[-1][0] == "entry_rejected"
    assert events[-1][1]["reason"] == "notional_exposure_limit_reached"
    assert events[-1][1]["position_risk"]["notional_exposure_usdc"] == 99.5


def test_quote_decision_logs_freshness_age_fields():
    bot = make_bot()
    bot.post_only_verified = True
    bot._hjb_params_snapshot = bot._params_snapshot("ETH")
    bot._hjb_param_fingerprint = bot._params_fingerprint(bot._hjb_params_snapshot)
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._collector_age_seconds = lambda symbol, now=None: 12.4
    bot._book_age_ms = lambda pair, current_time=None: 180.0

    bot._log_quote_decision(
        pair="ETH/USDC:USDC",
        symbol="ETH",
        side="bid",
        action="entry",
        decision="accept",
        reason="ok",
        mid_price=100.0,
        proposed_rate=99.5,
        raw_price=99.4,
        rounded_price=99.4,
        delta_model=0.5,
        fee_cushion=0.1,
        delta_total=0.6,
    )

    assert events[0][0] == "quote_decision"
    payload = events[0][1]
    assert payload["quote_id"] == "quote-000000000001"
    assert payload["param_age_seconds"] is not None
    assert payload["collector_age_seconds"] == 12.4
    assert payload["book_age_ms"] == 180.0
    assert payload["params_fresh"] is True
    assert payload["collector_fresh"] is True
    assert payload["book_fresh"] is True
    assert payload["expected_tif"] == "Alo"
    assert payload["expected_tif_canonical"] == "post_only"
    assert payload["post_only"] is True
    assert payload["post_only_verified"] is True
    assert payload["trading_enabled"] is False
    assert payload["dry_run"] is True
    assert payload["hjb_param_fingerprint"] == bot._hjb_param_fingerprint
    assert payload["hjb_params"]["sources"]["kappa"]["schema_version"] == 3
    assert payload["hjb_params"]["sources"]["kappa"]["generated_at"] == bot.kappas["ETH"]["generated_at"]
    assert payload["hjb_params"]["sources"]["lambda"]["lambda_source"] == "mo_survival_fit"
    assert payload["params"]["sources"]["epsilon"]["n_buy_events"] == 60
    assert payload["fee_snapshot"]["config_fee_matches_strategy"] is True
    assert payload["fee_snapshot"]["exchange_maker_fee_matches_strategy"] is True


def test_quote_decision_ids_are_monotonic():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    for side in ("bid", "ask"):
        bot._log_quote_decision(
            pair="ETH/USDC:USDC",
            symbol="ETH",
            side=side,
            action="entry" if side == "bid" else "exit",
            decision="reject",
            reason="test_reason",
            mid_price=100.0,
            proposed_rate=100.0,
        )

    assert [payload["quote_id"] for _, payload in events] == [
        "quote-000000000001",
        "quote-000000000002",
    ]


def test_hjb_refresh_logs_parameter_fingerprint():
    bot = make_bot()
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    class DummySolver:
        @staticmethod
        def compute_h_asymmetric(**kwargs):
            return {
                "method": "test_solver",
                "q_grid": np.array([-1, 0, 1]),
                "delta_plus": np.array([np.inf, 0.5, 0.4]),
                "delta_minus": np.array([0.4, 0.5, np.inf]),
            }

    bot.hjb_solver = DummySolver

    bot._refresh_hjb("ETH/USDC:USDC")

    refresh = [payload for event, payload in events if event == "hjb_refresh"][0]
    assert refresh["param_fingerprint"] == bot._hjb_param_fingerprint
    assert refresh["params"]["sources"]["kappa"]["schema_version"] == 3
    assert refresh["params"]["sources"]["epsilon"]["n_sell_events"] == 60
    assert bot._hjb_params_snapshot == refresh["params"]


def test_health_log_counts_open_orders_and_logs_position():
    bot = make_bot()
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))
    bot._collector_age_seconds = lambda symbol, now=None: 2.0
    bot._book_age_ms = lambda pair, current_time=None: 125.0
    bot.exchange.open_orders = [
        {"symbol": "ETH/USDC:USDC", "id": "a"},
        {"symbol": "ETH/USDC:USDC", "id": "b"},
        {"symbol": "BTC/USDC:USDC", "id": "c"},
    ]

    bot._log_health("ETH/USDC:USDC", now)

    assert events[0][0] == "health"
    payload = events[0][1]
    assert payload["pair"] == "ETH/USDC:USDC"
    assert payload["symbol"] == "ETH"
    assert payload["dry_run"] is True
    assert payload["stake_amount"] is None
    assert payload["open_orders"] == 2
    assert payload["open_orders_source"] == "exchange_open_orders"
    assert payload["position"] == 0.0
    assert payload["signed_base_position"] == 0.0
    assert payload["kill_on_taker_fill"] is True
    assert payload["deployment_stage"] == "research"
    assert payload["deployment_gate_reports_required"] is True
    assert payload["manual_monitoring_ack"] is False
    assert payload["max_deployment_report_age_seconds"] == bot.max_deployment_report_age_seconds
    assert payload["expected_entry_time_in_force"] == "GTC"
    assert payload["expected_entry_time_in_force_canonical"] == "gtc"
    assert payload["unrealized_pnl"] == 0.0
    assert payload["max_daily_loss_usdc"] == bot.max_daily_loss_usdc
    assert payload["max_consecutive_losses"] == bot.max_consecutive_losses
    assert payload["max_post_only_reject_rate"] == bot.max_post_only_reject_rate
    assert payload["max_abs_inventory_units"] == bot.max_abs_inventory_units
    assert payload["notional_exposure_usdc"] == 0.0
    assert payload["max_notional_exposure_usdc"] == bot.max_notional_exposure_usdc
    assert payload["margin_used_usdc"] == 0.0
    assert payload["max_margin_used_usdc"] == bot.max_margin_used_usdc
    assert payload["min_liquidation_buffer_usdc"] == bot.min_liquidation_buffer_usdc
    assert payload["fee_snapshot"]["config_fee_matches_strategy"] is True
    assert payload["fee_snapshot"]["exchange_maker_fee_matches_strategy"] is True


def test_health_log_counts_open_orders_from_trades_when_exchange_unavailable():
    bot = make_bot()
    # open_orders is absent by default now, matching real freqtrade.
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    events = []
    DummyTrade._open_trades = [
        DummyTrade(
            amount=0.01,
            orders=[
                {"status": "open", "remaining": 0.01},
                {"status": "closed", "remaining": 0.0},
            ],
        ),
        DummyTrade(amount=0.01, has_open_orders=True),
    ]
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot._log_health("ETH/USDC:USDC", now)

    assert events[0][0] == "health"
    assert events[0][1]["open_orders"] == 2
    assert events[0][1]["open_orders_source"] == "freqtrade_open_trades"


def test_health_log_counts_freqtrade_open_order_trades():
    bot = make_bot()
    # open_orders is absent by default now, matching real freqtrade.
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    events = []
    DummyTrade._open_order_trades = [
        DummyTrade(
            amount=0.01,
            orders=[
                {"status": "open", "remaining": 0.01},
                {"status": "cancelled", "remaining": 0.01},
            ],
        ),
        DummyTrade(amount=0.01, has_open_orders=True),
        DummyTrade(amount=0.01, has_open_orders=True, pair="BTC/USDC:USDC"),
    ]
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot._log_health("ETH/USDC:USDC", now)

    assert events[0][0] == "health"
    assert events[0][1]["open_orders"] == 2
    assert events[0][1]["open_orders_source"] == "freqtrade_open_order_trades"


def test_health_log_uses_accepted_order_estimate_when_runtime_sources_are_hidden():
    bot = make_bot()
    # open_orders is absent by default now, matching real freqtrade.
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    events = []
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]
    bot._accepted_order_attempts = 2
    bot._maker_fill_count = 1
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot._log_health("ETH/USDC:USDC", now)

    assert events[0][0] == "health"
    payload = events[0][1]
    assert payload["open_orders"] == 1
    assert payload["open_orders_source"] == "accepted_confirmation_estimate"
    assert payload["accepted_order_attempts"] == 2


def test_health_log_marks_open_trade_to_mid_price():
    bot = make_bot()
    bot.dp = DummyDP(best_bid=109.0, best_ask=111.0)
    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    events = []
    DummyTrade._open_trades = [DummyTrade(amount=0.02, open_rate=100.0)]
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot._log_health("ETH/USDC:USDC", now)

    assert events[0][0] == "health"
    payload = events[0][1]
    assert payload["signed_base_position"] == 0.02
    assert payload["q"] == 2
    assert payload["unrealized_pnl"] == 0.2


# ---------------------------------------------------------------------------
# Role-aware guards (two-instance market maker)
# ---------------------------------------------------------------------------


def test_long_role_rejects_short_entries():
    bot = make_bot()
    bot.can_short = False

    assert bot._entry_side_rejection_reason("short") == "short_entries_disabled"
    assert bot._entry_side_rejection_reason("sell") == "short_entries_disabled"
    assert bot._entry_side_rejection_reason("long") is None
    assert bot._entry_side_rejection_reason("buy") is None


def test_short_role_accepts_shorts_and_rejects_longs():
    """can_short=True means SHORT ONLY, not 'either direction'.

    Before this, line 1900 rejected any side outside the long set regardless of
    can_short, so the short instance could never place an order at all.
    """
    bot = make_bot()
    bot.can_short = True

    assert bot._entry_side_rejection_reason("short") is None
    assert bot._entry_side_rejection_reason("sell") is None
    assert bot._entry_side_rejection_reason("long") == "long_entries_disabled"
    assert bot._entry_side_rejection_reason("buy") == "long_entries_disabled"
    # An unspecified side is freqtrade not telling us, not a role violation.
    assert bot._entry_side_rejection_reason("") is None


def test_unsupported_sides_are_still_rejected_for_both_roles():
    for can_short in (False, True):
        bot = make_bot()
        bot.can_short = can_short
        assert bot._entry_side_rejection_reason("sideways") == "unsupported_entry_side"


def test_wrong_way_exposure_kills_each_role_independently():
    """The guard is mirrored, not removed: each leg dies on exposure pointing
    the wrong way for its own role."""
    long_bot = make_bot()
    long_bot.can_short = False
    assert long_bot._reject_unexpected_short_position("ETH/USDC:USDC", -0.5) is True
    assert long_bot.fail_closed_reason == "unexpected_short_position"
    assert long_bot._reject_unexpected_short_position("ETH/USDC:USDC", 0.5) is False

    short_bot = make_bot()
    short_bot.can_short = True
    assert short_bot._reject_unexpected_short_position("ETH/USDC:USDC", 0.5) is True
    assert short_bot.fail_closed_reason == "unexpected_long_position"
    assert short_bot._reject_unexpected_short_position("ETH/USDC:USDC", -0.5) is False


def test_flat_is_never_a_role_violation():
    for can_short in (False, True):
        bot = make_bot()
        bot.can_short = can_short
        assert bot._reject_unexpected_short_position("ETH/USDC:USDC", 0.0) is False
        assert bot.trading_enabled is not False or bot.fail_closed_reason != "unexpected_long_position"


def test_physical_quote_side_is_mirrored_for_the_short_leg():
    """Freqtrade speaks entries/exits; the book speaks bids/asks, and the map
    between them flips with the role. Getting this backwards would price the
    short leg's cover off the ask depth."""
    long_bot = make_bot()
    long_bot.can_short = False
    assert long_bot._physical_quote_side("long") == "bid"
    assert long_bot._physical_quote_side("exit") == "ask"

    short_bot = make_bot()
    short_bot.can_short = True
    assert short_bot._physical_quote_side("short") == "ask"
    assert short_bot._physical_quote_side("exit") == "bid"

    # An explicit book side is always taken literally, whatever the role.
    for bot in (long_bot, short_bot):
        assert bot._physical_quote_side("bid") == "bid"
        assert bot._physical_quote_side("ask") == "ask"


def test_short_leg_inventory_gates_are_mirrored():
    """For the short leg an ask ADDS (capped by q_max) and a bid COVERS (needs
    an existing short) -- the mirror of the long leg."""
    bot = make_bot()
    bot.can_short = True
    bot._reject_unexpected_short_position = lambda pair, signed_base=None: False

    bot._inventory_level = lambda pair: 0
    assert bot._inventory_allows_ask("ETH/USDC:USDC") is True   # can open a short
    assert bot._inventory_allows_bid("ETH/USDC:USDC") is False  # nothing to cover

    bot._inventory_level = lambda pair: -2
    assert bot._inventory_allows_ask("ETH/USDC:USDC") is True   # room to add
    assert bot._inventory_allows_bid("ETH/USDC:USDC") is True   # can cover

    bot._inventory_level = lambda pair: -3
    assert bot._inventory_allows_ask("ETH/USDC:USDC") is False  # at the cap
    assert bot._inventory_allows_bid("ETH/USDC:USDC") is True


def test_short_leg_emits_short_signals():
    """Instance S was dead on arrival while populate_*_trend only ever wrote
    enter_long / exit_long."""
    import pandas as pd

    bot = make_bot()
    bot.can_short = True
    bot.trading_enabled = True
    bot._model_ready = lambda pair: True
    bot._reject_unexpected_short_position = lambda pair, signed_base=None: False
    bot._inventory_level = lambda pair: 0

    df = pd.DataFrame({"close": [100.0, 101.0]})
    out = bot.populate_entry_trend(df.copy(), {"pair": "ETH/USDC:USDC"})
    assert out.iloc[-1]["enter_short"] == 1
    assert out.iloc[-1]["enter_long"] == 0
    assert out.iloc[-1]["enter_tag"] == "mm_ask"

    bot._inventory_level = lambda pair: -2
    out = bot.populate_exit_trend(df.copy(), {"pair": "ETH/USDC:USDC"})
    assert out.iloc[-1]["exit_short"] == 1
    assert out.iloc[-1]["exit_long"] == 0
    assert out.iloc[-1]["exit_tag"] == "mm_bid"


def test_target_inventory_steps_one_unit_toward_the_owned_side():
    """adjust_trade_position moves inventory one unit at a time in the direction
    of whichever side this leg is resting -- unit jumps, as eq. 10.2 assumes."""
    bot = make_bot()
    bot.role = "long"
    bot.can_short = False
    bot._inventory_level = lambda pair: 2

    bot._my_quote_side = lambda pair: "bid"
    assert bot._target_inventory_units("ETH/USDC:USDC") == 3
    bot._my_quote_side = lambda pair: "ask"
    assert bot._target_inventory_units("ETH/USDC:USDC") == 1
    # No side owned this cycle (e.g. the peer holds the only live side).
    bot._my_quote_side = lambda pair: None
    assert bot._target_inventory_units("ETH/USDC:USDC") is None


def test_no_peer_heartbeat_means_no_owned_side():
    """Fail closed: without a peer we cannot know net inventory, and 'assume
    flat' would silently mis-price every quote."""
    bot = make_bot()
    bot.role = "long"
    bot._net_inventory = lambda pair: (None, "peer_inventory_missing")
    assert bot._my_quote_side("ETH/USDC:USDC") is None
    assert bot._target_inventory_units("ETH/USDC:USDC") is None


def test_role_and_can_short_must_agree():
    """The role picks which side is quoted, can_short picks which exposure is a
    fault. If they disagree the guards contradict each other, so fail closed."""
    bot = make_bot()
    bot.trading_enabled = True
    bot._mm_config = lambda: {"role": "short", "can_short": False}
    bot._apply_runtime_safety_config()
    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "role_and_can_short_disagree"
def test_fixtures_do_not_invent_exchange_methods_freqtrade_lacks():
    """Guard against the fiction that hid the kill-switch bug.

    A fixture defining cancel_all_orders made every kill-switch test pass while
    the production path cancelled nothing. Any fixture standing in for the
    freqtrade exchange handle must expose only methods the real class has.
    """
    exchange = DummyExchange()
    for name in ABSENT_FROM_REAL_FREQTRADE_EXCHANGE:
        assert not hasattr(exchange, name), (
            f"DummyExchange defines {name}(), which freqtrade.exchange.Exchange "
            f"does not. Tests built on it assert against an API that does not exist."
        )
    # The strategy reads open orders off this attribute too; real freqtrade has
    # no such attribute either.
    assert not hasattr(exchange, "open_orders")


def test_kill_switch_cannot_cancel_resting_orders_on_real_freqtrade():
    """Documents a real limitation of the retiring strategy, rather than hiding it.

    With an exchange handle that has only the methods freqtrade actually
    provides, the cancel path finds no way to enumerate or cancel open orders and
    reports no_open_order_source. The trading flag still flips and the flatten
    request is still emitted, so the bot stops quoting -- but any order already
    resting on the book stays there until it fills or times out.

    The two-sided engine tracks its own resting order ids and cancels them
    through the Hyperliquid SDK, which is why it does not inherit this.
    """
    bot = make_bot()
    bot.trading_enabled = True
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    bot._trigger_kill_switch("manual_test", {"pair": "ETH/USDC:USDC"})

    assert not bot.trading_enabled
    payload = events[0][1]
    assert payload["cancel_method"] == "no_open_order_source"
    assert payload["cancel_open_orders_requested"] == 0
    assert bot._cancel_open_order_requests == 0


def test_kill_switch_suppresses_the_exit_signal_that_would_unwind_inventory():
    """The other half of the stranding problem, pinned so the engine must fix it.

    _trigger_kill_switch sets trading_enabled = False, and populate_exit_trend
    returns no signal while trading is disabled -- so a position open at the
    moment of a kill switch cannot be unwound by the bot at all. Only the -75%
    stoploss or a manual force-exit closes it.
    """
    import pandas as pd

    bot = make_bot()
    bot.trading_enabled = True
    DummyTrade._open_trades = [DummyTrade(amount=0.01, is_short=False)]
    frame = pd.DataFrame({"close": [100.0, 100.0], "date": [1, 2]})

    enabled = bot.populate_exit_trend(frame.copy(), {"pair": "ETH/USDC:USDC"})
    assert int(enabled["exit_long"].iloc[-1]) == 1, "quotes the ask while holding"

    bot._trigger_kill_switch("manual_test", {"pair": "ETH/USDC:USDC"})
    disabled = bot.populate_exit_trend(frame.copy(), {"pair": "ETH/USDC:USDC"})
    assert int(disabled["exit_long"].iloc[-1]) == 0, "inventory is stranded"
