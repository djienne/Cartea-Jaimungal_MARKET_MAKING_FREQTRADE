from __future__ import annotations

import sys
import types
import inspect
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

    def __init__(self, amount=0.0, is_short=False, trade_id=1, open_rate=100.0):
        self.amount = amount
        self.is_short = is_short
        self.id = trade_id
        self.open_rate = open_rate

    @classmethod
    def get_trades(cls, **kwargs):
        return list(cls._open_trades)


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


class DummyExchange:
    def __init__(self):
        self.markets = {
            "ETH/USDC:USDC": {
                "precision": {"amount": 2, "price": 2},
                "limits": {"amount": {"min": 0.01}, "cost": {"min": 1.0}},
            }
        }
        self.positions = []
        self.cancelled_pair = None

    def amount_to_precision(self, pair, amount):
        return f"{float(amount):.2f}"

    def price_to_precision(self, pair, price):
        return f"{float(price):.2f}"

    def fetch_positions(self, pairs=None):
        return list(self.positions)

    def cancel_all_orders(self, pair):
        self.cancelled_pair = pair


def make_bot() -> Market_Making:
    DummyTrade._open_trades = []
    bot = Market_Making()
    bot.dp = DummyDP()
    bot.exchange = DummyExchange()
    bot.config = {"dry_run": True}
    bot.debug_json_log = False
    bot.trading_enabled = False
    bot.post_only_verified = False
    bot.param_update_status_path = ""
    bot.hjb_cache = {
        "q_grid": np.array([-1, 0, 1]),
        "delta_plus": np.array([np.inf, 0.5, 0.4]),
        "delta_minus": np.array([0.4, 0.5, np.inf]),
    }
    bot._hjb_last_refresh_dt = datetime.now(timezone.utc)
    bot.kappas = {
        "ETH": {
            "schema_version": 2,
            "status": "ok",
            "kappa+": 2.0,
            "kappa-": 2.0,
            "n_points_plus": 3,
            "n_points_minus": 3,
            "r2_plus": 0.5,
            "r2_minus": 0.5,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    }
    bot.epsilons = {
        "ETH": {
            "schema_version": 2,
            "status": "ok",
            "epsilon+": 0.0,
            "epsilon-": 0.0,
            "n_buy_events": 3,
            "n_sell_events": 3,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    }
    bot.lambdas = {
        "ETH": {
            "schema_version": 2,
            "status": "ok",
            "lambda+": 0.1,
            "lambda-": 0.1,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    }
    bot._market_data_fresh = lambda symbol, max_age_seconds=None: (True, "ok")
    return bot


def test_trading_disabled_clears_entry_signal():
    bot = make_bot()
    df = pd.DataFrame({"close": [100.0, 101.0]})

    out = bot.populate_entry_trend(df, {"pair": "ETH/USDC:USDC"})

    assert out["enter_long"].sum() == 0


def test_custom_entry_price_signature_includes_trade_argument():
    params = list(inspect.signature(Market_Making.custom_entry_price).parameters)

    assert params[:7] == ["self", "pair", "trade", "current_time", "proposed_rate", "entry_tag", "side"]


def test_strategy_time_in_force_uses_runtime_supported_research_mode():
    assert Market_Making.order_time_in_force == {"entry": "GTC", "exit": "GTC"}
    assert Market_Making.post_only_verified is False


def test_dry_run_config_can_enable_research_mode_without_post_only():
    bot = make_bot()
    bot.config = {
        "dry_run": True,
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


def test_live_config_cannot_enable_without_post_only_verification():
    bot = make_bot()
    bot.config = {
        "dry_run": False,
        "market_making": {"trading_enabled": True, "post_only_verified": False},
    }

    bot._apply_runtime_safety_config()

    assert bot.trading_enabled is False
    assert bot.fail_closed_reason == "post_only_not_verified"


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
    ) == 40.0
    assert bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        4000.0,
        proposed_stake=25.0,
        min_stake=None,
        max_stake=1000.0,
        entry_tag="mm_bid",
        side="long",
    ) == 25.0


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


def test_param_snapshot_status_must_be_ok():
    bot = make_bot()
    bot.kappas["ETH"]["status"] = "seeded_unverified"

    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "param_status_not_ok")


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

    for exit_reason in ("stop_loss", "stoploss_on_exchange", "liquidation", "emergency_exit"):
        assert bot.confirm_trade_exit(
            "ETH/USDC:USDC",
            DummyTrade(amount=0.01),
            "limit",
            0.01,
            90.0,
            "GTC",
            exit_reason,
            datetime.now(timezone.utc),
        )


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
    assert payload["liquidity"] == "maker"
    assert payload["liquidity_normalized"] == "maker"
    assert payload["is_maker_fill"] is True
    assert payload["is_taker_fill"] is False
    assert payload["raw_order_side"] == "buy"
    assert payload["quote_side"] == "bid"
    assert payload["actual_fee_paid"] == 0.015
    assert payload["actual_fee_rate"] == 0.00015
    assert payload["expected_fee_rate"] == bot.fees_maker_HL
    assert payload["expected_tif"] == "Alo"
    assert payload["tif_canonical"] == "post_only"
    assert payload["expected_tif_canonical"] == "post_only"
    assert payload["tif_matches_expected"] is True
    assert bot._maker_fill_count == 1


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


def test_exchange_position_takes_priority_over_open_trade_count():
    bot = make_bot()
    bot.exchange.positions = [{"symbol": "ETH/USDC:USDC", "side": "long", "contracts": "0.02"}]
    DummyTrade._open_trades = [DummyTrade(amount=0.01)]

    assert bot._inventory_level("ETH/USDC:USDC") == 2


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
    assert bot.exchange.cancelled_pair == "ETH/USDC:USDC"
    assert [event for event, _ in events] == ["kill_switch", "entry_rejected"]
    assert events[0][1] == {
        "reason": "unexpected_short_position",
        "pair": "ETH/USDC:USDC",
        "signed_base_position": -0.02,
    }
    assert events[1][1]["reason"] == "unexpected_short_position"
    assert events[1][1]["signed_base_position"] == -0.02
    assert events[1][1]["q"] == 0


def test_daily_loss_triggers_kill_switch():
    bot = make_bot()
    bot.trading_enabled = True
    bot.max_daily_loss_usdc = 20
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


def test_adjust_exit_price_reprices_passive_ask():
    bot = make_bot()
    bot.trading_enabled = True
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
        exit_tag="mm_ask",
        side="long",
    )

    assert adjusted > 100.0


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


def test_quote_decision_logs_freshness_age_fields():
    bot = make_bot()
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
    assert payload["param_age_seconds"] is not None
    assert payload["collector_age_seconds"] == 12.4
    assert payload["book_age_ms"] == 180.0
    assert payload["params_fresh"] is True
    assert payload["collector_fresh"] is True
    assert payload["book_fresh"] is True
