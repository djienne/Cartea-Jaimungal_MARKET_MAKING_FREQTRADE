# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
import sys
import threading
from periodic_test_runner import schedule_tests
from pandas import DataFrame
from functools import reduce
import json
import logging
from pathlib import Path
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade, Order
from datetime import datetime, timedelta
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

def find_upwards(filename: str, start: Path, max_up: int = 10) -> Path:
    p = start.resolve()
    for _ in range(max_up + 1):
        candidate = p / filename
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError(f"Could not find {filename} from {start}")

def load_configs(start_dir: Path | None = None, max_up: int = 10):
    if start_dir is None:
        try:
            start_dir = Path(__file__).resolve().parent
        except NameError:  # e.g., interactive
            start_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()

    kappa = json.loads((find_upwards("kappa.json", start_dir, max_up)).read_text(encoding="utf-8"))
    epsilon = json.loads((find_upwards("epsilon.json", start_dir, max_up)).read_text(encoding="utf-8"))
    return kappa, epsilon

class Market_Making(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    #min_spread = 0.3/100.0 # minimum spread to avoid insane backtest results

    fees_maker_HL = 0.0150/100.0

    # Can this strategy go short?
    can_short: bool = False
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = True
    position_adjustment_enable: bool = False
    max_entry_position_adjustment = 0

    # exists ASAP
    minimal_roi = {
        "0": -1
    }

    kappas = None
    epsilons = None

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.75

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        "emergency_exit": "limit",
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        logger.info('Running calculation of Espilon and Kappa')
        pairs = self.dp.current_whitelist()
        if len(pairs)!=1:
            sys.exit()
        symbol = pairs[0].replace("/USDC:USDC","")
        logger.info(f"Current symbol: {symbol}")
        schedule_tests(run_once=True)
        self.kappas, self.epsilons = load_configs()
        logger.info(self.kappas)
        logger.info(self.epsilons)

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop). For each loop, it will run populate_indicators on all pairs.
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        logger.info('Running calculation of Espilon and Kappa')
        t = threading.Thread(target=schedule_tests, daemon=True)
        t.start()
        self.kappas, self.epsilons = load_configs()
        logger.info(self.kappas)
        logger.info(self.epsilons)

    def informative_pairs(self):
        """
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        if self.kappas is not None and self.epsilons is not None:
            dataframe.loc[:, 'enter_long'] = 1
        else:
            dataframe.loc[:, 'enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
        
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:

        orderbook = self.dp.orderbook(pair, maximum=1)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = proposed_rate
        symbol = pair.replace("/USDC:USDC","")
        kappa_m = self.kappas[symbol]['kappa-']
        epsilon_m = self.epsilons[symbol]['epsilon-']
        delta_m = (1.0/kappa_m + epsilon_m + self.fees_maker_HL*mid_price*2.0)
        returned_rate = mid_price - delta_m
        logger.info(f"Calculated bid: {returned_rate:.5f}  (mid_price -{delta_m/mid_price*100:.5f}%)")
        return returned_rate

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:
        orderbook = self.dp.orderbook(pair, maximum=1)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = proposed_rate

        symbol = pair.replace("/USDC:USDC","")
        kappa_p = self.kappas[symbol]['kappa+']
        epsilon_p = self.epsilons[symbol]['epsilon+']
        delta_p = (1.0/kappa_p + epsilon_p + self.fees_maker_HL*mid_price*2.0)
        returned_rate = mid_price + delta_p

        logger.info(f"Calculated ask: {returned_rate:.5f}  (mid_price +{delta_p/mid_price*100:.5f}%)")

        return returned_rate

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: str, side: str, **kwargs) -> float:
        orderbook = self.dp.orderbook(pair, maximum=1)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = proposed_rate

        symbol = pair.replace("/USDC:USDC","")
        kappa_m = self.kappas[symbol]['kappa-']
        epsilon_m = self.kappas[symbol]['epsilon-']
        returned_rate = mid_price - (1.0/kappa_m + epsilon_m)
        
        return returned_rate

    # @property
    # def protections(self):
    #     return [
    #         {
    #             "method": "MaxDrawdown",
    #             "lookback_period": 10080,  # 1 week
    #             "trade_limit": 0,  # Evaluate all trades since the bot started
    #             "stop_duration_candles": 10000000,  # Stop trading indefinitely
    #             "max_allowed_drawdown": 0.05  # Maximum drawdown of 5% before stopping
    #         },
    #     ]
