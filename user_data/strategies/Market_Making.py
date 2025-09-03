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

    # Market maker fees (1.5%)
    fees_maker_HL = 0.0150/100.0

    # Strategy configuration
    can_short: bool = False
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = False
    position_adjustment_enable: bool = False
    max_entry_position_adjustment = 0

    # Exit immediately when conditions are met
    minimal_roi = {
        "0": -1
    }

    # Configuration parameters loaded from external files
    kappas = None
    epsilons = None

    # Conservative stoploss at 75% loss
    stoploss = -0.75

    # Trailing stoploss disabled
    trailing_stop = False

    # Use 1-minute timeframe for high-frequency market making
    timeframe = '1m'

    # No startup candles required
    startup_candle_count: int = 0

    # Use limit orders for all operations to ensure maker fees
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        "emergency_exit": "limit",
        'stoploss_on_exchange': False
    }

    # Good-til-cancelled orders
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        logger.info('Loading market making parameters (Epsilon and Kappa)')
        pairs = self.dp.current_whitelist()
        if len(pairs) != 1:
            logger.error('Strategy requires exactly one trading pair')
            sys.exit()
        symbol = pairs[0].replace("/USDC:USDC", "")
        logger.info(f"Trading symbol: {symbol}")
        schedule_tests(run_once=True)
        self.kappas, self.epsilons = load_configs()
        logger.info(f'Loaded kappa values: {self.kappas}')
        logger.info(f'Loaded epsilon values: {self.epsilons}')

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of each bot iteration to refresh market making parameters.
        
        :param current_time: Current datetime
        :param **kwargs: Additional arguments
        """
        logger.info('Refreshing market making parameters')
        t = threading.Thread(target=schedule_tests, daemon=True)
        t.start()
        self.kappas, self.epsilons = load_configs()
        logger.info(f'Updated kappa values: {self.kappas}')
        logger.info(f'Updated epsilon values: {self.epsilons}')

    def informative_pairs(self):
        """
        No additional informative pairs required for this market making strategy.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No technical indicators needed for pure market making strategy.
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enter long positions when market making parameters are loaded.
        Entry price will be calculated dynamically in custom_entry_price.
        """
        if self.kappas is not None and self.epsilons is not None:
            dataframe.loc[:, 'enter_long'] = 1
        else:
            dataframe.loc[:, 'enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Never exit based on indicators - exits handled by custom_exit_price.
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
    
    def get_mid_price(self, pair: str, fallback_rate: float) -> float:
        """
        Calculate mid price from first order book bid and ask.
        
        :param pair: Trading pair
        :param fallback_rate: Rate to use if orderbook is not available
        :return: Mid price
        """
        orderbook = self.dp.orderbook(pair, maximum=1)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            return (best_bid + best_ask) / 2
        else:
            return fallback_rate
        
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:

        if side == 'short':
            return None

        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC", "")
        kappa_m = self.kappas[symbol]['kappa-']
        epsilon_m = self.epsilons[symbol]['epsilon-']
        # Calculate bid offset: inventory risk + market impact + trading fees
        delta_m = (1.0/kappa_m + epsilon_m + self.fees_maker_HL*mid_price*2.0)
        returned_rate = mid_price - delta_m
        logger.info(f"Calculated bid: {returned_rate:.5f} (mid_price -{delta_m/mid_price*100:.5f}%)")
        return returned_rate

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:
        
        if trade.is_short:
            return None
            
        mid_price = self.get_mid_price(pair, proposed_rate)

        symbol = pair.replace("/USDC:USDC", "")
        kappa_p = self.kappas[symbol]['kappa+']
        epsilon_p = self.epsilons[symbol]['epsilon+']
        # Calculate ask offset: inventory risk + market impact + trading fees
        delta_p = (1.0/kappa_p + epsilon_p + self.fees_maker_HL*mid_price*2.0)
        returned_rate = mid_price + delta_p

        logger.info(f"Calculated ask: {returned_rate:.5f} (mid_price +{delta_p/mid_price*100:.5f}%)")

        return returned_rate

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: str, side: str, **kwargs) -> float:
        
        if trade.is_short:
            return None
            
        mid_price = self.get_mid_price(pair, proposed_rate)

        symbol = pair.replace("/USDC:USDC", "")
        kappa_m = self.kappas[symbol]['kappa-']
        epsilon_m = self.epsilons[symbol]['epsilon-']  # Fixed: was using kappas instead of epsilons
        # Adjust bid price without fees (fees already paid on initial order)
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
