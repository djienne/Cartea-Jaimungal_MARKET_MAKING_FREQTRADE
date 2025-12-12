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
import importlib.util
from typing import Any
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
from importlib import import_module

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
    lambda_params = {}
    try:
        lambda_params = json.loads((find_upwards("lambda.json", start_dir, max_up)).read_text(encoding="utf-8"))
    except Exception:
        lambda_params = {}
    return kappa, epsilon, lambda_params


def load_hjb_solver():
    """
    Import HJB module.

    In the Freqtrade container this repo mounts `./scripts` to `/freqtrade/scripts`,
    but that path is not guaranteed to be on `sys.path`. We first try a normal
    import (`import hjb`), then fall back to loading `scripts/hjb.py` by file path.
    """
    try:
        return import_module("hjb")
    except Exception:
        pass

    try:
        start_dir = Path(__file__).resolve().parent
    except NameError:  # e.g., interactive
        start_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()

    for rel in ("scripts/hjb.py", "hjb.py"):
        try:
            hjb_path = find_upwards(rel, start_dir, max_up=10)
        except Exception:
            hjb_path = None
        if not hjb_path:
            continue
        try:
            spec = importlib.util.spec_from_file_location("hjb", str(hjb_path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue

    return None

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
    lambdas = None
    hjb_cache = None
    hjb_solver = load_hjb_solver()
    _hjb_import_error_logged: bool = False
    _hjb_generation: int = 0
    _hjb_last_refresh_ts: str | None = None

    debug_json_log: bool = True
    debug_json_log_filename: str = "mm_debug.jsonl"
    debug_json_log_max_bytes: int = 2_000_000
    _debug_log_lock = threading.Lock()

    _data_checked_and_available: bool = False # Added to track initial data presence
    # Conservative stoploss at 75% loss
    stoploss = -0.75

    # Trailing stoploss disabled
    trailing_stop = False

    # Use 1-minute timeframe for high-frequency market making
    timeframe = '1m'

    # No startup candles required
    startup_candle_count: int = 0

    # HJB risk settings (aligned with fq_market_making_introduction.ipynb)
    hjb_alpha = 0.0   # terminal inventory penalty
    hjb_phi = 0.0     # running inventory penalty
    hjb_q_max = 3     # inventory grid radius
    hjb_horizon_seconds = 60.0  # horizon in seconds for matrix exponential (λ is trades/sec)
    use_asymmetric_kappa = True  # always use backward-Euler asymmetric-κ solver (kappa+ != kappa-)

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
        self.kappas, self.epsilons, self.lambdas = load_configs()
        logger.info(f'Loaded kappa values: {self.kappas}')
        logger.info(f'Loaded epsilon values: {self.epsilons}')
        logger.info(f'Loaded lambda values: {self.lambdas}')
        if not self.use_asymmetric_kappa:
            logger.warning("Forcing use_asymmetric_kappa=True (always use kappa+/kappa-).")
            self.use_asymmetric_kappa = True
        if self.hjb_solver is None:
            self.hjb_solver = load_hjb_solver()
        logger.info(f"HJB module loaded: {self.hjb_solver is not None}")
        self._refresh_hjb(pairs[0])

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of each bot iteration to refresh market making parameters.
        
        :param current_time: Current datetime
        :param **kwargs: Additional arguments
        """
        logger.info('Refreshing market making parameters')
        t = threading.Thread(target=schedule_tests, daemon=True)
        t.start()
        self.kappas, self.epsilons, self.lambdas = load_configs()
        logger.info(f'Updated kappa values: {self.kappas}')
        logger.info(f'Updated epsilon values: {self.epsilons}')
        logger.info(f'Updated lambda values: {self.lambdas}')
        if not self.use_asymmetric_kappa:
            logger.warning("Forcing use_asymmetric_kappa=True (always use kappa+/kappa-).")
            self.use_asymmetric_kappa = True
        pairs = self.dp.current_whitelist()
        if pairs:
            self._refresh_hjb(pairs[0])

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

    def _check_data_files_exist(self) -> bool:
        if self._data_checked_and_available:
            return True

        # Path inside the Docker container where HL_data is mounted
        data_path_base = Path("/freqtrade/scripts/HL_data/ETH")
        
        found_data = False
        # Check in orderbooks, prices, and trades subdirectories
        for sub_dir in ["orderbooks", "prices", "trades"]:
            target_dir = data_path_base / sub_dir
            if target_dir.is_dir() and any(target_dir.glob("*.parquet")):
                found_data = True
                break
        
        if found_data:
            logger.info("Initial data files found in HL_data/ETH/**. Enabling trading.")
            self._data_checked_and_available = True
        else:
            logger.warning("No initial data files found in HL_data/ETH/**. Trading is currently disabled.")
        return found_data

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enter long positions when market making parameters are loaded AND initial data is available.
        Entry price will be calculated dynamically in custom_entry_price.
        """
        if (self.kappas is not None and self.epsilons is not None and self.hjb_cache is not None and
                self._check_data_files_exist()):
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
    
    def _refresh_hjb(self, pair: str) -> None:
        """
        Compute HJB surface using latest λ/κ/ε (asymmetric κ+/κ-).
        Keeps the last known-good cache if refresh fails.
        """
        symbol = pair.replace("/USDC:USDC", "")
        try:
            kappa_p = float(self.kappas[symbol]["kappa+"])
            kappa_m = float(self.kappas[symbol]["kappa-"])
            epsilon_p = float(self.epsilons[symbol]["epsilon+"])
            epsilon_m = float(self.epsilons[symbol]["epsilon-"])
            lambda_p = float(self.lambdas.get(symbol, {}).get("lambda+", 0.0)) if isinstance(self.lambdas, dict) else 0.0
            lambda_m = float(self.lambdas.get(symbol, {}).get("lambda-", 0.0)) if isinstance(self.lambdas, dict) else 0.0
        except Exception as e:
            logger.warning(f"HJB refresh skipped (missing/invalid params for {symbol}): {e}")
            self._debug_log_event(
                "hjb_refresh_skipped",
                {"pair": pair, "symbol": symbol, "reason": "missing_or_invalid_params", "error": str(e)},
            )
            return

        hjb_mod = self.hjb_solver
        if hjb_mod is None:
            hjb_mod = load_hjb_solver()
            self.hjb_solver = hjb_mod
        if hjb_mod is None:
            if not self._hjb_import_error_logged:
                logger.error("HJB module not available (could not import / load scripts/hjb.py). Trading will stay disabled.")
                self._debug_log_event(
                    "hjb_unavailable",
                    {"pair": pair, "symbol": symbol, "reason": "module_load_failed"},
                )
                self._hjb_import_error_logged = True
            return
        solver_name = "compute_h_asymmetric" if self.use_asymmetric_kappa else "compute_h_symmetric"
        solver = getattr(hjb_mod, solver_name, None)
        if solver is None:
            logger.error(f"HJB solver function not found: {solver_name}")
            self._debug_log_event(
                "hjb_unavailable",
                {"pair": pair, "symbol": symbol, "reason": "solver_not_found", "solver": solver_name},
            )
            return

        try:
            hjb_res = solver(
                lambda_plus=lambda_p,
                lambda_minus=lambda_m,
                epsilon_plus=epsilon_p,
                epsilon_minus=epsilon_m,
                kappa_plus=kappa_p,
                kappa_minus=kappa_m,
                alpha=self.hjb_alpha,
                phi=self.hjb_phi,
                T_seconds=self.hjb_horizon_seconds,
                q_max=self.hjb_q_max,
            )
            self.hjb_cache = hjb_res
            self._hjb_generation += 1
            self._hjb_last_refresh_ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            self._debug_log_event(
                "hjb_refresh",
                {
                    "pair": pair,
                    "symbol": symbol,
                    "solver": solver_name,
                    "inputs": {
                        "lambda_plus": lambda_p,
                        "lambda_minus": lambda_m,
                        "kappa_plus": kappa_p,
                        "kappa_minus": kappa_m,
                        "epsilon_plus": epsilon_p,
                        "epsilon_minus": epsilon_m,
                        "alpha": float(self.hjb_alpha),
                        "phi": float(self.hjb_phi),
                        "T_seconds": float(self.hjb_horizon_seconds),
                        "q_max": int(self.hjb_q_max),
                    },
                    "hjb_generation": self._hjb_generation,
                    "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                    "hjb": self._hjb_snapshot(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to compute HJB surfaces: {e}")
            self._debug_log_event(
                "hjb_refresh_failed",
                {"pair": pair, "symbol": symbol, "solver": solver_name, "error": str(e)},
            )
            # Keep last known-good cache (no static fallback).
            return

    def _inventory_level(self, pair: str) -> int:
        """
        Approximate inventory level as number of open trades for the pair.
        Limits to configured HJB grid.
        """
        try:
            open_trades = Trade.get_trades(is_open=True, pair=pair)
            q = len(open_trades)
        except Exception:
            q = 0
        q = max(-self.hjb_q_max, min(self.hjb_q_max, q))
        return q

    def _select_delta(self, side: str, q: int) -> float | None:
        """
        Select delta+/- from precomputed HJB grid for given inventory level.
        Returns None when HJB cache is unavailable.
        """
        if self.hjb_cache:
            q_grid = self.hjb_cache["q_grid"]
            if q < q_grid[0]:
                idx = 0
            elif q > q_grid[-1]:
                idx = -1
            else:
                idx = int(np.argmin(np.abs(q_grid - q)))
            if side == 'bid':
                return float(self.hjb_cache["delta_minus"][idx])
            else:
                return float(self.hjb_cache["delta_plus"][idx])

        return None

    def _log_spread(self, side: str, mid_price: float, delta: float, source: str) -> None:
        """
        Log the applied spread in basis points off mid, with its origin.
        """
        if mid_price <= 0:
            bps = float("nan")
        else:
            bps = (delta / mid_price) * 10_000.0
        logger.info(
            f"[spread] side={side} bps={bps:.2f} abs={delta:.6f} mid={mid_price:.6f} source={source}"
        )

    def _debug_log_path(self) -> Path:
        try:
            base = Path(__file__).resolve().parent.parent  # user_data
        except Exception:
            base = Path.cwd()
        return base / "logs" / self.debug_json_log_filename

    def _rotate_debug_log_if_needed(self, path: Path) -> None:
        max_bytes = int(getattr(self, "debug_json_log_max_bytes", 0) or 0)
        if max_bytes <= 0:
            return
        try:
            if path.exists() and path.stat().st_size > max_bytes:
                backup = path.with_suffix(path.suffix + ".1")
                try:
                    backup.unlink(missing_ok=True)
                except TypeError:
                    if backup.exists():
                        backup.unlink()
                path.replace(backup)
        except Exception:
            return

    def _debug_log_event(self, event: str, payload: dict[str, Any]) -> None:
        if not getattr(self, "debug_json_log", False):
            return
        record = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "event": event,
            **payload,
        }
        path = self._debug_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._debug_log_lock:
                self._rotate_debug_log_if_needed(path)
                with path.open("a", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, separators=(",", ":"), default=str)
                    f.write("\n")
        except Exception:
            return

    def _params_snapshot(self, symbol: str) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"symbol": symbol}
        try:
            if isinstance(self.kappas, dict):
                snapshot["kappa_plus"] = float(self.kappas[symbol]["kappa+"])
                snapshot["kappa_minus"] = float(self.kappas[symbol]["kappa-"])
        except Exception:
            pass
        try:
            if isinstance(self.epsilons, dict):
                snapshot["epsilon_plus"] = float(self.epsilons[symbol]["epsilon+"])
                snapshot["epsilon_minus"] = float(self.epsilons[symbol]["epsilon-"])
        except Exception:
            pass
        try:
            if isinstance(self.lambdas, dict):
                snapshot["lambda_plus"] = float(self.lambdas.get(symbol, {}).get("lambda+", 0.0))
                snapshot["lambda_minus"] = float(self.lambdas.get(symbol, {}).get("lambda-", 0.0))
        except Exception:
            pass

        snapshot["fees_maker_HL"] = float(self.fees_maker_HL)
        snapshot["hjb_alpha"] = float(self.hjb_alpha)
        snapshot["hjb_phi"] = float(self.hjb_phi)
        snapshot["hjb_q_max"] = int(self.hjb_q_max)
        snapshot["hjb_horizon_seconds"] = float(self.hjb_horizon_seconds)
        snapshot["use_asymmetric_kappa"] = bool(self.use_asymmetric_kappa)

        return snapshot

    def _hjb_snapshot(self) -> dict[str, Any] | None:
        cache = self.hjb_cache
        if not cache:
            return None

        def to_list(val: Any) -> Any:
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return [float(x) for x in val.tolist()]
            if isinstance(val, (list, tuple)):
                return [float(x) for x in val]
            if isinstance(val, (np.floating, np.integer)):
                return val.item()
            return val

        return {
            "method": cache.get("method", "matrix_exponential"),
            "q_grid": to_list(cache.get("q_grid")),
            "delta_plus": to_list(cache.get("delta_plus")),
            "delta_minus": to_list(cache.get("delta_minus")),
            "kappa_sym": to_list(cache.get("kappa_sym")),
            "kappa_plus": to_list(cache.get("kappa_plus")),
            "kappa_minus": to_list(cache.get("kappa_minus")),
            "dt": to_list(cache.get("dt")),
            "n_steps": to_list(cache.get("n_steps")),
        }

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:

        if side == 'short':
            return None

        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC", "")
        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        q_level = self._inventory_level(pair)
        delta_m = self._select_delta('bid', q_level)
        if delta_m is None:
            logger.warning("No HJB delta available for bid; skipping entry pricing.")
            self._debug_log_event(
                "quote_skipped",
                {
                    "action": "entry",
                    "side": "bid",
                    "pair": pair,
                    "symbol": symbol,
                    "q": q_level,
                    "mid": float(mid_price),
                    "proposed_rate": float(proposed_rate),
                    "reason": "no_hjb_delta",
                    "hjb_generation": int(self._hjb_generation),
                    "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                    "params": self._params_snapshot(symbol),
                },
            )
            return None
        delta_source = "hjb_grid"

        # Add maker fee cushion (price units)
        delta_model = float(delta_m)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        returned_rate = mid_price - delta_total
        self._log_spread("bid", mid_price, delta_total, delta_source)
        logger.info(f"Calculated bid: {returned_rate:.5f}")

        self._debug_log_event(
            "quote",
            {
                "action": "entry",
                "side": "bid",
                "pair": pair,
                "symbol": symbol,
                "q": q_level,
                "mid": float(mid_price),
                "proposed_rate": float(proposed_rate),
                "price": float(returned_rate),
                "delta_model": delta_model,
                "fee_cushion": fee_cushion,
                "delta_total": delta_total,
                "bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None,
                "hjb_generation": int(self._hjb_generation),
                "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                "params": self._params_snapshot(symbol),
            },
        )
        return returned_rate

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:
        
        if trade.is_short:
            return None
            
        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC", "")
        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        q_level = self._inventory_level(pair)
        delta_p = self._select_delta('ask', q_level)
        if delta_p is None:
            logger.error("No HJB delta available for ask; using proposed_rate for exit pricing.")
            self._debug_log_event(
                "quote_fallback",
                {
                    "action": "exit",
                    "side": "ask",
                    "pair": pair,
                    "symbol": symbol,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    "q": q_level,
                    "mid": float(mid_price),
                    "proposed_rate": float(proposed_rate),
                    "price": float(proposed_rate),
                    "reason": "no_hjb_delta",
                    "hjb_generation": int(self._hjb_generation),
                    "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                    "params": self._params_snapshot(symbol),
                },
            )
            return proposed_rate
        delta_source = "hjb_grid"

        delta_model = float(delta_p)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        returned_rate = mid_price + delta_total

        self._log_spread("ask", mid_price, delta_total, delta_source)
        logger.info(f"Calculated ask: {returned_rate:.5f}")

        self._debug_log_event(
            "quote",
            {
                "action": "exit",
                "side": "ask",
                "pair": pair,
                "symbol": symbol,
                "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                "open_rate": float(trade.open_rate) if getattr(trade, "open_rate", None) is not None else None,
                "current_profit": float(current_profit) if current_profit is not None else None,
                "exit_tag": exit_tag,
                "q": q_level,
                "mid": float(mid_price),
                "proposed_rate": float(proposed_rate),
                "price": float(returned_rate),
                "delta_model": delta_model,
                "fee_cushion": fee_cushion,
                "delta_total": delta_total,
                "bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None,
                "hjb_generation": int(self._hjb_generation),
                "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                "params": self._params_snapshot(symbol),
            },
        )

        return returned_rate

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: str, side: str, **kwargs) -> float:
        
        if trade.is_short:
            return None
            
        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC", "")

        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        q_level = self._inventory_level(pair)
        delta_m = self._select_delta('bid', q_level)
        if delta_m is None:
            logger.warning("No HJB delta available for bid adjust; keeping current order rate.")
            self._debug_log_event(
                "quote_fallback",
                {
                    "action": "adjust_entry",
                    "side": "bid",
                    "pair": pair,
                    "symbol": symbol,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    "q": q_level,
                    "mid": float(mid_price),
                    "proposed_rate": float(proposed_rate),
                    "current_order_rate": float(current_order_rate),
                    "price": float(current_order_rate),
                    "reason": "no_hjb_delta",
                    "hjb_generation": int(self._hjb_generation),
                    "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                    "params": self._params_snapshot(symbol),
                },
            )
            return current_order_rate
        delta_source = "hjb_grid"
        delta_model = float(delta_m)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        returned_rate = mid_price - delta_total

        self._log_spread("bid_adjust", mid_price, delta_total, delta_source)

        self._debug_log_event(
            "quote",
            {
                "action": "adjust_entry",
                "side": "bid",
                "pair": pair,
                "symbol": symbol,
                "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                "q": q_level,
                "mid": float(mid_price),
                "proposed_rate": float(proposed_rate),
                "current_order_rate": float(current_order_rate),
                "price": float(returned_rate),
                "delta_model": delta_model,
                "fee_cushion": fee_cushion,
                "delta_total": delta_total,
                "bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None,
                "hjb_generation": int(self._hjb_generation),
                "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
                "params": self._params_snapshot(symbol),
            },
        )
        
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
