# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import math
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
from datetime import datetime, timedelta, timezone
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

    # Hyperliquid base perp maker fee: 0.015% = 1.5 bps.
    fees_maker_HL = 0.0150/100.0
    trading_enabled: bool = False
    fail_closed_reason: str = "initial_safety_lock"
    post_only_verified: bool = False

    # Strategy configuration
    can_short: bool = False
    use_exit_signal: bool = True
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = False
    position_adjustment_enable: bool = False
    max_entry_position_adjustment = 0

    # Disable time-based ROI exits; passive exits are controlled explicitly.
    minimal_roi = {
        "0": 10
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
    _hjb_last_refresh_dt: datetime | None = None

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
    hjb_alpha = 0.001   # terminal inventory penalty
    hjb_phi = 0.0001    # running inventory penalty
    hjb_q_max = 3     # inventory grid radius
    hjb_horizon_seconds = 60.0  # horizon in seconds for matrix exponential (λ is trades/sec)
    use_asymmetric_kappa = True  # always use backward-Euler asymmetric-κ solver (kappa+ != kappa-)

    inventory_unit_base = 0.01
    max_abs_inventory_units = 3
    max_param_age_seconds = 90
    max_collector_age_seconds = 90
    max_book_age_seconds = 5
    collector_timestamp_cache_seconds = 90
    book_snapshot_cache_ms = 500
    max_toxicity = 1.5
    max_daily_loss_usdc = 20
    max_consecutive_losses = 10
    max_post_only_reject_rate = 0.80
    min_post_only_reject_samples = 10
    kill_on_taker_fill = True
    min_kappa_fit_points = 2
    min_kappa_r2 = 0.0
    min_epsilon_events = 1
    param_update_status_path: str | None = None
    _param_update_lock = threading.Lock()
    _param_update_running: bool = False
    _last_param_update: datetime | None = None
    _last_health_log: datetime | None = None
    _quote_decisions_count: int = 0
    _post_only_rejects: int = 0
    _maker_fill_count: int = 0
    _taker_fill_count: int = 0
    _daily_realized_pnl_usdc: float = 0.0
    _daily_risk_date: str | None = None
    _consecutive_losses: int = 0

    # Use limit orders for all operations to ensure maker fees
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        "emergency_exit": "limit",
        'stoploss_on_exchange': False
    }

    # Freqtrade 2025.4 rejects Hyperliquid PO at runtime. Keep GTC for
    # research/dry-run startup only; post_only_verified gates any live use.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _as_utc(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _parse_utc_timestamp(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return self._as_utc(value)
        try:
            if isinstance(value, (int, float)):
                ts = float(value)
                if ts > 1_000_000_000_000:
                    ts /= 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            text = str(value).strip()
            if not text:
                return None
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return self._as_utc(parsed)
        except Exception:
            return None

    def _symbol_from_pair(self, pair: str) -> str:
        return pair.split("/", 1)[0].split(":", 1)[0]

    def _mm_config(self) -> dict[str, Any]:
        config = getattr(self, "config", {}) or {}
        mm_config = config.get("market_making", {}) if isinstance(config, dict) else {}
        return mm_config if isinstance(mm_config, dict) else {}

    def _param_config_dir(self) -> Path | None:
        raw_dir = self._mm_config().get("param_dir")
        if not raw_dir:
            return None
        return Path(str(raw_dir))

    def _apply_runtime_safety_config(self) -> None:
        mm_config = self._mm_config()
        if not mm_config:
            return

        if "post_only_verified" in mm_config:
            self.post_only_verified = bool(mm_config.get("post_only_verified"))
        if "param_update_status_path" in mm_config:
            self.param_update_status_path = str(mm_config.get("param_update_status_path") or "")
        elif self._param_config_dir() is not None:
            self.param_update_status_path = str(self._param_config_dir() / "param_update_status.json")
        for numeric_key in (
            "max_param_age_seconds",
            "max_collector_age_seconds",
            "max_book_age_seconds",
            "max_toxicity",
        ):
            if numeric_key in mm_config:
                try:
                    setattr(self, numeric_key, float(mm_config[numeric_key]))
                except Exception:
                    self._debug_log_event(
                        "runtime_config_rejected",
                        {"key": numeric_key, "value": mm_config.get(numeric_key), "reason": "non_numeric"},
                    )

        if "trading_enabled" not in mm_config:
            return

        requested_enabled = bool(mm_config.get("trading_enabled"))
        is_dry_run = bool(getattr(self, "config", {}).get("dry_run", True))
        if requested_enabled and not is_dry_run and not self.post_only_verified:
            self.trading_enabled = False
            self.fail_closed_reason = "post_only_not_verified"
            self._debug_log_event(
                "trading_enable_rejected",
                {"reason": "post_only_not_verified", "dry_run": is_dry_run},
            )
            return

        self.trading_enabled = requested_enabled
        self.fail_closed_reason = "none" if self.trading_enabled else "initial_safety_lock"

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
        symbol = self._symbol_from_pair(pairs[0])
        logger.info(f"Trading symbol: {symbol}")
        self._apply_runtime_safety_config()
        self.kappas, self.epsilons, self.lambdas = load_configs(start_dir=self._param_config_dir())
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
        self._log_health(pairs[0], self._now_utc())

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of each bot iteration to refresh market making parameters.
        
        :param current_time: Current datetime
        :param **kwargs: Additional arguments
        """
        pairs = self.dp.current_whitelist()
        pair = pairs[0] if pairs else None
        now = self._as_utc(current_time) or self._now_utc()
        if pair:
            self._log_health(pair, now)

        if bool(self._mm_config().get("disable_param_refresh", False)):
            self._debug_log_event("param_update_skipped", {"reason": "disabled_by_config"})
            return

        if self._param_update_running:
            self._debug_log_event("param_update_skipped", {"reason": "estimator_running"})
            return
        if self._last_param_update and (now - self._last_param_update) < timedelta(seconds=60):
            return

        acquired = self._param_update_lock.acquire(blocking=False)
        if not acquired:
            self._debug_log_event("param_update_skipped", {"reason": "estimator_lock_busy"})
            return

        self._param_update_running = True
        try:
            logger.info('Refreshing market making parameters')
            schedule_tests(run_once=True, crypto=self._symbol_from_pair(pair) if pair else "ETH")
            new_kappas, new_epsilons, new_lambdas = load_configs(start_dir=self._param_config_dir())
            self._last_param_update = now
            old_params = (self.kappas, self.epsilons, self.lambdas)
            self.kappas, self.epsilons, self.lambdas = new_kappas, new_epsilons, new_lambdas
            if pair:
                ok, reason = self._params_are_valid(pair)
                if not ok:
                    self.kappas, self.epsilons, self.lambdas = old_params
                    self._debug_log_event("param_update_rejected", {"pair": pair, "reason": reason})
                    logger.warning(f"Parameter refresh rejected ({reason}); keeping last known values.")
                    return
            logger.info(f'Updated kappa values: {self.kappas}')
            logger.info(f'Updated epsilon values: {self.epsilons}')
            logger.info(f'Updated lambda values: {self.lambdas}')
            if not self.use_asymmetric_kappa:
                logger.warning("Forcing use_asymmetric_kappa=True (always use kappa+/kappa-).")
                self.use_asymmetric_kappa = True
            if pair:
                self._refresh_hjb(pair)
        except Exception as exc:
            self._debug_log_event("param_update_failed", {"error": str(exc)})
            logger.warning(f"Parameter refresh failed; keeping last known values: {exc}")
        finally:
            self._param_update_running = False
            self._param_update_lock.release()

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

    def _collector_symbol_dir(self, symbol: str) -> Path:
        candidates = [
            Path("/freqtrade/scripts/HL_data") / symbol,
            Path(__file__).resolve().parents[2] / "scripts" / "HL_data" / symbol,
            Path.cwd() / "scripts" / "HL_data" / symbol,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _latest_collector_timestamp(self, symbol: str) -> datetime | None:
        now = self._now_utc()
        cache = getattr(self, "_collector_timestamp_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._collector_timestamp_cache = cache
        cached = cache.get(symbol)
        if isinstance(cached, dict):
            checked_at = self._as_utc(cached.get("checked_at"))
            if (
                checked_at is not None
                and (now - checked_at).total_seconds() <= float(self.collector_timestamp_cache_seconds)
            ):
                return self._as_utc(cached.get("latest_ts"))

        base = self._collector_symbol_dir(symbol)
        newest: datetime | None = None
        for sub_dir in ("orderbooks", "prices", "trades"):
            target_dir = base / sub_dir
            if not target_dir.is_dir():
                continue
            for shard in target_dir.glob("*.parquet"):
                try:
                    ts = datetime.fromtimestamp(shard.stat().st_mtime, tz=timezone.utc)
                except Exception:
                    continue
                if newest is None or ts > newest:
                    newest = ts
        cache[symbol] = {"checked_at": now, "latest_ts": newest}
        return newest

    def _collector_age_seconds(self, symbol: str, now: datetime | None = None) -> float | None:
        latest_ts = self._latest_collector_timestamp(symbol)
        if latest_ts is None:
            return None
        reference = self._as_utc(now) or self._now_utc()
        return max(0.0, (reference - latest_ts).total_seconds())

    def _market_data_fresh(self, symbol: str, max_age_seconds: int | None = None) -> tuple[bool, str]:
        max_age = int(max_age_seconds or getattr(self, "max_collector_age_seconds", 90))
        age = self._collector_age_seconds(symbol)
        if age is None:
            return False, "no_collector_data"

        if age > max_age:
            return False, f"collector_data_stale_{age:.1f}s"

        return True, "ok"

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Emit a one-candle passive bid signal only when the model is ready.
        """
        dataframe.loc[:, 'enter_long'] = 0
        if dataframe.empty:
            return dataframe
        pair = metadata.get("pair", "")
        if not self.trading_enabled:
            return dataframe
        if not self._model_ready(pair):
            return dataframe
        if not self._inventory_allows_bid(pair):
            return dataframe
        dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
        dataframe.loc[dataframe.index[-1], 'enter_tag'] = "mm_bid"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Emit a one-candle passive ask signal only when inventory can be unwound.
        """
        dataframe.loc[:, 'exit_long'] = 0
        if dataframe.empty:
            return dataframe
        pair = metadata.get("pair", "")
        if not self.trading_enabled:
            return dataframe
        if not self._model_ready(pair):
            return dataframe
        if not self._inventory_allows_ask(pair):
            return dataframe
        dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
        dataframe.loc[dataframe.index[-1], 'exit_tag'] = "mm_ask"
        return dataframe
    
    def get_mid_price(self, pair: str, fallback_rate: float) -> float:
        """
        Calculate mid price from first order book bid and ask.
        
        :param pair: Trading pair
        :param fallback_rate: Rate to use if orderbook is not available
        :return: Mid price
        """
        orderbook = self.dp.orderbook(pair, maximum=1)
        try:
            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                best_bid = float(orderbook['bids'][0][0])
                best_ask = float(orderbook['asks'][0][0])
                if best_bid > 0 and best_ask > 0 and best_bid < best_ask:
                    return (best_bid + best_ask) / 2
        except Exception:
            pass
        return fallback_rate
    
    def _refresh_hjb(self, pair: str) -> None:
        """
        Compute HJB surface using latest λ/κ/ε (asymmetric κ+/κ-).
        Keeps the last known-good cache if refresh fails.
        """
        symbol = self._symbol_from_pair(pair)
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
            self._hjb_last_refresh_dt = self._now_utc()
            self._hjb_last_refresh_ts = self._hjb_last_refresh_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
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
        Long-only inventory unit, based on signed base exposure.
        """
        signed_base = self._signed_base_position(pair)
        if self._reject_unexpected_short_position(pair, signed_base):
            return 0
        unit = max(float(self.inventory_unit_base), 1e-12)
        q = int(round(max(0.0, signed_base) / unit))
        q = max(0, min(int(self.hjb_q_max), q))
        return q

    def _reject_unexpected_short_position(self, pair: str, signed_base: float | None = None) -> bool:
        if signed_base is None:
            signed_base = self._signed_base_position(pair)
        if float(signed_base) >= 0 or self.can_short:
            return False
        payload = {"pair": pair, "signed_base_position": float(signed_base)}
        if self.fail_closed_reason != "unexpected_short_position":
            self._trigger_kill_switch("unexpected_short_position", payload)
        return True

    def _signed_base_position(self, pair: str) -> float:
        exchange_position = self._signed_base_position_from_exchange(pair)
        if exchange_position is not None:
            return float(exchange_position)

        signed_base = 0.0
        try:
            open_trades = Trade.get_trades(is_open=True, pair=pair)
            for trade in open_trades:
                amount = abs(float(getattr(trade, "amount", 0.0) or 0.0))
                if bool(getattr(trade, "is_short", False)):
                    signed_base -= amount
                else:
                    signed_base += amount
        except Exception:
            pass
        if signed_base == 0.0:
            wallet_position = self._signed_base_position_from_wallet(pair)
            if wallet_position is not None:
                return float(wallet_position)
        return float(signed_base)

    def _signed_base_position_from_exchange(self, pair: str) -> float | None:
        exchange = getattr(self, "exchange", None)
        if exchange is None or not hasattr(exchange, "fetch_positions"):
            return None
        try:
            positions = exchange.fetch_positions([pair])
        except Exception:
            try:
                positions = exchange.fetch_positions()
            except Exception:
                return None

        symbol = self._symbol_from_pair(pair)
        for position in positions or []:
            if not isinstance(position, dict):
                continue
            info = position.get("info", {}) if isinstance(position.get("info", {}), dict) else {}
            identifiers = [
                position.get("symbol"),
                position.get("pair"),
                info.get("coin") if isinstance(info, dict) else None,
                info.get("symbol") if isinstance(info, dict) else None,
            ]
            if not any(value and (str(value) == pair or str(value).split("/", 1)[0] == symbol) for value in identifiers):
                continue

            signed_size = None
            for key in ("szi", "position", "contracts", "contractSize", "size", "amount"):
                value = info.get(key) if isinstance(info, dict) and key in info else position.get(key)
                if value is None:
                    continue
                try:
                    signed_size = float(value)
                    break
                except Exception:
                    continue
            if signed_size is None:
                return None

            side = str(
                position.get("side")
                or position.get("positionSide")
                or (info.get("side") if isinstance(info, dict) else "")
                or ""
            ).lower()
            if side == "short":
                return -abs(float(signed_size))
            if side == "long":
                return abs(float(signed_size))
            return float(signed_size)
        return None

    def _signed_base_position_from_wallet(self, pair: str) -> float | None:
        is_dry_run = bool(getattr(self, "config", {}).get("dry_run", True))
        wallets = getattr(self, "wallets", None)
        if not is_dry_run or wallets is None:
            return None
        symbol = self._symbol_from_pair(pair)
        for method_name in ("get_total", "get_free"):
            method = getattr(wallets, method_name, None)
            if not callable(method):
                continue
            try:
                value = method(symbol)
                if value is not None:
                    return max(0.0, float(value))
            except Exception:
                continue
        return None

    def _inventory_snapshot(self, pair: str) -> dict[str, Any]:
        signed_base = self._signed_base_position(pair)
        q = self._inventory_level(pair)
        return {
            "signed_base_position": float(signed_base),
            "inventory_unit_base": float(self.inventory_unit_base),
            "q": int(q),
            "max_abs_inventory_units": int(self.max_abs_inventory_units),
        }

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

    def _hjb_is_stale(self, current_time: datetime) -> bool:
        refreshed_at = self._as_utc(self._hjb_last_refresh_dt)
        now = self._as_utc(current_time) or self._now_utc()
        if refreshed_at is None:
            return True
        return (now - refreshed_at).total_seconds() > float(self.max_param_age_seconds)

    def _hjb_age_seconds(self, now: datetime | None = None) -> float | None:
        refreshed_at = self._as_utc(self._hjb_last_refresh_dt)
        if refreshed_at is None:
            return None
        reference = self._as_utc(now) or self._now_utc()
        return max(0.0, (reference - refreshed_at).total_seconds())

    def _combined_params(self, symbol: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for source in (self.kappas, self.epsilons, self.lambdas):
            if isinstance(source, dict) and isinstance(source.get(symbol), dict):
                payload.update(source[symbol])
        return payload

    def _param_source_entries(self, symbol: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for source in (self.kappas, self.epsilons, self.lambdas):
            if isinstance(source, dict) and isinstance(source.get(symbol), dict):
                entries.append(source[symbol])
        return entries

    def _param_age_seconds(self, symbol: str, now: datetime | None = None) -> float | None:
        ages: list[float] = []
        reference = self._as_utc(now) or self._now_utc()
        for entry in self._param_source_entries(symbol):
            generated_at = self._parse_utc_timestamp(entry.get("generated_at"))
            if generated_at is None:
                continue
            ages.append(max(0.0, (reference - generated_at).total_seconds()))
        return max(ages) if ages else None

    def _param_update_status(self) -> tuple[bool, str]:
        configured_path = getattr(self, "param_update_status_path", None)
        if configured_path is None:
            status_path = Path(__file__).resolve().parent / "param_update_status.json"
        elif str(configured_path) == "":
            return True, "ok"
        else:
            status_path = Path(configured_path)
        if not status_path.exists():
            return True, "ok"
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            return False, "param_status_unreadable"
        status = str(data.get("status", "")).lower()
        if status == "running":
            return False, "estimator_running"
        if status in {"failed", "interrupted"}:
            return False, "param_update_failed"
        return True, "ok"

    def _params_are_valid(self, pair: str) -> tuple[bool, str]:
        symbol = self._symbol_from_pair(pair)
        ok, reason = self._param_update_status()
        if not ok:
            return False, reason

        params = self._combined_params(symbol)
        schema_versions: list[int] = []
        source_entries = self._param_source_entries(symbol)
        for entry in source_entries:
            version = entry.get("schema_version")
            if version is not None:
                try:
                    schema_versions.append(int(version))
                except Exception:
                    return False, "param_schema_unsupported"
        if len(schema_versions) < 3 or any(version != 2 for version in schema_versions):
            return False, "param_schema_unsupported"
        if len(source_entries) < 3:
            return False, "param_schema_unsupported"

        for entry in source_entries:
            if str(entry.get("status", "")).lower() != "ok":
                return False, "param_status_not_ok"
            generated_at = entry.get("generated_at")
            if not generated_at:
                return False, "missing_param_timestamp"
            parsed = self._parse_utc_timestamp(generated_at)
            if parsed is None:
                return False, "invalid_param_timestamp"
            age = (self._now_utc() - parsed).total_seconds()
            if age > float(self.max_param_age_seconds):
                return False, "stale_params"

        required = ("kappa+", "kappa-", "lambda+", "lambda-", "epsilon+", "epsilon-")
        for key in required:
            if key not in params:
                return False, f"missing_{key}"
            try:
                value = float(params[key])
            except Exception:
                return False, f"nonfinite_{key}"
            if not np.isfinite(value):
                return False, f"nonfinite_{key}"

        if float(params["kappa+"]) <= 0 or float(params["kappa-"]) <= 0:
            return False, "invalid_kappa"
        if float(params["lambda+"]) < 0 or float(params["lambda-"]) < 0:
            return False, "invalid_lambda"
        if float(params["epsilon+"]) < 0 or float(params["epsilon-"]) < 0:
            return False, "invalid_epsilon"

        toxicity = max(
            float(params["kappa+"]) * float(params["epsilon+"]),
            float(params["kappa-"]) * float(params["epsilon-"]),
        )
        if toxicity > float(self.max_toxicity):
            return False, "toxicity_too_high"

        for key in ("n_points_plus", "n_points_minus"):
            if key not in params or int(params.get(key) or 0) < int(self.min_kappa_fit_points):
                return False, "insufficient_kappa_diagnostics"

        for key in ("r2_plus", "r2_minus"):
            if key not in params:
                return False, "insufficient_kappa_diagnostics"
            try:
                r2 = float(params[key])
            except Exception:
                return False, "insufficient_kappa_diagnostics"
            if not np.isfinite(r2) or r2 < float(self.min_kappa_r2):
                return False, "insufficient_kappa_diagnostics"

        buy_events = int(params.get("n_buy_events") or 0)
        sell_events = int(params.get("n_sell_events") or 0)
        if buy_events < int(self.min_epsilon_events) or sell_events < int(self.min_epsilon_events):
            return False, "insufficient_epsilon_diagnostics"

        return True, "ok"

    def _book_snapshot(self, pair: str) -> tuple[dict[str, float] | None, str]:
        now = self._now_utc()
        cache = getattr(self, "_book_snapshot_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._book_snapshot_cache = cache
        cached = cache.get(pair)
        if isinstance(cached, dict):
            checked_at = self._as_utc(cached.get("checked_at"))
            if (
                checked_at is not None
                and (now - checked_at).total_seconds() * 1000.0 <= float(self.book_snapshot_cache_ms)
            ):
                return cached.get("snapshot"), str(cached.get("reason", "ok"))

        try:
            ob = self.dp.orderbook(pair, maximum=1)
            if not ob or not ob.get("bids") or not ob.get("asks"):
                cache[pair] = {"checked_at": now, "snapshot": None, "reason": "empty_orderbook", "book_ts": None}
                return None, "empty_orderbook"
            best_bid = float(ob["bids"][0][0])
            best_ask = float(ob["asks"][0][0])
            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                cache[pair] = {
                    "checked_at": now,
                    "snapshot": None,
                    "reason": "crossed_or_invalid_book",
                    "book_ts": None,
                }
                return None, "crossed_or_invalid_book"
            book_ts = None
            if isinstance(ob, dict):
                for key in ("timestamp", "ts", "datetime"):
                    book_ts = self._parse_utc_timestamp(ob.get(key))
                    if book_ts is not None:
                        break
            snapshot = {"best_bid": best_bid, "best_ask": best_ask, "mid": (best_bid + best_ask) / 2.0}
            cache[pair] = {"checked_at": now, "snapshot": snapshot, "reason": "ok", "book_ts": book_ts}
            return snapshot, "ok"
        except Exception as exc:
            cache[pair] = {"checked_at": now, "snapshot": None, "reason": f"orderbook_error:{exc}", "book_ts": None}
            return None, f"orderbook_error:{exc}"

    def _book_age_ms(self, pair: str, current_time: datetime | None = None) -> float | None:
        self._book_snapshot(pair)
        cache = getattr(self, "_book_snapshot_cache", {})
        cached = cache.get(pair) if isinstance(cache, dict) else None
        if not isinstance(cached, dict):
            return None
        reference = self._as_utc(current_time) or self._now_utc()
        timestamp = self._as_utc(cached.get("book_ts"))
        if timestamp is None:
            timestamp = self._as_utc(cached.get("checked_at"))
        if timestamp is None:
            return None
        return max(0.0, (reference - timestamp).total_seconds() * 1000.0)

    def _book_is_fresh(self, pair: str, current_time: datetime) -> tuple[bool, str]:
        snapshot, reason = self._book_snapshot(pair)
        if snapshot is None:
            return False, reason
        age_ms = self._book_age_ms(pair, current_time)
        if age_ms is not None and age_ms > float(self.max_book_age_seconds) * 1000.0:
            return False, "stale_orderbook"
        return True, "ok"

    def _maker_safe(self, pair: str, quote_side: str, rate: float) -> tuple[bool, str]:
        snapshot, reason = self._book_snapshot(pair)
        if snapshot is None:
            return False, reason
        if not np.isfinite(float(rate)) or float(rate) <= 0:
            return False, "invalid_rate"
        if quote_side == "bid" and float(rate) >= snapshot["best_ask"]:
            return False, "bid_crosses_ask"
        if quote_side == "ask" and float(rate) <= snapshot["best_bid"]:
            return False, "ask_crosses_bid"
        return True, "ok"

    def _model_ready(self, pair: str) -> bool:
        ok, _ = self._params_are_valid(pair)
        if not ok or self.hjb_cache is None:
            return False
        ok, _ = self._market_data_fresh(self._symbol_from_pair(pair))
        return ok

    def _inventory_allows_bid(self, pair: str) -> bool:
        if self._reject_unexpected_short_position(pair):
            return False
        q = self._inventory_level(pair)
        return q < min(int(self.hjb_q_max), int(self.max_abs_inventory_units))

    def _inventory_allows_ask(self, pair: str) -> bool:
        if self._reject_unexpected_short_position(pair):
            return False
        return self._inventory_level(pair) > 0

    def _quote_state_valid(self, pair: str, side: str, rate: float, current_time: datetime) -> tuple[bool, str]:
        if not self.trading_enabled:
            return False, self.fail_closed_reason or "trading_disabled"
        if self.hjb_cache is None:
            return False, "no_hjb_cache"
        if not np.isfinite(float(rate)) or float(rate) <= 0:
            return False, "invalid_rate"
        if self._hjb_is_stale(current_time):
            return False, "stale_hjb"

        ok, reason = self._params_are_valid(pair)
        if not ok:
            return False, reason

        ok, reason = self._book_is_fresh(pair, current_time)
        if not ok:
            return False, "stale_orderbook" if reason == "empty_orderbook" else reason

        symbol = self._symbol_from_pair(pair)
        ok, reason = self._market_data_fresh(symbol)
        if not ok:
            return False, "stale_collector_data" if reason.startswith("collector_data_stale") else reason

        signed_base = self._signed_base_position(pair)
        if self._reject_unexpected_short_position(pair, signed_base):
            return False, "unexpected_short_position"

        q = self._inventory_level(pair)
        delta = self._select_delta("bid" if side in {"long", "bid"} else "ask", q)
        if delta is None or not np.isfinite(float(delta)):
            return False, "boundary_side_disabled"

        if side in {"long", "bid"} and not self._inventory_allows_bid(pair):
            return False, "position_limit_reached"
        if side in {"ask", "exit"} and not self._inventory_allows_ask(pair):
            return False, "position_limit_reached"

        is_dry_run = bool(getattr(self, "config", {}).get("dry_run", True))
        if not self.post_only_verified and self.trading_enabled and not is_dry_run:
            return False, "post_only_not_verified"

        return True, "ok"

    def _price_tick(self, pair: str) -> float | None:
        try:
            market = self.exchange.markets.get(pair, {}) if getattr(self, "exchange", None) else {}
            precision = market.get("precision", {}).get("price")
            if isinstance(precision, int):
                return 10 ** (-precision)
            if precision:
                return float(precision)
        except Exception:
            pass
        return None

    def _round_quote_price(self, pair: str, quote_side: str, raw_rate: float) -> float:
        tick = self._price_tick(pair)
        if tick and tick > 0:
            if quote_side == "bid":
                rounded = math.floor(float(raw_rate) / tick) * tick
            else:
                rounded = math.ceil(float(raw_rate) / tick) * tick
        else:
            rounded = float(raw_rate)
        try:
            if getattr(self, "exchange", None) and hasattr(self.exchange, "price_to_precision"):
                rounded = float(self.exchange.price_to_precision(pair, rounded))
        except Exception:
            pass
        return float(rounded)

    def _amount_step(self, pair: str) -> float | None:
        try:
            market = self.exchange.markets.get(pair, {}) if getattr(self, "exchange", None) else {}
            precision = market.get("precision", {}).get("amount")
            if isinstance(precision, int):
                return 10 ** (-precision)
            if precision:
                return float(precision)
        except Exception:
            pass
        return None

    def _round_quote_amount(self, pair: str, amount: float) -> float:
        amount = float(amount)
        step = self._amount_step(pair)
        if step and step > 0:
            rounded = math.floor(amount / step) * step
        else:
            rounded = amount
        try:
            if getattr(self, "exchange", None) and hasattr(self.exchange, "amount_to_precision"):
                exchange_rounded = float(self.exchange.amount_to_precision(pair, rounded))
                if step and step > 0 and exchange_rounded > amount:
                    exchange_rounded = max(0.0, exchange_rounded - step)
                rounded = min(exchange_rounded, amount)
        except Exception:
            pass
        return max(0.0, float(rounded))

    def _amount_lot_safe(self, pair: str, amount: float, rate: float) -> tuple[bool, str, float]:
        try:
            amount = float(amount)
            rate = float(rate)
        except Exception:
            return False, "invalid_amount", 0.0
        if not np.isfinite(amount) or amount <= 0:
            return False, "invalid_amount", 0.0
        if not np.isfinite(rate) or rate <= 0:
            return False, "invalid_rate", 0.0

        rounded = self._round_quote_amount(pair, amount)
        if rounded <= 0:
            return False, "invalid_amount", rounded
        if abs(rounded - amount) > max(1e-12, amount * 1e-9):
            return False, "amount_not_lot_safe", rounded

        try:
            market = self.exchange.markets.get(pair, {}) if getattr(self, "exchange", None) else {}
            limits = market.get("limits", {})
            min_amount = (limits.get("amount") or {}).get("min")
            min_cost = (limits.get("cost") or {}).get("min")
            if min_amount is not None and rounded < float(min_amount):
                return False, "amount_below_min", rounded
            if min_cost is not None and rounded * rate < float(min_cost):
                return False, "cost_below_min", rounded
        except Exception:
            pass

        return True, "ok", rounded

    def _trigger_kill_switch(self, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.trading_enabled = False
        self.fail_closed_reason = reason
        try:
            pair = (payload or {}).get("pair")
            if pair and getattr(self, "exchange", None) and hasattr(self.exchange, "cancel_all_orders"):
                self.exchange.cancel_all_orders(pair)
        except Exception as exc:
            payload = {**(payload or {}), "cancel_error": str(exc)}
        self._debug_log_event("kill_switch", {"reason": reason, **(payload or {})})

    def _record_quote_decision(self, pair: str, decision: str, reason: str) -> None:
        self._quote_decisions_count = int(getattr(self, "_quote_decisions_count", 0)) + 1
        post_only_reasons = {
            "bid_crosses_ask",
            "ask_crosses_bid",
            "crossed_or_invalid_book",
            "empty_orderbook",
            "post_only_not_verified",
            "not_post_only_supported",
        }
        if decision == "reject" and reason in post_only_reasons:
            self._post_only_rejects = int(getattr(self, "_post_only_rejects", 0)) + 1

        attempts = int(getattr(self, "_quote_decisions_count", 0))
        rejects = int(getattr(self, "_post_only_rejects", 0))
        if (
            self.trading_enabled
            and attempts >= int(self.min_post_only_reject_samples)
            and rejects / max(attempts, 1) > float(self.max_post_only_reject_rate)
        ):
            self._trigger_kill_switch(
                "post_only_reject_rate_exceeded",
                {"pair": pair, "quote_decisions": attempts, "post_only_rejects": rejects},
            )

    def _reset_daily_risk_if_needed(self, current_time: datetime) -> None:
        now = self._as_utc(current_time) or self._now_utc()
        day = now.date().isoformat()
        if getattr(self, "_daily_risk_date", None) != day:
            self._daily_risk_date = day
            self._daily_realized_pnl_usdc = 0.0
            self._consecutive_losses = 0

    def _extract_realized_pnl_usdc(self, trade: Trade, order: Order) -> float | None:
        for source in (order, trade):
            for attr in ("realized_pnl", "realized_profit", "close_profit_abs", "profit_abs"):
                value = getattr(source, attr, None)
                if value is None:
                    continue
                try:
                    value = float(value)
                except Exception:
                    continue
                if np.isfinite(value):
                    return value
        return None

    def _finite_float_or_none(self, value: Any) -> float | None:
        try:
            value_float = float(value)
        except Exception:
            return None
        return value_float if np.isfinite(value_float) else None

    def _normalize_liquidity(self, liquidity: Any) -> str:
        if liquidity is None:
            return "unknown"
        text = str(liquidity).strip().lower()
        if text in {"maker", "m", "add_liquidity", "added_liquidity", "post_only"}:
            return "maker"
        if text in {"taker", "t", "remove_liquidity", "removed_liquidity"}:
            return "taker"
        return text or "unknown"

    def _quote_side_from_order_side(self, side: Any) -> str | None:
        if side is None:
            return None
        text = str(side).strip().lower()
        if text in {"buy", "long", "bid", "entry", "open_long", "close_short"}:
            return "bid"
        if text in {"sell", "short", "ask", "exit", "open_short", "close_long"}:
            return "ask"
        return text or None

    def _canonical_tif(self, tif: Any) -> str | None:
        if tif is None:
            return None
        text = str(tif).strip().lower().replace("-", "_")
        if text in {"alo", "po", "post_only", "postonly"}:
            return "post_only"
        if text in {"gtc", "good_til_cancelled", "good_till_cancelled"}:
            return "gtc"
        if text in {"ioc", "immediate_or_cancel"}:
            return "ioc"
        return text or None

    def _expected_time_in_force(self, quote_side: str | None) -> str | None:
        configured = None
        if quote_side == "bid":
            configured = self.order_time_in_force.get("entry")
        elif quote_side == "ask":
            configured = self.order_time_in_force.get("exit")
        return "Alo" if self.post_only_verified else configured

    def _extract_order_fee_paid(self, order: Order, price: float | None, amount: float | None) -> float | None:
        fee = getattr(order, "fee", None)
        if isinstance(fee, dict):
            for key in ("cost", "fee_cost", "paid", "amount"):
                value = self._finite_float_or_none(fee.get(key))
                if value is not None:
                    return value
            rate = self._finite_float_or_none(fee.get("rate"))
            if rate is not None and price is not None and amount is not None:
                return abs(float(price) * float(amount) * rate)
        else:
            value = self._finite_float_or_none(fee)
            if value is not None:
                return value

        for attr in ("fee_cost", "ft_fee_cost", "cost"):
            value = self._finite_float_or_none(getattr(order, attr, None))
            if value is not None:
                return value
        return None

    def _extract_order_fee_rate(
        self,
        order: Order,
        fee_paid: float | None,
        price: float | None,
        amount: float | None,
    ) -> float | None:
        fee = getattr(order, "fee", None)
        if isinstance(fee, dict):
            value = self._finite_float_or_none(fee.get("rate"))
            if value is not None:
                return value

        for attr in ("fee_rate", "ft_fee_rate"):
            value = self._finite_float_or_none(getattr(order, attr, None))
            if value is not None:
                return value

        if fee_paid is not None and price is not None and amount is not None:
            notional = abs(float(price) * float(amount))
            if notional > 0:
                return abs(float(fee_paid)) / notional
        return None

    def _record_realized_pnl(self, pair: str, pnl_usdc: float, current_time: datetime, payload: dict[str, Any]) -> None:
        self._reset_daily_risk_if_needed(current_time)
        event_id = payload.get("order_id")
        if event_id is not None:
            seen = getattr(self, "_seen_pnl_event_ids", None)
            if seen is None:
                seen = set()
                self._seen_pnl_event_ids = seen
            if event_id in seen:
                return
            seen.add(event_id)

        self._daily_realized_pnl_usdc = float(getattr(self, "_daily_realized_pnl_usdc", 0.0)) + float(pnl_usdc)
        if pnl_usdc < 0:
            self._consecutive_losses = int(getattr(self, "_consecutive_losses", 0)) + 1
        elif pnl_usdc > 0:
            self._consecutive_losses = 0

        if self.trading_enabled and self._daily_realized_pnl_usdc <= -abs(float(self.max_daily_loss_usdc)):
            self._trigger_kill_switch(
                "drawdown_limit_reached",
                {"pair": pair, "daily_realized_pnl": self._daily_realized_pnl_usdc, **payload},
            )
        if self.trading_enabled and self._consecutive_losses >= int(self.max_consecutive_losses):
            self._trigger_kill_switch(
                "consecutive_losses_limit_reached",
                {"pair": pair, "consecutive_losses": self._consecutive_losses, **payload},
            )

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

    def _log_quote_decision(
        self,
        *,
        pair: str,
        symbol: str,
        side: str,
        action: str,
        decision: str,
        reason: str,
        mid_price: float,
        proposed_rate: float,
        raw_price: float | None = None,
        rounded_price: float | None = None,
        delta_model: float | None = None,
        fee_cushion: float | None = None,
        delta_total: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        now = self._now_utc()
        snapshot, book_snapshot_reason = self._book_snapshot(pair)
        params_ok, params_reason = self._params_are_valid(pair)
        collector_ok, collector_reason = self._market_data_fresh(symbol)
        book_ok, book_reason = self._book_is_fresh(pair, now)
        payload = {
            "action": action,
            "pair": pair,
            "symbol": symbol,
            "side": side,
            "decision": decision,
            "reason": reason,
            "mid": float(mid_price) if mid_price is not None else None,
            "best_bid": snapshot.get("best_bid") if snapshot else None,
            "best_ask": snapshot.get("best_ask") if snapshot else None,
            "proposed_rate": float(proposed_rate) if proposed_rate is not None else None,
            "raw_price": float(raw_price) if raw_price is not None else None,
            "rounded_price": float(rounded_price) if rounded_price is not None else None,
            "delta_model": float(delta_model) if delta_model is not None else None,
            "fee_cushion": float(fee_cushion) if fee_cushion is not None else None,
            "delta_total": float(delta_total) if delta_total is not None else None,
            "hjb_generation": int(self._hjb_generation),
            "hjb_last_refresh_ts": self._hjb_last_refresh_ts,
            "hjb_age_seconds": self._hjb_age_seconds(now),
            "param_age_seconds": self._param_age_seconds(symbol, now),
            "collector_age_seconds": self._collector_age_seconds(symbol, now),
            "book_age_ms": self._book_age_ms(pair, now),
            "params_fresh": bool(params_ok),
            "params_fresh_reason": params_reason,
            "collector_fresh": bool(collector_ok),
            "collector_fresh_reason": collector_reason,
            "book_fresh": bool(book_ok),
            "book_fresh_reason": book_reason if snapshot else book_snapshot_reason,
            "post_only_verified": bool(self.post_only_verified),
            "params": self._params_snapshot(symbol),
            **self._inventory_snapshot(pair),
            **(extra or {}),
        }
        if action in {"entry", "exit", "adjust_entry", "adjust_exit"}:
            self._record_quote_decision(pair, decision, reason)
        self._debug_log_event("quote_decision", payload)

    def _log_health(self, pair: str, current_time: datetime) -> None:
        now = self._as_utc(current_time) or self._now_utc()
        if self._last_health_log and (now - self._last_health_log) < timedelta(seconds=60):
            return
        symbol = self._symbol_from_pair(pair)
        params_ok, params_reason = self._params_are_valid(pair)
        collector_ok, collector_reason = self._market_data_fresh(symbol)
        book_ok, book_reason = self._book_is_fresh(pair, now)
        hjb_fresh = self.hjb_cache is not None and not self._hjb_is_stale(now)
        self._debug_log_event(
            "health",
            {
                "trading_enabled": bool(self.trading_enabled),
                "fail_closed_reason": self.fail_closed_reason,
                "collector_fresh": collector_ok,
                "collector_fresh_reason": collector_reason,
                "collector_age_seconds": self._collector_age_seconds(symbol, now),
                "params_fresh": params_ok,
                "params_fresh_reason": params_reason,
                "param_age_seconds": self._param_age_seconds(symbol, now),
                "book_fresh": book_ok,
                "book_fresh_reason": book_reason,
                "book_age_ms": self._book_age_ms(pair, now),
                "hjb_fresh": hjb_fresh,
                "hjb_age_seconds": self._hjb_age_seconds(now),
                "post_only_verified": bool(self.post_only_verified),
                "open_orders": None,
                "maker_fills": int(getattr(self, "_maker_fill_count", 0)),
                "taker_fills": int(getattr(self, "_taker_fill_count", 0)),
                "post_only_rejects": int(getattr(self, "_post_only_rejects", 0)),
                "quote_decisions": int(getattr(self, "_quote_decisions_count", 0)),
                "realized_pnl": float(getattr(self, "_daily_realized_pnl_usdc", 0.0)),
                "unrealized_pnl": None,
                "consecutive_losses": int(getattr(self, "_consecutive_losses", 0)),
                **self._inventory_snapshot(pair),
            },
        )
        self._last_health_log = now

    def custom_entry_price(self, pair: str, trade: Trade | None, current_time: datetime, proposed_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:

        if side == 'short':
            return proposed_rate

        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = self._symbol_from_pair(pair)
        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        q_level = self._inventory_level(pair)
        delta_m = self._select_delta('bid', q_level)
        if delta_m is None or not np.isfinite(float(delta_m)):
            logger.warning("No HJB delta available for bid; skipping entry pricing.")
            self._log_quote_decision(
                pair=pair,
                symbol=symbol,
                side="bid",
                action="entry",
                decision="reject",
                reason="boundary_side_disabled" if delta_m is not None else "no_hjb_delta",
                mid_price=mid_price,
                proposed_rate=proposed_rate,
            )
            return proposed_rate
        delta_source = "hjb_grid"

        # Add maker fee cushion (price units)
        delta_model = float(delta_m)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        raw_rate = mid_price - delta_total
        returned_rate = self._round_quote_price(pair, "bid", raw_rate)
        self._log_spread("bid", mid_price, delta_total, delta_source)
        logger.info(f"Calculated bid: {returned_rate:.5f}")

        ok, reason = self._maker_safe(pair, "bid", returned_rate)
        self._log_quote_decision(
            pair=pair,
            symbol=symbol,
            side="bid",
            action="entry",
            decision="accept" if ok else "reject",
            reason=reason,
            mid_price=mid_price,
            proposed_rate=proposed_rate,
            raw_price=raw_rate,
            rounded_price=returned_rate,
            delta_model=delta_model,
            fee_cushion=fee_cushion,
            delta_total=delta_total,
            extra={"bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None},
        )
        return returned_rate

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if side == "short":
            return proposed_stake
        try:
            rate = float(current_rate)
            proposed = float(proposed_stake)
            maximum = float(max_stake) if max_stake is not None else proposed
        except Exception:
            return proposed_stake
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(proposed) or proposed <= 0:
            return proposed_stake

        one_unit_stake = float(self.inventory_unit_base) * rate
        stake = min(proposed, maximum, one_unit_stake)
        if min_stake is not None:
            try:
                stake = max(float(min_stake), stake)
            except Exception:
                pass
        stake = min(stake, maximum)
        self._debug_log_event(
            "stake_sized",
            {
                "pair": pair,
                "side": side,
                "entry_tag": entry_tag,
                "current_rate": rate,
                "proposed_stake": proposed,
                "returned_stake": stake,
                "inventory_unit_base": float(self.inventory_unit_base),
            },
        )
        return float(stake)

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:
        
        if trade.is_short:
            return proposed_rate
            
        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = self._symbol_from_pair(pair)
        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        q_level = self._inventory_level(pair)
        delta_p = self._select_delta('ask', q_level)
        if delta_p is None or not np.isfinite(float(delta_p)):
            logger.error("No HJB delta available for ask; using proposed_rate for exit pricing.")
            self._log_quote_decision(
                pair=pair,
                symbol=symbol,
                side="ask",
                action="exit",
                decision="reject",
                reason="boundary_side_disabled" if delta_p is not None else "no_hjb_delta",
                mid_price=mid_price,
                proposed_rate=proposed_rate,
                extra={"trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None},
            )
            return proposed_rate
        delta_source = "hjb_grid"

        delta_model = float(delta_p)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        raw_rate = mid_price + delta_total
        returned_rate = self._round_quote_price(pair, "ask", raw_rate)

        self._log_spread("ask", mid_price, delta_total, delta_source)
        logger.info(f"Calculated ask: {returned_rate:.5f}")

        ok, reason = self._maker_safe(pair, "ask", returned_rate)
        self._log_quote_decision(
            pair=pair,
            symbol=symbol,
            side="ask",
            action="exit",
            decision="accept" if ok else "reject",
            reason=reason,
            mid_price=mid_price,
            proposed_rate=proposed_rate,
            raw_price=raw_rate,
            rounded_price=returned_rate,
            delta_model=delta_model,
            fee_cushion=fee_cushion,
            delta_total=delta_total,
            extra={
                "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                "open_rate": float(trade.open_rate) if getattr(trade, "open_rate", None) is not None else None,
                "current_profit": float(current_profit) if current_profit is not None else None,
                "exit_tag": exit_tag,
                "bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None,
            },
        )

        return returned_rate

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        ok, reason = self._quote_state_valid(pair, "bid", rate, current_time)
        if not ok:
            self._debug_log_event(
                "entry_rejected",
                {"pair": pair, "reason": reason, "rate": float(rate), "side": side, **self._inventory_snapshot(pair)},
            )
            return False

        amount_ok, amount_reason, rounded_amount = self._amount_lot_safe(pair, amount, rate)
        if not amount_ok:
            self._debug_log_event(
                "entry_rejected",
                {
                    "pair": pair,
                    "reason": amount_reason,
                    "rate": float(rate),
                    "amount": float(amount),
                    "rounded_amount": float(rounded_amount),
                    "side": side,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        ok, reason = self._maker_safe(pair, "bid", rate)
        if not ok:
            self._record_quote_decision(pair, "reject", reason)
            self._debug_log_event(
                "entry_rejected",
                {"pair": pair, "reason": reason, "rate": float(rate), "side": side, **self._inventory_snapshot(pair)},
            )
            return False

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        if exit_reason in {"stop_loss", "stoploss_on_exchange", "liquidation", "emergency_exit"}:
            return True

        ok, reason = self._quote_state_valid(pair, "ask", rate, current_time)
        if not ok:
            self._debug_log_event(
                "exit_rejected",
                {
                    "pair": pair,
                    "reason": reason,
                    "rate": float(rate),
                    "exit_reason": exit_reason,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        amount_ok, amount_reason, rounded_amount = self._amount_lot_safe(pair, amount, rate)
        if not amount_ok:
            self._debug_log_event(
                "exit_rejected",
                {
                    "pair": pair,
                    "reason": amount_reason,
                    "rate": float(rate),
                    "amount": float(amount),
                    "rounded_amount": float(rounded_amount),
                    "exit_reason": exit_reason,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        ok, reason = self._maker_safe(pair, "ask", rate)
        if not ok:
            self._record_quote_decision(pair, "reject", reason)
            self._debug_log_event(
                "exit_rejected",
                {
                    "pair": pair,
                    "reason": reason,
                    "rate": float(rate),
                    "exit_reason": exit_reason,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        return True

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        raw_liquidity = (
            getattr(order, "liquidity", None)
            or getattr(order, "ft_liquidity", None)
            or getattr(order, "filled_liquidity", None)
            or "unknown"
        )
        liquidity = self._normalize_liquidity(raw_liquidity)
        order_type = getattr(order, "order_type", None) or getattr(order, "type", None)
        tif = getattr(order, "time_in_force", None) or getattr(order, "ft_time_in_force", None)
        raw_side = getattr(order, "ft_order_side", None) or getattr(order, "side", None)
        quote_side = self._quote_side_from_order_side(raw_side)
        price = getattr(order, "price", None) or getattr(order, "safe_price", None)
        amount = getattr(order, "amount", None) or getattr(order, "filled", None)
        price_float = self._finite_float_or_none(price)
        amount_float = self._finite_float_or_none(amount)
        fee_paid = self._extract_order_fee_paid(order, price_float, amount_float)
        fee_rate = self._extract_order_fee_rate(order, fee_paid, price_float, amount_float)
        order_id = getattr(order, "id", None) or getattr(order, "order_id", None) or getattr(order, "ft_order_id", None)
        realized_pnl = self._extract_realized_pnl_usdc(trade, order)
        expected_tif = self._expected_time_in_force(quote_side)
        tif_canonical = self._canonical_tif(tif)
        expected_tif_canonical = self._canonical_tif(expected_tif)

        payload = {
            "pair": pair,
            "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
            "order_id": order_id,
            "raw_liquidity": raw_liquidity,
            "liquidity": liquidity,
            "liquidity_normalized": liquidity,
            "is_maker_fill": liquidity == "maker",
            "is_taker_fill": liquidity == "taker",
            "expected_fee_rate": float(self.fees_maker_HL),
            "actual_fee_paid": float(fee_paid) if fee_paid is not None else None,
            "actual_fee_rate": float(fee_rate) if fee_rate is not None else None,
            "order_type": order_type,
            "tif": tif,
            "tif_canonical": tif_canonical,
            "expected_tif": expected_tif,
            "expected_tif_canonical": expected_tif_canonical,
            "tif_matches_expected": (
                tif_canonical == expected_tif_canonical
                if tif_canonical is not None and expected_tif_canonical is not None
                else None
            ),
            "raw_order_side": raw_side,
            "quote_side": quote_side,
            "post_only_verified": bool(self.post_only_verified),
            "price": price_float,
            "amount": amount_float,
            "realized_pnl": float(realized_pnl) if realized_pnl is not None else None,
            **self._inventory_snapshot(pair),
        }
        self._debug_log_event("fill", payload)

        if liquidity == "maker":
            self._maker_fill_count = int(getattr(self, "_maker_fill_count", 0)) + 1
        if liquidity == "taker":
            self._taker_fill_count = int(getattr(self, "_taker_fill_count", 0)) + 1

        if realized_pnl is not None:
            self._record_realized_pnl(pair, realized_pnl, current_time, payload)

        if self.kill_on_taker_fill and liquidity == "taker":
            self._trigger_kill_switch("unexpected_taker_fill", payload)

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: str, side: str, **kwargs) -> float | None:
        
        if trade.is_short:
            return current_order_rate
            
        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = self._symbol_from_pair(pair)

        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        ok, reason = self._quote_state_valid(pair, "bid", current_order_rate, current_time)
        if not ok:
            self._log_quote_decision(
                pair=pair,
                symbol=symbol,
                side="bid",
                action="adjust_entry",
                decision="reject",
                reason=reason,
                mid_price=mid_price,
                proposed_rate=proposed_rate,
                rounded_price=current_order_rate,
                extra={
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    "current_order_rate": float(current_order_rate),
                    "cancel_open_order": True,
                },
            )
            return None

        q_level = self._inventory_level(pair)
        delta_m = self._select_delta('bid', q_level)
        if delta_m is None or not np.isfinite(float(delta_m)):
            logger.warning("No HJB delta available for bid adjust; cancelling open order.")
            self._log_quote_decision(
                pair=pair,
                symbol=symbol,
                side="bid",
                action="adjust_entry",
                decision="reject",
                reason="boundary_side_disabled" if delta_m is not None else "no_hjb_delta",
                mid_price=mid_price,
                proposed_rate=proposed_rate,
                rounded_price=current_order_rate,
                extra={
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    "current_order_rate": float(current_order_rate),
                    "cancel_open_order": True,
                },
            )
            return None
        delta_source = "hjb_grid"
        delta_model = float(delta_m)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        raw_rate = mid_price - delta_total
        returned_rate = self._round_quote_price(pair, "bid", raw_rate)

        self._log_spread("bid_adjust", mid_price, delta_total, delta_source)

        ok, reason = self._maker_safe(pair, "bid", returned_rate)
        self._log_quote_decision(
            pair=pair,
            symbol=symbol,
            side="bid",
            action="adjust_entry",
            decision="accept" if ok else "reject",
            reason=reason,
            mid_price=mid_price,
            proposed_rate=proposed_rate,
            raw_price=raw_rate,
            rounded_price=returned_rate,
            delta_model=delta_model,
            fee_cushion=fee_cushion,
            delta_total=delta_total,
            extra={
                "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                "current_order_rate": float(current_order_rate),
                "cancel_open_order": not ok,
                "bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None,
            },
        )

        if not ok:
            return None
        return returned_rate

    def adjust_exit_price(self, trade: Trade, order: Order, pair: str,
                          current_time: datetime, proposed_rate: float, current_order_rate: float,
                          exit_tag: str, side: str, **kwargs) -> float | None:
        if trade.is_short:
            return current_order_rate

        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = self._symbol_from_pair(pair)

        if self.hjb_cache is None:
            self._refresh_hjb(pair)

        ok, reason = self._quote_state_valid(pair, "ask", current_order_rate, current_time)
        if not ok:
            self._log_quote_decision(
                pair=pair,
                symbol=symbol,
                side="ask",
                action="adjust_exit",
                decision="reject",
                reason=reason,
                mid_price=mid_price,
                proposed_rate=proposed_rate,
                rounded_price=current_order_rate,
                extra={
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    "current_order_rate": float(current_order_rate),
                    "exit_tag": exit_tag,
                    "cancel_open_order": True,
                },
            )
            return None

        q_level = self._inventory_level(pair)
        delta_p = self._select_delta('ask', q_level)
        if delta_p is None or not np.isfinite(float(delta_p)):
            logger.warning("No HJB delta available for ask adjust; cancelling open order.")
            self._log_quote_decision(
                pair=pair,
                symbol=symbol,
                side="ask",
                action="adjust_exit",
                decision="reject",
                reason="boundary_side_disabled" if delta_p is not None else "no_hjb_delta",
                mid_price=mid_price,
                proposed_rate=proposed_rate,
                rounded_price=current_order_rate,
                extra={
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    "current_order_rate": float(current_order_rate),
                    "exit_tag": exit_tag,
                    "cancel_open_order": True,
                },
            )
            return None

        delta_source = "hjb_grid"
        delta_model = float(delta_p)
        fee_cushion = float(self.fees_maker_HL * mid_price * 2.0)
        delta_total = float(delta_model + fee_cushion)
        raw_rate = mid_price + delta_total
        returned_rate = self._round_quote_price(pair, "ask", raw_rate)

        self._log_spread("ask_adjust", mid_price, delta_total, delta_source)

        ok, reason = self._maker_safe(pair, "ask", returned_rate)
        self._log_quote_decision(
            pair=pair,
            symbol=symbol,
            side="ask",
            action="adjust_exit",
            decision="accept" if ok else "reject",
            reason=reason,
            mid_price=mid_price,
            proposed_rate=proposed_rate,
            raw_price=raw_rate,
            rounded_price=returned_rate,
            delta_model=delta_model,
            fee_cushion=fee_cushion,
            delta_total=delta_total,
            extra={
                "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                "current_order_rate": float(current_order_rate),
                "exit_tag": exit_tag,
                "cancel_open_order": not ok,
                "bps": (delta_total / float(mid_price)) * 10_000.0 if mid_price > 0 else None,
            },
        )

        if not ok:
            return None
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
