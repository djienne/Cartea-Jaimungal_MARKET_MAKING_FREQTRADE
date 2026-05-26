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
    max_collector_age_seconds = 30
    max_book_age_seconds = 5
    collector_timestamp_cache_seconds = 90
    collector_timestamp_newest_files_per_stream = 5
    collector_required_streams = ("orderbooks", "prices", "trades")
    book_snapshot_cache_ms = 500
    max_toxicity = 1.5
    max_daily_loss_usdc = 20
    max_consecutive_losses = 10
    max_post_only_reject_rate = 0.80
    min_post_only_reject_samples = 10
    kill_on_taker_fill = True
    kill_on_time_in_force_mismatch = True
    kill_on_unknown_liquidity_fill = True
    fill_markout_horizons_ms = (100, 1_000, 5_000, 30_000)
    fee_snapshot_cache_seconds = 300
    require_exchange_fee_match_live = True
    max_deployment_report_age_seconds = 86_400
    min_kappa_fit_points = 2
    min_kappa_r2 = 0.0
    min_epsilon_events = 1
    param_update_status_path: str | None = None
    _param_update_lock = threading.Lock()
    _param_update_running: bool = False
    _last_param_update: datetime | None = None
    _last_health_log: datetime | None = None
    _quote_decisions_count: int = 0
    _accepted_order_attempts: int = 0
    _cancel_open_order_requests: int = 0
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

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_repo_path(self, raw_path: Any, default_path: str) -> Path:
        path = Path(str(raw_path or default_path))
        if not path.is_absolute():
            path = self._repo_root() / path
        return path

    def _deployment_gate_report_paths(self) -> dict[str, Path]:
        mm_config = self._mm_config()
        return {
            "post_only": self._resolve_repo_path(
                mm_config.get("post_only_evidence_report_path"),
                "docs/post_only_evidence_report.json",
            ),
            "fee": self._resolve_repo_path(
                mm_config.get("fee_evidence_report_path"),
                "docs/fee_evidence_report.json",
            ),
            "replay": self._resolve_repo_path(
                mm_config.get("replay_acceptance_report_path"),
                "docs/replay_acceptance_report.json",
            ),
            "live_canary": self._resolve_repo_path(
                mm_config.get("live_canary_report_path"),
                "docs/live_canary_report.json",
            ),
        }

    def _read_gate_report_status(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"ok": False, "reason": "missing_report", "path": str(path)}
        except Exception as exc:
            return {"ok": False, "reason": "invalid_report", "path": str(path), "error": str(exc)}
        if not isinstance(payload, dict):
            return {"ok": False, "reason": "invalid_report", "path": str(path)}
        generated_at = payload.get("generated_at")
        generated_at_dt = self._parse_utc_timestamp(generated_at)
        age_seconds = (
            max(0.0, (self._now_utc() - generated_at_dt).total_seconds())
            if generated_at_dt is not None
            else None
        )
        max_age = float(self.max_deployment_report_age_seconds)
        report_ok = payload.get("ok") is True
        reason = "ok" if report_ok else "report_not_ok"
        ok = bool(report_ok)
        if report_ok and generated_at_dt is None:
            ok = False
            reason = "missing_generated_at"
        elif report_ok and max_age > 0 and age_seconds is not None and age_seconds > max_age:
            ok = False
            reason = "stale_report"
        return {
            "ok": ok,
            "reason": reason,
            "path": str(path),
            "reasons": payload.get("reasons", []),
            "generated_at": generated_at,
            "age_seconds": age_seconds,
            "max_age_seconds": max_age,
        }

    def _live_deployment_gate_state(self) -> tuple[bool, str, dict[str, Any]]:
        mm_config = self._mm_config()
        stage = str(mm_config.get("deployment_stage") or "research").strip().lower()
        payload: dict[str, Any] = {
            "deployment_gate_reports_required": True,
            "deployment_stage": stage,
            "gate_reports": {},
        }
        if stage not in {"canary", "production"}:
            return False, "deployment_stage_not_set", payload

        if stage in {"canary", "production"} and not bool(mm_config.get("manual_monitoring_ack", False)):
            return False, "manual_monitoring_not_acknowledged", payload

        required = ["post_only", "fee", "replay"]
        if stage == "production":
            required.append("live_canary")

        reports = self._deployment_gate_report_paths()
        for name in required:
            status = self._read_gate_report_status(reports[name])
            payload["gate_reports"][name] = status
            if not status.get("ok"):
                return False, f"{name}_gate_not_passed", payload

        return True, "ok", payload

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
        if "kill_on_unknown_liquidity_fill" in mm_config:
            self.kill_on_unknown_liquidity_fill = bool(mm_config.get("kill_on_unknown_liquidity_fill"))
        for numeric_key in (
            "max_param_age_seconds",
            "max_collector_age_seconds",
            "max_book_age_seconds",
            "max_toxicity",
            "collector_timestamp_newest_files_per_stream",
            "max_deployment_report_age_seconds",
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
        ok, reason = self._config_fee_state_valid()
        if requested_enabled and not ok:
            self.trading_enabled = False
            self.fail_closed_reason = reason
            self._debug_log_event(
                "trading_enable_rejected",
                {"reason": reason, "dry_run": is_dry_run},
            )
            return

        if requested_enabled and not is_dry_run and not self.post_only_verified:
            self.trading_enabled = False
            self.fail_closed_reason = "post_only_not_verified"
            self._debug_log_event(
                "trading_enable_rejected",
                {"reason": "post_only_not_verified", "dry_run": is_dry_run},
            )
            return

        if requested_enabled and not is_dry_run and self.post_only_verified:
            ok, reason, tif_payload = self._configured_post_only_tif_valid()
            if not ok:
                self.trading_enabled = False
                self.fail_closed_reason = reason
                self._debug_log_event(
                    "trading_enable_rejected",
                    {"reason": reason, "dry_run": is_dry_run, **tif_payload},
                )
                return
            ok, reason, gate_payload = self._live_deployment_gate_state()
            if not ok:
                self.trading_enabled = False
                self.fail_closed_reason = reason
                self._debug_log_event(
                    "trading_enable_rejected",
                    {"reason": reason, "dry_run": is_dry_run, **gate_payload},
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
            self._process_pending_fill_markouts(pair, now)
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

    def _latest_parquet_timestamp(self, path: Path) -> datetime | None:
        try:
            frame = pd.read_parquet(path, columns=["timestamp"])
        except Exception as exc:
            self._debug_log_event(
                "collector_data_read_error",
                {"path": str(path), "reason": str(exc)},
            )
            return None

        if frame.empty or "timestamp" not in frame:
            return None

        series = frame["timestamp"].dropna()
        if series.empty:
            return None

        try:
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if numeric.empty:
                    return None
                return self._parse_utc_timestamp(float(numeric.max()))

            parsed = pd.to_datetime(series, utc=True, errors="coerce").dropna()
            if parsed.empty:
                return None
            return self._as_utc(parsed.max().to_pydatetime())
        except Exception as exc:
            self._debug_log_event(
                "collector_data_timestamp_error",
                {"path": str(path), "reason": str(exc)},
            )
            return None

    def _collector_stream_timestamps(self, symbol: str) -> dict[str, datetime | None]:
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
                and isinstance(cached.get("stream_timestamps"), dict)
            ):
                return {
                    str(stream): self._as_utc(ts)
                    for stream, ts in cached.get("stream_timestamps", {}).items()
                }

        base = self._collector_symbol_dir(symbol)
        stream_timestamps: dict[str, datetime | None] = {}
        for sub_dir in self.collector_required_streams:
            target_dir = base / sub_dir
            if not target_dir.is_dir():
                stream_timestamps[sub_dir] = None
                continue
            newest: datetime | None = None
            shards = list(target_dir.glob("*.parquet"))
            shards = sorted(shards, key=lambda path: path.stat().st_mtime, reverse=True)
            try:
                scan_limit = int(self.collector_timestamp_newest_files_per_stream)
            except Exception:
                scan_limit = 5
            if scan_limit > 0:
                shards = shards[:scan_limit]
            for shard in shards:
                ts = self._latest_parquet_timestamp(shard)
                if ts is None:
                    continue
                if newest is None or ts > newest:
                    newest = ts
            stream_timestamps[sub_dir] = newest
        latest = max(
            (ts for ts in stream_timestamps.values() if ts is not None),
            default=None,
        )
        cache[symbol] = {
            "checked_at": now,
            "latest_ts": latest,
            "stream_timestamps": stream_timestamps,
        }
        return dict(stream_timestamps)

    def _latest_collector_timestamp(self, symbol: str) -> datetime | None:
        stream_timestamps = self._collector_stream_timestamps(symbol)
        newest = max(
            (ts for ts in stream_timestamps.values() if ts is not None),
            default=None,
        )
        return newest

    def _collector_age_seconds(self, symbol: str, now: datetime | None = None) -> float | None:
        stream_timestamps = self._collector_stream_timestamps(symbol)
        present = [ts for ts in stream_timestamps.values() if ts is not None]
        if not present:
            return None
        reference = self._as_utc(now) or self._now_utc()
        return max(0.0, max((reference - ts).total_seconds() for ts in present))

    def _market_data_fresh(self, symbol: str, max_age_seconds: int | None = None) -> tuple[bool, str]:
        max_age = int(max_age_seconds or getattr(self, "max_collector_age_seconds", 30))
        stream_timestamps = self._collector_stream_timestamps(symbol)
        missing = [stream for stream in self.collector_required_streams if stream_timestamps.get(stream) is None]
        if len(missing) == len(tuple(self.collector_required_streams)):
            return False, "no_collector_data"

        if missing:
            return False, "missing_collector_streams:" + ",".join(missing)

        reference = self._now_utc()
        stale: list[tuple[str, float]] = []
        for stream in self.collector_required_streams:
            ts = stream_timestamps.get(stream)
            if ts is None:
                continue
            age = max(0.0, (reference - ts).total_seconds())
            if age > max_age:
                stale.append((stream, age))

        if stale:
            stream, age = max(stale, key=lambda item: item[1])
            return False, f"collector_data_stale_{stream}_{age:.1f}s"

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
            "position": float(signed_base),
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

        ok, reason = self._fee_state_valid(pair)
        if not ok:
            return False, reason

        ok, reason = self._book_is_fresh(pair, current_time)
        if not ok:
            return False, "stale_orderbook" if reason == "empty_orderbook" else reason

        symbol = self._symbol_from_pair(pair)
        ok, reason = self._market_data_fresh(symbol)
        if not ok:
            if reason.startswith("collector_data_stale") or reason.startswith("missing_collector_streams"):
                return False, "stale_collector_data"
            return False, reason

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
        if self.trading_enabled and not is_dry_run:
            ok, reason, _ = self._live_deployment_gate_state()
            if not ok:
                return False, reason

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

    def _is_limit_order_type(self, order_type: str | None) -> bool:
        return str(order_type or "").strip().lower() == "limit"

    def _entry_side_rejection_reason(self, side: str | None) -> str | None:
        side_text = str(side or "").strip().lower()
        if side_text in {"short", "sell", "ask", "open_short"} and not self.can_short:
            return "short_entries_disabled"
        if side_text not in {"long", "buy", "bid", "entry", ""}:
            return "unsupported_entry_side"
        return None

    def _trigger_kill_switch(self, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.trading_enabled = False
        self.fail_closed_reason = reason
        payload = dict(payload or {})
        pair = payload.get("pair")
        try:
            payload.update(self._cancel_open_orders_for_kill_switch(pair))
        except Exception as exc:
            payload["cancel_error"] = str(exc)
        self._debug_log_event("kill_switch", {"reason": reason, **payload})

    def _record_quote_decision(self, pair: str, decision: str, reason: str) -> None:
        self._quote_decisions_count = int(getattr(self, "_quote_decisions_count", 0)) + 1
        post_only_reasons = {
            "bid_crosses_ask",
            "ask_crosses_bid",
            "crossed_or_invalid_book",
            "empty_orderbook",
            "post_only_not_verified",
            "not_post_only_supported",
            "time_in_force_not_post_only",
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

    def _record_order_attempt_accepted(self) -> None:
        self._accepted_order_attempts = int(getattr(self, "_accepted_order_attempts", 0)) + 1

    def _record_open_order_cancel_request(self, count: int = 1) -> None:
        self._cancel_open_order_requests = int(getattr(self, "_cancel_open_order_requests", 0)) + max(1, int(count))

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

    def _configured_time_in_force(self, quote_side: str | None) -> Any:
        config = getattr(self, "config", {}) or {}
        config_tif = config.get("order_time_in_force") if isinstance(config, dict) else None
        if not isinstance(config_tif, dict):
            config_tif = {}
        strategy_tif = self.order_time_in_force if isinstance(self.order_time_in_force, dict) else {}
        if quote_side == "bid":
            return config_tif.get("entry", strategy_tif.get("entry"))
        if quote_side == "ask":
            return config_tif.get("exit", strategy_tif.get("exit"))
        return None

    def _configured_post_only_tif_valid(self) -> tuple[bool, str, dict[str, Any]]:
        entry_tif = self._configured_time_in_force("bid")
        exit_tif = self._configured_time_in_force("ask")
        entry_canonical = self._canonical_tif(entry_tif)
        exit_canonical = self._canonical_tif(exit_tif)
        payload = {
            "entry_time_in_force": entry_tif,
            "exit_time_in_force": exit_tif,
            "entry_time_in_force_canonical": entry_canonical,
            "exit_time_in_force_canonical": exit_canonical,
        }
        if entry_canonical != "post_only" or exit_canonical != "post_only":
            return False, "time_in_force_not_post_only", payload
        return True, "ok", payload

    def _expected_time_in_force(self, quote_side: str | None) -> str | None:
        configured = self._configured_time_in_force(quote_side)
        return "Alo" if self.post_only_verified else configured

    def _time_in_force_valid(self, quote_side: str, time_in_force: str | None) -> tuple[bool, str]:
        expected_tif = self._expected_time_in_force(quote_side)
        expected_canonical = self._canonical_tif(expected_tif)
        actual_canonical = self._canonical_tif(time_in_force)
        if expected_canonical == "post_only" and actual_canonical != "post_only":
            return False, "time_in_force_not_post_only"
        return True, "ok"

    def _config_fee_rate(self) -> float | None:
        config = getattr(self, "config", {}) or {}
        if not isinstance(config, dict):
            return None
        return self._finite_float_or_none(config.get("fee"))

    def _config_fee_state_valid(self) -> tuple[bool, str]:
        config_fee = self._config_fee_rate()
        if config_fee is None:
            return False, "missing_config_fee"
        if not self._fee_rate_matches(float(self.fees_maker_HL), config_fee):
            return False, "config_fee_mismatch"
        return True, "ok"

    def _fee_rate_matches(self, expected: float | None, observed: float | None, tolerance: float = 1e-9) -> bool | None:
        if expected is None or observed is None:
            return None
        return abs(float(expected) - float(observed)) <= float(tolerance)

    def _extract_fee_rates_from_mapping(self, payload: Any) -> tuple[float | None, float | None]:
        if not isinstance(payload, dict):
            return None, None

        maker = None
        taker = None
        for key in ("maker", "maker_fee", "makerFee", "maker_rate", "makerRate"):
            if key in payload:
                maker = self._finite_float_or_none(payload.get(key))
                if maker is not None:
                    break
        for key in ("taker", "taker_fee", "takerFee", "taker_rate", "takerRate"):
            if key in payload:
                taker = self._finite_float_or_none(payload.get(key))
                if taker is not None:
                    break
        if maker is not None or taker is not None:
            return maker, taker

        for key in ("trading", "fees", "info"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                maker, taker = self._extract_fee_rates_from_mapping(nested)
                if maker is not None or taker is not None:
                    return maker, taker
        return None, None

    def _cached_exchange_fee_snapshot(self, pair: str) -> dict[str, Any] | None:
        cache = getattr(self, "_fee_snapshot_cache", None)
        if not isinstance(cache, dict):
            return None
        entry = cache.get(pair)
        if not isinstance(entry, dict):
            return None
        checked_at = self._as_utc(entry.get("checked_at"))
        if checked_at is None:
            return None
        age = (self._now_utc() - checked_at).total_seconds()
        if age > float(self.fee_snapshot_cache_seconds):
            return None
        snapshot = entry.get("snapshot")
        return snapshot if isinstance(snapshot, dict) else None

    def _exchange_fee_snapshot(self, pair: str) -> dict[str, Any]:
        cached = self._cached_exchange_fee_snapshot(pair)
        if cached is not None:
            return cached

        exchange = getattr(self, "exchange", None)
        maker = None
        taker = None
        source = "unavailable"

        if exchange is not None:
            try:
                market = getattr(exchange, "markets", {}).get(pair, {})
            except Exception:
                market = {}
            maker, taker = self._extract_fee_rates_from_mapping(market)
            if maker is not None or taker is not None:
                source = "exchange_markets"

            if maker is None and taker is None:
                maker, taker = self._extract_fee_rates_from_mapping(getattr(exchange, "fees", None))
                if maker is not None or taker is not None:
                    source = "exchange_fees"

            if (
                maker is None
                and taker is None
                and bool(self._mm_config().get("fetch_exchange_fee_snapshot", False))
            ):
                method = getattr(exchange, "fetch_trading_fee", None)
                if callable(method):
                    try:
                        fee_payload = method(pair)
                    except Exception as exc:
                        source = f"fetch_trading_fee_error:{exc}"
                    else:
                        maker, taker = self._extract_fee_rates_from_mapping(fee_payload)
                        source = "fetch_trading_fee"

        snapshot = {
            "source": source,
            "maker_fee_rate": float(maker) if maker is not None else None,
            "taker_fee_rate": float(taker) if taker is not None else None,
        }
        cache = getattr(self, "_fee_snapshot_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._fee_snapshot_cache = cache
        cache[pair] = {"checked_at": self._now_utc(), "snapshot": snapshot}
        return snapshot

    def _fee_snapshot(self, pair: str, actual_fee_rate: float | None = None) -> dict[str, Any]:
        exchange_snapshot = self._exchange_fee_snapshot(pair)
        strategy_fee = float(self.fees_maker_HL)
        config_fee = self._config_fee_rate()
        exchange_maker_fee = self._finite_float_or_none(exchange_snapshot.get("maker_fee_rate"))
        exchange_taker_fee = self._finite_float_or_none(exchange_snapshot.get("taker_fee_rate"))
        config_matches = self._fee_rate_matches(strategy_fee, config_fee)
        exchange_maker_matches = self._fee_rate_matches(strategy_fee, exchange_maker_fee)
        actual_matches = self._fee_rate_matches(strategy_fee, actual_fee_rate)
        observed_matches = [
            match
            for match in (config_matches, exchange_maker_matches, actual_matches)
            if match is not None
        ]
        return {
            "strategy_maker_fee_rate": strategy_fee,
            "config_fee_rate": float(config_fee) if config_fee is not None else None,
            "config_fee_matches_strategy": config_matches,
            "exchange_fee_source": exchange_snapshot.get("source"),
            "exchange_maker_fee_rate": exchange_maker_fee,
            "exchange_taker_fee_rate": exchange_taker_fee,
            "exchange_maker_fee_matches_strategy": exchange_maker_matches,
            "actual_fee_rate": float(actual_fee_rate) if actual_fee_rate is not None else None,
            "actual_fee_matches_strategy": actual_matches,
            "fee_agreement_ok": all(observed_matches) if observed_matches else None,
        }

    def _fee_state_valid(self, pair: str) -> tuple[bool, str]:
        ok, reason = self._config_fee_state_valid()
        if not ok:
            return False, reason

        snapshot = self._fee_snapshot(pair)
        if snapshot.get("exchange_maker_fee_matches_strategy") is False:
            return False, "exchange_fee_mismatch"

        is_dry_run = bool(getattr(self, "config", {}).get("dry_run", True))
        if (
            bool(self.trading_enabled)
            and not is_dry_run
            and bool(self.post_only_verified)
            and bool(self.require_exchange_fee_match_live)
            and snapshot.get("exchange_maker_fee_rate") is None
        ):
            return False, "exchange_fee_unavailable"

        return True, "ok"

    def _markout_value(self, quote_side: str | None, fill_price: float, future_mid_price: float) -> float | None:
        if quote_side == "bid":
            return float(future_mid_price) - float(fill_price)
        if quote_side == "ask":
            return float(fill_price) - float(future_mid_price)
        return None

    def _pending_fill_markout_entries(self) -> list[dict[str, Any]]:
        pending = getattr(self, "_pending_fill_markouts", None)
        if not isinstance(pending, list):
            pending = []
            self._pending_fill_markouts = pending
        return pending

    def _schedule_fill_markouts(self, fill_payload: dict[str, Any], current_time: datetime) -> None:
        quote_side = fill_payload.get("quote_side")
        fill_price = self._finite_float_or_none(fill_payload.get("price"))
        amount = self._finite_float_or_none(fill_payload.get("amount"))
        if quote_side not in {"bid", "ask"} or fill_price is None or amount is None or amount <= 0:
            return

        horizons = []
        for horizon in getattr(self, "fill_markout_horizons_ms", ()):
            try:
                horizon_int = int(horizon)
            except Exception:
                continue
            if horizon_int > 0:
                horizons.append(horizon_int)
        if not horizons:
            return

        fill_ts = self._as_utc(current_time) or self._now_utc()
        self._pending_fill_markout_entries().append(
            {
                "pair": fill_payload.get("pair"),
                "trade_id": fill_payload.get("trade_id"),
                "order_id": fill_payload.get("order_id"),
                "quote_side": quote_side,
                "fill_price": float(fill_price),
                "amount": float(amount),
                "fill_ts": fill_ts,
                "remaining_horizons_ms": sorted(set(horizons)),
            }
        )

    def _process_pending_fill_markouts(self, pair: str, current_time: datetime) -> None:
        pending = self._pending_fill_markout_entries()
        if not pending:
            return

        now = self._as_utc(current_time) or self._now_utc()
        keep: list[dict[str, Any]] = []
        for entry in pending:
            if entry.get("pair") != pair:
                keep.append(entry)
                continue

            fill_ts = self._as_utc(entry.get("fill_ts"))
            if fill_ts is None:
                continue

            due_horizons = []
            remaining_horizons = []
            for horizon_ms in entry.get("remaining_horizons_ms", []):
                due_at = fill_ts + timedelta(milliseconds=int(horizon_ms))
                if now >= due_at:
                    due_horizons.append(int(horizon_ms))
                else:
                    remaining_horizons.append(int(horizon_ms))

            if due_horizons:
                future_mid = self.get_mid_price(pair, float(entry["fill_price"]))
                for horizon_ms in due_horizons:
                    markout = self._markout_value(entry.get("quote_side"), float(entry["fill_price"]), future_mid)
                    if markout is None:
                        continue
                    self._debug_log_event(
                        "fill_markout",
                        {
                            "pair": pair,
                            "trade_id": entry.get("trade_id"),
                            "order_id": entry.get("order_id"),
                            "quote_side": entry.get("quote_side"),
                            "horizon_ms": int(horizon_ms),
                            "fill_ts": fill_ts.isoformat().replace("+00:00", "Z"),
                            "markout_ts": now.isoformat().replace("+00:00", "Z"),
                            "fill_price": float(entry["fill_price"]),
                            "future_mid": float(future_mid),
                            "amount": float(entry["amount"]),
                            "markout_usdc_per_base": float(markout),
                            "markout_usdc": float(markout) * float(entry["amount"]),
                        },
                    )

            if remaining_horizons:
                entry["remaining_horizons_ms"] = remaining_horizons
                keep.append(entry)

        self._pending_fill_markouts = keep

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

        risk_payload = {
            "pair": pair,
            "realized_pnl": float(pnl_usdc),
            "daily_realized_pnl": float(self._daily_realized_pnl_usdc),
            "max_daily_loss_usdc": float(self.max_daily_loss_usdc),
            "consecutive_losses": int(self._consecutive_losses),
            "max_consecutive_losses": int(self.max_consecutive_losses),
            "order_id": payload.get("order_id"),
            "trade_id": payload.get("trade_id"),
            "quote_side": payload.get("quote_side"),
            "liquidity": payload.get("liquidity"),
        }
        self._debug_log_event("risk_update", risk_payload)

        if self.trading_enabled and self._daily_realized_pnl_usdc <= -abs(float(self.max_daily_loss_usdc)):
            self._trigger_kill_switch(
                "drawdown_limit_reached",
                {**payload, **risk_payload},
            )
        if self.trading_enabled and self._consecutive_losses >= int(self.max_consecutive_losses):
            self._trigger_kill_switch(
                "consecutive_losses_limit_reached",
                {**payload, **risk_payload},
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

    def _open_order_count(self, pair: str) -> int | None:
        exchange = getattr(self, "exchange", None)
        if exchange is not None:
            for method_name in ("fetch_open_orders", "get_open_orders"):
                method = getattr(exchange, method_name, None)
                if not callable(method):
                    continue
                try:
                    orders = method(pair)
                except TypeError:
                    try:
                        orders = method()
                    except Exception:
                        continue
                except Exception:
                    continue
                try:
                    self._last_open_order_count_source = method_name
                    return len(list(orders or []))
                except Exception:
                    return None

            orders = getattr(exchange, "open_orders", None)
            if isinstance(orders, dict):
                if pair in orders and isinstance(orders[pair], list):
                    self._last_open_order_count_source = "exchange_open_orders"
                    return len(orders[pair])
                count = 0
                for value in orders.values():
                    if isinstance(value, list):
                        count += len(value)
                self._last_open_order_count_source = "exchange_open_orders"
                return count
            if isinstance(orders, list):
                count = 0
                for order in orders:
                    if not isinstance(order, dict):
                        continue
                    symbol = order.get("symbol") or order.get("pair")
                    if symbol is None or str(symbol) == pair:
                        count += 1
                self._last_open_order_count_source = "exchange_open_orders"
                return count
        return self._open_order_count_from_trades(pair)

    def _exchange_open_orders_for_pair(self, pair: str) -> tuple[list[Any] | None, str | None]:
        exchange = getattr(self, "exchange", None)
        if exchange is None:
            return None, None

        for method_name in ("fetch_open_orders", "get_open_orders"):
            method = getattr(exchange, method_name, None)
            if not callable(method):
                continue
            try:
                orders = method(pair)
            except TypeError:
                try:
                    orders = method()
                except Exception:
                    continue
            except Exception:
                continue
            try:
                order_list = list(orders or [])
            except TypeError:
                order_list = []
            return [order for order in order_list if self._order_matches_pair(order, pair)], method_name

        orders = getattr(exchange, "open_orders", None)
        if isinstance(orders, dict):
            if pair in orders and isinstance(orders[pair], list):
                return list(orders[pair]), "exchange_open_orders"
            order_list: list[Any] = []
            for value in orders.values():
                if isinstance(value, list):
                    order_list.extend(value)
            return [order for order in order_list if self._order_matches_pair(order, pair)], "exchange_open_orders"
        if isinstance(orders, list):
            return [order for order in orders if self._order_matches_pair(order, pair)], "exchange_open_orders"
        return None, None

    def _order_matches_pair(self, order: Any, pair: str) -> bool:
        if isinstance(order, dict):
            symbol = order.get("symbol") or order.get("pair")
        else:
            symbol = getattr(order, "symbol", None) or getattr(order, "pair", None)
        return symbol is None or str(symbol) == pair

    def _order_id_for_cancel(self, order: Any) -> Any:
        if isinstance(order, dict):
            for key in ("id", "order_id", "ft_order_id", "clientOrderId", "client_order_id"):
                value = order.get(key)
                if value not in (None, ""):
                    return value
            return None
        for attr in ("id", "order_id", "ft_order_id", "clientOrderId", "client_order_id"):
            value = getattr(order, attr, None)
            if value not in (None, ""):
                return value
        return None

    def _cancel_open_orders_for_kill_switch(self, pair: Any) -> dict[str, Any]:
        if not pair:
            return {"cancel_open_orders_requested": 0, "cancel_method": "no_pair"}
        exchange = getattr(self, "exchange", None)
        if exchange is None:
            return {"cancel_open_orders_requested": 0, "cancel_method": "no_exchange"}

        cancel_all = getattr(exchange, "cancel_all_orders", None)
        if callable(cancel_all):
            cancel_count = self._open_order_count(str(pair))
            cancel_all(pair)
            requested = int(cancel_count) if cancel_count is not None else 1
            if requested > 0:
                self._record_open_order_cancel_request(requested)
            return {
                "cancel_open_orders_requested": requested,
                "cancel_method": "cancel_all_orders",
            }

        cancel_one = getattr(exchange, "cancel_order", None) or getattr(exchange, "cancel", None)
        if not callable(cancel_one):
            return {"cancel_open_orders_requested": 0, "cancel_method": "unavailable"}

        orders, source = self._exchange_open_orders_for_pair(str(pair))
        if orders is None:
            return {"cancel_open_orders_requested": 0, "cancel_method": "no_open_order_source"}

        cancelled_ids: list[str] = []
        cancel_errors: list[dict[str, Any]] = []
        for order in orders:
            if not self._order_looks_open(order):
                continue
            order_id = self._order_id_for_cancel(order)
            if order_id is None:
                cancel_errors.append({"order_id": None, "error": "missing_order_id"})
                continue
            try:
                try:
                    cancel_one(order_id, str(pair))
                except TypeError:
                    cancel_one(order_id)
                cancelled_ids.append(str(order_id))
            except Exception as exc:
                cancel_errors.append({"order_id": str(order_id), "error": str(exc)})

        if cancelled_ids:
            self._record_open_order_cancel_request(len(cancelled_ids))
        return {
            "cancel_open_orders_requested": len(cancelled_ids),
            "cancel_method": getattr(cancel_one, "__name__", "cancel_order"),
            "cancel_source": source,
            "cancelled_order_ids": cancelled_ids,
            "cancel_errors": cancel_errors,
        }

    def _open_order_count_from_trades(self, pair: str) -> int | None:
        open_order_trades = self._freqtrade_open_order_trades(pair)
        if open_order_trades:
            count = 0
            for trade in open_order_trades:
                order_count = self._open_order_count_for_trade(trade)
                count += order_count if order_count is not None else 1
            self._last_open_order_count_source = "freqtrade_open_order_trades"
            return count

        try:
            open_trades = list(Trade.get_trades(is_open=True, pair=pair) or [])
        except Exception:
            return self._estimated_open_order_count()

        if not open_trades:
            self._last_open_order_count_source = "freqtrade_open_trades"
            return 0

        count = 0
        observed_order_state = False
        for trade in open_trades:
            orders = getattr(trade, "orders", None)
            if callable(orders):
                try:
                    orders = orders()
                except Exception:
                    orders = None
            if orders is not None:
                try:
                    order_list = list(orders)
                except TypeError:
                    order_list = []
                observed_order_state = True
                count += sum(1 for order in order_list if self._order_looks_open(order))
                continue

            has_open_orders = getattr(trade, "has_open_orders", None)
            if callable(has_open_orders):
                try:
                    has_open_orders = has_open_orders()
                except Exception:
                    has_open_orders = None
            if has_open_orders is not None:
                observed_order_state = True
                if bool(has_open_orders):
                    count += 1

        if observed_order_state:
            self._last_open_order_count_source = "freqtrade_open_trades"
            return count
        return self._estimated_open_order_count()

    def _estimated_open_order_count(self) -> int:
        attempts = int(getattr(self, "_accepted_order_attempts", 0))
        fills = int(getattr(self, "_maker_fill_count", 0)) + int(getattr(self, "_taker_fill_count", 0))
        cancels = int(getattr(self, "_cancel_open_order_requests", 0))
        self._last_open_order_count_source = "accepted_confirmation_estimate"
        return max(0, attempts - fills - cancels)

    def _freqtrade_open_order_trades(self, pair: str) -> list[Any] | None:
        method = getattr(Trade, "get_open_order_trades", None)
        if not callable(method):
            return None
        try:
            trades = list(method() or [])
        except Exception:
            return None

        matching = []
        for trade in trades:
            trade_pair = getattr(trade, "pair", None)
            if trade_pair is None or str(trade_pair) == pair:
                matching.append(trade)
        return matching

    def _open_order_count_for_trade(self, trade: Any) -> int | None:
        orders = getattr(trade, "orders", None)
        if callable(orders):
            try:
                orders = orders()
            except Exception:
                orders = None
        if orders is not None:
            try:
                order_list = list(orders)
            except TypeError:
                order_list = []
            return sum(1 for order in order_list if self._order_looks_open(order))

        has_open_orders = getattr(trade, "has_open_orders", None)
        if callable(has_open_orders):
            try:
                has_open_orders = has_open_orders()
            except Exception:
                has_open_orders = None
        if has_open_orders is not None:
            return 1 if bool(has_open_orders) else 0
        return None

    def _order_looks_open(self, order: Any) -> bool:
        if isinstance(order, dict):
            status = order.get("status")
            is_open = order.get("is_open")
            remaining = order.get("remaining", order.get("safe_remaining"))
        else:
            status = getattr(order, "status", None)
            is_open = getattr(order, "is_open", None)
            remaining = getattr(order, "remaining", getattr(order, "safe_remaining", None))

        if is_open is not None:
            return bool(is_open)

        text = str(status).strip().lower() if status is not None else ""
        if text in {"closed", "filled", "canceled", "cancelled", "expired", "rejected"}:
            return False

        remaining_float = self._finite_float_or_none(remaining)
        if remaining_float is not None and remaining_float > 0:
            return True

        if not text:
            return False
        if text in {"open", "new", "created", "pending", "partially_filled", "partially-filled"}:
            return True
        return False

    def _unrealized_pnl_usdc(self, pair: str) -> float | None:
        mid_price = self.get_mid_price(pair, 0.0)
        if not np.isfinite(float(mid_price)) or float(mid_price) <= 0:
            return None

        signed_base = self._signed_base_position(pair)
        if abs(float(signed_base)) <= 1e-12:
            return 0.0

        try:
            open_trades = list(Trade.get_trades(is_open=True, pair=pair) or [])
        except Exception:
            open_trades = []

        pnl = 0.0
        measured_base = 0.0
        for trade in open_trades:
            amount = self._finite_float_or_none(getattr(trade, "amount", None))
            open_rate = self._finite_float_or_none(getattr(trade, "open_rate", None))
            if amount is None or open_rate is None or open_rate <= 0:
                continue
            amount = abs(float(amount))
            if amount <= 0:
                continue
            if bool(getattr(trade, "is_short", False)):
                pnl += (float(open_rate) - float(mid_price)) * amount
                measured_base -= amount
            else:
                pnl += (float(mid_price) - float(open_rate)) * amount
                measured_base += amount

        if abs(measured_base) <= 1e-12:
            return None
        return float(pnl)

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
        expected_tif = self._expected_time_in_force(side)
        expected_tif_canonical = self._canonical_tif(expected_tif)
        config = getattr(self, "config", {}) if isinstance(getattr(self, "config", {}), dict) else {}
        payload = {
            "action": action,
            "pair": pair,
            "symbol": symbol,
            "side": side,
            "trading_enabled": bool(self.trading_enabled),
            "dry_run": bool(config.get("dry_run", True)),
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
            "expected_tif": expected_tif,
            "expected_tif_canonical": expected_tif_canonical,
            "post_only": expected_tif_canonical == "post_only",
            "post_only_verified": bool(self.post_only_verified),
            "params": self._params_snapshot(symbol),
            "fee_snapshot": self._fee_snapshot(pair),
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
        open_order_count = self._open_order_count(pair)
        open_order_source = getattr(self, "_last_open_order_count_source", "unavailable")
        config = getattr(self, "config", {}) if isinstance(getattr(self, "config", {}), dict) else {}
        mm_config = self._mm_config()
        entry_tif = self._expected_time_in_force("bid")
        exit_tif = self._expected_time_in_force("ask")
        self._debug_log_event(
            "health",
            {
                "pair": pair,
                "symbol": symbol,
                "trading_enabled": bool(self.trading_enabled),
                "fail_closed_reason": self.fail_closed_reason,
                "dry_run": bool(config.get("dry_run", True)),
                "stake_amount": config.get("stake_amount"),
                "tradable_balance_ratio": config.get("tradable_balance_ratio"),
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
                "deployment_stage": str(mm_config.get("deployment_stage") or "research"),
                "deployment_gate_reports_required": True,
                "manual_monitoring_ack": bool(mm_config.get("manual_monitoring_ack", False)),
                "max_deployment_report_age_seconds": float(self.max_deployment_report_age_seconds),
                "expected_entry_time_in_force": entry_tif,
                "expected_exit_time_in_force": exit_tif,
                "expected_entry_time_in_force_canonical": self._canonical_tif(entry_tif),
                "expected_exit_time_in_force_canonical": self._canonical_tif(exit_tif),
                "kill_on_taker_fill": bool(self.kill_on_taker_fill),
                "kill_on_time_in_force_mismatch": bool(self.kill_on_time_in_force_mismatch),
                "kill_on_unknown_liquidity_fill": bool(self.kill_on_unknown_liquidity_fill),
                "fee_snapshot": self._fee_snapshot(pair),
                "open_orders": open_order_count,
                "open_orders_source": open_order_source,
                "accepted_order_attempts": int(getattr(self, "_accepted_order_attempts", 0)),
                "open_order_cancels_requested": int(getattr(self, "_cancel_open_order_requests", 0)),
                "maker_fills": int(getattr(self, "_maker_fill_count", 0)),
                "taker_fills": int(getattr(self, "_taker_fill_count", 0)),
                "post_only_rejects": int(getattr(self, "_post_only_rejects", 0)),
                "max_post_only_reject_rate": float(self.max_post_only_reject_rate),
                "quote_decisions": int(getattr(self, "_quote_decisions_count", 0)),
                "realized_pnl": float(getattr(self, "_daily_realized_pnl_usdc", 0.0)),
                "max_daily_loss_usdc": float(self.max_daily_loss_usdc),
                "unrealized_pnl": self._unrealized_pnl_usdc(pair),
                "consecutive_losses": int(getattr(self, "_consecutive_losses", 0)),
                "max_consecutive_losses": int(self.max_consecutive_losses),
                "max_abs_inventory_units": int(self.max_abs_inventory_units),
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
        side_reason = self._entry_side_rejection_reason(side)
        if side_reason is not None:
            self._debug_log_event(
                "entry_rejected",
                {
                    "pair": pair,
                    "reason": side_reason,
                    "rate": float(rate),
                    "side": side,
                    "order_type": order_type,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        if not self._is_limit_order_type(order_type):
            self._debug_log_event(
                "entry_rejected",
                {
                    "pair": pair,
                    "reason": "non_limit_order_type",
                    "rate": float(rate),
                    "side": side,
                    "order_type": order_type,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        ok, reason = self._time_in_force_valid("bid", time_in_force)
        if not ok:
            self._record_quote_decision(pair, "reject", reason)
            self._debug_log_event(
                "entry_rejected",
                {
                    "pair": pair,
                    "reason": reason,
                    "rate": float(rate),
                    "side": side,
                    "order_type": order_type,
                    "time_in_force": time_in_force,
                    "expected_tif": self._expected_time_in_force("bid"),
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        ok, reason = self._quote_state_valid(pair, "bid", rate, current_time)
        if not ok:
            self._debug_log_event(
                "entry_rejected",
                {
                    "pair": pair,
                    "reason": reason,
                    "rate": float(rate),
                    "side": side,
                    "order_type": order_type,
                    **self._inventory_snapshot(pair),
                },
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

        self._record_order_attempt_accepted()
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
        protected_exit_reasons = {
            "stop_loss",
            "stoploss",
            "stoploss_on_exchange",
            "liquidation",
            "emergency_exit",
            "force_exit",
            "force_sell",
        }
        if exit_reason in protected_exit_reasons:
            return True

        if not self._is_limit_order_type(order_type):
            self._debug_log_event(
                "exit_rejected",
                {
                    "pair": pair,
                    "reason": "non_limit_order_type",
                    "rate": float(rate),
                    "order_type": order_type,
                    "exit_reason": exit_reason,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

        ok, reason = self._time_in_force_valid("ask", time_in_force)
        if not ok:
            self._record_quote_decision(pair, "reject", reason)
            self._debug_log_event(
                "exit_rejected",
                {
                    "pair": pair,
                    "reason": reason,
                    "rate": float(rate),
                    "order_type": order_type,
                    "time_in_force": time_in_force,
                    "expected_tif": self._expected_time_in_force("ask"),
                    "exit_reason": exit_reason,
                    "trade_id": int(trade.id) if getattr(trade, "id", None) is not None else None,
                    **self._inventory_snapshot(pair),
                },
            )
            return False

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

        self._record_order_attempt_accepted()
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
            "fee_snapshot": self._fee_snapshot(pair, fee_rate),
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
        self._schedule_fill_markouts(payload, current_time)

        if liquidity == "maker":
            self._maker_fill_count = int(getattr(self, "_maker_fill_count", 0)) + 1
        if liquidity == "taker":
            self._taker_fill_count = int(getattr(self, "_taker_fill_count", 0)) + 1

        if realized_pnl is not None:
            self._record_realized_pnl(pair, realized_pnl, current_time, payload)

        tif_missing_when_required = (
            expected_tif_canonical == "post_only"
            and tif_canonical is None
        )
        tif_mismatch = payload["tif_matches_expected"] is False or tif_missing_when_required
        if self.kill_on_time_in_force_mismatch and self.post_only_verified and tif_mismatch:
            self._trigger_kill_switch("unexpected_time_in_force", payload)

        if (
            self.kill_on_unknown_liquidity_fill
            and self.post_only_verified
            and liquidity not in {"maker", "taker"}
        ):
            self._trigger_kill_switch("unknown_fill_liquidity", payload)

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
            self._record_open_order_cancel_request()
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
            self._record_open_order_cancel_request()
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
            self._record_open_order_cancel_request()
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
            self._record_open_order_cancel_request()
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
            self._record_open_order_cancel_request()
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
            self._record_open_order_cancel_request()
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
