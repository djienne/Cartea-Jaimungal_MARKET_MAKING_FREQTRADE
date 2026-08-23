#!/usr/bin/env python3
"""
Hyperliquid Tick Data Collector
Collects time-tagged prices, executed orders, and order book data via websockets
"""

import asyncio
import json
import time
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import csv
import os
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from hyperliquid.info import Info
from hyperliquid.utils import constants


# float32 represents integers exactly only below 2**24; sizes on cheap assets
# are large integers, so narrowing is gated on this ceiling.
FLOAT32_EXACT_INT_LIMIT = 2 ** 24

# Every shard stream the collector writes, in one place so storage, flushing,
# pruning and compaction can never drift apart when a stream is added.
STREAMS = ('prices', 'trades', 'orderbooks', 'asset_ctx')


@dataclass
class TickData:
    """Base class for all tick data"""
    timestamp: float
    symbol: str
    exchange_timestamp: Optional[int] = None


@dataclass
class PriceData:
    """Price tick data"""
    timestamp: float
    symbol: str
    price: float
    size: float
    exchange_timestamp: Optional[int] = None
    side: Optional[str] = None  # 'bid' or 'ask' for BBO data


@dataclass
class TradeData:
    """Trade execution data"""
    timestamp: float
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    exchange_timestamp: Optional[int] = None
    trade_id: Optional[str] = None


@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: float


@dataclass
class OrderBookData:
    """Order book snapshot data"""
    timestamp: float
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    exchange_timestamp: Optional[int] = None
    sequence: Optional[int] = None


class DataStats:
    """Track data collection statistics"""
    def __init__(self):
        self.start_time = time.time()
        self.counters = defaultdict(int)
        self.last_update = time.time()
        self.recent_data = defaultdict(lambda: deque(maxlen=100))
    
    def update(self, data_type: str, data: Any = None):
        self.counters[data_type] += 1
        self.last_update = time.time()
        if data:
            self.recent_data[data_type].append(data)
    
    def get_summary(self) -> Dict[str, Any]:
        runtime = time.time() - self.start_time
        return {
            'runtime_seconds': runtime,
            'runtime_formatted': f"{runtime//3600:.0f}h {(runtime%3600)//60:.0f}m {runtime%60:.0f}s",
            'counters': dict(self.counters),
            'rates_per_minute': {k: v / (runtime / 60) for k, v in self.counters.items() if runtime > 0},
            'last_update': datetime.fromtimestamp(self.last_update).strftime('%H:%M:%S')
        }


class HyperliquidDataCollector:
    """Main data collector class"""
    
    def __init__(self, symbols: List[str], output_dir: str = "data", orderbook_depth: int = 20):
        self.symbols = symbols
        self.output_dir = output_dir
        self.orderbook_depth = orderbook_depth  # Configurable order book depth
        self.info = Info(constants.MAINNET_API_URL, skip_ws=False)
        self.stats = DataStats()
        self.subscription_ids = []
        
        # Lock for thread-safe buffer access
        self.lock = threading.Lock()
        
        # Create separate buffers for each symbol
        self.data_buffers = {}
        for symbol in symbols:
            self.data_buffers[symbol] = {
                'prices': deque(maxlen=100000),
                'trades': deque(maxlen=100000),
                'orderbooks': deque(maxlen=10000),
                'asset_ctx': deque(maxlen=100000)
            }
        # Per-symbol (signature, last_kept_ts) for asset_ctx change-detection.
        self._asset_ctx_last = {}
        
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Self-healing / watchdog configuration
        try:
            self.inactivity_timeout_sec = int(os.getenv("INACTIVITY_TIMEOUT_SEC", "180"))
        except ValueError:
            self.inactivity_timeout_sec = 180
        try:
            self.max_reconnect_attempts = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "3"))
        except ValueError:
            self.max_reconnect_attempts = 3
        try:
            self.reconnect_backoff_sec = float(os.getenv("RECONNECT_BACKOFF_SEC", "5"))
        except ValueError:
            self.reconnect_backoff_sec = 5.0
        # How long after a (re)connect to ignore a "socket is down" reading. The SDK
        # starts its websocket thread before the socket finishes connecting, so a
        # probe fired immediately would see a healthy startup as a failure and
        # reconnect in a loop.
        try:
            self.ws_health_grace_sec = float(os.getenv("WS_HEALTH_GRACE_SEC", "20"))
        except ValueError:
            self.ws_health_grace_sec = 20.0
        self._last_connect_time = time.time()
        # Flush cadence. The market-making strategy rejects collector data older
        # than max_collector_age_seconds (30s by default), so the flush interval
        # must stay comfortably below that window or quotes get rejected as
        # stale_collector_data. Default to 10s; override with FLUSH_INTERVAL_SEC.
        try:
            self.flush_interval_sec = float(os.getenv("FLUSH_INTERVAL_SEC", "10"))
        except ValueError:
            self.flush_interval_sec = 10.0
        if self.flush_interval_sec <= 0:
            self.flush_interval_sec = 10.0
        # Retention: prune shards older than this so the estimator (which reads
        # every shard each cycle) and disk usage stay bounded. 0 disables.
        try:
            self.retention_minutes = float(os.getenv("RETENTION_MINUTES", "60"))
        except ValueError:
            self.retention_minutes = 60.0
        # Compaction: merge shards older than this into one file per hour. A 10s
        # flush writes ~360 shards/hour/stream, and for the 83-column orderbook
        # schema that is pathological -- per-column metadata and the parquet
        # footer dwarf ~2 rows of payload, measured at 27,343 bytes/row against
        # 664 bytes of actual float64. Compacting an hour of ETH orderbooks:
        # 18.39 MB -> 0.27 MB (68x) and read time 2199ms -> 6ms. Must stay well
        # above the estimator's window edge so the live tail is never rewritten
        # underneath a reader. 0 disables.
        try:
            self.compact_after_minutes = float(os.getenv("COMPACT_AFTER_MINUTES", "15"))
        except ValueError:
            self.compact_after_minutes = 15.0
        self._reconnecting = False
        
        # Ensure output directory exists and organize by symbol/type
        self._init_storage()
    
    def _init_storage(self):
        """Initialize directory structure for data storage"""
        for symbol in self.symbols:
            for dtype in STREAMS:
                path = os.path.join(self.output_dir, symbol, dtype)
                os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def narrow_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Narrow columns to the width the data actually needs, before writing.

        Everything arrives as float64/object because that is what json + pandas
        produce, but not all of it needs 8 bytes:

        - Sizes are order quantities -- float32 carries ~7 significant digits,
          far more than any venue's lot precision.
        - ``side`` is two distinct strings; dictionary-encoded it costs bits per
          row instead of bytes.

        Two columns are deliberately left alone:

        - PRICES stay float64. float32's ~7 significant digits are not enough for
          BTC near 62,929.5 to preserve a 0.1 tick, and every depth and spread
          measurement in the project is a difference of prices.
        - ``timestamp`` stays float SECONDS. Milliseconds would be narrower, but
          estimator_common._ts_ms_from multiplies this column by 1000 to reach
          ms, get_lambda parses it with unit='s', and shards already on disk use
          seconds -- changing the unit would put two incompatible conventions in
          the same directory with nothing to tell them apart. Sorted float64
          timestamps compress well under zstd once compaction puts many rows in
          one file, which is where the real win is anyway.
        """
        for col in df.columns:
            try:
                if col == "side":
                    df[col] = df[col].astype("category")
                elif "size" in col and df[col].dtype == "float64":
                    values = df[col].to_numpy(dtype="float64", copy=False)
                    finite = values[np.isfinite(values)]
                    # float32 represents integers exactly only below 2**24. Sizes
                    # on cheap assets are large integers -- PENGU already trades
                    # 7.0M-unit orders against that 16.8M ceiling -- so narrow
                    # only when the whole column is comfortably inside it and
                    # keep float64 otherwise. Measured across every collected
                    # symbol, integer sizes round-trip exactly and fractional
                    # ones lose at most 6e-8 relative.
                    if finite.size == 0 or np.abs(finite).max() < FLOAT32_EXACT_INT_LIMIT:
                        df[col] = df[col].astype("float32")
            except (TypeError, ValueError, OverflowError):
                # A surprise value must not cost us the whole shard.
                continue
        return df

    def _write_to_parquet(self, symbol: str, dtype: str, data: List[Any]):
        """Write data to Parquet file using Pandas"""
        if not data:
            return

        try:
            # Convert to DataFrame
            # For list of objects (TickData, TradeData etc) or list of dicts
            if len(data) > 0 and not isinstance(data[0], dict):
                df = pd.DataFrame([asdict(item) for item in data])
            else:
                df = pd.DataFrame(data)

            df = self.narrow_dtypes(df)

            # Generate filename with timestamp
            timestamp = int(time.time() * 1000)
            filename = f"{dtype}_{timestamp}.parquet"
            file_path = os.path.join(self.output_dir, symbol, dtype, filename)

            # Write to a temporary name and rename into place. Readers glob
            # "*.parquet", which never matches the ".parquet.tmp" suffix, and
            # os.replace is atomic within a filesystem -- so a reader either sees
            # a complete shard or no shard, never a half-written one.
            #
            # Writing straight to the final path used to hand every reader a
            # window in which the file existed but its footer did not:
            # estimator_common._load_parquet_dir swallowed the failure and
            # silently DROPPED the shard (losing the newest data and biasing
            # n_trades / lambda down), and the strategy logged
            # collector_data_read_error with "Parquet magic bytes not found in
            # footer".
            tmp_path = file_path + ".tmp"
            try:
                df.to_parquet(tmp_path, engine='pyarrow', index=False, compression='zstd')
                os.replace(tmp_path, file_path)
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            print(f"Error writing to Parquet {symbol}/{dtype}: {e}")

    # _handle_bbo_data, _handle_trade_data, _handle_orderbook_data remain unchanged
    
    def _handle_bbo_data(self, data: Dict[str, Any]):
        """Handle best bid/offer data"""
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'bbo' and 'data' in data:
                bbo_data = data['data']
                symbol = bbo_data.get('coin', 'UNKNOWN')
                
                # BBO format: bbo array with [bid, ask]
                if 'bbo' in bbo_data and len(bbo_data['bbo']) >= 2:
                    bid_info = bbo_data['bbo'][0]  # First element is bid
                    ask_info = bbo_data['bbo'][1]  # Second element is ask
                    
                    # Create bid data
                    bid_data = {
                        'timestamp': timestamp,
                        'price': float(bid_info['px']),
                        'size': float(bid_info['sz']),
                        'side': 'bid',
                        'exchange_timestamp': bbo_data.get('time')
                    }
                    
                    # Create ask data
                    ask_data = {
                        'timestamp': timestamp,
                        'price': float(ask_info['px']),
                        'size': float(ask_info['sz']),
                        'side': 'ask',
                        'exchange_timestamp': bbo_data.get('time')
                    }

                    with self.lock:
                        self.data_buffers[symbol]['prices'].append(bid_data)
                        self.data_buffers[symbol]['prices'].append(ask_data)
                
                self.stats.update('bbo_updates')
            else:
                # Direct format fallback
                symbol = data.get('coin', 'UNKNOWN')
                
                with self.lock:
                    if 'bid' in data and data['bid']:
                        bid_data = {
                            'timestamp': timestamp,
                            'price': float(data['bid']['px']),
                            'size': float(data['bid']['sz']),
                            'side': 'bid',
                            'exchange_timestamp': data.get('time')
                        }
                        self.data_buffers[symbol]['prices'].append(bid_data)
                    
                    if 'ask' in data and data['ask']:
                        ask_data = {
                            'timestamp': timestamp,
                            'price': float(data['ask']['px']),
                            'size': float(data['ask']['sz']),
                            'side': 'ask',
                            'exchange_timestamp': data.get('time')
                        }
                        self.data_buffers[symbol]['prices'].append(ask_data)
                
                self.stats.update('bbo_updates')
        except Exception as e:
            print(f"Error handling BBO data: {e}")
    
    def _handle_trade_data(self, data: Dict[str, Any]):
        """Handle trade data"""
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'trades' and 'data' in data:
                trades = data['data']
                for trade in trades:
                    symbol = trade.get('coin', 'UNKNOWN')
                    # Convert A/B to buy/sell
                    side = 'sell' if trade['side'] == 'A' else 'buy'
                    
                    trade_data = {
                        'timestamp': timestamp,
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': side,
                        'trade_id': str(trade.get('tid')),
                        'exchange_timestamp': trade.get('time')
                    }
                    with self.lock:
                        self.data_buffers[symbol]['trades'].append(trade_data)
                
                self.stats.update('trades', len(trades))
            else:
                # Direct format fallback  
                if isinstance(data, list):
                    trades = data
                else:
                    trades = [data]
                    
                for trade in trades:
                    symbol = trade.get('coin', 'UNKNOWN')
                    side = 'sell' if trade['side'] == 'A' else 'buy'
                    
                    trade_data = {
                        'timestamp': timestamp,
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': side,
                        'trade_id': str(trade.get('tid')),
                        'exchange_timestamp': trade.get('time')
                    }
                    with self.lock:
                        self.data_buffers[symbol]['trades'].append(trade_data)
                
                self.stats.update('trades', len(trades))
        except Exception as e:
            print(f"Error handling trade data: {e}")
    
    def _handle_orderbook_data(self, data: Dict[str, Any]):
        """Handle order book data"""
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'l2Book' and 'data' in data:
                book_data = data['data']
                symbol = book_data.get('coin', 'UNKNOWN')
                
                # Parse bids and asks
                bids = []
                asks = []
                
                if 'levels' in book_data and len(book_data['levels']) >= 2:
                    # levels[0] is bids array, levels[1] is asks array
                    bids_array = book_data['levels'][0]
                    asks_array = book_data['levels'][1]
                    
                    # Parse bids
                    for bid in bids_array:
                        bids.append(OrderBookLevel(price=float(bid['px']), size=float(bid['sz'])))
                    
                    # Parse asks  
                    for ask in asks_array:
                        asks.append(OrderBookLevel(price=float(ask['px']), size=float(ask['sz'])))
                
                # Prepare flattened orderbook row (configurable depth levels)
                csv_row = {
                    'timestamp': timestamp,
                    'sequence': book_data.get('time'),
                    'exchange_timestamp': book_data.get('time')
                }
                
                for i in range(self.orderbook_depth):
                    if i < len(bids):
                        csv_row[f'bid_price_{i}'] = bids[i].price
                        csv_row[f'bid_size_{i}'] = bids[i].size
                    else:
                        csv_row[f'bid_price_{i}'] = None
                        csv_row[f'bid_size_{i}'] = None
                    
                    if i < len(asks):
                        csv_row[f'ask_price_{i}'] = asks[i].price
                        csv_row[f'ask_size_{i}'] = asks[i].size
                    else:
                        csv_row[f'ask_price_{i}'] = None
                        csv_row[f'ask_size_{i}'] = None
                
                with self.lock:
                    self.data_buffers[symbol]['orderbooks'].append(csv_row)
                self.stats.update('orderbook_updates')
            else:
                # Direct format fallback
                symbol = data.get('coin', 'UNKNOWN')
                
                # Parse bids and asks
                bids = []
                asks = []
                
                if 'levels' in data and len(data['levels']) >= 2:
                    # levels[0] is bids array, levels[1] is asks array
                    bids_array = data['levels'][0]
                    asks_array = data['levels'][1]
                    
                    # Parse bids
                    for bid in bids_array:
                        bids.append(OrderBookLevel(price=float(bid['px']), size=float(bid['sz'])))
                    
                    # Parse asks  
                    for ask in asks_array:
                        asks.append(OrderBookLevel(price=float(ask['px']), size=float(ask['sz'])))
                
                # Prepare flattened orderbook row (configurable depth levels)
                csv_row = {
                    'timestamp': timestamp,
                    'sequence': data.get('time'),
                    'exchange_timestamp': data.get('time')
                }
                
                for i in range(self.orderbook_depth):
                    if i < len(bids):
                        csv_row[f'bid_price_{i}'] = bids[i].price
                        csv_row[f'bid_size_{i}'] = bids[i].size
                    else:
                        csv_row[f'bid_price_{i}'] = None
                        csv_row[f'bid_size_{i}'] = None
                    
                    if i < len(asks):
                        csv_row[f'ask_price_{i}'] = asks[i].price
                        csv_row[f'ask_size_{i}'] = asks[i].size
                    else:
                        csv_row[f'ask_price_{i}'] = None
                        csv_row[f'ask_size_{i}'] = None
                
                with self.lock:
                    self.data_buffers[symbol]['orderbooks'].append(csv_row)
                self.stats.update('orderbook_updates')
            
        except Exception as e:
            print(f"Error handling order book data: {e}")

    def _handle_asset_ctx_data(self, data: Dict[str, Any]):
        """activeAssetCtx: oracle/mark/mid price, open interest, funding, premium.

        Recorded for the oracle-dislocation / OI-drop question deferred in
        docs/FLOW_GUARD_CANDIDATES.md: oraclePx-vs-midPx divergence is what
        distinguishes "the perp dislocated alone and will mean-revert" from
        "the whole market repriced", and openInterest collapsing is forced
        unwinding by definition. The 08-22 CASHCAT cascade was confirmed
        idiosyncratic from sibling tapes, but the signal itself could never be
        backtested because nothing recorded this channel -- this fixes that
        going forward.

        The venue pushes an unchanged ctx roughly every second, so a row is
        kept only when any tracked field changes, plus a 60 s heartbeat row so
        a quiet stream stays distinguishable from a dead subscription. The
        channel carries no exchange timestamp; `timestamp` is local receive
        time in float seconds like every other stream.
        """
        try:
            if not (isinstance(data, dict) and data.get('channel') == 'activeAssetCtx' and 'data' in data):
                return
            payload = data['data']
            symbol = payload.get('coin', 'UNKNOWN')
            if symbol not in self.data_buffers:
                return
            ctx = payload.get('ctx') or {}

            def _field(key):
                value = ctx.get(key)
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            impact = ctx.get('impactPxs') or []
            try:
                impact_bid = float(impact[0]) if len(impact) > 0 and impact[0] is not None else None
                impact_ask = float(impact[1]) if len(impact) > 1 and impact[1] is not None else None
            except (TypeError, ValueError):
                impact_bid = impact_ask = None

            now = time.time()
            row = {
                'timestamp': now,
                'oracle_px': _field('oraclePx'),
                'mark_px': _field('markPx'),
                'mid_px': _field('midPx'),
                'open_interest': _field('openInterest'),
                'funding': _field('funding'),
                'premium': _field('premium'),
                'impact_bid_px': impact_bid,
                'impact_ask_px': impact_ask,
                'day_ntl_vlm': _field('dayNtlVlm'),
            }
            signature = (
                row['oracle_px'], row['mark_px'], row['mid_px'],
                row['open_interest'], row['funding'], row['premium'],
            )
            last_signature, last_kept = self._asset_ctx_last.get(symbol, (None, 0.0))
            if signature == last_signature and now - last_kept < 60.0:
                return
            self._asset_ctx_last[symbol] = (signature, now)
            with self.lock:
                self.data_buffers[symbol]['asset_ctx'].append(row)
            self.stats.update('asset_ctx_updates')
        except Exception as e:
            print(f"Error handling asset ctx data: {e}")

    def _flush_buffers(self):
        """Flush data buffers to Parquet files"""
        try:
            # First, snapshot and clear buffers within the lock
            data_to_write = {}
            
            with self.lock:
                for symbol in self.symbols:
                    symbol_buffers = self.data_buffers[symbol]
                    data_to_write[symbol] = {}
                    for dtype in STREAMS:
                        if symbol_buffers[dtype]:
                            data_to_write[symbol][dtype] = list(symbol_buffers[dtype])
                            symbol_buffers[dtype].clear()
            
            # Then write to files (outside the lock)
            flushed_count = 0
            for symbol, buffers in data_to_write.items():
                for dtype in STREAMS:
                    if dtype in buffers:
                        self.executor.submit(self._write_to_parquet, symbol, dtype, buffers[dtype])
                        flushed_count += 1
            
            if flushed_count > 0:
                print(f"Flushed buffers for {flushed_count} data types across symbols")
                
        except Exception as e:
            print(f"Error flushing buffers: {e}")
    
    def _print_summary(self):
        """Print data collection summary"""
        summary = self.stats.get_summary()
        print("\n" + "="*60)
        print(f"DATA COLLECTION SUMMARY - {summary['last_update']}")
        print("="*60)
        print(f"Runtime: {summary['runtime_formatted']}")
        print(f"Data collected:")
        for data_type, count in summary['counters'].items():
            rate = summary['rates_per_minute'].get(data_type, 0)
            print(f"  {data_type}: {count:,} ({rate:.1f}/min)")
        
        print(f"\nBuffer sizes by symbol:")
        
        # Calculate buffer sizes with lock to ensure consistency
        with self.lock:
            for symbol in self.symbols:
                symbol_buffers = self.data_buffers[symbol]
                total_buffered = sum(len(buffer) for buffer in symbol_buffers.values())
                detail = ", ".join(f"{len(symbol_buffers[d])} {d}" for d in STREAMS)
                print(f"  {symbol}: {total_buffered} ({detail})")
        
        print("="*60)

    def _subscribe_all(self):
        """(Re)subscribe to all data feeds for all symbols."""
        self.subscription_ids = []
        for symbol in self.symbols:
            print(f"Subscribing to data feeds for {symbol}...")

            bbo_id = self.info.subscribe(
                {"type": "bbo", "coin": symbol},
                self._handle_bbo_data
            )
            self.subscription_ids.append(bbo_id)

            trades_id = self.info.subscribe(
                {"type": "trades", "coin": symbol},
                self._handle_trade_data
            )
            self.subscription_ids.append(trades_id)

            l2book_id = self.info.subscribe(
                {"type": "l2Book", "coin": symbol},
                self._handle_orderbook_data
            )
            self.subscription_ids.append(l2book_id)

            # Same WebSocket, one more multiplexed subscription -- this does
            # not consume another connection against the 10-per-IP budget.
            asset_ctx_id = self.info.subscribe(
                {"type": "activeAssetCtx", "coin": symbol},
                self._handle_asset_ctx_data
            )
            self.subscription_ids.append(asset_ctx_id)

        print(f"Subscribed to {len(self.subscription_ids)} data feeds")

    def _reconnect(self):
        """Reconnect websocket and resubscribe."""
        if self._reconnecting:
            return
        self._reconnecting = True
        try:
            print("Reconnecting Hyperliquid websocket...")
            try:
                self.info.disconnect_websocket()
            except Exception as e:
                print(f"Error disconnecting websocket during reconnect: {e}")
            time.sleep(self.reconnect_backoff_sec)
            self.info = Info(constants.MAINNET_API_URL, skip_ws=False)
            self._subscribe_all()
            self._last_connect_time = time.time()
            print("Reconnected successfully.")
        finally:
            self._reconnecting = False

    def _websocket_is_down(self) -> bool:
        """True when the SDK's websocket is provably gone.

        Hyperliquid expires a websocket session every few hours and sends a close
        frame ("Expired"); the SDK logs it and its manager thread exits. Nothing in
        the SDK reconnects. Before this probe existed, the only recovery was the
        inactivity timer below, so every routine server-side expiry cost a full
        INACTIVITY_TIMEOUT_SEC of missing data -- measured at 20 gaps of 3.1-3.5
        min over 60 h of CASHCAT, about 1.8% of the tape, on a clockwork ~3 h
        cadence. The close is knowable within seconds, so act on it.
        """
        manager = getattr(self.info, "ws_manager", None)
        if manager is None:
            return False  # skip_ws mode; nothing to watch
        if not manager.is_alive():
            return True
        ws = getattr(manager, "ws", None)
        sock = getattr(ws, "sock", None) if ws is not None else None
        return sock is None or not getattr(sock, "connected", False)

    def _watchdog_inactivity(self):
        """Watch for a dead socket or a long silence, and recover from either."""
        stale_attempts = 0
        down_readings = 0
        while self.running:
            time.sleep(5)

            # Fast path: the socket itself is closed. Require two consecutive
            # readings and a grace period since the last connect so a slow
            # handshake is never mistaken for a failure.
            socket_down = False
            if not self._reconnecting and (time.time() - self._last_connect_time) > self.ws_health_grace_sec:
                try:
                    socket_down = self._websocket_is_down()
                except Exception as e:
                    print(f"Websocket health probe failed: {e}")
                    socket_down = False
            down_readings = down_readings + 1 if socket_down else 0
            if down_readings >= 2:
                down_readings = 0
                stale_attempts += 1
                print(
                    f"Websocket closed. Reconnect attempt "
                    f"{stale_attempts}/{self.max_reconnect_attempts}..."
                )
                try:
                    self._reconnect()
                    stale_attempts = 0
                    self.stats.last_update = time.time()
                    continue
                except Exception as e:
                    print(f"Reconnect attempt failed: {e}")
                    if stale_attempts >= self.max_reconnect_attempts:
                        print("Max reconnect attempts reached. Flushing buffers and exiting for Docker restart.")
                        try:
                            self._flush_buffers()
                        except Exception as flush_e:
                            print(f"Flush before exit failed: {flush_e}")
                        os._exit(1)
                    continue

            # Slow path, unchanged: the socket looks fine but nothing is arriving.
            since_last = time.time() - self.stats.last_update
            if since_last <= self.inactivity_timeout_sec:
                stale_attempts = 0
                continue

            stale_attempts += 1
            print(
                f"No data for {since_last:.0f}s (> {self.inactivity_timeout_sec}s). "
                f"Reconnect attempt {stale_attempts}/{self.max_reconnect_attempts}..."
            )
            try:
                self._reconnect()
                stale_attempts = 0
                # Avoid immediate re-trigger after a reconnect
                self.stats.last_update = time.time()
            except Exception as e:
                print(f"Reconnect attempt failed: {e}")
                if stale_attempts >= self.max_reconnect_attempts:
                    print("Max reconnect attempts reached. Flushing buffers and exiting for Docker restart.")
                    try:
                        self._flush_buffers()
                    except Exception as flush_e:
                        print(f"Flush before exit failed: {flush_e}")
                    os._exit(1)
    
    def start_collection(self):
        """Start data collection"""
        print(f"Starting Hyperliquid data collection for symbols: {self.symbols}")
        print(f"Output directory: {self.output_dir}")
        
        self.running = True
        
        try:
            # Subscribe to data feeds for each symbol
            self._subscribe_all()
            
            # Start background tasks
            flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
            flush_thread.start()
            
            summary_thread = threading.Thread(target=self._periodic_summary, daemon=True)
            summary_thread.start()

            watchdog_thread = threading.Thread(target=self._watchdog_inactivity, daemon=True)
            watchdog_thread.start()
            
            # Keep running
            print("Data collection started. Press Ctrl+C to stop.")
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_collection()
        except Exception as e:
            print(f"Error during data collection: {e}")
            self.stop_collection()
    
    def _prune_old_shards(self):
        """Delete parquet shards older than the retention window.

        Shard age is derived from the millisecond timestamp embedded in the
        filename ("<dtype>_<ms>.parquet"), not os.stat(): statting thousands of
        files every cycle is pathologically slow on bind mounts and would stall
        the flush loop. Falls back to mtime only if the name can't be parsed.
        """
        if self.retention_minutes <= 0:
            return
        cutoff_ms = (time.time() - self.retention_minutes * 60.0) * 1000.0
        removed = 0
        for symbol in self.symbols:
            for dtype in STREAMS:
                directory = os.path.join(self.output_dir, symbol, dtype)
                if not os.path.isdir(directory):
                    continue
                try:
                    entries = os.listdir(directory)
                except OSError:
                    continue
                for name in entries:
                    # Sweep up .tmp files abandoned by a write that died between
                    # to_parquet and os.replace (SIGKILL, disk full). Readers
                    # ignore them, but nothing else would ever remove them.
                    if name.endswith('.parquet.tmp'):
                        try:
                            tmp_path = os.path.join(directory, name)
                            if os.path.getmtime(tmp_path) * 1000.0 < cutoff_ms:
                                os.remove(tmp_path)
                                removed += 1
                        except OSError:
                            pass
                        continue
                    if not name.endswith('.parquet'):
                        continue
                    stem = name[:-len('.parquet')]
                    ts_ms = None
                    if '_' in stem:
                        try:
                            ts_ms = float(stem.rsplit('_', 1)[1])
                        except ValueError:
                            ts_ms = None
                    path = os.path.join(directory, name)
                    try:
                        if ts_ms is None:
                            ts_ms = os.path.getmtime(path) * 1000.0
                        if ts_ms < cutoff_ms:
                            os.remove(path)
                            removed += 1
                    except OSError:
                        continue
        if removed:
            print(f"Pruned {removed} parquet shards older than {self.retention_minutes:.0f} min")

    def _compact_old_shards(self):
        """Merge settled shards into one file per hour.

        A 10s flush is right for freshness and wrong for storage: it writes ~360
        files per hour per stream, and a columnar format carrying 83 columns of
        schema and statistics for ~2 rows of payload spends 27,343 bytes/row on
        664 bytes of float64. Measured on an hour of ETH orderbooks, merging
        those shards takes 18.39 MB to 0.27 MB and the estimator's read of that
        directory from 2199ms to 6ms.

        Only shards older than ``compact_after_minutes`` are touched, so the
        window the estimators actually quote from is never rewritten under a
        reader.

        The compacted file is named for the NEWEST shard it absorbs, which keeps
        it selectable by estimator_common.select_shards_for_window (that cutoff
        compares against the newest timestamp) and prunable by the same
        filename-timestamp rule as any other shard.

        Ordering is write-then-delete, so a reader mid-scan can briefly see both
        the compacted file and its sources. Every consumer already collapses
        that: trades de-duplicate on trade_id, build_bbo_mid pivots on ts_ms and
        build_orderbook_mid drops duplicate ts_ms.
        """
        if self.compact_after_minutes <= 0:
            return
        cutoff_ms = (time.time() - self.compact_after_minutes * 60.0) * 1000.0
        compacted_files = 0
        reclaimed = 0

        for symbol in self.symbols:
            for dtype in STREAMS:
                directory = os.path.join(self.output_dir, symbol, dtype)
                if not os.path.isdir(directory):
                    continue
                try:
                    entries = os.listdir(directory)
                except OSError:
                    continue

                buckets: Dict[int, List[tuple]] = defaultdict(list)
                for name in entries:
                    if not name.endswith('.parquet') or '_compact_' in name:
                        continue
                    stem = name[:-len('.parquet')]
                    if '_' not in stem:
                        continue
                    try:
                        ts_ms = float(stem.rsplit('_', 1)[1])
                    except ValueError:
                        continue
                    if ts_ms >= cutoff_ms:
                        continue
                    hour_bucket = int(ts_ms // 3_600_000)
                    buckets[hour_bucket].append((ts_ms, os.path.join(directory, name)))

                for hour_bucket, items in buckets.items():
                    if len(items) < 2:
                        continue  # nothing to gain
                    items.sort()
                    paths = [path for _ts, path in items]
                    newest_ms = int(items[-1][0])
                    try:
                        frames = []
                        for path in paths:
                            try:
                                frames.append(pd.read_parquet(path))
                            except Exception as exc:
                                print(f"Compaction: skipping unreadable {path}: {exc}")
                        if not frames:
                            continue
                        merged = pd.concat(frames, ignore_index=True)
                        if 'timestamp' in merged.columns:
                            merged = merged.sort_values('timestamp', kind='stable')
                        merged = merged.reset_index(drop=True)
                        merged = self.narrow_dtypes(merged)

                        before = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
                        out_name = f"{dtype}_compact_{newest_ms}.parquet"
                        out_path = os.path.join(directory, out_name)
                        tmp_path = out_path + '.tmp'
                        merged.to_parquet(
                            tmp_path, engine='pyarrow', index=False, compression='zstd'
                        )
                        os.replace(tmp_path, out_path)

                        for path in paths:
                            try:
                                os.remove(path)
                            except OSError:
                                pass
                        after = os.path.getsize(out_path)
                        compacted_files += len(paths)
                        reclaimed += max(0, before - after)
                    except Exception as exc:
                        print(f"Compaction failed for {directory} hour {hour_bucket}: {exc}")
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except (OSError, UnboundLocalError):
                            pass

        if compacted_files:
            print(
                f"Compacted {compacted_files} shards, reclaimed {reclaimed / 1e6:.1f} MB"
            )

    def _periodic_flush(self):
        """Periodically flush buffers to disk; prune occasionally."""
        last_prune = 0.0
        last_compact = 0.0
        while self.running:
            time.sleep(self.flush_interval_sec)
            self._flush_buffers()
            # Pruning scans whole directories, so throttle it to ~once a minute
            # rather than running it on every flush.
            if time.time() - last_prune >= 60.0:
                self._prune_old_shards()
                last_prune = time.time()
            # Compaction rewrites whole hours, so it runs far less often still.
            # Every 5 minutes is ample: shards only become eligible once they are
            # compact_after_minutes old.
            if time.time() - last_compact >= 300.0:
                self._compact_old_shards()
                last_compact = time.time()
    
    def _periodic_summary(self):
        """Periodically print collection summary"""
        while self.running:
            time.sleep(30)  # Print summary every 30 seconds
            if self.running:
                self._print_summary()
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        
        # Unsubscribe from all feeds - SDK Info class manages this mostly via stop() but explicit unsubscribe would be here if needed
        # Since we are shutting down, we'll rely on disconnect_websocket
        
        # Final flush
        self._flush_buffers()
        
        # Wait for executor to finish
        self.executor.shutdown(wait=True)
        
        # Disconnect websocket
        try:
            self.info.disconnect_websocket()
        except Exception as e:
            print(f"Error disconnecting websocket: {e}")
        
        # Final summary
        self._print_summary()
        print(f"\nData files saved in: {self.output_dir}")


def main():
    """Main function"""
    # Configuration
    SYMBOLS = ["CASHCAT"]  # Add more symbols as needed
    OUTPUT_DIR = "HL_data"
    ORDERBOOK_DEPTH = 20  # Number of order book levels to capture (default: 20)
    
    print("Hyperliquid Tick Data Collector")
    print("================================")
    print(f"Order book depth: {ORDERBOOK_DEPTH} levels")
    
    # Create collector with configurable order book depth
    collector = HyperliquidDataCollector(SYMBOLS, OUTPUT_DIR, orderbook_depth=ORDERBOOK_DEPTH)
    
    # Start collection
    collector.start_collection()


if __name__ == "__main__":
    main()
