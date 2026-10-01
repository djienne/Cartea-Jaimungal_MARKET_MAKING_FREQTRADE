"""Offline fast-subscription and raw book/trade persistence check; no network."""
import sys
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hyperliquid_data_collector import DataStats, HyperliquidDataCollector


class FastCaptureCheck(unittest.TestCase):
    def test_subscription_and_published_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            c = HyperliquidDataCollector.__new__(HyperliquidDataCollector)
            c.symbols, c.output_dir, c.orderbook_depth = ['ETH'], directory, 20
            c.info, c.stats, c.lock = Mock(), DataStats(), threading.Lock()
            c.data_buffers = {'ETH': {'orderbooks': deque(), 'trades': deque()}}
            c._init_storage()
            for _ in range(2):  # reconnect follows the same subscription path
                c._subscribe_all()
            subscriptions = [call.args[0] for call in c.info.subscribe.call_args_list]
            self.assertEqual(sum(s == {'type': 'l2Book', 'coin': 'ETH', 'fast': True} for s in subscriptions), 2)
            self.assertEqual(sum(s == {'type': 'trades', 'coin': 'ETH'} for s in subscriptions), 2)
            for depth in (5, 20):
                levels = [[{'px': str(100 + sign * (i + 1)), 'sz': '2'} for i in range(depth)]
                          for sign in (-1, 1)]
                c._handle_orderbook_data({'channel': 'l2Book', 'data': {'coin': 'ETH', 'time': depth, 'levels': levels}})
                with patch('hyperliquid_data_collector.time.time', return_value=1000 + depth):
                    c._write_to_parquet('ETH', 'orderbooks', [c.data_buffers['ETH']['orderbooks'].pop()])
            books = pq.read_table(Path(directory) / 'ETH/orderbooks').to_pandas().sort_values('exchange_timestamp')
            self.assertEqual(len(books), 2)
            self.assertTrue(pd.isna(books.iloc[0].bid_price_5))
            self.assertEqual(books.iloc[1].bid_price_19, 80.)
            c._handle_trade_data({'channel': 'trades', 'data': [
                {'coin': 'ETH', 'time': 1234, 'tid': 77, 'side': 'A', 'px': '100', 'sz': '.5'}]})
            c._write_to_parquet('ETH', 'trades', list(c.data_buffers['ETH']['trades']))
            trade = pd.read_parquet(Path(directory) / 'ETH/trades').iloc[0]
            self.assertEqual((trade.exchange_timestamp, trade.trade_id, trade.side, trade['size']), (1234, '77', 'sell', .5))
            self.assertGreater(trade.timestamp, 0)


if __name__ == '__main__':
    unittest.main()
