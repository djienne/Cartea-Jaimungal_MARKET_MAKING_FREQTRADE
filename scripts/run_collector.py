import os
import signal
from hyperliquid_data_collector import HyperliquidDataCollector


def _install_sigterm_handler():
    """Route SIGTERM into the existing Ctrl-C shutdown path.

    This process is PID 1 in the container, and PID 1 gets no default signal
    handlers -- so without this, `docker stop`, `docker restart` and the
    autoheal watchdog all wait out the full stop timeout and then SIGKILL. A
    SIGKILL skips `stop_collection()`, which means the final `_flush_buffers()`
    never runs and everything still in memory (up to one flush interval of
    prices, trades and orderbooks) is lost on every restart.

    `start_collection` already catches KeyboardInterrupt and calls
    `stop_collection()`, so raising it here reuses the tested shutdown path
    instead of adding a second one. The main loop sits in `time.sleep(1)`,
    which the signal interrupts.
    """

    def _handle(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle)

def _symbols():
    raw = os.getenv("SYMBOLS", "ETH")
    return [s.strip() for s in raw.split(",") if s.strip()]

def main():
    symbols = _symbols()
    output_dir = os.getenv("OUTPUT_DIR", "HL_data")
    try:
        orderbook_depth = int(os.getenv("ORDERBOOK_DEPTH", "20"))
    except ValueError:
        orderbook_depth = 20

    print("Hyperliquid Tick Data Collector (Docker)")
    print("========================================")
    print(f"Symbols:         {symbols}")
    print(f"Output dir:      {output_dir}")
    print(f"Orderbook depth: {orderbook_depth}")

    _install_sigterm_handler()

    collector = HyperliquidDataCollector(
        symbols, output_dir, orderbook_depth=orderbook_depth
    )
    collector.start_collection()

if __name__ == "__main__":
    main()
