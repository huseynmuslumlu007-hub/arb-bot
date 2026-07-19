"""Collect historical OHLCV candles from Coinbase and Kraken, save them to CSV.

Fetches the past 90 days of 1-hour candles for BTC/USD and ETH/USD from each
exchange (public endpoints, no API key) and writes one CSV per
exchange+symbol into ``data_store/``.

Storage lives here, not in ``MarketFeed``: the feed's job is collecting and
normalizing market data, so the multi-page OHLCV fetch is a method on the feed
(``MarketFeed.fetch_ohlcv``), while turning candles into files is this script's
job. That keeps the data-source layer free of any opinion about how the data
gets persisted. ``fetch_ohlcv`` is exchange-agnostic, so Kraken reuses it with
no change.

Note: Kraken's public OHLC endpoint serves at most ~720 of the *most recent*
candles per interval regardless of ``since`` — so at 1h it yields ~30 days, not
90. That's an API cap, not a fetch bug; Kraken CSVs cover a shorter window than
Coinbase's. Deeper Kraken history would need a different source (e.g. its
trades endpoint aggregated into candles).

Run from the project root:
    python -m scripts.collect_history
"""

import asyncio
from pathlib import Path

import pandas as pd

from data.feed import MarketFeed

EXCHANGES = ("coinbase", "kraken")
SYMBOLS = ("BTC/USD", "ETH/USD")
TIMEFRAME = "1h"
DAYS = 90
OUTPUT_DIR = Path("data_store")

# OHLCV rows from ccxt are [timestamp_ms, open, high, low, close, volume].
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _csv_path(exchange: str, symbol: str) -> Path:
    """data_store/coinbase_BTC-USD_1h.csv for ('coinbase', 'BTC/USD')."""
    safe_symbol = symbol.replace("/", "-")
    return OUTPUT_DIR / f"{exchange}_{safe_symbol}_{TIMEFRAME}.csv"


def _save_candles(exchange: str, symbol: str, candles: list[list[float]]) -> Path:
    df = pd.DataFrame(candles, columns=COLUMNS)
    # Keep the raw epoch ms and add a human-readable UTC datetime column.
    df.insert(1, "datetime", pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    path = _csv_path(exchange, symbol)
    df.to_csv(path, index=False)
    return path


async def collect_exchange(exchange_id: str) -> None:
    """Fetch and save both symbols' 90-day 1h history from one exchange."""
    async with MarketFeed(exchange_id) as feed:
        # 'since' in epoch ms; the exchange's own clock avoids local-clock skew.
        since = feed.exchange.milliseconds() - DAYS * 24 * 60 * 60 * 1000
        # Symbols sequentially within an exchange so ccxt's per-instance rate
        # limiter paces the paginated requests.
        for symbol in SYMBOLS:
            candles = await feed.fetch_ohlcv(symbol, TIMEFRAME, since)
            path = _save_candles(exchange_id, symbol, candles)
            print(f"{exchange_id} {symbol}: {len(candles)} candles -> {path}")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Each exchange has its own MarketFeed (own session + rate limiter), so the
    # exchanges run concurrently without contending on a shared limiter.
    await asyncio.gather(*(collect_exchange(ex) for ex in EXCHANGES))


if __name__ == "__main__":
    asyncio.run(main())
