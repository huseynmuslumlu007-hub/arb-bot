"""Compare a live ticker across two exchanges side by side.

Fetches BTC/USD from Coinbase and Kraken *concurrently* — both requests are
in flight at the same time via ``asyncio.gather``, not one after the other —
so the two prices are as close to simultaneous as the network allows. That
matters for arbitrage: comparing prices pulled seconds apart would confuse a
stale quote for a real cross-exchange spread.

Run from the project root:
    python -m scripts.compare_feed
"""

import asyncio

from data.feed import MarketFeed

SYMBOL = "BTC/USD"
EXCHANGES = ("coinbase", "kraken")


async def main() -> None:
    feeds = [MarketFeed(exchange_id) for exchange_id in EXCHANGES]
    try:
        tickers = await asyncio.gather(
            *(feed.fetch_ticker(SYMBOL) for feed in feeds)
        )
        for feed, ticker in zip(feeds, tickers):
            print(f"{SYMBOL} on {feed.exchange_id:<10} last: {ticker['last']}")
    finally:
        # Close every session even if one fetch failed above.
        await asyncio.gather(*(feed.close() for feed in feeds))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
