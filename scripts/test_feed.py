"""Manual smoke test: poll a live ticker from Coinbase (public endpoint).

Prints the BTC/USD price every few seconds until interrupted with Ctrl+C.

Run from the project root:
    python -m scripts.test_feed
"""

import asyncio

from data.feed import MarketFeed

SYMBOL = "BTC/USD"
INTERVAL_SECONDS = 5


async def main() -> None:
    async with MarketFeed("coinbase") as feed:
        while True:
            ticker = await feed.fetch_ticker(SYMBOL)
            print(f"{SYMBOL} on coinbase  last: {ticker['last']}")
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
