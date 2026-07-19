"""Tests for the market feed."""

import asyncio

import pytest

from data.feed import MarketFeed


def test_market_feed_stores_exchange_id():
    feed = MarketFeed("coinbase")
    assert feed.exchange_id == "coinbase"
    asyncio.run(feed.close())


def test_fetch_ticker_rejects_unknown_symbol():
    """An unknown symbol must fail loudly instead of hitting the API."""

    async def scenario():
        async with MarketFeed("coinbase") as feed:
            # Stub out network calls: load_markets populates a known set, and
            # fetch_ticker would raise if the guard ever lets an unknown
            # symbol through to the actual request.
            async def fake_load_markets(*args, **kwargs):
                feed.exchange.markets = {"BTC/USD": {}}
                return feed.exchange.markets

            async def fail_fetch(*args, **kwargs):
                raise AssertionError("fetch_ticker reached the API for a bad symbol")

            feed.exchange.load_markets = fake_load_markets
            feed.exchange.fetch_ticker = fail_fetch

            with pytest.raises(ValueError, match="not a market"):
                await feed.fetch_ticker("BTC/NOPE")

    asyncio.run(scenario())
