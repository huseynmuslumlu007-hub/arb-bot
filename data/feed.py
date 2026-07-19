"""Market feed skeleton.

Fetches raw market data from exchanges. No trading or arbitrage logic here —
this module is only responsible for collecting and normalizing market data.
"""

import ccxt.async_support as ccxt


class MarketFeed:
    """Collects market data from a single exchange.

    Wraps a ccxt async exchange instance and exposes coroutine methods for
    pulling public market data. The underlying exchange holds an aiohttp
    session, so call ``close()`` (or use the async context manager) when done.
    """

    def __init__(self, exchange_id: str):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id)
        # Public endpoints only — no API key required yet.
        self.exchange = exchange_class({"enableRateLimit": True})

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch the latest ticker for a symbol (e.g. ``"BTC/USD"``).

        Validates ``symbol`` against the exchange's loaded markets first and
        raises ``ValueError`` if it is unknown, rather than issuing a blind API
        call. This fails loudly on typos or mismatched symbols — which, when
        comparing prices across exchanges, is what stops a wrong pair from
        producing a plausible-looking but bogus number.
        """
        await self._require_market(symbol)
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        until: int | None = None,
        page_limit: int = 300,
    ) -> list[list[float]]:
        """Fetch a full range of OHLCV candles, paginating as needed.

        Returns candles from ``since`` up to ``until`` (both epoch
        milliseconds; ``until`` defaults to now) as a list of
        ``[timestamp, open, high, low, close, volume]`` rows in ascending
        time order, with duplicate timestamps removed.

        Exchanges cap how many candles a single ``fetch_ohlcv`` call returns
        (Coinbase allows ~300), so a 90-day 1h history spans several requests.
        This method walks a cursor forward through those pages so callers don't
        have to reimplement the pagination — assembling the complete series is
        still just "collecting market data", the job this class exists for.
        Rate limiting between pages is handled by ``enableRateLimit``.

        Like ``fetch_ticker``, the symbol is validated up front so a mismatched
        pair fails loudly instead of silently returning an empty series.
        """
        await self._require_market(symbol)
        if not self.exchange.has.get("fetchOHLCV"):
            raise NotImplementedError(
                f"{self.exchange_id} does not support fetch_ohlcv"
            )

        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        if until is None:
            until = self.exchange.milliseconds()

        candles_by_ts: dict[int, list[float]] = {}
        cursor = since
        while cursor < until:
            batch = await self.exchange.fetch_ohlcv(
                symbol, timeframe, cursor, page_limit
            )
            if not batch:
                break
            for candle in batch:
                if candle[0] <= until:
                    candles_by_ts[candle[0]] = candle
            # Advance past the last candle we received. If the exchange
            # returned nothing newer than the cursor, stop rather than loop
            # forever on the same page.
            next_cursor = batch[-1][0] + timeframe_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        return [candles_by_ts[ts] for ts in sorted(candles_by_ts)]

    async def _require_market(self, symbol: str) -> None:
        """Load markets if needed and validate ``symbol`` against them.

        Raises ``ValueError`` on an unknown symbol before any data-fetching API
        call is made.
        """
        await self._ensure_markets_loaded()
        if symbol not in self.exchange.markets:
            raise ValueError(
                f"{symbol!r} is not a market on {self.exchange_id}. "
                f"Example available symbols: "
                f"{sorted(self.exchange.markets)[:5]}"
            )

    async def _ensure_markets_loaded(self) -> None:
        """Load and cache the exchange's markets if not already loaded."""
        if not self.exchange.markets:
            await self.exchange.load_markets()

    async def fetch_order_book(self, symbol: str):
        """Fetch the current order book for a symbol."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the underlying exchange's network session."""
        await self.exchange.close()

    async def __aenter__(self) -> "MarketFeed":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
