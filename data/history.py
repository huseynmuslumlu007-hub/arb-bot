"""Load collected OHLCV CSVs and align them across exchanges.

``scripts/collect_history.py`` writes one CSV per exchange+symbol. Those files
can cover *different* time ranges — Coinbase gives ~90 days at 1h while Kraken's
public OHLC endpoint caps at ~30 — and can have internal holes (a missing candle
hour). Comparing prices across exchanges is only valid where both actually have
a candle for the *same* hour.

``align`` handles that by inner-joining the per-exchange series on their
timestamp index: the result contains only timestamps present on **every**
requested exchange. A shorter history on one venue, or a gap on another, simply
shrinks the shared window rather than silently pairing a real quote against a
missing one — the same fail-honest stance as the symbol guard in ``MarketFeed``.

This module only reads what's already on disk; fetching lives in ``MarketFeed``.
"""

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data_store")
TIMEFRAME = "1h"


def candle_path(exchange: str, symbol: str, timeframe: str = TIMEFRAME) -> Path:
    """Path a collected CSV would live at, e.g. data_store/kraken_BTC-USD_1h.csv."""
    safe_symbol = symbol.replace("/", "-")
    return OUTPUT_DIR / f"{exchange}_{safe_symbol}_{timeframe}.csv"


def load_candles(
    exchange: str, symbol: str, timeframe: str = TIMEFRAME
) -> pd.DataFrame:
    """Load one exchange+symbol CSV, indexed by UTC ``datetime`` (ascending).

    Raises ``FileNotFoundError`` if the file hasn't been collected yet, rather
    than returning an empty frame that would quietly drop out of any join.
    """
    path = candle_path(exchange, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No collected data at {path}. "
            f"Run `python -m scripts.collect_history` first."
        )
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def load_stock_closes(ticker: str, timeframe: str = "1d") -> pd.Series:
    """Load one ticker's collected daily closes, indexed by ``date`` (ascending).

    Stock CSVs (from ``scripts.collect_stock_history``) have ``date``/``close``
    columns rather than the crypto candle schema, hence a separate loader. Raises
    ``FileNotFoundError`` if the file hasn't been collected yet.
    """
    path = OUTPUT_DIR / f"stock_{ticker}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No collected data at {path}. "
            f"Run `python -m scripts.collect_stock_history` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def align(
    symbol: str,
    exchanges,
    column: str = "close",
    timeframe: str = TIMEFRAME,
) -> pd.DataFrame:
    """Align one OHLCV column across exchanges onto their shared timestamps.

    Returns a DataFrame with one column per exchange (named by exchange id),
    indexed by the timestamps present on *all* of them (inner join). For
    example ``align("BTC/USD", ["coinbase", "kraken"])`` yields aligned close
    prices ready for a spread: ``df["coinbase"] - df["kraken"]``.
    """
    exchanges = list(exchanges)
    series = {
        exchange: load_candles(exchange, symbol, timeframe)[column]
        for exchange in exchanges
    }
    aligned = pd.concat(series, axis=1, join="inner")
    aligned.columns = exchanges
    return aligned
