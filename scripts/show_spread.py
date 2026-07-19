"""Show the Coinbase vs Kraken close-price spread over their shared window.

Reads the CSVs written by ``scripts.collect_history``, aligns each symbol's
close prices onto the timestamps both exchanges share (via
``data.history.align``), and prints the overlap window plus basic spread stats.

This is reporting only — no trading or arbitrage logic. It exists to make the
shared-window alignment tangible: note how the overlap is bounded by Kraken's
shorter (~30-day) history, not Coinbase's ~90 days.

Run from the project root:
    python -m scripts.show_spread
"""

from data.history import align

EXCHANGES = ("coinbase", "kraken")
SYMBOLS = ("BTC/USD", "ETH/USD")


def main() -> None:
    for symbol in SYMBOLS:
        df = align(symbol, EXCHANGES)
        spread = df["coinbase"] - df["kraken"]
        print(
            f"{symbol}: {len(df)} shared hours | "
            f"{df.index[0]} -> {df.index[-1]}"
        )
        print(
            f"  spread (coinbase - kraken)  "
            f"mean {spread.mean():+.2f}  "
            f"max |{spread.abs().max():.2f}|  "
            f"last {spread.iloc[-1]:+.2f}"
        )


if __name__ == "__main__":
    main()
