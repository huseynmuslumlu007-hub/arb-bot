"""Backtest a naive cross-exchange arbitrage signal on collected history.

For every hour that Coinbase and Kraken *both* have a BTC/USD or ETH/USD candle
(the shared window from ``data.history.align``), compare the two close prices.
When the gap between them exceeds a round-trip fee threshold, the hour is worth
flagging: in principle you could have bought on the cheaper venue and sold on
the dearer one and still cleared fees.

The spread percentage is measured against the *lower* price — the capital you'd
have to put up to buy — so it maps directly onto a round-trip fee rate: a 0.8%
threshold means the gross gap must beat 0.8% of the buy price before fees eat it.

Analysis only. This logs hypothetical signals from historical closes; it places
no orders and models no execution (slippage, depth, latency, withdrawal times).
Hourly candle closes are not tradable quotes, so a "signal" here is a research
flag, not a fill.

Run from the project root:
    python -m scripts.backtest_arbitrage                     # 0.8% threshold
    python -m scripts.backtest_arbitrage --fee-threshold 0.5
"""

import argparse

from data.history import align

EXCHANGES = ("coinbase", "kraken")
SYMBOLS = ("BTC/USD", "ETH/USD")
DEFAULT_FEE_THRESHOLD_PCT = 0.8


def backtest_symbol(symbol: str, fee_threshold_pct: float) -> dict:
    """Check every shared hour for one symbol; return summary stats.

    Prints one line per signal hour (spread over threshold) as a side effect.
    """
    a, b = EXCHANGES
    df = align(symbol, EXCHANGES)  # columns: 'coinbase', 'kraken'; shared hours

    cheaper = df[[a, b]].idxmin(axis=1)  # exchange id with the lower price
    lower = df[[a, b]].min(axis=1)
    spread_usd = df[[a, b]].max(axis=1) - lower
    spread_pct = spread_usd / lower * 100.0

    signals = df.index[spread_pct > fee_threshold_pct]

    print(f"\n{symbol}  ({len(df)} shared hours, threshold {fee_threshold_pct}%)")
    if len(signals) == 0:
        print("  no signals — no hour's spread exceeded the threshold")
    for ts in signals:
        print(
            f"  SIGNAL {ts}  cheaper={cheaper[ts]:<8} "
            f"spread ${spread_usd[ts]:,.2f}  {spread_pct[ts]:.3f}%"
        )

    # Largest spread seen at all (even if below threshold) — reported by pct,
    # since pct is what determines whether fees could be cleared.
    peak_ts = spread_pct.idxmax()
    return {
        "symbol": symbol,
        "hours": len(df),
        "signals": len(signals),
        "peak_ts": peak_ts,
        "peak_usd": spread_usd[peak_ts],
        "peak_pct": spread_pct[peak_ts],
        "peak_cheaper": cheaper[peak_ts],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fee-threshold",
        type=float,
        default=DEFAULT_FEE_THRESHOLD_PCT,
        metavar="PCT",
        help="round-trip fee threshold in percent (default: %(default)s)",
    )
    args = parser.parse_args()

    results = [backtest_symbol(s, args.fee_threshold) for s in SYMBOLS]

    total_hours = sum(r["hours"] for r in results)
    total_signals = sum(r["signals"] for r in results)
    biggest = max(results, key=lambda r: r["peak_pct"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  fee threshold      : {args.fee_threshold}% round-trip")
    print(f"  symbols            : {', '.join(SYMBOLS)}")
    print(f"  total hours checked: {total_hours}")
    print(f"  signals found      : {total_signals}")
    print(
        f"  largest spread seen: ${biggest['peak_usd']:,.2f} "
        f"({biggest['peak_pct']:.3f}%) on {biggest['symbol']} "
        f"at {biggest['peak_ts']}, cheaper on {biggest['peak_cheaper']}"
    )


if __name__ == "__main__":
    main()
