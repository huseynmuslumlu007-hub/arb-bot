"""Exploratory: momentum vs mean-reversion in BTC/USD hourly closes.

For each lookback window L, correlate the past L-hour return with the *next*
L-hour return over the collected Coinbase hourly history. A positive correlation
means recent winners keep winning (momentum); negative means recent moves tend to
reverse (mean-reversion); near-zero means no linear effect at that horizon.

Two honesty guards on the read:
  * Returns are logarithmic, so past and forward returns are on the same additive
    scale as everywhere else in this project.
  * The windows overlap (computed at every hour), which inflates the effective
    sample and autocorrelates observations — so a correlation is only worth
    noting if it clears a rough noise band based on the number of *independent*
    (non-overlapping) windows, ~2/sqrt(n_indep). Treat this as descriptive, not
    an inferential test.

Exploratory only — no strategy or trading logic.

Run from the project root:
    python -m scripts.check_momentum            # BTC/USD (default)
    python -m scripts.check_momentum ETH/USD
"""

import argparse

import numpy as np

from data.history import load_candles
from data.stats import momentum_autocorrelation

EXCHANGE = "coinbase"
LOOKBACKS_H = (6, 24, 72, 168)  # hours: 6h, 1d, 3d, 7d


def _label(hours: int) -> str:
    return f"{hours}h" if hours < 48 else f"{hours}h ({hours // 24}d)"


def _read(corr: float, band: float) -> str:
    if corr > band:
        return "momentum"
    if corr < -band:
        return "mean-reversion"
    return "~none (within noise)"


def analyze(symbol: str) -> None:
    close = load_candles(EXCHANGE, symbol)["close"]
    logc = np.log(close)

    print(f"{EXCHANGE}  {symbol}  momentum / mean-reversion by lookback")
    print(f"  {len(close)} hourly closes  ({close.index[0]} -> {close.index[-1]})\n")
    print(f"{'lookback':>12} {'n_obs':>7} {'corr':>8} {'n_indep':>8} {'noise±':>8}  read")
    print("-" * 62)

    for lb in LOOKBACKS_H:
        res = momentum_autocorrelation(logc, lb)
        if res.n_obs < 3:
            print(f"{_label(lb):>12} {res.n_obs:>7}  (too few observations)")
            continue
        print(
            f"{_label(lb):>12} {res.n_obs:>7} {res.corr:>+8.3f} {res.n_indep:>8} "
            f"{res.noise_band:>8.3f}  {_read(res.corr, res.noise_band)}"
        )

    print(
        "\nnoise± is a rough 95% band (~2/sqrt(n_indep)) from the count of "
        "non-overlapping\nwindows; |corr| below it is indistinguishable from zero "
        "at that horizon."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="BTC/USD", help="e.g. BTC/USD")
    args = parser.parse_args()
    analyze(args.symbol.upper())


if __name__ == "__main__":
    main()
