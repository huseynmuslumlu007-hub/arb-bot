"""Exploratory: momentum vs mean-reversion in collected stock closes.

Tests the classic academic momentum horizons — 1, 3, 6, and 12 months (~21, 63,
126, 252 trading days) — on each collected ticker (KO, PEP, GLD, GDX). For each
lookback L it correlates the past L-day return with the next L-day return: positive
suggests momentum (winners keep winning), negative mean-reversion, near-zero no
linear effect at that horizon.

Reuses the same honesty guard as ``scripts/check_momentum.py`` — the shared
``data.stats.momentum_autocorrelation``, whose noise band (``2/sqrt(n_indep)`` over
*non-overlapping* windows) keeps a small correlation from being mistaken for a
signal. That guard matters especially here: only ~2 years (~500 trading days) were
collected, so the longer horizons have very few independent windows — the 12-month
lookback needs 252 past + 252 forward days and so yields *no* testable observations
in a 500-day window at all. The table makes that lack of power explicit rather than
printing a number that looks like a result.

Exploratory only — no strategy or trading logic.

Run from the project root:
    python -m scripts.check_momentum_stocks
"""

import numpy as np

from data.history import load_stock_closes
from data.stats import momentum_autocorrelation

TICKERS = ("KO", "PEP", "GLD", "GDX")
LOOKBACKS_D = (21, 63, 126, 252)  # ~1mo, 3mo, 6mo, 12mo in trading days
TRADING_DAYS_PER_MONTH = 21


def _label(days: int) -> str:
    return f"{days}d (~{days // TRADING_DAYS_PER_MONTH}mo)"


def _read(corr: float, band: float) -> str:
    if corr > band:
        return "momentum"
    if corr < -band:
        return "mean-reversion"
    return "~none (within noise)"


def analyze(ticker: str) -> None:
    close = load_stock_closes(ticker)
    logc = np.log(close)

    print(f"{ticker}  ({len(close)} daily closes, "
          f"{close.index[0].date()} -> {close.index[-1].date()})")
    print(f"{'lookback':>14} {'n_obs':>7} {'corr':>8} {'n_indep':>8} {'noise±':>8}  read")
    print("  " + "-" * 62)

    for lb in LOOKBACKS_D:
        res = momentum_autocorrelation(logc, lb)
        if res.n_obs < 3:
            print(f"{_label(lb):>14} {res.n_obs:>7}  (window too short to test this horizon)")
            continue
        print(
            f"{_label(lb):>14} {res.n_obs:>7} {res.corr:>+8.3f} {res.n_indep:>8} "
            f"{res.noise_band:>8.3f}  {_read(res.corr, res.noise_band)}"
        )
    print()


def main() -> None:
    for ticker in TICKERS:
        analyze(ticker)
    print(
        "noise± is a rough 95% band (~2/sqrt(n_indep)) from the count of "
        "non-overlapping\nwindows; |corr| below it is indistinguishable from zero at "
        "that horizon.\nPower scales with n_indep, so judge each row against its own "
        "band: a horizon\nwith few independent windows (a long lookback on short "
        "history) is a weak test,\nnot evidence either way."
    )


if __name__ == "__main__":
    main()
