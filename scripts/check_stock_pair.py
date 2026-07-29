"""Exploratory: is a stock pair stable enough for a pairs strategy?

The stock-market analog of ``scripts/check_correlation.py``. Loads two tickers'
collected daily adjusted closes from ``data_store/stock_<TICKER>_1d.csv``, aligns
them on shared trading days, and runs the same battery — price-level correlation,
log-return correlation, an ADF stationarity test on the price ratio, and the full
Engle-Granger cointegration test (hedge ratio + spread half-life) via the shared
``data.stats`` routines. Output mirrors the crypto script for consistency; the
only substantive change is that half-life is reported in trading days, since the
bars are daily rather than hourly.

The pair is configurable, so the same script serves KO/PEP, GLD/GDX, and beyond:
    python -m scripts.check_stock_pair            # KO PEP (default)
    python -m scripts.check_stock_pair GLD GDX

Exploratory only — no strategy or trading logic.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.history import load_stock_closes
from data.stats import engle_granger, half_life

OUTPUT_DIR = Path("data_store")
TRADING_DAYS_PER_MONTH = 21  # ~21 trading days/month, for a friendlier second scale

_SPARK = "▁▂▃▄▅▆▇█"


def _hl_txt(hl: float) -> str:
    """Format a half-life (in trading days) for display."""
    if np.isinf(hl):
        return "inf (no reversion)"
    return f"{hl:.0f} trading days (~{hl / TRADING_DAYS_PER_MONTH:.1f} mo)"


def _sparkline(values: pd.Series, width: int = 70) -> str:
    """A compact unicode sparkline of a series, downsampled to ``width`` cols."""
    if len(values) > width:
        # Average within evenly spaced buckets so the shape survives downsampling.
        buckets = np.array_split(values.to_numpy(), width)
        points = np.array([b.mean() for b in buckets])
    else:
        points = values.to_numpy()
    lo, hi = points.min(), points.max()
    if hi == lo:
        return _SPARK[0] * len(points)
    scaled = (points - lo) / (hi - lo) * (len(_SPARK) - 1)
    return "".join(_SPARK[int(round(s))] for s in scaled)


def _stationarity_report(ratio: pd.Series) -> None:
    """Print the ADF test and half-life for the ratio, with a plain-English read."""
    print("\n  stationarity of the ratio (is it mean-reverting?)")
    print(f"    half-life          : {_hl_txt(half_life(ratio))}")

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("    ADF test           : (statsmodels not installed — skipped)")
        return

    # H0: the ratio has a unit root (non-stationary / not mean-reverting).
    stat, pvalue, _, _, crit, _ = adfuller(ratio.to_numpy(), autolag="AIC")
    reverting = pvalue < 0.05
    print(f"    ADF statistic      : {stat:.3f}  (5% crit {crit['5%']:.3f})")
    print(f"    ADF p-value        : {pvalue:.3f}")
    print(
        "    verdict            : "
        + (
            "stationary — reject unit root (mean-reverting)"
            if reverting
            else "NOT stationary — cannot reject unit root (drifts/wanders)"
        )
    )


def _cointegration_report(
    base: pd.Series, quote: pd.Series, base_name: str, quote_name: str
) -> None:
    """Engle-Granger: estimate the hedge ratio, test if the spread is stationary.

    Uses the shared ``data.stats.engle_granger`` so the maths matches the crypto
    script and the batch screen exactly; this function only formats it.
    """
    res = engle_granger(base, quote)
    cointegrated = res.p_value < 0.05

    print("\n  cointegration (Engle-Granger: does the hedged spread revert?)")
    print(
        f"    hedge ratio beta   : {res.beta:.3f}  "
        f"({base_name} ≈ {res.beta:.3f}·{quote_name} + {res.alpha:,.2f})"
    )
    print(f"    spread half-life   : {_hl_txt(res.half_life)}")
    print(f"    EG statistic       : {res.eg_stat:.3f}  (5% crit {res.crit_5pct:.3f})")
    print(f"    EG p-value         : {res.p_value:.3f}")
    print(
        "    verdict            : "
        + (
            "cointegrated — hedged spread is stationary (pairs-tradable)"
            if cointegrated
            else "NOT cointegrated — even the hedged spread drifts"
        )
    )


def _try_plot(ratio: pd.Series, base_name: str, quote_name: str) -> str | None:
    """Save a PNG of the ratio over time. Returns the path, or None if no matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path = str(OUTPUT_DIR / f"stock_{base_name}_{quote_name}_ratio.png")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ratio.index, ratio.to_numpy(), linewidth=0.8)
    ax.set_title(f"{base_name} / {quote_name} price ratio (daily, adjusted)")
    ax.set_xlabel("date")
    ax.set_ylabel(f"{base_name} / {quote_name}")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def analyze_pair(base_name: str, quote_name: str) -> None:
    base = load_stock_closes(base_name)
    quote = load_stock_closes(quote_name)
    # Inner join on date: only trading days both tickers have (a holiday or a
    # halt on one side drops that day rather than pairing a hole with a price).
    df = pd.concat({"base": base, "quote": quote}, axis=1, join="inner")

    price_corr = df["base"].corr(df["quote"])
    returns = np.log(df / df.shift(1)).dropna()
    return_corr = returns["base"].corr(returns["quote"])

    ratio = df["base"] / df["quote"]
    cv = ratio.std() / ratio.mean()  # coefficient of variation: spread relative to level

    print(f"{base_name} vs {quote_name}  (daily, adjusted close)")
    print(
        f"  shared days        : {len(df)}  "
        f"({df.index[0].date()} -> {df.index[-1].date()})"
    )
    print(f"  price-level corr    : {price_corr:+.4f}")
    print(f"  log-return corr     : {return_corr:+.4f}")
    print(
        f"  ratio {base_name}/{quote_name}   : "
        f"mean {ratio.mean():.3f}  std {ratio.std():.3f}  "
        f"cv {cv:.3%}  range [{ratio.min():.3f}, {ratio.max():.3f}]"
    )
    print(f"  ratio start -> end  : {ratio.iloc[0]:.3f} -> {ratio.iloc[-1]:.3f}")

    _stationarity_report(ratio)
    _cointegration_report(df["base"], df["quote"], base_name, quote_name)

    plot_path = _try_plot(ratio, base_name, quote_name)
    if plot_path:
        print(f"\n  ratio plot saved   : {plot_path}")
    else:
        print("\n  (matplotlib not installed — ASCII sparkline of the ratio)")
        print(f"  {ratio.min():.2f} {_sparkline(ratio)} {ratio.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="KO", help="base ticker (default: KO)")
    parser.add_argument("quote", nargs="?", default="PEP", help="quote ticker (default: PEP)")
    args = parser.parse_args()
    analyze_pair(args.base.upper(), args.quote.upper())


if __name__ == "__main__":
    main()
