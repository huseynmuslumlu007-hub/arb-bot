"""Exploratory: is the BTC/ETH relationship stable enough for a pairs strategy?

Loads Coinbase's collected BTC/USD and ETH/USD 1h closes (via
``data.history.load_candles``), aligns them on their shared hours, and reports:

  * the price-level correlation over the full ~90-day window (as asked), plus
    the log-return correlation, which is the more honest read on co-movement;
  * summary stats for the price ratio BTC/ETH, and the ratio over time as a
    saved PNG if matplotlib is available, otherwise an ASCII sparkline;
  * an Augmented Dickey-Fuller (ADF) stationarity test on the ratio and a
    mean-reversion half-life — the numbers that say whether the ratio actually
    reverts (pairs-tradable) or just wanders (a trap), rather than eyeballing it;
  * an Engle-Granger cointegration test: estimate the hedge ratio (regress BTC
    on ETH) and test whether that β-weighted spread is stationary. A hedged
    spread can revert even when the naive 1:1 ratio does not, so this is the
    fairer test of a pairs relationship.

Exploratory only — no strategy or trading logic. The point is to measure
whether the relationship looks stable before anyone builds on it.

Run from the project root:
    python -m scripts.check_correlation
"""

import numpy as np
import pandas as pd

from data.history import load_candles
from data.stats import engle_granger, half_life

EXCHANGE = "coinbase"
BASE, QUOTE = "BTC/USD", "ETH/USD"  # ratio is BASE / QUOTE
PLOT_PATH = "data_store/coinbase_btc_eth_ratio.png"

_SPARK = "▁▂▃▄▅▆▇█"


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
    hl_obs = half_life(ratio)
    hl_txt = "inf (no reversion)" if np.isinf(hl_obs) else (
        f"{hl_obs:.0f} h (~{hl_obs / 24:.1f} d)"
    )

    print("\n  stationarity of the ratio (is it mean-reverting?)")
    print(f"    half-life          : {hl_txt}")

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
        f"    verdict            : "
        + (
            "stationary — reject unit root (mean-reverting)"
            if reverting
            else "NOT stationary — cannot reject unit root (drifts/wanders)"
        )
    )


def _cointegration_report(base: pd.Series, quote: pd.Series) -> None:
    """Engle-Granger: estimate the hedge ratio, test if the spread is stationary.

    Uses the shared ``data.stats.engle_granger`` so the maths matches the batch
    screen in ``scripts/screen_pairs.py`` exactly; this function only formats it.
    """
    res = engle_granger(base, quote)
    hl = res.half_life
    hl_txt = "inf (no reversion)" if np.isinf(hl) else f"{hl:.0f} h (~{hl / 24:.1f} d)"
    cointegrated = res.p_value < 0.05

    print("\n  cointegration (Engle-Granger: does the hedged spread revert?)")
    print(
        f"    hedge ratio beta   : {res.beta:.3f}  "
        f"(BTC ≈ {res.beta:.2f}·ETH + {res.alpha:,.0f})"
    )
    print(f"    spread half-life   : {hl_txt}")
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


def _try_plot(ratio: pd.Series) -> bool:
    """Save a PNG of the ratio over time. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ratio.index, ratio.to_numpy(), linewidth=0.8)
    ax.set_title(f"{EXCHANGE}: {BASE} / {QUOTE} price ratio (1h)")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel(f"{BASE} / {QUOTE}")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    base = load_candles(EXCHANGE, BASE)["close"]
    quote = load_candles(EXCHANGE, QUOTE)["close"]
    # Inner join on timestamp: only hours both symbols have (drops the known
    # Coinbase candle gap on either side rather than pairing a hole with a price).
    df = pd.concat({"base": base, "quote": quote}, axis=1, join="inner")

    price_corr = df["base"].corr(df["quote"])
    returns = np.log(df / df.shift(1)).dropna()
    return_corr = returns["base"].corr(returns["quote"])

    ratio = df["base"] / df["quote"]
    cv = ratio.std() / ratio.mean()  # coefficient of variation: spread relative to level

    print(f"{EXCHANGE}  {BASE} vs {QUOTE}")
    print(f"  shared hours       : {len(df)}  ({df.index[0]} -> {df.index[-1]})")
    print(f"  price-level corr    : {price_corr:+.4f}")
    print(f"  log-return corr     : {return_corr:+.4f}")
    print(
        f"  ratio {BASE}/{QUOTE} : "
        f"mean {ratio.mean():.3f}  std {ratio.std():.3f}  "
        f"cv {cv:.3%}  range [{ratio.min():.3f}, {ratio.max():.3f}]"
    )
    print(f"  ratio start -> end  : {ratio.iloc[0]:.3f} -> {ratio.iloc[-1]:.3f}")

    _stationarity_report(ratio)
    _cointegration_report(df["base"], df["quote"])

    if _try_plot(ratio):
        print(f"\n  ratio plot saved   : {PLOT_PATH}")
    else:
        print("\n  (matplotlib not installed — ASCII sparkline of the ratio)")
        print(f"  {ratio.min():.2f} {_sparkline(ratio)} {ratio.max():.2f}")


if __name__ == "__main__":
    main()
