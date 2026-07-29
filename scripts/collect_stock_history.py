"""Collect daily stock closing prices via yfinance, save each to CSV.

Pulls ~2 years of daily closes for a few pairs-trading candidates (KO/PEP, a
consumer-staples pair; GLD/GDX, gold vs gold-miners) and writes one CSV per
ticker to ``data_store/`` as ``stock_<TICKER>_1d.csv`` — the same
``<source>_<symbol>_<interval>`` shape the crypto collector uses.

Deliberately *not* built on ``MarketFeed``. yfinance has a completely different
interface from ccxt (synchronous, batch download by date, its own adjustment
handling), and bending it into the async exchange abstraction would add a
leaky adapter for no benefit. The crypto and stock collectors share a naming
convention and an output folder — not a code path.

Closes are **dividend/split-adjusted** (``auto_adjust=True``): for pairs and
mean-reversion work you want the total-return series, otherwise a dividend or
split shows up as a fake jump in the spread.

Run from the project root:
    python -m scripts.collect_stock_history
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ("KO", "PEP", "GLD", "GDX")
PERIOD = "2y"
INTERVAL = "1d"
OUTPUT_DIR = Path("data_store")

# Rough sanity bands (USD) for the "sensible price level" check — not precise,
# just wide guards to catch a wrong ticker or a units/adjustment mistake (e.g.
# cents-vs-dollars, or the wrong symbol entirely). GLD/GDX bands are wide on the
# top end deliberately: gold roughly doubled over 2024-2026, so GLD ran to ~$496
# and GDX to ~$116 in-window — verified as a real, smooth trend (the two moved
# together, as gold and gold-miners should), not a data glitch.
PLAUSIBLE_RANGE = {
    "KO": (30, 120),
    "PEP": (100, 250),
    "GLD": (150, 600),
    "GDX": (20, 160),
}
# A gap between consecutive trading days longer than this (calendar days) is
# worth surfacing. Weekends give 3-day gaps; a holiday long weekend ~4. Beyond
# that usually means a holiday stretch (expected) or missing data (not).
GAP_FLAG_DAYS = 4


def _csv_path(ticker: str) -> Path:
    return OUTPUT_DIR / f"stock_{ticker}_{INTERVAL}.csv"


def fetch_closes(ticker: str) -> pd.DataFrame:
    """Return a DataFrame with columns ['date', 'close'] of adjusted daily closes."""
    hist = yf.Ticker(ticker).history(
        period=PERIOD, interval=INTERVAL, auto_adjust=True
    )
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")
    out = hist["Close"].rename("close").reset_index()
    out.columns = ["date", "close"]
    # yfinance dates are tz-aware midnights (US/Eastern); keep just the calendar date.
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    # yfinance appends a placeholder row for the pending/current session with a
    # NaN close; trim trailing NaNs. Interior NaNs are left in so verify() flags
    # them as a genuine gap rather than us silently swallowing real missing data.
    last_valid = out["close"].last_valid_index()
    if last_valid is not None:
        out = out.loc[:last_valid].reset_index(drop=True)
    return out


def verify(ticker: str, df: pd.DataFrame) -> list[str]:
    """Return a list of problems found; empty list means the data looks real."""
    problems = []
    lo, hi = PLAUSIBLE_RANGE[ticker]

    if df["close"].isna().any():
        problems.append(f"{df['close'].isna().sum()} NaN close values")
    if (df["close"] <= 0).any():
        problems.append("non-positive close prices")
    bad_level = df[(df["close"] < lo) | (df["close"] > hi)]
    if not bad_level.empty:
        problems.append(
            f"{len(bad_level)} closes outside plausible ${lo}-${hi} "
            f"(min {df['close'].min():.2f}, max {df['close'].max():.2f})"
        )

    span_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    if span_days < 700:  # ~2y should be ~730 calendar days
        problems.append(f"date span only {span_days} days (expected ~730)")
    # US markets trade ~252 days/year; 2y ≈ ~500 rows. Far fewer means truncation.
    if len(df) < 460:
        problems.append(f"only {len(df)} rows (expected ~500 trading days)")

    return problems


def _gap_report(df: pd.DataFrame) -> str:
    gaps = df["date"].diff().dt.days
    big = df[gaps > GAP_FLAG_DAYS]
    if big.empty:
        return "none > %dd (weekends/holidays only)" % GAP_FLAG_DAYS
    parts = [
        f"{int(g)}d before {d.date()}"
        for g, d in zip(gaps[big.index], big["date"])
    ]
    return f"{len(big)} gaps > {GAP_FLAG_DAYS}d: " + ", ".join(parts)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for ticker in TICKERS:
        df = fetch_closes(ticker)
        path = _csv_path(ticker)
        df.to_csv(path, index=False)

        problems = verify(ticker, df)
        status = "OK" if not problems else "CHECK"
        if problems:
            all_ok = False
        print(
            f"[{status}] {ticker:4} {len(df):>4} rows  "
            f"{df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}  "
            f"close ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f} "
            f"(range ${df['close'].min():.2f}-${df['close'].max():.2f}) -> {path}"
        )
        print(f"        gaps: {_gap_report(df)}")
        for p in problems:
            print(f"        ! {p}")

    print("\nAll downloads verified." if all_ok else "\nSome downloads need a look (see CHECK rows).")


if __name__ == "__main__":
    main()
