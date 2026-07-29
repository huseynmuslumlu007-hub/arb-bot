"""Collect the full available daily close history via yfinance, save each to CSV.

Pulls the *maximum available* history (``period="max"``) of daily
dividend/split-adjusted closes for a few pairs candidates (KO/PEP consumer
staples; GLD/GDX gold vs gold-miners) and writes one CSV per ticker to
``data_store/`` as ``stock_<TICKER>_1d.csv`` — the same
``<source>_<symbol>_<interval>`` shape the crypto collector uses. This overwrites
the earlier 2-year files: everything downstream now reads the full history.

Deliberately *not* built on ``MarketFeed``. yfinance has a completely different
interface from ccxt (synchronous, batch download by date, its own adjustment
handling), and bending it into the async exchange abstraction would add a leaky
adapter for no benefit. The two collectors share a naming convention and an output
folder — not a code path.

Closes are **dividend/split-adjusted** (``auto_adjust=True``): for pairs and
mean-reversion work you want the total-return series. A consequence that matters
over decades: adjustment scales *old* prices down (KO's 1960s adjusted close is a
few cents), so the raw price level is only meaningful at the recent end. The
verification below reflects that — it sanity-checks the *latest* close against
today's ballpark, requires positivity throughout, and scans for any single-day
jump that would betray a split the adjustment *failed* to remove.

Run from the project root:
    python -m scripts.collect_stock_history                       # full history (default)
    python -m scripts.collect_stock_history --period 2y           # a relative window
    python -m scripts.collect_stock_history --start 2024-07-29 --end 2026-07-28

An explicit ``--start``/``--end`` pins an exact date slice (used by
``scripts.collect_stock_history_2y`` to regenerate the original Model 6 window).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ("KO", "PEP", "GLD", "GDX")
PERIOD = "max"
INTERVAL = "1d"
OUTPUT_DIR = Path("data_store")

# Current-price sanity band (USD), applied to the LATEST close only. Over decades
# of adjusted history the old prices are legitimately tiny, so a fixed band can't
# apply to the whole series — but the most recent close should match today's real
# market price, which catches a wrong ticker or a units/adjustment mistake.
RECENT_PRICE_RANGE = {
    "KO": (40, 120),
    "PEP": (100, 250),
    "GLD": (150, 600),
    "GDX": (20, 160),
}
# Calendar-day gap beyond which we surface it. Weekends give 3 days, holiday long
# weekends ~4. Over decades, larger gaps are real market closures (9/11, storms,
# days of mourning) — reported for eyeballing, not treated as errors.
GAP_FLAG_DAYS = 4
# |1-day log return| above this flags a possible UNadjusted split or bad tick.
# auto_adjust should remove splits, so a 2:1 (~-0.69) or 3:2 (~-0.41) jump left in
# the adjusted series would be an artifact. Genuine 1-day moves this large don't
# happen for these large-cap names / ETFs.
JUMP_LOG_RETURN = 0.40


def _csv_path(ticker: str) -> Path:
    return OUTPUT_DIR / f"stock_{ticker}_{INTERVAL}.csv"


def fetch_history(ticker: str, period: str = PERIOD, start=None, end=None):
    """Return (DataFrame['date','close'], split_events) of adjusted history.

    Pulls ``period`` (e.g. "max", "2y") unless an explicit ``start``/``end`` date
    slice is given, which pins an exact window regardless of the run date.
    ``split_events`` is a list of ``(date, ratio)`` from yfinance's split record —
    reported for context; with ``auto_adjust`` these should already be reflected in
    the close, leaving no jump behind.
    """
    kwargs = {"interval": INTERVAL, "auto_adjust": True}
    if start or end:
        kwargs.update(start=start, end=end)
    else:
        kwargs["period"] = period
    hist = yf.Ticker(ticker).history(**kwargs)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")
    # yfinance appends a placeholder row for the pending session with a NaN close;
    # trim trailing NaNs. Interior NaNs stay so verify() can flag a genuine gap.
    last_valid = hist["Close"].last_valid_index()
    if last_valid is not None:
        hist = hist.loc[:last_valid]

    out = hist["Close"].rename("close").reset_index()
    out.columns = ["date", "close"]
    # yfinance dates are tz-aware midnights (US/Eastern); keep just the calendar date.
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.reset_index(drop=True)

    splits = hist["Stock Splits"]
    split_events = [(d.date(), float(r)) for d, r in splits[splits != 0].items()]
    return out, split_events


def _jumps(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose 1-day log return exceeds the artifact threshold."""
    ret = np.log(df["close"]).diff()
    hit = df[ret.abs() > JUMP_LOG_RETURN].copy()
    hit["ret"] = ret[hit.index]
    return hit


def verify(ticker: str, df: pd.DataFrame) -> list[str]:
    """Return a list of problems found; empty list means the data looks real."""
    problems = []

    if df["close"].isna().any():
        problems.append(f"{df['close'].isna().sum()} NaN close values")
    if (df["close"] <= 0).any():
        problems.append(f"{(df['close'] <= 0).sum()} non-positive close prices")
    if not df["date"].is_monotonic_increasing:
        problems.append("dates not sorted ascending")
    if df["date"].duplicated().any():
        problems.append(f"{df['date'].duplicated().sum()} duplicate dates")

    lo, hi = RECENT_PRICE_RANGE[ticker]
    last = df["close"].iloc[-1]
    if not (lo <= last <= hi):
        problems.append(f"latest close ${last:.2f} outside current range ${lo}-${hi}")

    # Row count vs the ~252 trading days/year the span implies (catches truncation).
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    expected = years * 252
    if years > 1 and len(df) < 0.85 * expected:
        problems.append(
            f"{len(df)} rows well below ~{expected:.0f} expected for {years:.1f}y"
        )

    jumps = _jumps(df)
    if not jumps.empty:
        dates = ", ".join(f"{d.date()} ({r:+.0%})" for d, r in zip(jumps["date"], jumps["ret"]))
        problems.append(
            f"{len(jumps)} day(s) with >{JUMP_LOG_RETURN:.0%} move — "
            f"possible unadjusted split/bad tick: {dates}"
        )

    return problems


def _gap_report(df: pd.DataFrame) -> str:
    gaps = df["date"].diff().dt.days
    big = df.assign(gap=gaps)[gaps > GAP_FLAG_DAYS].sort_values("gap", ascending=False)
    if big.empty:
        return f"none > {GAP_FLAG_DAYS}d (weekends/holidays only)"
    top = big.head(6)
    parts = [f"{int(g)}d->{d.date()}" for g, d in zip(top["gap"], top["date"])]
    tail = "" if len(big) <= 6 else f", +{len(big) - 6} more"
    return f"{len(big)} gaps > {GAP_FLAG_DAYS}d (largest: {', '.join(parts)}{tail})"


def _split_report(split_events: list) -> str:
    if not split_events:
        return "none on record"
    shown = ", ".join(f"{d} {r:g}:1" for d, r in split_events[-6:])
    prefix = "" if len(split_events) <= 6 else f"{len(split_events)} total, recent: "
    return prefix + shown


def collect(period: str = PERIOD, start=None, end=None) -> None:
    """Fetch, save, and verify every ticker for the given window."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    window = f"{start} .. {end}" if (start or end) else f"period={period}"
    print(f"Collecting stock history ({window}) into {OUTPUT_DIR}/\n")
    all_ok = True

    for ticker in TICKERS:
        df, splits = fetch_history(ticker, period=period, start=start, end=end)
        path = _csv_path(ticker)
        df.to_csv(path, index=False)

        problems = verify(ticker, df)
        status = "OK" if not problems else "CHECK"
        if problems:
            all_ok = False

        years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        print(
            f"[{status}] {ticker:4} {len(df):>6} rows  "
            f"{df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} ({years:.1f}y)  "
            f"close ${df['close'].iloc[0]:.4f} -> ${df['close'].iloc[-1]:.2f} "
            f"(min ${df['close'].min():.4f}, max ${df['close'].max():.2f}) -> {path}"
        )
        print(f"        splits (adjusted out): {_split_report(splits)}")
        print(f"        gaps: {_gap_report(df)}")
        for p in problems:
            print(f"        ! {p}")

    print(
        "\nAll downloads verified." if all_ok
        else "\nSome downloads need a look (see CHECK rows)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default=PERIOD, help='e.g. "max" (default), "2y"')
    parser.add_argument("--start", help="explicit start date YYYY-MM-DD (overrides --period)")
    parser.add_argument("--end", help="explicit end date YYYY-MM-DD, exclusive")
    args = parser.parse_args()
    collect(period=args.period, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
