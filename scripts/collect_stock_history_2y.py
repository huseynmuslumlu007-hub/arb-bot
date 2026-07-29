"""Regenerate the original Model 6 stock window (2 years: 2024-07 to 2026-07).

``data_store/`` now holds the FULL price history (``period="max"``) for the
momentum work, so Model 6's GLD/GDX cointegration analysis is no longer
reproducible from the live files. This rewrites the four stock CSVs with exactly
the 2-year slice Model 6 used, so ``scripts.check_stock_pair`` and
``scripts.validate_stock_pair`` reproduce that section's numbers.

The window is pinned by explicit dates (not a relative "2y"), so it yields the
same slice — first row 2024-07-29, last row 2026-07-27 — no matter when it runs.

WARNING: this OVERWRITES the full-history stock CSVs in place. When you are done
reproducing Model 6, restore full history for the momentum scripts with:
    python -m scripts.collect_stock_history

Run from the project root:
    python -m scripts.collect_stock_history_2y
"""

from scripts.collect_stock_history import collect

# Model 6 slice. yfinance's end is exclusive, so 2026-07-28 includes 2026-07-27.
MODEL6_START = "2024-07-29"
MODEL6_END = "2026-07-28"


def main() -> None:
    print("Regenerating the ORIGINAL Model 6 2-year window.")
    print("This OVERWRITES the full-history stock CSVs; restore afterward with:")
    print("    python -m scripts.collect_stock_history\n")
    collect(start=MODEL6_START, end=MODEL6_END)


if __name__ == "__main__":
    main()
