"""Screen every Coinbase asset pair for cointegration, with an FDR correction.

Tests all 21 unordered pairs of the 7 collected Coinbase assets (BTC, ETH, SOL,
ADA, LINK, LTC, XRP) with the shared Engle-Granger routine in ``data.stats``.
Because 21 tests run at once, raw p < 0.05 is not enough — at that threshold you
would expect roughly one false positive by chance alone. A Benjamini-Hochberg
false-discovery-rate correction rescales the p-values so "survives" means the
pair clears the bar *accounting for* how many tests were run.

Two deliberate choices, both to keep the correction honest:
  * All pairs share one common-timestamp sample (inner join across all 7 assets),
    so every test sees the same hours.
  * Engle-Granger is direction-dependent; each unordered pair is tested once in a
    fixed order (first listed asset regressed on the second), giving exactly 21
    tests — the number the BH correction is applied over.

Exploratory / analysis only — no trading logic.

Run from the project root:
    python -m scripts.screen_pairs
"""

import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from data.history import load_candles
from data.stats import engle_granger

EXCHANGE = "coinbase"
ASSETS = ("BTC", "ETH", "SOL", "ADA", "LINK", "LTC", "XRP")
ALPHA = 0.05  # target false discovery rate


def _load_common_closes() -> pd.DataFrame:
    """Close prices for every asset, inner-joined onto their shared timestamps."""
    closes = {a: load_candles(EXCHANGE, f"{a}/USD")["close"] for a in ASSETS}
    return pd.concat(closes, axis=1, join="inner")


def _hl(obs: float) -> str:
    return "inf" if np.isinf(obs) else f"{obs:.0f}h/{obs / 24:.1f}d"


def main() -> None:
    closes = _load_common_closes()

    rows = []
    for base, quote in itertools.combinations(ASSETS, 2):
        res = engle_granger(closes[base], closes[quote])
        rows.append(
            {
                "pair": f"{base}-{quote}",
                "beta": res.beta,
                "p_raw": res.p_value,
                "half_life": res.half_life,
            }
        )
    results = pd.DataFrame(rows)

    # Benjamini-Hochberg: rescale raw p-values to control the false discovery
    # rate across all 21 tests. 'reject' is the survival flag at the target FDR.
    reject, p_adj, _, _ = multipletests(
        results["p_raw"], alpha=ALPHA, method="fdr_bh"
    )
    results["p_bh"] = p_adj
    results["survives"] = reject
    results = results.sort_values("p_bh", kind="stable").reset_index(drop=True)

    print(
        f"Engle-Granger cointegration screen — {EXCHANGE}, "
        f"{len(ASSETS)} assets, {len(results)} pairs, {len(closes)} shared hours"
    )
    print(f"Benjamini-Hochberg FDR correction at alpha = {ALPHA}\n")
    print(f"{'pair':10} {'hedge β':>12} {'p_raw':>8} {'p_BH':>8} {'half-life':>10}  survives")
    print("-" * 62)
    for r in results.itertuples():
        print(
            f"{r.pair:10} {r.beta:>12.4g} {r.p_raw:>8.3f} {r.p_bh:>8.3f} "
            f"{_hl(r.half_life):>10}  {'YES' if r.survives else 'no'}"
        )

    n_raw = int((results["p_raw"] < ALPHA).sum())
    n_bh = int(results["survives"].sum())
    print(
        f"\nSummary: {n_raw}/{len(results)} pairs at raw p<{ALPHA}; "
        f"{n_bh} survive BH correction."
    )
    if n_bh == 0:
        print("No pair is cointegrated once multiple testing is accounted for.")


if __name__ == "__main__":
    main()
