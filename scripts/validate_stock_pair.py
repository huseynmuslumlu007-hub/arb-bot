"""Out-of-sample validation of a cointegrated stock pair.

The in-sample Engle-Granger test (``scripts.check_stock_pair``) estimates the
hedge ratio on the same data it tests, so a good p-value can just reflect a
relationship that happened to hold across that one window. This runs the honest
check:

    1. Split the aligned history into a first half (train) and second half (test).
    2. Estimate the hedge ratio beta on the *train* half only.
    3. Freeze that beta and form the spread on the *test* half with it.
    4. Test whether that out-of-sample spread is still stationary.

Because the hedge ratio is not re-estimated on the test half, the out-of-sample
spread is a fixed linear combination with no parameters fitted to the test data —
so a plain ADF with *standard* critical values is the correct test here (not the
Engle-Granger critical values, which exist precisely to penalize estimating beta
on the data under test). Using the ordinary ADF is what makes this a genuine
hold-out rather than the in-sample test in disguise.

Configurable; defaults to GLD/GDX (the pair that passed in-sample):
    python -m scripts.validate_stock_pair
    python -m scripts.validate_stock_pair KO PEP --split 0.5

Exploratory only — no trading logic.
"""

import argparse

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from data.history import load_stock_closes
from data.stats import engle_granger, half_life

TRADING_DAYS_PER_MONTH = 21


def _hl_txt(hl: float) -> str:
    if np.isinf(hl):
        return "inf (no reversion)"
    return f"{hl:.0f} trading days (~{hl / TRADING_DAYS_PER_MONTH:.1f} mo)"


def validate(base_name: str, quote_name: str, split: float) -> None:
    base = load_stock_closes(base_name)
    quote = load_stock_closes(quote_name)
    df = pd.concat({"base": base, "quote": quote}, axis=1, join="inner")

    cut = int(len(df) * split)
    train, test = df.iloc[:cut], df.iloc[cut:]

    print(f"{base_name} vs {quote_name}  out-of-sample validation (daily, adjusted)")
    print(
        f"  train : {len(train):>4} days  "
        f"{train.index[0].date()} -> {train.index[-1].date()}"
    )
    print(
        f"  test  : {len(test):>4} days  "
        f"{test.index[0].date()} -> {test.index[-1].date()}"
    )

    # Step 2: estimate the hedge ratio on the TRAIN half only.
    ins = engle_granger(train["base"], train["quote"])
    print("\n  in-sample (train half): estimate the hedge ratio here")
    print(
        f"    hedge ratio beta   : {ins.beta:.3f}  "
        f"({base_name} ≈ {ins.beta:.3f}·{quote_name} + {ins.alpha:,.2f})"
    )
    print(f"    EG p-value         : {ins.p_value:.3f}")
    print(f"    spread half-life   : {_hl_txt(ins.half_life)}")
    print(
        "    -> "
        + (
            "cointegrated in-sample"
            if ins.p_value < 0.05
            else "NOT cointegrated in-sample (nothing to validate)"
        )
    )

    # Steps 3-4: freeze train beta/alpha, build the test-half spread, ADF it.
    oos_spread = test["base"] - (ins.alpha + ins.beta * test["quote"])
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(oos_spread.to_numpy(), autolag="AIC")
    oos_holds = adf_p < 0.05

    print("\n  OUT-OF-SAMPLE (test half, frozen train beta): the real test")
    print(f"    ADF statistic      : {adf_stat:.3f}  (5% crit {adf_crit['5%']:.3f})")
    print(f"    ADF p-value        : {adf_p:.3f}")
    print(f"    spread half-life   : {_hl_txt(half_life(oos_spread))}")
    print(
        "    verdict            : "
        + (
            "HOLDS — frozen-beta spread still stationary out-of-sample"
            if oos_holds
            else "FAILS — frozen-beta spread not stationary out-of-sample"
        )
    )
    if ins.p_value >= 0.05:
        # A hold-out only means something if the training fit was itself a real
        # relationship. If train didn't cointegrate, a passing OOS ADF is luck,
        # not validation — flag it rather than let a green "HOLDS" mislead.
        print("                         (MOOT — train half did not cointegrate, so this")
        print("                          beta is not a validated relationship to carry forward)")

    # Diagnostic: re-estimate beta on the test half on its own. If the hedge ratio
    # drifted a lot, the relationship is unstable even if each half half-passes.
    refit = engle_granger(test["base"], test["quote"])
    drift = abs(refit.beta - ins.beta) / abs(ins.beta) if ins.beta else float("nan")
    print("\n  diagnostic: hedge-ratio stability across halves")
    print(f"    train beta {ins.beta:.3f}  vs  test-refit beta {refit.beta:.3f}  "
          f"(drift {drift:.0%})")

    print("\n  READ:")
    if ins.p_value >= 0.05:
        print("    In-sample fit itself failed on the train half — the full-window")
        print("    result did not survive using only the first half to estimate.")
    elif oos_holds:
        print("    Relationship estimated on the first half still holds on unseen data.")
        print("    Strongest evidence yet — but still one split of one 2-year regime.")
    else:
        print("    The in-sample cointegration did NOT survive out-of-sample: a hedge")
        print("    ratio fit on the first half fails to keep the second-half spread")
        print("    stationary. Consistent with a relationship riding a common trend")
        print("    (the 2024-26 gold move) rather than a stable structural link.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="GLD", help="base ticker (default: GLD)")
    parser.add_argument("quote", nargs="?", default="GDX", help="quote ticker (default: GDX)")
    parser.add_argument(
        "--split",
        type=float,
        default=0.5,
        help="train fraction of the window (default: 0.5)",
    )
    args = parser.parse_args()
    validate(args.base.upper(), args.quote.upper(), args.split)


if __name__ == "__main__":
    main()
