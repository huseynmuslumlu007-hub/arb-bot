"""Statistical tests for pair relationships: half-life and Engle-Granger.

Kept separate from ``data/history.py`` (which only loads and aligns CSVs) so the
data-access layer stays free of modeling. Shared by ``scripts/check_correlation.py``
(one pair, detailed narrative) and ``scripts/screen_pairs.py`` (all pairs, batch),
so the cointegration maths lives in exactly one place.

Requires statsmodels (a declared runtime dependency) for the cointegration test.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def half_life(series: pd.Series) -> float:
    """Mean-reversion half-life (in observations) from an AR(1) fit.

    Regress the change on the prior level: ``Δy_t = a + b·y_{t-1}``. A mean-
    reverting series pulls back toward its mean, so ``b`` is negative; the
    half-life — how long to close half the gap to the mean — is ``-ln(2)/b``.
    Returns ``inf`` when ``b >= 0`` (no reversion: the series drifts/random-walks).
    """
    lagged = series.shift(1)
    delta = series - lagged
    fit = pd.concat({"delta": delta, "lagged": lagged}, axis=1).dropna()
    design = np.column_stack([np.ones(len(fit)), fit["lagged"].to_numpy()])
    (_, b), *_ = np.linalg.lstsq(design, fit["delta"].to_numpy(), rcond=None)
    return float(-np.log(2) / b) if b < 0 else float("inf")


@dataclass
class CointResult:
    """Engle-Granger cointegration result for ``base ~ alpha + beta * quote``."""

    beta: float  # hedge ratio: units of quote per unit of base
    alpha: float  # regression intercept
    eg_stat: float  # Engle-Granger test statistic (more negative = more stationary)
    p_value: float  # MacKinnon p-value; H0 = no cointegration
    crit_5pct: float  # 5% critical value for eg_stat
    half_life: float  # of the hedged spread, in observations (inf if no reversion)


def engle_granger(base: pd.Series, quote: pd.Series) -> CointResult:
    """Engle-Granger two-step cointegration test with an estimated hedge ratio.

    Step 1: OLS regress ``base = alpha + beta * quote``; ``beta`` is the hedge
    ratio and the residual ``base - alpha - beta * quote`` is the spread.
    Step 2: test that spread for a unit root via statsmodels' ``coint``, which
    applies the correct critical values for a spread whose ``beta`` was
    *estimated* from the data (estimation makes residuals look more stationary
    than a plain ADF would credit).

    Note EG is direction-dependent: ``engle_granger(a, b)`` and
    ``engle_granger(b, a)`` are not identical. Callers pick a fixed convention.
    """
    from statsmodels.tsa.stattools import coint

    b = base.to_numpy()
    q = quote.to_numpy()
    design = np.column_stack([np.ones(len(q)), q])
    (alpha, beta), *_ = np.linalg.lstsq(design, b, rcond=None)
    spread = base - (alpha + beta * quote)

    eg_stat, p_value, crit = coint(b, q)  # crit = [1%, 5%, 10%]
    return CointResult(
        beta=float(beta),
        alpha=float(alpha),
        eg_stat=float(eg_stat),
        p_value=float(p_value),
        crit_5pct=float(crit[1]),
        half_life=half_life(spread),
    )
