"""Shared statistics: cointegration (half-life, Engle-Granger) and momentum.

Kept separate from ``data/history.py`` (which only loads and aligns CSVs) so the
data-access layer stays free of modeling. Shared across the analysis scripts so
each test's maths lives in exactly one place: ``check_correlation.py`` /
``screen_pairs.py`` (cointegration), and ``check_momentum.py`` /
``check_momentum_stocks.py`` (momentum autocorrelation).

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


@dataclass
class MomentumResult:
    """Correlation of a past return with the next same-length return."""

    corr: float  # NaN when n_obs < 3
    n_obs: int  # overlapping (per-bar) observations
    n_indep: int  # non-overlapping windows the noise band is based on
    noise_band: float  # ~95% band (2/sqrt(n_indep)); |corr| below it ≈ zero


def momentum_autocorrelation(log_prices: pd.Series, lookback: int) -> MomentumResult:
    """Correlate the past ``lookback``-step return with the next same-length return.

    Positive corr suggests momentum, negative suggests mean-reversion, near-zero
    no linear effect at that horizon. Inputs are log prices so returns are simple
    differences.

    The windows overlap (computed at every bar), which autocorrelates the
    observations and inflates the effective sample. The honesty guard is the noise
    band: ``2/sqrt(n_indep)`` from the count of *non-overlapping* windows
    (``n_obs // lookback``). A correlation smaller than the band is indistinguishable
    from zero and must not be read as a signal. Returns ``corr=NaN`` when fewer than
    three paired observations exist (e.g. a lookback too long for the window).
    """
    past = log_prices.diff(lookback)  # return over [t-lookback, t]
    fwd = log_prices.shift(-lookback) - log_prices  # return over [t, t+lookback]
    paired = np.column_stack([past.to_numpy(), fwd.to_numpy()])
    paired = paired[~np.isnan(paired).any(axis=1)]
    n_obs = len(paired)
    if n_obs < 3:
        return MomentumResult(corr=float("nan"), n_obs=n_obs, n_indep=0, noise_band=float("nan"))
    corr = float(np.corrcoef(paired[:, 0], paired[:, 1])[0, 1])
    n_indep = max(n_obs // lookback, 1)
    return MomentumResult(
        corr=corr, n_obs=n_obs, n_indep=n_indep, noise_band=2.0 / np.sqrt(n_indep)
    )
