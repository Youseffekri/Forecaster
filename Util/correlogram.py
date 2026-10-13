"""
Correlogram: Auto- and Cross-Correlation of Time Series and the Residual Tests

The correlation statistics of a series and the residual tests a fitted model is checked with:

    acov, acorr      the k-lag autocovariance / autocorrelation (STATIONARY form)
    ccorr, corr      the k-lag cross-correlation x_i vs y_i+k, Pearson's correlation
    durbinLevinson   the Durbin-Levinson recursion (its psi matrix's diagonal is the PACF)
    acfPacf          the ACF and PACF at lags 0 .. maxLag
    ljungBox         the Ljung-Box portmanteau test of whiteness, (Q, p-value)
    acfSpikes        the lags whose autocorrelation leaves the ~95 % band 2 / sqrt n
    collinearPair    the most collinear pair of columns of a matrix
    heteroscedastic  whether the spread of the residuals grows with the level

The Python counterpart of ScalaTion's `mathstat/Correlogram.scala` (`durbinLevinson`, the ACF / PACF of
`makeCorrelogram`, the top-level `ljungBox`, `acfSpikes`, `collinearPair`, `heteroscedastic`) and of the
statistics of `mathstat/VectorD.scala` it uses (`acov`, `acorr`, `ccorr`, `corr`, `cov`), with ScalaTion's
formulas and defaults (`acorr` adjusted by default, `corr` 1.0 / -0.0 when undefined).  Used by `EDA4TS`
(forecasting/eda4ts.py), `ExoScreen` (forecasting/exo_screen.py) and the search layer.  Differences: the
Ljung-Box p-value comes from scipy's chi-square (ScalaTion's `CDF.chiSquareCDF`, with the same shortcut
to 0 far in the tail); a lag k >= n raises ValueError (ScalaTion's `flaw` prints the same message).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import chi2


def _tooShort(fn: str, k: int, n: int) -> None:
    if k >= n:
        raise ValueError(f"{fn}: the vector is not long enough to compute {fn} for lag k = {k}")


# ---------------------------------------------------------------------------------------------
#  The statistics of a vector (ScalaTion's `VectorD.acov`, `acorr`, `ccorr`, `corr`)
# ---------------------------------------------------------------------------------------------

def acov(x: np.ndarray, k: int = 1, adjusted: bool = True) -> float:
    """
    The k-lag autocovariance of a STATIONARY series (ScalaTion's `VectorD.acov`): the sum of
    (x_i - mu) (x_i+k - mu) divided by n - k when `adjusted`, by n - 1 otherwise.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    _tooShort("acov", k, n)
    mu = x.mean()
    return float(np.sum((x[: n - k] - mu) * (x[k:] - mu))) / ((n - k) if adjusted else (n - 1))


def acorr(x: np.ndarray, k: int = 1, adjusted: bool = True) -> float:
    """
    The k-lag autocorrelation of a STATIONARY series (ScalaTion's `VectorD.acorr`): the MLE estimate
    sum_{i>=k} z_i z_i-k / sum z_i^2 (z = x - mean), times n / (n - k) when `adjusted` (the default,
    as ScalaTion's); NaN for a constant series.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    _tooShort("acorr", k, n)
    z = x - x.mean()
    den = float(z @ z)
    if den <= 0.0:
        return math.nan
    mle = float(z[k:] @ z[: n - k]) / den
    return mle * n / (n - k) if adjusted else mle


def ccorr(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """
    The k-lag cross-correlation of x with y (ScalaTion's `VectorD.ccorr`, STATIONARY form): x_i
    against y_i+k, the means and standard deviations over the whole vectors,

        ccorr = sum_{i < n-k} (x_i - mu_x) (y_i+k - mu_y) / (n - 1) / (sd_x sd_y)

    -- for an exogenous x and the response y, the correlation of y_t with x_t-k (x leading by k
    steps); NaN when a standard deviation is zero.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        The leading series.
    y : np.ndarray, shape (n,)
        The series it leads.
    k : int, default=1
        The lag (0 <= k < n).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    _tooShort("ccorr", k, n)
    mu_x, mu_y = x.mean(), y.mean()
    ccov = np.sum((x[: n - k] - mu_x) * (y[k:] - mu_y)) / (n - 1)
    den = np.std(x, ddof=1) * np.std(y, ddof=1)
    return float(ccov / den) if den > 0.0 else float("nan")


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson's correlation of x with y (ScalaTion's `VectorD.corr`: cov / sqrt (var_x var_y), with
    cov = (x . y - sum x sum y / n) / (n - 1)); when undefined (a zero variance) 1.0 if x and y are
    the same vector, -0.0 otherwise.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    cov = (x @ y - x.sum() * y.sum() / n) / (n - 1)
    den = np.sqrt(np.var(x, ddof=1) * np.var(y, ddof=1))
    if not den > 0.0 or not np.isfinite(den):
        return 1.0 if np.array_equal(x, y) else -0.0
    return float(cov / den)


# ---------------------------------------------------------------------------------------------
#  The correlogram (ScalaTion's `Correlogram`)
# ---------------------------------------------------------------------------------------------

def durbinLevinson(g: np.ndarray, ml: int) -> np.ndarray:
    """
    The Durbin-Levinson recursion on an autocovariance vector g (lags 0..ml): the psi matrix
    whose diagonal is the PACF (ScalaTion's `Correlogram.durbinLevinson`).
    """
    psi = np.zeros((ml + 1, ml + 1))
    r = np.zeros(ml + 1)
    r[0] = g[0]
    for k in range(1, ml + 1):
        s = sum(psi[k - 1, j] * g[k - j] for j in range(1, k))
        a = (g[k] - s) / r[k - 1] if r[k - 1] != 0.0 else 0.0
        psi[k, k] = a
        for j in range(1, k):
            psi[k, j] = psi[k - 1, j] - a * psi[k - 1, k - j]
        r[k] = r[k - 1] * (1.0 - a * a)
    return psi


def acfPacf(y: np.ndarray, maxLag: int, adjusted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    The ACF and PACF of a series at lags 0..maxLag (ScalaTion's `Correlogram.makeCorrelogram`: the
    autocovariances `acov` (adjusted), the ACF their ratio to lag 0, the PACF the diagonal of the
    Durbin-Levinson psi matrix).
    """
    y = np.asarray(y, dtype=float)
    ml = min(len(y) - 1, maxLag)
    acv = np.array([acov(y, k, adjusted) for k in range(ml + 1)])
    acf = acv / acv[0] if acv[0] != 0.0 else np.full(ml + 1, math.nan)
    pacf = np.diag(durbinLevinson(acv, ml)).copy()
    pacf[0] = 1.0
    return acf, pacf


# ---------------------------------------------------------------------------------------------
#  The residual tests (ScalaTion's `mathstat/Correlogram.scala` top-level functions)
# ---------------------------------------------------------------------------------------------

def ljungBox(e: np.ndarray, maxLag: int, dof: int = 0) -> Tuple[float, float]:
    """
    The Ljung-Box portmanteau test for the whiteness of a (residual) series e,
    Q = n (n + 2) sum_{k=1}^{L} r_k^2 / (n - k), r_k the lag-k autocorrelation (MLE form);
    under H0 Q ~ ChiSquare (L - dof).  Returns (Q, p-value); the p-value is NaN when L - dof < 1
    and 0 beyond Q = 100 + 3 (L - dof), where the tail is below 1E-22 (ScalaTion's shortcut).

    Parameters
    ----------
    e : np.ndarray, shape (n,)
        The series (e.g., residuals).
    maxLag : int
        The number of lags L pooled into Q.
    dof : int, default=0
        The number of fitted ARMA parameters to subtract from the degrees of freedom.
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    ml = min(maxLag, n - 1)
    q = n * (n + 2.0) * sum(acorr(e, k, adjusted=False) ** 2 / (n - k) for k in range(1, ml + 1))
    df = ml - dof
    if df < 1:
        return float(q), math.nan
    if q > 100.0 + 3.0 * df:
        return float(q), 0.0
    return float(q), float(max(0.0, chi2.sf(q, df)))


def acfSpikes(e: np.ndarray, maxLag: int, used: Set[int] = frozenset()) -> List[Tuple[int, float]]:
    """
    The lags at which the autocorrelation of e exceeds the approximate 95 % band 2 / sqrt(n),
    strongest first, as (lag, r_k) pairs, skipping the lags in `used` (e.g., the lags already present
    as predictors in the model whose residuals e are).
    """
    e = np.asarray(e, dtype=float)
    band = 2.0 / math.sqrt(len(e))
    out = [(k, acorr(e, k, adjusted=False)) for k in range(1, min(maxLag, len(e) - 1) + 1)]
    out = [(k, r) for k, r in out if abs(r) > band and k not in used]
    return sorted(out, key=lambda kr: -abs(kr[1]))


def collinearPair(x: np.ndarray, idx: Optional[Sequence[int]] = None) -> Tuple[int, int, float]:
    """
    The most collinear pair among the given columns of x: the pair with the largest |corr|, as
    (i, j, |corr|), or (-1, -1, 0) when fewer than two columns are given.
    """
    x = np.asarray(x, dtype=float)
    cols = list(range(x.shape[1])) if idx is None else list(idx)
    bi, bj, bc = -1, -1, 0.0
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            c = abs(corr(x[:, cols[a]], x[:, cols[b]]))
            if c > bc:
                bi, bj, bc = cols[a], cols[b], c
    return bi, bj, bc


def heteroscedastic(e: np.ndarray, y: np.ndarray, thresh: float = 0.3) -> bool:
    """Whether the spread of e grows (or shrinks) with the level of y: |corr (|e|, y)| > thresh."""
    return abs(corr(np.abs(np.asarray(e, dtype=float)), y)) > thresh
