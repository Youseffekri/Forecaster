"""
Regularized Module: the standard errors of a penalized fit

The Python counterpart of ScalaTion's `object Regularized` (rev5.4): ONE function, `stdErr`, that the
two penalized regressions (`CenteredRidge`, `ElasticNet`) call from their `stdErr` -- the covariance
of the penalized estimator over the active set (the sandwich form), NOT the OLS (X'X)^-1, which is
undefined (NaN) on the collinear lag columns the forecasters fit.

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import numpy as np

__all__ = ["stdErr"]


def stdErr(xc: np.ndarray, b: np.ndarray, sse: float, lam: float, alpha: float,
           xm: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    The standard errors, intercept first, of a penalized regression fitted on a centered (standardized
    when scaled) training window, from the covariance of the penalized estimator (ScalaTion's
    `Regularized.stdErr`):

        Cov (b_A) = sigma^2 M Xa'Xa M,   M = (Xa'Xa + D)^-1,   D = diag (lambda (1 - alpha) + lambda alpha / |b_j|)

    over the active set A (the columns with b_j != 0 that vary), the L1 part of the penalty replaced by
    its local quadratic lambda alpha b_j^2 / |b_j| (Tibshirani 1996; Zou & Hastie 2005), so alpha = 0 is
    the ridge sandwich (X'X + lambda I)^-1 X'X (X'X + lambda I)^-1 and lambda = 0 the OLS (X'X)^-1;
    sigma^2 = sse / (m - 1 - tr (M Xa'Xa)), the residual degrees of freedom of the penalized fit.  The
    slopes' variances are divided by the column scales squared (original units) and the intercept's is
    sigma^2 (1/m + xm' Cov xm) (the intercept is recovered from the means).  A coefficient the L1
    penalty set to zero has no standard error (NaN: the SUMMARY prints NA), nor has a constant column;
    every coefficient is NaN when the residual degrees of freedom are gone.

    Parameters
    ----------
    xc : np.ndarray, shape (m, n)
        The training window in the fitted (centered, scaled) space.
    b : np.ndarray, shape (n,)
        The fitted coefficients in that space (one response).
    sse : float
        The sum of squared residuals of the fit.
    lam : float
        The penalty.
    alpha : float
        The share of the L1 penalty (0 for a ridge).
    xm : np.ndarray, shape (n,)
        The column means of the window (original units).
    sd : np.ndarray, shape (n,)
        The column scales (all ones unless scaled).
    """
    m, n = xc.shape
    se = np.full(n + 1, np.nan)
    varies = np.sum(xc * xc, axis=0) > 0.0
    act = np.where(varies if alpha == 0.0 else (varies & (b != 0.0)))[0]   # the active set (every varying column for a ridge)
    if len(act) == 0:                                                     # the mean model: the intercept's error only
        if m > 1:
            se[0] = np.sqrt(sse / (m - 1.0) / m)
        return se
    xa = xc[:, act]
    xtx = xa.T @ xa
    d = lam * (1.0 - alpha) + (lam * alpha / np.abs(b[act]) if alpha > 0.0 else 0.0)
    mInv = np.linalg.inv(xtx + np.diag(np.broadcast_to(d, (len(act),))))   # M
    mx = mInv @ xtx                                                       # M Xa'Xa (its trace: the effective df of the slopes)
    cov = mx @ mInv                                                       # M Xa'Xa M
    dfRes = m - 1.0 - float(np.trace(mx))
    if dfRes <= 0.0:
        return se
    sig2 = sse / dfRes
    cov_s = cov / np.outer(sd[act], sd[act])                              # original units
    xa_m = xm[act]
    se[act + 1] = np.sqrt(sig2 * np.diag(cov_s))
    se[0] = np.sqrt(sig2 * (1.0 / m + float(xa_m @ cov_s @ xa_m)))       # the intercept: sigma^2 (1/m + xm' Cov xm)
    return se
