"""
CenteredRidge Module: Centered (Multi-Variate) Ridge Regression

The Python counterpart of ScalaTion's `CenteredRidge` / `CenteredRidgeMV` (the regression the
forecasting models delegate to): the columns of X and Y are centered on their training means (so
the intercept is not penalized and is recovered from the means), optionally standardized by their
(unbiased) standard deviations, and the coefficients are solved by ridge regression with penalty
lambda (lambda = 0 gives OLS).  `method="ridge"` of the forecasters and the coefficient fit of
`NARX_SR_D` use it.  As in ScalaTion, `parameter` holds the coefficients in the FITTED (centered,
standardized when scaled) space, `slopes` and `intercept` the original-unit ones, and `hp` the
default penalty (ScalaTion's `CenteredRidge.hp = RidgeRegression.hp`, lambda = 0.1).  `effDF` (the
trace of the ridge hat matrix plus the intercept) and `stdErr` (the sandwich covariance of the ridge
estimator) are the ridge-aware statistics the SUMMARY and the penalised selection score use.

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.7
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from modeling.hyper_parameter import HyperParameter, Knob
from modeling.optimization.varpro import ridge_solve


class CenteredRidge:
    """
    Ridge regression on centered (optionally standardized) columns with an unpenalized intercept.

    Parameters
    ----------
    lam : float, optional
        The ridge penalty lambda (0 => ordinary least squares); None => `CenteredRidge.hp["lambda"]`.
    scaled : bool, default=False
        Whether to standardize the centered columns (training mean and unbiased standard deviation)
        before the solve, so the penalty is unit-free.

    Attributes
    ----------
    parameter : np.ndarray, shape (n, ny)
        The coefficients in the fitted space (ScalaTion's `parameter` / `bb`).
    slopes : np.ndarray, shape (n, ny)
        The slopes in original units (ScalaTion's `slopes`): `parameter / xSig` when scaled.
    intercept : np.ndarray, shape (ny,)
        The intercepts in original units (ScalaTion's `intercept`).
    """

    #: ScalaTion's `CenteredRidge.hp` (= `RidgeRegression.hp`): the default ridge penalty.
    hp = HyperParameter().add("lambda", 0.1, 0.1)
    #: The knobs a tuner turns (`Forecaster.tune`): the penalty lambda (ScalaTion's `Regularized.tunable`).
    tunable = (Knob.LAMBDA,)

    def __init__(self, lam: Optional[float] = None, scaled: bool = False) -> None:
        self.lam = float(self.hp["lambda"] if lam is None else lam)
        self.scaled = scaled
        self.parameter: Optional[np.ndarray] = None
        self.slopes: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None

    @classmethod
    def ridge(cls, hparam: Optional[Mapping[str, Any]]) -> float:
        """The penalty a model's regression uses: hparam["lambda"] if present, `hp["lambda"]` otherwise
        (ScalaTion's `NARX_SR_D.ridge`)."""
        if hparam is not None and "lambda" in hparam:
            return float(hparam["lambda"])
        return float(cls.hp["lambda"])

    @staticmethod
    def xSig(Xc: np.ndarray) -> np.ndarray:
        """The column scales of a centered matrix: the unbiased standard deviations, 1 for a constant
        column (ScalaTion's `fitTransforms`)."""
        sd = Xc.std(axis=0, ddof=1) if Xc.shape[0] > 1 else np.ones(Xc.shape[1])
        return np.where((sd > 0.0) & np.isfinite(sd), sd, 1.0)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CenteredRidge":
        """
        Fits the coefficients on (X, Y) (the centering and scaling come from these rows only).

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            The input matrix (no constant column).
        Y : np.ndarray, shape (m,) or (m, ny)
            The response vector or matrix.
        """
        Y2 = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        xm, ym = X.mean(axis=0), Y2.mean(axis=0)
        Xc = X - xm
        sd = self.xSig(Xc) if self.scaled else np.ones(X.shape[1])
        Xs, Yc = Xc / sd, Y2 - ym
        self.parameter = ridge_solve(Xs, Yc, self.lam)
        self.slopes = self.parameter / sd[:, None]
        self.intercept = ym - xm @ self.slopes
        self._m, self._xm, self._sd = X.shape[0], xm, sd                 # for `effDF` and `stdErr`
        self._d, self._Vt = np.linalg.svd(Xs, full_matrices=False)[1:]  # the singular values of the fitted basis
        self._sse = np.sum((Yc - Xs @ self.parameter) ** 2, axis=0)     # the residual sum of squares per response
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The predictions (m, ny) of the rows of X (original units)."""
        return X @ self.slopes + self.intercept

    @property
    def params(self) -> np.ndarray:
        """The intercepts and the original-unit slopes stacked, intercept first: (n + 1, ny)
        (ScalaTion's `parameter_org`; the layout `Forecaster.params` uses)."""
        return np.vstack((self.intercept, self.slopes))

    # -- ridge-aware statistics --------------------------------------------------------------

    @property
    def effDF(self) -> float:
        """
        The effective number of parameters of the fit: 1 (the unpenalized intercept) + the trace of the
        ridge hat matrix H = X (X'X + lambda I)^-1 X' = sum_i d_i^2 / (d_i^2 + lambda) over the singular
        values d_i of the fitted basis -- n + 1 at lambda = 0 (OLS), falling toward 1 as lambda grows.
        The complexity a penalised criterion should charge a ridge fit (Hastie, Tibshirani & Friedman,
        sec. 3.4.1), where the OLS count n + 1 overstates it.
        """
        if self.parameter is None:
            raise ValueError("CenteredRidge.effDF: fit first")
        d2 = self._d ** 2
        return 1.0 + float(np.sum(d2 / (d2 + self.lam)))

    def stdErr(self, k: int = 0) -> np.ndarray:
        """
        The standard errors of the intercept and the original-unit slopes of response k, (n + 1,),
        from the covariance of the ridge estimator (the sandwich form, NOT the OLS (X'X)^-1):

            Cov (b) = sigma^2 (X'X + lambda I)^-1 X'X (X'X + lambda I)^-1 = sigma^2 V diag (d^2 / (d^2 + lambda)^2) V'

        with sigma^2 = SSE / (m - effDF) (the residual degrees of freedom of the ridge), the slopes'
        variances divided by the column scales squared when scaled, and the intercept's
        sigma^2 / m + xbar' Cov (slopes) xbar.  At lambda = 0 these are the OLS standard errors; with
        lambda > 0 they are finite even when X'X is singular (the NaN of the OLS formula on collinear lags).
        """
        if self.parameter is None:
            raise ValueError("CenteredRidge.stdErr: fit first")
        d2 = self._d ** 2
        dfRes = self._m - self.effDF
        sigma2 = float(self._sse[k]) / dfRes if dfRes > 0 else float("nan")
        shrink = d2 / (d2 + self.lam) ** 2                                # diag (d^2 / (d^2 + lambda)^2)
        cov_b = (self._Vt.T * shrink) @ self._Vt                         # V diag (.) V' in the fitted space
        cov_s = cov_b / np.outer(self._sd, self._sd)                      # the original-unit slopes
        var_s = np.diag(cov_s)
        var_0 = 1.0 / self._m + float(self._xm @ cov_s @ self._xm)
        return np.sqrt(sigma2 * np.append(var_0, var_s))
