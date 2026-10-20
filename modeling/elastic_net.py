"""
ElasticNet Module: Centered Elastic-Net Regression (L1 + L2)

The elastic net next to `CenteredRidge`, with the same contract (the regression the forecasters
delegate to under `method = "enet"`): the columns are centered on their training means (an
unpenalized intercept), optionally standardized, and the coefficients minimize

    (1/2) ||y - X b||^2 + lam (l1_ratio ||b||_1 + (1 - l1_ratio) ||b||_2^2 / 2)      (scikit-learn, alpha = lam / m)

so `lam` is on the scale of `CenteredRidge`'s penalty: l1_ratio = 0 is the ridge at lambda = lam
(the same coefficients), l1_ratio = 1 the lasso, and in between the L1 part sets weak columns to
exactly zero while the L2 part keeps correlated lags together (the grouping effect of Zou & Hastie,
2005) -- a continuous feature selection, the baseline every hard selection should be measured
against.  Fitted by scikit-learn's coordinate descent (one independent fit per response column, as
the direct models need); at l1_ratio = 0 by the ridge's own solver (`ridge_solve`: the exact solution,
so l1_ratio = 0 IS the ridge at the same lambda -- coordinate descent stops short of it on collinear
columns), as ScalaTion's `ElasticNetRegression` (rev5.4).  `effDF` is the number of nonzero
coefficients plus the intercept (the unbiased estimate of the lasso's degrees of freedom, Zou, Hastie
& Tibshirani, 2007); `stdErr` is the sandwich covariance over the active set (`regularized.stdErr`,
the L1 part as its local quadratic), NaN for a dropped coefficient.

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet as SkElasticNet

from modeling import regularized
from modeling.centered_ridge import CenteredRidge
from modeling.hyper_parameter import Knob
from modeling.optimization.varpro import ridge_solve

__all__ = ["ElasticNet"]


class ElasticNet:
    """
    Elastic-net regression on centered (optionally standardized) columns with an unpenalized
    intercept; the interface of `CenteredRidge` (`fit`, `predict`, `parameter`, `slopes`,
    `intercept`, `params`, `effDF`).

    Parameters
    ----------
    lam : float, optional
        The penalty on `CenteredRidge`'s scale (None => `CenteredRidge.hp["lambda"]`).
    l1_ratio : float, default=0.5
        The share of the L1 penalty: 0 => ridge, 1 => lasso (scikit-learn's `l1_ratio`).
    scaled : bool, default=False
        Whether to standardize the centered columns before the fit (the penalty is unit-free then;
        with raw columns of very different scales the penalty falls on the small-scale ones).
    maxIter : int, default=10000
        The coordinate-descent iterations.
    tol : float, default=1e-6
        Its tolerance on the dual gap (relative to the response's scale): ScalaTion's `solve` stops at a
        coefficient change of 1e-6, and at this tolerance the two agree to five digits (at scikit-learn's
        default 1e-4 the coefficients differ in the fourth digit).
    """

    #: The knobs a tuner turns (`Forecaster.tune`): the penalty lambda and the share l1_ratio
    #: (ScalaTion's `ElasticNetRegression.tunable`).
    tunable = (Knob.LAMBDA, Knob.L1_RATIO)

    def __init__(self, lam: Optional[float] = None, l1_ratio: float = 0.5, scaled: bool = False,
                 maxIter: int = 10000, tol: float = 1e-6) -> None:
        self.lam = float(CenteredRidge.hp["lambda"] if lam is None else lam)
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"ElasticNet: l1_ratio = {l1_ratio} must lie in [0, 1]")
        self.l1_ratio = float(l1_ratio)
        self.scaled = scaled
        self.maxIter, self.tol = int(maxIter), float(tol)
        self.parameter: Optional[np.ndarray] = None
        self.slopes: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ElasticNet":
        """
        Fits the coefficients on (X, Y): one elastic net per response column, on the centered
        (scaled) columns, alpha = lam / m so `lam` matches the ridge's penalty.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            The input matrix (no constant column).
        Y : np.ndarray, shape (m,) or (m, ny)
            The response vector or matrix.
        """
        Y2 = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        m = X.shape[0]
        xm, ym = X.mean(axis=0), Y2.mean(axis=0)
        Xc = X - xm
        sd = CenteredRidge.xSig(Xc) if self.scaled else np.ones(X.shape[1])
        Xs, Yc = Xc / sd, Y2 - ym
        if self.l1_ratio == 0.0:                                        # the ridge exactly (its own solver)
            self.parameter = ridge_solve(Xs, Yc, self.lam)
        else:
            alpha = max(self.lam / m, 1e-12)                             # scikit-learn's alpha (0 is refused)
            net = SkElasticNet(alpha=alpha, l1_ratio=self.l1_ratio, fit_intercept=False,
                               max_iter=self.maxIter, tol=self.tol)
            with warnings.catch_warnings():                              # a near-OLS lasso (tiny alpha) converges slowly;
                warnings.simplefilter("ignore", ConvergenceWarning)     # the coefficients are still at the tolerance's scale
                net.fit(Xs, Yc)
            self.parameter = np.atleast_2d(net.coef_).T + 0.0            # (n, ny) in the fitted space (no -0.0)
        self.slopes = self.parameter / sd[:, None]
        self.intercept = ym - xm @ self.slopes
        self._m, self._xm, self._sd, self._Xs = m, xm, sd, Xs             # the training window (`stdErr`)
        self._sse = np.sum((Yc - Xs @ self.parameter) ** 2, axis=0)     # the residual sum of squares per response
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The predictions (m, ny) of the rows of X (original units)."""
        return X @ self.slopes + self.intercept

    @property
    def params(self) -> np.ndarray:
        """The intercepts and the original-unit slopes stacked, intercept first: (n + 1, ny)."""
        return np.vstack((self.intercept, self.slopes))

    @property
    def nnz(self) -> int:
        """The number of nonzero coefficients (response 0): the columns the L1 penalty kept."""
        if self.parameter is None:
            raise ValueError("ElasticNet.nnz: fit first")
        return int(np.sum(self.parameter[:, 0] != 0.0))

    @property
    def effDF(self) -> float:
        """The effective number of parameters: the nonzero coefficients plus the intercept."""
        return 1.0 + self.nnz

    def stdErr(self, k: int = 0) -> np.ndarray:
        """
        The standard errors of the intercept and the original-unit slopes of response k, (n + 1,):
        `regularized.stdErr` on the last training window at this net's lambda and l1_ratio (the
        sandwich covariance over the active set); NaN for a coefficient the L1 penalty set to zero.
        """
        if self.parameter is None:
            raise ValueError("ElasticNet.stdErr: fit first")
        return regularized.stdErr(self._Xs, self.parameter[:, k], float(self._sse[k]), self.lam, self.l1_ratio,
                                  self._xm, self._sd)

    def __repr__(self) -> str:
        return f"ElasticNet(lam={self.lam:g}, l1_ratio={self.l1_ratio:g}, scaled={self.scaled})"
