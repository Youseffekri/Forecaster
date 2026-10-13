"""
Regression Module: Centered (Multi-Variate) Ridge Regression

The Python counterpart of ScalaTion's `CenteredRidgeMV` used by the `*_D` forecasting models:
the columns of X and Y are centered on their training means (so the intercept is not penalized
and is recovered from the means), optionally standardized, and the slopes are solved by ridge
regression with penalty lambda (lambda = 0 gives OLS).  `method="ridge"` of `ARX`/`ARX_D` and
the coefficient fit of `NARX_SR_D` use it.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from modeling.optimization.varpro import ridge_solve


class CenteredRidge:
    """
    Ridge regression on centered (optionally standardized) columns with an unpenalized intercept.

    Parameters
    ----------
    lam : float, default=1.0
        The ridge penalty lambda (0 => ordinary least squares).
    scaled : bool, default=False
        Whether to standardize the centered columns before the solve (the slopes are returned
        in original units either way).

    Attributes
    ----------
    intercept_ : np.ndarray, shape (ny,)
        The intercepts in original units (one per output column).
    coef_ : np.ndarray, shape (n, ny)
        The slopes in original units.
    """

    def __init__(self, lam: float = 1.0, scaled: bool = False) -> None:
        self.lam = float(lam)
        self.scaled = scaled
        self.intercept_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CenteredRidge":
        """
        Fits the slopes and intercepts on (X, Y).

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
        if self.scaled:
            sd = Xc.std(axis=0)
            sd[sd == 0.0] = 1.0
            Xc = Xc / sd
        else:
            sd = np.ones(X.shape[1])
        B = ridge_solve(Xc, Y2 - ym, self.lam) / sd[:, None]
        self.coef_ = B
        self.intercept_ = ym - xm @ B
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The predictions (m, ny) of the rows of X."""
        return X @ self.coef_ + self.intercept_

    @property
    def params(self) -> np.ndarray:
        """The parameters with the intercept first: (n + 1, ny)."""
        return np.vstack((self.intercept_, self.coef_))
