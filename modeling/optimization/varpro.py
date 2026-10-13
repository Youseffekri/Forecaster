"""
Variable Projection (VarPro) for Separable Nonlinear Least Squares

Minimizes, over the k nonlinear parameters a within a box [lb, ub],

    sum_j || y_j - X(a) b_j ||^2 + lambda || b_j ||^2        (j = 1..ny responses)

where the basis X(a) (m-by-n) depends on a and the linear coefficients b_j (one set per
response, all sharing a) are eliminated: for a given a they are the ridge solution on the
centered (optionally standardized) basis, with an implicit, unpenalized intercept.  The
outer problem is a bounded nonlinear least-squares problem on the reduced residual r(a),
solved by scipy's `least_squares` (trust region reflective, ScalaTion's `LevenbergMarquardt_TR`).

The alternatives that share the same objective -- L-BFGS-B on the reduced objective and one
least-squares problem over (a, b) together -- are the other cases of `OptimizerNL`, run by
`NARX_SR_D.fitNL` on this class's `objective`, `system` and `coefficients`.

The Python counterpart of ScalaTion's `optimization/nonlinear_ls/VarPro.scala`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


def ridge_solve(Xc: np.ndarray, Yc: np.ndarray, lam: float) -> np.ndarray:
    """
    The ridge coefficients of centered responses on a centered basis, shape (n, ny):
    the least-squares solution of the augmented system [Xc; sqrt(lam) I] b = [Yc; 0]
    (minimum-norm when the basis is rank deficient, so a degenerate a never fails).

    Parameters
    ----------
    Xc : np.ndarray, shape (m, n)
        The centered (and possibly scaled) basis.
    Yc : np.ndarray, shape (m, ny)
        The centered responses.
    lam : float
        The ridge penalty (0 => ordinary least squares).
    """
    n = Xc.shape[1]
    if lam > 0.0:
        A = np.vstack((Xc, np.sqrt(lam) * np.eye(n)))
        R = np.vstack((Yc, np.zeros((n, Yc.shape[1]))))
    else:
        A, R = Xc, Yc
    return np.linalg.lstsq(A, R, rcond=None)[0]


class VarPro:
    """
    Variable projection for a separable model y_j = c_j + X(a) b_j, j = 1..ny.

    Parameters
    ----------
    basis : callable
        a -> X(a), the design matrix (m, n) at the nonlinear parameters a (k,).
    Y : np.ndarray, shape (m,) or (m, ny)
        The responses (one column per horizon in a direct multi-horizon model).
    lb, ub : np.ndarray, shape (k,)
        The box (use -inf / inf for no bound).
    lam : float, default=0.0
        The ridge penalty on the coefficients (the intercept is not penalized).
    scaled : bool, default=False
        Whether the centered basis columns are standardized before the ridge solve
        (the coefficients are reported in the original units either way).

    Attributes
    ----------
    stepCounts : dict of str -> int
        The counts of the last `optimize` run (function and Jacobian evaluations, status), as
        ScalaTion's `VarPro.stepCounts` (the accepted steps of the outer solver by kind).
    """

    def __init__(self, basis: Callable[[np.ndarray], np.ndarray], Y: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray, lam: float = 0.0, scaled: bool = False) -> None:
        self._basis = basis
        Y = np.asarray(Y, dtype=float)
        self.Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        self.m, self.ny = self.Y.shape
        self.ymean = self.Y.mean(axis=0)
        self.Yc = self.Y - self.ymean
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.k = len(self.lb)
        self.lam = float(lam)
        self.scaled = scaled
        self.stepCounts: Dict[str, int] = {}
        self._cache: Tuple[Optional[np.ndarray], ...] = (None, None, None, None)

    # -- the inner (linear) problem --------------------------------------------------------

    @property
    def response(self) -> np.ndarray:
        """The centered responses, shape (m, ny) (ScalaTion's `VarPro.response`)."""
        return self.Yc

    def system(self, a: np.ndarray) -> np.ndarray:
        """The centered (and scaled) basis at a, shape (m, n); caches the column means/scales."""
        X = np.asarray(self._basis(np.asarray(a, dtype=float)), dtype=float)
        xm = X.mean(axis=0)
        Xc = X - xm
        if self.scaled:                                              # unbiased stdev, as ScalaTion's VarPro
            sd = Xc.std(axis=0, ddof=1) if Xc.shape[0] > 1 else np.ones(Xc.shape[1])
            sd = np.where((sd > 0.0) & np.isfinite(sd), sd, 1.0)
            Xc = Xc / sd
        else:
            sd = np.ones(X.shape[1])
        self._cache = (np.array(a, dtype=float), Xc, xm, sd)
        return Xc

    def coefficients(self, a: np.ndarray) -> np.ndarray:
        """The ridge coefficients at a in the (scaled) basis, shape (n, ny)."""
        Xc = self.system(a)
        return ridge_solve(Xc, self.Yc, self.lam)

    def linear(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The intercepts and coefficients at a, in the ORIGINAL column units.

        Returns
        -------
        (intercepts, B) : (np.ndarray (ny,), np.ndarray (n, ny))
        """
        B = self.coefficients(a)
        _, _, xm, sd = self._cache
        b = B / sd[:, None]
        c = self.ymean - xm @ b
        return c, b

    def residual(self, a: np.ndarray) -> np.ndarray:
        """The reduced residual r(a): the fit residuals of every response stacked, followed by
        the ridge rows sqrt(lambda) b_j when lambda > 0."""
        B = self.coefficients(a)
        Xc = self._cache[1]
        R = self.Yc - Xc @ B
        r = R.ravel(order="F")
        if self.lam > 0.0:
            r = np.concatenate((r, np.sqrt(self.lam) * B.ravel(order="F")))
        return r

    def objective(self, a: np.ndarray) -> float:
        """The reduced objective || r(a) ||^2."""
        r = self.residual(a)
        return float(r @ r)

    # -- the outer (nonlinear) problem ------------------------------------------------------

    def clip(self, a0: np.ndarray) -> np.ndarray:
        """The start a0 clipped into the box [lb, ub]."""
        return np.minimum(np.maximum(np.asarray(a0, dtype=float), self.lb), self.ub)

    def optimize(self, a0: np.ndarray, method: str = "trf", jac: str = "2-point",
                 **kwargs) -> np.ndarray:
        """
        Variable Projection: bounded nonlinear least squares on the reduced residual r(a).

        Parameters
        ----------
        a0 : np.ndarray, shape (k,)
            The start (clipped into the box).
        method : {"trf", "dogbox"}, default="trf"
            The scipy `least_squares` method ("lm" takes no bounds).
        jac : {"2-point", "3-point", "cs"}, default="2-point"
            The finite-difference scheme of the Jacobian.
        **kwargs
            Passed to `least_squares` (ftol, xtol, max_nfev, ...).

        Returns
        -------
        np.ndarray, shape (k,)
            The fitted nonlinear parameters.
        """
        res = least_squares(self.residual, self.clip(a0), jac=jac, bounds=(self.lb, self.ub),
                            method=method, **kwargs)
        self.stepCounts = {"nfev": int(res.nfev), "njev": int(res.njev or 0), "status": int(res.status)}
        return res.x
