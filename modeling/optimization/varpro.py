"""
Variable Projection (VarPro) for Separable Nonlinear Least Squares

Minimizes, over the k nonlinear parameters a within a box [lb, ub],

    sum_j || y_j - X(a) b_j ||^2 + lambda || b_j ||^2        (j = 1..ny responses)

where the basis X(a) (m-by-n) depends on a and the linear coefficients b_j (one set per
response, all sharing a) are eliminated: for a given a they are the ridge solution on the
centered (optionally standardized) basis, with an implicit, unpenalized intercept.  The
outer problem is a bounded nonlinear least-squares problem on the reduced residual.

Three optimizers share the same objective (the `optimizer` option of `NARX_SR_D`):

- "varpro"  : scipy `least_squares` (trust region reflective) on the reduced residual r(a);
- "lbfgsb"  : scipy `minimize` (L-BFGS-B) on the reduced objective || r(a) ||^2;
- "joint"   : scipy `least_squares` over (a, b_1, ..., b_ny) together -- the "one optimizer
              over everything" alternative VarPro is meant to beat.

The Python counterpart of ScalaTion's `VarPro` and `NARX_SR_D.fitNL`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize


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
    steps : dict
        Counts of the last optimization (function evaluations, iterations, status).
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
        self.steps: Dict[str, object] = {}
        self._cache: Tuple[Optional[np.ndarray], ...] = (None, None, None, None)

    # -- the inner (linear) problem --------------------------------------------------------

    def system(self, a: np.ndarray) -> np.ndarray:
        """The centered (and scaled) basis at a, shape (m, n); caches the column means/scales."""
        X = np.asarray(self._basis(np.asarray(a, dtype=float)), dtype=float)
        xm = X.mean(axis=0)
        Xc = X - xm
        if self.scaled:
            sd = Xc.std(axis=0)
            sd[sd == 0.0] = 1.0
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

    def _clip(self, a0: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(a0, dtype=float), self.lb), self.ub)

    def optimize(self, a0: np.ndarray, method: str = "trf", jac: str = "2-point",
                 **kwargs) -> np.ndarray:
        """
        VarPro: bounded nonlinear least squares on the reduced residual.

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
        res = least_squares(self.residual, self._clip(a0), jac=jac, bounds=(self.lb, self.ub),
                            method=method, **kwargs)
        self.steps = {"optimizer": "varpro", "nfev": int(res.nfev), "njev": int(res.njev or 0),
                      "status": int(res.status), "cost": float(2.0 * res.cost)}
        return res.x

    def optimize_lbfgsb(self, a0: np.ndarray, **kwargs) -> np.ndarray:
        """
        L-BFGS-B on the reduced objective (bounds honoured, gradient by finite differences).
        The `least_squares` keywords `max_nfev` and `ftol` are translated (maxfun, ftol); the
        others that only `least_squares` knows (`x_scale`, `jac`, `method`) are ignored.
        """
        bounds = [(None if not np.isfinite(l) else l, None if not np.isfinite(u) else u)
                  for l, u in zip(self.lb, self.ub)]
        options = dict(kwargs.pop("options", {}))
        if "max_nfev" in kwargs:
            options["maxfun"] = int(kwargs.pop("max_nfev"))
        if "ftol" in kwargs:
            options["ftol"] = float(kwargs.pop("ftol"))
        for k in ("x_scale", "jac", "method", "xtol", "gtol"):
            kwargs.pop(k, None)
        res = minimize(self.objective, self._clip(a0), method="L-BFGS-B", bounds=bounds, options=options, **kwargs)
        self.steps = {"optimizer": "lbfgsb", "nfev": int(res.nfev), "nit": int(res.nit),
                      "status": int(res.status), "cost": float(res.fun)}
        return res.x

    def optimize_joint(self, a0: np.ndarray, method: str = "trf", **kwargs) -> np.ndarray:
        """
        One bounded least-squares problem over w = (a, b_1, ..., b_ny) together, started at
        a0 and the ridge coefficients at a0; the residual is [y_j - X(a) b_j ; -sqrt(lambda) b_j].
        """
        a0 = self._clip(a0)
        B0 = self.coefficients(a0)
        n = B0.shape[0]
        k = self.k

        def res_joint(w: np.ndarray) -> np.ndarray:
            a = w[:k]
            B = w[k:].reshape(n, self.ny, order="F")
            Xc = self.system(a)
            r = (self.Yc - Xc @ B).ravel(order="F")
            if self.lam > 0.0:
                r = np.concatenate((r, -np.sqrt(self.lam) * B.ravel(order="F")))
            return r

        w0 = np.concatenate((a0, B0.ravel(order="F")))
        lb = np.concatenate((self.lb, np.full(n * self.ny, -np.inf)))
        ub = np.concatenate((self.ub, np.full(n * self.ny, np.inf)))
        res = least_squares(res_joint, w0, bounds=(lb, ub), method=method, **kwargs)
        self.steps = {"optimizer": "joint", "nfev": int(res.nfev), "njev": int(res.njev or 0),
                      "status": int(res.status), "cost": float(2.0 * res.cost)}
        return res.x[:k]

    def fit(self, a0: np.ndarray, optimizer: str = "varpro", **kwargs) -> np.ndarray:
        """
        Fits the nonlinear parameters with the named optimizer.

        Parameters
        ----------
        a0 : np.ndarray, shape (k,)
            The start.
        optimizer : {"varpro", "lbfgsb", "joint"}, default="varpro"
            The optimizer (see the module docstring).
        **kwargs
            Passed on to the optimizer.

        Returns
        -------
        np.ndarray, shape (k,)
        """
        if optimizer == "varpro":
            return self.optimize(a0, **kwargs)
        if optimizer == "lbfgsb":
            return self.optimize_lbfgsb(a0, **kwargs)
        if optimizer == "joint":
            return self.optimize_joint(a0, **kwargs)
        raise ValueError(f"optimizer '{optimizer}' unknown: varpro, lbfgsb or joint")
