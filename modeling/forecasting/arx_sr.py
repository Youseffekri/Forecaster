"""
ARX_SR Model: Auto-Regressive on Lagged y and xe with Symbolic Regression (SR) Terms

Extends the `ARX` model with trend, linear, power, root and cross terms: the lagged endogenous
variable y and its transformations (`fEndo`, e.g., `PowForm()` for y^1.5), the lagged exogenous
variables and their own transformations (`fExo`, one list per exogenous variable), and optional
ENDO-EXO cross terms (args.crx).  Multi-step forecasts are RECURSIVE: the forecast of
horizon h feeds the lags of horizon h + 1, and a transformed lag of a forecast is the transform of
the forecast.

With `rescale`, y and the exogenous variables are rescaled FIRST (`tFormT` fitted on the
training rows, as in ScalaTion) and the transformations are applied to the rescaled series.

The Python counterpart of ScalaTion's `ARX_SR` (`ARX_SR.apply` / `ARX_SR.rescale` /
`ARX_SR.buildMatrix` / `formNames` / `forge`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-17
Version: 2.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Util.make_matrix4ts import backfill_matrix, makeMatrix4L, makeMatrix4T, trend_names
from Util.transform import PowForm, Transform, TransformT
from modeling.forecasting.arx import ARX, _rescaleData
from modeling.forecasting.forecaster import Forecaster


def srMatrix(xe_bfill: Optional[np.ndarray], y: np.ndarray, hparam: Dict[str, Any],
             fEndo: Sequence[Transform], fExo: Sequence[Sequence[Transform]]) -> np.ndarray:
    """
    The design matrix of the SR models (ScalaTion's `ARX_SR.buildMatrix` / `NARX_SR_D.buildMatrix`),
    not checked for finiteness: the trend columns, the p lags of y and of each transform of y, then
    for each exogenous variable its px lags and those of each of its transforms, then (crx = 1) the
    px lags of y * xe_j for every j.
    """
    p, q, spec, cross = int(hparam["p"]), int(hparam["px"]), int(hparam["spec"]), int(hparam.get("crx", 0)) == 1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x_endo = np.column_stack([y] + [tr.f(y) for tr in fEndo])
    blocks = [makeMatrix4T(len(y), spec)] if spec > 0 else []
    blocks += [makeMatrix4L(x_endo[:, j], p) for j in range(x_endo.shape[1])] if p > 0 else []
    if xe_bfill is not None and xe_bfill.shape[1] > 0 and q > 0:
        cols = []
        for j in range(xe_bfill.shape[1]):
            xe_j = xe_bfill[:, j]
            cols.append(xe_j)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                cols += [tr.f(xe_j) for tr in (fExo[j] if j < len(fExo) else [])]
        if cross:
            cols += [y * xe_bfill[:, j] for j in range(xe_bfill.shape[1])]
        blocks += [makeMatrix4L(c, q) for c in cols]
    return np.column_stack(blocks)


class ARX_SR(ARX):
    """
    Auto-Regressive model with exogenous variables and Symbolic Regression terms (ARX_SR).

    Attributes
    ----------
    tForms : dict
        The transformations in play:
        - "tForm_y"   : the transform of y (None unless `rescale`), also used to invert forecasts
        - "fEndo"     : list of `Transform` applied to y before lagging (default `[PowForm()]`)
        - "fExo"      : list of lists of `Transform`, one list per exogenous variable
    fExo_sz : list of int
        The number of transformations per exogenous variable.
    fname : list of str
        The column labels (`formNames`): trend, yl{k}, f{i}(yl{k}), xe{j}l{k}, g{j},{i}(xe{j}l{k}), xe{j}l{k}*yl{k}.

    Notes
    -----
    Column order of X (the order `forge` relies on): trend terms (spec >= 1), the p lags of y,
    the p lags of each fEndo transform of y, then for each exogenous variable j its px raw lags
    followed by the px lags of each of its transforms, then (crx = 1) the px lags of y * xe_j
    for every j.

    See Also
    --------
    ARX : the recursive forecasting logic (train, forecast, _predict, hide).
    """

    ARGS = ("p", "px", "spec", "crx", "method", "lam", "scaled") + Forecaster.ARGS

    def __init__(self,
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 args: Any = None,
                 fEndo: Optional[Sequence[Transform]] = None,
                 fExo: Optional[Sequence[Sequence[Transform]]] = None,
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_SR forecaster, building the design matrix (ScalaTion's `ARX_SR.apply`)
        unless a ready matrix X is given.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The endogenous time series.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        xe : Optional[np.ndarray], shape (n_samples, n_exo)
            The exogenous variables aligned with y.
        args : Args or dict, optional
            The settings: p (lags of y), px (lags of each exogenous variable), spec (number of trend
            columns, ScalaTion's convention: 0 implicit constant, 1 linear, 2 quadratic, 3 cubic), crx
            (ENDO-EXO cross terms), nneg, skip, method ("sm_ols", "sk_lr", "ridge"), lam, scaled.
        fEndo : sequence of Transform, optional
            The transformations of y (default `[PowForm()]`, i.e., y^1.5; `[]` for none).
        fExo : sequence of sequences of Transform, optional
            One list of transformations per exogenous variable (default: none for every variable).
        X : Optional[np.ndarray]
            A ready design matrix (then no forging: hh must be 1).

        Raises
        ------
        ValueError
            If `fExo` does not have one entry per exogenous variable.
        """
        n_exo = xe.shape[1] if xe is not None else 0
        fEndo = [PowForm()] if fEndo is None else list(fEndo)
        fExo = [[] for _ in range(n_exo)] if fExo is None else [list(s) for s in fExo]
        if len(fExo) != n_exo:
            raise ValueError(f"Length of fExo {len(fExo)} must = number of exogenous variables {n_exo}")
        self._tForms = {"tForm_y": None, "fEndo": fEndo, "fExo": fExo}
        self._fExo_sz = [len(s) for s in fExo]
        super().__init__(y, hh, xe, args, X if X is not None else np.zeros((len(y), 1)))
        self._modelName = f"ARX_SR_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{self.hpTag}"
        if X is None:                                                    # ScalaTion's `ARX_SR.apply`
            xe_bfill = backfill_matrix(xe) if n_exo > 0 and self._hparam["px"] > 0 else None
            self._set_X(self.buildMatrix(xe_bfill, self._y, self._hparam, fEndo, fExo))
            self._useforge = True

    @classmethod
    def rescale(cls,
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                args: Any = None,
                fEndo: Optional[Sequence[Transform]] = None,
                fExo: Optional[Sequence[Sequence[Transform]]] = None,
                tFormT: TransformT = TransformT.MinMax) -> "ARX_SR":
        """
        Creates the model on rescaled data (ScalaTion's `ARX_SR.rescale`): y and the backfilled
        exogenous variables are transformed by `tFormT` fitted on the training rows (ScalaTion's
        `Model.trSize`; the fit rows while choosing; MinMax to [-2, 2] by default, where a power or
        log of a negative value is not defined -- use Log1p, as ScalaTion's tests do, or a sign-safe
        transform such as PowRForm / DQuadForm), THEN the transformations fEndo / fExo are applied;
        Norm turns the non-negativity off.

        Parameters
        ----------
        y, hh, xe, args, fEndo, fExo
            As in the constructor.
        tFormT : TransformT, default=TransformT.MinMax
            The transform type of the rescaling (ScalaTion's `rescale (..., tFormT)`).

        Returns
        -------
        ARX_SR
        """
        args, y_scl, xe_scl, tForm_y = _rescaleData(cls._argsOf(y, args), y, xe, tFormT)
        mod = cls(y_scl, hh, xe, args, fEndo, fExo, X=np.zeros((len(y_scl), 1)))
        mod._set_X(cls.buildMatrix(xe_scl, y_scl, mod._hparam, mod._tForms["fEndo"], mod._tForms["fExo"]))
        mod._useforge = True
        mod._tForms["tForm_y"] = tForm_y
        mod._setYForm(tForm_y, y)
        mod._recipe = lambda: cls.rescale(y, hh, xe, args, fEndo, fExo, tFormT)
        return mod

    @staticmethod
    def buildMatrix(xe_bfill: Optional[np.ndarray], y: np.ndarray, hparam: Dict[str, Any],
                    fEndo: Sequence[Transform], fExo: Sequence[Sequence[Transform]]) -> np.ndarray:
        """
        Builds the design matrix (ScalaTion's `ARX_SR.buildMatrix`): trend terms, the lags of y and
        of its transformations, the lags of each exogenous variable and of its transformations, and
        the lagged cross terms (see `srMatrix`).

        Parameters
        ----------
        xe_bfill : np.ndarray, shape (n_samples, n_exo), optional
            The backfilled (and rescaled) exogenous variables (None => none).
        y : np.ndarray, shape (n_samples,)
            The endogenous series (rescaled by `rescale`).
        hparam : dict
            "p", "px", "spec", "crx".
        fEndo : sequence of Transform
            The transformations of y.
        fExo : sequence of sequences of Transform
            The transformations of each exogenous variable.

        Returns
        -------
        np.ndarray, shape (n_samples, n_columns)

        Raises
        ------
        ValueError
            If a column is not finite (a power, root or log outside its domain).
        """
        X = srMatrix(xe_bfill, y, hparam, fEndo, fExo)
        bad = ~np.isfinite(X).all(axis=0)
        if bad.any():
            j = int(np.argmax(bad))
            raise ValueError(f"ARX_SR: column {j} of the design matrix is not finite -- a power, root or log of a negative "
                             "or zero value; rescale to a positive range (tFormT = TransformT.Log1p) or use a sign-safe "
                             "transform (PowRForm, DQuadForm)")
        return X

    @property
    def fname(self) -> List[str]:
        """The column labels of X, in ScalaTion's naming (`formNames`; generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return self.formNames(self._n_exo, self._hparam, len(self._tForms["fEndo"]), self._fExo_sz)
        return [f"x{j}" for j in range(self._X.shape[1])]

    @staticmethod
    def formNames(n_exo: int, hparam: Dict[str, Any], n_fEn: int, n_fExArr: Sequence[int]) -> List[str]:
        """
        The column labels in the order `buildMatrix` produces them (ScalaTion's `ARX_SR.formNames`):
        the trend names, yl{p}..yl1, f{i}(yl{k}), xe{j}l{k}, g{j},{i}(xe{j}l{k}), xe{j}l{k}*yl{k}.

        Parameters
        ----------
        n_exo : int
            The number of exogenous variables.
        hparam : dict
            "p", "px", "spec", "crx".
        n_fEn : int
            The number of transformations of y.
        n_fExArr : sequence of int
            The number of transformations per exogenous variable.
        """
        p, q, spec, cross = int(hparam["p"]), int(hparam["px"]), int(hparam["spec"]), int(hparam.get("crx", 0)) == 1
        names = trend_names(spec)
        names += [f"yl{k}" for k in range(p, 0, -1)]
        for i in range(n_fEn):
            names += [f"f{i}(yl{k})" for k in range(p, 0, -1)]
        if q > 0:
            for j in range(n_exo):
                names += [f"xe{j}l{k}" for k in range(q, 0, -1)]
                for i in range(n_fExArr[j] if j < len(n_fExArr) else 0):
                    names += [f"g{j},{i}(xe{j}l{k})" for k in range(q, 0, -1)]
            if cross:
                for j in range(n_exo):
                    names += [f"xe{j}l{k}*yl{k}" for k in range(q, 0, -1)]
        return names

    def forge(self, X_window: np.ndarray, y_fcast: np.ndarray, h: int) -> np.ndarray:
        """
        Builds the feature rows of horizon h from the actual lags and the forecasts of horizons
        1 .. h-1: the raw y block takes the forecasts, each transformed y block takes the
        transform of the forecasts, and the exogenous (raw, transformed, cross) blocks hide the
        values not yet known at horizon h.

        Parameters
        ----------
        X_window : np.ndarray, shape (n_rows, n_columns)
            The rows of X for the time steps being forecast.
        y_fcast : np.ndarray, shape (n_rows, hh)
            The forecasts so far (columns 0 .. h-2 are filled).
        h : int
            The horizon (>= 2).

        Returns
        -------
        np.ndarray, shape (n_rows, n_columns)
        """
        p, q = self._hparam["p"], self._hparam["px"]
        n_fEndo = len(self._tForms["fEndo"])
        idx_endo = self._hparam["spec"]                                     # trend columns before the y block
        idx_exo = idx_endo + (1 + n_fEndo) * p
        x_trend = X_window[:, :idx_endo]

        x_act = X_window[:, idx_endo + (h - 1): idx_endo + p]            # the actual lags still known at horizon h
        idx_fcast = max(x_act.shape[1] - (p - h + 1), 0)
        x_fcast = y_fcast[:, idx_fcast: h - 1]                           # the forecasts standing in for the rest
        blocks = [x_trend, x_act, x_fcast]
        for i, tr in enumerate(self._tForms["fEndo"]):                   # transformed y blocks
            lo = idx_endo + (i + 1) * p
            blocks += [X_window[:, lo + (h - 1): lo + p], tr.f(x_fcast) if x_fcast.shape[1] > 0 else x_fcast]
        X_h = np.column_stack(blocks)

        if self._n_exo > 0 and q > 0:
            n_blocks = self._n_exo + sum(self._fExo_sz) + (self._n_exo if self._hparam.get("crx", 0) == 1 else 0)
            x_exo = np.column_stack([self.hide(X_window[:, idx_exo + k * q: idx_exo + (k + 1) * q], h)
                                     for k in range(n_blocks)])
            X_h = np.column_stack((X_h, x_exo))
        return X_h

    def __repr__(self) -> str:
        s = f"{self.modelName}(p={self._hparam['p']}, px={self._hparam['px']}, n_exo={self._n_exo}, fEndo={self._tForms['fEndo']}"
        return s + f"), hh = {self.hh}, skip = {self._hparam['skip']}, method={self._method}"
