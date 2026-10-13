"""
ARX_SR Model: Auto-Regressive on Lagged y and xe with Symbolic Regression (SR) Terms

Extends the `ARX` model with trend, linear, power, root and cross terms: the lagged endogenous
variable y and its transformations (`fEndo`, e.g., `PowForm()` for y^1.5), the lagged exogenous
variables and their own transformations (`fExo`, one list per exogenous variable), and optional
ENDO-EXO cross terms (`args["cross"]`).  Multi-step forecasts are RECURSIVE: the forecast of
horizon h feeds the lags of horizon h + 1, and a transformed lag of a forecast is the transform of
the forecast.

With `rescale`, y and the exogenous variables are scaled FIRST (scalers fitted on the training
rows) and the transformations are applied to the scaled series, so no scale correction of the
transformed forecasts is needed (the `scaleCorrection` of the earlier version is gone).

The Python counterpart of ScalaTion's `ARX_SR` (`ARX_SR.apply` / `ARX_SR.rescale` /
`ARX_SR.buildMatrix` / `forge`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import (backfill_matrix, build_lagged_matrix, build_trend_matrix, fit_scaler,
                               tr_size_of, trend_names)
from Util.transform import PowForm, Transform
from modeling.forecasting.arx import ARX


class ARX_SR(ARX):
    """
    Auto-Regressive model with exogenous variables and Symbolic Regression terms (ARX_SR).

    Attributes
    ----------
    tForms : dict
        The transformations in play:
        - "tForm_y"   : the scaler of y (None unless `rescale`), also used to invert forecasts
        - "tForm_exo" : the scaler of the exogenous variables (None unless `rescale`)
        - "fEndo"     : list of `Transform` applied to y before lagging (default `[PowForm()]`)
        - "fExo"      : list of lists of `Transform`, one list per exogenous variable
        - "tr_size"   : the training rows the scalers are fitted on
    fExo_sz : list of int
        The number of transformations per exogenous variable.
    fname : list of str
        The column labels (`form_names`): trend, yl{k}, f{i}(yl{k}), xe{j}l{k}, g{j},{i}(xe{j}l{k}), xe{j}l{k}*yl{k}.

    Notes
    -----
    Column order of X (the order `forge` relies on): trend terms (spec > 1), the p lags of y,
    the p lags of each fEndo transform of y, then for each exogenous variable j its q raw lags
    followed by the q lags of each of its transforms, then (if `cross`) the q lags of y * xe_j
    for every j.

    See Also
    --------
    ARX : the recursive forecasting logic (train, forecast, _predict, _hide).
    """

    def __init__(self,
                 args: Dict[str, Any],
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 fEndo: Optional[Sequence[Transform]] = None,
                 fExo: Optional[Sequence[Sequence[Transform]]] = None,
                 method: str = "sm_ols",
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_SR forecaster.

        Parameters
        ----------
        args : dict
            Model configuration: "p" (lags of y), "q" (lags of each exogenous variable), "spec"
            (number of trend columns, ScalaTion's convention: 0 implicit constant, 1 linear, 2
            quadratic, 3 cubic), "cross" (ENDO-EXO cross terms, ScalaTion's crx), "skip".
        y : np.ndarray, shape (n_samples,)
            The endogenous time series.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        xe : Optional[np.ndarray], shape (n_samples, n_exo)
            The exogenous variables aligned with y.
        fEndo : sequence of Transform, optional
            The transformations of y (default `[PowForm()]`, i.e., y^1.5; `[]` for none).
        fExo : sequence of sequences of Transform, optional
            One list of transformations per exogenous variable (default: none for every variable).
        method : {"sm_ols", "sk_lr", "ridge"}, default="sm_ols"
            statsmodels OLS, sklearn LinearRegression or the centered ridge (args "lambda", "scaled").
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
        self._tForms = {"tForm_y": None, "tForm_exo": None, "fEndo": fEndo, "fExo": fExo}
        self._fExo_sz = [len(s) for s in fExo]
        super().__init__(args, y, hh, xe, method, X)
        self._modelName = f"ARX_SR_{self.args['p']}_{self.args['q']}_{self._n_exo}"

    @classmethod
    def rescale(cls,
                args: Dict[str, Any],
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                fEndo: Optional[Sequence[Transform]] = None,
                fExo: Optional[Sequence[Sequence[Transform]]] = None,
                method: str = "sm_ols",
                tForm: Optional[Type[TransformerMixin]] = None,
                X: Optional[np.ndarray] = None) -> "ARX_SR":
        """
        Alternative constructor that rescales y and the exogenous variables BEFORE the
        transformations are applied (scalers fitted on the training rows).

        Parameters
        ----------
        args, y, hh, xe, fEndo, fExo, method, X
            As in the constructor.
        tForm : Optional[Type[TransformerMixin]]
            The scaler class: `MinMaxScaler` (default, feature range `_lu`) or `StandardScaler`
            (turns the non-negativity constraint off, as ScalaTion's NormForm does).

        Returns
        -------
        ARX_SR
        """
        mod = cls(args, y, hh, xe, fEndo, fExo, method, np.array([0]))
        mod._X = X
        mod._tForms["tr_size"] = mod._tr_size
        if tForm is not None:
            mod._tForms["tForm_y"], mod._tForms["tForm_exo"] = tForm(), tForm()
            if isinstance(mod._tForms["tForm_y"], StandardScaler):
                mod._nneg = False
        else:
            mod._tForms["tForm_y"] = MinMaxScaler(feature_range=mod._lu)
            mod._tForms["tForm_exo"] = MinMaxScaler(feature_range=mod._lu)

        if mod._X is None:
            mod._useforge = True
            mod._set_X(mod._build_matrix(args=mod.args, y=mod._y, xe=mod._xe, tForms=mod._tForms))
        else:
            mod._X = fit_scaler(mod._tForms["tForm_exo"], mod._X, mod._tr_size).transform(mod._X)
            fit_scaler(mod._tForms["tForm_y"], mod._y, mod._tr_size)

        mod._yForm = mod._tForms["tForm_y"]
        mod._y = mod._tForms["tForm_y"].transform(mod._y.reshape(-1, 1)).flatten()
        return mod

    @staticmethod
    def _build_matrix(args: Dict[str, Any],
                      y: np.ndarray,
                      xe: Optional[np.ndarray] = None,
                      tForms: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Builds the design matrix: trend terms, the lags of y and of its transformations, the lags
        of each exogenous variable and of its transformations, and the lagged cross terms.

        Parameters
        ----------
        args : dict
            "p", "q", "spec", "cross".
        y : np.ndarray, shape (n_samples,)
            The endogenous series (unscaled; scaled here when `tForms["tForm_y"]` is set).
        xe : Optional[np.ndarray], shape (n_samples, n_exo)
            The exogenous variables (backfilled and scaled here).
        tForms : dict
            "tForm_y", "tForm_exo" (scalers or None), "fEndo", "fExo", "tr_size".

        Returns
        -------
        np.ndarray, shape (n_samples, n_columns)
        """
        tForms = tForms if tForms is not None else {"tForm_y": None, "fEndo": [PowForm()], "fExo": []}
        p, q, spec, cross = args["p"], args["q"], args["spec"], args.get("cross", False)
        fEndo: List[Transform] = tForms.get("fEndo", [])
        fExo: List[List[Transform]] = tForms.get("fExo", [])
        tr_size = tForms.get("tr_size", tr_size_of(len(y)))

        if tForms.get("tForm_y") is not None:                            # scale first ...
            y = fit_scaler(tForms["tForm_y"], y, tr_size).transform(y.reshape(-1, 1)).flatten()

        with np.errstate(invalid="ignore", divide="ignore"):             # a bad column is reported below, once
            x_endo = np.column_stack([y] + [tr.f(y) for tr in fEndo])    # ... then transform the scaled series
        X = np.column_stack([build_lagged_matrix(x_endo[:, j], p) for j in range(x_endo.shape[1])])

        if xe is not None and q > 0:
            xe_bfill = backfill_matrix(xe)
            if tForms.get("tForm_y") is not None:
                xe_bfill = fit_scaler(tForms["tForm_exo"], xe_bfill, tr_size).transform(xe_bfill)
            cols = []
            for j in range(xe_bfill.shape[1]):
                xe_j = xe_bfill[:, j]
                cols.append(xe_j)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cols += [tr.f(xe_j) for tr in (fExo[j] if j < len(fExo) else [])]
            if cross:
                cols += [y * xe_bfill[:, j] for j in range(xe_bfill.shape[1])]
            x_exo = np.column_stack(cols)
            X = np.column_stack((X, np.column_stack([build_lagged_matrix(x_exo[:, j], q) for j in range(x_exo.shape[1])])))

        if spec > 0:
            X = np.column_stack((build_trend_matrix(len(y), spec), X))
        bad = ~np.isfinite(X).all(axis=0)
        if bad.any():
            j = int(np.argmax(bad))
            raise ValueError(f"ARX_SR: column {j} of the design matrix is not finite -- a power, root or log of a negative "
                             "or zero value; scale to a positive range (MinMaxScaler, the default of rescale) or use a "
                             "sign-safe transform (PowRForm, DQuadForm) with StandardScaler")
        return X

    @property
    def fname(self) -> List[str]:
        """The column labels of X, in ScalaTion's naming (`form_names`; generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return self.form_names(self.args, self._n_exo, len(self._tForms["fEndo"]), self._fExo_sz)
        return [f"x{j}" for j in range(self._X.shape[1])]

    @staticmethod
    def form_names(args: Dict[str, Any], n_exo: int, n_fEn: int, n_fExArr: Sequence[int]) -> List[str]:
        """
        The column labels in the order `_build_matrix` produces them (ScalaTion's `ARX_SR.formNames`):
        the trend names, yl{p}..yl1, f{i}(yl{k}), xe{j}l{k}, g{j},{i}(xe{j}l{k}), xe{j}l{k}*yl{k}.

        Parameters
        ----------
        args : dict
            "p", "q", "spec", "cross".
        n_exo : int
            The number of exogenous variables.
        n_fEn : int
            The number of transformations of y.
        n_fExArr : sequence of int
            The number of transformations per exogenous variable.
        """
        p, q, spec, cross = args["p"], args["q"], args["spec"], args.get("cross", False)
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

    def _forge(self, X_window: np.ndarray, y_fcast: np.ndarray, h: int) -> np.ndarray:
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
        p, q = self.args["p"], self.args["q"]
        n_fEndo = len(self._tForms["fEndo"])
        idx_endo = self.args["spec"]                                     # trend columns before the y block
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
            n_blocks = self._n_exo + sum(self._fExo_sz) + (self._n_exo if self.args.get("cross", False) else 0)
            x_exo = np.column_stack([self._hide(X_window[:, idx_exo + k * q: idx_exo + (k + 1) * q], h)
                                     for k in range(n_blocks)])
            X_h = np.column_stack((X_h, x_exo))
        return X_h

    def __repr__(self) -> str:
        s = f"{self.modelName}(p={self.args['p']}, q={self.args['q']}, n_exo={self._n_exo}, fEndo={self._tForms['fEndo']}"
        return s + f"), hh = {self.hh}, skip = {self.args['skip']}, method={self._method}"
