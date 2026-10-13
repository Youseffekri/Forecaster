"""
ARX Model for Time Series Forecasting

Implements an ARX (Auto-Regressive model with eXogenous variables) by linear regression on the
lags of y (1 to p) and of each exogenous variable (1 to q), with optional trend columns (`spec`)
forecasting RECURSIVELY: the h-steps ahead forecast feeds the forecasts
of the earlier steps back into the lag columns (`_forge`), while the exogenous lags that would
lie in the future are hidden (`_hide`).  The Python counterpart of ScalaTion's `ARX.scala`
(`ARX.apply` / `ARX.rescale`, `formNames`).

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import (backfill_matrix, build_lagged_matrix, build_matrix_Y, build_trend_matrix,
                               fit_scaler, tr_size_of, trend_names)
from modeling.forecasting.forecaster import Forecaster
from modeling.regression import CenteredRidge


class ARX(Forecaster):
    """
    Auto-Regressive model with eXogenous variables (ARX), recursive multi-horizon forecasting.

    Attributes
    ----------
    method : {"sm_ols", "sk_lr", "ridge"}, default="sm_ols"
        The regression: statsmodels OLS, scikit-learn LinearRegression, or the centered ridge
        of ScalaTion's `CenteredRidge` (penalty args["lambda"], default 1.0; args["scaled"]).
    xe : np.ndarray, shape (m, n_exo), optional
        The exogenous variables aligned with `y`.
    n_exo : int
        The number of exogenous variables.
    useforge : bool
        Whether X was built by `_build_matrix` (needed to forge the lags for h > 1).
    tForms : dict
        "tForm_y" / "tForm_exo": the scalers of y and of the exogenous variables (None unless
        made by `rescale`), "tr_size": the rows the scalers are fitted on.
    modelName : str
        "ARX_{p}_{q}_{n_exo}".
    fname : list of str
        The column names: the trend names, yl{p} .. yl1, xe{j}l{q} .. xe{j}l1 (`form_names`).

    See Also
    --------
    Forecaster : the base forecasting logic (Yf, rollValidate, inSample_Test, tnT_Test).
    """

    def __init__(self,
                 args: Dict[str, Any],
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 method: str = "sm_ols",
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX forecaster.

        Parameters
        ----------
        args : dict
            The hyper-parameters: "p" (lags of y), "q" (lags of each exogenous variable),
            "spec" (number of trend columns, 0..3), "skip", "lambda" / "scaled" (for method "ridge").
        y : np.ndarray, shape (m,)
            The endogenous time series.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        xe : np.ndarray, shape (m, n_exo), optional
            The exogenous variables aligned with y.
        method : {"sm_ols", "sk_lr", "ridge"}, default="sm_ols"
            The regression method.
        X : np.ndarray, shape (m, n), optional
            A ready-made input matrix (built from y and xe when None).
        """
        super().__init__(args, y, hh)
        self._X = X
        self._method = method
        self._xe = xe
        self._n_exo = xe.shape[1] if xe is not None else 0
        self._useforge = self._X is None
        self._model = self._make_model()
        self._modelName = f"ARX_{self.args['p']}_{self.args['q']}_{self._n_exo}"

        if not hasattr(self, "_tForms"):
            self._tForms = {"tForm_y": None, "tForm_exo": None}

        if self._X is None:
            self._set_X(self._build_matrix(args=self.args, y=self._y, xe=self._xe, tForms=self._tForms))

    def _make_model(self):
        """The regression object of `method` (None for statsmodels, fitted per call)."""
        if self._method == "sk_lr":
            return LinearRegression()
        if self._method == "ridge":
            return CenteredRidge(lam=float(self.args.get("lambda", 1.0)), scaled=bool(self.args.get("scaled", False)))
        if self._method == "sm_ols":
            return None
        raise ValueError(f"method '{self._method}' unknown: sm_ols | sk_lr | ridge")

    @classmethod
    def rescale(cls,
                args: Dict[str, Any],
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                method: str = "sm_ols",
                tForm: Optional[Type[TransformerMixin]] = None,
                X: Optional[np.ndarray] = None) -> "ARX":
        """
        Alternative constructor that rescales y and the exogenous variables (ScalaTion's
        `ARX.rescale`): the scalers are fitted on the TRAINING rows only and the forecasts are
        inverted before being stored in `Yf`.

        Parameters
        ----------
        args, y, hh, xe, method, X
            As in the constructor.
        tForm : type of TransformerMixin, optional
            The scaler class: MinMaxScaler to (0, 4) by default; StandardScaler turns the
            non-negativity constraint off (ScalaTion's NormForm).

        Returns
        -------
        ARX
            The model on the scaled series.
        """
        mod = cls(args, y, hh, xe, method, np.array([0]))
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

    @classmethod
    def fromTerms(cls, formula, dataset: Dict[str, np.ndarray], hh: int, tFormT=None, trSize: int = -1,
                  scaled: bool = False, method: str = "sm_ols", args: Optional[Dict[str, Any]] = None):
        """
        Creates the model from the terms of a formula (ScalaTion's `ARX.fromTerms`), with one
        column per term (lagged variable, transform, interaction or wave), so skipped lags and
        interactions that contiguous lags cannot express are supported.  Rows are not trimmed
        (each term pads its first lag values).  With `tFormT`, every base column is transformed
        (fitted on its training slice, as in `rescale`) before the terms are evaluated, and the
        response's transform is kept for inverting forecasts.  The matrix has NO intercept
        column.  Recursive forecasting beyond h = 1 needs the standard lag layout (the plain
        lags of y then of each exogenous variable, in decreasing order); for any other formula
        use the direct `ARX_D.fromTerms` / `NARX_SR_D.fromTerms`.

        Parameters
        ----------
        formula : ARX_Formula
            The formula (response and predictor terms).
        dataset : dict of str -> np.ndarray
            The named columns.
        hh : int
            The maximum forecasting horizon.
        tFormT : TransformT, optional
            The transform type applied to every base column (None => none).
        trSize : int, default=-1
            The number of leading rows the transforms are fit on (-1 => `trSize`).
        scaled : bool, default=False
            Whether the regression standardizes the term columns (method "ridge").
        method : str, default="sm_ols"
            The regression method.
        args : dict, optional
            Hyper-parameter overrides (p, q and spec come from the formula; "skip" defaults to its largest lag).
        """
        from modeling.forecasting.arx_formula import ARX_Formula
        f = formula if tFormT is None else ARX_Formula(formula.resp, formula.preds, tFormT)
        td = f.prepare(dataset, trSize)                               # checks and base-column transforms
        X = f.matrix(td.data)                                         # one column per term
        hp = f.lagHP()
        a_ = {"p": hp["p"], "q": hp["q"], "spec": 0, "skip": f.maxLag, "scaled": scaled}
        if args:
            a_.update(args)
        mod = cls(a_, td.y, hh, None, method, X)
        mod._n_exo = hp["n_exo"]
        mod._modelName = f"{cls.__name__}_{hp['p']}_{hp['q']}_{hp['n_exo']}"
        mod._fname = f.labels
        mod._yForm = td.tForm_y
        if td.tForm_y is not None:                                    # column 0 of Yf stays in ORIGINAL units
            mod._Yf[:, 0] = np.asarray(dataset[f.resp.name], dtype=float)
        mod._tr_size = td.trSize
        mod._te_size = len(td.y) - td.trSize
        mod._useforge = ARX._standard_layout(f, hp)
        if hasattr(mod, "_Y"):
            mod._Y = build_matrix_Y(mod._y, hh)
        return mod

    @staticmethod
    def _standard_layout(f, hp: Dict[str, int]) -> bool:
        """Whether the terms are the plain lags yl{p}..yl1 then xe lags q..1 per variable (forgeable)."""
        from modeling.forecasting.arx_formula import FormulaVar
        yn = f.resp.name
        exo = sorted(f.vars - {yn})
        want = [f"{yn}~{k}" for k in range(hp["p"], 0, -1)]
        for v in exo:
            want += [f"{v}~{k}" for k in range(hp["q"], 0, -1)]
        return all(isinstance(t, FormulaVar) for t in f.preds) and f.labels == want

    @staticmethod
    def _build_matrix(args: Dict[str, Any],
                      y: np.ndarray,
                      xe: Optional[np.ndarray] = None,
                      tForms: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Builds the input matrix (ScalaTion's `ARX.buildMatrix`): the trend columns, the lags of
        y (yl{p} .. yl1) and the lags of each exogenous variable (backfilled; xe{j}l{q} .. xe{j}l1).
        ENDO-EXO cross terms belong to `ARX_SR` (ScalaTion's `crx`), not to `ARX`.

        Parameters
        ----------
        args : dict
            "p", "q", "spec".
        y : np.ndarray, shape (m,)
            The endogenous series.
        xe : np.ndarray, shape (m, n_exo), optional
            The exogenous variables.
        tForms : dict, optional
            "tForm_y" / "tForm_exo" scalers (fitted on the first "tr_size" rows) or None.

        Returns
        -------
        np.ndarray, shape (m, n)
        """
        tForms = {"tForm_y": None, "tForm_exo": None} if tForms is None else tForms
        p, q, spec = args["p"], args["q"], args["spec"]
        tr_size = tForms.get("tr_size", tr_size_of(len(y)))

        if tForms["tForm_y"] is not None:
            y = fit_scaler(tForms["tForm_y"], y, tr_size).transform(y.reshape(-1, 1)).flatten()

        x_endo = build_lagged_matrix(y, p)
        blocks = [x_endo]
        if xe is not None and xe.shape[1] > 0:
            xe = backfill_matrix(xe)
            if tForms["tForm_exo"] is not None:
                xe = fit_scaler(tForms["tForm_exo"], xe, tr_size).transform(xe)
            blocks.extend(build_lagged_matrix(xe[:, j], q) for j in range(xe.shape[1]))

        X = np.column_stack(blocks)
        if spec > 0:
            X = np.column_stack((build_trend_matrix(len(y), spec), X))
        return X

    @staticmethod
    def form_names(args: Dict[str, Any], n_exo: int) -> List[str]:
        """
        The column names of the input matrix (ScalaTion's `ARX.formNames`): the trend names,
        yl{p} .. yl1, then xe{j}l{q} .. xe{j}l1 per exogenous variable.
        """
        p, q, spec = args["p"], args["q"], args["spec"]
        names = trend_names(spec) + [f"yl{k}" for k in range(p, 0, -1)]
        for j in range(n_exo):
            names += [f"xe{j}l{k}" for k in range(q, 0, -1)]
        return names

    @property
    def fname(self) -> List[str]:
        """The column names of X (generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return self.form_names(self.args, self._n_exo)
        return [f"x{j}" for j in range(self._X.shape[1])]

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the regression on (X_, y_) (the full data by default); the parameters are
        stored with the intercept first.
        """
        X_ = self._X if X_ is None else X_
        y_ = self._y if y_ is None else y_

        if self._method == "sm_ols":
            self._model_fit = sm.OLS(y_, sm.add_constant(X_, has_constant="add")).fit()
            self._set_params(np.asarray(self._model_fit.params))
        elif self._method == "ridge":
            self._model.fit(X_, y_)
            self._set_params(self._model.params[:, 0])
        else:
            self._model.fit(X_, y_)
            self._set_params(np.append(self._model.intercept_, self._model.coef_))

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end) recursively and records
        them in `Yf` (original units).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
            The forecasts (model units).

        Raises
        ------
        ValueError
            If the model has not been trained, or if hh > 1 and X was not built by `_build_matrix`.
        """
        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = np.zeros((t_en - t_st, self.hh))
        X_window = self._X[t_st:t_en].copy()
        y_fcast[:, 0] = self._predict(X_window)

        if self.hh > 1 and not self._useforge:
            raise ValueError("Cannot use forge when the input matrix X was not built by _build_matrix.")

        for h in range(2, self.hh + 1):
            y_fcast[:, h - 1] = self._predict(self._forge(X_window, y_fcast, h))

        self._store(t_st, t_en, y_fcast)
        return y_fcast

    def _predict(self, X_horizon: np.ndarray) -> np.ndarray:
        """The one-step predictions of the rows of X_horizon (clipped when nneg)."""
        if self._method == "sm_ols":
            yp = self._model_fit.predict(sm.add_constant(X_horizon, has_constant="add"))
        elif self._method == "ridge":
            yp = self._model.predict(X_horizon)[:, 0]
        else:
            yp = self._model.predict(X_horizon)
        return self._rectify(yp)

    def _forge(self, X_window: np.ndarray, y_fcast: np.ndarray, h: int) -> np.ndarray:
        """
        The input rows for horizon h (ScalaTion's `forge`): the trend columns, the lags of y
        with the h - 1 most recent ones replaced by the forecasts of the earlier horizons, and
        the exogenous lags with the future ones hidden.

        Parameters
        ----------
        X_window : np.ndarray, shape (rows, n)
            The input rows being forecast.
        y_fcast : np.ndarray, shape (rows, hh)
            The forecasts so far (columns 0 .. h - 2 are used).
        h : int
            The horizon.
        """
        p, q, spec = self.args["p"], self.args["q"], self.args["spec"]
        idx_endo = spec
        idx_exo = idx_endo + p
        x_trend = X_window[:, :idx_endo]
        x_endo_act = X_window[:, idx_endo + (h - 1): idx_endo + p]            # the actual lags still usable
        idx_fcast = max(x_endo_act.shape[1] - (p - h + 1), 0)
        x_endo_fcast = y_fcast[:, idx_fcast:h - 1]                              # the forecasts of the earlier steps
        X_h = np.column_stack((x_trend, x_endo_act, x_endo_fcast))

        if self._n_exo > 0:
            x_exo = [self._hide(X_window[:, idx_exo + j * q: idx_exo + (j + 1) * q], h) for j in range(self._n_exo)]
            X_h = np.column_stack([X_h] + x_exo)
        return X_h

    def _hide(self, z: np.ndarray, h: int, fill: bool = True) -> np.ndarray:
        """
        Hides the exogenous lags that lie in the future at horizon h (ScalaTion's `hide`): the
        lag columns are shifted so that the last known value (lag 1 of the forecast origin)
        fills the h - 1 most recent slots (or zero when not `fill`).

        Parameters
        ----------
        z : np.ndarray, shape (rows, q)
            The lag matrix of one exogenous variable (lags q .. 1).
        h : int
            The horizon.
        fill : bool, default=True
            Whether to fill with the last known value (else zero).
        """
        z_last = z[:, -1]
        if h > z.shape[1]:
            return np.column_stack([z_last for _ in range(z.shape[1])]) if fill else np.zeros(z.shape)
        z_shift = np.column_stack([z_last for _ in range(h - 1)]) if h > 1 else np.zeros((z.shape[0], 0))
        z_ = z[:, h - 1:]
        return np.column_stack((z_, z_shift))
