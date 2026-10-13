"""
ARX_D Model for Time Series Forecasting

Implements ARX_D (Auto-Regressive model with eXogenous variables, DIRECT multi-horizon
forecasting): the same input matrix as `ARX` (trend columns, lags of y, lags of each exogenous
variable) with one regression per horizon on the target matrix Y, so no forecasts are fed back
into the lags.  The Python counterpart of ScalaTion's `ARX_D.scala` (`ARX_D.apply` /
`ARX_D.rescale`).

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import build_matrix_Y, fit_scaler
from modeling.forecasting.arx import ARX
from modeling.forecasting.forecaster_d import Forecaster_D
from modeling.regression import CenteredRidge


class ARX_D(Forecaster_D):
    """
    Auto-Regressive model with eXogenous variables, DIRECT multi-horizon forecasting (ARX_D).

    Attributes
    ----------
    method : {"sm_ols", "sk_lr", "ridge"}, default="sm_ols"
        The regression per horizon: statsmodels OLS, scikit-learn LinearRegression, or the
        centered ridge of ScalaTion's `CenteredRidgeMV` (penalty args["lambda"], default 1.0;
        args["scaled"]).
    xe : np.ndarray, shape (m, n_exo), optional
        The exogenous variables aligned with `y`.
    n_exo : int
        The number of exogenous variables.
    tForms : dict
        "tForm_y" / "tForm_exo": the scalers (None unless made by `rescale`), "tr_size".
    modelName : str
        "ARX_D_{p}_{q}_{n_exo}".
    fname : list of str
        The column names (`ARX.form_names`).

    See Also
    --------
    Forecaster_D : the direct forecasting logic (Y, fit_rows, rollValidate, tests).
    ARX : the recursive counterpart and the shared `_build_matrix` / `form_names`.
    """

    def __init__(self,
                 args: Dict[str, Any],
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 method: str = "sm_ols",
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_D forecaster (parameters as in `ARX`).
        """
        super().__init__(args, y, hh)
        self._X = X
        self._method = method
        self._xe = xe
        self._n_exo = xe.shape[1] if xe is not None else 0
        self._useforge = self._X is None
        self._model = self._make_model()
        self._modelName = f"ARX_D_{self.args['p']}_{self.args['q']}_{self._n_exo}"

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
                X: Optional[np.ndarray] = None) -> "ARX_D":
        """
        Alternative constructor that rescales y and the exogenous variables (ScalaTion's
        `ARX_D.rescale`): the scalers are fitted on the TRAINING rows only; `StandardScaler`
        turns the non-negativity constraint off.

        Returns
        -------
        ARX_D
            The model on the scaled series (Y rebuilt from the scaled y).
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
        mod._Y = build_matrix_Y(mod._y, hh)
        return mod

    @classmethod
    def fromTerms(cls, formula, dataset: Dict[str, np.ndarray], hh: int, tFormT=None, trSize: int = -1,
                  scaled: bool = False, method: str = "sm_ols", args: Optional[Dict[str, Any]] = None) -> "ARX_D":
        """
        Creates the direct model from the terms of a formula (ScalaTion's `ARX_D.fromTerms`): one
        column per term, the response's transform kept for inverting forecasts, no intercept
        column; every formula is supported since no forging is needed (see `ARX.fromTerms`).
        """
        mod = ARX.fromTerms.__func__(cls, formula, dataset, hh, tFormT, trSize, scaled, method, args)
        mod._useforge = True                                          # no forging in a direct model
        return mod

    @staticmethod
    def _build_matrix(args: Dict[str, Any],
                      y: np.ndarray,
                      xe: Optional[np.ndarray] = None,
                      tForms: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """The input matrix of `ARX` (same columns, same order)."""
        return ARX._build_matrix(args, y, xe, tForms)

    @property
    def fname(self) -> List[str]:
        """The column names of X (generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return ARX.form_names(self.args, self._n_exo)
        return [f"x{j}" for j in range(self._X.shape[1])]

    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains one regression per horizon on (X_, Y_) (the full data by default; the padded
        trailing rows are dropped by `fit_rows`); the parameters (n + 1, hh) are stored with the
        intercept first.
        """
        X_ = self._X if X_ is None else X_
        Y_ = self._Y if Y_ is None else Y_
        Y_, X_ = self.fit_rows(Y_, X_)

        if self._method == "sm_ols":
            Xc = sm.add_constant(X_, has_constant="add")
            self._models_fit = [sm.OLS(Y_[:, j], Xc).fit() for j in range(Y_.shape[1])]
            self._set_params(np.column_stack([np.asarray(f.params) for f in self._models_fit]))
        elif self._method == "ridge":
            self._model.fit(X_, Y_)
            self._set_params(self._model.params)
        else:
            self._model.fit(X_, Y_)
            self._set_params(np.column_stack([self._model.intercept_, self._model.coef_]).T)

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end) directly and records them
        in `Yf` (original units).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
            The forecasts (model units).

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        X_window = self._X[t_st:t_en]

        if self._method == "sm_ols":
            Xc = sm.add_constant(X_window, has_constant="add")
            y_fcast = np.column_stack([f.predict(Xc) for f in self._models_fit])
        else:
            y_fcast = self._model.predict(X_window)

        y_fcast = self._rectify(y_fcast)
        self._store(t_st, t_en, y_fcast)
        return y_fcast

    # -- feature selection: the sub-models (ScalaTion's `ARX_D.getModel` / `getModel_h`) -------

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "ARX_D":
        """
        Builds an `ARX_D` on the columns `cols` of X (ScalaTion's `ARX_D.getModel`): the same y, Y,
        hh, scaling, hyper-parameters and regression method, the feature names of the columns,
        e.g., the selected model after feature selection, `getModel(getBest().mod_cols)`;
        `getModel()` is a copy on all the columns.  The single-horizon `ARX` at h is
        `getModel_h(cols, h)` (`Forecaster_D`).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        """
        cols = self.mcols if cols is None else list(cols)
        sub = ARX_D(self.args, self.getY(), self.hh, None, self._method, self._X[:, cols])
        return self._subModel(sub, cols)
