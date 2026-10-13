"""
AR Model for Time Series Forecasting

Implements an Auto-Regressive model AR(p) estimated by the Yule-Walker equations (statsmodels
`yule_walker`, "adjusted" or "mle" auto-covariances, as ScalaTion's `AR` with its Durbin-Levinson
correlogram) or by ordinary least squares, forecasting RECURSIVELY from the last p values:

    y_t = delta + phi_1 y_t-1 + ... + phi_p y_t-p,    delta = mean(y) (1 - sum phi)

The Python counterpart of ScalaTion's `AR.scala` (`AR.apply` / `AR.rescale`).  `ar_yw.py`
keeps the former name `AR_YW` as an alias.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import numpy as np
import statsmodels.api as sm
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import build_lagged_matrix, fit_scaler
from modeling.forecasting.forecaster import Forecaster


class AR(Forecaster):
    """
    Auto-Regressive model AR(p), recursive multi-horizon forecasting.

    Attributes
    ----------
    method : {"adjusted", "mle", "sm_ols"}, default="adjusted"
        The estimator: Yule-Walker with adjusted (unbiased) or MLE auto-covariances, or OLS on
        the lagged matrix.
    params : np.ndarray, shape (p + 1,)
        delta (the drift/intercept) followed by phi_1 .. phi_p.
    parameter : np.ndarray, shape (p,)
        The AR coefficients phi (ScalaTion's `parameter` is delta +: b; the intercept is
        reported separately here, as for the regression models).
    modelName : str
        "AR_{p}".
    fname : list of str
        yl{p} .. yl1.
    """

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int, method: str = "adjusted") -> None:
        """
        Initializes the AR forecaster.

        Parameters
        ----------
        args : dict
            The hyper-parameters: "p" (the number of lags), "skip" (defaults to p).
        y : np.ndarray, shape (m,)
            The time series.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        method : {"adjusted", "mle", "sm_ols"}, default="adjusted"
            The estimator.
        """
        a = dict(args)
        a["q"] = 0
        a.setdefault("spec", 0)
        super().__init__(a, y, hh)
        self._method = method
        self._p = self.args["p"]
        self._modelName = f"AR_{self._p}"
        self._set_X(build_lagged_matrix(self._y, self._p))

    @classmethod
    def rescale(cls, args: Dict[str, Any], y: np.ndarray, hh: int, method: str = "adjusted",
                tForm: Optional[Type[TransformerMixin]] = None) -> "AR":
        """
        Alternative constructor that rescales y (scaler fitted on the training rows; MinMax to
        (0, 4) by default, StandardScaler turns the non-negativity constraint off).
        """
        mod = cls(args, y, hh, method)
        scaler = MinMaxScaler(feature_range=mod._lu) if tForm is None else tForm()
        if isinstance(scaler, StandardScaler):
            mod._nneg = False
        fit_scaler(scaler, mod._y, mod._tr_size)
        mod._yForm = scaler
        mod._y = scaler.transform(mod._y.reshape(-1, 1)).flatten()
        mod._set_X(build_lagged_matrix(mod._y, mod._p))
        return mod

    @property
    def fname(self) -> List[str]:
        """The lag names yl{p} .. yl1."""
        return [f"yl{k}" for k in range(self._p, 0, -1)]

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Estimates delta and phi on y_ (the full y by default) by Yule-Walker or OLS.
        """
        y_ = self._y if y_ is None else y_
        p = self._p
        if self._method == "sm_ols":
            X_ = build_lagged_matrix(y_, p) if X_ is None else X_
            fit = sm.OLS(y_[p:], sm.add_constant(X_[p:], has_constant="add")).fit()
            self._set_params(np.asarray(fit.params))
        else:
            phi = np.asarray(sm.regression.yule_walker(y_, order=p, method=self._method, result_object=False)[0])
            phi = phi[::-1]                                            # phi_p .. phi_1 to match the column order yl{p} .. yl1
            delta = y_.mean() * (1.0 - np.sum(phi))
            self._set_params(np.append(delta, phi))

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end) recursively (the lags of
        the earlier horizons replaced by their forecasts, ScalaTion's `forge`) and records them
        in `Yf` (original units).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
        """
        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        p, delta, b = self._p, self.params[0], self.params[1:]
        X_window = self._X[t_st:t_en]                                  # the actual lags yl{p} .. yl1
        y_fcast = np.zeros((t_en - t_st, self.hh))
        for h in range(1, self.hh + 1):
            x_act = X_window[:, h - 1:]                                # the actual lags still known at horizon h
            x_fcast = y_fcast[:, max(0, h - 1 - p):h - 1]              # the forecasts of the earlier horizons
            x = np.column_stack((x_act, x_fcast))[:, -p:]
            y_fcast[:, h - 1] = self._rectify(delta + x @ b)
        self._store(t_st, t_en, y_fcast)
        return y_fcast
