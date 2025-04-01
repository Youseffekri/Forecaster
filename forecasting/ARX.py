"""
ARX Model for Time Series Forecasting

Implements an ARX (Autoregressive with Exogenous Variables) model using linear regression. 
Supports forecasting with past observations and exogenous variables.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from typing import Any, Dict, Optional

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import *
from forecasting.Forecaster import Forecaster


class ARX(Forecaster):
    """
    Implements an Autoregressive (ARX) model with exogenous variables. 
    Uses past observations and external inputs for forecasting.

    Attributes
    ----------
    model : LinearRegression
        Trained linear regression model for forecasting.
    use_forge : bool
        Indicates whether the feature matrix X was automatically generated using buildMatrix4TS.
    n_exo : int
        Number of exogenous variables (equal to z.shape[1]).       
    """

    __BOUND = (0, 4)

    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray,
                 hh: int, 
                 z: Optional[np.ndarray] = None, 
                 method: str ="sk_lr", 
                 X: Optional[np.ndarray] = None):
        """
        Initializes the ARX Forecaster.

        Parameters
        ----------
        args : dict
            Model configuration parameters.
        y : np.ndarray
            Target time series (endogenous values).
        hh : int
            Forecast horizon (number of future steps).
        z : Optional[np.ndarray], default=None
            Exogenous variables used for forecasting.
        method : {"sm_ols", "sk_lr"}, default="sk_lr"
            Estimation method.
        X : Optional[np.ndarray], default=None
            Feature matrix (optional). If None, it will be constructed internally.
        """
        super().__init__(args, y, hh)
        self._z = z
        self._X = X
        self._useforge = True if self._X is None else False
        self._n_exo = z.shape[1] if z is not None else 0
        self._method = method
        self._model = LinearRegression()
        self._lu = ARX.__BOUND
        if not hasattr(self, '_tForms'):  self._tForms = {"tForm_y": None}
        
        if self._X is None:
            self._set_X(self._build_matrix(y      = self._y,
                                           z      = self._z,
                                           args   = self.args,
                                           tForms = self._tForms))

    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                z: Optional[np.ndarray] = None, 
                method: str ="sk_lr", 
                tForm = None, 
                X: Optional[np.ndarray] = None):
        """
        Alternative constructor that enables rescaling.

        Parameters
        ----------
        args : dict
            Model configuration parameters.
        y : np.ndarray
            Target time series.
        hh : int
            Forecast horizon.
        z : Optional[np.ndarray]
            Exogenous variables.
        method : {"sm_ols", "sk_lr"}
            Estimation method.
        tForm : Transformer, optional
            A transformation class such as MinMaxScaler or StandardScaler.
        X : Optional[np.ndarray]
            Feature matrix, if already available.

        Returns
        -------
        ARX
            An instance of ARX with scaling enabled.
        """

        ARX_re = cls(args, y, hh, z, method, np.array([0]))

        ARX_re._X = X
        if tForm is not None:
            ARX_re._tForms["tForm_y"],  ARX_re._tForms["tForm_exo"] = tForm(), tForm()
            if isinstance(ARX_re._tForms["tForm_y"], StandardScaler): ARX_re._nneg = False
        else:
            ARX_re._tForms["tForm_y"]    = MinMaxScaler(feature_range=ARX_re._lu)
            ARX_re._tForms["tForm_exo"]  = MinMaxScaler(feature_range=ARX_re._lu)

        if ARX_re._X is None:
            ARX_re._useforge = True
            ARX_re._set_X(ARX_re._build_matrix(y      = ARX_re._y,
                                               z      = ARX_re._z,
                                               args   = ARX_re.args,
                                               tForms = ARX_re._tForms))
        else:
            ARX_re._X = ARX_re._tForms["tForm_exo"].fit_transform(ARX_re._X)

        ARX_re._yForm = ARX_re._tForms["tForm_y"]
        ARX_re._y = ARX_re._tForms["tForm_y"].transform(ARX_re._y.reshape(-1, 1)).flatten()
        return ARX_re

    def _build_matrix(self, y, z, args, tForms):
        """
        Builds the feature matrix from endogenous and exogenous variables.

        Parameters
        ----------
        y : np.ndarray
            Target time series.
        z : Optional[np.ndarray]
            Exogenous variables.
        args : ??

        tForms : dict
            Dictionary of transformations to apply.

        Returns
        -------
        np.ndarray
            Feature matrix.
        """
        p = args["p"]
        q = args["q"]
        spec = args["spec"]

        if tForms["tForm_y"] is not None:
            y = tForms["tForm_y"].fit_transform(y.reshape(-1, 1)).flatten()
        X = build_lagged_matrix(y, p)
        if z is not None:
            z = backfill_matrix(z)
            if tForms["tForm_y"] is not None:
                z = tForms["tForm_exo"].fit_transform(z)
            x_exo = np.column_stack([build_lagged_matrix(z[:, j], q) for j in range(z.shape[1])])
            X = np.column_stack((X, x_exo))

        if spec > 1:
            xt = build_trend_matrix(len(y), spec)
            X = np.column_stack((xt, X))
        return X

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None):
        """
        Trains the ARX model.

        Parameters
        ----------
        y_ : Optional[np.ndarray]
            Target values.
        X_ : Optional[np.ndarray]
            Input feature matrix.
        """
        if X_ is None:
            X_ = self._X
        if y_ is None:
            y_ = self._y

        if self._method == "sm_ols":
            self._model = sm.OLS(y_, sm.add_constant(X_))
            self._model_fit = self._model.fit()
            self._set_params(self._model_fit.params)
        else:
            self._model.fit(X_, y_)
            self._set_params(np.append(self._model.intercept_, self._model.coef_))

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Generates a multi-step forecast matrix.

        Parameters
        ----------
        t_start : int
            Start index for forecasting.
        t_end : int
            End index for forecasting (exclusive).

        Returns
        -------
        np.ndarray
            Matrix containing forecast values for all hh horizons.

        Raises
        ------
        ValueError
            If the model has not been trained.
        """

        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        
        t_st, t_en = self._adjust_range(t_start, t_end)

        yf = np.zeros((t_en - t_st, self.hh))

        X_window = self._X[t_st: t_en].copy()
        

        if self._method == "sm_ols":
            yp = self._model_fit.predict(sm.add_constant(X_window))
        else:
            yp = self._model.predict(X_window)
        
        yf[:, 0] = self._rectify(yp)
        for h in range(2, self.hh + 1):
            xy = self._forge(X_window, yf, h)
            if self._method == "sm_ols":
                yfh = self._model_fit.predict(sm.add_constant(xy))
            else:
                yfh = self._model.predict(xy)
            yf[:, h - 1] = self._rectify(yfh)

        self._Yf[t_st: t_en, 1:-1] = yf    
        return yf

    def _forge(self, X_window: np.ndarray, yf: np.ndarray, h: int) -> np.ndarray:
        """
        Constructs feature vectors for horizon h using past actual and forecasted values.

        Parameters
        ----------
        X_window : np.ndarray
            Feature matrix window.
        yf : np.ndarray
            Previously forecasted values.
        h : int
            Forecasting horizon.

        Returns
        -------
        np.ndarray
            Feature matrix for current horizon.

        Raises
        ------
        ValueError
            If use_forge is False.
        """
        if not self._useforge:
            raise ValueError("Cannot use forge when the input matrix X was not built by buildMatrix.")      

        idx_endo = self.args['spec'] - 1
        idx_exo = idx_endo + self.args['p']
        x_trend = X_window[:, :idx_endo]
        x_endo_act = X_window[:, idx_endo + (h - 1): idx_endo + self.args['p']]
        idx_fcast = max(x_endo_act.shape[1] - (self.args['p'] - h + 1), 0)
        x_endo_fcast = yf[:, idx_fcast:h - 1]
        xy = np.column_stack((x_trend, x_endo_act, x_endo_fcast))

        if self._n_exo > 0:
            x_exo = np.column_stack([
                self._hide(X_window[:, idx_exo + j * self.args['q']: idx_exo + (j + 1) * self.args['q']], h)
                for j in range(self._n_exo)
            ])
            xy = np.column_stack((xy, x_exo))
        return xy

    def _hide(self, z: np.ndarray, h: int, fill: bool = False) -> np.ndarray:
        """
        Hides future values in a lagged exogenous matrix.

        Parameters
        ----------
        z : np.ndarray
            Lagged exogenous matrix.
        h : int
            Forecasting horizon.
        fill : bool, default=True
            If True, fills future values with last known value; otherwise with zero.

        Returns
        -------
        np.ndarray
            Modified matrix with future values hidden.
        """
        z_last = z[:, -1]
        if h > z.shape[1]:
            z_shift = np.column_stack([z_last for _ in range(z.shape[1])]) if fill else np.zeros(z.shape)
        else:
            z_shift = np.column_stack([z_last for _ in range(h - 1)])
            z_ = np.column_stack([z[:, i] for i in range(h - 1, z.shape[1])])
            z_shift = np.column_stack((z_, z_shift))
        return z_shift
