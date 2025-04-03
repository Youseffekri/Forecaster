"""
ARX Model for Time Series Forecasting

Implements an ARX (Autoregressive model with Exogenous variables) using linear regression.
Supports multi-step forecasting with lagged endogenous and optional exogenous variables.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from typing import Any, Dict, Optional, Type

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import *
from forecasting.Forecaster import Forecaster

class ARX(Forecaster):
    """
    Autoregressive model with Exogenous inputs (ARX) for time series forecasting.

    Extends the `Forecaster` base class and uses linear regression to perform 
    multi-step forecasting with lagged endogenous and optional exogenous variables.


    Attributes
    ----------
    model : LinearRegression
        Linear regression model used for training and forecasting.
    method : {"sm_ols", "sk_lr"}, default="sm_ols"
        Whether to use statsmodels or sklearn linear regression.
    xe : Optional[np.ndarray], shape (n_samples, n_exo), default=None
        Exogenous variables aligned with y for multivariate forecasting.
    n_exo : int
        Number of exogenous input variables used in the model.
    use_forge : bool
        Whether the input matrix `X` was constructed internally and supports dynamic forecasting.
    tForms : dict
        Dictionary of transformations to apply.

    See Also
    --------
    Forecaster : For base forecasting logic and shared attributes like y, X, Yf, tr_size, etc.
    """
    

    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray,
                 hh: int, 
                 xe: Optional[np.ndarray] = None, 
                 method: str ="sm_ols", 
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX Forecaster.

        Parameters
        ----------
        args : dict
            Dictionary of model configuration parameters.
        y : np.ndarray, shape (n_samples,)
            The input endogenous time series data.   
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        xe : Optional[np.ndarray], shape (n_samples, n_exo), default=None
            Exogenous variables aligned with y for multivariate forecasting.
        method : {"sm_ols", "sk_lr"}, default="sm_ols"
            Whether to use statsmodels OLS or sklearn Linear Regression.
        X : Optional[np.ndarray], shape (n_samples, n_lagged_columns), default=None
            Input feature matrix. If None, the model will build it from `y`, `xe`, and other configuration settings.
        """
        super().__init__(args, y, hh)
        self._X = X
        self._model = LinearRegression()
        self._method = method
        self._xe = xe
        self._n_exo = xe.shape[1] if xe is not None else 0
        self._useforge = True if self._X is None else False
        if not hasattr(self, "_tForms"):  self._tForms = {"tForm_y": None}
        if self._X is None:
            self._set_X(self._build_matrix(args   = self.args,
                                           y      = self._y,
                                           xe      = self._xe,
                                           tForms = self._tForms))
        self._modelName = f"{self.__class__.__name__}(p={self.args["p"]}, q={self.args["q"]}, n_exo={self._n_exo}), method={method}"


    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                xe: Optional[np.ndarray] = None, 
                method: str ="sm_ols", 
                tForm: Optional[Type[TransformerMixin]] = None, 
                X: Optional[np.ndarray] = None) -> "ARX":
        """
        Alternative constructor that enables rescaling.

        Parameters
        ----------
        args : dict
            Dictionary of model configuration parameters.
        y : np.ndarray, shape (n_samples,)
            The input endogenous time series data.   
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        xe : Optional[np.ndarray], shape (n_samples, n_exo), default=None
            Exogenous variables aligned with y for multivariate forecasting.
        method : {"sm_ols", "sk_lr"}, default="sm_ols"
            Whether to use statsmodels OLS or sklearn Linear Regression.
        tForm : Optional[Type[TransformerMixin]]
            Transformation used for scaling `y` and forecast outputs,
            e.g., MinMaxScaler or StandardScaler.
        X : Optional[np.ndarray], shape (n_samples, n_lagged_columns), default=None
            Input feature matrix. If None, it will be built from `y` and `xe`.

        Returns
        -------
        ARX
            An instance of ARX with scaling enabled.
        """

        ARX_re = cls(args, y, hh, xe, method, np.array([0]))

        ARX_re._X = X
        if tForm is not None:
            ARX_re._tForms["tForm_y"],  ARX_re._tForms["tForm_exo"] = tForm(), tForm()
            if isinstance(ARX_re._tForms["tForm_y"], StandardScaler): ARX_re._nneg = False
        else:
            ARX_re._tForms["tForm_y"]    = MinMaxScaler(feature_range=ARX_re._lu)
            ARX_re._tForms["tForm_exo"]  = MinMaxScaler(feature_range=ARX_re._lu)

        if ARX_re._X is None:
            ARX_re._useforge = True
            ARX_re._set_X(ARX_re._build_matrix(args   = ARX_re.args,
                                               y      = ARX_re._y,
                                               xe      = ARX_re._xe,
                                               tForms = ARX_re._tForms))
        else:
            ARX_re._X = ARX_re._tForms["tForm_exo"].fit_transform(ARX_re._X)

        ARX_re._yForm = ARX_re._tForms["tForm_y"]
        ARX_re._y = ARX_re._tForms["tForm_y"].transform(ARX_re._y.reshape(-1, 1)).flatten()
        return ARX_re

    def _build_matrix(self, args: Dict[str, Any], y: np.ndarray, xe: Optional[np.ndarray] = None, tForms : Optional[Type[TransformerMixin]] = {"tForm_y": None}) -> np.ndarray:
        """
        Builds the design matrix using lagged endogenous and optional exogenous inputs.

        Parameters
        ----------
        args : dict
            Model configuration parameters including lag orders and trend spec.
        y : np.ndarray, shape (n_samples,)
            Endogenous time series.
        xe : Optional[np.ndarray], shape (n_samples, n_exo)
            Exogenous input matrix aligned with `y`.
        tForms : dict
            Dictionary of transformations to apply.

        Returns
        -------
        np.ndarray
            Feature matrix combining trend, endogenous, and exogenous lagged variables.
        """
        p = args["p"]
        q = args["q"]
        spec = args["spec"]

        if tForms["tForm_y"] is not None:
            y = tForms["tForm_y"].fit_transform(y.reshape(-1, 1)).flatten()
        X = build_lagged_matrix(y, p)
        if xe is not None:
            xe = backfill_matrix(xe)
            if tForms["tForm_y"] is not None:
                xe = tForms["tForm_exo"].fit_transform(xe)
            x_exo = np.column_stack([build_lagged_matrix(xe[:, j], q) for j in range(xe.shape[1])])
            X = np.column_stack((X, x_exo))

        if spec > 1:
            xt = build_trend_matrix(len(y), spec)
            X = np.column_stack((xt, X))
        return X

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the ARX model.

        Parameters
        ----------
        y_ : Optional[np.ndarray], shape (n_samples,)
            The response vector (endogenous time series) used for training.
        X_ : Optional[np.ndarray], shape (n_samples, n_lagged_columns)
            Input matrix including lagged endogenous and exogenous variables.
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
        Generates a multi-step forecast matrix. This method updates the forecast values in `Yf`.

        Parameters
        ----------
        t_start : int
            Start index for forecasting. If -1, defaults to self.args["skip"].
        t_end : int
            End index for forecasting (exclusive). If -1, defaults to the length of the target series.

        Returns
        -------
        y_fcast : np.ndarray, shape (n_forecast_steps, hh)
            Matrix of forecasts for each time step and each horizon.

        Raises
        ------
        ValueError
            If the model has not been trained.
        """

        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = np.zeros((t_en - t_st, self.hh))
        X_window = self._X[t_st: t_en].copy()
        
        if self._method == "sm_ols":
            y_fcast_1 = self._model_fit.predict(sm.add_constant(X_window))
        else:
            y_fcast_1 = self._model.predict(X_window)
        y_fcast[:, 0] = self._rectify(y_fcast_1)

        for h in range(2, self.hh + 1):
            X_h = self._forge(X_window, y_fcast, h)
            if self._method == "sm_ols":
                y_fcast_h = self._model_fit.predict(sm.add_constant(X_h))
            else:
                y_fcast_h = self._model.predict(X_h)
            y_fcast[:, h - 1] = self._rectify(y_fcast_h)

        self._Yf[t_st: t_en, 1:-1] = y_fcast if self._yForm is None else self._yForm.inverse_transform(y_fcast)    
        return y_fcast

    def _forge(self, X_window: np.ndarray, y_fcast: np.ndarray, h: int) -> np.ndarray:
        """
        Constructs feature vectors for forecasting horizon `h` by combining actual and forecast values.

        Parameters
        ----------
        X_window : np.ndarray
            Input window of features for time steps being forecasted.
        y_fcast : np.ndarray, shape (n_forecast_steps, hh)
            Matrix of forecasts for each step and horizon.
        h : int
            Current forecasting step (horizon `h`).

        Returns
        -------
        X_h: np.ndarray
            Feature matrix with actual lags and prior forecasts for horizon `h`.

        Raises
        ------
        ValueError
            If the model was initialized with a custom feature matrix (i.e., use_forge is False).
        """
        if not self._useforge:
            raise ValueError("Cannot use forge when the input matrix X was not built by buildMatrix.")      

        idx_endo = self.args["spec"] - 1
        idx_exo = idx_endo + self.args["p"]
        x_trend = X_window[:, :idx_endo]
        x_endo_act = X_window[:, idx_endo + (h - 1): idx_endo + self.args["p"]]
        idx_fcast = max(x_endo_act.shape[1] - (self.args["p"] - h + 1), 0)
        x_endo_fcast = y_fcast[:, idx_fcast:h - 1]
        X_h = np.column_stack((x_trend, x_endo_act, x_endo_fcast))

        if self._n_exo > 0:
            x_exo = np.column_stack([self._hide(X_window[:, idx_exo + j * self.args["q"]: idx_exo + (j + 1) * self.args["q"]], h)
                                     for j in range(self._n_exo)])
            X_h = np.column_stack((X_h, x_exo))
        return X_h

    def _hide(self, z: np.ndarray, h: int, fill: bool = False) -> np.ndarray:
        """
        Hides future values in a lagged exogenous matrix.

        Parameters
        ----------
        z : np.ndarray, shape (n_samples, q)
            Lagged exogenous matrix for one variable.
        h : int
            Forecasting horizon.
        fill : bool, default=True
            If True, fills future values with last known value; otherwise with zero.

        Returns
        -------
        z_shift: np.ndarray
            Lag matrix with future values hidden based on the forecasting horizon.
        """
        z_last = z[:, -1]
        if h > z.shape[1]:
            z_shift = np.column_stack([z_last for _ in range(z.shape[1])]) if fill else np.zeros(z.shape)
        else:
            z_shift = np.column_stack([z_last for _ in range(h - 1)])
            z_ = np.column_stack([z[:, i] for i in range(h - 1, z.shape[1])])
            z_shift = np.column_stack((z_, z_shift))
        return z_shift
