"""
ARX_D Model for Time Series Forecasting

Implements an ARX_D (Autoregressive model with Exogenous variables) using linear regression.
Uses DIRECT (as opposed to RECURSIVE) multi-horizon forecasting.

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
from forecasting.Forecaster_D import Forecaster_D
from forecasting.ARX import ARX


class ARX_D(Forecaster_D):
    """
    Autoregressive model with Exogenous inputs (ARX_D) for time series forecasting.

    Extends the `Forecaster_D` class and uses linear regression to perform 
    DIRECT multi-steps forecasting. with lagged endogenous and optional exogenous variables.

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
    Forecaster_D : For base forecasting logic and shared attributes like y, X, Yf, tr_size, etc.
    """
    
    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray,
                 hh: int, 
                 xe: Optional[np.ndarray] = None, 
                 method: str ="sk_lr", 
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

        if not hasattr(self, "_tForms"): 
            self._tForms = {"tForm_y": None}

        if self._X is None:
            self._set_X(self._build_matrix(args   = self.args,
                                           y      = self._y,
                                           xe      = self._xe,
                                           tForms = self._tForms))
            self._Y = build_matrix_Y(self._y, hh)


    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                xe: Optional[np.ndarray] = None, 
                method: str ="sk_lr", 
                tForm = None, 
                X: Optional[np.ndarray] = None) -> "ARX_D":
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
            Input feature matrix. If None, the model will build it from `y`, `xe`, and other configuration settings.

        Returns
        -------
        ARX_D
            An instance of ARX_D with scaling enabled.
        """
        ARX_D_re = cls(args, y, hh, xe, method, np.array([0]))
        ARX_D_re._X = X
        if tForm is not None:
            ARX_D_re._tForms["tForm_y"],  ARX_D_re._tForms["tForm_exo"] = tForm(), tForm()
            if isinstance(ARX_D_re._tForms["tForm_y"], StandardScaler): ARX_D_re._nneg = False
        else:
            ARX_D_re._tForms["tForm_y"]    = MinMaxScaler(feature_range=ARX_D_re._lu)
            ARX_D_re._tForms["tForm_exo"]  = MinMaxScaler(feature_range=ARX_D_re._lu)

        if ARX_D_re._X is None:
            ARX_D_re._useforge = True
            ARX_D_re._set_X(ARX_D_re._build_matrix(y      = ARX_D_re._y,
                                                   xe     = ARX_D_re._xe,
                                                   args   = ARX_D_re.args,
                                                   tForms = ARX_D_re._tForms))
        else:
            ARX_D_re._X = ARX_D_re._tForms["tForm_exo"].fit_transform(ARX_D_re._X)

        ARX_D_re._yForm = ARX_D_re._tForms["tForm_y"]
        ARX_D_re._y = ARX_D_re._tForms["tForm_y"].transform(ARX_D_re._y.reshape(-1, 1)).flatten()
        ARX_D_re._Y = build_matrix_Y(ARX_D_re._y, hh)
        return ARX_D_re
   

    @staticmethod
    def _build_matrix(args: Dict[str, Any], 
                      y: np.ndarray, 
                      xe: Optional[np.ndarray] = None, 
                      tForms : Optional[Dict[str, Any]] = {"tForm_y": None}) -> np.ndarray:
        """
        Builds the design matrix using lagged endogenous and optional exogenous inputs.

        Parameters
        ----------
        args : dict
            Model configuration parameters including lag orders and trend specification.
        y : np.ndarray, shape (n_samples,)
            Endogenous time series.
        xe : Optional[np.ndarray], shape (n_samples, n_exo)
            Exogenous input matrix aligned with `y`.
        tForms : dict, optional
            Dictionary of transformations to apply. Can include:
            - "tForm_y"    : transformation applied to the target series `y`
            - "tForm_exo"  : transformation applied to derived exogenous features
            
        Returns
        -------
        np.ndarray
            Feature matrix combining trend, endogenous, and exogenous lagged variables.
        """
        return ARX._build_matrix(args, y, xe, tForms)
    

    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None):
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

        if Y_ is None:
            Y_ = self._Y

        if self._method == "sm_ols":
            self._models_fit = []
            n = Y_.shape[1]

            for j in range(n):
                mod = sm.OLS(Y_[:, j], sm.add_constant(X_))
                self._models_fit.append(mod.fit())

            self._set_params(np.column_stack([self._models_fit[j].params for j in range(n)]))
        else:
            self._model.fit(X_, Y_)
            self._set_params(np.column_stack([self._model.intercept_, self._model.coef_]).T)

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
        y_fcast = np.zeros((t_en - t_st, self.hh))
        X_window = self._X[t_st: t_en].copy()
        
        if self._method == "sm_ols":
            X_w_cst = sm.add_constant(X_window)
            y_fcast = np.column_stack([mod.predict(X_w_cst) for mod in self._models_fit])
        else:
            y_fcast = self._model.predict(X_window)

        y_fcast = self._rectify(y_fcast)
        self._Yf[t_st: t_en, 1:-1] = y_fcast if self._yForm is None else self._yForm.inverse_transform(y_fcast)      
        return y_fcast