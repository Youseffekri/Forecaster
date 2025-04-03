"""
AR_YW Forecaster Module

This module defines the AR_YW class, which implements an autoregressive (AR) 
forecasting model using Yule-Walker equations for parameter estimation.

Supports multiple estimation methods including statsmodels OLS, Yule-Walker MLE, and adjusted Yule-Walker.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from typing import Any, Dict, Optional
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

from forecasting.Forecaster import Forecaster


class AR_YW(Forecaster):
    """
    Autoregressive model (AR) using Yule-Walker equations for parameter estimation.

    Supports the following autoregressive estimation methods:
    - "sm_ols": Ordinary Least Squares via statsmodels' AutoReg
    - "mle": Maximum Likelihood Estimation via Yule-Walker
    - "adjusted": Bias-adjusted Yule-Walker method for bias correction

    Attributes
    ----------
    model : AutoReg
        Autoregressive model instance from statsmodels.
    method : str
        Estimation method used to compute parameters ('sm_ols', 'mle', or 'adjusted').
    dynamic : bool, default = True
        Whether to use dynamic forecasting.

    See Also
    --------
    Forecaster : For base forecasting logic and shared attributes like y, X, Yf, tr_size, etc.
    """

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int, method: str):
        """
        Initializes the AR_YW Forecaster.

        Parameters
        ----------
        args : dict
            Dictionary of model configuration parameters.
        y : np.ndarray, shape (n_samples,)
            The input endogenous time series data.   
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        method : {"sm_ols", "mle", "adjusted"}
            Method used to estimate AR parameters.    
        """
        super().__init__(args, y, hh)
        self._model = AutoReg(y, lags=self.args['p'])
        self._method = method
        self._dynamic = True

        if self.args['skip'] < self.args['p']:
            print("Warning: 'skip' < 'p' is not allowed. Automatically setting 'skip' to 'p'.")
            self.args['skip'] = self.args['p']


    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None):
        """
        Trains the AR model using the specified Yule-Walker method.

        Sets the estimated model parameters using either statsmodels OLS 
        or the Yule-Walker estimation method.

        Parameters
        ----------
        y_ : Optional[np.ndarray], shape (n_samples,)
            The response vector (endogenous time series) used for training.
        X_ : Optional[np.ndarray], shape (n_samples, n_lagged_columns)
            Input matrix including lagged endogenous and exogenous variables.
        """
        if y_ is None:
            y_ = self._y

        if self._method == "sm_ols":
            model_fit = AutoReg(y_, lags=self.args['p']).fit()
            self._set_params(model_fit.params)
        else:
            phi, _ = sm.regression.yule_walker(y_, order=self.args["p"], method=self._method)
            intercept = y_.mean() * (1 - np.sum(phi))
            self._set_params(np.append(intercept, phi))

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
        np.ndarray, shape (n_forecast_steps, hh)
            Forecast matrix where each row corresponds to a time step and each column 
            corresponds to a forecast horizon.

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = np.zeros((t_en - t_st, self.hh))

        for t in range(t_st, t_en):
            y_fcast[t - t_st] = self._model.predict(params=self.params,
                                                    start=t,
                                                    end=t + self.hh - 1,
                                                    dynamic=self._dynamic)

        self._Yf[t_st: t_en, 1:-1] = y_fcast    
        return y_fcast
