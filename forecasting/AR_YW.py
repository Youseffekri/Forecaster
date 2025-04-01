"""
AR_YW Forecaster Module

This module defines the AR_YW class, which implements a Yule-Walker-based 
autoregressive forecasting model.

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
    Implements an Autoregressive (AR) model using Yule-Walker equations 
    for parameter estimation.

    Attributes
    ----------
    model : AutoReg
        Autoregressive model instance from statsmodels.
    method : str
        Estimation method used to compute parameters ('sm_ols', 'mle', or 'adjusted').
    dynamic : bool
        Whether to use dynamic forecasting (default True).
    y : np.ndarray
        Response vector (endogenous time series).
    params : Optional[np.ndarray]
        Model parameters after training.
    tr_size : int
        Size of the training dataset.
    te_size : int
        Size of the testing dataset.
    Yf : np.ndarray
        Forecast output matrix of shape (n_samples, hh + 2).

    Notes
    -----
    The actual implementation uses leading underscores for internal attributes 
    (e.g., _model, _method, _dynamic, _y, _params).
    """

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int, method: str):
        """
        Initializes the AR_YW Forecaster.

        Parameters
        ----------
        args : dict
            Model configuration parameters, including:
            - 'skip' (int, default=0): Number of initial observations to skip.
            - 'p' (int): Number of lags for the endogenous variable (lags 1 to p).
        y : np.ndarray
            Target time series (endogenous values), shape (n_samples,).
        hh : int
            Forecast horizon (number of future steps).
        method : {"sm_ols", "mle", "adjusted"}
            Estimation method:
            - "sm_ols" : Statsmodels Ordinary Least Squares.
            - "mle" : Maximum Likelihood Estimation.
            - "adjusted" : Adjusted Yule-Walker method.
        """
        super().__init__(args, y, hh)
        self._method = method
        self._model = AutoReg(y, lags=self.args['p'])
        self._dynamic = True

        if self.args['skip'] < self.args['p']:
            print("Warning: 'skip' cannot be less than 'p'. Setting 'skip' to 'p'.")
            self.args['skip'] = self.args['p']

        print(f"\nAR_YW(p={self.args['p']}), skip={self.args['skip']}, method={self._method}")

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None):
        """
        Trains the AR model using the specified Yule-Walker method.

        Parameters
        ----------
        y_ : Optional[np.ndarray], default=None
            Target time series. If None, uses the internal stored series.
        X_ : Optional[np.ndarray], default=None
            Not used in this model (included for interface compatibility).

        Notes
        -----
        Sets the estimated model parameters using either statsmodels OLS 
        or the Yule-Walker estimation method.
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
        Generates a multi-step forecast matrix for all time points from t_start to t_end.

        Parameters
        ----------
        t_start : int, default=-1
            Start index for forecasting. If -1, uses the configured 'skip' value.
        t_end : int, default=-1
            End index for forecasting (exclusive). If -1, uses the length of the time series.

        Returns
        -------
        np.ndarray
            Forecast matrix of shape (t_end - t_start, hh), 
            containing predictions for each horizon h = 1 to hh.
        """
        t_st = t_start if t_start > -1 else self.args["skip"]
        t_en = t_end if t_end > -1 else len(self._y)
        yf = np.zeros((t_en - t_st, self.hh))

        for t in range(t_st, t_en):
            yf[t - t_st] = self._model.predict(
                params=self.params,
                start=t,
                end=t + self.hh - 1,
                dynamic=self._dynamic
            )
        return yf
