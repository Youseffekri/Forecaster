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
            
        y : np.ndarray, shape (n_samples,)
            Target time series (endogenous values).

        method : {"sm_ols", "mle", "adjusted"}
            Estimation method:
            - "sm_ols" : Statsmodel Ordinary Least Squares.
            - "mle" : Maximum Likelihood Estimation.
            - "adjusted" : Adjusted Yule-Walker method.


        Attributes
        ----------
        model : AutoReg
            Autoregressive model instance.
        """
        super().__init__(args, y, hh)
        self.method = method
        self.model = AutoReg(y, lags=self.args['p'])
        self.dynamic = True

        if self.args['skip'] < self.args['p']:
            print("Warning: 'skip' cannot be less than 'p'. Setting 'skip' to 'p'.")
            self.args['skip'] = self.args['p']
        
        print(f"\nAR_YW(p={self.args['p']}), skip={self.args['skip']}, method={self.method}")

            


    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None):
        """
        Trains the forecasting model.

        Parameters
        ----------
        y_ : Optional[np.ndarray]
            Target values.
        X_ : Optional[np.ndarray]
            Input feature matrix.
        

        Notes
        -----
        Subclasses must implement this method to train the model and store parameters in self.params,
        which can be accessed using get_params().
        """
        if y_ is None:
            y_ = self.y

        if self.method == "sm_ols":
            model_fit = AutoReg(y_, lags=self.args['p']).fit()
            self.set_params(model_fit.params)
        else:
            phi, _ = sm.regression.yule_walker(y_, order=self.args["p"], method=self.method)
            intercept = y_.mean() * (1 - np.sum(phi))
            self.set_params(np.append(intercept, phi))


    def forecast(self, t_start: int=-1, t_end: int=-1) -> np.ndarray:
        """
        Generates a multi-step forecast matrix and computes the quality of forecast (QoF) metrics.

        Parameters
        ----------
        t_start : int
            Start index for forecasting.
        t_end : int
            End index for forecasting (exclusive).

        Returns
        -------
        yf : np.ndarray, shape (t_end - t_start, hh)
            Matrix containing forecast values for all hh horizons.
        """
        t_st = t_start if t_start > -1 else self.args["skip"]
        t_en = t_end if t_end > -1 else len(self.y)
        yf = np.zeros((t_en - t_st, self.hh))

        for t in range(t_st, t_en):
            yf[t - t_st] = self.model.predict(params=self.get_params(), start=t, end=t + self.hh - 1, dynamic=self.dynamic)
        return yf

