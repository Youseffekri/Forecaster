"""
Forecaster_D Module

Defines an abstract `Forecaster_D` class for DIRECT (non-recursive) time series forecasting.
Subclasses must implement the `train` and `forecast` methods.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from forecasting.Forecaster import Forecaster


class Forecaster_D(Forecaster):
    """
    Abstract base class for DIRECT time series forecasting models.

    Attributes
    ----------
    Y : np.ndarray, shape (n_samples, hh)
        The full target matrix Y for direct multi-output regression. 
        (each column corresponds to a future step). 
    """

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int):
        """
        Initializes the Forecaster_D.

        Parameters
        ----------
        args : dict
            Dictionary of model configuration parameters, including:
            "spec" (int): Type of trend component to include in the model:
                1=Constant, 2=Linear, 3=Quadratic, 4=Sine, 5=Cosine.
            - "p" (int): Number of lags for the endogenous variable (lags 1 to p).
            - "q" (int): Number of lags for each exogenous variable (lags 1 to q).
            - "cross" (bool): Whether to add ENDO-EXO cross terms.
            - "skip" (int, default=0): Number of initial observations to skip.
        y : np.ndarray, shape (n_samples,)
            The input endogenous time series data.   
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        """
        super().__init__(args, y, hh)
        self._Y = None


    @property
    def Y(self):
        """
        Returns
        -------
        np.ndarray, shape (n_samples, hh)
            The multi-step response matrix.
        """
        return self._Y


    @abstractmethod
    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None):
        """
        Trains the forecasting model.

        Parameters
        ----------
        Y_ : Optional[np.ndarray]
            The response matrix for multi-step forecasting.
        X_ : Optional[np.ndarray]
            The data/input matrix (lagged columns of y and z).

        Notes
        -----
        Subclasses must implement this method to train the model and store parameters using self._set_params().
        """
        pass


    def diagnose_all(self, y_fcast: np.ndarray, TnT: bool = False) -> str:
        """
        Evaluate forecast accuracy for all prediction horizons.

        Computes quality-of-fit (QoF) metrics (MSE, MAE, R², etc.) across all forecast horizons.
        Optionally evaluates only on the test set when `TnT` is True.

        Parameters
        ----------
        y_fcast : np.ndarray, shape (n_samples, hh)
            Forecast values for each time step and each forecast horizon.
        TnT : bool, default=False
            If True, metrics are computed only on the test set (after training size).
            If False, metrics are computed from `args["skip"]` to the end.

        Returns
        -------
        str
            A formatted multi-line string showing the QoF metrics across all forecast horizons.
        """
        Y_true = self._Y[self._tr_size:].copy() if TnT else self._Y[self.args['skip']:].copy()
        ll, hh = Y_true.shape[0], y_fcast.shape[1]

        y_true_arr  = [Y_true[:ll - h, h] for h in range(hh)] if self._yForm is None else \
                      [self._yForm.inverse_transform(Y_true[:ll - h, h].reshape(-1, 1)).flatten() for h in range(hh)]
        y_fcast_arr = [y_fcast[:ll - h, h] for h in range(hh)] if self._yForm is None else \
                      [self._yForm.inverse_transform(y_fcast[:ll - h, h].reshape(-1, 1)).flatten() for h in range(hh)]

        return self._quality_of_fit(y_true_arr, y_fcast_arr)


    def rollValidate(self, rc: int = 2, growing: bool = False):
        """
        Performs rolling validation on the time series.

        Parameters
        ----------
        rc : int, default=2
            The retraining cycle (number of forecasts until retraining occurs).
        growing : bool, default=False
            Whether the training window grows with each step or remains fixed.

        Returns
        -------
        np.ndarray, shape (te_size, hh)
            Matrix including forecast values across rolling steps.
        """
        self._reset_Yf()
        yf = np.zeros((self._te_size, self.hh))
        
        for i in range(0, self._te_size, rc):
            is_ = 0 if growing else i
            t = self._tr_size + i
            X_ = self._X[is_:t] if self._X is not None else None
            Y_ = self._Y[is_:t]
            self.train(Y_, X_)

            if i+rc < self._te_size:
                yf[i:i + rc, :] = self.forecast(t, t + rc)
            else:
                yf[i:, :] = self.forecast(t)

        return yf