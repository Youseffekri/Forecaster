"""
Forecaster Module

Defines an abstract Forecaster class for time series forecasting. 
Subclasses must implement the train and forecast methods.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from Util.tools import diagnose
from Util.build_matrix import buildYf




class Forecaster(ABC):
    """
    Attributes
    ----------
    args : dict
        Model configuration parameters.
    y : np.ndarray, shape (n_samples,)
        Target (endogenous) values.
    X : Optional[np.ndarray], default=None
        Input feature matrix.
    params : Optional[np.ndarray], default=None
        Model parameters after training.
    use_forge : bool, default=False
        Internal flag.
    skip : int, default=0
        Number of initial training samples to skip.
    """
    

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int):
        """
        Initializes the Forecaster.

        Parameters
        ----------
        args : dict
            Model configuration parameters, including:
            - 'skip' (int, default=0): Number of initial observations to skip.
            - 'p' (int): Number of lags for the endogenous variable (lags 1 to p).
            - 'q' (int): Number of lags for each exogenous variable (lags 1 to q).
            - 'spec' (int): Number of trend terms to include:
                - 1 : Constant.
                - 2 : Linear.
                - 3 : Quadratic.
                - 4 : Sine.
                - 5 : Cosine.

        y : np.ndarray, shape (n_samples,)
            Target (endogenous) values.
        """
        self.args = { 
            'spec': args.get('spec', 1),
            'p': args.get('p', 6),
            'q': args.get('q', 0),
            **args
        }
        self.args['skip'] = args.get('skip', self.args['p'])
        self.y = y
        self.hh = hh
        self.yf = buildYf(y, hh)
        self.tr_size = int(0.8 * len(y))
        self.te_size = len(y) - self.tr_size
        self.X: Optional[np.ndarray] = None
        self.params: Optional[np.ndarray] = None
        self.use_forge = False
        self.nneg = True

    def set_X(self, X: np.ndarray) -> None:
        """
        Sets the input feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Raises
        ------
        TypeError
            If X is not a NumPy array.
        ValueError
            If X is empty.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.size == 0:
            raise ValueError("X cannot be empty.")
        self.X = X
    
    def set_params(self, params: np.ndarray) -> None:
        """
        Sets the trained model parameters.

        Parameters
        ----------
        params : np.ndarray
            The model parameters to store.

        Raises
        ------
        TypeError
            If params is not a NumPy array.
        ValueError
            If params is empty.
        """
        if not isinstance(params, np.ndarray):
            raise TypeError("params must be a NumPy array.")
        if params.size == 0:
            raise ValueError("params cannot be empty.")
        self.params = params

    def get_params(self) -> np.ndarray:
        """
        Returns the trained model parameters.

        Returns
        -------
        params : np.ndarray
            Trained model parameters.

        Raises
        ------
        ValueError
            If the model is not trained.
        """
        if self.params is None:
            raise ValueError("Model parameters are not set. Call train first.")
        return self.params

    
    @abstractmethod
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
        pass

    @abstractmethod
    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Generates a multi-step forecast matrix and computes the quality of forecast (QoF) metrics.

        Parameters
        ----------
        t_start : int
            Start index for forecasting.
        t_end : int
            End index for forecasting (exclusive).

         Notes
        -----

        """

        pass
    
    def rectify(self, yp:np.ndarray):
        if self.nneg:
            return np.maximum(yp, 0)
        else:
            return yp
        

    def diagnose_all(self, yf : np.ndarray, TnT : bool = False) -> str:
        """
        Generates a multi-step forecast matrix and computes the quality of forecast (QoF) metrics.

        Parameters
        ----------
        yf : np.ndarray
            Matrix containing forecast values for all hh horizons.

        Returns
        -------
        qof : str
            Quality of Forecast (QoF) metrics formatted as a string, including:
            
            - MSE (Mean Squared Error)
            - MAE (Mean Absolute Error)
            - R^2 (Coefficient of Determination)
            - Adjusted R^2 (R^2Bar)
            - SMAPE (Symmetric Mean Absolute Percentage Error)
            - m (Number of samples)

            Each metric is computed for different forecast horizons (h = 1 to hh).
        """
        y_true = self.y[self.tr_size:].copy() if TnT else self.y[self.args['skip']:].copy()
        hh = yf.shape[1]
        ll = len(y_true)

        horizon_all =f"horizon\t->"
        mse_all   = f"mse\t->"
        mae_all   = f"mae\t->"
        r2_all    = f"R^2\t->"
        r2Bar_all = f"R^2Bar\t->"
        smape_all = f"smape\t->"
        m_all     = f"m\t->"

        for h in range(hh):
            metrics = diagnose(y_true[h:], yf[:ll-h, h])
            horizon_all += f"\t  {h+1}\t"
            mse_all     += f"\t{metrics['MSE']:.4f}\t"
            mae_all     += f"\t{metrics['MAE']:.4f}\t"
            r2_all      += f"\t{metrics['R2']:.4f}\t"
            r2Bar_all   += f"\t{metrics['R2Bar']:.4f}\t"
            smape_all   += f"\t{metrics['SMAPE']:.4f}\t"
            m_all       += f"\t{metrics['m']}\t"

        qof = horizon_all + "\n" + mse_all + "\n" + mae_all + "\n" + r2_all + "\n" + r2Bar_all + "\n" + smape_all + "\n" + m_all + "\n"
        return qof
    
    def rollValidate(self, rc: int = 2, growing: bool = False):
        
        yf = np.zeros((self.te_size, self.hh))

        for i in range(0, self.te_size, rc):  
            is_ = 0 if growing else i
            t = self.tr_size + i  
            X_ = self.X[is_:t] if self.X is not None else None
            y_ = self.y[is_:t]
            self.train(y_, X_)  
            yf[i:i+rc, :] = self.forecast(t, t+rc)
        return yf

