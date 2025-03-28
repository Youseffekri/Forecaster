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

from Util.build_matrix import build_matrix_4ts
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
    __rescale  = False
    __useforge = False
    __bound    = (0, 1)

    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray,
                 hh: int, 
                 z: Optional[np.ndarray] = None, 
                 method: str ="sk_lr", 
                 xx: Optional[np.ndarray] = None):
        """
        Initializes the ARX Forecaster.

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
            Target time series (endogenous values).

        z : Optional[np.ndarray], shape (n_samples, n_exo), default=None
            Exogenous variables used for forecasting. If None, only endogenous values are used.

            
        method : {"sm_ols", "sk_lr"}
            Estimation method:
            - "sm_ols" : Statsmodel using Ordinary Least Squares.
            - "sk_lr" : Sklearn Linear Regression

        Attributes
        ----------
        model : LinearRegression
            LinearRegression model instance.
        """
        super().__init__(args, y, hh)
        self.X = xx
        self.method = method
        self.model = LinearRegression()
        self.n_exo = z.shape[1] if z is not None else 0
        if not self.__rescale and self.X is None:
            self.__useforge = True
            self.set_X(build_matrix_4ts(y    = y, 
                                        spec = self.args["spec"], 
                                        p    = self.args["p"], 
                                        z    = z, 
                                        q    = self.args["q"]))
            
        print(f"\nARX(p={self.args['p']}, n_exo={self.n_exo}, q={self.args['q']}), spec={self.args['spec']}, skip={self.args['skip']}, method={self.method}")

    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                z: Optional[np.ndarray] = None, 
                method: str ="sk_lr", 
                tForm = None, 
                xx: Optional[np.ndarray] = None):
        """
        Alternative constructor that enables rescaling.

        Parameters
        ----------
        args : dict
            Model configuration parameters.

        y : np.ndarray
            Target time series.

        z : Optional[np.ndarray]
            Exogenous variables.

        method : {"sm_ols", "sk_lr"}
            Estimation method:
            - "sm_ols" : Statsmodel using Ordinary Least Squares.
            - "sk_lr" : Sklearn Linear Regression

        Returns
        -------
        ARX_Symb
            An instance of ARX_Symb with scaling enabled.
        """

        cls.__rescale = True
        s_ARX = cls(args, y, hh, z, method, xx)

        if tForm is not None:
            tForms = [tForm(), tForm()]
            if isinstance(tForm, StandardScaler):
                s_ARX.nneg = False
        else:
            tForms = [MinMaxScaler(feature_range=s_ARX.__bound), MinMaxScaler(feature_range=s_ARX.__bound)]     
        s_ARX.tForm_y, s_ARX.tForm_exo  = tForms

        if s_ARX.X is None:
            s_ARX.__useforge = True
            s_ARX.set_X(build_matrix_4ts(y     = y, 
                                        spec   = s_ARX.args["spec"], 
                                        p      = s_ARX.args["p"], 
                                        z      = z, 
                                        q      = s_ARX.args["q"],
                                        tForms = tForms))
        else:
            s_ARX.X = s_ARX.tForm_exo.fit_transform(s_ARX.X)

        s_ARX.y = s_ARX.tForm_y.transform(s_ARX.y.reshape(-1, 1)).flatten()
        return s_ARX


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
        if X_ is None:
            X_ = self.X
        if y_ is None:
            y_ = self.y

        if self.method == "sm_ols":
            self.model =  sm.OLS(y_, sm.add_constant(X_))
            self.model_fit = self.model.fit()
            self.set_params(self.model_fit.params)
        else:
            self.model.fit(X_, y_)
            self.set_params(np.append(self.model.intercept_, self.model.coef_))

            

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
        yf : np.ndarray, shape (t_en - t_st, hh)
            Matrix containing forecast values for all hh horizons.
        """

        t_st = t_start if t_start > -1 else self.args["skip"]
        t_en = t_end if t_end > -1 else len(self.y)

        X_window = self.X[t_st: t_en].copy()

        yf = np.zeros((t_en - t_st, self.hh))

        if self.method == "sm_ols":
            yp = self.model_fit.predict(sm.add_constant(X_window))
        else:
            yp = self.model.predict(X_window)
        
        yf[:, 0] = self.rectify(yp)

        for h in range(2, self.hh+1):
            xy = self.forge(X_window, yf, h)
            if self.method == "sm_ols":
                yfh = self.model_fit.predict(sm.add_constant(xy))
            else:
                yfh = self.model.predict(xy)
            
            yf[:, h-1] = self.rectify(yfh)
        return yf

    def forge(self, X_window: np.ndarray, yf: np.ndarray, h: int) -> np.ndarray:
        """
        Constructs feature vectors using past actual values, forecasts, and exogenous variables.

        Parameters
        ----------
        X_window : np.ndarray, shape(n_window, n)
            Lagged actual values and forecasted values.
        h : int
            Forecasting horizon.

        Returns
        -------
        xy : np.ndarray
            Feature vector for forecasting.

        Raises
        ------
        ValueError
            If use_forge is False.
        """

        if not self.__useforge:
            raise ValueError("Cannot use forge when the input matrix X was not build by buildMatrix.")      

        idx_endo   = self.args['spec'] - 1
        idx_exo    = idx_endo + self.args['p']
        x_trend    = X_window[:, :idx_endo]
        x_endo_act = X_window[:, idx_endo + (h-1): idx_endo + self.args['p']]
        idx_fcast  = max(x_endo_act.shape[1] - (self.args['p'] - h +1), 0)
        x_endo_fcast = yf[:, idx_fcast:h-1]

        xy = np.column_stack((x_trend, x_endo_act, x_endo_fcast))
        if self.n_exo > 0:
            x_exo = np.column_stack([self.hide(X_window[:, idx_exo + j * self.args['q']: idx_exo + (j + 1) * self.args['q']], h) for j in range(self.n_exo)])
            xy    = np.column_stack((xy, x_exo))
        return xy


    def hide(self, z: np.ndarray, h: int, fill: bool = True) -> np.ndarray:
        """
        Hides future values in a vector.

        Parameters
        ----------
        z : np.ndarray, shape (n_window, self.args['q'])
            Input vector.
        h : int
            Forecasting horizon.
        fill : bool, default=True
            If True, fills hidden values with the last known value; otherwise, uses zero.

        Returns
        -------
        z_shift : np.ndarray
            Hide values at the end of vector z (last h-1 values) as the increasing horizon
            turns them in future values (hence unavailable).  Set these values to either 
            zero (the default) or the last available value.
        """

        z_last = z[:, -1]
        if h > z.shape[1]:
            z_shift = np.column_stack([z_last for _ in range(z.shape[1])]) if fill else np.zeros(z.shape)
        else:
            z_shift = np.column_stack([z_last for _ in range(h-1)])
            z_      = np.column_stack([z[:, i] for i in range(h-1, z.shape[1])])
            z_shift = np.column_stack((z_, z_shift))
        return z_shift
