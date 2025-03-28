"""
ARX Model for Time Series Forecasting

Implements an ARX (Autoregressive with Exogenous Variables) model using linear regression. 
Supports forecasting with past observations and exogenous variables.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from Util.build_matrix import build_matrix_symbolic_4ts
from forecasting.Forecaster import Forecaster


class ARX_Symb(Forecaster):
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
    __bound    = (0, 4)

    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray, 
                 hh: int,
                 z: Optional[np.ndarray] = None, 
                 cross: bool = False, 
                 fEndo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None, 
                 fExo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,  
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
        
        ff : Array of nonlinear transformations to be applied on endogenous values
        gg : Array of nonlinear transformations to be applied on exogenous values

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
        self.X      = xx
        self.n_exo  = z.shape[1] if z is not None else 0
        self.cross  = cross
        self.method = method
        self.model  = LinearRegression()
        functions   = [lambda x: np.power(x, 1.5),
                       lambda x: np.power(x, 0.5),
                       np.log1p,
                       np.sin,
                       np.cos]
        self.tForms = {"tForm_y": None}
        self.tForms["fEndo"] = fEndo if fEndo is not None else functions
        self.tForms["fExo"]  = fExo if fExo is not None else functions
        
        if not self.__rescale and self.X is None:
            self.__useforge = True
            self.set_X(build_matrix_symbolic_4ts(y     = y, 
                                                 spec  = self.args["spec"], 
                                                 p     = self.args["p"], 
                                                 z     = z, 
                                                 q     = self.args["q"], 
                                                 cross = self.cross, 
                                                 tForms= self.tForms))
            
        print(f"\nARX_Symb(p={self.args['p']}, n_exo={self.n_exo}, q={self.args['q']}), spec={self.args['spec']}, skip={self.args['skip']}, method={self.method}")

    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                z: Optional[np.ndarray] = None,
                cross: bool = False, 
                fEndo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None, 
                fExo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,   
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

        ff : list of functions, optional
            Nonlinear transformations applied to endogenous values.

        gg : list of functions, optional
            Nonlinear transformations applied to exogenous values.

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
        ARX_Symb_re = cls(args, y, hh, z, cross, fEndo, fExo, method, xx)

        if tForm is not None:
            ARX_Symb_re.tForms["tForm_y"], ARX_Symb_re.tForms["tForm_endo"], ARX_Symb_re.tForms["tForm_exo"] = tForm(), tForm(), tForm()
            if isinstance(tForm, StandardScaler):
                ARX_Symb_re.nneg = False
        else:
            ARX_Symb_re.tForms["tForm_y"]    = MinMaxScaler(feature_range=ARX_Symb_re.__bound)
            ARX_Symb_re.tForms["tForm_endo"] = MinMaxScaler(feature_range=ARX_Symb_re.__bound)
            ARX_Symb_re.tForms["tForm_exo"]  = MinMaxScaler(feature_range=ARX_Symb_re.__bound)

        if ARX_Symb_re.X is None:
            ARX_Symb_re.__useforge = True
            ARX_Symb_re.set_X(build_matrix_symbolic_4ts(y      = y, 
                                                        spec   = ARX_Symb_re.args["spec"], 
                                                        p      = ARX_Symb_re.args["p"], 
                                                        z      = z, 
                                                        q      = ARX_Symb_re.args["q"], 
                                                        cross  = ARX_Symb_re.cross, 
                                                        tForms = ARX_Symb_re.tForms))
        else:
            ARX_Symb_re.X = ARX_Symb_re.tForms["tForm_exo"].fit_transform(ARX_Symb_re.X)

        ARX_Symb_re.y = ARX_Symb_re.tForms["tForm_y"].transform(ARX_Symb_re.y.reshape(-1, 1)).flatten()
        
        return ARX_Symb_re
    
    
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
        hh : int
            Forecasting horizon: Number of future time steps to predict.
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
            raise ValueError("Cannot use forge when the input matrix X was provided by user!")

        n_fEndo = len(self.tForms["fEndo"])
        n_fExo = len(self.tForms["fExo"])

        idx_endo = self.args['spec'] - 1
        idx_exo  = idx_endo + (1+n_fEndo)*self.args['p']
        x_trend = X_window[:, :idx_endo]

        x_endo_act = [X_window[:, idx_endo + k*self.args['p'] + (h-1): idx_endo + (k+1)*self.args['p']] 
                         for k in range(1+n_fEndo)]
        
        idx_fcast = max(x_endo_act[0].shape[1] - (self.args['p'] - h +1), 0)

        x_endo_fcast  = [yf[:, idx_fcast:h - 1]] + self.scaleCorrection(yf[:, idx_fcast:h-1])

        x_endo = np.column_stack([np.column_stack((x_endo_act[i], x_endo_fcast[i])) for i in range(len(x_endo_act))])
        xy = np.column_stack((x_trend, x_endo))

        if self.n_exo > 0:
            n_cross = self.n_exo if self.cross else 0
            x_exo  = np.column_stack([self.hide(X_window[:, idx_exo + k*self.args['q']: idx_exo + (k+1)*self.args['q']], h)
                                      for k in range((1+n_fExo)*self.n_exo + n_cross)])
            xy = np.column_stack((xy, x_exo))

        return xy

    def scaleCorrection(self, yfh):
        if self.tForms["tForm_y"] is None:
            fcast_ff = [f(yfh) for f in self.tForms["fEndo"]]
        else:
            f_tForm    = [lambda x: f(self.tForms["tForm_y"].inverse_transform(x)) for f in self.tForms["fEndo"]]
            fcast_list = [self.tForms["tForm_endo"].transform(np.column_stack([f(yfh[:, j:j+1]) for f in f_tForm])) for j in range(yfh.shape[1])]
            fcast_ff   = [np.column_stack([fcast_list[i][:, k:k+1] for i in range(len(fcast_list))]) for k in range(len(f_tForm))]
        return fcast_ff
    

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
            z_shifted = np.column_stack([z_last for _ in range(z.shape[1])]) if fill else np.zeros(z.shape)
        else:
            z_lst = np.column_stack([z_last for _ in range(h-1)])
            z_rest     = np.column_stack([z[:, i] for i in range(h-1, z.shape[1])])
            z_shifted = np.column_stack((z_rest, z_lst))
        return z_shifted
