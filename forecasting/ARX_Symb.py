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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import *
from forecasting.ARX import ARX


class ARX_Symb(ARX):
    """
    Implements an Autoregressive model with Exogenous variables (ARX) using symbolic transformations. 
    Extends the ARX class by incorporating nonlinear transformations of lagged variables 
    and supports optional scaling of inputs and outputs.

    Attributes
    ----------
    model : LinearRegression
        Trained linear regression model for forecasting.
    use_forge : bool
        Indicates whether the feature matrix X was automatically generated using buildMatrix4TS.
    n_exo : int
        Number of exogenous variables (equal to z.shape[1]).
    tForms : dict
        Dictionary of transformations applied to endogenous and exogenous inputs.
    X : Optional[np.ndarray]
        Input feature matrix. Either provided by the user or built internally.
    y : np.ndarray
        Response vector (endogenous time series).
    params : Optional[np.ndarray]
        Model parameters after training.
    nneg : bool
        Whether to enforce non-negative predictions.
    tr_size : int
        Size of the training dataset.
    te_size : int
        Size of the testing dataset.
    yForm : optional
        Transformation applied to the target series for scaling.

    Notes
    -----
    The actual implementation uses leading underscores for internal attributes 
    (e.g., _X, _y, _params, _tForms, _useforge).
    Feature transformations include functions such as square root, power, log1p, sine, and cosine.
    """

    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray, 
                 hh: int,
                 z: Optional[np.ndarray] = None, 
                 fEndo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None, 
                 fExo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,  
                 method: str ="sk_lr", 
                 X: Optional[np.ndarray] = None):
        """
        Initializes the ARX_Symb forecaster.

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
        fEndo : list of callables, optional
            Nonlinear transformations applied to endogenous variables.
        fExo : list of callables, optional
            Nonlinear transformations applied to exogenous variables.
        method : {"sm_ols", "sk_lr"}, default="sk_lr"
            Estimation method.
        X : Optional[np.ndarray], default=None
            Feature matrix. If None, it will be constructed internally.
        """
        functions = [lambda x: np.power(x, 1.5),
                     lambda x: np.power(x, 0.5),
                     np.log1p]
                    #  ,
                    #  lambda x: np.sin(x*np.pi/40),
                    #  lambda x: np.cos(x*np.pi/40)]

        self._tForms = {"tForm_y": None}
        self._tForms["fEndo"] = fEndo if fEndo is not None else functions
        self._tForms["fExo"]  = fExo if fExo is not None else functions
        super().__init__(args, y, hh, z, method, X)
        self._modelName = f"ARX_Symb(p={self.args["p"]}, q={self.args["q"]}, n_exo={self._n_exo}), method={method}" 
    
    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                z: Optional[np.ndarray] = None,
                fEndo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None, 
                fExo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,   
                method: str ="sk_lr", 
                tForm = None, 
                X: Optional[np.ndarray] = None):
        """
        Alternative constructor that enables rescaling using a transformation function.

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
        fEndo : list of callables, optional
            Transformations for endogenous variables.
        fExo : list of callables, optional
            Transformations for exogenous variables.
        method : {"sm_ols", "sk_lr"}
            Estimation method.
        tForm : Transformer, optional
            Transformation class such as MinMaxScaler or StandardScaler.
        X : Optional[np.ndarray]
            Predefined feature matrix.

        Returns
        -------
        ARX_Symb
            Instance of ARX_Symb with scaling enabled.
        """
        ARX_Symb_re = cls(args, y, hh, z, fEndo, fExo, method, np.array([0]))

        ARX_Symb_re._X = X
        if tForm is not None:
            ARX_Symb_re._tForms["tForm_y"]    = tForm()
            ARX_Symb_re._tForms["tForm_endo"] = tForm()
            ARX_Symb_re._tForms["tForm_exo"]  = tForm()
            if isinstance(ARX_Symb_re._tForms["tForm_y"], StandardScaler): 
                ARX_Symb_re._nneg = False
        else:
            ARX_Symb_re._tForms["tForm_y"]    = MinMaxScaler(feature_range=ARX_Symb_re._lu)
            ARX_Symb_re._tForms["tForm_endo"] = MinMaxScaler(feature_range=ARX_Symb_re._lu)
            ARX_Symb_re._tForms["tForm_exo"]  = MinMaxScaler(feature_range=ARX_Symb_re._lu)

        if ARX_Symb_re._X is None:
            ARX_Symb_re._useforge = True
            ARX_Symb_re._set_X(ARX_Symb_re._build_matrix(y      = ARX_Symb_re._y, 
                                                         z      = ARX_Symb_re._z,
                                                         args   = ARX_Symb_re.args,
                                                         tForms = ARX_Symb_re._tForms))
        else:
            ARX_Symb_re._X = ARX_Symb_re._tForms["tForm_exo"].fit_transform(ARX_Symb_re._X)
        
        ARX_Symb_re._yForm = ARX_Symb_re._tForms["tForm_y"]
        ARX_Symb_re._y = ARX_Symb_re._tForms["tForm_y"].transform(ARX_Symb_re._y.reshape(-1, 1)).flatten()
        return ARX_Symb_re

    def _build_matrix(self, y, z, args, tForms):       
        """
        Builds the feature matrix from lagged endogenous and exogenous variables 
        with optional transformations and trend terms.

        Parameters
        ----------
        y : np.ndarray
            Target time series.
        z : Optional[np.ndarray]
            Exogenous variables.
        args : dict
            Model configuration including p, q, and spec.
        tForms : dict
            Dictionary of transformations to apply.

        Returns
        -------
        np.ndarray
            Feature matrix for regression.
        """
        p = args["p"]
        q = args["q"]
        spec = args["spec"]
        cross = args["cross"]
        y_fEndo = np.column_stack([f(y) for f in tForms["fEndo"]])

        if tForms["tForm_y"] is not None:
            y       = tForms["tForm_y"].fit_transform(y.reshape(-1, 1)).flatten()
            y_fEndo = tForms["tForm_endo"].fit_transform(y_fEndo)
        y_fEndo = np.column_stack((y, y_fEndo))
        X = np.column_stack([build_lagged_matrix(y_fEndo[:, j], p) for j in range(y_fEndo.shape[1])])

        if z is not None:
            z_bfill = backfill_matrix(z)
            if len(tForms["fExo"]) > 0:
                z_fExo = np.column_stack([f(z_bfill) for f in tForms["fExo"]])
                z_fExo = np.column_stack((z_bfill, z_fExo))
            else:
                z_fExo = z_bfill.copy()

            if cross:
                yz = np.column_stack([y*z_bfill[:, j] for j in range(z.shape[1])])
                z_fExo  = np.column_stack((z_fExo, yz))

            if tForms["tForm_y"] is not None:
                z_fExo = tForms["tForm_exo"].fit_transform(z_fExo)
            x_exo = np.column_stack([build_lagged_matrix(z_fExo[:, j], q) for j in range(z_fExo.shape[1])])
            X = np.column_stack((X, x_exo))

        if spec > 1:
            xt = build_trend_matrix(len(y), spec)
            X = np.column_stack((xt, X))
        return X

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
            raise ValueError("Cannot use forge when the input matrix X was provided by user!")

        n_fEndo = len(self._tForms["fEndo"])
        n_fExo = len(self._tForms["fExo"])

        idx_endo = self.args['spec'] - 1
        idx_exo  = idx_endo + (1+n_fEndo)*self.args['p']
        x_trend = X_window[:, :idx_endo]
        x_endo_act = [X_window[:, idx_endo + k*self.args['p'] + (h-1): idx_endo + (k+1)*self.args['p']] 
                         for k in range(1+n_fEndo)]
        
        idx_fcast = max(x_endo_act[0].shape[1] - (self.args['p'] - h +1), 0)
        x_endo_fcast  = [yf[:, idx_fcast:h - 1]] + self._scaleCorrection(yf[:, idx_fcast:h-1])
        x_endo = np.column_stack([np.column_stack((x_endo_act[i], x_endo_fcast[i])) for i in range(len(x_endo_act))])
        xy = np.column_stack((x_trend, x_endo))

        if self._n_exo > 0:
            n_cross = self._n_exo if self.args["cross"] else 0
            x_exo  = np.column_stack([self._hide(X_window[:, idx_exo + k*self.args['q']: idx_exo + (k+1)*self.args['q']], h)
                                      for k in range((1+n_fExo)*self._n_exo + n_cross)])
            xy = np.column_stack((xy, x_exo))
        return xy

    def _scaleCorrection(self, yfh: np.ndarray) -> List[np.ndarray]:
        """
        Applies inverse transformation to forecasted values and applies 
        symbolic transformations defined in fEndo.

        Parameters
        ----------
        yfh : np.ndarray
            Forecasted values for future time steps.

        Returns
        -------
        List[np.ndarray]
            List of transformed feature arrays for each fEndo function.
        """
        if self._tForms["tForm_y"] is None:
            fcast_ff = [f(yfh) for f in self._tForms["fEndo"]]
        else:
            f_tForm = [lambda x: f(self._tForms["tForm_y"].inverse_transform(x)) for f in self._tForms["fEndo"]]
            fcast_list = [self._tForms["tForm_endo"].transform(
                              np.column_stack([f(yfh[:, j:j+1]) for f in f_tForm]))
                          for j in range(yfh.shape[1])]
            fcast_ff = [np.column_stack([fcast_list[i][:, k:k+1] for i in range(len(fcast_list))]) 
                        for k in range(len(f_tForm))]
        return fcast_ff
