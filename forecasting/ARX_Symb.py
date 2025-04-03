"""
ARX_Symb Model for Symbolic Time Series Forecasting

Extends the ARX (Autoregressive with Exogenous Variables) model by applying nonlinear symbolic 
transformations to lagged inputs.

Supports multi-step forecasting using lagged endogenous variables, exogenous variables, 
and their symbolic transformations. Optionally supports cross terms between endogenous and exogenous variables.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import *
from forecasting.ARX import ARX


class ARX_Symb(ARX):
    """
    Symbolic Autoregressive model with Exogenous variables (ARX_Symb) for time series forecasting.

    Extends the ARX model by incorporating nonlinear symbolic transformations of lagged endogenous 
    and exogenous variables. Also supports optional scaling of input/output using sklearn transformers.

    Attributes
    ----------
    tForms : dict
        Dictionary containing symbolic functions and optional sklearn transformers applied to:
        - "tForm_y"    : transformation applied to the target series `y`
        - "tForm_endo" : transformation applied to derived endogenous features
        - "tForm_exo"  : transformation applied to derived exogenous features
        - "fEndo"      : symbolic functions applied to `y` before lagging
        - "fExo"       : symbolic functions applied to exogenous variables before lagging

    See Also
    --------
    ARX : For additional shared attributes such as model, method, n_exo, and use_forge.
    Forecaster : For base forecasting logic and shared attributes like y, X, Yf, tr_size, etc.
    """

    def __init__(self, 
                 args: Dict[str, Any], 
                 y: np.ndarray, 
                 hh: int,
                 xe: Optional[np.ndarray] = None, 
                 fEndo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None, 
                 fExo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,  
                 method: str ="sm_ols", 
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_Symb forecaster.

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
        fEndo : list of callables, optional
            Nonlinear transformations applied to endogenous variables.
        fExo : list of callables, optional
            Nonlinear transformations applied to exogenous variables.
        method : {"sm_ols", "sk_lr"}, default="sm_ols"
            Whether to use statsmodels OLS or sklearn Linear Regression.
        X : Optional[np.ndarray], shape (n_samples, n_lagged_columns), default=None
            Input feature matrix. If None, the model will build it from `y`, `xe`, and other configuration settings.
        
        Notes
        -----
        If no symbolic functions are provided, default transformations (e.g., sqrt, power, log) are used.
        """
        functions = [lambda x: np.power(x, 1.5),
                     lambda x: np.power(x, 0.5),
                     np.log1p]

        self._tForms = {"tForm_y": None}
        self._tForms["fEndo"] = fEndo if fEndo is not None else functions
        self._tForms["fExo"]  = fExo if fExo is not None else functions
        super().__init__(args, y, hh, xe, method, X)
    

    @classmethod
    def rescale(cls, 
                args: Dict[str, Any], 
                y: np.ndarray, 
                hh: int,
                xe: Optional[np.ndarray] = None,
                fEndo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None, 
                fExo: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,   
                method: str ="sm_ols", 
                tForm: Optional[Type[TransformerMixin]] = None, 
                X: Optional[np.ndarray] = None) -> "ARX_Symb":
        """
        Alternative constructor that enables rescaling using a transformation function.

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
        fEndo : list of callables, optional
            Nonlinear transformations applied to endogenous variables.
        fExo : list of callables, optional
            Nonlinear transformations applied to exogenous variables.
        method : {"sm_ols", "sk_lr"}, default="sm_ols"
            Whether to use statsmodels OLS or sklearn Linear Regression.
        tForm : Optional[Type[TransformerMixin]]
            Transformation used for scaling `y` and forecast outputs,
            e.g., MinMaxScaler or StandardScaler.
        X : Optional[np.ndarray], shape (n_samples, n_lagged_columns), default=None
            Input feature matrix. If None, the model will build it from `y`, `xe`, and other configuration settings.

        Returns
        -------
        ARX_Symb
            Instance of ARX_Symb with scaling enabled.
        """
        ARX_Symb_re = cls(args, y, hh, xe, fEndo, fExo, method, np.array([0]))

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
            ARX_Symb_re._set_X(ARX_Symb_re._build_matrix(args   = ARX_Symb_re.args,
                                                         y      = ARX_Symb_re._y, 
                                                         xe      = ARX_Symb_re._xe,                                                        
                                                         tForms = ARX_Symb_re._tForms))
        else:
            ARX_Symb_re._X = ARX_Symb_re._tForms["tForm_exo"].fit_transform(ARX_Symb_re._X)
        
        ARX_Symb_re._yForm = ARX_Symb_re._tForms["tForm_y"]
        ARX_Symb_re._y = ARX_Symb_re._tForms["tForm_y"].transform(ARX_Symb_re._y.reshape(-1, 1)).flatten()
        return ARX_Symb_re


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
            - "tForm_endo" : transformation applied to derived endogenous features
            - "tForm_exo"  : transformation applied to derived exogenous features
            - "fEndo"      : symbolic functions applied to `y` before lagging
            - "fExo"       : symbolic functions applied to exogenous variables before lagging

        Returns
        -------
        np.ndarray
            Feature matrix combining trend, endogenous, and exogenous lagged variables.
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

        if xe is not None:
            xe_bfill = backfill_matrix(xe)
            if len(tForms["fExo"]) > 0:
                xe_fExo = np.column_stack([f(xe_bfill) for f in tForms["fExo"]])
                xe_fExo = np.column_stack((xe_bfill, xe_fExo))
            else:
                xe_fExo = xe_bfill.copy()

            if cross:
                yxe = np.column_stack([y*xe_bfill[:, j] for j in range(xe.shape[1])])
                xe_fExo  = np.column_stack((xe_fExo, yxe))

            if tForms["tForm_y"] is not None:
                xe_fExo = tForms["tForm_exo"].fit_transform(xe_fExo)
            x_exo = np.column_stack([build_lagged_matrix(xe_fExo[:, j], q) for j in range(xe_fExo.shape[1])])
            X = np.column_stack((X, x_exo))

        if spec > 1:
            xt = build_trend_matrix(len(y), spec)
            X = np.column_stack((xt, X))

        return X


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
        np.ndarray
            Feature matrix with actual lags and prior forecasts for horizon `h`.
        """
        n_fEndo = len(self._tForms["fEndo"])
        n_fExo = len(self._tForms["fExo"])

        idx_endo = self.args['spec'] - 1
        idx_exo  = idx_endo + (1+n_fEndo)*self.args['p']
        x_trend = X_window[:, :idx_endo]
        x_endo_act = [X_window[:, idx_endo + k*self.args['p'] + (h-1): idx_endo + (k+1)*self.args['p']] 
                         for k in range(1+n_fEndo)]
        
        idx_fcast = max(x_endo_act[0].shape[1] - (self.args['p'] - h +1), 0)
        x_endo_fcast  = [y_fcast[:, idx_fcast:h - 1]] + self._scaleCorrection(y_fcast[:, idx_fcast:h-1])
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
        Applies inverse transformation to forecast values (if scaled), followed by 
        symbolic transformations defined in `fEndo`.

        Parameters
        ----------
        yfh : np.ndarray
            Forecast values for future time steps.

        Returns
        -------
        List[np.ndarray]
            List of feature arrays resulting from applying each `fEndo` transformation 
            to the forecast values.
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
