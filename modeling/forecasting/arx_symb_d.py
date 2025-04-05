"""
ARX_Symb_D Model for Symbolic Time Series Forecasting

Extends the ARX_D (Autoregressive with Exogenous Variables) model by applying nonlinear symbolic 
transformations to lagged inputs.

Supports multi-step forecasting using lagged endogenous variables, exogenous variables, 
and their symbolic transformations. Optionally includes cross terms between  endogenous and exogenous variables.

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
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_symb import ARX_Symb


class ARX_Symb_D(ARX_D):
    """
    Symbolic Autoregressive model with Exogenous variables (ARX_Symb_D) for time series forecasting.

    Extends modeling.the ARX_D model by incorporating nonlinear symbolic transformations of lagged endogenous 
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
    ARX_D : For additional shared attributes such as model, method, n_exo, and use_forge.
    Forecaster_D : For base forecasting logic and shared attributes like y, X, Yf, tr_size, etc.
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
        Initializes the ARX_Symb_D forecaster.

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
                X: Optional[np.ndarray] = None) -> "ARX_Symb_D":
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
        ARX_Symb_D
            Instance of ARX_Symb_D with scaling enabled.
        """
        ARX_Symb_D_re = cls(args, y, hh, xe, fEndo, fExo, method, np.array([0]))

        ARX_Symb_D_re._X = X
        if tForm is not None:
            ARX_Symb_D_re._tForms["tForm_y"]    = tForm()
            ARX_Symb_D_re._tForms["tForm_endo"] = tForm()
            ARX_Symb_D_re._tForms["tForm_exo"]  = tForm()
            if isinstance(ARX_Symb_D_re._tForms["tForm_y"], StandardScaler): 
                ARX_Symb_D_re._nneg = False
        else:
            ARX_Symb_D_re._tForms["tForm_y"]    = MinMaxScaler(feature_range=ARX_Symb_D_re._lu)
            ARX_Symb_D_re._tForms["tForm_endo"] = MinMaxScaler(feature_range=ARX_Symb_D_re._lu)
            ARX_Symb_D_re._tForms["tForm_exo"]  = MinMaxScaler(feature_range=ARX_Symb_D_re._lu)

        if ARX_Symb_D_re._X is None:
            ARX_Symb_D_re._useforge = True
            ARX_Symb_D_re._set_X(ARX_Symb_D_re._build_matrix(args   = ARX_Symb_D_re.args,
                                                             y      = ARX_Symb_D_re._y, 
                                                             xe     = ARX_Symb_D_re._xe,                                                        
                                                             tForms = ARX_Symb_D_re._tForms))
        else:
            ARX_Symb_D_re._X = ARX_Symb_D_re._tForms["tForm_exo"].fit_transform(ARX_Symb_D_re._X)
        
        ARX_Symb_D_re._yForm = ARX_Symb_D_re._tForms["tForm_y"]
        ARX_Symb_D_re._y = ARX_Symb_D_re._tForms["tForm_y"].transform(ARX_Symb_D_re._y.reshape(-1, 1)).flatten()
        ARX_Symb_D_re._Y = build_matrix_Y(ARX_Symb_D_re._y, hh)
        return ARX_Symb_D_re


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
        return ARX_Symb._build_matrix(args, y, xe, tForms)