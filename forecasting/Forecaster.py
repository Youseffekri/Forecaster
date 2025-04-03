"""
Forecaster Module

Defines an abstract `Forecaster` class for time series forecasting. 
Subclasses must implement the `train` and `forecast` methods.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import TransformerMixin

from Util.tools import diagnose

class Forecaster(ABC):
    """
    Abstract base class for time series forecasting models.

    Attributes
    ----------
    args : dict
        Dictionary of model configuration parameters such as trend specification, 
        number of lags, and optional cross terms.
    hh : int
        The maximum forecasting horizon (h = 1 to hh).
    y : np.ndarray, shape (n_samples,)
        The response vector representing the endogenous time series to be forecasted.
    X : Optional[np.ndarray], shape (n_samples, n_lagged_columns)
        The data/input matrix including lagged columns of the endogenous (`y`) and 
        exogenous variables, and optionally their transformations or cross terms.
    params : Optional[np.ndarray], shape (n_lagged_columns + 1,)
        Model parameters obtained after training.
    nneg : bool, default=True
        If True, applies non-negativity constraint to predictions.  
    tr_size : int
        Size of initial training set.
    te_size : int
        Size of testing set.   
    Yf : np.ndarray, shape (n_samples, hh + 2)
        Forecast matrix containing the raw input `y`, forecast outputs in the same scale 
        as the input `y`, and time indices.
    modelName : str
        The name for the model    
    yForm : Optional[Type[TransformerMixin]]
        Optional transformation used for scaling `y` and forecast outputs.
        e.g., MinMaxScaler or StandardScaler.
    TE_RATIO : float, default = 0.2
        Ratio of the testing set to the full dataset.
    lu : tuple(float, float)
        Default bounds for MinMaxScaler's feature_range

    Notes
    -----
    The actual implementation uses leading underscores for internal attributes (e.g., _y, _X, _params).
    The `train` method must be called before `forecast`.   
    """

    _TE_RATIO = 0.2
    _lu = (0, 4)

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int) -> None:
        """
        Initializes the Forecaster.

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
        self.args = {
            "spec": args.get("spec", 1),
            "p": args.get("p", 6),
            "q": args.get("q", 0),
            "cross": args.get("cross", False),
            **args
        }
        self.args["skip"] = args.get("skip", self.args["p"])
        self.hh = hh
        self._y = y
        self._X: Optional[np.ndarray] = None
        self._params: Optional[np.ndarray] = None
        self._nneg = True
        self._tr_size = int((1.0 - self._TE_RATIO) * len(y))
        self._te_size = len(y) - self._tr_size
        self._Yf = self.initialize_forecast_matrix()
        self._modelName = "Model"
        self._yForm = None

    def _set_X(self, X: np.ndarray) -> None:
        """
        Sets the input matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_lagged_columns)
            The data/input matrix including lagged columns of the endogenous (`y`) and 
            exogenous variables, and optionally their transformations or cross terms.

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
        self._X = X

    @property
    def X(self) -> Optional[np.ndarray]:
        """
        Returns
        -------
        np.ndarray, shape (n_samples, n_lagged_columns)
            The input matrix.
        """
        return self._X

    @property
    def y(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray, shape (n_samples,)
            The response vector (endogenous time series).
        """
        return self._y


    def initialize_forecast_matrix(self) -> np.ndarray:
        """
        Initializes a forecast matrix to store the observed time series and 
        future predictions in the same scale as the original data.

        Returns
        -------
        Yf : np.ndarray, shape (n_samples, hh + 2)
            Forecast matrix where:
            - Column 0 stores the original unscaled time series `y`.
            - Columns 1 to hh are initialized to zeros (to hold forecast values).
            - Last column stores time indices (0 to n_samples - 1).

        Notes
        -----
        All values are stored in the same scale as the raw input `y`, typically the original
        scale before any transformation.
        """
        y_fcast = np.zeros((self._y.shape[0], self.hh))
        time = np.arange(self._y.shape[0])
        Yf = np.column_stack([self._y, y_fcast, time])
        return Yf

    def _reset_Yf(self) -> None:
        """
        Resets the forecast columns in the Yf matrix to zero.
        """
        y, time = self._Yf[:, 0], self._Yf[:, -1]
        y_fcast = np.zeros((self._Yf.shape[0], self._Yf.shape[1]-2))
        self._Yf = np.column_stack([y, y_fcast, time])

    @property
    def Yf(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray, shape (n_samples, hh + 2)
            The forecast matrix containing observed values, forecast outputs, 
            and time indices.
        """
        return self._Yf
    
    def _set_params(self, params: np.ndarray) -> None:
        """
        Sets the trained model parameters.

        Parameters
        ----------
        params : np.ndarray, shape (n_lagged_columns + 1,)
            Model parameters to store.

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
        self._params = params

    @property
    def params(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray, shape (n_lagged_columns + 1,)
            Trained model parameters.

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self._params is None:
            raise ValueError("Model parameters are not set. Call `train` first.")
        return self._params

    @property
    def modelName(self) -> str:
        """
        Returns
        -------
        str
            The name of the model 
        """
        return self._modelName
    
    @abstractmethod
    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the forecasting model. This method updates the model parameters (see `params`).

        Parameters
        ----------
        y_ : Optional[np.ndarray], shape (n_samples,)
            The response vector (endogenous time series) used for training.
        X_ : Optional[np.ndarray], shape (n_samples, n_lagged_columns)
            The input matrix including lagged terms used as predictors.

        Notes
        -----
        Subclasses must implement this method to train the model and store parameters using self._set_params().
        """
        pass

    @abstractmethod
    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Generates a multi-step forecast matrix and updates the `Yf` attribute.

        Parameters
        ----------
        t_start : int
            Start index for forecasting. If -1, defaults to self.args["skip"].
        t_end : int
            End index for forecasting (exclusive). If -1, defaults to the length of the target series.

        Returns
        -------
        np.ndarray
            Multi-step forecast outputs for the specified time range.

        Notes
        -----
        Subclasses must implement this method to forecast values using the trained model.
        """  
        pass


    def _adjust_range(self, t_start: int, t_end: int) -> tuple[int, int]:
        """
        Validates and adjusts the forecast time range.

        Parameters
        ----------
        t_start : int
            Start index for forecasting. If -1, defaults to self.args["skip"].
        t_end : int
            End index for forecasting (exclusive). If -1, defaults to the length of the target series.

        Returns
        -------
        t_st, t_en: tuple[int, int]
            A tuple (t_st, t_en) representing the validated and adjusted start and end indices.

        Raises
        ------
        ValueError
            If the resulting range is invalid (e.g., out of bounds or t_start >= t_end).
        """
        t_st = t_start if t_start != -1 else self.args["skip"]
        t_en = t_end if t_end != -1  else len(self._y)

        if not (0 <= t_st < t_en <= len(self._y)):
            raise ValueError(f"Invalid range (t_start, t_end) = ({t_st}, {t_en})")

        return t_st, t_en

    def _rectify(self, yp: np.ndarray) -> np.ndarray:
        """
        Applies non-negativity constraint to forecast values if enabled.

        Parameters
        ----------
        yp : np.ndarray
            Forecast values.

        Returns
        -------
        np.ndarray
            Forecast values clipped at zero if _nneg is True.
        """
        if self._nneg:
            return np.maximum(yp, 0)
        else:
            return yp

    def diagnose_all(self, y_fcast: np.ndarray, TnT: bool = False) -> str:
        """
        Computes Quality of Fit (QoF) measures for all horizons.

        Parameters
        ----------
        y_fcast : np.ndarray
            Forecast values for each horizon.
        TnT : bool, default=False
            If True, only test set is used for evaluation.

        Returns
        -------
        qof : str
            Formatted metrics across all forecast horizons.
        """
        y_true = self._y[self._tr_size:].copy() if TnT else self._y[self.args["skip"]:].copy()
        hh = y_fcast.shape[1]
        ll = len(y_true)

        horizon_all = f"horizon\t->"
        mse_all     = f"mse\t->"
        mae_all     = f"mae\t->"
        r2_all      = f"R^2\t->"
        r2Bar_all   = f"R^2Bar\t->"
        smape_all   = f"smape\t->"
        m_all       = f"m\t->"

        if self._yForm is not None:
            y_true = self._yForm.inverse_transform(y_true.reshape(-1, 1)).flatten()

        for h in range(hh):
            y_fcast_h = y_fcast[:ll - h, h] if self._yForm is None else self._yForm.inverse_transform(y_fcast[:ll - h, h].reshape(-1, 1)).flatten()

            metrics = diagnose(y_true[h:], y_fcast_h)
            horizon_all += f"\t  {h + 1}\t"
            mse_all     += f"\t{metrics["MSE"]:.4f}\t"
            mae_all     += f"\t{metrics["MAE"]:.4f}\t"
            r2_all      += f"\t{metrics["R2"]:.4f}\t"
            r2Bar_all   += f"\t{metrics["R2Bar"]:.4f}\t"
            smape_all   += f"\t{metrics["SMAPE"]:.4f}\t"
            m_all       += f"\t{metrics["m"]}\t"

        qof = (
            horizon_all + "\n" +
            mse_all + "\n" +
            mae_all + "\n" +
            r2_all + "\n" +
            r2Bar_all + "\n" +
            smape_all + "\n" +
            m_all + "\n"
        )
        return qof

    def rollValidate(self, rc: int = 2, growing: bool = False) -> np.ndarray:
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
        y_fcast: np.ndarray, shape (te_size, hh)
            Forecast values across rolling steps.
        """
        self._reset_Yf()
        y_fcast = np.zeros((self._te_size, self.hh))
        for i in range(0, self._te_size, rc):
            is_ = 0 if growing else i
            t = self._tr_size + i
            X_ = self._X[is_:t] if self._X is not None else None
            y_ = self._y[is_:t]
            self.train(y_, X_)
            if i+rc < self._te_size:
                y_fcast[i:i + rc, :] = self.forecast(t, t + rc)
            else:
                y_fcast[i:, :] = self.forecast(t)
        return y_fcast


    def inSample_Test(self, showParams: bool = False, showYf: bool = False) -> None:
        """
        Runs an in-sample evaluation of the forecasting model and prints forecast quality metrics.

        Parameters
        ----------
        showParams : bool, default=False
            Whether to print trained model parameters.
        showYf : bool, default=False
            Whether to print the forecast matrix `Yf`.
        """
        print(f"In-Sample Test: {self.modelName}")
        self.train()
        y_fcast = self.forecast()
        qof = self.diagnose_all(y_fcast)
        print(f"QoF:\n{qof}")
        if showParams: print(f"params = {self.params}")
        if showYf: print(f"Yf = \n{self.Yf}")

    def trainNtest_Test(self, showParams: bool = False, showYf: bool = False) -> None:
        """
        Runs a rolling validation test and prints forecast quality metrics.

        Parameters
        ----------
        showParams : bool, default=False
            Whether to print trained model parameters.
        showYf : bool, default=False
            Whether to print the forecast matrix `Yf`.
        """
        print(f"Roll Validate Test: {self.modelName}")
        y_fcast = self.rollValidate()
        qof = self.diagnose_all(y_fcast, TnT = True)
        print(f"QoF:\n{qof}")
        if showParams: print(f"params = {self.params}")
        if showYf: print(f"Yf = \n{self.Yf}")