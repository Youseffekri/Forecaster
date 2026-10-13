"""
Forecaster Module

Defines the abstract `Forecaster` class for time series forecasting -- the Python counterpart of
ScalaTion's `Forecaster.scala` (with `ForecastMatrix.scala` and `Diagnoser.scala` folded in).
Subclasses implement `train` and `forecast`; the base class provides the forecast matrix `Yf`,
the rolling validation, the diagnosis over all horizons and the two standard tests,
`inSample_Test` and `tnT_Test`, whose printed output follows ScalaTion:

    -------------------------------
    | In-Sample Test: ARX_D_6_0_0 |
    -------------------------------
    b = VectorD(...), fn = Array(yl6, ..., yl1)

    REPORT
        ... modelName / hparameter / features / parameter / fitMap (horizon 1) ...
    -----------------------------
    | diagnoseAll: ARX_D_6_0_0 |
    -----------------------------
    fitMap QoF =
            rSq      -> VectorD(h = 1, h = 2, ..., h = hh)
            ...

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Util.tools import array_str, banner, hparam_str, matrix_str, report, vector_str
from modeling.fit import N_QoF, QoF, showFitMap
from modeling.forecasting.diagnoser import Diagnoser


def teSize(m: int, te_ratio: float = 0.2) -> int:
    """The size of the testing set (ScalaTion's `Model.teSize`: round up m * TE_RATIO)."""
    return int(math.floor(m * te_ratio + 1.0))


def trSize(m: int, te_ratio: float = 0.2) -> int:
    """The size of the training set (ScalaTion's `Model.trSize`): m - teSize(m)."""
    return m - teSize(m, te_ratio)


def teRng(m: int, te_ratio: float = 0.2) -> range:
    """The index range of the testing set (ScalaTion's `Forecaster.teRng`)."""
    return range(trSize(m, te_ratio), m)


class Forecaster(Diagnoser, ABC):
    """
    Abstract base class for time series forecasting models.

    Attributes
    ----------
    args : dict
        The hyper-parameters: "spec" (number of trend terms: 0 implicit constant, 1 linear,
        2 quadratic, 3 cubic -- ScalaTion's convention), "p" (lags of y), "q" (lags of each
        exogenous variable, ScalaTion's px), "cross" (ENDO-EXO cross terms, ScalaTion's crx),
        "skip" (rows skipped in the In-Sample diagnosis, defaults to p).
    hh : int
        The maximum forecasting horizon (h = 1 to hh).
    y : np.ndarray, shape (m,)
        The response vector (scaled when the model was made by `rescale`).
    X : np.ndarray, shape (m, n), optional
        The data/input matrix (lagged columns of y and of the exogenous variables).
    params : np.ndarray
        The regression parameters with the intercept FIRST: a vector (n + 1,) for a recursive
        model, a matrix (n + 1, hh) for a direct one.
    parameter : np.ndarray, shape (n,)
        The slopes (horizon 1 for a direct model), as ScalaTion's `parameter` (b).
    fname : list of str
        The column names of X (ScalaTion's `getFname`).
    modelName : str
        E.g., "ARX_D_6_6_2" (p, q, n_exo), as ScalaTion names its models.
    Yf : np.ndarray, shape (m, hh + 2)
        The FORECAST MATRIX in ORIGINAL units: column 0 the actual series, columns 1..hh the
        forecasts -- Yf[t, h] is the h-steps ahead forecast of y[t + h - 1], made from the
        information up to t - 1 -- and the last column the time index.
    yForm : TransformerMixin, optional
        The scaler of y (None unless made by `rescale`); forecasts are inverted before being
        stored in Yf.
    tr_size, te_size : int
        The sizes of the initial training set and of the testing set (ScalaTion's `Model.trSize`/
        `teSize`, 80/20 by default).

    Notes
    -----
    The actual implementation uses leading underscores for internal attributes (_y, _X, _params).
    `train` must be called before `forecast`.
    """

    TE_RATIO = 0.2
    _TE_RATIO = TE_RATIO
    _lu = (0, 4)                                                   # default feature range of the MinMax scaler
    DO_REPORT = True                                               # print the REPORT in trainNtest
    SHOW_QoF = True                                                # print the QoF table in diagnoseAll

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int, hasIntercept: bool = True) -> None:
        """
        Initializes the Forecaster.

        Parameters
        ----------
        args : dict
            The hyper-parameters (see the class attributes); missing ones take ScalaTion's defaults
            (spec 1, p 6, q 0, cross False, skip p).
        y : np.ndarray, shape (m,)
            The response vector (endogenous time series).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        hasIntercept : bool, default=True
            Whether the model has an intercept (counted in the degrees of freedom).
        """
        self.args = {"spec": args.get("spec", 1),
                     "p": args.get("p", 6),
                     "q": args.get("q", 0),
                     "cross": args.get("cross", False),
                     **args}
        self.args["skip"] = args.get("skip", self.args["p"])
        super().__init__(dfr=1.0, df=max(1.0, len(y) - 1.0), hasIntercept=hasIntercept, skip=self.args["skip"])
        self.hh = hh
        self._y = np.asarray(y, dtype=float)
        self._X: Optional[np.ndarray] = None
        self._params: Optional[np.ndarray] = None
        self._nneg = bool(self.args.get("nneg", True))
        self._tr_size = trSize(len(y), self.TE_RATIO)
        self._te_size = len(y) - self._tr_size
        self._Yf = self.initialize_forecast_matrix()
        self._yForm = None
        self._modelName: Optional[str] = None
        self._fname: Optional[List[str]] = None

    # -- names ----------------------------------------------------------------------------

    @property
    def modelName(self) -> str:
        """The model name, e.g., "ARX_D_6_6_2" (set by the subclasses; the class name otherwise)."""
        return self._modelName if self._modelName is not None else self.__class__.__name__

    @property
    def hparameter(self) -> Dict[str, Any]:
        """The hyper-parameters (ScalaTion's `hparameter`)."""
        return self.args

    @property
    def fname(self) -> List[str]:
        """The feature/column names of X (ScalaTion's `getFname`); generic names when unknown."""
        if self._fname is not None:
            return self._fname
        n = self._X.shape[1] if self._X is not None else 0
        return [f"x{j}" for j in range(n)]

    def getFname(self) -> List[str]:
        """ScalaTion's name for `fname`."""
        return self.fname

    # -- data -----------------------------------------------------------------------------

    def _set_X(self, X: np.ndarray) -> None:
        """
        Sets the input matrix.

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
        """The data/input matrix (m, n)."""
        return self._X

    def getX(self) -> Optional[np.ndarray]:
        """ScalaTion's name for `X`."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """The response vector (scaled when made by `rescale`)."""
        return self._y

    def getY(self) -> np.ndarray:
        """The response vector in ORIGINAL units (column 0 of the forecast matrix)."""
        return self._Yf[:, 0]

    @property
    def yForm(self):
        """The scaler of y (None unless made by `rescale`)."""
        return self._yForm

    @property
    def tr_size(self) -> int:
        """The size of the initial training set."""
        return self._tr_size

    @property
    def te_size(self) -> int:
        """The size of the testing set."""
        return self._te_size

    @property
    def nneg(self) -> bool:
        """Whether the forecasts are constrained to be non-negative."""
        return self._nneg

    def initialize_forecast_matrix(self) -> np.ndarray:
        """
        The FORECAST MATRIX in original units: column 0 holds y, columns 1..hh are zero (to hold
        the forecasts) and the last column holds the time index (ScalaTion's `makeForecastMatrix`).

        Returns
        -------
        np.ndarray, shape (m, hh + 2)
        """
        y_fcast = np.zeros((self._y.shape[0], self.hh))
        time = np.arange(self._y.shape[0])
        return np.column_stack([self._y, y_fcast, time])

    def _reset_Yf(self) -> None:
        """Resets the forecast columns of Yf to zero."""
        self._Yf[:, 1:-1] = 0.0

    @property
    def Yf(self) -> np.ndarray:
        """The forecast matrix (m, hh + 2) in original units."""
        return self._Yf

    def getYf(self) -> np.ndarray:
        """ScalaTion's name for `Yf`."""
        return self._Yf

    # -- parameters -----------------------------------------------------------------------

    def _set_params(self, params: np.ndarray) -> None:
        """
        Sets the trained model parameters (intercept first).

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
        The trained parameters with the intercept first: (n + 1,) or (n + 1, hh).

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self._params is None:
            raise ValueError("Model parameters are not set. Call `train` first.")
        return self._params

    @property
    def parameter(self) -> np.ndarray:
        """The slope parameters b (horizon 1 for a direct model), as ScalaTion's `parameter`."""
        p = self.params
        b = p[:, 0] if p.ndim == 2 else p
        return b[1:] if self._hasIntercept else b

    @property
    def intercept(self) -> Any:
        """The intercept (one per horizon for a direct model), None when the model has none."""
        if not self._hasIntercept:
            return None
        p = self.params
        return p[0, :] if p.ndim == 2 else float(p[0])

    @property
    def nparams(self) -> int:
        """The number of slope parameters."""
        return len(self.parameter)

    def mod_resetDF(self, size: int) -> None:
        """
        Resets the degrees of freedom from the number of slopes (ScalaTion's `mod_resetDF`):
        dfr = k and df = size - k - 1 (with an intercept).

        Parameters
        ----------
        size : int
            The number of rows diagnosed (after any skipped initial rows).
        """
        self.resetDF_k(size, self.nparams)

    def __repr__(self) -> str:
        """A string showing the model name, horizon and key hyper-parameters."""
        s = f"{self.modelName}(p={self.args['p']}"
        if hasattr(self, "_n_exo"):
            s += f", q={self.args['q']}, n_exo={self._n_exo}"
        s += f"), hh = {self.hh}, skip = {self.args['skip']}"
        if hasattr(self, "_method"):
            s += f", method={self._method}"
        return s

    # -- abstract methods -----------------------------------------------------------------

    @abstractmethod
    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the forecasting model (stores the parameters with `_set_params`).

        Parameters
        ----------
        y_ : np.ndarray, shape (m_,), optional
            The response vector used for training (the full y by default).
        X_ : np.ndarray, shape (m_, n), optional
            The input matrix (the full X by default).
        """

    @abstractmethod
    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows t_start <= t < t_end and records them in
        `Yf` (in original units); returns the block of forecasts (in the model's units).

        Parameters
        ----------
        t_start : int
            The first row (defaults to 0 when -1).
        t_end : int
            The row after the last (defaults to m when -1).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
        """

    def forecastAll(self) -> np.ndarray:
        """Forecasts all rows and all horizons (ScalaTion's `forecastAll`) and returns `Yf`."""
        self.forecast()
        return self._Yf

    def _adjust_range(self, t_start: int, t_end: int) -> Tuple[int, int]:
        """
        Validates the forecast time range (t_start defaults to 0, t_end to m).

        Raises
        ------
        ValueError
            If the range is out of bounds or empty.
        """
        t_st = t_start if t_start != -1 else 0
        t_en = t_end if t_end != -1 else len(self._y)
        if not (0 <= t_st < t_en <= len(self._y)):
            raise ValueError(f"Invalid range (t_start, t_end) = ({t_st}, {t_en})")
        return t_st, t_en

    def _rectify(self, yp: np.ndarray) -> np.ndarray:
        """Clips the forecasts at zero when the non-negativity constraint is on."""
        return np.maximum(yp, 0) if self._nneg else yp

    def _store(self, t_st: int, t_en: int, y_fcast: np.ndarray) -> None:
        """Records a block of forecasts (model units) in Yf (original units)."""
        self._Yf[t_st:t_en, 1:-1] = y_fcast if self._yForm is None else self._yForm.inverse_transform(y_fcast)

    # -- testing --------------------------------------------------------------------------

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tests the trained model in-sample for horizon 1 (ScalaTion's `test`): forecasts all rows,
        resets the degrees of freedom to m - skip rows and diagnoses column 1 of Yf against y
        (the first `skip` rows dropped).

        Returns
        -------
        (np.ndarray, np.ndarray)
            The h = 1 predictions (original units) and the QoF vector.
        """
        self.forecast()
        y0, yf1 = self._Yf[:, 0], self._Yf[:, 1]
        self.mod_resetDF(len(y0) - self.getSkip())
        return yf1, self.diagnose(y0, yf1)

    def report(self, qof: np.ndarray) -> str:
        """
        Prints "b = VectorD(...), fn = Array(...)" and returns the REPORT block (ScalaTion's
        `Model.report`): model name, hyper-parameters, feature names, parameters and the fit map.
        """
        b, fn = self.parameter, self.fname
        print(f"b = {vector_str(b)}, fn = {array_str(fn)}")
        if len(fn) != len(b):
            print(f"report: fn: # features = {len(fn)} != # parameters = {len(b)}")
        return report(self.modelName, self.hparameter, fn, b, qof)

    def trainNtest(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains on (X_, y_) (the full data by default), tests in-sample for horizon 1 and prints
        the REPORT (ScalaTion's `trainNtest`).

        Returns
        -------
        (np.ndarray, np.ndarray)
            The h = 1 predictions and the QoF vector.
        """
        self.train(y_, X_)
        yp, qof = self.test()
        if self.DO_REPORT:
            print(self.report(qof))
        return yp, qof

    def diagnoseAll(self, y_: Optional[np.ndarray] = None, yf: Optional[np.ndarray] = None,
                    tRng: Optional[range] = None, showYf: bool = False) -> np.ndarray:
        """
        Diagnoses the model for all horizons (ScalaTion's `ForecastMatrix.diagnoseAll`): for each
        h, the actual values y[t1 + h - 1 .. m) are compared with column h of the forecast matrix
        over rows [t1, m - h + 1), the degrees of freedom reset to the number of rows minus `skip`,
        and the first `skip` rows dropped.  Prints the QoF table ("fitMap QoF = ...").

        Parameters
        ----------
        y_ : np.ndarray, shape (m,), optional
            The actual series in original units (column 0 of Yf by default).
        yf : np.ndarray, shape (m, hh + 2), optional
            The forecast matrix (Yf by default).
        tRng : range, optional
            The time range to diagnose (the testing rows for TnT); all rows by default.
        showYf : bool, default=False
            Whether to print the forecast matrix.

        Returns
        -------
        np.ndarray, shape (hh, N_QoF)
            One QoF vector per horizon.
        """
        y_ = self._Yf[:, 0] if y_ is None else y_
        yf = self._Yf if yf is None else yf
        m = len(y_)
        t1 = 0 if tRng is None else tRng.start
        ftMat = np.zeros((self.hh, N_QoF))
        for h in range(1, self.hh + 1):
            yy = y_[t1 + h - 1: m]
            yfh = yf[t1: m - h + 1, h]
            assert len(yy) == len(yfh)
            self.mod_resetDF(len(yy) - self.getSkip())
            ftMat[h - 1] = self.diagnose(yy, yfh)
        if showYf:
            print(f"Final Forecast Matrix yf = {matrix_str(yf)}")
        if self.SHOW_QoF:
            print("fitMap QoF = ")
            print(showFitMap(ftMat.T))
        return ftMat

    def rollValidate(self, rc: int = 5, growing: bool = False) -> np.ndarray:
        """
        Rolling validation (ScalaTion's `rollValidate`): the data set is split into a TRAINING SET
        (tr) and a TESTING SET (te) [ <-- tr_size --> | <-- te_size --> ]; the model is retrained
        every `rc` rows on the sliding (or growing) training window and forecasts h = 1..hh
        steps ahead for each testing row.  The forecasts are recorded in `Yf`.

        Parameters
        ----------
        rc : int, default=5
            The retraining cycle (number of forecasts until retraining occurs).
        growing : bool, default=False
            Whether the training window grows or keeps a fixed size.

        Returns
        -------
        np.ndarray, shape (te_size, hh)
            The block of forecasts for the testing rows (model units); `Yf` holds them in
            original units.
        """
        banner(f"rollValidate: Evaluate {self.modelName}'s QoF for horizons 1 to {self.hh}:")
        self._reset_Yf()
        y_fcast = np.zeros((self._te_size, self.hh))
        for i in range(0, self._te_size, rc):
            is_ = 0 if growing else i
            t = self._tr_size + i
            X_ = self._X[is_:t] if self._X is not None else None
            self.train(self._train_target(is_, t), X_)
            t_end = min(t + rc, len(self._y))
            y_fcast[i:i + (t_end - t), :] = self.forecast(t, t_end)
        return y_fcast

    def _train_target(self, is_: int, t: int) -> np.ndarray:
        """The training target rows [is_, t) (a vector here; a matrix for direct models)."""
        return self._y[is_:t]

    def inSample_Test(self, showYf: bool = False, showParams: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        In-Sample testing (ScalaTion's `inSample_Test`): trains and tests on the full data set,
        prints the REPORT for horizon 1 and the QoF table for all horizons.

        Parameters
        ----------
        showYf : bool, default=False
            Whether to print the forecast matrix.
        showParams : bool, default=False
            Whether to print the full parameter array (intercept first).

        Returns
        -------
        (np.ndarray, np.ndarray)
            The h = 1 predictions and their QoF vector.
        """
        banner(f"In-Sample Test: {self.modelName}")
        yp, qof = self.trainNtest()
        self.forecastAll()
        banner(f"diagnoseAll: {self.modelName}")
        self.diagnoseAll()
        if showYf:
            print(f"Final In-Sample Forecast Matrix yf = {matrix_str(self._Yf)}")
        if showParams:
            print(f"params = \n{self.params}")
        return yp, qof

    def tnT_Test(self, rc: int = 5, showYf: bool = False, showParams: bool = False) -> np.ndarray:
        """
        Train-and-Test (TnT) testing with rolling validation (ScalaTion's `tnT_Test`): trains on
        the full data (REPORT), then rolls through the testing set retraining every `rc` rows and
        prints the QoF table on the testing rows (no rows skipped).

        Parameters
        ----------
        rc : int, default=5
            The retraining cycle.
        showYf : bool, default=False
            Whether to print the forecast matrix.
        showParams : bool, default=False
            Whether to print the full parameter array of the last retraining.

        Returns
        -------
        np.ndarray, shape (hh, N_QoF)
            One QoF vector per horizon on the testing rows.
        """
        banner(f"TnT Test: {self.modelName}")
        self.trainNtest()
        self.forecastAll()
        self.setSkip()
        self.rollValidate(rc)
        banner(f"diagnoseAll: {self.modelName}")
        ftMat = self.diagnoseAll(tRng=teRng(len(self._y), self.TE_RATIO))
        if showYf:
            print(f"Final TnT Forecast Matrix yf = {matrix_str(self._Yf)}")
        if showParams:
            print(f"params = \n{self.params}")
        return ftMat

    def trainNtest_Test(self, rc: int = 5, showYf: bool = False, showParams: bool = False) -> np.ndarray:
        """The former name of `tnT_Test` (kept for compatibility)."""
        return self.tnT_Test(rc, showYf, showParams)
