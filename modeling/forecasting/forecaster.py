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

The hyper-parameters start from `hp` (ScalaTion's `AR.hp`: p, d, q, P, D, Q, s, px, spec, crx, nneg
with their defaults), a `HyperParameter`, which the REPORT prints as ScalaTion does
("HyperParameter(p -> (6,1), ...)").  The regression the forecasters delegate to (`_make_model`) is
"sm_ols" (statsmodels), "sk_lr" (scikit-learn) or "ridge" -- ScalaTion's `CenteredRidge(MV)`, its
only regression, with penalty hparam["lambda"] (default `CenteredRidge.hp`, 0.1) and `scaled`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from Util.make_matrix4ts import centered
from Util.tools import array_str, banner, matrix_str, vector_str
from modeling.centered_ridge import CenteredRidge
from modeling.fit import N_QoF, fitMap, showFitMap
from modeling.forecasting.diagnoser import Diagnoser
from modeling.hyper_parameter import HyperParameter

#: ScalaTion's `AR.hp`: the base hyper-parameters every forecaster starts from (value, default);
#: `AR.hp` is this object.  Python-only entries ("skip", "lambda", ...) are added per model.
hp = HyperParameter()
hp.add("p", 1, 1)                  # number of Auto-Regressive (AR) parameters (lags of y)
hp.add("d", 1, 1)                  # number of Differences to take
hp.add("q", 1, 1)                  # number of Moving-Average (MA) parameters
hp.add("P", 1, 1)                  # number of Seasonal Auto-Regressive (AR) parameters
hp.add("D", 1, 1)                  # number of Seasonal Differences to take
hp.add("Q", 1, 1)                  # number of Seasonal Moving-Average (MA) parameters
hp.add("s", 7, 7)                  # length of the Seasonal Period
hp.add("px", 1, 1)                 # number of lags for the eXogenous variables
hp.add("spec", 1, 1)               # trend terms: 0 - implicit constant via centering, 1 - linear, 2 - quadratic, 3 - cubic
hp.add("crx", 0, 0)                # 0 - no ENDO-EXO cross terms, 1 - include ENDO-EXO cross terms
hp.add("nneg", 1, 1)               # 0 - unrestricted, 1 - predictions must be non-negative

REGRESSION_METHODS = ("sm_ols", "sk_lr", "ridge")


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
    hparam : HyperParameter
        The hyper-parameters: `hp` (ScalaTion's `AR.hp`: "p" lags of y, "px" lags of each exogenous
        variable, "spec" trend terms -- 0 implicit constant, 1 linear, 2 quadratic, 3 cubic --, "crx"
        ENDO-EXO cross terms, "nneg" non-negative forecasts, ...) updated by the given ones, plus
        "skip" (rows skipped in the In-Sample diagnosis, defaults to p) and, for method "ridge",
        "lambda".
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
        The coefficients b (horizon 1 for a direct model), as ScalaTion's `parameter`: the slopes, in
        the standardized space for method "ridge" with scaled = True (`slopes` are original units).
    fname : list of str
        The column names of X (ScalaTion's `getFname`).
    modelName : str
        E.g., "ARX_D_6_6_2" (p, px, n_exo), as ScalaTion names its models.
    Yf : np.ndarray, shape (m, hh + 2)
        The FORECAST MATRIX in ORIGINAL units: column 0 the actual series, columns 1..hh the
        forecasts -- Yf[t, h] is the h-steps ahead forecast of y[t + h - 1], made from the
        information up to t - 1 -- and the last column the time index.
    yForm : Transform, optional
        The transform of y (None unless made by `rescale` or by a formula with a TFORM); forecasts
        are inverted before being stored in Yf.
    tr_size, te_size : int
        The sizes of the initial training set and of the testing set (ScalaTion's `Model.trSize`/
        `teSize`, 80/20 by default).

    Notes
    -----
    The actual implementation uses leading underscores for internal attributes (_y, _X, _params).
    `train` must be called before `forecast`.
    """

    TE_RATIO = 0.2
    DO_REPORT = True                                               # print the REPORT in trainNtest
    SHOW_QoF = True                                                # print the QoF table in diagnoseAll

    def __init__(self, hparam: Dict[str, Any], y: np.ndarray, hh: int, hasIntercept: bool = True) -> None:
        """
        Initializes the Forecaster.

        Parameters
        ----------
        hparam : dict or HyperParameter, optional
            The hyper-parameters (see the class attributes); missing ones take the defaults of `hp`
            (ScalaTion's `AR.hp`: p 1, px 1, spec 1, crx 0, nneg 1, ...) and skip = p.
        y : np.ndarray, shape (m,)
            The response vector (endogenous time series).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        hasIntercept : bool, default=True
            Whether the model has an intercept (counted in the degrees of freedom).
        """
        self.hparam: HyperParameter = hp.copy()
        self.hparam.update(hparam or {})
        for k in ("crx", "nneg"):                                      # ScalaTion keeps the switches as 0 / 1
            if isinstance(self.hparam[k], (bool, np.bool_)):
                self.hparam[k] = int(self.hparam[k])
        self.hparam.setdefault("skip", self.hparam["p"])
        super().__init__(dfr=1.0, df=max(1.0, len(y) - 1.0), hasIntercept=hasIntercept, skip=self.hparam["skip"])
        self.hh = hh
        self._y = np.asarray(y, dtype=float)
        self._X: Optional[np.ndarray] = None
        self._params: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None                          # ScalaTion's b / bb (fitted space)
        self._method: Optional[str] = None
        self._scaled = False
        self._nneg = self.hparam["nneg"] == 1
        self._tr_size = trSize(len(y), self.TE_RATIO)
        self._te_size = len(y) - self._tr_size
        self._Yf = self.makeForecastMatrix()
        self._yForm = None
        self._modelName: Optional[str] = None
        self._fname: Optional[List[str]] = None

    # -- names ----------------------------------------------------------------------------

    @property
    def modelName(self) -> str:
        """The model name, e.g., "ARX_D_6_6_2" (set by the subclasses; the class name otherwise)."""
        return self._modelName if self._modelName is not None else self.__class__.__name__

    @property
    def hparameter(self) -> Optional[HyperParameter]:
        """The hyper-parameters (ScalaTion's `hparameter`)."""
        return self.hparam

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
        """The transform of y (None unless made by `rescale` or by a formula with a TFORM)."""
        return self._yForm

    def _setYForm(self, tForm_y: Any, y_org: Optional[np.ndarray] = None) -> None:
        """
        Records the transform of y (the model works on the transformed y): column 0 of `Yf` holds
        the series in ORIGINAL units (`y_org`, or the inverse of the model's y), and a transform into
        a space where y can be negative turns the non-negativity off (ScalaTion's `centered`).
        """
        self._yForm = tForm_y
        if tForm_y is None:
            return
        y0 = tForm_y.inverse_transform(self._y) if y_org is None else np.asarray(y_org, dtype=float)
        self._Yf[:, 0] = np.asarray(y0, dtype=float).ravel()
        if centered(tForm_y):
            self._nneg = False

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

    def makeForecastMatrix(self) -> np.ndarray:
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
        The intercept and the slopes stacked, intercept first: (n + 1,) or (n + 1, hh) -- the Python
        storage layout.  Prefer ScalaTion's names: `parameter` (the slopes b), `intercept`, and `bb`
        (the slope matrix of a direct model).

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
        """
        The coefficients b (horizon 1 for a direct model), as ScalaTion's `parameter`: the regression's
        own coefficients (in the standardized space for method "ridge" with scaled = True), the
        slopes otherwise.
        """
        if self._b is not None:
            return self._b[:, 0] if self._b.ndim == 2 else self._b
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

    # -- the regression (ScalaTion's `reg` of `Forecaster_Reg` / `Forecaster_D`) ---------------

    def _make_model(self):
        """
        The regression object of `method` (ScalaTion's `reg`): None for "sm_ols" (statsmodels, fitted
        per call), scikit-learn's `LinearRegression` for "sk_lr", and `CenteredRidge` (penalty
        hparam["lambda"], default `CenteredRidge.hp`; `scaled`) for "ridge".
        """
        if self._method == "sk_lr":
            return LinearRegression()
        if self._method == "ridge":
            return CenteredRidge(CenteredRidge.ridge(self.hparam), self._scaled)
        if self._method == "sm_ols":
            return None
        raise ValueError(f"method '{self._method}' unknown: sm_ols | sk_lr | ridge")

    @property
    def method(self) -> Optional[str]:
        """The regression method ("sm_ols", "sk_lr" or "ridge"; None for a model without one)."""
        return self._method

    @property
    def scaled(self) -> bool:
        """Whether the ridge regression standardizes the columns (ScalaTion's `scaled`)."""
        return self._scaled

    def _subModel(self, sub: "Forecaster", cols: Sequence[int], modelName: Optional[str] = None) -> "Forecaster":
        """
        Finishes a sub-model built on the columns `cols` of this model's input matrix (the model
        classes construct `sub` on `X[:, cols]` with the ORIGINAL y, then call this -- their
        `getModel` / `getModel_h`): this model's response in model units (scaled when it was made by
        `rescale`), its target matrix, transform of y, non-negativity, training/testing sizes, skip
        and the feature names of the columns, so the sub-model forecasts in the same units and on
        the same rows; a sub-model has no lag structure to forge, so its `_useforge` is False, and its
        skip is its own default (hparam["skip"]) -- as a new ScalaTion model's, not this model's current
        skip (a `tnT_Test` sets that to 0).

        Parameters
        ----------
        sub : Forecaster
            The freshly constructed sub-model (X given, so no lag matrix is rebuilt).
        cols : sequence of int
            The selected columns.
        modelName : str, optional
            The sub-model's name (defaults to this model's).

        Returns
        -------
        Forecaster
            `sub`, finished.
        """
        cols = list(cols)
        sub._y = self._y                                                  # model units
        if getattr(self, "_Y", None) is not None and hasattr(sub, "_Y"):
            sub._Y = self._Y                                              # the direct targets (same units)
        sub._yForm = self._yForm
        if hasattr(self, "_tForms"):
            sub._tForms = dict(self._tForms)
        sub._nneg = self._nneg
        sub._tr_size, sub._te_size = self._tr_size, self._te_size
        fn = self.fname
        sub._fname = [fn[j] if j < len(fn) else f"x{j}" for j in cols]
        if hasattr(self, "_n_exo"):
            sub._n_exo = self._n_exo
        sub._modelName = self.modelName if modelName is None else modelName
        sub._useforge = False                                             # the skip is the sub-model's own default
        return sub                                                        # (a new ScalaTion model), not this one's current

    def __repr__(self) -> str:
        """A string showing the model name, horizon and key hyper-parameters."""
        s = f"{self.modelName}(p={self.hparam['p']}"
        if hasattr(self, "_n_exo"):
            s += f", px={self.hparam['px']}, n_exo={self._n_exo}"
        s += f"), hh = {self.hh}, skip = {self.hparam['skip']}"
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

    def forecastAt(self, h: int) -> np.ndarray:
        """
        The h-steps ahead forecasts of all the rows (ScalaTion's `forecastAt`): column h of the
        forecast matrix (original units) after forecasting every row.

        Parameters
        ----------
        h : int
            The horizon (1 to hh).
        """
        if not 1 <= h <= self.hh:
            raise ValueError(f"forecastAt: horizon h = {h} out of range [1, {self.hh}]")
        self.forecast()
        return self._Yf[:, h]

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
        `Model.report`): model name, hyper-parameters, feature names, parameters and the fit map of
        horizon 1.
        """
        b, fn = self.parameter, self.fname
        print(f"b = {vector_str(b)}, fn = {array_str(fn)}")
        if len(fn) != len(b):
            print(f"report: fn: # features = {len(fn)} != # parameters = {len(b)}")
        rule = "    " + "-" * 76
        hpr = "null" if self.hparameter is None else repr(self.hparameter)
        return (f"\nREPORT\n{rule}\n    modelName  mn  = {self.modelName}\n{rule}\n"
                f"    hparameter hp  = {hpr}\n{rule}\n"
                f"    features   fn  = {array_str(fn)}\n{rule}\n"
                f"    parameter  b   = {vector_str(b)}\n{rule}\n"
                f"    fitMap     qof = {fitMap(qof)}\n{rule}\n        ")

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
