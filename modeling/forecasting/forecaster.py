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

Every model is built as `Model (y, hh, [xe,] args)`: the data in the call, the settings in an `Args`
(modeling/forecasting/args.py; None => the defaults; a dict of settings is checked the same way).
The model keeps a private snapshot of its args and its hyper-parameters, `AR.hp` (`hp`, ScalaTion's
p, d, q, P, D, Q, s, px, spec, crx, nneg with their defaults, read-only) with the values of args, as
a `HyperParameter` (`hparameter` returns a copy), which the REPORT prints as ScalaTion does
("HyperParameter(p -> (6,1), ...)").  The regression the forecasters delegate to (`_make_model`) is
args.method "sm_ols" (statsmodels), "sk_lr" (scikit-learn) or "ridge" -- ScalaTion's
`CenteredRidge(MV)`, with penalty args.lam (hparameter "lambda", default `CenteredRidge.hp`, 0.1) and
args.scaled, or "enet" (`ElasticNet`, args.lam and args.l1_ratio).  The KNOBS (rev2.2, ScalaTion
rev5.3): the regression declares its tunable hyper-parameters (`CenteredRidge.tunable` = [lambda],
`ElasticNet.tunable` = [lambda, l1_ratio], `Knob` objects), `knobs` reads them, `tune` turns them
(grid, golden-section, random), `withHP` copies a model at other knob values, and `hpTag` writes them
into the name: "ARX_D_6_6_2_L1.0" under the ridge, "ARX_D_6_6_2_L1.0_A0.5" under the elastic net; a
model without knobs (a plain least-squares fit, AR, ARIMA, RandomWalk) has no tag.

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 2.2
License: MIT License
"""

from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from Util.make_matrix4ts import centered
from Util.tools import array_str, banner, matrix_str, vector_str
from modeling.centered_ridge import CenteredRidge
from modeling.elastic_net import ElasticNet
from modeling.fit import N_QoF, QoF, fitMap, mae_n, showFitMap
from modeling.forecasting.args import ALIASES, PENALIZED_METHODS, REGRESSION_METHODS, Args, hp
from modeling.forecasting.diagnoser import Diagnoser
from modeling.hyper_parameter import HyperParameter, Knob
from modeling.model import Model

#: ScalaTion's `AR.hp` (`AR.hp` is this object): the hyper-parameters every forecaster starts from,
#: READ-ONLY, built from the settings table of `Args` (args.py):
#:     p 1 (AR lags of y), d 1 (differences), q 1 (MA), P 1, D 1, Q 1 (seasonal), s 7 (season),
#:     px 1 (lags of each exogenous variable), spec 1 (trend terms), crx 0 (ENDO-EXO cross terms),
#:     nneg 1 (non-negative forecasts).
#: `REGRESSION_METHODS` ("sm_ols", "sk_lr", "ridge") comes from there too; the split is `Model`'s.
__all__ = ["Forecaster", "hp", "REGRESSION_METHODS", "PENALIZED_METHODS", "Args", "Model", "teSize", "trSize", "teRng"]

#: The split functions are `Model`'s (modeling/model.py); these names stay for the callers that import them here.
teSize = Model.teSize
trSize = Model.trSize
teRng = Model.teRng


class Forecaster(Diagnoser, ABC):
    """
    Abstract base class for time series forecasting models.

    Attributes
    ----------
    hparameter : HyperParameter
        A copy of the model's hyper-parameters (ScalaTion's `hparameter`): `hp` (ScalaTion's `AR.hp`:
        "p" lags of y, "px" lags of each exogenous variable, "spec" trend terms -- 0 implicit constant,
        1 linear, 2 quadratic, 3 cubic --, "crx" ENDO-EXO cross terms, "nneg" non-negative forecasts,
        ...) with the values of the model's args, plus the Python-only "skip" (rows skipped in the
        In-Sample diagnosis, args.skip, by default p) and "lambda" (args.lam, the ridge penalty).
    ARGS : tuple of str (class attribute)
        The names of the `Args` settings the model reads (`Args.help (Model)` lists them).
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
    The actual implementation uses leading underscores for internal attributes (_y, _X, _params,
    _args, _hparam).  `train` must be called before `forecast`.
    """

    RC = 5                                                         # the retraining cycle of the rolling tests (ScalaTion's `Forecaster.RC`)
    DO_REPORT = True                                               # print the REPORT in trainNtest
    SHOW_QoF = True                                                # print the QoF table in diagnoseAll
    ARGS: Tuple[str, ...] = ("skip", "nneg", "te_ratio")           # the settings every forecaster reads
    METHOD: Optional[str] = None                                   # the default of args.method (None: no method)
    METHODS: Tuple[str, ...] = ()                                  # the methods the model accepts

    def __init__(self, y: np.ndarray, hh: int, args: Any = None, hasIntercept: bool = True) -> None:
        """
        Initializes the Forecaster.

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The response vector (endogenous time series).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        args : Args or dict, optional
            The settings (see `Args`; None => the defaults: `hp`, ScalaTion's `AR.hp`: p 1, px 1,
            spec 1, crx 0, nneg 1, ..., and skip = p); the model keeps a private snapshot.
        hasIntercept : bool, default=True
            Whether the model has an intercept (counted in the degrees of freedom).

        Raises
        ------
        TypeError
            If y is a dict or an Args (the order before rev1.4, hparam first).
        """
        self._args: Args = self._argsOf(y, args)                       # the private snapshot
        self._hparam: HyperParameter = self._args.hparam()
        self._hparam.setdefault("skip", self._hparam["p"])
        super().__init__(dfr=1.0, df=max(1.0, len(y) - 1.0), hasIntercept=hasIntercept, skip=self._hparam["skip"])
        self.hh = hh
        self._y = np.asarray(y, dtype=float)
        self._X: Optional[np.ndarray] = None
        self._params: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None                          # ScalaTion's b / bb (fitted space)
        self._method: Optional[str] = self._methodOf(self._args)
        self._scaled = False
        self._nneg = self._hparam["nneg"] == 1
        self._te_ratio = Model.TE_RATIO if self._args.te_ratio is None else self._args.te_ratio   # the split of THIS model
        self._tr_size = trSize(len(y), self._te_ratio)
        self._te_size = len(y) - self._tr_size
        self._lead = 0                                                # the offset of the targets: h - 1 for the single-horizon
                                                                      # model at h of a direct model (`getModel_h`), 0 for a series
        self._Yf = self.makeForecastMatrix()
        self._yForm = None
        self._recipe: Optional[Callable[[], "Forecaster"]] = None     # rebuilds the model by its factory (`refitTransform`)
        self._modelName: Optional[str] = None
        self._fname: Optional[List[str]] = None

    # -- names ----------------------------------------------------------------------------

    @property
    def modelName(self) -> str:
        """The model name, e.g., "ARX_D_6_6_2_L1.0" (set by the subclasses; the class name otherwise)."""
        return self._modelName if self._modelName is not None else self.__class__.__name__

    @property
    def lam(self) -> Optional[float]:
        """The penalty lambda of a regression-based model: hparameter "lambda" under a penalized method
        ("ridge", "enet"), 0.0 under the plain regressions ("sm_ols", "sk_lr": ordinary least squares
        is the ridge at lambda = 0); None for a model without a regression (AR, ARIMA, RandomWalk)."""
        if self._method is None:
            return None
        if self._method in PENALIZED_METHODS and "lambda" in self._hparam:
            return float(self._hparam["lambda"])
        return 0.0

    # -- the knobs (ScalaTion rev5.3) --------------------------------------------------------

    @property
    def knobs(self) -> Tuple[Knob, ...]:
        """The tunable hyper-parameters of this model's regression (ScalaTion's `REGRESSION.tunable`):
        `CenteredRidge.tunable` (lambda) under method "ridge", `ElasticNet.tunable` (lambda, l1_ratio)
        under "enet"; none for a plain regression or a model without one."""
        if self._method == "ridge":
            return tuple(CenteredRidge.tunable)
        if self._method == "enet":
            return tuple(ElasticNet.tunable)
        return ()

    @property
    def penalized(self) -> bool:
        """Whether the regression has knobs to tune (method "ridge" or "enet")."""
        return len(self.knobs) > 0

    @property
    def hpAll(self) -> Dict[str, Any]:
        """The hyper-parameters the regression reads (ScalaTion's merged `hparam`): the model's
        hyper-parameters plus the knobs that live in its args only (l1_ratio)."""
        h: Dict[str, Any] = dict(self._hparam)
        for k in self.knobs:
            if k.name not in h:
                h[k.name] = self._args[ALIASES.get(k.name, k.name)]
        return h

    @property
    def hpTag(self) -> str:
        """The tag the knobs write into a model's name: "_L1.0" for a ridge at lambda = 1, "_L1.0_A0.5" for
        an elastic net at l1_ratio = 0.5 (`Knob.tagOf`); "" without knobs."""
        h = self.hpAll
        return "".join(k.tagOf(float(h[k.name])) for k in self.knobs)

    def stdErrOf(self, X_fit: np.ndarray, resid: np.ndarray, k: int = 0) -> np.ndarray:
        """
        The standard errors, intercept first, of the regression last fitted on the rows `X_fit`
        (no constant column) with residuals `resid`: a ridge's own sandwich covariance (`CenteredRidge.stdErr`,
        response k), else the least-squares formula sqrt (diag ((X1'X1)^-1) SSE / (m - n - 1)) on
        X1 = [1, X_fit] -- what `summary` prints and what the t-statistic ranker divides by.
        """
        if self._ridgeSE():
            return self._model.stdErr(k)
        X1 = np.column_stack((np.ones(X_fit.shape[0]), X_fit))
        m, n1 = X1.shape
        sse = float(np.sum(np.asarray(resid, dtype=float) ** 2))
        try:
            _, R = np.linalg.qr(X1)
            Ri = np.linalg.inv(R)
            diag = np.sum(Ri * Ri, axis=1)
        except np.linalg.LinAlgError:
            diag = np.full(n1, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(diag * sse / max(m - n1, 1))

    def _ridgeSE(self) -> bool:
        """Whether the SUMMARY can take its standard errors from the regression's own covariance (a
        trained `CenteredRidge`: the sandwich form; an elastic net has none, least squares uses (X'X)^-1)."""
        return self._params is not None and hasattr(self._model, "stdErr")

    @property
    def effDF(self) -> Optional[np.ndarray]:
        """
        The effective number of parameters of the fitted regression, one value per response column
        (a ridge: 1 + tr H_lambda, the same for every horizon; an elastic net: 1 + the nonzero
        coefficients of each horizon) -- what `scoreOf` charges in its FPE factor (`nParams`) and
        `summary` uses for the residual degrees of freedom; None for a plain regression or before
        training, where the QoF's dfr + 1 is the count.
        """
        if not self.penalized or self._params is None:
            return None
        return np.atleast_1d(np.asarray(self._model.effDF, dtype=float))

    @property
    def hparameter(self) -> Optional[HyperParameter]:
        """A COPY of the model's hyper-parameters (ScalaTion's `hparameter`): read it, print it (the
        REPORT's form) or reuse its values; changing the copy does not change the model."""
        return self._hparam.copy()

    @classmethod
    def _argsOf(cls, y: Any, args: Any) -> Args:
        """
        The settings of the model being built: a copy of args (None => the defaults; a dict is checked
        as an Args), after checking that the data come first (the order before rev1.4 put hparam first).

        Raises
        ------
        TypeError
            If y is a dict or an Args.
        """
        if isinstance(y, (Mapping, Args)):
            name = cls.__name__
            raise TypeError(f"{name}: the data come first, the settings last: {name} (y, hh, ..., args) (since rev1.4)")
        return Args.of(args)

    def _methodOf(self, args: Args) -> Optional[str]:
        """The estimation method of the model: args.method, or the model's default `METHOD`, checked
        against `METHODS` (None for a model without a method, or for the library's default)."""
        if not self.METHODS:
            return None
        m = self.METHOD if args.method is None else args.method
        if m is not None and m not in self.METHODS:
            raise ValueError(f"{type(self).__name__}: method '{m}' is not one of {' | '.join(self.METHODS)}")
        return m

    @property
    def te_ratio(self) -> float:
        """The fraction of rows in the testing set of THIS model (args.te_ratio; `Model.TE_RATIO` when the
        model was built if None), ScalaTion's `Model.TE_RATIO`."""
        return self._te_ratio

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
    def mSeries(self) -> int:
        """The length of the series the split is computed on: len (y) + `lead` -- the single-horizon model at
        horizon h of a direct model (`getModel_h`) holds the m - h + 1 rows whose target y[r + h - 1] exists
        (ScalaTion keeps them as NO_DOUBLE rows), so its blocks are its parent's, cut at its last row."""
        return len(self._y) + self._lead

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
        args.lam, hparameter "lambda", default `CenteredRidge.hp`; args.scaled) for "ridge".
        """
        if self._method == "sk_lr":
            return LinearRegression()
        if self._method == "ridge":
            return CenteredRidge(CenteredRidge.ridge(self._hparam), self._scaled)
        if self._method == "enet":
            return ElasticNet(CenteredRidge.ridge(self._hparam), self._args.l1_ratio, self._scaled)
        if self._method == "sm_ols":
            return None
        raise ValueError(f"method '{self._method}' unknown: {' | '.join(REGRESSION_METHODS)}")

    @property
    def method(self) -> Optional[str]:
        """The regression method ("sm_ols", "sk_lr", "ridge" or "enet"; None for a model without one)."""
        return self._method

    @property
    def scaled(self) -> bool:
        """Whether the ridge regression standardizes the columns (ScalaTion's `scaled`)."""
        return self._scaled

    def _subArgs(self) -> Args:
        """The settings of a sub-model (`getModel`, `getModel_h`): a copy of this model's private args with
        its resolved method, so the sub-model fits as this model does."""
        return self._args.copy(method=self._method) if self.METHODS else self._args.copy()

    def _subModel(self, sub: "Forecaster", cols: Sequence[int], modelName: Optional[str] = None) -> "Forecaster":
        """
        Finishes a sub-model built on the columns `cols` of this model's input matrix (the model
        classes construct `sub` on `X[:, cols]` with the ORIGINAL y, then call this -- their
        `getModel` / `getModel_h`): this model's response in model units (scaled when it was made by
        `rescale`), its target matrix, transform of y, non-negativity, training/testing sizes, skip
        and the feature names of the columns, so the sub-model forecasts in the same units and on
        the same rows (the model classes build `sub` with this model's private args, so its
        hyper-parameters are this model's); a sub-model has no lag structure to forge, so its
        `_useforge` is False, and its skip is its own default (hparameter "skip") -- as a new ScalaTion
        model's, not this model's current skip (a `tnT_Test` sets that to 0).

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
        sub._te_ratio, sub._lead = self._te_ratio, self._lead             # the same split and boundary rule
        sub._tr_size, sub._te_size = self._tr_size, self._te_size
        fn = self.fname
        sub._fname = [fn[j] if j < len(fn) else f"x{j}" for j in cols]
        if hasattr(self, "_n_exo"):
            sub._n_exo = self._n_exo
        sub._modelName = self.modelName if modelName is None else modelName
        sub._useforge = False                                             # the skip is the sub-model's own default
        hp2 = self.bestHP(cols)                                           # (a new ScalaTion model), not this one's current
        if hp2 is not None and sub.penalized:
            sub._setHP(hp2)                                               # the best step's knobs for its own columns
        return sub

    def bestHP(self, cols: Sequence[int]) -> Optional[Dict[str, float]]:
        """The knobs the best step of this model's last feature selection tuned for the column set
        `cols` (hook: `SelectionReport` provides it; a model without a selection has none), so
        `getModel (getBest ().mod_cols)` and `selectedModel` carry them (ScalaTion's `bestHP`)."""
        return None

    def __repr__(self) -> str:
        """A string showing the model name, horizon and key hyper-parameters."""
        s = f"{self.modelName}(p={self._hparam['p']}"
        if hasattr(self, "_n_exo"):
            s += f", px={self._hparam['px']}, n_exo={self._n_exo}"
        s += f"), hh = {self.hh}, skip = {self._hparam['skip']}"
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

    # -- the selection score (ScalaTion's `Forecaster.scoreOf`) -----------------------------------

    ERROR_MEASURES = (QoF.sse, QoF.sde, QoF.rse, QoF.mse0, QoF.rmse, QoF.mae, QoF.smape, QoF.bmae, QoF.mape,
                      QoF.mase, QoF.smapeC)                               # smaller is better: the measures a chooser may score by

    @staticmethod
    def scoreOf(qof: np.ndarray, inSample: bool, extra: int = 0, measure: QoF = QoF.smape,
                hs: Optional[Sequence[int]] = None, nParams: Optional[Any] = None) -> float:
        """
        The ONE number every chooser compares (ScalaTion's `Forecaster.scoreOf`): the mean over the
        horizons `hs` of the error `measure` (sMAPE by default) of a QoF matrix (one row per
        horizon), multiplied, when the rows were FITTED (`inSample`), by the FPE factor
        (1 + 2 p_h / n_h) with p_h = the parameters of horizon h + `extra` (parameters held fixed)
        and n_h the rows diagnosed; rows that were never fitted (the validation rows) carry no
        penalty.  The parameters of horizon h are `nParams` when given -- the EFFECTIVE number of a
        penalized fit (`Forecaster.effDF`: 1 + tr H_lambda for a ridge, which falls as lambda grows,
        so more shrinkage is charged less, the way GCV and the ridge AIC count) -- else dfr_h + 1 of
        the QoF row (every column counts one: right for least squares, an overstatement for a ridge).
        inf when undefined.

        Parameters
        ----------
        qof : np.ndarray, shape (hh, N_QoF)
            The QoF matrix, one row per horizon (`diagnoseAll`, `rollTest`).
        inSample : bool
            Whether the rows diagnosed were fitted (penalised) or held out (not).
        extra : int, default=0
            Parameters fitted elsewhere and held fixed (a library's frozen nonlinear parameters).
        measure : QoF, default=QoF.smape
            The error measure averaged, one of `ERROR_MEASURES` (smaller is better; a goodness measure
            such as rSq would need the penalty inverted, so it is refused).
        hs : sequence of int, optional
            The horizons averaged, 1-based (None => all the rows of `qof`; [1] => the first horizon).
        nParams : float or sequence of float, optional
            The parameters charged per horizon (one value, or one per row of `qof`); None => dfr + 1.

        Raises
        ------
        ValueError
            If `measure` is not an error measure.
        """
        if measure not in Forecaster.ERROR_MEASURES:
            raise ValueError(f"scoreOf: measure {measure.name} is not an error measure (one of "
                             f"{[q.name for q in Forecaster.ERROR_MEASURES]})")
        q = np.atleast_2d(np.asarray(qof, dtype=float))
        p = q[:, QoF.dfr.ordinal] + 1.0 if nParams is None else np.broadcast_to(np.asarray(nParams, dtype=float).ravel(), (q.shape[0],))
        if hs is not None:
            rows = [int(h) - 1 for h in hs]
            q, p = q[rows], p[rows]
        s = q[:, measure.ordinal]
        if inSample:
            s = s * (1.0 + 2.0 * (p + extra) / q[:, QoF.m.ordinal])
        v = float(np.mean(s))
        return v if np.isfinite(v) else float("inf")

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

    def rollValidate(self, rc: Optional[int] = None, growing: bool = False, rng: Optional[range] = None,
                     verbose: bool = True) -> np.ndarray:
        """
        Rolling validation (ScalaTion's `rollValidate`): the data set is split into a TRAINING SET
        (tr) and a TESTING SET (te) [ <-- tr_size --> | <-- te_size --> ]; the model is retrained
        every `rc` rows on the sliding (or growing) training window and forecasts h = 1..hh
        steps ahead for each row of the block.  The forecasts are recorded in `Yf`.  The block is
        the testing rows by default; `rollTest` passes the validation rows.

        Parameters
        ----------
        rc : int, optional
            The retraining cycle (number of forecasts until retraining occurs; None => `RC`).
        growing : bool, default=False
            Whether the training window grows or keeps a fixed size.
        rng : range, optional
            The rows rolled over (the testing rows by default).
        verbose : bool, default=True
            Whether to print the banner.

        Returns
        -------
        np.ndarray, shape (len(rng), hh)
            The block of forecasts for the rows (model units); `Yf` holds them in original units.
        """
        rc = self.RC if rc is None else rc
        rng = teRng(self.mSeries, self._te_ratio) if rng is None else rng
        rng = range(rng.start, min(rng.stop, len(self._y)))              # a horizon-h model: its last rows have no target
        if verbose:
            banner(f"rollValidate: Evaluate {self.modelName}'s QoF for horizons 1 to {self.hh}:")
        self._reset_Yf()
        n = rng.stop - rng.start
        y_fcast = np.zeros((n, self.hh))
        for i in range(0, n, rc):
            is_ = 0 if growing else i
            t = rng.start + i
            self.train(*self._trainWindow(is_, t))                      # retrain on the sliding window (targets before t)
            t_end = min(t + rc, rng.stop)
            y_fcast[i:i + (t_end - t), :] = self.forecast(t, t_end)
        return y_fcast

    def _trainWindow(self, is_: int, t: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        The training window before the forecast origin t (ScalaTion's `trainWindow` hook): the target
        rows and the input rows [is_, t - lead), so NO training target lies at or after t -- the
        boundary rule.  `lead` is 0 for a series (the target of row r is y[r]); h - 1 for the
        single-horizon model at horizon h of a direct model (`getModel_h`: the target of row r is
        y[r + h - 1]); a direct model's window stops hh - 1 rows early (`Forecaster_D`).
        """
        te = t - self._lead
        if te <= is_:
            raise ValueError(f"_trainWindow: the window [{is_}, {te}) is empty (t = {t}, lead = {self._lead})")
        return self._y[is_:te], (self._X[is_:te] if self._X is not None else None)

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

    def rollTest(self, rng: range, rc: Optional[int] = None, label: str = "TnT", verbose: bool = True,
                 showYf: bool = False, showParams: bool = False) -> np.ndarray:
        """
        ONE rolling pass over a block of rows (ScalaTion's `Forecaster.rollTest`), the routine behind
        `valTest` (the validation rows) and `tnT_Test` (the testing rows): the model is trained on
        the rows before the block (the REPORT, when verbose), then rolls through the block retraining
        every `rc` rows on the sliding window (`rollValidate`) and is diagnosed on the block's rows
        only (no row skipped; the skip is saved and RESTORED, so a selection run afterwards does not
        score the burn-in rows).  The block's end bounds the diagnosis, so a validation pass never
        reads a testing row.

        Parameters
        ----------
        rng : range
            The rows rolled over and diagnosed (`Model.valRng` or `Model.teRng`).
        rc : int, optional
            The retraining cycle (None => `Forecaster.RC`).
        label : str, default="TnT"
            The banners' label ("Validation" for the validation rows).
        verbose : bool, default=True
            Whether to print the banners, the REPORT and the QoF table.
        showYf : bool, default=False
            Whether to print the forecast matrix (verbose only).
        showParams : bool, default=False
            Whether to print the full parameter array of the last retraining (verbose only).

        Returns
        -------
        np.ndarray, shape (hh, N_QoF)
            One QoF vector per horizon on the block's rows.
        """
        rc = self.RC if rc is None else rc
        m = len(self._y)
        rng = range(rng.start, min(rng.stop, m))                          # a horizon-h model: its last rows have no target
        if len(rng) == 0:                                                 # ScalaTion: a flaw, null
            print(f"flaw @ rollTest: {label}: the block {rng} is empty (no testing set when te_ratio = 0)")
            return None
        if not 0 < rng.start < rng.stop <= m:
            raise ValueError(f"rollTest: the block {rng} must lie inside 1 until {m}")
        skip0 = self.getSkip()
        if verbose:
            banner(f"{label} Test: {self.modelName}")
            self.trainNtest(*self._trainWindow(0, rng.start))           # the model that makes the first forecasts
            self.forecastAll()
        else:
            self.train(*self._trainWindow(0, rng.start))
        self.setSkip(0)
        try:
            self.rollValidate(rc, rng=rng, verbose=verbose)              # the sliding training windows
            if verbose:
                banner(f"diagnoseAll: {self.modelName}")
            show = self.SHOW_QoF
            if not verbose:
                Forecaster.SHOW_QoF = False
            try:
                ftMat = self.diagnoseAll(self._Yf[:rng.stop, 0], self._Yf[:rng.stop], tRng=rng)
            finally:
                Forecaster.SHOW_QoF = show
        finally:
            self.setSkip(skip0)                                           # restore the skip
        if verbose and showYf:
            print(f"Final {label} Forecast Matrix yf = {matrix_str(self._Yf)}")
        if verbose and showParams:
            print(f"params = \n{self.params}")
        return ftMat

    def errorTable(self, rng: range, scale: Optional[float] = None) -> pd.DataFrame:
        """
        The per-origin, per-horizon errors of the forecast matrix over a block, after `rollTest (rng)`
        (rev2.2, the measurement protocol): one row per origin t in the block and horizon h whose
        target y[t + h - 1] lies in the block (the pairs `qofOn` / `diagnoseAll` diagnose), in ORIGINAL
        units -- t, h, y, yhat, e = y - yhat, ae = |e|, sape = 200 |e| / (|y| + |yhat|) (the term of the
        sMAPE, so a horizon's mean sape IS its sMAPE) and ase = |e| / scale, the scaled error whose mean
        is the MASE, with scale the MAE of the one-step naive forecast on the rows before the block
        (`fit.mae_n`) unless given.  The long table the protocol's paired tests and summary tables read
        (`modeling/forecasting/protocol.py`).

        Parameters
        ----------
        rng : range
            The block the forecasts were rolled over (`Model.teRng` or `Model.valRng`).
        scale : float, optional
            MASE's denominator (None => the naive MAE of the rows before the block, original units).
        """
        y = self._Yf[:, 0]
        m = len(y)
        end = min(rng.stop, m - self._lead)
        if scale is None:
            scale = mae_n(y[:rng.start], 1) if rng.start > 1 else float("nan")
        rows = []
        for h in range(1, self.hh + 1):
            for t in range(rng.start, end - h + 1):
                yt, yp = float(y[t + h - 1]), float(self._Yf[t, h])
                e = yt - yp
                den = abs(yt) + abs(yp)
                rows.append((t, h, yt, yp, e, abs(e), 200.0 * abs(e) / den if den > 0.0 else 0.0,
                             abs(e) / scale if scale and scale > 0.0 else float("nan")))
        return pd.DataFrame(rows, columns=["t", "h", "y", "yhat", "e", "ae", "sape", "ase"])

    def valTest(self, rc: Optional[int] = None, showYf: bool = False, showParams: bool = False,
                verbose: bool = True) -> np.ndarray:
        """
        The validation pass (ScalaTion's `valTest`): `rollTest` over the validation rows `Model.valRng`,
        the model trained on the fit rows and refit every `rc` rows; what a chooser sees under
        `Model.validating`.  (ScalaTion's `Forecaster.validate (rando, ratio)` is Dr. Miller's one-step
        `Predictor.validate`, a random or trailing train-and-test split at h = 1, not ported.)

        Returns
        -------
        np.ndarray, shape (hh, N_QoF)
            One QoF vector per horizon on the validation rows (None when the block is empty).
        """
        return self.rollTest(Model.valRng(self.mSeries, self._te_ratio), rc, "Validation", verbose, showYf, showParams)

    def valScore(self, rc: Optional[int] = None, measure: QoF = QoF.smape, hs: Optional[Sequence[int]] = None) -> float:
        """The validation score: `scoreOf` (the error `measure` averaged over the horizons `hs` on the
        validation rows, no penalty) of a quiet `valTest`; the number `tune` tunes a
        forecaster by (NaN when there is no validation block)."""
        qv = self.valTest(rc, verbose=False)
        return float("nan") if qv is None else self.scoreOf(qv, inSample=False, measure=measure, hs=hs)

    def _setHP(self, hp2: Mapping[str, Any]) -> None:
        """Sets the knobs of an UNTRAINED model in place: its args (a knob's `Args` name, `lam` for
        "lambda") and hparameter, a new regression, and the name re-tagged (`hpTag` is the last part of
        every regression-based model's name)."""
        old = self.hpTag
        changes = {ALIASES.get(k, k): float(v) for k, v in hp2.items() if k in {kn.name for kn in self.knobs}}
        if not changes:
            return
        self._args = self._args.copy(**changes)
        self._hparam = self._hparam.copy()
        if "lambda" in hp2:
            self._hparam["lambda"] = float(hp2["lambda"])
        self._model = self._make_model()                                  # the regression at the new knobs
        name = self.modelName
        base = name[: len(name) - len(old)] if old and name.endswith(old) else name
        self._modelName = base + self.hpTag

    def withHP(self, hp2: Mapping[str, Any]) -> "Forecaster":
        """
        A copy of this model at other hyper-parameters (ScalaTion's `withHP`; rev2.2: was
        `withLambda (lam)`): the same rows, matrix and transforms, a new regression at the knobs in
        `hp2` (untrained), its name re-tagged.

        Parameters
        ----------
        hp2 : mapping
            The knobs by name ("lambda", "l1_ratio"; what `tune` returns); other keys are ignored.

        Raises
        ------
        ValueError
            If the model's regression has no knobs (not "ridge" or "enet").
        """
        if not self.penalized:
            raise ValueError(f"withHP: {self.modelName} is fitted by {self._method}, which has no knobs to turn")
        cp = copy.copy(self)
        cp._setHP(hp2)
        cp._params, cp._b = None, None
        cp._Yf = self._Yf.copy()
        cp._Yf[:, 1:-1] = 0.0
        if self._recipe is not None:                                      # a refit of the transforms keeps the knobs
            recipe = self._recipe
            hp3 = dict(hp2)
            cp._recipe = lambda: recipe().withHP(hp3)
        return cp

    def tune(self, knobs: Optional[Sequence[Knob]] = None, measure: QoF = QoF.smape,
             hs: Optional[Sequence[int]] = None, verbose: bool = False, search: str = "grid",
             folds: Optional[int] = None, trials: int = 0, seed: int = 0) -> Tuple[Dict[str, float], float]:
        """
        Tunes the knobs of this model's regression (ScalaTion's `Forecaster.tune`; rev2.2: was
        `findLambda`): the knob values with the best held-out score and that score, ({name: value},
        score).  The score is `valScore` -- the rolling validation: trained on the fit rows, rolled
        over the validation rows, `measure` averaged over the horizons `hs` -- or, with `folds = k`,
        `cvScore (k)`: k-fold cross-validation on the embedded rows of the regression (valid for a
        lag-input model with uncorrelated residuals, Bergmeir, Hyndman & Koo 2018), which scores every
        training row instead of the validation block.  The searches:
        "grid"    every combination of the knobs' grids (a ridge: the 7 decades of lambda; the elastic
                  net: 7 lambdas x 5 shares = 35 points);
        "golden"  the grid, then for each knob in turn a golden-section search between the neighbours
                  of the best grid value (in log10 for a log knob), the other knobs held at the best
                  point (the continuous optimum rather than the nearest grid point, without settling in
                  the wrong valley; tolerance 0.02 in the knob's coordinate, at most 40 evaluations a knob);
        "random"  `trials` points drawn uniformly in each knob's coordinate between the grid's ends
                  (log-uniform for a log knob) from one generator seeded by `seed` -- the baseline for
                  three or more knobs (Bergstra & Bengio 2012); trials = 0 => as many as the grid has.
        Every point is scored once (memoised).  Under `Model.tuning` every chooser calls it on each
        candidate and builds the candidate at the knobs found (`withHP`).  A model without knobs has
        nothing to tune: ({}, its score).

        Parameters
        ----------
        knobs : sequence of Knob, optional
            The knobs to turn (None => the regression's `knobs`).
        measure : QoF, default=QoF.smape
            The error measure compared (see `scoreOf`).
        hs : sequence of int, optional
            The horizons averaged (None => all; [1] => the first, a recursive sub-model's own).
        verbose : bool, default=False
            Prints every evaluation.
        search : {"grid", "golden", "random"}, default="grid"
        folds : int, optional
            None => `Model.folds`; 0 => the validation block (`valScore`, needs `Model.validating`);
            k => `cvScore (k)`.
        trials : int, default=0
            The number of random points ("random"; 0 => the grid's size).
        seed : int, default=0
            The seed of the random points ("random"; ScalaTion's `stream`).

        Raises
        ------
        ValueError
            Without folds, if `Model.validating` is off: in-sample a penalty always prefers its
            smallest value; or for an unknown `search`.
        """
        folds = Model.folds if folds is None else int(folds)
        if folds == 0 and not Model.validating:
            raise ValueError("tune: needs Model.validating (or Model.folds = k) -- in-sample a penalty always prefers its smallest value")
        if search not in ("grid", "golden", "random"):
            raise ValueError(f"tune: search must be 'grid', 'golden' or 'random', got {search!r}")

        def scoreAt(mod: "Forecaster") -> float:
            return mod.valScore(measure=measure, hs=hs) if folds == 0 else mod.cvScore(folds, measure, hs)

        ks = list(self.knobs if knobs is None else knobs)
        if not self.penalized or not ks:
            return {}, scoreAt(self)                                      # no knobs: nothing to tune
        memo: Dict[Tuple[float, ...], float] = {}

        def hpAt(pt: Tuple[float, ...]) -> Dict[str, float]:
            return {k.name: float(v) for k, v in zip(ks, pt)}

        def score(pt: Sequence[float]) -> float:
            pt = tuple(float(v) for v in pt)
            if pt not in memo:
                cand = self.withHP(hpAt(pt))
                memo[pt] = scoreAt(cand)
                if verbose:
                    at = ", ".join(f"{k.name} = {v:g}" for k, v in zip(ks, pt))
                    print(f"tune: {at} ({cand.modelName}): score = {memo[pt]:.4f}")
            return memo[pt]

        def best() -> Tuple[float, ...]:
            return min(memo, key=memo.get)

        grid = list(itertools.product(*(tuple(float(v) for v in k.grid) for k in ks)))   # the product of the knobs' grids

        def golden(i: int) -> None:                                       # golden-section on knob i, the others at the best point
            k = ks[i]
            srt = sorted(float(v) for v in k.grid)
            b0 = best()
            j = min(range(len(srt)), key=lambda jj: abs(srt[jj] - b0[i]))
            a, b = k.toCoord(srt[max(j - 1, 0)]), k.toCoord(srt[min(j + 1, len(srt) - 1)])
            if b - a <= 0.0:
                return

            def at(c: float) -> float:
                pt = list(b0)
                pt[i] = k.fromCoord(c)
                return score(pt)

            r = (np.sqrt(5.0) - 1.0) / 2.0                                # the golden ratio conjugate
            c, d = b - r * (b - a), a + r * (b - a)
            fc, fd = at(c), at(d)
            it = 0
            while b - a > 0.02 and it < 40:                               # tolerance 0.02 in the knob's coordinate
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - r * (b - a)
                    fc = at(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + r * (b - a)
                    fd = at(d)
                it += 1

        if search == "random":
            n = trials if trials > 0 else len(grid)
            rng = np.random.default_rng(seed)                             # ONE generator for every coordinate
            lo = [k.toCoord(min(k.grid)) for k in ks]
            hi = [k.toCoord(max(k.grid)) for k in ks]
            for _ in range(n):
                score([k.fromCoord(lo[i] + (hi[i] - lo[i]) * rng.random()) for i, k in enumerate(ks)])
        else:
            for pt in grid:                                               # the grid first, always
                score(pt)
            if search == "golden":                                        # then one sweep of golden-section
                for i in range(len(ks)):
                    golden(i)
        pt = best()
        return hpAt(pt), memo[pt]

    def cvScore(self, folds: int = 5, measure: QoF = QoF.smape, hs: Optional[Sequence[int]] = None) -> float:
        """
        The k-fold cross-validation score of the regression on the chooser's rows (the training rows,
        or the fit rows under `Model.validating`): the embedded rows (one row = one lag vector and its
        target(s)) in `folds` contiguous blocks, each block predicted by the regression fitted on the
        other blocks, the held-out predictions pooled per horizon in original units and scored by
        `scoreOf (measure, hs)` (no penalty: the rows were not fitted).  Ordinary k-fold on the
        embedded rows is valid for a lag-input model whose residuals are uncorrelated (Bergmeir,
        Hyndman & Koo, 2018) and scores every training row, where the validation block scores a
        fifth of the series; blocks rather than random folds keep neighbouring rows apart.  Only the
        regression's own horizons are scored: h = 1 for a recursive model (its h > 1 forecasts need the
        lag structure), every horizon for a direct one.  The model itself is untouched.

        Parameters
        ----------
        folds : int, default=5
            The number of contiguous blocks (at least 2).
        measure : QoF, default=QoF.smape
            The error measure (see `scoreOf`).
        hs : sequence of int, optional
            The horizons averaged (None => all the regression's).
        """
        if folds < 2:
            raise ValueError(f"cvScore: folds = {folds} must be at least 2")
        T, X = self._cvData()                                             # targets (rows, nh) and inputs, model units
        m = X.shape[0]
        cand = copy.copy(self)                                            # a disposable copy: its training leaves self as is
        cand._model = cand._make_model()
        cand._params, cand._b = None, None
        pred = np.full(T.shape, np.nan)
        for block in np.array_split(np.arange(m), folds):
            keep = np.setdiff1d(np.arange(m), block)
            cand._cvTrain(T[keep], X[keep])
            pred[block] = cand._cvPredict(X[block])
        if self._yForm is not None:                                       # original units, as every QoF of the platform
            T, pred = self._yForm.inverse_transform(T), self._yForm.inverse_transform(pred)
        qof = np.zeros((T.shape[1], N_QoF))
        for h in range(T.shape[1]):
            ok = np.isfinite(T[:, h]) & np.isfinite(pred[:, h])
            cand.mod_resetDF(int(ok.sum()))
            qof[h] = cand.diagnose(T[ok, h], pred[ok, h])
        return self.scoreOf(qof, inSample=False, measure=measure, hs=hs)

    def _cvData(self) -> Tuple[np.ndarray, np.ndarray]:
        """The rows `cvScore` folds (hook): the targets (rows, nh) and inputs of the embedded regression
        on the chooser's rows, in model units."""
        raise NotImplementedError(f"{self.modelName}: cvScore needs _cvData")

    def _cvTrain(self, T: np.ndarray, X: np.ndarray) -> None:
        """Fits the regression on (X, T) (hook; the recursive models take a vector, the direct a matrix)."""
        self.train(T[:, 0], X)

    def _cvPredict(self, X: np.ndarray) -> np.ndarray:
        """The regression's predictions (rows, nh) of the rows of X in model units (hook)."""
        return np.asarray(self._predict(X), dtype=float).reshape(X.shape[0], -1)

    def tnT_Test(self, rc: Optional[int] = None, showYf: bool = False, showParams: bool = False,
                 verbose: bool = True) -> np.ndarray:
        """
        Train-and-Test (TnT) testing with rolling validation (ScalaTion's `tnT_Test`): `rollTest`
        over the testing rows, the model trained on the training rows (REPORT), refit every `rc`
        rows, and the QoF table on the testing rows (no rows skipped; the skip is restored after).

        Parameters
        ----------
        rc : int, optional
            The retraining cycle (None => `Forecaster.RC`).
        showYf : bool, default=False
            Whether to print the forecast matrix.
        showParams : bool, default=False
            Whether to print the full parameter array of the last retraining.
        verbose : bool, default=True
            Whether to print the banners, the REPORT and the QoF table.

        Returns
        -------
        np.ndarray, shape (hh, N_QoF)
            One QoF vector per horizon on the testing rows.
        """
        return self.rollTest(teRng(self.mSeries, self._te_ratio), rc, "TnT", verbose, showYf, showParams)

    def tvT_Test(self, rc: Optional[int] = None, showYf: bool = False,
                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train-Validate-Test (ScalaTion's `tvT_Test`): `valTest` (fit rows -> validation rows), then
        `tnT_Test` (training rows -> testing rows), the two QoF tables and one summary line with the
        split and the two scores; what a chooser sees, then the test.

        Returns
        -------
        (np.ndarray, np.ndarray)
            The QoF tables on the validation rows and on the testing rows (hh, N_QoF each; None for
            an empty block).
        """
        m = self.mSeries
        qv = self.valTest(rc, showYf, verbose=verbose)
        qt = self.tnT_Test(rc, showYf, verbose=verbose)
        sv = float("nan") if qv is None else self.scoreOf(qv, False)
        st = float("nan") if qt is None else self.scoreOf(qt, False)
        print(f"tvT_Test: {self.modelName}: split {Model.splitStr(m, self._te_ratio)}: validation score = "
              f"{sv:.4f}, test score = {st:.4f} (mean sMAPE over h = 1..{self.hh})")
        return qv, qt

    def refitTransform(self) -> "Forecaster":
        """
        The model rebuilt by its factory with the switch OFF (`Model.inSample`), so its transforms
        (the response's and the exogenous variables' scaling, a formula's base-column transforms, a
        formula's nonlinear parameters) are fitted on the TRAINING rows: the model reported after
        choosing under `Model.validating`, whose candidates were built on the fit rows.  A model with
        no factory recipe (no transform) is returned as it is (ScalaTion rev5.2's `refitTransform` and
        `recipe`).
        """
        if self._recipe is None:
            return self
        with Model.inSample():
            return self._recipe()
