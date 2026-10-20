"""
NARX_SR_D Model: Direct Multi-Horizon NARX Symbolic Regression

Implements a DIRECT (one model per horizon) forecaster -- an `ARX_D` -- whose columns carry
NONLINEAR parameters, built either from transform TYPES (`apply` / `rescale`: the lags of y and of
each exogenous variable and of their transforms (y + s)^p, log (x + s), ..., whose parameters come
from `TransformT.wlu`) or from a `NARX_Formula` (`fromTerms`: lagged, transformed and interaction
terms, seasonal waves, and terms with FREE parameters -- a shift inside a log, an exponent, a
saturation constant, a threshold, a decay, ...).  The coefficients are fitted by the centered ridge
regression of `Forecaster_D` per horizon (penalty args.lam); the nonlinear parameters are fitted
ONCE, on the rows whose targets lie in the training rows, by the optimizer args.optimizer (an
`OptimizerNL`; None => the module's `OPTIMIZER_NL`):

    OptimizerNL.VarPro   Variable Projection: the coefficients are eliminated (ridge solution at
                         each a) and nonlinear least squares runs on the reduced residual (default)
    OptimizerNL.LBFGS_B  the coefficients are eliminated in the same way and the bounded
                         quasi-Newton L-BFGS-B runs on the reduced objective
    OptimizerNL.Joint    one least-squares problem over the coefficients and the parameters
                         together, started from the ridge coefficients at a0

and then held fixed: a rolling validation refits only the coefficients, as `ARX_D` does.  Every
fitted parameter counts as a degree of freedom (`nFree`), so the selection score pays for it.

A formula without free parameters (or args.tune = False) gives the plain direct ARX on its columns,
so one class serves the transform types and the whole formula language.  Feature selection uses
`ARX_D`'s sub-models (`getModel`, `getModel_h`), as in ScalaTion.

The Python counterpart of ScalaTion's `nonlinear/NARX_SR_D.scala` (`OptimizerNL`, `OPTIMIZER_NL`,
the class, and its companion's `apply`, `rescale`, `make`, `buildMatrix`, `optimize`, `fromTerms`,
`fitNL`, `joint`, `ridge`, `initializeW`, `tunable`, `getTransforms`, `formNames`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-19
Version: 2.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares, minimize

from Util.make_matrix4ts import backfill_matrix, makeMatrix4Y
from Util.tools import vector_str
from Util.transform import Transform, TransformT
from modeling.centered_ridge import CenteredRidge
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_formula import ARX_Formula
from modeling.forecasting.arx_sr import ARX_SR, srMatrix
from modeling.forecasting.forecaster import Args, Forecaster, Model
from modeling.forecasting.arx import _checkTFormT
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula, Par
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.optimization.varpro import VarPro


class OptimizerNL(Enum):
    """
    The nonlinear optimizers `NARX_SR_D.fitNL` can use for the free parameters of the terms
    (ScalaTion's `OptimizerNL`); all three minimize the same ridge least-squares criterion.

    VarPro   eliminates the coefficients b (the ridge solution b(a)) and runs bounded nonlinear
             least squares (scipy `least_squares`, trust region reflective) on the reduced residual
    LBFGS_B  eliminates b in the same way and runs the bounded quasi-Newton L-BFGS-B (scipy
             `minimize`) on the reduced objective
    Joint    runs `least_squares` over a and b together (b unbounded), starting from b(a0)
    """
    VarPro = "VarPro"
    LBFGS_B = "LBFGS_B"
    Joint = "Joint"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def of(cls, opt: Union["OptimizerNL", str, None]) -> "OptimizerNL":
        """The optimizer named by `opt` (an `OptimizerNL`, its name in any case, e.g., "lbfgsb";
        None => `OPTIMIZER_NL`)."""
        if opt is None:
            return OPTIMIZER_NL
        if isinstance(opt, OptimizerNL):
            return opt
        key = str(opt).replace("-", "_").replace("_", "").lower()
        for o in cls:
            if o.name.replace("_", "").lower() == key:
                return o
        raise ValueError(f"optimizer '{opt}' unknown: {', '.join(o.name for o in cls)}")


OPTIMIZER_NL: OptimizerNL = OptimizerNL.VarPro       # the default optimizer of `NARX_SR_D.fitNL` (a test may reassign it)

DEBUG = False                                        # True => `make` / `fitNL` print ScalaTion's debug lines

BIG = 1e9                                            # a bound this large means no bound (ScalaTion's `BIG`)

TSet = Sequence[TransformT]                          # a set of transform TYPES (ScalaTion's `LSET [TransformT]`)


def _nonlinear(t: TransformT, i: int) -> bool:
    """
    Whether parameter i (0 = shift, 1 = scale/power) of transform type t must be searched as a
    nonlinear parameter (ScalaTion's `NARX_SR_D.nonlinear`): one that only shifts or rescales the
    column is covered by the intercept and the linear coefficient --
        log (a x + s) = log a + log (x + s/a)  => the scale a is a constant plus a shift,
        exp (a x + s) = e^s exp (a x)          => the shift s is a linear coefficient --
    and the power of `PowR` is rounded to a rational in ScalaTion, so only its shift is searched.
    """
    if t is TransformT.Log:
        return i == 0
    if t in (TransformT.Exp, TransformT.NExp):
        return i == 1
    if t is TransformT.PowR:
        return i == 0
    return True


class NARX_SR_D(ARX_D):
    """
    Direct multi-horizon NARX symbolic regression: an `ARX_D` whose columns carry nonlinear
    parameters (ScalaTion's `NARX_SR_D`).

    Attributes
    ----------
    formula : NARX_Formula, optional
        The formula the columns come from (its parameters hold the fitted values); None for a
        model built from transform types (`apply` / `rescale`).
    fname : list of str
        The column labels.
    optimizer : OptimizerNL or None
        The optimizer that fitted the nonlinear parameters (None when nothing was fitted).
    nFree : int
        The number of nonlinear parameters that were FITTED (degrees of freedom they take).
    w_nl, parameter_nl : np.ndarray
        The nonlinear parameters (fitted or fixed): two per transform (shift, scale/power) in the
        order endogenous transforms, then the transforms of each exogenous variable; the formula's
        parameters in order of appearance.
    stepCounts : dict of str -> int
        The optimizer's counts of the parameter fit (empty when nothing was fitted).
    modelName : str
        "NARX_SR_D(p, px, n_exo)" (ScalaTion's name).

    See Also
    --------
    ARX_D : the direct forecasting logic (forecast, getModel, getModel_h); Forecaster_D the regression.
    """

    METHOD = "ridge"
    ARGS = ("p", "px", "spec", "crx", "lam", "scaled", "optimizer", "tune") + Forecaster.ARGS

    def __init__(self,
                 y: np.ndarray,
                 hh: int,
                 X: np.ndarray,
                 args: Any = None,
                 formula: Optional[ARX_Formula] = None,
                 fname: Optional[Sequence[str]] = None,
                 n_free: int = -1,
                 yForm=None,
                 w_nl: Optional[Sequence[float]] = None,
                 n_exo: Optional[int] = None) -> None:
        """
        Initializes the model on a GIVEN design matrix (ScalaTion's constructor; use `apply`,
        `rescale` or `fromTerms` to build one).  The coefficients are fitted by the centered ridge
        regression (penalty args.lam, default 0.1; args.method is not read: always "ridge").

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The endogenous series (already transformed when `yForm` is given).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        X : np.ndarray, shape (m, n)
            The design matrix (no constant column: the intercept is implicit).
        args : Args or dict, optional
            The settings: p, px, spec, crx, nneg, skip, lam, scaled (the regression standardizes the
            columns: training mean and standard deviation), te_ratio (`Args.help (NARX_SR_D)`).
        formula : NARX_Formula, optional
            The formula the columns come from.
        fname : sequence of str, optional
            The column labels (defaults to the formula's labels or x0, x1, ...).
        n_free : int, default=-1
            The number of nonlinear parameters that were FITTED (they take degrees of freedom like
            the coefficients, see `mod_resetDF`); -1 => all of w_nl; read back as `nFree`.
        yForm : Transform, optional
            The transform of `y` (forecasts are inverted into Yf); None => none.
        w_nl : sequence of float, optional
            The nonlinear parameters of the transforms (a model built from transform types).
        n_exo : int, optional
            The number of exogenous variables (defaults to the formula's).
        """
        args = self._argsOf(y, args)
        args.method = "ridge"                                            # the regression of NARX_SR_D (and its sub-models)
        super().__init__(y, hh, None, args, X=np.asarray(X, dtype=float))
        self.formula = NARX_Formula.of(formula) if formula is not None else None
        self._fname = list(fname) if fname is not None else \
            (self.formula.names if self.formula is not None else [f"x{j}" for j in range(self._X.shape[1])])
        self._w_nl = np.asarray(w_nl, dtype=float) if w_nl is not None else np.zeros(0)
        self._n_free = int(n_free)
        self.optimizer: Optional[OptimizerNL] = None
        self.stepCounts: Dict[str, int] = {}
        self._setYForm(yForm)
        hp = self.formula.lagHP() if self.formula is not None else \
            {"p": self._hparam["p"], "px": self._hparam["px"], "n_exo": 0 if n_exo is None else n_exo}
        self._n_exo = hp["n_exo"] if n_exo is None else n_exo
        self._modelName = f"NARX_SR_D({hp['p']}, {hp['px']}, {self._n_exo}){self.hpTag}"

    # -- factories: transform TYPES (ScalaTion's `apply` / `rescale`) ---------------------------

    @classmethod
    def apply(cls,
              y: np.ndarray,
              hh: int,
              xe: Optional[np.ndarray] = None,
              args: Any = None,
              fEndo: TSet = (TransformT.Root,),
              fExo: Optional[Sequence[TSet]] = None,
              trSize: int = -1,
              fname: Optional[Sequence[str]] = None,
              **opt_kwargs) -> "NARX_SR_D":
        """
        Creates the model on the data as given (ScalaTion's `NARX_SR_D.apply`; NOT rescaled, see
        `rescale`): the lags of y and of its transforms of the types fEndo, and of each exogenous
        variable and its transforms of the types fExo.  With args.tune = True the nonlinear parameters
        of the transforms are fitted by the optimizer (`optimize`) on the training rows; otherwise
        they keep their default values (`TransformT.wlu`), which gives the `ARX_SR_D` model with those
        transforms.  A transform that is not finite on some row (e.g., a root of a negative value) is
        reported.

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The endogenous series.
        hh : int
            The maximum forecasting horizon.
        xe : np.ndarray, shape (m, n_exo), optional
            The exogenous variables.
        args : Args or dict, optional
            The settings: p, px, spec, crx, lam, nneg, skip, te_ratio, tune (fit the nonlinear
            parameters; False => keep their defaults), scaled (the regression and the fit of the
            nonlinear parameters standardize the columns), optimizer (None => `OPTIMIZER_NL`).
        fEndo : sequence of TransformT, default=(TransformT.Root,)
            The transform types for the endogenous variable.
        fExo : sequence of sequences of TransformT, optional
            The transform types for each exogenous variable (None => `Pow` for each of them).
        trSize : int, default=-1
            The rows used for fitting the nonlinear parameters (-1 => `trSize` of args.te_ratio).
        fname : sequence of str, optional
            The feature names (defaults to `formNames`).
        **opt_kwargs
            Passed to the optimizer.
        """
        args = cls._argsOf(y, args)
        y = np.asarray(y, dtype=float)
        n_exo = xe.shape[1] if xe is not None else 0
        xe_bfill = backfill_matrix(np.asarray(xe, dtype=float)) if n_exo > 0 and args.px > 0 else None
        tr_size = Model.fitRows(len(y), args.te_ratio, trSize)
        return cls._make(xe_bfill, n_exo, y, hh, fname, args, list(fEndo), cls.exoSets(fExo, n_exo), None,
                         tr_size, None, **opt_kwargs)

    @classmethod
    def rescale(cls,
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                args: Any = None,
                fEndo: TSet = (TransformT.Root,),
                fExo: Optional[Sequence[TSet]] = None,
                tFormT: TransformT = TransformT.MinMax,
                lu: Sequence[float] = (1.0, 5.0),
                trSize: int = -1,
                fname: Optional[Sequence[str]] = None,
                **opt_kwargs) -> "NARX_SR_D":
        """
        Creates the model on rescaled data (ScalaTion's `NARX_SR_D.rescale`): rescales by `tFormT`
        (MinMax by default) with the statistics of the training rows (MinMax to lu = [1, 5] by default,
        which keeps the rescaled training values positive, as the power, root and log transforms
        require), then transforms, then lags; with args.tune the nonlinear parameters are fitted by the
        optimizer on the training rows (see `apply` for the other parameters).

        Parameters
        ----------
        tFormT : TransformT, default=TransformT.MinMax
            The transform type of the rescaling (ScalaTion's `rescale (..., tFormT)`).
        lu : sequence of float, default=(1, 5)
            The range [l, u] of the min-max rescaling.
        """
        args = cls._argsOf(y, args)
        y = np.asarray(y, dtype=float)
        n_exo = xe.shape[1] if xe is not None else 0
        tFormT = TransformT.MinMax if tFormT is None else tFormT
        _checkTFormT(args, tFormT)
        tr_size = Model.fitRows(len(y), args.te_ratio, trSize)
        tForm_y = tFormT.form(y[:tr_size])                           # the training rows' statistics, for In-ST and TnT
        tForm_y.setLU(lu)
        y_scl = tForm_y.f(y)
        xe_bfill = None
        if n_exo > 0 and args.px > 0:
            xe_bfill = backfill_matrix(np.asarray(xe, dtype=float))  # backfill each exogenous variable
            tForm_exo = tFormT.form(xe_bfill[:tr_size])
            tForm_exo.setLU(lu)
            xe_bfill = tForm_exo.f(xe_bfill)                         # rescale the backfilled exogenous variables
        mod = cls._make(xe_bfill, n_exo, y_scl, hh, fname, args, list(fEndo), cls.exoSets(fExo, n_exo), tForm_y,
                        tr_size, y, **opt_kwargs)
        mod._recipe = lambda: cls.rescale(y, hh, xe, args, fEndo, fExo, tFormT, lu, -1, fname, **opt_kwargs)
        return mod

    @classmethod
    def _make(cls, xe_bfill: Optional[np.ndarray], n_exo: int, y: np.ndarray, hh: int, fname: Optional[Sequence[str]],
              args: Args, fEndo: List[TransformT], fExo: List[List[TransformT]], tForm_y,
              trSize: int, y_org: Optional[np.ndarray], **opt_kwargs) -> "NARX_SR_D":
        """
        The common part of `apply` and `rescale` (ScalaTion's `NARX_SR_D.make`): the nonlinear
        parameters (fitted on the training rows when args.tune, or the defaults), the design matrix
        with the transforms at those parameters, and the model.
        """
        hp_ = args.hparam()
        tune = args.tune
        stepCounts: Dict[str, int] = {}
        if tune:
            w_nl, stepCounts = cls.optimize(xe_bfill, y, hh, hp_, fEndo, fExo, trSize, args.scaled, args.optimizer, **opt_kwargs)
        else:
            w_nl = cls.initializeW(fEndo, fExo)[0]
        if DEBUG:
            print(f"DEBUG @ NARX_SR_D.make: nonlinear parameters (shift, scale/power per transform) w_nl = {vector_str(w_nl)}")
        fEn, fEx = cls.getTransforms(w_nl, fEndo, fExo)
        X = cls.buildMatrix(xe_bfill, y, hp_, fEn, fEx)
        fname = list(fname) if fname is not None else cls.formNames(n_exo, hp_, len(fEndo), [len(s) for s in fExo])
        for j in range(X.shape[1]):
            if not np.isfinite(X[:, j]).all():
                print(f"[ NARX_SR_D.make ]: column {fname[j]} is not finite: a transform is outside its domain on some row")
        n_free = len(cls.tunable(fEndo, fExo)) if tune else 0      # the fitted parameters
        mod = cls(y, hh, X, args, None, fname, n_free, tForm_y, w_nl, n_exo)
        if tForm_y is not None and y_org is not None:
            mod._setYForm(tForm_y, y_org)
        mod._tForms = {"tForm_y": tForm_y, "fEndo": fEn, "fExo": fEx}
        mod.optimizer = OptimizerNL.of(args.optimizer) if tune and n_free > 0 else None
        mod.stepCounts = stepCounts
        mod._useforge = True
        return mod

    @staticmethod
    def buildMatrix(xe_bfill: Optional[np.ndarray], y: np.ndarray, hparam: Dict[str, Any],
                    fEndo: Sequence[Transform], fExo: Sequence[Sequence[Transform]]) -> np.ndarray:
        """
        Builds the input matrix (ScalaTion's `NARX_SR_D.buildMatrix`): the trend columns, the lags of
        y and of its transforms, the lags of each exogenous variable and of its transforms, and the
        lagged cross terms (crx = 1) -- the columns of `ARX_SR.buildMatrix`, not checked here (a
        column that is not finite is reported by `make`).
        """
        return srMatrix(xe_bfill, y, hparam, fEndo, fExo)

    @classmethod
    def optimize(cls, xe_bfill: Optional[np.ndarray], y: np.ndarray, hh: int, hparam: Dict[str, Any],
                 fEndo: TSet = (TransformT.Root,), fExo: Sequence[TSet] = (), trSize: int = -1,
                 scaled: bool = False, optimizer: Union[OptimizerNL, str, None] = None,
                 **opt_kwargs) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Fits the nonlinear parameters of the transforms (ScalaTion's `NARX_SR_D.optimize`) with the
        optimizer (`fitNL`), returning ALL of them (two per transform; the ones that are not tuned
        keep their defaults) and the step counts.  The model is separable: for given nonlinear
        parameters the coefficients of all horizons solve a linear (ridge) least-squares problem, so
        VarPro and L-BFGS-B search only the nonlinear parameters, within their bounds, while Joint
        searches them together with the coefficients; the parameters are shared by the hh horizons.
        The fit uses the rows whose hh targets all lie in the training rows (0 until trSize - hh + 1)
        and the centering, scaling and ridge penalty of the model's regression.

        Parameters
        ----------
        xe_bfill : np.ndarray, optional
            The backfilled (and rescaled) exogenous variables (None => none).
        y : np.ndarray, shape (m,)
            The endogenous series (rescaled by `rescale`).
        hh : int
            The maximum forecasting horizon.
        hparam : dict
            The hyper-parameters ("lambda" is the ridge penalty, `ridge`).
        fEndo, fExo : transform types
            As in `apply`.
        trSize : int, default=-1
            The number of training rows (-1 => `trSize`).
        scaled : bool, default=False
            Whether the regression standardizes the columns.
        optimizer : OptimizerNL or str, optional
            The optimizer (None => `OPTIMIZER_NL`).

        Returns
        -------
        (np.ndarray, dict)
            All the nonlinear parameters and the optimizer's step counts.
        """
        fEndo, fExo = list(fEndo), [list(s) for s in fExo]
        w0, l, u = cls.initializeW(fEndo, fExo)
        free = cls.tunable(fEndo, fExo)                              # indices of the parameters to tune
        if not free:
            return w0, {}                                            # nothing to tune
        tr_size = Model.fitRows(len(y), None, trSize)                 # None: Model.TE_RATIO (no model yet)
        fitRng = slice(0, tr_size - hh + 1)                          # rows whose targets lie in the training rows
        yy = makeMatrix4Y(y, hh)[fitRng]
        lb = np.array([-np.inf if l[i] <= -BIG else l[i] for i in free])
        ub = np.array([np.inf if u[i] >= BIG else u[i] for i in free])

        def full(a: np.ndarray) -> np.ndarray:                       # all parameters, given the tuned ones
            w = w0.copy()
            w[free] = a
            return w

        def basis(a: np.ndarray) -> np.ndarray:                      # the input matrix for the fitting rows
            fEn, fEx = cls.getTransforms(full(a), fEndo, fExo)
            return cls.buildMatrix(xe_bfill, y, hparam, fEn, fEx)[fitRng]

        if DEBUG:
            print(f"DEBUG @ NARX_SR_D.optimize: fit on rows Range 0 until {fitRng.stop} the parameters ({', '.join(map(str, free))})")
        a, stepCounts = cls.fitNL(basis, yy, w0[free], lb, ub, cls.ridge(hparam), scaled, optimizer, **opt_kwargs)
        return full(a), stepCounts

    # -- factory: a formula (ScalaTion's `fromTerms`) -------------------------------------------

    @classmethod
    def fromTerms(cls,
                  formula: Union[str, ARX_Formula],
                  dataset: Dict[str, np.ndarray],
                  hh: int,
                  args: Any = None,
                  tFormT: Optional[TransformT] = None,
                  trSize: int = -1,
                  maxLag: int = 8,
                  freqs: Optional[Sequence[float]] = None,
                  **opt_kwargs) -> "NARX_SR_D":
        """
        Builds the model from a formula (string or `ARX_Formula` / `NARX_Formula`) over a
        named-column dataset (ScalaTion's `NARX_SR_D.fromTerms`): checks the formula and
        transforms the base columns (`prepare`), resolves the parameters' defaults on the training
        rows, fits the free parameters (minus the redundant and the fixed ones) on the rows whose
        targets all lie in the training rows, then builds the design matrix at the fitted values.

        Parameters
        ----------
        formula : str or ARX_Formula
            E.g., "y ~= y~(3,1) + log(? + icu~1) + hinge(temp~1, ?)".
        dataset : dict of str -> np.ndarray
            The named columns (the response under its name, e.g., "y").
        hh : int
            The maximum forecasting horizon.
        args : Args or dict, optional
            The settings: lam (default 0.1), nneg, skip (default the formula's largest lag),
            te_ratio, scaled
            (standardize the columns in the regression, also inside the parameter fit), optimizer (of
            the nonlinear parameters: `OptimizerNL.VarPro`, `LBFGS_B` or `Joint`, or its name; None =>
            the module's `OPTIMIZER_NL`), tune (False => the free parameters stay at their start
            values, no optimizer run); p, px and spec = 0 come from the formula (ScalaTion's `lagHP`).
        tFormT : TransformT, optional
            The transform type of every base column (ScalaTion's `fromTerms (..., tFormT)`); None =>
            the formula's own (its "TFORM:" clause).
        trSize : int, default=-1
            The training rows (ScalaTion's `fromTerms (..., trSize)`; -1 => `Model.fitRows`: the training
            rows, or the fit rows while choosing).
        maxLag : int, default=8
            The largest lag the parser accepts (for a string formula).
        freqs : sequence of float, optional
            The frequencies `wave(k)` draws on (for a string formula).
        **opt_kwargs
            Passed to the optimizer (`fitNL`: e.g., max_nfev, ftol).

        Returns
        -------
        NARX_SR_D
            Untrained: call `train()` (or `inSample_Test`, `tnT_Test`, `rollValidate`).

        Raises
        ------
        ValueError
            For a formula that does not parse or does not fit the dataset.
        """
        args = Args.of(args)
        opt = OptimizerNL.of(args.optimizer)
        fit_params = args.tune
        f = NARX_FormulaParser.parseFormula(formula, list(dataset.keys()), maxLag, freqs) if isinstance(formula, str) \
            else NARX_Formula.of(formula)
        if tFormT is not None:
            f = NARX_Formula(f.resp, f.preds, tFormT)
        m = len(next(iter(dataset.values())))
        tr = Model.fitRows(m, args.te_ratio, trSize)
        if tr < hh:
            raise ValueError(f"fromTerms: trSize {tr} < hh = {hh}: no row has all its targets in the training rows")
        td = f.prepare(dataset, tr)                                  # checks, transforms, resolves

        pars = f.params()
        for p in pars:
            p.reset()
        f.resolve(td.data, td.trSize)
        redundant = f.redundant()
        for p in redundant:
            print(f"fromTerms: parameter {p.role.value} is covered by the linear fit: kept at its start value")
        free: List[Par] = [p for p in pars if p.free and not any(p is q for q in redundant)]

        stepCounts: Dict[str, int] = {}
        if free and fit_params:
            fitRng = slice(0, td.trSize - hh + 1)                    # rows whose targets lie in the training rows
            YY = makeMatrix4Y(td.y, hh)[fitRng]

            def basis(a: np.ndarray) -> np.ndarray:
                for i, p in enumerate(free):
                    p.update(a[i])
                return f.matrix(td.data, check=False)[fitRng]

            a0 = np.array([p.w0 for p in free])
            lb = np.array([p.lb for p in free])
            ub = np.array([p.ub for p in free])
            a, stepCounts = cls.fitNL(basis, YY, a0, lb, ub, args.lam, args.scaled, opt, **opt_kwargs)
            for i, p in enumerate(free):
                p.update(a[i])
        for p in pars:
            p.set(p.value)                                            # from now on the labels show the values

        X = f.matrix(td.data)                                        # one column per term, at the fitted values
        hp = f.lagHP()
        args.update(p=hp["p"], px=hp["px"], spec=0)
        if args.skip is None:
            args.skip = f.maxLag
        mod = cls(td.y, hh, X, args, formula=f, n_free=len(free) if fit_params else 0, yForm=td.tForm_y)
        mod.optimizer = opt if free and fit_params else None
        mod.stepCounts = stepCounts
        mod._tForms = {"tForm_y": td.tForm_y}
        mod._recipe = lambda: cls.fromTerms(f, dataset, hh, args, tFormT, -1, maxLag, freqs, **opt_kwargs)
        return mod

    @staticmethod
    def ridge(hparam: Optional[Dict[str, Any]]) -> float:
        """The ridge penalty the model's regression will use: hparam["lambda"] if present, the
        default `CenteredRidge.hp["lambda"]` otherwise (ScalaTion's `NARX_SR_D.ridge`)."""
        return CenteredRidge.ridge(hparam)

    # -- the nonlinear parameters --------------------------------------------------------------

    @staticmethod
    def fitNL(basis: Callable[[np.ndarray], np.ndarray], yy: np.ndarray, a0: np.ndarray, lb: np.ndarray,
              ub: np.ndarray, lam: float, scaled: bool, optimizer: Union[OptimizerNL, str, None] = None,
              **kwargs) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Fits the nonlinear parameters a, within [lb, ub], so that the ridge least-squares criterion
        of the linear fit of yy on basis(a) is minimal, with the given optimizer (ScalaTion's
        `NARX_SR_D.fitNL`, see `OptimizerNL`).

        Parameters
        ----------
        basis : callable
            a -> the input matrix (for the fitting rows) at the nonlinear parameters a.
        yy : np.ndarray, shape (m, hh)
            The response matrix (one column per horizon) for the fitting rows.
        a0 : np.ndarray, shape (k,)
            The starting point.
        lb, ub : np.ndarray, shape (k,)
            The bounds (-inf / inf for none).
        lam : float
            The ridge penalty of the model's regression.
        scaled : bool
            Whether the regression standardizes the columns.
        optimizer : OptimizerNL or str, optional
            The optimizer (None => `OPTIMIZER_NL`).
        **kwargs
            Passed to the optimizer (`least_squares`: ftol, xtol, max_nfev, ...; for L-BFGS-B
            `max_nfev` and `ftol` are translated to its `maxfun` and `ftol` options).

        Returns
        -------
        (a, stepCounts) : (np.ndarray (k,), dict of str -> int)
            The fitted parameters and the optimizer's step counts.
        """
        opt = OptimizerNL.of(optimizer)
        varPro = VarPro(basis, yy, lb, ub, lam=lam, scaled=scaled)
        if opt is OptimizerNL.VarPro:
            a = varPro.optimize(a0, **kwargs)
            stepCounts = dict(varPro.stepCounts)
        elif opt is OptimizerNL.LBFGS_B:
            a, stepCounts = NARX_SR_D.lbfgs_b(varPro, a0, **kwargs)
        else:
            a, stepCounts = NARX_SR_D.joint(varPro, a0, lb, ub, lam, **kwargs)
        if DEBUG:
            print(f"fitNL: {opt}: a = {vector_str(a)}, objective = {varPro.objective(a)}, steps = {stepCounts}")
        return a, stepCounts

    @staticmethod
    def lbfgs_b(varPro: VarPro, a0: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Fits the nonlinear parameters with the bounded quasi-Newton L-BFGS-B (scipy `minimize`) on
        VarPro's reduced objective (gradient by finite differences); the `least_squares` keywords
        `max_nfev` and `ftol` become the options `maxfun` and `ftol`, the others only
        `least_squares` knows (`x_scale`, `jac`, `method`, `xtol`, `gtol`) are dropped.

        Returns
        -------
        (a, stepCounts) : (np.ndarray (k,), dict of str -> int)
        """
        bounds = [(None if not np.isfinite(l) else l, None if not np.isfinite(u) else u)
                  for l, u in zip(varPro.lb, varPro.ub)]
        options = dict(kwargs.pop("options", {}))
        if "max_nfev" in kwargs:
            options["maxfun"] = int(kwargs.pop("max_nfev"))
        if "ftol" in kwargs:
            options["ftol"] = float(kwargs.pop("ftol"))
        for k in ("x_scale", "jac", "method", "xtol", "gtol"):
            kwargs.pop(k, None)
        res = minimize(varPro.objective, varPro.clip(a0), method="L-BFGS-B", bounds=bounds, options=options, **kwargs)
        return res.x, {"nfev": int(res.nfev), "nit": int(res.nit), "status": int(res.status)}

    @staticmethod
    def joint(varPro: VarPro, a0: np.ndarray, lb: np.ndarray, ub: np.ndarray, lam: float,
              method: str = "trf", **kwargs) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Fits the nonlinear parameters a and the linear coefficients b of all the responses TOGETHER
        with one bounded least-squares problem (the alternative to Variable Projection), on the
        residual [y_j - X(a) b_j; -sqrt(lambda) b_j] stacked over the responses j, with the
        centering and scaling of varPro's inner problem, started at a0 and the ridge coefficients
        at a0 (ScalaTion's `NARX_SR_D.joint`).

        Parameters
        ----------
        varPro : VarPro
            The Variable Projection solver holding the problem (basis, responses).
        a0 : np.ndarray, shape (k,)
            The starting point for the nonlinear parameters.
        lb, ub : np.ndarray, shape (k,)
            The bounds on the nonlinear parameters (the coefficients are unbounded).
        lam : float
            The ridge penalty on the coefficients.
        method : {"trf", "dogbox"}, default="trf"
            The scipy `least_squares` method.
        **kwargs
            Passed to `least_squares`.

        Returns
        -------
        (a, stepCounts) : (np.ndarray (k,), dict of str -> int)
        """
        a0 = varPro.clip(a0)
        B0 = varPro.coefficients(a0)
        n, ny = B0.shape
        k = len(a0)
        slam = np.sqrt(lam)

        def res_joint(w: np.ndarray) -> np.ndarray:                 # w = (a, b_1, ..., b_ny)
            B = w[k:].reshape(n, ny, order="F")
            Xc = varPro.system(w[:k])
            r = (varPro.response - Xc @ B).ravel(order="F")
            if lam > 0.0:
                r = np.concatenate((r, -slam * B.ravel(order="F")))
            return r

        w0 = np.concatenate((a0, B0.ravel(order="F")))
        lb_ = np.concatenate((np.asarray(lb, dtype=float), np.full(n * ny, -np.inf)))
        ub_ = np.concatenate((np.asarray(ub, dtype=float), np.full(n * ny, np.inf)))
        res = least_squares(res_joint, w0, bounds=(lb_, ub_), method=method, **kwargs)
        return res.x[:k], {"nfev": int(res.nfev), "njev": int(res.njev or 0), "status": int(res.status)}

    # -- the transform types and their nonlinear parameters (ScalaTion's companion helpers) -----

    @staticmethod
    def exoSets(fExo: Optional[Sequence[TSet]], n_exo: int) -> List[List[TransformT]]:
        """The sets of transform types for the n_exo exogenous variables: `Pow` for each when fExo is
        None, an empty set for a variable beyond fExo's length (ScalaTion's `exoSets`)."""
        if fExo is None:
            return [[TransformT.Pow] for _ in range(n_exo)]
        return [list(fExo[j]) if j < len(fExo) else [] for j in range(n_exo)]

    @staticmethod
    def allForms(fEndo: TSet, fExo: Sequence[TSet]) -> List[TransformT]:
        """All the transform types in the order of the nonlinear parameters: endo's transforms, exo1's
        transforms, exo2's transforms, ... (ScalaTion's `allForms`)."""
        return list(fEndo) + [t for s in fExo for t in s]

    @classmethod
    def initializeW(cls, fEndo: TSet, fExo: Sequence[TSet]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The default nonlinear parameters and their bounds (ScalaTion's `initializeW`): two per
        transform (shift, scale/power) from `TransformT.wlu`; a type without parameters (e.g., Log1p)
        contributes none.

        Returns
        -------
        (w, l, u) : three np.ndarray
        """
        w, l, u = [], [], []
        for t in cls.allForms(fEndo, fExo):
            wlu = t.wlu
            if wlu is not None:
                w.extend(wlu[0])
                l.extend(wlu[1])
                u.extend(wlu[2])
        return np.array(w, dtype=float), np.array(l, dtype=float), np.array(u, dtype=float)

    @classmethod
    def tunable(cls, fEndo: TSet, fExo: Sequence[TSet]) -> List[int]:
        """The indices (into the nonlinear parameters) of the parameters to tune: those whose bounds
        differ (equal bounds fix a parameter) and that are truly nonlinear (ScalaTion's `tunable`)."""
        idx, k = [], 0
        for t in cls.allForms(fEndo, fExo):
            wlu = t.wlu
            if wlu is None:
                continue
            for i in range(2):
                if wlu[1][i] < wlu[2][i] and _nonlinear(t, i):
                    idx.append(k + i)
            k += 2
        return idx

    @staticmethod
    def getTransforms(w_nl: np.ndarray, fEndo: TSet, fExo: Sequence[TSet]) -> Tuple[List[Transform], List[List[Transform]]]:
        """The transform objects at the nonlinear parameters w_nl, two per parameterized type, in
        `allForms` order (ScalaTion's `getTransforms`)."""
        k = 0

        def make(t: TransformT) -> Transform:
            nonlocal k
            if t.wlu is None:
                return t.form(None)                                  # a transform without parameters
            k += 2
            return t.form(np.array([w_nl[k - 2], w_nl[k - 1]]))

        listEndo = [make(t) for t in fEndo]
        listExo = [[make(t) for t in s] for s in fExo]
        return listEndo, listExo

    @staticmethod
    def formNames(n_exo: int, hparam: Dict[str, Any], n_fEn: int, n_fExArr: Sequence[int]) -> List[str]:
        """The column names (ScalaTion's `NARX_SR_D.formNames`, the labels of `ARX_SR.formNames`)."""
        return ARX_SR.formNames(n_exo, hparam, n_fEn, n_fExArr)

    # -- properties --------------------------------------------------------------------------

    @property
    def w_nl(self) -> np.ndarray:
        """The nonlinear parameters: the formula's in order of appearance, or those of the transforms
        (two per transform) for a model built from transform types (ScalaTion's `w_nl`)."""
        if self.formula is not None:
            return np.array([p.value for p in self.formula.params()])
        return self._w_nl

    @property
    def parameter_nl(self) -> np.ndarray:
        """The nonlinear parameters (ScalaTion's `parameter_nl`)."""
        return self.w_nl

    @property
    def lam(self) -> float:
        """The ridge penalty of the regression (args.lam, hparameter "lambda", default 0.1)."""
        return self.ridge(self._hparam)

    @property
    def n_coef(self) -> int:
        """The number of coefficients per horizon, intercept included."""
        return self._X.shape[1] + 1

    @property
    def nFree(self) -> int:
        """The number of fitted nonlinear parameters (ScalaTion's `nFree`: n_free, or all of w_nl when -1)."""
        return len(self.w_nl) if self._n_free < 0 else self._n_free

    @property
    def dof(self) -> int:
        """The degrees of freedom the model takes per horizon: the predictors plus the fitted
        nonlinear parameters (the base of the parsimony-adjusted scores)."""
        return self._X.shape[1] + self.nFree

    def mod_resetDF(self, size: int) -> None:
        """Resets the degrees of freedom: the k slopes plus the fitted nonlinear parameters
        (ScalaTion's `NARX_SR_D.mod_resetDF`)."""
        self.resetDF_k(size, self.nparams + self.nFree)

    @property
    def formulaStr(self) -> str:
        """The formula as text (the terms' labels joined when the model has no formula)."""
        return self.formula.formulaStr if self.formula is not None else " + ".join(self.fname)

    def __repr__(self) -> str:
        s = f"{self.modelName}(nTerms={self._X.shape[1]}, nFree={self.nFree}, lam={self.lam}), hh = {self.hh}, skip = {self._hparam['skip']}"
        return s + (f", optimizer={self.optimizer}" if self.optimizer is not None else "")
