"""
NARX_SR_D Model: Direct Multi-Horizon NARX Symbolic Regression from a Formula

Implements a DIRECT (one model per horizon) forecaster whose design matrix is built from a
`NARX_Formula` -- lagged, transformed and interaction terms, seasonal waves, and terms with FREE
nonlinear parameters (a shift inside a log, an exponent, a saturation constant, a threshold, a
decay, ...).  The coefficients are fitted by ridge regression per horizon (centered, with an
unpenalized intercept, `CenteredRidge`); the nonlinear parameters are fitted ONCE, on the rows
whose targets lie in the training rows, by one of three optimizers (the `optimizer` option,
ScalaTion's `OptimizerNL`):

    "varpro"  variable projection: nonlinear least squares on the reduced residual (default)
    "lbfgsb"  L-BFGS-B on the reduced objective, bounds honoured
    "joint"   one least-squares problem over the coefficients and the parameters together

and then held fixed: a rolling validation refits only the coefficients, as `ARX_D` does.  Every
fitted parameter counts as a degree of freedom (`nFree`), so the selection score pays for it.

A formula without free parameters gives the plain direct ARX on its terms (an `ARX_D` whose
columns are the terms), so one class serves the whole formula language.

The Python counterpart of ScalaTion's `nonlinear/NARX_SR_D.scala` (`NARX_SR_D.fromTerms`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from Util.build_matrix import build_matrix_Y
from modeling.forecasting.forecaster import trSize as tr_size_of
from modeling.forecasting.forecaster_d import Forecaster_D
from modeling.forecasting.arx_formula import ARX_Formula
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula, Par
from modeling.forecasting.nonlinear.narx_formula_parser import parse_formula
from modeling.optimization.varpro import VarPro
from modeling.regression import CenteredRidge


class NARX_SR_D(Forecaster_D):
    """
    Direct multi-horizon NARX symbolic regression on the columns of a formula.

    Attributes
    ----------
    formula : NARX_Formula
        The formula the columns come from (its parameters hold the fitted values).
    fname : list of str
        The column (term) labels.
    lam : float
        The ridge penalty on the coefficients (0 => OLS).
    scaled : bool
        Whether the centered columns are standardized before the ridge solve.
    optimizer : str
        The optimizer used for the nonlinear parameters ("varpro", "lbfgsb", "joint", or "none").
    nFree : int
        The number of nonlinear parameters that were FITTED (degrees of freedom they take).
    w_nl : np.ndarray
        The values of all the formula's parameters (fitted or fixed), in order of appearance.
    steps : dict
        The optimizer's counts of the parameter fit (empty when nothing was fitted).
    modelName : str
        "NARX_SR_D(p, q, n_exo)" (ScalaTion's name).

    See Also
    --------
    Forecaster_D : the base forecasting logic (y, X, Y, Yf, tr_size, rollValidate, the tests).
    """

    OPTIMIZERS = ("varpro", "lbfgsb", "joint")

    def __init__(self,
                 args: Dict[str, Any],
                 y: np.ndarray,
                 hh: int,
                 X: np.ndarray,
                 formula: Optional[ARX_Formula] = None,
                 fname: Optional[Sequence[str]] = None,
                 lam: float = 1.0,
                 scaled: bool = False,
                 n_free: int = 0,
                 nneg: bool = True,
                 yForm=None) -> None:
        """
        Initializes the model on a GIVEN design matrix (use `fromTerms` / `from_formula` to build one).

        Parameters
        ----------
        args : dict
            The hyper-parameters ("p", "q", "spec", "skip", ...; see `Forecaster`).
        y : np.ndarray, shape (m,)
            The endogenous series (already transformed when `yForm` is given).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        X : np.ndarray, shape (m, n_terms)
            The design matrix (no constant column: the intercept is implicit).
        formula : NARX_Formula, optional
            The formula the columns come from.
        fname : sequence of str, optional
            The column labels (defaults to the formula's labels or x0, x1, ...).
        lam : float, default=1.0
            The ridge penalty on the coefficients.
        scaled : bool, default=False
            Whether to standardize the centered columns before the ridge solve.
        n_free : int, default=0
            The number of fitted nonlinear parameters (degrees of freedom they take).
        nneg : bool, default=True
            Whether forecasts are clipped at zero (a count series); False for series that can
            be negative (e.g., a temperature).
        yForm : optional
            The transform of `y` (with `inverse_transform`), e.g., a `Transform`; None => none.
        """
        super().__init__(args, y, hh)
        self._set_X(np.asarray(X, dtype=float))
        self._Y = build_matrix_Y(self._y, hh)
        self.formula = NARX_Formula.of(formula) if formula is not None else None
        self._fname = list(fname) if fname is not None else \
            (self.formula.labels if self.formula is not None else [f"x{j}" for j in range(X.shape[1])])
        self.lam = float(lam)
        self.scaled = scaled
        self.optimizer = "none"
        self.n_free = int(n_free)
        self.steps: Dict[str, Any] = {}
        self._nneg = nneg
        self._yForm = yForm
        if yForm is not None:                                         # column 0 of Yf holds y in ORIGINAL units
            self._Yf[:, 0] = np.asarray(yForm.inverse_transform(self._y.reshape(-1, 1))).ravel()
        self._method = "ridge"
        self._coef: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None
        hp = self.formula.lagHP() if self.formula is not None else {"p": self.args["p"], "q": self.args["q"], "n_exo": 0}
        self._n_exo = hp["n_exo"]
        self._modelName = f"NARX_SR_D({hp['p']}, {hp['q']}, {hp['n_exo']})"

    # -- factories ---------------------------------------------------------------------------

    @classmethod
    def fromTerms(cls,
                  formula: Union[str, ARX_Formula],
                  dataset: Dict[str, np.ndarray],
                  hh: int,
                  args: Optional[Dict[str, Any]] = None,
                  tFormT=None,
                  trSize: int = -1,
                  lam: float = 1.0,
                  scaled: bool = False,
                  optimizer: str = "varpro",
                  maxLag: int = 8,
                  freqs: Optional[Sequence[float]] = None,
                  nneg: bool = True,
                  fit_params: bool = True,
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
        args : dict, optional
            Hyper-parameter overrides; "skip" defaults to the formula's largest lag.
        tFormT : TransformT, optional
            The transform type of every base column (defaults to the formula's; None => none).
        trSize : int, default=-1
            The training rows (-1 => ScalaTion's `trSize`, 80 %).
        lam : float, default=1.0
            The ridge penalty on the coefficients.
        scaled : bool, default=False
            Whether to standardize the columns before the ridge solve (also inside VarPro).
        optimizer : {"varpro", "lbfgsb", "joint"}, default="varpro"
            The optimizer of the nonlinear parameters.
        maxLag : int, default=8
            The largest lag the parser accepts (for a string formula).
        freqs : sequence of float, optional
            The frequencies `wave(k)` draws on (for a string formula).
        nneg : bool, default=True
            Whether forecasts are clipped at zero.
        fit_params : bool, default=True
            False => the free parameters stay at their start values (no optimizer run).
        **opt_kwargs
            Passed to the optimizer (e.g., max_nfev, ftol).

        Returns
        -------
        NARX_SR_D
            Untrained: call `train()` (or `inSample_Test`, `tnT_Test`, `rollValidate`).

        Raises
        ------
        ValueError
            For a formula that does not parse or does not fit the dataset.
        """
        if optimizer not in cls.OPTIMIZERS:
            raise ValueError(f"optimizer '{optimizer}' unknown: {cls.OPTIMIZERS}")
        f = parse_formula(formula, list(dataset.keys()), maxLag, freqs) if isinstance(formula, str) else NARX_Formula.of(formula)
        if tFormT is not None:
            f = NARX_Formula(f.resp, f.preds, tFormT)
        m = len(next(iter(dataset.values())))
        tr = trSize if trSize > 0 else tr_size_of(m, cls.TE_RATIO)
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

        steps: Dict[str, Any] = {}
        if free and fit_params:
            fitRng = slice(0, td.trSize - hh + 1)                    # rows whose targets lie in the training rows
            YY = build_matrix_Y(td.y, hh)[fitRng]

            def basis(a: np.ndarray) -> np.ndarray:
                for i, p in enumerate(free):
                    p.update(a[i])
                return f.matrix(td.data, check=False)[fitRng]

            a0 = np.array([p.w0 for p in free])
            lb = np.array([p.lb for p in free])
            ub = np.array([p.ub for p in free])
            vp = VarPro(basis, YY, lb, ub, lam=lam, scaled=scaled)
            a = vp.fit(a0, optimizer=optimizer, **opt_kwargs)
            for i, p in enumerate(free):
                p.update(a[i])
            steps = dict(vp.steps)
            steps["objective"] = vp.objective(a)
        for p in pars:
            p.set(p.value)                                            # from now on the labels show the values

        X = f.matrix(td.data)                                        # one column per term, at the fitted values
        hp = f.lagHP()
        a_ = {"p": hp["p"], "q": hp["q"], "spec": 0, "cross": False, "skip": f.maxLag}
        if args:
            a_.update(args)
        mod = cls(a_, td.y, hh, X, formula=f, lam=lam, scaled=scaled, n_free=len(free) if fit_params else 0,
                  nneg=nneg, yForm=td.tForm_y)
        mod.optimizer = optimizer if free and fit_params else "none"
        mod.steps = steps
        mod._tr_size = td.trSize
        mod._te_size = m - td.trSize
        return mod

    @classmethod
    def from_formula(cls, formula, data: Dict[str, np.ndarray], hh: int, args: Optional[Dict[str, Any]] = None,
                     tr_size: int = -1, lam: float = 1.0, scaled: bool = False, optimizer: str = "varpro",
                     max_lag: int = 8, freqs: Optional[Sequence[float]] = None, nneg: bool = True,
                     fit_params: bool = True, **opt_kwargs) -> "NARX_SR_D":
        """The former name of `fromTerms` (snake_case parameters)."""
        return cls.fromTerms(formula, data, hh, args=args, trSize=tr_size, lam=lam, scaled=scaled,
                             optimizer=optimizer, maxLag=max_lag, freqs=freqs, nneg=nneg,
                             fit_params=fit_params, **opt_kwargs)

    # -- properties --------------------------------------------------------------------------

    @property
    def w_nl(self) -> np.ndarray:
        """The values of the formula's parameters, in order of appearance (ScalaTion's `parameter_nl`)."""
        return np.array([p.value for p in self.formula.params()]) if self.formula is not None else np.array([])

    @property
    def parameter_nl(self) -> np.ndarray:
        return self.w_nl

    @property
    def n_coef(self) -> int:
        """The number of coefficients per horizon, intercept included."""
        return self._X.shape[1] + 1

    @property
    def nFree(self) -> int:
        """The number of fitted nonlinear parameters (ScalaTion's `nFree`)."""
        return self.n_free

    @property
    def dof(self) -> int:
        """The degrees of freedom the model takes per horizon: the predictors plus the fitted
        nonlinear parameters (the base of the parsimony-adjusted scores)."""
        return self._X.shape[1] + self.n_free

    def mod_resetDF(self, size: int) -> None:
        """Resets the degrees of freedom: the k slopes plus the fitted nonlinear parameters
        (ScalaTion's `NARX_SR_D.mod_resetDF`)."""
        self.resetDF_k(size, self.nparams + self.n_free)

    @property
    def formula_str(self) -> str:
        return self.formula.formulaStr if self.formula is not None else " + ".join(self.fname)

    @property
    def formulaStr(self) -> str:
        return self.formula_str

    def __repr__(self) -> str:
        s = f"{self.modelName}(n_terms={self._X.shape[1]}, n_free={self.n_free}, lam={self.lam}), hh = {self.hh}, skip = {self.args['skip']}"
        return s + (f", optimizer={self.optimizer}" if self.optimizer != "none" else "")

    # -- training and forecasting --------------------------------------------------------

    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Fits the coefficients of every horizon by centered ridge regression on the given rows
        (the nonlinear parameters are already fixed in the columns; padded rows dropped).

        Parameters
        ----------
        Y_ : np.ndarray, shape (n_rows, hh), optional
            The target matrix (defaults to the model's).
        X_ : np.ndarray, shape (n_rows, n_terms), optional
            The design matrix (defaults to the model's).
        """
        X_ = self._X if X_ is None else X_
        Y_ = self._Y if Y_ is None else Y_
        Y_, X_ = self.fit_rows(Y_, X_)
        reg = CenteredRidge(self.lam, self.scaled).fit(X_, Y_)
        self._coef = reg.coef_
        self._intercept = reg.intercept_
        self._set_params(reg.params)

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        The multi-horizon forecasts for rows [t_start, t_end), written into `Yf` (in the original
        scale when a transform was used).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
        """
        if self._coef is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = self._rectify(self._X[t_st:t_en] @ self._coef + self._intercept)
        self._store(t_st, t_en, y_fcast)
        return y_fcast

    def predict_rows(self, rows: np.ndarray) -> np.ndarray:
        """The forecasts of the given rows without touching `Yf` (for the scorer)."""
        if self._coef is None:
            raise ValueError("The model has not been trained. Call train first.")
        return self._rectify(self._X[rows] @ self._coef + self._intercept)

    # -- feature selection: the sub-models (ScalaTion's `NARX_SR_D` inherits `ARX_D.getModel` / `getModel_h`)

    def _nFreeOf(self, cols: Sequence[int]) -> int:
        """The fitted nonlinear parameters the terms of the columns `cols` carry (distinct, at most `n_free`)."""
        if self.formula is None or self.n_free == 0:
            return 0
        seen: List[Par] = []
        for j in cols:
            if j < len(self.formula.preds):
                for p in self.formula.preds[j].params():
                    if p.free and not any(p is q for q in seen):
                        seen.append(p)
        return min(self.n_free, len(seen))

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "NARX_SR_D":
        """
        Builds a `NARX_SR_D` on the columns `cols` of the design matrix (the nonlinear parameters
        are fixed in the columns, so the sub-model is a direct ridge model on the selected terms, as
        ScalaTion's inherited `ARX_D.getModel`): the same y, Y, hh, transform of y, ridge penalty,
        scaling, hyper-parameters and the labels of the selected terms; the fitted parameters the
        kept terms carry still count as degrees of freedom.

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the design matrix with the selected terms (defaults to all).
        """
        from modeling.forecasting.feature_selection_d import subModel
        cols = self.mcols if cols is None else list(cols)
        fn = self.fname
        sub = NARX_SR_D(self.args, self.getY(), self.hh, self._X[:, cols], formula=None,
                        fname=[fn[j] for j in cols], lam=self.lam, scaled=self.scaled, n_free=self._nFreeOf(cols),
                        nneg=self.nneg, yForm=self.yForm)
        sub.optimizer, sub.steps = self.optimizer, dict(self.steps)
        return subModel(self, sub, cols)

    def getModel_h(self, cols: Optional[Sequence[int]] = None, h: int = 1):
        """
        Builds the single-horizon `ARX` (method "ridge", the same penalty and scaling) at horizon h
        on the columns `cols` (ScalaTion's inherited `ARX_D.getModel_h`).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the design matrix with the selected terms (defaults to all).
        h : int, default=1
            The horizon.
        """
        from modeling.forecasting.feature_selection_d import subModel_h
        cols = self.mcols if cols is None else list(cols)
        return subModel_h(self, cols, h, "ridge", {**self.args, "lambda": self.lam, "scaled": self.scaled})
