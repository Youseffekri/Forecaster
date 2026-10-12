"""
NARX_SR_D Model: Direct Multi-Horizon NARX Symbolic Regression from a Formula

Implements a DIRECT (one model per horizon) forecaster whose design matrix is built from a
`Formula` -- lagged, transformed and interaction terms, seasonal waves, and terms with FREE
nonlinear parameters (a shift inside a log, an exponent, a saturation constant, a threshold,
a decay, ...).  The coefficients are fitted by ridge regression per horizon (centered, with an
unpenalized intercept); the nonlinear parameters are fitted ONCE, on the rows whose targets lie
in the training rows, by one of three optimizers (the `optimizer` option):

    "varpro"  variable projection: nonlinear least squares on the reduced residual (default)
    "lbfgsb"  L-BFGS-B on the reduced objective, bounds honoured
    "joint"   one least-squares problem over the coefficients and the parameters together

and then held fixed: a rolling validation refits only the coefficients, as `ARX_D` does.  Every
fitted parameter counts as a degree of freedom (`n_free`), so the selection score pays for it.

A formula without free parameters gives the plain direct ARX on its terms (an `ARX_D` whose
columns are the terms), so one class serves the whole formula language.

The Python counterpart of ScalaTion's `NARX_SR_D.fromTerms` (and `ARX_D.fromTerms`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from Util.build_matrix import build_matrix_Y
from modeling.forecasting.forecaster_d import Forecaster_D
from modeling.forecasting.formula import Formula, Par
from modeling.forecasting.formula_parser import parse_formula
from modeling.optimization.varpro import VarPro, ridge_solve


class NARX_SR_D(Forecaster_D):
    """
    Direct multi-horizon NARX symbolic regression on the columns of a formula.

    Attributes
    ----------
    formula : Formula
        The formula the columns come from (its parameters hold the fitted values).
    fname : list of str
        The column (term) labels.
    lam : float
        The ridge penalty on the coefficients (0 => OLS).
    scaled : bool
        Whether the centered columns are standardized before the ridge solve.
    optimizer : str
        The optimizer used for the nonlinear parameters ("varpro", "lbfgsb", "joint").
    n_free : int
        The number of nonlinear parameters that were fitted (degrees of freedom they take).
    w_nl : np.ndarray
        The values of all the formula's parameters (fitted or fixed), in order of appearance.
    steps : dict
        The optimizer's counts of the parameter fit (empty when nothing was fitted).

    See Also
    --------
    Forecaster_D : the base forecasting logic (y, X, Y, Yf, tr_size, rollValidate, ...).
    """

    OPTIMIZERS = ("varpro", "lbfgsb", "joint")

    def __init__(self,
                 args: Dict[str, Any],
                 y: np.ndarray,
                 hh: int,
                 X: np.ndarray,
                 formula: Optional[Formula] = None,
                 fname: Optional[Sequence[str]] = None,
                 lam: float = 1.0,
                 scaled: bool = False,
                 n_free: int = 0,
                 nneg: bool = True,
                 yForm=None) -> None:
        """
        Initializes the model on a GIVEN design matrix (use `from_formula` to build one).

        Parameters
        ----------
        args : dict
            Model configuration ("p", "q", "spec", "skip", ...; see `Forecaster`).
        y : np.ndarray, shape (n_samples,)
            The endogenous series (already transformed when `yForm` is given).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        X : np.ndarray, shape (n_samples, n_terms)
            The design matrix (no constant column: the intercept is implicit).
        formula : Formula, optional
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
            Whether forecasts are clipped at zero (a count series); False for series that
            can be negative (e.g., a temperature).
        yForm : optional
            The transform of `y` (with `inverse_transform`), e.g., a `TForm`; None => none.
        """
        super().__init__(args, y, hh)
        self._set_X(np.asarray(X, dtype=float))
        self._Y = build_matrix_Y(self._y, hh)
        self.formula = formula
        self.fname = list(fname) if fname is not None else \
            (formula.labels if formula is not None else [f"x{j}" for j in range(X.shape[1])])
        self.lam = float(lam)
        self.scaled = scaled
        self.optimizer = "none"
        self.n_free = int(n_free)
        self.steps: Dict[str, Any] = {}
        self._nneg = nneg
        self._yForm = yForm
        self._n_exo = formula.lag_hp()["n_exo"] if formula is not None else 0
        self._method = "ridge"
        self._coef: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None

    # -- factory -----------------------------------------------------------------------------

    @classmethod
    def from_formula(cls,
                     formula: Union[str, Formula],
                     data: Dict[str, np.ndarray],
                     hh: int,
                     args: Optional[Dict[str, Any]] = None,
                     tr_size: int = -1,
                     lam: float = 1.0,
                     scaled: bool = False,
                     optimizer: str = "varpro",
                     max_lag: int = 6,
                     freqs: Optional[Sequence[float]] = None,
                     nneg: bool = True,
                     fit_params: bool = True,
                     **opt_kwargs) -> "NARX_SR_D":
        """
        Builds the model from a formula (string or `Formula`) over a named-column dataset:
        parses, resolves the parameters' defaults on the training rows, fits the free
        parameters on the rows whose targets all lie in the training rows, then builds the
        design matrix at the fitted values.

        Parameters
        ----------
        formula : str or Formula
            E.g., "y ~= y~(3,1) + log(? + icu~1) + hinge(temp~1, ?)".
        data : dict of str -> np.ndarray
            The named columns (the response under its name, e.g., "y").
        hh : int
            The maximum forecasting horizon.
        args : dict, optional
            Model configuration; "skip" defaults to the formula's largest lag.
        tr_size : int, default=-1
            The training rows (-1 => 80 % of the rows, the base class's split).
        lam : float, default=1.0
            The ridge penalty on the coefficients.
        scaled : bool, default=False
            Whether to standardize the columns before the ridge solve (also inside VarPro).
        optimizer : {"varpro", "lbfgsb", "joint"}, default="varpro"
            The optimizer of the nonlinear parameters.
        max_lag : int, default=6
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
            Untrained: call `train()` (or `inSample_Test`, `trainNtest_Test`, `rollValidate`).

        Raises
        ------
        ValueError
            For a formula that does not parse or does not fit the dataset.
        """
        if optimizer not in cls.OPTIMIZERS:
            raise ValueError(f"optimizer '{optimizer}' unknown: {cls.OPTIMIZERS}")
        f = parse_formula(formula, list(data.keys()), max_lag, freqs) if isinstance(formula, str) else formula
        m = len(next(iter(data.values())))
        tr = tr_size if tr_size > 0 else int((1.0 - cls._TE_RATIO) * m)
        if tr < hh:
            raise ValueError(f"from_formula: tr_size {tr} < hh = {hh}: no row has all its targets in the training rows")
        cols = f.prepare(data, tr)                                   # checks, transform, resolve
        y = cols[f.resp.name]

        pars = f.params()
        for p in pars:
            p.reset()
        f.resolve(cols, tr)
        redundant = f.redundant()
        for p in redundant:
            print(f"from_formula: parameter {p.role.value} is covered by the linear fit: kept at its start value")
        free: List[Par] = [p for p in pars if p.free and not any(p is q for q in redundant)]

        steps: Dict[str, Any] = {}
        if free and fit_params:
            fit_rows = slice(0, tr - hh + 1)                          # rows whose targets lie in the training rows
            YY = build_matrix_Y(y, hh)[fit_rows]

            def basis(a: np.ndarray) -> np.ndarray:
                for i, p in enumerate(free):
                    p.update(a[i])
                return f.columns(cols, check=False)[fit_rows]

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

        X = f.columns(cols)
        hp = f.lag_hp()
        a_ = {"p": hp["p"], "q": hp["q"], "spec": 1, "cross": False, "skip": f.max_lag}
        if args:
            a_.update(args)
        mod = cls(a_, y, hh, X, formula=f, lam=lam, scaled=scaled, n_free=len(free) if fit_params else 0,
                  nneg=nneg, yForm=f.tform)
        mod.optimizer = optimizer if free and fit_params else "none"
        mod.steps = steps
        mod._tr_size = tr
        mod._te_size = m - tr
        return mod

    # -- properties --------------------------------------------------------------------------

    @property
    def w_nl(self) -> np.ndarray:
        """The values of the formula's parameters, in order of appearance."""
        return np.array([p.value for p in self.formula.params()]) if self.formula is not None else np.array([])

    @property
    def n_coef(self) -> int:
        """The number of coefficients per horizon, intercept included."""
        return self._X.shape[1] + 1

    @property
    def dfr(self) -> int:
        """The degrees of freedom the model takes per horizon: the predictors plus the fitted
        nonlinear parameters (the base of the parsimony-adjusted scores)."""
        return self._X.shape[1] + self.n_free

    @property
    def formula_str(self) -> str:
        return self.formula.formula_str if self.formula is not None else " + ".join(self.fname)

    def __repr__(self) -> str:
        s = f"NARX_SR_D(n_terms={self._X.shape[1]}, n_free={self.n_free}, lam={self.lam}), hh = {self.hh}, skip = {self.args['skip']}"
        return s + (f", optimizer={self.optimizer}" if self.optimizer != "none" else "")

    # -- training and forecasting --------------------------------------------------------

    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Fits the coefficients of every horizon by ridge regression on the given rows (the
        nonlinear parameters are already fixed in the columns).

        Parameters
        ----------
        Y_ : np.ndarray, shape (n_rows, hh), optional
            The target matrix (defaults to the model's).
        X_ : np.ndarray, shape (n_rows, n_terms), optional
            The design matrix (defaults to the model's).
        """
        if X_ is None:
            X_ = self._X
        if Y_ is None:
            Y_ = self._Y
        xm = X_.mean(axis=0)
        ym = Y_.mean(axis=0)
        Xc = X_ - xm
        if self.scaled:
            sd = Xc.std(axis=0)
            sd[sd == 0.0] = 1.0
            Xc = Xc / sd
        else:
            sd = np.ones(X_.shape[1])
        B = ridge_solve(Xc, Y_ - ym, self.lam) / sd[:, None]
        self._coef = B
        self._intercept = ym - xm @ B
        self._set_params(np.vstack((self._intercept, B)))

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Generates the multi-horizon forecasts for rows t_start to t_end (exclusive) and
        writes them into `Yf` (in the original scale when a transform was used).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
        """
        if self._coef is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = self._X[t_st:t_en] @ self._coef + self._intercept
        y_fcast = self._rectify(y_fcast)
        self._Yf[t_st:t_en, 1:-1] = y_fcast if self._yForm is None else self._yForm.inverse_transform(y_fcast)
        return y_fcast

    def predict_rows(self, rows: np.ndarray) -> np.ndarray:
        """The forecasts of the given rows without touching `Yf` (for the scorer)."""
        return self._rectify(self._X[rows] @ self._coef + self._intercept)
