"""
ARX_Formula: Formula Terms and Formulas for Linear ARX Models

The term language of the formula-driven ARX models -- the Python counterpart of ScalaTion's
`ARX_Formula.scala` (same term classes, labels, canonical key and grammar):

    FormulaVar       y~3               a lagged variable (lag 0 is the variable itself, the response)
    TransformedTerm  log1p(icu~1)      a parameter-free function of a term (`FUNCTIONS`)
    PowerTerm        pow(y~1, 1.5)     a fixed power; powr(y~1, 0.5) the sign-safe sign(x) |x|^p
    ProductTerm      y~1 * icu~1       an interaction of two terms
    WaveTerm         sin(w=0.1234)     one seasonal component at a fixed angular frequency

An `ARX_Formula` is a response variable plus distinct predictor terms (and an optional
`TransformT` applied to every base column, written TFORM: Log1p); `matrix` builds the design
matrix from a named-column dataset, `fit` / `fit_D` fit the `ARX` / `ARX_D` models with one
column per term.  The DSL mirrors ScalaTion's:

    from modeling.forecasting.arx_formula import FormulaVar as V, log1p, pow, wave
    y, icu = V("y"), V("icu")
    f = ARX_Formula(y, y.lags(3, 1) + [log1p(icu.lag(1))] + [y.lag(1) * icu.lag(1)])
    f.formulaStr   # 'y ~= y~3 + y~2 + y~1 + log1p(icu~1) + y~1 * icu~1'

The nonlinear terms with free parameters (shift, exponent, threshold, ...) extend these in
`modeling/forecasting/nonlinear/narx_formula.py` (ScalaTion's `NARX_Formula.scala`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-17
Version: 2.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from Util.transform import HP as TRANSFORM_HP, TransformT, transform_t
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.forecaster import trSize as tr_size_of


class LabelMode(Enum):
    """
    How a free parameter prints inside a term label (ScalaTion's `Par.LabelMode`): the plain value
    once fitted (VALUE), `?` (SLOT, the canonical key) or `?w` with the current value as the
    start value (SLOT_VALUE, a proposal).  Parameter-free terms print the same in every mode.
    """
    VALUE = 0
    SLOT = 1
    SLOT_VALUE = 2



def fmt(v: float) -> str:
    """A number in a term label: four significant digits (ScalaTion's `Par.fmt`, "%.4g")."""
    return f"{v:.4g}"


def lagv(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Lags a series by `lag` steps, padding the first `lag` rows with the first value (ScalaTion's
    `VectorD.lagv`, the convention of `makeMatrix4L`).

    Parameters
    ----------
    x : np.ndarray, shape (m,)
        The series.
    lag : int
        The lag (0 => the series itself).

    Returns
    -------
    np.ndarray, shape (m,)
        Row t holds x[t - lag] (x[0] for t < lag).
    """
    x = np.asarray(x, dtype=float)
    if lag <= 0:
        return x
    return np.concatenate((np.full(lag, x[0]), x[:-lag]))


def _dq(x: np.ndarray, mx: float = TRANSFORM_HP["mx"]) -> np.ndarray:
    """The dampened quadratic x |x| / sqrt(1 + (x / mx)^2) (ScalaTion's `DampenedQuad.f`)."""
    return x * np.abs(x) / np.sqrt(1.0 + (x / mx) ** 2)


#: The unary functions of the grammar, by name (ScalaTion's `ARX_Formula.functions`): adding one
#: here adds it to the DSL, the parser and the grammar text at once.
FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log": np.log,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "sq": np.square,
    "dq": _dq,
    "exp": np.exp,
    "expm1": np.expm1,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
}


# ---------------------------------------------------------------------------------------------
#  Terms
# ---------------------------------------------------------------------------------------------

class Term(ABC):
    """
    A term of a formula: one column of the design matrix, built from a named-column dataset
    (a dict name -> np.ndarray of equal length).  ScalaTion's `Term` trait.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the variable the term is built on."""

    @abstractmethod
    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        """The term as it prints in a formula (`y~3`, `log1p(icu~1)`, `pow(y~1, 1.5)`, ...)."""

    @property
    @abstractmethod
    def lag(self) -> int:
        """The largest lag the term reads (0 for a wave)."""

    @property
    def vars(self) -> set:
        """The dataset variables the term reads."""
        return {self.name}

    @abstractmethod
    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        """The term's column over the rows of the dataset (at the parameters' current values)."""

    def params(self) -> list:
        """The free parameters the term carries, in order of appearance (none here)."""
        return []

    def sources(self) -> List["Term"]:
        """The direct sub-terms."""
        return []

    def fresh(self, copies: Dict[int, object]) -> "Term":
        """A copy with fresh parameters (`copies` maps id(par) -> its copy, so shared parameters
        stay shared); parameter-free terms return themselves."""
        return self

    @property
    def key(self) -> str:
        """The canonical label (free parameters as `?`)."""
        return self.label(LabelMode.SLOT)

    def __mul__(self, other: Union["Term", Sequence["Term"]]):
        """`a * b` is the interaction ProductTerm(a, b); `a * [b1, b2]` one per term of the list."""
        if isinstance(other, Term):
            return ProductTerm(self, other)
        return [ProductTerm(self, t) for t in other]

    def __add__(self, other: Union["Term", Sequence["Term"]]) -> List["Term"]:
        """`a + b` starts a list of terms (the right-hand side of a formula)."""
        return [self] + (list(other) if not isinstance(other, Term) else [other])

    def __radd__(self, other: Sequence["Term"]) -> List["Term"]:
        return list(other) + [self]

    def __repr__(self) -> str:
        return self.label()


class FormulaVar(Term):
    """A lagged variable `name~lag` (lag 0 is the variable itself, used for the response)."""

    def __init__(self, name: str, lag: int = 0) -> None:
        if lag < 0:
            raise ValueError(f"FormulaVar: lag {lag} for variable '{name}' must be >= 0")
        self._name = name
        self._lag = lag

    @property
    def name(self) -> str:
        return self._name

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return self._name if self._lag == 0 else f"{self._name}~{self._lag}"

    @property
    def lag(self) -> int:
        return self._lag

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        return lagv(dataset[self._name], self._lag)

    def lagged(self, l: int) -> "FormulaVar":
        """The variable lagged l steps (ScalaTion's `x ~ l`)."""
        return FormulaVar(self._name, l)

    def lags(self, max_lag: int, min_lag: int = 1) -> List["FormulaVar"]:
        """The lag range `name~(max_lag, min_lag)` as a list of terms, lag decreasing (ScalaTion's `x ~ (h, l)`)."""
        if max_lag < min_lag:
            raise ValueError(f"FormulaVar: lag range ({max_lag},{min_lag}) for '{self._name}' must be decreasing")
        return [FormulaVar(self._name, l) for l in range(max_lag, min_lag - 1, -1)]

    def __invert__(self):
        raise TypeError("write x.lagged(k) or x.lags(h, l) for the lags of a variable")

    def __eq__(self, other) -> bool:
        return isinstance(other, FormulaVar) and other._name == self._name and other._lag == self._lag

    def __hash__(self) -> int:
        return hash((self._name, self._lag))


class TransformedTerm(Term):
    """A parameter-free function of a term: `log(x~1)`, `sq(y~2)`, ... (`FUNCTIONS`)."""

    def __init__(self, source: Term, op: str) -> None:
        if op not in FUNCTIONS:
            raise ValueError(f"unknown function '{op}'")
        self.source = source
        self.op = op

    @property
    def name(self) -> str:
        return self.source.name

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.op}({self.source.label(mode)})"

    @property
    def lag(self) -> int:
        return self.source.lag

    @property
    def vars(self) -> set:
        return self.source.vars

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        return FUNCTIONS[self.op](self.source.column(dataset))

    def params(self) -> list:
        return self.source.params()

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, object]) -> Term:
        return TransformedTerm(self.source.fresh(copies), self.op)


class PowerTerm(Term):
    """A fixed power `pow(x~1, 1.5)`; with `safe`, the sign-safe `powr(x~1, q)` = sign(x) |x|^q."""

    def __init__(self, source: Term, p: float, safe: bool = False) -> None:
        self.source = source
        self.p = float(p)
        self.safe = safe

    @property
    def name(self) -> str:
        return self.source.name

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{'powr' if self.safe else 'pow'}({self.source.label(mode)}, {fmt(self.p)})"

    @property
    def lag(self) -> int:
        return self.source.lag

    @property
    def vars(self) -> set:
        return self.source.vars

    def f(self, x: np.ndarray) -> np.ndarray:
        """The power applied element-wise."""
        return np.sign(x) * np.abs(x) ** self.p if self.safe else np.power(x, self.p)

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        return self.f(self.source.column(dataset))

    def params(self) -> list:
        return self.source.params()

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, object]) -> Term:
        return PowerTerm(self.source.fresh(copies), self.p, self.safe)


class ProductTerm(Term):
    """An interaction of two terms `a * b`."""

    def __init__(self, left: Term, right: Term) -> None:
        self.left = left
        self.right = right

    @property
    def name(self) -> str:
        return self.left.name

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.left.label(mode)} * {self.right.label(mode)}"

    @property
    def lag(self) -> int:
        return max(self.left.lag, self.right.lag)

    @property
    def vars(self) -> set:
        return self.left.vars | self.right.vars

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        return self.left.column(dataset) * self.right.column(dataset)

    def params(self) -> list:
        return self.left.params() + self.right.params()

    def sources(self) -> List[Term]:
        return [self.left, self.right]

    def fresh(self, copies: Dict[int, object]) -> Term:
        return ProductTerm(self.left.fresh(copies), self.right.fresh(copies))


class WaveTerm(Term):
    """
    One seasonal component sin(w t + phase) at a FIXED angular frequency w; a `wave(k)` pair is
    a sin (phase 0) and a cos (phase pi/2) at the k-th periodogram frequency (`idx` = k - 1).
    """

    def __init__(self, w: float, phase: float = 0.0, idx: int = 0) -> None:
        self.w = float(w)
        self.phase = float(phase)
        self.idx = idx

    @property
    def name(self) -> str:
        kind = "sin" if self.phase == 0.0 else "cos"
        return f"{kind}(w={self.w:.4f})"

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return self.name

    @property
    def lag(self) -> int:
        return 0

    @property
    def vars(self) -> set:
        return set()

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        m = len(next(iter(dataset.values())))
        return np.sin(self.w * np.arange(m) + self.phase)


class AvgTerm(Term):
    """
    The MEAN of the lags lo..hi of a variable, `avg(x~(hi,lo))`: one column for a block of
    consecutive lags, the lag grouping that keeps the information of a dense lag structure while
    removing the collinearity of its near-copies (yl2, yl3, yl4 correlate above 0.95 on a smooth
    series; their mean is one interpretable "recent past" regressor).  Parameter-free; the discrete
    cousin of the exponential moving average `ema (x, decay)` of the nonlinear terms.
    """

    def __init__(self, name: str, hi: int, lo: int) -> None:
        if lo < 1 or hi < lo:
            raise ValueError(f"AvgTerm: lag range ({hi},{lo}) for variable '{name}' must satisfy 1 <= lo <= hi")
        self._name, self.hi, self.lo = name, int(hi), int(lo)

    @property
    def name(self) -> str:
        return self._name

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"avg({self._name}~({self.hi},{self.lo}))"

    @property
    def lag(self) -> int:
        return self.hi

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        x = dataset[self._name]
        return np.mean([lagv(x, l) for l in range(self.lo, self.hi + 1)], axis=0)

    def __eq__(self, other) -> bool:
        return isinstance(other, AvgTerm) and (other._name, other.hi, other.lo) == (self._name, self.hi, self.lo)

    def __hash__(self) -> int:
        return hash(("avg", self._name, self.hi, self.lo))


# ---------------------------------------------------------------------------------------------
#  The DSL (ScalaTion's top-level `log`, `pow`, `wave`, ...)
# ---------------------------------------------------------------------------------------------

def avg(x: Union[FormulaVar, str], hi: int, lo: int = 1) -> AvgTerm:
    """The mean of the lags lo..hi of a variable: `avg(y, 4, 2)` = (y~4 + y~3 + y~2) / 3."""
    return AvgTerm(x.name if isinstance(x, FormulaVar) else str(x), hi, lo)


def _each(ts, f):
    """Applies f to a term or to every term of a list (a lag range)."""
    return [f(t) for t in ts] if isinstance(ts, (list, tuple)) else f(ts)


def fn(name: str, t):
    """`log(t)`, `sqrt(t)`, ... by name, for a term or a list of terms (a lag range)."""
    return _each(t, lambda x: TransformedTerm(x, name))


def log(t):   return fn("log", t)
def log1p(t): return fn("log1p", t)
def sqrt(t):  return fn("sqrt", t)
def sq(t):    return fn("sq", t)
def exp(t):   return fn("exp", t)
def expm1(t): return fn("expm1", t)
def sin(t):   return fn("sin", t)
def cos(t):   return fn("cos", t)
def tanh(t):  return fn("tanh", t)


def pow(t, p: float):
    """`pow(t, p)`: a fixed exponent (a `Par` exponent is `nonlinear.narx_formula.pow`)."""
    return _each(t, lambda x: PowerTerm(x, float(p)))



def powr(t, q: float):
    """Sign-safe power `powr(t, q)` = sign(t) |t|^q (fixed exponent only)."""
    return _each(t, lambda x: PowerTerm(x, float(q), safe=True))


def wave_pair(freq: float, idx: int = 0) -> List[Term]:
    """One sin/cos pair at the frequency `freq` (cycles per time unit): ScalaTion's `wave(freq, idx)`."""
    w = 2.0 * math.pi * float(freq)
    return [WaveTerm(w, 0.0, idx), WaveTerm(w, math.pi / 2, idx)]


def wave(k: int, freqs: Optional[Sequence[float]] = None) -> List[Term]:
    """
    Seasonal sin/cos pairs for the k strongest frequencies: `wave(k, freqs)` gives k pairs at the
    first k entries of `freqs` (cycles per time unit, e.g., the periodogram's top frequencies) --
    ScalaTion's `wave(freqs, k)`.
    """
    k = int(k)
    if freqs is None or len(freqs) < k:
        raise ValueError(f"wave({k}) needs {k} frequencies, {0 if freqs is None else len(freqs)} given")
    out: List[Term] = []
    for i in range(k):
        out += wave_pair(freqs[i], i)
    return out


def distinct(preds: Sequence[Term]) -> List[Term]:
    """
    Collapses duplicate terms (same label), which would make the design matrix singular, keeping
    the first occurrence in order (ScalaTion's `ARX_Formula.distinct`).
    """
    seen = set()
    out: List[Term] = []
    for t in preds:
        k = t.label()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


# ---------------------------------------------------------------------------------------------
#  ARX_Formula
# ---------------------------------------------------------------------------------------------

class TermData:
    """
    A formula's dataset as the terms see it (ScalaTion's `ARX_Formula.TermData`).

    Attributes
    ----------
    data : dict of str -> np.ndarray
        The dataset with every base column transformed (when a tFormT was given).
    y : np.ndarray
        The (transformed) response column.
    tForm_y : Transform, optional
        The transform applied to the response (None => none), for inverting forecasts.
    trSize : int
        The number of leading rows the transforms were fit on.
    """

    def __init__(self, data: Dict[str, np.ndarray], y: np.ndarray, tForm_y, trSize: int) -> None:
        self.data, self.y, self.tForm_y, self.trSize = data, y, tForm_y, trSize

    def __iter__(self):
        return iter((self.data, self.y, self.tForm_y, self.trSize))


class ARX_Formula:
    """
    A formula: a response variable, distinct predictor terms and an optional column transform.

    Attributes
    ----------
    resp : FormulaVar
        The response variable (lag 0).
    preds : list of Term
        The predictor terms (distinct by label).
    tFormT : TransformT, optional
        The transform type applied to every base column before the terms are built
        (`TFORM: Log1p`), fitted on the training rows; None => none.

    Notes
    -----
    `key` is the canonical identity (terms sorted), so `y~1 + hosp~2` and `hosp~2 + y~1` collide as
    intended; `formulaStr` is the specification the parser reads back.
    """

    P_MIN = 0.2                                                  # the exponent range of the grammar
    P_MAX = 3.0

    def __init__(self, resp: Union[FormulaVar, str], preds: Sequence[Term],
                 tFormT: Union[TransformT, str, None] = None) -> None:
        self.resp = resp if isinstance(resp, FormulaVar) else FormulaVar(str(resp))
        self.preds = self._distinct(list(preds))
        self.tFormT = transform_t(tFormT) if isinstance(tFormT, str) else tFormT

    @classmethod
    def _distinct(cls, preds: Sequence[Term]) -> List[Term]:
        return distinct(preds)

    # -- strings ----------------------------------------------------------------------------

    def _wave_parts(self, mode: LabelMode):
        """The non-wave labels and the wave part (wave(k) when the pairs are 1..k, else their labels)."""
        waves = [t for t in self.preds if isinstance(t, WaveTerm)]
        rest = [t.label(mode) for t in self.preds if not isinstance(t, WaveTerm)]
        if not waves:
            return rest, []
        idx = sorted({t.idx for t in waves})
        if idx[0] >= 0 and idx == list(range(0, idx[-1] + 1)):
            return rest, [f"wave({idx[-1] + 1})"]
        return rest, [t.label(mode) for t in waves]

    def _str(self, mode: LabelMode = LabelMode.VALUE) -> str:
        rest, wpart = self._wave_parts(mode)
        head = f"{self.resp.name} ~= " + " + ".join(rest + wpart)
        return head if self.tFormT is None else f"{head} TFORM: {self.tFormT.name}"

    @property
    def formulaStr(self) -> str:
        """The formula as a specification string in the grammar (`FormulaParser.parseFormula` rebuilds it)."""
        return self._str(LabelMode.VALUE)

    @property
    def key(self) -> str:
        """The canonical key for de-duplication (labels sorted, plus the transform type)."""
        s = f"{self.resp.name} ~= " + " + ".join(sorted(t.label(LabelMode.SLOT) for t in self.preds))
        return s if self.tFormT is None else f"{s} TFORM: {self.tFormT.name}"

    def __repr__(self) -> str:
        return self.formulaStr

    # -- terms ------------------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        """The terms' labels: the feature names of the matrix (ScalaTion's `ARX_Formula.names (preds)`)."""
        return [t.label() for t in self.preds]

    @property
    def nTerms(self) -> int:
        """The number of predictor terms."""
        return len(self.preds)

    @property
    def vars(self) -> set:
        """The dataset variables the terms read."""
        v: set = set()
        for t in self.preds:
            v |= t.vars
        return v

    @property
    def maxLag(self) -> int:
        """The largest lag used by any term (0 if none)."""
        return max((t.lag for t in self.preds), default=0)

    def fresh(self) -> "ARX_Formula":
        """A copy of the formula (with its own parameters, for a formula with free ones)."""
        copies: Dict[int, object] = {}
        return self.__class__(self.resp, [t.fresh(copies) for t in self.preds], self.tFormT)

    # -- data -------------------------------------------------------------------------------

    def prepare(self, dataset: Dict[str, np.ndarray], trSize: int = -1) -> TermData:
        """
        Checks the formula against the dataset (variables present, lengths equal, lags inside the
        series) and transforms every base column on its own TRAINING slice (as in `rescale`)
        before the terms are evaluated (ScalaTion's `ARX_Formula.prepare`).

        Parameters
        ----------
        dataset : dict of str -> np.ndarray
            The named columns, all of the same length.
        trSize : int, default=-1
            The number of leading rows the transforms are fit on (-1 => `trSize` of the series).

        Returns
        -------
        TermData
            (data, y, tForm_y, trSize).

        Raises
        ------
        ValueError
            For a missing variable, a length mismatch, a lag beyond the series, or no terms.
        """
        if not self.preds:
            raise ValueError("fromTerms: no predictor terms (the mean model has no design matrix)")
        if self.resp.name not in dataset:
            raise ValueError(f"fromTerms: response '{self.resp.name}' not in dataset")
        y = np.asarray(dataset[self.resp.name], dtype=float)
        for nm in sorted(self.vars):
            if nm not in dataset:
                raise ValueError(f"fromTerms: variable '{nm}' not in dataset")
            if len(dataset[nm]) != len(y):
                raise ValueError(f"fromTerms: '{nm}' length {len(dataset[nm])} != {len(y)}")
        if self.maxLag >= len(y):
            raise ValueError(f"fromTerms: lag {self.maxLag} >= series length {len(y)}")

        tr_size = trSize if trSize > 0 else tr_size_of(len(y))
        if tr_size > len(y):
            raise ValueError(f"fromTerms: trSize {tr_size} > series length {len(y)}")
        if self.tFormT is None:
            data = {k: np.asarray(v, dtype=float) for k, v in dataset.items()}
            tForm_y = None
        else:
            data = {}
            for k, v in dataset.items():
                v = np.asarray(v, dtype=float)
                data[k] = self.tFormT.form(v[:tr_size]).f(v)
            tForm_y = self.tFormT.form(y[:tr_size])
        return TermData(data, data[self.resp.name], tForm_y, tr_size)

    def matrix(self, data: Dict[str, np.ndarray], check: bool = True) -> np.ndarray:
        """
        The data/input matrix of the formula: one column per term, full length (first lag values
        padded) and NO intercept column (the regression centers / adds its own).  With `check`,
        a term whose column is not finite is rejected before the regression sees it.

        Parameters
        ----------
        data : dict of str -> np.ndarray
            The dataset as the terms see it (`prepare(...).data`).
        check : bool, default=True
            Whether to reject a non-finite column (log of 0, exp of a large value).

        Returns
        -------
        np.ndarray, shape (m, nTerms)
        """
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):   # a bad column is reported below, once
            X = np.column_stack([t.column(data) for t in self.preds])
        if check:
            bad = ~np.isfinite(X).all(axis=0)
            if bad.any():
                j = int(np.argmax(bad))
                raise ValueError(f"fromTerms: term '{self.preds[j].label()}' is not finite (e.g., log of 0 or exp of a large value)")
        return X

    def lagHP(self) -> Dict[str, int]:
        """
        The lag hyper-parameters of a model built from the formula (ScalaTion's `lagHP`): p (the
        largest lag of the response), px (the largest lag of the exogenous variables), spec = 0, and n_exo (the distinct exogenous variables in the terms).
        """
        yn = self.resp.name
        p = max((t.lag for t in self.preds if t.vars == {yn}), default=0)
        q = max((t.lag for t in self.preds if t.vars and t.vars != {yn}), default=0)
        n_exo = len(self.vars - {yn})
        return {"p": p, "px": q, "spec": 0, "n_exo": n_exo}

    # -- fitting ----------------------------------------------------------------------------

    def fit(self, dataset: Dict[str, np.ndarray], hh: int = 1, args: Any = None,
            tFormT: Optional[TransformT] = None, trSize: int = -1) -> "ARX":
        """
        Fits the recursive `ARX` model with one column per term (ScalaTion's `ARX_Formula.fit (hh,
        tFormT, trSize, scaled)`: `ARX.fromTerms` then `trainNtest` in-sample).

        Parameters
        ----------
        dataset : dict of str -> np.ndarray
            The named columns.
        hh : int, default=1
            The maximum forecasting horizon.
        args : Args or dict, optional
            The settings of the model: method ("ridge" is ScalaTion's `CenteredRidge`), lam (default
            0.1), scaled (the regression standardizes the term columns), nneg, skip, te_ratio.
        tFormT : TransformT, optional
            The transform type of every base column (ScalaTion's `fit (..., tFormT)`); None => the
            formula's own (its "TFORM:" clause).
        trSize : int, default=-1
            The training rows the transforms are fit on (-1 => the training rows, or the fit rows
            while choosing).
        """
        mod = ARX.fromTerms(self, dataset, hh, args, tFormT, trSize)
        mod.trainNtest()
        return mod

    def fit_D(self, dataset: Dict[str, np.ndarray], hh: int = 1, args: Any = None,
              tFormT: Optional[TransformT] = None, trSize: int = -1) -> "ARX_D":
        """
        Fits the direct `ARX_D` model with one column per term (ScalaTion's `ARX_Formula.fit_D`:
        `ARX_D.fromTerms` then `trainNtest` in-sample).  Parameters as in `fit`.
        """
        mod = ARX_D.fromTerms(self, dataset, hh, args, tFormT, trSize)
        mod.trainNtest()
        return mod

    # -- the grammar ------------------------------------------------------------------------

    @staticmethod
    def grammar(maxLag: int = 8, pMin: float = 0.2, pMax: float = 3.0) -> str:
        """The grammar as text (for prompts and documentation), ScalaTion's `ARX_Formula.grammar`."""
        fns = ", ".join(sorted(FUNCTIONS.keys()))
        return (f"formula := y ~= element {{ + element }} [ TFORM: Log1p | Norm ]\n"
                f"element := term | term * term                      (two factors, no wave inside)\n"
                f"term    := x~k | x~(h,l) | FN(term) | pow(term, p) | powr(term, p) | wave(k)\n"
                f"            x~k is x lagged k steps (1 <= k <= {maxLag}), x~(h,l) is x~h, ..., x~l\n"
                f"            FN in {{{fns}}}, {pMin} <= p <= {pMax}\n"
                f"            wave(k) adds sin/cos pairs for the k strongest seasonal frequencies")

