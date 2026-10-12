"""
NARX Formula Terms with Free Nonlinear Parameters

Defines the term language of the NARX formula models: lagged variables, transformed
terms, powers, products, seasonal waves, and the terms with FREE nonlinear parameters
(a shift inside a log, an exponent, a saturation constant, a threshold, a decay, ...).
A `Formula` is a response variable plus a list of distinct terms; its `columns` build
the design matrix from a named-column dataset, so `NARX_SR_D.from_formula` can run
parse -> fit -> score on any formula string.

The Python counterpart of ScalaTion's `ARX_Formula` and `NARX_Formula` (same labels,
same canonical key, same parameter roles and bounds), written for numpy.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------------------------
#  Free nonlinear parameters
# ------------------------------------------------------------------------------------

class ParRole(Enum):
    """
    The role a free parameter plays; the role decides its default start value and bounds
    (see `Par.resolve`).

    FREE    fully given by the user, no defaults.
    SHIFT   x + s inside log, pow, ...: keeps the argument positive on the training rows,
            at most one training range above that (a larger shift makes log(s + x) linear).
    PHASE   x + s inside sin or cos: in [-pi, pi].
    RATE    a * x inside exp, sin, cos, tanh: 1 / max|x| and [0.01, 20] / max|x|.
    POWER   the exponent of pow: 1.5 in [0.5, 3] (an exponent near 1 is a plain lag).
    FREQ    the frequency of wave in cycles per time unit: 0.1 in [0.01, 0.5].
    DECAY   the decay of ema in (0, 1): 0.5 in [0.05, 0.9].
    THRESH  the threshold of hinge and trans: the middle of the training range, in it.
    WIDTH   the width of trans: a tenth of the training range, in [1/100, 1] of it.
    HALF    the half-saturation of sat: the middle of the training range, in [1/20, 1] of it.
    CAP     the cap of dq: half the largest |x|, in [0.05, 2] of it.
    """
    FREE = "Free"
    SHIFT = "Shift"
    PHASE = "Phase"
    RATE = "Rate"
    POWER = "Power"
    FREQ = "Freq"
    DECAY = "Decay"
    THRESH = "Thresh"
    WIDTH = "Width"
    HALF = "Half"
    CAP = "Cap"


class LabelMode(Enum):
    """How a parameter prints inside a term label: `?` (canonical key), `?w` (a proposal,
    the current value as the start value) or the plain value (a fitted formula)."""
    SLOT = 0
    START = 1
    VALUE = 2


#: Default (start, lower bound, upper bound) per role for the roles that need no data.
#: These are ScalaTion's rev3.2 ranges; the pre-rev3.2 ones were POWER (1.5, 1.1, 4.0)
#: and DECAY (0.5, 0.01, 0.99).
PAR_DEFAULTS: Dict[ParRole, tuple] = {
    ParRole.POWER: (1.5, 0.5, 3.0),
    ParRole.FREQ: (0.1, 0.01, 0.5),
    ParRole.DECAY: (0.5, 0.05, 0.9),
}

#: A SHIFT keeps the argument this fraction of the training range above zero.
SHIFT_MARGIN = 0.01


def fmt(v: float) -> str:
    """
    Formats a parameter value with four significant digits (the format of term labels).

    Parameters
    ----------
    v : float
        The value.

    Returns
    -------
    str
        The formatted value, e.g., "1.5", "0.3595", "9.99e+04".
    """
    return f"{v:.4g}"


class Par:
    """
    A free nonlinear parameter of a term: its role, start value, bounds and current value.

    Attributes
    ----------
    role : ParRole
        The role (decides the defaults).
    w0 : float
        The start value (NaN until resolved).
    lb, ub : float
        The bounds (NaN until resolved; equal bounds make a constant).
    value : float
        The current value (the start value until fitted).
    fitted : bool
        Whether `set` was called (the labels then show the value as fitted).

    Notes
    -----
    A `Par` is shared by every term built from one lag range (`pow(x~(3,1), ?)` is three
    terms with ONE exponent) and is mutable: fitting a formula changes its parameters.
    `fresh` returns an independent copy for a formula built from another's terms.
    """

    def __init__(self, role: ParRole, w0: float = math.nan, lb: float = math.nan,
                 ub: float = math.nan) -> None:
        self.role = role
        self.w0 = float(w0)
        self.lb = float(lb)
        self.ub = float(ub)
        self.value = self.w0
        self.fitted = False
        self._resolved = False
        if role in PAR_DEFAULTS:                                   # table roles need no data
            self._fill(*PAR_DEFAULTS[role])

    # -- construction helpers -------------------------------------------------------

    @staticmethod
    def fixed(v: float) -> "Par":
        """A constant written in a parameter position (equal bounds)."""
        p = Par(ParRole.FREE, v, v, v)
        p._resolved = True
        return p

    def fresh(self) -> "Par":
        """
        Returns an independent copy: the same role and bounds, starting at the fitted
        value when this parameter was fitted (a term taken from a fitted equation starts
        its next fit where the data put it), otherwise at the start value.
        """
        p = Par(self.role, self.value if self.fitted else self.w0, self.lb, self.ub)
        p._resolved = self._resolved
        return p

    def reset(self) -> None:
        """Forgets the fit: the value goes back to the start value."""
        self.value = self.w0
        self.fitted = False

    # -- state ----------------------------------------------------------------------

    @property
    def free(self) -> bool:
        """Whether a fit may move this parameter (bounds not equal, or not yet resolved)."""
        return not (self._resolved and self.lb == self.ub)

    @property
    def resolved(self) -> bool:
        return self._resolved

    def update(self, v: float) -> None:
        """Sets the value during a fit (the labels keep showing the start value)."""
        self.value = float(v)

    def set(self, v: float) -> None:
        """Sets the value after a fit (the labels now show it)."""
        self.value = float(v)
        self.fitted = True

    def _fill(self, w: float, lo: float, hi: float) -> None:
        if math.isnan(self.lb):
            self.lb = lo
        if math.isnan(self.ub):
            self.ub = hi
        if math.isnan(self.w0):
            self.w0 = w
        if self.lb > self.ub:
            raise ValueError(f"Par ({self.role.value}): lower bound {self.lb} > upper bound {self.ub}")
        self.w0 = max(self.lb, min(self.ub, self.w0))      # keep the start inside the box
        if not self.fitted:
            self.value = self.w0
        self._resolved = True

    def resolve(self, x: np.ndarray, tr_size: int = -1) -> None:
        """
        Fills the unset start value and bounds from the role and the source column's
        training rows (the data decide what "a shift" or "a threshold" can be).

        Parameters
        ----------
        x : np.ndarray, shape (n_samples,)
            The source column the parameter acts on.
        tr_size : int, default=-1
            The training rows considered (-1 => 80 % of the rows).
        """
        if self._resolved:
            return
        m = min(tr_size, len(x)) if tr_size > 0 else int(0.8 * len(x))
        xt = x[:m]
        lo, hi = float(np.min(xt)), float(np.max(xt))
        rng = hi - lo if hi > lo else max(abs(hi), 1.0)
        big = max(abs(lo), abs(hi)) if max(abs(lo), abs(hi)) > 0.0 else 1.0
        r = self.role
        if r is ParRole.SHIFT:
            lo_s = -lo + SHIFT_MARGIN * rng                       # the argument stays positive
            self._fill(max(0.0, lo_s), lo_s, lo_s + rng)
        elif r is ParRole.PHASE:
            self._fill(0.0, -math.pi, math.pi)
        elif r is ParRole.RATE:
            self._fill(1.0 / big, 0.01 / big, 20.0 / big)
        elif r is ParRole.THRESH:
            self._fill(0.5 * (lo + hi), lo, hi)
        elif r is ParRole.WIDTH:
            self._fill(0.1 * rng, 0.01 * rng, rng)
        elif r is ParRole.HALF:
            self._fill(max(0.5 * (lo + hi), 0.05 * rng), 0.05 * rng, max(hi, rng))
        elif r is ParRole.CAP:
            self._fill(0.5 * big, 0.05 * big, 2.0 * big)
        elif r in PAR_DEFAULTS:
            self._fill(*PAR_DEFAULTS[r])
        else:
            self._fill(0.0, -math.inf, math.inf)

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        """The parameter as it prints inside a term label."""
        if not self.free and self._resolved:
            return fmt(self.value)                                 # a constant
        if mode is LabelMode.SLOT:
            return "?"
        if mode is LabelMode.START or not self.fitted:
            return "?" if math.isnan(self.value) else "?" + fmt(self.value)
        return fmt(self.value)

    def __repr__(self) -> str:
        return f"Par({self.role.value}, w0={self.w0}, lb={self.lb}, ub={self.ub}, value={self.value})"


# -- role constructors (the `?` slots of the parser; also usable by hand) --------------

def par(role: ParRole, w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:
    """A free parameter of the given role (NaN => filled from the data by `resolve`)."""
    return Par(role, w0, lb, ub)


def shift(w0: float = math.nan) -> Par:
    return Par(ParRole.SHIFT, w0)


def power(w0: float = math.nan) -> Par:
    return Par(ParRole.POWER, w0)


def rate(w0: float = math.nan) -> Par:
    return Par(ParRole.RATE, w0)


def phase(w0: float = math.nan) -> Par:
    return Par(ParRole.PHASE, w0)


def freq(w0: float = math.nan) -> Par:
    return Par(ParRole.FREQ, w0)


def decay(w0: float = math.nan) -> Par:
    return Par(ParRole.DECAY, w0)


def thresh(w0: float = math.nan) -> Par:
    return Par(ParRole.THRESH, w0)


def width(w0: float = math.nan) -> Par:
    return Par(ParRole.WIDTH, w0)


def half(w0: float = math.nan) -> Par:
    return Par(ParRole.HALF, w0)


def cap(w0: float = math.nan) -> Par:
    return Par(ParRole.CAP, w0)


# ------------------------------------------------------------------------------------
#  Terms
# ------------------------------------------------------------------------------------

def lagv(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Lags a series by `lag` steps, padding the first `lag` rows with the first value
    (the convention of `Util.build_matrix.build_lagged_matrix`).

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        The series.
    lag : int
        The lag (0 => the series itself).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Row t holds x[t - lag] (x[0] for t < lag).
    """
    if lag <= 0:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.concatenate((np.full(lag, x[0]), x[:-lag]))


#: The parameter-free functions a term may apply (name -> vectorized function).
FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log": np.log,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "sq": np.square,
    "exp": np.exp,
    "expm1": np.expm1,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
}


class Term(ABC):
    """
    A term of a formula: one column of the design matrix, built from a named-column
    dataset (a dict name -> np.ndarray of equal length).
    """

    @abstractmethod
    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        """The term as it prints in a formula (`y~3`, `log(0.3 + icu~1)`, ...)."""

    @property
    @abstractmethod
    def vars(self) -> set:
        """The dataset variables the term reads."""

    @property
    @abstractmethod
    def lag(self) -> int:
        """The largest lag the term reads (0 for a wave)."""

    @abstractmethod
    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """The term's column over the rows of the dataset, at the parameters' current values."""

    def params(self) -> List[Par]:
        """The parameters the term carries, in order of appearance."""
        return []

    def sources(self) -> List["Term"]:
        """The direct sub-terms."""
        return []

    @abstractmethod
    def fresh(self, copies: Dict[int, Par]) -> "Term":
        """A copy with fresh parameters (`copies` maps id(par) -> its copy, so shared
        parameters stay shared)."""

    def _cp(self, p: Par, copies: Dict[int, Par]) -> Par:
        if id(p) not in copies:
            copies[id(p)] = p.fresh()
        return copies[id(p)]

    def __repr__(self) -> str:
        return self.label()

    @property
    def key(self) -> str:
        return self.label(LabelMode.SLOT)


class Var(Term):
    """A lagged variable `name~lag` (lag 0 is the variable itself, used for the response)."""

    def __init__(self, name: str, lag: int = 0) -> None:
        if lag < 0:
            raise ValueError(f"Var: lag {lag} for variable '{name}' must be >= 0")
        self.name = name
        self._lag = lag

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return self.name if self._lag == 0 else f"{self.name}~{self._lag}"

    @property
    def vars(self) -> set:
        return {self.name}

    @property
    def lag(self) -> int:
        return self._lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return lagv(data[self.name], self._lag)

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Var(self.name, self._lag)

    def lags(self, max_lag: int, min_lag: int = 1) -> List["Var"]:
        """The lag range `name~(max_lag, min_lag)` as a list of terms (decreasing lag)."""
        if max_lag < min_lag:
            raise ValueError(f"Var: lag range ({max_lag},{min_lag}) for '{self.name}' must be decreasing")
        return [Var(self.name, l) for l in range(max_lag, min_lag - 1, -1)]


class Transformed(Term):
    """A parameter-free function of a term: `log(x~1)`, `sq(y~2)`, ... (see `FUNCTIONS`)."""

    def __init__(self, source: Term, fn: str) -> None:
        if fn not in FUNCTIONS:
            raise ValueError(f"unknown function '{fn}'")
        self.source = source
        self.fn = fn

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.fn}({self.source.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return FUNCTIONS[self.fn](self.source.column(data))

    def params(self) -> List[Par]:
        return self.source.params()

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Transformed(self.source.fresh(copies), self.fn)


class Shift(Term):
    """A free shift added to a term, `? + x~1`, written inside a function: `log(? + x~1)`."""

    def __init__(self, source: Term, s: Par) -> None:
        self.source = source
        self.s = s

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.s.label(mode)} + {self.source.label(mode)}"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return self.source.column(data) + self.s.value

    def params(self) -> List[Par]:
        return self.source.params() + [self.s]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Shift(self.source.fresh(copies), self._cp(self.s, copies))


class Scale(Term):
    """A free scale multiplying a term, `? * x~1`, written inside a function: `exp(? * x~1)`."""

    def __init__(self, source: Term, a: Par) -> None:
        self.source = source
        self.a = a

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.a.label(mode)} * {self.source.label(mode)}"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return self.a.value * self.source.column(data)

    def params(self) -> List[Par]:
        return self.source.params() + [self.a]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Scale(self.source.fresh(copies), self._cp(self.a, copies))


class Power(Term):
    """A fixed power `pow(x~1, 1.5)`; `safe` is the sign-safe `powr(x~1, q)` = sign(x) |x|^q."""

    def __init__(self, source: Term, p: float, safe: bool = False) -> None:
        self.source = source
        self.p = float(p)
        self.safe = safe

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{'powr' if self.safe else 'pow'}({self.source.label(mode)}, {fmt(self.p)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        x = self.source.column(data)
        return np.sign(x) * np.abs(x) ** self.p if self.safe else np.power(x, self.p)

    def params(self) -> List[Par]:
        return self.source.params()

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Power(self.source.fresh(copies), self.p, self.safe)


class ParPower(Term):
    """A power with a free exponent `pow(x~1, ?)`."""

    def __init__(self, source: Term, p: Par) -> None:
        self.source = source
        self.p = p

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"pow({self.source.label(mode)}, {self.p.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return np.power(self.source.column(data), self.p.value)

    def params(self) -> List[Par]:
        return self.source.params() + [self.p]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return ParPower(self.source.fresh(copies), self._cp(self.p, copies))


class Product(Term):
    """An interaction of two terms `a * b`."""

    def __init__(self, left: Term, right: Term) -> None:
        self.left = left
        self.right = right

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.left.label(mode)} * {self.right.label(mode)}"

    @property
    def vars(self) -> set:
        return self.left.vars | self.right.vars

    @property
    def lag(self) -> int:
        return max(self.left.lag, self.right.lag)

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return self.left.column(data) * self.right.column(data)

    def params(self) -> List[Par]:
        return self.left.params() + self.right.params()

    def sources(self) -> List[Term]:
        return [self.left, self.right]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Product(self.left.fresh(copies), self.right.fresh(copies))


class Wave(Term):
    """One seasonal component sin(w t + phase) at a FIXED angular frequency w (a `wave(k)`
    pair is a sin and a cos at the k-th periodogram frequency)."""

    def __init__(self, w: float, ph: float = 0.0, idx: int = 0) -> None:
        self.w = float(w)
        self.ph = float(ph)
        self.idx = idx

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        kind = "sin" if self.ph == 0.0 else "cos"
        return f"{kind}(w={self.w:.4f})"

    @property
    def vars(self) -> set:
        return set()

    @property
    def lag(self) -> int:
        return 0

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        m = len(next(iter(data.values())))
        return np.sin(self.w * np.arange(m) + self.ph)

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Wave(self.w, self.ph, self.idx)


class ParWave(Term):
    """One seasonal component with a FREE frequency f (cycles per step): sin(2 pi f t + phase)."""

    def __init__(self, f: Par, ph: float = 0.0) -> None:
        self.f = f
        self.ph = float(ph)

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        kind = "sin" if self.ph == 0.0 else "cos"
        return f"{kind}(f={self.f.label(mode)})"

    @property
    def vars(self) -> set:
        return set()

    @property
    def lag(self) -> int:
        return 0

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        m = len(next(iter(data.values())))
        return np.sin(2.0 * math.pi * self.f.value * np.arange(m) + self.ph)

    def params(self) -> List[Par]:
        return [self.f]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return ParWave(self._cp(self.f, copies), self.ph)


class Sat(Term):
    """Saturation `sat(x~1, ?)` = x / (x + k) with a free half-saturation k."""

    def __init__(self, source: Term, k: Par) -> None:
        self.source = source
        self.k = k

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"sat({self.source.label(mode)}, {self.k.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        x = self.source.column(data)
        return x / (x + self.k.value)

    def params(self) -> List[Par]:
        return self.source.params() + [self.k]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Sat(self.source.fresh(copies), self._cp(self.k, copies))


class Hinge(Term):
    """Hinge `hinge(x~1, ?)` = max(0, x - c) with a free threshold c."""

    def __init__(self, source: Term, c: Par) -> None:
        self.source = source
        self.c = c

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"hinge({self.source.label(mode)}, {self.c.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return np.maximum(0.0, self.source.column(data) - self.c.value)

    def params(self) -> List[Par]:
        return self.source.params() + [self.c]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Hinge(self.source.fresh(copies), self._cp(self.c, copies))


class Trans(Term):
    """Smooth transition `trans(x~1, ?, ?)` = 1 / (1 + exp(-(x - c) / w)) with a free
    threshold c and width w; `trans(y~1, ?, ?) * y~(3,1)` is a STAR model."""

    def __init__(self, source: Term, c: Par, w: Par) -> None:
        self.source = source
        self.c = c
        self.w = w

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"trans({self.source.label(mode)}, {self.c.label(mode)}, {self.w.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        x = self.source.column(data)
        z = np.clip(-(x - self.c.value) / self.w.value, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(z))

    def params(self) -> List[Par]:
        return self.source.params() + [self.c, self.w]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Trans(self.source.fresh(copies), self._cp(self.c, copies), self._cp(self.w, copies))


class Ema(Term):
    """Exponentially weighted moving average `ema(x~1, ?)`: e_t = d x_t + (1 - d) e_{t-1}
    (a geometric distributed lag) with a free decay d in (0, 1)."""

    def __init__(self, source: Term, d: Par) -> None:
        self.source = source
        self.d = d

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"ema({self.source.label(mode)}, {self.d.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        x = self.source.column(data)
        d = self.d.value
        e = np.empty_like(x)
        e[0] = x[0]
        for t in range(1, len(x)):
            e[t] = d * x[t] + (1.0 - d) * e[t - 1]
        return e

    def params(self) -> List[Par]:
        return self.source.params() + [self.d]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Ema(self.source.fresh(copies), self._cp(self.d, copies))


class Dq(Term):
    """Dampened quadratic `dq(x~1, ?)` = x |x| / sqrt(1 + x^2 / mx^2) with a free cap mx
    (quadratic for |x| << mx, linear beyond)."""

    def __init__(self, source: Term, mx: Par) -> None:
        self.source = source
        self.mx = mx

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"dq({self.source.label(mode)}, {self.mx.label(mode)})"

    @property
    def vars(self) -> set:
        return self.source.vars

    @property
    def lag(self) -> int:
        return self.source.lag

    def column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        x = self.source.column(data)
        return x * np.abs(x) / np.sqrt(1.0 + (x / self.mx.value) ** 2)

    def params(self) -> List[Par]:
        return self.source.params() + [self.mx]

    def sources(self) -> List[Term]:
        return [self.source]

    def fresh(self, copies: Dict[int, Par]) -> "Term":
        return Dq(self.source.fresh(copies), self._cp(self.mx, copies))


# -- term constructors (the DSL) -------------------------------------------------------

def _each(ts, f):
    return [f(t) for t in ts] if isinstance(ts, (list, tuple)) else f(ts)


def fn(name: str, t):
    """`log(t)`, `sqrt(t)`, ... for a term or a list of terms (a lag range)."""
    return _each(t, lambda x: Transformed(x, name))


def pow_(t, p) -> Term:
    """`pow(t, p)`: a fixed exponent (float) or a free one (`Par`)."""
    if isinstance(p, Par):
        return _each(t, lambda x: ParPower(x, p))
    return _each(t, lambda x: Power(x, float(p)))


def powr(t, p: float):
    """Sign-safe power `powr(t, p)` = sign(t) |t|^p (fixed exponent only)."""
    return _each(t, lambda x: Power(x, float(p), safe=True))


def sat(t, k: Par):
    return _each(t, lambda x: Sat(x, k))


def hinge(t, c: Par):
    return _each(t, lambda x: Hinge(x, c))


def trans(t, c: Par, w: Par):
    return _each(t, lambda x: Trans(x, c, w))


def ema(t, d: Par):
    return _each(t, lambda x: Ema(x, d))


def dq(t, mx: Par):
    return _each(t, lambda x: Dq(x, mx))


def wave(k_or_par, freqs: Optional[Sequence[float]] = None) -> List[Term]:
    """
    Seasonal sin/cos pairs: `wave(k)` gives k pairs at the first k frequencies of `freqs`
    (cycles per step, e.g., the periodogram's top frequencies); `wave(par)` one pair with
    a free frequency.
    """
    if isinstance(k_or_par, Par):
        return [ParWave(k_or_par, 0.0), ParWave(k_or_par, math.pi / 2)]
    k = int(k_or_par)
    if freqs is None or len(freqs) < k:
        raise ValueError(f"wave({k}) needs {k} frequencies, {0 if freqs is None else len(freqs)} given")
    out: List[Term] = []
    for i in range(k):
        w = 2.0 * math.pi * float(freqs[i])
        out += [Wave(w, 0.0, i), Wave(w, math.pi / 2, i)]
    return out


# ------------------------------------------------------------------------------------
#  Formula
# ------------------------------------------------------------------------------------

class TForm:
    """
    A response/column transform applied to every dataset column before the terms are
    built (`TFORM: Log1p` in a formula string) and inverted on the forecasts.
    Supported: "Log1p", "Log", "Sqrt" (parameter-free, so nothing is fitted), with the
    sklearn-like `transform` / `inverse_transform` on 2-D arrays the `Forecaster` base
    expects of `_yForm`.
    """

    KINDS = {
        "log1p": (np.log1p, np.expm1),
        "log": (np.log, np.exp),
        "sqrt": (np.sqrt, np.square),
    }

    def __init__(self, kind: str) -> None:
        k = kind.lower()
        if k not in self.KINDS:
            raise ValueError(f"TFORM '{kind}' unknown (Log1p, Log, Sqrt)")
        self.kind = k[0].upper() + k[1:]
        self._f, self._fi = self.KINDS[k]

    def f(self, x: np.ndarray) -> np.ndarray:
        return self._f(np.asarray(x, dtype=float))

    def fi(self, x: np.ndarray) -> np.ndarray:
        return self._fi(np.asarray(x, dtype=float))

    def fit(self, x):
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fi(x)

    def __repr__(self) -> str:
        return f"TForm({self.kind})"


def distinct(terms: Sequence[Term]) -> List[Term]:
    """Collapses terms with the same canonical label (duplicates would make the design
    matrix singular), keeping the first of each."""
    seen = set()
    out: List[Term] = []
    for t in terms:
        k = t.key
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


class Formula:
    """
    A NARX formula: a response variable, the predictor terms and an optional transform.

    Attributes
    ----------
    resp : Var
        The response variable (lag 0).
    preds : list of Term
        The predictor terms (distinct by canonical label).
    tform : Optional[TForm]
        The transform applied to every column before the terms are built (`TFORM: Log1p`).

    Notes
    -----
    `key` is the canonical identity (parameters as `?`, terms sorted), so two proposals
    that differ only in start or fitted values are one candidate; `formula_str` shows the
    parameters as fitted (or their start values); `proposal_str` shows them as `?w`
    slots, the string a proposer is given back.
    """

    def __init__(self, resp, preds: Sequence[Term], tform=None) -> None:
        self.resp = resp if isinstance(resp, Var) else Var(str(resp))
        self.preds = distinct(list(preds))
        self.tform = TForm(tform) if isinstance(tform, str) else tform

    # -- strings ----------------------------------------------------------------------

    def _str(self, mode: LabelMode) -> str:
        parts: List[str] = []
        seen_free: set = set()
        n_fixed = 0
        for t in self.preds:
            if isinstance(t, ParWave):                             # a free-frequency pair prints once, as wave(?w)
                if id(t.f) not in seen_free:
                    seen_free.add(id(t.f))
                    parts.append(f"wave({t.f.label(mode)})")
            elif isinstance(t, Wave):                              # k fixed pairs print once, as wave(k)
                n_fixed = max(n_fixed, t.idx + 1)
                if t.idx == 0 and t.ph == 0.0:
                    parts.append("wave(K)")
            else:
                parts.append(t.label(mode))
        parts = [p if p != "wave(K)" else f"wave({n_fixed})" for p in parts]
        s = f"{self.resp.name} ~= " + " + ".join(parts)
        return s + (f" TFORM: {self.tform.kind}" if self.tform is not None else "")

    @property
    def key(self) -> str:
        labels = sorted(t.key for t in self.preds)
        s = f"{self.resp.name} ~= " + " + ".join(labels)
        return s + (f" TFORM: {self.tform.kind}" if self.tform is not None else "")

    @property
    def formula_str(self) -> str:
        return self._str(LabelMode.VALUE)

    @property
    def proposal_str(self) -> str:
        return self._str(LabelMode.START)

    def __repr__(self) -> str:
        return self.formula_str

    # -- terms and parameters ---------------------------------------------------------

    @property
    def labels(self) -> List[str]:
        return [t.label() for t in self.preds]

    @property
    def n_terms(self) -> int:
        return len(self.preds)

    @property
    def vars(self) -> set:
        v: set = set()
        for t in self.preds:
            v |= t.vars
        return v

    @property
    def max_lag(self) -> int:
        return max((t.lag for t in self.preds), default=0)

    def params(self) -> List[Par]:
        """The distinct parameters of the formula, in order of appearance (a parameter
        shared by a lag range appears once)."""
        out: List[Par] = []
        seen = set()
        for t in self.preds:
            for p in t.params():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    @property
    def n_params(self) -> int:
        return len(self.params())

    @property
    def n_free(self) -> int:
        """The number of distinct FREE parameters (the ones a fit searches)."""
        return sum(1 for p in self.params() if p.free)

    def fresh(self) -> "Formula":
        """A copy with its own parameters (fitting it never touches this one)."""
        copies: Dict[int, Par] = {}
        return Formula(self.resp, [t.fresh(copies) for t in self.preds], self.tform)

    def reset(self) -> None:
        for p in self.params():
            p.reset()

    # -- data ---------------------------------------------------------------------------

    def prepare(self, data: Dict[str, np.ndarray], tr_size: int = -1) -> Dict[str, np.ndarray]:
        """
        Checks the formula against a dataset and returns the (transformed) columns.

        Parameters
        ----------
        data : dict of str -> np.ndarray
            The named columns, all of the same length.
        tr_size : int, default=-1
            The training rows (-1 => 80 % of the rows); the parameters are resolved on them.

        Returns
        -------
        dict of str -> np.ndarray
            The columns the terms are built on (transformed when `tform` is set).

        Raises
        ------
        ValueError
            For a missing variable, a length mismatch, a lag beyond the series, or no terms.
        """
        if not self.preds:
            raise ValueError("from_formula: no predictor terms (the mean model has no design matrix)")
        if self.resp.name not in data:
            raise ValueError(f"from_formula: response '{self.resp.name}' not in dataset")
        y = np.asarray(data[self.resp.name], dtype=float)
        for nm in self.vars:
            if nm not in data:
                raise ValueError(f"from_formula: variable '{nm}' not in dataset")
            if len(data[nm]) != len(y):
                raise ValueError(f"from_formula: '{nm}' length {len(data[nm])} != {len(y)}")
        if self.max_lag >= len(y):
            raise ValueError(f"from_formula: lag {self.max_lag} >= series length {len(y)}")
        if self.tform is None:
            cols = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        else:
            cols = {k: self.tform.f(v) for k, v in data.items()}
        self.resolve(cols, tr_size)
        return cols

    def resolve(self, data: Dict[str, np.ndarray], tr_size: int = -1) -> None:
        """Resolves every unset parameter from its source column's training rows."""
        for t in self.preds:
            self._resolve_term(t, data, tr_size)

    def _resolve_term(self, t: Term, data: Dict[str, np.ndarray], tr_size: int) -> None:
        for s in t.sources():
            self._resolve_term(s, data, tr_size)
        own = [p for p in t.params() if not any(p is q for s in t.sources() for q in s.params())]
        if own:
            if t.sources():
                x = t.sources()[0].column(data)
            else:
                x = np.arange(len(next(iter(data.values()))), dtype=float)
            for p in own:
                p.resolve(x, tr_size)

    def columns(self, data: Dict[str, np.ndarray], check: bool = True) -> np.ndarray:
        """
        The design matrix: one column per term at the parameters' current values.

        Parameters
        ----------
        data : dict of str -> np.ndarray
            The (prepared) columns.
        check : bool, default=True
            Whether to reject a non-finite column (log of 0, exp of a large value).

        Returns
        -------
        np.ndarray, shape (n_samples, n_terms)
        """
        X = np.column_stack([t.column(data) for t in self.preds])
        if check:
            bad = ~np.isfinite(X).all(axis=0)
            if bad.any():
                j = int(np.argmax(bad))
                raise ValueError(f"term '{self.preds[j].label()}' is not finite (e.g., log of 0 or exp of a large value)")
        return X

    def redundant(self) -> List[Par]:
        """
        The parameters the linear fit already covers (kept at their start value): a top-level
        shift or scale, a shift inside exp (e^s factors out), a scale inside log (log a adds a
        constant) or under a free power, a scale of a product factor or inside sat, and the
        affine parameters under hinge, trans and ema (their threshold, width and coefficient
        cover them).
        """
        out: List[Par] = []
        for t in self.preds:
            out += self._redundant(t)
        return out

    @staticmethod
    def _affine(t: Term) -> List[Par]:
        if isinstance(t, Shift):
            return [t.s] + Formula._affine(t.source)
        if isinstance(t, Scale):
            return [t.a] + Formula._affine(t.source)
        return []

    @staticmethod
    def _redundant(t: Term) -> List[Par]:
        if isinstance(t, Shift):
            return [t.s]
        if isinstance(t, Scale):
            return [t.a]
        if isinstance(t, Transformed):
            s = t.source
            if t.fn in ("exp", "expm1") and isinstance(s, Shift):
                return [s.s]
            if t.fn == "log" and isinstance(s, Scale):
                return [s.a]
            if t.fn == "log" and isinstance(s, Shift) and isinstance(s.source, Scale):
                return [s.source.a]
            return []
        if isinstance(t, ParPower):
            s = t.source
            if isinstance(s, Scale):
                return [s.a]
            if isinstance(s, Shift) and isinstance(s.source, Scale):
                return [s.source.a]
            return []
        if isinstance(t, Product):
            out = []
            if isinstance(t.left, Scale):
                out.append(t.left.a)
            if isinstance(t.right, Scale):
                out.append(t.right.a)
            return out
        if isinstance(t, Sat) and isinstance(t.source, Scale):
            return [t.source.a]
        if isinstance(t, (Hinge, Trans, Ema)):
            return Formula._affine(t.source)
        return []

    def lag_hp(self) -> Dict[str, int]:
        """The lag hyper-parameters of the formula: p (largest response lag), q (largest
        exogenous lag), n_exo (exogenous variables used)."""
        yn = self.resp.name
        p = max((t.lag for t in self.preds if t.vars == {yn}), default=0)
        q = max((t.lag for t in self.preds if t.vars and t.vars != {yn}), default=0)
        n_exo = len(self.vars - {yn})
        return {"p": p, "q": q, "n_exo": n_exo}
