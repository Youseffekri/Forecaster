"""
NARX_Formula: Formula Terms with FREE Nonlinear Parameters

Extends the linear term language of `arx_formula.py` with terms that hold free nonlinear
parameters (`Par`) -- the Python counterpart of ScalaTion's `nonlinear/NARX_Formula.scala`:

    ShiftTerm     log(? + x~1)        a free shift inside a function (role Shift; Phase inside sin/cos)
    ScaleTerm     exp(? * x~1)        a free rate inside a function (role Rate)
    ParPowerTerm  pow(x~1, ?)         a free exponent (role Power)
    SatTerm       sat(x~1, ?)         saturation x / (x + k), free half-saturation k (Half)
    HingeTerm     hinge(x~1, ?)       max(0, x - c), free threshold c (Thresh)
    TransTerm     trans(x~1, ?, ?)    smooth step 1 / (1 + exp(-(x - c) / w)), threshold c and width w
    EmaTerm       ema(x~1, ?)         exponentially weighted moving average, free decay d (Decay)
    DqTerm        dq(x~1, ?)          dampened quadratic, free cap mx (Cap)
    ParWaveTerm   wave(?)             one sin/cos pair with a free frequency (Freq)

A `Par` has a role that decides its default start value and bounds (from a table, or from the
training rows of the column it acts on, `Par.resolve`); a parameter written as a number is a
constant (equal bounds).  A `NARX_Formula` is an `ARX_Formula` whose terms may carry such
parameters: `params`, `nFree`, the canonical `key` (parameters as ?), `proposalStr` (parameters
as ?w), `fresh` (own copies of the parameters), `resolve`, `redundant` (the parameters the
linear fit already covers) and `fitNL_D` (the `NARX_SR_D` model with the parameters fitted).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from modeling.forecasting.arx_formula import (ARX_Formula, FormulaVar, LabelMode, PowerTerm, ProductTerm, Term,
                                              TransformedTerm, _each, fmt, wave as wave_k)

from modeling.forecasting.forecaster import trSize as tr_size_of

if TYPE_CHECKING:
    from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D


# ---------------------------------------------------------------------------------------------
#  Free nonlinear parameters
# ---------------------------------------------------------------------------------------------

class ParRole(Enum):
    """
    The role a free parameter plays; the role decides its default start value and bounds
    (ScalaTion's `ParRole`, see `Par.resolve`).

    FREE    fully given by the user (start value and bounds), no defaults.
    SHIFT   x + s inside log, pow, ...: keeps the argument positive on the training rows.
    PHASE   x + s inside sin or cos: in [-pi, pi].
    RATE    a * x inside exp, sin, cos, tanh: 1 / max|x| in [0.01, 20] / max|x|.
    POWER   the exponent of pow: 1.5 in [1.1, 4] (`PAR_DEFAULTS`).
    FREQ    the frequency of wave in cycles per time unit: 0.1 in [0.01, 0.5].
    DECAY   the decay of ema in (0, 1): 0.5 in [0.01, 0.99].
    THRESH  the threshold of hinge and trans: the middle of the training range, in it.
    WIDTH   the width of trans: a tenth of the training range, in [1/100, 1] of it.
    HALF    the half-saturation of sat: the middle of the training range, in [1/100, 10] of it.
    CAP     the cap of dq: half the largest |x|, in [0.05, 5] of it.
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


#: Default (start, lower bound, upper bound) per unit-free role (ScalaTion's `Par.defaults`).
PAR_DEFAULTS: Dict[ParRole, Tuple[float, float, float]] = {
    ParRole.POWER: (1.5, 1.1, 4.0),
    ParRole.FREQ: (0.1, 0.01, 0.5),
    ParRole.DECAY: (0.5, 0.01, 0.99),
}

#: A SHIFT keeps the argument this fraction of the training range above zero (`Par.SHIFT_MARGIN`).
SHIFT_MARGIN = 0.01


class Par:
    """
    A free nonlinear parameter of a term: its role, start value, bounds and current value
    (ScalaTion's `Par`).

    Parameters
    ----------
    role : ParRole
        The role (decides the defaults).
    w0 : float, default=nan
        The start value (nan => from the role / the data).
    lb, ub : float, default=nan
        The bounds (nan => from the role / the data); equal bounds make a constant.

    Attributes
    ----------
    value : float
        The current value (the start value until fitted).
    fitted : bool
        Whether `set` was called (the labels then show the value as fitted).

    Notes
    -----
    A `Par` is shared by every term built from one lag range (`pow(x~(3,1), ?)` is three terms
    with ONE exponent) and is mutable: fitting a formula changes its parameters.  `fresh` gives
    an independent copy (same role, numbers as given) for a formula built from another's terms.
    """

    def __init__(self, role: ParRole, w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> None:
        self.role = role
        self._w0_, self._lb_, self._ub_ = float(w0), float(lb), float(ub)     # as given (for `fresh`)
        self.w0, self.lb, self.ub = self._w0_, self._lb_, self._ub_
        self.value = self.w0
        self.fitted = False
        if role is ParRole.FREE:
            if not self.resolved:
                raise ValueError("Par: a Free parameter needs its start value and bounds")
        elif role in PAR_DEFAULTS:                                 # unit-free role: resolve from the table
            self._fill(*PAR_DEFAULTS[role])

    @staticmethod
    def const(v: float) -> "Par":
        """A constant written in a parameter position (equal bounds)."""
        return Par(ParRole.FREE, v, v, v)

    @property
    def resolved(self) -> bool:
        """Whether the start value and both bounds are set."""
        return not (math.isnan(self.w0) or math.isnan(self.lb) or math.isnan(self.ub))

    @property
    def free(self) -> bool:
        """Whether a fit may move this parameter (bounds not equal; unresolved => free)."""
        return self.lb < self.ub if self.resolved else True

    @property
    def fixed(self) -> bool:
        """Whether the parameter is a constant (equal bounds), ScalaTion's `fixed`."""
        return self.resolved and self.lb == self.ub

    def fresh(self) -> "Par":
        """An independent copy with the role and the numbers as given at construction."""
        return Par(self.role, self._w0_, self._lb_, self._ub_)

    def _fill(self, w: float, l: float, u: float) -> None:
        if math.isnan(self.lb):
            self.lb = l
        if math.isnan(self.ub):
            self.ub = u
        if math.isnan(self.w0):
            self.w0 = w
        if self.lb > self.ub:
            raise ValueError(f"Par ({self.role.value}): lower bound {self.lb} > upper bound {self.ub}")
        self.w0 = max(self.lb, min(self.ub, self.w0))                # keep the start inside the box
        if not self.fitted:
            self.value = self.w0

    def resolve(self, x: np.ndarray, trSize: int = -1) -> None:
        """
        Fills the unset start value and bounds from the role and the TRAINING rows of the column
        the parameter acts on (the data decide what "a shift" or "a threshold" can be).

        Parameters
        ----------
        x : np.ndarray, shape (m,)
            The source column the parameter acts on.
        trSize : int, default=-1
            The training rows considered (-1 => `trSize` of the series).
        """
        if self.resolved:
            return
        m = min(trSize, len(x)) if trSize > 0 else tr_size_of(len(x))
        xt = np.asarray(x[:m], dtype=float)
        lo, hi = float(np.min(xt)), float(np.max(xt))
        rng = hi - lo if hi > lo else max(abs(hi), 1.0)
        big = max(abs(lo), abs(hi)) if max(abs(lo), abs(hi)) > 0.0 else 1.0
        r = self.role
        if r is ParRole.SHIFT:
            l = -lo + SHIFT_MARGIN * rng                             # the argument stays positive
            self._fill(max(0.0, l), l, max(hi, l + rng))
        elif r is ParRole.PHASE:
            self._fill(0.0, -math.pi, math.pi)
        elif r is ParRole.RATE:
            self._fill(1.0 / big, 0.01 / big, 20.0 / big)
        elif r is ParRole.THRESH:
            self._fill(0.5 * (lo + hi), lo, hi)
        elif r is ParRole.WIDTH:
            self._fill(0.1 * rng, 0.01 * rng, rng)
        elif r is ParRole.HALF:
            self._fill(max(0.5 * (lo + hi), 0.01 * rng), 0.01 * rng, 10.0 * rng)
        elif r is ParRole.CAP:
            self._fill(0.5 * big, 0.05 * big, 5.0 * big)
        else:
            self._fill(0.0, -math.inf, math.inf)

    def update(self, v: float) -> None:
        """Sets the value during a fit (the labels keep showing the start value)."""
        self.value = float(v)

    def set(self, v: float) -> None:
        """Sets the value after a fit (the labels now show it)."""
        self.value = float(v)
        self.fitted = True

    def reset(self) -> None:
        """Forgets the fit: the value goes back to the start value."""
        self.value = self.w0
        self.fitted = False

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        """
        The parameter as it prints inside a term label (ScalaTion's `Par.label`): a constant prints
        its value; SLOT prints `?`; SLOT_VALUE prints `?w` (w the current value); VALUE prints the
        value once fitted, `?w` before.
        """
        if self.fixed:
            return fmt(self.value)
        if mode is LabelMode.SLOT:
            return "?"
        if mode is LabelMode.SLOT_VALUE or not self.fitted:
            return "?" if math.isnan(self.value) else "?" + fmt(self.value)
        return fmt(self.value)

    def __add__(self, t):
        """`s + term` is the shifted term `ShiftTerm(term, s)` (one per term of a list)."""
        return _each(t, lambda x: ShiftTerm(x, self))

    def __mul__(self, t):
        """`a * term` is the scaled term `ScaleTerm(term, a)` (one per term of a list)."""
        return _each(t, lambda x: ScaleTerm(x, self))

    def __repr__(self) -> str:
        return f"Par({self.role.value}, w0={self.w0}, lb={self.lb}, ub={self.ub}, value={self.value})"

    def __str__(self) -> str:
        return self.label()


# -- role constructors (the `?` slots of the parser; also usable by hand) ------------------------

def par(w0: float, lb: float = -math.inf, ub: float = math.inf) -> Par:
    """A fully specified free parameter (role Free)."""
    return Par(ParRole.FREE, w0, lb, ub)


def shift(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:  return Par(ParRole.SHIFT, w0, lb, ub)
def phase(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:  return Par(ParRole.PHASE, w0, lb, ub)
def rate(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:   return Par(ParRole.RATE, w0, lb, ub)
def power(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:  return Par(ParRole.POWER, w0, lb, ub)
def freq(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:   return Par(ParRole.FREQ, w0, lb, ub)
def decay(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:  return Par(ParRole.DECAY, w0, lb, ub)
def thresh(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par: return Par(ParRole.THRESH, w0, lb, ub)
def width(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:  return Par(ParRole.WIDTH, w0, lb, ub)
def half(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:   return Par(ParRole.HALF, w0, lb, ub)
def cap(w0: float = math.nan, lb: float = math.nan, ub: float = math.nan) -> Par:    return Par(ParRole.CAP, w0, lb, ub)


# ---------------------------------------------------------------------------------------------
#  Parameter terms
# ---------------------------------------------------------------------------------------------

class ParTerm(Term):
    """
    A term built from a source term and free parameters (ScalaTion's `ParTerm` trait); the
    parameters are resolved from the source column's training rows when first needed.
    """

    source: Term

    @property
    def pars(self) -> List[Par]:
        """The term's own parameters."""
        raise NotImplementedError

    def apply(self, x: np.ndarray) -> np.ndarray:
        """The term's function of the source column at the parameters' current values."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def lag(self) -> int:
        return self.source.lag

    @property
    def vars(self) -> set:
        return self.source.vars

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        x = self.source.column(dataset)
        for p in self.pars:
            if not p.resolved:
                p.resolve(x)
        return self.apply(x)

    def params(self) -> List[Par]:
        return self.source.params() + list(self.pars)

    def sources(self) -> List[Term]:
        return [self.source]

    def _cp(self, p: Par, copies: Dict[int, Par]) -> Par:
        if id(p) not in copies:
            copies[id(p)] = p.fresh()
        return copies[id(p)]


class ShiftTerm(ParTerm):
    """A free shift added to a term, `? + x~1`, written inside a function: `log(? + x~1)`."""

    def __init__(self, source: Term, s: Par) -> None:
        self.source, self.s = source, s

    @property
    def pars(self) -> List[Par]:
        return [self.s]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.s.label(mode)} + {self.source.label(mode)}"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x + self.s.value

    def params(self) -> List[Par]:
        return self.source.params() + [self.s]

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return ShiftTerm(self.source.fresh(copies), self._cp(self.s, copies))


class ScaleTerm(ParTerm):
    """A free scale multiplying a term, `? * x~1`, written inside a function: `exp(? * x~1)`."""

    def __init__(self, source: Term, a: Par) -> None:
        self.source, self.a = source, a

    @property
    def pars(self) -> List[Par]:
        return [self.a]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"{self.a.label(mode)} * {self.source.label(mode)}"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.a.value * x

    def params(self) -> List[Par]:
        return self.source.params() + [self.a]

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return ScaleTerm(self.source.fresh(copies), self._cp(self.a, copies))


class ParPowerTerm(ParTerm):
    """A power with a free exponent `pow(x~1, ?)`."""

    def __init__(self, source: Term, p: Par) -> None:
        self.source, self.p = source, p

    @property
    def pars(self) -> List[Par]:
        return [self.p]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"pow({self.source.label(mode)}, {self.p.label(mode)})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.power(x, self.p.value)

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return ParPowerTerm(self.source.fresh(copies), self._cp(self.p, copies))


class SatTerm(ParTerm):
    """Saturation `sat(x~1, ?)` = x / (x + k) with a free half-saturation k."""

    def __init__(self, source: Term, k: Par) -> None:
        self.source, self.k = source, k

    @property
    def pars(self) -> List[Par]:
        return [self.k]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"sat({self.source.label(mode)}, {self.k.label(mode)})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x / (x + self.k.value)

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return SatTerm(self.source.fresh(copies), self._cp(self.k, copies))


class HingeTerm(ParTerm):
    """Hinge `hinge(x~1, ?)` = max(0, x - c) with a free threshold c."""

    def __init__(self, source: Term, c: Par) -> None:
        self.source, self.c = source, c

    @property
    def pars(self) -> List[Par]:
        return [self.c]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"hinge({self.source.label(mode)}, {self.c.label(mode)})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x - self.c.value)

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return HingeTerm(self.source.fresh(copies), self._cp(self.c, copies))


class TransTerm(ParTerm):
    """Smooth transition `trans(x~1, ?, ?)` = 1 / (1 + exp(-(x - c) / w)) with a free threshold c
    and width w; `trans(y~1, ?, ?) * y~(3,1)` is a STAR model."""

    def __init__(self, source: Term, c: Par, w: Par) -> None:
        self.source, self.c, self.w = source, c, w

    @property
    def pars(self) -> List[Par]:
        return [self.c, self.w]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"trans({self.source.label(mode)}, {self.c.label(mode)}, {self.w.label(mode)})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(-(x - self.c.value) / self.w.value, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(z))

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return TransTerm(self.source.fresh(copies), self._cp(self.c, copies), self._cp(self.w, copies))


class EmaTerm(ParTerm):
    """Exponentially weighted moving average `ema(x~1, ?)`: e_t = d x_t + (1 - d) e_{t-1} (a
    geometric distributed lag) with a free decay d in (0, 1)."""

    def __init__(self, source: Term, d: Par) -> None:
        self.source, self.d = source, d

    @property
    def pars(self) -> List[Par]:
        return [self.d]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"ema({self.source.label(mode)}, {self.d.label(mode)})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        d = self.d.value
        e = np.empty_like(x)
        e[0] = x[0]
        for t in range(1, len(x)):
            e[t] = d * x[t] + (1.0 - d) * e[t - 1]
        return e

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return EmaTerm(self.source.fresh(copies), self._cp(self.d, copies))


class DqTerm(ParTerm):
    """Dampened quadratic `dq(x~1, ?)` = x |x| / sqrt(1 + x^2 / mx^2) with a free cap mx
    (quadratic for |x| << mx, linear beyond)."""

    def __init__(self, source: Term, mx: Par) -> None:
        self.source, self.mx = source, mx

    @property
    def pars(self) -> List[Par]:
        return [self.mx]

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        return f"dq({self.source.label(mode)}, {self.mx.label(mode)})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x * np.abs(x) / np.sqrt(1.0 + (x / self.mx.value) ** 2)

    def fresh(self, copies: Dict[int, Par]) -> Term:
        return DqTerm(self.source.fresh(copies), self._cp(self.mx, copies))


class ParWaveTerm(Term):
    """One seasonal component with a FREE frequency f (cycles per time unit): sin(2 pi f t + phase)."""

    def __init__(self, freq: Par, phase: float = 0.0, idx: int = 0) -> None:
        self.freq, self.phase, self.idx = freq, float(phase), idx

    @property
    def ph(self) -> float:
        return self.phase

    @property
    def name(self) -> str:
        return self.label()

    def label(self, mode: LabelMode = LabelMode.VALUE) -> str:
        kind = "sin" if self.phase == 0.0 else "cos"
        return f"{kind}(f={self.freq.label(mode)})"

    @property
    def lag(self) -> int:
        return 0

    @property
    def vars(self) -> set:
        return set()

    def column(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        m = len(next(iter(dataset.values())))
        return np.sin(2.0 * math.pi * self.freq.value * np.arange(m) + self.phase)

    def params(self) -> List[Par]:
        return [self.freq]

    def fresh(self, copies: Dict[int, Par]) -> Term:
        if id(self.freq) not in copies:
            copies[id(self.freq)] = self.freq.fresh()
        return ParWaveTerm(copies[id(self.freq)], self.phase, self.idx)


# ---------------------------------------------------------------------------------------------
#  The DSL (ScalaTion's `pow (t, p: Par)`, `sat`, `hinge`, `trans`, `ema`, `dq`, `wave (freq: Par)`)
# ---------------------------------------------------------------------------------------------

def pow(t, p):
    """`pow(t, p)`: a fixed exponent (number) or a free one (`Par`), for a term or a lag range."""
    if isinstance(p, Par):
        return _each(t, lambda x: ParPowerTerm(x, p))
    return _each(t, lambda x: PowerTerm(x, float(p)))



def sat(t, k: Par):    return _each(t, lambda x: SatTerm(x, k))
def hinge(t, c: Par):  return _each(t, lambda x: HingeTerm(x, c))
def trans(t, c: Par, w: Par): return _each(t, lambda x: TransTerm(x, c, w))
def ema(t, d: Par):    return _each(t, lambda x: EmaTerm(x, d))


def dq(t, mx: Optional[Par] = None):
    """`dq(t, ?)` with a free cap, or the parameter-free `dq(t)` of the linear grammar."""
    if mx is None:
        return _each(t, lambda x: TransformedTerm(x, "dq"))
    return _each(t, lambda x: DqTerm(x, mx))


def wave(k_or_par: Union[int, Par], freqs: Optional[Sequence[float]] = None) -> List[Term]:
    """
    Seasonal sin/cos pairs: `wave(k, freqs)` gives k pairs at the first k frequencies of `freqs`
    (ScalaTion's `wave(freqs, k)`); `wave(par)` one pair with a free frequency (`wave(freq: Par)`).
    """
    if isinstance(k_or_par, Par):
        return [ParWaveTerm(k_or_par, 0.0), ParWaveTerm(k_or_par, math.pi / 2)]
    return wave_k(k_or_par, freqs)


# ---------------------------------------------------------------------------------------------
#  NARX_Formula
# ---------------------------------------------------------------------------------------------

class NARX_Formula(ARX_Formula):
    """
    An `ARX_Formula` whose terms may hold free nonlinear parameters (`Par`).

    Notes
    -----
    `key` is the canonical identity (parameters as `?`, terms sorted), so two proposals that
    differ only in start or fitted values are one candidate; `formulaStr` shows the parameters
    as fitted (or their start values); `proposalStr` shows them as `?w` slots, the string a
    proposer is given back; `fresh` copies the terms with their own parameters.
    """

    def __init__(self, resp: Union[FormulaVar, str], preds: Sequence[Term], tFormT=None) -> None:
        super().__init__(resp, preds, tFormT)

    @classmethod
    def of(cls, f: ARX_Formula) -> "NARX_Formula":
        """The formula as an `NARX_Formula` (the same object when it already is one), ScalaTion's `NARX_Formula (f)`."""
        return f if isinstance(f, NARX_Formula) else cls(f.resp, f.preds, f.tFormT)

    @classmethod
    def _distinct(cls, preds: Sequence[Term]) -> List[Term]:
        """Distinct by STRUCTURE: every free parameter written as ? (the first copy is kept)."""
        seen = set()
        out: List[Term] = []
        for t in preds:
            k = t.label(LabelMode.SLOT)
            if k not in seen:
                seen.add(k)
                out.append(t)
        return out

    # -- strings ----------------------------------------------------------------------------

    def _str(self, mode: LabelMode = LabelMode.VALUE) -> str:
        pw = [t for t in self.preds if isinstance(t, ParWaveTerm)]
        if not pw:
            return super()._str(mode)
        rest = [t for t in self.preds if not isinstance(t, ParWaveTerm)]
        base = ARX_Formula(self.resp, rest)._str(mode) if rest else f"{self.resp.name} ~="
        seen: List[int] = []
        pairs = []
        for t in pw:
            if id(t.freq) not in seen:
                seen.append(id(t.freq))
                pairs.append(f"wave({t.freq.label(mode)})")
        head = f"{base} + {' + '.join(pairs)}" if rest else f"{base} {' + '.join(pairs)}"
        return head if self.tFormT is None else f"{head} TFORM: {self.tFormT.name}"

    @property
    def proposalStr(self) -> str:
        """The formula with every free parameter written as ?w, w its current (start or fitted)
        value -- the form a proposer is shown, which `NARX_FormulaParser` reads back as a start value."""
        return self._str(LabelMode.SLOT_VALUE)

    # -- parameters -------------------------------------------------------------------------

    def params(self) -> List[Par]:
        """The distinct free parameters of the terms (by identity), in the order they appear."""
        out: List[Par] = []
        seen = set()
        for t in self.preds:
            for p in t.params():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    @property
    def nParams(self) -> int:
        """The number of distinct parameters (free ones and constants in parameter positions)."""
        return len(self.params())

    @property
    def nFree(self) -> int:
        """The number of distinct FREE parameters (the ones a fit searches; the count the DF
        penalty of `NARX_SR_D` and the scorer use)."""
        return sum(1 for p in self.params() if p.free)

    def reset(self) -> None:
        """Forgets the fits: every parameter goes back to its start value."""
        for p in self.params():
            p.reset()

    def resolve(self, dataset: Dict[str, np.ndarray], trSize: int = -1) -> None:
        """
        Resolves the free parameters that are still unset from the training rows of the columns
        they act on (`Par.resolve`); `fromTerms` calls this with its own number of training rows.
        """
        for t in self.preds:
            self._resolve_term(t, dataset, trSize)

    @classmethod
    def _resolve_term(cls, t: Term, dataset: Dict[str, np.ndarray], trSize: int) -> None:
        if isinstance(t, ParTerm):
            if any(not p.resolved for p in t.pars):
                x = t.source.column(dataset)
                for p in t.pars:
                    p.resolve(x, trSize)
            cls._resolve_term(t.source, dataset, trSize)
        elif isinstance(t, ParWaveTerm):
            if not t.freq.resolved:
                t.freq.resolve(np.arange(len(next(iter(dataset.values()))), dtype=float), trSize)
        else:
            for s in t.sources():
                cls._resolve_term(s, dataset, trSize)

    def prepare(self, dataset: Dict[str, np.ndarray], trSize: int = -1):
        """`ARX_Formula.prepare`, then the unset parameters resolved on the training rows."""
        td = super().prepare(dataset, trSize)
        self.resolve(td.data, td.trSize)
        return td

    def redundant(self) -> List[Par]:
        """
        The free parameters the LINEAR fit already covers, so that searching them would make the
        problem singular (kept at their start values by `NARX_SR_D.fromTerms`):

            x + s or a x alone              the intercept / the coefficient covers s / a
            exp (x + s) = e^s exp (x)       the coefficient covers e^s (also expm1)
            log (a x) = log a + log (x)     the intercept covers log a (also with a shift)
            (a x)^p = a^p x^p               the coefficient covers a^p (also with a shift)
            (a x) * t                       the coefficient covers a
            sat (a x, k) = sat (x, k / a)   k covers a
            hinge (a x + s, c), trans (a x + s, c, w): c and w cover s and a
            ema (a x + s) = a ema (x) + s   the coefficient and the intercept cover a and s
        """
        out: List[Par] = []
        for t in self.preds:
            out += self.redundant_of(t)
        return out

    @staticmethod
    def _affine(t: Term) -> List[Par]:
        """The shift and scale parameters at the top of a term (a x + s or s + a x)."""
        if isinstance(t, ShiftTerm):
            return [t.s] + NARX_Formula._affine(t.source)
        if isinstance(t, ScaleTerm):
            return [t.a] + NARX_Formula._affine(t.source)
        return []

    @staticmethod
    def redundant_of(t: Term) -> List[Par]:
        """The redundant parameters of one term (ScalaTion's `NARX_Formula.redundant (t)`)."""
        if isinstance(t, ShiftTerm):
            return [t.s]
        if isinstance(t, ScaleTerm):
            return [t.a]
        if isinstance(t, TransformedTerm):
            s = t.source
            if t.op in ("exp", "expm1") and isinstance(s, ShiftTerm):
                return [s.s]
            if t.op == "log" and isinstance(s, ScaleTerm):
                return [s.a]
            if t.op == "log" and isinstance(s, ShiftTerm) and isinstance(s.source, ScaleTerm):
                return [s.source.a]
            return []
        if isinstance(t, ParPowerTerm):
            s = t.source
            if isinstance(s, ScaleTerm):
                return [s.a]
            if isinstance(s, ShiftTerm) and isinstance(s.source, ScaleTerm):
                return [s.source.a]
            return []
        if isinstance(t, ProductTerm):
            out = []
            if isinstance(t.left, ScaleTerm):
                out.append(t.left.a)
            if isinstance(t.right, ScaleTerm):
                out.append(t.right.a)
            return out
        if isinstance(t, SatTerm) and isinstance(t.source, ScaleTerm):
            return [t.source.a]
        if isinstance(t, (HingeTerm, TransTerm, EmaTerm)):
            return NARX_Formula._affine(t.source)
        return []

    @staticmethod
    def slots(s: str) -> str:
        """Removes the start values from the free-parameter slots of a label or key, e.g.,
        pow(y~1, ?1.500) -> pow(y~1, ?)."""
        return re.sub(r"\?[-+.0-9eE]*", "?", s)

    # -- fitting ----------------------------------------------------------------------------

    def fitNL_D(self, dataset: Dict[str, np.ndarray], hh: int = 1, tFormT=None, trSize: int = -1,
                scaled: bool = False, optimizer=None, hparam: Optional[Dict] = None, **opt_kwargs) -> "NARX_SR_D":
        """
        Fits the `NARX_SR_D` model with one column per term: the free parameters are fitted on the
        training rows by the optimizer, then the coefficients in-sample (`NARX_SR_D.fromTerms`
        followed by `trainNtest`), as `ARX_Formula.fit_D` does for `ARX_D`.

        Parameters
        ----------
        dataset : dict of str -> np.ndarray
            The named columns.
        hh : int, default=1
            The maximum forecasting horizon.
        tFormT : TransformT, optional
            The transform type of every base column (defaults to the formula's).
        trSize : int, default=-1
            The training rows the transforms and the free parameters are fit on (-1 => `trSize`).
        scaled : bool, default=False
            Whether the regression (and the fit of the parameters) standardizes the term columns.
        optimizer : OptimizerNL or str, optional
            The optimizer of the nonlinear parameters (None => `narx_sr_d.OPTIMIZER_NL`).
        hparam : dict, optional
            The hyper-parameters of the model ("lambda", the ridge penalty, default 0.1; "nneg"; ...).
        """
        # the one deferred import of the package: `narx_sr_d` builds on this module (a genuine cycle)
        from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
        mod = NARX_SR_D.fromTerms(self, dataset, hh, hparam, tFormT=tFormT if tFormT is not None else self.tFormT,
                                  trSize=trSize, scaled=scaled, optimizer=optimizer, **opt_kwargs)
        mod.trainNtest()
        return mod

    # -- the grammar ------------------------------------------------------------------------

    @staticmethod
    def grammar(maxLag: int = 8, pMin: float = 0.2, pMax: float = 3.0) -> str:
        """The grammar as text: `ARX_Formula.grammar` plus the free parameters."""
        return ARX_Formula.grammar(maxLag, pMin, pMax) + (
            "\nfree parameters: a number in a term may be ?, a parameter fitted on the training rows (?w0: from the start value w0)\n"
            f"            pow(term, ?)          free exponent, {pMin} <= p <= {pMax}\n"
            "            FN(? + term)          free shift inside log, log1p, pow, sin, cos (not exp: e^s is a coefficient)\n"
            "            exp(? * term)         free rate (not inside log or pow: a is a coefficient there)\n"
            "            sat(term, ?)          saturation x / (x + k), free half-saturation k\n"
            "            hinge(term, ?)        max(0, x - c), free threshold c\n"
            "            trans(term, ?, ?)     smooth step 1 / (1 + exp(-(x - c) / w)), free threshold c and width w\n"
            "            ema(term, ?)          exponentially weighted moving average, free decay in (0, 1)\n"
            "            dq(term, ?)           dampened quadratic, free cap\n"
            "            wave(?)               free frequency in cycles per time unit (one sin/cos pair)\n"
            "            the same parameter may be shared by the terms of a lag range, e.g., pow(y~(3,1), ?)")

