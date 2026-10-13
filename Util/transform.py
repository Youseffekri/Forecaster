"""
Transformations for Symbolic Regression Terms

The parameterized transformations the `ARX_SR` / `ARX_SR_D` models apply to the lagged
endogenous and exogenous variables (and that `NARX_SR_D` fits the parameters of): each has a
forward map `f` and an inverse `fi`, and a default parameter vector taken from `HP`, the
counterpart of ScalaTion's `Transform.hp`.

    PowForm   (x + s)^p            RootForm  (x + s)^r          RecipForm  (x + s)^c
    PowRForm  sign(x+s) |x+s|^q    LogForm   log(a x + ss)      Log1pForm  log(x + 1)
    ExpForm   exp(a x + s)         NExpForm  exp(an x + s)      BoxcoxForm ((x^l) - 1) / l
    DQuadForm x|x| / sqrt(1 + x^2/mx^2)  (dampened quadratic)   DIQuadForm  its dampened inverse
    SinForm   sin(2 pi f x + s)    CosForm   cos(2 pi f x + s)

    NormForm  (x - mu) / sigma     MinMaxForm  l + (x - min)(u - l)/(max - min)    CenterForm  x - mu
    (fitted on the data given at construction -- the training slice -- column by column for a
    matrix; MinMaxForm maps to the CENTERED range [l, u] = [-2, 2] by default, `setLU` changes it)

The Python counterpart of ScalaTion's `scalation.mathstat.Transform`: `TransformT` names the types
and `TransformT.X.form(x)` makes one -- fitted on x (the training slice) for Norm, MinMax and
Center, from the parameter vector x = (w0, w1) for the parameterized ones (the defaults when x is
None), as ScalaTion's `t.form (VectorD (w0, w1))`; `TransformT.X.wlu` holds the default parameters
and their bounds (ScalaTion's `TransformS`), the search box of `NARX_SR_D.optimize`.  The `rescale`
factories of the models take a `TransformT` (`tFormT`), as in ScalaTion.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: The default parameters of the transformations (ScalaTion's `Transform.hp`).
HP: Dict[str, float] = {
    "a": 1.0,      # the multiplier (positive)
    "an": -1.0,    # the multiplier (negative)
    "c": -1.0,     # the reciprocal power
    "f": 0.1,      # the frequency for cos, sin
    "l": 0.4,      # the Box-Cox lambda
    "mx": 5.0,     # the maximum for the dampened quadratics
    "p": 1.5,      # the real power
    "q": 0.67,     # the rational power
    "r": 0.5,      # the root
    "s": 0.0,      # the amount of shift
    "ss": 1.0,     # the amount of shift for log
}


def _quiet(fn):
    """Runs a transform's map with numpy's floating-point warnings off (inf / nan are returned, not reported)."""
    def wrapped(self, x):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return fn(self, x)
    wrapped.__doc__ = fn.__doc__
    wrapped.__name__ = fn.__name__
    wrapped._quiet = True
    return wrapped


class Transform:
    """
    A parameterized transformation of a vector (or of every column of a matrix).

    Attributes
    ----------
    w : np.ndarray
        The parameter vector (e.g., (shift, power) for `PowForm`).
    name : str
        The class name, used in column labels and reports.
    """

    def __init__(self, w) -> None:
        self.w = np.asarray(w, dtype=float)
        self.lu = np.array([-2.0, 2.0])                         # the min-max range (used by MinMaxForm)

    def setLU(self, lu: Sequence[float]) -> None:
        """Sets the min-max range [l, u] (ScalaTion's `setLU`; only `MinMaxForm` uses it)."""
        self.lu = np.asarray(lu, dtype=float)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def f(self, x: np.ndarray) -> np.ndarray:
        """The forward transformation (element-wise)."""
        raise NotImplementedError

    def fi(self, y: np.ndarray) -> np.ndarray:
        """The inverse transformation (element-wise)."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def __init_subclass__(cls, **kwargs) -> None:
        """Wraps every subclass's `f` and `fi` in `np.errstate` (a log of 0 or an exp of a huge forecast
        yields inf / nan without a numpy warning; the callers check finiteness and report it once)."""
        super().__init_subclass__(**kwargs)
        for name in ("f", "fi"):
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "_quiet", False):
                setattr(cls, name, _quiet(fn))

    # -- the sklearn-like interface, so a Transform can serve as a model's `yForm` ----------------

    def fit(self, x):
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fi(x)

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{v:g}' for v in self.w)})"


class PowForm(Transform):
    """(x + s)^p -- a shifted power; the default of `ARX_SR` (p = 1.5)."""

    def __init__(self, s: float = HP["s"], p: float = HP["p"]) -> None:
        super().__init__([s, p])

    def f(self, x):
        return np.power(np.asarray(x, dtype=float) + self.w[0], self.w[1])

    def fi(self, y):
        return np.power(np.asarray(y, dtype=float), 1.0 / self.w[1]) - self.w[0]


class RootForm(PowForm):
    """(x + s)^r -- a shifted root (r = 0.5)."""

    def __init__(self, s: float = HP["s"], r: float = HP["r"]) -> None:
        super().__init__(s, r)


class RecipForm(PowForm):
    """(x + s)^c -- a shifted reciprocal power (c = -1)."""

    def __init__(self, s: float = HP["s"], c: float = HP["c"]) -> None:
        super().__init__(s, c)


class PowRForm(Transform):
    """sign(x + s) |x + s|^q -- a sign-safe (rational) power defined for negative values (q = 0.67)."""

    def __init__(self, s: float = HP["s"], q: float = HP["q"]) -> None:
        super().__init__([s, q])

    def f(self, x):
        z = np.asarray(x, dtype=float) + self.w[0]
        return np.sign(z) * np.power(np.abs(z), self.w[1])

    def fi(self, y):
        y = np.asarray(y, dtype=float)
        return np.sign(y) * np.power(np.abs(y), 1.0 / self.w[1]) - self.w[0]


class DQuadForm(Transform):
    """x |x| / sqrt(1 + x^2 / mx^2) -- a dampened quadratic: quadratic for |x| << mx, linear beyond."""

    def __init__(self, mx: float = HP["mx"]) -> None:
        super().__init__([0.0, mx])

    def f(self, x):
        x = np.asarray(x, dtype=float)
        k = 1.0 / self.w[1] ** 2
        return x * np.abs(x) / np.sqrt(1.0 + k * x * x)

    def fi(self, z):
        z = np.asarray(z, dtype=float)
        k = 1.0 / self.w[1] ** 2
        z_sq = z * z
        kz_sq = k * z_sq
        u = (kz_sq + np.sqrt(kz_sq * kz_sq + 4.0 * z_sq)) / 2.0
        return np.sign(z) * np.sqrt(u)


class DIQuadForm(Transform):
    """sign(x) sqrt(|x|) / (1 + |x| / mx)^0.25-like dampened inverse quadratic: the inverse map of `DQuadForm`."""

    def __init__(self, mx: float = HP["mx"]) -> None:
        super().__init__([0.0, mx])
        self._dq = DQuadForm(mx)

    def f(self, x):
        return self._dq.fi(x)

    def fi(self, y):
        return self._dq.f(y)


class BoxcoxForm(Transform):
    """(x^l - 1) / l -- the Box-Cox transformation (l = 0.4)."""

    def __init__(self, l: float = HP["l"]) -> None:
        super().__init__([l])

    def f(self, x):
        l = self.w[0]
        return (np.power(np.asarray(x, dtype=float), l) - 1.0) / l

    def fi(self, y):
        l = self.w[0]
        return np.power(np.asarray(y, dtype=float) * l + 1.0, 1.0 / l)


class LogForm(Transform):
    """log(a x + ss) -- a shifted, scaled log (ss = 1, a = 1: log1p)."""

    def __init__(self, ss: float = HP["ss"], a: float = HP["a"]) -> None:
        super().__init__([ss, a])

    def f(self, x):
        return np.log(np.asarray(x, dtype=float) * self.w[1] + self.w[0])

    def fi(self, y):
        return (np.exp(np.asarray(y, dtype=float)) - self.w[0]) / self.w[1]


class Log1pForm(LogForm):
    """log(x + 1)."""

    def __init__(self) -> None:
        super().__init__(1.0, 1.0)


class ExpForm(Transform):
    """exp(a x + s) (a = 1)."""

    def __init__(self, s: float = HP["s"], a: float = HP["a"]) -> None:
        super().__init__([s, a])

    def f(self, x):
        return np.exp(np.asarray(x, dtype=float) * self.w[1] + self.w[0])

    def fi(self, y):
        return (np.log(np.asarray(y, dtype=float)) - self.w[0]) / self.w[1]


class NExpForm(ExpForm):
    """exp(an x + s) with a negative multiplier (an = -1)."""

    def __init__(self, s: float = HP["s"], an: float = HP["an"]) -> None:
        super().__init__(s, an)


class CosForm(Transform):
    """cos(2 pi f x + s) (f = 0.1)."""

    def __init__(self, s: float = HP["s"], f: float = HP["f"]) -> None:
        super().__init__([s, f])

    def f(self, x):
        return np.cos(np.asarray(x, dtype=float) * (self.w[1] * 2.0 * np.pi) + self.w[0])

    def fi(self, y):
        return (np.arccos(np.asarray(y, dtype=float)) - self.w[0]) / (self.w[1] * 2.0 * np.pi)


class SinForm(Transform):
    """sin(2 pi f x + s) (f = 0.1)."""

    def __init__(self, s: float = HP["s"], f: float = HP["f"]) -> None:
        super().__init__([s, f])

    def f(self, x):
        return np.sin(np.asarray(x, dtype=float) * (self.w[1] * 2.0 * np.pi) + self.w[0])

    def fi(self, y):
        return (np.arcsin(np.asarray(y, dtype=float)) - self.w[0]) / (self.w[1] * 2.0 * np.pi)


class NormForm(Transform):
    """(x - mu) / (sigma + eps) -- standardization with the mean and (unbiased) standard deviation of
    the data given at construction (the training slice; column by column for a matrix); ScalaTion's
    robust `NormForm` (eps = 1e-8 keeps a constant column finite)."""

    EPS = 1e-8

    def __init__(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        self._mu = x.mean(axis=0)
        self._sig = x.std(axis=0, ddof=1) if n > 1 else np.zeros_like(self._mu)
        super().__init__(np.concatenate((np.atleast_1d(self._mu), np.atleast_1d(self._sig))))

    def f(self, x):
        return (np.asarray(x, dtype=float) - self._mu) / (self._sig + self.EPS)

    def fi(self, y):
        return np.asarray(y, dtype=float) * (self._sig + self.EPS) + self._mu


class MinMaxForm(Transform):
    """l + (x - min) (u - l) / (max - min) -- min-max scaling to [l, u] with the extremes of the data
    given at construction (the training slice; column by column for a matrix); ScalaTion's
    `MinMaxForm`: CENTERED to [-2, 2] by default (to reduce collinearity), [0, 4] with
    centered = False, any range with `setLU` (e.g., [1, 5] keeps powers, roots and logs defined)."""

    def __init__(self, x: np.ndarray, centered: bool = True) -> None:
        x = np.asarray(x, dtype=float)
        self._lo, self._hi = x.min(axis=0), x.max(axis=0)
        self._rng = np.where(self._hi > self._lo, self._hi - self._lo, 1.0)     # a constant column maps to l
        super().__init__(np.concatenate((np.atleast_1d(self._lo), np.atleast_1d(self._hi))))
        self.lu = np.array([-2.0, 2.0]) if centered else np.array([0.0, 4.0])

    def f(self, x):
        l, u = self.lu
        return (np.asarray(x, dtype=float) - self._lo) * (u - l) / self._rng + l

    def fi(self, y):
        l, u = self.lu
        return (np.asarray(y, dtype=float) - l) * self._rng / (u - l) + self._lo


class CenterForm(Transform):
    """x - mu -- centering with the mean of the data given at construction (column by column for a
    matrix); ScalaTion's `CenterForm`."""

    def __init__(self, x: np.ndarray) -> None:
        self._mu = np.asarray(x, dtype=float).mean(axis=0)
        super().__init__(np.atleast_1d(self._mu))

    def f(self, x):
        return np.asarray(x, dtype=float) - self._mu

    def fi(self, y):
        return np.asarray(y, dtype=float) + self._mu


#: The transformations by ScalaTion's `TransformT` name.
TRANSFORMS = {
    "Pow": PowForm, "Root": RootForm, "Recip": RecipForm, "PowR": PowRForm, "DQuad": DQuadForm,
    "DIQuad": DIQuadForm, "Boxcox": BoxcoxForm, "Log": LogForm, "Log1p": Log1pForm, "Exp": ExpForm,
    "NExp": NExpForm, "Cos": CosForm, "Sin": SinForm,
}

#: The transformations fitted on data (the training slice) by ScalaTion's `TransformT` name.
FITTED_TRANSFORMS = {"Norm": NormForm, "MinMax": MinMaxForm, "Center": CenterForm}

#: ScalaTion's `TransformT.wlu` (`TransformS`): the default parameters w = (w0, w1) of each
#: parameterized type and their lower (l) and upper (u) bounds; equal bounds fix a parameter.
WLU: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    "Pow":    ((HP["s"], HP["p"]),   (0.0, 1.1),     (10.0, 4.0)),     # (x + s)^p, p in [1.1, 4]
    "PowR":   ((HP["s"], HP["q"]),   (0.0, 0.1),     (10.0, 0.9)),     # sign-safe power q in [0.1, 0.9]
    "DQuad":  ((0.0, HP["mx"]),      (0.0, -1e9),    (0.0, 1e9)),      # dampened quadratic (max mx)
    "DIQuad": ((0.0, HP["mx"]),      (0.0, -1e9),    (0.0, 1e9)),      # dampened inverse quadratic
    "Boxcox": ((HP["l"], 0.0),       (-4.0, 0.0),    (4.0, 0.0)),      # Box-Cox l in [-4, 4]
    "Root":   ((HP["s"], HP["r"]),   (0.0, 0.1),     (10.0, 0.9)),     # (x + s)^r, r in [0.1, 0.9]
    "Recip":  ((HP["s"], HP["c"]),   (0.0, -4.0),    (10.0, -0.1)),    # (x + s)^c, c in [-4, -0.1]
    "Log":    ((HP["ss"], HP["a"]),  (0.0, 0.1),     (10.0, 10.0)),    # log (a x + ss)
    "Exp":    ((HP["s"], HP["a"]),   (0.0, 0.1),     (10.0, 10.0)),    # exp (a x + s)
    "NExp":   ((HP["s"], HP["an"]),  (-10.0, -10.0), (0.0, -0.1)),     # exp (an x + s), an < 0
    "Cos":    ((HP["s"], HP["f"]),   (0.0, 0.01),    (10.0, 10.0)),    # cos (2 pi f x + s)
    "Sin":    ((HP["s"], HP["f"]),   (0.0, 0.01),    (10.0, 10.0)),    # sin (2 pi f x + s)
}


def _from_w(name: str, w: Sequence[float]) -> Transform:
    """The transformation of a parameterized type at the parameter vector w = (w0, w1)."""
    w0, w1 = float(w[0]), float(w[1])
    if name in ("DQuad", "DIQuad"):
        return TRANSFORMS[name](w1)                              # (0, mx): the cap only
    if name == "Boxcox":
        return BoxcoxForm(w0)                                    # (l, 0): lambda only
    return TRANSFORMS[name](w0, w1)


def transform(name: str, *args) -> Transform:
    """A transformation by name (`transform("Pow", 0, 2)` is (x + 0)^2)."""
    if name not in TRANSFORMS:
        raise ValueError(f"transform '{name}' unknown: {', '.join(TRANSFORMS)}")
    return TRANSFORMS[name](*args)


class TransformT(Enum):
    """
    The transform TYPES (ScalaTion's `TransformT` enum): `form(x)` makes the transformation -- fitted
    on x (the training slice) for the data-dependent ones (Norm, MinMax, Center), from the parameter
    vector x = (w0, w1) for the parameterized ones (ScalaTion's `t.form (VectorD (w0, w1))`), with
    the default parameters of `HP` when x is None (or is data); `wlu` gives the default parameters
    and their bounds (None for a type without parameters).  `TFORM: Log1p` in a formula names one.
    """
    Norm = "NormForm"
    MinMax = "MinMaxForm"
    Center = "CenterForm"
    Pow = "PowForm"
    PowR = "PowRForm"
    DQuad = "DQuadForm"
    DIQuad = "DIQuadForm"
    Boxcox = "BoxcoxForm"
    Root = "RootForm"
    Recip = "RecipForm"
    Log = "LogForm"
    Log1p = "Log1pForm"
    Exp = "ExpForm"
    NExp = "NExpForm"
    Cos = "CosForm"
    Sin = "SinForm"

    def form(self, x: Optional[np.ndarray] = None) -> Transform:
        """
        The transformation of this type: fitted on x when data-dependent; at the parameters x = (w0, w1)
        when parameterized and x is such a pair; with the default parameters otherwise.
        """
        if self.name in FITTED_TRANSFORMS:
            return FITTED_TRANSFORMS[self.name](x)
        if x is not None and self.name in WLU and np.ndim(x) == 1 and len(x) == 2:
            return _from_w(self.name, x)
        return TRANSFORMS[self.name]()

    @property
    def wlu(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The default parameters w and their bounds (l, u) -- ScalaTion's `TransformS` -- or None."""
        if self.name not in WLU:
            return None
        return tuple(np.array(v, dtype=float) for v in WLU[self.name])

    def __str__(self) -> str:
        return self.name


def transform_t(name: str) -> TransformT:
    """
    The transform type by name, case-insensitive (`transform_t("log1p")` is `TransformT.Log1p`).

    Raises
    ------
    ValueError
        For an unknown name.
    """
    for t in TransformT:
        if t.name.lower() == name.lower():
            return t
    raise ValueError(f"unknown TFORM '{name}'")
