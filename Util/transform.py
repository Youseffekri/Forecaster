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
    (fitted on the data given at construction -- the training slice)

The Python counterpart of ScalaTion's `scalation.mathstat.Transform` (`TransformT` names the
types, `TransformT.Log1p.form(x_train)` makes one).  The `rescale` factories of the models
take sklearn scalers (MinMaxScaler / StandardScaler) instead, which play the roles of
MinMaxForm / NormForm there.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

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
    """(x - mu) / sigma -- standardization with the mean and (unbiased) standard deviation of the
    data given at construction (the training slice); ScalaTion's `NormForm`."""

    def __init__(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
        super().__init__([float(np.mean(x)), sd if sd > 0.0 else 1.0])

    def f(self, x):
        return (np.asarray(x, dtype=float) - self.w[0]) / self.w[1]

    def fi(self, y):
        return np.asarray(y, dtype=float) * self.w[1] + self.w[0]


class MinMaxForm(Transform):
    """l + (x - min) (u - l) / (max - min) -- min-max scaling to (l, u) with the extremes of the
    data given at construction (the training slice); ScalaTion's `MinMaxForm` (lu = (0, 4))."""

    def __init__(self, x: np.ndarray, lu=(0.0, 4.0)) -> None:
        x = np.asarray(x, dtype=float)
        lo, hi = float(np.min(x)), float(np.max(x))
        super().__init__([lo, hi if hi > lo else lo + 1.0, float(lu[0]), float(lu[1])])

    def f(self, x):
        lo, hi, l, u = self.w
        return l + (np.asarray(x, dtype=float) - lo) * (u - l) / (hi - lo)

    def fi(self, y):
        lo, hi, l, u = self.w
        return lo + (np.asarray(y, dtype=float) - l) * (hi - lo) / (u - l)


class CenterForm(Transform):
    """x - mu -- centering with the mean of the data given at construction; ScalaTion's `CenterForm`."""

    def __init__(self, x: np.ndarray) -> None:
        super().__init__([float(np.mean(np.asarray(x, dtype=float)))])

    def f(self, x):
        return np.asarray(x, dtype=float) - self.w[0]

    def fi(self, y):
        return np.asarray(y, dtype=float) + self.w[0]


#: The transformations by ScalaTion's `TransformT` name.
TRANSFORMS = {
    "Pow": PowForm, "Root": RootForm, "Recip": RecipForm, "PowR": PowRForm, "DQuad": DQuadForm,
    "DIQuad": DIQuadForm, "Boxcox": BoxcoxForm, "Log": LogForm, "Log1p": Log1pForm, "Exp": ExpForm,
    "NExp": NExpForm, "Cos": CosForm, "Sin": SinForm,
}

#: The transformations fitted on data (the training slice) by ScalaTion's `TransformT` name.
FITTED_TRANSFORMS = {"Norm": NormForm, "MinMax": MinMaxForm, "Center": CenterForm}


def transform(name: str, *args) -> Transform:
    """A transformation by name (`transform("Pow", 0, 2)` is (x + 0)^2)."""
    if name not in TRANSFORMS:
        raise ValueError(f"transform '{name}' unknown: {', '.join(TRANSFORMS)}")
    return TRANSFORMS[name](*args)


class TransformT(Enum):
    """
    The transform TYPES (ScalaTion's `TransformT` enum): `form(x)` makes the transformation, fitting
    it on x (the training slice) for the data-dependent ones (Norm, MinMax, Center) and with the
    default parameters of `HP` for the others.  `TFORM: Log1p` in a formula names one.
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

    def form(self, x: np.ndarray) -> Transform:
        """The transformation of this type, fitted on x when data-dependent."""
        if self.name in FITTED_TRANSFORMS:
            return FITTED_TRANSFORMS[self.name](x)
        return TRANSFORMS[self.name]()

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
