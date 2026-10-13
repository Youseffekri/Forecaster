"""
Utility Functions: ScalaTion-style Printing and Small Helpers

The Python counterpart of the printing helpers of ScalaTion (`banner`, `fmt`, `VectorD.toString`,
`MatrixD.toString`, `stringOf (Array)`, Java's `Double.toString`), so that the output of the
forecasting models reads as ScalaTion's: "b = VectorD(0.946883,\t...), fn = Array(yl1, ...)",
"fitMap QoF = ...".  Also the `dotdict`.  The QoF measures live in `modeling/fit.py` (ScalaTion's
`Fit`), the REPORT block in `Forecaster.report` (ScalaTion's `Model.report`) and the hyper-parameter
printing in `modeling/hyper_parameter.py`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Sequence

import numpy as np


class dotdict(dict):
    """
    A dictionary subclass that allows dot notation for attribute access.

    Examples
    --------
    d = dotdict({"a": 1, "b": 2})
    print(d.a)    # Outputs: 1
    d.c = 3
    print(d["c"]) # Outputs: 3
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'dotdict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


# ---------------------------------------------------------------------------------------------
# ScalaTion-style printing
# ---------------------------------------------------------------------------------------------

def banner(s: str) -> None:
    """Prints the string in a box of dashes (ScalaTion's `banner`)."""
    line = "-" * (len(s) + 4)
    print(line)
    print("| " + s + " |")
    print(line)


def _java_special(x: float) -> str:
    """Java's spelling of NaN and the infinities ("NaN", "Infinity", "-Infinity"), or "" for a finite value."""
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return ""


def fmt(x: float) -> str:
    """Six decimals (ScalaTion's `fmt`, Java's "%.6f": NaN prints as "NaN")."""
    xf = float(x)
    return _java_special(xf) or "%.6f" % xf


def jdouble(x: float) -> str:
    """
    A double as Java's `Double.toString` prints it (ScalaTion's string interpolation of a Double):
    decimal notation with at least one fraction digit for 1e-3 <= |x| < 1e7 ("91.0",
    "12.521340230318195"), computerized scientific notation otherwise ("6.308091682338812E-7").
    """
    xf = float(x)
    special = _java_special(xf)
    if special:
        return special
    if xf == 0.0:
        return "-0.0" if np.signbit(xf) else "0.0"
    if 1e-3 <= abs(xf) < 1e7:
        s = repr(xf)
        return s if "." in s else s + ".0"
    sign, digits, exp = Decimal(repr(xf)).normalize().as_tuple()      # the shortest digits that round-trip
    ds = "".join(str(d) for d in digits)
    e10 = exp + len(ds) - 1                                           # the exponent of the leading digit
    mant = ds[0] + "." + (ds[1:] or "0")
    return ("-" if sign else "") + f"{mant}E{e10}"


def fmt_g(x: float) -> str:
    """Six significant digits keeping trailing zeros, as Java's "%g" used by `VectorD.toString`."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    xf = float(x)
    return _java_special(xf) or "%#.6g" % xf


def vector_str(v: Sequence[float], name: str = "VectorD") -> str:
    """The vector as ScalaTion prints it: "VectorD(0.946883,\\t0.676751,\\t...)"."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0:
        return f"{name}()"
    return f"{name}(" + ",\t".join(fmt_g(x) for x in v) + ")"


def matrix_str(a: np.ndarray, name: str = "MatrixD") -> str:
    """The matrix as ScalaTion prints it: "\\nMatrixD (v,\\tv, ...\\n \\tv,\\tv, ...)"."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.size == 0:
        return f"\n{name} ()"
    rows = [",\t".join(fmt_g(x) for x in row) for row in a]
    return f"\n{name} (" + ",\n \t".join(rows) + ")"


def array_str(items: Iterable[Any]) -> str:
    """An array as Scala's `stringOf` prints it: "Array(cons, yl6, yl5)"."""
    return "Array(" + ", ".join(str(x) for x in items) + ")"


def set_str(items: Iterable[Any]) -> str:
    """A column set as Scala prints a `LinkedHashSet`: "LinkedHashSet(0, 1, 3)"."""
    return "LinkedHashSet(" + ", ".join(str(x) for x in items) + ")"
