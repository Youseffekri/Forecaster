"""
Utility Functions: ScalaTion-style Printing and Small Helpers

The Python counterpart of the printing helpers of ScalaTion (`banner`, `fmt`, `VectorD.toString`,
`MatrixD.toString`, `stringOf (Array)`), so that the output of the forecasting models reads as
ScalaTion's: "b = VectorD(0.946883,\t...), fn = Array(yl1, ...)", "fitMap QoF = ...".  Also the
`dotdict` and the legacy metric functions kept for compatibility (the QoF measures now live in
`modeling/fit.py`, in ScalaTion's `Fit` form).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

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


def fmt(x: float) -> str:
    """Six decimals (ScalaTion's `fmt`, "%.6f")."""
    return "%.6f" % x


def fmt_g(x: float) -> str:
    """Six significant digits keeping trailing zeros, as Java's "%g" used by `VectorD.toString`."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    xf = float(x)
    if np.isnan(xf):
        return "NaN"
    if np.isinf(xf):
        return "Infinity" if xf > 0 else "-Infinity"
    return "%#.6g" % xf


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


def hparam_str(hp: Mapping[str, Any]) -> str:
    """The hyper-parameters as ScalaTion prints a `HyperParameter`: "HyperParameter(p -> 6, q -> 0, ...)"."""
    return "HyperParameter(" + ", ".join(f"{k} -> {v}" for k, v in hp.items()) + ")"


def report(modelName: str, hparam: Mapping[str, Any], fname: Sequence[str], b: np.ndarray,
           qof: np.ndarray, qoF_names: Optional[Sequence[str]] = None) -> str:
    """
    The REPORT block of ScalaTion's `Model.report` (model name, hyper-parameters, feature names,
    parameters, fit map); `b` may be a vector (one line) or a matrix (one column per horizon).

    Parameters
    ----------
    modelName : str
        The model name, e.g., "ARX_D_6_6_2".
    hparam : Mapping[str, Any]
        The hyper-parameters.
    fname : Sequence[str]
        The feature (column) names.
    b : np.ndarray
        The parameter vector (or matrix for a direct model).
    qof : np.ndarray, shape (N_QoF,)
        The QoF vector (horizon 1).
    qoF_names : Sequence[str], optional
        The QoF labels (defaults to ScalaTion's `QoF` names).
    """
    from modeling.fit import fitMap, qoF_names as _names
    labels = _names if qoF_names is None else qoF_names
    b = np.asarray(b, dtype=float)
    b_str = vector_str(b) if b.ndim == 1 else matrix_str(b)
    rule = "    " + "-" * 76
    return (f"\nREPORT\n{rule}\n    modelName  mn  = {modelName}\n{rule}\n"
            f"    hparameter hp  = {hparam_str(hparam)}\n{rule}\n"
            f"    features   fn  = {array_str(fname)}\n{rule}\n"
            f"    parameter  b   = {b_str}\n{rule}\n"
            f"    fitMap     qof = {fitMap(qof, labels)}\n{rule}\n        ")


# ---------------------------------------------------------------------------------------------
# Legacy metric functions (kept for compatibility; see `modeling/fit.py` for the QoF measures)
# ---------------------------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The Mean Squared Error (sse / m)."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The coefficient of determination R^2 (0 when y is constant)."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_total == 0:
        return 0.0
    return float(1 - np.sum((y_true - y_pred) ** 2) / ss_total)


def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """The adjusted R^2 with n_features slopes and an intercept (nan when m <= n_features + 1)."""
    m = len(y_true)
    if m <= n_features + 1:
        return np.nan
    return float(1 - (1 - r2(y_true, y_pred)) * (m - 1) / (m - n_features - 1))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The symmetric Mean Absolute Percentage Error in percent (ScalaTion's `smapeF`, zero errors skipped)."""
    from modeling.fit import smapeF
    return smapeF(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))


def diagnose(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> Dict[str, float]:
    """
    The QoF measures of a prediction as a dictionary keyed by ScalaTion's QoF names
    (rSq, rSqBar, sst, sse, ..., smapeC); `n_features` is the number of slopes (dfr).
    """
    from modeling.fit import diagnose as _diagnose, fitMapDict
    return fitMapDict(_diagnose(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), n_features))


def fit_map(y_true_arr: Sequence[np.ndarray], y_fcast_arr: Sequence[np.ndarray], dfr: int = 1) -> str:
    """
    The QoF table of per-horizon (actual, forecast) pairs in ScalaTion's `showFitMap` form
    (one "label -> VectorD(...)" line per measure, the horizons as the columns).
    """
    from modeling.fit import N_QoF, diagnose as _diagnose, showFitMap
    ft = np.zeros((N_QoF, len(y_true_arr)))
    for h, (yy, yfh) in enumerate(zip(y_true_arr, y_fcast_arr)):
        ft[:, h] = _diagnose(np.asarray(yy, dtype=float), np.asarray(yfh, dtype=float), dfr)
    return showFitMap(ft)
