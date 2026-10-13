"""
ExoScreen: Screening of Candidate Exogenous Variables for Forecasting

Which columns of a data set are eligible as drivers of the response, which drivers lead it, at
which lag, with which sign -- the cheap data-only filter that runs BEFORE any model is fitted and
before the formula search ranks the drivers.  The Python counterpart of ScalaTion's
`ExoScreen.scala` (`ExoRank`, `ExoPool`, the `ExoScreen` object) together with `Model.screen`
of `Model.scala` (`model_screen` here).  Three steps, each usable alone:

    pool       from a data matrix with a header (or a pandas DataFrame), every column that can be a
               driver of the response: not the response, not excluded, finite, and neither constant
               nor cumulative (monotone) on the training rows
    rank       each candidate's cross-correlation with y at lags 1..maxLag on the TRAINING rows only
               (a driver must LEAD the response; `ccorr (x, y, k)` correlates x_i with y_i+k),
               strongest first, next to its lag-0 correlation
    shortlist  `Model.screen` fed the time-series dependency matrix of `dependency` (x-vs-y: the
               strongest lead; x-vs-x: lag-0 correlation): in decreasing dependency, keep a candidate
               when its |ccf| exceeds thr1 and its correlation with every candidate already kept
               stays below thr2 (collinearity), the first k kept

and `screen` does the three in one call, returning the (xe, y, names) the forecasters take
(`ARX_D (hparam, y, hh, xe)`), with the k chosen columns in the file's order.  What the data cannot
decide is whether a column is a driver or another measurement of the response (excess mortality
next to Covid-19 deaths) or comparable over the whole period (case counts under changing testing
regimes); that is domain knowledge, given through `exclude`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Util.tools import banner
from modeling.forecasting.forecaster import trSize


# ---------------------------------------------------------------------------------------------
#  Statistics of ScalaTion's VectorD / MatrixD used by the screen
# ---------------------------------------------------------------------------------------------

def corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson's correlation of x with y (ScalaTion's `VectorD.corr`): 1 when both are the same
    constant vector, -0.0 when undefined.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    cov = (x @ y - x.sum() * y.sum() / n) / (n - 1)
    den = np.sqrt(np.var(x, ddof=1) * np.var(y, ddof=1))
    if not den > 0.0 or not np.isfinite(den):                            # a zero variance: undefined
        return 1.0 if np.array_equal(x, y) else -0.0
    return float(cov / den)


def ccorr(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """
    The k-lag cross-correlation of x with y (ScalaTion's `VectorD.ccorr`, STATIONARY form):
    x_i against y_i+k, the means and standard deviations over the whole vectors,

        ccorr = sum_{i < n-k} (x_i - mu_x) (y_i+k - mu_y) / (n - 1) / (sd_x sd_y)

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        The leading series.
    y : np.ndarray, shape (n,)
        The series it leads.
    k : int, default=1
        The lag (0 <= k < n).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if k >= n:
        raise ValueError(f"ccorr: the vector is not long enough to compute ccorr for lag k = {k}")
    mu_x, mu_y = x.mean(), y.mean()
    ccov = np.sum((x[:n - k] - mu_x) * (y[k:] - mu_y)) / (n - 1)
    den = np.std(x, ddof=1) * np.std(y, ddof=1)
    return float(ccov / den) if den > 0.0 else float("nan")


def corr_matrix(xy: np.ndarray) -> np.ndarray:
    """The correlation matrix of the columns of xy (ScalaTion's `MatrixD.corr`)."""
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[1]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = corr(xy[:, i], xy[:, j]) if i != j else 1.0
    return out


def model_screen(xy: np.ndarray, thr1: float = 0.05, thr2: float = 0.95,
                 dep: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Screens the x-columns of the [ x, y ] matrix xy on two thresholds (ScalaTion's
    `Model.screen`): a column is a candidate when its dependency with y (the last column of
    `dep`) exceeds thr1 in absolute value; in decreasing dependency, a candidate is kept when its
    dependency with every column already kept stays below thr2.

    Parameters
    ----------
    xy : np.ndarray, shape (m, n + 1)
        The combined data-response matrix, y in the last column.
    thr1 : float, default=0.05
        The smallest dependency with y kept.
    thr2 : float, default=0.95
        The largest dependency/collinearity with an already kept column tolerated.
    dep : np.ndarray, shape (n + 1, n + 1), optional
        The dependency measure (the correlation matrix of xy by default).

    Returns
    -------
    (np.ndarray, np.ndarray)
        The reduced matrix xy[:, selected] and the selected column indices (ascending).
    """
    xy = np.asarray(xy, dtype=float)
    dep = corr_matrix(xy) if dep is None else np.asarray(dep, dtype=float)
    lst = dep.shape[1] - 1                                               # the index of the last column (holds y)
    depY, depX = dep[:, lst], dep[:lst, :lst]
    indices = [i for i in range(lst) if abs(depY[i]) > thr1]             # above the minimal dependency level
    s_indices = sorted(indices, key=lambda i: -abs(depY[i]))             # from the highest dependency to the lowest
    selected: List[int] = []
    for i in s_indices:
        if all(abs(depX[i, k]) < thr2 for k in selected):               # below the collinearity cut-off
            selected.append(i)
    selected_ = np.array(sorted(selected), dtype=int)
    return xy[:, selected_], selected_


# ---------------------------------------------------------------------------------------------
#  The screening results
# ---------------------------------------------------------------------------------------------

@dataclass
class ExoRank:
    """
    One candidate's screening result (ScalaTion's `ExoRank`).

    Attributes
    ----------
    name : str
        The candidate exogenous variable.
    bestLag : int
        The lag (1..maxLag) at which |ccf| with the response is largest.
    ccf : float
        The cross-correlation at that lag (signed).
    corr0 : float
        The lag-0 (contemporaneous) correlation with the response.
    """
    name: str
    bestLag: int
    ccf: float
    corr0: float

    @property
    def dep(self) -> float:
        """The dependency with the response, |ccf|."""
        return abs(self.ccf)

    @property
    def leads(self) -> bool:
        """Whether the candidate is stronger ahead of y than alongside it."""
        return self.dep > abs(self.corr0)

    def __str__(self) -> str:
        return (f"{self.name:<40s} lag {self.bestLag:2d}  ccf {self.ccf:7.3f}  corr0 {self.corr0:7.3f}"
                + ("  leads" if self.leads else ""))


@dataclass
class ExoPool:
    """
    The eligible pool cut from a data matrix (ScalaTion's `ExoPool`).

    Attributes
    ----------
    y : np.ndarray
        The response series (full length).
    xe : dict of str -> np.ndarray
        The candidate series by name, in column order (full length).
    dropped : list of (str, str)
        The columns left out, each with the reason.
    """
    y: np.ndarray
    xe: Dict[str, np.ndarray]
    dropped: List[Tuple[str, str]]


# ---------------------------------------------------------------------------------------------
#  ExoScreen
# ---------------------------------------------------------------------------------------------

class ExoScreen:
    """
    Pools, ranks and screens candidate exogenous variables (ScalaTion's `ExoScreen` object).
    Every method is static; a data set is a matrix with a header or a pandas DataFrame.
    """

    debug = False                                                      # print the debug lines of the Scala object

    @staticmethod
    def _data(data, header: Optional[Sequence[str]]) -> Tuple[np.ndarray, List[str]]:
        """The (matrix, header) pair of a matrix with a header or of a DataFrame (columns coerced to
        float; a non-numeric column becomes NaN and is dropped by `pool` with the reason)."""
        if header is None:
            if not hasattr(data, "columns"):
                raise ValueError("pool: a header is needed for a matrix (or pass a DataFrame)")
            cols = [str(c) for c in data.columns]
            mat = np.column_stack([pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float) for c in data.columns]) \
                if len(cols) > 0 else np.zeros((len(data), 0))
            return mat, cols
        return np.asarray(data, dtype=float), [str(h) for h in header]

    @staticmethod
    def finite(x: np.ndarray) -> bool:
        """Whether every value of the vector is finite."""
        return bool(np.all(np.isfinite(x)))

    @staticmethod
    def monotone(x: np.ndarray) -> float:
        """The fraction of the steps x_i - x_i-1 that go one way (all up or all down => 1.0), the
        signature of a cumulative series."""
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            return 0.0
        dx = np.diff(x)
        up, dn = int(np.sum(dx >= 0)), int(np.sum(dx <= 0))
        return max(up, dn) / max(1, len(x) - 1)

    @classmethod
    def pool(cls, data, header: Optional[Sequence[str]] = None, resp: str = "y", tr: int = 0,
             exclude: Iterable[str] = (), thrMono: float = 0.98) -> ExoPool:
        """
        Cuts the eligible pool from a data matrix with a header: every column but the response
        that is not excluded, has finite values, and is neither constant nor cumulative on the
        training rows.  A cumulative (running total) column moves in one direction for at least
        thrMono of its steps: it carries a trend, not a driver signal, so its correlation with the
        response would be spurious.

        Parameters
        ----------
        data : np.ndarray, shape (m, n) or pandas.DataFrame
            The data matrix (rows = time, columns = named series).
        header : sequence of str, optional
            The column names (one per column of data; the DataFrame's columns when None).
        resp : str
            The name of the response column.
        tr : int
            The number of leading (training) rows the rules look at (0 => `trSize` of the rows).
        exclude : iterable of str
            The columns domain knowledge rules out (e.g., proxies of the response).
        thrMono : float, default=0.98
            The fraction of one-direction steps from which a column is cumulative.

        Returns
        -------
        ExoPool
        """
        mat, names = cls._data(data, header)
        m = mat.shape[0]
        tr = tr if tr > 0 else trSize(m)
        if len(names) != mat.shape[1]:
            raise ValueError(f"pool: {len(names)} names for {mat.shape[1]} columns")
        if not 0 < tr <= m:
            raise ValueError(f"pool: tr = {tr} must be in 1..{m}")
        if resp not in names:
            raise ValueError(f"pool: response '{resp}' not in the header {', '.join(names)}")
        jy = names.index(resp)
        y = mat[:, jy]
        if not cls.finite(y):
            print(f"pool: the response '{resp}' has non-finite values")
        exclude = set(exclude)
        xe: Dict[str, np.ndarray] = {}                                   # insertion (column) order kept
        dropped: List[Tuple[str, str]] = []
        for j, nm in enumerate(names):
            if j == jy:
                continue
            x = mat[:, j]
            if nm in exclude:
                why = "excluded (domain knowledge)"
            elif not cls.finite(x):
                why = "non-finite values (not a numeric column?)"
            elif np.std(x[:tr], ddof=1) == 0.0:
                why = "constant on the training rows"
            else:
                mono = cls.monotone(x[:tr])
                why = f"cumulative: {mono:.2f} of its steps go one way on the training rows" if mono >= thrMono else None
            if why is None:
                xe[nm] = x
            else:
                if cls.debug:
                    print(f"pool: {nm} dropped: {why}")
                dropped.append((nm, why))
        return ExoPool(y, xe, dropped)

    @staticmethod
    def ccf(y: np.ndarray, x: np.ndarray, maxLag: int, tr: int) -> np.ndarray:
        """
        The cross-correlation of the response y_t with x_t-k for lags k = 1..maxLag on the first
        tr rows (NaN where undefined, e.g., a constant x).

        Parameters
        ----------
        y : np.ndarray
            The response series (full length).
        x : np.ndarray
            The candidate series (full length).
        maxLag : int
            The maximum lag examined.
        tr : int
            The number of leading (training) rows used.
        """
        yTr, xTr = np.asarray(y, dtype=float)[:tr], np.asarray(x, dtype=float)[:tr]
        return np.array([ccorr(xTr, yTr, k) for k in range(1, maxLag + 1)])

    @classmethod
    def rank(cls, y: np.ndarray, xe: Dict[str, np.ndarray], maxLag: int, tr: int) -> List[ExoRank]:
        """
        Ranks the candidates by the largest |ccf| over lags 1..maxLag, strongest first, each with
        its lag-0 correlation; candidates whose ccf is undefined (non-finite data, a constant
        series) are left out.

        Parameters
        ----------
        y : np.ndarray
            The response series (full length).
        xe : dict of str -> np.ndarray
            The candidate series by name (full length).
        maxLag : int
            The maximum lag examined.
        tr : int
            The number of leading (training) rows used.
        """
        ranks: List[ExoRank] = []
        for nm, x in sorted(xe.items(), key=lambda kv: kv[0]):
            c = cls.ccf(y, x, maxLag, tr)
            if not cls.finite(c):
                if cls.debug:
                    print(f"rank: {nm}: ccf undefined, left out")
                continue
            k = int(np.argmax(np.abs(c)))
            ranks.append(ExoRank(nm, k + 1, float(c[k]), corr(np.asarray(x, dtype=float)[:tr], np.asarray(y, dtype=float)[:tr])))
        return sorted(ranks, key=lambda r: -r.dep)

    @staticmethod
    def xyOf(ranks: Sequence[ExoRank], y: np.ndarray, xe: Dict[str, np.ndarray], tr: int) -> np.ndarray:
        """
        The [ x, y ] matrix of the training rows with the candidates in ranking order (strongest
        first) and y as the last column -- the layout `model_screen` expects.
        """
        cols = [np.asarray(xe[r.name], dtype=float)[:tr] for r in ranks] + [np.asarray(y, dtype=float)[:tr]]
        return np.column_stack(cols)

    @staticmethod
    def dependency(ranks: Sequence[ExoRank], xy: np.ndarray) -> np.ndarray:
        """
        The dependency matrix `model_screen` needs for time-series candidates: its x-vs-x block
        is the lag-0 correlation of the candidates (collinearity), while its x-vs-y column and
        row hold each candidate's strongest cross-correlation LEADING y over lags 1..maxLag
        (signed, from the ranking) instead of the lag-0 correlation -- a driver is judged by how
        it moves ahead of the response, not alongside it.

        Parameters
        ----------
        ranks : sequence of ExoRank
            The ranking, in the column order of xy.
        xy : np.ndarray
            The [ x, y ] matrix of the training rows (`xyOf`).
        """
        n = len(ranks)
        if xy.shape[1] != n + 1:
            raise ValueError(f"dependency: {xy.shape[1]} columns for {n} candidates + y")
        dep = corr_matrix(xy)                                            # lag-0 correlations on the training rows
        for i in range(n):
            dep[i, n] = ranks[i].ccf                                     # x_i vs y: its strongest lead
            dep[n, i] = ranks[i].ccf
        return dep

    @classmethod
    def shortlist(cls, y: np.ndarray, xe: Dict[str, np.ndarray], maxLag: int, tr: int, k: int,
                  thr1: float = 0.05, thr2: float = 0.95) -> Tuple[List[ExoRank], List[str]]:
        """
        Ranks the candidates and keeps a shortlist of at most k of them with `model_screen` on the
        time-series dependency matrix: in decreasing lead dependency, keep a candidate when its
        |ccf| exceeds thr1 and its lag-0 correlation with every candidate already kept is below
        thr2; the screen's first k (its greedy pass accepts in dependency order, so the first k of
        its uncapped result are exactly the capped result).

        Parameters
        ----------
        y : np.ndarray
            The response series (full length).
        xe : dict of str -> np.ndarray
            The candidate series by name (full length).
        maxLag : int
            The maximum lag examined.
        tr : int
            The number of leading (training) rows used.
        k : int
            The largest shortlist wanted.
        thr1 : float, default=0.05
            The smallest dependency with the response kept.
        thr2 : float, default=0.95
            The largest correlation with an already kept candidate tolerated.

        Returns
        -------
        (list of ExoRank, list of str)
            The ranking and the shortlist (in ranking order).
        """
        ranks = cls.rank(y, xe, maxLag, tr)
        if not ranks:
            return ranks, []
        xy = cls.xyOf(ranks, y, xe, tr)
        _, sel = model_screen(xy, thr1, thr2, cls.dependency(ranks, xy))
        kept = [ranks[i].name for i in sorted(sel.tolist())[:k]]        # ascending index = ranking order
        if cls.debug:
            rest = [r.name for r in ranks if r.dep > thr1 and r.name not in kept]
            print(f"shortlist: passed thr1 but not kept (collinear or beyond k = {k}): {', '.join(rest)}")
        return ranks, kept

    @classmethod
    def screen(cls, data, header: Optional[Sequence[str]] = None, resp: str = "y", k: int = 2, maxLag: int = 6,
               tr: int = 0, exclude: Iterable[str] = (), thr1: float = 0.05, thr2: float = 0.95,
               show: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        The one-call screen for a forecaster: from a data matrix with a header (or a DataFrame),
        the response's name and k, returns the exogenous matrix of the k variables chosen on the
        training rows -- all rows, columns in the file's order -- the response vector and the
        chosen names (in that order): the (xe, y) pair `ARX_D (hparam, y, hh, xe)` and the other
        forecasters take.

        Parameters
        ----------
        data : np.ndarray or pandas.DataFrame
            The data matrix (rows = time, columns = named series).
        header : sequence of str, optional
            The column names (the DataFrame's when None).
        resp : str
            The name of the response column.
        k : int
            The number of exogenous variables wanted (at most).
        maxLag : int, default=6
            The maximum lead examined (the lag window of the forecaster).
        tr : int, default=0
            The number of training rows (0 => `trSize` of the rows).
        exclude : iterable of str
            The columns domain knowledge rules out (proxies of the response).
        thr1, thr2 : float
            The dependency thresholds of `shortlist`.
        show : bool, default=True
            Whether to print the dropped columns, the ranking and the choice.

        Returns
        -------
        (np.ndarray, np.ndarray, list of str)
            xe (m, n_chosen), y (m,) and the chosen names in the file's order.
        """
        mat, names = cls._data(data, header)
        tr_ = tr if tr > 0 else trSize(mat.shape[0])
        p = cls.pool(mat, names, resp, tr_, exclude)
        ranks, kept = cls.shortlist(p.y, p.xe, maxLag, tr_, k, thr1, thr2)
        chosen = [nm for nm in names if nm in kept]                       # the chosen columns in the file's order
        xe = np.column_stack([p.xe[nm] for nm in chosen]) if chosen else np.zeros((mat.shape[0], 0))
        if show:
            banner(f"ExoScreen.screen: response {resp}, {len(p.xe)} candidates, first {tr_} rows, leads 1..{maxLag}")
            for nm, why in p.dropped:
                print(f"dropped {nm:<40s} {why}")
            for r in ranks:
                print(r)
            print(f"chosen (k = {k}, in the file's order): {', '.join(chosen)}")
        return xe, p.y, chosen
