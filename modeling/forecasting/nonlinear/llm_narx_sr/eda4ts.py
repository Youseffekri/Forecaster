"""
EDA4TS: Exploratory Data Analysis (EDA) Profile of a Time Series for the LLM-NARX-SR Search
(prompt block and residual report)

Everything here is computed from the TRAINING rows only (rows 0 until trSize, ScalaTion's
`Model.trSize`), so nothing from the testing rows reaches a proposer or the exogenous screening.
The profile is what a statistician would look at before writing an ARX formula: the ACF and PACF
of the response (which lags matter), the periodogram's top frequencies (seasonality), the
cross-correlation of the response with each exogenous variable by lag (which lag of which driver,
and with which sign), the level and spread (whether a log transform is plausible), the share of
zeros (whether log is safe) and a KPSS stationarity test.  `promptBlock` renders it in words for
the LLM prompt (blind variable names by default); `residualReport` applies the residual tests
(Ljung-Box, ACF spikes, collinearity, heteroscedasticity) to a fitted model for the plausibility
table.

The Python counterpart of ScalaTion's `llm_narx_sr/EDA4TS.scala` together with the `mathstat`
helpers it calls (`Correlogram`'s Durbin-Levinson PACF, `ljungBox`, `acfSpikes`, `collinearPair`,
`heteroscedastic`, `VectorD.acorr` / `ccorr`).  Differences: the periodogram amplitudes come from
numpy's FFT (ScalaTion: a direct DFT; the same values); the KPSS test is statsmodels' `kpss`
(regression "c", the "short" lag rule 4 (n/100)^(1/4); its p-value is interpolated inside
[0.01, 0.10] and clipped outside, as ScalaTion's table lookup is); the Ljung-Box p-value comes
from scipy's chi-square.  The module functions `top_freqs`, `periods`, `amplitudes` of the earlier
core are kept.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------------------------
#  Periodogram (ScalaTion's `EDA4TS` companion object)
# ---------------------------------------------------------------------------------------------

def amplitudes(y: np.ndarray) -> np.ndarray:
    """
    The single-sided amplitude spectrum of a centered series: index j is the amplitude of the
    frequency j / n cycles per step, j = 0 .. n // 2 (ScalaTion's direct DFT; numpy's FFT here).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        The series (the mean is removed).

    Returns
    -------
    np.ndarray, shape (n // 2 + 1,)
    """
    y = np.asarray(y, dtype=float)
    yc = y - y.mean()
    return np.abs(np.fft.rfft(yc)) / len(yc)


def topFreqs(y: np.ndarray, k: int = 2, min_period: float = 2.0) -> np.ndarray:
    """
    The k frequencies (cycles per step) with the largest periodogram amplitude, strongest first;
    the zero frequency and periods of `min_period` steps or shorter are excluded (ScalaTion's
    `EDA4TS.topFreqs` ranks the indices 1 .. n/2 - 1: the mean and the Nyquist bin left out).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        The series (use the training rows only).
    k : int, default=2
        The number of frequencies.
    min_period : float, default=2.0
        Periods below this (in steps) are not considered.

    Returns
    -------
    np.ndarray, shape (k,)
        E.g., for a weekly series with an annual cycle, 1/52 = 0.0192 comes first.
    """
    n = len(y)
    amp = amplitudes(y)
    freqs = np.arange(len(amp)) / n
    ok = (freqs > 0.0) & (freqs < 1.0 / min_period)
    idx = np.where(ok)[0]
    order = idx[np.argsort(-amp[idx], kind="stable")]
    return freqs[order[:k]]


top_freqs = topFreqs                                                   # the former name


def periods(y: np.ndarray, k: int = 2) -> np.ndarray:
    """The periods (steps per cycle) of the top k frequencies."""
    return 1.0 / topFreqs(y, k)


# ---------------------------------------------------------------------------------------------
#  Correlation helpers (ScalaTion's `VectorD.acorr` / `ccorr` / `corr`, `Correlogram`)
# ---------------------------------------------------------------------------------------------

def acorr(x: np.ndarray, k: int = 1, adjusted: bool = False) -> float:
    """
    The lag-k autocorrelation of a (stationary) series (ScalaTion's `VectorD.acorr`): the MLE
    estimate sum_{i>=k} z_i z_{i-k} / sum z_i^2, times n / (n - k) when `adjusted`.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k >= n or n < 2:
        return math.nan
    z = x - x.mean()
    den = float(z @ z)
    if den <= 0.0:
        return math.nan
    mle = float(z[k:] @ z[:n - k]) / den
    return mle * n / (n - k) if adjusted else mle


def acov(x: np.ndarray, k: int = 1, adjusted: bool = True) -> float:
    """The lag-k autocovariance (ScalaTion's `VectorD.acov`): divided by n - k when `adjusted`, n - 1 otherwise."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    s = float(np.sum((x[: n - k] - mu) * (x[k:] - mu)))
    return s / ((n - k) if adjusted else (n - 1))


def ccorr(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """
    The lag-k cross-correlation of x with y (ScalaTion's `VectorD.ccorr`): x.ccorr (y, k)
    correlates x_i with y_{i+k} -- for an exogenous x and the response y, the correlation of
    y_t with x_{t-k} (x leading by k steps).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if k >= n:
        return math.nan
    mu, mu_y = x.mean(), y.mean()
    cov = float(np.sum((x[: n - k] - mu) * (y[k:] - mu_y))) / (n - 1)
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    return cov / (sx * sy) if sx > 0.0 and sy > 0.0 else math.nan


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """The (lag-0) correlation of two vectors (ScalaTion's `VectorD.corr`; -0.0 when undefined)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    if sx <= 0.0 or sy <= 0.0:
        return 1.0 if np.array_equal(x, y) else -0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1)) / (sx * sy)


def durbinLevinson(g: np.ndarray, ml: int) -> np.ndarray:
    """
    The Durbin-Levinson recursion on an autocovariance vector g (lags 0..ml): the psi matrix
    whose diagonal is the PACF (ScalaTion's `Correlogram.durbinLevinson`).
    """
    psi = np.zeros((ml + 1, ml + 1))
    r = np.zeros(ml + 1)
    r[0] = g[0]
    for k in range(1, ml + 1):
        s = sum(psi[k - 1, j] * g[k - j] for j in range(1, k))
        a = (g[k] - s) / r[k - 1] if r[k - 1] != 0.0 else 0.0
        psi[k, k] = a
        for j in range(1, k):
            psi[k, j] = psi[k - 1, j] - a * psi[k - 1, k - j]
        r[k] = r[k - 1] * (1.0 - a * a)
    return psi


def acfPacf(y: np.ndarray, maxLag: int, adjusted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    The ACF and PACF of a series at lags 0..maxLag (ScalaTion's `Correlogram.makeCorrelogram`:
    the autocovariances `acov` (adjusted), the ACF their ratio to lag 0, the PACF the diagonal of
    the Durbin-Levinson psi matrix).
    """
    y = np.asarray(y, dtype=float)
    ml = min(len(y) - 1, maxLag)
    acv = np.array([acov(y, k, adjusted) for k in range(ml + 1)])
    acf = acv / acv[0] if acv[0] != 0.0 else np.full(ml + 1, math.nan)
    pacf = np.diag(durbinLevinson(acv, ml)).copy()
    pacf[0] = 1.0
    return acf, pacf


# ---------------------------------------------------------------------------------------------
#  Residual tests (ScalaTion's `mathstat/Correlogram.scala` top-level functions)
# ---------------------------------------------------------------------------------------------

def ljungBox(e: np.ndarray, maxLag: int, dof: int = 0) -> Tuple[float, float]:
    """
    The Ljung-Box portmanteau test for the whiteness of a (residual) series e,
    Q = n (n + 2) sum_{k=1}^{L} r_k^2 / (n - k), r_k the lag-k autocorrelation (MLE form);
    under H0 Q ~ ChiSquare (L - dof).  Returns (Q, p-value); the p-value is NaN when L - dof < 1.
    """
    from scipy.stats import chi2
    e = np.asarray(e, dtype=float)
    n = len(e)
    ml = min(maxLag, n - 1)
    q = n * (n + 2.0) * sum(acorr(e, k, adjusted=False) ** 2 / (n - k) for k in range(1, ml + 1))
    df = ml - dof
    if df < 1:
        return float(q), math.nan
    return float(q), float(max(0.0, chi2.sf(q, df)))


def acfSpikes(e: np.ndarray, maxLag: int, used: Set[int] = frozenset()) -> List[Tuple[int, float]]:
    """
    The lags at which the autocorrelation of e exceeds the approximate 95 % band 2 / sqrt(n),
    strongest first, as (lag, r_k) pairs, skipping the lags in `used`.
    """
    e = np.asarray(e, dtype=float)
    band = 2.0 / math.sqrt(len(e))
    out = [(k, acorr(e, k, adjusted=False)) for k in range(1, min(maxLag, len(e) - 1) + 1)]
    out = [(k, r) for k, r in out if abs(r) > band and k not in used]
    return sorted(out, key=lambda kr: -abs(kr[1]))


def collinearPair(x: np.ndarray, idx: Optional[Sequence[int]] = None) -> Tuple[int, int, float]:
    """
    The most collinear pair among the given columns of x: the pair with the largest |corr|, as
    (i, j, |corr|), or (-1, -1, 0) when fewer than two columns are given.
    """
    x = np.asarray(x, dtype=float)
    cols = list(range(x.shape[1])) if idx is None else list(idx)
    bi, bj, bc = -1, -1, 0.0
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            c = abs(corr(x[:, cols[a]], x[:, cols[b]]))
            if c > bc:
                bi, bj, bc = cols[a], cols[b], c
    return bi, bj, bc


def heteroscedastic(e: np.ndarray, y: np.ndarray, thresh: float = 0.3) -> bool:
    """Whether the spread of e grows (or shrinks) with the level of y: |corr (|e|, y)| > thresh."""
    return abs(corr(np.abs(np.asarray(e, dtype=float)), y)) > thresh


# ---------------------------------------------------------------------------------------------
#  EDA4TS
# ---------------------------------------------------------------------------------------------

def _g3(v: float) -> str:
    """Three significant digits as Java's "%.3g" (trailing zeros kept)."""
    return "%#.3g" % v


class EDA4TS:
    """
    The profile of a response series and its exogenous variables on their training rows
    (ScalaTion's `EDA4TS`).

    Parameters
    ----------
    y : np.ndarray
        The response series (full length; only the first trSize rows are read).
    exo : dict of str -> np.ndarray
        The exogenous series by name (same length as y).
    trSize : int
        The number of leading (training) rows the profile is computed on.
    maxLag : int, default=6
        The maximum lag examined (ACF, PACF, CCF).
    nFreqs : int, default=2
        The number of periodogram frequencies reported.
    respName : str, default="y"
        The response variable's name (the key used in `promptBlock`'s name map).

    Attributes
    ----------
    yTr, exoTr : np.ndarray, dict
        The training rows of the response and of the exogenous series.
    acf, pacf : np.ndarray
        The ACF and PACF of the response at lags 0..maxLag (index = lag).
    band : float
        The ~95 % band for a correlation at this sample size (2 / sqrt n).
    freqs, periods : np.ndarray
        The top periodogram frequencies (cycles per step) and their periods.
    kpssStat, kpssP : float
        The KPSS statistic and p-value (small p => reject level stationarity).
    zeroShare, yMin, yMax, yMean, yStd : float
        The share of zeros in the response (log is unsafe when > 0) and its level and spread.
    """

    def __init__(self, y: np.ndarray, exo: Dict[str, np.ndarray], trSize: int,
                 maxLag: int = 6, nFreqs: int = 2, respName: str = "y") -> None:
        if trSize <= 2 * maxLag + 2:
            raise ValueError(f"EDA4TS: trSize = {trSize} too small for maxLag = {maxLag}")
        self.maxLag, self.nFreqs, self.respName = int(maxLag), int(nFreqs), respName
        self.trSize = int(trSize)
        self.yTr = np.asarray(y, dtype=float)[: trSize]
        self.exoTr = {k: np.asarray(v, dtype=float)[: trSize] for k, v in exo.items()}
        self.n = len(self.yTr)
        self.acf, self.pacf = acfPacf(self.yTr, self.maxLag)
        self.band = 2.0 / math.sqrt(self.n)
        self.freqs = topFreqs(self.yTr, self.nFreqs)
        self.periods = np.array([1.0 / f if f > 0.0 else math.nan for f in self.freqs])
        self.kpssStat, self.kpssP = self._kpss(self.yTr)
        self.zeroShare = float(np.sum(self.yTr == 0.0)) / self.n
        self.yMin, self.yMax = float(self.yTr.min()), float(self.yTr.max())
        self.yMean, self.yStd = float(self.yTr.mean()), float(np.std(self.yTr, ddof=1))

    @staticmethod
    def _kpss(y: np.ndarray) -> Tuple[float, float]:
        """The KPSS level-stationarity test (statsmodels; the "short" lag rule), (NaN, NaN) on failure."""
        try:
            from statsmodels.tsa.stattools import kpss
            lags = int(4 * (len(y) / 100.0) ** 0.25)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p, _, _ = kpss(y, regression="c", nlags=lags)
            return float(stat), float(p)
        except Exception:
            return math.nan, math.nan

    def pacfSpikes(self) -> List[Tuple[int, float]]:
        """The lags (1..maxLag) at which the PACF exceeds the band, strongest first: the lags an AR term should carry."""
        out = [(k, float(self.pacf[k])) for k in range(1, self.maxLag + 1) if abs(self.pacf[k]) > self.band]
        return sorted(out, key=lambda kr: -abs(kr[1]))

    def ccf(self, name: str) -> np.ndarray:
        """The cross-correlation of the response y_t with x_{t-k} for lags k = 1..maxLag of an exogenous variable."""
        x = self.exoTr[name]
        return np.array([ccorr(x, self.yTr, k) for k in range(1, self.maxLag + 1)])

    def bestLags(self) -> List[Tuple[str, int, float]]:
        """Per exogenous variable, its best lag (largest |ccf|) and the correlation there, strongest variable first."""
        out = []
        for nm in sorted(self.exoTr):
            c = self.ccf(nm)
            if np.all(np.isnan(c)):
                continue
            k = int(np.nanargmax(np.abs(c)))
            out.append((nm, k + 1, float(c[k])))
        return sorted(out, key=lambda t: -abs(t[2]))

    def signs(self) -> Dict[str, int]:
        """The expected sign of each exogenous variable's effect from the training correlation at its
        best lag (+1, -1, or 0 when inside the band)."""
        return {nm: (0 if abs(r) <= self.band else (1 if r > 0.0 else -1)) for nm, _, r in self.bestLags()}

    def promptBlock(self, names: Optional[Dict[str, str]] = None) -> str:
        """
        Renders the profile in words for a prompt.  Variables are shown under the given display
        names (blind names y, x1, x2, ... by default), nothing identifies the series, and only
        training-row statistics appear.
        """
        names = {} if names is None else names

        def nm(v: str) -> str:
            return names.get(v, v)

        yn = nm(self.respName)
        pac = ", ".join(f"{k} ({r:.2f})" for k, r in self.pacfSpikes()[:3])
        sb = ["Data profile (training rows only):\n",
              f"- {yn}: level {_g3(self.yMean)}, spread {_g3(self.yStd)}, range {_g3(self.yMin)} to {_g3(self.yMax)}" +
              (f", {self.zeroShare * 100:.0f}% zeros (log is unsafe, prefer log1p)" if self.zeroShare > 0.0 else "") + "\n",
              f"- {yn} autocorrelation at lags 1..{self.maxLag}: " +
              ", ".join(f"{self.acf[k]:.2f}" for k in range(1, self.maxLag + 1)) + "\n",
              f"- {yn} partial autocorrelation spikes (lag (value)): " + (pac if pac else "none") + "\n"]
        if len(self.freqs) > 0:
            sb.append("- seasonality: strongest periods " +
                      ", ".join(f"{self.periods[i]:.1f} steps (wave({i + 1}))" for i in range(len(self.freqs))) + "\n")
        for v, k, r in self.bestLags():
            sb.append(f"- {nm(v)}: strongest correlation with {yn} at lag {k} (r = {r:.2f}" +
                      (", weak)" if abs(r) <= self.band else (", positive)" if r > 0.0 else ", negative)")) + "\n")
        if not math.isnan(self.kpssP):
            sb.append(f"- KPSS level-stationarity p-value {self.kpssP:.2f}" +
                      (f" (non-stationary: lags of {yn} matter)" if self.kpssP < 0.05 else " (stationary)") + "\n")
        return "".join(sb)

    def residualReport(self, e: np.ndarray, x: np.ndarray, names: Sequence[str], yv: np.ndarray
                       ) -> Tuple[List[str], float]:
        """
        The residual tests of a fitted model in words: Ljung-Box whiteness, ACF spikes of the
        residuals, the most collinear pair of design columns and heteroscedasticity (spread
        growing with the level).  Returns the sentences and a cleanliness score in [0, 1] (the
        share of the four tests passed).
        """
        out: List[str] = []
        passed = 0
        q, p = ljungBox(e, self.maxLag)
        if math.isnan(p) or p >= 0.05:
            passed += 1
        out.append(f"Ljung-Box Q({self.maxLag}) = {q:.2f}, p = {p:.3f}" +
                   (" (residuals autocorrelated)" if (not math.isnan(p) and p < 0.05) else " (white)"))
        spikes = acfSpikes(e, self.maxLag)
        if not spikes:
            passed += 1
        else:
            out.append("residual ACF spikes at lags " + ", ".join(f"{k} ({r:.2f})" for k, r in spikes))
        x = np.asarray(x, dtype=float)
        if x.ndim == 2 and x.shape[1] > 1:
            i, j, c = collinearPair(x)
            if c < 0.95:
                passed += 1
            else:
                out.append(f"collinear columns {names[i]} and {names[j]} (|corr| = {c:.2f})")
        else:
            passed += 1
        if heteroscedastic(e, yv):
            out.append("residual spread grows with the level (consider TFORM: Log1p)")
        else:
            passed += 1
        return out, passed / 4.0
