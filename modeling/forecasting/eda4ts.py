"""
EDA4TS: Exploratory Data Analysis (EDA) Profile of a Time Series

Everything here is computed from the TRAINING rows only (rows 0 until trSize, ScalaTion's
`Model.trSize`), so nothing from the testing rows reaches a model choice.  The profile is what a
statistician looks at before writing an ARX formula: the ACF and PACF of the response (which lags
matter), the periodogram's top frequencies (seasonality), the cross-correlation of the response with
each exogenous variable by lag (which lag of which driver, and with which sign), the level and spread
(whether a log transform is plausible), the share of zeros (whether log is safe) and a KPSS
stationarity test.  `residualReport` applies the residual tests (Ljung-Box, ACF spikes, collinearity,
heteroscedasticity: `Util/correlogram.py`) to a fitted model.  The LLM-NARX-SR search writes the
profile in words for its prompts (`LLMProposer.promptBlock`) and ranks wave frequencies by `topFreqs`.

The Python counterpart of ScalaTion's `forecasting/EDA4TS.scala` (the class and its companion's
`amplitudes`, `topFreqs`).  Differences: the periodogram amplitudes come from numpy's FFT (ScalaTion: a
direct DFT; the same values); `periods` (the periods of `topFreqs`) is a Python helper.  The KPSS test
is ScalaTion's `Stationarity_KPSS (yTr, 0, "short", "c")` (forecasting/stationarity_kpss.py).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.6
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from Util.correlogram import acfPacf, acfSpikes, ccorr, collinearPair, heteroscedastic, ljungBox
from modeling.forecasting.stationarity_kpss import Stationarity_KPSS

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



def periods(y: np.ndarray, k: int = 2) -> np.ndarray:
    """The periods (steps per cycle) of the top k frequencies."""
    return 1.0 / topFreqs(y, k)


# ---------------------------------------------------------------------------------------------
#  EDA4TS
# ---------------------------------------------------------------------------------------------

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
        The response variable's name (the key `LLMProposer.promptBlock`'s name map uses).

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
        """The KPSS level-stationarity test (`Stationarity_KPSS`, the "short" lag rule), (NaN, NaN) on failure."""
        try:
            test = Stationarity_KPSS(y, 0, "short", "c")
            return test.statistic(), test.pvalue()
        except (ValueError, ZeroDivisionError, FloatingPointError):
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
