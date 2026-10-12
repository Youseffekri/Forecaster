"""
EDA Helpers for the Formula Search: Periodogram Frequencies

The seasonal terms `wave(k)` of a formula need frequencies; this module finds the top
frequencies of a series' periodogram (cycles per step) so that `wave(1)` is the strongest
cycle of the training rows, `wave(2)` the two strongest, and so on.

The Python counterpart of ScalaTion's `EDA4TS.topFreqs` (a direct DFT there; numpy's FFT
here, the same amplitudes).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import numpy as np


def amplitudes(y: np.ndarray) -> np.ndarray:
    """
    The periodogram amplitudes of a (centered) series: index j is the amplitude of the
    frequency j / n cycles per step, j = 0 .. n // 2.

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


def top_freqs(y: np.ndarray, k: int = 2, min_period: float = 2.0) -> np.ndarray:
    """
    The k frequencies (cycles per step) with the largest periodogram amplitude, strongest
    first; the zero frequency and periods shorter than `min_period` steps are excluded.

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
    ok = (freqs > 0.0) & (freqs <= 1.0 / min_period)
    idx = np.where(ok)[0]
    order = idx[np.argsort(-amp[idx], kind="stable")]
    return freqs[order[:k]]


def periods(y: np.ndarray, k: int = 2) -> np.ndarray:
    """The periods (steps per cycle) of the top k frequencies."""
    return 1.0 / top_freqs(y, k)
