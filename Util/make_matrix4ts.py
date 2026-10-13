"""
MakeMatrix4TS: Matrices from Lagged Endogenous and Exogenous Variables

The Python counterpart of ScalaTion's `MakeMatrix4TS.scala` (its object's methods, same names):

    makeMatrix4L   the lag matrix of a series (lags p .. 1, DESCENDING, first values padded)
    makeMatrix4T   the trend columns (`spec`: 1 linear, 2 + quadratic, 3 + cubic; 4, 5 sin / cos)
    makeMatrix4Y   the target matrix of a direct model (column h - 1 = y shifted by h - 1, NaN = NO_DOUBLE)
    backcast, backfill, backfill_matrix   weighted backcasting of the zero prefix of an exogenous variable
    trend_names    the trend-column names of `MakeMatrix4TS.trend` ("cons", "lin", "quad", ...)
    centered       whether a response transform maps y into a space where it can be negative
                   (CenterForm, NormForm, MinMaxForm): forecasts there must not be rectified

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
"""

from typing import Any

import numpy as np

from Util.transform import CenterForm, MinMaxForm, NormForm

q_mean = 2  # Number of prior values used for weighted average in backcasting

def backcast(z: np.ndarray, i: int = 0) -> float:
    """
    Compute a weighted average of past values for backcasting.

    Parameters
    ----------
    z : np.ndarray, shape (n_samples,)
        1D array representing the time series data.
    i : int, default=0
        Index at which to begin backcasting.

    Returns
    -------
    float
        Weighted average of the past `q_mean` values.
    """
    q_m = q_mean
    ww = np.arange(1, q_m + 1)
    b = ww / ww.sum()
    z_inv = z[i:q_m + i][::-1]
    return np.dot(b, z_inv)

def backfill(xej: np.ndarray) -> np.ndarray:
    """
    Estimate and replace leading zeros in a 1D time series using weighted backcasting.

    Parameters
    ----------
    xej : np.ndarray, shape (n_samples,)
        Input array containing potential leading zeros.

    Returns
    -------
    np.ndarray
        Array with leading zeros replaced by backcast values.
    """
    xe_j = np.concatenate(([0.0], xej))
    ii = np.argmax(xe_j != 0.0) - 1

    for i_ in range(ii + 1):
        i = ii - i_
        xe_j[i] = backcast(xe_j, i)

    return xe_j[1:]

def backfill_matrix(xe: np.ndarray) -> np.ndarray:
    """
    Apply backfilling to each column of a matrix.

    Parameters
    ----------
    xe : np.ndarray, shape (n_samples, n_exo)
        The matrix of exogenous variables with potential initial zeros.

    Returns
    -------
    np.ndarray, shape (n_samples, n_exo)
        Matrix with each column backfilled independently.
    """
    xe_bfill = xe.copy().astype(float)

    for j in range(xe.shape[1]):
        xe_bfill[:, j] = backfill(xe[:, j])

    return xe_bfill

TREND = ["cons", "lin", "quad", "sin", "cos"]      # the trend-column names of ScalaTion's MakeMatrix4TS.trend

def trend_names(spec: int) -> list:
    """
    The names of the `spec` trend columns, exactly as ScalaTion's `MakeMatrix4TS.formNames` labels
    them (`MakeMatrix4TS.trend`); note that ScalaTion labels the first trend column, which holds
    the linear trend t / m, "cons".
    """
    return [TREND[j] for j in range(spec)]


def makeMatrix4T(m: int, spec: int = 1, lwave: int = 20) -> np.ndarray:
    """
    Constructs the matrix of trend columns (ScalaTion's `MakeMatrix4TS.makeMatrix4T`).

    Parameters
    ----------
    m : int
        Number of time steps (rows).
    spec : int, default=1
        The number of trend columns (cumulative): 0 = implicit constant (the intercept of the
        regression), 1 = linear t / m, 2 = + quadratic ((t - m/2) / (m/2))^2, 3 = + cubic,
        4 = + sin(w t), 5 = + cos(w t) (the last two are Python extras).
    lwave : int, default=20
        Wavelength used for the periodic trends (sine and cosine).

    Returns
    -------
    np.ndarray, shape (m, spec)
        Matrix of trend features based on the selected specification.

    Raises
    ------
    ValueError
        If `spec` is not in [0, 5].
    """
    if spec not in [0, 1, 2, 3, 4, 5]:
        raise ValueError("Trend specification out of range")

    m2 = m / 2.0
    w = (2 * np.pi) / lwave
    x_t = np.zeros((m, spec))
    t_0m = np.arange(m)

    if spec >= 1:
        x_t[:, 0] = t_0m / m
    if spec >= 2:
        x_t[:, 1] = ((t_0m - m2) ** 2) / (m2 ** 2)
    if spec >= 3:
        x_t[:, 2] = ((t_0m - m2) ** 3) / (m2 ** 3)
    if spec >= 4:
        x_t[:, 3] = np.sin(t_0m * w)
    if spec == 5:
        x_t[:, 4] = np.cos(t_0m * w)

    return x_t

def makeMatrix4L(z: np.ndarray, lag: int) -> np.ndarray:
    """
    Constructs the lag matrix of a 1D time series (ScalaTion's `MakeMatrix4TS.makeMatrix4L`).

    Parameters
    ----------
    z : np.ndarray, shape (n_samples,)
        The time series data to be transformed into lagged features.
    lag : int
        Number of lags (columns) to generate.

    Returns
    -------
    np.ndarray, shape (n_samples, lag)
        Matrix whose columns hold the lags in DESCENDING order (lag, lag - 1, ..., 1), as
        ScalaTion's `makeMatrix4L` (column names yl{lag} .. yl1); rows before the lag is
        available are padded with the first value.

    Raises
    ------
    ValueError
        If the input array `z` is not one-dimensional.
    """
    if z.ndim != 1:
        raise ValueError("Input time series z must be 1-dimensional")
    
    first = z[0]
    ones_ = first * np.ones(lag)
    zRow = np.concatenate((ones_, z[:-1]))
    z_lagged = np.column_stack([zRow[i:len(zRow) - lag + i + 1] for i in range(lag)])
    return z_lagged



def makeMatrix4Y(y: np.ndarray, hh: int) -> np.ndarray:
    """
    Builds the matrix of future target values for DIRECT multi-step forecasting (ScalaTion's
    `MakeMatrix4TS.makeMatrix4Y`).

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        The time series target values.
    hh : int
        The maximum forecasting horizon.

    Returns
    -------
    np.ndarray, shape (n_samples, hh)
        Target matrix where column h - 1 holds y shifted by h - 1 (Y[t, h-1] = y[t + h - 1]);
        the entries beyond the end of the series are NaN (ScalaTion's NO_DOUBLE), and the
        training methods drop those rows (`Forecaster_D.fit_rows`).
    """
    m = len(y)
    Y = np.full((m, hh), np.nan)
    for h in range(hh):
        Y[:m - h, h] = y[h:]
    return Y


def centered(tForm_y: Any) -> bool:
    """
    Whether the response transform maps the response into a space where it can be negative --
    centered (`CenterForm`), standardized (`NormForm`) or min-max scaled (`MinMaxForm`, centered to
    [-2, 2] by default) -- so predictions made there must NOT be rectified to be non-negative
    (non-negativity is a property of the ORIGINAL scale); a transform that keeps zero at zero (e.g.,
    `Log1pForm`) may be rectified (ScalaTion's `MakeMatrix4TS.centered`).

    Parameters
    ----------
    tForm_y : Transform or None
        The transform of the response (None => no transform).
    """
    return isinstance(tForm_y, (CenterForm, NormForm, MinMaxForm))
