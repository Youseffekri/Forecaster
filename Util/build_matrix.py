"""
Matrix Utilities for Time Series Forecasting

This module provides utility functions for preprocessing time series data
in preparation for forecasting tasks. It includes functionality for:

- Constructing lagged feature matrices from the time series data.
- Generating trend-based feature matrices (e.g., linear, quadratic, sine, cosine).
- Handling missing or initial zero values using weighted backcasting and backfilling.

The Python counterpart of ScalaTion's `MakeMatrix4TS.scala` (lags in descending order, trend
columns named as `MakeMatrix4TS.trend`, the target matrix Y padded with NaN for NO_DOUBLE).

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""


import numpy as np

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


def build_trend_matrix(m: int, spec: int = 1, lwave: int = 20) -> np.ndarray:
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

def build_lagged_matrix(z: np.ndarray, lag: int) -> np.ndarray:
    """
    Constructs a lagged matrix from a 1D time series.

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



def build_matrix_Y(y: np.ndarray, hh: int) -> np.ndarray:
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


def fit_scaler(scaler, data: np.ndarray, tr_size: int):
    """
    Fits an sklearn-style scaler on the TRAINING rows only (the first `tr_size` rows), the
    convention of ScalaTion's `rescale` factories: the same (min, max) or (mean, std) serve the
    in-sample and the train-and-test evaluations, and the testing rows never leak into the scale.

    Parameters
    ----------
    scaler : TransformerMixin
        E.g., MinMaxScaler or StandardScaler.
    data : np.ndarray, shape (n_samples,) or (n_samples, n_cols)
        The data (a vector is treated as one column).
    tr_size : int
        The number of training rows.

    Returns
    -------
    TransformerMixin
        The fitted scaler.
    """
    d = data.reshape(-1, 1) if data.ndim == 1 else data
    scaler.fit(d[:tr_size])
    return scaler


def tr_size_of(n: int, te_ratio: float = 0.2) -> int:
    """The number of training rows of a series of n rows (ScalaTion's `Model.trSize`: n - round_up(n * TE_RATIO))."""
    from modeling.forecasting.forecaster import trSize
    return trSize(n, te_ratio)
