"""
Matrix Utilities for Time Series Forecasting

Provides utility functions to construct lagged and trend-based feature matrices,
and handle missing initial values via backcasting and backfilling.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""


import numpy as np

q_mean = 2  # Number of prior values used for weighted average in backcasting

def backcast(y_: np.ndarray, i: int = 0) -> float:
    """
    Computes a weighted backward average using previous values.

    Parameters
    ----------
    y_ : np.ndarray, shape (n_samples,)
        Input time series.
    i : int, default=0
        Starting index for backcasting.

    Returns
    -------
    float
        Weighted average of the past `q_mean` values.
    """
    q_m = q_mean
    ww = np.arange(1, q_m + 1)
    b = ww / ww.sum()
    yy = y_[i:q_m + i][::-1]
    return np.dot(b, yy)

def backfill(zj: np.ndarray) -> np.ndarray:
    """
    Replaces leading zeros in a 1D array using weighted backcasting.

    Parameters
    ----------
    zj : np.ndarray, shape (n_samples,)
        A 1D array with possible leading zeros.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Array with initial zeros replaced by estimated values.
    """
    z_j = zj.copy().astype(float)
    z_j = np.concatenate(([0.0], z_j))
    ii = np.argmax(z_j != 0.0) - 1

    for i_ in range(ii + 1):
        i = ii - i_
        z_j[i] = backcast(z_j, i)
    return z_j[1:]

def backfill_matrix(z: np.ndarray) -> np.ndarray:
    """
    Applies backfilling column-wise to a matrix.

    Parameters
    ----------
    z : np.ndarray, shape (n_samples, n_features)
        Matrix of exogenous variables with potential initial zeros.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Matrix with each column backfilled independently.
    """
    z_bfill = z.copy().astype(float)
    for j in range(z.shape[1]):
        z_bfill[:, j] = backfill(z[:, j])
    return z_bfill

def build_trend_matrix(m: int, spec: int = 1, lwave: int = 20) -> np.ndarray:
    """
    Constructs a matrix of trend components (e.g., linear, quadratic, sine, cosine).

    Parameters
    ----------
    m : int
        Number of time steps (rows).
    spec : int, default=1
        Type of trend to include:
            1 = Constant
            2 = Linear
            3 = Quadratic
            4 = Sine
            5 = Cosine
    lwave : int, default=20
        Wavelength used for periodic trends (sine and cosine).

    Returns
    -------
    np.ndarray, shape (m, spec - 1)
        Trend matrix matching the specified trend type.
    """
    m2 = m / 2.0
    w = (2 * np.pi) / lwave  
    x = np.zeros((m, spec - 1))  
    t_0m = np.arange(m)

    if spec >= 2:
        x[:, 0] = t_0m / m                        
    if spec >= 3:
        x[:, 1] = ((t_0m - m2) ** 2) / (m2 ** 2)  
    if spec >= 4:
        x[:, 2] = np.sin(t_0m * w)                
    if spec == 5:
        x[:, 3] = np.cos(t_0m * w)                
    return x

def build_lagged_matrix(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Constructs a lagged matrix from a 1D time series.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        1D time series data.
    lag : int
        Number of lags (columns) to generate.

    Returns
    -------
    np.ndarray, shape (n_samples, lag)
        Lagged matrix where each column i contains x shifted by i+1 time steps.
    """
    first = x[0]
    ones_ = first * np.ones(lag)
    xRow = np.concatenate((ones_, x[:-1]))
    xx = np.column_stack([xRow[i:len(xRow) - lag + i + 1] for i in range(lag)])
    return xx



def build_matrix_Y(y: np.ndarray, hh: int) -> np.ndarray:
    """
    Builds a matrix of future target values for multi-step forecasting.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        1D time series target data.
    hh : int
        Maximum forecasting horizon.

    Returns
    -------
    np.ndarray, shape (n_samples, hh)
        Target matrix where column i contains the i-step ahead value of `y`.
    """
    zeros_ = np.zeros(hh-1)
    yRow = np.concatenate((y, zeros_))
    Y = np.column_stack([yRow[i:len(yRow) - hh + i + 1] for i in range(hh)])
    return Y

