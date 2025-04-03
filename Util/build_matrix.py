"""
Matrix Utilities for Time Series Forecasting

This module provides utility functions for preprocessing time series data
in preparation for forecasting tasks. It includes functionality for:

- Constructing lagged feature matrices from the time series data.
- Generating trend-based feature matrices (e.g., linear, quadratic, sine, cosine).
- Handling missing or initial zero values using weighted backcasting and backfilling.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
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

def build_trend_matrix(m: int, spec: int = 1, lwave: int = 20) -> np.ndarray:
    """
    Construct a matrix of trend components (e.g., linear, quadratic, sine, cosine).

    Parameters
    ----------
    m : int
        Number of time steps (rows).
    spec : int, default=1
        Type of trend to include (cumulative):
            1=Constant, 2=Linear, 3=Quadratic, 4=Sine, 5=Cosine.
    lwave : int, default=20
        Wavelength used for periodic trends (sine and cosine).

    Returns
    -------
    np.ndarray, shape (m, spec - 1)
        Matrix of trend features based on the selected specification.
    
    Raises
    ------
    ValueError
        If `spec` is not in the valid range [1, 2, 3, 4, 5].
    """
    if spec not in [1, 2, 3, 4, 5]:
        raise ValueError("Trend specification out of range")
    
    m2 = m / 2.0
    w = (2 * np.pi) / lwave  
    x_t = np.zeros((m, spec - 1))  
    t_0m = np.arange(m)

    if spec >= 2:
        x_t[:, 0] = t_0m / m                        
    if spec >= 3:
        x_t[:, 1] = ((t_0m - m2) ** 2) / (m2 ** 2)  
    if spec >= 4:
        x_t[:, 2] = np.sin(t_0m * w)                
    if spec == 5:
        x_t[:, 3] = np.cos(t_0m * w)  

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
        Matrix where each column i contains values lagged by i+1 steps.

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
    Build a matrix of future target values for multi-step forecasting.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        The time series target values.
    hh : int
        The maximum forecasting horizon.

    Returns
    -------
    np.ndarray, shape (n_samples, hh)
        Target matrix where column i contains the i-step-ahead values of `y`.
    """
    if hh > 1:
        zeros_ = np.zeros(hh-1)
        yRow = np.concatenate((y, zeros_))
        Y = np.column_stack([yRow[i:len(yRow) - hh + i + 1] for i in range(hh)])
    else:
        Y = y.reshape(-1, 1)

    return Y