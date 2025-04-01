"""
Utility functions for building input matrices and handling missing values 
in time series forecasting models.

Includes:
- Backcasting and backfilling methods
- Trend and lag matrix builders
- Initial forecast matrix creation

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

import numpy as np

q_mean = 2  # Number of prior values used for weighted average in backcasting

def backcast(y_: np.ndarray, i: int = 0) -> float:
    """
    Computes a weighted average of previous values in reverse order.

    Parameters
    ----------
    y_ : np.ndarray
        Input time series.
    i : int, default=0
        Starting index to apply the backcast.

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
    Fills initial zeros in a 1D array using weighted backcasting.

    Parameters
    ----------
    zj : np.ndarray
        A 1D array (e.g., column of exogenous variable) with possible leading zeros.

    Returns
    -------
    np.ndarray
        A 1D array with initial zeros replaced by backcasted values.
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
    Applies backfill to each column of a 2D matrix.

    Parameters
    ----------
    z : np.ndarray
        Matrix of exogenous variables with possible initial zeros.

    Returns
    -------
    np.ndarray
        Backfilled version of input matrix.
    """
    z_bfill = z.copy().astype(float)
    for j in range(z.shape[1]):
        z_bfill[:, j] = backfill(z[:, j])
    return z_bfill

def build_trend_matrix(m: int, spec: int = 1, lwave: int = 20) -> np.ndarray:
    """
    Builds a trend component matrix for time series modeling.

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
        Wavelength used for sine/cosine components.

    Returns
    -------
    np.ndarray
        Trend matrix of shape (m, spec - 1).
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
    Constructs a lagged feature matrix from a 1D time series.

    Parameters
    ----------
    x : np.ndarray
        1D time series data.
    lag : int
        Number of lag steps.

    Returns
    -------
    np.ndarray
        A 2D matrix of shape (len(x), lag) where each column is a lagged version of x.
    """
    first = x[0]
    ones_ = first * np.ones(lag)
    xRow = np.concatenate((ones_, x[:-1]))
    xx = np.column_stack([xRow[i:len(xRow) - lag + i + 1] for i in range(lag)])
    return xx

def initial_Yf(y: np.ndarray, hh: int) -> np.ndarray:
    """
    Creates an initial forecast matrix with placeholder values.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        The response vector (time series data).
    hh : int
        The maximum forecasting horizon (h = 1 to hh).

    Returns
    -------
    np.ndarray, shape (n_samples, hh + 2)
        A matrix where:
        - First column contains the original time series y.
        - Next hh columns are initialized to zeros (for forecasts).
        - Last column contains the time index.
    """
    y_fcast = np.zeros((y.shape[0], hh))
    time = np.arange(0, y.shape[0]).reshape(-1, 1)
    yf = np.hstack([y.reshape(-1, 1), y_fcast, time])
    return yf
