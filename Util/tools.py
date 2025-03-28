"""
Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""

import numpy as np


class dotdict(dict):
    """
    A dictionary subclass that allows dot notation for attribute access.

    Example
    -------
    d = dotdict({"a": 1, "b": 2})
    print(d.a)  # Outputs: 1
    d.c = 3
    print(d["c"])  # Outputs: 3
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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray, shape (m,)
        Ground truth values.
    y_pred : np.ndarray, shape (m,)
        Predicted values.

    Returns
    -------
    mse_ : float
        Mean squared error.
    """
    mse_ = np.mean((y_true - y_pred) ** 2)
    return mse_


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray, shape (m,)
        Ground truth values.
    y_pred : np.ndarray, shape (m,)
        Predicted values.

    Returns
    -------
    mae_ : float
        Mean absolute error.
    """
    mae_ = np.mean(np.abs(y_true - y_pred))
    return mae_


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the coefficient of determination (R-squared).

    Parameters
    ----------
    y_true : np.ndarray, shape (m,)
        Ground truth values.
    y_pred : np.ndarray, shape (m,)
        Predicted values.

    Returns
    -------
    r2_ : float
        R-squared score.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_total == 0:
        return 0.0
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_ = 1 - (ss_residual / ss_total)
    return r2_


def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Computes the Adjusted R-squared score.

    Parameters
    ----------
    y_true : np.ndarray, shape (m,)
        Ground truth values.
    y_pred : np.ndarray, shape (m,)
        Predicted values.
    n_features : int
        The number of features used in the model.

    Returns
    -------
    r2_adj : float
        Adjusted R-squared score.
    """
    m = len(y_true)
    if m <= n_features + 1:
        return np.nan
    r2_ = r2(y_true, y_pred)
    r2_adj = 1 - (1 - r2_) * (m - 1) / (m - n_features - 1)
    return r2_adj


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : np.ndarray, shape (m,)
        Ground truth values.
    y_pred : np.ndarray, shape (m,)
        Predicted values.

    Returns
    -------
    smape_ : float
        SMAPE score, expressed as a percentage.
    """
    smape_ = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    return smape_



def diagnose(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> dict:
    """
    Computes multiple evaluation metrics for model performance.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    n_features : int, optional
        Number of features in the model (default is 1).

    Returns
    -------
    metrics : dict
        Dictionary containing MSE, MAE, R^2, Adjusted R^2, SMAPE, and m.
    """
    
    return {
        "MSE"   : mse(y_true, y_pred),
        "MAE"   : mae(y_true, y_pred),
        "R2"    : r2(y_true, y_pred),
        "R2Bar" : r2_adjusted(y_true, y_pred, n_features),
        "SMAPE" : smape(y_true, y_pred),
        "m"     : len(y_true)
    }
