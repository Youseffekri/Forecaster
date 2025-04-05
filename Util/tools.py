"""
Utility Functions for Model Evaluation and Dot-Accessible Dictionaries

Includes common regression evaluation metrics (MSE, MAE, R2, Adjusted R2, SMAPE)
and a dot-accessible dictionary class for convenience.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05
Version: 1.0
License: MIT License
"""


import numpy as np


class dotdict(dict):
    """
    A dictionary subclass that allows dot notation for attribute access.

    Examples
    --------
    d = dotdict({"a": 1, "b": 2})
    print(d.a)    # Outputs: 1
    d.c = 3
    print(d["c"]) # Outputs: 3
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
    y_true : np.ndarray, shape (n_samples,)
        Ground truth values.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth values.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the coefficient of determination (R-squared).

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth values.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        R-squared score.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_total == 0:
        return 0.0
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return float(1 - (ss_residual / ss_total))


def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Computes the Adjusted R-squared score.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth values.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values.
    n_features : int
        The number of features used in the model.

    Returns
    -------
    float
        Adjusted R-squared score.
    """
    m = len(y_true)
    if m <= n_features + 1:
        return np.nan
    r2_ = r2(y_true, y_pred)
    return float(1 - (1 - r2_) * (m - 1) / (m - n_features - 1))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth values.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        SMAPE score, expressed as a percentage.
    """
    return float(100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)))


def diagnose(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> dict:
    """
    Computes multiple evaluation metrics for model performance.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    n_features : int, default = 1
        The number of features used in the model.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - MSE   : Mean Squared Error.
        - MAE   : Mean Absolute Error.
        - R2    : R-squared.
        - R2Bar : Adjusted R-squared.
        - SMAPE : Symmetric Mean Absolute Percentage Error.
        - m     : The number of samples.
    """
    return {
        "MSE"   : mse(y_true, y_pred),
        "MAE"   : mae(y_true, y_pred),
        "R2"    : r2(y_true, y_pred),
        "R2Bar" : r2_adjusted(y_true, y_pred, n_features),
        "SMAPE" : smape(y_true, y_pred),
        "m"     : len(y_true)
    }



def fit_map(y_true_arr: list[np.ndarray], y_fcast_arr: list[np.ndarray]) -> str:
    """
    Calculate multiple quality-of-fit metrics for each forecast horizon.

    For each horizon, computes:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Coefficient of Determination (R² and Adjusted R²)
    - Symmetric Mean Absolute Percentage Error (SMAPE)
    - Number of samples (m)

    Parameters
    ----------
    y_true_arr : list of np.ndarray
        List of true values for each horizon (aligned to forecast values).
    y_fcast_arr : list of np.ndarray
        List of forecasted values for each horizon.

    Returns
    -------
    str
        A formatted string reporting the metrics across all horizons.
    """
    horizon_all = f"horizon\t->"
    mse_all     = f"mse\t->"
    mae_all     = f"mae\t->"
    r2_all      = f"R^2\t->"
    r2Bar_all   = f"R^2Bar\t->"
    smape_all   = f"smape\t->"
    m_all       = f"m\t->"

    for h in range(len(y_true_arr)):
        metrics = diagnose(y_true_arr[h], y_fcast_arr[h])
        horizon_all += f"\t  {h + 1}\t"
        mse_all     += f"\t{metrics['MSE']:.4f}\t"
        mae_all     += f"\t{metrics['MAE']:.4f}\t"
        r2_all      += f"\t{metrics['R2']:.4f}\t"
        r2Bar_all   += f"\t{metrics['R2Bar']:.4f}\t"
        smape_all   += f"\t{metrics['SMAPE']:.4f}\t"
        m_all       += f"\t{metrics['m']}\t"

    qof = (horizon_all + "\n" +
            mse_all    + "\n" +
            mae_all    + "\n" +
            r2_all     + "\n" +
            r2Bar_all  + "\n" +
            smape_all  + "\n" +
            m_all      + "\n")
    return qof