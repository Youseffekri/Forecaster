"""
Forecaster_D Module

Defines the abstract `Forecaster_D` class for DIRECT (non-recursive) multi-horizon forecasting --
the Python counterpart of ScalaTion's `Forecaster_D.scala`: one regression per horizon on the
same input matrix, the targets held in the matrix Y (column h - 1 = y shifted by h - 1, padded
with NaN where the future is unknown, ScalaTion's NO_DOUBLE).  Subclasses implement `train` and
`forecast`; `fit_rows` drops the padded rows before training, as ScalaTion's `train` does.
The feature-selection scenarios of ScalaTion's `Forecaster_D` (`selectFeatures`,
`selectFeaturesPerH`, `selectFeaturesAtH`, `reportSelection`) are provided by the
`FeatureSelection_D` mixin of `modeling/forecasting/feature_selection_d.py`.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from Util.build_matrix import build_matrix_Y
from modeling.forecasting.feature_selection_d import FeatureSelection_D
from modeling.forecasting.forecaster import Forecaster


class Forecaster_D(FeatureSelection_D, Forecaster):
    """
    Abstract base class for DIRECT multi-horizon forecasting models.

    Attributes
    ----------
    Y : np.ndarray, shape (m, hh)
        The target matrix: Y[t, h - 1] = y[t + h - 1] (NaN beyond the end of the series).
    params : np.ndarray, shape (n + 1, hh)
        The parameters per horizon, intercept first.
    bb : np.ndarray, shape (n, hh)
        The slopes per horizon (ScalaTion's `bb`); `parameter` is its first column.
    """

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int, hasIntercept: bool = True) -> None:
        """
        Initializes the Forecaster_D (parameters as in `Forecaster`); the target matrix Y is built
        from y (subclasses made by `rescale` rebuild it from the scaled y).
        """
        super().__init__(args, y, hh, hasIntercept)
        self._Y: Optional[np.ndarray] = build_matrix_Y(self._y, hh)

    @property
    def Y(self) -> np.ndarray:
        """The multi-horizon target matrix (m, hh), NaN where the future is unknown."""
        return self._Y

    @property
    def bb(self) -> np.ndarray:
        """The slope matrix (n, hh): one column per horizon."""
        p = self.params
        return p[1:, :] if self._hasIntercept else p

    @staticmethod
    def fit_rows(Y_: np.ndarray, X_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The rows usable for training: those before the first NaN in the last horizon column
        (ScalaTion's `Forecaster_D.train` drops the trailing rows whose last horizon is undefined).

        Parameters
        ----------
        Y_ : np.ndarray, shape (m_, hh)
            The target matrix (window).
        X_ : np.ndarray, shape (m_, n)
            The input matrix (window).

        Returns
        -------
        (np.ndarray, np.ndarray)
            The trimmed (Y_, X_).
        """
        bad = np.where(np.isnan(Y_[:, -1]))[0]
        if len(bad) == 0:
            return Y_, X_
        idx = int(bad[0])
        return Y_[:idx], X_[:idx]

    @abstractmethod
    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the forecasting model, one regression per horizon (stores the parameters with
        `_set_params`, intercept first, one column per horizon).

        Parameters
        ----------
        Y_ : np.ndarray, shape (m_, hh), optional
            The target matrix (the full Y by default); padded rows are dropped by `fit_rows`.
        X_ : np.ndarray, shape (m_, n), optional
            The input matrix (the full X by default).
        """

    def _train_target(self, is_: int, t: int) -> np.ndarray:
        """The training target rows [is_, t) of the matrix Y."""
        return self._Y[is_:t]

    def diagnose_all(self, y_fcast: np.ndarray, TnT: bool = False) -> str:
        """
        The former per-horizon QoF table (kept for compatibility): the forecasts `y_fcast`
        (rows from `skip`, or the testing rows when TnT) against the targets, in original
        units, in ScalaTion's `showFitMap` form.
        """
        from Util.tools import fit_map
        start = self._tr_size if TnT else self.args["skip"]
        y_true = self._Yf[:, 0]
        ll, hh = len(y_true) - start, y_fcast.shape[1]
        y_true_arr = [y_true[start + h: start + ll] for h in range(hh)]
        yf = y_fcast if self._yForm is None else self._yForm.inverse_transform(y_fcast)
        y_fcast_arr = [yf[:ll - h, h] for h in range(hh)]
        return fit_map(y_true_arr, y_fcast_arr, self.nparams)
