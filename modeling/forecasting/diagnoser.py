"""
Diagnoser Module: Forecast Diagnostics with Skipped Initial Rows

The Python counterpart of ScalaTion's `Diagnoser.scala`: a `Fit` whose `diagnose` drops the first
`skip` rows, since forecasting at early time points may not be feasible (the lag columns are
padded there).  For In-Sample testing skip defaults to p (the maximum lag); for Train-and-Test
`setSkip(0)` is called, as the training rows supply the past values.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from modeling.fit import Fit


class Diagnoser(Fit):
    """
    Quality of Fit measures for time series where the first `skip` predictions are left out.

    Parameters
    ----------
    dfr : float
        The degrees of freedom taken by the model.
    df : float
        The degrees of freedom left for the errors.
    hasIntercept : bool, default=True
        Whether the model has an intercept.
    skip : int, default=1
        The number of beginning elements to skip in `diagnose` (defaults to p for In-Sample tests).
    """

    def __init__(self, dfr: float = 1.0, df: float = 1.0, hasIntercept: bool = True, skip: int = 1) -> None:
        super().__init__(dfr, df, hasIntercept)
        self._skip = max(0, int(skip))

    def getSkip(self) -> int:
        """The current value of skip (the number of elements at the beginning to skip)."""
        return self._skip

    def setSkip(self, skip_: int = 0) -> None:
        """
        Sets the number of data points to skip at the beginning of a time series for diagnosis
        (In-Sample: usually p; Train-and-Test: 0, the default here).
        """
        self._skip = max(0, int(skip_))

    def diagnose(self, y: np.ndarray, yp: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The QoF measures of (y, yp) with the first `skip` elements dropped (`Fit.diagnose` otherwise).
        """
        s = self._skip
        if s > 0:
            return super().diagnose(y[s:], yp[s:], None if w is None else w[s:])
        return super().diagnose(y, yp, w)

    def ssef(self, y: np.ndarray, yp: np.ndarray) -> float:
        """The sum of squared errors, the first `skip` errors taken as zero."""
        s = self._skip
        return float(np.sum((y[s:] - yp[s:]) ** 2))
