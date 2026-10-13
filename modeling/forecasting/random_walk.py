"""
RandomWalk Model for Time Series Forecasting

Implements the Random Walk (RW) baseline: the next value of the series is forecast by its last
observed value,

    y_t+1 = y_t + e_t+1,    so   yf(t, h) = y_t-1   for every horizon h = 1 to hh

(the forecast of every horizon made at time t is the last value known at t, y_t-1).  The model
has no parameters to train: `train` is a no-op that records the empty parameter vector, as
ScalaTion's `NoTrainModel` trait does, and the forecasts are the observed values themselves, so
the non-negativity constraint never applies.  The Python counterpart of ScalaTion's
`RandomWalk.scala` (`RandomWalk (y, hh)` and the companion's `make`, with `Forecaster`'s default
`predict (t, y) = y(max0 (t-1))` and `forecastAt (h) = yf(?, 1)` folded in).

Differences from ScalaTion: row 0 of the forecast matrix holds y(0) (ScalaTion's `predict (0, y)`),
where ScalaTion's `predictAll` writes NO_DOUBLE, a row the default skip (1, ScalaTion's
`max (1, AR.hp("p"))`) drops from the In-Sample diagnosis anyway.  As in ScalaTion the model has
no hyper-parameters (`hparameter` is None, the REPORT prints "null"); of its args it reads skip
(default max (1, args.p), ScalaTion's `max (1, AR.hp("p"))`) and te_ratio.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from modeling.forecasting.forecaster import Forecaster


class RandomWalk(Forecaster):
    """
    Random Walk forecasting model: every horizon's forecast is the last observed value.

    Attributes
    ----------
    params : np.ndarray, shape (0,)
        Empty: there is nothing to train (ScalaTion's `b = VectorD (0)`).
    parameter : np.ndarray, shape (0,)
        Empty, as `params`.
    fname : list of str
        ["no-x features"] (ScalaTion's `Forecast.getFname` default for models without x).
    modelName : str
        "RandomWalk".
    hparameter : None
        The model has no hyper-parameters (ScalaTion's `hparam = null`).

    See Also
    --------
    Forecaster : the base forecasting logic (Yf, rollValidate, inSample_Test, tnT_Test).
    """

    ARGS = Forecaster.ARGS

    def __init__(self, y: np.ndarray, hh: int, args: Any = None) -> None:
        """
        Initializes the RandomWalk forecaster (the argument order of every model of the package:
        the data, the horizon, the settings).

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The response vector (time series data).
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        args : Args or dict, optional
            The settings: only skip (default max (1, p) = 1) and te_ratio are read; None => defaults.
        """
        args = self._argsOf(y, args)
        if args.skip is None:
            args.skip = max(1, args.p)                                    # ScalaTion: skip = max (1, AR.hp("p"))
        super().__init__(y, int(hh), args)
        self._modelName = "RandomWalk"
        self._params = np.zeros(0)                                      # nothing to train (NoTrainModel)

    @property
    def hparameter(self) -> None:
        """None: the model has no hyper-parameters (ScalaTion's `hparam = null`, printed "null")."""
        return None

    # -- names and parameters -------------------------------------------------------------

    @property
    def fname(self) -> List[str]:
        """["no-x features"], ScalaTion's `Forecast.getFname` for models without an input matrix."""
        return ["no-x features"]

    @property
    def params(self) -> np.ndarray:
        """The empty parameter vector (nothing is trained)."""
        return np.zeros(0)

    @property
    def parameter(self) -> np.ndarray:
        """The empty parameter vector b (ScalaTion's `b = VectorD (0)`)."""
        return np.zeros(0)

    @property
    def intercept(self) -> Any:
        """None: the model has no intercept to report."""
        return None

    def __repr__(self) -> str:
        return f"RandomWalk(), hh = {self.hh}, skip = {self._hparam['skip']}"

    # -- training and forecasting ---------------------------------------------------------

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Does nothing (ScalaTion's `NoTrainModel.train`): the model has no parameters; the empty
        parameter vector is recorded.

        Parameters
        ----------
        y_ : np.ndarray, optional
            Ignored.
        X_ : np.ndarray, optional
            Ignored.
        """
        self._params = np.zeros(0)

    def predict(self, t: int, y_: Optional[np.ndarray] = None) -> float:
        """
        The 1-step ahead prediction of y_t: the previous value y_t-1 (y_0 for t = 0), ScalaTion's
        `Forecaster.predict (t, y_)`.

        Parameters
        ----------
        t : int
            The time point being predicted.
        y_ : np.ndarray, optional
            The actual values to use (the model's y by default).
        """
        y_ = self._y if y_ is None else y_
        return float(y_[max(0, t - 1)])

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end): every horizon takes the
        last observed value y[t - 1] (ScalaTion's `forecast (t, y_)`), recorded in `Yf`.

        Parameters
        ----------
        t_start : int
            The first row (defaults to 0 when -1).
        t_end : int
            The row after the last (defaults to m when -1).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
        """
        t_st, t_en = self._adjust_range(t_start, t_end)
        prev = self._y[np.maximum(np.arange(t_st, t_en) - 1, 0)]       # y[t-1], y[0] at t = 0
        y_fcast = np.repeat(prev[:, None], self.hh, axis=1)
        self._store(t_st, t_en, y_fcast)
        return y_fcast

    # -- the companion's factory ----------------------------------------------------------

    @staticmethod
    def make(y0: int, d: int, n: int, stream: int = 0) -> np.ndarray:
        """
        Makes a random walk of n integer steps from y0, each step drawn uniformly from
        {-d, ..., d} (ScalaTion's `RandomWalk.make` with `Randi (-d, d, stm)`; numpy's
        `default_rng (stream)` replaces ScalaTion's random stream, so the walks differ).

        Parameters
        ----------
        y0 : int
            The previous value (e.g., the last value of a training set).
        d : int
            The maximum step size.
        n : int
            The number of steps to make.
        stream : int, default=0
            The random number stream (the seed of `default_rng`).

        Returns
        -------
        np.ndarray, shape (n,)
            The walk y0 + s_1, y0 + s_1 + s_2, ...
        """
        rng = np.random.default_rng(stream)
        steps = rng.integers(-d, d + 1, size=n)
        return (y0 + np.cumsum(steps)).astype(float)
