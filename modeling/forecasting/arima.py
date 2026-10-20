"""
ARIMA Model for Time Series Forecasting (statsmodels-backed)

Implements the Auto-Regressive Integrated Moving-Average model ARIMA(p, d, q) under the ScalaTion
`Forecaster` interface, with the estimation and the state-space machinery of statsmodels
(`statsmodels.tsa.arima.model.ARIMA`, exact maximum likelihood by the Kalman filter):

    (1 - phi_1 B - ... - phi_p B^p) (1 - B)^d y_t = c + (1 + theta_1 B + ... + theta_q B^q) e_t

Multi-horizon forecasts are made from the Kalman filter's predicted state: for every row t the
h = 1..hh steps ahead forecasts of y_t, y_t+1, ..., y_t+hh-1 use the information up to t - 1
(the filter is run once over the known rows, padded with hh - 1 missing values so that statsmodels
extends any trend term into the future, and the state is propagated with the transition matrix),
which equals `res.apply (y[:t]).forecast (hh)` for every t to machine precision -- in a fraction
of the time.  Rows t < max (p, q + 1) + d hold the model's own start-up forecasts (the diffuse
states forecast 0 until they are initialised), which the default skip drops from the In-Sample
diagnosis.  `rollValidate` of the base retrains every rc rows on the sliding window.

ScalaTion's own ARIMA family (`ARMA.scala`: conditional sum of squares / MLE by its own
optimizer; `ARIMA_diff.scala`: the differencing and back-forming helpers; `ARIMA.scalaa`,
`SARIMA.scala`) estimates the parameters differently, so the coefficients and QoF values of the
two differ while the interface (`train`, `forecast`, `inSample_Test`, `tnT_Test`, `rollValidate`,
`parameter`, `fname`, `modelName` = "ARIMA_p_d_q") is the same.  The hyper-parameter defaults
p = 1, d = 1, q = 1 are ScalaTion's `AR.hp`.  The ScalaTion `parameter` of an ARMA includes the
constant (delta +: phi); here too `parameter` returns every fitted coefficient but the innovation
variance sigma2, under the statsmodels names ("const", "ar.L1", "ma.L1", ...).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA

from modeling.forecasting.args import ARIMA_METHODS, hp as AR_hp
from modeling.forecasting.forecaster import Forecaster


class ARIMA(Forecaster):
    """
    ARIMA(p, d, q) forecasting model backed by statsmodels.

    Attributes
    ----------
    hp : HyperParameter (class attribute)
        The default hyper-parameters, ScalaTion's `AR.hp` (p = 1, d = 1, q = 1, ...).
    hparameter : HyperParameter
        `AR.hp` with the model's p, d, q (from its args), plus "trend" and "skip" (args.skip, default
        max (p, q + 1) + d).
    trend : str or sequence
        The statsmodels trend: "n" (none), "c" (constant), "t" (linear drift), "ct", or a
        polynomial indicator list; the default is "c" for d = 0 and "n" otherwise (statsmodels'
        rule: a trend term of order below d is not allowed).
    params, parameter : np.ndarray
        The fitted coefficients without sigma2 (trend terms first, then ar.L*, ma.L*).
    fname : list of str
        Their statsmodels names, e.g., ["const", "ar.L1", "ma.L1"].
    intercept : float or None
        The "const" coefficient when the trend has one.
    modelName : str
        "ARIMA_{p}_{d}_{q}".
    res : statsmodels ARIMAResults
        The last fit (None before `train`).

    See Also
    --------
    Forecaster : the base forecasting logic (Yf, rollValidate, inSample_Test, tnT_Test).
    AR : the Yule-Walker / OLS auto-regressive model of `ar.py`.
    """

    hp = AR_hp                                                         # ScalaTion's AR.hp
    ARGS = ("p", "d", "q", "method") + Forecaster.ARGS
    METHODS = ARIMA_METHODS                                            # METHOD None: statsmodels' default

    def __init__(self, y: np.ndarray, hh: int, args: Any = None,
                 trend: Union[str, Sequence[int], None] = None, **fit_kwargs: Any) -> None:
        """
        Initializes the ARIMA forecaster.

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The time series.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        args : Args or dict, optional
            The settings: p (AR order), d (differences), q (MA order), each defaulting to `ARIMA.hp`
            (1, 1, 1); skip (default max (p, q + 1) + d); nneg; te_ratio; method, the statsmodels
            estimation method ("statespace", "innovations_mle", "hannan_rissanen", "burg",
            "innovations", "yule_walker"; None => statsmodels' default, exact MLE via the Kalman filter).
        trend : str or sequence of int, optional
            The trend specification (statsmodels'); None => "c" when d = 0, "n" otherwise.
        **fit_kwargs
            Further keyword arguments of `ARIMAResults.fit` (e.g., maxiter).
        """
        args = self._argsOf(y, args)
        p, d, q = args.p, args.d, args.q
        self._trend = ("c" if d == 0 else "n") if trend is None else trend
        has_const = self._trend[0] == 1 if not isinstance(self._trend, str) else "c" in self._trend
        if args.skip is None:
            args.skip = max(p, q + 1) + d
        super().__init__(y, int(hh), args, hasIntercept=bool(has_const))
        self._hparam["trend"] = self._trend
        self._p, self._d, self._q = p, d, q
        self._fit_kwargs = dict(fit_kwargs)
        self._res = None
        self._modelName = f"ARIMA_{p}_{d}_{q}"
        names = list(self._model(self._y).param_names)                  # the names before training
        self._fname = [nm for nm in names if nm != "sigma2"]

    # -- the statsmodels model ----------------------------------------------------------------

    def _model(self, y_: np.ndarray) -> _SM_ARIMA:
        """The (unfitted) statsmodels ARIMA on the rows y_."""
        return _SM_ARIMA(np.asarray(y_, dtype=float), order=(self._p, self._d, self._q), trend=self._trend)

    @property
    def res(self):
        """The statsmodels results of the last `train` (None before training)."""
        return self._res

    @property
    def order(self):
        """(p, d, q)."""
        return self._p, self._d, self._q

    @property
    def trend(self):
        """The trend specification."""
        return self._trend

    def summary(self):
        """The statsmodels summary of the last fit."""
        if self._res is None:
            raise ValueError("The model has not been trained. Call train first.")
        return self._res.summary()

    # -- parameters ----------------------------------------------------------------------------

    @property
    def fname(self) -> List[str]:
        """The statsmodels names of the coefficients (sigma2 left out)."""
        return list(self._fname)

    @property
    def parameter(self) -> np.ndarray:
        """Every fitted coefficient but sigma2 (the constant included, as ScalaTion's delta +: phi)."""
        return self.params

    @property
    def intercept(self) -> Any:
        """The "const" coefficient when the trend has one, None otherwise."""
        if not self._hasIntercept or self._params is None:
            return None
        return float(self.params[self._fname.index("const")]) if "const" in self._fname else None

    @property
    def sigma2(self) -> float:
        """The innovation variance of the last fit."""
        if self._res is None:
            raise ValueError("The model has not been trained. Call train first.")
        return float(self._res.params[list(self._res.param_names).index("sigma2")])

    def mod_resetDF(self, size: int) -> None:
        """
        Resets the degrees of freedom (ScalaTion's `Forecaster.mod_resetDF`): the parameter vector
        includes the constant when the trend has one, so the slopes are one fewer.
        """
        k = len(self.parameter) - (1 if self._hasIntercept else 0)
        self.resetDF_k(size, max(0, k))

    def __repr__(self) -> str:
        return f"{self.modelName}(trend={self._trend!r}), hh = {self.hh}, skip = {self._hparam['skip']}"

    # -- training and forecasting --------------------------------------------------------------

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Fits the ARIMA(p, d, q) on the rows y_ (the full y by default) by statsmodels; the
        coefficients (without sigma2) are stored as the parameters.

        Parameters
        ----------
        y_ : np.ndarray, shape (m_,), optional
            The training rows.
        X_ : np.ndarray, optional
            Ignored (no exogenous variables).
        """
        y_ = self._y if y_ is None else np.asarray(y_, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kw = dict(self._fit_kwargs)
            if self._method is not None:
                kw["method"] = self._method
            self._res = self._model(y_).fit(**kw)
        names = list(self._res.param_names)
        keep = [i for i, nm in enumerate(names) if nm != "sigma2"]
        self._fname = [names[i] for i in keep]
        self._params = np.asarray(self._res.params, dtype=float)[keep]

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end) from the Kalman filter's
        predicted state at each row (the information up to t - 1) and records them in `Yf`:
        row t holds the forecasts of y[t], y[t + 1], ..., y[t + hh - 1].

        Parameters
        ----------
        t_start : int
            The first row (defaults to 0 when -1).
        t_end : int
            The row after the last (defaults to m when -1).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self._res is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        hh, n = self.hh, t_en - t_st
        padded = np.concatenate([self._y[:t_en], np.full(hh - 1, np.nan)])   # missing rows: the trend is extended
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rk = self._res.apply(padded, refit=False)                 # the filter over the known rows, same parameters
        ssm = rk.model.ssm
        T, Z = ssm["transition"], ssm["design"]
        c, d = ssm["state_intercept"], ssm["obs_intercept"]
        A = rk.filter_results.predicted_state[:, t_st:t_en].T.copy()  # a_{t|t-1}: (n, k_states)

        def at(M: np.ndarray, nd: int, s: np.ndarray) -> np.ndarray:
            """The system matrix at the times s (time-varying when it has a time axis)."""
            return np.moveaxis(M[..., s], -1, 0) if M.ndim == nd + 1 else np.broadcast_to(M, (n,) + M.shape)

        y_fcast = np.zeros((n, hh))
        for h in range(1, hh + 1):
            s = np.arange(t_st, t_en) + h - 1                         # the time forecast at horizon h
            y_fcast[:, h - 1] = np.einsum("nik,nk->ni", at(Z, 2, s), A)[:, 0] + at(d, 1, s)[:, 0]
            if h < hh:
                A = np.einsum("nik,nk->ni", at(T, 2, s), A) + at(c, 1, s)
        y_fcast = self._rectify(y_fcast)
        self._store(t_st, t_en, y_fcast)
        return y_fcast
