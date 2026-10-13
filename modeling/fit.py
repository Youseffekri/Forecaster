"""
Fit Module: Quality of Fit (QoF) Measures

The Python counterpart of ScalaTion's `FitM.scala` / `Fit.scala`: the `QoF` enumeration (the
26 measures in ScalaTion's order), the `Fit` mixin that computes them from the actual and the
predicted responses given the degrees of freedom, and the ScalaTion printing conventions
(`fitMap` -> "LinkedHashMap(rSq -> 0.910754, ...)", `showFitMap` -> one "label -> VectorD(...)"
line per measure, `help`), and ScalaTion's coefficient table `Fit.summary` (estimates, standard
errors, t values, p-values and VIFs -- `vif` is ScalaTion's `Predictor.vif`).

    rSq, rSqBar, sst, sse, sde, rse, mse0, rmse, mae, smape, bmae,
    m, dfr, df, fStat, aic, bic, mape, mase, smapeC,
    picp, pinc, ace, pinaw, mis, wis            (the last six are -1 unless intervals are diagnosed)

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from enum import Enum
from math import log, pi, sqrt
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import f as fisher_dist, t as student_dist

from Util.tools import array_str, fmt, jdouble, vector_str


class QoF(Enum):
    """
    The Quality of Fit (QoF) measures/metrics, in ScalaTion's order (the index into a QoF vector).
    """
    rSq    = 0      # 0-3 related to R^2
    rSqBar = 1
    sst    = 2
    sse    = 3
    sde    = 4      # 4-10 various error metrics
    rse    = 5
    mse0   = 6
    rmse   = 7
    mae    = 8
    smape  = 9
    bmae   = 10
    m      = 11     # 11-16 degrees of freedom and information criteria
    dfr    = 12
    df     = 13
    fStat  = 14
    aic    = 15
    bic    = 16
    mape   = 17     # 17-19 time series metrics (also 8-9)
    mase   = 18
    smapeC = 19
    picp   = 20     # 20-25 for prediction intervals
    pinc   = 21
    ace    = 22
    pinaw  = 23
    mis    = 24
    wis    = 25

    @property
    def ordinal(self) -> int:
        """The index of the measure in a QoF vector (ScalaTion's `ordinal`)."""
        return self.value


N_QoF = len(QoF)                                                 # the number of QoF measures
qoF_names: List[str] = [q.name for q in QoF]                     # the QoF names from the QoF enum
LOG_2PI = log(2.0 * pi)
MAXI = {QoF.rSq.ordinal, QoF.rSqBar.ordinal}                     # maximize these QoF metrics (minimize the rest)

HELP = """
help: Quality of Fit (QoF) metrics/measures:
    rSq    =  R-squared, the Coefficient of Determination (R^2)
    rSqBar =  adjusted R-squared (R^2-bar)
    sst    =  Sum of Squares Total (SST) [ssr + sse]
    sse    =  Sum of Squares for Error (SSE = RSS)

    sde    =  Standard Deviation of Errors (SDE)
    rse    =  Residual Standard Error (RSE) [sqrt (sse / df)]
    mse0   =  raw Mean Square Error (MSE = SSE / m)
    rmse   =  Root Mean Square Error (RMSE)
    mae    =  Mean Absolute Error (MAE)
    smape  =  symmetric Mean Absolute Percentage Error (sMAPE)
    bmae   =  Blended-denominator Mean Absolute Error (BMAE) [WAPE <-> NMAE]

    m      =  Number of Observations
    dfr    =  Degrees of Freedom (DFr) taken by the regression/model, e.g., one lost per parameter estimated
    df     =  Degrees of Freedom (DF) left for residuals/errors
    fStat  =  Fisher's Statistic
    aic    =  Akaike Information Criterion (AIC)
    bic    =  Bayesian Information Criterion (BIC)

    mape   =  Mean Absolute Percentage Error (MAPE)
    mase   =  Mean Absolute Scaled Error (MASE)
    smapeC =  symmetric Mean Absolute Percentage Error information Criterion (sMAPE-IC)

    picp   =  Prediction Interval empirical Coverage Probability (PICP)
    pinc   =  Prediction Interval Nominal Coverage probability (PINC) [1 - alpha/2]
    ace    =  Average Coverage Error (ACE) [empirical - nominal coverage, i.e., picp - pinc]
    pinaw  =  Prediction Interval Normalized Average Width (PINAW)
    mis    =  Mean Interval Score (MIS) [ over given instances ]
    wis    =  Weighted Interval Score (WIS) [ over several alpha values ]
"""


# ---------------------------------------------------------------------------------------------
# Stand-alone measures (ScalaTion's `FitM`/`Fit` companion objects)
# ---------------------------------------------------------------------------------------------

def smapeF(y: np.ndarray, yp: np.ndarray, e: Optional[np.ndarray] = None) -> float:
    """
    The symmetric Mean Absolute Percentage Error (sMAPE, in percent) -- ScalaTion's `FitM.smapeF`:
    the terms with a zero error are skipped (y_i = yp_i = 0 => no error => no percentage error).

        smape = 200 / m * sum_{e_i != 0} |e_i| / (|y_i| + |yp_i|)
    """
    e = y - yp if e is None else e
    nz = e != 0.0
    if not np.any(nz):
        return 0.0
    s = np.sum(np.abs(e[nz]) / (np.abs(y[nz]) + np.abs(yp[nz])))
    return float(200.0 * s / len(e))


def mae(y: np.ndarray, yp: np.ndarray, h: int = 1) -> float:
    """The Mean Absolute Error of an h-steps ahead forecast (ScalaTion's `Fit.mae`): yp(t-h) forecasts y(t)."""
    s = float(np.sum(np.abs(y[h:] - yp[:len(y) - h]))) if h > 0 else float(np.sum(np.abs(y - yp)))
    return s / len(yp)


def mae_n(y: np.ndarray, h: int = 1) -> float:
    """The Mean Absolute Error of the Naive model (last value) at horizon h (ScalaTion's `Fit.mae_n`)."""
    return float(np.sum(np.abs(y[h:] - y[:-h]))) / (len(y) - h)


def mase(y: np.ndarray, yp: np.ndarray, h: int = 1) -> float:
    """The Mean Absolute Scaled Error: MAE of the model over MAE of the one-step Naive model."""
    den = mae_n(y, 1)
    return mae(y, yp, h) / den if den != 0.0 else float("inf")


def bmae(y: np.ndarray, yp: np.ndarray, h: int = 1, omega: float = 0.5) -> float:
    """
    The Blended-denominator Mean Absolute Error (percent): 100 mae / ((1 - omega) mu + omega sigma),
    a convex combination of WAPE (omega = 0) and NMAE (omega = 1); 0 when the denominator is 0.
    """
    if not 0.0 <= omega <= 1.0:
        raise ValueError("BMAE omega blending parameter must be between 0.0 and 1.0")
    mu, sigma = float(np.mean(y)), float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    denom = (1.0 - omega) * mu + omega * sigma
    return 0.0 if denom == 0.0 else 100.0 * mae(y, yp, h) / denom


def mape(y: np.ndarray, yp: np.ndarray) -> float:
    """The Mean Absolute Percentage Error (percent): 100 mean(|e| / |y|) (inf when some y is 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(100.0 * np.sum(np.abs(y - yp) / np.abs(y)) / len(y))


def extreme(qk: int) -> float:
    """A contrary/starting value: -inf for a maximized measure (rSq, rSqBar), +inf for a minimized one."""
    return float("-inf") if qk in MAXI else float("inf")


def better(qk: int, a: float, b: float) -> bool:
    """Whether value a of measure qk is better than value b."""
    return a > b if qk in MAXI else a < b


# ---------------------------------------------------------------------------------------------
# Printing (ScalaTion's `FitM.fitMap` / `FitM.showFitMap`)
# ---------------------------------------------------------------------------------------------

def fitMap(ftVec: Sequence[float], ftLab: Sequence[str] = qoF_names) -> str:
    """
    The QoF vector as ScalaTion prints it in a REPORT: "LinkedHashMap(rSq -> 0.910754, rSqBar -> ...)"
    (each value with six decimals, `fmt`).
    """
    items = ", ".join(f"{lab} -> {fmt(float(v))}" for lab, v in zip(ftLab, ftVec))
    return f"LinkedHashMap({items})"


def fitMapDict(ftVec: Sequence[float], ftLab: Sequence[str] = qoF_names) -> Dict[str, float]:
    """The QoF vector as an ordered dictionary label -> value."""
    return {lab: float(v) for lab, v in zip(ftLab, ftVec)}


def showFitMap(ftMat: np.ndarray, ftLab: Sequence[str] = qoF_names) -> str:
    """
    The QoF table as ScalaTion's `diagnoseAll` prints it: one line per measure,
    "\\t\\t{label} \\t -> VectorD(v1,\\tv2, ...) " for the horizons (columns of `ftMat`).

    Parameters
    ----------
    ftMat : np.ndarray, shape (N_QoF, hh)
        The QoF values, one row per measure and one column per horizon (ScalaTion's `ftMat.transpose`).
    """
    lines = ["\n"]
    for i, lab in enumerate(ftLab):
        lines.append(f"\t\t{lab} \t -> {vector_str(ftMat[i])} \n")
    return "".join(lines)


# ---------------------------------------------------------------------------------------------
# The `Fit` mixin
# ---------------------------------------------------------------------------------------------

class Fit:
    """
    Quality of Fit (QoF) measures for a model with `dfr` degrees of freedom taken by the model
    and `df` left for the errors (ScalaTion's `Fit` trait, built on `FitM`).  `diagnose` fills
    and returns the 26-element QoF vector in `QoF` order.

    Parameters
    ----------
    dfr : float
        The degrees of freedom taken by the regression/model (the number of slope parameters).
    df : float
        The degrees of freedom left for the errors.
    hasIntercept : bool, default=True
        Whether the model has an intercept (counted in kParam for AIC/BIC and in r_df).

    Attributes
    ----------
    qof : np.ndarray, shape (N_QoF,)
        The last computed QoF vector (-1 before the first `diagnose`).

    Notes
    -----
        sse = |e|^2, sst = |y - mean(y)|^2, ssr = sst - sse, sde = stdev(e) (unbiased),
        rSq = 1 - sse / sst, mse0 = sse / m, rmse = sqrt(mse0), mae = |e|_1 / m,
        msr = ssr / dfr (0 if dfr = 0), mse = sse / df, rse = sqrt(mse),
        rSqBar = 1 - (1 - rSq) r_df with r_df = (dfr + df) / df if df > 1 else dfr (+1 with intercept),
        fStat = msr / mse, kParam = dfr + (2 if intercept else 1),
        ll = m (log(mse) + mse0 / mse + log 2 pi), aic = ll + 2 kParam, bic = aic + kParam (ln m - 2),
        smapeC = smape + 2 kParam / m.
    """

    PIC = 2.0                                                    # penalty multiplier for sMAPE IC

    def __init__(self, dfr: float = 1.0, df: float = 1.0, hasIntercept: bool = True) -> None:
        self._dfr = float(dfr)
        self._df = float(df)
        self._hasIntercept = hasIntercept
        self._r_df = self._ratio_df()
        self.qof: np.ndarray = -np.ones(N_QoF)
        self._sig2e = -1.0                                       # variance of the errors (set once, as in ScalaTion)
        self._m = 0
        self._mse = self._rse = self._rSq = self._rSqBar = self._fStat = float("nan")   # for `summary`
        self._p_fS = -1.0                                        # p-value of the F statistic

    # -- degrees of freedom ---------------------------------------------------------------

    def _ratio_df(self) -> float:
        dfr, df = self._dfr, self._df
        return (dfr + df) / df if df > 1.0 else dfr + (1.0 if self._hasIntercept else 0.0)

    @property
    def dfr(self) -> float:
        """The degrees of freedom taken by the model."""
        return self._dfr

    @property
    def df(self) -> float:
        """The degrees of freedom left for the errors."""
        return self._df

    @property
    def hasIntercept(self) -> bool:
        """Whether the model has an intercept."""
        return self._hasIntercept

    def resetDF(self, dfr: float, df: float) -> None:
        """
        Resets the degrees of freedom (ScalaTion's `Fit.resetDF`).

        Parameters
        ----------
        dfr : float
            The degrees of freedom taken by the model.
        df : float
            The degrees of freedom left for the errors.
        """
        self._dfr, self._df = float(dfr), float(df)
        self._r_df = self._ratio_df()

    def resetDF_k(self, size: int, k: int) -> None:
        """
        Resets the degrees of freedom from the number of slope parameters k (ScalaTion's
        `Forecaster.resetDF_k`): dfr = k (never counting the intercept) and df = size - k - 1 with an
        intercept, size - k otherwise, so that dfr + df is m - 1 with an intercept and m without.
        An error DF <= 0 is reported and clamped to 1 (the DF-based measures are then unreliable).

        Parameters
        ----------
        size : int
            The number of rows diagnosed (after any skipped initial rows).
        k : int
            The number of slope parameters (excluding any intercept).
        """
        df0 = size - k - (1 if self._hasIntercept else 0)
        if df0 <= 0:
            print(f"resetDF_k: error DF = {df0} <= 0 (size = {size}, k = {k}): clamped to 1, DF-based QoF unreliable")
        self.resetDF(float(k), float(max(1, df0)))

    # -- diagnosis ------------------------------------------------------------------------

    def diagnose(self, y: np.ndarray, yp: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Diagnoses the health of the model by computing the QoF measures from the actual and the
        predicted responses (ScalaTion's `Fit.diagnose`).  The prediction-interval measures
        (picp .. wis) are left at -1.

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The actual response/output vector (test/full).
        yp : np.ndarray, shape (m,)
            The predicted response/output vector (test/full).
        w : np.ndarray, optional
            The weights on the instances (defaults to None).

        Returns
        -------
        np.ndarray, shape (N_QoF,)
            The QoF vector in `QoF` order.
        """
        y, yp = np.asarray(y, dtype=float), np.asarray(yp, dtype=float)
        m = len(y)
        if m < 2:
            print(f"diagnose: requires at least 2 responses to evaluate m = {m}")
        if len(yp) != m:
            print(f"diagnose: yp.dim = {len(yp)} != y.dim = {m}")
        self._m = m
        q = -np.ones(N_QoF)

        e = y - yp                                               # residual/error vector
        sse = float(np.dot(e, e))                                # Sum of Squares for Error
        if w is None:
            sst = float(np.sum((y - y.mean()) ** 2))             # Sum of Squares Total (ssr + sse)
            ssr = sst - sse                                      # Sum of Squares Regression
        else:
            ssr = float(np.sum(w * (yp - np.sum(w * yp / w.sum())) ** 2))
            sst = ssr + sse
        sde = float(np.std(e, ddof=1)) if m > 1 else 0.0         # Standard Deviation of Error (unbiased)

        with np.errstate(divide="ignore", invalid="ignore"):
            rSq = 1.0 - sse / sst if sst != 0.0 else float("nan")
        mse0 = sse / m                                           # raw/MLE Mean Squared Error
        rmse = sqrt(mse0)                                        # Root Mean Squared Error
        mae_ = float(np.sum(np.abs(e))) / m                      # Mean Absolute Error
        smape_ = smapeF(y, yp, e)                                # symmetric Mean Absolute Percentage Error

        dfr, df = self._dfr, self._df
        if dfr < 0 or df < 0:
            print(f"diagnose: degrees of freedom dfr = {dfr} and df = {df} must be non-negative")
        msr = 0.0 if dfr == 0 else ssr / dfr                     # Mean Squared Regression
        mse = sse / df                                           # Mean Squared Error (unbiased)
        rse = sqrt(mse)                                          # Residual Standard Error
        rSqBar = 1.0 - (1.0 - rSq) * self._r_df                  # adjusted R-squared
        fStat = msr / mse if mse != 0.0 else float("inf")        # F statistic
        if dfr == 0:
            p_fS = -0.0
        else:
            p_fS = float(1.0 - fisher_dist.cdf(fStat, int(dfr), int(df)))   # p-value of the F statistic
            if np.isnan(p_fS):
                p_fS = -0.0
        self._mse, self._rse, self._rSq, self._rSqBar, self._fStat, self._p_fS = mse, rse, rSq, rSqBar, fStat, p_fS
        if self._sig2e == -1.0:
            self._sig2e = float(np.var(e))

        kParam = dfr + (2.0 if self._hasIntercept else 1.0)     # slopes + intercept + residual variance
        ln_m = log(m)
        ll = m * (log(mse) + mse0 / mse + LOG_2PI) if mse > 0.0 else float("-inf")
        aic = ll + 2.0 * kParam                                  # Akaike Information Criterion
        bic = aic + kParam * (ln_m - 2.0)                        # Bayesian Information Criterion
        bmae_ = bmae(y, yp)                                      # Blended-denominator MAE
        mape_ = mape(y, yp)                                      # Mean Absolute Percentage Error
        mase_ = mase(y, yp)                                      # Mean Absolute Scaled Error
        smapeC = smape_ + self.PIC * kParam / m                  # sMAPE Information Criterion

        q[:20] = [rSq, rSqBar, sst, sse, sde, rse, mse0, rmse, mae_, smape_, bmae_,
                  m, dfr, df, fStat, aic, bic, mape_, mase_, smapeC]
        self.qof = q
        return q

    @property
    def fit(self) -> np.ndarray:
        """The last computed QoF vector."""
        return self.qof

    def ll(self, mse0_: Optional[float] = None, mse_: Optional[float] = None, m_: Optional[int] = None) -> float:
        """-2 log-likelihood: m (ln(mse) + mse0 / mse + ln 2 pi) (from the last diagnosis unless given)."""
        mse0_ = self.qof[QoF.mse0.ordinal] if mse0_ is None else mse0_
        mse_ = self.qof[QoF.sse.ordinal] / self._df if mse_ is None else mse_
        m_ = self._m if m_ is None else m_
        return m_ * (log(mse_) + mse0_ / mse_ + LOG_2PI)

    # -- printing -------------------------------------------------------------------------

    def fitMap(self, ftVec: Optional[Sequence[float]] = None) -> str:
        """The QoF vector (the last one by default) in ScalaTion's "LinkedHashMap(...)" form."""
        return fitMap(self.qof if ftVec is None else ftVec)

    @staticmethod
    def showFitMap(ftMat: np.ndarray, ftLab: Sequence[str] = qoF_names) -> str:
        """The QoF table (measures x horizons) as `diagnoseAll` prints it."""
        return showFitMap(ftMat, ftLab)

    def showQoF(self, qof: Optional[np.ndarray] = None) -> None:
        """Prints the QoF vector in `fitMap` form."""
        print(self.fitMap(qof))

    @staticmethod
    def help() -> str:
        """The help string that describes the QoF measures."""
        return HELP

    # -- the coefficient table (ScalaTion's `Fit.summary`) ----------------------------------

    def summary(self, x_: np.ndarray, fname: Optional[Sequence[str]] = None, b: Optional[np.ndarray] = None,
                vifs: Optional[np.ndarray] = None) -> str:
        """
        The coefficient table of a fit (ScalaTion's `Fit.summary`): the standard errors come from the
        diagonal of (x_' x_)^-1 (a QR factorization of x_) times the unbiased MSE of the LAST
        `diagnose` call, so the table reflects that diagnosis (the last horizon after `diagnoseAll`).

        Parameters
        ----------
        x_ : np.ndarray, shape (m, n)
            The data/input matrix the coefficients belong to (column 0 the intercept, when there is one).
        fname : sequence of str, optional
            The feature names (printed first as "fname = Array(...)").
        b : np.ndarray, shape (n,)
            The coefficients.
        vifs : np.ndarray, shape (n - 1,), optional
            The Variance Inflation Factors of columns 1 .. n-1 (column 0 prints "NA").
        """
        x_ = np.asarray(x_, dtype=float)
        try:
            _, R = np.linalg.qr(x_)
            Ri = np.linalg.inv(R)
            diag = np.sum(Ri * Ri, axis=1)                            # the diagonal of (x' x)^-1 = R^-1 R^-T
        except np.linalg.LinAlgError:
            diag = np.full(x_.shape[1], np.nan)
        with np.errstate(invalid="ignore"):
            stdErr = np.sqrt(diag * self._mse)
        return self.showSummary(stdErr, fname, b, vifs)

    def showSummary(self, stdErr: Optional[np.ndarray], fname: Optional[Sequence[str]] = None,
                    b: Optional[np.ndarray] = None, vifs: Optional[np.ndarray] = None) -> str:
        """The SUMMARY block of ScalaTion's `Fit.showSummary`."""
        rule = "-" * 82
        return ((f"fname = {array_str(fname)}" if fname is not None else "") +
                f"\nSUMMARY\n    Parameters/Coefficients:\n    Var      Estimate    Std. Error \t t value \t Pr(>|t|) \t VIF\n"
                f"{rule}\n{self.sumCoeff(np.asarray(b, dtype=float), stdErr, vifs)}\n"
                f"    Residual standard error: {fmt(self._rse)} on {jdouble(self._df)} degrees of freedom\n"
                f"    Multiple R-squared:  {fmt(self._rSq)},\tAdjusted R-squared:  {fmt(self._rSqBar)}\n"
                f"    F-statistic: {jdouble(self._fStat)} on {jdouble(self._dfr)} and {jdouble(self._df)} DF,  "
                f"p-value: {jdouble(self._p_fS)}\n{rule}\n        ")

    def sumCoeff(self, b: np.ndarray, stdErr: Optional[np.ndarray], vf: Optional[np.ndarray]) -> str:
        """The coefficient rows of the summary (ScalaTion's `Fit.sumCoeff`): x_j, estimate, standard
        error, t value, p-value (2 P(T < -|t|), 0 when undefined) and VIF ("NA" for column 0)."""
        out = []
        t = p = None
        if stdErr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = b / stdErr
            p = 2.0 * student_dist.cdf(-np.abs(t), self._df) if self._df > 0 else -np.ones(len(b))
        for j in range(len(b)):
            if stdErr is None:
                out.append(f"    x{j}\t {fmt(b[j])}?")
                continue
            p_j = 0.0 if np.isnan(p[j]) else p[j]
            v = "NA" if j == 0 or vf is None else fmt(vf[j - 1])
            out.append(f"    x{j}\t {fmt(b[j])}\t {fmt(stdErr[j])}\t {fmt(t[j])}\t {fmt(p_j)}\t {v}\n")
        return "".join(out)


def vif(x: np.ndarray, skip: int = 1) -> np.ndarray:
    """
    The Variance Inflation Factors of the columns skip .. n-1 of x (ScalaTion's `Predictor.vif`):
    VIF_j = 1 / (1 - R^2_j), R^2_j that of regressing column j on all the other columns (x keeps its
    column of ones, so the regression has an intercept).  One column besides the skipped => [1].
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    out = np.zeros(n - skip)
    if len(out) == 1:
        out[0] = 1.0
        return out
    for j in range(skip, n):
        x_j = x[:, j]
        x_noj = np.delete(x, j, axis=1)
        b = np.linalg.lstsq(x_noj, x_j, rcond=None)[0]
        e = x_j - x_noj @ b
        sst = float(np.sum((x_j - x_j.mean()) ** 2))
        rSq_j = 1.0 - float(e @ e) / sst if sst > 0.0 else float("nan")
        with np.errstate(divide="ignore"):
            out[j - skip] = 1.0 / (1.0 - rSq_j)
    return out


def diagnose(y: np.ndarray, yp: np.ndarray, dfr: float = 1.0, df: Optional[float] = None,
             hasIntercept: bool = True) -> np.ndarray:
    """
    Computes the QoF vector of a prediction without a model object: a stand-alone `Fit.diagnose`.

    Parameters
    ----------
    y, yp : np.ndarray, shape (m,)
        The actual and the predicted responses.
    dfr : float, default=1
        The degrees of freedom taken by the model (number of slopes).
    df : float, optional
        The degrees of freedom left for the errors (defaults to m - dfr - 1 with an intercept,
        clamped to 1 as in `resetDF_k`).
    hasIntercept : bool, default=True
        Whether the model has an intercept.

    Returns
    -------
    np.ndarray, shape (N_QoF,)
        The QoF vector in `QoF` order (index by `QoF.smape.ordinal`, ...).
    """
    f = Fit(dfr, 1.0, hasIntercept)
    if df is None:
        f.resetDF_k(len(y), int(round(dfr)))
    else:
        f.resetDF(dfr, df)
    return f.diagnose(y, yp)
