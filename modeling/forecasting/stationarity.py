"""
Stationarity: the Framework of the Unit Root Tests for Time Series Stationarity

`UnitRoot` holds what the unit root tests share: the number of lags (the L4 "short" or L12 "long"
rule of Schwert 1989, or a given number), the regression trend ("c" constant, "ct" constant and
linear trend), the critical values of the test at 15 probabilities from response-surface
coefficients, and the p-value by linear interpolation between them:

    cv(p) = sum_j c0_j / n^j + sum_j c1_j (lags / n)^(j+1),     n = nobs - lags - 1

The Python counterpart of ScalaTion's `forecasting/Stationarity.scala` (trait `UnitRoot`, translated
there from the C++ of github.com/olmallet81/URT); `Stationarity_KPSS` (forecasting/stationarity_kpss.py)
is the test built on it.  Differences: the lag-length optimization of the ADF / DF-GLS tests (not
ported in ScalaTion either: `optim` is false) is left out; `setTrend` does not print its
"setTrend: trend = ..., npar = ..." line.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.6
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from Util.tools import banner

#: The probabilities at which the critical values are computed (ScalaTion's `UnitRoot.probas`).
PROBAS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999)

#: The number of parameters (excluding lagged differences) of each regression trend.
NPAR = {"nc": 1, "c": 2, "ct": 3, "ctt": 4}
TREND_TYPE = {"nc": "no constant", "c": "constant", "ct": "constant trend", "ctt": "quadratic trend"}


class UnitRoot:
    """
    The common framework of the unit root tests (ScalaTion's `UnitRoot` trait).

    Parameters
    ----------
    testName : str
        The name of the test, e.g., "KPSS".
    nobs : int
        The number of observations (length of the series).
    validTrends : sequence of str
        The trends the test accepts, e.g., ("c", "ct").
    lagsType : str
        "short" (L4 rule), "long" (L12 rule) or "" (use `lags` as given).
    lags : int
        The number of lags (replaced by the rule's value when `lagsType` is set).
    trend : str
        The regression trend ("c" by default; an unknown one falls back to "c").
    """

    COEFF: Dict[str, Dict[float, Dict[int, Sequence[float]]]] = {}      # the critical value coefficients (per test)

    def __init__(self, testName: str, nobs: int, validTrends: Sequence[str], lagsType: str, lags: int,
                 trend: str = "c") -> None:
        self.testName, self.nobs, self.validTrends = testName, int(nobs), tuple(validTrends)
        self.lagsType, self.lags, self.trend = lagsType, int(lags), trend
        self.pval = 1.0
        self.criticalVals = np.zeros(len(PROBAS))
        self.trendType: Optional[str] = None
        self._prevLags = 0                                               # the lags of the previous call
        self.npar = 0

    # -- settings (ScalaTion's setLagsType, setLags, setTrend) --------------------------------------

    def setLagsType(self) -> None:
        """Checks the default type of lags: an unknown one becomes "long" (with a warning)."""
        if self.lagsType and self.lagsType not in ("long", "short"):
            print("\n  WARNING: unknown default type of lags, long has been selected by default.\n")
            self.lagsType = "long"

    def setLags(self) -> None:
        """Sets the number of lags: the L4 (short) or L12 (long) rule when a lags type is given,
        unless a number of lags of one's own was chosen (it then replaces the lags type)."""
        if self.lags < 0:
            self.lags = 0
            print("\n  WARNING: number of lags cannot be negative, it has been set to 0 by default.\n")
        if self.lagsType and self.lags != self._prevLags:
            self.lagsType = ""                                           # a value of one's own (ScalaTion's rule)
        if self.lagsType:
            factor = 4 if self.lagsType == "short" else 12
            self.lags = int(factor * (0.01 * self.nobs) ** 0.25)          # Schwert (1989)
        self._prevLags = self.lags

    def setTrend(self) -> None:
        """Sets the regression trend: an unknown one becomes "c" (with a warning)."""
        if self.trend not in self.validTrends:
            print("\n  WARNING: unknown regression trend selected, 'constant term' has been selected by default.\n")
            print(f"  Possible trends for this test are {self.validTrends}")
            self.trend = "c"
        self.trendType, self.npar = TREND_TYPE[self.trend], NPAR[self.trend]

    # -- critical values and p-value -----------------------------------------------------------------

    def computeCV(self, coeff: Dict[float, Dict[int, Sequence[float]]]) -> np.ndarray:
        """The critical values at the 15 probabilities from the response-surface coefficients."""
        n = self.nobs - self.lags - 1                                    # the adjusted number of observations
        for i, p in enumerate(PROBAS):
            c0, c1 = coeff[p][0], coeff[p][1]
            self.criticalVals[i] = (sum(c / n ** j for j, c in enumerate(c0)) +
                                    sum(c * (self.lags / n) ** (j + 1) for j, c in enumerate(c1)))
        return self.criticalVals

    def computePval(self, stat: float) -> float:
        """The probability of `stat` by linear interpolation between the critical values (clamped
        to [0.001, 0.999] outside them)."""
        cv = self.criticalVals
        if stat <= cv[0]:
            return PROBAS[0]
        if stat > cv[-1]:
            return PROBAS[-1]
        for i in range(1, len(PROBAS)):
            if stat <= cv[i]:
                return PROBAS[i - 1] + (stat - cv[i - 1]) * (PROBAS[i] - PROBAS[i - 1]) / (cv[i] - cv[i - 1])
        return PROBAS[-1]

    def pvalue(self, stat: float) -> float:
        """The probability of the statistic (NaN for a NaN statistic)."""
        if math.isnan(stat):
            self.pval = math.nan
        else:
            self.computeCV(self.COEFF[self.trend])
            self.pval = self.computePval(stat)
        return self.pval

    def show(self, stat: float) -> None:
        """Prints the test results (ScalaTion's `show`): statistic, p-value, lags, trend, hypotheses,
        the 1 %, 5 % and 10 % critical values and the conclusion."""
        banner(f"{self.testName} Test Results")
        print(f"  Statistic = {stat}")
        p = self.pval
        print("  P-value = " + ("< 0.001" if p <= PROBAS[0] else "> 0.999" if p >= PROBAS[-1] else f"{p}"))
        print(f"  Lags  = {self.lags}")
        print(f"  Trend = {self.trendType}")
        print("  ------------------------------------\n")
        print("  Test Hypothesis")
        print("  ------------------------------------")
        print("  H0: The process is weakly stationary")
        print("  H1: The process contains a unit root\n")
        print("  Critical Values")
        print("  ---------------")
        for label, i in (("   1%", 12), ("   5%", 10), ("  10%", 9)):
            print(f"{label} {math.nan if math.isnan(stat) else self.criticalVals[i]}")
        print("  ---------------")
        print("  Test Conclusion")
        if p <= 0.01:
            print("We can reject H0 at the 1% significance level")
        elif p <= 0.05:
            print("We can reject H0 at the 5% significance level")
        elif p <= 0.10:
            print("We can reject H0 at the 10% significance level")
        elif not math.isnan(p):
            print("We cannot reject H0")
        else:
            print("We cannot conclude, NaN produced")
