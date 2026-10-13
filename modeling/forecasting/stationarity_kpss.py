"""
Stationarity_KPSS: the KPSS Test for Time Series Stationarity

The Kwiatkowski-Phillips-Schmidt-Shin test of the null hypothesis that a series is (weakly) stationary
around a constant ("c") or a linear trend ("ct"), against a unit root.  The series is demeaned (or
detrended by OLS), its partial sums s_t give eta = sum s_t^2 / n^2, the long-run variance s2 is the
variance plus Bartlett-weighted autocovariances up to `lags`, and the statistic is eta / s2; its
p-value is one minus the probability of the statistic under the critical values of `UnitRoot`
(response surfaces at 15 probabilities, linear interpolation between them, so p lies in
[0.001, 0.999]).  Small p => reject stationarity.

    test = Stationarity_KPSS(y, 0, "short", "c")       # lags by the L4 rule 4 (n / 100)^(1/4)
    stat, p = test.statistic(), test.pvalue()

The Python counterpart of ScalaTion's `forecasting/Stationarity_KPSS.scala` (Hao Peng, John Miller;
translated there from github.com/olmallet81/URT), with its coefficient table.  `EDA4TS` uses it for
the profile's KPSS p-value.  (statsmodels' `kpss` gives the same statistic but interpolates its
p-value between four critical values only, clipped to [0.01, 0.10].)

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.6
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from modeling.forecasting.stationarity import UnitRoot

#: The critical value coefficients of the KPSS test (ScalaTion's `Stationarity_KPSS.coeffKpss`): per
#: trend and probability, 0 -> the coefficients of 1 / n^j, 1 -> those of (lags / n)^(j+1).
COEFF_KPSS: Dict[str, Dict[float, Dict[int, Tuple[float, ...]]]] = {
    "c": {
        0.001 : {0: (0.0167565, 0.1628078, -0.01441111),
                 1: (0.2036426, 0.1065976, -0.5134231)},
        0.005 : {0: (0.02160067, 0.1649124, -0.005748196),
                 1: (0.1939746, 0.180736, -0.6535936)},
        0.01  : {0: (0.02466286, 0.1635288, 0.04500264),
                 1: (0.1893402, 0.2089429, -0.6943796)},
        0.025 : {0: (0.03026605, 0.1623029, 0.09417188),
                 1: (0.1847549, 0.2238625, -0.6818411)},
        0.05  : {0: (0.03650665, 0.1643194, 0.118059),
                 1: (0.1819509, 0.2103519, -0.5991279)},
        0.1   : {0: (0.04597858, 0.1695003, 0.1046518),
                 1: (0.1809212, 0.1596069, -0.4150671)},
        0.2   : {0: (0.06222337, 0.1785042, 0.06728867),
                 1: (0.1794514, 0.1031811, -0.2438007)},
        0.5   : {0: (0.1188952, 0.2136505, -0.1988974),
                 1: (0.1738762, -0.04517147, 0.1061083)},
        0.8   : {0: (0.2413558, 0.2427114, -0.4925517),
                 1: (0.07161341, 0.3635483, -0.8456676)},
        0.9   : {0: (0.347477, 0.1815891, -0.8897294),
                 1: (-0.1026364, 0.4255538, -0.6916551)},
        0.95  : {0: (0.461564, -0.03241585, -0.4093091),
                 1: (-0.4579306, 1.037593, -0.9517771)},
        0.975 : {0: (0.5808698, -0.4198131, 1.286591),
                 1: (-1.021283, 2.743161, -2.926945)},
        0.99  : {0: (0.7434379, -1.282285, 7.296387),
                 1: (-2.025517, 6.584114, -8.361188)},
        0.995 : {0: (0.8686703, -2.108955, 13.70689),
                 1: (-3.003148, 11.11568, -15.74966)},
        0.999 : {0: (1.162377, -4.578843, 34.27324),
                 1: (-5.808165, 25.97152, -41.79132)},
    },
    "ct": {
        0.001 : {0: (0.01234275, 0.1655343, 0.01672944, -1.675843),
                 1: (0.2106948, 0.03970888, -0.3368414)},
        0.005 : {0: (0.01526251, 0.1588324, 0.2212439, -3.176791),
                 1: (0.2015772, 0.1179992, -0.5024903)},
        0.01  : {0: (0.01699618, 0.1544975, 0.3604586, -4.262545),
                 1: (0.1969925, 0.1538411, -0.5702144)},
        0.025 : {0: (0.02005522, 0.1485785, 0.5907839, -6.211238),
                 1: (0.1880815, 0.2169062, -0.6777426)},
        0.05  : {0: (0.02326679, 0.1414555, 0.8224264, -7.893967),
                 1: (0.1804144, 0.2639471, -0.7486399)},
        0.1   : {0: (0.02781531, 0.1377224, 0.917243, -8.218599),
                 1: (0.170477, 0.3089857, -0.7857759)},
        0.2   : {0: (0.03489574, 0.1345627, 0.9657117, -7.923499),
                 1: (0.1579158, 0.3369968, -0.7515043)},
        0.5   : {0: (0.0556109, 0.1397416, 0.4620311, -1.559948),
                 1: (0.1300005, 0.2833583, -0.4283284)},
        0.8   : {0: (0.09159799, 0.1345566, -0.619965, 9.817577),
                 1: (0.05585951, 0.2773908, -0.1216728)},
        0.9   : {0: (0.1193112, 0.09514355, -1.079983, 16.87997),
                 1: (-0.03271029, 0.4379497, -0.2279858)},
        0.95  : {0: (0.1478756, 0.03863841, -1.89105, 30.13683),
                 1: (-0.149998, 0.6506317, -0.2305911)},
        0.975 : {0: (0.1773233, -0.03755152, -3.125617, 51.66143),
                 1: (-0.3091562, 1.057115, -0.4266199)},
        0.99  : {0: (0.2172606, -0.2049983, -4.308647, 82.65278),
                 1: (-0.5765132, 1.915267, -1.057833)},
        0.995 : {0: (0.2480191, -0.3967449, -4.07506, 101.2076),
                 1: (-0.8216993, 2.85189, -1.957697)},
        0.999 : {0: (0.3203646, -1.046171, -0.8434172, 144.566),
                 1: (-1.515429, 6.124994, -6.22971)},
    },
}


class Stationarity_KPSS(UnitRoot):
    """
    The KPSS test of stationarity around a deterministic trend (ScalaTion's `Stationarity_KPSS`).

    Parameters
    ----------
    yy : np.ndarray, shape (n,)
        The series.
    lags : int
        The number of lags of the long-run variance (replaced by the rule's value when
        `lagsType` is set).
    lagsType : str
        "short" (L4 rule), "long" (L12 rule) or "" (use `lags`).
    trend : str, default="c"
        "c" (level stationarity) or "ct" (trend stationarity).
    """

    COEFF = COEFF_KPSS

    def __init__(self, yy: np.ndarray, lags: int, lagsType: str, trend: str = "c") -> None:
        self.yy = np.asarray(yy, dtype=float)
        super().__init__("KPSS", len(self.yy), ("c", "ct"), lagsType, lags, trend)
        self.stat = -0.0

    def olsDetrend(self) -> np.ndarray:
        """The series demeaned ("c") or minus its OLS line ("ct")."""
        y = self.yy.copy()
        if self.trend == "c":
            return y - y.mean()
        w = np.column_stack([np.ones(self.nobs), np.arange(self.nobs, dtype=float)])
        b, *_ = np.linalg.lstsq(w, y, rcond=None)
        return y - w @ b

    def statistic(self) -> float:
        """The KPSS statistic eta / s2 (the lags and the trend set first)."""
        self.setLagsType()
        self.setLags()
        self.setTrend()
        y = self.olsDetrend()
        n = self.nobs
        s = np.cumsum(y)
        eta = float(s @ s) / (n * n)
        s2 = float(y @ y) / n
        tmp = 0.0
        for i in range(1, self.lags + 1):                                # the Bartlett-weighted autocovariances
            tmp += (1.0 - i / (self.lags + 1.0)) * float(y[i:] @ y[: n - i])
        s2 += 2.0 * tmp / n
        self.stat = eta / s2
        return self.stat

    def pvalue(self, stat: Optional[float] = None) -> float:
        """The p-value of the statistic: 1 - its probability under the critical values."""
        st = self.statistic() if stat is None else stat
        p = super().pvalue(st)
        self.pval = math.nan if math.isnan(p) else 1.0 - p
        return self.pval

    def show(self, stat: Optional[float] = None) -> None:
        """Prints the test results (`UnitRoot.show`)."""
        st = self.statistic() if stat is None else stat
        self.pvalue(st)
        super().show(st)
