"""
Example: Forecasting Weekly Influenza-Like Illness (ScalaTion's Example_ILI scenarios)

Forecasting the weekly US ILI patient count ILITOTAL of the CDC's ILINet (data/ILINetNov25.csv, 1,208
weeks from 2002-10-06 to 2025-11-23) h = 1..12 weeks ahead, with the population-weighted ILI
percentage (%WEIGHTED ILI) and the total patient visits (TOTAL PATIENTS, ScalaTion's "OT") as the
exogenous variables.  One scenario per `*_test` function, in the shape of ScalaTion's
`example_ILITest3..6` and `example_ILITest10`:

    randomWalk_test   RandomWalk                         In-Sample (example_ILITest3) and TnT (example_ILITest4)
    aR_test           AR(p) by Yule-Walker, p = 1..10    In-Sample (example_ILITest5) and TnT
    aRIMA_test        ARIMA(p, 1, q) for a few orders    In-Sample (the ARMA grid of example_ILITest6)
    aRX_D_test        ARX_D (p = q = 10, spec 1)         In-Sample and TnT
    aRX_SR_D_test     ARX_SR_D.rescale (Log1p; Pow of y and of each exogenous lag range, spec 1, cross terms)
                                                          In-Sample and TnT (example_ILITest10's model, direct)

Run from the repository root:  python Examples/Example_ILI.py [rw|ar|arima|arx_d|arx_sr_d|all]
(no argument: the quick default -- RandomWalk and ARX_D TnT).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from Util.tools import banner
from Util.transform import PowForm, TransformT
from modeling.forecasting.ar import AR
from modeling.forecasting.arima import ARIMA
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_ILI
from modeling.forecasting.random_walk import RandomWalk

np.set_printoptions(linewidth=200, precision=4, suppress=True)

HH = 12                                                                 # maximum forecasting horizon (weeks)
RC = 5                                                                  # retraining cycle of the rolling validation
EXO = Example_ILI.EXO                                                   # %WEIGHTED ILI, TOTAL PATIENTS
RESPONSE = Example_ILI.RESPONSE                                         # ILITOTAL


def load_ili():
    """(y, xe): the weekly ILI total and the two exogenous variables (ScalaTion's `loadData`)."""
    return Example_ILI.loadData(EXO)


def randomWalk_test(y, hh=HH):
    """RandomWalk: In-Sample Testing (example_ILITest3) and TnT with rolling validation (example_ILITest4)."""
    banner(f"In-ST Forecasts: RandomWalk on ILI Dataset ({RESPONSE})")
    RandomWalk(None, y, hh).inSample_Test()
    banner(f"TnT Forecasts: RandomWalk on ILI Dataset ({RESPONSE})")
    RandomWalk(None, y, hh).tnT_Test(RC)


def aR_test(y, hh=HH, ps=range(1, 11), method="adjusted"):
    """AR(p) for several p: In-Sample (example_ILITest5) and TnT (the AR(1) row of example_ILITest4)."""
    for p in ps:
        mod = AR({"p": p}, y, hh, method)
        banner(f"In-ST Forecasts: {mod.modelName} on ILI Dataset")
        mod.inSample_Test()
    for p in ps:
        mod = AR({"p": p}, y, hh, method)
        banner(f"TnT Forecasts: {mod.modelName} on ILI Dataset")
        mod.tnT_Test(RC)


def aRIMA_test(y, hh=HH, orders=((1, 1, 0), (1, 1, 1), (2, 1, 1), (3, 1, 2))):
    """ARIMA(p, d, q) for a few orders: In-Sample (example_ILITest6's ARMA grid, statsmodels-backed)."""
    for p, d, q in orders:
        mod = ARIMA({"p": p, "d": d, "q": q}, y, hh)
        banner(f"In-ST Forecasts: {mod.modelName} on ILI Dataset")
        mod.inSample_Test()


def aRX_D_test(y, xe, hh=HH, hparam=None, method="sm_ols"):
    """ARX_D (direct) with p = q = 10 lags and a linear trend (spec 1): In-Sample and TnT."""
    hparam = {"p": 10, "px": 10, "spec": 1} if hparam is None else hparam
    mod = ARX_D(hparam, y, hh, xe, method=method)
    banner(f"In-ST Forecasts: {mod.modelName} on ILI Dataset")
    mod.inSample_Test()
    mod = ARX_D(hparam, y, hh, xe, method=method)
    banner(f"TnT Forecasts: {mod.modelName} on ILI Dataset")
    mod.tnT_Test(RC)


def aRX_SR_D_test(y, xe, hh=HH, hparam=None, method="ridge"):
    """ARX_SR_D (direct, SR terms) as example_ILITest10's model: y^1.5 next to the lags of y, xe_j^1.5 next to
    each exogenous variable's lags, a linear trend and ENDO-EXO cross terms, rescaled by Log1p first (as
    ScalaTion's tests); the centered ridge (lambda 1) of ScalaTion's RidgeRegression: In-Sample and TnT."""
    hparam = {"p": 10, "px": 10, "spec": 1, "crx": 1, "lambda": 1.0} if hparam is None else hparam
    fEndo = [PowForm()]                                                  # (y + 0)^1.5
    fExo = [[PowForm()] for _ in range(xe.shape[1])]                     # (xe_j + 0)^1.5
    mod = ARX_SR_D.rescale(hparam, y, hh, xe, fEndo, fExo, method=method, tFormT=TransformT.Log1p)
    print(f"columns ({len(mod.fname)}): {mod.fname}")
    banner(f"In-ST Forecasts: {mod.modelName} on ILI Dataset")
    mod.inSample_Test()
    mod = ARX_SR_D.rescale(hparam, y, hh, xe, fEndo, fExo, method=method, tFormT=TransformT.Log1p)
    banner(f"TnT Forecasts: {mod.modelName} on ILI Dataset")
    mod.tnT_Test(RC)


SCENARIOS = {
    "rw": lambda y, xe: randomWalk_test(y),
    "ar": lambda y, xe: aR_test(y),
    "arima": lambda y, xe: aRIMA_test(y),
    "arx_d": lambda y, xe: aRX_D_test(y, xe),
    "arx_sr_d": lambda y, xe: aRX_SR_D_test(y, xe),
}


def main(argv):
    y, xe = load_ili()
    print(f"y.shape = {y.shape} ({RESPONSE}), xe.shape = {xe.shape} ({', '.join(EXO)}), hh = {HH}")
    which = [a.lower() for a in argv] or ["default"]
    if "all" in which:
        which = list(SCENARIOS)
    for w in which:
        if w == "default":
            randomWalk_test(y)
            mod = ARX_D({"p": 10, "px": 10, "spec": 1}, y, HH, xe)
            banner(f"TnT Forecasts: {mod.modelName} on ILI Dataset")
            mod.tnT_Test(RC)
        elif w in SCENARIOS:
            SCENARIOS[w](y, xe)
        else:
            print(f"unknown scenario '{w}': choose from {', '.join(SCENARIOS)} or all")


if __name__ == "__main__":
    main(sys.argv[1:])
