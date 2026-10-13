"""
Example: Forecasting the Daily Peak Electricity Load of GEFCom2014 (ScalaTion's Example_GEFCom)

The GEFCom2014 load track (Hong et al. 2016) aggregated to days (data/gefcom2014_load_daily.csv, 2,099
days from 2005-01-01 to 2010-09-30): the daily peak load of a US utility, forecast h = 1..6 days ahead
from its own lags and the lagged daily mean (temp) and maximum (tmax) temperature -- no weather
forecasts, so the temperatures enter lagged, as the loads do.  A week of lags (p = q = 7) carries the
weekly cycle.  The load-temperature relation is V-shaped (heating below ~55 F, cooling above ~65 F),
which the nonlinear scenario lets a free hinge threshold find.  One scenario per `*_test` function,
each printing ScalaTion's blocks:

    randomWalk_test   RandomWalk                                            In-Sample and TnT
    aR_test           AR(7) by Yule-Walker                                  In-Sample and TnT
    aRX_D_test        ARX_D (p = q = 7, spec 0; temp, tmax)                 TnT
    aRX_SR_D_test     ARX_SR_D.rescale (Log1p; Pow of y, Pow and Log of the temperatures)   TnT
    nARX_Formula_test NARX_SR_D.fromTerms with a free hinge threshold on temp~1 (the V shape)   TnT

Run from the repository root:  python Examples/Example_GEFCom.py [rw|ar|arx_d|arx_sr_d|narx|all]
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
from Util.transform import LogForm, PowForm, TransformT
from modeling.forecasting.ar import AR
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_GEFCom
from modeling.forecasting.random_walk import RandomWalk

np.set_printoptions(linewidth=200, precision=4, suppress=True)

HH = 6                                                                  # maximum forecasting horizon (days)
RC = 5                                                                  # retraining cycle of the rolling validation
LAG = 7                                                                 # a week of lags
EXO = ["temp", "tmax"]                                                  # daily mean and maximum temperature (deg F)


def load_gefcom():
    """(y, xe): the daily peak load and the two temperature series."""
    return Example_GEFCom.loadData(EXO)


def randomWalk_test(y, hh=HH):
    """RandomWalk: In-Sample Testing and TnT with rolling validation."""
    banner("In-ST Forecasts: RandomWalk on GEFCom2014 Dataset (daily peak load)")
    RandomWalk(None, y, hh).inSample_Test()
    banner("TnT Forecasts: RandomWalk on GEFCom2014 Dataset (daily peak load)")
    RandomWalk(None, y, hh).tnT_Test(RC)


def aR_test(y, hh=HH, p=LAG, method="adjusted"):
    """AR(7), a week of lags: In-Sample and TnT."""
    mod = AR({"p": p}, y, hh, method)
    banner(f"In-ST Forecasts: {mod.modelName} on GEFCom2014 Dataset")
    mod.inSample_Test()
    mod = AR({"p": p}, y, hh, method)
    banner(f"TnT Forecasts: {mod.modelName} on GEFCom2014 Dataset")
    mod.tnT_Test(RC)


def aRX_D_test(y, xe, hh=HH, hparam=None, method="sm_ols"):
    """ARX_D (direct) on a week of lags of the peak load and of the two temperatures: TnT."""
    hparam = {"p": LAG, "px": LAG, "spec": 0} if hparam is None else hparam
    mod = ARX_D(hparam, y, hh, xe, method=method)
    print(f"columns ({len(mod.fname)}): {mod.fname}")
    banner(f"TnT Forecasts: {mod.modelName} on GEFCom2014 Dataset")
    mod.tnT_Test(RC)


def aRX_SR_D_test(y, xe, hh=HH, hparam=None, method="sm_ols"):
    """ARX_SR_D (direct, SR terms): y^1.5 next to the lags of the load, temp^1.5 and log (1 + temp) next to the
    lags of each temperature (a bend for the V shape), rescaled by Log1p first (as ScalaTion's tests): TnT."""
    hparam = {"p": LAG, "px": 3, "spec": 0, "crx": 0} if hparam is None else hparam
    fEndo = [PowForm()]                                                  # (y + 0)^1.5
    fExo = [[PowForm(), LogForm()] for _ in range(xe.shape[1])]          # xe_j^1.5 and log (1 + xe_j)
    mod = ARX_SR_D.rescale(hparam, y, hh, xe, fEndo, fExo, method=method, tFormT=TransformT.Log1p)
    print(f"columns ({len(mod.fname)}): {mod.fname}")
    banner(f"TnT Forecasts: {mod.modelName} on GEFCom2014 Dataset")
    mod.tnT_Test(RC)


def nARX_Formula_test(hh=HH):
    """NARX_SR_D from a string formula: a week of load lags, the lagged mean temperature and a hinge of it with
    a FREE threshold (the cooling branch of the V), fitted by VarPro on the training rows; the recovered
    threshold is printed next to the V table of the data; then TnT with rolling validation."""
    data = Example_GEFCom.context(EXO)                                   # {"y", "temp", "tmax"}
    print(f"mean peak load per 10 F band of the daily mean temperature (20..90 F): {np.round(Example_GEFCom.v_table(), 1)}")
    formula = f"y ~= y~({LAG},1) + temp~1 + hinge(temp~1, ?)"
    mod = NARX_SR_D.fromTerms(formula, data, hh, {"lambda": 1.0})
    print(f"formula  = {mod.formulaStr}\nparam_nl = {mod.w_nl} (the hinge threshold, deg F), nFree = {mod.nFree}, stepCounts = {mod.stepCounts}")
    banner(f"TnT Forecasts: {mod.modelName} on GEFCom2014 Dataset")
    mod.tnT_Test(RC)


SCENARIOS = {
    "rw": lambda y, xe: randomWalk_test(y),
    "ar": lambda y, xe: aR_test(y),
    "arx_d": lambda y, xe: aRX_D_test(y, xe),
    "arx_sr_d": lambda y, xe: aRX_SR_D_test(y, xe),
    "narx": lambda y, xe: nARX_Formula_test(),
}


def main(argv):
    y, xe = load_gefcom()
    print(f"y.shape = {y.shape} (peak load), xe.shape = {xe.shape} ({', '.join(EXO)}), hh = {HH}, lag = {LAG}")
    which = [a.lower() for a in argv] or ["default"]
    if "all" in which:
        which = list(SCENARIOS)
    for w in which:
        if w == "default":
            randomWalk_test(y)
            aRX_D_test(y, xe)
        elif w in SCENARIOS:
            SCENARIOS[w](y, xe)
        else:
            print(f"unknown scenario '{w}': choose from {', '.join(SCENARIOS)} or all")


if __name__ == "__main__":
    main(sys.argv[1:])
