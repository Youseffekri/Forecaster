"""
Example: Forecasting the Hourly Oil Temperature of an Electricity Transformer (ScalaTion's Example_ETTh1)

ETTh1 (Zhou et al. 2021, Informer): two years of hourly rows of one transformer -- six load series
(HUFL, HULL, MUFL, MULL, LUFL, LULL: high / middle / low useful and useless load) and the oil
temperature OT, the target.  Here OT is forecast h = 1..6 hours ahead from its own lags and the lagged
loads over the FIRST YEAR, `range (0, 8760)` (ScalaTion's `context (rng = 0 until 8760)`): a physical,
lagged response (thermal accumulation and decay) that a symbolic model may recover.  OT goes below zero
(26 of the 8760 hours), so every model is built with `nneg = False` (ScalaTion's `hp("nneg") = 0`).
One scenario per `*_test` function, each printing ScalaTion's blocks:

    randomWalk_test   RandomWalk                                  In-Sample and TnT
    aR_test           AR(24) by Yule-Walker (a day of lags)       In-Sample and TnT
    aRX_D_test        ARX_D (p = px = 6, spec 0, the six loads)   TnT (the paper's baseline layout)
    aRX_SR_D_test     ARX_SR_D.rescale (sign-safe PowR of y and of each load; MinMax to [-2, 2])   TnT
    nARX_Formula_test NARX_SR_D.fromTerms: lags of y, an ema (thermal lag) of two loads with free decays
                                                                   TnT

The rolling validation retrains every RC = 24 rows (a day) to keep the run short: the testing set has
1,753 hours.  Run from the repository root:  python Examples/Example_ETTh1.py [rw|ar|arx_d|arx_sr_d|narx|all]
(no argument: the quick default -- RandomWalk and ARX_D TnT).

Author: Yousef Fekri Dabanloo
Date: 2026-10-16
Version: 1.6
License: MIT License
Note: AI assisted code
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from Util.tools import banner
from Util.transform import PowRForm, TransformT
from modeling.forecasting.ar import AR
from modeling.forecasting.args import Args
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.datasets import Example_ETTh1
from modeling.forecasting.random_walk import RandomWalk

np.set_printoptions(linewidth=200, precision=4, suppress=True)

HH = 6                                                                  # maximum forecasting horizon (hours)
RC = 24                                                                 # retraining cycle: a day
RNG = range(0, 8760)                                                    # the first year (2016-07-01 .. 2017-06-30)
LOADS = Example_ETTh1.LOADS
BASE = Args(nneg=False)                                                 # the preset: OT goes negative, no clipping at zero


def load_etth1():
    """(xe, y): the six loads and the oil temperature over the first year."""
    return Example_ETTh1.loadData(LOADS, Example_ETTh1.response, RNG)


def randomWalk_test(y, hh=HH):
    """RandomWalk: In-Sample Testing and TnT with rolling validation."""
    mod = RandomWalk(y, hh)
    banner(f"In-ST Forecasts: {mod.modelName} on ETTh1 Dataset (OT, first year)")
    mod.inSample_Test()
    banner(f"TnT Forecasts: {mod.modelName} on ETTh1 Dataset (OT, first year)")
    mod.tnT_Test(RC)


def aR_test(y, hh=HH, p=24, method="adjusted"):
    """AR(p) with a day of lags: In-Sample and TnT."""
    args = BASE.copy(p=p, method=method)
    mod = AR(y, hh, args)
    banner(f"In-ST Forecasts: {mod.modelName} on ETTh1 Dataset")
    mod.inSample_Test()
    banner(f"TnT Forecasts: {mod.modelName} on ETTh1 Dataset")
    mod.tnT_Test(RC)


def aRX_D_test(y, xe, hh=HH, args=None):
    """ARX_D (direct) on the lags of OT and of the six loads: TnT."""
    args = BASE.copy(p=6, px=6, spec=0) if args is None else args
    mod = ARX_D(y, hh, xe, args)
    print(f"columns ({len(mod.fname)}): {mod.fname}")
    banner(f"TnT Forecasts: {mod.modelName} on ETTh1 Dataset")
    mod.tnT_Test(RC)


def aRX_SR_D_test(y, xe, hh=HH, args=None):
    """ARX_SR_D (direct, SR terms): the sign-safe power sign(y) |y|^0.67 next to the lags of OT and of each load
    (`PowRForm`), rescaled by MinMax to ScalaTion's centered [-2, 2] first -- a power or log of a negative value
    is not defined there, and OT and the loads go below zero anyway: TnT."""
    args = BASE.copy(p=6, px=3, spec=0) if args is None else args
    fEndo = [PowRForm()]                                                 # sign (y) |y|^0.67 on the [-2, 2] scale
    fExo = [[PowRForm()] for _ in range(xe.shape[1])]                    # sign (xe_j) |xe_j|^0.67
    mod = ARX_SR_D.rescale(y, hh, xe, args, fEndo, fExo, TransformT.MinMax)
    print(f"columns ({len(mod.fname)}): {mod.fname}")
    banner(f"TnT Forecasts: {mod.modelName} on ETTh1 Dataset")
    mod.tnT_Test(RC)


def nARX_Formula_test(hh=HH):
    """NARX_SR_D from a string formula: the lags of OT and exponentially weighted moving averages (the thermal
    lag of the oil) of the high and middle useful load with FREE decays, fitted by VarPro on the training
    rows; then TnT with rolling validation (the coefficients refitted, the decays held)."""
    data = Example_ETTh1.context(LOADS, rng=RNG)                         # {"y", "hufl", "hull", "mufl", "mull", "lufl", "lull"}
    formula = "y ~= y~(3,1) + ema(hufl~1, ?) + ema(mufl~1, ?) + lufl~1"
    mod = NARX_SR_D.fromTerms(formula, data, hh, BASE.copy(lam=1.0))
    print(f"formula  = {mod.formulaStr}\nparam_nl = {mod.w_nl}, nFree = {mod.nFree}, stepCounts = {mod.stepCounts}")
    banner(f"TnT Forecasts: {mod.modelName} on ETTh1 Dataset")
    mod.tnT_Test(RC)


SCENARIOS = {
    "rw": lambda y, xe: randomWalk_test(y),
    "ar": lambda y, xe: aR_test(y),
    "arx_d": lambda y, xe: aRX_D_test(y, xe),
    "arx_sr_d": lambda y, xe: aRX_SR_D_test(y, xe),
    "narx": lambda y, xe: nARX_Formula_test(),
}


def main(argv):
    xe, y = load_etth1()
    print(f"y.shape = {y.shape} (OT: min {y.min():.2f}, max {y.max():.2f}, {int((y < 0).sum())} hours below zero), "
          f"xe.shape = {xe.shape} ({', '.join(LOADS)}), hh = {HH}, rc = {RC}")
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
