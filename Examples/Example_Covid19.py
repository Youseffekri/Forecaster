"""
Example: Forecasting Weekly Covid-19 Deaths (ScalaTion's Example_Covid scenarios)

Forecasting the weekly US Covid-19 deaths (data/covid_19_weekly.csv, the first 116 weeks, ScalaTion's
`Example_Covid.clip`) h = 1..6 weeks ahead, with icu_patients and positive_rate as the exogenous
variables (`Example_Covid.BEST2_EXO`).  One scenario per `*_test` function, in the shape of the Scala
tests (`example_CovidTest3..6`, `randomWalkTest3/4`, `aRX_DTest4`, `aRX_SR_DTest`, `aRX_FormulaTest5`,
`nARX_FormulaTest5`, `nARX_SR_DTest5`): build the model, In-Sample test it, then Train-and-Test it with
rolling validation; every model prints ScalaTion's blocks (REPORT, "fitMap QoF =").

    randomWalk_test   RandomWalk                                 In-Sample and TnT
    aR_test           AR(p) by Yule-Walker, p = 1..10            In-Sample and TnT (example_CovidTest5/6)
    aRIMA_test        ARIMA(1,1,1), ARIMA(2,1,0), ARIMA(1,0,1)   In-Sample and TnT (statsmodels-backed)
    aRX_test          ARX (p = q = 6, spec 0), also rescaled     TnT (example_CovidTest16, recursive)
    aRX_D_test        ARX_D (p = q = 6, spec 0)                  TnT (aRX_DTest4, the paper's baseline, direct)
    aRX_SR_test       ARX_SR.rescale with PowForm / LogForm      TnT (recursive, SR terms)
    aRX_SR_D_test     ARX_SR_D.rescale with PowForm, RootForm, LogForm   In-Sample and TnT (aRX_SR_DTest)
    aRX_Formula_test  ARX_Formula DSL: fit (ARX) and fit_D (ARX_D), TFORM: Log1p   (aRX_FormulaTest5)
    nARX_Formula_test NARX_Formula fitNL_D and NARX_SR_D.fromTerms on a string formula  (nARX_FormulaTest5)
    mHAttn_test       the multi-head attention regressor's feature scores (needs torch; skipped otherwise)

Hyper-parameters follow ScalaTion: `spec` counts the trend columns (0 implicit constant, 1 linear
trend, 2 quadratic, 3 cubic), `p` the lags of y, `q` the lags of each exogenous variable, `skip` the
rows dropped from the In-Sample diagnosis (p by default).

Run from the repository root:  python Examples/Example_Covid19.py [rw|ar|arima|arx|arx_d|arx_sr|arx_sr_d|formula|narx|mhattn|all]
(no argument: the quick default -- RandomWalk, AR(6) and ARX_D).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.preprocessing import StandardScaler

from Util.tools import banner
from Util.transform import LogForm, PowForm, RootForm
from modeling.forecasting.ar import AR
from modeling.forecasting.arima import ARIMA
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_formula import ARX_Formula, FormulaVar, log1p
from modeling.forecasting.arx_sr import ARX_SR
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula, decay, ema, half, sat, thresh, trans, width
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid
from modeling.forecasting.random_walk import RandomWalk

np.set_printoptions(linewidth=200, precision=4, suppress=True)

HH = 6                                                                  # maximum forecasting horizon
RC = 5                                                                  # retraining cycle of the rolling validation
EXO = Example_Covid.BEST2_EXO                                           # icu_patients, positive_rate


def load_covid():
    """(y, xe): the weekly deaths and the two best exogenous variables, the first 116 weeks."""
    return Example_Covid.load_data(EXO)


def randomWalk_test(y, hh=HH):
    """RandomWalk: In-Sample Testing and TnT with rolling validation (randomWalkTest3, randomWalkTest4)."""
    banner(f"In-ST Forecasts: RandomWalk on COVID-19 Dataset")
    RandomWalk(y, hh).inSample_Test()
    banner(f"TnT Forecasts: RandomWalk on COVID-19 Dataset")
    RandomWalk(y, hh).tnT_Test(RC)


def aR_test(y, hh=HH, ps=range(1, 11), method="adjusted"):
    """AR(p) for several p (example_CovidTest5 In-Sample, example_CovidTest6 TnT); Yule-Walker as ScalaTion's AR."""
    for p in ps:
        mod = AR({"p": p}, y, hh, method)
        banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test()
    for p in ps:
        mod = AR({"p": p}, y, hh, method)
        banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test(RC)


def aRIMA_test(y, hh=HH, orders=((1, 1, 1), (2, 1, 0), (1, 0, 1))):
    """ARIMA(p, d, q) for several orders (the Python counterpart of example_CovidTest7/8 with ARMA): In-Sample and TnT."""
    for p, d, q in orders:
        mod = ARIMA({"p": p, "d": d, "q": q}, y, hh)
        banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test()
        mod = ARIMA({"p": p, "d": d, "q": q}, y, hh)
        banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test(RC)


def aRX_test(y, xe, hh=HH, args=None, method="sm_ols"):
    """ARX (recursive) on the raw series and rescaled with a StandardScaler (ScalaTion's NormForm): TnT."""
    args = {"p": 6, "q": 6, "spec": 0} if args is None else args
    mod = ARX(args, y, hh, xe, method=method)
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)
    mod = ARX.rescale(args, y, hh, xe, method=method, tForm=StandardScaler)
    banner(f"TnT Forecasts: {mod.modelName} (rescaled, NormForm) on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_D_test(y, xe, hh=HH, args=None, method="sm_ols"):
    """ARX_D (direct, one regression per horizon): In-Sample and TnT (aRX_DTest4 with p = px = 6, spec = 0)."""
    args = {"p": 6, "q": 6, "spec": 0} if args is None else args
    mod = ARX_D(args, y, hh, xe, method=method)
    banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test()
    mod = ARX_D(args, y, hh, xe, method=method)
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_SR_test(y, xe, hh=HH, args=None, method="sm_ols"):
    """ARX_SR (recursive, symbolic regression terms): y^1.5 next to the lags of y and log (1 + xe_j) next to
    each exogenous variable's lags; rescaled to (0, 4) first so the powers and logs stay defined."""
    args = {"p": 6, "q": 4, "spec": 0, "cross": False} if args is None else args
    fEndo = [PowForm()]                                                  # (y + 0)^1.5
    fExo = [[LogForm()] for _ in range(xe.shape[1])]                     # log (1 + xe_j)
    mod = ARX_SR.rescale(args, y, hh, xe, fEndo, fExo, method=method)
    print(f"columns: {mod.fname}")
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_SR_D_test(y, xe, hh=HH, args=None, method="sm_ols"):
    """ARX_SR_D (direct, SR terms): PowForm and RootForm of y, LogForm of each exogenous variable (aRX_SR_DTest)."""
    args = {"p": 6, "q": 4, "spec": 0, "cross": False} if args is None else args
    fEndo = [PowForm(), RootForm()]                                      # y^1.5 and y^0.5
    fExo = [[LogForm()] for _ in range(xe.shape[1])]                     # log (1 + xe_j)
    mod = ARX_SR_D.rescale(args, y, hh, xe, fEndo, fExo, method=method)
    print(f"columns: {mod.fname}")
    banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test()
    mod = ARX_SR_D.rescale(args, y, hh, xe, fEndo, fExo, method=method)
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_Formula_test(hh=HH):
    """ARX_Formula written in the DSL (aRX_FormulaTest5): fit_D gives the direct ARX_D on the terms, fit the
    recursive ARX for h = 1; with TFORM: Log1p every base column is log1p-transformed on its training slice
    and the forecasts come back in deaths."""
    data = Example_Covid.context(EXO)                                     # {"y", "icu", "pRate"}
    y, icu, pRate = FormulaVar("y"), FormulaVar("icu"), FormulaVar("pRate")
    f = ARX_Formula(y, y.lags(6, 1) + icu.lags(4, 1) + [log1p(pRate.lag_(1))] + [y.lag_(1) * icu.lag_(1)])
    banner(f"ARX_Formula: {f.formulaStr}")
    print(f"key     = {f.key}\nlagHP   = {f.lagHP()}\nnTerms  = {f.nTerms}")
    mod_D = f.fit_D(data, hh)                                             # ARX_D.fromTerms + trainNtest
    mod_D.forecastAll()
    banner(f"diagnoseAll: {mod_D.modelName}")
    mod_D.diagnoseAll()
    mod_D.setSkip()
    mod_D.rollValidate(RC)
    banner(f"diagnoseAll (TnT): {mod_D.modelName}")
    mod_D.diagnoseAll(tRng=range(mod_D.tr_size, len(data["y"])))

    mod = f.fit(data, 1)                                                  # ARX.fromTerms + trainNtest (h = 1)
    print(f"{mod.modelName}: h = 1 forecasts of the recursive model, first 10 = {mod.forecast()[:10, 0]}")

    g = ARX_Formula(y, y.lags(6, 1) + icu.lags(4, 1) + pRate.lags(4, 1), "Log1p")
    banner(f"ARX_Formula with a column transform: {g.formulaStr}")
    mod_T = g.fit_D(data, hh)
    print(f"yForm = {mod_T.yForm}; forecasts in deaths, rows 92..97 at h = 1: {mod_T.Yf[92:98, 1]}")


def nARX_Formula_test(hh=HH):
    """NARX_Formula (nARX_FormulaTest5): lags, a saturating icu effect, a distributed lag of pRate and a
    smooth-transition switch of the y lags; fit_D keeps the start values, fitNL_D fits the four free
    parameters (VarPro); then the same model from a string formula through NARX_SR_D.fromTerms, TnT."""
    data = Example_Covid.context(EXO)
    y, icu, pRate = FormulaVar("y"), FormulaVar("icu"), FormulaVar("pRate")
    f = NARX_Formula(y, y.lags(3, 1) + [sat(icu.lag_(1), half()), ema(pRate.lag_(1), decay())]
                     + trans(y.lag_(1), thresh(), width()) * y.lags(3, 1))
    banner(f"NARX_Formula: {f.formulaStr}")
    print(f"key     = {f.key}\nnParams = {f.nParams}, nFree = {f.nFree}")

    banner("fit_D: ARX_D with the parameters at their start values")
    mod0 = f.fit_D(data, hh, scaled=True)
    print(f"formula = {f.formulaStr}")
    mod0.forecastAll()
    mod0.diagnoseAll()

    banner("fitNL_D: NARX_SR_D with the parameters fitted by varpro")
    mod = f.fitNL_D(data, hh, scaled=True)
    print(f"formula = {f.formulaStr}\nparam_nl = {mod.w_nl}, steps = {mod.steps}")
    mod.forecastAll()
    mod.diagnoseAll()

    formula = "y ~= y~(3,1) + log(? + icu~1) + pow(y~1, ?) + sat(pRate~1, ?)"
    mod2 = NARX_SR_D.fromTerms(formula, data, hh, lam=1.0, scaled=True)
    banner(f"TnT Forecasts: {mod2.modelName} on {mod2.formulaStr}")
    mod2.tnT_Test(RC)


def mHAttn_test(y, xe, hh=HH, args=None):
    """The multi-head attention regressor on the ARX_SR design matrix: trains on all rows and ranks the
    columns by their mean attention score (needs torch; skipped with a message when it is missing)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from modeling.forecasting.neuralforecasting.mhattn_regressor import MHAttn_Regressor
    except ImportError as e:
        print(f"mHAttn_test skipped: torch is not installed ({e})")
        return
    args = {"p": 6, "q": 4, "spec": 0, "cross": False} if args is None else args
    model = ARX_SR.rescale(args, y, hh, xe, fExo=[[] for _ in range(xe.shape[1])])
    X_torch = torch.tensor(model.X, dtype=torch.float32)
    y_torch = torch.tensor(model.y.reshape(-1, 1), dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=len(y), shuffle=False)
    net = MHAttn_Regressor(input_dim=X_torch.shape[1], d_model=72, num_heads=4)
    criterion = nn.MSELoss()
    net.trainX(train_loader, criterion)
    heatMap = net.testX(train_loader, criterion)
    scores = heatMap.mean(dim=0).numpy()
    ranked = sorted(enumerate(scores), key=lambda s: s[1], reverse=True)[:20]
    print("selected columns:", [(model.fname[i], round(float(s), 6)) for i, s in ranked])
    print([i for i, _ in ranked])


SCENARIOS = {
    "rw": lambda y, xe: randomWalk_test(y),
    "ar": lambda y, xe: aR_test(y),
    "arima": lambda y, xe: aRIMA_test(y),
    "arx": lambda y, xe: aRX_test(y, xe),
    "arx_d": lambda y, xe: aRX_D_test(y, xe),
    "arx_sr": lambda y, xe: aRX_SR_test(y, xe),
    "arx_sr_d": lambda y, xe: aRX_SR_D_test(y, xe),
    "formula": lambda y, xe: aRX_Formula_test(),
    "narx": lambda y, xe: nARX_Formula_test(),
    "mhattn": lambda y, xe: mHAttn_test(y, xe),
}


def main(argv):
    y, xe = load_covid()
    print(f"y.shape = {y.shape}, xe.shape = {xe.shape} ({', '.join(EXO)}), hh = {HH}")
    which = [a.lower() for a in argv] or ["default"]
    if "all" in which:
        which = list(SCENARIOS)
    for w in which:
        if w == "default":                                               # the quick default
            randomWalk_test(y)
            aR_test(y, ps=[6])
            aRX_D_test(y, xe)
        elif w in SCENARIOS:
            SCENARIOS[w](y, xe)
        else:
            print(f"unknown scenario '{w}': choose from {', '.join(SCENARIOS)} or all")


if __name__ == "__main__":
    main(sys.argv[1:])
