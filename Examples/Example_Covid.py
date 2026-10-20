"""
Example: Forecasting Weekly Covid-19 Deaths (ScalaTion's Example_Covid scenarios)

Forecasting the weekly US Covid-19 deaths (data/covid_19_weekly.csv, the first 116 weeks, ScalaTion's
`Example_Covid.clip`) h = 1..6 weeks ahead, with icu_patients and positive_rate as the exogenous
variables (`Example_Covid.BEST2_EXO`).  One scenario per `*_test` function, in the shape of the Scala
tests (`example_CovidTest3..6`, `randomWalkTest3/4`, `aRX_DTest4`, `aRX_SR_DTest`, `aRX_FormulaTest5`,
`nARX_FormulaTest5`, `nARX_SR_DTest5..8`): build the model, In-Sample test it, then Train-and-Test it with
rolling validation; every model prints ScalaTion's blocks (REPORT, "fitMap QoF =", SUMMARY).

    randomWalk_test   RandomWalk                                 In-Sample and TnT
    aR_test           AR(p) by Yule-Walker, p = 1..10            In-Sample and TnT (example_CovidTest5/6)
    aRIMA_test        ARIMA(1,1,1), ARIMA(2,1,0), ARIMA(1,0,1)   In-Sample and TnT (statsmodels-backed)
    aRX_test          ARX (p = px = 6, spec 0), also rescaled (Norm)   TnT (example_CovidTest16, recursive)
    aRX_D_test        ARX_D (p = px = 6, spec 0)                 TnT (aRX_DTest4, the paper's baseline, direct)
    aRX_D_test2       ARX_D ridge: tune on the validation rows (grid, golden, random, a wider block, 5-fold), In-Sample
                      and TnT at the knobs found, then every feature selection technique with Model.tuning on
    aRX_D_enet_test   ridge vs elastic net on standardized columns: the knobs (lambda; lambda and l1_ratio) by tune, TnT,
                      the columns the L1 part kept
    aRX_D_diff_test   ARX_D on the differenced series (d = 1), forecasts integrated back to levels, TnT against the
                      undifferenced model on the same testing rows
    aRX_SR_test       ARX_SR.rescale (Log1p) with PowForm / LogForm   TnT (recursive, SR terms)
    aRX_SR_D_test     ARX_SR_D.rescale (Log1p) with PowForm, RootForm, LogForm   In-Sample, SUMMARY, TnT (aRX_SR_DTest)
    aRX_Formula_test  ARX_Formula DSL: fit (ARX) and fit_D (ARX_D), TFORM: Log1p   (aRX_FormulaTest5)
    nARX_Formula_test NARX_Formula fitNL_D and NARX_SR_D.fromTerms on a string formula  (nARX_FormulaTest5)
    nARX_SR_D_test    NARX_SR_D.rescale with Root transforms, default vs fitted parameters, Backward FS  (nARX_SR_DTest5)
    optimizers_test   the three OptimizerNL on Pow / Log transforms, lambda = 6  (nARX_SR_DTest8)
    mHAttn_test       the multi-head attention regressor's feature scores (needs torch; skipped otherwise)
    run_test          the one-object style: dataset, model, hyper-parameters and test in one Args, run (args)

Every model is built as `Model (y, hh, [xe,] args)`: the data in the call, the settings in an `Args`
(`args = Args (); args.p = 6; ...`, checked when set; `Args.help (ARX_D)` lists what ARX_D reads).  The
hyper-parameters follow ScalaTion: `spec` counts the trend columns (0 implicit constant, 1 linear trend,
2 quadratic, 3 cubic), `p` the lags of y, `px` the lags of each exogenous variable, `skip` the rows
dropped from the In-Sample diagnosis (p by default), `lam` the ridge penalty (RidgeRegression.hp("lambda")).

Run from the repository root:  python Examples/Example_Covid.py [rw|ar|arima|arx|arx_d|arx_d2|enet|diff|arx_sr|arx_sr_d|formula|narx|narx_sr|optimizers|mhattn|run|all]
(no argument: the quick default -- RandomWalk, AR(6) and ARX_D).

Author: Yousef Fekri Dabanloo
Date: 2026-10-17
Version: 1.8
License: MIT License
Note: AI assisted code
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time

import numpy as np

from Util.tools import banner
from Util.transform import LogForm, PowForm, RootForm, TransformT
from modeling.feature_selection import FeatureSelection, SelectionTech
from modeling.forecasting.ar import AR
from modeling.forecasting.arima import ARIMA
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_formula import ARX_Formula, FormulaVar, log1p
from modeling.forecasting.arx_sr import ARX_SR
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula, decay, ema, half, sat, thresh, trans, width
from modeling.forecasting.nonlinear.narx_sr_d import OPTIMIZER_NL, NARX_SR_D, OptimizerNL
from modeling.forecasting.datasets import Example_Covid
from modeling.forecasting.forecaster import Forecaster, Model
from modeling.hyper_parameter import Knob
from modeling.forecasting.random_walk import RandomWalk
from modeling.forecasting.runner import run

np.set_printoptions(linewidth=200, precision=4, suppress=True)

HH = 6                                                                  # maximum forecasting horizon
RC = 5                                                                  # retraining cycle of the rolling validation
EXO = Example_Covid.BEST2_EXO                                           # icu_patients, positive_rate


def load_covid():
    """(xe, y): the two best exogenous variables and the weekly deaths, the first 116 weeks (ScalaTion's
    clip (loadData (BEST2_EXO)))."""
    return Example_Covid.clip(Example_Covid.loadData(EXO))


def randomWalk_test(y, hh=HH):
    """RandomWalk: In-Sample Testing and TnT with rolling validation (randomWalkTest3, randomWalkTest4)."""
    mod = RandomWalk(y, hh)
    banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test()
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aR_test(y, hh=HH, ps=range(1, 11), method="adjusted"):
    """AR(p) for several p (example_CovidTest5 In-Sample, example_CovidTest6 TnT); Yule-Walker as ScalaTion's AR."""
    args = Args(method=method)
    for p in ps:
        args.p = p
        mod = AR(y, hh, args)
        banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test()
    for p in ps:
        mod = AR(y, hh, args.copy(p=p))
        banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test(RC)


def aRIMA_test(y, hh=HH, orders=((1, 1, 1), (2, 1, 0), (1, 0, 1))):
    """ARIMA(p, d, q) for several orders (the Python counterpart of example_CovidTest7/8 with ARMA): In-Sample and TnT."""
    for p, d, q in orders:
        args = Args(p=p, d=d, q=q)
        mod = ARIMA(y, hh, args)
        banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test()
        banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test(RC)


def aRX_test(y, xe, hh=HH, args=None):
    """ARX (recursive) on the raw series and rescaled by NormForm (`rescale (..., TransformT.Norm)`): TnT."""
    if args is None:
        args = Args()
        args.p, args.px, args.spec = 6, 6, 0
    mod = ARX(y, hh, xe, args)
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)
    mod = ARX.rescale(y, hh, xe, args, TransformT.Norm)
    banner(f"TnT Forecasts: {mod.modelName} (rescaled, NormForm) on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_D_test(y, xe, hh=HH, args=None):
    """ARX_D (direct, one regression per horizon): In-Sample and TnT (aRX_DTest4 with p = px = 6, spec = 0)."""
    if args is None:
        args = Args()
        args.p, args.px, args.spec = 6, 6, 0                            # method sm_ols (ARX_D's default)
    mod = ARX_D(y, hh, xe, args)
    banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test()
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_D_test2(y, xe, hh=HH, args=None):
    """ARX_D fitted by the ridge (p = px = 6, spec = 1) with its knobs TUNED on the validation rows
    (`tune`: the knobs the regression declares -- `Knob.LAMBDA`, decades from 1e-3 to 1e3 --, the best mean
    validation sMAPE over h = 1..hh, the split 68 / 24 / 24) -- with the lambda landscape on raw and on
    standardized columns, and the knobs found by the golden-section refinement, by random search, on a wider
    validation block and by 5-fold cross-validation on the training rows (`Model.folds`) --, then In-Sample
    and TnT at the knobs found; then every feature selection technique with `Model.tuning` on (each
    candidate column set at its own best knobs), the selected model of each reported and TnT-tested at
    the knobs its best step found (the tag of every name, _L1.0, says which).  Under `method = "enet"` the
    same code tunes two knobs (lambda, l1_ratio: the 35-point grid) and the names carry _L.._A.. ."""
    if args is None:
        args = Args(p=6, px=6, spec=1, lam=1.0, method="ridge")        # a ridge: lambda is the knob tune turns
    with Model.withValidation():                                        # choose on the validation rows
        mod = ARX_D(y, hh, xe, args)
        banner(f"tune: {mod.modelName} on COVID-19 Dataset, knobs {[k.name for k in mod.knobs]}, fit / validation / test = {Model.splitStr(len(y))}")
        hpG, score = mod.tune(verbose=True)                             # the model's own knobs play no part
        lam = hpG["lambda"]
        print(f"best knobs = {hpG}, validation score (mean sMAPE over h = 1..{hh}) = {score:.4f}")
        banner("the lambda landscape depends on the column scales: raw columns vs standardized (scaled = True)")
        print(f"column sd: y {np.std(y):.0f}, {EXO[0]} {np.std(xe[:, 0]):.0f}, {EXO[1]} {np.std(xe[:, 1]):.3f}  "
              f"-- on raw columns lambda = 1 is heavy for {EXO[1]}'s lags and invisible to the others")
        modS = ARX_D(y, hh, xe, args.copy(scaled=True))
        for name, m_ in (("raw", mod), ("scaled", modS)):
            sc = {l: m_.withHP({"lambda": l}).valScore() for l in Knob.LAMBDA.grid}
            print(f"  {name:>6s}: " + "  ".join(f"L{l:g} {v:6.2f}" for l, v in sc.items()))
        banner("the knobs found five ways (raw columns): the search and the rows it is scored on")
        print(f"  grid, 24 validation rows:   {hpG}, score {score:.4f}")
        hpS, sb = mod.tune(search="golden")
        print(f"  golden, 24 validation rows: lambda = {hpS['lambda']:.3g}, score {sb:.4f}   (the grid refined between its neighbours)")
        hpR, sr = mod.tune(search="random", trials=10, seed=7)
        print(f"  random, 10 points:          lambda = {hpR['lambda']:.3g}, score {sr:.4f}   (log-uniform between the grid's ends)")
    with Model.withValidation(ratio=0.4):                               # a wider block: fit 45, score 47 (rollTest IS the
        hp47, s47 = mod.tune()                                          # expanding window, refit every rc rows)
        print(f"  grid, 47 validation rows:   {hp47}, score {s47:.4f}   (fit / validation / test = {Model.splitStr(len(y))})")
    hp5, s5 = mod.tune(folds=5)                                         # k-fold on the 92 training rows (no switch needed)
    print(f"  grid, 5-fold on 92 rows:    {hp5}, score {s5:.4f}   (Bergmeir, Hyndman & Koo: valid for a lag-input model)")
    args = args.copy(lam=lam)
    mod = ARX_D(y, hh, xe, args)                                        # ONE model at the knobs found: In-Sample, TnT
    banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")     # and the selections below (a selection
    mod.inSample_Test()                                                 # never trains or alters its model)
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)

    with Model.withValidation(tuning=True):                             # every candidate set at its own best knobs
        for tech in SelectionTech:
            banner(f"Feature Selection ({tech}, tuning on): {mod.modelName} on COVID-19 Dataset")
            colsArr, rSqArr = mod.selectFeatures(tech)
            best = mod.getBest()
            print(f"{tech}: best step {best.mod.modelName}: k = {len(best.mod_cols)} of n = {len(mod.mcols)}, "
                  f"knobs = {mod.bestHP(best.mod_cols)}, score = {best.qof[best.qk]:.4f}, "
                  f"columns = {[mod.fname[j] for j in best.mod_cols]}")
            mod.reportSelection(colsArr, rSqArr, inSample=False, rc=RC)   # TnT of the selected model at its knobs


def aRX_D_enet_test(y, xe, hh=HH, args=None):
    """The ELASTIC NET next to the ridge (p = px = 6, spec = 1, standardized columns): the ridge's one knob
    (lambda) and the elastic net's two (lambda, l1_ratio: the 35-point grid, then one golden-section sweep per
    knob) tuned on the validation rows by the SAME `tune`, then TnT at the knobs found with the columns the
    L1 part kept at h = 1 -- the continuous selection every hard selection should be measured against (the
    grouping effect keeps correlated lags together); the lasso (l1_ratio fixed at 1) next to them."""
    args = Args(p=6, px=6, spec=1, lam=1.0, scaled=True) if args is None else args
    banner("ridge vs elastic net on COVID-19 (standardized columns): the knobs by tune (grid + golden), then TnT")
    print(f"{'method':<16s} {'knobs':>26s} {'val score':>10s} {'TnT mean sMAPE':>15s} {'nonzero at h = 1':>17s}")
    rows = (("ridge", None, None), ("enet", None, None), ("enet", 1.0, (Knob.LAMBDA,)))
    for method, l1, knobs in rows:
        a = args.copy(method=method) if l1 is None else args.copy(method=method, l1_ratio=l1)
        mod = ARX_D(y, hh, xe, a)
        with Model.withValidation():
            hp2, score = mod.tune(knobs, search="golden")
        mod = mod.withHP(hp2)
        Forecaster.SHOW_QoF, keep = False, Forecaster.SHOW_QoF
        try:
            qt = mod.tnT_Test(RC, verbose=False)
        finally:
            Forecaster.SHOW_QoF = keep
        nz = int(np.sum(np.asarray(mod.params)[1:, 0] != 0.0))
        label = method if l1 is None else f"{method} (l1 {l1:g} fixed)"
        kn = ", ".join(f"{k} {v:.3g}" for k, v in hp2.items())
        print(f"{label:<16s} {kn:>26s} {score:10.3f} {Forecaster.scoreOf(qt, False):15.2f} {nz:17d}   {mod.modelName}")


def aRX_D_diff_test(y, xe, hh=HH, args=None):
    """ARX_D on the DIFFERENCED series (d = 1), an experiment without a base change: the model is fitted on
    z_k = y_{k+1} - y_k (the exogenous variables aligned with z, the non-negativity off: a difference may be
    negative), its TnT forecasts of the differences are integrated back to levels -- from the origin t - 1,
    y_{t+h-1} = y_{t-1} + sum_{j <= h} z-forecast_j, i.e. the cumulative sum along row t - 1 of the forecast
    matrix -- and the levels are diagnosed on the SAME testing rows as the undifferenced model (`diagnoseAll`),
    so the two QoF tables compare directly.  Differencing removes the level's trend and most of the
    collinearity of its lags; whether it forecasts better is what the table says."""
    args = Args(p=6, px=6, spec=0, lam=1.0, method="ridge") if args is None else args
    mod = ARX_D(y, hh, xe, args)
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset (levels)")
    qofLevel = mod.tnT_Test(RC)

    z, xez = np.diff(y), xe[1:]                                          # z_k = y_{k+1} - y_k, xe aligned with z
    modz = ARX_D(z, hh, xez, args.copy(nneg=False))
    banner(f"TnT Forecasts: {modz.modelName} on the DIFFERENCED COVID-19 Dataset (d = 1)")
    modz.tnT_Test(RC)                                                   # its QoF on the differences
    Yf = mod.Yf.copy()                                                  # the level layout: Yf[t, h] = y_{t+h-1} forecast at t - 1
    te = range(mod.tr_size, len(y))
    for t in te:                                                         # row t - 1 of the difference model = origin t - 1
        Yf[t, 1:hh + 1] = np.maximum(y[t - 1] + np.cumsum(modz.Yf[t - 1, 1:hh + 1]), 0.0)
    banner(f"diagnoseAll: {modz.modelName} integrated back to LEVELS on the testing rows {te.start}..{te.stop - 1}")
    qofDiff = mod.diagnoseAll(y, Yf, te)
    print(f"mean sMAPE over h = 1..{hh}: levels {Forecaster.scoreOf(qofLevel, False):.4f}, "
          f"differenced then integrated {Forecaster.scoreOf(qofDiff, False):.4f}")


def aRX_SR_test(y, xe, hh=HH, args=None):
    """ARX_SR (recursive, symbolic regression terms): y^1.5 next to the lags of y and log (1 + xe_j) next to
    each exogenous variable's lags; rescaled by Log1p first (as ScalaTion's tests), so the powers and logs
    stay defined."""
    args = Args(p=6, px=4, spec=0, crx=False) if args is None else args
    fEndo = [PowForm()]                                                  # (y + 0)^1.5
    fExo = [[LogForm()] for _ in range(xe.shape[1])]                     # log (1 + xe_j)
    mod = ARX_SR.rescale(y, hh, xe, args, fEndo, fExo, TransformT.Log1p)
    print(f"columns: {mod.fname}")
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_SR_D_test(y, xe, hh=HH, args=None):
    """ARX_SR_D (direct, SR terms): PowForm and RootForm of y, LogForm of each exogenous variable, rescaled by
    Log1p (aRX_SR_DTest5): In-Sample with the coefficient table (`summary`), then TnT."""
    args = Args(p=6, px=4, spec=0, crx=False) if args is None else args
    fEndo = [PowForm(), RootForm()]                                      # y^1.5 and y^0.5
    fExo = [[LogForm()] for _ in range(xe.shape[1])]                     # log (1 + xe_j)
    mod = ARX_SR_D.rescale(y, hh, xe, args, fEndo, fExo, TransformT.Log1p)
    print(f"columns: {mod.fname}")
    banner(f"In-ST Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test()
    print(mod.summary())                                                 # statistical summary of fit
    banner(f"TnT Forecasts: {mod.modelName} on COVID-19 Dataset")
    mod.tnT_Test(RC)


def aRX_Formula_test(hh=HH):
    """ARX_Formula written in the DSL (aRX_FormulaTest5): fit_D gives the direct ARX_D on the terms, fit the
    recursive ARX for h = 1; with TFORM: Log1p every base column is log1p-transformed on its training slice
    and the forecasts come back in deaths."""
    data = Example_Covid.context(EXO, ["icu", "pRate"])                   # {"y", "icu", "pRate"}
    y, icu, pRate = FormulaVar("y"), FormulaVar("icu"), FormulaVar("pRate")
    f = ARX_Formula(y, y.lags(6, 1) + icu.lags(4, 1) + [log1p(pRate.lagged(1))] + [y.lagged(1) * icu.lagged(1)])
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
    data = Example_Covid.context(EXO, ["icu", "pRate"])
    y, icu, pRate = FormulaVar("y"), FormulaVar("icu"), FormulaVar("pRate")
    f = NARX_Formula(y, y.lags(3, 1) + [sat(icu.lagged(1), half()), ema(pRate.lagged(1), decay())]
                     + trans(y.lagged(1), thresh(), width()) * y.lags(3, 1))
    banner(f"NARX_Formula: {f.formulaStr}")
    print(f"key     = {f.key}\nnParams = {f.nParams}, nFree = {f.nFree}")

    banner("fit_D: ARX_D with the parameters at their start values")
    args = Args(method="ridge", lam=1.0, scaled=True)                    # RidgeRegression.hp("lambda") = 1.0
    mod0 = f.fit_D(data, hh, args)
    print(f"formula = {f.formulaStr}")
    mod0.forecastAll()
    mod0.diagnoseAll()

    banner(f"fitNL_D: NARX_SR_D with the parameters fitted by {OPTIMIZER_NL}")
    mod = f.fitNL_D(data, hh, args)                                      # always ridge (args.method not read)
    print(f"formula = {f.formulaStr}\nparam_nl = {mod.w_nl}, stepCounts = {mod.stepCounts}")
    mod.forecastAll()
    mod.diagnoseAll()

    formula = "y ~= y~(3,1) + log(? + icu~1) + pow(y~1, ?) + sat(pRate~1, ?)"
    mod2 = NARX_SR_D.fromTerms(formula, data, hh, args)
    banner(f"TnT Forecasts: {mod2.modelName} on {mod2.formulaStr}")
    mod2.tnT_Test(RC)


def nARX_SR_D_test(y, xe, hh=HH):
    """NARX_SR_D from transform TYPES (nARX_SR_DTest5): a ROOT transform (y + s)^r of the y lags and of each
    exogenous variable's lags, rescaled by MinMax to [1, 5]; first with the DEFAULT nonlinear parameters
    (tune = False, the ARX_SR_D model with those transforms), then with the parameters fitted by
    `OPTIMIZER_NL`; Backward feature selection at h = 1 on the fitted model, the selected model in TnT."""
    args = Args()
    args.p, args.px, args.spec = 6, 4, 1
    args.lam = 6.0                                                      # RidgeRegression.hp("lambda") = 6.0
    fEndo = [TransformT.Root]                                           # transform types for the endogenous variable
    fExo = [[TransformT.Root], [TransformT.Root]]                       # and for each exogenous variable

    banner("NARX_SR_D with the DEFAULT nonlinear parameters")
    mod0 = NARX_SR_D.rescale(y, hh, xe, args.copy(tune=False), fEndo, fExo)
    mod0.inSample_Test()                                                # In-sample Testing
    mod0.tnT_Test(rc=2)                                                 # TnT with Rolling Validation

    banner(f"NARX_SR_D with the nonlinear parameters fitted by {OPTIMIZER_NL}")
    mod = NARX_SR_D.rescale(y, hh, xe, args, fEndo, fExo)
    mod.inSample_Test()
    mod.tnT_Test(rc=2)

    banner("nonlinear parameters (shift, root) of f0 (y), g0,0 (icu_patients), g1,0 (positive_rate)")
    print(f"default: {mod0.parameter_nl}")
    print(f"fitted:  {mod.parameter_nl}")

    banner("Feature Selection (Backward, horizon 1 decides) on the model with the fitted nonlinear parameters")
    FeatureSelection.fullset_FS = False                                 # TnT test: select on the training rows
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.Backward, "none", h=1)
    mod.reportSelection(colsArr, rSqArr, inSample=False, rc=2)          # TnT of the selected multi-horizon model


def optimizers_test(y, xe, hh=HH):
    """The NONLINEAR OPTIMIZERS of `OptimizerNL` (nARX_SR_DTest8) on a POWER term of y, (y + s)^p, and a LOG
    term of each exogenous variable, log (xe + s), lambda = 6: for each optimizer the fitted parameters, the
    fitting time and TnT with rolling validation -- they minimize the same objective, so they agree on the
    parameters when they converge and differ in cost."""
    args = Args(p=6, px=4, spec=1, lam=6.0)
    fEndo = [TransformT.Pow]                                            # (y + s)^p: shift s and power p are tuned
    fExo = [[TransformT.Log], [TransformT.Log]]                         # log (xe + s): shift s is tuned
    w_nl, secs = {}, {}
    for opt in OptimizerNL:
        banner(f"NARX_SR_D with the nonlinear parameters fitted by {opt}")
        t0 = time.time()
        mod = NARX_SR_D.rescale(y, hh, xe, args.copy(optimizer=opt), fEndo, fExo)
        secs[opt] = time.time() - t0
        w_nl[opt] = mod.parameter_nl
        mod.tnT_Test(rc=2)
    banner("nonlinear parameters: (shift, power) of f0 (y); (shift, scale) of g0,0 (icu_patients), g1,0 (positive_rate)")
    for opt in OptimizerNL:
        print(f"{opt}: fitted {w_nl[opt]}, fitting time {secs[opt]:.4f} s")


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
    args = Args(p=6, px=4, spec=0) if args is None else args
    model = ARX_SR.rescale(y, hh, xe, args, fExo=[[] for _ in range(xe.shape[1])], tFormT=TransformT.Log1p)
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


def run_test():
    """The one-object style of an experiment script: the dataset, the exogenous columns, the model, its
    hyper-parameters, the horizon and the test in one Args (every name and value checked when set), then
    `run (args)`, which loads the data, builds the model and tests it -- the same ARX_D as `aRX_D_test`."""
    args = Args()
    args.dataset = "Covid-19"
    args.exo = EXO                                                      # icu_patients, positive_rate
    args.model = "ARX_D"
    args.p = 6
    args.px = 6
    args.spec = 0
    args.hh = HH
    args.rc = RC
    args.validation = "both"                                            # In-Sample, then TnT
    args.help(ARX_D)                                                    # the settings ARX_D reads
    mod = run(args)
    print(f"{mod.modelName}: hparameter = {mod.hparameter}")


SCENARIOS = {
    "rw": lambda y, xe: randomWalk_test(y),
    "ar": lambda y, xe: aR_test(y),
    "arima": lambda y, xe: aRIMA_test(y),
    "arx": lambda y, xe: aRX_test(y, xe),
    "arx_d": lambda y, xe: aRX_D_test(y, xe),
    "arx_d2": lambda y, xe: aRX_D_test2(y, xe),
    "enet": lambda y, xe: aRX_D_enet_test(y, xe),
    "diff": lambda y, xe: aRX_D_diff_test(y, xe),
    "arx_sr": lambda y, xe: aRX_SR_test(y, xe),
    "arx_sr_d": lambda y, xe: aRX_SR_D_test(y, xe),
    "formula": lambda y, xe: aRX_Formula_test(),
    "narx": lambda y, xe: nARX_Formula_test(),
    "narx_sr": lambda y, xe: nARX_SR_D_test(y, xe),
    "optimizers": lambda y, xe: optimizers_test(y, xe),
    "mhattn": lambda y, xe: mHAttn_test(y, xe),
    "run": lambda y, xe: run_test(),
}


def main(argv):
    xe, y = load_covid()
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
