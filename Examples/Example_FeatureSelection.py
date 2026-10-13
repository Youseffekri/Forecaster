"""
Example: Feature Selection of Direct Forecasting Models (ScalaTion's aRX_DTest5 .. aRX_DTest9,
subsetSearchTest and cEM_SelectorTest)

Forecasting weekly COVID-19 deaths (data/covid_19_weekly.csv, the first 116 weeks, exogenous
variables icu_patients and positive_rate -- ScalaTion's `Example_Covid.BEST2_EXO`) with an `ARX_D`
and selecting its columns by the techniques of `SelectionTech`: Forward, Backward, Stepwise, Beam
and CEM.  One scenario per `*_test` function, in the shape of the Scala tests: build the model,
test it (the reference), select, report and test the selected model(s).

    aRX_DTest5_test   selectFeatures (h = 1 decides)     ARX_D.rescale (Log1p)  In-Sample, the multi-horizon model
    aRX_DTest6_test   selectFeaturesAtH (h = 1)          ARX_D.rescale (Log1p)  TnT, the h = 1 model
    aRX_DTest7_test   selectFeatures (mean J)            ARX_D                  In-Sample, the multi-horizon model
    aRX_DTest8_test   selectFeatures (h = 1 decides)     ARX_D.rescale (Log1p)  TnT, the multi-horizon model
    aRX_DTest9_test   selectFeaturesPerH (one set per h) ARX_D                  TnT, one model per horizon
    subsetSearch_test  every technique of `SubsetSearch` on a synthetic set function (no model)
    cEM_Selector_test  `CEM_Selector` on a synthetic regression with a planted column set

Selection sees the rows the test allows it to see: `FeatureSelection.fullset_FS = True` for the
in-sample tests, False for the TnT tests (select on the training rows, test on the rest).  Tests 5
and 6 use p = 10, q = 5, spec = 2 (a rescaled model); tests 7 to 9 use p = 6, q = 6, spec = 1 (19
columns), so their tables compare with one another.  With `method="ridge"` and lambda = 1 the
regression is ScalaTion's `CenteredRidgeMV`, and `aRX_DTest8_test` reproduces the Scala log
(/logs/modeling_forecasting_aRX_DTest8.log) to the printed digits.

Run from the repository root:  python Examples/Example_FeatureSelection.py [5|6|7|8|9|search|cem|all]

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

from Util.tools import banner, set_str, vector_str
from Util.transform import TransformT
from modeling.cem_selector import CEM_Selector, Random
from modeling.feature_selection import BestStep, FeatureSelection, SelectionTech
from modeling.fit import N_QoF, QoF, diagnose
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid
from modeling.subset_search import SubsetSearch

np.set_printoptions(linewidth=200, precision=4, suppress=True)

HH = 6                                                                  # maximum forecasting horizon


def load_covid():
    """(xe, y): the two best exogenous variables and the weekly deaths, the first 116 weeks (ScalaTion's clip (loadData (BEST2_EXO)))."""
    y, xe = Example_Covid.loadData(Example_Covid.BEST2_EXO)
    return xe, y


# -- the feature selection of a direct model (aRX_DTest5 .. aRX_DTest9) ----------------------------

def aRX_DTest5_test(techs=(SelectionTech.CEM,)):
    """
    In-Sample Testing: Feature Selection AT HORIZON 1 (`selectFeatures (h = 1)`) on a RESCALED model
    (Log1p); the selected multi-horizon model is tested in-sample.  The single-horizon model at
    h = 1 is printed first as the reference for reading the selection traces in ORIGINAL space.
    """
    xe, y = load_covid()
    hparam = {"p": 10, "px": 5, "spec": 2, "lambda": 1.0}               # endo lags, exo lags, trend spec, regularization
    FeatureSelection.fullset_FS = True                                 # in-sample test: select on all the rows

    mod = ARX_D.rescale(hparam, y, HH, xe, method="ridge", tFormT=TransformT.Log1p)   # create model for time series data
    mod.inSample_Test()                                                # the full model: the reference

    banner(f"reference for the selection trace: {mod.modelName} at h = 1 as a single-horizon ARX")
    mod.getModel_h(h=1).inSample_Test()                                # in-sample QoF of the h = 1 model

    for tech in techs:
        banner(f"Feature Selection Technique: {tech} (at h = 1)")
        colsArr, rSqArr = mod.selectFeatures(tech, "none", h=1)       # horizon 1 decides, its columns shared by all horizons
        mod.reportSelection(colsArr, rSqArr, tnt=False)               # print; In-sample Test of the selected model


def aRX_DTest6_test(techs=(SelectionTech.CEM,)):
    """
    Train-n-Test with Rolling Validation: Feature Selection AT HORIZON 1 (`selectFeaturesAtH`) on a
    RESCALED model (Log1p); what is tested is the single-horizon model the selection returns (an
    `ARX` at h = 1, so its QoF has one horizon).
    """
    xe, y = load_covid()
    hparam = {"p": 10, "px": 5, "spec": 2, "lambda": 1.0}
    FeatureSelection.fullset_FS = False                                # TnT test: select on the training rows

    mod = ARX_D.rescale(hparam, y, HH, xe, method="ridge", tFormT=TransformT.Log1p)
    mod.getModel().tnT_Test(showYf=False)                              # the full model: the reference -- a COPY,
                                                                       # as `tnT_Test` sets skip = 0 and the
                                                                       # selection below must keep the burn-in
    for tech in techs:
        banner(f"Feature Selection Technique: {tech} (at h = 1)")
        cols, rSq = mod.selectFeaturesAtH(tech, "none")                # R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection([cols], [rSq], h=1, inSample=False)       # print; TnT of the selected h = 1 model


def aRX_DTest7_test(techs=(SelectionTech.Backward,)):
    """
    In-Sample Testing: Feature Selection over ALL HORIZONS JOINTLY (`selectFeatures`, h = 0): every
    candidate column set is fit as ONE multi-horizon model and scored by J, the mean FPE-form sMAPE
    over the horizons (`aggFPE`); the selected model is tested in-sample against the full model.
    """
    xe, y = load_covid()
    hparam = {"p": 6, "px": 6, "spec": 1, "lambda": 1.0}
    FeatureSelection.fullset_FS = True

    mod = ARX_D(hparam, y, HH, xe, method="ridge")                       # create model for time series data
    mod.inSample_Test()                                                # the full model: the reference

    for tech in techs:
        banner(f"Feature Selection Technique: {tech} (all horizons, J = mean FPE-form sMAPE)")
        colsArr, rSqArr = mod.selectFeatures(tech)                     # one multi-horizon model per candidate set
        mod.reportSelection(colsArr, rSqArr, tnt=False)               # print; In-sample Test of the selected model


def aRX_DTest8_test(techs=(SelectionTech.CEM,)):
    """
    Train-n-Test with Rolling Validation: the single-horizon model at h = 1 decides the ONE column
    set of the multi-horizon model (`selectFeatures (h = 1)`) on the training rows, and the selected
    multi-horizon model is tested with TnT against the full model (ScalaTion's aRX_DTest8, whose
    log this reproduces).
    """
    xe, y = load_covid()
    hparam = {"p": 6, "px": 6, "spec": 1, "lambda": 1.0}
    FeatureSelection.fullset_FS = False

    mod = ARX_D.rescale(hparam, y, HH, xe, method="ridge", tFormT=TransformT.Log1p)   # the rescaled version
    mod.getModel().tnT_Test(showYf=False)                              # the full model: the reference -- a COPY

    for tech in techs:
        banner(f"Feature Selection Technique: {tech} (all horizons, J = mean FPE-form sMAPE)")
        colsArr, rSqArr = mod.selectFeatures(tech, h=1)               # the single-horizon model at h = 1 decides
        mod.reportSelection(colsArr, rSqArr, inSample=False)          # print; TnT of the selected multi-horizon model


def aRX_DTest9_test(techs=(SelectionTech.Backward,)):
    """
    Train-n-Test with Rolling Validation: Feature Selection PER HORIZON (`selectFeaturesPerH`): one
    column set and one single-horizon model per horizon, each rolling-validated on the testing rows.
    The table at the end lists, per horizon, how many columns were kept, that model's TnT sMAPE and
    its features.
    """
    xe, y = load_covid()
    hparam = {"p": 6, "px": 6, "spec": 1, "lambda": 1.0}
    FeatureSelection.fullset_FS = False

    mod = ARX_D(hparam, y, HH, xe, method="ridge")
    mod.getModel().tnT_Test(showYf=False)                              # the full model: the reference -- a COPY

    for tech in techs:
        banner(f"Feature Selection Technique: {tech} (per horizon)")
        colsArr, rSqArr = mod.selectFeaturesPerH(tech)                 # one column set per horizon
        _, _, ftMat = mod.reportSelection(colsArr, rSqArr, inSample=False)   # print; TnT of each horizon's model

        banner(f"{mod.modelName}: per-horizon selection ({tech}), TnT sMAPE on the testing rows")
        print("  h   k   TnT smape   features")
        for h in range(1, HH + 1):
            print(f"{h:3d} {len(colsArr[h - 1]):3d} {ftMat[h - 1, QoF.smape.ordinal]:11.3f}   "
                  + ", ".join(mod.fname[j] for j in sorted(colsArr[h - 1])))
        kept = set.intersection(*(set(c) for c in colsArr))
        print("kept at every horizon: " + ", ".join(mod.fname[j] for j in sorted(kept)))


# -- the algorithms alone (subsetSearchTest, cEM_SelectorTest) -----------------------------------------

def subsetSearch_test():
    """
    Every technique of `SubsetSearch` on a synthetic set function that needs no model: with a
    planted set P over columns 0 until 8 (column 0 fixed), the criterion of a set S is |S delta P|
    plus a small tie-breaker, to be MINIMIZED (qk = smapeC), so every technique must return exactly
    P (ScalaTion's subsetSearchTest).
    """
    qk = QoF.smapeC.ordinal                                            # minimized criterion

    def evalFor(planted):
        def ev(cols):
            s = set(cols)
            qof = np.zeros(N_QoF)
            qof[qk] = len(s - planted) + len(planted - s) + 0.001 * len(s)
            return BestStep(-1, qof, None, cols, qk, qof[qk])
        return ev

    n, planted = 8, {0, 1, 3, 6}
    ev = evalFor(planted)

    banner("forward")
    pf = SubsetSearch.forward(n, ev, qk=qk)
    print(f"forward:  cols = {set_str(pf.cols)}, best = {pf.best.qof[qk]}, steps = {len(pf.steps)}, nEval = {pf.nEval}")
    assert set(pf.cols) == planted and len(pf.steps) == n and pf.nEval == 29
    banner("backward")
    pb = SubsetSearch.backward(n, ev, qk=qk)
    print(f"backward: cols = {set_str(pb.cols)}, best = {pb.best.qof[qk]}, steps = {len(pb.steps)}, nEval = {pb.nEval}")
    assert set(pb.cols) == planted and len(pb.steps) == n
    banner("stepwise")
    ps = SubsetSearch.stepwise(n, ev, slack=0.0, qk=qk)
    print(f"stepwise: cols = {set_str(ps.cols)}, best = {ps.best.qof[qk]}, steps = {len(ps.steps)}, nEval = {ps.nEval}")
    assert set(ps.cols) == planted and len(ps.steps) == 3
    banner("beam")
    pm = SubsetSearch.beam(n, ev, bk=3, qk=qk)
    print(f"beam:     cols = {set_str(pm.cols)}, best = {pm.best.qof[qk]}, steps = {len(pm.steps)}, nEval = {pm.nEval}")
    assert set(pm.cols) == planted
    banner("cem")
    pc = SubsetSearch.cem(n, ev, stream=0, qk=qk)
    print(f"cem:      cols = {set_str(pc.cols)}, best = {pc.best.qof[qk]}, rounds = {len(pc.steps)}, nEval = {pc.nEval}")
    print(f"cem:      probs = {vector_str(pc.probs)}")
    assert set(pc.cols) == planted and pc.probs is not None and len(pc.probs) == n

    banner("run: every technique through the dispatcher agrees")
    for tech in SelectionTech:
        p = SubsetSearch.run(tech, n, ev, qk=qk)
        print(f"{tech}: cols = {set_str(p.cols)}")
        assert set(p.cols) == planted

    banner("units: a wave pair (columns 4, 5) moves as a whole")
    pair = {4, 5}
    unitOf = lambda j: pair if j in pair else {j}
    ev2 = evalFor({0, 2, 4, 5})
    pf2 = SubsetSearch.forward(n, ev2, units=unitOf, qk=qk)
    pb2 = SubsetSearch.backward(n, ev2, units=unitOf, qk=qk)
    print(f"forward with units:  cols = {set_str(pf2.cols)}, steps = {[len(s.mod_cols) for s in pf2.steps]}")
    print(f"backward with units: cols = {set_str(pb2.cols)}, steps = {[len(s.mod_cols) for s in pb2.steps]}")
    assert set(pf2.cols) == {0, 2, 4, 5} and set(pb2.cols) == {0, 2, 4, 5}

    banner("a failed evaluation is skipped, not chosen")
    ev3 = lambda cols: BestStep(-1, None, None, cols, qk) if 7 in cols else ev(cols)
    pf3 = SubsetSearch.forward(n, ev3, qk=qk)
    print(f"forward with column 7 failing: cols = {set_str(pf3.cols)}, steps = {len(pf3.steps)}")
    assert set(pf3.cols) == planted and all(7 not in s.mod_cols for s in pf3.steps)
    print("subsetSearch_test: passed")


def cEM_Selector_test():
    """
    `CEM_Selector` on a synthetic regression: 12 candidate columns of which 3 (plus the intercept)
    generate y, compared by BIC; the selector must recover exactly the true set, its inclusion
    probabilities must separate true from noise columns, the same stream must reproduce the same
    run, and a prior favouring the true columns must also recover them (ScalaTion's cEM_SelectorTest).
    """
    rk = QoF.bic.ordinal                                               # compare models by BIC
    m, n = 120, 13                                                     # rows, columns (0 = intercept)
    rn = Random(5)
    x = np.zeros((m, n))
    for i in range(m):
        x[i, 0] = 1.0
        for j in range(1, n):
            x[i, j] = rn.gen() * 10.0
    truth = {0, 2, 5, 9}
    y = np.array([3.0 + 2.0 * x[i, 2] - 1.5 * x[i, 5] + 0.8 * x[i, 9] + 6.0 * (rn.gen() - 0.5) for i in range(m)])

    def evalCols(cols):                                                # build, train and test a regression on cols
        xc = x[:, cols]
        b = np.linalg.lstsq(xc, y, rcond=None)[0]
        qof = diagnose(y, xc @ b, dfr=len(cols) - 1, df=m - len(cols))   # column 0 is the explicit intercept
        return BestStep(-1, qof, b, cols, rk, qof[rk])

    banner("CEM over 12 candidate columns (intercept fixed), criterion BIC")
    cem = CEM_Selector(n, evalCols, fixed=[0], stream=0, qk=rk)
    best = cem.run()
    print(f"sample size per round = {cem.nS}, elite fraction = {cem.rho}")
    print(f"best cols = {set_str(best.mod_cols)}, BIC = {best.qof[rk]}, evaluated = {cem.nEvaluated}, rounds = {len(cem.bestByRound)}")
    print(f"inclusion probabilities = {vector_str(cem.probs)}")
    assert set(best.mod_cols) == truth, f"expected exactly the true columns {truth}, got {best.mod_cols}"
    pr = cem.probs
    pT = min(pr[j] for j in truth if j != 0)
    pN = max(pr[j] for j in range(1, n) if j not in truth)
    print(f"min p (true columns) = {pT:.3f}, max p (noise columns) = {pN:.3f}")
    assert pT > pN

    banner("reproducibility: the same stream gives the same result")
    again = CEM_Selector(n, evalCols, fixed=[0], stream=0, qk=rk)
    best2 = again.run()
    assert set(best2.mod_cols) == set(best.mod_cols) and again.nEvaluated == cem.nEvaluated

    banner("a prior that favours the true columns")
    prior = np.array([0.8 if j in truth else 0.2 for j in range(n)])
    cemP = CEM_Selector(n, evalCols, fixed=[0], prior=prior, stream=0, qk=rk)
    bestP = cemP.run()
    print(f"with prior: best cols = {set_str(bestP.mod_cols)}, evaluated = {cemP.nEvaluated} (uniform prior: {cem.nEvaluated})")
    assert set(bestP.mod_cols) == truth
    print("cEM_Selector_test: passed")


TESTS = {"5": aRX_DTest5_test, "6": aRX_DTest6_test, "7": aRX_DTest7_test, "8": aRX_DTest8_test,
         "9": aRX_DTest9_test, "search": subsetSearch_test, "cem": cEM_Selector_test}

if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["8"]
    if which == ["all"]:
        which = list(TESTS)
    for w in which:
        TESTS[w]()
