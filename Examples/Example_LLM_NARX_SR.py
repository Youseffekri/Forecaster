"""
Example: the LLM-NARX-SR core on the Covid-19, ETTh1 and GEFCom2014 datasets

Shows the formula layer end to end -- parse -> fit -> score -> rolling test -- without any
LLM: hand-written formulas with free nonlinear parameters, the selection score J, the
rejection rules, the three optimizers of the nonlinear parameters, and the V-shaped
load-temperature relation of GEFCom2014 recovered by a free hinge threshold.

Run from the repository root:   python Examples/Example_LLM_NARX_SR.py [covid|etth1|gef14|optimizers|all]

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
License: MIT License
Note: AI assisted code
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from modeling.forecasting.narx_sr_d import NARX_SR_D
from modeling.forecasting.llm_narx_sr.datasets import Datasets, Example_GEFCom
from modeling.forecasting.llm_narx_sr.formula_scorer import FormulaScorer, qof_table

np.set_printoptions(linewidth=200, precision=4, suppress=True)


def banner(s: str) -> None:
    print("\n" + "-" * (len(s) + 4) + f"\n| {s} |\n" + "-" * (len(s) + 4))


def score_table(sc: FormulaScorer, formulas, tnt: bool = True) -> None:
    """Scores each formula and prints J, the fit / validation sMAPE and (optionally) the rolling TnT sMAPE."""
    print(f"{'J':>9s}  {'fit':>6s}  {'val':>6s}  {'TnT':>6s}  n_free  formula")
    for s in formulas:
        c = sc.score(s)
        if c.failed:
            print(f"{'rejected':>9s}  {'':>6s}  {'':>6s}  {'':>6s}  {c.n_free:6d}  {c.formula_str}{c.why}")
            continue
        t = ""
        if tnt:
            _, qt = sc.tnt(c.formula.fresh())
            t = f"{np.mean([q['SMAPE'] for q in qt]):6.2f}"
        print(f"{c.score:9.3f}  {c.smape:6.2f}  {c.smape_val:6.2f}  {t:>6s}  {c.n_free:6d}  {c.formula_str}")


def covid_test() -> None:
    """Covid-19 weekly deaths with icu patients and the positivity rate (116 weeks, hh = 6)."""
    banner("Covid-19 (icu, pRate): formulas with and without free parameters")
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6)
    print(f"rows {sc.m}: training {sc.tr_size} (fit {sc.fit_size} + validation blocks {[(b.start, b.stop) for b in sc.blocks]}), "
          f"testing {sc.te_size}; periodogram periods {np.round(1 / np.array(sc.freqs), 1)} weeks")
    score_table(sc, [
        cfg.baseline_formula_str(6),                           # the plain-ARX baseline: 18 lag terms
        "y ~= y~(3,1) + icu~1 + pRate~1",
        "y ~= y~(2,1) + log1p(icu~1) + pRate~1",
        "y ~= y~(2,1) + icu~1 + sat(pRate~1, ?)",
        "y ~= y~(2,1) + ema(icu~1, ?) + pRate~1",
        "y ~= y~1 + pow(y~1, ?) + icu~1",
        "y ~= y~(3,1) + log(? + icu~1) + pRate~1",
        "y ~= y~1 + hinge(icu~1, ?) + ema(pRate~1, ?) + dq(y~2, ?)",
        "y ~= y~(2,1) + icu~(2,1) + pRate~(2,1) TFORM: Log1p",
    ])
    banner("the report of one candidate")
    print(sc.report(sc.score("y ~= y~(3,1) + icu~1 + pRate~1")))


def optimizer_test() -> None:
    """The three optimizers of the nonlinear parameters on one Covid-19 formula."""
    banner("optimizers of the nonlinear parameters: varpro, lbfgsb, joint (Covid-19)")
    data = Datasets.COVID2.load()
    formula = "y ~= y~(3,1) + log(? + icu~1) + pow(y~1, ?) + sat(pRate~1, ?)"
    print(f"formula: {formula}\n{'optimizer':10s} {'scaled':6s} {'objective':>16s} {'evals':>6s} {'time':>7s}  fitted")
    for scaled in (False, True):
        for opt in NARX_SR_D.OPTIMIZERS:
            t0 = time.time()
            m = NARX_SR_D.from_formula(formula, data, hh=6, lam=1.0, scaled=scaled, optimizer=opt, max_nfev=600)
            print(f"{opt:10s} {str(scaled):6s} {m.steps['objective']:16.2f} {m.steps['nfev']:6d} {time.time() - t0:6.2f}s  "
                  f"{m.formula_str.split('+ log')[1]}")
    print("VarPro eliminates the coefficients (6 horizons x 7 columns here) and searches the 3 parameters only;")
    print("the joint fit searches all 45 unknowns at once -- the same optimum when the columns are standardized, "
          "far more evaluations when they are not.")


def gefcom_test() -> None:
    """GEFCom2014 daily peak load: the V-shaped load-temperature relation and a free hinge threshold."""
    banner("GEFCom2014 daily peak load (temp, tmax): the V shape")
    print("mean peak load per 10 F band, 20-30 .. 80-90 F:", [round(v) for v in Example_GEFCom.v_table()])
    data = Datasets.GEF14.load()
    sc = FormulaScorer(data, hh=6, hp={"maxLag": 7, "skip": 7})      # the weekly cycle
    print(f"rows {sc.m}: training {sc.tr_size}, testing {sc.te_size}; periodogram periods {np.round(1 / np.array(sc.freqs), 1)} days")
    for opt in NARX_SR_D.OPTIMIZERS:
        m = NARX_SR_D.from_formula("y ~= y~(7,1) + hinge(temp~1, ?)", data, hh=6, lam=1.0, scaled=True,
                                   optimizer=opt, max_lag=7)
        print(f"{opt:7s}: hinge threshold = {m.formula.params()[0].value:6.2f} F   "
              f"({m.steps['nfev']} evaluations) -- the cooling point of the V")
    score_table(sc, [
        Datasets.GEF14.baseline_formula_str(7),                # 21 lag terms
        "y ~= y~(7,1) + hinge(temp~1, ?)",
        "y ~= y~(7,1) + hinge(temp~1, ?) + wave(1)",
        "y ~= y~(7,1) + temp~1 + hinge(temp~1, ?)",            # rejected: collinear with temp~1
        "y ~= y~(7,1) + hinge(temp~1, ?) + hinge(tmax~1, ?)",  # rejected: the two hinges are collinear
        "y ~= y~1 + y~7 + hinge(temp~1, ?) + wave(1)",
    ])


def etth1_test() -> None:
    """ETTh1 first year: hourly transformer oil temperature with the six loads (nneg = False: OT goes negative)."""
    banner("ETTh1 first year (8,760 hours): oil temperature from the loads")
    data = Datasets.ETTH1_1Y.load()
    sc = FormulaScorer(data, hh=6, hp={"nneg": False})
    print(f"rows {sc.m}: training {sc.tr_size}, testing {sc.te_size}; periodogram periods {np.round(1 / np.array(sc.freqs), 1)} hours")
    score_table(sc, [
        "y ~= y~(6,1) + hufl~(6,1) + lull~(6,1)",
        "y ~= y~(6,1) + hufl~1 + lull~1",
        "y ~= y~(2,1) + ema(hufl~1, ?) + lull~1",
        "y ~= y~(2,1) + sat(hufl~1, ?) + lull~1",
        "y ~= y~(2,1) + hufl~1 + lull~1 + wave(1)",
    ])


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("covid", "all"):
        covid_test()
    if which in ("optimizers", "all"):
        optimizer_test()
    if which in ("gef14", "gefcom", "all"):
        gefcom_test()
    if which in ("etth1", "all"):
        etth1_test()
    print(f"\ndone in {time.time() - t0:.1f} s")
