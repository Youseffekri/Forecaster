"""
Example: LLM-NARX-SR on the Covid-19, ETTh1 and GEFCom2014 datasets -- the formula layer and the search layer

Part 1 (the core: covid | optimizers | gef14 | etth1) shows the formula layer end to end -- parse ->
fit -> score -> rolling test -- without any LLM: hand-written formulas with free nonlinear parameters,
the selection score J, the rejection rules, the three optimizers of the nonlinear parameters, and the
V-shaped load-temperature relation of GEFCom2014 recovered by a free hinge threshold.

Part 2 (the search layer: baseline | genetic | random | mockllm | grid) runs the arms of the paper
offline: the four classical anchors (`BaselineArm`), the genetic and the random proposer through the
search loop (`PipelineArm`) on Covid-19 and on a slice of GEFCom2014, the LLM arms with a scripted
`MockLLMClient` (no key, no network), and the harness's small zero-LLM grid (`LLM_NARX_SR.runGrid`)
with its report, significance and plausibility tables.

Run from the repository root:
    python Examples/Example_LLM_NARX_SR.py [covid|optimizers|gef14|etth1|baseline|genetic|random|mockllm|grid|all]

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
License: MIT License
Note: AI assisted code
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from modeling.feature_selection import SelectionTech
from modeling.llm_client import MockLLMClient
from modeling.forecasting.nonlinear import narx_sr_d
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D, OptimizerNL
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import BaselineArm
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Datasets, Example_GEFCom
from modeling.forecasting.nonlinear.llm_narx_sr.eda4ts import EDA4TS
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.llm_narx_sr import LLM_NARX_SR
from modeling.forecasting.nonlinear.llm_narx_sr.pipeline_arm import PipelineArm
from modeling.forecasting.nonlinear.llm_narx_sr.plausibility_checker import PlausibilityChecker
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import ArmSpec, Method
from modeling.forecasting.nonlinear.llm_narx_sr.significance import Significance

np.set_printoptions(linewidth=200, precision=4, suppress=True)

#: The budgets of the search scenarios (small: the example runs in seconds).
SMALL = {"maxRounds": 6, "patience": 3, "cemSample": 60, "cemIter": 5}


def banner(s: str) -> None:
    print("\n" + "-" * (len(s) + 4) + f"\n| {s} |\n" + "-" * (len(s) + 4))


def score_table(sc: FormulaScorer, formulas, tnt: bool = True) -> None:
    """Scores each formula and prints J, the fit / validation sMAPE and (optionally) the rolling TnT sMAPE."""
    print(f"{'J':>9s}  {'fit':>6s}  {'val':>6s}  {'TnT':>6s}  nFree  formula")
    for s in formulas:
        c = sc.score(s)
        if c.failed:
            print(f"{'rejected':>9s}  {'':>6s}  {'':>6s}  {'':>6s}  {c.nFree:6d}  {c.formulaStr}{c.why}")
            continue
        t = ""
        if tnt:
            _, qt = sc.tnt(c.formula.fresh())
            t = f"{np.mean([q['smape'] for q in qt]):6.2f}"
        print(f"{c.score:9.3f}  {c.smape:6.2f}  {c.smapeVal:6.2f}  {t:>6s}  {c.nFree:6d}  {c.formulaStr}")


def run_table(runs) -> None:
    """One line per run: the arm, its best equation, the scores and the budget."""
    print(f"{'arm':<22s} {'score':>7s} {'val':>7s} {'TnT':>7s} {'terms':>5s} {'fits':>6s}  best equation")
    for r in runs:
        print(f"{r.name:<22s} {r.best.score:7.3f} {r.valSmape:7.3f} {r.testSmape:7.3f} {r.best.nTerms:5d} {r.nEval:6d}  {r.best.formulaStr}")


# ---------------------------------------------------------------------------------------------
#  Part 1: the formula layer
# ---------------------------------------------------------------------------------------------

def covid_test() -> None:
    """Covid-19 weekly deaths with icu patients and the positivity rate (116 weeks, hh = 6)."""
    banner("Covid-19 (icu, pRate): formulas with and without free parameters")
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6)
    print(f"rows {sc.m}: training {sc.trSize} (fit {sc.fitSize} + validation blocks {[(b.start, b.stop) for b in sc.blocks]}), "
          f"testing {sc.teSize}; periodogram periods {np.round(1 / np.array(sc.freqs), 1)} weeks")
    score_table(sc, [
        cfg.baselineFormulaStr(6),                             # the plain-ARX baseline: 18 lag terms
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
    """The three optimizers of `OptimizerNL` on one Covid-19 formula (the counterpart of nARX_SR_DTest8):
    the fitted parameters, the evaluations and the fitting time; the objective is in the debug line
    of `NARX_SR_D.fitNL` (`narx_sr_d.DEBUG = True`)."""
    banner("optimizers of the nonlinear parameters: OptimizerNL.VarPro, LBFGS_B, Joint (Covid-19)")
    data = Datasets.COVID2.load()
    formula = "y ~= y~(3,1) + log(? + icu~1) + pow(y~1, ?) + sat(pRate~1, ?)"
    narx_sr_d.DEBUG = True                                                 # fitNL prints a, the objective and the steps
    print(f"formula: {formula}\n{'optimizer':10s} {'scaled':6s} {'evals':>6s} {'time':>7s}  fitted")
    for scaled in (False, True):
        for opt in OptimizerNL:
            t0 = time.time()
            m = NARX_SR_D.fromTerms(formula, data, 6, {"lambda": 1.0}, scaled=scaled, optimizer=opt, max_nfev=600)
            print(f"{str(opt):10s} {str(scaled):6s} {m.stepCounts['nfev']:6d} {time.time() - t0:6.2f}s  "
                  f"{m.formulaStr.split('+ log')[1]}")
    narx_sr_d.DEBUG = False
    print("VarPro eliminates the coefficients (6 horizons x 7 columns here) and searches the 3 parameters only;")
    print("the joint fit searches all 45 unknowns at once -- the same optimum when the columns are standardized, "
          "far more evaluations when they are not.")


def gefcom_test() -> None:
    """GEFCom2014 daily peak load: the V-shaped load-temperature relation and a free hinge threshold."""
    banner("GEFCom2014 daily peak load (temp, tmax): the V shape")
    print("mean peak load per 10 F band, 20-30 .. 80-90 F:", [round(v) for v in Example_GEFCom.v_table()])
    data = Datasets.GEF14.load()
    sc = FormulaScorer(data, hh=6, hp={"maxLag": 7, "skip": 7})      # the weekly cycle
    print(f"rows {sc.m}: training {sc.trSize}, testing {sc.teSize}; periodogram periods {np.round(1 / np.array(sc.freqs), 1)} days")
    for opt in OptimizerNL:
        m = NARX_SR_D.fromTerms("y ~= y~(7,1) + hinge(temp~1, ?)", data, 6, {"lambda": 1.0}, scaled=True, optimizer=opt, maxLag=7)
        print(f"{str(opt):7s}: hinge threshold = {m.formula.params()[0].value:6.2f} F   "
              f"({m.stepCounts['nfev']} evaluations) -- the cooling point of the V")
    score_table(sc, [
        Datasets.GEF14.baselineFormulaStr(7),                  # 21 lag terms: TnT 12.78, the rev3.3 ScalaTion number
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
    print(f"rows {sc.m}: training {sc.trSize}, testing {sc.teSize}; periodogram periods {np.round(1 / np.array(sc.freqs), 1)} hours")
    print(f"the screening's shortlist of the six loads: {Datasets.ETTH1_1Y.shortlist(sc)}")
    score_table(sc, [
        "y ~= y~(6,1) + hufl~(6,1) + lull~(6,1)",
        "y ~= y~(6,1) + hufl~1 + lull~1",
        "y ~= y~(2,1) + ema(hufl~1, ?) + lull~1",
        "y ~= y~(2,1) + sat(hufl~1, ?) + lull~1",
        "y ~= y~(2,1) + hufl~1 + lull~1 + wave(1)",
    ])


# ---------------------------------------------------------------------------------------------
#  Part 2: the search layer
# ---------------------------------------------------------------------------------------------

def baseline_test() -> None:
    """The four classical anchors on Covid-19 (icu, pRate): baseline, tuned-arx, sel-arx, random-walk --
    the rev3.2 ScalaTion numbers are TnT 43.5, 30.7 (y~1 + icu~(3,1) + pRate~(3,1)), 51.3 and 45.7."""
    banner("BaselineArm: the classical anchors on Covid-19 (icu, pRate)")
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6)
    runs = [BaselineArm(k, verbose=False).run(cfg, sc, 0) for k in (Method.Baseline, Method.TunedARX, Method.SelARX, Method.RandomWalk)]
    run_table(runs)
    print("\nthe tuned ARX's five best structures (score, in-sample sMAPE, terms):")
    print(runs[1].equations)
    runs[0].showQoF()


def genetic_test() -> None:
    """The genetic arm (crossover / mutation through the search loop) on Covid-19 and on the first 700 days
    of GEFCom2014, seeded with the plain-ARX baseline; the same stream reproduces the run."""
    banner("PipelineArm (genetic): Covid-19 (icu, pRate), streams 0 and 1")
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6, hp=SMALL)
    arm = PipelineArm(ArmSpec.genetic, verbose=False)
    runs = [arm.run(cfg, sc, s) for s in (0, 1)]
    run_table(runs)
    r = runs[0]
    print(f"\nstream 0: proposed {r.nProposed}, valid {r.nValid}, failed {r.nFailed}, fits {r.nEval}, rounds {len(r.conv)}")
    print("convergence (best score per round):", [round(c.bestScore, 3) for c in r.conv])
    LLM_NARX_SR.showTraces(r, 8)
    banner("PipelineArm (genetic): GEFCom2014, the first 700 days, lags 1..7")
    data = {k: v[:700] for k, v in Datasets.GEF14.load().items()}
    scg = FormulaScorer(data, hh=6, hp={**SMALL, "maxLag": 7, "skip": 7})
    rg = PipelineArm(ArmSpec.genetic, verbose=False).run(Datasets.GEF14, scg, 0)
    base = BaselineArm(Method.Baseline, verbose=False).run(Datasets.GEF14, scg, 0)
    run_table([base, rg])


def random_test() -> None:
    """The random arm (grammar sampling, no feedback) on Covid-19 and on the first 700 days of GEFCom2014."""
    banner("PipelineArm (random): Covid-19 (icu, pRate) and GEFCom2014 (700 days)")
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6, hp=SMALL)
    rc = PipelineArm(ArmSpec.random, verbose=False).run(cfg, sc, 0)
    data = {k: v[:700] for k, v in Datasets.GEF14.load().items()}
    scg = FormulaScorer(data, hh=6, hp={**SMALL, "maxLag": 7, "skip": 7})
    rg = PipelineArm(ArmSpec.random, verbose=False).run(Datasets.GEF14, scg, 0)
    run_table([rc, rg])
    print(f"\nCovid-19: {rc.nProposed} proposals, {rc.nValid} legal, {rc.nFailed} failed or rejected fits")
    for t in rc.traces[:6]:
        print(" ", t)


REPLIES = ["y ~= y~1 + x1~2\ny ~= y~(2,1) + log1p(x2~1)\ny ~= y~1 + x9~1",
           "```\ny ~= y~1 + y~2 + x1~1 * y~1\nTFORM: Log1p\ny ~= y~(3,1) + x1~(2,1)\n```",
           "Sure! y ~= y~1\ny ~= y~1 + wave(1)\ny ~= y~1 + sq(x2~2)\ny ~= y~(4,1) + x2~(3,1) + x1~3 + x2~4",
           "y ~= y~1 + hinge(x1~1, ?) + log1p(x2~1)\ny ~= y~(2,1) + pow(x1~1, ?) + x2~3"]


def mockllm_test() -> None:
    """The LLM arms with a scripted MockLLMClient (no key, no network): llm-loop and llm-cem, blind names
    in the prompt, rejects counted as invalid, the selector over the union library."""
    banner("PipelineArm (llm-loop, llm-cem) with a scripted MockLLMClient on Covid-19")
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6, hp=SMALL)
    mock = lambda: MockLLMClient(list(REPLIES))
    runs = [PipelineArm(spec, mock, verbose=False).run(cfg, sc, 0)
            for spec in (ArmSpec.llmLoop, ArmSpec.llmCEM, ArmSpec.llmCEM.copy(trimmer=SelectionTech.Backward))]
    run_table(runs)
    r = runs[1]
    print(f"\nllm-cem: LLM calls {r.llmCalls}, proposals {r.nProposed}, valid {r.nValid} ({r.validity:.0%}), "
          f"library of {len(r.labels)} terms; top inclusion probabilities:")
    order = np.argsort(-r.probs)[:6]
    print("  " + ", ".join(f"{r.labels[j]} {r.probs[j]:.2f}" for j in order))
    LLM_NARX_SR.showTraces(r)


def grid_test() -> None:
    """The harness's small zero-LLM grid on Covid-19 (anchors, random, genetic, stat-cem; two streams):
    the results table, the best run shown the base's way, the significance and plausibility tables."""
    banner("LLM_NARX_SR.runGrid: the zero-LLM arms on Covid-19 (icu, pRate), streams 0 and 1")
    cfg = Datasets.COVID2
    data = cfg.load()
    LLM_NARX_SR.VERBOSE = False
    runs = LLM_NARX_SR.runGrid("Covid-19 icu/pRate (zero-LLM arms, example)", cfg, LLM_NARX_SR.zeroLLMArms(), data,
                               streams=[0, 1], show=True, csv=False, hp=SMALL)
    Significance.reportSignificance("Covid-19 icu/pRate", runs, Method.Genetic)
    print(f"selection check: Spearman (selection score vs TnT sMAPE) over {len(runs)} runs = {Significance.selectionCheck(runs):.3f}")
    sc = LLM_NARX_SR.scorerOf(data, SMALL)
    eda = EDA4TS(data["y"], {v: data[v] for v in cfg.exo}, sc.trSize)
    PlausibilityChecker(cfg.expectedSigns).report("Covid-19 icu/pRate", runs, sc, eda)


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
    if which in ("baseline", "all"):
        baseline_test()
    if which in ("genetic", "all"):
        genetic_test()
    if which in ("random", "all"):
        random_test()
    if which in ("mockllm", "llm", "all"):
        mockllm_test()
    if which in ("grid", "all"):
        grid_test()
    print(f"\ndone in {time.time() - t0:.1f} s")
