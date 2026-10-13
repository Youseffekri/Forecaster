"""Tests of the arms and the harness (counterparts of baselineArmTest, pipelineArmTest / 2,
lLM_NARX_SRTest / 2 / 8, significanceTest, plausibilityCheckerTest): offline, small budgets, the
anchors checked against the rev3.2 / rev3.3 ScalaTion numbers on Covid-19 and GEFCom2014."""
import math

import numpy as np
import pytest

from modeling.feature_selection import SelectionTech
from modeling.llm_client import MockLLMClient
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import BaselineArm, arxFormulaStr
from modeling.forecasting.datasets import Datasets
from modeling.forecasting.eda4ts import EDA4TS
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.llm_narx_sr import LLMModel, LLM_NARX_SR
from modeling.forecasting.nonlinear.llm_narx_sr.pipeline_arm import PipelineArm
from modeling.forecasting.nonlinear.llm_narx_sr.plausibility_checker import PlausibilityChecker
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import (ArmSpec, ExpRun, LibraryKind, Method,
                                                                   ProposerKind, Source)
from modeling.forecasting.nonlinear.llm_narx_sr.significance import Significance, wilcoxon_test

SMALL = {"maxRounds": 3, "patience": 2, "cemSample": 40, "cemIter": 3, "refitTop": 2}   # the budgets of the tests

REPLIES = ["y ~= y~1 + x1~2\ny ~= y~(2,1) + log1p(x2~1)\ny ~= y~1 + x9~1",
           "```\ny ~= y~1 + y~2 + x1~1 * y~1\nTFORM: Log1p\ny ~= y~(3,1) + x1~(2,1)\n```",
           "Sure! y ~= y~1\ny ~= y~1 + wave(1)\ny ~= y~1 + sq(x2~2)\ny ~= y~(4,1) + x2~(3,1) + x1~3 + x2~4"]


def mock():
    return MockLLMClient(list(REPLIES))


@pytest.fixture(scope="module")
def covid():
    cfg = Datasets.COVID2
    return cfg, FormulaScorer(cfg.load(), hh=6, hp=SMALL)


def test_anchors_match_scalation(covid):
    """baselineArmTest (the rev4.1 log, Oct 13): baseline score 29.456, in-sample 18.590, validation 18.068,
    TnT 43.483; tuned-arx 22.298 / 20.953 / 17.636 / 30.711 (y~1 + icu~(3,1) + pRate~(3,1)); sel-arx
    18.271 / 21.622 / 14.817 / 51.286 (y~1 + icu~5 + icu~3 + pRate~6 + pRate~4 + pRate~1); random-walk
    41.221 / 33.410 / 39.887 / 45.712; the baseline IS the seed."""
    cfg, sc = covid
    runs = {k: BaselineArm(k, verbose=False).run(cfg, sc, 0) for k in (Method.Baseline, Method.TunedARX, Method.SelARX, Method.RandomWalk)}
    base, tuned, sel, rw = runs[Method.Baseline], runs[Method.TunedARX], runs[Method.SelARX], runs[Method.RandomWalk]
    assert all(not r.failed and r.stream == -1 for r in runs.values()) and all(BaselineArm(k).deterministic for k in runs)
    def nums(r):
        return [round(v, 3) for v in (r.best.score, r.best.smape, r.valSmape, r.testSmape)]
    assert nums(base) == [29.456, 18.590, 18.068, 43.483] and base.best.nTerms == 18 and base.exoUsed == ["icu", "pRate"]
    assert nums(tuned) == [22.298, 20.953, 17.636, 30.711] and tuned.best.formulaStr == "y ~= y~1 + icu~3 + icu~2 + icu~1 + pRate~3 + pRate~2 + pRate~1"
    assert tuned.best.score <= base.best.score + 1e-12 and len(tuned.top) == 5 and tuned.nEval == 126
    assert nums(sel) == [18.271, 21.622, 14.817, 51.286] and sel.nEval == 1033
    assert sel.best.formulaStr == "y ~= y~1 + icu~5 + icu~3 + pRate~6 + pRate~4 + pRate~1"
    assert nums(rw) == [41.221, 33.410, 39.887, 45.712] and rw.best.formula is None and rw.best.model is not None
    assert rw.best.label.startswith("RandomWalk") and rw.exoUsed == [] and all(v > 0 for v in rw.smapeTest)
    seed = sc.score(NARX_FormulaParser.parseFormula(cfg.baselineFormulaStr(6), sc.vars, 6))
    assert seed.key == base.best.key and seed.score == base.best.score, "the baseline must be the search seed"
    assert arxFormulaStr("y", 2, 0, ["icu"]) == "y ~= y~(2,1)" and arxFormulaStr("y", 6, 6, ["icu", "pRate"]) == cfg.baselineFormulaStr(6)
    assert base.name == "baseline" and tuned.method is Method.TunedARX and base.equations == ""
    with pytest.raises(ValueError):
        BaselineArm(Method.Genetic)


def test_auto_exo_anchors_match_scalation():
    """lLM_NARX_SRTest5 (the rev4.1 log, Oct 13): Covid-19 auto exo, the screening's shortlist exMort, cases;
    the deterministic arms' score, in-sample, validation and TnT sMAPE and their equations."""
    cfg = Datasets.COVID_ALL
    sc = FormulaScorer(cfg.load(), hh=6)
    assert sc.shortlist(cfg) == ["exMort", "cases"]
    expect = {Method.Baseline: ([40.589, 19.114, 24.868, 61.807], None),
              Method.TunedARX: ([31.569, 21.951, 26.313, 31.109], "y ~= y~1 + exMort~2 + exMort~1 + cases~2 + cases~1"),
              Method.SelARX: ([25.447, 26.053, 22.454, 54.515], "y ~= y~1 + exMort~4 + cases~1"),
              Method.RandomWalk: ([41.221, 33.410, 39.887, 45.712], None)}
    for k, (nums, eq) in expect.items():
        r = BaselineArm(k, verbose=False).run(cfg, sc, 0)
        assert [round(v, 3) for v in (r.best.score, r.best.smape, r.valSmape, r.testSmape)] == nums, k
        assert eq is None or r.best.formulaStr == eq


def test_zero_llm_pipeline_arms(covid):
    """pipelineArmTest: random and genetic return a fitted best no worse than the seed, every budget
    column filled, a trace behind the winner, the same stream reproducing the result; stat-cem's
    library, its traced equation from the library, a backward selector; the derived labels."""
    cfg, sc = covid
    base = sc.score(NARX_FormulaParser.parseFormula(cfg.baselineFormulaStr(6), sc.vars, 6))
    for spec in (ArmSpec.genetic, ArmSpec.random):
        arm = PipelineArm(spec, verbose=False)
        r = arm.run(cfg, sc, 0)
        assert not r.best.failed and r.best.score <= base.score + 1e-12
        assert r.nProposed > 0 and r.nEval > 0 and r.conv and r.nValid == r.nProposed
        assert r.originOf(r.best) is not None and not np.isnan(r.smapeTest).any()
        assert arm.run(cfg, sc, 0).best.key == r.best.key, "the same stream must reproduce the same best formula"
        assert r.equations and r.validity == 1.0
    arm = PipelineArm(ArmSpec.statCEM, verbose=False)
    r = arm.run(cfg, sc, 0)
    assert arm.name == "stat-cem" and r.method is Method.StatCEM and not r.best.failed
    nLin = 3 * 6 * 4 + 4
    assert nLin <= len(r.labels) <= nLin + 3 * 6 * 5 and r.probs is not None and len(r.probs) == len(r.labels)
    assert r.best.nFree <= 2 and all(t.label(__import__("modeling.forecasting.arx_formula", fromlist=["LabelMode"]).LabelMode.SLOT) in r.labels
                                     for t in r.best.formula.preds), "fitted, from the library"
    assert len(r.traces) == 1 and r.traces[0].source is Source.Selected and r.traces[0].trimmer is SelectionTech.CEM
    assert arm.run(cfg, sc, 0).best.key == r.best.key
    small = FormulaScorer(cfg.load(), hh=6, hp={**SMALL, "nlTerms": 0, "maxLag": 3})
    armB = PipelineArm(ArmSpec.statCEM.copy(selector=SelectionTech.Backward), verbose=False)
    rB = armB.run(cfg, small, 0)
    assert armB.name == "stat-cem-sel-backward" and rB.probs is None and not rB.best.failed and len(rB.labels) == 3 * 3 * 4 + 4
    assert ArmSpec.genetic.copy(selector=SelectionTech.CEM).name == "genetic-sel-cem"
    assert ArmSpec.genetic.copy(trimmer=SelectionTech.Backward, selector=SelectionTech.CEM).name == "genetic-trim-backward-sel-cem"
    assert ArmSpec.llmCEM.name == "llm-cem" and ArmSpec.llmLoop.name == "llm-loop" and ArmSpec.random.name == "random"
    assert ArmSpec.llmCEM.copy(eda=False, rounds=2, named=True).name == "llm-cem-blind-named-2r"
    assert ArmSpec.llmCEM.copy(library=LibraryKind.TopK, tag="ft").name == "llm-cem-topk-ft"
    with pytest.raises(ValueError):
        ArmSpec(ProposerKind.NoLoop)
    with pytest.raises(ValueError):
        ArmSpec(ProposerKind.LLM, rounds=2)
    with pytest.raises(ValueError):
        PipelineArm(ArmSpec.llmLoop)


def test_llm_arms_with_a_scripted_client(covid):
    """pipelineArmTest2: llm-loop (calls counted, validity below one, a fitted best no worse than the
    seed, the adaptive stop, a trace behind the winner, reproducible), llm-cem (the union library,
    the selected equation from it, probabilities and the selection trace) and the ablations."""
    cfg, sc = covid
    base = sc.score(NARX_FormulaParser.parseFormula(cfg.baselineFormulaStr(6), sc.vars, 6))
    loopArm = PipelineArm(ArmSpec.llmLoop, mock, maxRounds=6, patience=2, verbose=False)
    r0 = loopArm.run(cfg, sc, 0)
    assert loopArm.name == "llm-loop" and r0.method is Method.LLMLoop
    assert r0.llmCalls == len(r0.conv) and r0.llmCalls >= 2, "one LLM call per round"
    assert r0.validity < 1.0 and r0.nValid > 0 and r0.tokens > 0 and r0.cost == 0.0
    assert not r0.best.failed and r0.best.score <= base.score + 1e-12 and len(r0.conv) < 6
    assert r0.originOf(r0.best) is not None and len(r0.traces) == 1 + r0.nValid
    assert loopArm.run(cfg, sc, 0).best.key == r0.best.key
    cemArm = PipelineArm(ArmSpec.llmCEM, mock, maxRounds=6, patience=2, verbose=False)
    r = cemArm.run(cfg, sc, 0)
    assert cemArm.name == "llm-cem" and r.method is Method.LLMCEM and r.probs is not None and len(r.probs) == len(r.labels)
    assert all(t.label() in set(r.labels) for t in r.best.formula.preds), "the selected equation must come from the library"
    assert not r.best.failed and any(t.source is Source.Selected for t in r.traces)
    assert r.best.score <= r0.best.score + 1e-12, "the selector over the library must score no worse than the loop's best"
    specs = [ArmSpec.llmCEM.copy(eda=False), ArmSpec.llmCEM.copy(trimmer=SelectionTech.CEM),
             ArmSpec.llmCEM.copy(trimmer=SelectionTech.Backward), ArmSpec.llmCEM.copy(selector=SelectionTech.Backward),
             ArmSpec.llmCEM.copy(rounds=2)]
    runs = []
    for s in specs:
        a = PipelineArm(s, mock, maxRounds=4, patience=2, verbose=False)
        ra = a.run(cfg, sc, 0)
        assert not ra.best.failed, f"{a.name} must return a fitted equation"
        runs.append((a.name, ra))
    assert [n for n, _ in runs] == ["llm-cem-blind", "llm-cem-trim", "llm-cem-trim-backward", "llm-cem-sel-backward", "llm-cem-2r"]
    assert any(t.trimmed for t in runs[2][1].traces), "the padded LLM proposal must be trimmed"
    rSelB = runs[3][1]
    assert rSelB.probs is None and any(t.source is Source.Selected and t.trimmer is SelectionTech.Backward for t in rSelB.traces)
    r2r = runs[4][1]
    assert r2r.llmCalls > r.llmCalls / 2 and sum(1 for t in r2r.traces if t.source is Source.Selected) >= 2


def test_grid_report_and_folds(covid, tmp_path, capsys):
    """lLM_NARX_SRTest / 2: the grid over anchors and arms (one stream), the report, bestOf, the CSV,
    showBest, the Pareto front, the traces; runFolds on two nested prefixes; a failing arm is recorded."""
    cfg, _ = covid
    LLM_NARX_SR.RESULTS_DIR = str(tmp_path)
    LLM_NARX_SR.VERBOSE = False
    arms = LLM_NARX_SR.mainArms(mock, (0.0, 0.0)) + LLM_NARX_SR.ablationArms(mock, (0.0, 0.0))
    assert [a.name for a in arms] == ["baseline", "tuned-arx", "sel-arx", "random-walk", "stat-cem", "genetic", "llm-loop", "llm-cem",
                                      "random", "llm-cem-blind", "llm-cem-trim", "llm-cem-2r"]
    assert len(LLM_NARX_SR.zeroLLMArms()) == 7 and len(LLM_NARX_SR.llmArms(mock)) == 4 and len(LLM_NARX_SR.ftArms(mock)) == 2
    assert LLMModel.GPT4oMini.prices == (0.15, 0.60) and LLMModel.GPT4o.id == "gpt-4o"

    class Broken(BaselineArm):
        def run(self, cfg, scorer, stream):
            raise RuntimeError("boom")

    runs = LLM_NARX_SR.runGrid("Covid-19 icu/pRate (test)", cfg, arms[:5] + arms[6:8] + [Broken()], streams=[0], show=True, hp=SMALL)
    out = capsys.readouterr().out
    names = [r.name for r in runs]
    assert names == ["baseline", "tuned-arx", "sel-arx", "random-walk", "stat-cem", "llm-loop", "llm-cem", "baseline"]
    assert runs[-1].failed and runs[-1].error and "ERROR @ LLM_NARX_SR.runGrid" in out
    assert all(not r.failed for r in runs[:-1]) and all(not np.isnan(r.smapeTest).any() for r in runs[:-1])
    assert all(r.llmCalls > 0 for r in runs if r.method in (Method.LLMLoop, Method.LLMCEM))
    assert "results (in-sample on the fit rows | validation | TnT rolling on the testing rows)" in out
    assert "per-method means over streams" in out and "FAILED: " in out and "REPORT" in out
    assert "complexity vs accuracy front" in out and "traces (" in out and "term importance at h = 1" in out
    top = LLM_NARX_SR.bestOf(runs)
    assert not top.failed and top is LLM_NARX_SR.bestOf(runs, top.method)
    with pytest.raises(ValueError):
        LLM_NARX_SR.bestOf(runs, Method.Random)
    tab, hdr = LLM_NARX_SR.resultsTable(runs)
    assert tab.shape == (8, len(hdr)) and hdr[6] == "train_mean" and hdr[-1] == "n_exo" and tab[0, -1] == 2
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) == 1 and csvs[0].name.startswith("Covid-19_icu_pRate__test__")
    lines = csvs[0].read_text().splitlines()
    assert lines[0].startswith("method,stream,train_h1") and len(lines) == 9 and lines[1].startswith("baseline,-1,")
    front = LLM_NARX_SR.paretoFront(top)
    assert front and all(a.nTerms < b.nTerms and a.score > b.score for a, b in zip(front, front[1:])) and any(c.key == top.best.key for c in front)
    assert len(LLM_NARX_SR.byMethod(runs)) == 7
    m = len(cfg.load()["y"])
    folds = LLM_NARX_SR.runFolds("Covid-19 folds (test)", cfg, LLM_NARX_SR.anchors()[:1] + LLM_NARX_SR.armsOf([ArmSpec.genetic]),
                                 [m - 16, m], streams=[0, 1], csv=False, hp=SMALL)
    assert len(folds) == 2 and [len(f) for f in folds] == [3, 3] and folds[0][0].best.nTerms == 18
    Significance.reportFolds("Covid-19 folds (test)", [m - 16, m], folds)
    out = capsys.readouterr().out
    assert "TnT sMAPE per fold" in out and "vs baseline" in out and "genetic" in out
    rho = Significance.selectionCheck([r for f in folds for r in f])
    assert math.isnan(rho) or -1.0 <= rho <= 1.0


def test_significance_on_hand_made_runs(capsys):
    """significanceTest: a uniformly better method is significant, one against itself is not, the
    one-sample test against a far-off constant is significant, the selection check is +1."""
    f = NARX_FormulaParser.parseFormula("y ~= y~1", ["y"], 1)

    def run(method, stream, smape, tnt):
        q = [{"smape": smape, "m": 50.0, "dfr": 1.0}]
        return ExpRun(method, "", stream, Candidate(f, smape, q, 1), [{"smape": tnt}])

    a = [run(Method.LLMCEM, s, 8.0 + 0.1 * s, 9.0 + 0.1 * s) for s in range(8)]
    b = [run(Method.StatCEM, s, 12.0 + 0.1 * s, 13.0 + 0.1 * s) for s in range(8)]
    ab = Significance.compareMethods(a, b)
    assert ab.nPairs == 8 and ab.pValue < 0.05 and "*significant" in str(ab) and ab.meanA == pytest.approx(9.35)
    aa = Significance.compareMethods(a, a)
    assert aa.pValue > 0.5 and aa.wStat == 0.0
    ac = Significance.compareToConstant(a, 20.0)
    assert ac.pValue < 0.05 and ac.sdB == 0.0
    assert Significance.selectionCheck(a + b) == pytest.approx(1.0)
    assert math.isnan(Significance.selectionCheck(a[:2]))
    Significance.reportSignificance("hand-made", a + b + [run(Method.Baseline, -1, 10.0, 11.0)], Method.LLMCEM)
    out = capsys.readouterr().out
    assert "llm-cem vs stat-cem" in out and "(stream-paired)" in out and "llm-cem vs baseline" in out and "(deterministic)" in out
    with pytest.raises(ValueError):
        Significance.compareMethods(a, [run(Method.StatCEM, 99, 1.0, 1.0)])
    # the signed-rank statistic against scipy's normal approximation (continuity corrected)
    from scipy.stats import wilcoxon
    x = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 0.90, 2.1, 1.1])
    y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29, 0.95, 1.9, 1.5])
    w, z, p = wilcoxon_test(x, y)
    assert p == pytest.approx(wilcoxon(x, y, method="approx", correction=True).pvalue, abs=1e-9) and w == min(w, abs(w))
    assert wilcoxon_test([1.0, 2.0], [1.0, 2.0]) == (0.0, 0.0, 1.0)


def test_plausibility_checks():
    """plausibilityCheckerTest: z drives y positively at lag 2 and w negatively at lag 1."""
    rng = np.random.default_rng(31)
    n = 300
    z = 20.0 + 2.0 * rng.standard_normal(n)
    w = 20.0 + 2.0 * rng.standard_normal(n)
    y = np.zeros(n)
    y[:2] = [40.0, 41.0]
    for t in range(2, n):
        y[t] = 10.0 + 0.6 * y[t - 1] + 1.5 * z[t - 2] - 1.0 * w[t - 1] + rng.standard_normal()
    data = {"y": y, "z": z, "w": w}
    sc = FormulaScorer(data, hh=2)
    eda = EDA4TS(y, {"z": z, "w": w}, sc.trSize)
    checker = PlausibilityChecker({"z": 1, "w": -1})
    c1 = sc.score(NARX_FormulaParser.parseFormula("y ~= y~1 + z~(2,1) + log1p(w~1)", sc.vars, 3))
    p1 = checker.evaluate(c1, sc, eda)
    assert p1.netEffects["z"] > 0.0 > p1.netEffects["w"] and p1.signConsistency == 1.0 and p1.nSignChecked == 2
    assert 0.0 <= p1.cleanliness <= 1.0 and "sign= 1.00" in p1.tableRow
    p2 = PlausibilityChecker({"z": 1, "w": 1}).evaluate(c1, sc, eda)
    assert p2.signConsistency == pytest.approx(0.5)
    c2 = sc.score(NARX_FormulaParser.parseFormula("y ~= y~1 + z~2", sc.vars, 3))
    assert checker.stability([c1, c1]) == 1.0 and 0.0 < checker.stability([c1, c2]) < 1.0 and math.isnan(checker.stability([c1]))
    base = Candidate(None, 1.0, [], 0, label="RandomWalk")
    assert math.isnan(checker.evaluate(base, sc, eda).signConsistency)
    runs = [ExpRun(Method.Genetic, "", 0, c1, []), ExpRun(Method.Genetic, "", 1, c2, [])]
    checker.report("planted", runs, sc, eda)


def test_gefcom_checks():
    """lLM_NARX_SRTest8 / rev3.3: the 21-lag baseline's rolling TnT sMAPE 12.783, the free hinge on
    yesterday's mean temperature at the cooling point (~65.8 F), two hinges rejected as collinear (VIF 18.7),
    and a small genetic run on the daily series with lags 1..7."""
    cfg = Datasets.GEF14
    data = cfg.load()
    sc = FormulaScorer(data, hh=6, hp={**SMALL, "maxLag": 7, "skip": 7})
    assert sc.trSize == 1679 and sc.teSize == 420
    c = sc.score(cfg.baselineFormulaStr(7))
    assert c.nTerms == 21 and not c.failed
    _, q = sc.tnt(c.formula.fresh())
    assert np.mean([x["smape"] for x in q]) == pytest.approx(12.783, abs=0.005)
    ch = sc.score("y ~= y~(7,1) + hinge(temp~1, ?)")
    assert not ch.failed and 55.0 < ch.formula.params()[0].value < 80.0 and ch.formula.params()[0].value == pytest.approx(65.8, abs=0.3)
    c2 = sc.score("y ~= y~(7,1) + hinge(temp~1, ?) + hinge(tmax~1, ?)")
    assert c2.failed and "collinear" in c2.reason and "VIF 18.7" in c2.reason
    r = PipelineArm(ArmSpec.genetic, verbose=False).run(cfg, sc, 0)
    assert not r.best.failed and r.best.score <= c.score + 1e-12 and r.exoUsed <= ["temp", "tmax"]
    base = BaselineArm(Method.Baseline, verbose=False).run(cfg, sc, 0)
    assert base.testSmape == pytest.approx(12.783, abs=0.005) and base.exoUsed == ["temp", "tmax"]
