"""Tests of TermLibrary (counterparts of termLibraryTest / termLibraryTest3) and SearchLoop
(counterpart of searchLoopTest) on planted structures."""
import math

import numpy as np
import pytest

from modeling.feature_selection import SelectionTech
from modeling.forecasting.arx_formula import FormulaVar, LabelMode, WaveTerm
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser, parse_formula
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import Source, Trace
from modeling.forecasting.nonlinear.llm_narx_sr.search_loop import Proposer, Reject, SearchLoop
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import TermLibrary, fresh_terms, params_of


def planted_ar3(n=400, seed=11):
    """y_t = 5 + 0.7 y_t-1 - 0.2 y_t-3 + 1.5 z_t-2 + e_t."""
    rng = np.random.default_rng(seed)
    z = 10.0 + 2.0 * rng.standard_normal(n)
    y = np.zeros(n)
    y[:3] = [30.0, 31.0, 32.0]
    for t in range(3, n):
        y[t] = 5.0 + 0.7 * y[t - 1] - 0.2 * y[t - 3] + 1.5 * z[t - 2] + rng.standard_normal()
    return {"y": y, "z": z}


def planted_log(n=300, burn=50, seed=41):
    """y_t = 5 + 0.6 y_t-1 + 4 log (z_t-1 + 0.3) + e_t."""
    rng = np.random.default_rng(seed)
    zAll = np.maximum(3.0 + 1.5 * rng.standard_normal(n + burn), 0.1)
    yAll = np.zeros(n + burn)
    yAll[0] = 20.0
    for t in range(1, n + burn):
        yAll[t] = 5.0 + 0.6 * yAll[t - 1] + 4.0 * math.log(zAll[t - 1] + 0.3) + 0.3 * rng.standard_normal()
    return {"y": yAll[burn:], "z": zAll[burn:]}


def baseOf(t):
    """The (variable, lag) under any transform."""
    return (t.name, t.lag) if t.vars else (t.label(), 0)


def test_grammar_library_and_union_library():
    data = planted_ar3()
    sc = FormulaScorer(data, hh=1)
    lib = TermLibrary.fromGrammar("y", ["z"], maxLag=4)
    assert lib.size == 8 + 16 + 8 and lib.labels[:4] == ["y~1", "y~2", "y~3", "y~4"] and lib.indexOf("z~2") == 5
    assert lib.indexOf("nope") == -1 and lib.unitOf(3) == {3}
    f1 = parse_formula("y ~= y~1 + z~2", ["y", "z"], 4)
    f2 = parse_formula("y ~= y~1 + y~3", ["y", "z"], 4)
    f3 = parse_formula("y ~= y~1 + log1p(z~1)", ["y", "z"], 4)
    cands = [sc.score(f) for f in (f1, f2, f3)]
    lib2 = TermLibrary.from_(cands)
    prior = TermLibrary.frequencyPrior(cands, lib2)
    assert lib2.size == 4 and prior[lib2.indexOf("y~1")] == pytest.approx(0.95) and prior[lib2.indexOf("z~2")] == pytest.approx(1 / 3)
    truth = {"y~1", "y~3", "z~2"}
    scores = []
    for tech in SelectionTech:
        c, probs, nEv = lib2.selectBy(tech, sc, prior, stream=1)
        assert {t.label() for t in c.formula.preds} == truth, f"{tech} must recover the planted structure"
        assert (probs is not None) == (tech is SelectionTech.CEM) and nEv > 0
        scores.append(c.score)
    assert max(scores) - min(scores) < 1e-9, "the same equation scores the same under every technique"
    cG, probs, nEv = lib.selectBy(SelectionTech.CEM, sc, stream=0, nSample=60, maxIter=6)
    assert not cG.failed and {("y", 1), ("y", 3), ("z", 2)} <= {baseOf(t) for t in cG.formula.preds}
    assert len(probs) == lib.size and "0." in lib.showProbs(probs, 3)
    assert cG.score <= sc.score(lib.full).score + 1e-9


def test_free_parameter_library_mechanics():
    """termLibraryTest3: one term per structure, the prior finds it, the library model is built at the
    current values without a fit, the selected equation has its OWN fitted parameters, the grammar
    library with nlTerms has one parameter per form, `fitted` makes at most one fit per term."""
    data = planted_log()
    sc = FormulaScorer(data, hh=1, hp={"lambda": 1e-6, "boundTol": -1.0, "maxVIF": 0.0, "valFrac": 0.0})
    vars_ = sc.vars
    cands = [sc.score(parse_formula(s, vars_, 4)) for s in
             ("y ~= y~1 + log(? + z~1)", "y ~= y~1 + log(?3 + z~1) + z~2", "y ~= y~2 + pow(z~1, ?)")]
    lib = TermLibrary.from_(cands)
    assert lib.size == 5 and lib.labels.count("log(? + z~1)") == 1
    prior = TermLibrary.frequencyPrior(cands, lib)
    assert prior[lib.indexOf("log(? + z~1)")] == pytest.approx(2 / 3) and lib.indexOf("log(?2.000 + z~1)") == lib.indexOf("log(? + z~1)")
    n0 = sc.evaluations
    libMod = sc.library(lib.resp, lib.terms, lib.tFormT)
    assert libMod.X.shape[1] == lib.size and libMod.nFree == 0 and sc.evaluations == n0, "built at the current values, no fit"
    sel, _, nEv = lib.selectBy(SelectionTech.Backward, sc)
    assert not sel.failed and any(t.label().startswith("log(") for t in sel.formula.preds)
    libPars, selPars = params_of(lib.terms), sel.formula.params()
    assert all(not any(p is q for q in libPars) for p in selPars), "no parameter object shared with the library"
    assert all(p.fitted for p in selPars) and "?" not in sel.formulaStr
    gl = TermLibrary.fromGrammar("y", ["z"], maxLag=3, nlTerms=True)
    assert gl.size == 6 + 12 + 6 + 6 * len(TermLibrary.LIB_NL_FORMS) and len(params_of(gl.terms)) == 6 * len(TermLibrary.LIB_NL_FORMS)
    n0 = sc.evaluations
    glf = gl.fitted(sc)
    assert 6 + 12 + 6 <= glf.size <= gl.size and sc.evaluations - n0 <= 6 * len(TermLibrary.LIB_NL_FORMS)
    fp = params_of(glf.terms)
    assert fp and all(p.fitted for p in fp) and glf.freeCount(range(glf.size)) == sum(1 for p in fp if p.free)
    cap = glf.withinCap(list(range(glf.size)), 2)
    assert glf.freeCount(cap) <= 2 and glf.finite(data).size == glf.size


def test_waves_stay_whole_and_fresh_terms():
    freqs = [1 / 12.0, 1 / 50.0]
    lib = TermLibrary.fromGrammar("y", [], 2, freqs=freqs, nWaves=2)
    j = [i for i, t in enumerate(lib.terms) if isinstance(t, WaveTerm)]
    assert len(j) == 4 and lib.unitOf(j[0]) == {j[0], j[1]} and lib.completeWaves([0, j[2]]) == [0, j[2], j[3]]
    f = lib.formulaOf([0, j[0]])
    assert "wave(1)" in f.formulaStr and f.nTerms == 3
    g = parse_formula("y ~= y~1 + pow(y~(2,1), ?)", ["y"], 3)
    ft = fresh_terms(g.preds)
    assert len(params_of(ft)) == 1 and not any(p is q for p in params_of(ft) for q in g.params()), "one shared copy, none of the originals"


class Scripted(Proposer):
    """Round 1 two weak formulas and one reject, round 2 the planted formula and a duplicate, then nothing new."""

    def __init__(self, vars_):
        self.round = 0
        self.vars_ = vars_

    @property
    def source(self):
        return Source.LLM

    def parse(self, s):
        f, why = NARX_FormulaParser.tryParseFormula(s, self.vars_, 4)
        return f if f is not None else Reject(why)

    def propose(self, top, n):
        self.round += 1
        if self.round == 1:
            return [self.parse("y ~= y~2"), self.parse("y ~= z~1"), self.parse("y ~= w~1")]
        if self.round == 2:
            return [self.parse("y ~= y~1 + z~2"), self.parse("y ~= z~2 + y~1")]
        return [self.parse("y ~= y~2")]


def planted_z2(n=200, burn=50, seed=21):
    rng = np.random.default_rng(seed)
    zAll = 10.0 + 2.0 * rng.standard_normal(n + burn)
    yAll = np.zeros(n + burn)
    yAll[:2] = [30.0, 31.0]
    for t in range(2, n + burn):
        yAll[t] = 5.0 + 0.7 * yAll[t - 1] + 1.5 * zAll[t - 2] + rng.standard_normal()
    return {"y": yAll[burn:], "z": zAll[burn:], "u": rng.standard_normal(n)}


def test_search_loop_scripted():
    data = planted_z2()
    sc = FormulaScorer(data, hh=2)
    vars_ = sc.vars
    loop = SearchLoop(sc, topK=3, verbose=False)
    loop.seed((parse_formula("y ~= y~1", vars_, 4), Source.Seed))
    seedScore = loop.best.score
    done = loop.run(Scripted(vars_), perRound=3, maxRounds=10, patience=2)
    assert done == 4 and loop.nProposed == 7, "4 rounds (3 + 2 + 1 + 1 proposals), then stagnant/dry stop"
    assert loop.nValid == loop.nProposed - 1 and loop.size == 4
    assert {t.label() for t in loop.best.formula.preds} == {"y~1", "z~2"} and loop.best.score < seedScore
    assert len(loop.traces) == 1 + loop.nValid and all(loop.originOf(c) is not None for c in loop.all)
    org = loop.originOf(loop.best)
    assert org.round == 2 and org.source is Source.LLM and sum(1 for t in loop.traces if t.key == loop.best.key) == 2
    assert len(loop.convergence) == 4 and loop.convergence[-1].bestScore == pytest.approx(loop.best.score)
    assert loop.fits > 0 and loop.error is None


def test_trimmers_on_a_padded_proposal():
    data = planted_z2()
    sc = FormulaScorer(data, hh=2)
    vars_ = sc.vars
    padded = parse_formula("y ~= y~1 + z~2 + z~1 + y~4 + z~4 + u~1 + u~2", vars_, 4)
    for tech in (SelectionTech.CEM, SelectionTech.Backward):
        loopT = SearchLoop(sc, topK=3, trimmer=tech, stream=0, verbose=False)
        tr = loopT.enter(padded, Source.LLM, 1)
        assert tr.trimmer is tech and tr.fTrim.nTerms < tr.f.nTerms, f"{tech} must drop an idle term"
        kept = {t.label() for t in tr.fTrim.preds}
        assert kept <= {t.label() for t in tr.f.preds} and {"y~1", "z~2", "z~1"} <= kept
        assert tr.candTrim.score < tr.cand.score and loopT.size == 1 and loopT.best.key == tr.key
        assert tr.dropped and tr.gain > 0 and str(tr).count("->") >= 2
    loopK = SearchLoop(sc, topK=3, trimmer=SelectionTech.Backward, stream=0, verbose=False)
    trK = loopK.enter(parse_formula("y ~= y~1 + z~2 + z~1", vars_, 4), Source.LLM, 1)
    assert trK.candTrim.score <= trK.cand.score and (trK.trimmer is None) == (trK.fTrim.key == trK.f.key)


def test_loop_rejects_over_the_cap_and_budget():
    data = planted_z2()
    sc = FormulaScorer(data, hh=1, hp={"maxFree": 1, "maxFits": 2})
    loop = SearchLoop(sc, topK=2, verbose=False)
    f = parse_formula("y ~= y~1 + pow(z~1, ?) + log(? + z~2)", sc.vars, 4)
    n = loop.evaluate([f, Reject("bad")], Source.Genetic, 1)
    assert n == 0 and loop.nProposed == 2 and loop.nValid == 0 and loop.size == 0

    class Many(Proposer):
        source = Source.Random

        def propose(self, top, n):
            return [parse_formula(f"y ~= y~1 + z~{k}", sc.vars, 4) for k in range(1, n + 1)]

    done = loop.run(Many(), perRound=3, maxRounds=5, patience=5)
    assert done == 1 and loop.fits >= 2, "the fit budget stops the loop after the first round"
