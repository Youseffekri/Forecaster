"""Tests of the proposers (counterparts of geneticProposerTest, lLMProposerTest), the explainer
(explainerTest) and the LLM clients (lLMClientTest) -- all offline, with a MockLLMClient."""
import math

import numpy as np
import pytest

from modeling.feature_selection import SelectionTech
from modeling.llm_client import CachedClient, CountingClient, Json, LLMClient, LocalClient, MockLLMClient
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser, parse_formula
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Datasets
from modeling.forecasting.nonlinear.llm_narx_sr.eda4ts import EDA4TS
from modeling.forecasting.nonlinear.llm_narx_sr.explainer import Explainer, TermEffect
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.genetic_proposer import GeneticProposer, RandomTerms
from modeling.forecasting.nonlinear.llm_narx_sr.llm_proposer import LLMProposer, genericSchema
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import Source, Trace
from modeling.forecasting.nonlinear.llm_narx_sr.search_loop import Reject

VARS = ["y", "hosp", "icu"]
FREQS = [0.13043, 0.06521]
MAX_LAG = 6


def test_random_sampling_is_legal_and_reproducible():
    gp = GeneticProposer("y", ["hosp", "icu"], MAX_LAG, FREQS, stream=0, random=True)
    assert gp.source is Source.Random
    ps = gp.propose([], 30)
    for f in ps:
        assert f.preds and len(f.preds) <= 12
        assert all(t.lag <= MAX_LAG and t.vars <= set(VARS) for t in f.preds), f.formulaStr
        back, why = NARX_FormulaParser.tryParseFormula(f.formulaStr, VARS, MAX_LAG, FREQS)
        assert back is not None and back.key == f.key, (f.formulaStr, why)
        assert f.nFree <= 2, "within the free-parameter cap"
    assert sum(1 for f in ps if f.nFree > 0) > 0, "random sampling must draw free-parameter forms (hp nlTerms = 1)"
    again = GeneticProposer("y", ["hosp", "icu"], MAX_LAG, FREQS, stream=0, random=True).propose([], 30)
    assert [f.key for f in ps] == [f.key for f in again], "the same stream must reproduce the same proposals"
    other = GeneticProposer("y", ["hosp", "icu"], MAX_LAG, FREQS, stream=1, random=True).propose([], 30)
    assert [f.key for f in ps] != [f.key for f in other]
    lin = GeneticProposer("y", ["hosp", "icu"], MAX_LAG, FREQS, stream=0, random=True, nonlinear=False).propose([], 30)
    assert all(f.nFree == 0 for f in lin)
    rt = RandomTerms("y", [], 3, None, 5)
    assert all(t.name == "y" for t in rt.formula(4).preds) and rt.normalizeWaves([]) == []


def test_offspring_inherit_and_never_share_parameters():
    f1 = parse_formula("y ~= y~1 + hosp~2", VARS, MAX_LAG, FREQS)
    f2 = parse_formula("y ~= y~(2,1) + log1p(icu~1) + wave(1)", VARS, MAX_LAG, FREQS)
    c1, c2 = Candidate(f1, 10.0, [], 2), Candidate(f2, 12.0, [], 5)
    top = [Trace(0, Source.Seed, c1, None, c1), Trace(0, Source.Seed, c2, None, c2)]
    gp = GeneticProposer("y", ["hosp", "icu"], MAX_LAG, FREQS, stream=2)
    assert gp.source is Source.Genetic
    kids = gp.propose(top, 20)
    parentTerms = {t.label() for t in f1.preds + f2.preds}
    inherited = sum(1 for k in kids if any(t.label() in parentTerms for t in k.preds))
    assert inherited >= len(kids) // 2 and all(k.preds and len(k.preds) <= 12 for k in kids)
    f3 = parse_formula("y ~= y~1 + pow(hosp~1, ?) + log(? + icu~1)", VARS, MAX_LAG, FREQS)
    c3 = Candidate(f3, 9.0, [], 3)
    kid3 = GeneticProposer("y", ["hosp", "icu"], MAX_LAG, FREQS, stream=3).propose([Trace(0, Source.Seed, c3, None, c3)], 10)
    pp = f3.params()
    for k in kid3:
        for p in k.params():
            assert not any(p is q for q in pp), f"shared parameter object in {k.formulaStr}"
    assert any(k.nFree > 0 for k in kid3)
    shifted = gp.shift(parse_formula("y ~= log1p(hosp~6)", VARS, MAX_LAG).preds[0])
    assert shifted.label() == "log1p(hosp~5)"
    capped = gp.cap(parse_formula("y ~= y~1 + pow(hosp~1, ?) + log(? + icu~1) + sat(y~2, ?)", VARS, MAX_LAG))
    assert capped.nFree <= 2


REPLY = """Here are my proposals:
```
y ~= y~1 + y~2 + log1p(x1~2)
TFORM: Log1p
y ~= y~(3,1) + x2~1 * y~1
y ~= y~1 + pow(x1~1, ?) + sat(x2~1, ?)
y ~= y~1 + wave(3)
y ~= y~1 + x9~1
```"""


def test_llm_proposer_blind_prompt_and_parsing():
    cfg = Datasets.COVID2
    data = cfg.load()
    sc = FormulaScorer(data, hh=6, hp={"boundTol": -1.0, "maxVIF": 0.0})     # the rejection rules off: this test is about the proposer
    eda = EDA4TS(data["y"], {"icu": data["icu"], "pRate": data["pRate"]}, sc.trSize)
    lp = LLMProposer(MockLLMClient([REPLY]), cfg, sc.vars, 6, FREQS, eda, tag="[run 0]")
    assert lp.source is Source.LLM
    assert "x1" in lp.system and "icu" not in lp.system and "pow(x1~1, ?)" in lp.system and "sat(x1~1, ?)" in lp.system
    assert genericSchema(2) in lp.system and "at most 2 are allowed" in lp.system
    c0 = sc.score(parse_formula("y ~= y~1 + icu~2", sc.vars, 6))
    cF = sc.score(parse_formula("y ~= y~1 + icu~2 + pRate~3", sc.vars, 6))
    top = [Trace(0, Source.Seed, c0, None, c0), Trace(1, Source.LLM, cF, SelectionTech.CEM, c0)]
    out = lp.propose(top, 5)
    assert "proposed y ~= y~1 + x1~2 + x2~3  ->  kept y ~= y~1 + x1~2  (dropped: x2~3)" in lp.lastPrompt
    assert "icu" not in lp.lastPrompt and "testing rows" not in lp.lastPrompt and "TnT" not in lp.lastPrompt
    assert "validation sMAPE" in lp.lastPrompt and "Data profile" in lp.lastPrompt and "[run 0]" in lp.lastPrompt
    ok = [p for p in out if not isinstance(p, Reject)]
    bad = [p for p in out if isinstance(p, Reject)]
    assert len(ok) == 3 and len(bad) == 2
    assert ok[0].tFormT is not None and any(t.label() == "log1p(icu~2)" for t in ok[0].preds), "TFORM attached and x1 mapped to icu"
    assert any(t.label() == "pRate~1 * y~1" for t in ok[1].preds)
    assert ok[2].nFree == 2 and any(t.label() == "sat(pRate~1, ?)" for t in ok[2].preds)
    cN = sc.score(ok[2])
    assert not cN.failed and cN.nFree == 2
    out2 = lp.propose([Trace(2, Source.LLM, cN, None, cN)], 2)
    assert "pow(x1~1, ?" in lp.lastPrompt and "pow(x1~1, ?)" not in lp.lastPrompt, "a fitted equation is shown with ?w slots"
    assert "Rejected last round" in lp.lastPrompt and out2 and lp.calls == 2
    assert any("wave(3)" in r.reason for r in bad) and any("x9" in r.reason for r in bad)
    lp.feedback([("y ~= y~1 + ema(icu~1, ?)", "Decay 0.99 at its upper bound 0.99")], "residual autocorrelation of the best at lags 1..6: [0.1]")
    lp.setCore(["y~1", "icu~2"])
    lp.propose([], 1)
    assert "ema(x1~1, ?)" in lp.lastPrompt and "stable by the selection stage" in lp.lastPrompt and "# residual autocorrelation" in lp.lastPrompt
    named = LLMProposer(MockLLMClient(["y ~= y~1 + icu~1"]), cfg, sc.vars, 6, None, named=True, nonlinear=False)
    assert "icu" in named.system and cfg.schema in named.system and "wave(1)" not in named.system and "free exponent" not in named.system
    assert not isinstance(named.propose([], 1)[0], Reject)


def test_explainer_checks_the_claims():
    cfg = Datasets.COVID2
    data = cfg.load()
    sc = FormulaScorer(data, hh=6)
    f = parse_formula("y ~= y~(2,1) + icu~1 + log1p(pRate~3)", sc.vars, 6)
    cand = sc.score(f)
    mod = sc.fit(f)
    eff = Explainer.importance(mod, f, sc.trSize)
    assert [e.label for e in eff] == ["y~2", "y~1", "icu~1", "log1p(pRate~3)"]
    for j, e in enumerate(eff):
        assert e.effect == pytest.approx(e.coef * np.std(mod.X[: sc.trSize, j], ddof=1))
    table = Explainer.importanceTable(eff)
    assert table.splitlines()[0].startswith("term") and len(table.splitlines()) == 5
    rankOf = {e.label: i + 1 for i, e in enumerate(sorted(eff, key=lambda e: -abs(e.effect)))}

    def line(e, sign=0, rank=0):
        s = sign if sign else e.sign
        r = rank if rank else rankOf[e.label]
        return f"TERM: {e.label} | SIGN: {'+' if s > 0 else '-'} | LAG: {e.lag} | RANK: {r}"

    good = "\n".join(line(e) for e in eff) + "\nEXPLANATION: Last week's deaths carry the forecast; icu patients a week earlier add to it."
    ex1 = Explainer(MockLLMClient([good]), cfg)
    e1 = ex1.explain(cand, sc)
    assert len(e1.claims) == 4 and e1.coverage == 1.0 and e1.faithfulness == 1.0 and e1.text.startswith("Last week")
    assert "Formula:" in ex1.lastPrompt and "standardized effect" in ex1.lastPrompt and ex1.calls == 1
    r1, r2 = rankOf[eff[0].label], rankOf[eff[1].label]
    bad = "\n".join([line(eff[0], rank=r2), line(eff[1], rank=r1), line(eff[2], sign=-eff[2].sign)]) + "\nEXPLANATION: a narrative."
    e2 = Explainer(MockLLMClient([bad]), cfg).explain(cand, sc)
    assert len(e2.claims) == 3 and e2.coverage == 0.75
    assert e2.signRate == pytest.approx(2 / 3) and e2.lagRate == 1.0 and e2.rankRate == pytest.approx(1 / 3) and e2.faithfulness == 0.0
    assert "coverage" in e2.tableRow and all(str(c) for c in e2.claims)
    ex3 = Explainer(MockLLMClient([good.replace("icu", "x1").replace("pRate", "x2")]), cfg, named=False)
    e3 = ex3.explain(cand, sc)
    assert "icu" not in ex3.lastPrompt and "x1~1" in ex3.lastPrompt
    assert len(e3.claims) == 4 and e3.faithfulness == 1.0 and "icu" in e3.text
    base = Candidate(None, 1.0, [], 0, label="RandomWalk")
    assert ex1.explain(base, sc).nTerms == 0
    assert Explainer.variableOf("log1p(icu~2)") == "icu" and Explainer.variableOf("log(? + pRate~1)") == "pRate"
    assert Explainer.variableOf("hinge(temp~1, 65.8)") == "temp"


def test_llm_clients_offline(tmp_path):
    mock = MockLLMClient(["y ~= y~(3,1) + hosp~1", "y ~= y~(5,1) + wave(2)"])
    assert mock.complete("s", "u1") == "y ~= y~(3,1) + hosp~1" and mock.complete("s", "u2") == "y ~= y~(5,1) + wave(2)"
    assert mock.complete("s", "u3") == "y ~= y~(3,1) + hosp~1", "cycling"

    class Counting(LLMClient):
        calls = 0

        def complete(self, system, user):
            self.calls += 1
            return f"response-{self.calls}"

    inner = Counting()
    cached = CachedClient(inner, str(tmp_path))
    a = cached.complete("sys", "prompt-A")
    b = cached.complete("sys", "prompt-A")
    c = cached.complete("sys", "prompt-B")
    assert a == b and inner.calls == 2 and c == "response-2"
    cachedB = CachedClient(inner, str(tmp_path), tag="other-model@0.8")
    d = cachedB.complete("sys", "prompt-A")
    assert inner.calls == 3 and d == "response-3" and d != a, "a different model tag must not share entries"
    assert len(cached.keyOf("sys", "prompt-A")) == 64 and cached.keyOf("a", "bc") != cached.keyOf("ab", "c")
    counter = CountingClient(cached, priceIn=1.0, priceOut=4.0)
    counter.complete("sys", "prompt-A")
    counter.complete("sys", "prompt-C")
    assert counter.calls == 2 and inner.calls == 4 and counter.chars > 0 and counter.tokens == counter.chars // 4 and counter.cost > 0.0
    sysP = 'grammar:\n  y ~= y~1 + "quoted" \\ backslash\ttab'
    body = Json.chatBody("gpt-4o-mini", 0.8, sysP, "user text")
    assert Json.string(body, "content") == sysP
    reply = ('{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"y ~= y~(3,1) + x1~2\\nTFORM: Log1p\\n\\u00e9"},'
             '"finish_reason":"stop"}],"usage":{"prompt_tokens":10}}')
    assert Json.string(reply, "content") == "y ~= y~(3,1) + x1~2\nTFORM: Log1p\né"
    assert Json.string('{"type":"message","content":[{"type":"text","text":"hello"}]}', "text") == "hello"
    with pytest.raises(RuntimeError):
        Json.string('{"a": 1}', "content")
    assert LocalClient("http://127.0.0.1:9/v1/chat/completions").reachable(probeMs=200) is False
