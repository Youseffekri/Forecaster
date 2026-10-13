"""Tests of the fit-and-score engine (counterparts of formulaScorerTest, formulaScorerTest4/6/7)."""
import math

import numpy as np
import pytest

from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.datasets import Datasets


def planted():
    rng = np.random.default_rng(3)
    n = 300
    z = rng.uniform(0.5, 4.0, n)
    y = np.zeros(n)
    y[0] = 10.0
    for t in range(1, n):
        y[t] = 5 + 0.6 * y[t - 1] + 4 * np.log(z[t - 1] + 0.3) + 0.2 * rng.standard_normal()
    return {"y": y, "z": z}


def test_exact_linear_series():
    """formulaScorerTest (the rev4.1 log, Oct 13): y = 10 + 2 t, y ~= y~1, near-OLS ridge, in-sample J; every
    horizon h is scored on the rows whose targets y(t+h-1) lie in the fit rows (ScalaTion's diagnoseH)."""
    y = np.array([10.0 + 2.0 * t for t in range(40)])
    sc = FormulaScorer({"y": y}, 3, hp={"lambda": 1e-9, "valFrac": 0.0})
    mod = sc.fit("y ~= y~1")
    c = sc.score("y ~= y~1")
    assert sc.trRng == range(0, 31) and sc.teRng == range(31, 40) and sc.skip == 6
    assert np.allclose(c.smapeH, [0.212800, 0.202128, 0.193098], atol=5e-7)
    assert mod.intercept[0] == pytest.approx(1.717557253184502, abs=1e-9) and mod.slopes[0, 0] == pytest.approx(1.00572519080617, abs=1e-9)
    assert c.score == pytest.approx(0.23644833224898068, abs=1e-12) and c.nTerms == 1
    assert [q["m"] for q in c.qofTrain] == [25, 24, 23]                  # 31 - skip - (h - 1) rows per horizon


def test_rows_and_blocks():
    """ScalaTion's split (Model.trSize: te = floor(0.2 m + 1) = 61 of 300), a quarter of the training
    rows as three rolling-origin validation blocks."""
    sc = FormulaScorer(planted(), hh=3)
    assert sc.trSize == 239 and sc.teSize == 61 and sc.nVal == 60 and sc.fitSize == 179
    assert [b.start for b in sc.blocks] == [179, 199, 219] and sc.blocks[-1].stop == 239
    assert sc.trRng == range(0, 239) and sc.teRng == range(239, 300) and sc.fitRng == range(0, 179) and sc.valRng == range(179, 239)
    sc0 = FormulaScorer(planted(), hh=3, hp={"valFrac": 0.0})
    assert sc0.nVal == 0 and sc0.fitSize == sc0.trSize and sc0.blocks == []


def test_fit_cache_and_scala_names():
    """formulaScorerTest6: a second parse of the same structure (another start value) hits the cache
    -- one candidate, one fit -- and the formula takes the fitted values; the Scala names are there."""
    sc = FormulaScorer(planted(), hh=1, hp={"lambda": 1e-6, "skip": 2, "valFrac": 0.0})
    f = sc.parse("y ~= y~1 + log(? + z~1)")
    mod = sc.fit(f)
    n = sc.evaluations
    other = sc.parse("y ~= log(?0.5 + z~1) + y~1")
    assert sc.fit(other) is mod and sc.evaluations == n, "the same structure with another start value must hit the cache"
    assert other.params()[0].value == pytest.approx(f.params()[0].value) and other.key == f.key
    assert FormulaScorer.keyOf(other) == f.key
    c = sc.score(f)
    assert c.nTerms == 2 and c.nFree == 1 and c.formulaStr == repr(c.formula) and c.proposalStr.count("?") == 1
    assert c.qofVal is not None and c.qofV == c.qofTrain and c.smapeVal == pytest.approx(c.smape)   # in-sample J: the validation QoF is the fit rows'
    assert np.allclose(c.scoreH, FormulaScorer.jOf(c.qofTrain, c.qofV)) and c.score == pytest.approx(sc.scoreOf(c.qofTrain))
    assert c.qofTrain[0]["dfr"] == 3.0, "two slopes + one free parameter = 3 degrees of freedom"


def test_free_parameter_formula_beats_the_fixed_one():
    """formulaScorerTest6: log(? + z~1) recovers the shift (h = 1, where it is planted) and scores
    better than the fixed log1p."""
    sc = FormulaScorer(planted(), hh=1, hp={"lambda": 1e-6, "skip": 2, "nFreqs": 1})
    free = sc.score("y ~= y~1 + log(? + z~1)")
    fixed = sc.score("y ~= y~1 + log1p(z~1)")
    assert not free.failed and not fixed.failed
    assert free.nFree == 1 and free.formula.params()[0].value == pytest.approx(0.3, abs=0.2)   # a shift inside a log is weakly identified
    assert free.score < fixed.score
    assert free.key == "y ~= log(? + z~1) + y~1"


def test_rejections_and_caps():
    sc = FormulaScorer(planted(), hh=3, hp={"skip": 2, "maxFree": 1})
    c = sc.score("y ~= y~1 + log(? + z~1) + pow(z~1, ?)")
    assert c.failed and "2 free parameters, at most 1 allowed" in c.reason
    c = sc.score("y ~= y~1 + log(? + w~1)")
    assert c.failed and "variable 'w' not in dataset" in c.reason
    c = sc.score("y ~= y~1 + pow(z~1, ?0.99)")           # an exponent fitted near 1 is a plain lag ...
    assert c.failed and ("plain lag" in c.reason or "bound" in c.reason or "collinear" in c.reason)


def test_j_is_fpe_adjusted_validation_smape():
    sc = FormulaScorer(planted(), hh=3, hp={"skip": 2})
    c = sc.score("y ~= y~(2,1) + log1p(z~1)")
    for h in range(3):
        q, qv = c.qofTrain[h], c.qofV[h]
        assert q["dfr"] == 3
        assert c.scoreH[h] == pytest.approx(qv["smape"] * (1 + 2 * (3 + 1) / q["m"]))
    assert c.score == pytest.approx(float(c.scoreH.mean()))


def test_covid_baseline_and_tnt():
    """The Covid-19 baseline (18 lag terms) scores and rolls over the testing rows."""
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6)
    c = sc.score(cfg.baselineFormulaStr(6))
    assert not c.failed and c.nTerms == 18 and 10 < c.score < 60
    mod, qof = sc.tnt(c.formula.fresh())
    assert len(qof) == 6 and all(math.isfinite(q["smape"]) for q in qof)
    assert mod.X.shape == (116, 18)
    assert sc.evaluations >= 4                               # 1 fit + 2 folds + the final model
