"""Tests of the fit-and-score engine (counterparts of formulaScorerTest, formulaScorerTest4/6/7)."""
import math

import numpy as np
import pytest

from modeling.forecasting.forecaster import Model
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
    """formulaScorerTest (rev5.3 / rev2.2): y = 10 + 2 t, y ~= y~1, near-OLS ridge, in-sample J; the candidate is
    fitted AND diagnosed on the FS rows of the training rows (0 until 31 - hh + 1 = 29: the boundary rule, as
    `Forecaster_D.fitCols`), every horizon h on the rows whose targets y(t+h-1) lie in them (ScalaTion's qofOn)."""
    y = np.array([10.0 + 2.0 * t for t in range(40)])
    sc = FormulaScorer({"y": y}, 3, hp={"lambda": 1e-9}, validating=False)
    mod = sc.fit("y ~= y~1")
    c = sc.score("y ~= y~1")
    assert sc.trRng == range(0, 31) and sc.teRng == range(31, 40) and sc.skip == 6 and sc.fs_rng(sc.trRng) == range(0, 29)
    assert np.allclose(c.smapeH, [0.2267184, 0.2150100, 0.2051937], atol=5e-7)
    assert mod.intercept[0] == pytest.approx(1.6963123657217238, abs=1e-9) and mod.slopes[0, 0] == pytest.approx(1.0065075921549427, abs=1e-9)
    assert c.score == pytest.approx(0.2548428618706658, abs=1e-12) and c.nTerms == 1
    assert [q["m"] for q in c.qofTrain] == [23, 22, 21]                  # 29 - skip - (h - 1) rows per horizon (the FS rows)
    assert c.hp == {"lambda": 1e-9} and c.lam == 1e-9                     # the knobs of the fit (the scorer's penalty)
    assert mod.modelName.endswith("_L1.0E-9")                             # the tag of a small penalty: all its digits


def test_rows_and_the_switch():
    """`Model`'s split (te = floor(0.2 m + 1) = 61 of 300; the validation block the same size, 61, so
    fit = 178) and the one switch: a scorer reads `Model.validating` at construction or takes it as
    an argument; a stale hp entry (valFrac, valFolds, rc, skip, hSel) is refused."""
    sc = FormulaScorer(planted(), hh=3, validating=True)
    assert sc.trSize == 239 and sc.teSize == 61 and sc.valSize == 61 and sc.fitSize == 178 and sc.validating
    assert sc.trRng == range(0, 239) and sc.teRng == range(239, 300) and sc.fitRng == range(0, 178) and sc.valRng == range(178, 239)
    assert sc.splitStr == "178 / 61 / 61" and sc.rc == 5 and sc.skip == 6
    sc0 = FormulaScorer(planted(), hh=3)                                  # Model.validating is False by default
    assert not sc0.validating and sc0.valSize == 0 and sc0.fitSize == sc0.trSize and sc0.splitStr == "239 / 0 / 61"
    with Model.withValidation(0.1):
        sc1 = FormulaScorer(planted(), hh=3)
        assert sc1.validating and sc1.valSize == 31 and sc1.fitSize == 208
    with pytest.raises(ValueError, match="unknown hp"):
        FormulaScorer(planted(), hh=3, hp={"valFrac": 0.0})
    with pytest.raises(ValueError, match="unknown hp"):
        FormulaScorer(planted(), hh=3, hp={"skip": 2})


def test_fit_cache_and_scala_names():
    """formulaScorerTest6: a second parse of the same structure (another start value) hits the cache
    -- one candidate, one fit -- and the formula takes the fitted values; the Scala names are there."""
    sc = FormulaScorer(planted(), hh=1, hp={"lambda": 1e-6, "maxLag": 2}, validating=False)
    f = sc.parse("y ~= y~1 + log(? + z~1)")
    mod = sc.fit(f)
    n = sc.evaluations
    other = sc.parse("y ~= log(?0.5 + z~1) + y~1")
    assert sc.fit(other) is mod and sc.evaluations == n, "the same structure with another start value must hit the cache"
    assert other.params()[0].value == pytest.approx(f.params()[0].value) and other.key == f.key
    assert FormulaScorer.keyOf(other) == f.key
    c = sc.score(f)
    assert c.nTerms == 2 and c.nFree == 1 and c.formulaStr == repr(c.formula) and c.proposalStr.count("?") == 1
    assert c.qofVal is None and c.qofV == c.qofTrain and c.smapeVal == pytest.approx(c.smape)   # in-sample J: no validation QoF
    assert c.score == pytest.approx(sc.scoreOf(c.qofTrain, nParams=c.nParams)) and c.score == pytest.approx(float(c.scoreH.mean()))
    assert c.nParams is not None and 1.0 < float(c.nParams[0]) < c.nTerms + 1.0      # a ridge: 1 + tr H_lambda < the columns + 1
    assert c.qofTrain[0]["dfr"] == 3.0, "two slopes + one free parameter = 3 degrees of freedom"


def test_free_parameter_formula_beats_the_fixed_one():
    """formulaScorerTest6: log(? + z~1) recovers the shift (h = 1, where it is planted) and scores
    better than the fixed log1p."""
    sc = FormulaScorer(planted(), hh=1, hp={"lambda": 1e-6, "maxLag": 2, "nFreqs": 1}, validating=True)
    free = sc.score("y ~= y~1 + log(? + z~1)")
    fixed = sc.score("y ~= y~1 + log1p(z~1)")
    assert not free.failed and not fixed.failed
    assert free.nFree == 1 and free.formula.params()[0].value == pytest.approx(0.3, abs=0.2)   # a shift inside a log is weakly identified
    assert free.score < fixed.score
    assert free.key == "y ~= log(? + z~1) + y~1"


def test_rejections_and_caps():
    sc = FormulaScorer(planted(), hh=3, hp={"maxLag": 2, "maxFree": 1})
    c = sc.score("y ~= y~1 + log(? + z~1) + pow(z~1, ?)")
    assert c.failed and "2 free parameters, at most 1 allowed" in c.reason
    c = sc.score("y ~= y~1 + log(? + w~1)")
    assert c.failed and "variable 'w' not in dataset" in c.reason
    c = sc.score("y ~= y~1 + pow(z~1, ?0.99)")           # an exponent fitted near 1 is a plain lag ...
    assert c.failed and ("plain lag" in c.reason or "bound" in c.reason or "collinear" in c.reason)


def test_j_is_the_fpe_form_in_sample_and_the_plain_validation_smape_otherwise():
    """J = Forecaster.scoreOf: in-sample the mean sMAPE times (1 + 2 p_h / n_h) with p_h the ridge's
    effective parameters (1 + tr H_lambda < dfr + 1), on the validation rows (a fresh copy rolled by
    `rollTest`, refit every rc rows) the plain mean sMAPE."""
    sc = FormulaScorer(planted(), hh=3, hp={"maxLag": 2}, validating=False)
    c = sc.score("y ~= y~(2,1) + log1p(z~1)")
    p = float(c.nParams[0])
    assert 1.0 < p < 4.0
    for h in range(3):
        q = c.qofTrain[h]
        assert q["dfr"] == 3 and c.scoreH[h] == pytest.approx(q["smape"] * (1 + 2 * p / q["m"]))
    assert c.qofVal is None and c.score == pytest.approx(float(c.scoreH.mean()))
    scv = FormulaScorer(planted(), hh=3, hp={"maxLag": 2}, validating=True)
    cv = scv.score("y ~= y~(2,1) + log1p(z~1)")
    assert cv.qofVal is not None and len(cv.qofVal) == 3 and cv.qofTrain[0]["m"] == len(scv.fs_rng(scv.fitRng)) - scv.skip   # the FS rows of the 178 fit rows
    assert np.allclose(cv.scoreH, [q["smape"] for q in cv.qofVal]) and cv.score == pytest.approx(cv.smapeVal)
    assert cv.qofVal[0]["m"] == scv.valSize and cv.qofVal[2]["m"] == scv.valSize - 2    # the block's rows per horizon
    mod = scv.build(scv.fresh(cv.formula), scv.fitSize)                # the validation pass is rollTest
    direct = scv._qofList(mod.rollTest(scv.valRng, scv.rc, verbose=False), mod)
    assert np.allclose([q["smape"] for q in direct], cv.smapeValH)


def test_covid_baseline_and_tnt():
    """The Covid-19 baseline (18 lag terms) scores and rolls over the testing rows."""
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6)
    c = sc.score(cfg.baselineFormulaStr(6))
    assert not c.failed and c.nTerms == 18 and 10 < c.score < 60
    mod, qof = sc.tnt(c.formula.fresh())
    assert len(qof) == 6 and all(math.isfinite(q["smape"]) for q in qof)
    assert mod.X.shape == (116, 18)
    assert sc.evaluations >= 2                               # the fit on the fit rows + the TnT pass
    sct = FormulaScorer(cfg.load(), hh=6, validating=True)
    ct = sct.score(cfg.baselineFormulaStr(6))
    assert not ct.failed and ct.qofVal is not None and ct.score == pytest.approx(ct.smapeVal) and sct.splitStr == "68 / 24 / 24"
