"""Tests of the fit-and-score engine (counterparts of formulaScorerTest4/6/7)."""
import math

import numpy as np
import pytest

from modeling.forecasting.llm_narx_sr.formula_scorer import FormulaScorer, Candidate
from modeling.forecasting.llm_narx_sr.datasets import Datasets, Example_Covid


def planted():
    rng = np.random.default_rng(3)
    n = 300
    z = rng.uniform(0.5, 4.0, n)
    y = np.zeros(n)
    y[0] = 10.0
    for t in range(1, n):
        y[t] = 5 + 0.6 * y[t - 1] + 4 * np.log(z[t - 1] + 0.3) + 0.2 * rng.standard_normal()
    return {"y": y, "z": z}


def test_rows_and_blocks():
    sc = FormulaScorer(planted(), hh=3)
    assert sc.tr_size == 240 and sc.n_val == 60 and sc.fit_size == 180
    assert [b.start for b in sc.blocks] == [180, 200, 220] and sc.blocks[-1].stop == 240
    sc0 = FormulaScorer(planted(), hh=3, hp={"valFrac": 0.0})
    assert sc0.n_val == 0 and sc0.fit_size == sc0.tr_size and sc0.blocks == []


def test_free_parameter_formula_beats_the_fixed_one():
    """formulaScorerTest6: log(? + z~1) recovers the shift (h = 1, where it is planted) and scores
    better than the fixed log1p."""
    sc = FormulaScorer(planted(), hh=1, hp={"lambda": 1e-6, "skip": 2, "nFreqs": 1})
    free = sc.score("y ~= y~1 + log(? + z~1)")
    fixed = sc.score("y ~= y~1 + log1p(z~1)")
    assert not free.failed and not fixed.failed
    assert free.n_free == 1 and free.formula.params()[0].value == pytest.approx(0.3, abs=0.2)   # a shift inside a log is weakly identified
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
        q, qv = c.qof_train[h], c.qof_v[h]
        assert q["dfr"] == 3
        assert c.score_h[h] == pytest.approx(qv["SMAPE"] * (1 + 2 * (3 + 1) / q["m"]))
    assert c.score == pytest.approx(float(c.score_h.mean()))


def test_covid_baseline_and_tnt():
    """The Covid-19 baseline (18 lag terms) scores and rolls over the testing rows."""
    cfg = Datasets.COVID2
    sc = FormulaScorer(cfg.load(), hh=6)
    c = sc.score(cfg.baseline_formula_str(6))
    assert not c.failed and c.n_terms == 18 and 10 < c.score < 60
    mod, qof = sc.tnt(c.formula.fresh())
    assert len(qof) == 6 and all(math.isfinite(q["SMAPE"]) for q in qof)
    assert mod.X.shape == (116, 18)
    assert sc.evaluations >= 4                               # 1 fit + 2 folds + the final model
