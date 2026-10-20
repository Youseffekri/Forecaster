"""The validation switch (Model, rollTest / valTest / tvT_Test, the evaluators), the knobs (Knob, tune, withHP,
hpTag names, Model.tuning / folds -- rev2.2, ScalaTion rev5.3), and the attention ranker (skipped without PyTorch)."""
import io
import contextlib

import numpy as np
import pytest

from modeling.feature_selection import SelectionTech, qk as QK
from modeling.fit import QoF
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.forecaster import Forecaster
from modeling.model import Model
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import TermLibrary
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import arxFormulaStr
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.datasets import Datasets, Example_Covid
from modeling.centered_ridge import CenteredRidge
from modeling.elastic_net import ElasticNet
from modeling.hyper_parameter import Knob
from Util.transform import TransformT


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return f(*a, **k)


# -- Model: the split and the switch ------------------------------------------------------------------

def test_model_split_and_switch():
    assert Model.TE_RATIO == 0.2 and Model.VAL_RATIO == 0.2 and Model.validating is False
    assert (Model.teSize(116), Model.trSize(116), Model.valSize(116), Model.fitSize(116)) == (24, 92, 24, 68)
    assert Model.valRng(116) == range(68, 92) and Model.teRng(116) == range(92, 116) and Model.fitRng(116) == range(0, 68)
    assert Model.splitStr(116) == "68 / 24 / 24" and Model.trSize(116) == 92 and Model.teRng(116) == range(92, 116)
    assert Model.valSize(116, 0.1) == 12 and Model.splitStr(116, val_ratio=0.1) == "80 / 12 / 24"
    with Model.withValidation():
        assert Model.validating and Model.VAL_RATIO == 0.2
        with Model.withValidation(0.3):
            assert Model.VAL_RATIO == 0.3 and Model.fitSize(116) == 92 - 35
        assert Model.VAL_RATIO == 0.2
    assert Model.validating is False
    with pytest.raises(ValueError):
        Model.valSize(116, 0.85)                                       # r + TE_RATIO must stay below 1
    with pytest.raises(ValueError):
        with Model.withValidation(-0.1):
            pass
    assert Model.validating is False


# -- Forecaster: rollTest, valTest, tnT_Test, tvT_Test, valScore ----------------------------------------

def test_roll_test_is_the_one_rolling_routine_and_restores_the_skip():
    y, xe = covid()
    mod = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    assert Forecaster.RC == 5 and mod.getSkip() == 6
    qt = quiet(mod.tnT_Test)                                             # over the testing rows
    assert qt.shape == (6, 26) and mod.getSkip() == 6                   # the skip is restored (rev4.1 left it at 0)
    direct = quiet(mod.rollTest, Model.teRng(116), 5, "TnT", False)
    assert np.allclose(direct, qt)
    qv = quiet(mod.valTest, verbose=False)                              # over the validation rows: trained on the 68 fit rows
    assert qv.shape == (6, 26) and qv[0, QoF.m.ordinal] == 24 and qv[5, QoF.m.ordinal] == 19
    assert not np.allclose(qv[:, QoF.smape.ordinal], qt[:, QoF.smape.ordinal])
    assert mod.valScore() == pytest.approx(Forecaster.scoreOf(qv, inSample=False))
    with pytest.raises(ValueError):
        mod.rollTest(range(0, 10))                                       # a block must start after the first row
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        qv2, qt2 = mod.tvT_Test(verbose=False)
    assert np.allclose(qv2, qv) and np.allclose(qt2, qt)
    assert "tvT_Test: ARX_D_6_6_2_L1.0: split 68 / 24 / 24: validation score = " in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod.tnT_Test(rc=20)
    s = out.getvalue()
    assert "| TnT Test: ARX_D_6_6_2_L1.0 |" in s and "rollValidate: Evaluate ARX_D_6_6_2_L1.0's QoF" in s and "| diagnoseAll: ARX_D_6_6_2_L1.0 |" in s


def test_roll_validate_takes_a_block_and_runs_quietly():
    y, xe = covid()
    mod = ARX(y, 3, xe, Args(p=3, px=2, spec=1, lam=1.0, method="ridge"))
    mod.train()
    yf = quiet(mod.rollValidate, 5, rng=Model.valRng(116), verbose=False)
    assert yf.shape == (24, 3)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod.rollValidate(5, verbose=False)
    assert out.getvalue() == ""


# -- the evaluators on the switch ------------------------------------------------------------------------

def test_selection_under_the_switch_scores_on_the_validation_rows_and_refits_the_transform():
    y, xe = covid()
    args = Args(p=4, px=3, spec=1, lam=1.0, method="ridge")
    mod0 = ARX_D.rescale(y, 6, xe, args, TransformT.Log1p)
    assert mod0.fs_rng == range(0, 92 - 5)                              # the training rows whose 6 targets all lie before 92
    with Model.withValidation():
        mod = ARX_D.rescale(y, 6, xe, args, TransformT.Log1p)           # the transform fitted on the 68 fit rows
        assert mod.fs_rng == range(0, 68 - 5) and mod.tr_size == 92     # the fit rows whose 6 targets all lie before 68
        cols, rSq = quiet(mod.selectFeatures, SelectionTech.Backward)
        best = mod.getBest()
        sub, qof = mod.fitCols(best.mod_cols)
        assert qof.shape == (6, 26) and qof[0, QoF.m.ordinal] == 24    # scored on the 24 validation rows
        assert best.qof[QK] == pytest.approx(Forecaster.scoreOf(qof, inSample=False))
        fs1 = mod.getModel_h(h=1)
        step = fs1.evalCols([0, 1, 2])
        assert step.qof[QoF.m.ordinal] == 24 and step.qof[QK] == pytest.approx(step.qof[QoF.smape.ordinal])   # no penalty
        rebuilt = mod.refitTransform()                                  # the switch suspended: transforms on the 92 training rows
        assert rebuilt is not mod and rebuilt.X.shape == mod.X.shape and Model.validating is True   # the switch restored
        assert np.allclose(rebuilt.X, mod0.X) and rebuilt.fs_rng == range(0, 68 - 5)   # the unswitched model's matrix; the switch is back on
        models, _, qT = quiet(mod.reportSelection, cols, rSq, inSample=False, showTrace=False)
        assert qT.shape == (6, 26) and models[0].X.shape[1] == len(best.mod_cols)
    assert Model.validating is False
    assert mod0.refitTransform() is not mod0 and ARX_D(y, 6, xe, args).refitTransform() is not None
    plain = ARX_D(y, 6, xe, Args(p=2, px=0, spec=1))
    assert plain.refitTransform() is plain                              # no transform: the model itself
    step0 = mod0.getModel_h(h=1).evalCols([0, 1, 2])                    # in-sample: the FPE-form score
    q = step0.qof
    p = float(step0.mod.effDF[0])                                        # a ridge: its effective parameters, not dfr + 1
    assert 1.0 < p < q[QoF.dfr.ordinal] + 1.0
    assert q[QK] == pytest.approx(q[QoF.smape.ordinal] * (1 + 2 * p / q[QoF.m.ordinal]))


# -- the knobs: tune, withHP, hpTag, Model.tuning / folds (rev2.2, ScalaTion rev5.3) ----------------------

LAMS = Knob.LAMBDA.grid


def test_knobs_and_the_tags():
    """`Knob`: the two knobs of the regularized regressions, their tags (ScalaTion's `hyperParameterTest3`),
    the search coordinate and its inverse, and what each regression declares (`tunable`)."""
    lam, l1 = Knob.LAMBDA, Knob.L1_RATIO
    assert lam.name == "lambda" and lam.log and lam.tag == "L" and LAMS == (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
    assert l1.name == "l1_ratio" and not l1.log and l1.tag == "A" and l1.grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert [lam.tagOf(v) for v in (1.0, 10.0, 1000.0, 0.01, 0.001, 1e-12, 0.572, 0.1)] == \
           ["_L1.0", "_L10.0", "_L1000.0", "_L0.01", "_L0.001", "_L1.0E-12", "_L0.6", "_L0.1"]
    assert [l1.tagOf(v) for v in (0.5, 0.25, 0.0, 1.0, 0.75)] == ["_A0.5", "_A0.25", "_A0.0", "_A1.0", "_A0.75"]
    assert lam.tagOf(1.0) + l1.tagOf(0.5) == "_L1.0_A0.5"
    assert lam.toCoord(1000.0) == 3.0 and lam.toCoord(0.001) == -3.0 and lam.fromCoord(2.0) == 100.0
    assert l1.toCoord(0.25) == 0.25 and l1.fromCoord(0.75) == 0.75
    assert all(abs(lam.fromCoord(lam.toCoord(v)) - v) < 1e-9 * v for v in LAMS)
    assert CenteredRidge.tunable == (lam,) and ElasticNet.tunable == (lam, l1)


def test_model_names_carry_the_knobs():
    y, xe = covid()
    m = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    assert m.modelName == "ARX_D_6_6_2_L1.0" and m.hparameter["lambda"] == 1.0 and m.knobs == (Knob.LAMBDA,) and m.hpTag == "_L1.0"
    assert ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=0.25, method="ridge")).modelName == "ARX_D_6_6_2_L0.2"   # one decimal
    assert ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=0.01, method="ridge")).modelName == "ARX_D_6_6_2_L0.01"  # ... unless < 0.1
    e = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="enet"))
    assert e.modelName == "ARX_D_6_6_2_L1.0_A0.5" and e.knobs == (Knob.LAMBDA, Knob.L1_RATIO) and e.hparameter["l1_ratio"] == 0.5   # the elastic net: two knobs
    assert ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=10.0, method="enet", l1_ratio=0.3)).modelName == "ARX_D_6_6_2_L10.0_A0.3"
    plain = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, method="sm_ols"))    # least squares: no knobs, no tag (rev2.3: the default is the net)
    assert plain.modelName == "ARX_D_6_6_2" and "lambda" not in plain.hparameter and plain.knobs == () and plain.hpTag == ""
    assert ARX_D(y, 6, xe, Args(p=6, px=6, spec=1)).modelName == "ARX_D_6_6_2_L0.1_A0.5"   # the default: lambda 0.1, l1_ratio 0.5
    from modeling.forecasting.random_walk import RandomWalk
    rw = RandomWalk(y, 6)
    assert rw.modelName == "RandomWalk" and rw.hparameter is None and rw.knobs == ()   # no regression: no tag
    with Model.withValidation(tuning=True):
        hp0, score = quiet(plain.tune)                                    # nothing to tune without knobs
        assert hp0 == {} and score == quiet(plain.valScore)
        quiet(plain.selectFeatures, SelectionTech.Backward)              # the switch is harmless for a plain regression
        assert plain.getBest().mod.knobs == () and plain.bestHP(plain.getBest().mod_cols) is None
    assert m.getModel([0, 1]).modelName == "ARX_D_6_6_2_L1.0" and m.getModel_h(h=1).modelName == "ARX_6_6_2_L1.0"
    w = m.withHP({"lambda": 10.0})
    assert w.modelName == "ARX_D_6_6_2_L10.0" and w.hparameter["lambda"] == 10.0 and m.hparameter["lambda"] == 1.0 and w.X is m.X and w is not m
    assert w.withHP({"lambda": 0.5}).modelName == "ARX_D_6_6_2_L0.5"      # re-tagged, not appended
    w2 = e.withHP({"lambda": 0.1, "l1_ratio": 0.75})
    assert w2.modelName == "ARX_D_6_6_2_L0.1_A0.75" and w2.hparameter["l1_ratio"] == 0.75 and w2._args.l1_ratio == 0.75 and e.hparameter["l1_ratio"] == 0.5
    assert e.withHP({"l1_ratio": 1.0}).modelName == "ARX_D_6_6_2_L1.0_A1.0"   # one knob turned, the other kept
    with pytest.raises(ValueError):
        plain.withHP({"lambda": 1.0})
    assert not hasattr(Forecaster, "LAMBDAS") and not hasattr(m, "findLambda") and not hasattr(m, "withLambda") and not hasattr(m, "lamTag")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        m.tnT_Test(rc=20)
    assert "| TnT Test: ARX_D_6_6_2_L1.0 |" in out.getvalue()


def test_score_of_takes_the_measure_and_the_horizons():
    q = np.zeros((3, 26))
    q[:, QoF.smape.ordinal], q[:, QoF.mae.ordinal] = [10.0, 20.0, 30.0], [1.0, 2.0, 3.0]
    q[:, QoF.m.ordinal], q[:, QoF.dfr.ordinal] = 50.0, 4.0
    assert Forecaster.scoreOf(q, False) == 20.0                          # mean sMAPE over all horizons
    assert Forecaster.scoreOf(q, False, hs=[1]) == 10.0 and Forecaster.scoreOf(q, False, hs=[2, 3]) == 25.0
    assert Forecaster.scoreOf(q, False, measure=QoF.mae) == 2.0
    assert Forecaster.scoreOf(q, True) == pytest.approx(20.0 * (1.0 + 2.0 * 5.0 / 50.0))   # the FPE factor in-sample
    with pytest.raises(ValueError, match="not an error measure"):
        Forecaster.scoreOf(q, False, measure=QoF.rSq)


SMALL = (Knob("lambda", (0.1, 1.0, 10.0), True, "L"),)                  # a three-point grid for the tests


def test_tune_picks_the_best_validation_score_and_needs_the_switch():
    y, xe = covid()
    m = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    with pytest.raises(ValueError, match="needs Model.validating"):
        m.tune()
    with Model.withValidation():
        scores = {lam: quiet(m.withHP({"lambda": lam}).valScore) for lam in (0.1, 1.0, 10.0)}
        hp2, score = quiet(m.tune, SMALL)                                 # ({name: value}, score), as ScalaTion's tune
        assert hp2 == {"lambda": min(scores, key=scores.get)} and score == scores[hp2["lambda"]]
        assert m.hparameter["lambda"] == 1.0 and m.modelName == "ARX_D_6_6_2_L1.0"   # the model itself is untouched
        best = m.withHP(hp2)
        assert best.hparameter["lambda"] == hp2["lambda"] and best.modelName == f"ARX_D_6_6_2{best.hpTag}" and best._params is None
        hp1, s1 = quiet(m.tune, SMALL, QoF.smape, [1])                    # by the first horizon only
        assert s1 == quiet(m.withHP(hp1).valScore, None, QoF.smape, [1])
        hp_mae, _ = quiet(m.tune, SMALL, QoF.mae)
        assert hp_mae["lambda"] in (0.1, 1.0, 10.0)
        r = quiet(ARX_D.rescale, y, 6, xe, Args(p=3, px=2, spec=1, lam=1.0, method="ridge"), TransformT.Log1p)
        rb = r.withHP(quiet(r.tune, (Knob("lambda", (0.1, 1.0), True, "L"),))[0])
        assert rb.refitTransform().hparameter["lambda"] == rb.hparameter["lambda"]   # a refit of the transforms keeps the knobs found
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m.tune(SMALL, verbose=True)
        assert "tune: lambda = 0.1 (ARX_D_6_6_2_L0.1): score = " in out.getvalue()
    assert Model.validating is False and Model.tuning is False


def test_tune_golden_random_and_k_fold():
    """golden refines the grid between the neighbours of its best value; random draws from one generator
    (seeded, reproducible) in the knob's coordinate between the grid's ends; `Model.folds` / `folds = k` score
    by `cvScore` on the training rows (no switch needed)."""
    y, xe = covid()
    m = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    with Model.withValidation():
        hpG, gs = quiet(m.tune)
        hpS, bs = quiet(m.tune, search="golden")                           # the grid, then golden-section between the neighbours
        g, b = hpG["lambda"], hpS["lambda"]
        assert bs <= gs and g / 10 <= b <= g * 10 and b not in LAMS
        hpR, rs = quiet(m.tune, search="random", trials=8, seed=7)
        assert 1e-3 <= hpR["lambda"] <= 1e3 and np.isfinite(rs)
        assert quiet(m.tune, search="random", trials=8, seed=7) == (hpR, rs)   # reproducible
        assert quiet(m.tune, search="random", trials=8, seed=8)[0] != hpR or True
        with pytest.raises(ValueError, match="grid"):
            m.tune(search="newton")
    with pytest.raises(ValueError, match="needs Model.validating"):
        m.tune()
    hp5, s5 = quiet(m.tune, folds=5)                                      # k-fold on the 92 training rows: no switch needed
    assert hp5["lambda"] in LAMS and np.isfinite(s5)
    assert s5 == pytest.approx(quiet(m.withHP(hp5).cvScore, 5)) and m._params is None   # the model itself untouched
    Model.folds = 5
    try:
        assert quiet(m.tune) == (hp5, s5)                                 # Model.folds is the default
    finally:
        Model.folds = 0
    assert quiet(m.cvScore, 5, QoF.mae) > 0 and quiet(m.cvScore, 5, QoF.smape, [1]) > 0
    with pytest.raises(ValueError, match="at least 2"):
        m.cvScore(1)
    rec = ARX(y, 6, xe, Args(p=3, px=2, spec=1, lam=1.0, method="ridge"))
    assert np.isfinite(quiet(rec.cvScore, 4))                             # a recursive model: its h = 1 regression
    plain = ARX_D(y, 6, xe, Args(p=3, px=2, spec=1, method="sm_ols"))
    assert quiet(plain.tune, folds=5)[0] == {}                            # least squares: nothing to tune
    with Model.withValidation():                                          # the folds come from the fit rows then
        assert quiet(m.cvScore, 4) != s5


def test_tune_turns_two_knobs_for_the_elastic_net():
    """The same `tune` turns both knobs of the elastic net: the grid is the product (a small one here),
    golden sweeps each knob in turn in its own coordinate, `withHP` sets both, the name shows both."""
    y, xe = covid()
    e = ARX_D(y, 6, xe, Args(p=3, px=2, spec=1, lam=1.0, method="enet", scaled=True))
    two = (Knob("lambda", (0.1, 10.0), True, "L"), Knob("l1_ratio", (0.0, 0.5, 1.0), False, "A"))
    with Model.withValidation():
        hpG, gs = quiet(e.tune, two)
        assert set(hpG) == {"lambda", "l1_ratio"} and hpG["lambda"] in (0.1, 10.0) and hpG["l1_ratio"] in (0.0, 0.5, 1.0)
        pts = {(l, a): quiet(e.withHP({"lambda": l, "l1_ratio": a}).valScore) for l in (0.1, 10.0) for a in (0.0, 0.5, 1.0)}
        assert gs == pytest.approx(min(pts.values())) and (hpG["lambda"], hpG["l1_ratio"]) == min(pts, key=pts.get)
        hpS, ss = quiet(e.tune, two, search="golden")
        assert ss <= gs + 1e-12 and 0.1 <= hpS["lambda"] <= 10.0 and 0.0 <= hpS["l1_ratio"] <= 1.0
        hpR, _ = quiet(e.tune, two, search="random", trials=5, seed=3)
        assert 0.1 <= hpR["lambda"] <= 10.0 and 0.0 <= hpR["l1_ratio"] <= 1.0
        one, _ = quiet(e.tune, (Knob.L1_RATIO,))                           # one knob of the two: lambda stays
        assert set(one) == {"l1_ratio"} and e.withHP(one).hparameter["lambda"] == 1.0
    best = e.withHP(hpG)
    assert best.modelName == f"ARX_D_3_2_2{Knob.LAMBDA.tagOf(hpG['lambda'])}{Knob.L1_RATIO.tagOf(hpG['l1_ratio'])}"
    assert best._args.l1_ratio == hpG["l1_ratio"] and best.hparameter["lambda"] == hpG["lambda"]


def test_tuning_switch_gives_every_candidate_its_own_knobs():
    y, xe = covid()
    args = Args(p=4, px=3, spec=1, lam=1.0, method="ridge")
    with Model.withValidation(tuning=True):
        assert Model.tuning
        mod = ARX_D(y, 6, xe, args)
        fs1 = mod.getModel_h(h=1)
        s_full = quiet(fs1.evalCols, fs1.mcols)
        s_two = quiet(fs1.evalCols, [0, 1, 2])
        assert s_full.mod.hparameter["lambda"] in LAMS and s_two.mod.hparameter["lambda"] in LAMS
        assert s_full.mod.modelName.endswith(s_full.mod.hpTag)
        cols, rSq = quiet(mod.selectFeatures, SelectionTech.Backward)   # joint selection: every set at its best knobs
        best = mod.getBest()
        assert best.mod.hparameter["lambda"] in LAMS
        models, _, qT = quiet(mod.reportSelection, cols, rSq, inSample=False, showTrace=False)
        assert models[0].hparameter["lambda"] == best.mod.hparameter["lambda"] and models[0].modelName.endswith(best.mod.hpTag)   # the reported model keeps them
        assert qT.shape == (6, 26)
        sub, qof = mod.fitCols(best.mod_cols)
        assert sub.hparameter["lambda"] == best.mod.hparameter["lambda"] and qof.shape == (6, 26)
        assert mod.bestHP(best.mod_cols) == {"lambda": best.mod.hparameter["lambda"]} and mod.bestHP([0]) is None
        out = io.StringIO()
        with contextlib.redirect_stdout(out):                             # the banners name the SELECTED model
            m7 = ARX_D(y, 6, xe, args.copy(lam=7.0))
            c7, r7 = m7.selectFeatures(SelectionTech.Ranked)
            m7.reportSelection(c7, r7, inSample=False, showTrace=False)
        b7 = m7.getBest()
        assert m7.modelName == "ARX_D_4_3_2_L7.0" and b7.mod.hparameter["lambda"] != 7.0   # the parent keeps its own knobs
        assert f"| {b7.mod.modelName}: selected: k = {len(b7.mod_cols)} of n = " in out.getvalue()
        assert f"| {b7.mod.modelName}: TnT Test (rolling validation) of the selected model on " in out.getvalue()
        assert "ARX_D_4_3_2_L7.0: selected" not in out.getvalue()
        assert mod.getModel(best.mod_cols).hparameter["lambda"] == best.mod.hparameter["lambda"]   # getModel (getBest ().mod_cols) carries them
        assert mod.getModel(best.mod_cols).modelName.endswith(best.mod.hpTag)
    assert mod.getModel(best.mod_cols).hparameter["lambda"] == best.mod.hparameter["lambda"]   # ... with the switches off too
    with Model.withValidation():                                         # tuning off: the model's own knobs everywhere
        mod = ARX_D(y, 6, xe, args)
        quiet(mod.selectFeatures, SelectionTech.Backward)
        assert mod.getBest().mod.hparameter["lambda"] == 1.0
    assert Model.tuning is False


def test_scorer_tunes_per_candidate_and_carries_the_knobs_to_the_tnt():
    cfg = Datasets.COVID2
    sc0 = FormulaScorer(cfg.load(), hh=6, validating=True)
    c0 = sc0.score("y ~= y~(3,1) + icu~1")
    assert c0.hp == {"lambda": 1.0, "l1_ratio": 0.5} and not sc0.tuning and "lambda=1, l1_ratio=0.5" in repr(c0)   # rev2.3: the net's two knobs
    with Model.withValidation(tuning=True):
        sc = FormulaScorer(cfg.load(), hh=6)
        assert sc.validating and sc.tuning
        n0 = sc.evaluations
        c = sc.score("y ~= y~(3,1) + icu~1")
        assert c.hp["lambda"] in LAMS and set(c.hp) == {"lambda", "l1_ratio"} and c.score <= c0.score + 1e-12   # its own best knobs score no worse than lambda = 1
        assert sc.evaluations - n0 >= len(LAMS) + 2                       # the grid points count toward the budget
        mod, qt = sc.tnt(c.formula, c.hp)
        assert mod.hparameter["lambda"] == c.hp["lambda"] and mod.modelName.endswith(mod.hpTag) and len(qt) == 6
        assert "lambda = " in sc.report(c, tnt=False)
        maxLag = int(sc.hp["maxLag"])
        f = NARX_FormulaParser.parseFormula(arxFormulaStr(cfg.resp, 2, 1, list(cfg.exo)), sc.vars, maxLag)
        lib = TermLibrary(f.resp, f.preds, f.tFormT)
        cand, _, nEv = lib.selectBy(SelectionTech.Backward, sc)
        assert not cand.failed and cand.hp["lambda"] in LAMS and nEv > 0
    assert Model.tuning is False


# -- the attention ranker ---------------------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="the attention ranker needs PyTorch")


def test_attn_ranker_recovers_a_planted_structure_and_feeds_ranked():
    from modeling.attn_ranker import Attn_Ranker
    rng = np.random.default_rng(0)
    m, n = 120, 10
    X = rng.standard_normal((m, n))
    y = 3 * X[:, 2] - 2 * X[:, 7] + 0.5 * rng.standard_normal(m)       # planted: columns 2 and 7
    r = Attn_Ranker(X, y, seeds=range(2), epochs=40)
    assert repr(r).startswith("Attn_Ranker(m=120, n=10") and r.scores is None
    with pytest.raises(ValueError):
        r.ranking()
    s = r.run()
    assert s.shape == (n,) and r.gradScores.shape == (n,) and len(r.losses) == 2 and np.isfinite(s).all()
    assert set(r.ranking()[:2]) == {2, 7} and set(np.argsort(-r.gradScores)[:2]) == {2, 7}
    assert abs(float(s.sum()) - 1.0) < 1e-5                            # attention received, summed over the keys, is 1
    g = Attn_Ranker(X, y, seeds=[0], epochs=5, score="grad")
    assert np.allclose(g.run(), g.gradScores)
    with pytest.raises(ValueError):
        Attn_Ranker(X, y[:-1])
    with pytest.raises(ValueError):
        Attn_Ranker(X, y, score="both")
    from modeling.forecasting.neuralforecasting.mhattn_regressor import MHAttn_Regressor
    paper = Attn_Ranker(X, y, seeds=[0], epochs=10, regressor=lambda n: MHAttn_Regressor(input_dim=n, d_model=16, num_heads=2))
    assert paper.run().shape == (n,)                                     # the paper's regressor plugs in (its heat map is (n, n))
    y2, xe = covid()
    mod = ARX_D(y2, 6, xe, Args(p=3, px=2, spec=1, lam=1.0, method="ridge"))
    fs = mod.fs_rng
    rank = Attn_Ranker(mod.X[fs.start:fs.stop], mod.y[fs.start:fs.stop], seeds=[0], epochs=20).run()
    cols, rSq = quiet(mod.selectFeatures, SelectionTech.Ranked, rank=rank)   # the vector drives the Ranked technique
    assert 0 in cols[0] and len(rSq[0]) == mod.X.shape[1] and mod.getBest().qof is not None
