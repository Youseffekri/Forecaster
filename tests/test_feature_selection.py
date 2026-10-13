"""Tests of the feature selection port: `SubsetSearch` (subsetSearchTest), `CEM_Selector`
(cEM_SelectorTest), `BestStep` / `qofVector`, and the selection scenarios of the direct models
(`selectFeatures`, `selectFeaturesAtH`, `selectFeaturesPerH`, `reportSelection`, `getModel`,
`getModel_h`; the aRX_DTest8 log)."""
import numpy as np
import pytest

from Util.transform import PowForm, TransformT
from modeling.cem_selector import CEM_Selector, Random
from modeling.feature_selection import BestStep, FeatureSelection, SelectionTech, qk as QK, qofVector
from modeling.fit import N_QoF, QoF, diagnose
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.forecaster_d import aggFPE, smapeFPE
from modeling.forecasting.forecaster_reg import Forecaster_Reg
from modeling.forecasting.forecaster import Forecaster
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid
from modeling.subset_search import SubsetSearch


@pytest.fixture(autouse=True)
def quiet_and_training_rows():
    """No REPORT / QoF tables in the tests; selection on the training rows (ScalaTion's default)."""
    Forecaster.DO_REPORT, Forecaster.SHOW_QoF = False, False
    FeatureSelection.fullset_FS = False
    yield
    Forecaster.DO_REPORT, Forecaster.SHOW_QoF = True, True
    FeatureSelection.fullset_FS = False


def covid():
    y, xe = Example_Covid.loadData(["icu_patients", "positive_rate"])
    return y, xe


def small_arx_d(method="ridge", hh=3):
    y, xe = covid()
    return ARX_D({"p": 3, "px": 2, "spec": 1, "lambda": 1.0}, y, hh, xe, method=method)   # 8 columns


# -- SubsetSearch on a synthetic set function (ScalaTion's subsetSearchTest) ----------------------

def eval_for(planted, qk=QK):
    def ev(cols):
        s = set(cols)
        qof = np.zeros(N_QoF)
        qof[qk] = len(s - planted) + len(planted - s) + 0.001 * len(s)
        return BestStep(-1, qof, None, cols, qk, qof[qk])
    return ev


def test_subset_search_finds_the_planted_set_with_the_scala_counts():
    n, planted = 8, {0, 1, 3, 6}
    ev = eval_for(planted)
    pf = SubsetSearch.forward(n, ev)
    assert set(pf.cols) == planted and len(pf.steps) == n and pf.nEval == 1 + 7 + 6 + 5 + 4 + 3 + 2 + 1
    pb = SubsetSearch.backward(n, ev)
    assert set(pb.cols) == planted and len(pb.steps) == n and pb.nEval == 29
    ps = SubsetSearch.stepwise(n, ev, slack=0.0)
    assert set(ps.cols) == planted and len(ps.steps) == 3 and ps.nEval == 29
    pm = SubsetSearch.beam(n, ev, bk=3)
    assert set(pm.cols) == planted and len(pm.steps) == 6 and pm.nEval == 123
    for tech in SelectionTech:
        assert set(SubsetSearch.run(tech, n, ev).cols) == planted


def test_subset_search_cem_reproduces_the_scala_log():
    """The MRG31k3p port draws ScalaTion's sets: 3 rounds, 104 sets and the log's probabilities."""
    pc = SubsetSearch.cem(8, eval_for({0, 1, 3, 6}), stream=0)
    assert set(pc.cols) == {0, 1, 3, 6} and len(pc.steps) == 3 and pc.nEval == 104
    assert np.allclose(pc.probs, [1.0, 0.71825, 0.16425, 0.813, 0.1695, 0.06425, 0.73575, 0.23625], atol=1e-6)


def test_subset_search_units_and_failed_evaluations():
    pair = {4, 5}
    unit_of = lambda j: pair if j in pair else {j}
    ev2 = eval_for({0, 2, 4, 5})
    pf2 = SubsetSearch.forward(8, ev2, units=unit_of)
    pb2 = SubsetSearch.backward(8, ev2, units=unit_of)
    assert set(pf2.cols) == {0, 2, 4, 5} and set(pb2.cols) == {0, 2, 4, 5}
    assert [len(s.mod_cols) for s in pf2.steps] == [1, 3, 4, 5, 6, 7, 8]
    whole = lambda s: pair <= set(s.mod_cols) or not (pair & set(s.mod_cols))
    assert all(whole(s) for s in pf2.steps) and all(whole(s) for s in pb2.steps)
    ev = eval_for({0, 1, 3, 6})
    ev3 = lambda cols: BestStep(-1, None, None, cols) if 7 in cols else ev(cols)
    pf3 = SubsetSearch.forward(8, ev3)
    assert set(pf3.cols) == {0, 1, 3, 6} and len(pf3.steps) == 7 and all(7 not in s.mod_cols for s in pf3.steps)


# -- CEM_Selector on a synthetic regression (ScalaTion's cEM_SelectorTest) --------------------------

def planted_regression():
    rk = QoF.bic.ordinal
    m, n = 120, 13
    rn = Random(5)
    x = np.zeros((m, n))
    for i in range(m):
        x[i, 0] = 1.0
        for j in range(1, n):
            x[i, j] = rn.gen() * 10.0
    y = np.array([3.0 + 2.0 * x[i, 2] - 1.5 * x[i, 5] + 0.8 * x[i, 9] + 6.0 * (rn.gen() - 0.5) for i in range(m)])

    def eval_cols(cols):
        xc = x[:, cols]
        b = np.linalg.lstsq(xc, y, rcond=None)[0]
        qof = diagnose(y, xc @ b, dfr=len(cols) - 1, df=m - len(cols))
        return BestStep(-1, qof, b, cols, rk, qof[rk])
    return n, eval_cols, rk


def test_cem_selector_recovers_the_truth_and_beats_the_full_set():
    n, eval_cols, rk = planted_regression()
    cem = CEM_Selector(n, eval_cols, fixed=[0], stream=0, qk=rk)
    best = cem.run()
    assert set(best.mod_cols) == {0, 2, 5, 9}
    assert best.qof[rk] <= eval_cols(list(range(n))).qof[rk]           # the selected set scores no worse than the full set
    pr = cem.probs
    assert min(pr[j] for j in (2, 5, 9)) > max(pr[j] for j in range(1, n) if j not in (2, 5, 9))
    assert cem.nS == 60 and 0.05 <= cem.rho <= 0.3
    again = CEM_Selector(n, eval_cols, fixed=[0], stream=0, qk=rk)
    assert set(again.run().mod_cols) == set(best.mod_cols) and again.nEvaluated == cem.nEvaluated
    prior = np.array([0.8 if j in (0, 2, 5, 9) else 0.2 for j in range(n)])
    assert set(CEM_Selector(n, eval_cols, fixed=[0], prior=prior, stream=0, qk=rk).run().mod_cols) == {0, 2, 5, 9}


def test_cem_selector_argument_checks():
    ev = eval_for({0, 1})
    with pytest.raises(ValueError):
        CEM_Selector(4, ev, fixed=[7])
    with pytest.raises(ValueError):
        CEM_Selector(4, ev, prior=[0.5, 0.5])
    with pytest.raises(ValueError):
        CEM_Selector(4, ev, alpha=0.0)


# -- BestStep and qofVector ----------------------------------------------------------------------------

def test_best_step_direction_and_qof_vector():
    q1, q2 = np.zeros(N_QoF), np.zeros(N_QoF)
    q1[QK], q2[QK] = 10.0, 12.0                                        # sMAPE-IC: lower is better
    s1, s2 = BestStep(1, q1, None, [0, 1]), BestStep(2, q2, None, [0, 2])
    assert s1.gt(12.0) and s1.ge(10.0) and not s2.gt(10.0)
    assert BestStep().better(s2) is s2 and s2.better(s1) is s1 and s1.better(s2) is s1
    assert s1.compare(s2) < 0 and s1 < s2
    rk = QoF.rSqBar.ordinal                                            # R^2 bar: higher is better
    r1, r2 = np.zeros(N_QoF), np.zeros(N_QoF)
    r1[rk], r2[rk] = 0.8, 0.9
    assert BestStep(1, r1, None, [0], rk).better(2, r2, None, [0, 2]).col == 2
    fit = np.zeros(N_QoF)
    fit[QoF.rSq.ordinal], fit[QoF.rSqBar.ordinal], fit[QoF.smape.ordinal], fit[QoF.smapeC.ordinal] = 0.9, 0.85, 8.0, 8.5
    assert np.allclose(qofVector(fit), [90.0, 85.0, 8.0, 8.5])
    assert np.allclose(qofVector(fit, np.vstack([fit, fit])), [90.0, 85.0, 8.0, 90.0])


def test_agg_fpe_is_the_mean_fpe_form_smape():
    qof = np.zeros((2, N_QoF))
    qof[:, QoF.smape.ordinal] = [10.0, 20.0]
    qof[:, QoF.dfr.ordinal] = [4.0, 4.0]
    qof[:, QoF.m.ordinal] = [100.0, 50.0]
    assert np.allclose(smapeFPE(qof), [10.0 * 1.1, 20.0 * 1.2])
    assert aggFPE(qof)[QoF.smapeC.ordinal] == pytest.approx((11.0 + 24.0) / 2) and aggFPE(qof)[QoF.smape.ordinal] == 15.0


# -- the selection of the direct models ---------------------------------------------------------------

def test_forward_selection_adds_the_best_column_first():
    mod = small_arx_d()
    n = mod.X.shape[1]
    fs = mod.getModel_h(h=1)                                           # the single-horizon ARX: its Forecaster_Reg selection
    colsArr, rSqArr = fs.forwardSelAll("none")
    scores = {j: fs.evalCols([0, j]).qof[QK] for j in range(1, n)}
    assert colsArr[0][0] == 0 and colsArr[0][1] == min(scores, key=scores.get)
    assert len(colsArr[0]) == n and len(rSqArr[0]) == n and all(len(r) == 4 for r in rSqArr[0])
    colsArr, _ = mod.selectFeatures(SelectionTech.Forward)             # all horizons jointly: J = mean FPE-form sMAPE
    joint = {j: aggFPE(mod.fitCols([0, j])[1])[QK] for j in range(1, n)}
    assert colsArr[0][1] == min(joint, key=joint.get)
    assert mod.getBest().qof[QK] == min(aggFPE(mod.fitCols(colsArr[0][:k])[1])[QK] for k in range(1, n + 1))


def test_backward_elimination_removes_the_worst_column_first():
    mod = small_arx_d()
    n = mod.X.shape[1]
    fs = mod.getModel_h(h=1)
    path = SubsetSearch.backward(n, fs.evalCols, [0], minCols=2)
    scores = {j: fs.evalCols([c for c in range(n) if c != j]).qof[QK] for j in range(1, n)}
    assert path.steps[1].col == min(scores, key=scores.get)
    colsArr, rSqArr = fs.backwardElimAll(1, "none")
    assert colsArr[0][-1] == path.steps[1].col and colsArr[0][0] == 0   # reverse order of elimination
    assert len(rSqArr[0]) == len(path.steps) + 1                         # the trace plus the intercept-only row


def test_cem_selection_scores_no_worse_than_the_full_set():
    mod = small_arx_d()
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.CEM)
    full = aggFPE(mod.fitCols(mod.mcols)[1])[QK]
    assert mod.getBest().qof[QK] <= full and set(colsArr[0]) <= set(mod.mcols) and 0 in colsArr[0]
    assert mod.cemProbs is not None and len(mod.cemProbs) == mod.X.shape[1] and mod.cemProbs[0] == 1.0
    assert len(rSqArr[0]) >= 1 and all(len(r) == 4 for r in rSqArr[0])


def test_select_features_prints_the_banners_and_returns_cols_and_trace(capsys):
    mod = small_arx_d()
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.CEM, h=1)       # the single-horizon model at h = 1 decides
    out = capsys.readouterr().out
    assert "| cemSelAll: " in out and " column sets evaluated => cols = LinkedHashSet(" in out
    assert "cemSelAll: selected features = LinkedHashSet(cons, " in out
    assert "cemSelAll: inclusion probabilities = VectorD(1.00000," in out
    assert isinstance(colsArr, list) and len(colsArr) == 1 and len(rSqArr) == 1
    assert all(isinstance(j, int) for j in colsArr[0]) and colsArr[0][0] == 0
    assert all(r.shape == (4,) for r in rSqArr[0])
    Forecaster.DO_REPORT, Forecaster.SHOW_QoF = True, True
    models, qofIn, qofTnT = mod.reportSelection(colsArr, rSqArr, inSample=False)
    out = capsys.readouterr().out
    k = len(colsArr[0])
    assert f"| ARX_D_3_2_2: selected: k = {k} of n = 8, cols = LinkedHashSet(" in out
    assert "trace per step: 100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv (or sMAPE-IC)" in out
    assert "| ARX_D_3_2_2: TnT Test (rolling validation) of the selected model on Array(cons, " in out
    assert "b = VectorD(" in out and ", fn = Array(cons, " in out and "fitMap QoF = " in out
    assert qofIn is None and qofTnT.shape == (mod.hh, N_QoF) and models[0].X.shape[1] == k
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.Backward)       # all horizons jointly
    out = capsys.readouterr().out
    assert "| selectFeatures (Backward, all horizons): " in out and " column sets evaluated => cols = LinkedHashSet(" in out


def test_select_features_at_h_and_per_h():
    mod = small_arx_d()
    cols, rSq = mod.selectFeaturesAtH(SelectionTech.Stepwise, h=2)
    assert 0 in cols and len(rSq) >= 1
    models, qofIn, qofTnT = mod.reportSelection([cols], [rSq], h=2)
    assert isinstance(models[0], ARX) and models[0].hh == 1 and qofIn.shape == (1, N_QoF) and qofTnT.shape == (1, N_QoF)
    colsArr, rSqArr = mod.selectFeaturesPerH(SelectionTech.Backward)
    assert len(colsArr) == mod.hh == len(rSqArr)
    models, qofIn, qofTnT = mod.reportSelection(colsArr, rSqArr, inSample=False)
    assert len(models) == mod.hh and qofIn is None and qofTnT.shape == (mod.hh, N_QoF)
    with pytest.raises(ValueError):
        mod.reportSelection(colsArr, rSqArr, h=1)
    with pytest.raises(ValueError):
        mod.selectFeatures(SelectionTech.Forward, h=mod.hh + 1)


def test_get_model_and_get_model_h_shapes():
    y, xe = covid()
    cols = [0, 2, 5]
    for mod in (small_arx_d("ridge"), small_arx_d("sm_ols"),
                ARX_D.rescale({"p": 3, "px": 2, "spec": 1}, y, 3, xe, method="ridge", tFormT=TransformT.Log1p)):
        sub = mod.getModel(cols)
        assert isinstance(sub, ARX_D) and sub.X.shape[1] == len(cols) and sub.fname == [mod.fname[j] for j in cols]
        assert sub.modelName == mod.modelName and sub.hh == mod.hh and sub.yForm is mod.yForm and not sub._useforge
        assert sub.tr_size == mod.tr_size and np.allclose(sub.Y, mod.Y, equal_nan=True) and sub.getSkip() == mod.getSkip()
        assert mod.getModel().X.shape == mod.X.shape
        sub_h = mod.getModel_h(cols, 2)
        assert isinstance(sub_h, ARX) and sub_h.hh == 1 and sub_h.X.shape == (len(y) - 1, len(cols))
        assert sub_h.tr_size == mod.tr_size and np.allclose(sub_h.y, mod.Y[:len(y) - 1, 1])
        sub.train()
        sub.forecast()
        assert np.all(np.isfinite(sub.Yf[:, 1:-1]))
    m3 = ARX_SR_D.rescale({"p": 3, "px": 2, "spec": 1}, y, 3, xe, [PowForm()], [[], []], method="sm_ols",
                          tFormT=TransformT.Log1p)
    sub = m3.getModel(cols)                                            # ScalaTion: the inherited ARX_D.getModel
    assert type(sub) is ARX_D and sub.X.shape[1] == 3 and sub.fname == [m3.fname[j] for j in cols]
    assert sub.modelName == "ARX_D_3_2_2" and sub.yForm is m3.yForm
    data = Example_Covid.context(["icu_patients", "positive_rate"])
    m4 = NARX_SR_D.fromTerms("y ~= y~(3,1) + log(? + icu~1) + pRate~1", data, 3, {"lambda": 1.0})
    sub = m4.getModel([1, 3, 4])                                       # the inherited ARX_D.getModel (no nFree)
    assert type(sub) is ARX_D and sub.X.shape[1] == 3 and sub.fname == [m4.fname[j] for j in (1, 3, 4)]
    assert sub.modelName == "ARX_D_3_1_2" and isinstance(m4, ARX_D)
    assert isinstance(m4.getModel_h([1, 3], 1), ARX)
    colsArr, _ = m4.selectFeatures(SelectionTech.Backward, first=0)   # no intercept column in a fromTerms matrix
    assert len(colsArr[0]) >= 1


def test_fullset_fs_switches_the_selection_rows():
    mod = small_arx_d()
    assert mod.getModel_h(h=1).fs_rng == range(0, mod.tr_size)
    FeatureSelection.fullset_FS = True
    assert mod.getModel_h(h=1).fs_rng == range(0, len(mod.y))
    assert mod.getModel_h(h=3).fs_rng == range(0, len(mod.y) - 2)
    FeatureSelection.fullset_FS = False


def test_forecaster_reg_selection_on_a_recursive_arx():
    """`ARX` is a `Forecaster_Reg`: it selects its own features (ScalaTion's `Forecaster_Reg.selectFeatures`)."""
    y, xe = covid()
    arx = ARX({"p": 3, "px": 2, "spec": 1, "lambda": 1.0}, y, 3, xe, method="ridge")
    assert isinstance(arx, Forecaster_Reg)
    colsArr, rSqArr = arx.selectFeatures(SelectionTech.Stepwise, cross="none")
    assert 0 in colsArr[0] and arx.getBest().qof is not None
    models, qofIn, qofTnT = arx.reportSelection(colsArr, rSqArr)
    assert models[0].hh == 1 and qofIn.shape == (1, N_QoF) and qofTnT.shape == (1, N_QoF)
    assert models[0].modelName == arx.modelName and models[0].X.shape[1] == len(arx.bestCols(colsArr[0]))


def test_arx_d_test8_log_is_reproduced():
    """CEM at h = 1 on ARX_D_6_6_2 (Log1p, lambda = 1): 990 sets, the log's columns and score."""
    y, xe = covid()
    mod = ARX_D.rescale({"p": 6, "px": 6, "spec": 1, "lambda": 1.0}, y, 6, xe, method="ridge", tFormT=TransformT.Log1p)
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.CEM, h=1)
    assert colsArr[0] == [0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 17, 18]
    assert mod.getBest().qof[QK] == pytest.approx(8.901091449490549, rel=1e-9)
    assert len(rSqArr[0]) == 11 and np.allclose(rSqArr[0][-1], [92.5975, 91.7351, 8.57551, 8.90109], rtol=1e-5)
    assert np.allclose(mod.cemProbs[:4], [1.0, 0.874167, 0.05, 0.95], atol=1e-6)
    sub = mod.getBest().mod
    assert isinstance(sub, ARX_D) and sub.X.shape[1] == 12
