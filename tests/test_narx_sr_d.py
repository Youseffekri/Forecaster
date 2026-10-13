"""Tests of NARX_SR_D and the three optimizers of its nonlinear parameters (counterparts of
varProTest, nARX_FormulaTest and nARX_SR_DTest8)."""
import numpy as np
import pytest

from Util.tools import dotdict
from Util.transform import TransformT
from modeling.feature_selection import SelectionTech
from modeling.fit import QoF
from modeling.optimization.varpro import VarPro
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.nonlinear import narx_sr_d
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D, OptimizerNL
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid

OPTIMIZERS = list(OptimizerNL)


def test_optimizer_nl_names():
    """`OptimizerNL.of` accepts the enum, its name in any case and None (the module default)."""
    assert OptimizerNL.of("lbfgsb") is OptimizerNL.LBFGS_B and OptimizerNL.of("VarPro") is OptimizerNL.VarPro
    assert OptimizerNL.of(OptimizerNL.Joint) is OptimizerNL.Joint and OptimizerNL.of(None) is narx_sr_d.OPTIMIZER_NL
    assert str(OptimizerNL.LBFGS_B) == "LBFGS_B"
    with pytest.raises(ValueError):
        OptimizerNL.of("newton")


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_varpro_recovers_a_noise_free_model(optimizer):
    """y = 1 + 0.5 x^2 + 3 log(x + 2): every optimizer recovers (1, 0.5, 3; a = 2) from a = 8
    (the counterpart of varProTest)."""
    x = np.linspace(0.5, 5.0, 60)
    y = 1 + 0.5 * x ** 2 + 3 * np.log(x + 2.0)
    basis = lambda a: np.column_stack([x ** 2, np.log(x + a[0])])
    a, steps = NARX_SR_D.fitNL(basis, y, np.array([8.0]), np.array([0.1]), np.array([20.0]), 0.0, False, optimizer)
    vp = VarPro(basis, y, lb=[0.1], ub=[20.0])
    c, b = vp.linear(a)
    assert a[0] == pytest.approx(2.0, abs=1e-3) and steps["nfev"] > 0
    assert c[0] == pytest.approx(1.0, abs=1e-3) and np.allclose(b.ravel(), [0.5, 3.0], atol=1e-3)
    if optimizer is OptimizerNL.VarPro:
        assert vp.optimize(np.array([8.0]))[0] == pytest.approx(a[0]) and vp.stepCounts == steps


def test_varpro_two_responses_and_ridge_agree_across_optimizers():
    """Two responses sharing two decay rates; with a ridge penalty all three optimizers reach the
    same penalized objective (the nARX_SR_DTest8 property)."""
    t = np.arange(50.0)
    Y = np.column_stack([2 * np.exp(-t / 10) + 0.3 * np.exp(-0.02 * t), 1.5 * np.exp(-t / 10) - 0.7 * np.exp(-0.02 * t)])
    basis = lambda a: np.column_stack([np.exp(-a[0] * t / 10), np.exp(-a[1] * t / 10)])
    vp = VarPro(basis, Y, lb=[0.01, 0.01], ub=[5, 5], lam=0.1)
    objs = []
    for opt in OPTIMIZERS:
        a, _ = NARX_SR_D.fitNL(basis, Y, np.array([0.5, 0.6]), vp.lb, vp.ub, 0.1, False, opt)
        objs.append(vp.objective(a))
    assert max(objs) - min(objs) < 1e-6 * max(objs)


def covid_context():
    return Example_Covid.context(["icu_patients", "positive_rate"])


def test_plain_lags_equal_arx_d():
    """y ~= y~(6,1) with no ridge is the base's ARX_D with p = 6 (the compatibility test)."""
    data = covid_context()
    hh = 6
    hp = dotdict(p=6, px=0, spec=0, skip=6, crx=False)
    ref = ARX_D(hp, data["y"], hh, None, method="sk_lr")
    ref.train()
    yf_ref = ref.forecast()
    mod = NARX_SR_D.fromTerms("y ~= y~(6,1)", data, hh, hparam={'lambda': 0.0})
    mod.train()
    yf = mod.forecast()
    assert np.allclose(yf, yf_ref, atol=1e-6)
    assert mod.params.shape == ref.params.shape == (7, hh)
    assert mod.nFree == 0 and mod.optimizer is None and mod.stepCounts == {}


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_planted_log_shift_is_recovered(optimizer):
    """y_t = 5 + 0.6 y_t-1 + 4 log(z_t-1 + 0.3) + e: the shift is fitted close to 0.3 and counted as a DF."""
    rng = np.random.default_rng(7)
    n = 400
    z = rng.uniform(0.2, 3.0, n)
    y = np.zeros(n)
    y[0] = 8.0
    for t in range(1, n):
        y[t] = 5 + 0.6 * y[t - 1] + 4 * np.log(z[t - 1] + 0.3) + 0.05 * rng.standard_normal()
    data = {"y": y, "z": z}
    mod = NARX_SR_D.fromTerms("y ~= y~1 + log(? + z~1)", data, hh=2, hparam={'lambda': 1e-8}, optimizer=optimizer)
    s = mod.formula.params()[0]
    assert s.role.value == "Shift" and s.value == pytest.approx(0.3, abs=0.05), mod.formulaStr
    assert mod.nFree == 1 and mod.dof == 3 and mod.optimizer is optimizer and mod.stepCounts["nfev"] > 0
    mod.train()
    yf = mod.forecast()
    assert yf.shape == (n, 2)                                         # every row is forecast (skip affects the diagnosis only)


def test_three_optimizers_agree_on_covid_when_scaled(capsys):
    """On Covid-19 with standardized columns the three optimizers reach the same optimum (the
    nonlinear parameters agree; the objective is in the debug line of `fitNL`)."""
    data = covid_context()
    narx_sr_d.DEBUG = True
    try:
        w, objs = {}, {}
        for opt in OPTIMIZERS:
            m = NARX_SR_D.fromTerms("y ~= y~(3,1) + log(? + icu~1) + pow(y~1, ?)", data, hh=6, hparam={'lambda': 1.0},
                                    scaled=True, optimizer=opt.name.lower())
            w[opt] = m.parameter_nl
            line = capsys.readouterr().out.strip().splitlines()[-1]
            assert line.startswith(f"fitNL: {opt}: a = VectorD(")
            objs[opt] = float(line.split("objective = ")[1].split(",")[0])
    finally:
        narx_sr_d.DEBUG = False
    for opt in OPTIMIZERS[1:]:
        assert np.allclose(w[opt], w[OptimizerNL.VarPro], rtol=1e-3) and objs[opt] == pytest.approx(objs[OptimizerNL.VarPro], rel=1e-5)


def test_module_default_optimizer():
    """With no `optimizer` argument `fromTerms` uses the module's `OPTIMIZER_NL` (a test may reassign it)."""
    data = covid_context()
    narx_sr_d.OPTIMIZER_NL = OptimizerNL.LBFGS_B
    try:
        m = NARX_SR_D.fromTerms("y ~= y~1 + log(? + icu~1)", data, hh=2, hparam={'lambda': 1.0})
        assert m.optimizer is OptimizerNL.LBFGS_B and "nit" in m.stepCounts
    finally:
        narx_sr_d.OPTIMIZER_NL = OptimizerNL.VarPro


def test_redundant_parameter_is_not_fitted(capsys):
    """exp(? + x): the shift is a factor the coefficient absorbs; it stays at its start value."""
    data = covid_context()
    m = NARX_SR_D.fromTerms("y ~= y~1 + exp(? + pRate~1)", data, hh=3, hparam={'lambda': 1.0})
    assert m.nFree == 0 and "covered by the linear fit" in capsys.readouterr().out


def test_tform_log1p_forecasts_in_original_scale():
    data = covid_context()
    m = NARX_SR_D.fromTerms("y ~= y~(2,1) + log1p(icu~1) TFORM: Log1p", data, hh=3, hparam={'lambda': 1.0})
    m.train()
    yf = m.forecast()
    assert np.all(m.Yf[6:, 1] > 50), "Yf holds the forecasts back in the original (deaths) scale"
    assert np.all(yf < 20), "the model works in log1p space"


# -- the transform-type path (ScalaTion's `NARX_SR_D.apply` / `rescale`, nARX_SR_DTest5..8) ---------

def covid_xe():
    return Example_Covid.loadData(["icu_patients", "positive_rate"])


HP5 = {"p": 6, "px": 4, "spec": 1, "crx": 0, "lambda": 6.0}               # nARX_SR_DTest5..8


def test_transform_types_parameters_and_bounds():
    """initializeW / tunable / getTransforms follow ScalaTion's TransformT.wlu and `nonlinear` rules."""
    R, P, L, L1 = TransformT.Root, TransformT.Pow, TransformT.Log, TransformT.Log1p
    w, l, u = NARX_SR_D.initializeW([R], [[L], [L1, P]])
    assert w.tolist() == [0.0, 0.5, 1.0, 1.0, 0.0, 1.5] and l.tolist() == [0.0, 0.1, 0.0, 0.1, 0.0, 1.1]
    assert u.tolist() == [10.0, 0.9, 10.0, 10.0, 10.0, 4.0]
    assert NARX_SR_D.tunable([R], [[L], [L1, P]]) == [0, 1, 2, 4, 5]     # the log's scale is covered by the fit
    assert NARX_SR_D.tunable([TransformT.Exp], []) == [1]                 # exp's shift is a coefficient
    fEn, fEx = NARX_SR_D.getTransforms(np.array([0.2, 0.7, 2.0, 1.0, 0.0, 2.0]), [R], [[L], [L1, P]])
    assert fEn[0].w.tolist() == [0.2, 0.7] and fEx[0][0].w.tolist() == [2.0, 1.0]
    assert type(fEx[1][0]).__name__ == "Log1pForm" and fEx[1][1].w.tolist() == [0.0, 2.0]
    assert NARX_SR_D.exoSets(None, 2) == [[P], [P]] and NARX_SR_D.exoSets([[L]], 2) == [[L], []]


def test_rescale_default_parameters_reproduce_nARX_SR_DTest5(capsys):
    """tune = False: Root transforms at (0, 0.5), MinMax to [1, 5], lambda = 6 -- the Scala log's numbers
    (In-Sample rSq 0.951773, sMAPE 10.3656 ..; TnT sMAPE 17.8772 ..) and REPORT layout."""
    y, xe = covid_xe()
    R = TransformT.Root
    mod = NARX_SR_D.rescale(HP5, y, 6, xe, [R], [[R], [R]], tune=False)
    assert isinstance(mod, ARX_D) and mod.modelName == "NARX_SR_D(6, 4, 2)" and mod.nFree == 0
    assert mod.parameter_nl.tolist() == [0.0, 0.5, 0.0, 0.5, 0.0, 0.5] and not mod._nneg   # MinMax: no rectification
    assert mod.fname[:8] == ["cons", "yl6", "yl5", "yl4", "yl3", "yl2", "yl1", "f0(yl6)"] and len(mod.fname) == 29
    _, qof = mod.inSample_Test()
    out = capsys.readouterr().out
    assert "b = VectorD(-0.131082,\t-0.0369535,\t0.0124104," in out
    assert ("hparameter hp  = HyperParameter(p -> (6,1), d -> (1,1), q -> (1,1), P -> (1,1), D -> (1,1), Q -> (1,1), "
            "s -> (7,7), px -> (4,1), spec -> (1,1), crx -> (0,0), nneg -> (1,1))") in out
    assert qof[QoF.rSq.ordinal] == pytest.approx(0.951773, abs=1e-6) and qof[QoF.dfr.ordinal] == 29
    ftMat = mod.tnT_Test(rc=2)
    assert np.allclose(ftMat[:, QoF.smape.ordinal], [17.8772, 15.8783, 18.9987, 25.8737, 34.0134, 44.1487], atol=1e-4)


def test_rescale_varpro_reproduces_nARX_SR_DTest8():
    """Pow of y, Log of each exogenous variable, lambda = 6, fitted by VarPro: the Scala log's parameters
    (10, 1.99494, 0, 1, 0, 1) and TnT sMAPE (14.1927 ..); the fitted parameters take degrees of freedom."""
    y, xe = covid_xe()
    mod = NARX_SR_D.rescale(HP5, y, 6, xe, [TransformT.Pow], [[TransformT.Log], [TransformT.Log]],
                            optimizer=OptimizerNL.VarPro)
    assert np.allclose(mod.parameter_nl, [10.0, 1.99494, 0.0, 1.0, 0.0, 1.0], atol=1e-4)
    assert mod.nFree == 4 and mod.optimizer is OptimizerNL.VarPro and mod.stepCounts["nfev"] > 0
    ftMat = mod.tnT_Test(rc=2)
    assert ftMat[0, QoF.dfr.ordinal] == 29 + 4
    assert np.allclose(ftMat[:, QoF.smape.ordinal], [14.1927, 18.2151, 23.9818, 32.7923, 42.8505, 53.1538], atol=1e-4)


def test_apply_on_raw_data_and_feature_selection_submodel():
    """`apply` (no rescaling) with tune = False is the ARX_SR_D model with the default transforms; the feature
    selection sub-model is ARX_D's (ScalaTion's inherited `getModel`)."""
    y, xe = covid_xe()
    mod = NARX_SR_D.apply({"p": 3, "px": 2, "spec": 1, "lambda": 1.0}, y, 3, xe, [TransformT.Root], tune=False)
    assert mod.yForm is None and mod._nneg and mod.fname[-1] == "g1,0(xe1l1)"
    sub = mod.getModel([0, 1, 4])
    assert type(sub) is ARX_D and sub.modelName == "ARX_D_3_2_2" and sub.X.shape[1] == 3


def test_backward_selection_after_tnt_reproduces_nARX_SR_DTest5():
    """Root transforms fitted by VarPro, TnT, then Backward at h = 1 (nARX_SR_DTest5): the Scala log's columns and
    trace -- the sub-models keep their own default skip (a new model's), not the parent's skip 0 after `tnT_Test`."""
    y, xe = covid_xe()
    R = TransformT.Root
    mod = NARX_SR_D.rescale(HP5, y, 6, xe, [R], [[R], [R]])
    assert np.allclose(mod.parameter_nl[[1, 2, 3, 5]], [0.9, 0.7899, 0.9, 0.9], atol=2e-3)   # Scala: 0.789879
    mod.tnT_Test(rc=2)
    assert mod.getSkip() == 0
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.Backward, "none", h=1)
    assert colsArr[0] == [0, 1, 6, 12, 13, 28]
    assert np.allclose(rSqArr[0][0], [-1.38036, -2.26967, 50.5695, 50.6393], atol=1e-4)
    assert np.allclose(rSqArr[0][-1], [96.4862, 95.3013, 11.2285, 11.9495], atol=1e-4)
