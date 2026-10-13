"""Tests of NARX_SR_D and the three optimizers of its nonlinear parameters (counterparts of
varProTest, nARX_FormulaTest and nARX_SR_DTest8)."""
import numpy as np
import pytest

from Util.tools import dotdict
from modeling.optimization.varpro import VarPro
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid

OPTIMIZERS = ["varpro", "lbfgsb", "joint"]


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_varpro_recovers_a_noise_free_model(optimizer):
    """y = 1 + 0.5 x^2 + 3 log(x + 2): VarPro recovers (1, 0.5, 3; a = 2) from a = 8."""
    x = np.linspace(0.5, 5.0, 60)
    y = 1 + 0.5 * x ** 2 + 3 * np.log(x + 2.0)
    vp = VarPro(lambda a: np.column_stack([x ** 2, np.log(x + a[0])]), y, lb=[0.1], ub=[20.0])
    a = vp.fit(np.array([8.0]), optimizer=optimizer)
    c, b = vp.linear(a)
    assert a[0] == pytest.approx(2.0, abs=1e-3)
    assert c[0] == pytest.approx(1.0, abs=1e-3) and np.allclose(b.ravel(), [0.5, 3.0], atol=1e-3)


def test_varpro_two_responses_and_ridge_agree_across_optimizers():
    """Two responses sharing two decay rates; with a ridge penalty all three optimizers reach the
    same penalized objective (the nARX_SR_DTest8 property)."""
    t = np.arange(50.0)
    Y = np.column_stack([2 * np.exp(-t / 10) + 0.3 * np.exp(-0.02 * t), 1.5 * np.exp(-t / 10) - 0.7 * np.exp(-0.02 * t)])
    basis = lambda a: np.column_stack([np.exp(-a[0] * t / 10), np.exp(-a[1] * t / 10)])
    objs = []
    for opt in OPTIMIZERS:
        vp = VarPro(basis, Y, lb=[0.01, 0.01], ub=[5, 5], lam=0.1)
        a = vp.fit(np.array([0.5, 0.6]), optimizer=opt)
        objs.append(vp.objective(a))
    assert max(objs) - min(objs) < 1e-6 * max(objs)


def covid_context():
    return Example_Covid.context(["icu_patients", "positive_rate"])


def test_plain_lags_equal_arx_d():
    """y ~= y~(6,1) with no ridge is the base's ARX_D with p = 6 (the compatibility test)."""
    data = covid_context()
    hh = 6
    args = dotdict(p=6, q=0, spec=0, skip=6, cross=False)
    ref = ARX_D(args, data["y"], hh, None, method="sk_lr")
    ref.train()
    yf_ref = ref.forecast()
    mod = NARX_SR_D.from_formula("y ~= y~(6,1)", data, hh, lam=0.0)
    mod.train()
    yf = mod.forecast()
    assert np.allclose(yf, yf_ref, atol=1e-6)
    assert mod.params.shape == ref.params.shape == (7, hh)
    assert mod.n_free == 0 and mod.optimizer == "none"


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
    mod = NARX_SR_D.from_formula("y ~= y~1 + log(? + z~1)", data, hh=2, lam=1e-8, optimizer=optimizer)
    s = mod.formula.params()[0]
    assert s.role.value == "Shift" and s.value == pytest.approx(0.3, abs=0.05), mod.formula_str
    assert mod.n_free == 1 and mod.dof == 3 and mod.optimizer == optimizer
    mod.train()
    yf = mod.forecast()
    assert yf.shape == (n, 2)                                         # every row is forecast (skip affects the diagnosis only)


def test_three_optimizers_agree_on_covid_when_scaled():
    """On Covid-19 with standardized columns, VarPro and the joint fit reach the same optimum."""
    data = covid_context()
    objs = {}
    for opt in ["varpro", "joint"]:
        m = NARX_SR_D.from_formula("y ~= y~(3,1) + log(? + icu~1) + pow(y~1, ?)", data, hh=6, lam=1.0,
                                   scaled=True, optimizer=opt)
        objs[opt] = m.steps["objective"]
    assert objs["varpro"] == pytest.approx(objs["joint"], rel=1e-5)


def test_redundant_parameter_is_not_fitted(capsys):
    """exp(? + x): the shift is a factor the coefficient absorbs; it stays at its start value."""
    data = covid_context()
    m = NARX_SR_D.from_formula("y ~= y~1 + exp(? + pRate~1)", data, hh=3, lam=1.0)
    assert m.n_free == 0 and "covered by the linear fit" in capsys.readouterr().out


def test_tform_log1p_forecasts_in_original_scale():
    data = covid_context()
    m = NARX_SR_D.from_formula("y ~= y~(2,1) + log1p(icu~1) TFORM: Log1p", data, hh=3, lam=1.0)
    m.train()
    yf = m.forecast()
    assert np.all(m.Yf[6:, 1] > 50), "Yf holds the forecasts back in the original (deaths) scale"
    assert np.all(yf < 20), "the model works in log1p space"
