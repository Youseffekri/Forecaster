"""Tests of NARX_Formula (the formula terms with FREE nonlinear parameters): the Python counterparts of
ScalaTion's nARX_FormulaTest (a noise-free toy whose shift, exponent and shift are recovered by fitNL_D
while fit_D keeps the start values), nARX_FormulaTest3 (labels with ? slots, a shared parameter, key,
a redundant parameter, plain labels after the fit, a free wave frequency), nARX_FormulaTest4 (sat,
hinge, trans, ema, dq each recovered from a noise-free series) and nARX_FormulaTest5 (Covid-19, fit_D
and fitNL_D followed by rolling validation)."""
import math

import numpy as np
import pytest

from modeling.forecasting.args import Args
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_formula import FormulaVar as V, LabelMode, fmt, log, exp
from modeling.forecasting.datasets import Example_Covid
from modeling.forecasting.forecaster import Forecaster, teRng
from modeling.forecasting.nonlinear.narx_formula import (NARX_Formula, Par, ParRole, ParWaveTerm, cap, decay, dq, ema,
                                                        freq, half, hinge, par, pow, power, rate, sat, shift, thresh,
                                                        trans, wave, width)
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D, OptimizerNL

LAM = 1e-12                                                              # ~OLS, so the parameters can be recovered exactly


@pytest.fixture(autouse=True)
def quiet():
    """The fits print the REPORT and the QoF tables; keep the test output short."""
    r, q = Forecaster.DO_REPORT, Forecaster.SHOW_QoF
    Forecaster.DO_REPORT, Forecaster.SHOW_QoF = False, False
    yield
    Forecaster.DO_REPORT, Forecaster.SHOW_QoF = r, q


def toy_power_log():
    """nARX_FormulaTest: y_t = 1 + 0.5 (x_t-1 + 2)^1.7 + 3 log (z_t-1 + 1.5), noise free."""
    m = 100
    t = np.arange(m)
    x = 1.0 + 0.05 * t + 0.6 * np.sin(0.7 * t)
    z = 0.5 + 2.0 * np.abs(np.cos(0.3 * t))
    y = np.zeros(m)
    y[1:] = 1.0 + 0.5 * (x[:-1] + 2.0) ** 1.7 + 3.0 * np.log(z[:-1] + 1.5)
    y[0] = y[1]
    return {"y": y, "x": x, "z": z}


# -- nARX_FormulaTest --------------------------------------------------------------------------------

def test_toy_shift_exponent_shift_fit_d_keeps_the_start_values_and_fitnl_d_recovers_them():
    data = toy_power_log()
    y, x, z = V("y"), V("x"), V("z")
    f = NARX_Formula(y, [pow(shift() + x.lagged(1), power()), log(shift() + z.lagged(1))])
    assert f.formulaStr == "y ~= pow(? + x~1, ?1.5) + log(? + z~1)", "shifts are bare until resolved; the power has its default"
    assert f.key == "y ~= log(? + z~1) + pow(? + x~1, ?)"
    assert f.nParams == f.nFree == 3

    mod0 = f.fit_D(data, 1)                                              # ARX_D with the parameters at their start values
    assert isinstance(mod0, ARX_D) and mod0.modelName == "ARX_D_0_1_2"
    s1, p, s2 = f.params()
    assert (s1.role, p.role, s2.role) == (ParRole.SHIFT, ParRole.POWER, ParRole.SHIFT)
    assert all(q.resolved for q in f.params()) and s1.lb < 0 < s1.ub and (p.lb, p.ub) == (1.1, 4.0)
    assert s1.value == s1.w0 and p.value == 1.5 and s2.value == s2.w0, "fit_D moves no parameter"
    assert f.formulaStr == f"y ~= pow(?{fmt(s1.w0)} + x~1, ?1.5) + log(?{fmt(s2.w0)} + z~1)", "start values shown as ?w"
    assert mod0.fname == f.names and mod0.params.shape == (3, 1)

    mod = f.fitNL_D(data, 1, Args(lam=LAM))                                          # NARX_SR_D with the parameters fitted
    assert isinstance(mod, NARX_SR_D) and mod.modelName == f"NARX_SR_D(0, 1, 2){mod.hpTag}"
    assert np.allclose(mod.w_nl, [2.0, 1.7, 1.5], atol=1e-3), f"fitNL_D must recover (2, 1.7, 1.5), got {mod.w_nl}"
    assert np.allclose(mod.intercept, [1.0], atol=1e-3) and np.allclose(mod.parameter, [0.5, 3.0], atol=1e-3)
    assert mod.nFree == 3 and mod.optimizer is OptimizerNL.VarPro and mod.dof == 5
    assert f.formulaStr == "y ~= pow(2 + x~1, 1.7) + log(1.5 + z~1)", "plain labels after the fit"
    assert mod.fname == ["pow(2 + x~1, 1.7)", "log(1.5 + z~1)"]
    assert f.key == "y ~= log(? + z~1) + pow(? + x~1, ?)", "the key never changes"
    assert f.proposalStr == "y ~= pow(?2 + x~1, ?1.7) + log(?1.5 + z~1)", "the proposal form carries the fitted values as start values"
    g = NARX_FormulaParser.parseFormula(f.proposalStr, ["y", "x", "z"], 6)
    assert g.key == f.key and [q.w0 for q in g.params()] == pytest.approx([2.0, 1.7, 1.5], abs=1e-3)


# -- nARX_FormulaTest3: the grammar side -----------------------------------------------------------

def test_labels_with_slots_shared_parameter_and_key():
    y, hosp = V("y"), V("hosp")
    p = power()
    f1 = NARX_Formula(y, y.lags(3, 1) + pow(y.lags(3, 1), p) + [log(shift() + hosp.lagged(1)), exp(rate() * hosp.lagged(1))])
    assert f1.names == ["y~3", "y~2", "y~1", "pow(y~3, ?1.5)", "pow(y~2, ?1.5)", "pow(y~1, ?1.5)",
                         "log(? + hosp~1)", "exp(? * hosp~1)"]
    assert f1.nParams == 3, "the three power terms share one parameter"
    assert f1.key == "y ~= exp(? * hosp~1) + log(? + hosp~1) + pow(y~1, ?) + pow(y~2, ?) + pow(y~3, ?) + y~1 + y~2 + y~3"
    assert [q.role for q in f1.params()] == [ParRole.POWER, ParRole.SHIFT, ParRole.RATE]
    assert f1.params()[0] is p
    assert "free parameters" in NARX_Formula.grammar() and "pow(term, ?)" in NARX_Formula.grammar()
    assert NARX_Formula.slots("pow(y~1, ?1.500) + log(?-2.1e-3 + x~1)") == "pow(y~1, ?) + log(? + x~1)"


def test_fresh_copies_the_parameters_and_distinct_is_by_structure():
    y, x = V("y"), V("x")
    p = power()
    f = NARX_Formula(y, pow(y.lags(2, 1), p) + pow(y.lags(2, 1), power(1.3)) + [sat(x.lagged(1), half())])
    assert f.nTerms == 3, "pow(y~k, ?) twice with different start values is one term per lag (distinct by structure)"
    g = f.fresh()
    assert g.key == f.key and g.nParams == f.nParams
    assert all(a is not b for a, b in zip(f.params(), g.params())), "fresh parameters"
    assert g.params()[0] is g.preds[0].params()[0] is g.preds[1].params()[0], "the shared parameter stays shared in the copy"
    f.params()[0].set(2.2)
    assert g.params()[0].value == 1.5 and f.formulaStr.startswith("y ~= pow(y~2, 2.2)")
    f.reset()
    assert f.params()[0].value == 1.5 and not f.params()[0].fitted
    assert NARX_Formula.of(f) is f and isinstance(NARX_Formula.of(f.fresh()), NARX_Formula)


def test_par_roles_defaults_and_constants():
    assert Par(ParRole.POWER).w0 == 1.5 and (Par(ParRole.POWER).lb, Par(ParRole.POWER).ub) == (1.1, 4.0)
    assert freq().w0 == 0.1 and decay().w0 == 0.5 and not shift().resolved
    c = Par.const(2.5)
    assert c.fixed and not c.free and c.label() == "2.5" and c.label(LabelMode.SLOT) == "2.5"
    q = par(0.3, 0.0, 1.0)
    assert q.free and q.label() == "?0.3" and q.label(LabelMode.SLOT) == "?"
    q.set(0.42)
    assert q.label() == "0.42" and q.label(LabelMode.SLOT_VALUE) == "?0.42"
    with pytest.raises(ValueError):
        Par(ParRole.FREE)
    with pytest.raises(ValueError):
        power(lb=3.0, ub=1.0)
    s = shift()
    s.resolve(np.array([-1.0, 0.0, 2.0, 3.0]), 4)
    assert s.resolved and s.lb == pytest.approx(1.0 + 0.01 * 4.0) and s.w0 >= 0.0


def test_redundant_parameters_are_the_ones_the_linear_fit_covers():
    y, x = V("y"), V("x")
    s, a, k, c, w, d = shift(), rate(), half(), thresh(), width(), decay()
    f = NARX_Formula(y, [s + x.lagged(1),                                  # x + s alone: the intercept covers s
                         a * x.lagged(2),                                  # a x alone: the coefficient covers a
                         exp(shift() + x.lagged(1)),                       # e^s is a factor
                         log(rate() * x.lagged(1)),                        # log a is a constant
                         pow(rate() * x.lagged(1), power()),               # a^p is a factor
                         (rate() * x.lagged(1)) * y.lagged(1),               # the coefficient covers a
                         sat(rate() * x.lagged(1), k),                     # k covers a
                         hinge(shift() + x.lagged(1), c),                  # c covers s
                         ema(rate() * x.lagged(1), d)])                    # the coefficient covers a
    red = f.redundant()
    assert len(red) == 9 and {q.role for q in red} == {ParRole.SHIFT, ParRole.RATE}
    assert s in red and a in red and k not in red and c not in red and w not in red and d not in red
    g = NARX_Formula(y, [log(shift() + x.lagged(1)), pow(shift() + x.lagged(1), power()), exp(rate() * x.lagged(1)),
                         sat(x.lagged(1), half()), hinge(x.lagged(1), thresh())])
    assert g.redundant() == [], "a shift inside log or pow and a rate inside exp are genuine"


def test_redundant_exp_shift_is_kept_and_the_rate_is_recovered(capsys):
    """y = 2 + 1.5 exp (0.8 x_t-1) + 0.4 x_t-1^1.3: the shift inside exp is reported and kept at its start
    value, the rate is recovered, and the labels print plainly after fitNL_D."""
    m = 80
    t = np.arange(m)
    x = 0.5 + 0.02 * t + 0.3 * np.sin(0.5 * t)
    yv = np.zeros(m)
    yv[1:] = 2.0 + 1.5 * np.exp(0.8 * x[:-1]) + 0.4 * x[:-1] ** 1.3
    yv[0] = yv[1]
    y, xv = V("y"), V("x")
    s = shift()
    f2 = NARX_Formula(y, [exp(s + rate() * xv.lagged(1)), pow(xv.lagged(1), 1.3)])
    assert f2.formulaStr == "y ~= exp(? + ? * x~1) + pow(x~1, 1.3)" and f2.nParams == 2
    mod = f2.fitNL_D({"y": yv, "x": x}, 1, Args(lam=LAM))
    out = capsys.readouterr().out
    assert "fromTerms: parameter Shift is covered by the linear fit: kept at its start value" in out
    assert s.value == s.w0, "the redundant shift must keep its start value"
    r = [q for q in f2.params() if q.role is ParRole.RATE][0]
    assert r.value == pytest.approx(0.8, abs=1e-4), f"fitNL_D must recover the rate 0.8, got {mod.w_nl}"
    assert mod.nFree == 1
    assert f2.formulaStr == f"y ~= exp({fmt(s.w0)} + {fmt(r.value)} * x~1) + pow(x~1, 1.3)", "plain labels expected"
    assert np.allclose(mod.intercept, [2.0], atol=1e-3) and np.allclose(mod.parameter, [1.5 * math.exp(s.w0), 0.4], atol=1e-3)


def test_free_wave_frequency_is_recovered():
    """A noise-free seasonal series at 0.13 cycles per step; the start 0.12 is given, the bounds come from the role."""
    m = 80
    t = np.arange(m)
    w = 5.0 + 2.0 * np.sin(2 * math.pi * 0.13 * t) + np.cos(2 * math.pi * 0.13 * t)
    f3 = NARX_Formula("w", wave(freq(0.12)))
    assert f3.formulaStr == "w ~= wave(?0.12)" and f3.nTerms == 2
    assert f3.key == "w ~= cos(f=?) + sin(f=?)", "the key lists the pair's two terms (the wave(...) form is formulaStr's)"
    assert NARX_FormulaParser.parseFormula(f3.formulaStr, ["w"], 6).key == f3.key
    assert all(isinstance(tm, ParWaveTerm) for tm in f3.preds) and f3.nParams == 1, "the sin/cos pair shares the frequency"
    mod3 = f3.fitNL_D({"w": w}, 1, Args(lam=LAM))
    assert mod3.w_nl[0] == pytest.approx(0.13, abs=1e-4), f"fitNL_D must recover the frequency 0.13, got {mod3.w_nl}"
    assert np.allclose(mod3.intercept, [5.0], atol=1e-3) and np.allclose(mod3.parameter, [2.0, 1.0], atol=1e-3)
    assert f3.formulaStr == "w ~= wave(0.13)"


# -- nARX_FormulaTest4: sat, hinge, trans, ema, dq -----------------------------------------------

def _x4():
    m = 120
    t = np.arange(m)
    x = 2.5 + 2.2 * np.sin(0.37 * t) + 0.004 * t                        # in [0.3, 5.2]
    xl = np.concatenate([[x[0]], x[:-1]])                                # the x~1 column
    return x, xl


def _ema(xl, d):
    e = np.empty_like(xl)
    e[0] = xl[0]
    for i in range(1, len(xl)):
        e[i] = d * xl[i] + (1 - d) * e[i - 1]
    return e


CASES = [
    ("sat", lambda xl: 1.0 + 4.0 * xl / (xl + 1.5), lambda x: sat(x.lagged(1), half()), [1.5], 1e-3),
    ("hinge", lambda xl: 0.5 + 2.0 * np.maximum(0.0, xl - 2.0), lambda x: hinge(x.lagged(1), thresh()), [2.0], 1e-2),
    ("trans", lambda xl: 1.0 + 2.0 / (1.0 + np.exp(-(xl - 2.0) / 0.3)), lambda x: trans(x.lagged(1), thresh(), width()), [2.0, 0.3], 1e-3),
    ("ema", lambda xl: 2.0 * _ema(xl, 0.3) + 1.0, lambda x: ema(x.lagged(1), decay()), [0.3], 1e-3),
    ("dq", lambda xl: 2.0 * xl * np.abs(xl) / np.sqrt(1.0 + (xl / 3.0) ** 2) + 1.0, lambda x: dq(x.lagged(1), cap()), [3.0], 1e-3),
]


@pytest.mark.parametrize("name, series, term, truth, tol", CASES, ids=[c[0] for c in CASES])
def test_parameter_terms_recover_their_planted_parameter(name, series, term, truth, tol):
    x, xl = _x4()
    f = NARX_Formula("y", [term(V("x"))])
    assert f.key == ("y ~= trans(x~1, ?, ?)" if name == "trans" else f"y ~= {name}(x~1, ?)")
    mod = f.fitNL_D({"y": series(xl), "x": x}, 1, Args(lam=LAM))
    assert np.linalg.norm(mod.w_nl - np.array(truth)) < tol, f"{name}: fitNL_D must recover {truth}, got {mod.w_nl}"
    assert mod.nFree == len(truth)
    assert np.allclose(mod.intercept, [1.0 if name != "hinge" else 0.5], atol=1e-2)
    assert np.allclose(mod.parameter, [4.0 if name == "sat" else 2.0], atol=1e-2)


# -- nARX_FormulaTest5: Covid-19 -------------------------------------------------------------------

def test_covid_fit_d_and_fitnl_d_with_rolling_validation():
    data = Example_Covid.context(Example_Covid.BEST2_EXO, ["icu", "pRate"])
    y, icu, pRate = V("y"), V("icu"), V("pRate")
    hh = 6
    formula = NARX_Formula(y, y.lags(3, 1) + [sat(icu.lagged(1), half()), ema(pRate.lagged(1), decay())]
                           + trans(y.lagged(1), thresh(), width()) * y.lags(3, 1))
    assert formula.nParams == 4 and formula.nTerms == 8
    assert formula.key == ("y ~= ema(pRate~1, ?) + sat(icu~1, ?) + trans(y~1, ?, ?) * y~1 + trans(y~1, ?, ?) * y~2 + "
                           "trans(y~1, ?, ?) * y~3 + y~1 + y~2 + y~3")

    mod0 = formula.fit_D(data, hh, Args(scaled=True))                    # ARX_D with the parameters at their start values
    assert mod0.modelName == "ARX_D_3_1_2" and mod0.fname == formula.names and mod0.X.shape == (116, 8)
    assert all(not q.fitted for q in formula.params())
    mod0.setSkip()
    yf0 = mod0.rollValidate(rc=2)
    te = teRng(116)
    assert yf0.shape == (len(te), hh)
    ft0 = mod0.diagnoseAll(tRng=te)
    assert ft0.shape == (hh, 26) and np.all(np.isfinite(ft0[:, 9]))

    mod = formula.fitNL_D(data, hh, Args(scaled=True))                   # NARX_SR_D with the parameters fitted
    assert mod.modelName == "NARX_SR_D(3, 1, 2)_L0.1" and mod.fname == formula.names and mod.nFree == 4   # the default ridge penalty 0.1
    assert all(q.fitted and q.lb <= q.value <= q.ub for q in formula.params()), "fitted inside their bounds"
    assert "?" not in formula.formulaStr and formula.nParams == 4
    mod.setSkip()
    yf = mod.rollValidate(rc=2)
    assert yf.shape == (len(te), hh) and np.all(yf >= 0)
    ft = mod.diagnoseAll(tRng=te)
    assert ft.shape == (hh, 26) and np.all(ft[:, 9] < 100)
    assert ft[0, 12] == mod0.diagnoseAll(tRng=te)[0, 12] + 4, "the fitted parameters count as degrees of freedom"
