"""Tests of ARX_Formula (the linear term language and its DSL) and FormulaParser (the counterparts of
ScalaTion's aRX_FormulaTest .. aRX_FormulaTest5 and formulaParserTest, where no model fit is involved
beyond fit / fit_D on Covid-19)."""
import math

import numpy as np
import pytest

from Util.transform import Log1pForm, TransformT
from modeling.fit import QoF
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_formula import (ARX_Formula, FormulaVar, LabelMode, ProductTerm, TransformedTerm,
                                              WaveTerm, distinct, lagv, log, log1p, pow, powr, sq, wave, wave_pair)
from modeling.forecasting.formula_parser import FormulaError, FormulaParser, parse_formula, try_parse_formula
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid

VARS = ["y", "icu", "pRate"]
FREQS = [0.13043, 0.06521]
HH = 6


def covid():
    return Example_Covid.context(["icu_patients", "positive_rate"])


# -- terms and the DSL ----------------------------------------------------------------------------

def test_lagv_pads_with_the_first_value():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(lagv(x, 0), x)
    assert np.array_equal(lagv(x, 2), [1.0, 1.0, 1.0, 2.0])


def test_term_labels_lags_and_vars():
    y, icu = FormulaVar("y"), FormulaVar("icu")
    assert y.label() == "y" and y.lag_(3).label() == "y~3" and y.lag_(3).lag == 3
    assert [t.label() for t in y.lags(3, 1)] == ["y~3", "y~2", "y~1"]
    assert log1p(icu.lag_(1)).label() == "log1p(icu~1)" and isinstance(log1p(icu.lag_(1)), TransformedTerm)
    assert pow(y.lag_(1), 1.5).label() == "pow(y~1, 1.5)" and powr(y.lag_(2), 0.5).label() == "powr(y~2, 0.5)"
    pr = y.lag_(1) * icu.lag_(2)
    assert isinstance(pr, ProductTerm) and pr.label() == "y~1 * icu~2" and pr.lag == 2 and pr.vars == {"y", "icu"}
    assert [t.label() for t in y.lag_(1) * icu.lags(2, 1)] == ["y~1 * icu~2", "y~1 * icu~1"]
    assert [t.label() for t in sq(y.lags(2, 1))] == ["sq(y~2)", "sq(y~1)"]
    w = wave_pair(0.1)
    assert [t.name for t in w] == ["sin(w=0.6283)", "cos(w=0.6283)"] and all(isinstance(t, WaveTerm) for t in w)
    with pytest.raises(ValueError):
        FormulaVar("y", -1)
    with pytest.raises(ValueError):
        y.lags(1, 3)
    with pytest.raises(ValueError):
        TransformedTerm(y.lag_(1), "cube")
    with pytest.raises(ValueError):
        wave(3, FREQS)
    assert (y.lag_(1) + icu.lag_(1)) == [y.lag_(1), icu.lag_(1)]          # `+` builds the term list


def test_columns_of_the_terms():
    data = covid()
    y, icu = FormulaVar("y"), FormulaVar("icu")
    assert np.array_equal(y.lag_(2).column(data), lagv(data["y"], 2))
    assert np.allclose(log1p(icu.lag_(1)).column(data), np.log1p(lagv(data["icu"], 1)))
    assert np.allclose(pow(y.lag_(1), 1.5).column(data), lagv(data["y"], 1) ** 1.5)
    assert np.allclose(powr(y.lag_(1), 0.5).column(data), np.sqrt(lagv(data["y"], 1)))
    assert np.allclose((y.lag_(1) * icu.lag_(1)).column(data), lagv(data["y"], 1) * lagv(data["icu"], 1))
    s = WaveTerm(2 * math.pi * 0.1).column(data)
    assert s.shape == (116,) and s[0] == 0.0


def test_distinct_drops_duplicate_labels():
    y = FormulaVar("y")
    ts = distinct(y.lags(2, 1) + y.lags(3, 1))
    assert [t.label() for t in ts] == ["y~2", "y~1", "y~3"]


# -- ARX_Formula -----------------------------------------------------------------------------------

def test_dsl_and_parser_give_the_same_formula_str_and_key():
    y, icu, pRate = FormulaVar("y"), FormulaVar("icu"), FormulaVar("pRate")
    f = ARX_Formula(y, y.lags(3, 1) + [log1p(icu.lag_(1))] + [y.lag_(1) * icu.lag_(1)] + pow(pRate.lags(2, 1), 1.5))
    assert f.formulaStr == "y ~= y~3 + y~2 + y~1 + log1p(icu~1) + y~1 * icu~1 + pow(pRate~2, 1.5) + pow(pRate~1, 1.5)"
    g = FormulaParser.parseFormula(f.formulaStr, VARS, 6)
    assert g.formulaStr == f.formulaStr and g.key == f.key
    assert f.key == "y ~= " + " + ".join(sorted(t.label(LabelMode.SLOT) for t in f.preds))
    assert f.labels == f.names() == [t.label() for t in f.preds] and f.nTerms == 7
    assert f.vars == {"y", "icu", "pRate"} and f.maxLag == 3
    assert f.lagHP() == {"p": 3, "q": 2, "spec": 0, "n_exo": 2}
    h = ARX_Formula(y, [icu.lag_(2), y.lag_(1)])
    k = ARX_Formula("y", [y.lag_(1), icu.lag_(2)])
    assert h.key == k.key and h.formulaStr != k.formulaStr, "the key is order-free, formulaStr is not"
    assert repr(h) == h.formulaStr == h.formula_str


def test_waves_print_as_wave_k_and_parse_back():
    y = FormulaVar("y")
    f = ARX_Formula(y, y.lags(2, 1) + wave(2, FREQS))
    assert f.formulaStr == "y ~= y~2 + y~1 + wave(2)"
    g = parse_formula(f.formulaStr, VARS, 6, FREQS)
    assert g.nTerms == 6 and g.key == f.key
    f1 = ARX_Formula(y, [y.lag_(1)] + wave_pair(0.2, 1))                 # the second frequency alone: not wave(k)
    assert f1.formulaStr == "y ~= y~1 + sin(w=1.2566) + cos(w=1.2566)"


def test_tform_prints_parses_and_transforms_the_base_columns():
    y, icu = FormulaVar("y"), FormulaVar("icu")
    f = ARX_Formula(y, y.lags(2, 1) + [icu.lag_(1)], "Log1p")
    assert f.tFormT is TransformT.Log1p and f.formulaStr.endswith(" TFORM: Log1p") and f.key.endswith(" TFORM: Log1p")
    g = FormulaParser.parseFormula(f.formulaStr, VARS, 6)
    assert g.tFormT is TransformT.Log1p and g.key == f.key
    data = covid()
    td = f.prepare(data)
    assert np.allclose(td.y, np.log1p(data["y"])) and isinstance(td.tForm_y, Log1pForm) and td.trSize == 92
    assert np.allclose(td.data["icu"], np.log1p(data["icu"]))
    X = f.matrix(td.data)
    assert X.shape == (116, 3) and np.allclose(X[:, 2], lagv(np.log1p(data["icu"]), 1))
    g2 = f.fresh()
    assert g2.key == f.key and g2 is not f


def test_prepare_checks_the_dataset():
    y, icu, z = FormulaVar("y"), FormulaVar("icu"), FormulaVar("z")
    data = covid()
    with pytest.raises(ValueError, match="not in dataset"):
        ARX_Formula(y, [z.lag_(1)]).prepare(data)
    with pytest.raises(ValueError, match="no predictor terms"):
        ARX_Formula(y, []).prepare(data)
    with pytest.raises(ValueError, match="length"):
        ARX_Formula(y, [icu.lag_(1)]).prepare({"y": data["y"], "icu": data["icu"][:50]})
    with pytest.raises(ValueError, match="lag"):
        ARX_Formula(y, [y.lag_(200)]).prepare(data)
    with pytest.raises(ValueError, match="not finite"):
        ARX_Formula(y, [log(icu.lag_(1))]).matrix(data)                  # icu has zeros: log (0)


def test_grammar_text():
    g = ARX_Formula.grammar(6)
    assert "formula := y ~= element" in g and "1 <= k <= 6" in g and "wave(k)" in g and "log1p" in g


# -- fit / fit_D on Covid-19 ----------------------------------------------------------------------

def test_fit_d_and_fit_return_the_scala_named_models(capsys):
    data = covid()
    y, icu, pRate = FormulaVar("y"), FormulaVar("icu"), FormulaVar("pRate")
    f = ARX_Formula(y, y.lags(3, 1) + icu.lags(2, 1) + [log1p(pRate.lag_(1))])
    mod_D = f.fit_D(data, HH)
    out = capsys.readouterr().out
    assert isinstance(mod_D, ARX_D) and mod_D.modelName == "ARX_D_3_2_2"
    assert "modelName  mn  = ARX_D_3_2_2" in out and "features   fn  = Array(y~3, y~2, y~1, icu~2, icu~1, log1p(pRate~1))" in out
    assert mod_D.fname == f.labels and mod_D.params.shape == (7, HH) and mod_D.X.shape == (116, 6)
    assert mod_D.args["p"] == 3 and mod_D.args["q"] == 2 and mod_D.args["skip"] == 3 and mod_D.getSkip() == 3
    yf = mod_D.forecast()
    assert yf.shape == (116, HH) and np.all(yf >= 0)
    ft = mod_D.diagnoseAll()
    assert ft.shape == (HH, 26) and ft[0, QoF.m.ordinal] == 116 - 3

    mod = f.fit(data, 1)
    out = capsys.readouterr().out
    assert isinstance(mod, ARX) and mod.modelName == "ARX_3_2_2" and "modelName  mn  = ARX_3_2_2" in out
    assert mod.params.shape == (7,) and mod.forecast().shape == (116, 1)
    mod_D1 = f.fit_D(data, 1)
    assert np.allclose(mod.forecast()[:, 0], mod_D1.forecast()[:, 0], atol=1e-6), "with hh = 1 the direct model is the recursive one"
    assert not np.allclose(mod.forecast()[:, 0], mod_D.forecast()[:, 0]), "with hh = 6 the direct fit drops the 5 padded rows"
    with pytest.raises(ValueError, match="forge"):
        f.fit(data, 2)                                                   # log1p(pRate~1) is not the standard lag layout

    g = ARX_Formula(y, y.lags(2, 1) + icu.lags(2, 1))                    # the standard layout forges fine
    mod2 = g.fit(data, 3)
    assert mod2.modelName == "ARX_2_2_1" and mod2.forecast().shape == (116, 3)


def test_tform_log1p_keeps_the_forecasts_of_yf_in_original_units():
    data = covid()
    y, icu = FormulaVar("y"), FormulaVar("icu")
    f = ARX_Formula(y, y.lags(2, 1) + [icu.lag_(1)], "Log1p")
    mod = f.fit_D(data, 3)
    assert mod.modelName == "ARX_D_2_1_1"
    assert np.allclose(mod.y, np.log1p(data["y"])), "the model works on the transformed response"
    assert isinstance(mod.yForm, Log1pForm)
    yf = mod.forecast()
    assert np.all(yf[5:] < 15), "the forecasts of the model are in log1p units"
    assert np.all(mod.Yf[5:, 1] > 50) and np.allclose(mod.Yf[5:, 1], np.expm1(yf[5:, 0])), "Yf holds them back in deaths"
    from modeling.fit import diagnose
    q = diagnose(data["y"][2:], mod.Yf[2:, 1], 3)
    assert q[QoF.smape.ordinal] < 40 and q[QoF.rSq.ordinal] > 0.8


def test_tform_log1p_keeps_column_0_of_yf_in_original_units():
    data = covid()
    y, icu = FormulaVar("y"), FormulaVar("icu")
    f = ARX_Formula(y, y.lags(2, 1) + [icu.lag_(1)], "Log1p")
    mod = f.fit_D(data, 3)
    assert np.array_equal(mod.Yf[:, 0], data["y"]), "column 0 of Yf is y in original units"
    mod.forecast()
    ft = mod.diagnoseAll()
    assert ft[0, QoF.smape.ordinal] < 40


# -- the linear parser and its rejects -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "y ~= pow(y~1, ?)",
    "y ~= y~(2,1) + log(? + icu~1)",
    "y ~= exp(? * icu~1)",
    "y ~= sat(icu~1, ?)",
    "y ~= hinge(y~1, ?) + y~1",
    "y ~= trans(y~1, ?, ?) * y~(2,1)",
    "y ~= ema(pRate~1, ?)",
    "y ~= dq(y~1, ?)",
    "y ~= y~1 + wave(?)",
])
def test_free_parameter_forms_are_rejected_by_the_linear_parser(text):
    with pytest.raises(FormulaError, match=r"free parameters \(\?\) need nonlinear.NARX_FormulaParser"):
        FormulaParser.parseFormula(text, VARS, 6, FREQS)
    f, why = try_parse_formula(text, VARS, 6, FREQS)
    assert f is None and "free parameters (?) need nonlinear.NARX_FormulaParser" in why


@pytest.mark.parametrize("bad, reason", [
    ("y ~= y~0", "lag 0 outside [1,6]"),
    ("y ~= y~(9,1)", "lag range (9,1) outside [1,6]"),
    ("y ~= z~1", "variable 'z' not in dataset"),
    ("w ~= y~1", "response 'w' not a dataset variable"),
    ("y ~= pow(y~1, 5)", "pow exponent 5.0 outside [0.2,3.0]"),
    ("y ~= cube(y~1)", "unknown function 'cube'"),
    ("y ~= wave(3)", "wave(3) outside [1,2]"),
    ("y ~= y~1 * wave(1)", "wave(k) not allowed inside interactions"),
    ("y ~= y~1 TFORM: Foo", "unknown TFORM 'Foo'"),
    ("y ~= y~1 * icu~1 * pRate~1", "interaction with 3 factors"),
    ("y ~= y", "a variable needs a lag"),
])
def test_linear_parser_rejects_carry_the_scala_reasons(bad, reason):
    with pytest.raises(FormulaError) as e:
        parse_formula(bad, VARS, 6, FREQS)
    assert reason in str(e.value) and str(e.value).startswith("parse: ")


def test_wave_without_frequencies_and_term_cap():
    with pytest.raises(FormulaError, match="no frequencies supplied"):
        parse_formula("y ~= wave(1)", VARS, 6)
    with pytest.raises(FormulaError, match="exceeds maxTerms"):
        FormulaParser.parseFormula("y ~= y~(6,1) + icu~(6,1)", VARS, 6, maxTerms=10)
    f = parse_formula("y ~= y~(2,1) * icu~(2,1)", VARS, 6)
    assert f.labels == ["y~2 * icu~2", "y~2 * icu~1", "y~1 * icu~2", "y~1 * icu~1"], "a lag range distributes over the product"
