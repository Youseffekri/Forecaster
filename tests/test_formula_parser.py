"""Tests of the formula language and its parser (the Python counterpart of nARX_FormulaParserTest)."""
import numpy as np
import pytest

from modeling.forecasting.arx_formula import lagv
from modeling.forecasting.nonlinear.narx_formula import ParRole
from modeling.forecasting.formula_parser import FormulaError
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser

VARS = ["y", "icu", "pRate"]
FREQS = [0.13043, 0.06521]

CASES = [
    ("y ~= y~(3,1) + pow(y~1, ?)", ["Power"], 1, 4),
    ("y ~= pow(y~(3,1), ?1.3) + log(? + icu~1)", ["Power", "Shift"], 2, 4),
    ("y ~= exp(? * icu~1) + sin(? + y~1) + log1p(?2 + pRate~1)", ["Rate", "Phase", "Shift"], 3, 3),
    ("y ~= sat(icu~1, ?) + hinge(y~1, ?) + ema(pRate~(2,1), ?)", ["Half", "Thresh", "Decay"], 3, 4),
    ("y ~= trans(y~1, ?, ?) * y~(2,1) + dq(icu~1, ?) + wave(?)", ["Thresh", "Width", "Cap", "Freq"], 4, 5),
    ("y ~= pow(? + y~1, ?) + pow(? + icu~1, 1.5) TFORM: Log1p", ["Shift", "Power", "Shift"], 3, 2),
]


@pytest.mark.parametrize("text, roles, n_params, n_terms", CASES)
def test_roles_and_round_trip(text, roles, n_params, n_terms):
    f = NARX_FormulaParser.parseFormula(text, VARS, 6, FREQS)
    assert [p.role.value for p in f.params()] == roles
    assert f.nParams == n_params, "a lag range shares one parameter"
    assert f.nTerms == n_terms
    assert NARX_FormulaParser.parseFormula(f.formulaStr, VARS, 6, FREQS).key == f.key, "formulaStr round trip"
    assert NARX_FormulaParser.parseFormula(f.proposalStr, VARS, 6, FREQS).key == f.key, "proposalStr round trip"


def test_canonical_key_ignores_start_values():
    f1 = NARX_FormulaParser.parseFormula("y ~= pow(y~1, ?) + log(? + icu~1)", VARS, 6)
    f2 = NARX_FormulaParser.parseFormula("y ~= log(?3 + icu~1) + pow(y~1, ?1.7)", VARS, 6)
    assert f1.key == f2.key == "y ~= log(? + icu~1) + pow(y~1, ?)"
    assert f2.formulaStr == "y ~= log(?3 + icu~1) + pow(y~1, ?1.7)"


def test_constants_in_parameter_positions_are_not_free():
    f = NARX_FormulaParser.parseFormula("y ~= sat(icu~1, 2.5) + pow(y~1, 1.5) + dq(y~1, 3) + y~2 + log(2 + y~1)", VARS, 6)
    assert f.nFree == 0
    assert f.names == ["sat(icu~1, 2.5)", "pow(y~1, 1.5)", "dq(y~1, 3)", "y~2", "log(2 + y~1)"]


def test_duplicates_collapse_and_waves_print_once():
    f = NARX_FormulaParser.parseFormula("y ~= y~1 + y~1 + wave(2) + log1p(icu~2)", VARS, 6, FREQS)
    assert f.nTerms == 6                                 # y~1, 2 sin/cos pairs, log1p
    assert f.formulaStr == "y ~= y~1 + log1p(icu~2) + wave(2)"           # the wave part prints last (ScalaTion)


@pytest.mark.parametrize("bad, reason", [
    ("y ~= powr(y~1, ?)", "powr keeps a fixed exponent"),
    ("y ~= hinge(y~1)", "hinge needs (term, parameter)"),
    ("y ~= trans(y~1, ?)", "trans takes 2 parameters"),
    ("y ~= sat(y~1, ?, ?)", "sat takes 1 parameter"),
    ("y ~= ? + y~1", "syntax error"),
    ("y ~= pow(y~1, ?) * wave(?)", "wave(?) not allowed inside interactions"),
    ("y ~= log(? + z~1)", "variable 'z' not in dataset"),
    ("y ~= cube(? + y~1)", "unknown function 'cube'"),
    ("y ~= y~7", "lag 7 outside [1,6]"),
    ("y ~= icu~0", "lag 0 outside [1,6]"),
    ("y ~= wave(3)", "wave(3) outside [1,2]"),
    ("z ~= y~1", "response 'z' not a dataset variable"),
    ("y ~= y~1 * y~2 * y~3", "interaction with 3 factors (at most two)"),
])
def test_rejects_carry_a_reason(bad, reason):
    f, why = NARX_FormulaParser.tryParseFormula(bad, VARS, 6, FREQS)
    assert f is None and reason in why and "in formula:" in why
    with pytest.raises(FormulaError):
        NARX_FormulaParser.parseFormula(bad, VARS, 6, FREQS)


def test_columns_match_lagging_and_functions():
    rng = np.random.default_rng(1)
    y = rng.uniform(1, 5, 40); z = rng.uniform(1, 5, 40)
    data = {"y": y, "z": z}
    f = NARX_FormulaParser.parseFormula("y ~= y~2 + log(z~1) + y~1 * z~1 + pow(z~1, 2)", ["y", "z"], 6)
    td = f.prepare(data)                                               # TermData: (data, y, tForm_y, trSize)
    X = f.matrix(td.data)
    assert np.allclose(X[:, 0], lagv(y, 2)) and X[0, 0] == y[0] and X[5, 0] == y[3]
    assert np.allclose(X[:, 1], np.log(lagv(z, 1)))
    assert np.allclose(X[:, 2], lagv(y, 1) * lagv(z, 1))
    assert np.allclose(X[:, 3], lagv(z, 1) ** 2)


def test_resolve_fills_role_defaults_from_training_rows():
    z = np.linspace(2.0, 12.0, 50)
    f = NARX_FormulaParser.parseFormula("y ~= log(? + z~1) + sat(z~1, ?) + hinge(z~1, ?)", ["y", "z"], 6)
    f.prepare({"y": z.copy(), "z": z}, trSize=40)
    s, k, c = f.params()
    assert s.role is ParRole.SHIFT and s.lb < 0 and s.ub > s.lb        # keeps log(s + z) defined
    assert k.role is ParRole.HALF and k.lb > 0
    z1 = lagv(z, 1)[:40]                                               # the parameters see the LAGGED column
    assert c.role is ParRole.THRESH and c.lb == pytest.approx(z1.min()) and c.ub == pytest.approx(z1.max())
    assert c.value == pytest.approx(0.5 * (z1.min() + z1.max()))


def test_fresh_copies_share_nothing():
    f = NARX_FormulaParser.parseFormula("y ~= pow(y~(2,1), ?)", VARS, 6)
    g = f.fresh()
    f.params()[0].set(2.5)
    assert g.params()[0].value == pytest.approx(1.5) and f.formulaStr != g.formulaStr
    assert f.key == g.key
