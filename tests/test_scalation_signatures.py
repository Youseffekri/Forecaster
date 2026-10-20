"""The signatures made consistent with ScalaTion rev5.0 in ForecasterPy rev2.0: `rescale` without `trSize`
(the rows come from `Model`), `tFormT` before `trSize` in `fromTerms` / `fit` / `fit_D` / `fitNL_D`,
`ARX_Formula.names`, the checked `Model.TE_RATIO`, `Forecaster_D.tnT_Test (showYf = True)`,
`refitTransform ()` under `Model.inSample`."""
import contextlib
import inspect
import io

import numpy as np
import pytest

from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_formula import ARX_Formula, FormulaVar
from modeling.forecasting.arx_sr import ARX_SR
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.datasets import Example_Covid
from modeling.forecasting.forecaster import Forecaster
from modeling.model import Model
from modeling.forecasting.forecaster_d import Forecaster_D
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from Util.transform import TransformT


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def params(f):
    return list(inspect.signature(f).parameters)


# -- Model.TE_RATIO is checked when set (ScalaTion's `TE_RATIO_=`) --------------------------------------

def test_te_ratio_is_a_checked_class_property():
    assert Model.TE_RATIO == 0.2 and Model.trSize(116) == 92
    try:
        Model.TE_RATIO = 0.3
        assert Model.TE_RATIO == 0.3 and Model.trSize(116) == 81 and Model.teSize(116) == 35
        with pytest.raises(ValueError, match=r"should be in \(0.05, 0.95\)"):
            Model.TE_RATIO = 0.95
        with pytest.raises(ValueError):
            Model.TE_RATIO = -0.1
        assert Model.TE_RATIO == 0.3                                        # a refused value leaves it as it was
        Model.TE_RATIO = 0.0                                                # 0: no testing set (a competition)
        assert Model.teSize(116) == 0 and Model.trSize(116) == 116 and Model.valRng(116) == range(92, 116)
        assert Model.splitStr(116) == "92 / 24 / 0" and Model.fitRows(116) == 116
    finally:
        Model.TE_RATIO = 0.2
    assert Model.TE_RATIO == 0.2 and Model.trSize(116) == 92


def test_in_sample_suspends_the_switch_and_restores_it():
    assert Model.validating is False
    with Model.withValidation(tuning=True):
        assert Model.validating and Model.tuning and Model.fitRows(116, 0.2) == 68
        with Model.inSample():
            assert not Model.validating and not Model.tuning and Model.fitRows(116, 0.2) == 92
        assert Model.validating and Model.tuning
    assert Model.validating is False and Model.tuning is False


# -- rescale: no trSize (the rows come from Model), tFormT last ------------------------------------------

def test_rescale_signatures_have_no_tr_size():
    assert params(ARX.rescale) == ["y", "hh", "xe", "args", "tFormT"]
    assert params(ARX_D.rescale) == ["y", "hh", "xe", "args", "tFormT"]
    assert params(ARX_SR.rescale) == ["y", "hh", "xe", "args", "fEndo", "fExo", "tFormT"]
    assert params(ARX_SR_D.rescale) == ["y", "hh", "xe", "args", "fEndo", "fExo", "tFormT"]
    assert params(NARX_SR_D.rescale) == ["y", "hh", "xe", "args", "fEndo", "fExo", "tFormT", "lu", "trSize", "fname", "opt_kwargs"]
    assert params(NARX_SR_D.apply)[:7] == ["y", "hh", "xe", "args", "fEndo", "fExo", "trSize"]   # ScalaTion keeps trSize here
    y, xe = covid()
    with pytest.raises(TypeError):
        ARX_D.rescale(y, 6, xe, Args(p=3, px=2), TransformT.Log1p, 92)    # no sixth argument any more


def test_rescale_fits_on_the_rows_model_gives_and_refit_transform_rebuilds_on_the_training_rows():
    y, xe = covid()
    args = Args(p=3, px=2, spec=1, lam=1.0, method="ridge")
    mod = ARX_D.rescale(y, 6, xe, args)                                    # MinMax on the 92 training rows
    lo, hi = mod._tForms["tForm_y"]._lo, mod._tForms["tForm_y"]._hi
    assert lo == pytest.approx(y[:92].min()) and hi == pytest.approx(y[:92].max())
    with Model.withValidation():
        modV = ARX_D.rescale(y, 6, xe, args)                               # the 68 fit rows while choosing
        assert modV._tForms["tForm_y"]._hi == pytest.approx(y[:68].max())
        back = modV.refitTransform()                                       # the switch suspended: the 92 training rows
        assert back._tForms["tForm_y"]._hi == pytest.approx(hi) and Model.validating is True
        assert np.allclose(back.X, mod.X)
    assert mod.refitTransform() is not mod and np.allclose(mod.refitTransform().X, mod.X)
    assert ARX_D(y, 6, xe, args).refitTransform() is not None


# -- fromTerms / fit / fit_D / fitNL_D: tFormT before trSize -------------------------------------------

def test_from_terms_and_fit_take_t_form_t_before_tr_size():
    assert params(ARX.fromTerms) == ["formula", "dataset", "hh", "args", "tFormT", "trSize"]
    assert params(ARX_D.fromTerms) == ["formula", "dataset", "hh", "args", "tFormT", "trSize"]
    assert params(NARX_SR_D.fromTerms) == ["formula", "dataset", "hh", "args", "tFormT", "trSize", "maxLag", "freqs", "opt_kwargs"]
    assert params(ARX_Formula.fit) == ["self", "dataset", "hh", "args", "tFormT", "trSize"]
    assert params(ARX_Formula.fit_D) == ["self", "dataset", "hh", "args", "tFormT", "trSize"]
    assert params(NARX_Formula.fitNL_D) == ["self", "dataset", "hh", "args", "tFormT", "trSize", "opt_kwargs"]
    y, xe = covid()
    data = {"y": y, "icu": xe[:, 0], "pRate": xe[:, 1]}
    yv, icu = FormulaVar("y"), FormulaVar("icu")
    f = ARX_Formula(yv, yv.lags(3, 1) + [icu.lagged(1)])
    m1 = ARX_D.fromTerms(f, data, 3, Args(lam=1.0), TransformT.Log1p)      # positional tFormT
    m2 = ARX_D.fromTerms(f, data, 3, Args(lam=1.0), TransformT.Log1p, 92)  # then trSize
    assert m1.X.shape == m2.X.shape == (116, 4) and np.allclose(m1.X, m2.X)
    m3 = quiet(f.fit_D, data, 3, Args(lam=1.0), TransformT.Log1p)
    assert m3.X.shape == (116, 4) and m3.fname == f.names


def test_arx_formula_names_is_the_one_name():
    yv, icu = FormulaVar("y"), FormulaVar("icu")
    f = ARX_Formula(yv, yv.lags(2, 1) + [icu.lagged(1)])
    assert f.names == ["y~2", "y~1", "icu~1"] and isinstance(ARX_Formula.names, property)
    assert not hasattr(f, "labels"), "ScalaTion's name is `names`; `labels` belongs to TermLibrary only"


# -- tnT_Test is the base's for every model (rev2.3: the Forecaster_D override that flipped showYf is gone); the
#    matrix is printed on request, slanted (ScalaTion's `Forecaster.slant`, rev5.4) ---------------------------

def test_tnt_test_prints_the_slanted_matrix_on_request_only():
    assert inspect.signature(Forecaster.tnT_Test).parameters["showYf"].default is False
    assert "tnT_Test" not in Forecaster_D.__dict__, "no override: a test that wants the matrix passes showYf = True"
    y, xe = covid()
    mod = ARX_D(y, 2, xe, Args(p=2, px=1, spec=1))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod.tnT_Test(rc=24)
    assert "Final TnT Forecast Matrix yf" not in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod.tnT_Test(rc=24, showYf=True)
    assert "Final TnT Forecast Matrix yf" in out.getvalue()
    s = mod.slant()
    assert s.shape == mod.Yf.shape and np.allclose(s[:, 0], mod.Yf[:, 0]) and np.allclose(s[:, -1], np.arange(len(y)))
    assert np.allclose(s[1:, 1], mod.Yf[:-1, 1]) and np.allclose(s[2:, 2], mod.Yf[:-2, 2]) and s[0, 1] == 0.0 and s[1, 2] == 0.0
