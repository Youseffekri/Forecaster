"""Tests of ARX_SR / ARX_SR_D (the Python counterparts of ScalaTion's ARX_SR / ARX_SR_D) and of the
transformations they apply."""
import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.tools import dotdict
from Util.transform import (PowForm, RootForm, LogForm, Log1pForm, ExpForm, DQuadForm, DIQuadForm, BoxcoxForm,
                            PowRForm, SinForm, CosForm, RecipForm, NExpForm, transform, TRANSFORMS)
from Util.build_matrix import build_lagged_matrix, backfill_matrix
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr import ARX_SR
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid


def covid():
    y, xe = Example_Covid.load_data(["icu_patients", "positive_rate"])
    return y, xe


def args_(p=3, q=2, spec=0, cross=False, skip=3):
    return dotdict(p=p, q=q, spec=spec, cross=cross, skip=skip)


@pytest.mark.parametrize("tr", [PowForm(), RootForm(), LogForm(), Log1pForm(), ExpForm(), DQuadForm(), DIQuadForm(),
                                BoxcoxForm(), PowRForm(), RecipForm(), NExpForm(), SinForm(), CosForm()])
def test_transforms_invert(tr):
    x = np.linspace(0.2, 2.0, 25)
    assert np.allclose(tr.fi(tr.f(x)), x, atol=1e-8), tr


def test_transform_defaults_and_registry():
    assert PowForm().w.tolist() == [0.0, 1.5] and RootForm().w.tolist() == [0.0, 0.5]
    assert LogForm().w.tolist() == [1.0, 1.0] and DQuadForm().w.tolist() == [0.0, 5.0]
    assert np.allclose(transform("Pow", 0, 2).f(np.array([3.0])), [9.0])
    assert set(TRANSFORMS) >= {"Pow", "Root", "Log", "Log1p", "Exp", "DQuad", "Boxcox", "Sin", "Cos"}


def test_no_transforms_is_arx():
    """ARX_SR with fEndo = [] and no fExo is the ARX on the same lags (same X, same forecasts)."""
    y, xe = covid()
    a = args_()
    ref = ARX(a, y, 6, xe, method="sk_lr")
    mod = ARX_SR(a, y, 6, xe, fEndo=[], fExo=[[], []], method="sk_lr")
    assert np.allclose(mod.X, ref.X)
    ref.train(); mod.train()
    assert np.allclose(mod.forecast(), ref.forecast())
    assert mod.fname == ["yl3", "yl2", "yl1", "xe0l2", "xe0l1", "xe1l2", "xe1l1"]


def test_column_layout_and_names():
    """Columns: y lags, f0(y) lags, xe0 lags, g0,0(xe0) lags, xe1 lags, cross lags -- as ScalaTion builds them."""
    y, xe = covid()
    a = args_(p=2, q=2, cross=True)
    mod = ARX_SR(a, y, 6, xe, fEndo=[PowForm()], fExo=[[LogForm()], []])
    names = mod.fname
    assert names == ["yl2", "yl1", "f0(yl2)", "f0(yl1)", "xe0l2", "xe0l1", "g0,0(xe0l2)", "g0,0(xe0l1)",
                     "xe1l2", "xe1l1", "xe0l2*yl2", "xe0l1*yl1", "xe1l2*yl2", "xe1l1*yl1"]
    assert mod.X.shape == (116, len(names))
    xb = backfill_matrix(xe)
    assert np.allclose(mod.X[:, 2:4], build_lagged_matrix(y ** 1.5, 2))
    assert np.allclose(mod.X[:, 6:8], build_lagged_matrix(np.log1p(xb[:, 0]), 2))
    assert np.allclose(mod.X[:, 10:12], build_lagged_matrix(y * xb[:, 0], 2))


def test_fexo_must_match_the_exogenous_variables():
    y, xe = covid()
    with pytest.raises(ValueError):
        ARX_SR(args_(), y, 6, xe, fEndo=[PowForm()], fExo=[[PowForm()]])
    with pytest.raises(ValueError):
        ARX_SR_D(args_(), y, 6, xe, fExo=[[], [], []])


def test_forge_uses_transformed_forecasts():
    """At horizon h the transformed y block of the forged row equals the transform of the forecasts."""
    y, xe = covid()
    a = args_(p=3, q=2)
    mod = ARX_SR(a, y, 6, xe, fEndo=[PowForm()], fExo=[[], []], method="sk_lr")
    mod.train()
    yf = mod.forecast()
    assert yf.shape == (116, 6) and np.isfinite(yf).all()               # all rows are forecast (skip only affects the diagnosis)
    t0 = a.skip
    X_window = mod.X[t0: t0 + 5]
    X_3 = mod._forge(X_window, yf[t0: t0 + 5], 3)
    # columns are in DESCENDING lag order (yl3, yl2, yl1): at horizon 3 the lag-3 slot takes the actual
    # yl1 (= y_{t-1}), the lag-2 and lag-1 slots the forecasts of horizons 1 and 2 (y_t, y_{t+1})
    assert np.allclose(X_3[:, 0], X_window[:, 2])                            # yl3 slot <- actual yl1
    assert np.allclose(X_3[:, 1:3], yf[t0: t0 + 5, 0:2])                     # yl2, yl1 slots <- forecasts h = 1, 2
    assert np.allclose(X_3[:, 3], X_window[:, 5])                            # f0(yl3) slot <- actual f0(yl1)
    assert np.allclose(X_3[:, 4:6], yf[t0: t0 + 5, 0:2] ** 1.5)              # f0 slots <- transformed forecasts
    assert np.allclose(X_3[:, 6], X_window[:, 7]) and np.allclose(X_3[:, 7], X_window[:, 7])   # xe0 lags hidden (h > q)


def test_rescale_fits_scalers_on_training_rows():
    """The MinMax / Norm scalers see the training rows only (ScalaTion's rescale): on a rising series
    the maximum of the testing rows is not the scaler's maximum."""
    t = np.arange(120.0)
    y = 100.0 + 2.0 * t + 10.0 * np.sin(t / 5.0)
    xe = np.column_stack([50.0 + t, 10.0 + 0.5 * t])
    mod = ARX_SR.rescale(args_(), y, 6, xe, fEndo=[PowForm()], fExo=[[], []], method="sk_lr")
    sc = mod._tForms["tForm_y"]
    assert isinstance(sc, MinMaxScaler) and sc.data_min_[0] == y[: mod._tr_size].min()
    assert sc.data_max_[0] == y[: mod._tr_size].max() < y.max()              # the testing rows do not leak
    assert mod._tForms["tForm_exo"].data_max_[0] == xe[: mod._tr_size, 0].max() < xe[:, 0].max()
    assert mod._nneg
    mod_n = ARX_SR_D.rescale(args_(), y, 6, xe, fEndo=[PowRForm()], method="sk_lr", tForm=StandardScaler)
    assert not mod_n._nneg                                                   # NormForm => non-negativity off
    with pytest.raises(ValueError):                                          # y^1.5 of a standardized (negative) value
        ARX_SR_D.rescale(args_(), y, 6, xe, fEndo=[PowForm()], method="sk_lr", tForm=StandardScaler)
    mod.train(); yf = mod.forecast()
    assert np.isfinite(yf).all() and mod.Yf[10, 1] > 50, "Yf holds forecasts in the original scale"


def test_arx_sr_d_equals_narx_sr_d_on_the_equivalent_formula():
    """The direct SR model and the formula model build the same columns (in another order), so
    with OLS / no ridge they give the same forecasts -- the two paths of the project agree."""
    y, xe = covid()
    a = args_(p=3, q=2)
    sr = ARX_SR_D(a, y, 6, xe, fEndo=[PowForm()], fExo=[[LogForm()], []], method="sm_ols")   # exact least squares
    sr.train()                                     # (sklearn's LinearRegression truncates the rank of this ill-conditioned X)
    yf_sr = sr.forecast()
    data = {"y": y, "icu": backfill_matrix(xe)[:, 0], "pRate": backfill_matrix(xe)[:, 1]}
    f = "y ~= y~(3,1) + pow(y~(3,1), 1.5) + icu~(2,1) + log1p(icu~(2,1)) + pRate~(2,1)"
    nx = NARX_SR_D.from_formula(f, data, 6, lam=0.0, args={"skip": 3})
    nx.train()
    yf_nx = nx.forecast()
    assert np.allclose(yf_sr, yf_nx, atol=1e-6)


def test_arx_d_rescale_uses_training_rows_too():
    t = np.arange(120.0)
    y = 100.0 + 2.0 * t
    xe = np.column_stack([50.0 + t])
    mod = ARX_D.rescale(args_(), y, 6, xe, method="sk_lr")
    assert mod._tForms["tForm_y"].data_max_[0] == y[: mod._tr_size].max() < y.max()
