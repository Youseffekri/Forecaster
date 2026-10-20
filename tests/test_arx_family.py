"""Tests of the ARX family -- ARX, ARX_D, ARX_SR, ARX_SR_D (the Python counterparts of ScalaTion's
classes of those names) -- and of the transformations the SR models apply."""
import numpy as np
import pytest

from Util.transform import (PowForm, RootForm, LogForm, Log1pForm, ExpForm, DQuadForm, DIQuadForm, BoxcoxForm,
                            PowRForm, SinForm, CosForm, RecipForm, NExpForm, MinMaxForm, NormForm, TransformT,
                            transform, TRANSFORMS)
from Util.make_matrix4ts import makeMatrix4L, backfill_matrix
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr import ARX_SR
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.datasets import Example_Covid


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


def hp_(p=3, px=2, spec=0, crx=False, skip=3, **settings):
    return Args(p=p, px=px, spec=spec, crx=crx, skip=skip, **settings)


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
    a = hp_(method="sk_lr")
    ref = ARX(y, 6, xe, a)
    mod = ARX_SR(y, 6, xe, a, fEndo=[], fExo=[[], []])
    assert np.allclose(mod.X, ref.X)
    ref.train(); mod.train()
    assert np.allclose(mod.forecast(), ref.forecast())
    assert mod.fname == ["yl3", "yl2", "yl1", "xe0l2", "xe0l1", "xe1l2", "xe1l1"]


def test_column_layout_and_names():
    """Columns: y lags, f0(y) lags, xe0 lags, g0,0(xe0) lags, xe1 lags, cross lags -- as ScalaTion builds them."""
    y, xe = covid()
    a = hp_(p=2, px=2, crx=True)
    mod = ARX_SR(y, 6, xe, a, fEndo=[PowForm()], fExo=[[LogForm()], []])
    names = mod.fname
    assert names == ["yl2", "yl1", "f0(yl2)", "f0(yl1)", "xe0l2", "xe0l1", "g0,0(xe0l2)", "g0,0(xe0l1)",
                     "xe1l2", "xe1l1", "xe0l2*yl2", "xe0l1*yl1", "xe1l2*yl2", "xe1l1*yl1"]
    assert mod.X.shape == (116, len(names))
    xb = backfill_matrix(xe)
    assert np.allclose(mod.X[:, 2:4], makeMatrix4L(y ** 1.5, 2))
    assert np.allclose(mod.X[:, 6:8], makeMatrix4L(np.log1p(xb[:, 0]), 2))
    assert np.allclose(mod.X[:, 10:12], makeMatrix4L(y * xb[:, 0], 2))


def test_fexo_must_match_the_exogenous_variables():
    y, xe = covid()
    with pytest.raises(ValueError):
        ARX_SR(y, 6, xe, hp_(), fEndo=[PowForm()], fExo=[[PowForm()]])
    with pytest.raises(ValueError):
        ARX_SR_D(y, 6, xe, hp_(), fExo=[[], [], []])


def test_forge_uses_transformed_forecasts():
    """At horizon h the transformed y block of the forged row equals the transform of the forecasts."""
    y, xe = covid()
    a = hp_(p=3, px=2, method="sk_lr")
    mod = ARX_SR(y, 6, xe, a, fEndo=[PowForm()], fExo=[[], []])
    mod.train()
    yf = mod.forecast()
    assert yf.shape == (116, 6) and np.isfinite(yf).all()               # all rows are forecast (skip only affects the diagnosis)
    t0 = a.skip
    X_window = mod.X[t0: t0 + 5]
    X_3 = mod.forge(X_window, yf[t0: t0 + 5], 3)
    # columns are in DESCENDING lag order (yl3, yl2, yl1): at horizon 3 the lag-3 slot takes the actual
    # yl1 (= y_{t-1}), the lag-2 and lag-1 slots the forecasts of horizons 1 and 2 (y_t, y_{t+1})
    assert np.allclose(X_3[:, 0], X_window[:, 2])                            # yl3 slot <- actual yl1
    assert np.allclose(X_3[:, 1:3], yf[t0: t0 + 5, 0:2])                     # yl2, yl1 slots <- forecasts h = 1, 2
    assert np.allclose(X_3[:, 3], X_window[:, 5])                            # f0(yl3) slot <- actual f0(yl1)
    assert np.allclose(X_3[:, 4:6], yf[t0: t0 + 5, 0:2] ** 1.5)              # f0 slots <- transformed forecasts
    assert np.allclose(X_3[:, 6], X_window[:, 7]) and np.allclose(X_3[:, 7], X_window[:, 7])   # xe0 lags hidden (h > q)


def test_rescale_fits_the_transform_on_training_rows():
    """The rescaling transform sees the training rows only (ScalaTion's rescale): on a rising series the
    maximum of the testing rows is not the transform's maximum; MinMax maps the training rows to
    [-2, 2] (ScalaTion's centered default), so forecasts are not rectified there, and a power of a
    negative value is rejected."""
    t = np.arange(120.0)
    y = 100.0 + 2.0 * t + 10.0 * np.sin(t / 5.0)
    xe = np.column_stack([50.0 + t, 10.0 + 0.5 * t])
    mod = ARX_SR.rescale(y, 6, xe, hp_(method="sk_lr"), fEndo=[PowRForm()], fExo=[[], []])
    tf = mod._tForms["tForm_y"]
    tr = mod._tr_size
    assert isinstance(tf, MinMaxForm) and np.allclose(tf.f(y[:tr]).min(), -2.0) and np.allclose(tf.f(y[:tr]).max(), 2.0)
    assert tf.f(y).max() > 2.0                                               # the testing rows do not leak
    assert not mod._nneg                                                     # a centered space: no rectification
    with pytest.raises(ValueError):                                          # y^1.5 of a negative (rescaled) value
        ARX_SR.rescale(y, 6, xe, hp_(method="sk_lr"), fEndo=[PowForm()], fExo=[[], []])
    mod_n = ARX_SR_D.rescale(y, 6, xe, hp_(method="sk_lr"), fEndo=[PowRForm()], tFormT=TransformT.Norm)
    assert isinstance(mod_n.yForm, NormForm) and not mod_n._nneg and mod_n.hparameter["nneg"] == 0   # NormForm => nneg 0
    mod_l = ARX_SR_D.rescale(y, 6, xe, hp_(method="sk_lr"), fEndo=[PowForm()], tFormT=TransformT.Log1p)
    assert mod_l._nneg                                                       # Log1p keeps zero at zero
    mod.train(); yf = mod.forecast()
    assert np.isfinite(yf).all() and mod.Yf[10, 1] > 50, "Yf holds forecasts in the original scale"
    assert np.allclose(mod.Yf[:, 0], y)


def test_arx_sr_d_equals_narx_sr_d_on_the_equivalent_formula():
    """The direct SR model and the formula model build the same columns (in another order), so
    with OLS / no ridge they give the same forecasts -- the two paths of the project agree."""
    y, xe = covid()
    a = hp_(p=3, px=2, method="sm_ols")
    sr = ARX_SR_D(y, 6, xe, a, fEndo=[PowForm()], fExo=[[LogForm()], []])   # exact least squares
    sr.train()                                     # (sklearn's LinearRegression truncates the rank of this ill-conditioned X)
    yf_sr = sr.forecast()
    data = {"y": y, "icu": backfill_matrix(xe)[:, 0], "pRate": backfill_matrix(xe)[:, 1]}
    f = "y ~= y~(3,1) + pow(y~(3,1), 1.5) + icu~(2,1) + log1p(icu~(2,1)) + pRate~(2,1)"
    nx = NARX_SR_D.fromTerms(f, data, 6, Args(skip=3, lam=0.0))
    nx.train()
    yf_nx = nx.forecast()
    assert np.allclose(yf_sr, yf_nx, atol=1e-6)


def test_arx_d_rescale_uses_training_rows_too():
    t = np.arange(120.0)
    y = 100.0 + 2.0 * t
    xe = np.column_stack([50.0 + t])
    mod = ARX_D.rescale(y, 6, xe, hp_(method="sk_lr"))
    tf = mod._tForms["tForm_y"]
    assert np.isclose(tf.f(np.array([y[: mod._tr_size].max()]))[0], 2.0) and tf.f(y).max() > 2.0
