"""ForecasterPy rev2.1 (consistent with ScalaTion rev5.2): the boundary rule of the rolling tests (a direct
model trains only on the rows whose hh targets lie before the forecast origin; the single-horizon model at
horizon h stops h - 1 rows early), `valTest` (ScalaTion's name), `Model.fitRows`, `Model.TE_RATIO = 0` (no
testing set: a competition's dataset) with `args.te_ratio = None` (the model takes `Model.TE_RATIO` when it
is built), and `tune (search = "golden")` (rev2.2: was `findLambda`)."""
import contextlib
import inspect
import io

import numpy as np
import pytest

from Util.transform import TransformT
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.datasets import Example_Covid
from modeling.forecasting.forecaster import Forecaster, Model, teRng, trSize
from modeling.forecasting.forecaster_d import Forecaster_D
from modeling.centered_ridge import CenteredRidge
from modeling.hyper_parameter import Knob


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def params(f):
    return list(inspect.signature(f).parameters)


# -- the boundary rule -----------------------------------------------------------------------------------

def test_direct_training_window_stops_hh_minus_1_rows_before_the_origin():
    y, xe = covid()
    hh = 6
    mod = ARX_D(y, hh, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    Y_, X_ = mod._trainWindow(0, 92)                                     # the window before origin 92
    assert Y_.shape[0] == X_.shape[0] == 92 - hh + 1                       # rows 0 .. 86: their targets y[r .. r+5] < 92
    assert np.array_equal(Y_, mod.Y[:87]) and np.array_equal(X_, mod.X[:87])
    assert np.all(np.isfinite(Y_)) and mod.fs_rng == range(0, 87)          # the FS rows are cut the same way
    with pytest.raises(ValueError, match="empty"):
        mod._trainWindow(85, 90)
    # the first retraining of the rolling test IS a fit on that window: the parameters agree
    ridge = CenteredRidge(lam=1.0)
    ridge.fit(mod.X[:87], mod.Y[:87])
    quiet(mod.rollValidate, rc=100)                                        # one retraining at i = 0 (origin 92)
    assert np.allclose(mod._b, ridge.parameter)


def test_direct_tnt_numbers_moved_from_rev2_0_by_the_boundary_rule():
    """The rev2.0 window [0, 92) held five rows with targets inside the testing block; the rev2.1 window is
    [0, 87) -- the honest number is a little worse (sMAPE h = 1 on Covid, lambda = 1: 22.4 vs 21.9)."""
    y, xe = covid()
    mod = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    ft = quiet(mod.tnT_Test, rc=5, verbose=False)
    assert ft.shape == (6, 26) and ft[0, 11] == 24                         # diagnosed on the 24 testing rows
    leaky = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    leaky._trainWindow = lambda is_, t: (leaky.Y[is_:t], leaky.X[is_:t])    # rev2.0's window, for the comparison only
    ftL = quiet(leaky.tnT_Test, rc=5, verbose=False)
    assert not np.allclose(ft[:, 4], ftL[:, 4])                             # the sMAPE column differs


def test_single_horizon_model_at_h_has_lead_h_minus_1():
    y, xe = covid()
    mod = ARX_D(y, 6, xe, Args(p=4, px=3, spec=1, lam=1.0, method="ridge"))
    m3 = mod.getModel_h(h=3)
    assert m3._lead == 2 and m3.mSeries == 116 and len(m3.y) == 114 and m3.tr_size == 92
    y_, X_ = m3._trainWindow(0, 92)
    assert y_.shape[0] == X_.shape[0] == 90                                 # rows 0 .. 89: their target y[r + 2] < 92
    assert m3.fs_rng == range(0, 90) and m3.val_rng == range(68, 92)
    ft = quiet(m3.tnT_Test, verbose=False)                                 # the testing rows 92 .. 114 (the last two have no target)
    assert ft.shape == (1, 26) and ft[0, 11] == 22
    m1 = mod.getModel_h(h=1)
    assert m1._lead == 0 and m1.fs_rng == range(0, 92) and quiet(m1.tnT_Test, verbose=False)[0, 11] == 24
    assert mod.getModel([0, 1, 2])._lead == 0 and m3.getModel([0, 1])._lead == 2   # a sub-model keeps its lead


def test_recursive_models_are_unchanged_by_the_rule():
    y, xe = covid()
    mod = ARX(y, 6, xe, Args(p=4, px=3, spec=1, lam=1.0, method="ridge"))
    y_, X_ = mod._trainWindow(0, 92)
    assert y_.shape[0] == X_.shape[0] == 92 and mod._lead == 0


# -- valTest, tvT_Test ------------------------------------------------------------------------------------

def test_val_test_is_scalations_name_and_signature():
    assert not hasattr(Forecaster, "validate")
    assert params(Forecaster.valTest) == ["self", "rc", "showYf", "showParams", "verbose"]
    assert params(Forecaster.tnT_Test) == ["self", "rc", "showYf", "showParams", "verbose"]
    y, xe = covid()
    mod = ARX_D(y, 6, xe, Args(p=3, px=2, spec=1, lam=1.0, method="ridge"))
    qv = quiet(mod.valTest, verbose=False)
    assert qv.shape == (6, 26) and qv[0, 11] == 24                         # rows 68 .. 92
    assert mod.valScore() == pytest.approx(Forecaster.scoreOf(qv, inSample=False))
    qv2, qt = quiet(mod.tvT_Test, verbose=False)
    assert np.allclose(qv, qv2) and qt[0, 11] == 24


# -- Model.fitRows, TE_RATIO = 0 --------------------------------------------------------------------------

def test_model_fit_rows():
    assert params(Model.fitRows) == ["m", "te_ratio", "trSize"]
    assert Model.fitRows(116) == 92 and Model.fitRows(116, 0.2, 50) == 50 and Model.fitRows(116, 0.3) == 81
    with Model.withValidation():
        assert Model.fitRows(116) == 68 and Model.fitRows(116, None, 70) == 70
    import modeling.forecasting.arx as arx
    assert not hasattr(arx, "fitRows")                                      # one name: Model.fitRows


def test_te_ratio_zero_is_the_competition_mode():
    y, xe = covid()
    args = Args(p=3, px=2, spec=1, lam=1.0, method="ridge")
    assert args.te_ratio is None                                            # None: Model.TE_RATIO when the model is built
    try:
        Model.TE_RATIO = 0.0
        assert Model.teSize(116) == 0 and teRng(116) == range(116, 116) and Model.splitStr(116) == "92 / 24 / 0"
        mod = ARX_D.rescale(y, 6, xe, args, TransformT.MinMax)            # transforms on all 116 rows
        assert mod.te_ratio == 0.0 and mod.tr_size == 116 and mod.te_size == 0
        assert mod._tForms["tForm_y"]._hi == pytest.approx(y.max())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert mod.tnT_Test() is None                                   # nothing to test: a flaw, None
        assert "flaw @ rollTest" in out.getvalue()
        qv = quiet(mod.valTest, verbose=False)                              # the chooser's view: rows 92 .. 116
        assert qv.shape == (6, 26) and qv[0, 11] == 24
        qv2, qt = quiet(mod.tvT_Test, verbose=False)
        assert qt is None and np.allclose(qv, qv2)
        with Model.withValidation():
            modV = ARX_D.rescale(y, 6, xe, args, TransformT.MinMax)        # transforms on the first 92 rows
            assert modV._tForms["tForm_y"]._hi == pytest.approx(y[:92].max())
            hp2, sc = quiet(modV.tune, (Knob("lambda", (0.1, 1.0, 10.0), True, "L"),))
            assert hp2["lambda"] in (0.1, 1.0, 10.0) and np.isfinite(sc)
        final = mod.withHP(hp2)
        quiet(final.inSample_Test)                                          # the model entered: fit on all 116 rows
        assert final.lam == hp2["lambda"] and final.tr_size == 116
    finally:
        Model.TE_RATIO = 0.2
    assert ARX_D(y, 6, xe, args).tr_size == 92                              # back to the default split
    direct = ARX_D(y, 6, xe, args.copy(te_ratio=0.0))                       # or per model
    assert direct.tr_size == 116 and quiet(direct.tnT_Test) is None
    assert ARX_D(y, 6, xe, args.copy(te_ratio=0.3)).tr_size == trSize(116, 0.3) == 81
    with pytest.raises(ValueError, match="te_ratio"):
        Args(te_ratio=0.96)


def test_a_model_takes_model_te_ratio_when_it_is_built():
    y, xe = covid()
    try:
        Model.TE_RATIO = 0.3
        mod = ARX_D(y, 3, xe, Args(p=2, px=1, spec=0))
        assert mod.te_ratio == 0.3 and mod.tr_size == 81
        Model.TE_RATIO = 0.2
        assert mod.te_ratio == 0.3 and mod.getModel([0, 1]).te_ratio == 0.3   # a built model keeps its split
    finally:
        Model.TE_RATIO = 0.2


# -- tune: golden -----------------------------------------------------------------------------------------

def test_tune_golden_is_the_one_search():
    y, xe = covid()
    mod = ARX_D(y, 6, xe, Args(p=4, px=3, spec=1, lam=1.0, method="ridge"))
    small = (Knob("lambda", (0.1, 1.0, 10.0), True, "L"),)
    with Model.withValidation():
        g, gs = quiet(mod.tune, small)
        hp2, sc = quiet(mod.tune, small, search="golden")
        assert sc <= gs + 1e-12 and 0.1 <= hp2["lambda"] <= 10.0
        with pytest.raises(ValueError, match="golden"):
            mod.tune(search="brent")                                        # scipy's Brent is gone: one algorithm in both code bases
    import modeling.forecasting.forecaster as fc
    assert "minimize_scalar" not in dir(fc)
