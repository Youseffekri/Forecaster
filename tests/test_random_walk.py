"""Tests of RandomWalk (the Python counterpart of ScalaTion's RandomWalk: randomWalkTest3/4 on Covid-19,
randomWalkTest5/6 on small vectors, the companion's make)."""
import numpy as np
import pytest

from modeling.fit import QoF
from modeling.forecasting.forecaster import teRng, trSize
from modeling.forecasting.random_walk import RandomWalk
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Example_Covid

HH = 3


def covid_y():
    y, _ = Example_Covid.loadData(["icu_patients", "positive_rate"])
    return y


def test_no_parameters_and_the_scala_names():
    y = covid_y()
    mod = RandomWalk(None, y, HH)
    assert mod.modelName == "RandomWalk"
    assert mod.fname == ["no-x features"]
    assert mod.parameter.shape == (0,) and mod.params.shape == (0,)
    assert mod.intercept is None
    assert mod.hparam == {"skip": 1} and mod.getSkip() == 1               # ScalaTion: skip = max (1, AR.hp("p")) = 1
    mod.train()                                                          # a no-op
    assert mod.parameter.shape == (0,)
    assert repr(mod) == f"RandomWalk(), hh = {HH}, skip = 1"


def test_args_first_order_is_accepted():
    y = covid_y()
    a = RandomWalk({"skip": 2}, y, HH)
    b = RandomWalk({"skip": 2}, y, HH)
    assert a.hparam == b.hparam == {"skip": 2} and a.hh == b.hh == HH


def test_forecast_is_the_last_value_for_every_horizon():
    y = covid_y()
    mod = RandomWalk(None, y, HH)
    mod.train()
    yf = mod.forecast()
    assert yf.shape == (len(y), HH)
    assert np.array_equal(yf[1:, 0], y[:-1]), "h = 1 forecasts are y shifted by one"
    assert yf[0, 0] == y[0], "row 0 holds y(0) (ScalaTion's predict (0, y) = y(max0 (-1)))"
    for h in range(1, HH):
        assert np.array_equal(yf[:, h], yf[:, 0]), "every horizon repeats the last value"
    assert np.array_equal(mod.Yf[:, 1:HH + 1], yf) and np.array_equal(mod.Yf[:, 0], y)
    assert mod.predict(10) == y[9] and mod.predict(0) == y[0]


def test_in_sample_prints_the_scala_blocks(capsys):
    y = covid_y()
    mod = RandomWalk(None, y, HH)
    yp, qof = mod.inSample_Test()
    out = capsys.readouterr().out
    assert "| In-Sample Test: RandomWalk |" in out and "| diagnoseAll: RandomWalk |" in out
    assert "b = VectorD(), fn = Array(no-x features)" in out
    assert "modelName  mn  = RandomWalk" in out and "parameter  b   = VectorD()" in out
    assert "features   fn  = Array(no-x features)" in out and "fitMap QoF =" in out
    assert qof[QoF.dfr.ordinal] == 0 and qof[QoF.m.ordinal] == len(y) - 1     # skip 1, no slopes
    assert qof[QoF.rSq.ordinal] == pytest.approx(qof[QoF.rSqBar.ordinal])     # dfr = 0 => r_df = 1
    assert np.array_equal(yp[1:], y[:-1])


def test_tnt_h1_forecasts_equal_y_shifted_and_reproduce_the_scala_smape():
    """TnT with rolling validation (rc 20): the testing rows' h = 1 forecasts are y shifted by one and
    the sMAPE per horizon is ScalaTion's example_CovidTest4 RW row (18.1532, 27.2211, 40.3519, ...)."""
    y = covid_y()
    mod = RandomWalk(None, y, 6)
    ft = mod.tnT_Test(rc=20)
    te = teRng(len(y))
    assert np.array_equal(mod.Yf[te.start:, 1], y[te.start - 1:-1])
    assert ft.shape == (6, 26)
    assert np.allclose(ft[:, QoF.smape.ordinal], [18.1532, 27.2211, 40.3519, 52.3739, 62.5276, 73.6424], atol=1e-4)
    assert ft[0, QoF.m.ordinal] == len(y) - trSize(len(y))              # no rows skipped in TnT
    assert mod.getSkip() == 0


def test_small_vectors_of_the_scala_tests():
    y5 = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=float)      # randomWalkTest5
    mod = RandomWalk(None, y5, 3)
    mod.train()
    yf = mod.forecast()
    assert np.array_equal(yf[1:, 0], y5[:-1]) and np.all(yf[1:, 0] == y5[1:] - 2)
    y6 = np.array([1, 3, 4, 2, 5, 7, 9, 8, 6, 3], dtype=float)           # randomWalkTest6
    _, qof = RandomWalk(None, y6, 1).inSample_Test()
    e = y6[1:] - y6[:-1]                                                 # the errors of rows 1..9 (skip 1)
    assert qof[QoF.sse.ordinal] == pytest.approx(float(e @ e)) == 40.0


def test_make_is_a_random_walk_of_bounded_integer_steps():
    yy = RandomWalk.make(3, 2, 50, stream=2)
    assert yy.shape == (50,) and yy.dtype == float
    steps = np.diff(np.concatenate([[3.0], yy]))
    assert np.all(np.abs(steps) <= 2) and np.all(steps == np.round(steps))
    assert np.array_equal(yy, RandomWalk.make(3, 2, 50, stream=2)), "the same stream gives the same walk"
    assert not np.array_equal(yy, RandomWalk.make(3, 2, 50, stream=3))
