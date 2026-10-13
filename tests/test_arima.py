"""Tests of the statsmodels-backed ARIMA under the ScalaTion Forecaster interface (Covid-19, 116 rows,
hh 3, rc 20): shapes, the forecast matrix convention, agreement with statsmodels' own apply/forecast,
ARIMA(1,0,0) against AR(1), the printed REPORT and the rolling validation."""
import warnings

import numpy as np
import pytest

from modeling.fit import QoF
from modeling.forecasting.ar import AR
from modeling.forecasting.arima import ARIMA
from modeling.forecasting.args import Args
from modeling.forecasting.forecaster import teRng, trSize
from modeling.forecasting.datasets import Example_Covid

HH = 3
RC = 20


def covid_y():
    return Example_Covid.clip((None, Example_Covid.loadData_y()))[1]      # ScalaTion: clip ((null, loadData_y ()))


def test_defaults_names_and_shapes():
    y = covid_y()
    mod = ARIMA(y, HH)                                                   # ScalaTion's AR.hp defaults p = d = q = 1
    assert mod.modelName == "ARIMA_1_1_1" and mod.order == (1, 1, 1) and mod.trend == "n"
    assert {k: mod.hparameter[k] for k in ("p", "d", "q", "trend", "skip")} == {"p": 1, "d": 1, "q": 1, "trend": "n", "skip": 3}   # skip = max (p, q + 1) + d
    assert mod.fname == ["ar.L1", "ma.L1"] and not mod.hasIntercept
    mod.train()
    assert mod.parameter.shape == (2,) and mod.params.shape == (2,) and mod.intercept is None
    assert mod.sigma2 > 0
    yf = mod.forecast()
    assert yf.shape == (len(y), HH) and mod.Yf.shape == (len(y), HH + 2)
    assert np.array_equal(mod.Yf[:, 1:HH + 1], yf) and np.all(yf >= 0)
    assert repr(mod) == f"ARIMA_1_1_1(trend='n'), hh = {HH}, skip = 3"


def test_trend_c_gives_a_constant_and_an_intercept():
    y = covid_y()
    mod = ARIMA(y, HH, Args(p=2, d=0, q=1))                              # d = 0 => trend "c" by default
    assert mod.trend == "c" and mod.hasIntercept and mod.fname == ["const", "ar.L1", "ar.L2", "ma.L1"]
    mod.train()
    assert mod.intercept == pytest.approx(float(mod.parameter[0]))
    assert mod.modelName == "ARIMA_2_0_1"
    with pytest.raises(ValueError):
        ARIMA(y, HH, Args(p=-1))


@pytest.mark.parametrize("args, trend", [(Args(p=1, d=0, q=0), "c"), (Args(p=1, d=1, q=1), None),
                                         (Args(p=2, d=1, q=0), "t"), (Args(p=1, d=0, q=2), "ct")])
def test_row_t_equals_apply_forecast_from_the_information_up_to_t_minus_1(args, trend):
    """Row t of the block holds the h = 1..hh forecasts of y[t], ..., y[t + hh - 1] made from y[:t]:
    statsmodels' own res.apply (y[:t]).forecast (hh)."""
    y = covid_y()
    mod = ARIMA(y, HH, args, trend=trend)
    mod.train(y[:trSize(len(y))])
    yf = mod.forecast()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in (1, 2, 5, 40, 92, 115):
            ref = np.asarray(mod.res.apply(y[:t], refit=False).forecast(HH))
            assert np.allclose(yf[t], np.maximum(ref, 0.0), atol=1e-8), (args, trend, t)


def test_arima_1_0_0_with_trend_c_is_close_to_ar_1():
    """ARIMA(1,0,0) with a constant is the AR(1) in mean form: phi close to the Yule-Walker / OLS phi
    and const close to the mean; the h = 1 forecasts agree closely."""
    y = covid_y()
    mod = ARIMA(y, HH, Args(p=1, d=0, q=0), trend="c")
    mod.train()
    phi_sm = mod.parameter[mod.fname.index("ar.L1")]
    for method in ("sm_ols", "adjusted"):
        ar = AR(y, HH, Args(p=1, method=method))
        ar.train()
        phi_ar, delta = ar.parameter[0], ar.params[0]
        assert phi_sm == pytest.approx(phi_ar, abs=0.03)
        assert mod.intercept * (1.0 - phi_sm) == pytest.approx(delta, rel=0.35)   # delta = mean (1 - phi)
    yf_ar = ar.forecast()
    yf = mod.forecast()
    rel = np.abs(yf[2:, 0] - yf_ar[2:, 0]) / np.maximum(np.abs(yf_ar[2:, 0]), 1.0)
    assert np.median(rel) < 0.05


def test_in_sample_report_and_tnt_rolling(capsys):
    y = covid_y()
    mod = ARIMA(y, HH, Args(p=1, d=1, q=1))
    yp, qof = mod.inSample_Test()
    out = capsys.readouterr().out
    assert "| In-Sample Test: ARIMA_1_1_1 |" in out and "modelName  mn  = ARIMA_1_1_1" in out
    assert "features   fn  = Array(ar.L1, ma.L1)" in out and "fitMap QoF =" in out
    assert "hparameter hp  = HyperParameter(p -> (1,1), d -> (1,1), q -> (1,1), P -> (1,1), D -> (1,1), Q -> (1,1), s -> (7,7), px -> (1,1), spec -> (1,1), crx -> (0,0), nneg -> (1,1))" in out
    assert qof[QoF.dfr.ordinal] == 2 and qof[QoF.m.ordinal] == len(y) - 3 and qof[QoF.rSq.ordinal] > 0.85

    ft = mod.tnT_Test(rc=RC)
    out = capsys.readouterr().out
    assert "| TnT Test: ARIMA_1_1_1 |" in out and "rollValidate: Evaluate ARIMA_1_1_1's QoF" in out
    assert ft.shape == (HH, 26)
    te = teRng(len(y))
    assert ft[0, QoF.m.ordinal] == len(te) and ft[HH - 1, QoF.m.ordinal] == len(te) - HH + 1
    assert np.all(mod.Yf[te.start:, 1] > 0) and ft[0, QoF.smape.ordinal] < 30
    assert mod.res.nobs == len(y) - te.start + 0 or mod.res.nobs == trSize(len(y))   # the last window is tr_size rows


def test_roll_validate_retrains_every_rc_rows_on_the_sliding_window():
    y = covid_y()
    mod = ARIMA(y, HH, Args(p=1, d=1, q=0))
    mod.train()
    b_full = mod.parameter.copy()
    yf = mod.rollValidate(rc=RC)
    te = teRng(len(y))
    assert yf.shape == (len(te), HH)
    assert mod.res.nobs == trSize(len(y)), "the last retraining used a window of tr_size rows"
    assert not np.allclose(mod.parameter, b_full), "the window's parameters differ from the full fit's"
    assert np.all(mod.Yf[:te.start, 1:HH + 1] == 0), "rollValidate resets Yf and forecasts the testing rows only"
