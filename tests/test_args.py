"""Tests of `Args` (the checked settings of a model and of a run) and of `run (args)`: names, checks and
messages, copies and presets, the private snapshot a model keeps, the settings each model reads, and the
runner reproducing a model built by hand."""
import copy
import pickle

import numpy as np
import pytest

from Util.transform import NormForm, TransformT
from modeling.centered_ridge import CenteredRidge
from modeling.forecasting.ar import AR
from modeling.forecasting.arima import ARIMA
from modeling.forecasting.args import OPTIMIZERS, SETTINGS, Args, hp
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.forecaster import Forecaster, teRng, trSize
from modeling.forecasting.datasets import Example_Covid
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D, OptimizerNL
from modeling.forecasting.random_walk import RandomWalk
from modeling.forecasting.runner import loadData, makeModel, run


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


# -- the settings and their checks ------------------------------------------------------------

def test_defaults_are_scalation_defaults():
    a = Args()
    assert a.hparam() == {**{k: hp[k] for k in hp}, "lambda": CenteredRidge.hp["lambda"]}
    assert repr(a.hparam()) == repr(hp)                                 # printed as ScalaTion prints AR.hp
    assert (a.p, a.px, a.spec, a.crx, a.nneg, a.skip, a.lam, a.te_ratio) == (1, 1, 1, False, True, None, 0.1, 0.2)
    assert repr(a) == "Args()" and a.changed() == {}
    assert OPTIMIZERS == tuple(o.name for o in OptimizerNL)             # the names Args checks are OptimizerNL's
    assert [n for n, s in SETTINGS.items() if "AR.hp" in s.section] == list(hp.registered())


def test_unknown_names_say_what_was_meant():
    a = Args()
    with pytest.raises(AttributeError, match=r"no setting 'pX'; did you mean 'px'\?"):
        a.pX = 4
    with pytest.raises(AttributeError, match=r"did you mean 'lam'"):
        a.lamda = 1.0
    with pytest.raises(AttributeError, match=r"'te_ratio' \(the TESTING fraction"):
        a.training_ratio = 0.8
    with pytest.raises(AttributeError, match=r"did you mean 'hh'"):
        Args(max_horizon=6)
    with pytest.raises(AttributeError, match="no setting 'nothing_like_it'  "):
        a.nothing_like_it
    with pytest.raises(AttributeError):
        a.update({"spc": 2})


@pytest.mark.parametrize("name, value", [("p", -1), ("p", 1.5), ("p", True), ("spec", 6), ("s", 0), ("crx", 2),
                                         ("nneg", "yes"), ("lam", -0.1), ("te_ratio", 1.0), ("te_ratio", 0.0),
                                         ("method", "ols"), ("tFormT", "Square"), ("optimizer", "Newton"),
                                         ("model", "LSTM"), ("validation", "kfold"), ("exo", 3), ("skip", -2)])
def test_bad_values_are_refused(name, value):
    with pytest.raises(ValueError, match=f"Args.{name} = "):
        setattr(Args(), name, value)


def test_values_are_normalized():
    a = Args(p=6.0, crx=1, nneg=0, lam=1, tFormT="log1p", optimizer="lbfgs-b", model="arx_sr_d", dataset="covid",
             validation="TnT", exo="icu_patients", method="SM_OLS")
    assert a.p == 6 and type(a.p) is int and a.crx is True and a.nneg is False and a.lam == 1.0
    assert a.tFormT is TransformT.Log1p and a.optimizer == "LBFGS_B" and a.model == "ARX_SR_D"
    assert a.dataset == "Covid-19" and a.validation == "tnT" and a.exo == ("icu_patients",) and a.method == "sm_ols"
    assert Args(tFormT=TransformT.Norm).tFormT is TransformT.Norm and Args(tFormT="MinMaxForm").tFormT is TransformT.MinMax
    assert Args(optimizer=OptimizerNL.Joint).optimizer == "Joint" == Args(optimizer="joint").optimizer   # kept by name
    b = Args(**{"lambda": 2.0})                                         # "lambda" (a Python keyword) is an alias of lam
    assert b.lam == 2.0 and b["lambda"] == 2.0 and b.hparam()["lambda"] == 2.0


def test_copies_changes_presets(tmp_path):
    a = Args(p=6, px=4)
    b = a.copy(px=2)
    assert (a.px, b.px) == (4, 2) and a != b and a == a.copy()
    a.update(spec=0, crx=True).update({"skip": 6})
    assert a.changed() == {"p": 6, "px": 4, "spec": 0, "crx": True, "skip": 6}
    assert repr(a) == "Args(p=6, px=4, spec=0, crx=True, skip=6)"
    del a.crx                                                            # back to the default
    a.reset("skip")
    assert a.crx is False and a.skip is None and Args.default("p") == 1
    a.update(tFormT=TransformT.Log1p, optimizer=OptimizerNL.VarPro, exo=["icu_patients", "positive_rate"])
    path = tmp_path / "covid.json"
    a.save(path)
    assert Args.load(path) == a == Args(p=6, px=4, spec=0, tFormT="Log1p", optimizer="VarPro", exo=["icu_patients", "positive_rate"])
    assert pickle.loads(pickle.dumps(a)).changed().keys() == a.changed().keys() and copy.deepcopy(a).p == 6
    assert Args.of(None) == Args() and Args.of({"p": 3}) == Args(p=3) and Args.of(a) is not a
    with pytest.raises(TypeError):
        Args.of(3)
    assert "px" in dir(a) and "lam" in dir(a)                            # tab completion lists the settings


def test_help_lists_the_settings_a_model_reads(capsys):
    a = Args(p=6)
    text = a.describe(ARX_D)
    assert text.startswith("Args read by ARX_D")
    assert "  * p          = 6" in text and "    px         = 1" in text and " d " not in text and "dataset" not in text
    assert "optimizer" not in text and "lam" in text
    assert "optimizer" in Args.describe(NARX_SR_D) and "  method " not in Args.describe(NARX_SR_D)
    assert set(RandomWalk.ARGS) == {"skip", "nneg", "te_ratio"} and "d" in ARIMA.ARGS and "method" in AR.ARGS
    Args.help()
    out = capsys.readouterr().out
    assert out.startswith("Args  (* = changed from the default)") and "*" not in out.split("\n", 1)[1].split("default")[0]
    a.help(ARX)
    assert capsys.readouterr().out.startswith("Args read by ARX")


# -- the models and their snapshots ----------------------------------------------------------------

def test_a_model_keeps_a_snapshot_of_its_args():
    y, xe = covid()
    args = Args(p=3, px=2, spec=0)
    mod = ARX_D(y, 3, xe, args)
    args.p = 6                                                           # later changes do not reach the model
    assert mod.hparameter["p"] == 3 and mod.modelName == "ARX_D_3_2_2" and mod.X.shape[1] == 3 + 2 * 2
    sub = mod.getModel([0, 1])
    assert sub.hparameter["p"] == 3 and sub.method == mod.method == "sm_ols"


def test_the_report_prints_the_args_values(capsys):
    """The REPORT prints the hyper-parameters as ScalaTion prints the hp it is given: every Scala entry of args."""
    y, xe = covid()
    ARX_D(y, 2, xe, Args(p=2, px=1, spec=0, q=2, nneg=False)).inSample_Test()
    out = capsys.readouterr().out
    assert ("hparameter hp  = HyperParameter(p -> (2,1), d -> (1,1), q -> (2,1), P -> (1,1), D -> (1,1), Q -> (1,1), "
            "s -> (7,7), px -> (1,1), spec -> (0,1), crx -> (0,0), nneg -> (0,1))") in out


def test_each_model_checks_its_method_and_ignores_what_it_does_not_read():
    y, xe = covid()
    with pytest.raises(ValueError, match="ARX_D: method 'adjusted'"):
        ARX_D(y, 3, xe, Args(method="adjusted"))
    with pytest.raises(ValueError, match="AR: method 'ridge'"):
        AR(y, 3, Args(method="ridge"))
    with pytest.raises(ValueError, match="ARIMA: method 'sm_ols'"):
        ARIMA(y, 3, Args(method="sm_ols"))
    shared = Args(p=2, px=1, spec=0, d=0, q=1, method=None, lam=1.0, tFormT="Log1p")
    rw, ar, arima, arx_d = RandomWalk(y, 3, shared), AR(y, 3, shared), ARIMA(y, 3, shared), ARX_D(y, 3, xe, shared)
    assert rw.method is None and ar.method == "adjusted" and arima.method is None and arx_d.method == "sm_ols"
    assert arima.order == (2, 0, 1) and ar.hparameter["px"] == 0 and ar.hparameter["spec"] == 0
    nx = NARX_SR_D.apply(y, 3, xe, Args(p=2, px=1, spec=0, method="sm_ols", tune=False))
    assert nx.method == "ridge" and nx.getModel([0, 1]).method == "ridge"    # NARX_SR_D always fits by ridge


def test_te_ratio_sets_the_split():
    y, xe = covid()
    mod = ARX_D(y, 3, xe, Args(p=2, px=1, spec=0, te_ratio=0.3))
    assert mod.te_ratio == 0.3 and mod.tr_size == trSize(len(y), 0.3) == 81 and mod.te_size == 35
    ft = mod.tnT_Test(rc=5)
    assert ft[0, 11] == len(teRng(len(y), 0.3))                          # the QoF's m: the testing rows
    assert ARX_D(y, 3, xe).tr_size == trSize(len(y)) == 92 and Forecaster.TE_RATIO == 0.2


def test_the_old_argument_order_is_refused():
    y, xe = covid()
    for build in (lambda: ARX_D({"p": 3}, y, 3, xe), lambda: AR(Args(p=2), y, 3), lambda: ARX.rescale({"p": 3}, y, 3, xe),
                  lambda: NARX_SR_D.apply(Args(), y, 3, xe)):
        with pytest.raises(TypeError, match=r"the data come first.*\(since rev1.4\)"):
            build()


# -- the runner ------------------------------------------------------------------------------------

def test_load_data_and_make_model():
    xe, y = loadData(Args(dataset="Covid-19"))                          # (xe, y), as the loaders return them
    y0, xe0 = covid()
    assert np.array_equal(y, y0) and np.array_equal(xe, xe0)
    assert loadData(Args(dataset="covid", exo=[]))[0] is None
    assert loadData(Args(dataset="covid", exo=["hosp_patients"]))[0].shape == (116, 1)
    with pytest.raises(ValueError, match="set args.dataset"):
        loadData(Args())
    m = makeModel(Args(model="ARX_D", tFormT="Norm", hh=3, p=2, px=1), y, xe)
    assert isinstance(m, ARX_D) and isinstance(m.yForm, NormForm) and m.hh == 3
    assert isinstance(makeModel(Args(model="AR", tFormT="MinMax"), y, xe).yForm, NormForm)   # ScalaTion's AR.rescale
    assert makeModel(Args(model="RandomWalk", tFormT="Norm"), y, xe).yForm is None
    assert isinstance(makeModel(Args(model="NARX_SR_D", tune=False, hh=2), y, xe), NARX_SR_D)


def test_run_reproduces_the_model_built_by_hand(capsys):
    args = Args(dataset="Covid-19", model="ARX_D", p=6, px=6, spec=0, hh=6, validation="tnT")
    mod = run(args)
    out = capsys.readouterr().out
    assert "| run: ARX_D on Covid-19 |" in out and "| TnT Test: ARX_D_6_6_2 |" in out and "In-Sample Test" not in out
    y, xe = covid()
    ref = ARX_D(y, 6, xe, args)
    ref.tnT_Test(args.rc)
    assert np.allclose(mod.Yf, ref.Yf)
    both = run(args.copy(validation="both", model="RandomWalk", hh=2))
    out = capsys.readouterr().out
    assert isinstance(both, RandomWalk) and "In-Sample Test: RandomWalk" in out and "TnT Test: RandomWalk" in out
