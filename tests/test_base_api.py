"""Tests of the base API shared with ScalaTion: the hyper-parameters (`AR.hp`, `HyperParameter`, the model's
private copy behind `hparameter`), the regression in the bases (`Forecaster_Reg` / `Forecaster_D`), the
coefficient table (`summary`, `vif`), `forecastAt`, `ARX.trend`, the rescaling by a `TransformT`, and the module
layout (imports at the top).  The settings object `Args` and `run (args)` are tested in test_args.py."""
import ast
import pathlib

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from Util.transform import MinMaxForm, NormForm, TransformT
from modeling.centered_ridge import CenteredRidge
from modeling.fit import QoF, vif
from modeling.forecasting.ar import AR
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.forecaster import hp as AR_hp
from modeling.forecasting.forecaster_d import Forecaster_D
from modeling.forecasting.random_walk import RandomWalk
from modeling.hyper_parameter import HyperParameter
from modeling.forecasting.datasets import Example_Covid

ROOT = pathlib.Path(__file__).resolve().parents[1]


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


def test_hyper_parameter_prints_as_scalation():
    hp = HyperParameter().add("p", 1, 1).add("lambda", 0.1, 0.1)
    hp["p"] = 6
    hp["skip"] = 6                                                     # an unregistered (Python-only) entry: not printed
    assert repr(hp) == "HyperParameter(p -> (6,1), lambda -> (0.1,0.1))" and hp.default("p") == 1
    assert dict(hp.updateReturn(("p", 0))) == {"p": 0, "lambda": 0.1, "skip": 6} and hp["p"] == 6
    assert AR.hp is AR_hp and list(AR_hp.registered()) == ["p", "d", "q", "P", "D", "Q", "s", "px", "spec", "crx", "nneg"]
    assert repr(AR_hp) == ("HyperParameter(p -> (1,1), d -> (1,1), q -> (1,1), P -> (1,1), D -> (1,1), Q -> (1,1), "
                           "s -> (7,7), px -> (1,1), spec -> (1,1), crx -> (0,0), nneg -> (1,1))")


def test_ar_hp_is_read_only_and_hparameter_is_a_copy():
    """`AR.hp` cannot be changed (a model could otherwise change the defaults of the models made after it); a model
    keeps a private HyperParameter, and `hparameter` returns a copy."""
    with pytest.raises(TypeError, match="read-only"):
        AR_hp["p"] = 6
    with pytest.raises(TypeError):
        AR_hp.update(p=6)
    c = AR_hp.copy()
    c["p"] = 6                                                         # a copy is writable
    assert AR_hp["p"] == 1 and c["p"] == 6 and not c.frozen and AR_hp.frozen
    y, xe = covid()
    mod = ARX_D(y, 3, xe, Args(p=3, px=2))
    h = mod.hparameter
    h["p"] = 9
    assert mod.hparameter["p"] == 3 and mod.modelName == "ARX_D_3_2_2" and not hasattr(mod, "hparam")


def test_defaults_are_ar_hp_and_report_line_is_the_scala_log(capsys):
    """ARX_D_1_0_0 on Covid-19 (aRX_DTest4): the REPORT's hyper-parameter line is the Scala log's."""
    y, _ = covid()
    mod = ARX_D(y, 6, None, Args(p=1, px=0, spec=0, method="ridge"))
    hp = mod.hparameter
    assert hp["d"] == 1 and hp["nneg"] == 1 and hp["skip"] == 1 and mod.method == "ridge"
    mod.inSample_Test()
    out = capsys.readouterr().out
    assert ("    hparameter hp  = HyperParameter(p -> (1,1), d -> (1,1), q -> (1,1), P -> (1,1), D -> (1,1), "
            "Q -> (1,1), s -> (7,7), px -> (0,1), spec -> (0,1), crx -> (0,0), nneg -> (1,1))") in out
    m = ARX(y, 2, None, Args(crx=True))                                # the switches are kept as 0 / 1
    assert m.hparameter["crx"] == 1 and m.hparameter["p"] == 1 and m.hparameter["px"] == 1 and m.modelName == "ARX_1_1_0"
    RandomWalk(y, 2).inSample_Test()
    assert "hparameter hp  = null" in capsys.readouterr().out          # ScalaTion's RandomWalk has hparam = null


def test_regression_lives_in_the_bases():
    """`Forecaster_D` fits every direct model (ScalaTion's `reg`); lambda defaults to `CenteredRidge.hp` (0.1)."""
    y, xe = covid()
    assert "train" in Forecaster_D.__dict__ and "train" not in ARX_D.__dict__
    mod = ARX_D(y, 3, xe, Args(p=3, px=2, spec=1, method="ridge", scaled=True))
    assert mod._model.lam == CenteredRidge.hp["lambda"] == 0.1 and mod.scaled
    mod.train()
    reg = CenteredRidge(0.1, scaled=True).fit(*[a[: len(y) - 2] for a in (mod.X, mod.Y)])
    assert np.allclose(mod.bb, reg.parameter) and np.allclose(mod.slopes, reg.slopes) and np.allclose(mod.intercept, reg.intercept)
    assert np.allclose(mod.parameter, reg.parameter[:, 0])            # b: the fitted (standardized) coefficients
    assert not np.allclose(mod.parameter, mod.slopes[:, 0])           # ... not the original-unit slopes
    with pytest.raises(NotImplementedError):
        Forecaster_D.getModel_h(mod)                                   # implemented by ARX_D, as in ScalaTion


def test_summary_and_vif():
    """The coefficient table (ScalaTion's `Fit.summary`): standard errors from (X' X)^-1 and the MSE of the last
    diagnosis, Student t p-values, VIFs as statsmodels' (column 0 the intercept)."""
    y, xe = covid()
    mod = ARX(y, 1, xe, Args(p=2, px=1, spec=0))
    mod.train()
    yp = mod.forecast()[:, 0]
    mod.mod_resetDF(len(y))
    mod.diagnose(y, yp)
    x1 = np.column_stack((np.ones(len(y)), mod.X))
    v = vif(x1)
    assert np.allclose(v, [variance_inflation_factor(x1, j) for j in range(1, x1.shape[1])])
    s = mod.summary()
    assert s.startswith("fname = Array(intercept, yl2, yl1, xe0l1, xe1l1)\nSUMMARY\n    Parameters/Coefficients:")
    rows = [ln for ln in s.splitlines() if ln.startswith("    x")]
    assert len(rows) == 5 and rows[0].endswith("\t NA")
    se = np.sqrt(np.diag(np.linalg.inv(x1.T @ x1)) * mod.qof[QoF.sse.ordinal] / mod.df)
    assert float(rows[1].split("\t ")[2]) == pytest.approx(se[1], rel=1e-5)
    ols = sm.OLS(y, x1).fit()
    assert np.allclose(mod.params, ols.params) and "F-statistic: " in s and " on 4.0 and " in s
    md = ARX_D(y, 3, xe, Args(p=2, px=1, spec=0))
    md.inSample_Test()
    assert md.summary(2).startswith("\nhorizon h = 2\nfname = Array(intercept,")    # ScalaTion's summary (h)


def test_forecast_at_and_trend():
    y, xe = covid()
    mod = ARX(y, 3, xe, Args(p=2, px=1, spec=1))
    mod.train()
    assert np.allclose(mod.forecastAt(2), mod.Yf[:, 2]) and mod.forecastAt(3).shape == (len(y),)
    with pytest.raises(ValueError):
        mod.forecastAt(4)
    tr = ARX.trend(y, 3, None, Args(spec=2))
    assert tr.fname == ["cons", "lin"] and tr.X.shape == (len(y), 2) and tr.modelName == "ARX_0_0_0"
    tr.train()
    assert np.isfinite(tr.forecast()).all()


def test_rescale_takes_a_transform_type():
    """`rescale (..., tFormT)` (ScalaTion's): MinMax to the centered [-2, 2] by default (no rectification there),
    Norm sets nneg = 0, Log1p keeps zero at zero; the transform is fitted on the training rows, column by column."""
    y, xe = covid()
    hp = Args(p=3, px=2, spec=0)
    m = ARX_D.rescale(y, 3, xe, hp)
    assert isinstance(m.yForm, MinMaxForm) and np.isclose(m.y[: m.tr_size].min(), -2) and not m._nneg
    assert np.allclose(m.getY(), y) and np.isclose(m.X[: m.tr_size, -1].max(), 2.0)   # xe rescaled per column
    n = ARX.rescale(y, 3, xe, hp.copy(tFormT=TransformT.Norm))
    assert isinstance(n.yForm, NormForm) and n.hparameter["nneg"] == 0 and hp.nneg   # the caller's args unchanged
    l1 = ARX_D.rescale(y, 3, xe, hp.copy(tFormT="Log1p"))
    assert l1._nneg and np.allclose(l1.y, np.log1p(y))
    a = AR.rescale(y, 3, Args(p=2))                                    # ScalaTion's AR.rescale: NormForm
    assert isinstance(a.yForm, NormForm) and not a._nneg


def _nested_imports(path: pathlib.Path):
    """The imports of a module that are not at its top level (nor under `if TYPE_CHECKING:`)."""
    tree = ast.parse(path.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top.add(id(node))
        elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            top.update(id(n) for n in node.body)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top:
            out.append(node.module if isinstance(node, ast.ImportFrom) else node.names[0].name)
    return out


def test_imports_are_at_the_top():
    """Every import of the package is at the top of its module, except the documented ones: the two genuine
    import cycles and the optional dependencies (torch, matplotlib)."""
    allowed = {("modeling/forecasting/nonlinear/narx_formula.py", "modeling.forecasting.nonlinear.narx_sr_d"),
               ("modeling/forecasting/nonlinear/llm_narx_sr/formula_scorer.py", "modeling.forecasting.nonlinear.llm_narx_sr.explainer"),
               ("modeling/forecasting/neuralforecasting/mhattn_regressor.py", "matplotlib.pyplot"),
               ("Util/data_loader.py", "torch")}
    found = set()
    for sub in ("modeling", "Util"):
        for f in (ROOT / sub).rglob("*.py"):
            rel = f.relative_to(ROOT).as_posix()
            found.update((rel, m) for m in _nested_imports(f))
    assert found <= allowed, sorted(found - allowed)
