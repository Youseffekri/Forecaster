"""The measurement protocol (rev2.2): Forecaster.errorTable, the HAC variance and the Diebold-Mariano test,
the Protocol runner (base models and arms), its summary and comparisons, and the CSV output."""
import contextlib
import io
import math

import numpy as np
import pytest

from modeling.fit import QoF, mae_n
from modeling.forecasting.args import Args
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.datasets import Datasets, Example_Covid
from modeling.forecasting.forecaster import Forecaster, Model
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import BaselineArm
from modeling.forecasting.nonlinear.llm_narx_sr.protocol import ERROR_COLUMNS, Protocol
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import Method
from modeling.forecasting.nonlinear.llm_narx_sr.significance import dm_test, hacVar, wilcoxon_test
from modeling.forecasting.random_walk import RandomWalk


def covid():
    xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO))
    return y, xe


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def test_error_table_reproduces_the_qof_smape_and_scales_by_the_naive_mae():
    y, xe = covid()
    mod = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
    qt = quiet(mod.tnT_Test)
    rng = Model.teRng(116)
    tab = mod.errorTable(rng)
    assert list(tab.columns) == ["t", "h", "y", "yhat", "e", "ae", "sape", "ase"]
    assert tab.groupby("h").size().tolist() == [24, 23, 22, 21, 20, 19]   # the pairs qofOn diagnoses: y[t+h-1] in the block
    assert tab["t"].min() == 92 and tab["t"].max() == 115 and (tab["t"] + tab["h"] - 1 <= 115).all()
    assert np.allclose(tab.groupby("h")["sape"].mean().to_numpy(), qt[:, QoF.smape.ordinal])   # a horizon's mean sape IS its sMAPE
    assert np.allclose(tab["e"], tab["y"] - tab["yhat"]) and np.allclose(tab["ae"], tab["e"].abs())
    scale = mae_n(y[:92], 1)                                                # MASE's denominator: the naive MAE of the rows before the block
    assert np.allclose(tab["ase"], tab["ae"] / scale)
    assert np.allclose(mod.errorTable(rng, scale=2.0)["ase"], tab["ae"] / 2.0)
    row = tab.iloc[0]
    assert row["t"] == 92 and row["h"] == 1 and row["y"] == y[92] and row["yhat"] == pytest.approx(mod.Yf[92, 1])
    rw = RandomWalk(y, 6)
    quiet(rw.tnT_Test)
    tw = rw.errorTable(rng)
    assert len(tw) == 129 and np.allclose(tw[tw["h"] == 1]["yhat"].to_numpy(), y[91:115])   # y_t+h = y_t-1


def test_hac_variance_and_the_diebold_mariano_test():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(200)
    assert hacVar(d, 0) == pytest.approx(d.var() / 200)                    # lag 0: the plain variance of the mean
    assert hacVar(d, 3) > 0 and math.isnan(hacVar(d[:1], 0))
    ma = np.convolve(rng.standard_normal(1001), np.ones(3), "valid")        # an MA(2): errors overlap two periods
    assert hacVar(ma, 2) > 1.5 * hacVar(ma, 0)                              # the overlap inflates the variance of the mean
    a, b = d, d + 0.5 + 0.3 * rng.standard_normal(200)                   # B's loss is larger by 0.5 on average
    dbar, dm, p = dm_test(a, b, h=1)
    assert dbar == pytest.approx(float((a - b).mean())) and dm < -4 and p < 1e-4   # A's loss is smaller: a negative DM*, tiny p
    dbar2, dm2, p2 = dm_test(b, a, h=1)
    assert dbar2 == pytest.approx(-dbar) and dm2 == pytest.approx(-dm) and p2 == pytest.approx(p)
    _, dm3, p3 = dm_test(a, b, h=4)                                          # the HLN correction shrinks |DM| for h > 1
    assert abs(dm3) < abs(dm) and p3 > p
    assert dm_test(a, a) == (0.0, 0.0, 1.0) and dm_test([1.0, 2.0], [2.0, 1.0])[2] == 1.0
    with pytest.raises(ValueError):
        dm_test([1.0, 2.0, 3.0], [1.0, 2.0])
    w, z, pw = wilcoxon_test(a, b)
    assert pw < 1e-4


def test_protocol_runs_models_and_arms_summarises_and_compares(tmp_path):
    y, xe = covid()
    P = Protocol(te_ratio=0.4, hh=6)
    assert P.rc == Forecaster.RC and P.loss == "sape" and list(P.errors.columns) == list(ERROR_COLUMNS)
    models = {"ARX_D ridge L1": lambda y, xe, hh: ARX_D(y, hh, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge")),
              "RandomWalk": lambda y, xe, hh: RandomWalk(y, hh)}
    cfg = Datasets.COVID2
    added = quiet(P.runModels, cfg.name, y, xe, models)
    assert Model.TE_RATIO == 0.2                                            # restored
    assert len(added) == 2 * (47 + 46 + 45 + 44 + 43 + 42) and set(added["model"]) == set(models)   # 47 origins: rows 69..115
    assert added["t"].min() == 69 and added["t"].max() == 115 and (added["stream"] == -1).all()
    arms = quiet(P.runArms, cfg, [BaselineArm(Method.RandomWalk, verbose=False)])
    assert set(arms["model"]) == {"random-walk"} and len(arms) == 267 and set(P.errors["model"]) == {"ARX_D ridge L1", "RandomWalk", "random-walk"}
    rwA = P.errors[P.errors["model"] == "RandomWalk"].sort_values(["t", "h"])["sape"].to_numpy()
    rwB = P.errors[P.errors["model"] == "random-walk"].sort_values(["t", "h"])["sape"].to_numpy()
    assert np.allclose(rwA, rwB)                                           # the arm's random walk IS the base's
    s = P.summary()
    assert list(s.columns) == ["dataset", "model", "h", "n", "smape", "smape_lo", "smape_hi", "mase", "mase_lo", "mase_hi"]
    assert sorted(s["h"].unique()) == [0, 1, 2, 3, 4, 5, 6] and (s[s["h"] == 1]["n"] == 47).all()
    r1 = s[(s["model"] == "RandomWalk") & (s["h"] == 1)].iloc[0]
    assert r1["smape_lo"] < r1["smape"] < r1["smape_hi"]
    allrow = s[(s["model"] == "RandomWalk") & (s["h"] == 0)].iloc[0]
    assert allrow["smape"] == pytest.approx(s[(s["model"] == "RandomWalk") & (s["h"] > 0)]["smape"].mean())   # the mean of the per-h means
    c = P.compare("ARX_D ridge L1", "RandomWalk")
    assert list(c["h"]) == [1, 2, 3, 4, 5, 6, 0] and (c["n"].iloc[:6] == [47, 46, 45, 44, 43, 42]).all() and c["n"].iloc[6] == 267
    assert np.allclose(c["diff"], c["lossA"] - c["lossB"]) and c.iloc[6]["p_dm"] < 0.01 and c.iloc[6]["diff"] < 0   # the ridge beats the random walk on this span
    same = P.compare("RandomWalk", "random-walk")
    assert len(same) == 7 and (same["diff"] == 0).all() and (same["p_dm"] == 1.0).all()
    ca = P.compareAll("RandomWalk")
    assert set(ca["modelA"]) == {"ARX_D ridge L1", "random-walk"} and (ca["modelB"] == "RandomWalk").all()
    with pytest.raises(ValueError):
        Protocol(loss="mse")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        P.report(reference="RandomWalk")
    assert "summary (mean over origins, 95 % HAC interval)" in out.getvalue() and "every model against RandomWalk" in out.getvalue()
    paths = P.write(str(tmp_path), "covid", reference="RandomWalk")
    assert [p.name for p in paths] == ["covid_errors.csv", "covid_summary.csv", "covid_comparisons.csv"] and all(p.exists() for p in paths)
