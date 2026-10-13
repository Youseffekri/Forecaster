"""Tests of the dataset loaders and configurations (counterpart of datasetsTest2)."""
import pytest

from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Datasets, Example_Covid, Example_GEFCom, Example_ILI


def test_covid():
    ctx = Datasets.COVID2.load()
    assert set(ctx) == {"y", "icu", "pRate"} and len(ctx["y"]) == 116
    y, xe = Example_Covid.loadData(["icu_patients", "hosp_patients"])
    assert xe.shape == (116, 2) and y[0] == 1.0


def test_etth1():
    ctx = Datasets.ETTH1.load()
    assert len(ctx["y"]) == 17420 and set(ctx) == {"y", "hufl", "hull", "mufl", "mull", "lufl", "lull"}
    ctx1 = Datasets.ETTH1_1Y.load()
    assert len(ctx1["y"]) == 8760 and ctx1["y"][0] == pytest.approx(30.531, abs=1e-3)
    assert (ctx["y"] < 0).any(), "OT goes negative: use nneg=False for this series"


def test_gefcom():
    ctx = Datasets.GEF14.load()
    assert len(ctx["y"]) == 2099 and set(ctx) == {"y", "temp", "tmax"}
    v = Example_GEFCom.v_table()
    assert [round(x) for x in v] == [272, 226, 185, 148, 136, 214, 273], "the V shape of load vs temperature"


def test_ili_loads_ilinet():
    ctx = Datasets.ILI.load()
    assert len(ctx["y"]) == 1208 and set(ctx) == {"y", "wili", "ot"}
    assert ctx["y"][0] == 2060 and ctx["ot"][0] == 176569                  # 10/6/2002: ILITOTAL, TOTAL PATIENTS
    y, xe = Example_ILI.loadData(["%WEIGHTED ILI", "NUM. OF PROVIDERS"], trim=8)
    assert y.shape == (1200,) and xe.shape == (1200, 2)


def test_config_helpers():
    cfg = Datasets.COVID2
    assert cfg.baselineFormulaStr(6) == "y ~= y~(6,1) + icu~(6,1) + pRate~(6,1)"
    assert cfg.blindNames == {"y": "y", "icu": "x1", "pRate": "x2"}
    assert Datasets.get("gef14") is Datasets.GEF14
    with pytest.raises(KeyError):
        Datasets.get("nope")
