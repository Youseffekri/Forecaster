"""Tests of the example loaders (ScalaTion's `Example_Covid`, `Example_ILI`, `Example_ETTh1`, `Example_GEFCom`:
loadData -> (xe, y), clip, loadData_y, loadData_yy, context) and the dataset configurations (datasetsTest2)."""
import numpy as np
import pytest

from modeling.forecasting.datasets import Datasets, Example_Covid, Example_ETTh1, Example_GEFCom, Example_ILI, loadH


def test_covid():
    ctx = Datasets.COVID2.load()
    assert set(ctx) == {"y", "icu", "pRate"} and len(ctx["y"]) == 116
    xe, y = Example_Covid.loadData(["icu_patients", "hosp_patients"])        # the whole file, as ScalaTion's
    assert xe.shape == (169, 2) and y.shape == (169,) and y[0] == 1.0
    xe, y = Example_Covid.clip((xe, y))                                       # ScalaTion: clip (loadData (...))
    assert xe.shape == (116, 2) and y.shape == (116,)
    _, y1 = Example_Covid.clip((None, Example_Covid.loadData_y()))
    assert _ is None and np.array_equal(y1, y)
    assert Example_Covid.loadData(Example_Covid.NO_EXO)[0].shape == (169, 0) and Example_Covid.BEST_EXO == ("icu_patients",)
    yy = Example_Covid.loadData_yy(["new_deaths", "icu_patients"], trim=1)
    assert yy.shape == (168, 2) and yy[0, 0] == Example_Covid.loadData_y()[1]
    assert set(Example_Covid.context(Example_Covid.BEST2_EXO)) == {"y", "icu_patients", "positive_rate"}   # names default x_strs
    with pytest.raises(KeyError, match="unknown column"):
        Example_Covid.loadData(["icu"])
    mat, header = loadH(Example_Covid.fileName)
    assert mat.shape == (169, 16) and tuple(header) == Example_Covid.header


def test_etth1():
    ctx = Datasets.ETTH1.load()
    assert len(ctx["y"]) == 17420 and set(ctx) == {"y", "hufl", "hull", "mufl", "mull", "lufl", "lull"}
    ctx1 = Datasets.ETTH1_1Y.load()
    assert len(ctx1["y"]) == 8760 and ctx1["y"][0] == pytest.approx(30.531, abs=1e-3)
    assert (ctx["y"] < 0).any(), "OT goes negative: use nneg=False for this series"
    xe, y = Example_ETTh1.loadData(Example_ETTh1.LOADS, Example_ETTh1.response, range(0, 24))
    assert xe.shape == (24, 6) and np.array_equal(y, Example_ETTh1.loadData_y(rng=range(0, 24)))


def test_gefcom():
    ctx = Datasets.GEF14.load()
    assert len(ctx["y"]) == 2099 and set(ctx) == {"y", "temp", "tmax"}
    v = Example_GEFCom.v_table()
    assert [round(x) for x in v] == [272, 226, 185, 148, 136, 214, 273], "the V shape of load vs temperature"
    assert set(Example_GEFCom.context()) == {"y", "temp", "tmax", "tmin"} and len(Example_GEFCom.loadData_y()) == 2099


def test_ili_loads_ilinet():
    ctx = Datasets.ILI.load()
    assert len(ctx["y"]) == 1208 and set(ctx) == {"y", "wili", "ot"}
    assert ctx["y"][0] == 2060 and ctx["ot"][0] == 176569                  # 10/6/2002: ILITOTAL, TOTAL PATIENTS
    xe, y = Example_ILI.loadData(["%WEIGHTED ILI", "NUM. OF PROVIDERS"], Example_ILI.response, trim=8)
    assert y.shape == (1200,) and xe.shape == (1200, 2) and np.array_equal(y, Example_ILI.loadData_y("ILITOTAL", trim=8))
    assert Example_ILI.loadData_yy(["ILITOTAL", "TOTAL PATIENTS"]).shape == (1208, 2)


def test_config_helpers():
    cfg = Datasets.COVID2
    assert cfg.baselineFormulaStr(6) == "y ~= y~(6,1) + icu~(6,1) + pRate~(6,1)"
    assert cfg.blindNames == {"y": "y", "icu": "x1", "pRate": "x2"}
    assert cfg.shortlist(cfg.load(), 92, 2, 6) == ["icu", "pRate"]          # Given: the list as is
    assert Datasets.get("gef14") is Datasets.GEF14
    with pytest.raises(KeyError):
        Datasets.get("nope")
