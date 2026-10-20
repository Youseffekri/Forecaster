"""Tests of EDA4TS (counterpart of eDA4TSTest), of Stationarity_KPSS, of Util/correlogram.py (correlogramTest2), of
`LLMProposer.promptBlock` and of the dataset configurations (ExoMode, the screening's shortlist -- the generic
`DatasetConfig.shortlist` and `FormulaScorer.shortlist`; counterparts of datasetsTest / datasetsTest2)."""
import math

import numpy as np
import pytest

from Util.correlogram import acfSpikes, acorr, ccorr, collinearPair, heteroscedastic, ljungBox
from modeling.forecasting.args import Args
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.datasets import Datasets, ExoMode
from modeling.forecasting.eda4ts import EDA4TS, topFreqs
from modeling.forecasting.exo_screen import ExoScreen
from modeling.model import Model
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.llm_proposer import LLMProposer
from modeling.forecasting.stationarity_kpss import Stationarity_KPSS


def planted():
    """y depends on its own lag 1, on z at lag 2 (positive) and has a 12-step cycle."""
    rng = np.random.default_rng(3)
    n = 320
    z = 10.0 + 2.0 * rng.standard_normal(n)
    y = np.zeros(n)
    y[0], y[1] = 20.0, 21.0
    for t in range(2, n):
        y[t] = 5.0 + 0.7 * y[t - 1] + 1.5 * z[t - 2] + 4.0 * math.sin(2.0 * math.pi * t / 12.0) + rng.standard_normal()
    return y, z


def test_profile_finds_the_planted_structure():
    y, z = planted()
    eda = EDA4TS(y, {"z": z}, trSize=256, maxLag=6, nFreqs=2)
    assert eda.pacfSpikes() and eda.pacfSpikes()[0][0] == 1, "PACF spike at lag 1 expected"
    nm, k, r = eda.bestLags()[0]
    assert nm == "z" and k == 2 and r > 0.0
    assert eda.signs()["z"] == 1
    assert any(abs(p - 12.0) < 2.0 for p in eda.periods), eda.periods
    assert eda.acf[0] == pytest.approx(1.0) and eda.pacf[0] == 0.0 and len(eda.acf) == 7
    assert 0.0 <= eda.kpssP <= 1.0 or math.isnan(eda.kpssP)
    block = LLMProposer.promptBlock(eda, {"y": "y", "z": "x1"})
    assert "z:" not in block and "x1" in block and block.startswith("Data profile (training rows only):")
    assert "autocorrelation at lags 1..6" in block and "wave(1)" in block


def test_residual_report_fails_a_wrong_model():
    """eDA4TSTest (rev4.1): y ~= y~1 as ARX_D (p = 1, px = 0, spec = 0) trained on the training rows;
    its h = 1 residuals miss z~2 and the cycle, so at least one residual test fails."""
    y, z = planted()
    tr = Model.trSize(len(y))
    eda = EDA4TS(y, {"z": z}, 256, maxLag=6)
    mod = ARX_D(y, 1, None, Args(p=1, px=0, spec=0))
    mod.train(mod.Y[:tr], mod.X[:tr])
    yf = mod.predict_rows(np.arange(len(y)))[:, 0]                       # the h = 1 forecast of every row
    e = (y - yf)[1:tr]
    lines, clean = eda.residualReport(e, mod.X[1:tr], mod.fname, y[1:tr])
    assert clean < 1.0 and any(l.startswith("Ljung-Box") for l in lines)


def test_kpss_is_scalations():
    """Stationarity_KPSS: the statistic of statsmodels' kpss, the p-value of ScalaTion's response surfaces
    (the Covid-19 training rows: 0.43 in lLMProposerTest's EDA block; statsmodels clips it to 0.10)."""
    y = Datasets.COVID2.load()["y"][:92]
    t = Stationarity_KPSS(y, 0, "short", "c")
    assert t.statistic() == pytest.approx(0.155764867609451, abs=1e-12) and t.lags == 3
    assert t.pvalue() == pytest.approx(0.42835275908751, abs=1e-9)
    assert EDA4TS(y, {}, 92).kpssP == pytest.approx(t.pvalue())
    assert Stationarity_KPSS(y, 5, "short").pvalue() == pytest.approx(0.5912343124293, abs=1e-9)   # own lags replace the rule
    assert Stationarity_KPSS(np.cumsum(np.arange(200.0)), 0, "short").pvalue() == pytest.approx(0.001)


def test_correlogram_helpers():
    """correlogramTest2: the hand-checked Ljung-Box statistic, AR(1) spikes, a proportional pair, level-proportional noise."""
    e0 = np.array([1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 8, 12], dtype=float)
    q, p = ljungBox(e0, 3)
    assert q == pytest.approx(10.931086991432, abs=1e-9) and p == pytest.approx(0.012104410741, abs=1e-4)
    rng = np.random.default_rng(7)
    n = 300
    g = rng.standard_normal(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.9 * e[t - 1] + g[t]
    assert ljungBox(g, 10)[1] > 0.001 and ljungBox(e, 10)[1] < 0.001
    sp = acfSpikes(e, 3)
    assert sp and sp[0][0] == 1 and acfSpikes(e, 3, {1, 2, 3}) == []
    xm = np.column_stack([np.arange(n, dtype=float), 2.0 * np.arange(n) + 1.0, g])
    bi, bj, bc = collinearPair(xm)
    assert (bi, bj) == (0, 1) and bc > 0.999 and collinearPair(xm, [2]) == (-1, -1, 0.0)
    lev = 10.0 + np.arange(n, dtype=float)
    assert heteroscedastic(lev * g, lev) and not heteroscedastic(g, lev)
    assert acorr(e, 1) == pytest.approx(float(np.corrcoef(e[1:], e[:-1])[0, 1]), abs=0.02)
    assert acorr(e, 3) == pytest.approx(acorr(e, 3, adjusted=False) * n / (n - 3))   # ScalaTion's default: adjusted
    assert ljungBox(e, 10)[1] == 0.0 and ljungBox(e, 10)[0] > 100.0 + 3.0 * 10      # the tail shortcut
    with pytest.raises(ValueError, match="not long enough"):
        acorr(e0, 12)
    x = rng.standard_normal(n)
    yy = np.concatenate([[0.0, 0.0], x[:-2]]) * 0.8 + 0.3 * rng.standard_normal(n)
    assert ccorr(x, yy, 2) > 0.8 > abs(ccorr(x, yy, 1))
    t = np.arange(256)
    s = np.sin(2 * np.pi * t / 12.0) + 0.5 * np.sin(2 * np.pi * t / 50.0)
    assert np.allclose(1.0 / topFreqs(s, 2), [12.19, 51.2], atol=0.1)


def test_exo_modes_and_shortlists():
    """datasetsTest: the file-based Covid-19 pool (8 eligible columns) and its shortlist (hospital
    patients and the positivity rate); COVID_ALL (auto) and the Given configurations."""
    assert Datasets.COVID2.exoMode is ExoMode.Given and Datasets.COVID_ALL.exoMode is ExoMode.Auto
    assert Datasets.ETTH1_1Y.exoMode is ExoMode.Auto and Datasets.GEF14.exoMode is ExoMode.Given
    cf = Datasets.COVID_FILE
    assert cf is Datasets.COVID_FILE and cf.exoMode is ExoMode.Auto                 # lazy, read once
    assert len(cf.exo) == 8 and "excess_mortality" not in cf.exo and "new_cases" not in cf.exo and "people_vaccinated" not in cf.exo
    data = cf.load()
    assert len(data) == len(cf.exo) + 1 and len(data["y"]) == 116
    sc = FormulaScorer(data, hh=6)
    assert sc.shortlist(cf) == ["hosp_patients", "positive_rate"], "the shortlist of exoScreenTest2"
    assert cf.shortlist(data, Model.trSize(116), 2, 6) == sc.shortlist(cf)          # the generic one, as datasetsTest calls it
    assert len(cf.blindNames) == 9 and cf.blindNames["y"] == "y"
    ca = Datasets.COVID_ALL
    short = FormulaScorer(ca.load(), hh=6).shortlist(ca)
    assert len(short) == 2 and set(short) <= set(ca.exo)
    assert sc.shortlist(Datasets.COVID2) == ["icu", "pRate"] and sc.shortlist(Datasets.GEF14) == ["temp", "tmax"]
    assert Datasets.get("covid_all") is Datasets.COVID_ALL and Datasets.get("covid_file") is cf
    assert Datasets.COVID2.copy(name="x").name == "x" and Datasets.COVID2.expectedSigns == {"icu": 1, "pRate": 1}


def test_covid_file_shortlist():
    """The screening of the file-based Covid configuration (ScalaTion's datasetsTest): hosp_patients, positive_rate."""
    cf = Datasets.COVID_FILE
    data = cf.load()
    xe = {v: data[v] for v in cf.exo}
    ranks, kept = ExoScreen.shortlist(data["y"], xe, 6, 92, 2)
    assert kept == ["hosp_patients", "positive_rate"] and ranks[0].dep >= ranks[1].dep
    assert ExoScreen.monotone(np.arange(10.0)) == 1.0 and ExoScreen.monotone(np.array([1.0, 3.0, 2.0, 4.0])) == pytest.approx(2 / 3)


def test_etth1_auto_picks_exok_loads():
    cfg = Datasets.ETTH1_1Y
    sc = FormulaScorer(cfg.load(), hh=6, hp={"nneg": False})
    short = sc.shortlist(cfg)
    assert len(short) == 2 and set(short) <= set(cfg.exo) and sc.trSize == 7007 and sc.teSize == 1753
