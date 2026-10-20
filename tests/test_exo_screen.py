"""Tests of ExoScreen (the Python counterpart of ScalaTion's exoScreenTest, exoScreenTest2 and
exoScreenTest3 on the Covid-19 file) and of the statistics it rests on (ccorr, corr, Model.screen)."""
import numpy as np
import pandas as pd
import pytest

from Util.correlogram import ccorr, corr
from modeling.forecasting.args import Args
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.datasets import Example_Covid, loadH
from modeling.forecasting.exo_screen import ExoPool, ExoRank, ExoScreen, corr_matrix, model_screen
from modeling.model import Model

POOL = ["new_cases", "reproduction_rate", "icu_patients", "hosp_patients", "new_tests",
        "positive_rate", "tests_per_case", "new_vaccinations", "excess_mortality"]
CUMULATIVE = {"people_vaccinated", "people_fully_vaccinated", "total_boosters",
              "excess_mortality_cumulative_absolute", "excess_mortality_cumulative_per_million"}
EXCLUDE = {"excess_mortality", "new_cases"}


def covid_frame() -> pd.DataFrame:
    """The 116 rows every Covid-19 test uses, the date column skipped (MatrixD.loadH (skipCol = 1))."""
    mat, header = loadH(Example_Covid.fileName, 1)
    return pd.DataFrame(mat[:116], columns=header)


def covid_pool():
    df = covid_frame()
    y = df["new_deaths"].to_numpy(dtype=float)
    return y, {c: df[c].to_numpy(dtype=float) for c in POOL}, Model.trSize(len(y))


def test_ccorr_and_corr_follow_the_scalation_definitions():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(40), rng.standard_normal(40)
    n, k = len(x), 3
    expect = np.sum((x[:n - k] - x.mean()) * (y[k:] - y.mean())) / (n - 1) / (x.std(ddof=1) * y.std(ddof=1))
    assert ccorr(x, y, k) == pytest.approx(expect)
    assert ccorr(x, x, 0) == pytest.approx(1.0)                          # lag 0 of a vector with itself
    assert corr(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    c = np.full(10, 2.0)
    assert corr(c, c) == 1.0 and corr(c, x[:10]) == 0.0 and np.isnan(ccorr(c, x[:10], 1))
    assert np.allclose(corr_matrix(np.column_stack([x, y])), np.corrcoef(x, y))
    with pytest.raises(ValueError):
        ccorr(x, y, 40)


def test_model_screen_keeps_dependent_non_collinear_columns():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(200)
    x1 = y + 0.3 * rng.standard_normal(200)                              # strongly dependent
    x2 = x1 + 1e-3 * rng.standard_normal(200)                            # collinear with x1
    x3 = rng.standard_normal(200)                                        # independent of y
    x4 = -0.5 * y + rng.standard_normal(200)                             # dependent, negative
    xy = np.column_stack([x1, x2, x3, x4, y])
    reduced, sel = model_screen(xy, 0.1, 0.95)
    assert len(sel) == 2 and sel[0] in (0, 1) and sel[1] == 3 and reduced.shape == (200, 2)   # one of the collinear pair, x4
    _, sel2 = model_screen(xy, 0.1, 1.0)                                 # no collinearity cut-off
    assert sel2.tolist() == [0, 1, 3]
    _, sel3 = model_screen(xy, 0.05, 1.0)                                # x3 (|corr| 0.06) passes thr1 = 0.05
    assert sel3.tolist() == [0, 1, 2, 3]


def test_exo_rank_prints_as_the_scala_case_class():
    r = ExoRank("positive_rate", 3, 0.7123, 0.4971)
    assert r.dep == pytest.approx(0.7123) and r.leads
    assert str(r) == "positive_rate                            lag  3  ccf   0.712  corr0   0.497  leads"
    assert not ExoRank("new_tests", 1, 0.556, 0.561).leads and "leads" not in str(ExoRank("new_tests", 1, 0.556, 0.561))


def test_ranking_and_shortlist_of_exo_screen_test():
    """exoScreenTest: the data alone ranks excess mortality and new cases first, new cases lead by two
    weeks, tests per case and vaccinations correlate negatively, and the shortlist of two is the proxy
    and the cases; with both excluded, hospital patients and the positivity rate (BEST2_EXO with hosp
    for icu, which is collinear), tests per case joining at k = 3."""
    y, xe, tr = covid_pool()
    assert tr == 92
    ranks, short = ExoScreen.shortlist(y, xe, 6, tr, 2)
    assert [r.name for r in ranks[:5]] == ["excess_mortality", "new_cases", "hosp_patients", "positive_rate", "icu_patients"]
    assert all(ranks[i].dep >= ranks[i + 1].dep for i in range(len(ranks) - 1))
    assert all(1 <= r.bestLag <= 6 for r in ranks)
    by = {r.name: r for r in ranks}
    assert by["new_cases"].bestLag == 2 and by["excess_mortality"].bestLag == 1
    assert by["excess_mortality"].ccf == pytest.approx(0.863, abs=5e-4)
    assert by["tests_per_case"].ccf < 0 and by["new_vaccinations"].ccf < 0
    assert short == ["excess_mortality", "new_cases"]

    xe2 = {k: v for k, v in xe.items() if k not in EXCLUDE}
    _, short2 = ExoScreen.shortlist(y, xe2, 6, tr, 2)
    _, short3 = ExoScreen.shortlist(y, xe2, 6, tr, 3)
    assert short2 == ["hosp_patients", "positive_rate"]
    assert short3 == ["hosp_patients", "positive_rate", "tests_per_case"]


def test_lead_dependency_against_the_lag_0_screen():
    """The same Model.screen with its default lag-0 dependency would take new tests before the positivity
    rate, which leads deaths by three weeks (2nd by lead, only 4th alongside)."""
    y, xe, tr = covid_pool()
    xe2 = {k: v for k, v in xe.items() if k not in EXCLUDE}
    ranks2 = ExoScreen.rank(y, xe2, 6, tr)
    xy = ExoScreen.xyOf(ranks2, y, xe2, tr)
    assert xy.shape == (tr, len(xe2) + 1) and np.array_equal(xy[:, -1], y[:tr])
    dep = ExoScreen.dependency(ranks2, xy)
    assert np.allclose(dep[:-1, -1], [r.ccf for r in ranks2]) and np.allclose(dep[-1, :-1], [r.ccf for r in ranks2])
    assert dep[0, 0] == 1.0
    _, sel = model_screen(xy, 0.05, 0.95)                                 # dep = corr (lag 0 only)
    lag0 = [ranks2[i].name for i in sorted(sel.tolist(), key=lambda i: -abs(ranks2[i].corr0))[:3]]
    assert lag0 == ["hosp_patients", "new_tests", "positive_rate"]
    by_lead = [r.name for r in ranks2]
    by_lag0 = [r.name for r in sorted(ranks2, key=lambda r: -abs(r.corr0))]
    assert by_lead.index("positive_rate") == 1 and by_lag0.index("positive_rate") == 3
    with pytest.raises(ValueError):
        ExoScreen.dependency(ranks2, xy[:, :-2])


def test_pool_drops_exactly_the_five_running_totals():
    """exoScreenTest2: from the file with its header, `pool` keeps the ten other columns in column order."""
    df = covid_frame()
    tr = Model.trSize(len(df))
    p = ExoScreen.pool(df, None, "new_deaths", tr)
    assert isinstance(p, ExoPool) and "new_deaths" not in p.xe
    assert {nm for nm, _ in p.dropped} == CUMULATIVE
    assert all(why.startswith("cumulative") for _, why in p.dropped)
    assert len(p.xe) == 10 and list(p.xe) == [c for c in df.columns if c != "new_deaths" and c not in CUMULATIVE]
    assert np.array_equal(p.y, Example_Covid.context(["icu_patients"])["y"])
    ranks, short = ExoScreen.shortlist(p.y, p.xe, 6, tr, 2)
    assert len(ranks) == 10 and ranks[0].name == "excess_mortality" and short == ["excess_mortality", "new_cases"]

    p2 = ExoScreen.pool(df.to_numpy(dtype=float), list(df.columns), "new_deaths", tr, exclude=EXCLUDE)
    assert sum(1 for nm, why in p2.dropped if nm in EXCLUDE and why.startswith("excluded")) == 2
    assert ExoScreen.shortlist(p2.y, p2.xe, 6, tr, 2)[1] == ["hosp_patients", "positive_rate"]
    assert ExoScreen.shortlist(p2.y, p2.xe, 6, tr, 3)[1] == ["hosp_patients", "positive_rate", "tests_per_case"]


def test_pool_argument_checks_and_constant_columns():
    df = covid_frame()
    with pytest.raises(ValueError):
        ExoScreen.pool(df, None, "deaths", 92)
    with pytest.raises(ValueError):
        ExoScreen.pool(df.to_numpy(dtype=float), ["a", "b"], "a", 92)
    with pytest.raises(ValueError):
        ExoScreen.pool(df, None, "new_deaths", 500)
    df2 = df[["new_deaths", "icu_patients"]].copy()
    df2["flat"] = 1.0
    df2["text"] = "a"
    p = ExoScreen.pool(df2, None, "new_deaths", 92)
    assert list(p.xe) == ["icu_patients"]
    assert dict(p.dropped) == {"flat": "constant on the training rows", "text": "non-finite values (not a numeric column?)"}
    assert ExoScreen.monotone(np.arange(5.0)) == 1.0 and ExoScreen.monotone(np.array([1.0, 3.0, 2.0, 4.0])) == pytest.approx(2 / 3)


def test_screen_in_one_call_prints_and_feeds_arx_d(capsys):
    """exoScreenTest3: file -> screen -> forecaster; with k = 3 the chosen columns come back in the file's
    order, xe holds the full columns, and the ARX_D on the k = 2 choice trains and tests as in aRX_DTest4."""
    df = covid_frame()
    xe3, y3, names3 = ExoScreen.screen(df, None, "new_deaths", k=3, exclude=EXCLUDE)
    out = capsys.readouterr().out
    assert names3 == ["hosp_patients", "positive_rate", "tests_per_case"]
    assert xe3.shape == (116, 3) and np.array_equal(xe3[:, 1], df["positive_rate"].to_numpy(dtype=float))
    assert "| ExoScreen.screen: response new_deaths, 8 candidates, first 92 rows, leads 1..6 |" in out
    assert "dropped new_cases                                excluded (domain knowledge)" in out
    assert "chosen (k = 3, in the file's order): hosp_patients, positive_rate, tests_per_case" in out
    assert "hosp_patients                            lag  1  ccf   0.736  corr0   0.708  leads" in out

    xe, y, names = ExoScreen.screen(df.to_numpy(dtype=float), list(df.columns), "new_deaths", k=2,
                                    exclude=EXCLUDE, show=False)
    assert capsys.readouterr().out == ""
    assert names == ["hosp_patients", "positive_rate"] and xe.shape == (116, 2) and y.shape == (116,)
    mod = ARX_D(y, 6, xe, Args(p=6, px=6, spec=0))
    assert mod.modelName == "ARX_D_6_6_2_L0.1_A0.5"                    # rev2.3: the elastic net by default
    ft = mod.tnT_Test(rc=20)
    assert ft.shape == (6, 26) and np.all(ft[:, 9] < 100)

    xe0, _, names0 = ExoScreen.screen(df, None, "new_deaths", k=0, show=False)
    assert names0 == [] and xe0.shape == (116, 0)
