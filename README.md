# Forecaster

The Python counterpart of the forecasting package of [ScalaTion 2.0](https://github.com/scalation/scalation)
(`scalation.modeling.forecasting`): the same model names, factory methods, hyper-parameters,
formula grammar, feature-selection techniques and printed reports, written for numpy / pandas /
scipy / statsmodels.  It covers the baseline models (RandomWalk, AR, ARIMA), the ARX family
(`ARX`, `ARX_D`), symbolic-regression terms (`ARX_SR`, `ARX_SR_D`), formula-driven models
(`ARX_Formula`, `NARX_Formula`, `NARX_SR_D`), feature selection (Forward, Backward, Stepwise, Beam,
CEM) and **LLM-NARX-SR**: the search for NARX formulas with free nonlinear parameters, fitted by
variable projection and scored for multi-horizon forecasting, proposed by an LLM, a genetic or a
random proposer and selected over a term library.

## Getting started

```bash
git clone https://github.com/Youseffekri/Forecaster.git
cd Forecaster
pip install -e .            # numpy, pandas, scipy, scikit-learn, statsmodels
pip install -e .[neural]    # + torch, for the neural models (MHAttn_Regressor)
pip install -e .[test]      # + pytest
pytest                      # 178 tests, ~25 s, all offline
python Examples/Example_Covid19.py            # rw | ar | arima | arx | arx_d | arx_sr | arx_sr_d | formula | narx | mhattn | all
python Examples/Example_ILI.py                # rw | ar | arima | arx_d | arx_sr_d | all
python Examples/Example_ETTh1.py              # rw | ar | arx_d | arx_sr_d | narx | all
python Examples/Example_GEFCom.py             # rw | ar | arx_d | arx_sr_d | narx | all
python Examples/Example_FeatureSelection.py   # 5 | 6 | 7 | 8 | 9 | search | cem | all   (aRX_DTest5-9, subsetSearchTest, cEM_SelectorTest)
python Examples/Example_LLM_NARX_SR.py        # covid | optimizers | gef14 | etth1 | baseline | genetic | random | mockllm | grid | all
```

With no argument each example runs a quick default.  `pip install -e .` makes `modeling` and `Util`
importable from anywhere; the examples also run from the repository root without it.

## The ScalaTion conventions

Every model is used the way its ScalaTion counterpart is, and prints the same blocks:

```python
from modeling.forecasting.arx_d import ARX_D
mod = ARX_D({"p": 6, "q": 6, "spec": 1}, y, hh=6, xe=xe)      # or ARX_D.rescale(...), ARX_D.fromTerms(formula, dataset, hh)
mod.inSample_Test()        # banner, "b = VectorD(...), fn = Array(cons, yl6, ...)", REPORT, "fitMap QoF =" table over h = 1..hh
mod.tnT_Test(rc=5)         # train, rolling validation on the testing rows (retrain every rc rows), QoF table on the testing rows
ftMat = mod.diagnoseAll()  # (hh, 26) QoF matrix: rSq, rSqBar, sst, sse, sde, rse, mse0, rmse, mae, smape, bmae, m, dfr, df, fStat, aic, bic, mape, mase, smapeC, ...
mod.Yf                     # the forecast matrix in ORIGINAL units: column 0 = y, Yf[t, h] = h-steps ahead forecast of y[t + h - 1]
mod.parameter, mod.fname, mod.modelName, mod.hparameter          # the slopes b, the column names, "ARX_D_6_6_2", the hyper-parameters
```

* Hyper-parameters (`args` dict): `p` (lags of y), `q` (lags of each exogenous variable, ScalaTion's `px`),
  `spec` (number of trend columns, ScalaTion's convention: **0 implicit constant, 1 linear t/m, 2 + quadratic,
  3 + cubic**; default 1; the columns are named as `MakeMatrix4TS.trend`: "cons", "lin", "quad"), `cross`
  (ENDO-EXO cross terms of `ARX_SR*`, ScalaTion's `crx`), `skip` (rows left out of the In-Sample diagnosis,
  default p), `lambda` / `scaled` (the centered ridge of `method="ridge"`).
* Quality of Fit (`modeling/fit.py`, ScalaTion's `Fit`): the 26 measures in `QoF` order with ScalaTion's
  formulas (R²-bar through the DF ratio, AIC/BIC with kParam = dfr + 2, sMAPE-IC), the first `skip` rows
  dropped (`Diagnoser`), the degrees of freedom reset per horizon.  The Python runs reproduce the ScalaTion
  logs digit for digit (`ARX_D_1_0_0` on Covid-19: TnT sMAPE 18.5489, 31.4088, 45.3246, 57.0302, 64.1919,
  70.4402; the CEM selection of `aRX_DTest8`; `subsetSearchTest`).
* Train/test split: `Model.trSize` / `teSize` (te = floor(0.2 m + 1): 92/24 on Covid-19, 1679/420 on GEFCom2014,
  966/242 on ILI, 13935/3485 on ETTh1).
* Printing helpers (`Util/tools.py`): `banner`, `vector_str` ("VectorD(0.946883,\t...)"), `matrix_str`,
  `array_str`, `set_str` ("LinkedHashSet(0, 1, 3)"), `report`.
* `docs/CONVENTIONS.md` maps every ScalaTion file to its Python file and states the rules.

## Models (`modeling/forecasting/`)

| Python | ScalaTion | what |
|---|---|---|
| `RandomWalk(y, hh)` | RandomWalk.scala | the last-value baseline (no training) |
| `AR(args, y, hh, method)` | AR.scala | AR(p) by Yule-Walker ("adjusted", "mle") or OLS; `AR_YW` is an alias |
| `ARIMA(args, y, hh)` | ARMA.scala / ARIMA_diff.scala | ARIMA(p, d, q) backed by statsmodels (exact MLE) under the ScalaTion interface |
| `ARX`, `ARX_D` | ARX.scala, ARX_D.scala | lags of y and of the exogenous variables; recursive (forge / hide) and direct; methods `sm_ols`, `sk_lr`, `ridge` |
| `ARX_SR`, `ARX_SR_D` | ARX_SR.scala, ARX_SR_D.scala | + transformations of the lags: `fEndo=[PowForm(), RootForm()]`, `fExo=[[LogForm()], []]` (`Util/transform.py` = `mathstat/Transform.scala`), cross terms |
| `ARX_Formula` + `FormulaParser` | ARX_Formula.scala, FormulaParser.scala | the linear formula DSL and grammar: `y ~= y~(3,1) + log1p(icu~1) + y~1 * icu~1 + wave(2) TFORM: Log1p`; `fit(data, hh)` -> `ARX.fromTerms`, `fit_D` -> `ARX_D.fromTerms` |
| `NARX_Formula` + `NARX_FormulaParser` | nonlinear/NARX_Formula.scala, NARX_FormulaParser.scala | + free parameters `Par` with roles (Shift, Phase, Rate, Power, Freq, Decay, Thresh, Width, Half, Cap): `log(? + icu~1)`, `pow(y~(3,1), ?)`, `hinge(temp~1, ?)`, `trans(y~1, ?, ?)`, `ema(x~1, ?)`, `dq(x~1, ?)`, `wave(?)`; `fitNL_D(data, hh)` |
| `NARX_SR_D.fromTerms(formula, dataset, hh, lam, scaled, optimizer)` | nonlinear/NARX_SR_D.scala | the direct model of a formula: coefficients by centered ridge per horizon, free parameters by `varpro` (variable projection, default), `lbfgsb` or `joint`; each fitted parameter is a degree of freedom (`nFree`) |
| `ExoScreen.screen(data, header, resp, k, maxLag)` | ExoScreen.scala | pool / rank (lead cross-correlation) / shortlist of candidate exogenous variables |
| `MHAttn_Regressor` (optional torch) | -- | multi-head attention for feature selection |

### Feature selection (`modeling/feature_selection.py`, `subset_search.py`, `cem_selector.py`; the scenarios inside `Forecaster_Reg` / `Forecaster_D`)

ScalaTion's `FeatureSelection`, `SubsetSearch` and `CEM_Selector` (with a port of ScalaTion's MRG31k3p `Random`
so a CEM run with the same stream draws the same samples).  As in ScalaTion, the selection methods live in the two
forecaster bases: `Forecaster_Reg` (the regression-based recursive models `ARX`, `ARX_SR`: `forwardSelAll`,
`backwardElimAll`, `stepwiseSelAll`, `beamSelAll`, `cemSelAll`, `correctQoF`, `getModel`) and `Forecaster_D` (the direct
models: the three scenarios below).

```python
from modeling.feature_selection import SelectionTech
colsArr, rSqArr = mod.selectFeatures(SelectionTech.CEM)            # one column set for all horizons, J = mean FPE-form sMAPE
colsArr, rSqArr = mod.selectFeatures(SelectionTech.Stepwise, h=1)  # the h = 1 model decides
mod.reportSelection(colsArr, rSqArr, inSample=False)               # prints the selection and TnT-tests the selected model (getModel(cols))
```

`selectFeaturesAtH`, `selectFeaturesPerH`, `getModel(cols)` / `getModel_h(cols, h)`; a recursive `ARX` selects its own
features the same way (`arx.selectFeatures(SelectionTech.Stepwise)`); `FeatureSelection.fullset_FS = True` selects on all rows.

### Transformations (`Util/transform.py`, ScalaTion's `mathstat/Transform.scala`)

`PowForm (x+s)^p`, `RootForm`, `RecipForm`, `PowRForm` (sign-safe), `DQuadForm`, `DIQuadForm`, `BoxcoxForm`,
`LogForm log(a x + ss)`, `Log1pForm`, `ExpForm`, `NExpForm`, `SinForm`, `CosForm` with ScalaTion's default
parameters (`HP`); `NormForm`, `MinMaxForm`, `CenterForm` fitted on the training slice; `TransformT` names the
types (`TransformT.Log1p.form(x_train)`) and is what `TFORM: Log1p` in a formula means.  The `rescale` factories
take sklearn scalers (`MinMaxScaler` to (0, 4) by default; `StandardScaler` turns the non-negativity off, like NormForm).

## LLM-NARX-SR (`modeling/forecasting/nonlinear/llm_narx_sr/`, ScalaTion's `llm_narx_sr` package)

The structure of a forecasting model is written as a formula over a named-column dataset
(`{"y": ..., "icu": ..., "pRate": ...}`), parsed, fitted and scored by one engine, and searched for:

```python
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import Datasets
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D

data = Datasets.GEF14.load()                                   # daily peak load, temp, tmax
sc   = FormulaScorer(data, hh=6, hp={"maxLag": 7, "skip": 7})  # the engine: training / validation rows, the score J
cand = sc.score("y ~= y~(7,1) + hinge(temp~1, ?)")             # parse -> fit -> score (the threshold lands at 65.8 F)
model, qof = sc.tnt(cand.formula.fresh())                      # rolling validation on the testing rows

m = NARX_SR_D.fromTerms("y ~= y~(7,1) + hinge(temp~1, ?)", data, hh=6, lam=1.0, scaled=True, optimizer="lbfgsb", maxLag=7)
m.inSample_Test(); m.tnT_Test()                                # the model alone, another optimizer

from modeling.forecasting.nonlinear.llm_narx_sr.llm_narx_sr import LLM_NARX_SR
runs = LLM_NARX_SR.runGrid("Covid-19", Datasets.COVID2, LLM_NARX_SR.zeroLLMArms(), streams=[0, 1])   # baseline, tuned-arx, sel-arx, random-walk, stat-cem, genetic, random
```

| level | files | what |
|---|---|---|
| L1 | `formula_scorer.py`, `eda4ts.py` | `FormulaScorer` (all ScalaTion hp entries; parse -> fit -> `Candidate`; J_h = sMAPE_h(validation)(1 + 2p/n); rejection rules: too many free parameters, a parameter at a bound, an exponent near 1, a collinear free-parameter column; a fit cache; `tnt`); `EDA4TS` (ACF/PACF, periodogram, CCF, KPSS, the prompt block, the residual report) |
| L2 | `datasets.py`, `term_library.py` | `Example_Covid` / `Example_ILI` / `Example_ETTh1` / `Example_GEFCom`, `DatasetConfig` (`ExoMode.Given` / `Auto` with the screening), `Datasets` (COVID2, COVID, COVID_ALL, ILI, ETTH1, ETTH1_1Y, GEF14, `covidFile()`, `fromFile`); `TermLibrary` (grammar or union libraries, `selectBy` CEM / Forward / Backward / Stepwise / Beam on J) |
| L3 | `search_arm.py` | `ArmSpec` (llm-loop, llm-cem, stat-cem, genetic, random and the ablations), `ExpRun` / `Trace` records, `SearchArm` |
| L4 | `search_loop.py`, `genetic_proposer.py`, `llm_proposer.py`, `explainer.py` | the propose -> [trim] -> score -> feed back loop with adaptive stop; `RandomTerms` / `GeneticProposer`; `LLMProposer` (blind prompt: generic sentence, grammar, EDA profile, scored population, rejects); `Explainer` (claims checked against the fit) |
| L5 | `baseline_arm.py`, `pipeline_arm.py`, `plausibility_checker.py`, `significance.py` | the anchors (baseline, tuned-arx, sel-arx, random-walk), every searching arm, sign/stability plausibility, paired Wilcoxon / fold tables / selection check |
| L6 | `llm_narx_sr.py` | the harness: `runGrid`, `runFolds`, `report`, `showBest`, `paretoFront`, CSV rows in `llm_narx_sr_results/`; env vars `LLM_NARX_SR_DATASET`, `LLM_NARX_SR_FT_ENDPOINT` / `LLM_NARX_SR_FT_MODEL`, `OPENAI_API_KEY`; cache `llm_cache/` |

`modeling/llm_client.py`: `OpenAIClient`, `AnthropicClient`, `LocalClient` (Ollama / vLLM), `CachedClient`,
`CountingClient`, `MockLLMClient` (scripted replies; the tests and the `mockllm` example use it, nothing calls
the network in the tests).

### Formula grammar

```
formula  := ID "~=" element { "+" element } [ "TFORM:" ID ]      # TFORM: Log1p | Norm | MinMax | ...
element  := atom { "*" atom }                                     # at most two factors (a lag range distributes)
atom     := x~k | x~(k1,k2)                                       # a lag, a lag range (one term per lag)
          | log|log1p|sqrt|sq|dq|exp|expm1|sin|cos|tanh ( atom )  # a function of a term
          | pow(atom, p) | powr(atom, p)                          # fixed exponent in [0.2, 3], sign-safe power
          | wave(k)                                               # k periodogram sin/cos pairs
          -- free parameters (NARX_FormulaParser):
          | FN( ? + atom ) | FN( ? * atom )                       # a free shift / rate inside a function
          | pow(atom, ?) | sat(atom, ?) | hinge(atom, ?) | trans(atom, ?, ?) | ema(atom, ?) | dq(atom, ?) | wave(?)
?        := a free parameter; ?w a free parameter starting at w; a number a constant
```

### Datasets (`data/`)

| config | file | rows | response | drivers |
|---|---|---|---|---|
| `Datasets.COVID2`, `COVID`, `COVID_ALL` | `covid_19_weekly.csv` | 116 weeks | new deaths | icu, pRate / hosp, icu / the screened pool |
| `Datasets.ILI` | `ILINetNov25.csv` (CDC ILINet, 2002-10-06 .. 2025-11-23) | 1,208 weeks | ILITOTAL | wili (%WEIGHTED ILI), ot (TOTAL PATIENTS) |
| `Datasets.ETTH1`, `ETTH1_1Y` | `ETTh1.csv` (Zhou et al. 2021) | 17,420 / 8,760 hours | OT (oil temperature, goes negative: `nneg=False`) | two of hufl, hull, mufl, mull, lufl, lull picked by the screening |
| `Datasets.GEF14` | `gefcom2014_load_daily.csv` (Hong et al. 2016; built by `tools/build_gefcom_daily.py`) | 2,099 days | peak load | temp, tmax |

### Results reproduced

On GEFCom2014 the lag baseline `y ~= y~(7,1) + temp~(7,1) + tmax~(7,1)` gives a rolling TnT sMAPE of 12.783
(mean over h = 1..6; ScalaTion rev3.3: 12.783) and the free hinge threshold of `y ~= y~(7,1) + hinge(temp~1, ?)`
is fitted at 65.8 F by all three optimizers (the cooling point of the V-shaped load-temperature curve: mean peak
per 10 F band 272, 226, 185, 148, 136, 214, 273).  On Covid-19 the anchors give TnT sMAPE 43.48 (baseline),
30.72 (tuned-arx, `y ~= y~1 + icu~(3,1) + pRate~(3,1)`), 45.71 (random-walk), as ScalaTion; a shift inside a
log and a decay in an ema run to their bounds and are rejected.

## Repository layout

```
modeling/fit.py                      QoF (ScalaTion Fit / FitM)            modeling/regression.py   CenteredRidge
modeling/feature_selection.py        SelectionTech, BestStep, FeatureSelection       modeling/subset_search.py   SubsetSearch
modeling/cem_selector.py             CEM_Selector, Random (MRG31k3p)        modeling/llm_client.py   LLM clients
modeling/forecasting/                diagnoser, forecaster, forecaster_reg, forecaster_d, random_walk, ar (ar_yw alias), arima,
                                     arx, arx_d, arx_sr, arx_sr_d, arx_formula, formula_parser, exo_screen
modeling/forecasting/nonlinear/      narx_formula, narx_formula_parser, narx_sr_d
modeling/forecasting/nonlinear/llm_narx_sr/   formula_scorer, eda4ts, datasets, term_library, search_arm, search_loop, genetic_proposer,
                                     llm_proposer, explainer, baseline_arm, pipeline_arm, plausibility_checker, significance, llm_narx_sr
modeling/forecasting/neuralforecasting/  mhattn_regressor
modeling/optimization/               varpro (VarPro, L-BFGS-B, joint)
Util/                                build_matrix (MakeMatrix4TS), tools (banner, VectorD/Array printing, report), transform (Transform.scala), data_loader
data/                                covid_19_weekly.csv, ILINetNov25.csv, ETTh1.csv, gefcom2014_load_daily.csv
Examples/                            Example_Covid19, Example_ILI, Example_ETTh1, Example_GEFCom, Example_FeatureSelection, Example_LLM_NARX_SR
tests/                               178 pytest tests (fit, parsers, formulas, models, feature selection, scorer, datasets, search layer)
docs/                                CONVENTIONS.md (the ScalaTion -> Python map and rules), SCALA_SOURCES.md
```

## License

MIT License

## Author

Yousef Fekri Dabanloo
March 2025 -- October 2026
