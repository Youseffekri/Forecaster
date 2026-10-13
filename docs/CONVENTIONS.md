# Porting conventions: ScalaTion `scalation.modeling.forecasting` -> Python `modeling/forecasting`

The Python package mirrors ScalaTion 2.0 file by file.  Keep these rules when adding a file.

## Files and names

| ScalaTion | Python |
|---|---|
| `modeling/Fit.scala`, `FitM.scala` (QoF enum, diagnose, fitMap, showFitMap, summary) | `modeling/fit.py` (`QoF`, `Fit`, `diagnose`, `fitMap`, `showFitMap`, `Fit.summary`, `vif` = `Predictor.vif`) |
| `HyperParameter.scala` | `modeling/hyper_parameter.py` (`HyperParameter`: a dict of values that keeps the defaults, printed as ScalaTion) |
| `modeling/CenteredRidge*.scala` | `modeling/centered_ridge.py` (`CenteredRidge`: `hp` (lambda 0.1), `fit`, `predict`, `parameter` (fitted space), `slopes`, `intercept`, `params`, `ridge`) |
| `modeling/FeatureSelection.scala`, `SubsetSearch.scala`, `CEM_Selector.scala` | `modeling/feature_selection.py`, `subset_search.py`, `cem_selector.py` |
| `forecasting/Diagnoser.scala` | `forecasting/diagnoser.py` (`Diagnoser`: `getSkip`, `setSkip`, skip-aware `diagnose`) |
| `forecasting/Forecaster.scala` (+ `ForecastMatrix.scala`), `AR.hp` | `forecasting/forecaster.py` (`Forecaster`, `hp` = `AR.hp`, `trSize`, `teSize`, `teRng`, `makeForecastMatrix`, `forecastAt`) |
| `forecasting/Forecaster_D.scala` (direct base: `reg`, `train`, `summary`, feature-selection scenarios) | `forecasting/forecaster_d.py` (`Forecaster_D`: `_make_model`, `train`, `slopes`, `bb`, `summary` / `summary (h)`; `smapeFPE`, `aggFPE`) |
| `forecasting/AR.scala`, `RandomWalk.scala`, `ARX.scala`, `ARX_D.scala`, `ARX_SR.scala`, `ARX_SR_D.scala` | `ar.py`, `random_walk.py`, `arx.py`, `arx_d.py`, `arx_sr.py`, `arx_sr_d.py` |
| `forecasting/ARMA.scala`, `ARIMA_diff.scala` (ScalaTion's ARIMA family) | `arima.py` (`ARIMA`, statsmodels-backed under the same interface) |
| `forecasting/Forecaster_Reg.scala` (regression-based recursive base: `reg`, `train`, `summary`, feature selection) | `forecasting/forecaster_reg.py` (`Forecaster_Reg`; `ARX` and `ARX_SR` extend it) |
| `forecasting/ARX_Formula.scala`, `FormulaParser.scala`, `ExoScreen.scala` | `arx_formula.py`, `formula_parser.py`, `exo_screen.py` |
| `forecasting/nonlinear/NARX_Formula.scala`, `NARX_FormulaParser.scala`, `NARX_SR_D.scala` (`OptimizerNL`, `OPTIMIZER_NL`, the class (an `ARX_D`), `apply`, `rescale`, `make`, `buildMatrix`, `optimize`, `fromTerms`, `fitNL`, `joint`, `ridge`, `initializeW`, `tunable`, `getTransforms`, `formNames`) | `nonlinear/narx_formula.py`, `narx_formula_parser.py`, `narx_sr_d.py` (the enum and the module default live with `NARX_SR_D`, as in Scala) |
| `optimization/nonlinear_ls/VarPro.scala` | `modeling/optimization/varpro.py` (`VarPro`: `system`, `coefficients`, `residual`, `objective`, `optimize`, `stepCounts`) |
| `forecasting/nonlinear/llm_narx_sr/*.scala` (14 files) | `nonlinear/llm_narx_sr/*.py`, same base names in snake_case (`formula_scorer.py`, `eda4ts.py`, `datasets.py`, `term_library.py`, `search_arm.py`, `search_loop.py`, `genetic_proposer.py`, `llm_proposer.py`, `explainer.py`, `baseline_arm.py`, `pipeline_arm.py`, `plausibility_checker.py`, `significance.py`, `llm_narx_sr.py`) |
| `mathstat/Transform.scala` | `Util/transform.py` (`Transform` with `setLU`, `PowForm` .., `NormForm`, `MinMaxForm` (centered [-2, 2]), `TransformT` with `form` and `wlu`) |
| `forecasting/MakeMatrix4TS.scala` | `Util/make_matrix4ts.py` (`makeMatrix4L`, `makeMatrix4T`, `makeMatrix4Y`, `backcast`, `backfill`, `trend_names`, `centered`) |
| `LLMClient.scala` | `modeling/llm_client.py` |

File names are the Scala file names in snake_case; a file is named after the class it holds
(`centered_ridge.py` holds `CenteredRidge`, never a generic name such as `regression.py`).

Imports go at the top of a module.  The exceptions are the two genuine import cycles (`NARX_Formula.fitNL_D`
builds a `NARX_SR_D`, which builds on `NARX_Formula`; `FormulaScorer.notes` uses the `Explainer`, which
imports the scorer) and the optional dependencies (torch, matplotlib); `tests/test_base_api.py` checks it.

Class names are ScalaTion's (`ARX_SR_D`, `NARX_Formula`, `FormulaScorer`, `CEM_Selector`, `OptimizerNL`).
Method, property and attribute names that users touch keep ScalaTion's camelCase (`inSample_Test`,
`tnT_Test`, `rollValidate`, `selectFeatures`, `formulaStr`, `proposalStr`, `nTerms`, `nFree`, `maxLag`,
`lagHP`, `fromTerms`, `parseFormula`, `tryParseFormula`, `grammarLines`, `topFreqs`, `loadData`,
`stepCounts`, `qofTrain`, `trSize`, `fitRng`, ...).  There are NO snake_case aliases of them: one name
per thing, the ScalaTion one.  Python-only helpers that have no Scala counterpart (`fit_rows`,
`predict_rows`, `vector_str`, `tr_size`) use snake_case, so they are recognizable as such.  Model names
printed in reports are ScalaTion's: `ARX_D_6_6_2` (p, px, n_exo), `AR_6`, `RandomWalk`, `NARX_SR_D(p, px, n_exo)`.

## Hyper-parameters (`hparam`)

The constructor argument is `hparam` (ScalaTion's `hparam: HyperParameter`).  Every model completes it from
`hp` (ScalaTion's `AR.hp`, also `AR.hp`): `p` (lags of y, default 1), `d`, `q`, `P`, `D`, `Q`, `s` (ARIMA),
`px` (lags of each exogenous variable, default 1), `spec` (number of trend columns: 0 implicit constant,
1 linear t/m, 2 + quadratic, 3 + cubic; default 1), `crx` (1 => ENDO-EXO cross terms; only `ARX_SR*`),
`nneg` (1 => forecasts clipped at 0); the switches are 0 / 1 (a bool is converted).  Python-only entries:
`skip` (rows skipped in the In-Sample diagnosis, default p) and `lambda` (the ridge penalty of method
"ridge", default `CenteredRidge.hp["lambda"]` = 0.1, ScalaTion's `RidgeRegression.hp`).  `scaled` is a
constructor / factory keyword, as in ScalaTion.  `mod.hparam` (= `mod.hparameter`) is a `HyperParameter`;
the REPORT prints the registered (AR.hp) entries as "(value,default)" pairs, exactly as the Scala logs
(RandomWalk has none: "null").  Trend columns are named as ScalaTion's `MakeMatrix4TS.trend`.

## The base API every model follows

```
mod = ARX_D(hparam, y, hh, xe, method, scaled=False)   # or ARX_D.rescale(..., tFormT) / ARX_D.fromTerms(formula, dataset, hh)
mod.train(Y_, X_)                                # direct: Y_ (m, hh) NaN-padded; recursive: train(y_, X_)
mod.forecast(t_start, t_end) -> (rows, hh)       # writes Yf (ORIGINAL units); Yf[t, h] forecasts y[t + h - 1]
mod.forecastAt(h)                                # column h of Yf after forecasting every row
mod.inSample_Test() ; mod.tnT_Test(rc=5)         # ScalaTion's printed output (banner, b = VectorD(...), REPORT, fitMap QoF)
mod.diagnoseAll(tRng=None) -> (hh, N_QoF)        # skip rows dropped (Diagnoser), DF reset per horizon
mod.summary() ; mod.summary(h)                   # ScalaTion's SUMMARY (reflects the last diagnose call)
mod.parameter (b / bb(?, 0): the regression's coefficients) ; mod.slopes ; mod.intercept ; mod.params (intercept first)
mod.fname ; mod.modelName ; mod.hparameter ; mod.tr_size, mod.te_size (Model.trSize / teSize: te = floor(0.2 m + 1))
```

The regression lives in the bases, as ScalaTion's `reg`: `Forecaster._make_model` ("sm_ols", "sk_lr", or
"ridge" = `CenteredRidge(lambda, scaled)`), `Forecaster_Reg.train` (recursive) and `Forecaster_D.train`
(direct, after `fit_rows` drops the NaN-padded rows, ScalaTion's NO_DOUBLE).  The model classes add the
matrix (`buildMatrix (xe_bfill, y, hparam, ...)`, `formNames (n_exo, hparam, ...)` -- ScalaTion's argument
order), the forecasts and the sub-models: `getModel` (ARX, ARX_D -- inherited by ARX_SR_D and NARX_SR_D)
and `getModel_h` (ARX_D; abstract in Forecaster_D).  Python differences kept on purpose: the "sm_ols" /
"sk_lr" choices, one solve per horizon (no MV class), the constructor builds the matrix (ScalaTion's
`apply`), range-based `forecast`, the `_train_target` hook instead of overriding the tests.

`rescale (..., tFormT)` takes a `TransformT` fitted on the training rows (column by column): MinMax (the
default) maps to ScalaTion's centered [-2, 2], Norm standardizes and sets nneg = 0, Log1p keeps zero at
zero; a response transform into a space where y can be negative turns the rectification off (`centered`;
ScalaTion applies it in `ARX_D` only -- here every model, since clipping at 0 in a centered space is wrong).
`AR.rescale` standardizes (NormForm), as ScalaTion's.

QoF vectors are `np.ndarray(26)` in `QoF` order (`QoF.smape.ordinal` indexes them); `Fit.diagnose`
reproduces ScalaTion's formulas (R^2-bar via r_df, AIC/BIC via kParam = dfr + 2, smapeC = smape + 2 kParam / m).

## Nonlinear parameters (`NARX_SR_D`)

`NARX_SR_D` is an `ARX_D` whose columns carry nonlinear parameters: from transform TYPES (`apply` /
`rescale (hparam, y, hh, xe, fEndo = [TransformT.Root], fExo = None => Pow, tFormT = MinMax, lu = (1, 5),
tune = True)`, the parameters' start values and bounds from `TransformT.wlu`, `initializeW`, `tunable`,
`getTransforms`, `optimize`) or from a formula (`fromTerms`).  `OptimizerNL.VarPro` (default), `LBFGS_B`,
`Joint` -- `narx_sr_d.OPTIMIZER_NL` is the module default a test may reassign (ScalaTion's `var
OPTIMIZER_NL`); the factories take the enum, its name or None (the default).  `NARX_SR_D.fitNL` returns
`(a, stepCounts)`; the model keeps `optimizer`, `stepCounts`, `nFree` (ctor argument `n_free`, -1 => all
of w_nl, as Scala's), `parameter_nl` / `w_nl`.  `narx_sr_d.DEBUG = True` prints ScalaTion's debug lines of
`make`, `optimize` and `fitNL`.

## Printing

`Util/tools.py`: `banner`, `fmt` ("%.6f", NaN as "NaN"), `jdouble` (Java's `Double.toString`), `vector_str`
("VectorD(0.946883,\t...)", Java's %g), `matrix_str`, `array_str` ("Array(yl6, ...)"), `set_str`
("LinkedHashSet(0, 1)"); the REPORT block is `Forecaster.report`, the SUMMARY block `Fit.summary`.
Use them so the Python logs read as the Scala ones.

## Formulas

`ARX_Formula` (linear terms) and `NARX_Formula` (free parameters `Par` with roles) print and parse the
same grammar as ScalaTion (`formulaStr`, `key`, `proposalStr`, `FormulaParser.parseFormula`,
`NARX_FormulaParser.parseFormula`); the reject reasons are ScalaTion's strings.  In the Python DSL
`x.lagged(k)` and `x.lags(h, l)` stand for Scala's `x ~ k` and `x ~ (h, l)`.  `fit`, `fit_D`, `fitNL_D`
take `hparam` (e.g., {"lambda": 1.0}, ScalaTion's `RidgeRegression.hp("lambda") = 1.0`).

## Docstrings

numpy style; the module docstring names the ScalaTion file it ports and the differences, if any.
Keep the header block (Author, Date, Version, License, Note: AI assisted code).
