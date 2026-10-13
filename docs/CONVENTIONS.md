# Porting conventions: ScalaTion `scalation.modeling.forecasting` -> Python `modeling/forecasting`

The Python package mirrors ScalaTion 2.0 file by file.  Keep these rules when adding a file.

## Files and names

| ScalaTion | Python |
|---|---|
| `modeling/Fit.scala`, `FitM.scala` (QoF enum, diagnose, fitMap, showFitMap) | `modeling/fit.py` (`QoF`, `Fit`, `diagnose`, `fitMap`, `showFitMap`) |
| `modeling/CenteredRidge*.scala` | `modeling/regression.py` (`CenteredRidge`) |
| `modeling/FeatureSelection.scala`, `SubsetSearch.scala`, `CEM_Selector.scala` | `modeling/feature_selection.py`, `subset_search.py`, `cem_selector.py` |
| `forecasting/Diagnoser.scala` | `forecasting/diagnoser.py` (`Diagnoser`: `getSkip`, `setSkip`, skip-aware `diagnose`) |
| `forecasting/Forecaster.scala` (+ `ForecastMatrix.scala`) | `forecasting/forecaster.py` (`Forecaster`, `trSize`, `teSize`, `teRng`) |
| `forecasting/Forecaster_D.scala` (direct base, with its feature-selection scenarios) | `forecasting/forecaster_d.py` (`Forecaster_D`, `smapeFPE`, `aggFPE`) |
| `forecasting/AR.scala`, `RandomWalk.scala`, `ARX.scala`, `ARX_D.scala`, `ARX_SR.scala`, `ARX_SR_D.scala` | `ar.py`, `random_walk.py`, `arx.py`, `arx_d.py`, `arx_sr.py`, `arx_sr_d.py` |
| `forecasting/ARMA.scala`, `ARIMA_diff.scala` (ScalaTion's ARIMA family) | `arima.py` (`ARIMA`, statsmodels-backed under the same interface) |
| `forecasting/Forecaster_Reg.scala` (regression-based recursive base, with its feature selection) | `forecasting/forecaster_reg.py` (`Forecaster_Reg`; `ARX` and `ARX_SR` extend it) |
| `forecasting/ARX_Formula.scala`, `FormulaParser.scala`, `ExoScreen.scala` | `arx_formula.py`, `formula_parser.py`, `exo_screen.py` |
| `forecasting/nonlinear/NARX_Formula.scala`, `NARX_FormulaParser.scala`, `NARX_SR_D.scala` | `nonlinear/narx_formula.py`, `narx_formula_parser.py`, `narx_sr_d.py` |
| `forecasting/nonlinear/llm_narx_sr/*.scala` (14 files) | `nonlinear/llm_narx_sr/*.py`, same base names in snake_case (`formula_scorer.py`, `eda4ts.py`, `datasets.py`, `term_library.py`, `search_arm.py`, `search_loop.py`, `genetic_proposer.py`, `llm_proposer.py`, `explainer.py`, `baseline_arm.py`, `pipeline_arm.py`, `plausibility_checker.py`, `significance.py`, `llm_narx_sr.py`) |
| `mathstat/Transform.scala` | `Util/transform.py` (`Transform`, `PowForm` .., `NormForm`, `MinMaxForm`, `TransformT`) |
| `forecasting/MakeMatrix4TS.scala` | `Util/build_matrix.py` |
| `LLMClient.scala` | `modeling/llm_client.py` |

Class names are ScalaTion's (`ARX_SR_D`, `NARX_Formula`, `FormulaScorer`, `CEM_Selector`).  Method
names that users call keep ScalaTion's camelCase (`inSample_Test`, `tnT_Test`, `rollValidate`,
`selectFeatures`, `formulaStr`, `nFree`, `fromTerms`, `parseFormula`); a snake_case alias may be
kept for compatibility (`formula_str`, `from_formula`) but the camelCase one is documented first.
Model names printed in reports are ScalaTion's: `ARX_D_6_6_2` (p, q, n_exo), `AR_6`, `RandomWalk`,
`NARX_SR_D(p, q, n_exo)`.

## Hyper-parameters (`args` dict)

`p` (lags of y), `q` (lags of each exogenous variable -- ScalaTion's `px`), `spec` (number of trend
columns: 0 implicit constant, 1 linear t/m, 2 + quadratic, 3 + cubic; ScalaTion's convention, default 1),
`cross` (ENDO-EXO cross terms, ScalaTion's `crx`; only `ARX_SR*` use it), `skip` (rows skipped in the
In-Sample diagnosis, default p), `lambda` / `scaled` (centered ridge).  Trend columns are named as
ScalaTion's `MakeMatrix4TS.trend` ("cons", "lin", "quad", ...).

## The base API every model follows

```
mod = ARX_D(args, y, hh, xe, method)            # or ARX_D.rescale(...) / ARX_D.fromTerms(formula, dataset, hh)
mod.train(Y_, X_)                                # direct: Y_ (m, hh) NaN-padded; recursive: train(y_, X_)
mod.forecast(t_start, t_end) -> (rows, hh)       # writes Yf (ORIGINAL units); Yf[t, h] forecasts y[t + h - 1]
mod.inSample_Test() ; mod.tnT_Test(rc=5)         # ScalaTion's printed output (banner, b = VectorD(...), REPORT, fitMap QoF)
mod.diagnoseAll(tRng=None) -> (hh, N_QoF)        # skip rows dropped (Diagnoser), DF reset per horizon
mod.params (intercept first) ; mod.parameter (slopes, ScalaTion's b) ; mod.fname ; mod.modelName ; mod.hparameter
mod.tr_size, mod.te_size (ScalaTion's Model.trSize / teSize: te = floor(0.2 m + 1))
```

QoF vectors are `np.ndarray(26)` in `QoF` order (`QoF.smape.ordinal` indexes them); `Fit.diagnose`
reproduces ScalaTion's formulas (R^2-bar via r_df, AIC/BIC via kParam = dfr + 2, smapeC = smape + 2 kParam / m).
`Forecaster_D.fit_rows` drops the NaN-padded trailing rows before a fit (ScalaTion's NO_DOUBLE rows).

## Printing

`Util/tools.py`: `banner`, `vector_str` ("VectorD(0.946883,\t...)", Java's %g), `matrix_str`,
`array_str` ("Array(yl6, ...)"), `set_str` ("LinkedHashSet(0, 1)"), `hparam_str`, `report`.
Use them so the Python logs read as the Scala ones.

## Formulas

`ARX_Formula` (linear terms) and `NARX_Formula` (free parameters `Par` with roles) print and parse the
same grammar as ScalaTion (`formulaStr`, `key`, `proposalStr`, `FormulaParser.parseFormula`,
`NARX_FormulaParser.parseFormula`); the reject reasons are ScalaTion's strings.

## Docstrings

numpy style; the module docstring names the ScalaTion file it ports and the differences, if any.
Keep the header block (Author, Date, Version, License, Note: AI assisted code).
