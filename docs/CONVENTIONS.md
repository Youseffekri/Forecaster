# Porting conventions: ScalaTion `scalation.modeling.forecasting` -> Python `modeling/forecasting`

The Python package mirrors ScalaTion 2.0 file by file.  Keep these rules when adding a file.

## Files and names

| ScalaTion | Python |
|---|---|
| `modeling/Fit.scala`, `FitM.scala` (QoF enum, diagnose, fitMap, showFitMap, summary) | `modeling/fit.py` (`QoF`, `Fit`, `diagnose`, `fitMap`, `showFitMap`, `Fit.summary`, `vif` = `Predictor.vif`) |
| `HyperParameter.scala` | `modeling/hyper_parameter.py` (`HyperParameter`: a dict of values that keeps the defaults, printed as ScalaTion; `freeze` makes `AR.hp` read-only) |
| `AR.hp` (+ `RidgeRegression.hp`, `Model.TE_RATIO`) | `forecasting/args.py` (`Args`: the checked settings of a model and a run; `hp` = `AR.hp`, read-only, built from its settings table) -- Python only as an object: ScalaTion passes `HyperParameter`s |
| -- (ScalaTion's test mains) | `forecasting/runner.py` (`run (args)`, `loadData`, `makeModel`: the one-object run of an experiment script) |
| `modeling/CenteredRidge*.scala` | `modeling/centered_ridge.py` (`CenteredRidge`: `hp` (lambda 0.1), `fit`, `predict`, `parameter` (fitted space), `slopes`, `intercept`, `params`, `ridge`, `effDF` (1 + tr H_lambda), `stdErr` (the sandwich covariance)) |
| `modeling/ElasticNetRegression.scala` (rev5.1: `solve` by cyclic coordinate descent, `l1_ratio`, `cdIter`, `nonZero`; `CenteredElasticNet` / `neuralnet.ElasticNetRegressionMV` / `CenteredElasticNetMV`, chosen by the `REGRESSION` import alias) | `modeling/elastic_net.py` (`ElasticNet`: scikit-learn's coordinate descent on the same objective (`alpha = lambda / m`), `l1_ratio`, `nnz`, `effDF` = 1 + nnz; chosen at run time by `Args.method = "enet"`) |
| `modeling/Model.scala` (`object Model`: `TE_RATIO` with the checked setter `TE_RATIO_=` ((0.05, 0.95) or 0 = no testing set), `teSize`, `trSize`; rev5.2: `VAL_RATIO`, `validating`, `valSize`, `fitSize`, `fitRng`, `valRng`, `fitRows`, `splitStr`, `inSample (block)`; rev5.3: `tuning` (was `tuneLambda`), `folds`, `withValidation (ratio, tuning_)(block)`) | `modeling/model.py` (`Model`: the same members; `TE_RATIO` a checked class property, `withValidation` / `inSample` are context managers, `fitRows (m, te_ratio, trSize)` a static method) |
| `HyperParameter.scala` (`Knob (name, grid, log, tag)`: `tagOf`, `toCoord`, `fromCoord`; `Knob.LAMBDA`, `Knob.L1_RATIO`; rev5.3) | `modeling/hyper_parameter.py` (`Knob`, a frozen dataclass with the same members and the two knobs) |
| `modeling/Regularized.scala` / `neuralnet/RegularizedMV.scala` (`tunable: Seq [Knob]`, lambda; `ElasticNetRegression(MV).tunable` adds l1_ratio; the `Centered*` companions carry it) | `CenteredRidge.tunable = (Knob.LAMBDA,)`, `ElasticNet.tunable = (Knob.LAMBDA, Knob.L1_RATIO)` (class attributes) |
| `modeling/FeatureSelection.scala`, `SubsetSearch.scala` (+ `ranked`), `CEM_Selector.scala` | `modeling/feature_selection.py` (`SelectionTech.Ranked`), `subset_search.py` (`ranked`), `cem_selector.py` |
| `autograd/Attn_Ranker.scala` (to follow the experiment) | `modeling/attn_ranker.py` (`Attn_Ranker`, the shape of `CEM_Selector`: `run ()` -> `scores`; optional PyTorch) |
| `forecasting/Diagnoser.scala` | `forecasting/diagnoser.py` (`Diagnoser`: `getSkip`, `setSkip`, skip-aware `diagnose`) |
| `forecasting/Forecaster.scala` (+ `ForecastMatrix.scala`; `object Forecaster { RC; ERROR_MEASURES; scoreOf; teRng }`, `inSample_Test`, `rollTest (rng, rc, label, verbose, showYf, hmax)`, `tnT_Test (rc, showYf)`, `valTest (rc, showYf)`, `tvT_Test`, `valScore`, `qofOn`, `getYf_org`, `recipe` / `refitTransform`, the hooks `trainNtestFull` / `trainNtestWindow` / `trainWindow` and `lead`, `rollValidate (rc, growing, doPlot, rng, verbose, hmax)`, `cvScore`, `forecastAt`, `diagnoseAll` -- rev5.2; rev5.3: `knobs`, `hpTag`, `withHP (hp2)`, `tune (knobs, measure, hs, verbose, search = "grid" | "golden" | "random", folds = Model.folds, trials, stream)`; rev5.4: `hparameter` the merged map, `slant (yf)`, `dfWarned` -- `lam`, `penalized`, `hpAll` gone) | `forecasting/forecaster.py` (`Forecaster`: the same names; `hparameter`, `hp` = `AR.hp` re-exported, `trSize`, `teSize`, `teRng` (= `Model`'s), `RC`, `scoreOf`, `rollTest (rng, rc, label, verbose, showYf, showParams)`, `tnT_Test`, `valTest (rc, showYf, showParams, verbose)` (ScalaTion's `Forecaster.validate (rando, ratio)`, Dr. Miller's one-step random split, is not ported), `tvT_Test`, `valScore`, `cvScore`, `knobs`, `hpTag`, `withHP (hp)`, `tune (knobs, measure, hs, verbose, search, folds, trials, seed)`, `bestHP`, `refitTransform ()`, `getYf_org`, `slant (yf)`, `qofOn (rng, hmax)`, `trainNtestWindow (n)`, the hook `_trainWindow (is_, t)` and `_lead` / `mSeries`; Python only: `effDF`, `stdErrOf`, `showParams`, `makeForecastMatrix`, `errorTable (rng, scale)` (the protocol)) |
| `forecasting/Forecaster_D.scala` (direct base: `reg`, `train`, `summary`, feature-selection scenarios) | `forecasting/forecaster_d.py` (`Forecaster_D`: `_make_model`, `train`, `slopes`, `bb`, `summary` / `summary (h)`, `fitCols`, `jointQoF`, `coefImportance`) |
| `forecasting/AR.scala`, `RandomWalk.scala`, `ARX.scala`, `ARX_D.scala`, `ARX_SR.scala`, `ARX_SR_D.scala` | `ar.py`, `random_walk.py`, `arx.py`, `arx_d.py`, `arx_sr.py`, `arx_sr_d.py` |
| `forecasting/ARMA.scala`, `ARIMA_diff.scala` (ScalaTion's ARIMA family) | `arima.py` (`ARIMA`, statsmodels-backed under the same interface) |
| `forecasting/Forecaster_Reg.scala` (regression-based recursive base: `reg`, `train`, `summary`, feature selection) | `forecasting/forecaster_reg.py` (`Forecaster_Reg`; `ARX` and `ARX_SR` extend it) |
| `forecasting/ARX_Formula.scala`, `FormulaParser.scala`, `ExoScreen.scala` | `arx_formula.py`, `formula_parser.py`, `exo_screen.py` |
| `forecasting/Example_Covid.scala`, `Example_ILI.scala`, `Example_ETTh1.scala`, `Example_GEFCom.scala` (the objects) + `forecasting/Datasets.scala` | `forecasting/datasets.py`: the four loader classes with the objects' members (`fileName`, `header`, `response`, `loadData (x_strs, y_str, ...)` -> (xe, y), `loadData_y`, `loadData_yy`, `context`, Covid's `clip`, `NO_EXO`, `BEST_EXO`, `BEST2_EXO`), `ExoMode`, `DatasetConfig` (`shortlist (data, trSize, k, maxLag)`), `Datasets` (`COVID_FILE` lazy), `loadH` (= `MatrixD.loadH`). One file for all of them on purpose: ScalaTion keeps each loader with its test mains, which are the scenarios of `Examples/Example_*.py` here |
| `forecasting/EDA4TS.scala` | `forecasting/eda4ts.py` (`EDA4TS`: the profile and `residualReport`; `amplitudes`, `topFreqs`) |
| `forecasting/Stationarity.scala` (`UnitRoot`), `forecasting/Stationarity_KPSS.scala` | `forecasting/stationarity.py` (`UnitRoot`: lags rule, critical values at 15 probabilities, interpolated p-value), `forecasting/stationarity_kpss.py` (`Stationarity_KPSS`: `statistic`, `pvalue`, `show`; the coefficient table) |
| `forecasting/nonlinear/NARX_Formula.scala`, `NARX_FormulaParser.scala`, `NARX_SR_D.scala` (`OptimizerNL`, `OPTIMIZER_NL`, the class (an `ARX_D`), `apply`, `rescale`, `make`, `buildMatrix`, `optimize`, `fromTerms`, `fitNL`, `joint`, `ridge`, `initializeW`, `tunable`, `getTransforms`, `formNames`) | `nonlinear/narx_formula.py`, `narx_formula_parser.py`, `narx_sr_d.py` (the enum and the module default live with `NARX_SR_D`, as in Scala) |
| `optimization/nonlinear_ls/VarPro.scala` | `modeling/optimization/varpro.py` (`VarPro`: `system`, `coefficients`, `residual`, `objective`, `optimize`, `stepCounts`) |
| `forecasting/nonlinear/llm_narx_sr/*.scala` (12 files) | `nonlinear/llm_narx_sr/*.py`, same base names in snake_case (`formula_scorer.py` with `shortlist (cfg)`, the `validating` switch and `tuning`, `term_library.py`, `search_arm.py`, `search_loop.py`, `genetic_proposer.py`, `llm_proposer.py` with `promptBlock (eda, names)`, `explainer.py`, `baseline_arm.py`, `pipeline_arm.py`, `plausibility_checker.py`, `significance.py` (+ `hacVar`, `dm_test`), `llm_narx_sr.py`; Python only: `protocol.py`, the measurement protocol); the package builds on `forecasting`, never the other way round |
| `mathstat/Transform.scala` | `Util/transform.py` (`Transform` with `setLU`, `PowForm` .., `NormForm`, `MinMaxForm` (centered [-2, 2]), `TransformT` with `form` and `wlu`) |
| `mathstat/Correlogram.scala` (+ `VectorD.acov`, `acorr`, `ccorr`, `corr`) | `Util/correlogram.py` (`durbinLevinson`, `acfPacf`, `ljungBox`, `acfSpikes`, `collinearPair`, `heteroscedastic`; ScalaTion's defaults: `acorr` adjusted) |
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

## Settings (`Args`) and hyper-parameters (`hparameter`)

Every model is built as `Model (y, hh, [xe,] args)` -- the data first, the settings last (ScalaTion's order:
`ARX_D.apply (xe, y, hh, ..., hparam = AR.hp)`); `args` is an `Args` (None => the defaults; a dict of
settings is checked the same way); the old order (hparam first) raises a TypeError.  `Args` is the one table of
every setting: name, default, check (type and range) and meaning; an unknown name raises AttributeError with the
closest names ("did you mean 'px'?"), a bad value ValueError; `Args.help (Model)` lists the settings a model
reads (its class attribute `ARGS`).  The settings:

* ScalaTion's `AR.hp` (printed in the REPORT): `p` (lags of y, 1), `d`, `q`, `P`, `D`, `Q`, `s` (ARIMA), `px`
  (lags of each exogenous variable, 1), `spec` (number of trend columns: 0 implicit constant, 1 linear t/m,
  2 + quadratic, 3 + cubic; 1), `crx` (ENDO-EXO cross terms; only `ARX_SR*`), `nneg` (forecasts clipped at 0);
  the switches are bools in `Args`, 0 / 1 in the HyperParameter (as ScalaTion keeps them).
* Python only: `skip` (rows skipped in the In-Sample diagnosis; None => the model's rule: p, max (1, p) for
  RandomWalk, max (p, q + 1) + d for ARIMA, the largest lag for `fromTerms`, 0 for `ARX.trend`), `lam` (the
  penalty, ScalaTion's `RidgeRegression.hp("lambda")`, 0.1; the alias "lambda" is accepted) and `l1_ratio` (the
  share of the L1 penalty, ScalaTion's `ElasticNetRegression.hp("l1_ratio")`, 0.5); `mod.hparameter` is the
  MERGED map, the model's keys plus the knobs its regression reads (ScalaTion rev5.4's `hparameter`).
* Model settings: `method` (None => the model's `METHOD`, "enet" for every regression-based forecaster since
  rev2.3 -- ScalaTion rev5.4's `REGRESSION` import `CenteredElasticNet(MV)`; checked against its `METHODS`),
  `scaled`, `optimizer`, `tune` (NARX_SR_D), `te_ratio` (ScalaTion's `TE_RATIO`, here per model).
* Run settings (`runner.run`): `dataset`, `exo`, `model`, `tFormT` (passed to the model's `rescale`; None => no
  rescaling), `hh`, `validation`, `rc`.  `tFormT` is an argument of `rescale` / `fromTerms` (ScalaTion's
  signature), never read by a model; `rescale` refuses an `args.tFormT` that differs from the one given.

A model keeps a private snapshot (`_args`) and a private `HyperParameter` (`_hparam` = `Args.hparam ()`: `AR.hp`
with the Scala values of args, plus the unregistered "skip" and "lambda"); there is no public `hparam`.
`mod.hparameter` (ScalaTion's name) returns a COPY; the REPORT prints its registered entries as "(value,default)"
pairs, exactly as the Scala logs (RandomWalk has none: "null").  `AR.hp` is read-only (`HyperParameter.freeze`):
a model or script cannot change the defaults of the models made after it.  A factory that fixes values does it
on the snapshot (`fromTerms`: p, px, spec = 0; `ARX.trend`: p = px = 0; `AR`: px = spec = 0; rescale by Norm:
nneg off), and sub-models get the parent's snapshot with its resolved method (`_subArgs`).  Trend columns are
named as ScalaTion's `MakeMatrix4TS.trend`.

## The base API every model follows

```
mod = ARX_D(y, hh, xe, args)                     # or ARX_D.rescale(y, hh, xe, args) / ARX_D.fromTerms(formula, dataset, hh, args)
mod.train(Y_, X_)                                # direct: Y_ (m, hh) NaN-padded; recursive: train(y_, X_)
mod.forecast(t_start, t_end) -> (rows, hh)       # writes Yf (ORIGINAL units); Yf[t, h] forecasts y[t + h - 1]
mod.forecastAt(h)                                # column h of Yf after forecasting every row
mod.inSample_Test() ; mod.tnT_Test(rc=5)         # ScalaTion's printed output (banner, b = VectorD(...), REPORT, fitMap QoF)
mod.diagnoseAll(tRng=None) -> (hh, N_QoF)        # skip rows dropped (Diagnoser), DF reset per horizon
mod.summary() ; mod.summary(h)                   # ScalaTion's SUMMARY (reflects the last diagnose call)
mod.parameter (b / bb(?, 0): the regression's coefficients) ; mod.slopes ; mod.intercept ; mod.params (intercept first)
mod.fname ; mod.modelName ; mod.hparameter (a copy) ; mod.tr_size, mod.te_size (Model.trSize / teSize: te = floor(args.te_ratio m + 1))
```

The regression lives in the bases, as ScalaTion's `reg`: `Forecaster._make_model` ("enet" = `ElasticNet (args.lam,
args.l1_ratio, args.scaled)` by default, "ridge" = `CenteredRidge(args.lam, args.scaled)`, "sm_ols", "sk_lr"),
`Forecaster_Reg.train` (recursive) and `Forecaster_D.train`
(direct, after `fit_rows` drops the NaN-padded rows, ScalaTion's NO_DOUBLE).  The model classes add the
matrix (`buildMatrix (xe_bfill, y, hparam, ...)`, `formNames (n_exo, hparam, ...)` -- ScalaTion's argument
order), the forecasts and the sub-models: `getModel` (ARX, ARX_D -- inherited by ARX_SR_D and NARX_SR_D)
and `getModel_h` (ARX_D; abstract in Forecaster_D).  Python differences kept on purpose: the "sm_ols" /
"sk_lr" choices, one solve per horizon (no MV class), the constructor builds the matrix (ScalaTion's
`apply`), range-based `forecast`, the `_trainWindow (is_, t)` hook (ScalaTion's `trainWindow`) instead of
overriding the tests.  The boundary rule of both code bases (ScalaTion rev5.2, Python rev2.1): a training
window ends where its last target would reach the forecast origin -- `[is_, t)` for a series, `[is_, t - hh + 1)`
for a direct model (`Forecaster_D._trainWindow`), `[is_, t - h + 1)` for the single-horizon model at horizon h
(`getModel_h`: `_lead = h - 1`; ScalaTion keeps its h - 1 undefined rows as NO_DOUBLE, Python drops them, so the
split of such a model is computed on `mSeries = len (y) + lead`); the FS rows (`fs_rng`) and `cvScore`'s rows
are cut the same way.

`rescale (y, hh, xe, args, tFormT = MinMax)` takes a `TransformT` fitted on the training rows (column by column;
`Model.trSize` rows as ScalaTion's `rescale`, the fit rows while choosing under `Model.validating` -- no `trSize`
argument, as in ScalaTion): MinMax (the default) maps to ScalaTion's centered [-2, 2], Norm standardizes and sets
nneg = 0, Log1p keeps zero at zero; a response transform into a space where y can be negative turns the
rectification off (`centered`; ScalaTion applies it in `ARX_D` only -- here every model, since clipping at 0 in a
centered space is wrong).  `AR.rescale` standardizes (NormForm), as ScalaTion's.  `refitTransform ()` rebuilds a
model by its factory under `Model.inSample` (the switch suspended), so a model chosen on the fit rows is reported
with its transforms fitted on the training rows.

## ScalaTion rev5.0 <-> Python rev2.0: the factory signatures

The data come first and in Python's order `(y, hh, xe, args)` (ScalaTion: `(xe, y, hh, fname_, tRng, hparam, bakcast)`
-- `fname_`, `tRng` and `bakcast` are not ported; `hparam` and `scaled` are in `args`); after them, the arguments
follow ScalaTion's order and defaults:

| ScalaTion rev5.0 | Python rev2.0 |
|---|---|
| `ARX.rescale (xe, y, hh, fname_, tRng, hparam, bakcast, tFormT = MinMax)` (rows: `Model.fitRows`; sets `recipe`) | `ARX.rescale (y, hh, xe, args, tFormT = MinMax)` (rows: `Model.fitRows`; sets the recipe) |
| `ARX_D.rescale (...)`, `ARX_SR.rescale (..., fEndo, fExo, bakcast, tFormT)`, `ARX_SR_D.rescale (...)` | `ARX_D.rescale (y, hh, xe, args, tFormT)`, `ARX_SR.rescale (y, hh, xe, args, fEndo, fExo, tFormT)`, `ARX_SR_D.rescale (...)` |
| `NARX_SR_D.apply (xe, y, hh, fname_, tRng, hparam, fEndo, fExo, bakcast, tune, trSize, scaled)` | `NARX_SR_D.apply (y, hh, xe, args, fEndo, fExo, trSize, fname, **opt)` (`tune`, `scaled`, `optimizer` in `args`) |
| `NARX_SR_D.rescale (..., fEndo, fExo, bakcast, tFormT = MinMax, lu, tune, trSize, scaled)` | `NARX_SR_D.rescale (y, hh, xe, args, fEndo, fExo, tFormT, lu, trSize, fname, **opt)` |
| `ARX.fromTerms (resp, preds, hh, fname_, tRng, hparam, bakcast, tFormT = null, trSize = -1, scaled)(using dataset)`, `ARX_D.fromTerms`, `NARX_SR_D.fromTerms` | `ARX.fromTerms (formula, dataset, hh, args, tFormT = None, trSize = -1)`, `ARX_D.fromTerms (...)`, `NARX_SR_D.fromTerms (formula, dataset, hh, args, tFormT, trSize, maxLag, freqs, **opt)` |
| `ARX_Formula.fit (hh, tFormT = this.tFormT, trSize = -1, scaled)(using dataset)`, `fit_D (...)` | `f.fit (dataset, hh, args, tFormT = None, trSize = -1)`, `f.fit_D (...)` (None => the formula's `tFormT`) |
| `NARX_Formula.fitNL_D (hh, tFormT, trSize, scaled)(using dataset)` | `f.fitNL_D (dataset, hh, args, tFormT = None, trSize = -1, **opt)` |
| `ARX_Formula.names (preds)` | `f.names` (a property; `labels` is `TermLibrary`'s only) |
| `Forecaster.tnT_Test (rc = 5, showYf = false)` (rev5.4: no `Forecaster_D` override; a test passes `showYf = true`) | `Forecaster.tnT_Test (rc = None => RC, showYf = False, showParams, verbose)` (the same) |
| `Model.TE_RATIO_= (ratio)`: a flaw outside (0.05, 0.95), 0 allowed (no testing set) | `Model.TE_RATIO = ratio`: `ValueError` outside (0.05, 0.95), 0 allowed; `args.te_ratio` None => `Model.TE_RATIO` when the model is built |
| `Forecaster.validate (rando, ratio)(idx)` (Miller's one-step split test, not ported) | -- |

### ScalaTion rev5.2 <-> Python rev2.1: the protocol

| ScalaTion rev5.2 | Python rev2.1 |
|---|---|
| `Model.VAL_RATIO`, `validating`, `valSize`, `fitSize`, `fitRng`, `valRng`, `fitRows (m)`, `splitStr`, `withValidation (ratio, tuning_) { }`, `inSample { }` | the same names (`Model.fitRows (m, te_ratio, trSize)` replaces `arx.fitRows`; `with Model.withValidation (ratio, tuning):`, `with Model.inSample():`) |
| `Model.TE_RATIO = 0`: no testing set; `tnT_Test` flaws and returns null | `Model.TE_RATIO = 0` (or `Args (te_ratio = 0)`): `tnT_Test` prints the flaw and returns None; `tvT_Test` reports NaN for the test |
| `rollTest (rng, rc, label, verbose, showYf, hmax)`, `tnT_Test (rc, showYf): MatrixD`, `valTest (rc, showYf)`, `tvT_Test (rc, showYf)`, `valScore (rc, measure, hs)`, `qofOn (rng, hmax)` | `rollTest (rng, rc, label, verbose, showYf, showParams)`, `tnT_Test (rc, showYf, showParams, verbose)`, `valTest (...)` (was `validate` until rev2.0), `tvT_Test (rc, showYf, verbose)`, `valScore (rc, measure, hs)` (`hmax`: Python's recursive sub-models are scored through `rollTest` as well, at every horizon) |
| the hooks `trainNtestFull ()`, `trainNtestWindow (n)`, `trainWindow (is, t)`; `lead` | `_trainWindow (is_, t)` -> (targets, X rows); `_lead`, `mSeries` |
| the boundary rule: `Forecaster_D.trainWindow (is, t)` = rows `is until t - hh + 1`; `fs_rng` cut hh - 1 (direct) / `lead` (recursive) | `Forecaster_D._trainWindow` = rows `[is_, t - hh + 1)`; `fs_rng` cut the same way; the TnT pins of every direct model moved in rev2.1 |
| `getYf_org` (the forecast matrix in original units; `getYf` in model space) | `getYf_org` (= `Yf`: Python's forecast matrix is kept in original units) |
| `recipe`, `refitTransform ()` | `_recipe`, `refitTransform ()` |
| `cvScore (folds, measure, hs)` (`lam` and `penalized`: gone in rev5.4) | the same |
| `Forecaster.RC`, `ERROR_MEASURES`, `scoreOf (qof, inSample, extra, measure, hs)` | `Forecaster.RC`, `scoreOf (qof, inSample, nParams, measure, hs)` (`nParams`: the effective parameters, Python's `effDF`) |
| `Forecaster_Reg.evalCols` / `select0` on the switch (h = 1), `scored`, `fs_rng`, `val_rng`, `getModel (cols, hparam2)`, `selectedModel` (refit + bestHP), `testSelected` through `rollTest` | the same (`getModel (cols)`: the sub-model carries `bestHP` through `_subModel`) |
| `Forecaster_D.fitCols` on the switch, `jointQoF`, `selectFeatures` without `agg`, `getModel (cols, hparam2)`, `getModel_h (cols, h, hparam2)`, `selectFeaturesAtH` carrying the tuned knobs | the same (`jointQoF (qof, nParams)`) |
| `HyperParameter.clone` a deep copy; `RidgeRegressionMV.apply` on `dim2`; `RidgeRegression.lambda < 0` a flaw; the old `findLambda`s commented out | -- (Python's `HyperParameter` copies; `Args` is immutable) |

### ScalaTion rev5.3 <-> Python rev2.2: the knobs

| ScalaTion rev5.3 | Python rev2.2 |
|---|---|
| `Knob (name, grid: VectorD, log, tag)`, `tagOf (v)` (`_L1.0`, `_L0.01`, `_L1.0E-12`, `_A0.5`, `_A0.25`), `toCoord`, `fromCoord`; `Knob.LAMBDA`, `Knob.L1_RATIO` | `Knob (name, grid: tuple, log, tag)` with the same members and tags (`1e-12` prints `_L1.0E-12` as Java does) |
| `Regularized.tunable` / `RegularizedMV.tunable` (lambda), `ElasticNetRegression(MV).tunable` (lambda, l1_ratio), `Centered*.tunable` | `CenteredRidge.tunable`, `ElasticNet.tunable` |
| `Model.tuning` (was `tuneLambda`), `Model.folds` (0 => the validation block, k => k-fold), `withValidation (ratio, tuning_ = false)` | `Model.tuning`, `Model.folds`, `withValidation (ratio = None, tuning = False)` (always sets `tuning`, as ScalaTion's does) |
| `Forecaster.knobs` (`REGRESSION.tunable`; `Seq ()` without a regularized regression), `penalized = knobs.nonEmpty`, `hpAll` (the merged map), `hpTag`, `withHP (hp2: HyperParameter)` (was `withLambda`), `tune (knobs, measure, hs, verbose, search = "grid" \| "golden" \| "random", folds = Model.folds, trials, stream) -> (HyperParameter, score)` (was `findLambda`), `LAMBDAS` gone | `knobs` (by `method`: ridge / enet; `()` otherwise), `penalized`, `hpAll` (a dict: the hparameter plus the args-only knobs), `hpTag`, `withHP (hp: Mapping)`, `tune (knobs, measure, hs, verbose, search, folds = None => Model.folds, trials, seed) -> ({name: value}, score)` (the knob values only; `{}` without knobs), `LAMBDAS` gone |
| the name tags `_L1.0`, `_L1.0_A0.5` on `ARX`, `ARX_D`, `ARX_SR`, `ARX_SR_D`, `ARX_Quad_D`, `NARX_SR_D` | the same on `ARX`, `ARX_D`, `ARX_SR`, `ARX_SR_D`, `NARX_SR_D`; a plain least-squares model (sm_ols, sk_lr) has no knobs and no tag (was `_L0.0`) |
| `Forecaster_Reg` / `Forecaster_D`: `hpOf`, `bestHP (cols)` (were `lamOf`, `bestLambda`), `withHP (tune ()._1)` in `evalCols` / `fitCols` under `Model.tuning` | `bestHP (cols)` (a dict; `SelectionReport`), `withHP (tune ()[0])` in `evalCols` / `fitCols` under `Model.tuning` |
| `resetDF_k` warns once per model (`dfWarned`; rev5.4: once per RUN, `Forecaster.dfWarned`) | `Fit.resetDF_k` warns once per run (`Fit.dfWarned`) |
| `FormulaScorer (hh, scaled, validating = Model.validating)`: `tuning`, `fs_rng (rows)`, `build (f, size, hp2)`, `fresh (f)`, `tuned (mod)`, `fit`, `fitFull (f, hp2)`, `score`, `scoreH`, `tnt (f, hp2)`, `train` (the FS rows), `qofFit` / `qofTrain` on the FS rows, `qofValidation` (`rollTest`), `qofByH` (`qofOn`), `scoreOf (qof, extra)`, `evalCols`; `Candidate.hp`, `lam` | `FormulaScorer (data, hh, hp, scaled, freqs, validating)`: `tuning`, `fs_rng`, `build (f, size, fit_params, hp)`, `fresh`, `tuned`, `fit`, `fitFull (f, hp)`, `score`, `scoreH`, `tnt (f, hp)`, `train`, `qofFit` / `qofTrain`, `qofValidation`, `qofByH`, `scoreOf (qof, extra, nParams)`, `evalCols`; `Candidate.hp` (a dict), `lam` (a property) |
| `LLM_NARX_SR.VALIDATING`, `scorerOf (validating)`, CSV columns `val_ratio` / `rc`, `ExpRun.valRatio` / `rc`, `showBest` via `tnT_Test`, `toModel` / `record` / `showPareto` at the candidate's knobs | the same |
| -- | Python only (rev2.2): `Forecaster.errorTable (rng, scale)`; `llm_narx_sr/protocol.py` (`Protocol`: `runModels`, `runArms`, `summary`, `compare`, `compareAll`, `report`, `write`); `significance.hacVar`, `dm_test`; `Examples/Example_Protocol.py` |

### ScalaTion rev5.4 <-> Python rev2.3: the elastic net by default, the standard errors of a penalized fit

| ScalaTion rev5.4 | Python rev2.3 |
|---|---|
| the `REGRESSION` import of `Forecaster_Reg` / `Forecaster_D` is `CenteredElasticNet` / `CenteredElasticNetMV` (the ridge commented); `ElasticNetRegression.hp` IS `RidgeRegression.hp` (lambda 0.1, l1_ratio 0.5, no `sparse`) | `METHOD = "enet"` in both bases (`ElasticNet (lam, l1_ratio, scaled)`; the tests that need least squares or the ridge say so); `NARX_SR_D` reads `method` like every direct model |
| `ElasticNetRegression(MV).train`: l1_ratio = 0 => `RidgeRegression(MV)`'s own solver (exact), else coordinate descent (tol 1e-6 on the coefficient change); NO `sparsify` (the zeros are the soft threshold's own) | `ElasticNet.fit`: l1_ratio = 0 => `ridge_solve`, else scikit-learn's descent at `tol = 1e-6` (the two agree to five digits; a coefficient AT the soft threshold can still be a tiny nonzero on one side and an exact zero on the other, which moves a selection's FPE charge by one) |
| `object Regularized { stdErr (xc, b, sse, lambda, alpha, xMu, xSig) }`: the sandwich covariance over the active set, sigma^2 = sse / (m - 1 - tr (M Xa'Xa)), intercept first, original units, NaN for a dropped or constant column | `modeling/regularized.py` `stdErr (xc, b, sse, lam, alpha, xm, sd)`: the same function (numpy) |
| `CenteredRidge(MV).stdErr (k)`, `CenteredElasticNet(MV).stdErr (k)` (on the last training window `xC`, `yC`), `summary` = `showSummary (stdErr, "intercept" +: fname, parameter_org, vifs)` | `CenteredRidge.stdErr (k)`, `ElasticNet.stdErr (k)` (on `_Xs`, `_sse`); `Forecaster.stdErrOf` takes them (`_penalizedSE`), else the OLS formula |
| `Forecaster_Reg.summary` / `Forecaster_D.summary (h)` through `reg.stdErr (h - 1)`; `Fit.summary`: diag ((X'X)^-1) = the squared row norms of R^-1; `Fit.diagnose`: no F p-value when fStat < 0; `sumCoeff`: `NA NA NA` for a row without a standard error | the same (`summary`, `summary (h)`; `Fit.summary`, `Fit.diagnose`, `NA` rows) |
| `hparameter` IS the merged map (`REGRESSION.hp ++ hparam`); `hpAll`, `penalized`, `lam` gone; `knobs.nonEmpty` says whether there is anything to tune; `tune` flaws (verbose) when a log knob's best grid value sits at the grid's edge | `hparameter` merges the knobs' values (from the args) into the model's map; `hpAll`, `penalized`, `lam` gone (`hparameter["lambda"]`); the same edge warning |
| `Forecaster.slant (yf = getYf_org)` (moved from `ForecastMatrix`, original units), `diagnoseAll (y_, yf, tRng)` without `showYf`; `inSample_Test` / `rollTest` print `slant ()` | `Forecaster.slant (yf = None => getYf_org)`, `diagnoseAll (y_, yf, tRng)` without `showYf`; the prints the same |
| `Forecaster.dfWarned` (once per run); `FeatureSelection.reportSelection`: `bestOf (cols)` (a per-horizon set is its own best step) | `Fit.dfWarned`; `bestOf` |
| `FormulaScorer`: `Candidate.failed = score.isInfinite`, infinity the failed score, `Candidate.lam` gone, `hpOf` = `hparameter`, `score (f)` through `qofOn (fs_rng (fitRng))`; `qofTrain`, `qofFit`, `qofByH`, `diagnoseH`, the `scoreOf` wrapper gone (`qofValidation` stays) | `FormulaScorer.hp` is the ONE map (the class attribute; `LLM_NARX_SR.hp` its alias, `DEFAULT_HP` gone), `Candidate.lam` gone (`hp["lambda"]`), `qofIn (mod, rows)`, `residuals (mod, rows, noSkip)`, infinity the failed score |
| `LLM_NARX_SR.showBest`: `mod.qofOn (fs)`, per horizon `qofOn (fs, h)` before `summary (h)`; `SearchLoop.bestSoFar` starts at infinity | the same |

Python-only members (the rankers, `effDF`, `stdErrOf`, `avg (x~(hi,lo))`, `showParams`, `Model.teRng` / `trSize` as the
models' split) keep their names until ScalaTion gets them.

QoF vectors are `np.ndarray(26)` in `QoF` order (`QoF.smape.ordinal` indexes them); `Fit.diagnose`
reproduces ScalaTion's formulas (R^2-bar via r_df, AIC/BIC via kParam = dfr + 2, smapeC = smape + 2 kParam / m).

## Nonlinear parameters (`NARX_SR_D`)

`NARX_SR_D` is an `ARX_D` whose columns carry nonlinear parameters: from transform TYPES (`apply` /
`rescale (y, hh, xe, args, fEndo = [TransformT.Root], fExo = None => Pow, tFormT = MinMax, lu = (1, 5))` with
args.tune, args.optimizer, the parameters' start values and bounds from `TransformT.wlu`, `initializeW`, `tunable`,
`getTransforms`, `optimize`) or from a formula (`fromTerms`).  `OptimizerNL.VarPro` (default), `LBFGS_B`,
`Joint` -- `narx_sr_d.OPTIMIZER_NL` is the module default a test may reassign (ScalaTion's `var
OPTIMIZER_NL`); args.optimizer takes the enum, its name or None (the default).  `NARX_SR_D.fitNL` returns
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
take `args` (e.g., `Args (lam = 1.0)`, ScalaTion's `RidgeRegression.hp("lambda") = 1.0`), then `tFormT`, then
`trSize`, in ScalaTion's order.

## Docstrings

numpy style; the module docstring names the ScalaTion file it ports and the differences, if any.
Keep the header block (Author, Date, Version, License, Note: AI assisted code).
