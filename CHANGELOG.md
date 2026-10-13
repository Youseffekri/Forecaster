# Changelog

## 2.1.0 (ForecasterPy_rev1.1) -- 2026-10-13: feature selection inside the forecaster bases, as in ScalaTion

- New `modeling/forecasting/forecaster_reg.py` -- `Forecaster_Reg(FeatureSelection, Forecaster)`, ScalaTion's
  `Forecaster_Reg`: the regression-based RECURSIVE base (the regression methods "sm_ols" / "sk_lr" / "ridge",
  `train`, `_predict`, the recursive `forecast` through the subclass's `_forge`) and its feature selection
  (`forwardSelAll`, `backwardElimAll`, `stepwiseSelAll`, `beamSelAll`, `cemSelAll`, `correctQoF`, `evalCols`,
  `select0`, `getBest`, `cemProbs`, `importance`, `fs_rng`, the `reportSelection` hooks).  `ARX` (and so `ARX_SR`)
  extends it: `arx.selectFeatures(SelectionTech.Stepwise)` replaces the former `FeatureSelection_Reg(arx)` adapter;
  `ARX.getModel(cols)` builds the sub-model (hh = 1, ScalaTion's `ARX.getModel`).  `RandomWalk`, `AR`, `ARIMA`
  stay on `Forecaster`.
- `Forecaster_D` now holds its own feature-selection scenarios (ScalaTion's `Forecaster_D`): `selectFeatures`,
  `selectFeaturesAtH`, `selectFeaturesPerH`, `fitCols`, `getBest`, `cemProbs`, `getModel` (abstract), a default
  `getModel_h(cols, h, method, args)` (ScalaTion's `ARX_D.getModel_h` recipe), the `reportSelection` hooks, and
  the companion functions `smapeFPE` / `aggFPE` (module level, also `Forecaster_D.aggFPE`); `getYy` added.
- `modeling/forecasting/feature_selection_forecaster.py` (ex `feature_selection_d.py`: `FeatureSelection_D`,
  `FeatureSelection_Reg`, `subModel`, `subModel_h`) is REMOVED; the sub-model helper is `Forecaster._subModel`.
  `modeling/feature_selection.py`, `subset_search.py`, `cem_selector.py` (ScalaTion's base files) are unchanged.
- `requirements.txt` lists every library (core, pytest, and torch + matplotlib for the neural models);
  `pip install -e .[neural]` now pulls matplotlib too; the matplotlib import of `MHAttn_Regressor` is lazy.
- Printed output, numbers and the 178 tests are unchanged (aRX_DTest8 / subsetSearchTest still reproduced).

## 2.0.0 (ForecasterPy_rev1.0) -- 2026-10-12: the complete ScalaTion-consistent package

The package now mirrors ScalaTion 2.0's `scalation.modeling.forecasting` file by file (see
`docs/CONVENTIONS.md`): names, algorithms, hyper-parameters, formula grammars and the printed output.

Output (the Util rewrite)
- `modeling/fit.py` -- ScalaTion's `Fit` / `FitM`: the `QoF` enum (26 measures), `Fit.diagnose` with ScalaTion's
  formulas (sde, rse, R^2-bar via the DF ratio, F, AIC/BIC with kParam = dfr + 2, bmae, mape, mase, sMAPE-IC),
  `resetDF` / `resetDF_k`, `fitMap` ("LinkedHashMap(rSq -> 0.910754, ...)"), `showFitMap` (the "label -> VectorD(...)"
  table), `help`; `modeling/forecasting/diagnoser.py` -- `Diagnoser` (skip rows dropped from the diagnosis).
- `Util/tools.py` -- `banner`, `fmt`, `fmt_g` (Java's %g), `vector_str`, `matrix_str`, `array_str`, `set_str`,
  `hparam_str`, `report` (the REPORT block); the old metric functions are kept and now wrap `modeling.fit`.
- `Forecaster.inSample_Test` / `tnT_Test` print ScalaTion's blocks: banner, "b = VectorD(...), fn = Array(...)",
  REPORT (modelName, hparameter, features, parameter, fitMap of horizon 1), "diagnoseAll" banner and the
  "fitMap QoF =" table (`diagnoseAll(tRng)`, `teRng`).  The Covid-19 `ARX_D_1_0_0` run reproduces the ScalaTion
  log digit for digit (in-sample b = 0.946883, rSq 0.910754; TnT sMAPE 18.5489 .. 70.4402, m = 24 .. 19).

Base classes (ScalaTion's API)
- `Forecaster`: `train`, `forecast` (every row; `skip` only affects the diagnosis), `forecastAll`, `test`, `trainNtest`,
  `report`, `diagnoseAll`, `rollValidate(rc=5)` (ScalaTion's default), `inSample_Test`, `tnT_Test` (`trainNtest_Test`
  kept as an alias), `params` (intercept first), `parameter` (slopes = ScalaTion's b), `intercept`, `fname` /
  `getFname`, `modelName`, `hparameter`, `mod_resetDF`; `trSize` / `teSize` / `teRng` = ScalaTion's `Model.trSize`
  (te = floor(0.2 m + 1); 17,420 rows now split 13935/3485, 300 rows 239/61).
- `Forecaster_D`: the target matrix Y is NaN-padded (ScalaTion's NO_DOUBLE) and `fit_rows` drops the trailing rows
  before a fit, as `Forecaster_D.train` does; `bb`; mixes in `FeatureSelection_D`.
- `spec` follows ScalaTion: 0 = implicit constant (was 1), 1 = linear t/m (was 2), 2 = + quadratic, 3 = + cubic;
  default 1; trend columns named as `MakeMatrix4TS.trend` ("cons", "lin", "quad").  `ARX` has no cross terms
  (only `ARX_SR*`, ScalaTion's `crx`).  Model names: `ARX_6_6_2`, `ARX_D_6_6_2`, `ARX_SR_6_6_2`, `ARX_SR_D_6_6_2`,
  `AR_6`, `RandomWalk`, `ARIMA_1_1_1`, `NARX_SR_D(p, q, n_exo)`.
- `modeling/regression.py` -- `CenteredRidge` (ScalaTion's CenteredRidgeMV); `ARX`, `ARX_D` take `method="ridge"`
  (args "lambda", "scaled") besides "sm_ols" and "sk_lr"; `NARX_SR_D` fits its coefficients with it.
- `Util/build_matrix.py` -- `trend_names`, `build_trend_matrix` (ScalaTion's spec), `build_matrix_Y` (NaN padding),
  `tr_size_of` = `trSize`.

Models
- New `ar.py` (`AR`, Yule-Walker / OLS, forecasts for every row; `ar_yw.py` keeps `AR_YW` as an alias), `random_walk.py`
  (`RandomWalk`), `arima.py` (`ARIMA`, statsmodels-backed under the ScalaTion interface; forecasts from the Kalman
  filter's predicted state; `rollValidate` retrains every rc rows), `exo_screen.py` (`ExoScreen`, `ExoRank`, `ExoPool`,
  `model_screen` = ScalaTion's `Model.screen`).
- `ARX.fromTerms` / `ARX_D.fromTerms` (one column per formula term; recursive forecasting beyond h = 1 only for the
  standard lag layout) and `NARX_SR_D.fromTerms` (`from_formula` kept as an alias) with ScalaTion's parameter names
  (`tFormT`, `trSize`, `scaled`, `optimizer`, `maxLag`); column 0 of `Yf` stays in original units under a TFORM.
- `NARX_SR_D.mod_resetDF` counts the fitted nonlinear parameters (`nFree`) as degrees of freedom, as ScalaTion;
  `dof` = terms + nFree (the former `dfr` property, whose name now belongs to `Fit`).

Formulas (ScalaTion's names, the DSL and the parsers consistent with the Scala)
- `formula.py` -> `arx_formula.py` (`Term`, `FormulaVar`, `TransformedTerm`, `PowerTerm`, `ProductTerm`, `WaveTerm`,
  `FUNCTIONS` (with `dq`), the DSL `log1p`, `pow`, `powr`, `wave`, ..., `distinct`, `TermData`, `ARX_Formula` with
  `formulaStr`, `key`, `nTerms`, `maxLag`, `prepare`, `matrix`, `lagHP`, `fit`, `fit_D`, `grammar`; the wave part prints
  last as in Scala) and `nonlinear/narx_formula.py` (`ParRole`, `Par` with Miller's current defaults -- Power
  (1.5, 1.1, 4), Decay (0.5, 0.01, 0.99), Half (.., 0.01 rng, 10 rng), Cap (.., 0.05 big, 5 big) -- `ShiftTerm`,
  `ScaleTerm`, `ParPowerTerm`, `SatTerm`, `HingeTerm`, `TransTerm`, `EmaTerm`, `DqTerm`, `ParWaveTerm`, `NARX_Formula`
  with `params`, `nParams`, `nFree`, `proposalStr`, `fresh`, `resolve`, `redundant`, `slots`, `fitNL_D`).  The former
  short names (`Var`, `Shift`, `Formula`, ...) remain as aliases.
- `formula_parser.py` -> the linear `FormulaParser` (`parseFormula`, `tryParseFormula`; rejects `?` with
  "free parameters (?) need nonlinear.NARX_FormulaParser"; maxLag 8, maxTerms 40, exponents in [0.2, 3]) and
  `nonlinear/narx_formula_parser.py` -> `NARX_FormulaParser` (the free-parameter forms, `grammarLines`); the reject
  reasons are ScalaTion's strings ("lag 7 outside [1,6]", "wave(3) outside [1,2]", "interaction with 3 factors (at most two)").
- `Util/transform.py` -- `NormForm`, `MinMaxForm`, `CenterForm` (fitted on the training slice), `TransformT` enum with
  `form(x)` (what `TFORM: Log1p` names), `transform_t`; every `Transform` has the sklearn-like `transform` /
  `inverse_transform`.

Feature selection
- `modeling/feature_selection.py` (`SelectionTech`, `BestStep`, `updateQoF`, `qofVector`, `SelectionReport`, `FeatureSelection`),
  `modeling/subset_search.py` (`SubsetSearch`: forward, backward, stepwise, beam, cem, run), `modeling/cem_selector.py`
  (`CEM_Selector` with ScalaTion's hyper-parameters and a port of ScalaTion's MRG31k3p `Random`),
  `modeling/forecasting/feature_selection_forecaster.py` (`FeatureSelection_D`: `selectFeatures`, `selectFeaturesAtH`,
  `selectFeaturesPerH`, `reportSelection`, `getBest`, `cemProbs`, `smapeFPE`, `aggFPE`; `FeatureSelection_Reg` for
  recursive forecasters); `getModel(cols)` on ARX_D, ARX_SR_D, NARX_SR_D and `getModel_h(cols, h)`.  The aRX_DTest8
  CEM selection and subsetSearchTest logs are reproduced to the printed digits.

LLM-NARX-SR search layer (`modeling/forecasting/nonlinear/llm_narx_sr/`, ScalaTion rev4.0)
- Moved from `modeling/forecasting/llm_narx_sr/`; `eda.py` -> `eda4ts.py` (the full `EDA4TS`: ACF/PACF by Durbin-Levinson,
  periodogram, CCF, KPSS, prompt block, Ljung-Box / ACF-spike / collinearity / heteroscedasticity residual report).
- `formula_scorer.py`: all ScalaTion hp entries, the Scala method names (`qofFit`, `qofValidationAll`, `scoreOf`, `jOf`,
  `keyOf`, `fitFull`, ...; snake_case aliases kept), a 64-entry fit cache by canonical key, `score(mod, label)`, `notes`,
  `library` / `buildFixed` / `evalCols`; QoF dicts keyed by ScalaTion's names (rSq, rSqBar, mse0, mae, smape, m, dfr).
- `datasets.py`: `ExoMode`, `DatasetConfig.shortlist` (via `ExoScreen`), `COVID_ALL`, `covidFile()`, `fromFile`, `ettH1(rng)`;
  `Example_ILI` now loads `data/ILINetNov25.csv` (CDC ILINet, 1,208 weeks; response ILITOTAL, drivers %WEIGHTED ILI -> wili,
  TOTAL PATIENTS -> ot, NUM. OF PROVIDERS -> prov).
- New `term_library.py`, `search_arm.py`, `search_loop.py`, `genetic_proposer.py`, `llm_proposer.py`, `explainer.py`,
  `baseline_arm.py`, `pipeline_arm.py`, `plausibility_checker.py`, `significance.py`, `llm_narx_sr.py` and
  `modeling/llm_client.py` (OpenAI / Anthropic / local clients, disk cache, call counter, mock client; `urllib` only).
- Checks: GEFCom2014 baseline TnT sMAPE 12.783, hinge threshold 65.8 F, VIF 18.7 rejection; Covid-19 anchors 43.48 /
  30.72 / 45.71 as the ScalaTion runs.

Examples and tests
- `Examples/Example_Covid19.py` rewritten (one `*_test` per model, scenario names, torch imported lazily); new
  `Example_ILI.py`, `Example_ETTh1.py`, `Example_GEFCom.py`, `Example_FeatureSelection.py`; `Example_LLM_NARX_SR.py`
  gains baseline | genetic | random | mockllm | grid.
- 178 tests (was 67): fit, parsers, formulas, models, feature selection, scorer, datasets, search layer; all offline.
- `pyproject.toml` version 2.0.0, packages `modeling.forecasting.nonlinear(.llm_narx_sr)`.

Known differences from ScalaTion (documented in the module docstrings)
- `ARIMA` estimates by statsmodels' exact MLE, not ScalaTion's CSS/MLE optimizer.  The genetic / random proposers draw
  from numpy's `default_rng(stream)`; the CEM selector uses the ported MRG31k3p stream.  No plots.  `tnT_Test` does not
  print the forecast matrix unless `showYf=True`.

## 1.1.0 -- 2026-10-12: LLM-NARX-SR core layer, four datasets, optimizer options

New
- `modeling/forecasting/formula.py` -- the NARX formula terms (lags, functions, powers, products,
  waves, and the free-parameter terms shift / rate / pow / sat / hinge / trans / ema / dq / wave(?)),
  `Par` with roles and data-driven bounds, `Formula` (key, formula_str, proposal_str, columns, redundancy).
- `modeling/forecasting/formula_parser.py` -- `parse_formula` / `try_parse_formula` with reasons, `grammar_lines`.
- `modeling/forecasting/narx_sr_d.py` -- `NARX_SR_D(Forecaster_D)`: `from_formula` factory, ridge per
  horizon, free parameters by `optimizer="varpro" | "lbfgsb" | "joint"`, `n_free` degrees of freedom.
- `modeling/optimization/varpro.py` -- `VarPro` (reduced residual / objective, `fit(a0, optimizer)`), `ridge_solve`.
- `modeling/forecasting/llm_narx_sr/formula_scorer.py` -- `FormulaScorer` (hp, fit rows and validation blocks,
  `score` -> `Candidate`, rejection rules, `tnt`, `report`), `qof_table`.
- `modeling/forecasting/llm_narx_sr/datasets.py` -- `Example_Covid`, `Example_ILI`, `Example_ETTh1`,
  `Example_GEFCom`, `DatasetConfig`, `Datasets` (COVID2, COVID, ILI, ETTH1, ETTH1_1Y, GEF14).
- `modeling/forecasting/llm_narx_sr/eda.py` -- periodogram `top_freqs` for `wave(k)`.
- `data/ETTh1.csv` (original ETDataset file, md5 8381763947c85f4be6ac456c508460d6),
  `data/gefcom2014_load_daily.csv` (2,099 days) and `tools/build_gefcom_daily.py` that builds it.
- `tests/` -- 67 pytest tests (parser, model + optimizers, scorer, datasets, ARX_SR / transforms); `Examples/Example_LLM_NARX_SR.py`.
- `pyproject.toml` (`pip install -e .`), `__init__.py` files, `requirements.txt` re-saved as UTF-8 (it was UTF-16).

Changed, for consistency with the current ScalaTion code
- `ARX_Symb` / `ARX_Symb_D` are now `ARX_SR` / `ARX_SR_D` (`modeling/forecasting/arx_sr.py`, `arx_sr_d.py`),
  ScalaTion's names.  Transformations are `Transform` objects from the new `Util/transform.py` (PowForm,
  RootForm, RecipForm, PowRForm, LogForm, Log1pForm, ExpForm, NExpForm, DQuadForm, DIQuadForm, BoxcoxForm,
  SinForm, CosForm, with ScalaTion's default parameters) instead of bare lambdas; `fEndo` defaults to
  `[PowForm()]`; `fExo` is one list per exogenous variable (an error otherwise, as the `require` in Scala).
- `rescale` scales y and the exogenous variables FIRST and applies the transformations to the scaled series
  (ScalaTion's current design); the transformed lags of forecasts are the transforms of the forecasts, so
  `_scaleCorrection` and the extra `tForm_endo` / `tForm_exo` scaling of transformed columns are gone.
- Column order and labels follow `ARX_SR.buildMatrix` / `formNames`: trend, y lags, f_i(y) lags, per exogenous
  variable its raw then transformed lags, then the cross terms (`model.fname`).
- All `rescale` factories (`ARX`, `ARX_D`, `ARX_SR`, `ARX_SR_D`) fit their scalers on the TRAINING rows only
  (`Util.build_matrix.fit_scaler`), as ScalaTion's `Model.trSize` split does; in-sample numbers of rescaled
  models move slightly.  `StandardScaler` turns the non-negativity constraint off (NormForm).
- `Examples/Example_Covid19.py`: `ARX_SR_test` / `ARX_SR_D_test` replace the `ARX_Symb*` tests; the `NN_3L_test`
  that imported the removed `modeling.neuralnet.neuralnet_3l` is dropped.

Unchanged
- `AR_YW`, `ARX`, `ARX_D` (apart from the scaler rows above), `MHAttn_Regressor`, `Util.tools`.
  `NARX_SR_D` on a plain-lag formula with `lam=0` reproduces `ARX_D`; `ARX_SR_D` (OLS) and `NARX_SR_D` on the
  equivalent formula give the same forecasts (tested).

Known
- `data/national_illness_clip.csv` (ILI) is not shipped; `Example_ILI` says where it goes.
- sklearn's `LinearRegression` (method "sk_lr") truncates the rank of an ill-conditioned design matrix
  (e.g., y with y^1.5 and log(1 + xe) with xe); "sm_ols" is exact least squares, as ScalaTion's `Regression`.
