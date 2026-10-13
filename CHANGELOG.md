# Changelog

## 2.6.0 (ForecasterPy_rev1.6) -- 2026-10-13: fixes found by the ScalaTion rev4.1 logs

Yousef's rev4.1 runs (Oct 13: `datasetsTest`, `datasetsTest2`, `eDA4TSTest`, `formulaScorerTest`, `lLMProposerTest`,
`baselineArmTest`, `lLM_NARX_SRTest5`) compared with the Python port, number by number.  After these fixes the
deterministic results are ScalaTion's to the printed digit: the dataset configurations and the screening, the
exact-linear `formulaScorerTest` (sMAPE per horizon 0.212800 0.202128 0.193098, J 0.236448332249), the four
deterministic arms of `baselineArmTest` (Covid-19 icu/pRate) and of `lLM_NARX_SRTest5` (auto exo: exMort, cases),
including sel-arx's equation, and the EDA block of the LLM prompt.  225 tests pass.

Fixes
- `FormulaScorer` scored horizon h on the rows t <= t2 - 1 of a range (targets y(t + h - 1) up to h - 1 rows PAST
  the range, i.e., in-sample QoF reaching into the validation rows, a validation block into the next block);
  ScalaTion's `diagnoseH` keeps every target inside the range (t <= t2 - h, m_h = m_1 - (h - 1)).  In-sample,
  validation and selection scores change for h >= 2 (TnT does not); e.g., baseline 28.948 / 20.13 / 18.06 ->
  29.456 / 18.59 / 18.07, and sel-arx now selects ScalaTion's y~1 + icu~5 + icu~3 + pRate~6 + pRate~4 + pRate~1.
- The KPSS p-value is ScalaTion's: NEW `forecasting/stationarity.py` (`UnitRoot`) and
  `forecasting/stationarity_kpss.py` (`Stationarity_KPSS`, the URT response surfaces at 15 probabilities) replace
  statsmodels' `kpss` in `EDA4TS`, whose p-value is clipped to [0.01, 0.10] (Covid-19: 0.10 -> 0.43, as the rev4.1
  prompt shows).  The statistic was already the same.
- `acfPacf`: pacf[0] = psi(0, 0) = 0, as ScalaTion prints it (was 1).
- `fmt_g` (Java's "%g"): no trailing decimal point (728386 was printed 728386.).
- The parsers keep the dataset's variable order in "variable 'x9' not in dataset {y,icu,pRate}" (was sorted).

Tests
- NEW `test_exact_linear_series` (formulaScorerTest), `test_auto_exo_anchors_match_scalation` (lLM_NARX_SRTest5),
  `test_kpss_is_scalations`; `test_anchors_match_scalation` asserts the baselineArmTest numbers;
  `test_residual_report_fails_a_wrong_model` uses ARX_D (p = 1, px = 0, spec = 0) as rev4.1's eDA4TSTest.

Examples: Covid, ILI, GEFCom, ETTh1, FeatureSelection print what rev1.5 printed (apart from 6-digit integers in
`fmt_g`); LLM_NARX_SR's in-sample / validation / score columns change (the first fix), its TnT columns do not.

Still different from ScalaTion (by design or not yet decided)
- The scorer builds every formula as `NARX_SR_D` (ScalaTion: `ARX_D` when it has no free parameter), so banners
  read "NARX_SR_D(6, 6, 2)" where ScalaTion prints "ARX_D_6_6_2" (the fits are the same).
- The nonlinear optimizer is scipy's `least_squares` (ScalaTion: its `LevenbergMarquardt_TR`): fitted parameters
  agree to about 3 digits on flat objectives (lLMProposerTest: 1.947 / 7.702 vs 1.9475 / 7.6982).
- The validation-block refits run on copies of the formula (ScalaTion refits the candidate's own parameters, so its
  formula then shows the LAST block's values: lLMProposerTest's 2.012 / 4.742).
- `UnitRoot.setTrend` does not print ScalaTion's "setTrend: trend = c, npar = 2" line.

## 2.5.0 (ForecasterPy_rev1.5) -- 2026-10-13: datasets and EDA4TS at the forecasting level; loaders as ScalaTion's

The data loaders, the dataset configurations and the EDA profile move out of the LLM package to `forecasting`, as in
ScalaTion rev4.1 (whose `Datasets.scala` and `EDA4TS.scala` move to `forecasting/`); the loaders take ScalaTion's
API.  Every example prints the same output as rev1.4 (and rev1.3).  222 tests pass.

Files
- `modeling/forecasting/datasets.py` (was `nonlinear/llm_narx_sr/datasets.py`): the four loaders, `ExoMode`,
  `DatasetConfig`, `Datasets`; one file on purpose (ScalaTion keeps each loader object in its `Example_*.scala` with
  its test mains, which are the scenarios of `Examples/Example_*.py` here).  `DATA_DIR` still `<repo>/data`.
- `modeling/forecasting/eda4ts.py` (was `nonlinear/llm_narx_sr/eda4ts.py`): `EDA4TS` (the profile, `residualReport`),
  `amplitudes`, `topFreqs`, `periods`.
- NEW `Util/correlogram.py` (ScalaTion's `mathstat/Correlogram.scala` + the `VectorD` statistics): `acov`, `acorr`,
  `ccorr`, `corr`, `durbinLevinson`, `acfPacf`, `ljungBox`, `acfSpikes`, `collinearPair`, `heteroscedastic`;
  `exo_screen.py` uses its `corr` / `ccorr` (its duplicates removed; `corr_matrix` stays).

The loaders follow ScalaTion's objects
- `loadData (x_strs, y_str, trim | rng)` returns (xe, y) (was (y, xe)); ILI / ETTh1 / GEFCom take `y_str`
  explicitly, as ScalaTion's; `loadData_y`; `loadData_yy` (Covid, ILI); `context (x_strs, names, y_str, ...)` with
  names defaulting to x_strs (ETTh1: `LOAD_NAMES`).
- Covid: `loadData` reads the whole file (169 weeks) and `clip ((xe, y), rng = 0 until 116)` keeps the 116 weeks the
  experiments use (`xe, y = clip (loadData (BEST2_EXO))`; `context` clips by default); `NO_EXO`, `BEST_EXO`,
  `BEST2_EXO`.  Members renamed to ScalaTion's: `fileName`, `header`, `response` (were `FILE`, `RESPONSE`, `HEADER`).
- Removed Python-only members: `frame`, `N_ROWS`, `SHORT` (short names are passed to `context`), ILI's `EXO` /
  `NAMES`, ETTh1's `ONE_YEAR` (use `range (0, 8760)`); kept `Example_GEFCom.v_table` (a Python helper).
- `loadH (fileName, skipCol)` = ScalaTion's `MatrixD.loadH` (used by `Datasets.fromFile`).

Configurations and the LLM layer
- `DatasetConfig.shortlist (data, trSize, k, maxLag)` is generic; `FormulaScorer.shortlist (cfg)` passes the scorer's
  data, training rows, hp["exoK"] and hp["maxLag"] (callers: `scorer.shortlist (cfg)` in BaselineArm, PipelineArm,
  the LLM_NARX_SR harness).  `DatasetConfig.expectedSigns` is the field (was `expected_signs` + a property).
- `Datasets.COVID_FILE` is a lazy class attribute (ScalaTion's `lazy val`; was `covidFile ()`).
- `LLMProposer.promptBlock (eda, names)` (was `EDA4TS.promptBlock`): the EDA profile in words for a prompt.
- `runner.loadData (args)` returns (xe, y) as the loaders do.

Fix found on the way
- `acorr` defaults to adjusted = True, as ScalaTion's `VectorD.acorr`: `FormulaScorer.notes` prints the residual ACF
  of `e.acorr (l)` as ScalaTion's (the Python default was the MLE form, e.g., lag 2 0.07 -> 0.08 on Covid);
  `ljungBox` / `acfSpikes` keep the MLE form explicitly, as ScalaTion's.  `ljungBox` returns p = 0 beyond
  Q = 100 + 3 df, ScalaTion's shortcut (printed the same).

## 2.4.0 (ForecasterPy_rev1.4) -- 2026-10-13: the settings in one checked `Args`; hparam private

Every model is built as `Model (y, hh, [xe,] args)`: the data in the call, every setting in one `Args` -- the
dotdict style of experiment scripts, but checked.  The hyper-parameters stay ScalaTion's (`AR.hp`, the REPORT line,
`hparameter`); the results and the printed output are unchanged (every example scenario prints the same text as
rev1.3; the Scala-log reproductions still pass).  222 tests pass.

`Args` (new `modeling/forecasting/args.py`)
- One table of every setting (name, default, check, meaning): ScalaTion's `AR.hp` (p, d, q, P, D, Q, s, px, spec, crx,
  nneg), the Python-only `skip` and `lam` (ScalaTion's `RidgeRegression.hp("lambda")`; "lambda" is a Python keyword,
  accepted as an alias), the model settings `method`, `scaled`, `tFormT`, `optimizer`, `tune`, `te_ratio` (ScalaTion's
  `TE_RATIO`, here per model), and the run settings `dataset`, `exo`, `model`, `hh`, `validation`, `rc`.
- Checked when set: an unknown name raises AttributeError with the closest names or a hint ("did you mean 'px'?",
  "training_ratio" -> te_ratio), a bad value ValueError ("Args.p = -1: must be an integer >= 0"); values are
  normalized (bools for the switches, `TransformT` / optimizer / model / dataset names in any case).
- `Args (p=6, px=4)` or attribute by attribute; `copy (**changes)`, `update`, `reset`, `del args.p`, `changed`,
  `toDict`, `save` / `load` (JSON presets), `hparam ()` (the HyperParameter of a model), `help ()` / `describe ()`
  (the table; `Args.help (ARX_D)` lists the settings ARX_D reads, its class attribute `ARGS`), tab completion.

The models
- New argument order (ScalaTion's: data first, hparam optional last): `RandomWalk (y, hh, args)`, `AR (y, hh, args)`,
  `ARIMA (y, hh, args, trend)`, `ARX (y, hh, xe, args)`, `ARX_D (y, hh, xe, args)`, `ARX_SR (y, hh, xe, args, fEndo,
  fExo)`, `ARX_SR_D` likewise, `NARX_SR_D (y, hh, X, args, ...)`; `rescale (y, hh, xe, args, ...)` (args.tFormT,
  MinMax by default), `ARX.trend (y, hh, xe, args)`, `fromTerms (formula, dataset, hh, args, trSize)`,
  `NARX_SR_D.apply` / `rescale` (args.tune, scaled, optimizer), `ARX_Formula.fit` / `fit_D` and
  `NARX_Formula.fitNL_D (dataset, hh, args, trSize)`.  The keyword arguments `method`, `scaled`, `tFormT`,
  `optimizer`, `tune`, `fit_params` and `hparam` are gone (all in args).  The old order raises a TypeError that says so.
- hparam is private: a model keeps a snapshot of its args and its `HyperParameter`; `mod.hparam` is gone and
  `mod.hparameter` (ScalaTion's name) returns a COPY.  The REPORT prints the Scala hyper-parameters of args as
  ScalaTion prints the hp it is given.
- `AR.hp` is read-only (`HyperParameter.freeze`; `copy` is writable), so a model or script cannot change the
  defaults of the models made after it.
- Each model checks args.method against its own methods (`METHODS`, default `METHOD`): ARX* "sm_ols" | "sk_lr" |
  "ridge"; AR "adjusted" | "mle" | "sm_ols"; ARIMA the statsmodels methods; NARX_SR_D always ridge.  Settings a model
  does not read are ignored, so one args drives several models.  Sub-models get the parent's snapshot with its
  resolved method (`_subArgs`).
- `te_ratio` sets a model's train/test split (`tr_size`, the rescaling statistics, the TnT rows, the FS rows).
- `FormulaScorer.args (**changes)` (ScalaTion's `hparam (nvs*)`) builds the Args of the models it fits.

The runner (new `modeling/forecasting/runner.py`)
- `run (args)`: loads args.dataset (Covid-19, ILI, ETTh1, GEFCom2014) with args.exo, builds args.model (rescaled when
  args.tFormT is set), runs the In-Sample test, the TnT test or both, returns the model; `loadData`, `makeModel` on
  your own data.  The models never load data, make folders or plot.

Examples, tests, docs
- Every example and test uses `Args`; `Example_Covid.py run` shows the one-object style (`run (args)`), ETTh1 a preset
  (`BASE = Args (nneg=False)`, `BASE.copy (...)`).  New tests/test_args.py (29 tests).
- README ("Settings: Args"), docs/CONVENTIONS.md, headers Version 1.4.

## 2.3.0 (ForecasterPy_rev1.3) -- 2026-10-13: the structure of ScalaTion where it fits Python

Consistency with ScalaTion's latest code wherever it is possible; Python-only choices are kept on purpose
(`_make_model` with "sm_ols" / "sk_lr" / "ridge", one solve per horizon instead of an MV class, constructors
that build their matrix, range-based `forecast`).  192 tests pass with `-W error::RuntimeWarning`.

The regression lives in the bases (ScalaTion's `reg`)
- `Forecaster._make_model` (shared) and `Forecaster_D.train` / `_predict` fit every direct model (moved from `ARX_D`);
  `Forecaster_D` exposes `slopes` (original units), `bb` / `parameter` (the regression's own coefficients:
  standardized when "ridge" is scaled, as ScalaTion's `reg.parameter`) and `intercept`; `Forecaster_Reg` likewise.
- `getModel_h` is implemented in `ARX_D` and abstract in `Forecaster_D` (was the reverse); `ARX_D.getModel` names its
  sub-model `ARX_D_p_px_n` and is inherited by `ARX_SR_D` (its own `getModel` removed) and `NARX_SR_D`.
- `NARX_SR_D` extends `ARX_D` (its own train / forecast / getModel / getModel_h copies removed).
- Fix: a sub-model (`getModel`, `getModel_h`) keeps its own default skip, as a new ScalaTion model does, instead of
  the parent's current skip (0 after a `tnT_Test`): Backward selection after TnT now reproduces nARX_SR_DTest5's
  columns (0, 1, 6, 12, 13, 28), its trace and the selected model's coefficients and TnT sMAPE (15.9482 .. 50.1970).

Hyper-parameters
- New `modeling/hyper_parameter.py`: `HyperParameter` (ScalaTion's; a dict of values that keeps each registered
  parameter's default).  `modeling.forecasting.forecaster.hp` is ScalaTion's `AR.hp` (p 1, d 1, q 1, P 1, D 1, Q 1,
  s 7, px 1, spec 1, crx 0, nneg 1; also `AR.hp`); every model completes its hparam from it (Python defaults were
  p 6, px 0); `crx` / `nneg` are 0 / 1.
- The REPORT prints the hyper-parameters as the Scala logs: "HyperParameter(p -> (6,1), ..., nneg -> (1,1))"
  (RandomWalk: "null"); `Util.tools.report` / `hparam_str` moved into `Forecaster.report`.
- `lambda` is read from hparam everywhere, default `CenteredRidge.hp["lambda"]` = 0.1 (ScalaTion's
  `RidgeRegression.hp`; Python used 1.0); `NARX_SR_D.fromTerms` loses `lam` (pass {"lambda": ...}) and `nneg` (hparam);
  `scaled` is a constructor / factory keyword everywhere (was hparam["scaled"] for ARX / ARX_D);
  `ARX_Formula.fit` / `fit_D` and `NARX_Formula.fitNL_D` take `hparam`.
- `CenteredRidge`: `hp`, `ridge (hparam)`, `parameter` in the fitted space, `slopes` / `intercept` in original units;
  the standardization uses the unbiased standard deviation (also in `VarPro`), as ScalaTion.

Rescaling by a `TransformT` (ScalaTion's `rescale (..., tFormT)`)
- `ARX`, `ARX_D`, `ARX_SR`, `ARX_SR_D`: `rescale (hparam, y, hh, xe, [fEndo, fExo,] method, tFormT=TransformT.MinMax,
  scaled)` replaces the sklearn `tForm=` classes; `AR.rescale` standardizes (NormForm) as ScalaTion's.
- `Util/transform.py`: `NormForm` / `MinMaxForm` / `CenterForm` fitted column by column; `MinMaxForm` maps to
  ScalaTion's centered [-2, 2] (was (0, 4)), `setLU` changes the range; robust `NormForm` (eps 1e-8);
  `TransformT.form (w)` builds a parameterized transform at w = (w0, w1); `TransformT.wlu` (ScalaTion's `TransformS`).
- `Util.make_matrix4ts.centered` (ScalaTion's `MakeMatrix4TS.centered`): a response transform into a space where y
  can be negative turns the rectification off (applied by every model; ScalaTion applies it in `ARX_D`);
  `fit_scaler` removed.  The SR examples rescale by Log1p as ScalaTion's tests (ETTh1: sign-safe `PowRForm`).
- `buildMatrix (xe_bfill, y, hparam, [fEndo, fExo])` and `formNames (n_exo, hparam, ...)` take ScalaTion's arguments.

New
- `summary ()` (`Forecaster_Reg`, `Forecaster_D`) and `summary (h)` (direct): ScalaTion's SUMMARY block
  (`Fit.summary`, `showSummary`, `sumCoeff`: estimates, standard errors, t, p-values, VIFs; `modeling.fit.vif` =
  `Predictor.vif`); `Util.tools.jdouble` (Java's `Double.toString`), `fmt` prints NaN as "NaN".
- `Forecaster.forecastAt (h)`; `ARX.trend (hparam, y, hh)` (the trend-only model; the matrix holds the trend
  columns, ScalaTion's passes the lags too while its hparam says p = 0).
- `NARX_SR_D.apply` / `rescale (hparam, y, hh, xe, fEndo = [TransformT.Root], fExo = None => Pow, tFormT, lu = (1, 5),
  tune)` with `buildMatrix`, `optimize`, `initializeW`, `tunable`, `getTransforms`, `exoSets`, `allForms`,
  `formNames`, `ridge` (ScalaTion's companion): reproduces nARX_SR_DTest5 (defaults: the REPORT and QoF tables
  digit for digit; fitted: In-Sample rSq 0.954374) and nARX_SR_DTest8 (VarPro to (10, 1.99494, 0, 1, 0, 1), TnT
  sMAPE 14.1927 ..).  `Example_Covid` scenarios `narx_sr` and `optimizers`; `aRX_SR_D_test` prints the SUMMARY.

Imports
- Every import is at the top of its module except the two genuine cycles (`NARX_Formula.fitNL_D` -> `NARX_SR_D`,
  `FormulaScorer.notes` -> `Explainer`) and the optional torch / matplotlib; `tests/test_base_api.py` checks it.
  Unused imports removed (pyflakes clean).

## 2.2.0 (ForecasterPy_rev1.2) -- 2026-10-13: one name per thing, the ScalaTion one

Every misleading or duplicated name is gone; the Python names are the ScalaTion names (files, classes,
methods, properties, hyper-parameter keys).  Behaviour, printed output and the reproduced ScalaTion
numbers are unchanged; 180 tests pass with `-W error::RuntimeWarning`.

Files
- `modeling/regression.py` -> `modeling/centered_ridge.py` (it holds `CenteredRidge` only: `fit`, `predict`,
  `slopes`, `intercept`, `parameter`, `params`).
- `Util/build_matrix.py` -> `Util/make_matrix4ts.py` (ScalaTion's `MakeMatrix4TS`): `makeMatrix4L`, `makeMatrix4T`,
  `makeMatrix4Y`, `backcast`, `backfill`, `backfill_matrix`, `trend_names`, `fit_scaler`; `tr_size_of` removed
  (use `modeling.forecasting.forecaster.trSize`).
- `modeling/forecasting/ar_yw.py` removed (`AR` is the class; no `AR_YW` alias).
- `Examples/Example_Covid19.py` -> `Examples/Example_Covid.py`; `tests/test_arx_sr.py` -> `tests/test_arx_family.py`.

Hyper-parameters
- The constructor argument is `hparam` everywhere (was `args`), the completed dict is `mod.hparam` (= `hparameter`),
  with ScalaTion's keys: `px` (was "q": the exogenous lags) and `crx` (was "cross"); `ARIMA` keeps `p`, `d`, `q`.
  `ARX_Formula.lagHP()` returns `{"p", "px", "spec", "n_exo"}`.  `RandomWalk(hparam, y, hh)` takes the dict like
  every other model (`RandomWalk(None, y, hh)` => skip 0).

Models
- `Forecaster`: `makeForecastMatrix` (was `initialize_forecast_matrix`), `TE_RATIO` (the `_TE_RATIO` duplicate removed),
  `trainNtest_Test` removed (use `inSample_Test`), `params` documented (prefer `parameter` / `intercept` / `bb`).
- `Forecaster_D`: `diagnose_all` removed (`diagnoseAll`); `fit_rows` kept (Python-only helper).
- `ARX`: `buildMatrix` (was `build_matrix`), `formNames`, `forge`, `hide` (were `_forge` / `_hide`); `ARX_D.from_formula`
  removed (`fromTerms(formula, dataset, hh, ..., hparam=...)`).
- `Util/tools.py` keeps the ScalaTion printing helpers only (`banner`, `fmt`, `fmt_g`, `vector_str`, `matrix_str`,
  `array_str`, `set_str`, `hparam_str`, `report`); the legacy `mse` / `mae` / `r2` / `smape` / `diagnose` / `fit_map`
  wrappers and `modeling.fit.help()` are removed (`Fit.help`, `modeling.fit.diagnose`).

Nonlinear parameters (ScalaTion's `NARX_SR_D.scala`)
- `OptimizerNL` enum (`VarPro`, `LBFGS_B`, `Joint`) and the module default `OPTIMIZER_NL` live in
  `modeling/forecasting/nonlinear/narx_sr_d.py`, as in Scala; `NARX_SR_D.fromTerms(..., optimizer=None)` takes the
  enum, its name ("lbfgsb") or None (the module default); `NARX_SR_D.OPTIMIZERS` and the strings "varpro" / "lbfgsb" /
  "joint" are gone.
- `NARX_SR_D.fitNL(basis, yy, a0, lb, ub, lam, scaled, optimizer)` -> `(a, stepCounts)`, with the `lbfgs_b` and `joint`
  helpers (ScalaTion's `fitNL` / `joint`); `VarPro` keeps `optimize` only (`optimize_lbfgsb`, `optimize_joint`, `fit`
  removed) and reports `stepCounts` (was `steps`); `narx_sr_d.DEBUG = True` prints the `fitNL` debug line (a,
  objective, steps).
- `NARX_SR_D`: `nFree` (the attribute `n_free` is private; the constructor keyword `n_free` is Scala's),
  `stepCounts` (was `steps`), `optimizer` is an `OptimizerNL` or None (was "none"), `formulaStr` only.

Formulas and the search layer
- `FormulaVar.lagged(k)` (was `lag_`) beside `lags(h, l)`; the short aliases `Var`, `Transformed`, `Power`, `Product`,
  `Wave`, `Shift`, `Scale`, `ParPower`, `Sat`, `Hinge`, `Trans`, `Ema`, `Dq`, `ParWave`, `Formula`, `pow_`, `START`
  and the attribute aliases `fn` / `ph` / `f` are removed (the `*Term` classes, `LabelMode.SLOT_VALUE`).
- Only the ScalaTion names remain: `formulaStr`, `proposalStr`, `nTerms`, `maxLag`, `nParams`, `nFree`, `lagHP`,
  `matrix` (the snake_case `formula_str`, `proposal_str`, `n_terms`, `max_lag`, `n_params`, `n_free`, `lag_hp`,
  `columns` are removed); `FormulaParser.parseFormula` / `tryParseFormula` and `NARX_FormulaParser.parseFormula` /
  `tryParseFormula` / `grammarLines` (the module-level `parse_formula`, `try_parse_formula`, `grammar_lines` are removed).
- `FormulaScorer`: attributes `trSize`, `teSize`, `trRng`, `teRng`, `fitSize`, `fitRng`, `valRng`, `nVal`, `valFrac`,
  `valFolds`, `hSel`, `maxFree`, `boundTol`, `maxVIF`, `FormulaScorer.hp` (the snake_case attributes, the duplicate
  properties and the method aliases `fit_full`, `qof_train`, `qof_fit`, `qof_validation(_all)`, `qof_by_h`,
  `diagnose_h`, `qof_tnt`, `j_of`, `score_of` are removed); `DEFAULT_HP["optimizer"]` is None (=> `OPTIMIZER_NL`).
- `Candidate(formula, score, qofTrain, nTerms, qofVal, reason, label, extraDF, model)` -- Scala's field names (were
  `qof_train`, `n_terms`, `qof_val`, `extra_df`); the aliases `formula_str`, `proposal_str`, `n_free`, `smape_h`,
  `qof_v`, `smape_val`, `score_h` are removed.
- `eda4ts.topFreqs` only (`top_freqs` removed); `DatasetConfig.baselineFormulaStr`, `blindNames` only;
  `Example_Covid.loadData`, `Example_ILI.loadData` / `loadData_y`, `Example_ETTh1.loadData`, `Example_GEFCom.loadData`
  (were `load_data*`).
- `docs/CONVENTIONS.md` restated: file names after the class they hold, ScalaTion's camelCase for everything users
  touch, no aliases, snake_case only for Python-only helpers.

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
- No numpy RuntimeWarnings any more: `ARX_Formula.matrix` builds the columns under `np.errstate` (a non-finite term
  is reported once by its check) and every `Transform.f` / `fi` runs quietly (a log of 0 or an exp of a huge
  forecast returns inf / nan for the caller's finiteness check); `pytest -W error::RuntimeWarning` passes.
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
