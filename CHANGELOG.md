# Changelog

## 3.2.0 (ForecasterPy_rev2.2) -- 2026-10-20: consistent with ScalaTion rev5.3 (the knobs), plus the measurement protocol

ScalaTion rev5.3 generalized the penalty search to the KNOBS of whatever regression is bound, moved the `llm_narx_sr`
engine onto the platform's switch and renamed the second switch; this release does the same in Python and adds the
measurement protocol the roadmap's Stage 1 asked for.

1. The knobs.  `Knob (name, grid, log, tag)` in `modeling/hyper_parameter.py` with `tagOf` (`_L1.0`, `_L0.01`,
   `_L1.0E-12`, `_A0.5`, `_A0.25`: ScalaTion's tags, digit for digit), `toCoord` / `fromCoord`; `Knob.LAMBDA`
   (1e-3 .. 1e3 by decades, log) and `Knob.L1_RATIO` (0 .. 1 by quarters, linear); the regressions declare theirs:
   `CenteredRidge.tunable = (LAMBDA,)`, `ElasticNet.tunable = (LAMBDA, L1_RATIO)`.
2. `Forecaster.knobs` (by the model's method), `penalized` (= knobs present), `hpAll`, `hpTag`, `withHP (hp)` (was
   `withLambda (lam)`), `tune (knobs, measure, hs, verbose, search = "grid" | "golden" | "random", folds, trials, seed)
   -> ({name: value}, score)` (was `findLambda`): the grid is the product of the knobs' grids (7 for the ridge, 35 for
   the elastic net), golden sweeps each knob in its coordinate between the neighbours of the best grid value, random
   draws from one seeded generator in the knobs' box; `bestHP (cols)` (was `bestLambda`).  `Forecaster.LAMBDAS`,
   `lamTag`, `_setLambda` are gone.  The names: `ARX_D_6_6_2_L1.0` under the ridge, `ARX_D_6_6_2_L1.0_A0.5` under the
   elastic net, and NO tag for a plain least-squares model (was `_L0.0`: no knobs, no tag) or a model without a regression.
3. `Model.tuning` (was `tuneLambda`) and `Model.folds` (0 => the validation block, k => `cvScore (k)`: `tune`'s default);
   `withValidation (ratio = None, tuning = False)` always sets `tuning`, as ScalaTion's does.  `evalCols` / `fitCols`
   tune each candidate set by `withHP (tune ()[0])` under `Model.tuning`; `_subModel` and `selectedModel` apply `bestHP`.
4. `FormulaScorer`: `tuning` (was `tuneLambda`), `build (f, size, fit_params, hp)`, `fresh (f)`, `tuned (mod)`,
   `fitFull (f, hp)`, `tnt (f, hp)`, `Candidate.hp` (a dict; `lam` is a property of it); and two rules now shared with
   ScalaTion rev5.3: (a) in-sample a candidate is fitted AND diagnosed on the FS rows of its block (`fs_rng (rows)` =
   rows.stop - hh + 1, the boundary rule, exactly as `Forecaster_D.fitCols`) -- the in-sample numbers moved (Covid
   baseline arm 17.048, was 18.117; the validation and TnT numbers did not); (b) `scoreH` and the validation pass
   roll a FRESH copy at the fit's knobs, never the cached model.  `toModel` / `record` / `showPareto` build the final
   model at the candidate's knobs.  `Fit.resetDF_k` warns once per model (hundreds of identical lines before).
5. The measurement protocol (roadmap 2.2; Python first).  `Forecaster.errorTable (rng, scale)`: after `rollTest`,
   one row per origin and horizon whose target lies in the block, in original units -- y, yhat, e, ae, sape (a
   horizon's mean sape IS its sMAPE), ase (|e| / the naive one-step MAE of the rows before the block: MASE's
   denominator; the QoF table's `mase` scales by the block).  `llm_narx_sr/protocol.py`: `Protocol (te_ratio = 0.4,
   rc, hh, loss)` with `runModels` (named factories) and `runArms` (the harness's arms, each best model rolled over
   the testing rows at its knobs) appending to one long table; `summary ()` (sMAPE and MASE per dataset, model and
   horizon and over all, with a 95 % HAC interval: Newey-West, Bartlett weights, lag h - 1); `compare (A, B)` on the
   paired per-origin losses: the Diebold-Mariano test with the Harvey-Leybourne-Newbold correction
   (`significance.dm_test`, `hacVar`) next to the Wilcoxon signed-rank test; `compareAll`, `report`, `write` (CSV).
   `Examples/Example_Protocol.py [covid|ili|gef14|etth1|all] [--arms]` writes the "before" tables of the base
   models and the deterministic arms (`results/protocol/`): on Covid-19's 47 origins the ridge `ARX_D` beats the
   random walk (DM* -3.7, p < 0.001 pooled over the horizons), `sel-arx` is worse than the random walk at h = 1
   (DM* +4.3) -- the selection variance, quantified.
6. Examples: `Example_Covid.py arx_d2` (tune: grid, golden, random, a wider block, 5-fold; selection with
   `Model.tuning`), `enet` (the ridge's one knob and the elastic net's two by the same `tune`, the lasso at a fixed
   share); docs: README (the knobs, the protocol), `docs/CONVENTIONS.md` (the table "ScalaTion rev5.3 <-> Python
   rev2.2: the knobs").
7. Tests: `tests/test_validation_tune.py` (was `test_validation_find_lambda.py`: the knobs and tags, `tune` grid /
   golden / random / k-fold, the elastic net's two knobs, the tuning switch, the scorer), `tests/test_protocol.py`
   (the error table against the QoF table, HAC and DM, the runner, summary, comparisons, CSV); the `_L0.0` pins
   dropped; the in-sample pins of the scorer and the anchors re-pinned; 261 tests pass.

## 3.1.0 (ForecasterPy_rev2.1) -- 2026-10-19: consistent with ScalaTion rev5.2 (the protocol)

ScalaTion rev5.2 gave the tree the switch, ONE rolling routine, the penalty search and the boundary rule; this
release makes the names and the numbers agree again.

1. `Forecaster.valTest (rc, showYf, showParams, verbose)` is the validation pass (was `validate (rc, verbose)`):
   ScalaTion's name, ScalaTion's signature next to `tnT_Test`'s.  `tvT_Test` and `valScore` go through it.
2. The boundary rule (ScalaTion rev5.2's `trainWindow`): a rolling test retrains only on rows whose targets lie
   before the forecast origin.  `Forecaster._trainWindow (is_, t)` -> (targets, X rows) replaces the
   `_train_target` hook; `Forecaster_D._trainWindow` returns the rows `[is_, t - hh + 1)` (the old `[is_, t)` had
   hh - 1 rows with targets inside the block rolled over); the single-horizon model at horizon h (`getModel_h`)
   carries `_lead = h - 1` and stops that many rows early (its split is computed on `mSeries = len (y) + lead`,
   the series length, so its blocks are its parent's, cut at its last row with a target).  `fs_rng` of both
   bases and `cvScore`'s rows are cut the same way.  EVERY rolling number of a direct model moved a little
   (honestly so); the anchors are re-pinned (Covid baseline arm: validation 23.217, TnT 47.684, were 20.030 /
   43.483; `nARX_SR_DTest5` TnT sMAPE 22.8084 .., was 17.8772 ..; GEFCom's 21-lag baseline 12.798, was 12.783 --
   a long series barely moves).  The recursive models are unchanged.
3. `Model.TE_RATIO = 0` is allowed: no testing set (a competition's dataset), `teSize = 0`, `trSize = m`, the
   chooser works on the last `VAL_RATIO` of the dataset (`valTest`, `findLambda` under `withValidation`) and the
   model entered is fit on all the rows; `tnT_Test` prints the flaw and returns None, `tvT_Test` reports NaN for
   the test.  `args.te_ratio` defaults to None = `Model.TE_RATIO` when the model is built (it was a copy of 0.2
   taken at import time, so `Model.TE_RATIO = r` never reached a model); `Args (te_ratio = 0)` per model; the
   `TE_RATIO` constant of `args.py` and `Forecaster.TE_RATIO` are gone (one name: `Model.TE_RATIO`).
4. `Model.fitRows (m, te_ratio, trSize)` replaces `arx.fitRows` (ScalaTion's `Model.fitRows`).
5. `findLambda (search = "golden")` replaces `"brent"`: the grid, then a golden-section search in log10 lambda
   between the neighbours of the best grid value (tolerance 0.02 decades, at most 40 evaluations, memoised) --
   the one algorithm of both code bases; scipy's `minimize_scalar` is no longer imported by `forecaster.py`.
6. Docs: `docs/CONVENTIONS.md` gets the table "ScalaTion rev5.2 <-> Python rev2.1: the protocol" and the boundary
   rule under the regression paragraph; README: `te_ratio`, TE_RATIO = 0, the rule; the Covid example prints the
   golden-section lambda.
7. Tests: `tests/test_rev21_protocol.py` (9 tests: the direct window `[0, 87)` on Covid and its fit, the moved TnT
   numbers, `_lead` of `getModel_h`, the recursive models unchanged, `valTest`'s signature, `Model.fitRows`,
   TE_RATIO = 0 end to end, a model keeps the split it was built with, golden); the pins of the arms, the
   NARX_SR_D TnT and the FS rows re-pinned; 256 tests pass.

## 3.0.0 (ForecasterPy_rev2.0) -- 2026-10-17: consistent with ScalaTion rev5.0, argument for argument

A major version: the factory signatures change (the Python code was ahead of ScalaTion in places where it should
not have been).  Base: ScalaTion rev5.0 (`scalation_beforeElNet.zip`) and its rev5.1 (the elastic net).

1. `rescale` has no `trSize`: `ARX.rescale (y, hh, xe, args, tFormT = MinMax)`, `ARX_D.rescale (...)`,
   `ARX_SR.rescale (y, hh, xe, args, fEndo, fExo, tFormT)`, `ARX_SR_D.rescale (...)`.  The rows the transform is
   fitted on come from `Model`, as in ScalaTion (`Model.trSize`; `Model.fitSize` while choosing under the switch,
   `arx.fitRows`).  `NARX_SR_D.apply` / `rescale` and every `fromTerms` keep `trSize = -1`, as ScalaTion's do.
2. `tFormT` comes before `trSize`: `fromTerms (formula, dataset, hh, args, tFormT = None, trSize = -1)` (ARX, ARX_D;
   `NARX_SR_D.fromTerms (formula, dataset, hh, args, tFormT, trSize, maxLag, freqs, **opt)`), `ARX_Formula.fit` /
   `fit_D (dataset, hh, args, tFormT = None, trSize = -1)`, `NARX_Formula.fitNL_D (dataset, hh, args, tFormT, trSize,
   **opt)` -- ScalaTion's `fit (hh, tFormT, trSize, scaled)`; `tFormT` was an `args` entry in the old docstrings of
   `fit*`, which was wrong since rev1.10 (it is a RUN setting there).
3. `ARX_Formula.names` (a property) is the one name of the terms' labels (ScalaTion's `names (preds)`); the
   `labels` property and the `names ()` method are gone (`TermLibrary.labels` stays: a different thing).
4. `Model.TE_RATIO` is a checked class property (a metaclass): `Model.TE_RATIO = r` raises `ValueError` unless
   0.05 < r < 0.95, ScalaTion's `TE_RATIO_=` flaw.
5. `Forecaster_D.tnT_Test (rc, showYf = True, showParams, verbose)` overrides the base's `showYf = False`, as
   ScalaTion's `Forecaster_D.tnT_Test (rc = 5, showYf = true)`: the direct models print their forecast matrix.
6. `refitTransform ()` takes no size: it rebuilds the model by its factory under the new `Model.inSample ()` (the
   switch suspended), so the transforms land on the training rows; the recipes are zero-argument.  `withLambda`
   keeps the lambda through a refit as before.  (ScalaTion has no `refitTransform`: its `rescale` always fits on
   `Model.trSize`.)
7. Documented, not changed: Python's `Forecaster.validate (rc)` is the rolling pass over the validation rows;
   ScalaTion's `Forecaster.validate (rando, ratio)` is Dr. Miller's one-step split test, not ported (docstring,
   CONVENTIONS).  Kept on purpose: the data order `(y, hh, xe, args)`, `Args` for `hparam` / `scaled` / `tune`, the
   un-ported `fname_` / `tRng` / `bakcast`, the Python-first features (the switch, `scoreOf`, `findLambda`, the
   rankers, `effDF`, `enet`, `avg`), which ScalaTion will get step by step from rev5.2.
8. `docs/CONVENTIONS.md`: the table "ScalaTion rev5.0 <-> Python rev2.0: the factory signatures"; the Model and
   Forecaster rows now separate ScalaTion's members from the Python-first ones; the ElasticNet row names rev5.1's
   `ElasticNetRegression` / `CenteredElasticNet` (+ MV; `l1_ratio`, `cdIter`).
9. Tests: `tests/test_scalation_signatures.py` (7 tests: the signatures by `inspect`, the checked TE_RATIO,
   `inSample`, the rows `rescale` fits on and `refitTransform`, `fromTerms` / `fit_D` with a positional `tFormT`,
   `names`, `tnT_Test`'s default); the refitTransform and labels pins re-pinned; 247 tests pass.  No numeric pin
   moved: with the switch off `fitRows` is `Model.trSize`, the rows the old `trSize = -1` gave.

## 2.11.0 (ForecasterPy_rev1.11) -- 2026-10-17: the ten changes of the "naming options" review

The penalty
1. Standardized columns (`scaled = True`) are shown next to the raw ones wherever lambda is read: `arx_d2` prints both
   lambda landscapes with the column scales (sd deaths 5.6e3, icu 6.0e4, positive_rate 0.38: on raw columns lambda = 1
   is heavy for positive_rate's lags and invisible to the others, which is why every raw-column search lands on 1;
   standardized, the landscape is flat).  The default stays `False` (ScalaTion's `CenteredRidge`, the parity pins), and
   on Covid-19 the raw-column ridge at lambda 1 tests BETTER (full model TnT sMAPE 32.8 against 49.2 standardized): the
   shrinkage of the positive_rate block is what helps.
2. More scoring rows for the chooser: `findLambda (folds = k)` / `Forecaster.cvScore (folds, measure, hs)`, k-fold
   cross-validation on the embedded rows of the regression (contiguous blocks; the recursive models by their h = 1
   regression, the direct ones every horizon; original units; the model itself untouched), which scores every training
   row and is valid for a lag-input model with uncorrelated residuals (Bergmeir, Hyndman & Koo 2018) -- the k-fold
   ScalaTion's `findLambda` intended, minus its bugs.  The wider validation block needs no code: `withValidation
   (ratio = 0.4)` fits 45 and scores 47 rows, and `rollTest` IS the expanding window (refit every rc rows), so the
   "validation folds" of the review add nothing to it.  On Covid-19 the four ways disagree (grid on 24 rows: 1; Brent:
   1.38; 47 rows: 0.01; 5-fold on 92 rows: 1 raw, 100 standardized): the choice of lambda depends on the rows it is
   scored on more than on the search.
3. Ridge-aware statistics: `CenteredRidge.effDF` (1 + tr H_lambda) and `CenteredRidge.stdErr (k)` (the sandwich
   covariance sigma^2 (X'X + lI)^-1 X'X (X'X + lI)^-1 with sigma^2 = SSE / (m - effDF); the OLS errors at lambda 0);
   `Fit.summary (..., stdErr)`, `Forecaster.stdErrOf`, `summary ()` / `summary (h)` of a ridge print them (finite on
   19 collinear lags).  `Forecaster.scoreOf (..., nParams)` charges a penalized fit its effective parameters in the FPE
   factor (`Forecaster.effDF`, one value per horizon; `Candidate.nParams`; `jointQoF (qof, nParams)`,
   `FormulaScorer.scoreOf (qof, extra, nParams)`), so in-sample a ridge is no longer charged one parameter per column:
   the in-sample ridge pins move -- aRX_DTest8 CEM keeps [0, 3, 6, 7, 10, 11, 12, 13, 17, 18] at 10.3174 (effDF 7.4;
   was [0, 6, 7, 18] at 10.5218), nARX_SR_DTest5 Backward [0, 6, 8, 12, 13, 28] with the trace 52.2695 / 13.4114.
4. The elastic net: `method = "enet"` (`modeling/elastic_net.py`, `ElasticNet` with `CenteredRidge`'s interface;
   scikit-learn's coordinate descent on the centered, optionally standardized columns; alpha = lam / m so l1_ratio = 0
   reproduces the ridge; `Args.l1_ratio`, 0.5; `nnz`, `effDF` = 1 + nnz).  `PENALIZED_METHODS` = (ridge, enet) is the
   rule behind `lam`, `penalized`, `withLambda`, `findLambda`, `bestLambda`, `_subModel`, the evaluators and `train` /
   `_predict`.  `Example_Covid enet` compares ridge and enet (l1_ratio 0.3, 1) at their found lambdas: on standardized
   columns enet l1 0.3 at lambda 100 tests at 25.6 against the ridge's 41.3 and the lasso's 38.1 (one run, 24 test
   weeks: a hint, not a result).  NARX_SR_D stays ridge-only (VarPro's linear step is a ridge), so the LLM pipeline has
   no enet arm yet.
5. `findLambda (search = "brent")`: the grid, then Brent's bounded minimization in log10 lambda between the neighbours
   of the best grid value (the continuous optimum, ~8 more evaluations, without settling in the wrong valley); the
   grid stays the default.
6. `avg (x ~ (hi, lo))`: the mean of a lag block as ONE term (`AvgTerm`, the DSL's `avg (x, hi, lo)`, both parsers,
   "avg needs a lag range" otherwise); `TermLibrary.fromGrammar` adds the two blocks of 2+ lags per variable for
   maxLag >= 4 (`avg(v~(mid,2))`, `avg(v~(maxLag,mid+1))`, mid = (maxLag + 2) // 2; lag 1 stays its own term); the
   proposer's grammar text names it.
7. `Example_Covid diff`: ARX_D on the differenced series (d = 1, nneg off), the forecasts integrated back to levels
   from the origin t - 1 (the cumulative sum along row t - 1 of the forecast matrix) and diagnosed on the same testing
   rows: better at h = 1 (13.0 against 13.9), much worse from h = 3 on (mean 52.5 against 30.0).
8. The name tag: `lamTag` -- one decimal, except below 0.1 where the digits are kept (`_L0.01`, `_L0.001`), since the
   grid's small values all printed `_L0.0`; `Candidate` prints `lam=1` (`:g`).
9. / 10. The rankers by name: `selectFeatures (Ranked, rank = "t")`, the t-statistics |b_j| / se_j re-ranked per step
   (backward elimination by p-value, a ridge with its own standard errors) and `rank = "vif"`, minus the variance
   inflation factors (the most collinear column first); `SelectionReport.ranker`, `tImportance` / `vifImportance` in
   `Forecaster_Reg`, `Forecaster_D` and `FormulaScorer` (for `TermLibrary.selectBy`), `Example_RankedSelection` runs
   them.  On Covid-19 (ARX_D ridge, joint over h = 1..6) the t-ranker is competitive but not better than Backward / CEM
   (raw, in-sample: 34.5 against 28.4; standardized, validating: 38.6 against 54.3 for Backward), the VIF ranker keeps
   almost everything (no column has a VIF below 10): a statistic for the report, not a gate.

Tests: `test_ridge_summary_uses_the_sandwich_covariance_and_the_effective_df`,
`test_find_lambda_brent_refines_the_grid_and_k_fold_scores_the_training_rows`, `test_avg_is_the_mean_of_a_lag_block`,
the named rankers in `test_ranked_elimination...` (test_feature_selection), the enet name; the library size (+1 avg term per variable at maxLag 4).
240 tests pass.

## 2.10.0 (ForecasterPy_rev1.10) -- 2026-10-16: `tFormT` is an argument of `rescale`; one model per scenario

`tFormT` back in ScalaTion's place
- The transform type of a rescaling is a parameter of the factory again, as in ScalaTion and as `fEndo` / `fExo`
  already were: `ARX.rescale (y, hh, xe, args, tFormT = TransformT.MinMax, trSize = -1)`, the same for `ARX_D`,
  `ARX_SR.rescale (y, hh, xe, args, fEndo, fExo, tFormT, trSize)`, `ARX_SR_D`, and
  `NARX_SR_D.rescale (y, hh, xe, args, fEndo, fExo, tFormT, lu, trSize, fname, **opt)`; `fromTerms (..., tFormT = None)`
  (ARX, ARX_D, NARX_SR_D: None => the formula's own `TFORM:` clause).  The NormForm rule (nneg off) lives in
  `_rescaleData`, where ScalaTion has it.  It had moved into `Args` in rev1.4 with the other keyword settings, for
  the one-object `run (args)` style -- but it is a construction choice of the factory, not a setting of the model.
- `Args.tFormT` is a RUN setting now (next to `dataset`, `model`, `validation`), the transform `run (args)` passes to
  the model's `rescale` (None => no rescaling); no model reads it and the model classes' `ARGS` no longer list it.
  `rescale` raises when `args.tFormT` is set to another type than the one given, so a script that still writes
  `Args (tFormT = Log1p)` fails loudly instead of rescaling by MinMax.
- Call sites: the examples (Covid, ILI, ETTh1, GEFCom, FeatureSelection) and tests pass `TransformT.Log1p` /
  `Norm` / `MinMax` to `rescale`; `test_base_api` pins the refusal.

One model per scenario
- `Example_Covid.aRX_D_test2` builds the model at the lambda found once and uses it for In-Sample, TnT and the six
  selections (a selection never trains or alters its model; `inSample_Test` and `tnT_Test` retrain from scratch,
  the skip restored), as ScalaTion's `aRX_DTest4` does; the duplicate constructions between In-Sample and TnT in
  `randomWalk_test`, `aRIMA_test`, `aRX_D_test`, `aRX_SR_D_test` (Covid), `aRX_D_test`, `aRX_SR_D_test` (ILI),
  `aR_test` (ETTh1, GEFCom) and the RandomWalk scenarios are gone (the per-p loops of `aR_test` stay: one model per p).
- `Forecaster_D.selectFeatures` clears `_theBest` at the start, so the previous selection's best step (and its
  lambda, `bestLambda`) cannot leak into the next search on the same model.

The selection banners name the selected model
- `reportSelection`'s "selected: k = ... of n = ..." banner and `testSelected`'s "In-Sample Test / TnT Test ... of the
  selected model" banners (Forecaster_Reg, Forecaster_D) print the SELECTED model's name, whose `_L` tag is the lambda
  its best step found, instead of the host model's -- a host built at lambda 7 whose best step chose 1 printed
  `ARX_D_6_6_2_L7.0: selected ...` above a model fitted at `_L1.0`.  The host keeps its own name (it is the full
  column set at its own lambda).  Pinned in `test_tune_lambda_switch_gives_every_candidate_its_own_lambda`.

Findings (Covid-19, `arx_d2`): `findLambda` works (grid 22.3 / 22.3 / 21.5 / 15.7 / 21.3 / 23.6 / 23.8, a smooth U
with its minimum near 1.2; small sets pick 10 or 1000), but every selected set lands on lambda 1.0 because the ridge is
centered, not standardized (`scaled` False) and the series differ in scale by 10^5 (sd: deaths 5.6e3, icu 6.0e4,
positive_rate 0.38): lambda is one knob, "how much to shrink the positive_rate lags", the same for every set that has
them; with `scaled = True` the landscape is flat and prefers the smallest lambda.  `tuneLambda` on gives the same sets,
J and TnT as lambda fixed at 1 for all techniques, at 6-8x the time -- keep it off by default and call `findLambda`
once.  237 tests pass.

## 2.9.0 (ForecasterPy_rev1.9) -- 2026-10-16: `_L` on every regression-based model; `findLambda` -> (lambda, score)

The `_L{lambda}` tag
- EVERY regression-based model carries its ridge penalty in its name, written by the model class where the name is
  set (`f"ARX_D_{p}_{px}_{n_exo}_L{self.lam:.1f}"`, as ScalaTion's `modelName` strings will): `ARX_D_6_6_2_L0.0`
  under the default `sm_ols` (`Forecaster.lam` is 0.0 for a plain regression: least squares is the ridge at 0),
  `ARX_D_6_6_2_L1.0` under `method = "ridge", lam = 1`; the sub-models of `getModel` / `getModel_h` and
  `ARX.fromTerms` alike.  A model without a regression (AR, ARIMA, RandomWalk; `lam` None) has no tag.
  `Forecaster.modelName` is the plain property again; `lamSuffix` is gone.  `withLambda` re-tags its copy
  (`_setLambda`: the tag is the last part of the name).
- `SelectionReport.bestLambda (cols)`: the lambda the best step of the last selection found for that column set
  (None otherwise); `_subModel` applies it, so `getModel (getBest ().mod_cols)` -- the docstrings' "selected
  forecaster" -- carries the best step's lambda, and `selectedModel` (Forecaster_Reg, Forecaster_D) uses the same
  hook on its refit model instead of its own guard.

`findLambda` and the score
- `lam, score = mod.findLambda (lams, measure, hs, verbose)` returns the pair, as ScalaTion's
  `RidgeRegression.findLambda` does; the model is untouched and `mod.withLambda (lam)` is the copy at the lambda
  found.  A model under a plain regression has nothing to search: (0.0, its validation score).  The verbose line
  prints the raw lambda (`lambda = 0.001 (ARX_D_6_6_2_L0.0)`), since the tag rounds to one decimal.
- `Forecaster.scoreOf (qof, inSample, extra, measure = QoF.smape, hs = None)`: the error measure averaged and the
  horizons averaged are arguments (`Forecaster.ERROR_MEASURES`; a goodness measure such as rSq raises, since the
  FPE factor would reward complexity); `valScore (rc, measure, hs)` passes them on.  Defaults unchanged: the mean
  sMAPE over all horizons, so no pinned number moves ...
- ... except under `tuneLambda` for a recursive sub-model: `Forecaster_Reg.evalCols` now tunes by `hs = [1]`, the
  one horizon a sub-model without the lag structure forecasts (the horizon `evalCols` scores by), where before the
  mean over all hh horizons entered; `Forecaster_D.fitCols` keeps all horizons (direct models).  Both call
  `findLambda` only for a ridge (nothing to tune otherwise), the scorer always has one.

Examples and tests
- `Example_Covid.aRX_D_test2` (`arx_d2`): ARX_D ridge, `findLambda` on the validation rows (68 / 24 / 24), In-Sample
  and TnT at the lambda found (1.0, mean validation sMAPE 15.70), then all six selection techniques with
  `tuneLambda` on, each selected model reported and TnT-tested at its best step's lambda.
- Tests: `test_score_of_takes_the_measure_and_the_horizons`; `findLambda`'s pair, `bestLambda`,
  `getModel (getBest ().mod_cols)` at the best lambda, the re-tag of `withLambda`; the plain models' names carry
  `_L0.0` (`test_args`, `test_arx_formula`, `test_base_api`, `test_exo_screen`, `test_feature_selection`,
  `test_narx_formula`, `test_narx_sr_d`).  237 tests pass.

## 2.8.0 (ForecasterPy_rev1.8) -- 2026-10-16: findLambda and the tuneLambda switch; HyperSearch removed

The ridge penalty
- Every model fitted by the ridge carries its penalty in its name: `ARX_D_6_6_2_L1.0`, `NARX_SR_D(6, 4, 2)_L6.0`
  (`Forecaster.modelName` appends `lamSuffix` = `_L` + lambda with ONE digit after the decimal point; a model
  without a ridge, method sm_ols / sk_lr, has no tag).  `Forecaster.lam` is the penalty (None without a ridge).
- `Forecaster.LAMBDAS` = (1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3); `withLambda (lam)` (a copy with a new regression,
  same rows, matrix and transforms, untrained; a factory-made model's `refitTransform` keeps the new lambda);
  `findLambda (lams, verbose)`: the copy at the grid value with the best `valScore`, ScalaTion's
  `RidgeRegression.findLambda` with the forecaster's rolling validation in place of k-fold CV (which is wrong for a
  time series).  Needs `Model.validating`: in-sample a ridge always prefers the smallest lambda.
- `Model.tuneLambda` (the second switch, `withValidation (ratio, tuneLambda)`): every candidate a chooser evaluates
  gets its own best lambda -- `Forecaster_Reg.evalCols` / `select0`, `Forecaster_D.fitCols`, `FormulaScorer.fit`
  and `evalCols` -- so the best step of a feature selection carries the best lambda for its columns (its model's
  name shows it), `selectedModel` / `reportSelection` build the reported model at that lambda, and a `Candidate`
  carries `lam`, which `tnt (f, lam)`, `fitFull (f, lam)`, `record` and `toModel` use; `repr` and `report` print it;
  the scorer counts one fit per grid value in `evaluations`.  Off (the default) is exactly rev1.7.

Removed
- `modeling/hyper_search.py` (HyperRange, HyperSpace, Trial, Sampler, Grid, Random, HyperSearch) and
  `llm_narx_sr/llm_sampler.py` (LLMSampler), `LLM_NARX_SR.llmTunedArm`, `BaselineArm (sampler, maxTrials)`: the one
  numeric knob the models need is lambda, found by `findLambda`; p and px come from the EDA (PACF, CCF), feature
  selection or the nested tuned-arx loop, which is its own double loop again (the same 42 structures, the same J).

Tests: `tests/test_validation_find_lambda.py` replaces `test_validation_hyper_search.py` (the model names, findLambda,
the tuneLambda switch on ARX_D, the scorer and a library selection); the model-name assertions carry `_L`.
236 tests pass.

## 2.7.0 (ForecasterPy_rev1.7) -- 2026-10-15: one split, one switch, one score; Ranked selection; HyperSearch

The validation, tuning and selection plan (the Oct 15 plan doc / PDF), implemented Python-first; ScalaTion rev4.2
is to follow it.  237 tests pass (PyTorch optional).

The switch (`modeling/model.py`, NEW: ScalaTion's `Model` object)
- `Model.TE_RATIO` (0.2), `Model.VAL_RATIO` (= TE_RATIO: the validation block has the testing block's size, 116
  Covid-19 weeks -> 68 / 24 / 24), `Model.validating` (the ONE switch), `teSize`, `trSize`, `valSize`, `fitSize`,
  `fitRng`, `valRng`, `teRng`, `splitStr`, `withValidation (ratio)` (a context manager: the switch on, then
  restored).  `forecaster.py` re-exports `trSize`, `teSize`, `teRng` from it; `args.py`'s `TE_RATIO` is `Model`'s.
  No folds and no modes: validation always rolls over `valRng` with the test's retraining cycle.
- Every chooser reads the switch: `Forecaster_Reg.evalCols` / `select0`, `Forecaster_D.fitCols` (ON: the candidate
  is trained on the fit rows and rolled over the validation rows, `rollTest`; OFF: today's rows, penalised),
  `FormulaScorer` (`validating` argument, else `Model.validating` at construction), the factories' transforms
  (`rescale (..., trSize)`, `fromTerms (..., trSize)`: `fitRows` = the fit rows while validating) and
  `reportSelection` (`selectedModel` refits the transforms on the training rows, `refitTransform`).

Forecaster (one rolling routine, three tests, one score)
- `Forecaster.RC = 5` (the retraining cycle; `args.rc` None => it), `Forecaster.scoreOf (qof, inSample, extra)`:
  the mean sMAPE over the horizons, times the FPE factor (1 + 2 (dfr + 1 + extra) / n_h) when the rows were fitted.
  `Forecaster_D.smapeFPE` / `aggFPE` and `FormulaScorer.jOf` / `hSel` are gone: `scoreOf` is the one score
  (`Forecaster_D.jointQoF` aggregates a QoF matrix for the joint selection with it).
- `rollTest (rng, rc, label, verbose, showYf, showParams)`: trains on the rows before the block, `rollValidate`
  over it (now with a `rng` block and `verbose`), diagnoses the block's rows only (a validation pass never reads a
  testing row), and RESTORES the skip (ScalaTion rev4.1's `tnT_Test` left it at 0, so a selection run afterwards
  scored the burn-in rows).  `tnT_Test` = rollTest over the testing rows (its REPORT is now the model trained on
  the training rows; the QoF tables are unchanged), NEW `validate ()` = rollTest over `Model.valRng`, NEW
  `tvT_Test` = validate then tnT_Test with one summary line, NEW `valScore ()` = scoreOf (validate ()), NEW
  `refitTransform (n)` (a factory-made model rebuilt with its transforms fitted on n rows; `_recipe`).
- `runner.run`: `args.validation = "tvT"`.

Feature selection
- NEW `SelectionTech.Ranked` and `SubsetSearch.ranked (n, eval, rank, fixed, units, minCols)`: backward
  elimination in the order of a ranking (recursive feature elimination): the ranker decides which column leaves,
  the criterion where to stop; n evaluations.  `rank` is a vector (one ranking of the full set) or a function of
  the current set (re-ranked after every removal).  Hosts: `Forecaster_Reg.rankedElimAll (rank, first, cross,
  minCols)` with the rankers `coefImportance` (|b_j| sd(x_j), the default) and `permImportance`;
  `Forecaster_D.selectFeatures (..., rank)` (joint `coefImportance`), `selectFeaturesAtH (..., rank)`;
  `TermLibrary.selectBy (Ranked, scorer, rank)` with `FormulaScorer.coefImportance (lib, cols)` as the default.
- The in-sample selection score of the base evaluators is `scoreOf (inSample = True)`, the FPE form, in the
  sMAPE-IC slot (rev4.1: `Fit.smapeC`'s additive 2(k+2)/m): aRX_DTest8's CEM keeps 4 columns at 10.522 (was 12
  at 8.901); the traces' sMAPE-IC column changes, the selected sets of the other pinned tests do not.

HyperSearch (`modeling/hyper_search.py`, NEW, at the model level)
- `HyperRange` (`Discrete`, `IntRange`, `LogRange`, `Continuous`), `HyperSpace` (`size`, `points`, `sample`,
  `keyOf`), `Trial`, `Sampler` (`Grid`, `Random`), `HyperSearch (space, objective, sampler, maxTrials, stream,
  minimize, verbose)`: `run ()`, `best`, `trials` (a cache by key: a point is never evaluated twice; a failing
  objective scores inf), `ranked ()`, `report ()`, `saveCSV (path)`.  The objective is a closure the caller
  writes, e.g. `lambda hp: ARX_D (y, hh, xe, Args (**hp)).valScore ()`.
- `BaselineArm (Method.TunedARX, sampler, maxTrials)` runs through it (the grid by default; the same J as every
  arm); NEW `llm_narx_sr/llm_sampler.py`: `LLMSampler (client, eda, perCall, maxCalls, shown)`, a `Sampler` that
  asks the LLM for the next points (snapped into the space, de-duplicated, random fallback);
  `LLM_NARX_SR.llmTunedArm (llm, eda, maxTrials)` is the tuned-arx-llm arm (run per stream).

FormulaScorer on the switch
- `FormulaScorer (data, hh, hp, scaled, freqs, validating)`: score = fit on the fit rows + `qofValidation` (a
  fresh copy rolled by `rollTest`) + `scoreOf` when validating, else the in-sample FPE-form J; `tnt` / `tntModel`
  through `tnT_Test`; `splitStr`; `rc` = `Forecaster.RC`; `skip` = hp["maxLag"] (one skip for every candidate).
  REMOVED: the hp entries valFrac, valFolds, rc, skip, hSel (an unknown hp key is now an error) and the methods
  `qofValidationAll`, `jOf`, `qofTnt`.  `LLM_NARX_SR.VALIDATING = True`: the harness's scorers score on the
  validation rows (`scorerOf (dataset, hp, validating)`); the results CSV gains val_ratio and rc; `ExpRun.valRatio`,
  `rc`; the grid banner prints the split; `showBest` tests through `tnT_Test`.
- The anchors change with the protocol (fit 68, validation 24 rolled with rc = 5): Covid-19 icu/pRate baseline
  20.030 / 18.117 / 20.030 / 43.483 (score, in-sample, validation, TnT), tuned-arx 16.169 / 20.128 / 16.169 /
  44.586 (y~1 + icu~(5,1) + pRate~(5,1)), sel-arx 14.570 / 21.686 / 14.570 / 51.228, random-walk 42.117 / 32.978
  / 42.117 / 45.712; auto exo (exMort, cases): 25.089 / 18.417 / 25.089 / 61.807, 24.730 / 19.761 / 24.730 /
  61.913, 22.837 / 20.386 / 22.837 / 73.976.  The TnT numbers of the baseline and the random walk are the rev4.1
  log's: the test pass did not change.

The attention ranker (`modeling/attn_ranker.py`, NEW, optional PyTorch: `pip install -e .[neural]`)
- `Attn_Ranker (x, y, dModel, heads, hidden, dropout, epochs, lr, weightDecay, batch, seeds, stream, score,
  regressor)`, the shape of `CEM_Selector`: `run ()` returns the importance per column (`scores`: the mean
  attention a column receives as a key, the NARX_SR paper's eq. 4; `gradScores`: gradient x input), averaged over
  seeds, on standardized rows.  Its `AttnRegressor` gives every column its own embedding (x_j w_j + b_j), which
  the paper's `MHAttn_Regressor` (a shared `Linear (1, d)` and mean pooling, permutation-invariant over the
  columns) lacks; that regressor plugs in through `regressor` for the comparison.
- NEW `Examples/Example_RankedSelection.py`: the experiment that decides the ScalaTion port -- the rankings
  (coefficients, permutation, attention, gradient x input, the paper's regressor), `Ranked` with each against
  Backward and CEM, the switch off and on, Spearman between the rankings, J, sets evaluated, TnT sMAPE.

Not changed (base-file decisions still open in the plan): the trend column names (cons, lin, quad), column 0 under
first = 1, `Forecaster_D.train` fitting every horizon on the rows of the last one, `Fit.smapeC` itself.

Tests
- NEW `tests/test_validation_hyper_search.py` (the split and the switch, rollTest / validate / tvT_Test, the
  evaluators under the switch, HyperSearch and its samplers, the LLM-tuned arm, the attention ranker (skipped
  without torch)); `test_feature_selection.py` (Ranked, scoreOf / jointQoF, aRX_DTest8 re-pinned),
  `test_formula_scorer.py` (the switch, the unknown-hp error, J in-sample vs validation),
  `test_llm_narx_sr_arms.py` (the anchors re-pinned, the CSV header), `test_random_walk.py` / `test_narx_sr_d.py`
  (the skip restored).

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
- `UnitRoot.setTrend` does not print ScalaTion's "setTrend: trend = c, npar = 2" line.

Same in both now: the validation blocks refit COPIES of a candidate's formula (Python always did; ScalaTion's
re-issued rev4.1 `FormulaScorer.score` does too -- before, it refitted the candidate's own parameters, which then
showed the LAST block's values: the first rev4.1 lLMProposerTest printed 2.012 / 4.742 for a model fitted with
1.9475 / 7.698).

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
