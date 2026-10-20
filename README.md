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
pytest                      # 240 tests, ~1 min, all offline (1 needs PyTorch and is skipped without it)
python Examples/Example_Covid.py              # rw | ar | arima | arx | arx_d | arx_d2 | enet | diff | arx_sr | arx_sr_d | formula | narx | narx_sr | optimizers | mhattn | run | all
python Examples/Example_ILI.py                # rw | ar | arima | arx_d | arx_sr_d | all
python Examples/Example_ETTh1.py              # rw | ar | arx_d | arx_sr_d | narx | all
python Examples/Example_GEFCom.py             # rw | ar | arx_d | arx_sr_d | narx | all
python Examples/Example_FeatureSelection.py   # 5 | 6 | 7 | 8 | 9 | search | cem | all   (aRX_DTest5-9, subsetSearchTest, cEM_SelectorTest)
python Examples/Example_LLM_NARX_SR.py        # covid | optimizers | gef14 | etth1 | baseline | genetic | random | mockllm | grid | all
```

With no argument each example runs a quick default.  `pip install -e .` makes `modeling` and `Util`
importable from anywhere; the examples also run from the repository root without it.

## Settings: `Args`

A model is built as `Model (y, hh, [xe,] args)`: the data in the call, every setting in one `Args`
(`modeling/forecasting/args.py`) -- the dotdict style of experiment scripts, but checked:

```python
from modeling.forecasting.args import Args
args = Args()                      # every setting at its default (ScalaTion's AR.hp, lam 0.1, ...)
args.p = 6                         # checked when set: name, type and range
args.px = 6
args.spec = 1
args.method = "ridge"              # the regression ("enet" by default | "ridge" | "sm_ols" | "sk_lr"); args.lam its penalty
args.pX = 6                        # AttributeError: Args has no setting 'pX'; did you mean 'px'?
args.p = -1                        # ValueError: Args.p = -1: must be an integer >= 0
args.help()                        # the table: value, default, meaning, allowed values; * = changed
Args.help(ARX_D)                   # only the settings ARX_D reads
args2 = args.copy(px=2)            # a variant (args unchanged); Args(p=6, px=6) works too
args.save("covid.json"); args = Args.load("covid.json")      # a preset (the changed settings)
```

| group | settings |
|---|---|
| hyper-parameters (ScalaTion's `AR.hp`, printed in the REPORT) | `p` (lags of y, 1), `d`, `q`, `P`, `D`, `Q`, `s` (ARIMA), `px` (lags of each exogenous variable, 1), `spec` (trend columns, 1), `crx` (cross terms, False), `nneg` (forecasts clipped at 0, True) |
| hyper-parameters (Python only) | `skip` (rows left out of the In-Sample diagnosis; None: the model's rule, usually p), `lam` (ScalaTion's `RidgeRegression.hp("lambda")`, 0.1; `Args(**{"lambda": 1.0})` is accepted), `l1_ratio` (ScalaTion's `ElasticNetRegression.hp("l1_ratio")`, 0.5); `mod.hparameter` is the MERGED map, the model's keys plus the knobs of its regression, as ScalaTion's since rev5.4 |
| model settings | `method` (None: the model's own -- the elastic net for every regression-based forecaster, ScalaTion's `REGRESSION` import), `scaled` (the penalized regression standardizes the columns), `optimizer`, `tune` (NARX_SR_D), `te_ratio` (the testing fraction; None: `Model.TE_RATIO` when the model is built; 0: no testing set) |
| run settings (`run (args)`) | `dataset`, `exo`, `model`, `tFormT` (the transform `run` passes to the model's `rescale`; None: no rescaling), `hh`, `validation` ("inSample", "tnT", "both"), `rc` |

The transform type of a rescaling is an argument of the factory, as in ScalaTion -- `ARX_D.rescale(y, hh, xe, args,
TransformT.Log1p)`, `ARX_SR_D.rescale(y, hh, xe, args, fEndo, fExo, TransformT.Log1p)`, `fromTerms(formula, dataset,
hh, args, tFormT=None, trSize=-1)` (None: the formula's own `TFORM:` clause) -- not a setting of the model: a model
never reads `args.tFormT`, and `rescale` refuses an `args.tFormT` that differs from the one it is given.  `rescale`
fits the transform on the rows `Model` gives (the training rows, ScalaTion's `Model.trSize`; the fit rows while
choosing under `Model.validating`), so it takes no `trSize`, exactly as ScalaTion's; `fromTerms` and `NARX_SR_D.apply`
/ `rescale` keep ScalaTion's `trSize = -1`.  The signatures follow ScalaTion rev5.0 argument for argument
(`docs/CONVENTIONS.md`, "ScalaTion rev5.0 <-> Python rev2.0", "ScalaTion rev5.2 <-> Python rev2.1: the protocol", "ScalaTion rev5.3 <-> Python rev2.2: the knobs" and "ScalaTion rev5.4 <-> Python rev2.3: the elastic net by default").

A model keeps a private SNAPSHOT of its args (later changes to `args` do not reach it); `mod.hparameter` returns a copy
of its `HyperParameter` (`AR.hp` with the values of args, as ScalaTion's `hparameter`), which the REPORT prints as
ScalaTion: "HyperParameter(p -> (6,1), ..., nneg -> (1,1))".  `AR.hp` itself is read-only.  A setting a model does
not read is ignored by it, so one `args` can drive RandomWalk, ARIMA and ARX_D side by side; a method a model does not
have is refused (`AR` with method "ridge").  A dict of settings is accepted where an `Args` is (checked the same way).

The one-object run of an experiment script (`modeling/forecasting/runner.py`):

```python
from modeling.forecasting.runner import run
args = Args(dataset="Covid-19", model="ARX_D", p=6, px=6, spec=0, hh=6, validation="both")
mod = run(args)                    # loads the data, builds the model (rescale(..., args.tFormT) when set), In-Sample + TnT
mod.Yf                             # the forecasts (original units); makeModel (args, y, xe) on your own data
```

## The ScalaTion conventions

Every model is used the way its ScalaTion counterpart is, and prints the same blocks:

```python
from modeling.forecasting.arx_d import ARX_D
mod = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1))               # or ARX_D.rescale(y, hh, xe, args), ARX_D.fromTerms(formula, dataset, hh, args)
mod.inSample_Test()        # banner, "b = VectorD(...), fn = Array(cons, yl6, ...)", REPORT, "fitMap QoF =" table over h = 1..hh
mod.tnT_Test(rc=5)         # train, rolling validation on the testing rows (retrain every rc rows), QoF table on the testing rows
ftMat = mod.diagnoseAll()  # (hh, 26) QoF matrix: rSq, rSqBar, sst, sse, sde, rse, mse0, rmse, mae, smape, bmae, m, dfr, df, fStat, aic, bic, mape, mase, smapeC, ...
mod.Yf                     # the forecast matrix in ORIGINAL units: column 0 = y, Yf[t, h] = h-steps ahead forecast of y[t + h - 1]
mod.forecastAt(h)          # column h of the forecast matrix (ScalaTion's forecastAt)
print(mod.summary())       # the coefficient table (SUMMARY: estimate, std. error, t, p-value, VIF), mod.summary(h) per horizon
mod.parameter, mod.slopes, mod.intercept, mod.fname, mod.modelName, mod.hparameter   # b, original-unit slopes, ..., "ARX_D_6_6_2", hp (a copy)
```

* Hyper-parameters: ScalaTion's keys in `Args` (above); `spec` counts the trend columns with ScalaTion's convention
  (**0 implicit constant, 1 linear t/m, 2 + quadratic, 3 + cubic**; the columns are named as `MakeMatrix4TS.trend`:
  "cons", "lin", "quad").
* The regression lives in the bases, as ScalaTion's `reg`: `Forecaster_Reg` (recursive) and `Forecaster_D` (direct)
  make it (`_make_model`: "enet" = `ElasticNet` by default, "ridge" = `CenteredRidge`, "sm_ols", "sk_lr"), fit it (`train`) and expose `parameter`
  (ScalaTion's b / bb), `slopes`, `intercept`; the model classes add the matrix, the forecasts and the sub-models.
* Quality of Fit (`modeling/fit.py`, ScalaTion's `Fit`): the 26 measures in `QoF` order with ScalaTion's
  formulas (R²-bar through the DF ratio, AIC/BIC with kParam = dfr + 2, sMAPE-IC), the first `skip` rows
  dropped (`Diagnoser`), the degrees of freedom reset per horizon.  The Python runs reproduce the ScalaTion
  logs digit for digit (`ARX_D_1_0_0` on Covid-19: TnT sMAPE 18.5489, 31.4088, 45.3246, 57.0302, 64.1919,
  70.4402; the CEM selection of `aRX_DTest8`; `subsetSearchTest`).
* Train/test split: `Model.trSize` / `teSize` (te = floor(0.2 m + 1): 92/24 on Covid-19, 1679/420 on GEFCom2014,
  966/242 on ILI, 13935/3485 on ETTh1); `Model.TE_RATIO = r` is checked as ScalaTion's `TE_RATIO_=` (0.05 < r < 0.95,
  or 0: no testing set -- a competition's dataset, the examiner holds the test rows; `tnT_Test` then has nothing to
  test and the chooser works on the last `VAL_RATIO` of the dataset).
* The boundary rule of the rolling tests (rev2.1, ScalaTion rev5.2): a model is retrained only on rows whose
  targets lie before the forecast origin -- a direct model's window stops hh - 1 rows early (`Forecaster_D._trainWindow`),
  the single-horizon model at horizon h stops h - 1 rows early -- so no training target ever lies in the block being
  rolled over.  Every TnT number of a direct model moved a little (honestly so) from rev2.0; the recursive models are
  untouched.
* Printing helpers (`Util/tools.py`): `banner`, `fmt`, `jdouble` (Java's `Double.toString`), `vector_str`
  ("VectorD(0.946883,\t...)"), `matrix_str`, `array_str`, `set_str` ("LinkedHashSet(0, 1, 3)").
* `docs/CONVENTIONS.md` maps every ScalaTion file to its Python file and states the rules.

## Models (`modeling/forecasting/`)

| Python | ScalaTion | what |
|---|---|---|
| `RandomWalk(y, hh, args)` | RandomWalk.scala | the last-value baseline (no training; no hyper-parameters, the REPORT prints "null") |
| `AR(y, hh, args)`, `AR.rescale` | AR.scala | AR(p) by Yule-Walker (args.method "adjusted", "mle") or OLS; `AR.hp` the base hyper-parameters (read-only); `rescale` standardizes (NormForm) |
| `ARIMA(y, hh, args, trend)` | ARMA.scala / ARIMA_diff.scala | ARIMA(p, d, q) backed by statsmodels (exact MLE) under the ScalaTion interface |
| `ARX(y, hh, xe, args)`, `ARX_D` | ARX.scala, ARX_D.scala | lags of y and of the exogenous variables; recursive (forge / hide) and direct; methods `sm_ols`, `sk_lr`, `ridge`, `enet`; `rescale(y, hh, xe, args, tFormT=MinMax)`, `ARX.trend`, `fromTerms(formula, dataset, hh, args, tFormT=None, trSize=-1)`; `ARX_D.tnT_Test` prints the forecast matrix by default (`showYf=True`, ScalaTion's `Forecaster_D.tnT_Test`) |
| `ARX_SR`, `ARX_SR_D` | ARX_SR.scala, ARX_SR_D.scala | + transformations of the lags: `fEndo=[PowForm(), RootForm()]`, `fExo=[[LogForm()], []]` (instances, as ScalaTion's `LSET [Transform]`; `Util/transform.py` = `mathstat/Transform.scala`), cross terms (`crx`) |
| `ARX_Formula` + `FormulaParser` | ARX_Formula.scala, FormulaParser.scala | the linear formula DSL and grammar: `y ~= y~(3,1) + log1p(icu~1) + y~1 * icu~1 + wave(2) TFORM: Log1p`; `fit(data, hh)` -> `ARX.fromTerms`, `fit_D` -> `ARX_D.fromTerms` |
| `NARX_Formula` + `NARX_FormulaParser` | nonlinear/NARX_Formula.scala, NARX_FormulaParser.scala | + free parameters `Par` with roles (Shift, Phase, Rate, Power, Freq, Decay, Thresh, Width, Half, Cap): `log(? + icu~1)`, `pow(y~(3,1), ?)`, `hinge(temp~1, ?)`, `trans(y~1, ?, ?)`, `ema(x~1, ?)`, `dq(x~1, ?)`, `wave(?)`; `fitNL_D(data, hh)` |
| `NARX_SR_D` (an `ARX_D`): `apply` / `rescale(y, hh, xe, args, fEndo=[TransformT.Root], fExo)`, `fromTerms(formula, dataset, hh, args)` (args.tune, scaled, optimizer, lam) | nonlinear/NARX_SR_D.scala | the direct model whose columns carry nonlinear parameters -- of transform TYPES (`TransformT.wlu` start values and bounds, `initializeW`, `tunable`, `getTransforms`, `optimize`) or of a formula's free parameters: coefficients by the model's regression per horizon (the elastic net by default), the nonlinear parameters by `OptimizerNL.VarPro` (its inner ridge) (variable projection, the default `OPTIMIZER_NL`), `LBFGS_B` or `Joint` (`fitNL`, `stepCounts`); each fitted parameter is a degree of freedom (`nFree`); reproduces nARX_SR_DTest5 and nARX_SR_DTest8 |
| `ExoScreen.screen(data, header, resp, k, maxLag)` | ExoScreen.scala | pool / rank (lead cross-correlation) / shortlist of candidate exogenous variables |
| `MHAttn_Regressor` (optional torch) | -- | the NARX_SR paper's multi-head attention regressor (as written) |
| `Attn_Ranker(x, y, ...)` (`modeling/attn_ranker.py`, optional torch) | autograd/Attn_Ranker (to follow) | attention scores as an importance per column (`run()` -> `scores`, eq. 4; `gradScores`), per-column embedding, seed-averaged; the ranking `Ranked` consumes |

### One split, one switch, one score (`modeling/model.py`, `Forecaster.rollTest` / `scoreOf`)

`Model` holds the split of a dataset -- `TE_RATIO` 0.2, `VAL_RATIO` = TE_RATIO (116 Covid-19 weeks -> 68 fit / 24
validation / 24 test, `Model.splitStr(116)`) -- and the ONE switch every chooser reads, `Model.validating`: ON, a
candidate (a column set, a hyper-parameter point, a formula) is fitted on the fit rows and scored on the validation
rows, rolled with the test's retraining cycle (`Forecaster.rollTest`, `valTest()`); OFF (the default), it is fitted
and scored on the training rows with the FPE-form penalty.  Either way the number compared is `Forecaster.scoreOf`
(the mean sMAPE over the horizons, times (1 + 2(k+1)/n_h) when the rows were fitted).  The testing rows are read
once, by `tnT_Test`; `tvT_Test` shows the validation pass, then the test.

```python
from modeling.forecasting.forecaster import Model
with Model.withValidation():                                       # 80/20 -> 60/20/20 for everything inside
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.Backward)   # scored on the validation rows
    mod.reportSelection(colsArr, rSqArr)                            # the chosen set rebuilt on the training rows (refitTransform), then TnT
mod.tvT_Test()                                                     # validation QoF table, test QoF table, one summary line
```

### Feature selection (`modeling/feature_selection.py`, `subset_search.py`, `cem_selector.py`; the scenarios inside `Forecaster_Reg` / `Forecaster_D`)

ScalaTion's `FeatureSelection`, `SubsetSearch` and `CEM_Selector` (with a port of ScalaTion's MRG31k3p `Random`
so a CEM run with the same stream draws the same samples), plus the `Ranked` technique: backward elimination in the
order of a ranking (recursive feature elimination; the ranker decides which column leaves, the score where to stop;
n evaluations against n(n+1)/2).  As in ScalaTion, the selection methods live in the two forecaster bases:
`Forecaster_Reg` (the regression-based recursive models `ARX`, `ARX_SR`: `forwardSelAll`, `backwardElimAll`,
`stepwiseSelAll`, `beamSelAll`, `cemSelAll`, `rankedElimAll` with the rankers `coefImportance` / `tImportance` /
`vifImportance` / `permImportance`, `correctQoF`, `getModel`) and `Forecaster_D` (the direct models: the three
scenarios below).  A ranker is named in the call (`rank="coef"`, the default, the standardized coefficients;
`"t"` the t-statistics, i.e. backward elimination by p-value, a penalized fit using its own standard errors; `"vif"` minus
the variance inflation factors, the most collinear column first; `"perm"`), or given as a callable or an array.

```python
from modeling.feature_selection import SelectionTech
colsArr, rSqArr = mod.selectFeatures(SelectionTech.CEM)            # one column set for all horizons, J = Forecaster.scoreOf
colsArr, rSqArr = mod.selectFeatures(SelectionTech.Stepwise, h=1)  # the h = 1 model decides
colsArr, rSqArr = mod.selectFeatures(SelectionTech.Ranked)         # RFE with the standardized coefficients re-ranked per step
colsArr, rSqArr = mod.selectFeatures(SelectionTech.Ranked, rank="t")   # elimination by p-value (|t| re-ranked per step)
colsArr, rSqArr = mod.selectFeatures(SelectionTech.Ranked, rank=Attn_Ranker(X_fit, y_fit).run())   # an attention ranking
mod.reportSelection(colsArr, rSqArr, inSample=False)               # prints the selection and TnT-tests the selected model (getModel(cols))
```

`selectFeaturesAtH`, `selectFeaturesPerH`, `getModel(cols)` / `getModel_h(cols, h)`; a recursive `ARX` selects its own
features the same way (`arx.selectFeatures(SelectionTech.Stepwise)`); `FeatureSelection.fullset_FS = True` selects on all rows.
`Examples/Example_RankedSelection.py` compares the rankings (coefficients, t-statistics, VIF, permutation, attention,
gradient x input) and the techniques on Covid-19 with the switch off and on.  On Covid-19 the t-ranker is competitive
with Backward and CEM (a 4-column set at TnT sMAPE 38.6 against 54 for Backward with standardized columns under
validation) but not better in general, and the VIF ranker keeps almost everything: the collinearity of consecutive
lags is structural (no column of the 19 has a VIF below 10), so a VIF gate or ranker removes the lag structure first.

### The knobs: `tune`, `withHP`, the `Model.tuning` switch, the elastic net and the standard errors of a penalized fit

Every regression-based model class writes its KNOBS into its name (`hpTag`): `ARX_D_6_6_2_L0.1_A0.5` under the
elastic net, the default (`_L` + lambda with one decimal; below 0.1 the digits are kept, `_L0.01`, since one decimal
would print every small value as 0.0; `_A` + l1_ratio with up to two decimals), `ARX_D_6_6_2_L1.0` under the ridge;
a plain least-squares fit (`method = "sm_ols"`) has no knobs and no tag, `ARX_D_6_6_2`, as have AR, ARIMA and
RandomWalk -- so a printed number always says which knobs made it.  Two penalized methods: `method = "enet"`, the
elastic net (`modeling/elastic_net.py`, scikit-learn's coordinate descent on the centered, optionally standardized
columns, at a tolerance of 1e-6 so its coefficients agree with ScalaTion's own descent to five digits; `Args.l1_ratio`,
0 the ridge, 1 the lasso, 0.5 by default; `lam` on the ridge's scale; at l1_ratio = 0 the ridge's own solver, so
l1_ratio = 0 IS the ridge), whose L1 part sets weak columns to exactly zero while the L2 part keeps correlated lags
together -- the continuous selection every hard selection should be measured against -- and `method = "ridge"`
(ScalaTion's `CenteredRidge`).  Since rev2.3 the elastic net is the default of every regression-based forecaster, as
`CenteredElasticNet(MV)` is ScalaTion rev5.4's `REGRESSION` import (`python Examples/Example_Covid.py enet`).

A `Knob` (`modeling/hyper_parameter.py`, ScalaTion rev5.3) is a hyper-parameter a tuner may turn: its name, the grid
a grid search tries, whether a continuous search moves in log10 (`Knob.LAMBDA`: 0.001 .. 1000 by decades) or
linearly (`Knob.L1_RATIO`: 0, 0.25, 0.5, 0.75, 1), and the letter of its tag.  The REGRESSION declares its knobs
(`CenteredRidge.tunable` = (lambda,), `ElasticNet.tunable` = (lambda, l1_ratio)), `mod.knobs` reads them by the
model's method, and ONE tuner turns whatever it finds: `hp, score = mod.tune()` returns the knob values with the
best held-out score, `{"lambda": 1.0}` or `{"lambda": 10.0, "l1_ratio": 0.75}`, and that score; the model itself is
untouched, `mod.withHP(hp)` is the copy at those knobs (re-tagged, untrained).  The score is `valScore`, the
forecaster's rolling validation on the validation rows, or with `folds=k` (or `Model.folds = k`) `cvScore`, k-fold
cross-validation on the embedded rows of the regression (contiguous blocks), which scores every training row and is
valid for a lag-input model with uncorrelated residuals (Bergmeir, Hyndman & Koo, 2018).  The searches: `"grid"`,
every combination of the knobs' grids (7 points for the ridge, 7 x 5 = 35 for the elastic net); `"golden"`, the
grid, then for each knob in turn a golden-section search between the neighbours of the best grid value in the
knob's coordinate, the other knobs held at the best point (the continuous optimum; tolerance 0.02, memoised);
`"random"`, `trials` points drawn log-uniformly / uniformly in the knobs' box from one seeded generator (the
baseline for three or more knobs, Bergstra & Bengio 2012).  The criterion is `Forecaster.scoreOf`: the error
`measure` (sMAPE by default; MAE, RMSE, MAPE ... are accepted, a goodness measure is refused) averaged over the
horizons `hs` (all by default; `hs=[1]` the first alone), with the FPE factor only in-sample -- charging a
penalized fit its EFFECTIVE parameters (`effDF`: 1 + tr H_lambda for a ridge, 1 + the nonzero coefficients for an
elastic net), not one per column.  Without folds, `tune` needs `Model.validating`, because in-sample a penalty
always prefers its smallest value.  The second switch, `Model.tuning`, makes every chooser call it on each
candidate -- a column set in feature selection (a recursive sub-model by its own horizon h = 1, a direct model over
all horizons), a formula in the scorer -- so the best step carries the knobs that made its score (the
chicken-and-egg of the penalty and the selection is solved by scoring the pair), and `getModel(getBest().mod_cols)`,
`reportSelection` and the TnT keep them (`bestHP(cols)`):

```python
with Model.withValidation():
    hp, score = mod.tune(verbose=True)                              # ({"lambda": 1.0}, mean validation sMAPE over h = 1..hh)
    hp, score = mod.tune(search="golden")                           # the grid, then golden-section per knob
mod = mod.withHP(hp)                                                # ... or ARX_D(y, hh, xe, args.copy(lam=hp["lambda"]))
mod.tnT_Test()
with Model.withValidation(tuning=True):
    colsArr, rSqArr = mod.selectFeatures(SelectionTech.Backward)   # every set at its own best knobs
    mod.getBest().mod.modelName                                    # ... "_L10.0": the knobs the best step found
    mod.bestHP(mod.getBest().mod_cols)                              # {"lambda": 10.0}: the selected model carries them
    mod.reportSelection(colsArr, rSqArr)                            # ... and so does the reported model
```

The SUMMARY of a penalized fit takes its standard errors from the covariance of the penalized estimator
(`modeling/regularized.py`, ScalaTion rev5.4's `Regularized.stdErr`, through `CenteredRidge.stdErr` and
`ElasticNet.stdErr`): over the active set A (the columns the L1 penalty kept), Cov = sigma^2 M Xa'Xa M with
M = (Xa'Xa + D)^-1 and D = diag (lambda (1 - alpha) + lambda alpha / |b_j|) (the L1 part as its local quadratic), sigma^2 =
SSE / (m - 1 - tr (M Xa'Xa)) -- the ridge sandwich at alpha = 0, the OLS (X'X)^-1 at lambda = 0, finite on collinear lags
where the OLS formula prints NaN -- the slopes back in original units, the intercept sigma^2 (1/m + xm' Cov xm); a
coefficient set to zero prints `NA` (no standard error, no t, no p), as does a constant column.  `summary()` /
`summary(h)` use them; a least-squares fit keeps the textbook errors (diag ((X'X)^-1) = the squared row norms of
R^-1); the VIF is a property of X and stays.

`python Examples/Example_Covid.py arx_d2` runs the whole story on Covid-19: `tune` on the validation rows, the
lambda landscape on raw and on standardized columns (`scaled=True`; on raw columns lambda = 1 is heavy for
positive_rate's lags, sd 0.38, and invisible to deaths and icu, sd 5,600 and 60,000, so "the best lambda" is a
statement about one block of columns), the knobs found by the golden-section and the random search, on a wider
validation block (`withValidation(ratio=0.4)`: `rollTest` is already the expanding window, refit every `rc` rows)
and by 5-fold cross-validation, In-Sample and TnT at the knobs found, then every selection technique with
`Model.tuning` on; `enet` tunes the ridge's one knob and the elastic net's two with the same `tune`.
`python Examples/Example_Covid.py diff` is the differencing experiment (d = 1): the forecasts of the differenced
model integrated back to levels and diagnosed on the same testing rows (on Covid-19 better at h = 1, much worse
from h = 3 on: the integration accumulates the errors).

## LLM-NARX-SR (`modeling/forecasting/nonlinear/llm_narx_sr/`, ScalaTion's `llm_narx_sr` package)

The structure of a forecasting model is written as a formula over a named-column dataset
(`{"y": ..., "icu": ..., "pRate": ...}`), parsed, fitted and scored by one engine, and searched for:

```python
from modeling.forecasting.datasets import Datasets
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D, OptimizerNL

data = Datasets.GEF14.load()                                   # daily peak load, temp, tmax
sc   = FormulaScorer(data, hh=6, hp={"maxLag": 7}, validating=True)   # the engine: the split, the switch, the score J
cand = sc.score("y ~= y~(7,1) + hinge(temp~1, ?)")             # parse -> fit -> score (the threshold lands at 65.8 F)
model, qof = sc.tnt(cand.formula.fresh())                      # rolling validation on the testing rows

m = NARX_SR_D.fromTerms("y ~= y~(7,1) + hinge(temp~1, ?)", data, 6, Args(lam=1.0, scaled=True, optimizer=OptimizerNL.LBFGS_B), maxLag=7)
m.inSample_Test(); m.tnT_Test()                                # the model alone, another optimizer

from modeling.forecasting.nonlinear.llm_narx_sr.llm_narx_sr import LLM_NARX_SR
runs = LLM_NARX_SR.runGrid("Covid-19", Datasets.COVID2, LLM_NARX_SR.zeroLLMArms(), streams=[0, 1])   # baseline, tuned-arx, sel-arx, random-walk, stat-cem, genetic, random
```

| level | files | what |
|---|---|---|
| base | `forecasting/datasets.py`, `forecasting/eda4ts.py`, `forecasting/stationarity_kpss.py`, `Util/correlogram.py` | the loaders `Example_Covid` / `Example_ILI` / `Example_ETTh1` / `Example_GEFCom` (ScalaTion's objects: `loadData` -> (xe, y), `clip`, `loadData_y`, `loadData_yy`, `context`), `DatasetConfig` (`ExoMode.Given` / `Auto`, the generic `shortlist (data, trSize, k, maxLag)`), `Datasets` (COVID2, COVID, COVID_ALL, COVID_FILE, ILI, ETTH1, ETTH1_1Y, GEF14, `fromFile`); `EDA4TS` (ACF/PACF, periodogram, CCF, KPSS (`Stationarity_KPSS`, ScalaTion's p-value), the residual report); the correlogram and the residual tests (`acorr`, `ccorr`, `corr`, `acfPacf`, `ljungBox`, `acfSpikes`, `collinearPair`, `heteroscedastic`) |
| L1 | `formula_scorer.py` | `FormulaScorer` (ScalaTion's hp entries; parse -> fit -> `Candidate` (with its knobs `hp`); J = `Forecaster.scoreOf`: the mean validation sMAPE when validating (a fresh copy rolled over `Model.valRng`), the FPE-form in-sample sMAPE otherwise, in-sample on the FS rows of the block (the boundary rule, as `Forecaster_D.fitCols`); under `Model.tuning` every candidate at its own best knobs (`tune`); rejection rules: too many free parameters, a parameter at a bound, an exponent near 1, a collinear free-parameter column; a fit cache; `tnt` (`tnT_Test`); `shortlist (cfg)`, the exogenous variables of a configuration; `coefImportance`) |
| L2 | `term_library.py` | `TermLibrary` (grammar or union libraries, `selectBy` CEM / Forward / Backward / Stepwise / Beam / Ranked on J) |
| L3 | `search_arm.py` | `ArmSpec` (llm-loop, llm-cem, stat-cem, genetic, random and the ablations), `ExpRun` / `Trace` records, `SearchArm` |
| L4 | `search_loop.py`, `genetic_proposer.py`, `llm_proposer.py`, `explainer.py` | the propose -> [trim] -> score -> feed back loop with adaptive stop; `RandomTerms` / `GeneticProposer`; `LLMProposer` (blind prompt: generic sentence, grammar, the EDA profile in words (`promptBlock`), scored population, rejects); `Explainer` (claims checked against the fit) |
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
          | avg(x~(k1,k2))                                        # the mean of a lag range as ONE term (lag grouping)
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
| `Datasets.ETTH1`, `ETTH1_1Y` | `ETTh1.csv` (Zhou et al. 2021) | 17,420 / 8,760 hours | OT (oil temperature, goes negative: `args.nneg = False`) | two of hufl, hull, mufl, mull, lufl, lull picked by the screening |
| `Datasets.GEF14` | `gefcom2014_load_daily.csv` (Hong et al. 2016; built by `tools/build_gefcom_daily.py`) | 2,099 days | peak load | temp, tmax |

### Results reproduced

On GEFCom2014 the lag baseline `y ~= y~(7,1) + temp~(7,1) + tmax~(7,1)` gives a rolling TnT sMAPE of 12.783
(mean over h = 1..6; ScalaTion rev3.3: 12.783) and the free hinge threshold of `y ~= y~(7,1) + hinge(temp~1, ?)`
is fitted at 65.8 F by all three optimizers (the cooling point of the V-shaped load-temperature curve: mean peak
per 10 F band 272, 226, 185, 148, 136, 214, 273).  On Covid-19 the anchors give TnT sMAPE 43.48 (baseline) and
45.71 (random-walk), as ScalaTion rev4.1 (tuned-arx chose `y ~= y~1 + icu~(3,1) + pRate~(3,1)` at 30.72 under
rev4.1's three validation blocks; under rev1.7's one rolled block it chooses `y~1 + icu~(5,1) + pRate~(5,1)`,
44.59); a shift inside a log and a decay in an ema run to their bounds and are rejected.  `NARX_SR_D.rescale` reproduces nARX_SR_DTest5
(Root transforms at their defaults: In-Sample rSq 0.951773, TnT sMAPE 17.8772, 15.8783, 18.9987, 25.8737, 34.0134,
44.1487 -- the whole REPORT and QoF tables digit for digit) and nARX_SR_DTest8 (Pow / Log fitted by VarPro to
(10, 1.99494, 0, 1, 0, 1), TnT sMAPE 14.1927, 18.2151, 23.9818, 32.7923, 42.8505, 53.1538); after TnT, Backward
selection at h = 1 keeps ScalaTion's 6 columns (0, 1, 6, 12, 13, 28) with its trace and the selected model's TnT
sMAPE 15.9482 .. 50.1970.  Since rev1.7 the selection scores follow the plan's protocol (one switch, the FPE-form
in-sample score, the validation block rolled); the anchors under it are in `CHANGELOG.md`, and ScalaTion rev4.2 is
to reproduce them.

### The measurement protocol (`llm_narx_sr/protocol.py`, `Forecaster.errorTable`)

One mean sMAPE per model on 24 testing rows cannot separate the arms (the selection variance dominates every
refinement, see the roadmap).  The protocol makes every row of a LONG test span a forecast origin and keeps the
per-origin, per-horizon errors in one table: `mod.errorTable(rng)` after `rollTest(rng)` gives, in original
units, one row per origin t and horizon h whose target lies in the block -- y, yhat, e, |e|, sape (the sMAPE's
term: a horizon's mean sape IS its sMAPE) and ase (the MASE's term, scaled by the naive one-step MAE of the rows
before the block; the QoF table's MASE scales by the block itself).  `Protocol(te_ratio=0.4, rc=5, hh=6)` runs
named model factories (`runModels`) or the arms of the harness (`runArms`: each run's best model rolled over the
testing rows at its knobs) on a dataset, appending to `errors`; `summary()` gives sMAPE and MASE per dataset,
model and horizon (and "all") with a 95 % interval from a HAC standard error (Newey-West, lag h - 1: the h-step
errors of neighbouring origins overlap); `compare(A, B)` tests A against B on the paired per-origin losses with
the Diebold-Mariano test (HLN small-sample correction, `significance.dm_test`) and the Wilcoxon signed-rank test
on the same pairs; `report(reference)` prints both tables, `write(dir)` saves them as CSV.
`python Examples/Example_Protocol.py all --arms` writes the "before" tables of the base models and the
deterministic arms on Covid-19, ILI, GEFCom2014 and ETTh1 to `results/protocol/` -- what every later stage is
measured against (on Covid-19's 47 origins the ridge `ARX_D` beats the random walk, DM* -3.7, p < 0.001 pooled
over the horizons, where `sel-arx`'s h = 1 is worse than the random walk, DM* +4.3: the selection variance,
quantified).

## Repository layout

```
modeling/fit.py                      QoF, summary, vif (ScalaTion Fit / FitM)   modeling/centered_ridge.py   CenteredRidge
modeling/regularized.py              stdErr: the standard errors of a penalized fit (ScalaTion's Regularized)   modeling/elastic_net.py   ElasticNet
modeling/model.py                    Model: the split (TE_RATIO, VAL_RATIO, trSize, valSize, fitSize, valRng) and the switches (validating, tuning, folds, withValidation)
modeling/hyper_parameter.py          HyperParameter (ScalaTion's; `freeze` for the read-only AR.hp), Knob (LAMBDA, L1_RATIO)
modeling/feature_selection.py        SelectionTech (+ Ranked), BestStep, FeatureSelection   modeling/subset_search.py   SubsetSearch (+ ranked)
modeling/cem_selector.py             CEM_Selector, Random (MRG31k3p)        modeling/attn_ranker.py   Attn_Ranker (optional torch)
modeling/llm_client.py               LLM clients
modeling/forecasting/                args (Args, AR.hp), runner (run), datasets (Example_* loaders, Datasets), eda4ts, diagnoser,
                                     forecaster, forecaster_reg, forecaster_d, random_walk, ar, arima, arx, arx_d, arx_sr, arx_sr_d,
                                     arx_formula, formula_parser, exo_screen
modeling/forecasting/nonlinear/      narx_formula, narx_formula_parser, narx_sr_d
modeling/forecasting/nonlinear/llm_narx_sr/   formula_scorer, term_library, search_arm, search_loop, genetic_proposer,
                                     llm_proposer, explainer, baseline_arm, pipeline_arm, plausibility_checker, significance (+ dm_test),
                                     llm_narx_sr, protocol (the measurement protocol: error tables, HAC intervals, DM / Wilcoxon, the runner)
modeling/forecasting/neuralforecasting/  mhattn_regressor
modeling/optimization/               varpro (VarPro; the OptimizerNL enum, fitNL and joint live in nonlinear/narx_sr_d.py as in Scala)
Util/                                make_matrix4ts (MakeMatrix4TS, centered), tools (banner, VectorD/Array printing), transform (Transform.scala),
                                     correlogram (Correlogram.scala + VectorD statistics), data_loader
data/                                covid_19_weekly.csv, ILINetNov25.csv, ETTh1.csv, gefcom2014_load_daily.csv
Examples/                            Example_Covid, Example_ILI, Example_ETTh1, Example_GEFCom, Example_FeatureSelection, Example_LLM_NARX_SR,
                                     Example_RankedSelection, Example_Protocol (the "before" tables)
results/protocol/                    the "before" tables (errors / summary / comparisons per dataset, CSV)
tests/                               261 pytest tests (fit, base API, parsers, formulas, models, feature selection, scorer, datasets, search layer,
                                     the validation switch, the knobs (tune), the protocol, the attention ranker)
docs/                                CONVENTIONS.md (the ScalaTion -> Python map and rules), SCALA_SOURCES.md
```

## License

MIT License

## Author

Yousef Fekri Dabanloo
March 2025 -- October 2026
