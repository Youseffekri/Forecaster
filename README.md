# Forecaster

A modular time series forecasting framework that integrates ARX models, symbolic regression,
attention-based feature selection and **LLM-NARX-SR**: NARX formulas with free nonlinear
parameters, fitted by variable projection and scored for multi-horizon forecasting.
The Python counterpart of the forecasting package of [ScalaTion](https://github.com/scalation/scalation)
(same factory methods, same documentation conventions).

## Features

- AR and ARX models with customizable lag structures (recursive `ARX` and direct `ARX_D`)
- Symbolic regression with nonlinear transformations (`ARX_Symb`, `ARX_Symb_D`)
- Attention-based feature selection using PyTorch (`MHAttn_Regressor`)
- **Formula language and parser**: `y ~= y~(3,1) + log(? + icu~1) + hinge(temp~1, ?) + wave(1)`
- **`NARX_SR_D`**: a direct multi-horizon NARX model built from a formula; coefficients by ridge,
  free nonlinear parameters by one of three optimizers (`varpro`, `lbfgsb`, `joint`)
- **`FormulaScorer`**: parse -> fit -> score with the parsimony-adjusted validation sMAPE (the
  selection score J), rolling-origin validation blocks inside the training rows, rejection rules
  (parameter at a bound, exponent ~ 1, collinear free-parameter column), rolling test on the
  testing rows
- Four datasets: Covid-19 weekly deaths, ILI (bring your own file), ETTh1 hourly transformer
  temperature, GEFCom2014 daily peak load
- Rolling window validation and quality-of-forecast metrics (MSE, MAE, R², adjusted R², sMAPE)

## Getting Started

```bash
git clone https://github.com/Youseffekri/Forecaster.git
cd Forecaster
pip install -e .            # numpy, pandas, scipy, scikit-learn, statsmodels
pip install -e .[neural]    # + torch, for the neural models
pip install -e .[test]      # + pytest
pytest                      # 46 tests, ~3 s
python Examples/Example_LLM_NARX_SR.py           # covid | etth1 | gef14 | optimizers | all
python Examples/Example_Covid19.py
```

`pip install -e .` makes `modeling` and `Util` importable from anywhere; the example scripts also
run from the repository root without it.

## Models

- `AR_YW`: autoregressive model using Yule-Walker estimation
- `ARX`, `ARX_D`: AR model with exogenous inputs (recursive / direct multi-horizon)
- `ARX_Symb`, `ARX_Symb_D`: symbolic-regression ARX (nonlinear transformations of the lags)
- `NARX_SR_D`: direct multi-horizon NARX symbolic regression from a formula (below)
- `MHAttn_Regressor`: multi-head attention for feature selection

## LLM-NARX-SR (core layer)

The structure of a forecasting model is written as a formula over a named-column dataset
(`{"y": ..., "icu": ..., "pRate": ...}`), parsed, fitted and scored by one engine:

```python
from modeling.forecasting.llm_narx_sr.datasets import Datasets
from modeling.forecasting.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.narx_sr_d import NARX_SR_D

data = Datasets.GEF14.load()                                   # daily peak load, temp, tmax
sc   = FormulaScorer(data, hh=6, hp={"maxLag": 7, "skip": 7})  # the engine: training / validation rows, J
cand = sc.score("y ~= y~(7,1) + hinge(temp~1, ?)")             # parse -> fit -> score
print(cand.formula_str, cand.score)                            # the threshold fitted at ~65.8 F, J
model, qof = sc.tnt(cand.formula.fresh())                      # rolling validation on the testing rows

m = NARX_SR_D.from_formula("y ~= y~(7,1) + hinge(temp~1, ?)", data, hh=6, lam=1.0, scaled=True,
                           optimizer="lbfgsb", max_lag=7)      # the model alone, another optimizer
m.inSample_Test(); m.trainNtest_Test()                         # the base class's tests
```

### Formula grammar

```
formula  := ID "~=" element { "+" element } [ "TFORM:" ID ]      # TFORM: Log1p | Log | Sqrt
element  := atom { "*" atom }                                     # at most two factors
atom     := x~k | x~(k1,k2)                                       # a lag, a lag range (one term per lag)
          | log|log1p|sqrt|sq|exp|expm1|sin|cos|tanh ( atom )     # a function of a term
          | FN( ? + atom ) | FN( ? * atom )                       # a free shift / rate inside a function
          | pow(atom, 1.5) | pow(atom, ?) | powr(atom, 1.5)       # fixed, free, sign-safe exponent
          | sat(atom, ?) | hinge(atom, ?) | trans(atom, ?, ?) | ema(atom, ?) | dq(atom, ?)
          | wave(k) | wave(?)                                     # k periodogram sin/cos pairs, or a free frequency
?        := a free parameter; ?w a free parameter starting at w; a number a constant
```

A lag range shares one parameter (`pow(y~(3,1), ?)` is three terms, one exponent).  Every
free parameter has a role (shift, power, decay, threshold, ...) that fixes its default start
value and bounds from the training rows (`modeling/forecasting/formula.py`, `ParRole`).

### Fitting and scoring

| step | where | what |
|---|---|---|
| parse | `formula_parser.parse_formula` | string -> `Formula`; rejects carry a reason (unknown variable, lag out of range, `powr(?)`, ...) |
| fit | `NARX_SR_D.from_formula` | free parameters fitted once on the rows whose targets lie in the training rows (`optimizer`); coefficients by ridge per horizon (centered, unpenalized intercept); each fitted parameter counts as a degree of freedom |
| score | `FormulaScorer.score` | J_h = sMAPE_h(validation) × (1 + 2p/n), p = terms + 1 + fitted parameters; validation = the last 25 % of the training rows in 3 rolling-origin blocks (`valFrac`, `valFolds`); mean over horizons |
| reject | `FormulaScorer.rejection` | > `maxFree` free parameters; a parameter within `boundTol` of its box; an exponent within 0.05 of 1; a free-parameter column with VIF > `maxVIF` |
| test | `FormulaScorer.tnt` | the model refitted on all training rows, then `rollValidate(rc)` on the testing rows (sliding window, retrain every `rc` rows) |

### Optimizer options (`optimizer=` of `NARX_SR_D.from_formula`, `hp["optimizer"]` of the scorer)

| option | method | notes |
|---|---|---|
| `varpro` (default) | variable projection: scipy `least_squares` (trust region reflective, bounds) on the reduced residual, the coefficients eliminated by ridge | few evaluations; the one to use |
| `lbfgsb` | scipy `minimize` L-BFGS-B on the reduced objective, bounds honoured | a check on VarPro; can land on another local optimum |
| `joint` | one `least_squares` over coefficients and parameters together | the "one optimizer over everything" alternative; same optimum when the columns are standardized (`scaled=True`), many more evaluations when not |

All three live in `modeling/optimization/varpro.py` (`VarPro.fit(a0, optimizer=...)`).

### Datasets (`modeling/forecasting/llm_narx_sr/datasets.py`)

| config | file | rows | response | drivers |
|---|---|---|---|---|
| `Datasets.COVID2`, `COVID` | `data/covid_19_weekly.csv` | 116 weeks | new deaths | icu, pRate / hosp, icu |
| `Datasets.ILI` | `data/national_illness_clip.csv` (not shipped) | — | ILI patient count | wili, ot |
| `Datasets.ETTH1`, `ETTH1_1Y` | `data/ETTh1.csv` (Zhou et al. 2021) | 17,420 / 8,760 hours | OT (oil temperature) | hufl, hull, mufl, mull, lufl, lull |
| `Datasets.GEF14` | `data/gefcom2014_load_daily.csv` (Hong et al. 2016; built by `tools/build_gefcom_daily.py`) | 2,099 days | peak load | temp, tmax |

Each configuration carries the response, the pool of exogenous variables, a domain sentence,
the expected coefficient signs and a loader (`cfg.load()` -> the named-column map).  OT goes
negative: use `hp={"nneg": False}` / `nneg=False` for ETTh1.

### What the example shows (`Examples/Example_LLM_NARX_SR.py`)

On GEFCom2014 the free hinge threshold of `y ~= y~(7,1) + hinge(temp~1, ?)` is fitted at 65.8 F by
all three optimizers -- the cooling point of the V-shaped load-temperature curve (mean peak per
10 F band 272, 226, 185, 148, 136, 214, 273).  The 21-term lag baseline gives a rolling TnT sMAPE of
12.78 (mean over h = 1..6), the hinge formula with one seasonal pair 12.99 with 10 terms.  On
Covid-19 a shift inside a log and a decay in an ema run to their bounds and are rejected, which is
what the rules are for.

### Not in this layer (next)

The search layer of LLM-NARX-SR -- the LLM / genetic / random proposers, the term library with
cross-entropy selection, the arms and the grid harness, the significance tests -- builds on
`FormulaScorer` and `Datasets` and is the next delivery.

## Evaluation Metrics

- Mean Squared Error (MSE), Mean Absolute Error (MAE)
- Coefficient of Determination (R² and adjusted R²)
- Symmetric Mean Absolute Percentage Error (sMAPE); the selection score J is its FPE-form parsimony adjustment

## Repository layout

```
modeling/forecasting/            forecaster.py, forecaster_d.py, ar_yw.py, arx.py, arx_d.py, arx_symb.py, arx_symb_d.py
                                 formula.py, formula_parser.py, narx_sr_d.py        (LLM-NARX-SR core: formula layer)
modeling/forecasting/llm_narx_sr/  formula_scorer.py, datasets.py, eda.py            (the engine, the datasets, periodogram frequencies)
modeling/forecasting/neuralforecasting/  mhattn_regressor.py
modeling/optimization/           varpro.py                                          (VarPro, L-BFGS-B, joint)
Util/                            build_matrix.py, tools.py, data_loader.py
data/                            covid_19_weekly.csv, ETTh1.csv, gefcom2014_load_daily.csv
tools/                           build_gefcom_daily.py
Examples/                        Example_Covid19.py, Example_LLM_NARX_SR.py
tests/                           pytest suite (parser, model and optimizers, scorer, datasets)
```

## License

MIT License

## Author

Yousef Fekri Dabanloo
March 2025 -- October 2026
