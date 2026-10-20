# The "before" tables of the measurement protocol (rev2.2, Oct 20, 2026)

`python Examples/Example_Protocol.py all --arms`: every row of the last 40 % of each series is a forecast origin (refit every
rc = 5 rows, h = 1..6), the base models at the ridge penalty lambda = 1 (raw columns) and the deterministic arms of the
harness (the validation J, `Model.tuning` off; ScalaTion rev5.3 / Python rev2.2 anchors).  Per dataset: `<name>_errors.csv`
(the long table: dataset, model, stream, t, h, y, yhat, e, ae, sape, ase), `<name>_summary.csv` (per model and horizon, h = 0
= all: n origins, sMAPE and MASE with the 95 % HAC interval) and `<name>_comparisons.csv` (every model against the random
walk on the paired per-origin sape: Diebold-Mariano with the HLN correction, Wilcoxon signed-rank).  MASE scales by the
naive one-step MAE of the rows before the testing span (so a series whose level grew, ILI, has large MASEs).

The all-horizon rows (the mean over h = 1..6 of the per-horizon means; a negative DM* with a small p says the model
beats the random walk):

## Covid-19 (icu, pRate), 116 weeks

| model | origins | sMAPE | 95 % interval | MASE | diff vs RW | DM* | p (DM) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|---|
| ARX_D_6_6 ridge L1 | 44 | 26.264 | [13.51, 39.02] | 1.319 | -15.339 | -3.74 | 0.0002 | 0.0000 |
| ARX_6_6 ridge L1 | 44 | 36.849 | [21.86, 51.84] | 1.890 | -5.004 | -0.91 | 0.3641 | 0.0002 |
| AR_6 | 44 | 38.248 | [29.09, 47.41] | 2.721 | -3.402 | -0.66 | 0.5082 | 0.1450 |
| RandomWalk | 44 | 41.800 | [28.47, 55.13] | 2.888 | -- | -- | -- | -- |
| baseline | 44 | 43.458 | [17.31, 69.60] | 2.621 | +1.917 | +0.28 | 0.7785 | 0.3343 |
| tuned-arx | 44 | 32.933 | [19.46, 46.40] | 3.620 | -8.496 | -1.67 | 0.0969 | 0.0033 |
| sel-arx | 44 | 42.464 | [23.52, 61.41] | 3.708 | +1.421 | +0.26 | 0.7960 | 0.5246 |
| random-walk | 44 | 41.800 | [28.47, 55.13] | 2.888 | +0.000 | +0.00 | 1.0000 | 1.0000 |

## ILI, 1208 weeks

| model | origins | sMAPE | 95 % interval | MASE | diff vs RW | DM* | p (DM) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|---|
| ARX_D_6_6 ridge L1 | 481 | 26.899 | [23.77, 30.02] | 9.724 | +0.260 | +0.26 | 0.7932 | 0.0048 |
| ARX_6_6 ridge L1 | 481 | 26.604 | [23.71, 29.50] | 9.800 | -0.034 | -0.04 | 0.9662 | 0.0049 |
| AR_6 | 481 | 24.718 | [21.61, 27.82] | 9.304 | -1.919 | -2.31 | 0.0210 | 0.0000 |
| RandomWalk | 481 | 26.639 | [23.98, 29.30] | 10.719 | -- | -- | -- | -- |
| baseline | 481 | 26.704 | [23.77, 29.64] | 9.746 | +0.066 | +0.07 | 0.9436 | 0.0172 |
| tuned-arx | 481 | 24.966 | [21.72, 28.21] | 9.436 | -1.672 | -1.91 | 0.0562 | 0.0000 |
| sel-arx | 481 | 22.773 | [19.97, 25.58] | 9.080 | -3.861 | -5.85 | 0.0000 | 0.0000 |
| random-walk | 481 | 26.639 | [23.98, 29.30] | 10.719 | +0.000 | +0.00 | 1.0000 | 1.0000 |

## GEFCom2014 daily load (temp, tmax), 2099 days, lags 1..7

| model | origins | sMAPE | 95 % interval | MASE | diff vs RW | DM* | p (DM) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|---|
| ARX_D_7_7 ridge L1 | 837 | 13.314 | [12.40, 14.23] | 1.450 | -2.131 | -8.72 | 0.0000 | 0.0000 |
| ARX_7_7 ridge L1 | 837 | 13.831 | [12.89, 14.77] | 1.495 | -1.615 | -5.68 | 0.0000 | 0.0000 |
| AR_7 | 837 | 13.892 | [13.01, 14.77] | 1.519 | -1.554 | -7.28 | 0.0000 | 0.0000 |
| RandomWalk | 837 | 15.446 | [14.37, 16.52] | 1.687 | -- | -- | -- | -- |
| baseline | 837 | 13.301 | [12.40, 14.20] | 1.448 | -2.145 | -8.87 | 0.0000 | 0.0000 |
| tuned-arx | 837 | 13.418 | [12.53, 14.31] | 1.462 | -2.028 | -8.48 | 0.0000 | 0.0000 |
| sel-arx | 837 | 13.468 | [12.58, 14.36] | 1.467 | -1.977 | -8.26 | 0.0000 | 0.0000 |
| random-walk | 837 | 15.446 | [14.37, 16.52] | 1.687 | +0.000 | +0.00 | 1.0000 | 1.0000 |

## ETTh1 first year (auto exo), 8760 hours

| model | origins | sMAPE | 95 % interval | MASE | diff vs RW | DM* | p (DM) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|---|
| ARX_D_6_6 ridge L1 | 3502 | 11.720 | [10.86, 12.58] | 1.670 | -1.206 | -11.83 | 0.0000 | 0.0000 |
| ARX_6_6 ridge L1 | 3502 | 12.068 | [11.21, 12.93] | 1.748 | -0.858 | -8.72 | 0.0000 | 0.0000 |
| AR_6 | 3502 | 12.713 | [11.85, 13.57] | 1.929 | -0.213 | -4.72 | 0.0000 | 0.0000 |
| RandomWalk | 3502 | 12.927 | [12.03, 13.82] | 1.961 | -- | -- | -- | -- |
| baseline | 3502 | 12.394 | [11.53, 13.26] | 1.856 | -0.532 | -8.71 | 0.0000 | 0.0000 |
| tuned-arx | 3502 | 12.353 | [11.49, 13.22] | 1.853 | -0.574 | -9.35 | 0.0000 | 0.0000 |
| sel-arx | 3502 | 12.342 | [11.48, 13.21] | 1.851 | -0.584 | -9.55 | 0.0000 | 0.0000 |
| random-walk | 3502 | 12.927 | [12.03, 13.82] | 1.961 | +0.000 | +0.00 | 1.0000 | 1.0000 |

Reading: on Covid-19 (47 origins) only the ridge `ARX_D` beats the random walk clearly (DM* -3.7); the seed formula
(`baseline`, standardized columns) and `sel-arx` do not, and `sel-arx`'s h = 1 is worse than the random walk (DM* +4.3) --
the selection variance.  On ILI (481 origins) `sel-arx` is the best arm (DM* -5.9) and the plain lag models tie the random
walk; on GEFCom2014 (837 origins) every model beats the random walk by about two sMAPE points (DM* -6 .. -9) and the arms
add nothing over the ridge baseline; on ETTh1 (3502 origins) the lags of y carry everything and the loads add little.
These are the tables every later stage is measured against (roadmap Stage 1, item 2.2.4).

The long tables of GEFCom2014 and ETTh1 (6 and 27 MB) are not in the zip; `python Examples/Example_Protocol.py gef14 --arms`
(33 s) and `etth1 --arms` (6.5 min) rewrite them.
