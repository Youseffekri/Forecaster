"""
Feature Selection Module: Techniques, Best Step and the Selection Report

The Python counterpart of ScalaTion's `modeling/FeatureSelection.scala`:

    SelectionTech     the techniques: Forward, Backward, Stepwise, Beam, CEM, Ranked
    newFname          the feature names restricted to the selected columns
    qk, slack_base    the "given" instances: the QoF index the models are compared on (sMAPE-IC)
                      and the slack base of the stepwise rule (+25 when qk is maximized, -25 otherwise)
    FeatureSelection  the trait: `selectFeatures` dispatches on the technique to `forwardSelAll`,
                      `backwardElimAll`, `stepwiseSelAll`, `beamSelAll`, `cemSelAll`, `rankedElimAll`
                      (the subclasses
                      implement them: `Forecaster_Reg` in `modeling/forecasting/forecaster_reg.py`); its class
                      attribute `fullset_FS` is ScalaTion's `FeatureSelection` object (whether the
                      selection sees the FULL dataset or the TRAINING rows only)
    SelectionReport   `reportSelection`: prints every selected column set (size, columns, features,
                      QoF trace) and tests the selected model(s) through the two hooks
                      `selectedModel` and `testSelected`
    BestStep          the best improvement step found so far (column, QoF, model, columns)
    updateQoF         appends the key-metrics row of a step to the trace (`qofVector`)
    qofVector         ScalaTion's `Fit.qofVector`: 100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv or sMAPE-IC

A column set is a `list` of column indices in insertion order (ScalaTion's `LinkedHashSet`) and
prints as "LinkedHashSet(0, 1, 3)" (`Util.tools.set_str`).  No plotting (ScalaTion's `PlotM` of the
trace is not reproduced; the `plot` argument is accepted and ignored).

Author: Yousef Fekri Dabanloo
Date: 2026-10-17
Version: 1.4
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Util.tools import banner, set_str, vector_str
from modeling.fit import MAXI, N_QoF, QoF, extreme, showFitMap

#: A column set: the selected columns in insertion order (ScalaTion's `LinkedHashSet [Int]`).
Cols = List[int]


class SelectionTech(Enum):
    """The available feature selection techniques (ScalaTion's `SelectionTech`)."""
    Forward = 0         # Forward Selection (add best x_j)
    Backward = 1        # Backward Elimination (remove worst x_j)
    Stepwise = 2        # Stepwise Selection (best step add/remove)
    Beam = 3            # Beam Search (improve multiple candidate sub-models)
    CEM = 4             # Cross-Entropy Method (see `CEM_Selector`)
    Ranked = 5          # Ranked Elimination (remove the lowest-RANKED x_j; see `SubsetSearch.ranked`)

    def __str__(self) -> str:
        return self.name


def newFname(fname: Sequence[str], cols: Sequence[int]) -> List[str]:
    """
    The feature names restricted to the selected columns (ScalaTion's `newFname`).

    Parameters
    ----------
    fname : sequence of str
        The original/full set of feature names.
    cols : sequence of int
        The selected columns.
    """
    return [fname[j] for j in cols]


# G I V E N S  (change as needed, as the given instances of the Scala file)

#: The QoF metric index used for comparing models by default (ScalaTion's given `qk`):
#: sMAPE-IC for time series (`QoF.rSqBar.ordinal` for regression, `QoF.aic.ordinal` for either).
qk: int = QoF.smapeC.ordinal

#: The slack base of stepwise selection (ScalaTion's given `slack_base`): new vs. old,
#: +25 when a high value of qk is good (R^2 bar), -25 when a low value is good (sMAPE-IC).
slack_base: float = 25.0 if qk in MAXI else -25.0


def given_qk(k: Optional[int] = None) -> int:
    """The QoF index to compare on: `k` when given, else the package default `qk` (the given instance)."""
    return qk if k is None else int(k)


#: The labels of the key metrics of a trace row (ScalaTion's `Regression.metrics`).
METRICS = ["R^2", "R^2 bar", "sMAPE", "R^2 cv"]


def qofVector(fit: np.ndarray, cv_fit: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The key metrics of a model as one row of a selection trace (ScalaTion's `Fit.qofVector`):
    100 R^2, 100 R^2 bar, sMAPE and 100 R^2 cv (the mean R^2 of the cross-validation folds), or,
    when no cross-validation was run, the sMAPE-IC.

    Parameters
    ----------
    fit : np.ndarray, shape (N_QoF,)
        The QoF vector of the model (on the training/FS rows).
    cv_fit : np.ndarray, optional
        The QoF vectors of the cross-validation folds (one row per fold) or their mean; None => no cv.
    """
    fit = np.asarray(fit, dtype=float)
    if cv_fit is None:
        cv = fit[QoF.smapeC.ordinal]                                   # cv not computed => use sMAPE_IC
    else:
        cv = 100.0 * float(np.mean(np.atleast_2d(np.asarray(cv_fit, dtype=float))[:, QoF.rSq.ordinal]))
    return np.array([100.0 * fit[QoF.rSq.ordinal], 100.0 * fit[QoF.rSqBar.ordinal], fit[QoF.smape.ordinal], cv])


qofVectorSize = 4                                                      # must correspond to the size of qofVector


# ---------------------------------------------------------------------------------------------
# BestStep
# ---------------------------------------------------------------------------------------------

class BestStep:
    """
    The best improvement step found so far during feature selection (ScalaTion's `BestStep`);
    "best" depends on whether the metric qk is maximized (R^2, R^2 bar) or minimized (the rest).

    Parameters
    ----------
    col : int, default=-1
        The column/variable to ADD/REMOVE for this step (-1 => none).
    qof : np.ndarray, optional
        The Quality of Fit (QoF) vector for this step (None => the fit failed / no step yet).
    mod : object, optional
        The model including the selected features/variables for this step (the sub-model).
    mod_cols : list of int, optional
        The columns selected for `mod`.
    qk : int, optional
        The index of the QoF measure used for comparison (defaults to the given `qk`).
    bestq : float, optional
        The best QoF for metric qk so far (defaults to `Fit.extreme (qk)`; informational).
    """

    def __init__(self, col: int = -1, qof: Optional[np.ndarray] = None, mod: Any = None,
                 mod_cols: Optional[Sequence[int]] = None, qk: Optional[int] = None,
                 bestq: Optional[float] = None) -> None:
        self.col = int(col)
        self.qof = None if qof is None else np.asarray(qof, dtype=float)
        self.mod = mod
        self.mod_cols: Optional[Cols] = None if mod_cols is None else list(mod_cols)
        self.qk = given_qk(qk)
        self.bestq = extreme(self.qk) if bestq is None else float(bestq)

    def gt(self, that_qof: float) -> bool:
        """Whether this step is better than a step with QoF value `that_qof` (strictly)."""
        v = float(self.qof[self.qk])
        return v > that_qof if self.qk in MAXI else v < that_qof

    def ge(self, that_qof: float) -> bool:
        """Whether this step is better than or equal to a step with QoF value `that_qof`."""
        v = float(self.qof[self.qk])
        return v >= that_qof if self.qk in MAXI else v <= that_qof

    def better(self, cand_or_j, qof_j: Optional[np.ndarray] = None, mod_j: Any = None,
               mod_cols: Optional[Sequence[int]] = None) -> "BestStep":
        """
        The better of this step and a candidate: `better (cand)` with a `BestStep`, or
        `better (j, qof_j, mod_j, mod_cols)` with the parts of the candidate to be formed.
        """
        if isinstance(cand_or_j, BestStep):
            cand = cand_or_j
        else:
            cand = BestStep(int(cand_or_j), qof_j, mod_j, mod_cols, self.qk, float(np.asarray(qof_j)[self.qk]))
        if self.qof is None:
            return cand
        return cand if cand.gt(float(self.qof[self.qk])) else self

    def compare(self, that: "BestStep") -> int:
        """The comparison of this and that step on metric qk: -1, 0 or 1 (ScalaTion's `compare`)."""
        if self.qof is None and that.qof is None:
            return 0
        a, b = float(self.qof[self.qk]), float(that.qof[self.qk])
        return (a > b) - (a < b)

    def __lt__(self, that: "BestStep") -> bool:
        return self.compare(that) < 0

    def __repr__(self) -> str:
        q = "None" if self.qof is None else f"{float(self.qof[self.qk])}"
        cols = "None" if self.mod_cols is None else set_str(self.mod_cols)
        return f"BestStep(col = {self.col}, qof({self.qk}) = {q}, mod_cols = {cols})"


def updateQoF(rSq: List[np.ndarray], cross: str, best: BestStep) -> None:
    """
    Appends the key metrics of a step (`qofVector`) to the trace `rSq` for the next iteration of
    feature selection (ScalaTion's `updateQoF`).

    Parameters
    ----------
    rSq : list of np.ndarray
        The trace: one row (100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv or sMAPE-IC) per step.
    cross : {"many", "one", "none"}
        "many" => cross-validation (`best.mod.crossValidate ()`), "one" => validation
        (`best.mod.validate ()`), anything else => no validation column (the sMAPE-IC is shown).
        The forecasting models provide no cross-validation (a candidate is a forecaster, not a
        regression), so they use "none".
    best : BestStep
        The best step so far.
    """
    if cross == "many":
        if not hasattr(best.mod, "crossValidate"):
            raise NotImplementedError("updateQoF: cross = 'many' needs a model with crossValidate ()")
        rSq.append(qofVector(best.qof, best.mod.crossValidate()))      # results for model mod_l, with cross-validation
    elif cross == "one":
        if not hasattr(best.mod, "validate"):
            raise NotImplementedError("updateQoF: cross = 'one' needs a model with validate ()")
        res = best.mod.validate()
        qof = res[1] if isinstance(res, tuple) else res
        rSq.append(qofVector(best.qof, np.atleast_2d(qof)[0]))        # results for model mod_l, with validation
    else:
        rSq.append(qofVector(best.qof, None))                          # results for model mod_l, with nothing


# ---------------------------------------------------------------------------------------------
# SelectionReport
# ---------------------------------------------------------------------------------------------

class SelectionReport:
    """
    Reports and tests the result of a feature selection the same way for every model family
    (ScalaTion's `SelectionReport` trait): `reportSelection` prints, for each selected column set,
    its size, columns and feature names and its QoF trace, then builds the selected model(s) and
    tests them In-Sample and/or Out-of-Sample, the QoF of every model in ONE table.  What a
    "selected model" is and how it is tested differ per family and are the two hooks
    `selectedModel` and `testSelected`.  The arrays may hold one column set (`selectFeatures`: one
    model) or one per horizon (`selectFeaturesPerH`: one single-horizon model per horizon).

    The host provides `modelName`, `fname`, `getX ()` and `_theBest` (the best step of its last
    selection; None or empty before one).
    """

    def selectedModel(self, cols: Cols, h: int) -> Any:
        """
        Builds the model of a selected column set (hook).

        Parameters
        ----------
        cols : list of int
            The selected columns.
        h : int
            The horizon of a single-horizon selection (0 => none, or all horizons).
        """
        raise NotImplementedError("selectedModel: the model family must provide it")

    def testSelected(self, mod: Any, inSample: bool, tnt: bool, rc: int,
                     showYf: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Tests a selected model (hook) in-sample (train on all the rows, report) and/or out-of-sample
        (a predictor: k-fold cross-validation; a forecaster: rolling validation on the testing rows)
        and returns the two QoF tables -- one row per horizon for a forecaster, one row for a
        predictor; None for a test not run.

        Parameters
        ----------
        mod : object
            The selected model.
        inSample : bool
            Whether to test in-sample.
        tnt : bool
            Whether to test out-of-sample (cross-validation / rolling validation).
        rc : int
            The retraining cycle of a rolling validation (a forecaster).
        showYf : bool
            Whether to print the forecast matrix (a forecaster).
        """
        raise NotImplementedError("testSelected: the model family must provide it")

    def bestCols(self, cols: Cols) -> Cols:
        """
        The column set to build the model from (hook): the set itself, or for a family whose
        selection returns the path (see `FeatureSelection`) the best step along it.
        """
        return cols

    RANKERS = {"coef": "coefImportance", "t": "tImportance", "vif": "vifImportance", "perm": "permImportance"}

    def ranker(self, rank: Any) -> Any:
        """
        The ranking of the Ranked technique: a name -> the host's ranker (re-run after every removal):
        "coef" the standardized coefficients |b_j| sd (x_j) (the default), "t" the t-statistics
        |b_j| / se_j (backward elimination by p-value: the least significant column leaves first; a
        ridge uses its own standard errors), "vif" minus the variance inflation factors (the most
        collinear column leaves first), "perm" the permutation importance (the recursive models);
        a callable or an array passes through, None stays None.
        """
        if not isinstance(rank, str):
            return rank
        meth = self.RANKERS.get(rank.lower())
        if meth is None or not hasattr(self, meth):
            raise ValueError(f"ranker: {rank!r} is not one of {[k for k, v in self.RANKERS.items() if hasattr(self, v)]}")
        return getattr(self, meth)

    def bestHP(self, cols: Cols) -> Optional[Dict[str, float]]:
        """
        The knobs the best step of the last selection tuned for the column set `cols` (under
        `Model.tuning` every candidate is fitted at its own best knobs), as {name: value}; None when
        the set is not the best step's or no selection has run -- so `getModel (getBest ().mod_cols)`
        and `selectedModel` build the selected model at the knobs that made its score (ScalaTion's
        `bestHP`; rev2.2: was `bestLambda`).
        """
        best = getattr(self, "_theBest", None)
        if best is None or best.mod is None or best.mod_cols is None or list(best.mod_cols) != list(cols):
            return None
        knobs = getattr(best.mod, "knobs", ())
        if not knobs:
            return None
        h = best.mod.hparameter
        return {k.name: float(h[k.name]) for k in knobs}

    def colName(self, j: int) -> str:
        """The name of column j for the report: the feature name, or x_j when the model names fewer columns."""
        fn = self.fname
        return fn[j] if fn is not None and j < len(fn) else f"x{j}"

    def reportSelection(self, colsArr: Sequence[Cols], rSqArr: Sequence[Sequence[np.ndarray]], h: int = 0,
                        inSample: bool = True, tnt: bool = True, rc: int = 5, showTrace: bool = True,
                        plot: bool = True, showYf: bool = False
                        ) -> Tuple[List[Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reports and tests a feature selection (ScalaTion's `reportSelection`): prints, for each
        selected column set, its size k (of n), columns and feature names and its QoF trace per
        step (`qofVector`: 100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv or sMAPE-IC), then builds the
        model(s) of the best set(s) and tests them (see the hooks); returns the model(s) and the
        in-sample and out-of-sample QoF tables (None when not run), row h-1 = horizon h for a
        forecaster.  One set => one model; one set per horizon (hh sets) => one single-horizon
        model per horizon, whose QoF rows are assembled into the table and printed.

        Parameters
        ----------
        colsArr : sequence of list of int
            The selected column set(s): one, or one per horizon.
        rSqArr : sequence of list of np.ndarray
            The QoF trace(s) of the selection(s), one per column set.
        h : int, default=0
            The horizon of a single-horizon selection (`selectFeaturesAtH`); 0 => none / all horizons.
        inSample : bool, default=True
            Whether to test the selected model(s) in-sample.
        tnt : bool, default=True
            Whether to test them out-of-sample (a forecaster's rolling validation).
        rc : int, default=5
            The retraining cycle of a rolling validation.
        showTrace : bool, default=True
            Whether to print the QoF trace(s).
        plot : bool, default=True
            Accepted for signature parity; nothing is plotted in the Python port.
        showYf : bool, default=False
            Whether to print the forecast matrices.

        Returns
        -------
        (list, np.ndarray or None, np.ndarray or None)
            The selected model(s), the in-sample QoF table (hh, N_QoF) and the out-of-sample one.
        """
        if len(colsArr) != len(rSqArr):
            raise ValueError("reportSelection: one trace per column set")
        if len(colsArr) < 1:
            raise ValueError("reportSelection: no column set")
        perH = len(colsArr) > 1                                          # one single-horizon model per horizon
        if h != 0 and perH:
            raise ValueError("reportSelection: a single-horizon selection has one column set")
        n = self.getX().shape[1]

        def bestOf(cols: Cols) -> Cols:                                  # rev2.3: `bestCols` reads the LAST selection's best step,
            return cols if perH else self.bestCols(cols)                 #   so per horizon it returned horizon hh's columns for EVERY
                                                                         #   horizon; a per-horizon set is already its own best step
        models = [self.selectedModel(bestOf(list(colsArr[k])), k + 1 if perH else h)
                  for k in range(len(colsArr))]
        for k in range(len(colsArr)):                                    # report the selection(s), under the SELECTED model's
            # name: its _L tag is the lambda its best step found, which may differ from this model's
            cols = list(colsArr[k])
            best = bestOf(cols)
            what = f" for horizon {k + 1}" if perH else (f" for horizon {h}" if h > 0 else "")
            banner(f"{models[k].modelName}: selected{what}: k = {len(cols)} of n = {n}, cols = {set_str(cols)}")
            if list(best) != cols:
                print(f"best step: k = {len(best)}, cols = {set_str(best)}")
            print("features: " + ", ".join(self.colName(j) for j in sorted(best)))
            if showTrace:
                print("trace per step: 100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv (or sMAPE-IC)")
                for r_ in rSqArr[k]:
                    print(f"    {vector_str(r_)}")
            # plot: ScalaTion's PlotM of the trace is not reproduced

        qofIn: Optional[np.ndarray] = None
        qofTnT: Optional[np.ndarray] = None
        if not perH:
            qofIn, qofTnT = self.testSelected(models[0], inSample, tnt, rc, showYf)
        else:
            hh = len(colsArr)
            if inSample:
                qofIn = np.zeros((hh, N_QoF))
            if tnt:
                qofTnT = np.zeros((hh, N_QoF))
            for k in range(hh):
                qi, qt = self.testSelected(models[k], inSample, tnt, rc, showYf)
                if inSample:
                    qofIn[k] = np.asarray(qi)[0]
                if tnt:
                    qofTnT[k] = np.asarray(qt)[0]
            if inSample:
                banner(f"{self.modelName}: In-Sample QoF per horizon, one model per horizon")
                print(showFitMap(qofIn.T))
            if tnt:
                banner(f"{self.modelName}: Out-of-Sample QoF per horizon, one model per horizon")
                print(showFitMap(qofTnT.T))
        return models, qofIn, qofTnT


# ---------------------------------------------------------------------------------------------
# FeatureSelection
# ---------------------------------------------------------------------------------------------

class FeatureSelection(SelectionReport):
    """
    The framework for feature selection (ScalaTion's `FeatureSelection` trait): selecting the
    features (a variable x_j, a cross term x_j x_k, a functional form x_j^2) to include in the
    model.  `selectFeatures` dispatches on the technique; the five techniques are implemented by
    the model family (see `Forecaster_Reg` in `modeling/forecasting/forecaster_reg.py`; the direct
    models' own scenarios are in `Forecaster_D`, `modeling/forecasting/forecaster_d.py`).
    Every selection returns ARRAYS -- (list of column sets, list of traces) with ONE element: the
    column set (the path's order for forward/backward, see `importance`) and its QoF trace -- the
    shape `Forecaster_D.selectFeaturesPerH` fills with one element per horizon, so
    `reportSelection` prints and tests every selection alike; the best step along the path is
    `getBest ()`.

    Attributes
    ----------
    fullset_FS : bool (class attribute)
        ScalaTion's `FeatureSelection.fullset_FS` getter/setter: True => feature selection is
        performed on the entire (full) dataset (e.g., for In-Sample Testing); False (the default)
        => restricted to the training set to prevent data leakage and ensure an unbiased
        evaluation on the testing set.  Set it with `FeatureSelection.fullset_FS = True`.
    """

    fullset_FS: bool = False                                         # FS uses: True => full dataset, False => training set

    def bestCols(self, cols: Cols) -> Cols:
        """The columns of the best model of the last selection, if available (else `cols`)."""
        getBest = getattr(self, "getBest", None)
        best = getBest() if callable(getBest) else None
        if best is not None and best.mod_cols is not None and best.qof is not None:
            return best.mod_cols
        return cols

    def selectFeatures(self, tech: SelectionTech, cross: str = "many", first: int = 1, swap: bool = True,
                       rank: Any = None, qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        Performs feature selection to find the most predictive features/variables to have in the
        model, returning the features/variables added and the Quality of Fit (QoF) measures for
        all steps, as one-element lists (ScalaTion's `FeatureSelection.selectFeatures`).

        Parameters
        ----------
        tech : SelectionTech
            The feature selection technique to apply.
        cross : str, default="many"
            Indicator to include the cross-validation/validation QoF measure ("many", "one", "none").
        first : int, default=1
            The first variable to consider for elimination (1 => the intercept x_0 stays in any model).
        swap : bool, default=True
            Whether to allow a swap step (swap out a feature for a new feature in one step).
        rank : callable or array, optional
            The importance per column driving the Ranked technique (None => the model family's own
            ranker, the standardized coefficients re-fit at every step; see `rankedElimAll`).
        qk : int, optional
            The index of the QoF measure to compare on (defaults to the given `qk`).
        """
        qk = given_qk(qk)
        if tech is SelectionTech.Forward:
            return self.forwardSelAll(cross, qk=qk)
        if tech is SelectionTech.Backward:
            return self.backwardElimAll(first, cross, qk=qk)
        if tech is SelectionTech.Stepwise:
            return self.stepwiseSelAll(cross, swap, qk=qk)
        if tech is SelectionTech.Beam:
            return self.beamSelAll(cross, qk=qk)
        if tech is SelectionTech.CEM:
            return self.cemSelAll(cross, first, qk=qk)
        if tech is SelectionTech.Ranked:
            return self.rankedElimAll(rank, first, cross, qk=qk)
        raise ValueError(f"selectFeatures: unknown technique {tech}")

    def forwardSelAll(self, cross: str = "many", qk: Optional[int] = None):
        """FORWARD SELECTION: adds the MOST predictive feature at each step (the model family implements it)."""
        raise NotImplementedError("forwardSelAll is not supported by this model")

    def backwardElimAll(self, first: int = 1, cross: str = "many", qk: Optional[int] = None):
        """BACKWARD ELIMINATION: removes the LEAST predictive feature at each step (the model family implements it)."""
        raise NotImplementedError("backwardElimAll is not supported by this model")

    def stepwiseSelAll(self, cross: str = "many", swap: bool = True, qk: Optional[int] = None):
        """STEPWISE SELECTION: the best of a forward and a backward step, or a swap (the model family implements it)."""
        raise NotImplementedError("stepwiseSelAll is not supported by this model")

    def beamSelAll(self, cross: str = "many", bk: int = 3, qk: Optional[int] = None):
        """BEAM SEARCH SELECTION: keeps the top bk sets and expands them (the model family implements it)."""
        raise NotImplementedError("beamSelAll is not supported by this model")

    def cemSelAll(self, cross: str = "many", first: int = 1, prior: Optional[np.ndarray] = None,
                  nSample: int = -1, maxIter: int = 30, maxCols: int = -1, stream: int = 0,
                  qk: Optional[int] = None):
        """
        CROSS-ENTROPY SELECTION (see `CEM_Selector`): samples feature sets from per-feature
        inclusion probabilities, keeps the elite sets and moves the probabilities towards them;
        the default reports it as unsupported (the model families override it).
        """
        raise NotImplementedError(
            f"cemSelAll (cross = {cross}, first = {first}, prior = {prior}, nSample = {nSample}, "
            f"maxIter = {maxIter}, maxCols = {maxCols}, stream = {stream}, qk = {given_qk(qk)}) "
            "is not supported by this model")

    def rankedElimAll(self, rank: Any = None, first: int = 1, cross: str = "many", qk: Optional[int] = None):
        """
        RANKED ELIMINATION (see `SubsetSearch.ranked`): removes the lowest-RANKED feature at each
        step; the default reports it as unsupported (the model families override it).
        """
        raise NotImplementedError(
            f"rankedElimAll (rank = {rank}, first = {first}, cross = {cross}, qk = {given_qk(qk)}) "
            "is not supported by this model")
