"""
Forecaster_D Module

Defines the abstract `Forecaster_D` class for DIRECT (non-recursive) multi-horizon forecasting --
the Python counterpart of ScalaTion's `Forecaster_D.scala`: one regression per horizon on the
same input matrix, the targets held in the matrix Y (column h - 1 = y shifted by h - 1, padded
with NaN where the future is unknown, ScalaTion's NO_DOUBLE).  As in ScalaTion the base holds the
regression (`_make_model`: "sm_ols", "sk_lr" or the centered ridge "ridge" of `CenteredRidgeMV`)
and fits it (`train`: `fit_rows` drops the padded rows first), exposes `intercept`, `slopes`, `bb`
and the coefficient tables `summary` / `summary (h)`; the model classes implement `forecast`,
`getModel` and `getModel_h`.

The FEATURE SELECTION scenarios of ScalaTion's `Forecaster_D` live here as well (the section
"F E A T U R E   S E L E C T I O N" of the Scala file):

    selectFeatures      ONE column set for all the horizons (h = 0: every candidate set is fit as ONE
                        multi-horizon model and scored over ALL horizons by `agg`, the mean FPE-form
                        sMAPE J by default; h >= 1: the single-horizon model at h decides)
    selectFeaturesAtH   the column set of the single-horizon model at h (`getModel_h`, an `ARX` whose
                        `Forecaster_Reg` selection runs)
    selectFeaturesPerH  one column set PER HORIZON
    reportSelection     prints and tests any of them, in-sample and/or TnT (`SelectionReport`)

The model classes provide `getModel (cols)` (the direct sub-model) and `getModel_h (cols, h)` (the
single-horizon model at h); `smapeFPE` / `aggFPE` are the companion object's functions.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import statsmodels.api as sm

from Util.make_matrix4ts import makeMatrix4Y
from Util.tools import array_str, banner, set_str
from modeling.feature_selection import (BestStep, Cols, FeatureSelection, SelectionReport, SelectionTech, given_qk,
                                        updateQoF)
from modeling.fit import N_QoF, Fit, QoF, vif
from modeling.forecasting.forecaster import Forecaster, trSize
from modeling.subset_search import SubsetSearch


# ---------------------------------------------------------------------------------------------
# The companion object: the FPE-form sMAPE and its aggregation over the horizons
# ---------------------------------------------------------------------------------------------

def smapeFPE(qof: np.ndarray) -> np.ndarray:
    """
    The FPE-form parsimony-adjusted sMAPE per horizon (ScalaTion's `Forecaster_D.smapeFPE`),

        sMAPE_h (1 + 2 p / n_h),  p = |S| + 1 (the slopes and the intercept),

    n_h the rows diagnosed at horizon h (the `dfr` and `m` slots of the QoF matrix).  Akaike's
    FPE = sigma^2 (n + p) / (n - p) ~ sigma^2 (1 + 2p/n) charges a term in proportion to the loss,
    so the weight of the penalty against the loss is the same at every error level.

    Parameters
    ----------
    qof : np.ndarray, shape (hh, N_QoF)
        The QoF matrix, one row per horizon.
    """
    qof = np.asarray(qof, dtype=float)
    p = qof[:, QoF.dfr.ordinal] + 1.0
    n = qof[:, QoF.m.ordinal]
    return qof[:, QoF.smape.ordinal] * (1.0 + 2.0 * p / n)


def aggFPE(qof: np.ndarray) -> np.ndarray:
    """
    Aggregates a QoF matrix over the horizons for `selectFeatures` (h = 0) (ScalaTion's
    `Forecaster_D.aggFPE`): the mean of every measure, with the sMAPE-IC slot holding the mean
    FPE-form sMAPE over the horizons (`smapeFPE`), the selection score J of the LLM-NARX-SR search;
    with qk = smapeC the joint selection therefore minimizes J.

    Parameters
    ----------
    qof : np.ndarray, shape (hh, N_QoF)
        The QoF matrix, one row per horizon.
    """
    qof = np.asarray(qof, dtype=float)
    v = qof.mean(axis=0)
    v[QoF.smapeC.ordinal] = float(np.mean(smapeFPE(qof)))
    return v


class Forecaster_D(SelectionReport, Forecaster):
    """
    Abstract base class for DIRECT multi-horizon forecasting models.

    Attributes
    ----------
    Y : np.ndarray, shape (m, hh)
        The target matrix: Y[t, h - 1] = y[t + h - 1] (NaN beyond the end of the series).
    method : {"sm_ols", "sk_lr", "ridge"}
        The regression per horizon (ScalaTion's `reg`, a `CenteredRidgeMV`, is "ridge").
    params : np.ndarray, shape (n + 1, hh)
        The parameters per horizon, intercept first (original units).
    bb : np.ndarray, shape (n, hh)
        The coefficients per horizon (ScalaTion's `bb`, the regression's own: standardized when
        "ridge" is scaled); `parameter` is its first column, `slopes` the original-unit ones.
    """

    def __init__(self, hparam: Dict[str, Any], y: np.ndarray, hh: int, method: str = "ridge",
                 scaled: bool = False, hasIntercept: bool = True) -> None:
        """
        Initializes the Forecaster_D (hparam, y, hh as in `Forecaster`); the target matrix Y is built
        from y.

        Parameters
        ----------
        method : {"sm_ols", "sk_lr", "ridge"}, default="ridge"
            The regression per horizon.
        scaled : bool, default=False
            Whether the ridge regression standardizes the columns (ScalaTion's `scaled`).
        hasIntercept : bool, default=True
            Whether the model has an intercept (counted in the degrees of freedom).
        """
        super().__init__(hparam, y, hh, hasIntercept)
        self._method = method
        self._scaled = scaled
        self._model = self._make_model()
        self._Y: Optional[np.ndarray] = makeMatrix4Y(self._y, hh)
        self._theBest: Optional[BestStep] = None                          # the best step of the last selection
        self._lastProbs: Optional[np.ndarray] = None                      # the last CEM selection's probabilities

    @property
    def Y(self) -> np.ndarray:
        """The multi-horizon target matrix (m, hh), NaN where the future is unknown."""
        return self._Y

    def getYy(self) -> np.ndarray:
        """ScalaTion's name for `Y`."""
        return self._Y

    @property
    def bb(self) -> np.ndarray:
        """The coefficient matrix (n, hh), one column per horizon (ScalaTion's `bb` = `reg.parameter`)."""
        if self._b is not None:
            return self._b
        return self.slopes

    @property
    def slopes(self) -> np.ndarray:
        """The slopes in ORIGINAL units (n, hh), one column per horizon, so that forecast (t) =
        intercept + X[t] @ slopes (ScalaTion's `slopes`)."""
        p = self.params
        return p[1:, :] if self._hasIntercept else p

    @staticmethod
    def fit_rows(Y_: np.ndarray, X_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The rows usable for training: those before the first NaN in the last horizon column
        (ScalaTion's `Forecaster_D.train` drops the trailing rows whose last horizon is undefined).

        Parameters
        ----------
        Y_ : np.ndarray, shape (m_, hh)
            The target matrix (window).
        X_ : np.ndarray, shape (m_, n)
            The input matrix (window).

        Returns
        -------
        (np.ndarray, np.ndarray)
            The trimmed (Y_, X_).
        """
        bad = np.where(np.isnan(Y_[:, -1]))[0]
        if len(bad) == 0:
            return Y_, X_
        idx = int(bad[0])
        return Y_[:idx], X_[:idx]

    def train(self, Y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the regression of every horizon on an (X_, Y_) window (ScalaTion's `Forecaster_D.train`):
        the trailing rows whose last horizon is undefined are dropped (`fit_rows`), the regression
        refits its centering (and scaling) on the rows that remain, and the parameters (n + 1, hh)
        are stored with the intercept first; `bb` holds the regression's own coefficients.

        Parameters
        ----------
        Y_ : np.ndarray, shape (m_, hh), optional
            The target matrix (the full Y by default).
        X_ : np.ndarray, shape (m_, n), optional
            The input matrix (the full X by default).
        """
        X_ = self._X if X_ is None else X_
        Y_ = self._Y if Y_ is None else Y_
        Y_, X_ = self.fit_rows(Y_, X_)
        self._b = None
        if self._method == "sm_ols":
            Xc = sm.add_constant(X_, has_constant="add")
            self._models_fit = [sm.OLS(Y_[:, j], Xc).fit() for j in range(Y_.shape[1])]
            self._set_params(np.column_stack([np.asarray(f.params) for f in self._models_fit]))
        elif self._method == "ridge":
            self._model.fit(X_, Y_)
            self._set_params(self._model.params)
            self._b = self._model.parameter                              # ScalaTion's bb = reg.parameter
        else:
            self._model.fit(X_, Y_)
            self._set_params(np.column_stack([self._model.intercept_, self._model.coef_]).T)

    def _predict(self, X_rows: np.ndarray) -> np.ndarray:
        """The forecasts (rows, hh) of the given input rows, in model units, not rectified
        (ScalaTion's `reg.predict`)."""
        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        if self._method == "sm_ols":
            Xc = sm.add_constant(X_rows, has_constant="add")
            return np.column_stack([f.predict(Xc) for f in self._models_fit])
        return np.asarray(self._model.predict(X_rows), dtype=float).reshape(len(X_rows), -1)

    def summary(self, x_: Any = None, fname_: Optional[Sequence[str]] = None, b_: Optional[np.ndarray] = None,
                vifs: Optional[np.ndarray] = None) -> str:
        """
        The coefficient table (ScalaTion's `Forecaster_D.summary`): the intercept followed by the
        slopes of horizon 1 in original units, with a column of ones prepended to x so `Fit.summary`'s
        standard errors and VIF indexing (column 0 = intercept) apply.  Reflects the last `diagnose`
        call (the last horizon after `diagnoseAll`).  `summary (h)` -- an int as the first argument,
        as ScalaTion's overload -- gives the table of horizon h.

        Parameters
        ----------
        x_ : np.ndarray, shape (m, n), or int, optional
            The data/input matrix without an intercept column (defaults to X), or the horizon h.
        fname_ : sequence of str, optional
            The feature names (defaults to `fname`).
        b_ : np.ndarray, shape (n + 1,), optional
            The parameters, intercept first (defaults to horizon 1's).
        vifs : np.ndarray, shape (n,), optional
            The Variance Inflation Factors (computed when None).
        """
        if isinstance(x_, (int, np.integer)):                            # summary (h)
            h = int(x_)
            if not 1 <= h <= self.hh:
                raise ValueError(f"summary: horizon h = {h} out of range [1, {self.hh}]")
            return f"\nhorizon h = {h}\n" + self.summary(b_=self.params[:, h - 1])
        x_ = self._X if x_ is None else x_
        fname_ = self.fname if fname_ is None else fname_
        x1 = np.column_stack((np.ones(x_.shape[0]), x_))              # column of ones for the implicit intercept
        b1 = self.params[:, 0] if b_ is None else b_                   # intercept +: slopes of horizon 1
        v1 = vif(x1) if vifs is None else vifs                         # one VIF per slope column
        return Fit.summary(self, x1, ["intercept"] + list(fname_), b1, v1)

    def _train_target(self, is_: int, t: int) -> np.ndarray:
        """The training target rows [is_, t) of the matrix Y."""
        return self._Y[is_:t]

    # -- F E A T U R E   S E L E C T I O N  (ScalaTion's `Forecaster_D`) ---------------------------

    # Note: `selectFeatures` and `selectFeaturesPerH` return LISTS (one column set and trace, or one per
    #       horizon); `selectFeaturesAtH` returns the column set and trace (the model is `getModel_h`);
    #       `reportSelection` prints and tests any of them, in-sample and/or TnT, one QoF table per test

    @property
    def mcols(self) -> Cols:
        """The set of columns (numbers) for the features in this model."""
        return list(range(self.getX().shape[1]))

    def getBest(self) -> BestStep:
        """
        The best step/model of the last feature selection run.  After `selectFeatures` its `mod` is
        the selected multi-horizon model (fitted on the FS rows, feature names kept) and `mod_cols`
        its columns, so `getModel (getBest ().mod_cols)` is the selected forecaster.  After
        `selectFeaturesAtH` (or `selectFeaturesPerH`: its last horizon) it is the best step of
        that single-horizon model.

        Raises
        ------
        RuntimeError
            If no feature selection has run.
        """
        if self._theBest is None:
            raise RuntimeError("Forecaster_D.getBest: no feature selection has run, so theBest is null")
        return self._theBest

    @property
    def cemProbs(self) -> Optional[np.ndarray]:
        """The inclusion probability per column left by the last CEM selection, or None if none ran."""
        return self._lastProbs

    def bestCols(self, cols: Cols) -> Cols:
        """The columns of the best model of the last selection, if available (else `cols`)."""
        best = self._theBest
        if best is not None and best.mod_cols is not None and best.qof is not None:
            return best.mod_cols
        return cols

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "Forecaster_D":
        """
        Builds the direct model on the columns `cols` of the input matrix, e.g., the selected model
        after feature selection, `getModel (getBest ().mod_cols)` (the model classes implement it,
        constructing the sub-model on `X[:, cols]` and finishing it with `_subModel`).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        """
        raise NotImplementedError(f"{self.modelName}.getModel: the model class must implement it")

    def getModel_h(self, cols: Optional[Sequence[int]] = None, h: int = 1):
        """
        Builds the single-horizon model at horizon h on the columns `cols` (the model classes
        implement it, ScalaTion's `ARX_D.getModel_h`: an `ARX` with hh = 1 bound to column h - 1 of Y),
        whose `Forecaster_Reg` selection `selectFeaturesAtH` runs.

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        h : int, default=1
            The horizon (1 to hh).
        """
        raise NotImplementedError(f"{self.modelName}.getModel_h: the model class must implement it")

    def fitCols(self, cols: Sequence[int]) -> Tuple["Forecaster_D", np.ndarray]:
        """
        Fits the multi-horizon model of a column set on the FS rows (the training rows, or all
        rows when `FeatureSelection.fullset_FS`) and returns it with its QoF matrix (one row per
        horizon) on those rows in ORIGINAL space with the first `getSkip ()` rows left out -- the
        evaluation of every candidate set of `selectFeatures` (h = 0) and the fit behind
        `getBest ()` (h >= 1).

        Parameters
        ----------
        cols : sequence of int
            The columns of the set.
        """
        m = len(self._y)
        rng = range(0, m) if FeatureSelection.fullset_FS else range(0, trSize(m, self.TE_RATIO))
        sub = self.getModel(cols)                                         # this model's transforms and scaling
        sub.setSkip(self.getSkip())
        sub.train(sub.Y[rng.start:rng.stop], sub.X[rng.start:rng.stop])
        sub.forecast()                                                    # every row and horizon into Yf
        qof = np.zeros((self.hh, N_QoF))
        y_orig, yf = sub.getY(), sub.Yf
        for h in range(1, self.hh + 1):
            yy = y_orig[rng.start + h - 1: rng.stop]                      # align the actual response values
            yfh = yf[rng.start: rng.stop - h + 1, h]                      # align column h of the forecast matrix
            sub.mod_resetDF(len(yy) - sub.getSkip())
            qof[h - 1] = sub.diagnose(yy, yfh)                            # original space, skip rows dropped
        return sub, qof

    def selectFeatures(self, tech: SelectionTech, cross: str = "none", first: int = 1, swap: bool = True,
                       h: int = 0, prior: Optional[Sequence[float]] = None, stream: int = 0, bk: int = 3,
                       agg: Callable[[np.ndarray], np.ndarray] = aggFPE,
                       qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        Performs feature selection for this multi-horizon model, returning the columns selected and
        the QoF measures for all steps -- as LISTS of ONE column set and ONE trace, the shape
        `selectFeaturesPerH` returns per horizon, so `reportSelection` prints and tests both alike;
        the selected model is `getBest ()` and `getModel (getBest ().mod_cols)` (the forecaster, to
        train or roll-validate).  Two ways to score a candidate column set:

        h = 0  -- ALL HORIZONS JOINTLY: every set is fit as ONE multi-horizon model (`fitCols`: this
                  model's transforms and scaling, the FS rows, per-horizon QoF in original space with
                  the skip rows dropped) and scored by `agg` over the horizons (default `aggFPE`: the
                  mean of every measure, the sMAPE-IC slot holding the mean FPE-form sMAPE, so with
                  qk = smapeC the criterion is the selection score J of the LLM-NARX-SR search).  The
                  trace holds the best-so-far per step (aggregated measures); no cross-validation
                  column; a set whose fit fails is skipped.
        h >= 1 -- AT ONE HORIZON: the single-horizon model at h decides (`selectFeaturesAtH`:
                  `Forecaster_Reg`'s selection on `getModel_h (h)`, so `cross` applies and the trace
                  is that model's) and its columns are shared by every horizon; `getBest ()` then
                  carries that step's QoF with the multi-horizon model of the columns.

        Parameters
        ----------
        tech : SelectionTech
            The selection technique (Forward, Backward, Stepwise, Beam, CEM).
        cross : str, default="none"
            Indicator to include the cross-validation/validation QoF measure (h >= 1 only).
        first : int, default=1
            The number of leading columns always kept (1 => the intercept column 0; 0 => none,
            e.g., a matrix built by `fromTerms`).
        swap : bool, default=True
            Whether stepwise may swap a column in for one out.
        h : int, default=0
            The horizon whose single-horizon model decides (1 to hh), or 0 => all horizons jointly.
        prior : sequence of float, optional
            The prior inclusion probability per column (CEM only; None => 0.5 each).
        stream : int, default=0
            The random number stream (CEM only, h = 0).
        bk : int, default=3
            The beam width (h = 0; the single-horizon model's own default otherwise).
        agg : callable, default=aggFPE
            The aggregation of the (hh, N_QoF) matrix (one row per horizon) into one QoF vector (h = 0).
        qk : int, optional
            The index of the QoF measure to compare on (defaults to the given `qk`, sMAPE-IC).

        Returns
        -------
        (list of list of int, list of list of np.ndarray)
            One column set and its trace.
        """
        qk = given_qk(qk)
        if not 0 <= h <= self.hh:
            raise ValueError(f"selectFeatures: horizon h = {h} out of range [0, {self.hh}]")

        if h > 0:                                                         # the single-horizon model at h decides
            cols, rSq = self.selectFeaturesAtH(tech, cross, first, swap, h, prior, qk)
            best = self._theBest                                          # that model's best step
            if best.qof is not None:                                      # its QoF, the multi-horizon model
                try:
                    self._theBest = BestStep(best.col, best.qof, self.fitCols(cols)[0], cols, qk, float(best.qof[qk]))
                except Exception as e:
                    print(f"selectFeatures: fit of {set_str(cols)} failed: {e}")
            return [cols], [rSq]

        rSq: List[np.ndarray] = []                                        # the best-so-far per step

        def evalJoint(cols: Cols) -> BestStep:                            # fit and score one set over all horizons
            try:
                sub, qof = self.fitCols(cols)
                q = agg(qof)
                return BestStep(-1, q, sub, cols, qk, float(q[qk]))
            except Exception as e:
                print(f"selectFeatures: cols = {set_str(cols)} failed: {e}")
                return BestStep(-1, None, None, cols, qk)

        path = SubsetSearch.run(tech, self.getX().shape[1], evalJoint, list(range(first)), swap=swap, bk=bk,
                                prior=prior, stream=stream, qk=qk)
        for step in path.steps:
            updateQoF(rSq, "none", step)
        self._theBest = path.best
        self._lastProbs = path.probs
        banner(f"selectFeatures ({tech}, all horizons): {path.nEval} column sets evaluated => cols = {set_str(path.cols)}"
               + (f" @ {float(path.best.qof[qk])}" if path.found else " (no set could be fit)"))
        return [path.cols], [rSq]

    def selectFeaturesAtH(self, tech: SelectionTech, cross: str = "none", first: int = 1, swap: bool = True,
                          h: int = 1, prior: Optional[Sequence[float]] = None,
                          qk: Optional[int] = None) -> Tuple[Cols, List[np.ndarray]]:
        """
        Performs regression-based feature selection on the single-horizon model at horizon h
        (`getModel_h (h = h)`, an `ARX` with hh = 1, whose `Forecaster_Reg` selection runs),
        returning the variables left and the QoF measures for all steps; the selected single-horizon
        model is `getModel_h (cols, h)` (`reportSelection` with h prints and tests it).  Unlike
        `selectFeatures` the selection is horizon (h) specific.

        Parameters
        ----------
        tech : SelectionTech
            The type of the feature selection to use.
        cross : str, default="none"
            Indicator to include the cross-validation/validation QoF measure.
        first : int, default=1
            The first variable to consider for elimination (1 => the intercept x_0 stays in any model).
        swap : bool, default=True
            Whether to allow a swap step.
        h : int, default=1
            The horizon.
        prior : sequence of float, optional
            The prior inclusion probability per column (CEM only; None => 0.5 each).
        qk : int, optional
            The index of the QoF measure to compare on.

        Returns
        -------
        (list of int, list of np.ndarray)
            The selected columns and the trace of the single-horizon model's selection.
        """
        qk = given_qk(qk)
        if not 1 <= h <= self.hh:
            raise ValueError(f"horizon h = {h} out of range [1, {self.hh}]")

        mod_h = self.getModel_h(h=h)                                      # the single-horizon model bound to Y[:, h-1], hh = 1
        if tech is SelectionTech.Forward:
            res = mod_h.forwardSelAll(cross, qk=qk)
        elif tech is SelectionTech.Backward:
            res = mod_h.backwardElimAll(first, cross, qk=qk)
        elif tech is SelectionTech.Stepwise:
            res = mod_h.stepwiseSelAll(cross, swap, qk=qk)
        elif tech is SelectionTech.Beam:
            res = mod_h.beamSelAll(cross, qk=qk)
        elif tech is SelectionTech.CEM:
            res = mod_h.cemSelAll(cross, first, prior, qk=qk)
        else:
            raise ValueError(f"selectFeaturesAtH: unknown technique {tech}")
        rSq = res[1][0]                                                   # its trace (the single-horizon model's selection)
        best = mod_h.getBest()
        cols = list(best.mod_cols) if best.mod_cols is not None else []
        self._theBest = best
        self._lastProbs = mod_h.cemProbs
        return cols, rSq

    def selectFeaturesPerH(self, tech: SelectionTech, cross: str = "none", first: int = 1, swap: bool = True,
                           qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        Performs regression-based feature selection PER HORIZON: `selectFeaturesAtH` for h = 1 to
        hh, one column set per horizon; returns the column sets and the QoF traces (element h-1 =
        horizon h), the shape `reportSelection` prints and tests (one single-horizon model per
        horizon, `getModel_h`).  `getBest ()` is left at the last horizon's best step.

        Parameters
        ----------
        tech, cross, first, swap, qk
            As in `selectFeaturesAtH`.
        """
        qk = given_qk(qk)
        colsArr: List[Cols] = []
        rSqArr: List[List[np.ndarray]] = []
        for h in range(1, self.hh + 1):
            cls_, rSq = self.selectFeaturesAtH(tech, cross, first, swap, h, qk=qk)
            colsArr.append(cls_)
            rSqArr.append(rSq)
        return colsArr, rSqArr

    # -- R E P O R T I N G   T H E   S E L E C T I O N  (the hooks of `SelectionReport.reportSelection`)

    def selectedModel(self, cols: Cols, h: int) -> Any:
        """
        Builds the forecaster of a selected column set (the hook of `reportSelection`): the
        multi-horizon model on the columns, or the single-horizon model at h (h > 0).
        """
        return self.getModel_h(cols, h) if h > 0 else self.getModel(cols)

    def testSelected(self, mod: Any, inSample: bool, tnt: bool, rc: int,
                     showYf: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Tests a selected forecaster (the hook of `reportSelection`) with the steps of
        `inSample_Test` (train on all the rows, forecast, diagnose every horizon) and/or of
        `tnT_Test` (train, forecast, `setSkip`, `rollValidate (rc)`, diagnose on the testing rows);
        each QoF table has one row per horizon of the model (one row for a single-horizon model).

        Parameters
        ----------
        mod : Forecaster
            The selected forecaster (multi- or single-horizon).
        inSample : bool
            Whether to test in-sample.
        tnt : bool
            Whether to test with rolling validation.
        rc : int
            The retraining cycle of the rolling validation.
        showYf : bool
            Whether to print the forecast matrices.
        """
        qofIn: Optional[np.ndarray] = None
        qofTnT: Optional[np.ndarray] = None
        if inSample:
            banner(f"{self.modelName}: In-Sample Test of the selected model on {array_str(mod.fname)}")
            mod.trainNtest()                                              # train on all the rows
            mod.forecastAll()                                             # forecast h-steps ahead for all y
            qofIn = mod.diagnoseAll()                                     # QoF for each horizon
            if showYf:
                print(f"Forecast Matrix yf = {mod.Yf}")
        if tnt:
            banner(f"{self.modelName}: TnT Test (rolling validation) of the selected model on {array_str(mod.fname)}")
            mod.trainNtest()                                              # initial training updated by `rollValidate`
            mod.forecastAll()
            mod.setSkip()                                                 # diagnose: don't skip any rows
            mod.rollValidate(rc)                                          # TnT with Rolling Validation
            qofTnT = mod.diagnoseAll(tRng=range(mod.tr_size, len(mod.y)))   # the testing rows only
            if showYf:
                print(f"Forecast Matrix yf = {mod.Yf}")
        return qofIn, qofTnT

    # the companion object's functions, reachable from the class as in ScalaTion (`Forecaster_D.aggFPE`)
    smapeFPE = staticmethod(smapeFPE)
    aggFPE = staticmethod(aggFPE)
