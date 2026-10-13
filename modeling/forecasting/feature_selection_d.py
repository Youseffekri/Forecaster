"""
Feature Selection for Forecasters: the Direct Multi-Horizon Scenarios and the Regression-Based Selection

The Python counterpart of the feature-selection part of ScalaTion's `forecasting/Forecaster_D.scala`
(lines "F E A T U R E   S E L E C T I O N" to the companion object) and of the selection methods of
`forecasting/Forecaster_Reg.scala`:

    FeatureSelection_D    the mixin of `Forecaster_D`: `selectFeatures` (ONE column set for all the
                          horizons: h = 0 => every candidate set is fit as ONE multi-horizon model and
                          scored over ALL horizons by `agg`, the mean FPE-form sMAPE J by default;
                          h >= 1 => the single-horizon model at h decides), `selectFeaturesAtH` (the
                          column set of the single-horizon model at h), `selectFeaturesPerH` (one set
                          per horizon), `fitCols`, `getBest`, `cemProbs`, and the two hooks of
                          `reportSelection` (`selectedModel`, `testSelected`); the models provide
                          `getModel (cols)` (the direct sub-model) and `getModel_h (cols, h)` (the
                          single-horizon `ARX` at h)
    FeatureSelection_Reg  ScalaTion's `Forecaster_Reg` selection (`forwardSelAll`, `backwardElimAll`,
                          `stepwiseSelAll`, `beamSelAll`, `cemSelAll`, `correctQoF`, `getBest`,
                          `cemProbs`, `importance`, `reportSelection`) as an ADAPTER over a recursive
                          forecaster with an input matrix (an `ARX`, `ARX_SR`, ...), since the Python
                          `ARX` does not inherit it: `FeatureSelection_Reg (arx).selectFeatures (tech)`
    smapeFPE, aggFPE      the companion object: the FPE-form parsimony-adjusted sMAPE per horizon and
                          its mean over the horizons (the selection score J of the `llmsr` search)
    subModel, subModel_h  the helpers the models' `getModel` / `getModel_h` build on

Every candidate set is scored on the FS rows (the training rows, or all rows when
`FeatureSelection.fullset_FS`) in ORIGINAL space with the skip rows dropped (`correctQoF`); the
algorithms are `SubsetSearch`'s; with qk = smapeC and h = 0 the criterion is J, so the base's models
and the structure search select on the same criterion through the same algorithms.

Differences from the Scala: the `mod` of a `BestStep` is the sub-FORECASTER on the columns (not the
bare regression); a single-horizon model at h > 1 holds the m - h + 1 rows whose horizon-h response
is defined (the Scala keeps the NaN-padded rows) and the parent's training size.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Util.tools import array_str, banner, set_str, vector_str
from modeling.cem_selector import CEM_Selector
from modeling.feature_selection import (BestStep, Cols, FeatureSelection, SelectionReport, SelectionTech,
                                        given_qk, updateQoF)
from modeling.fit import N_QoF, QoF, diagnose as diagnose_, showFitMap
from modeling.forecasting.arx import ARX
from modeling.forecasting.forecaster import trSize
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
    FPE-form sMAPE over the horizons (`smapeFPE`), the selection score J of the `llmsr` search;
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


# ---------------------------------------------------------------------------------------------
# Sub-models on a column set
# ---------------------------------------------------------------------------------------------

def _inv(yForm, v: np.ndarray) -> np.ndarray:
    """Maps a vector back to ORIGINAL units through the model's transform of y (None => as is)."""
    v = np.asarray(v, dtype=float)
    return v if yForm is None else np.asarray(yForm.inverse_transform(v.reshape(-1, 1)), dtype=float).ravel()


def subModel(parent: Any, sub: Any, cols: Sequence[int], modelName: Optional[str] = None) -> Any:
    """
    Finishes a sub-model built on the columns `cols` of the parent's input matrix (the model
    classes construct `sub` on `parent.getX ()[:, cols]` with the ORIGINAL y, then call this):
    the parent's response in model units (scaled when the parent was made by `rescale`), its
    target matrix, transform of y, non-negativity, training/testing sizes, skip and the feature
    names of the columns, so the sub-model forecasts in the same units and on the same rows.

    Parameters
    ----------
    parent : Forecaster
        The model the columns are taken from.
    sub : Forecaster
        The freshly constructed sub-model (X given, so no lag matrix is rebuilt).
    cols : sequence of int
        The selected columns.
    modelName : str, optional
        The sub-model's name (defaults to the parent's).

    Returns
    -------
    Forecaster
        `sub`, finished.
    """
    cols = list(cols)
    sub._y = parent.y                                                     # model units
    if getattr(parent, "_Y", None) is not None and hasattr(sub, "_Y"):
        sub._Y = parent._Y                                                # the direct targets (same units)
    sub._yForm = parent.yForm
    if hasattr(parent, "_tForms"):
        sub._tForms = dict(parent._tForms)
    sub._nneg = parent.nneg
    sub._tr_size, sub._te_size = parent.tr_size, parent.te_size
    fn = parent.fname
    sub._fname = [fn[j] if j < len(fn) else f"x{j}" for j in cols]
    if hasattr(parent, "_n_exo"):
        sub._n_exo = parent._n_exo
    sub._modelName = parent.modelName if modelName is None else modelName
    sub._useforge = False                                                 # the lag structure is gone: no forging
    sub.setSkip(parent.getSkip())
    return sub


def subModel_h(parent: Any, cols: Sequence[int], h: int, method: str,
               args: Optional[Dict[str, Any]] = None) -> ARX:
    """
    The single-horizon `ARX` at horizon h on the columns `cols` of a direct model (ScalaTion's
    `ARX_D.getModel_h`): an `ARX` with hh = 1 whose response is column h - 1 of the target matrix,
    on the m - h + 1 rows where that response is defined, with the parent's training size (so the
    FS rows and the testing rows are the parent's), scaling and skip.

    Parameters
    ----------
    parent : Forecaster_D
        The direct model.
    cols : sequence of int
        The selected columns.
    h : int
        The horizon (1 to hh).
    method : str
        The regression method of the `ARX` ("sm_ols", "sk_lr", "ridge").
    args : dict, optional
        The hyper-parameters of the `ARX` (defaults to the parent's).
    """
    if not 1 <= h <= parent.hh:
        raise ValueError(f"getModel_h: horizon h = {h} out of range [1, {parent.hh}]")
    cols = list(cols)
    args = parent.args if args is None else args
    m = len(parent.y)
    mh = m - h + 1                                                        # the rows whose horizon-h response is defined
    sub = ARX(args, parent.getY()[h - 1:], 1, None, method, parent.getX()[:mh, cols])
    n_exo = getattr(parent, "_n_exo", 0)
    subModel(parent, sub, cols, modelName=f"ARX_{args['p']}_{args['q']}_{n_exo}")
    sub._y = parent.Y[:mh, h - 1]                                         # the horizon-h response (model units)
    sub._te_size = mh - sub._tr_size
    return sub


# ---------------------------------------------------------------------------------------------
# FeatureSelection_Reg: the selection of a recursive forecaster (ScalaTion's Forecaster_Reg)
# ---------------------------------------------------------------------------------------------

class FeatureSelection_Reg(FeatureSelection):
    """
    The feature selection of ScalaTion's `Forecaster_Reg` for a recursive forecaster `mod` that
    has an input matrix (`ARX`, `ARX_SR`, or the single-horizon model `getModel_h` of a direct
    model): every technique evaluates a column set the same way (`evalCols`: build the sub-model
    on the columns, train it on the FS rows, correct its QoF to ORIGINAL space with the skip rows
    dropped) through the algorithms of `SubsetSearch`, records the best-so-far per step into the
    trace (`updateQoF`) and keeps the best step (`getBest`).

    Parameters
    ----------
    mod : Forecaster
        The forecaster whose columns are selected (its `getModel (cols)` builds a sub-model when
        it has one; otherwise a model of the same class is built on `X[:, cols]`).

    Attributes
    ----------
    fs_rng : range
        The rows feature selection fits on: the training rows (or all rows when
        `FeatureSelection.fullset_FS`), cut off at the first undefined (NaN) response.
    """

    def __init__(self, mod: Any) -> None:
        self.mod = mod
        self._theBest = BestStep()                                        # the best model from Feature Selection (FS)
        self._lastProbs: Optional[np.ndarray] = None                      # the last CEM run's inclusion probabilities
        m = len(mod.y)
        t_rng = range(0, m) if FeatureSelection.fullset_FS else range(0, mod.tr_size)
        nan = np.where(np.isnan(np.asarray(mod.y, dtype=float)))[0]
        end = int(nan[0]) if len(nan) else -1
        self.fs_rng = t_rng if end < 0 else range(t_rng.start, min(t_rng.stop, end))

    # -- the host's attributes `SelectionReport` reads -------------------------------------------

    @property
    def modelName(self) -> str:
        return self.mod.modelName

    @property
    def fname(self) -> List[str]:
        return self.mod.fname

    @property
    def X(self) -> np.ndarray:
        return self.mod.X

    def getX(self) -> np.ndarray:
        return self.mod.X

    @property
    def y(self) -> np.ndarray:
        return self.mod.y

    @property
    def yForm(self):
        return self.mod.yForm

    def getSkip(self) -> int:
        return self.mod.getSkip()

    @property
    def mcols(self) -> Cols:
        """The set of columns (numbers) for the features in this model."""
        return list(range(self.mod.X.shape[1]))

    # -- sub-models ------------------------------------------------------------------------------

    def getModel(self, cols: Optional[Sequence[int]] = None) -> Any:
        """
        Builds the forecaster on the selected columns (e.g., the best model after feature selection:
        `getModel (getBest ().mod_cols)`): the model's own `getModel` when it has one, else a
        SINGLE-HORIZON model of the same class on `X[:, cols]` (the `ARX` constructor signature),
        finished by `subModel`.  A sub-model holds the selected columns only while the recursive
        forecasts for h > 1 need the full lag structure, so (as ScalaTion's `testSelected` tests a
        selected forecaster at h = 1 only) the sub-model built here has hh = 1.
        """
        cols = self.mcols if cols is None else list(cols)
        mod = self.mod
        if hasattr(mod, "getModel"):
            return mod.getModel(cols)
        sub = type(mod)(mod.args, mod.getY(), 1, None, mod._method, mod.X[:, cols])
        return subModel(mod, sub, cols)

    def resetBest(self) -> None:
        """Resets the best step to the default."""
        self._theBest = BestStep()

    def getBest(self) -> BestStep:
        """The best step/model found from feature selection (`getModel (getBest ().mod_cols)` is the forecaster)."""
        return self._theBest

    def _updateBest(self, best: BestStep, qk: int) -> None:
        """Replaces the best step when the new one is better on metric qk."""
        if best.qof is not None:
            if self._theBest.qof is None or best.gt(float(self._theBest.qof[qk])):
                self._theBest = best

    def buildNrun(self, cols_j: Sequence[int]) -> Tuple[Any, np.ndarray]:
        """
        Builds the sub-model on the columns `cols_j`, trains it on the FS rows and tests it there
        (in model units, with the regression's degrees of freedom), returning the model and its QoF.
        """
        fs = self.fs_rng
        mod_j = self.getModel(cols_j)
        mod_j.train(mod_j.y[fs.start:fs.stop], mod_j.X[fs.start:fs.stop])    # train on the FS rows
        yp = mod_j.forecast(fs.start, fs.stop)[:, 0]                           # predictions (model units)
        k, m_full = mod_j.nparams, len(mod_j.y)
        qof_j = diagnose_(mod_j.y[fs.start:fs.stop], yp, dfr=k, df=max(1, m_full - k - 1))
        return mod_j, qof_j

    def correctQoF(self, skp: int, best: BestStep, qk: Optional[int] = None) -> BestStep:
        """
        Corrects the QoF in `best` (ScalaTion's `Forecaster_Reg.correctQoF`): re-diagnoses the
        sub-model on the rows feature selection trains on (`fs_rng`), leaving out the first `skp`
        rows (their lags are clamped), in ORIGINAL space -- when the data were transformed
        (`rescale`, `fromTerms` with a tFormT) both the responses and the predictions are mapped
        back before `diagnose`, so the criterion compared, printed and stored by every selection
        step is the one the final model reports.  The degrees of freedom are the regression's as
        ScalaTion's `CenteredRidge` sets them at construction (dfr = k, df = m - k - 1 with m the
        rows of the sub-model's matrix).

        Parameters
        ----------
        skp : int
            The number of time-steps to skip at the beginning.
        best : BestStep
            The best model selected so far (its `mod` is the sub-forecaster).
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        if (skp > 0 or self.yForm is not None) and best.mod is not None:
            bmod, fs = best.mod, self.fs_rng
            bmod.forecast(fs.start, fs.stop)                              # predictions on the FS rows only
            yo = bmod.getY()[fs.start:fs.stop]                            # the responses in ORIGINAL units
            ypo = bmod.Yf[fs.start:fs.stop, 1]                            # the predictions in ORIGINAL units
            k, m_full = bmod.nparams, len(bmod.y)
            qof = diagnose_(yo[skp:], ypo[skp:], dfr=k, df=max(1, m_full - k - 1))
            return BestStep(best.col, qof, bmod, best.mod_cols, qk, float(qof[qk]))
        return best

    # -- T H E   S E L E C T I O N   T E C H N I Q U E S  (see `SubsetSearch`) --------------------

    def evalCols(self, cols: Sequence[int], qk: Optional[int] = None) -> BestStep:
        """
        Evaluates one column set the way every selection step does: builds and runs the sub-model
        on the FS rows (`buildNrun`) and corrects its QoF (`correctQoF`: skip rows dropped, original
        space).  The evaluator of every technique; a set whose fit fails returns a step with qof
        None, so the search skips it.
        """
        qk = given_qk(qk)
        cols = list(cols)
        try:
            mod_j, qof_j = self.buildNrun(cols)
        except Exception as e:
            print(f"evalCols: cols = {set_str(cols)} failed: {e}")
            return BestStep(-1, None, None, cols, qk)
        return self.correctQoF(self.getSkip(), BestStep(-1, qof_j, mod_j, cols, qk, float(qof_j[qk])), qk)

    def cname(self, j: int) -> str:
        """The name of column j for the banners and traces: fname[j], or x_j when the model names fewer columns."""
        fn = self.fname
        return fn[j] if j < len(fn) else f"x{j}"

    def _record(self, path: SubsetSearch.Path, rSq: List[np.ndarray], cross: str, qk: int) -> None:
        """Records a search path: the best-so-far after each step into the trace and the best step found."""
        for step in path.steps:
            updateQoF(rSq, cross, step)
        self._updateBest(path.best, qk)

    def select0(self, qk: Optional[int] = None) -> BestStep:
        """
        Evaluates the model with only column 0 (e.g., the intercept-only model; ScalaTion's
        `Predictor.select0`): trained on the FS rows, tested on all the rows.
        """
        qk = given_qk(qk)
        fs = self.fs_rng
        mod_0 = self.getModel([0])
        mod_0.train(mod_0.y[fs.start:fs.stop], mod_0.X[fs.start:fs.stop])
        yp = mod_0.forecast()[:, 0]
        k, m = mod_0.nparams, len(mod_0.y)
        qof_0 = diagnose_(mod_0.y, yp, dfr=k, df=max(1, m - k - 1))
        return BestStep(0, qof_0, mod_0, [0], qk, float(qof_0[qk]))

    def forwardSelAll(self, cross: str = "many", qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        FORWARD SELECTION: finds the MOST predictive variables to have in the model, returning the
        variables added and the QoF measures for all steps (`SubsetSearch.forward`: the full path
        from the intercept-only model, the best set along it is `getBest ()`).

        Parameters
        ----------
        cross : str, default="many"
            Indicator to include the cross-validation/validation QoF measure ("none" for forecasters).
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        self.resetBest()
        rSq: List[np.ndarray] = []                                        # QoF: R^2, R^2 Bar, sMAPE, R^2 cv
        banner(f"forwardSelAll: (qk = {qk}, l = 0) INITIAL variable (0, {self.cname(0)}) => cols = {set_str([0])}")
        path = SubsetSearch.forward(self.X.shape[1], lambda c: self.evalCols(c, qk), [0], qk=qk)
        self._record(path, rSq, cross, qk)
        for l, step in enumerate(path.steps):
            if l > 0:
                banner(f"forwardSelAll: (l = {l}) ADD variable ({step.col}, {self.cname(step.col)}) => "
                       f"cols = {set_str(step.mod_cols)} @ {float(step.qof[qk])}")
        cols = [0] if not path.steps else path.steps[-1].mod_cols
        return [cols], [rSq]

    def backwardElimAll(self, first: int = 1, cross: str = "none",
                        qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        BACKWARD ELIMINATION: finds the LEAST predictive variables to remove from the full model,
        returning the variables in reverse order of elimination and the QoF measures for all steps,
        from the smallest model up (`SubsetSearch.backward`: the full path down to two columns, the
        best set along it is `getBest ()`).

        Parameters
        ----------
        first : int, default=1
            The first variable to consider for elimination (the leading `first` columns are kept).
        cross : str, default="none"
            Indicator to include the cross-validation/validation QoF measure.
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        self.resetBest()
        rSq: List[np.ndarray] = []                                        # R^2, R^2 Bar, sMAPE, sMAPEC
        path = SubsetSearch.backward(self.X.shape[1], lambda c: self.evalCols(c, qk), list(range(first)),
                                     minCols=2, qk=qk)
        self._record(path, rSq, cross, qk)
        updateQoF(rSq, cross, self.correctQoF(self.getSkip(), self.select0(qk), qk))   # the 0-th variable row
        rem = [step.col for step in path.steps[1:]]                       # the columns removed, in order
        cols = self.mcols if not path.steps else path.steps[-1].mod_cols
        rem.append(max(cols))                                             # remove last non-zero column
        rem.append(0)                                                     # remove column 0
        return [list(reversed(rem))], [list(reversed(rSq))]               # reverse the order results

    def stepwiseSelAll(self, cross: str = "none", swap: bool = True,
                       qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        STEPWISE SELECTION: finds a GOOD COMBINATION of predictive variables, returning the variables
        selected and the QoF measures for all steps (`SubsetSearch.stepwise`: at each step the best
        forward and the best backward move, a swap when neither improves, stop when no move improves).

        Parameters
        ----------
        cross : str, default="none"
            Indicator to include the cross-validation/validation QoF measure.
        swap : bool, default=True
            Whether to allow a swap step (swap out a feature for a new feature in one step).
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        self.resetBest()
        rSq: List[np.ndarray] = []                                        # QoF: R^2, R^2 Bar, sMAPE, sMAPEC
        banner(f"stepwiseSelAll: (qk = {qk}, l = 0) INITIAL variable (0, {self.cname(0)}) => cols = {set_str([0])}")
        path = SubsetSearch.stepwise(self.X.shape[1], lambda c: self.evalCols(c, qk), [0], swap=swap, qk=qk)
        self._record(path, rSq, cross, qk)
        prev = 1                                                          # size of the previous set (intercept only)
        for l, step in enumerate(path.steps):
            cols = step.mod_cols
            move = "ADD" if len(cols) > prev else ("REMOVE" if len(cols) < prev else "SWAP")
            banner(f"stepwiseSelAll: (l = {l + 1}) {move} variable ({step.col}, {self.cname(step.col)}) => "
                   f"cols = {set_str(cols)} @ {float(step.qof[qk])}")
            prev = len(cols)
        cols = [0] if not path.steps else path.steps[-1].mod_cols
        print(f"stepwiseSelAll: selected features = {set_str(cols)}")
        print(f"stepwiseSelAll: selected features = {set_str(self.cname(j) for j in cols)}")
        return [cols], [rSq]

    def beamSelAll(self, cross: str = "many", bk: int = 8,
                   qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        BEAM SEARCH SELECTION: finds a GOOD COMBINATION of predictive features, returning the best set
        found and the QoF measures after each level (`SubsetSearch.beam`: phase I keeps the bk best
        single features, phase II expands every set of the beam by the move set (1,0), (2,0), (0,1),
        (0,2), (1,1), (2,1) and keeps the bk best, until patience runs out).

        Parameters
        ----------
        cross : str, default="many"
            Indicator to include the cross-validation/validation QoF measure.
        bk : int, default=8
            The beam width holding the top k models.
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        self.resetBest()
        rSq: List[np.ndarray] = []                                        # QoF: R^2, R^2 Bar, sMAPE, R^2 cv
        updateQoF(rSq, cross, self.correctQoF(self.getSkip(), self.select0(qk), qk))   # the 0-th variable row
        banner(f"beamSelAll: (qk = {qk}, l = 0) INITIAL variable (0, {self.cname(0)}) => cols = {set_str([0])}")
        path = SubsetSearch.beam(self.X.shape[1], lambda c: self.evalCols(c, qk), [0], bk=bk, qk=qk)
        self._record(path, rSq, cross, qk)
        for l, step in enumerate(path.steps):
            banner(f"beamSelAll: level {l} best so far => cols = {set_str(step.mod_cols)} @ {float(step.qof[qk])}")
        return [path.cols], [rSq]

    def cemSelAll(self, cross: str = "none", first: int = 1, prior: Optional[Sequence[float]] = None,
                  nSample: int = CEM_Selector.N_SAMPLE, maxIter: int = CEM_Selector.MAX_ITER,
                  maxCols: int = -1, stream: int = 0,
                  qk: Optional[int] = None) -> Tuple[List[Cols], List[List[np.ndarray]]]:
        """
        CROSS-ENTROPY SELECTION (`SubsetSearch.cem`, `CEM_Selector`): each round samples column sets
        from per-column inclusion probabilities, evaluates each set as every other technique does
        (`evalCols`), keeps the elite sets and moves the probabilities towards them.  Returns the
        best set found and the QoF of the best-so-far model after each round; `cemProbs` returns the
        final inclusion probabilities.

        Parameters
        ----------
        cross : str, default="none"
            Indicator to include the cross-validation/validation QoF measure.
        first : int, default=1
            The number of leading columns always kept (1: the intercept x_0; 0 when x has no
            intercept column, e.g., a matrix built by `fromTerms`).
        prior : sequence of float, optional
            The prior inclusion probability per column of x (None => 0.5 each).
        nSample : int
            The number of column sets sampled per round.
        maxIter : int
            The maximum number of rounds.
        maxCols : int, default=-1
            The maximum number of columns in a sampled set (-1 => no cap).
        stream : int, default=0
            The random number stream.
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        self.resetBest()
        rSq: List[np.ndarray] = []                                        # QoF of the best-so-far model per round
        path = SubsetSearch.cem(self.X.shape[1], lambda c: self.evalCols(c, qk), list(range(first)), prior,
                                nSample, maxIter, maxCols, stream, qk=qk)
        self._record(path, rSq, cross, qk)
        self._lastProbs = path.probs
        if not path.found:
            print("cemSelAll: no column set could be evaluated")
            return [[]], [rSq]
        best = path.best
        banner(f"cemSelAll: {path.nEval} column sets evaluated => cols = {set_str(best.mod_cols)} @ {float(best.qof[qk])}")
        print(f"cemSelAll: selected features = {set_str(self.cname(j) for j in best.mod_cols)}")
        print(f"cemSelAll: inclusion probabilities = {vector_str(path.probs)}")
        return [best.mod_cols], [rSq]

    @property
    def cemProbs(self) -> Optional[np.ndarray]:
        """The inclusion probability per column left by the last `cemSelAll` run (None if none ran)."""
        return self._lastProbs

    @staticmethod
    def importance(cols: Sequence[int], rSq: np.ndarray) -> List[Tuple[int, float]]:
        """
        The relative importance of the selected variables, ordered highest to lowest, rescaled so
        the highest is one (ScalaTion's `Predictor.importance`): the scaled improvement in R^2 of
        each step of the trace.

        Parameters
        ----------
        cols : sequence of int
            The selected columns/features/variables (in the path's order).
        rSq : np.ndarray, shape (steps, 4)
            The trace (column 0 holds 100 R^2).
        """
        r2 = np.asarray(rSq, dtype=float)[:, 0]
        sf = 1.0 / (r2[1] - r2[0])                                        # scale factor, so most important = 1
        imp = [(int(cols[0]), -0.0)]
        for j in range(1, len(r2)):
            imp.append((int(cols[j]), float(sf * (r2[j] - r2[j - 1]))))
        return imp

    # -- R E P O R T I N G   T H E   S E L E C T I O N ---------------------------------------------

    def selectedModel(self, cols: Cols, h: int) -> Any:
        """The forecaster on a selected column set (the hook of `reportSelection`): `getModel`; h is ignored."""
        return self.getModel(cols)

    def testSelected(self, mod: Any, inSample: bool, tnt: bool, rc: int,
                     showYf: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Tests a selected forecaster (the hook of `reportSelection`) with the steps of
        `inSample_Test` (train on all the rows, forecast, diagnose) and/or of `tnT_Test` (train,
        `setSkip`, rolling validation with retraining every rc rows, diagnose on the testing rows);
        each QoF table has one row per horizon.  A sub-model holds the selected columns only while
        the recursive forecasts for h > 1 assume the full lag structure, so a selected forecaster
        with hh > 1 is tested at h = 1 only (one row): its predictions, rolled over the testing rows.
        """
        qofIn: Optional[np.ndarray] = None
        qofTnT: Optional[np.ndarray] = None
        h1 = mod.hh > 1                                                   # hh > 1: h = 1 only (see the note)
        note = " (h = 1 only: a sub-model's forge needs the full lag structure)" if h1 else ""

        if inSample:
            banner(f"{self.modelName}: In-Sample Test of the selected model on {array_str(mod.fname)}{note}")
            _, qof = mod.trainNtest()                                     # train on all the rows, test on all the rows
            if h1:
                qofIn = np.atleast_2d(qof)                                # QoF at h = 1
                print("fitMap QoF = ")
                print(showFitMap(qofIn.T))
            else:
                mod.forecastAll()                                         # forecast over all horizons
                qofIn = mod.diagnoseAll()                                 # QoF for all horizons
                if showYf:
                    print(f"Forecast Matrix yf = {mod.Yf}")

        if tnt:
            banner(f"{self.modelName}: TnT Test (rolling validation, rc = {rc}) of the selected model on "
                   f"{array_str(mod.fname)}{note}")
            mod.trainNtest()                                              # initial training updated by the rolling validation
            mod.setSkip()                                                 # diagnose: don't skip any rows
            if h1:
                X, y = mod.X, mod.y
                te = range(mod.tr_size, len(y))                           # the testing rows
                for i in range(len(te)):
                    t = te.start + i                                      # the next time point to predict
                    if i % rc == 0:
                        mod.train(y[i:t], X[i:t])                         # retrain on the sliding training rows
                    mod.forecast(t, t + 1)                                # its 1-step ahead prediction (Yf[t, 1])
                mod.mod_resetDF(len(te))                                  # reset the Degrees of Freedom (DF)
                qofTnT = np.atleast_2d(mod.diagnose(mod.getY()[te.start:te.stop], mod.Yf[te.start:te.stop, 1]))
                print("fitMap QoF = ")
                print(showFitMap(qofTnT.T))
            else:
                mod.rollValidate(rc)                                      # TnT with Rolling Validation
                qofTnT = mod.diagnoseAll(tRng=range(mod.tr_size, len(mod.y)))   # the testing rows only
                if showYf:
                    print(f"Forecast Matrix yf = {mod.Yf}")
        return qofIn, qofTnT


# ---------------------------------------------------------------------------------------------
# FeatureSelection_D: the mixin of Forecaster_D
# ---------------------------------------------------------------------------------------------

class FeatureSelection_D(SelectionReport):
    """
    The feature-selection scenarios of a DIRECT multi-horizon forecaster (the mixin of
    `Forecaster_D`; ScalaTion's `Forecaster_D` with `SelectionReport`).  The three selections and
    what they return:

    `selectFeatures`      ONE column set for all the horizons (h = 0: every candidate set is fit as
                          ONE multi-horizon model and scored over ALL horizons by `agg`; h >= 1:
                          the single-horizon model at h decides, its columns are shared by every
                          horizon); the selected multi-horizon model is `getBest ()`.
    `selectFeaturesAtH`   the column set of the single-horizon model at h (`getModel_h`).
    `selectFeaturesPerH`  one column set PER HORIZON.

    `selectFeatures` and `selectFeaturesPerH` return LISTS (one column set and trace, or one per
    horizon); `selectFeaturesAtH` returns the column set and trace (the model is `getModel_h`);
    `reportSelection` prints and tests any of them, in-sample and/or TnT, one QoF table per test.
    The model classes provide `getModel (cols)` and `getModel_h (cols, h)`.
    """

    _theBest: Optional[BestStep] = None                                   # the best step of the last selection
    _lastProbs: Optional[np.ndarray] = None                               # the last CEM selection's probabilities

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

    def getModel(self, cols: Optional[Sequence[int]] = None) -> Any:
        """
        Builds the direct model on the columns `cols` of the input matrix (the model classes
        implement it; see `subModel`).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        """
        raise NotImplementedError(f"{self.modelName}.getModel: the model class must implement it")

    def getModel_h(self, cols: Optional[Sequence[int]] = None, h: int = 1) -> Any:
        """
        Builds the single-horizon forecaster at horizon h on the columns `cols` (the model classes
        implement it; see `subModel_h`).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        h : int, default=1
            The horizon.
        """
        raise NotImplementedError(f"{self.modelName}.getModel_h: the model class must implement it")

    def fitCols(self, cols: Sequence[int]) -> Tuple[Any, np.ndarray]:
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
        m = len(self.y)
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
                  qk = smapeC the criterion is the selection score J of the `llmsr` search).  The
                  trace holds the best-so-far per step (aggregated measures); no cross-validation
                  column; a set whose fit fails is skipped.
        h >= 1 -- AT ONE HORIZON: the single-horizon model at h decides (`selectFeaturesAtH`:
                  `FeatureSelection_Reg`'s selection on `getModel_h (h)`, so `cross` applies and the
                  trace is that model's) and its columns are shared by every horizon; `getBest ()`
                  then carries that step's QoF with the multi-horizon model of the columns.

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
        (`getModel_h (h = h)`, an `ARX` with hh = 1, through `FeatureSelection_Reg`), returning the
        variables left and the QoF measures for all steps; the selected single-horizon model is
        `getModel_h (cols, h)` (`reportSelection` with h prints and tests it).  Unlike
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
        fs = FeatureSelection_Reg(mod_h)                                  # its selection (ScalaTion's Forecaster_Reg)
        if tech is SelectionTech.Forward:
            res = fs.forwardSelAll(cross, qk=qk)
        elif tech is SelectionTech.Backward:
            res = fs.backwardElimAll(first, cross, qk=qk)
        elif tech is SelectionTech.Stepwise:
            res = fs.stepwiseSelAll(cross, swap, qk=qk)
        elif tech is SelectionTech.Beam:
            res = fs.beamSelAll(cross, qk=qk)
        elif tech is SelectionTech.CEM:
            res = fs.cemSelAll(cross, first, prior, qk=qk)
        else:
            raise ValueError(f"selectFeaturesAtH: unknown technique {tech}")
        rSq = res[1][0]                                                   # its trace (the single-horizon model's selection)
        best = fs.getBest()
        cols = list(best.mod_cols) if best.mod_cols is not None else []
        self._theBest = best
        self._lastProbs = fs.cemProbs
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

    # -- R E P O R T I N G   T H E   S E L E C T I O N  (see `SelectionReport.reportSelection`) ----

    # Note: `reportSelection (colsArr, rSqArr, h, inSample, tnt, rc, ...)` prints every column set with
    #       its trace and tests the selected model(s): from `selectFeatures` (one set, h = 0) the
    #       multi-horizon model on the columns (`getModel`); from `selectFeaturesAtH` (one set, h given)
    #       the single-horizon model at h (`getModel_h`); from `selectFeaturesPerH` (hh sets) one
    #       single-horizon model per horizon, their QoF rows assembled into one table.  The two hooks:

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
