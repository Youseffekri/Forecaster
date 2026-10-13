"""
Forecaster_Reg Module

Defines the abstract `Forecaster_Reg` class, the base of the forecasting models that use a
REGRESSION on an input matrix X (lagged columns of y and of the exogenous variables) and forecast
RECURSIVELY -- `ARX`, `ARX_SR` and the models built by `ARX.fromTerms`.  The Python counterpart of
ScalaTion's `Forecaster_Reg.scala`: the regression (`train`, `_predict`; methods "sm_ols",
"sk_lr" and the centered ridge "ridge" of ScalaTion's `CenteredRidge`), the recursive forecasting
(`forecast` through the subclass's `_forge`), the sub-model on a column set (`getModel`) and the
FEATURE SELECTION of ScalaTion's `Forecaster_Reg`: `forwardSelAll`, `backwardElimAll`,
`stepwiseSelAll`, `beamSelAll`, `cemSelAll` (through `SubsetSearch` / `CEM_Selector`), `correctQoF`
(every candidate set scored on the FS rows in ORIGINAL space with the skip rows dropped),
`select0`, `getBest`, `cemProbs`, `importance` and the two hooks of `reportSelection`
(`selectedModel`, `testSelected`).  `RandomWalk`, `AR` and `ARIMA` have no input matrix and
extend `Forecaster` directly, as in ScalaTion.

Differences from the Scala: the `mod` of a `BestStep` is the sub-FORECASTER on the columns (not the
bare regression); a sub-model holds the selected columns only while the recursive forecasts for
h > 1 need the full lag structure, so `getModel` builds it with hh = 1 (ScalaTion's `testSelected`
tests a selected recursive model at h = 1 as well).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from Util.tools import array_str, banner, set_str, vector_str
from modeling.cem_selector import CEM_Selector
from modeling.feature_selection import BestStep, Cols, FeatureSelection, given_qk, updateQoF
from modeling.fit import diagnose as diagnose_, showFitMap
from modeling.forecasting.forecaster import Forecaster
from modeling.regression import CenteredRidge
from modeling.subset_search import SubsetSearch


class Forecaster_Reg(FeatureSelection, Forecaster):
    """
    Abstract base class of the regression-based RECURSIVE forecasting models (ScalaTion's
    `Forecaster_Reg`).

    Attributes
    ----------
    X : np.ndarray, shape (m, n)
        The data/input matrix (no intercept column: the regression adds / recovers its own).
    method : {"sm_ols", "sk_lr", "ridge"}
        The regression: statsmodels OLS, scikit-learn LinearRegression, or the centered ridge of
        ScalaTion's `CenteredRidge` (penalty args["lambda"], default 1.0; args["scaled"]).
    useforge : bool
        Whether X has the lag structure `_forge` needs for the horizons h > 1 (False for a given X).
    fs_rng : range
        The rows feature selection fits on: the training rows (or all rows when
        `FeatureSelection.fullset_FS`), cut off at the first undefined (NaN) response.
    """

    def __init__(self, args: Dict[str, Any], y: np.ndarray, hh: int, X: Optional[np.ndarray] = None,
                 method: str = "sm_ols", hasIntercept: bool = True) -> None:
        """
        Initializes the regression-based forecaster.

        Parameters
        ----------
        args : dict
            The hyper-parameters (see `Forecaster`; "lambda" / "scaled" for method "ridge").
        y : np.ndarray, shape (m,)
            The response vector.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        X : np.ndarray, shape (m, n), optional
            The input matrix (a subclass builds it when None).
        method : {"sm_ols", "sk_lr", "ridge"}, default="sm_ols"
            The regression method.
        hasIntercept : bool, default=True
            Whether the model has an intercept.
        """
        super().__init__(args, y, hh, hasIntercept)
        self._X = X
        self._method = method
        self._model = self._make_model()
        self._useforge = False
        self._theBest = BestStep()                                        # the best model from Feature Selection (FS)
        self._lastProbs: Optional[np.ndarray] = None                      # the last CEM run's inclusion probabilities

    # -- the regression -------------------------------------------------------------------

    def _make_model(self):
        """The regression object of `method` (None for statsmodels, fitted per call)."""
        if self._method == "sk_lr":
            return LinearRegression()
        if self._method == "ridge":
            return CenteredRidge(lam=float(self.args.get("lambda", 1.0)), scaled=bool(self.args.get("scaled", False)))
        if self._method == "sm_ols":
            return None
        raise ValueError(f"method '{self._method}' unknown: sm_ols | sk_lr | ridge")

    @property
    def method(self) -> str:
        """The regression method."""
        return self._method

    def train(self, y_: Optional[np.ndarray] = None, X_: Optional[np.ndarray] = None) -> None:
        """
        Trains the regression on (X_, y_) (the full data by default; rows after the first NaN
        response are dropped, as ScalaTion's `train` drops NO_DOUBLE rows); the parameters are
        stored with the intercept first.
        """
        X_ = self._X if X_ is None else X_
        y_ = self._y if y_ is None else y_
        nan = np.where(np.isnan(np.asarray(y_, dtype=float)))[0]
        if len(nan):
            X_, y_ = X_[:nan[0]], y_[:nan[0]]

        if self._method == "sm_ols":
            self._model_fit = sm.OLS(y_, sm.add_constant(X_, has_constant="add")).fit()
            self._set_params(np.asarray(self._model_fit.params))
        elif self._method == "ridge":
            self._model.fit(X_, y_)
            self._set_params(self._model.params[:, 0])
        else:
            self._model.fit(X_, y_)
            self._set_params(np.append(self._model.intercept_, self._model.coef_))

    def _predict(self, X_rows: np.ndarray) -> np.ndarray:
        """The one-step predictions of the given input rows (clipped at zero when nneg)."""
        if self._method == "sm_ols":
            yp = self._model_fit.predict(sm.add_constant(X_rows, has_constant="add"))
        elif self._method == "ridge":
            yp = self._model.predict(X_rows)[:, 0]
        else:
            yp = self._model.predict(X_rows)
        return self._rectify(yp)

    # -- recursive forecasting ------------------------------------------------------------

    def _forge(self, X_window: np.ndarray, y_fcast: np.ndarray, h: int) -> np.ndarray:
        """
        The input rows for horizon h (ScalaTion's `forge`): the lags of the earlier horizons
        replaced by their forecasts (the subclass knows the lag structure of its X).

        Parameters
        ----------
        X_window : np.ndarray, shape (rows, n)
            The input rows being forecast.
        y_fcast : np.ndarray, shape (rows, hh)
            The forecasts so far (columns 0 .. h - 2 are used).
        h : int
            The horizon (>= 2).
        """
        raise NotImplementedError(f"{self.modelName}._forge: the model class must implement it")

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end) recursively (horizon 1 from
        the input rows, the later horizons from the forged rows) and records them in `Yf` (original
        units).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
            The forecasts (model units).

        Raises
        ------
        ValueError
            If the model has not been trained, or if hh > 1 and X has no lag structure to forge.
        """
        if self._params is None:
            raise ValueError("The model has not been trained. Call train first.")
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = np.zeros((t_en - t_st, self.hh))
        X_window = self._X[t_st:t_en]
        y_fcast[:, 0] = self._predict(X_window)

        if self.hh > 1 and not self._useforge:
            raise ValueError("Cannot use forge when the input matrix X was not built by _build_matrix.")

        for h in range(2, self.hh + 1):
            y_fcast[:, h - 1] = self._predict(self._forge(X_window, y_fcast, h))

        self._store(t_st, t_en, y_fcast)
        return y_fcast

    # -- F E A T U R E   S E L E C T I O N  (ScalaTion's `Forecaster_Reg`) -----------------------

    @property
    def mcols(self) -> Cols:
        """The set of columns (numbers) for the features in this model."""
        return list(range(self._X.shape[1]))

    @property
    def fs_rng(self) -> range:
        """The rows feature selection fits on (training rows, or all rows when `fullset_FS`), cut off
        at the first undefined (NaN) response."""
        m = len(self._y)
        t_rng = range(0, m) if FeatureSelection.fullset_FS else range(0, self._tr_size)
        nan = np.where(np.isnan(np.asarray(self._y, dtype=float)))[0]
        end = int(nan[0]) if len(nan) else -1
        return t_rng if end < 0 else range(t_rng.start, min(t_rng.stop, end))

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "Forecaster_Reg":
        """
        Builds the forecaster on the columns `cols` of the input matrix, e.g., the best model after
        feature selection: `getModel (getBest ().mod_cols)` (the model classes implement it, see
        `ARX.getModel`).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        """
        raise NotImplementedError(f"{self.modelName}.getModel: the model class must implement it")

    def resetBest(self) -> None:
        """Resets the best step to the default."""
        self._theBest = BestStep()

    def getBest(self) -> BestStep:
        """The best step/model found from feature selection (`getModel (getBest ().mod_cols)` is the forecaster)."""
        return self._theBest

    def _updateBest(self, best: BestStep, qk: int) -> None:
        """Replaces the best step when the new one is better on measure qk."""
        if best.qof is not None:
            if self._theBest.qof is None or best.gt(float(self._theBest.qof[qk])):
                self._theBest = best

    def buildNrun(self, cols_j: Sequence[int]) -> Tuple["Forecaster_Reg", np.ndarray]:
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
        path = SubsetSearch.forward(self._X.shape[1], lambda c: self.evalCols(c, qk), [0], qk=qk)
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
        path = SubsetSearch.backward(self._X.shape[1], lambda c: self.evalCols(c, qk), list(range(first)),
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
        path = SubsetSearch.stepwise(self._X.shape[1], lambda c: self.evalCols(c, qk), [0], swap=swap, qk=qk)
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
        path = SubsetSearch.beam(self._X.shape[1], lambda c: self.evalCols(c, qk), [0], bk=bk, qk=qk)
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
        path = SubsetSearch.cem(self._X.shape[1], lambda c: self.evalCols(c, qk), list(range(first)), prior,
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

    # -- R E P O R T I N G   T H E   S E L E C T I O N  (the hooks of `reportSelection`) --------

    def selectedModel(self, cols: Cols, h: int) -> "Forecaster_Reg":
        """The forecaster on a selected column set (the hook of `reportSelection`): `getModel`; h is ignored."""
        return self.getModel(cols)

    def testSelected(self, mod: "Forecaster_Reg", inSample: bool, tnt: bool, rc: int,
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
