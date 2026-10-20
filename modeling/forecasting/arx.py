"""
ARX Model for Time Series Forecasting

Implements an ARX (Auto-Regressive model with eXogenous variables) by linear regression on the
lags of y (1 to p) and of each exogenous variable (1 to px), with optional trend columns (`spec`)
forecasting RECURSIVELY: the h-steps ahead forecast feeds the forecasts
of the earlier steps back into the lag columns (`forge`), while the exogenous lags that would
lie in the future are hidden (`hide`).  The regression, the recursive forecasting and the
feature selection come from `Forecaster_Reg` (ScalaTion's `Forecaster_Reg`); `ARX` adds the lag
matrix, its names, the forge and the sub-model on a column set (`getModel`).  The Python counterpart
of ScalaTion's `ARX.scala`: the constructor `ARX (y, hh, xe, args)` builds the matrix as `ARX.apply`
does (a ready matrix X may be given instead), and `rescale`, `trend`, `fromTerms`, `buildMatrix`,
`formNames` are the companion object's factories and helpers; the settings (p, px, spec, method,
lam, scaled, ...) come from the `Args`; the transform type of a rescaling, `tFormT`, is an argument of
`rescale` / `fromTerms`, as in ScalaTion, and `rescale` fits it on the rows `Model` gives (the training
rows; the fit rows while choosing under `Model.validating`), as ScalaTion's fits on `Model.trSize`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-19
Version: 2.1
License: MIT License
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Util.make_matrix4ts import backfill_matrix, makeMatrix4L, makeMatrix4T, makeMatrix4Y, trend_names
from Util.transform import TransformT
from modeling.forecasting.forecaster import Args, Forecaster, Model
from modeling.forecasting.forecaster_reg import Forecaster_Reg


def _checkTFormT(args: Args, tFormT: TransformT) -> None:
    """A flaw when args.tFormT (the RUN setting `run (args)` passes to `rescale`) is set to another type than
    the `tFormT` given: a model never reads it, so a script that still sets it would silently rescale by MinMax."""
    if args.tFormT is not None and args.tFormT is not tFormT:
        raise ValueError(f"rescale: tFormT is an argument of rescale (ScalaTion's signature), not a setting of the model: "
                         f"pass tFormT = {args.tFormT} to rescale (args.tFormT is the run setting of run (args))")


def _rescaleData(args: Args, y: np.ndarray, xe: Optional[np.ndarray], tFormT: Optional[TransformT] = TransformT.MinMax):
    """
    The common part of the `rescale` factories (ScalaTion's `ARX*.rescale`): the transform of type
    `tFormT` (None => MinMax) fitted on the leading rows `Model.fitRows` gives (the training
    rows, ScalaTion's `Model.trSize`, or the fit rows while choosing) of y and of the backfilled
    exogenous variables (column by column), applied to the whole series; NormForm turns the
    non-negativity off.

    Parameters
    ----------
    args : Args
        The settings of the model being built (changed in place: nneg off for Norm).
    tFormT : TransformT, default=TransformT.MinMax
        The transform type of the rescaling (ScalaTion's `rescale (..., tFormT)`), a construction
        choice of the factory like `fEndo` / `fExo`, not a setting of the model.

    Returns
    -------
    (Args, np.ndarray, np.ndarray or None, Transform)
        The settings, the rescaled y, the rescaled backfilled xe (None without exogenous lags) and
        the transform of y.
    """
    tFormT = TransformT.MinMax if tFormT is None else tFormT
    _checkTFormT(args, tFormT)
    if tFormT is TransformT.Norm:
        args.nneg = False                                                # ScalaTion: if tFormT.name == "NormForm" then hparam("nneg") = 0
    y = np.asarray(y, dtype=float)
    tr_size = Model.fitRows(len(y), args.te_ratio)
    tForm_y = tFormT.form(y[:tr_size])                               # the statistics of the leading rows, for In-ST and TnT
    y_scl = tForm_y.f(y)
    xe_scl = None
    if xe is not None and xe.shape[1] > 0 and args.px > 0:
        xe_bfill = backfill_matrix(np.asarray(xe, dtype=float))     # backfill each exogenous variable
        tForm_exo = tFormT.form(xe_bfill[:tr_size])
        xe_scl = tForm_exo.f(xe_bfill)                               # rescale the backfilled exogenous variables
    return args, y_scl, xe_scl, tForm_y


class ARX(Forecaster_Reg):
    """
    Auto-Regressive model with eXogenous variables (ARX), recursive multi-horizon forecasting.

    Attributes
    ----------
    method : {"sm_ols", "sk_lr", "ridge"}, default="sm_ols"
        The regression (args.method): statsmodels OLS, scikit-learn LinearRegression, or the
        centered ridge of ScalaTion's `CenteredRidge` (penalty args.lam, default 0.1; args.scaled).
    xe : np.ndarray, shape (m, n_exo), optional
        The exogenous variables aligned with `y`.
    n_exo : int
        The number of exogenous variables.
    useforge : bool
        Whether X was built by `buildMatrix` (needed to forge the lags for h > 1).
    tForms : dict
        "tForm_y": the transform of y (None unless made by `rescale` or `fromTerms` with a tFormT).
    modelName : str
        "ARX_{p}_{px}_{n_exo}".
    fname : list of str
        The column names: the trend names, yl{p} .. yl1, xe{j}l{px} .. xe{j}l1 (`formNames`).

    See Also
    --------
    Forecaster_Reg : the regression, the recursive forecasting and the feature selection.
    Forecaster : the base forecasting logic (Yf, rollValidate, inSample_Test, tnT_Test).
    """

    ARGS = ("p", "px", "spec", "method", "lam", "scaled") + Forecaster.ARGS

    def __init__(self,
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 args: Any = None,
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX forecaster, building the input matrix from y and xe (ScalaTion's
        `ARX.apply`) unless a ready matrix X is given.

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            The endogenous time series.
        hh : int
            The maximum forecasting horizon (h = 1 to hh).
        xe : np.ndarray, shape (m, n_exo), optional
            The exogenous variables aligned with y.
        args : Args or dict, optional
            The settings: p (lags of y), px (lags of each exogenous variable), spec (number of trend
            columns, 0..5), nneg, skip, method ("sm_ols", "sk_lr", "ridge"), lam and scaled (for
            "ridge"), te_ratio; None => the defaults (`Args.help (ARX)`).
        X : np.ndarray, shape (m, n), optional
            A ready-made input matrix (built from y and xe when None).
        """
        self._xe = xe
        self._n_exo = xe.shape[1] if xe is not None else 0
        if not hasattr(self, "_tForms"):
            self._tForms = {"tForm_y": None}
        super().__init__(y, hh, args, X)
        self._useforge = self._X is None
        self._modelName = f"ARX_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{self.hpTag}"

        if self._X is None:
            xe_bfill = backfill_matrix(xe) if xe is not None and xe.shape[1] > 0 and self._hparam["px"] > 0 else None
            self._set_X(self.buildMatrix(xe_bfill, self._y, self._hparam))

    @classmethod
    def rescale(cls,
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                args: Any = None,
                tFormT: TransformT = TransformT.MinMax) -> "ARX":
        """
        Creates the model on rescaled data (ScalaTion's `ARX.rescale`): y and the backfilled exogenous
        variables are transformed by `tFormT` fitted on the training rows (ScalaTion's
        `Model.trSize`; the fit rows while choosing under `Model.validating`: `Model.fitRows`), the matrix
        is built from the rescaled series, and forecasts are inverted before being stored in `Yf`.
        `refitTransform ()` rebuilds the model with the transforms fitted on the training rows.

        Parameters
        ----------
        y, hh, xe, args
            As in the constructor.
        tFormT : TransformT, default=TransformT.MinMax
            The transform type of the rescaling (MinMax to [-2, 2]; Norm turns the non-negativity
            off; Log1p ...), as ScalaTion's `rescale (..., tFormT)`.

        Returns
        -------
        ARX
            The model on the rescaled series.
        """
        args, y_scl, xe_scl, tForm_y = _rescaleData(cls._argsOf(y, args), y, xe, tFormT)
        mod = cls(y_scl, hh, xe_scl, args, X=np.zeros((len(y_scl), 1)))
        mod._set_X(cls.buildMatrix(xe_scl, y_scl, mod._hparam))        # xe_scl is already backfilled
        mod._useforge = True
        mod._tForms["tForm_y"] = tForm_y
        mod._setYForm(tForm_y, y)
        mod._recipe = lambda: cls.rescale(y, hh, xe, args, tFormT)
        return mod

    @classmethod
    def trend(cls,
              y: np.ndarray,
              hh: int,
              xe: Optional[np.ndarray] = None,
              args: Any = None) -> "ARX":
        """
        Creates the TREND model (ScalaTion's `ARX.trend`): the settings with p = px = 0, so the input
        matrix holds the trend columns of `spec` only (ScalaTion passes the given hparam to
        `buildMatrix`, which keeps the lags of y under p = 0; here the matrix matches p = px = 0).

        Parameters
        ----------
        y, hh, xe, args
            As in the constructor (args.spec must be at least 1; skip defaults to 0).
        """
        args = cls._argsOf(y, args)
        if args.spec < 1:
            raise ValueError("trend: spec must be at least 1 (a trend column)")
        args.update(p=0, px=0)
        if args.skip is None:
            args.skip = 0
        return cls(y, hh, xe, args)

    @classmethod
    def fromTerms(cls, formula, dataset: Dict[str, np.ndarray], hh: int, args: Any = None,
                  tFormT: Optional[TransformT] = None, trSize: int = -1):
        """
        Creates the model from the terms of a formula (ScalaTion's `ARX.fromTerms`), with one
        column per term (lagged variable, transform, interaction or wave), so skipped lags and
        interactions that contiguous lags cannot express are supported.  Rows are not trimmed
        (each term pads its first lag values).  With `tFormT`, every base column is transformed
        (fitted on its training slice, as in `rescale`) before the terms are evaluated, and the
        response's transform is kept for inverting forecasts.  The matrix has NO intercept
        column.  Recursive forecasting beyond h = 1 needs the standard lag layout (the plain
        lags of y then of each exogenous variable, in decreasing order); for any other formula
        use the direct `ARX_D.fromTerms` / `NARX_SR_D.fromTerms`.

        Parameters
        ----------
        formula : ARX_Formula
            The formula (response and predictor terms).
        dataset : dict of str -> np.ndarray
            The named columns.
        hh : int
            The maximum forecasting horizon.
        args : Args or dict, optional
            The settings: method, lam and scaled (the ridge regression standardizes the term columns),
            nneg, skip (defaults to the formula's largest lag), te_ratio; p, px and spec = 0 come from
            the formula (ScalaTion's `lagHP`).
        tFormT : TransformT, optional
            The transform type applied to every base column (ScalaTion's `fromTerms (..., tFormT)`);
            None => the formula's own (its "TFORM:" clause).
        trSize : int, default=-1
            The number of leading rows the transforms are fit on (ScalaTion's `fromTerms (...,
            trSize)`; -1 => `Model.fitRows`: the training rows, or the fit rows while choosing).
        """
        args = Args.of(args)
        f = formula if tFormT is None else formula.__class__(formula.resp, formula.preds, tFormT)
        m = len(next(iter(dataset.values())))
        n_fit = Model.fitRows(m, args.te_ratio, trSize)
        td = f.prepare(dataset, n_fit)                                   # checks and base-column transforms
        X = f.matrix(td.data)                                         # one column per term
        hp = f.lagHP()
        args.update(p=hp["p"], px=hp["px"], spec=0)
        if args.skip is None:
            args.skip = f.maxLag
        mod = cls(td.y, hh, None, args, X=X)
        mod._n_exo = hp["n_exo"]
        mod._modelName = f"{cls.__name__}_{hp['p']}_{hp['px']}_{hp['n_exo']}{mod.hpTag}"
        mod._fname = f.names
        mod._tForms["tForm_y"] = td.tForm_y
        mod._setYForm(td.tForm_y, dataset[f.resp.name])               # column 0 of Yf stays in ORIGINAL units
        mod._useforge = ARX._standard_layout(f, hp)
        if hasattr(mod, "_Y"):
            mod._Y = makeMatrix4Y(mod._y, hh)
        mod._recipe = lambda: cls.fromTerms(formula, dataset, hh, args, tFormT)
        return mod

    @staticmethod
    def _standard_layout(f, hp: Dict[str, int]) -> bool:
        """Whether the terms are the plain lags y~p..y~1 then x~px..x~1 per exogenous variable, in that
        order (the layout `forge` can extend to h > 1; only plain lagged variables have such labels)."""
        yn = f.resp.name
        want = [f"{yn}~{k}" for k in range(hp["p"], 0, -1)]
        for v in sorted(f.vars - {yn}):
            want += [f"{v}~{k}" for k in range(hp["px"], 0, -1)]
        return f.names == want

    @staticmethod
    def buildMatrix(xe_bfill: Optional[np.ndarray], y: np.ndarray, hparam: Dict[str, Any]) -> np.ndarray:
        """
        Builds the input matrix (ScalaTion's `ARX.buildMatrix`): the trend columns, the lags of y
        (yl{p} .. yl1) and the lags of each (backfilled) exogenous variable (xe{j}l{px} .. xe{j}l1).
        ENDO-EXO cross terms belong to `ARX_SR` (ScalaTion's `crx`), not to `ARX`.

        Parameters
        ----------
        xe_bfill : np.ndarray, shape (m, n_exo), optional
            The backfilled exogenous variables (None => none).
        y : np.ndarray, shape (m,)
            The endogenous series.
        hparam : dict
            "p", "px", "spec".

        Returns
        -------
        np.ndarray, shape (m, n)
        """
        p, q, spec = int(hparam["p"]), int(hparam["px"]), int(hparam["spec"])
        blocks = [makeMatrix4T(len(y), spec)] if spec > 0 else []
        if p > 0:
            blocks.append(makeMatrix4L(y, p))
        if xe_bfill is not None and xe_bfill.shape[1] > 0 and q > 0:
            blocks.extend(makeMatrix4L(xe_bfill[:, j], q) for j in range(xe_bfill.shape[1]))
        return np.column_stack(blocks)

    @staticmethod
    def formNames(n_exo: int, hparam: Dict[str, Any], n_fEn: int = 0, n_fExArr: Optional[Sequence[int]] = None) -> List[str]:
        """
        The column names of the input matrix (ScalaTion's `ARX.formNames`): the trend names,
        yl{p} .. yl1, then xe{j}l{px} .. xe{j}l1 per exogenous variable (n_fEn, n_fExArr unused here).
        """
        p, q, spec = int(hparam["p"]), int(hparam["px"]), int(hparam["spec"])
        names = trend_names(spec) + [f"yl{k}" for k in range(p, 0, -1)]
        for j in range(n_exo):
            names += [f"xe{j}l{k}" for k in range(q, 0, -1)]
        return names

    @property
    def fname(self) -> List[str]:
        """The column names of X (generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return self.formNames(self._n_exo, self._hparam)
        return [f"x{j}" for j in range(self._X.shape[1])]

    def forge(self, X_window: np.ndarray, y_fcast: np.ndarray, h: int) -> np.ndarray:
        """
        The input rows for horizon h (ScalaTion's `forge`): the trend columns, the lags of y
        with the h - 1 most recent ones replaced by the forecasts of the earlier horizons, and
        the exogenous lags with the future ones hidden.

        Parameters
        ----------
        X_window : np.ndarray, shape (rows, n)
            The input rows being forecast.
        y_fcast : np.ndarray, shape (rows, hh)
            The forecasts so far (columns 0 .. h - 2 are used).
        h : int
            The horizon.
        """
        p, q, spec = self._hparam["p"], self._hparam["px"], self._hparam["spec"]
        idx_endo = spec
        idx_exo = idx_endo + p
        x_trend = X_window[:, :idx_endo]
        x_endo_act = X_window[:, idx_endo + (h - 1): idx_endo + p]            # the actual lags still usable
        idx_fcast = max(x_endo_act.shape[1] - (p - h + 1), 0)
        x_endo_fcast = y_fcast[:, idx_fcast:h - 1]                              # the forecasts of the earlier steps
        X_h = np.column_stack((x_trend, x_endo_act, x_endo_fcast))

        if self._n_exo > 0 and q > 0:
            x_exo = [self.hide(X_window[:, idx_exo + j * q: idx_exo + (j + 1) * q], h) for j in range(self._n_exo)]
            X_h = np.column_stack([X_h] + x_exo)
        return X_h

    def hide(self, z: np.ndarray, h: int, fill: bool = True) -> np.ndarray:
        """
        Hides the exogenous lags that lie in the future at horizon h (ScalaTion's `hide`): the
        lag columns are shifted so that the last known value (lag 1 of the forecast origin)
        fills the h - 1 most recent slots (or zero when not `fill`).

        Parameters
        ----------
        z : np.ndarray, shape (rows, q)
            The lag matrix of one exogenous variable (lags q .. 1).
        h : int
            The horizon.
        fill : bool, default=True
            Whether to fill with the last known value (else zero).
        """
        z_last = z[:, -1]
        if h > z.shape[1]:
            return np.column_stack([z_last for _ in range(z.shape[1])]) if fill else np.zeros(z.shape)
        z_shift = np.column_stack([z_last for _ in range(h - 1)]) if h > 1 else np.zeros((z.shape[0], 0))
        z_ = z[:, h - 1:]
        return np.column_stack((z_, z_shift))

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "ARX":
        """
        Builds an `ARX` on the columns `cols` of X (ScalaTion's `ARX.getModel`): the same y, transforms,
        scaling, hyper-parameters and regression method, the feature names of the columns -- e.g., the
        best model after feature selection, `getModel (getBest ().mod_cols)`.  A sub-model holds the
        selected columns only, so it has no lag structure to forge: it is built with hh = 1 (the
        horizon ScalaTion's `testSelected` tests a selected recursive model at).

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        """
        cols = self.mcols if cols is None else list(cols)
        sub = ARX(self.getY(), 1, None, self._subArgs(), X=self._X[:, cols])
        return self._subModel(sub, cols, modelName=f"ARX_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{sub.hpTag}")
