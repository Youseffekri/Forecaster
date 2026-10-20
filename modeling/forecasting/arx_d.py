"""
ARX_D Model for Time Series Forecasting

Implements ARX_D (Auto-Regressive model with eXogenous variables, DIRECT multi-horizon
forecasting): the same input matrix as `ARX` (trend columns, lags of y, lags of each exogenous
variable) with one regression per horizon on the target matrix Y, so no forecasts are fed back
into the lags.  The regression and its fit live in `Forecaster_D` (ScalaTion's `reg`); `ARX_D`
adds the matrix, the forecasts (`forecast`, `predict_rows`) and the sub-models (`getModel`,
`getModel_h`).  The Python counterpart of ScalaTion's `ARX_D.scala` (the constructor
`ARX_D (y, hh, xe, args)` builds the matrix as `ARX_D.apply` does; `rescale`, `fromTerms`,
`buildMatrix`, `formNames` as the companion's; the settings come from the `Args`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-19
Version: 2.1
License: MIT License
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Util.make_matrix4ts import backfill_matrix
from Util.transform import TransformT
from modeling.forecasting.arx import ARX, _rescaleData
from modeling.forecasting.forecaster import Forecaster
from modeling.forecasting.forecaster_d import Forecaster_D


class ARX_D(Forecaster_D):
    """
    Auto-Regressive model with eXogenous variables, DIRECT multi-horizon forecasting (ARX_D).

    Attributes
    ----------
    method : {"enet", "ridge", "sm_ols", "sk_lr"}, default="enet"
        The regression per horizon (args.method): the centered elastic net of ScalaTion's
        `CenteredElasticNetMV` (penalty args.lam, default 0.1, share args.l1_ratio, default 0.5;
        args.scaled) -- the default of every regression-based forecaster, as ScalaTion's `REGRESSION`
        import --, the centered ridge, statsmodels OLS or scikit-learn LinearRegression.
    xe : np.ndarray, shape (m, n_exo), optional
        The exogenous variables aligned with `y`.
    n_exo : int
        The number of exogenous variables.
    tForms : dict
        "tForm_y": the transform of y (None unless made by `rescale` or `fromTerms` with a tFormT).
    modelName : str
        "ARX_D_{p}_{px}_{n_exo}".
    fname : list of str
        The column names (`ARX.formNames`).

    See Also
    --------
    Forecaster_D : the direct forecasting logic (Y, the regression, rollValidate, tests).
    ARX : the recursive counterpart and the shared `buildMatrix` / `formNames`.
    """

    ARGS = ("p", "px", "spec", "method", "lam", "l1_ratio", "scaled") + Forecaster.ARGS

    def __init__(self,
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 args: Any = None,
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_D forecaster (parameters as in `ARX`: the settings p, px, spec, method --
        default "sm_ols" --, lam, scaled, nneg, skip, te_ratio in args), building the input matrix from
        y and xe (ScalaTion's `ARX_D.apply`) unless a ready matrix X is given.
        """
        super().__init__(y, hh, args)
        self._X = X
        self._xe = xe
        self._n_exo = xe.shape[1] if xe is not None else 0
        self._useforge = self._X is None
        self._modelName = f"ARX_D_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{self.hpTag}"

        if not hasattr(self, "_tForms"):
            self._tForms = {"tForm_y": None}

        if self._X is None:
            xe_bfill = backfill_matrix(xe) if xe is not None and xe.shape[1] > 0 and self._hparam["px"] > 0 else None
            self._set_X(self.buildMatrix(xe_bfill, self._y, self._hparam))

    @classmethod
    def rescale(cls,
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                args: Any = None,
                tFormT: TransformT = TransformT.MinMax) -> "ARX_D":
        """
        Creates the model on rescaled data (ScalaTion's `ARX_D.rescale`): y and the backfilled
        exogenous variables are transformed by `tFormT` fitted on the training rows (ScalaTion's
        `Model.trSize`; the fit rows while choosing under `Model.validating`: `Model.fitRows`; MinMax to
        [-2, 2] by default; Norm turns the non-negativity off; Log1p ...); forecasts are inverted
        before being stored in `Yf`, and they are not rectified in a space where y can be negative
        (`centered`).  `refitTransform ()` rebuilds the model with the transforms fitted on the
        training rows.

        Returns
        -------
        ARX_D
            The model on the rescaled series (Y built from the rescaled y).
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
    def fromTerms(cls, formula, dataset: Dict[str, np.ndarray], hh: int, args: Any = None,
                  tFormT: Optional[TransformT] = None, trSize: int = -1) -> "ARX_D":
        """
        Creates the direct model from the terms of a formula (ScalaTion's `ARX_D.fromTerms`): one
        column per term, the response's transform kept for inverting forecasts, no intercept
        column; every formula is supported since no forging is needed (see `ARX.fromTerms` for the
        settings).
        """
        mod = ARX.fromTerms.__func__(cls, formula, dataset, hh, args, tFormT, trSize)
        mod._useforge = True                                          # no forging in a direct model
        return mod

    @staticmethod
    def buildMatrix(xe_bfill: Optional[np.ndarray], y: np.ndarray, hparam: Dict[str, Any]) -> np.ndarray:
        """The input matrix of `ARX` (same columns, same order; ScalaTion's `ARX_D` calls `ARX.buildMatrix`)."""
        return ARX.buildMatrix(xe_bfill, y, hparam)

    @staticmethod
    def formNames(n_exo: int, hparam: Dict[str, Any], n_fEn: int = 0, n_fExArr: Optional[Sequence[int]] = None) -> List[str]:
        """The column names (ScalaTion's `ARX_D.formNames` = `ARX.formNames`)."""
        return ARX.formNames(n_exo, hparam, n_fEn, n_fExArr)

    @property
    def fname(self) -> List[str]:
        """The column names of X (generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return ARX.formNames(self._n_exo, self._hparam)
        return [f"x{j}" for j in range(self._X.shape[1])]

    def forecast(self, t_start: int = -1, t_end: int = -1) -> np.ndarray:
        """
        Forecasts h = 1..hh steps ahead for the rows [t_start, t_end) directly (ScalaTion's
        `ARX_D.forecast` / `predict`: the regression's predictions, rectified when nneg) and records
        them in `Yf` (original units).

        Returns
        -------
        np.ndarray, shape (t_end - t_start, hh)
            The forecasts (model units).

        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        t_st, t_en = self._adjust_range(t_start, t_end)
        y_fcast = self._rectify(self._predict(self._X[t_st:t_en]))
        self._store(t_st, t_en, y_fcast)
        return y_fcast

    def predict_rows(self, rows: np.ndarray) -> np.ndarray:
        """The forecasts (len(rows), hh) of the given rows in model units, without touching `Yf`
        (the scorer's evaluation)."""
        return self._rectify(self._predict(self._X[rows]))

    # -- feature selection: the sub-models (ScalaTion's `ARX_D.getModel` / `getModel_h`) -------

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "ARX_D":
        """
        Builds an `ARX_D` on the columns `cols` of X (ScalaTion's `ARX_D.getModel`, inherited by
        `ARX_SR_D` and `NARX_SR_D`): the same y, Y, hh, transforms, scaling, hyper-parameters and
        regression method, the feature names of the columns, e.g., the selected model after feature
        selection, `getModel(getBest().mod_cols)`; `getModel()` is a copy on all the columns.

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        """
        cols = self.mcols if cols is None else list(cols)
        sub = ARX_D(self.getY(), self.hh, None, self._subArgs(), X=self._X[:, cols])
        return self._subModel(sub, cols, modelName=f"ARX_D_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{sub.hpTag}")

    def getModel_h(self, cols: Optional[Sequence[int]] = None, h: int = 1) -> ARX:
        """
        Builds the single-horizon `ARX` at horizon h on the columns `cols` (ScalaTion's
        `ARX_D.getModel_h`: `ARX` is the base model of the ARX*_D family): an `ARX` with hh = 1 whose
        response is column h - 1 of the target matrix, on the m - h + 1 rows where that response is
        defined (ScalaTion keeps the NO_DOUBLE rows), with this model's regression method, scaling,
        training size (so the FS rows and the testing rows are this model's) and skip.

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the input matrix with the selected features (defaults to all).
        h : int, default=1
            The horizon (1 to hh).
        """
        if not 1 <= h <= self.hh:
            raise ValueError(f"getModel_h: horizon h = {h} out of range [1, {self.hh}]")
        cols = self.mcols if cols is None else list(cols)
        mh = len(self._y) - h + 1                                         # the rows whose horizon-h response is defined
        sub = ARX(self.getY()[h - 1:], 1, None, self._subArgs(), X=self.getX()[:mh, cols])
        self._subModel(sub, cols, modelName=f"ARX_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{sub.hpTag}")
        sub._y = self._Y[:mh, h - 1]                                      # the horizon-h response (model units)
        sub._te_size = mh - sub._tr_size
        sub._lead = h - 1                                                 # its row r targets y[r + h - 1]: the boundary rule
        return sub
