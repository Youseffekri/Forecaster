"""
ARX_SR_D Model: Direct Multi-Horizon ARX with Symbolic Regression (SR) Terms

The DIRECT counterpart of `ARX_SR`: one regression per horizon on the same design matrix --
trend, the lags of y and of its transformations (`fEndo`), the lags of each exogenous variable
and of its transformations (`fExo`), and optional ENDO-EXO cross terms -- so no forging of
forecasts into lags is needed.  With `rescale`, y and the exogenous variables are rescaled first
(`tFormT` fitted on the training rows) and the transformations act on the rescaled series.
The sub-models of feature selection are `ARX_D`'s (`getModel`, `getModel_h`), as in ScalaTion.

The Python counterpart of ScalaTion's `ARX_SR_D` (`ARX_SR_D.apply` / `ARX_SR_D.rescale`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-17
Version: 2.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Util.make_matrix4ts import backfill_matrix
from Util.transform import PowForm, Transform, TransformT
from modeling.forecasting.arx import _rescaleData
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr import ARX_SR


class ARX_SR_D(ARX_D):
    """
    Direct multi-horizon ARX with Symbolic Regression terms (ARX_SR_D).

    Attributes
    ----------
    tForms : dict
        "tForm_y" (the transform of y or None), "fEndo" (transforms of y), "fExo" (one list per
        exogenous variable).
    fExo_sz : list of int
        The number of transformations per exogenous variable.
    fname : list of str
        The column labels (`ARX_SR.formNames`).

    See Also
    --------
    ARX_D : the direct forecasting logic (forecast, getModel, getModel_h).
    ARX_SR : the recursive counterpart and the shared `buildMatrix` / `formNames`.
    """

    ARGS = ARX_SR.ARGS

    def __init__(self,
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 args: Any = None,
                 fEndo: Optional[Sequence[Transform]] = None,
                 fExo: Optional[Sequence[Sequence[Transform]]] = None,
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_SR_D forecaster (parameters as in `ARX_SR`), building the design matrix
        (ScalaTion's `ARX_SR_D.apply`) unless a ready matrix X is given.

        Raises
        ------
        ValueError
            If `fExo` does not have one entry per exogenous variable.
        """
        n_exo = xe.shape[1] if xe is not None else 0
        fEndo = [PowForm()] if fEndo is None else list(fEndo)
        fExo = [[] for _ in range(n_exo)] if fExo is None else [list(s) for s in fExo]
        if len(fExo) != n_exo:
            raise ValueError(f"Length of fExo {len(fExo)} must = number of exogenous variables {n_exo}")
        self._tForms = {"tForm_y": None, "fEndo": fEndo, "fExo": fExo}
        self._fExo_sz = [len(s) for s in fExo]
        super().__init__(y, hh, xe, args, X if X is not None else np.zeros((len(y), 1)))
        self._modelName = f"ARX_SR_D_{self._hparam['p']}_{self._hparam['px']}_{self._n_exo}{self.hpTag}"
        if X is None:                                                    # ScalaTion's `ARX_SR_D.apply`
            xe_bfill = backfill_matrix(xe) if n_exo > 0 and self._hparam["px"] > 0 else None
            self._set_X(self.buildMatrix(xe_bfill, self._y, self._hparam, fEndo, fExo))
            self._useforge = True

    @classmethod
    def rescale(cls,
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                args: Any = None,
                fEndo: Optional[Sequence[Transform]] = None,
                fExo: Optional[Sequence[Sequence[Transform]]] = None,
                tFormT: TransformT = TransformT.MinMax) -> "ARX_SR_D":
        """
        Creates the model on rescaled data (ScalaTion's `ARX_SR_D.rescale`): y and the backfilled
        exogenous variables are transformed by `tFormT` fitted on the training rows (ScalaTion's
        `Model.trSize`; the fit rows while choosing; MinMax to [-2, 2] by default; Log1p keeps
        powers and logs defined, as in ScalaTion's tests), then the transformations are applied; a
        space where y can be negative turns the non-negativity off.
        """
        args, y_scl, xe_scl, tForm_y = _rescaleData(cls._argsOf(y, args), y, xe, tFormT)
        mod = cls(y_scl, hh, xe, args, fEndo, fExo, X=np.zeros((len(y_scl), 1)))
        mod._set_X(cls.buildMatrix(xe_scl, y_scl, mod._hparam, mod._tForms["fEndo"], mod._tForms["fExo"]))
        mod._useforge = True
        mod._tForms["tForm_y"] = tForm_y
        mod._setYForm(tForm_y, y)
        mod._recipe = lambda: cls.rescale(y, hh, xe, args, fEndo, fExo, tFormT)
        return mod

    @staticmethod
    def buildMatrix(xe_bfill: Optional[np.ndarray], y: np.ndarray, hparam: Dict[str, Any],
                    fEndo: Sequence[Transform], fExo: Sequence[Sequence[Transform]]) -> np.ndarray:
        """The design matrix of `ARX_SR` (same columns, same order)."""
        return ARX_SR.buildMatrix(xe_bfill, y, hparam, fEndo, fExo)

    @staticmethod
    def formNames(n_exo: int, hparam: Dict[str, Any], n_fEn: int, n_fExArr: Sequence[int]) -> List[str]:
        """The column labels (ScalaTion's `ARX_SR_D.formNames` = `ARX_SR.formNames`)."""
        return ARX_SR.formNames(n_exo, hparam, n_fEn, n_fExArr)

    @property
    def fname(self) -> List[str]:
        """The column labels of X, in ScalaTion's naming (generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return ARX_SR.formNames(self._n_exo, self._hparam, len(self._tForms["fEndo"]), self._fExo_sz)
        return [f"x{j}" for j in range(self._X.shape[1])]

    def __repr__(self) -> str:
        s = f"{self.modelName}(p={self._hparam['p']}, px={self._hparam['px']}, n_exo={self._n_exo}, fEndo={self._tForms['fEndo']}"
        return s + f"), hh = {self.hh}, skip = {self._hparam['skip']}, method={self._method}"
