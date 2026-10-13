"""
ARX_SR_D Model: Direct Multi-Horizon ARX with Symbolic Regression (SR) Terms

The DIRECT counterpart of `ARX_SR`: one regression per horizon on the same design matrix --
trend, the lags of y and of its transformations (`fEndo`), the lags of each exogenous variable
and of its transformations (`fExo`), and optional ENDO-EXO cross terms -- so no forging of
forecasts into lags is needed.  With `rescale`, y and the exogenous variables are scaled first
(scalers fitted on the training rows) and the transformations act on the scaled series.

The Python counterpart of ScalaTion's `ARX_SR_D` (`ARX_SR_D.apply` / `ARX_SR_D.rescale`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Util.build_matrix import build_matrix_Y, fit_scaler
from Util.transform import PowForm, Transform
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr import ARX_SR


class ARX_SR_D(ARX_D):
    """
    Direct multi-horizon ARX with Symbolic Regression terms (ARX_SR_D).

    Attributes
    ----------
    tForms : dict
        "tForm_y", "tForm_exo" (scalers or None), "fEndo" (transforms of y), "fExo" (one list per
        exogenous variable), "tr_size".
    fExo_sz : list of int
        The number of transformations per exogenous variable.
    fname : list of str
        The column labels (`ARX_SR.form_names`).

    See Also
    --------
    ARX_D : the direct forecasting logic (train per horizon, forecast, rollValidate).
    ARX_SR : the recursive counterpart and the shared `_build_matrix` / `form_names`.
    """

    def __init__(self,
                 args: Dict[str, Any],
                 y: np.ndarray,
                 hh: int,
                 xe: Optional[np.ndarray] = None,
                 fEndo: Optional[Sequence[Transform]] = None,
                 fExo: Optional[Sequence[Sequence[Transform]]] = None,
                 method: str = "sm_ols",
                 X: Optional[np.ndarray] = None) -> None:
        """
        Initializes the ARX_SR_D forecaster (parameters as in `ARX_SR`).

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
        self._tForms = {"tForm_y": None, "tForm_exo": None, "fEndo": fEndo, "fExo": fExo}
        self._fExo_sz = [len(s) for s in fExo]
        super().__init__(args, y, hh, xe, method, X)
        self._modelName = f"ARX_SR_D_{self.args['p']}_{self.args['q']}_{self._n_exo}"

    @classmethod
    def rescale(cls,
                args: Dict[str, Any],
                y: np.ndarray,
                hh: int,
                xe: Optional[np.ndarray] = None,
                fEndo: Optional[Sequence[Transform]] = None,
                fExo: Optional[Sequence[Sequence[Transform]]] = None,
                method: str = "sm_ols",
                tForm: Optional[Type[TransformerMixin]] = None,
                X: Optional[np.ndarray] = None) -> "ARX_SR_D":
        """
        Alternative constructor that rescales y and the exogenous variables before the
        transformations (scalers fitted on the training rows); `StandardScaler` turns the
        non-negativity constraint off.
        """
        mod = cls(args, y, hh, xe, fEndo, fExo, method, np.array([0]))
        mod._X = X
        mod._tForms["tr_size"] = mod._tr_size
        if tForm is not None:
            mod._tForms["tForm_y"], mod._tForms["tForm_exo"] = tForm(), tForm()
            if isinstance(mod._tForms["tForm_y"], StandardScaler):
                mod._nneg = False
        else:
            mod._tForms["tForm_y"] = MinMaxScaler(feature_range=mod._lu)
            mod._tForms["tForm_exo"] = MinMaxScaler(feature_range=mod._lu)

        if mod._X is None:
            mod._useforge = True
            mod._set_X(mod._build_matrix(args=mod.args, y=mod._y, xe=mod._xe, tForms=mod._tForms))
        else:
            mod._X = fit_scaler(mod._tForms["tForm_exo"], mod._X, mod._tr_size).transform(mod._X)
            fit_scaler(mod._tForms["tForm_y"], mod._y, mod._tr_size)

        mod._yForm = mod._tForms["tForm_y"]
        mod._y = mod._tForms["tForm_y"].transform(mod._y.reshape(-1, 1)).flatten()
        mod._Y = build_matrix_Y(mod._y, hh)
        return mod

    @staticmethod
    def _build_matrix(args: Dict[str, Any],
                      y: np.ndarray,
                      xe: Optional[np.ndarray] = None,
                      tForms: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """The design matrix of `ARX_SR` (same columns, same order)."""
        return ARX_SR._build_matrix(args, y, xe, tForms)

    @property
    def fname(self) -> List[str]:
        """The column labels of X, in ScalaTion's naming (generic when X was given)."""
        if self._fname is not None:
            return self._fname
        if self._useforge:
            return ARX_SR.form_names(self.args, self._n_exo, len(self._tForms["fEndo"]), self._fExo_sz)
        return [f"x{j}" for j in range(self._X.shape[1])]

    def __repr__(self) -> str:
        s = f"{self.modelName}(p={self.args['p']}, q={self.args['q']}, n_exo={self._n_exo}, fEndo={self._tForms['fEndo']}"
        return s + f"), hh = {self.hh}, skip = {self.args['skip']}, method={self._method}"

    # -- feature selection: the sub-model (ScalaTion's `ARX_SR_D` inherits `ARX_D.getModel`) -----

    def getModel(self, cols: Optional[Sequence[int]] = None) -> "ARX_SR_D":
        """
        Builds an `ARX_SR_D` on the columns `cols` of X: the same y, Y, hh, scaling, transforms,
        hyper-parameters and method, the column labels of the selected terms (ScalaTion's
        `ARX_SR_D` inherits `ARX_D.getModel`, a direct model on the selected columns; here the
        sub-model keeps the class and name of its parent).  `getModel_h` is `ARX_D`'s.

        Parameters
        ----------
        cols : sequence of int, optional
            The columns of the design matrix with the selected features (defaults to all).
        """
        cols = self.mcols if cols is None else list(cols)
        sub = ARX_SR_D(self.args, self.getY(), self.hh, None, self._tForms["fEndo"], None, self._method,
                       self._X[:, cols])
        return self._subModel(sub, cols)
