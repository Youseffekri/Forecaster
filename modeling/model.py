"""
Model Module: the Split of a Dataset and the Validation Switch

The Python counterpart of ScalaTion's `Model` object (the parts every chooser reads): ONE split of
the m rows of a dataset into fit, validation and testing rows, and ONE switch that says whether
a chooser (feature selection, hyper-parameter tuning, the search arms of LLM-NARX-SR) scores a
candidate on the validation rows or in-sample:

    [ <-- fitSize --> | <-- valSize --> | <-- teSize --> ]
    [ <--------- trSize -------------> ]

    TE_RATIO    the share of the FULL dataset for testing (0.2): teSize = round up (m TE_RATIO);
                0 => NO testing set (a competition: the examiner holds it), trSize = m, and the
                chooser's validation block is the last VAL_RATIO of the dataset
    VAL_RATIO   the share of the FULL dataset held out while choosing, = TE_RATIO by default so the
                validation block has the testing block's size (116 Covid-19 weeks -> 68 / 24 / 24)
    validating  the switch: True => a candidate is fitted on the fit rows and scored on the
                validation rows (rolling, `Forecaster.rollTest`); False => fitted and scored on the
                training rows, penalised (today's protocol)
    tuning      the second switch: True => every candidate a chooser evaluates (a column set, a
                formula) is tuned over its regression's knobs by `Forecaster.tune` (the ridge's
                lambda; lambda and l1_ratio for the elastic net) before it is scored, so the best
                step carries the knobs that made its score; needs `validating` (rev2.2: was `tuneLambda`)
    folds       what a tuner scores on: 0 => the validation block (`Forecaster.valScore`); k =>
                k-fold `cvScore (k)` on the chooser's rows (rev2.2)

Validation rolls over `valRng` with the retraining cycle of the test (`Forecaster.RC`).
`withValidation` runs a block of code with the switch on and restores it, so one line turns 80/20
into 60/20/20 for the base models, the scorer and the LLM models alike; `inSample` runs a block with
the switch OFF (the factories then fit their transforms on the training rows, whatever the switch
was: `Forecaster.refitTransform`):

    with Model.withValidation():          # or withValidation (0.25), withValidation (tuning = True)
        mod.selectFeatures (SelectionTech.Backward)

A model's own `te_ratio` (args.te_ratio; None => `TE_RATIO` when the model is built) may differ
from `TE_RATIO`; every size function takes the ratio as an argument for that, defaulting to the
class's.  `TE_RATIO` is checked when set, as ScalaTion's `Model.TE_RATIO_=` does: a ratio outside
(0.05, 0.95) is refused, except 0.

`fitRows (m, te_ratio, trSize)` is the number of leading rows a factory fits its transforms (and a
formula's nonlinear parameters) on: `trSize` when given (> 0), else `fitSize` while choosing and
`trSize` otherwise (ScalaTion's `Model.fitRows`).

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.4
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = ["Model"]


class _ModelMeta(type):
    """The metaclass of `Model`: `TE_RATIO` is a checked class property (ScalaTion's `Model.TE_RATIO`
    getter and `TE_RATIO_=` setter), so `Model.TE_RATIO = 0.3` is checked like a method call."""

    _TE_RATIO: float = 0.2                                               # share of the FULL dataset for testing

    @property
    def TE_RATIO(cls) -> float:
        """The ratio of the size of the TESTING set to the size of the FULL dataset."""
        return cls._TE_RATIO

    @TE_RATIO.setter
    def TE_RATIO(cls, ratio: float) -> None:
        """Sets the ratio; a flaw unless 0.05 < ratio < 0.95 or ratio = 0 (no testing set), as
        ScalaTion's `TE_RATIO_=`."""
        if ratio != 0.0 and not (0.05 < ratio < 0.95):
            raise ValueError(f"Model: testing ratio = {ratio} should be in (0.05, 0.95), or 0 for no testing set")
        cls._TE_RATIO = float(ratio)


class Model(metaclass=_ModelMeta):
    """
    The split of a dataset and the validation switch (ScalaTion's `Model` object).  Class
    attributes only: set `Model.TE_RATIO` (checked), `Model.VAL_RATIO`, `Model.validating` (or use
    `withValidation` / `inSample`), read the sizes through the static methods.
    """

    VAL_RATIO: float = 0.2                                               # share of the FULL dataset held out for choosing (= TE_RATIO)
    validating: bool = False                                             # the switch: choose on the validation rows (True) or in-sample (False)
    tuning: bool = False                                                 # the second switch: every candidate at its own best knobs (needs validating)
    folds: int = 0                                                       # what a tuner scores on: 0 => the validation block, k => k-fold cvScore

    # -- sizes -----------------------------------------------------------------------------

    @staticmethod
    def teSize(m: int, te_ratio: Optional[float] = None) -> int:
        """The size of the testing set (ScalaTion's `Model.teSize`: round up m * TE_RATIO; 0 when the
        ratio is 0: no testing set)."""
        r = Model.TE_RATIO if te_ratio is None else te_ratio
        return 0 if r == 0.0 else int(math.floor(m * r + 1.0))

    @staticmethod
    def trSize(m: int, te_ratio: Optional[float] = None) -> int:
        """The size of the training set (ScalaTion's `Model.trSize`): m - teSize (m)."""
        return m - Model.teSize(m, te_ratio)

    @staticmethod
    def valSize(m: int, val_ratio: Optional[float] = None) -> int:
        """The size of the validation block: round up m * VAL_RATIO (the testing block's rule, so the
        two blocks have one size when the ratios agree)."""
        r = Model.VAL_RATIO if val_ratio is None else val_ratio
        Model._checkRatio(r)
        return int(math.floor(m * r + 1.0))

    @staticmethod
    def fitSize(m: int, te_ratio: Optional[float] = None, val_ratio: Optional[float] = None) -> int:
        """The rows a candidate is fitted on while choosing: trSize (m) - valSize (m)."""
        return Model.trSize(m, te_ratio) - Model.valSize(m, val_ratio)

    # -- ranges ----------------------------------------------------------------------------

    @staticmethod
    def teRng(m: int, te_ratio: Optional[float] = None) -> range:
        """The index range of the testing rows (ScalaTion's `Forecaster.teRng`)."""
        return range(Model.trSize(m, te_ratio), m)

    @staticmethod
    def valRng(m: int, te_ratio: Optional[float] = None, val_ratio: Optional[float] = None) -> range:
        """The index range of the validation rows: fitSize (m) until trSize (m)."""
        return range(Model.fitSize(m, te_ratio, val_ratio), Model.trSize(m, te_ratio))

    @staticmethod
    def fitRng(m: int, te_ratio: Optional[float] = None, val_ratio: Optional[float] = None) -> range:
        """The index range of the fit rows: 0 until fitSize (m)."""
        return range(0, Model.fitSize(m, te_ratio, val_ratio))

    @staticmethod
    def splitStr(m: int, te_ratio: Optional[float] = None, val_ratio: Optional[float] = None) -> str:
        """The split as the banners print it: "fit / validation / test", e.g., "68 / 24 / 24"."""
        return f"{Model.fitSize(m, te_ratio, val_ratio)} / {Model.valSize(m, val_ratio)} / {Model.teSize(m, te_ratio)}"

    @staticmethod
    def fitRows(m: int, te_ratio: Optional[float] = None, trSize: int = -1) -> int:
        """
        The leading rows a factory fits its transforms (and a formula's nonlinear parameters) on
        (ScalaTion's `Model.fitRows`): `trSize` when given (> 0; `fromTerms` and `NARX_SR_D.apply` /
        `rescale` take it, as ScalaTion's do), else `fitSize` while choosing (`validating`: the fit
        rows, so a candidate's transform never sees the validation rows) and `Model.trSize` otherwise.
        """
        if trSize > 0:
            return trSize
        return Model.fitSize(m, te_ratio) if Model.validating else Model.trSize(m, te_ratio)

    # -- the switch ------------------------------------------------------------------------

    @staticmethod
    def _checkRatio(r: float) -> None:
        """A flaw unless 0 <= r and r + TE_RATIO < 1 (the fit rows must remain)."""
        if not (0.0 <= r and r + Model.TE_RATIO < 1.0):
            raise ValueError(f"Model: VAL_RATIO = {r} must satisfy 0 <= r and r + TE_RATIO ({Model.TE_RATIO}) < 1")

    @staticmethod
    @contextmanager
    def withValidation(ratio: Optional[float] = None, tuning: bool = False) -> Iterator[None]:
        """
        Runs the block with the switch ON (and `VAL_RATIO` = `ratio`, `tuning` as given), then
        restores all three (ScalaTion's `withValidation (ratio, tuning_)`):

            with Model.withValidation():
                mod.selectFeatures (SelectionTech.Backward)
            with Model.withValidation(tuning=True):                       # every candidate at its best knobs
                mod.selectFeatures (SelectionTech.Backward)
        """
        saved = (Model.validating, Model.VAL_RATIO, Model.tuning)
        if ratio is not None:
            Model._checkRatio(ratio)
            Model.VAL_RATIO = ratio
        Model.validating = True
        Model.tuning = bool(tuning)
        try:
            yield
        finally:
            Model.validating, Model.VAL_RATIO, Model.tuning = saved

    @staticmethod
    @contextmanager
    def inSample() -> Iterator[None]:
        """
        Runs the block with the switch OFF (fitting on the training rows, scoring in-sample) and
        restores it: what `Forecaster.refitTransform` does to rebuild a chosen model with its
        transforms fitted on the training rows after choosing under `withValidation`.
        """
        saved = (Model.validating, Model.tuning)
        Model.validating = False
        Model.tuning = False
        try:
            yield
        finally:
            Model.validating, Model.tuning = saved
