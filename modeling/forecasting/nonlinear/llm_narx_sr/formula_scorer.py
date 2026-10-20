"""
FormulaScorer: the Fit-and-Score Engine of LLM-NARX-SR

One engine takes a formula (a string or a `NARX_Formula`), fits it as a direct multi-horizon
NARX model (`NARX_SR_D.fromTerms`: coefficients by ridge, free nonlinear parameters by VarPro,
L-BFGS-B or a joint fit) and scores it with the selection score J used by every search arm:

    J = Forecaster.scoreOf (qof, inSample)      the ONE selection score of the platform

under the ONE switch (`Model.validating`, read at construction, or the `validating` argument):
ON => the candidate is fitted on the fit rows (`Model.fitRng`: 68 of the 116 Covid-19 rows) and
rolled over the validation rows (`Model.valRng`, the next 24, refit every `Forecaster.RC` rows,
`rollTest`), J = the mean validation sMAPE over the horizons, no penalty; OFF => fitted on the
training rows and J = the mean in-sample sMAPE times the FPE factor (1 + 2 p / n), p = terms + 1 +
fitted parameters.  In-sample, a candidate is fitted AND diagnosed on the FS rows of its block (the
rows whose hh targets all lie inside it, `fs_rng`: the boundary rule, exactly as `Forecaster_D.fitCols`).
`Model.tuning` (needs the switch) makes every candidate pick its own knobs (the ridge's lambda; lambda
and l1_ratio for the elastic net) by `Forecaster.tune` before it is scored; the knobs travel in
`Candidate.hp` and in the model name's tag (_L1.0, _L1.0_A0.5).  The testing rows `teRng` (the last
24) are read once, by `tnt` (`tnT_Test`).

Rejection rules (the reason travels with the `Candidate`, so a proposer can be told):
more than hp["maxFree"] free parameters; a fitted parameter within hp["boundTol"] of its
box; an exponent within 0.05 of 1 (a plain lag); a free-parameter column with a VIF above
hp["maxVIF"] on the fit rows (collinear with the others).

Any failure (non-finite column, singular fit, ...) scores inf: a search generates degenerate
formulas and they must rank last, never crash.  Quiet by construction: nothing here prints a
REPORT or plots; `LLM_NARX_SR.showBest` does that for the winners.

The Python counterpart of ScalaTion's `llm_narx_sr/FormulaScorer.scala` (rev5.3).  Differences:
the QoF "matrices" are lists of one dict per horizon keyed by ScalaTion's QoF names (`QOF_KEYS`
are the ones the tables show); the hyper-parameters are ONE dict (`FormulaScorer.hp`, ScalaTion's
`FormulaScorer.hp`) copied into every scorer (`scorer.hp`), so a harness sets them on
`FormulaScorer.hp` BEFORE building a scorer; every formula (linear or not) is built by
`NARX_SR_D.fromTerms` (a formula without free parameters is the plain direct ridge model on
its terms, which is what `ARX_D.fromTerms` gives in ScalaTion); the fit cache hands a formula
the fitted values of the cached model of its structure (ScalaTion returns the cached model
only); the validation pass rolls a fresh copy of a formula, so the candidate keeps the fit on the
fit rows.

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.11
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from Util.correlogram import acorr
from Util.tools import vector_str
from modeling.feature_selection import BestStep, Cols
from modeling.fit import MAXI, QoF, qoF_names, showFitMap, vif
from modeling.forecasting.datasets import DatasetConfig
from modeling.forecasting.eda4ts import topFreqs
from modeling.forecasting.forecaster import Forecaster
from modeling.forecasting.args import Args
from modeling.model import Model
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula as Formula, Par, ParRole
from modeling.forecasting.formula_parser import FormulaError
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D, OptimizerNL


QOF_KEYS = ("rSq", "rSqBar", "mse0", "mae", "smape", "m", "dfr")        # ScalaTion's QoF names (the tables)

#: The QoF dict (one horizon) as a QoF vector in `QoF` order (26 entries, -1 where absent).
def qof_vector(q: Dict[str, float]) -> np.ndarray:
    """The QoF dict of one horizon as ScalaTion's QoF vector (`QoF` order)."""
    return np.array([float(q.get(k, -1.0)) for k in qoF_names])


@dataclass
class Candidate:
    """
    One scored member of a search population: a formula (or a base model under a label) with its
    in-sample QoF on the fit rows and its QoF on the validation rows (ScalaTion's `Candidate`).

    Attributes
    ----------
    formula : NARX_Formula, optional
        The formula (its parameters hold the fitted values); None for a base model or a failed parse.
    score : float
        The selection score J (lower is better); inf for a failed or rejected fit.
    qofTrain : list of dict
        The QoF per horizon on the fit rows (ScalaTion's hh x N_QoF matrix; empty when failed).
    nTerms : int
        The number of terms / columns (the parsimony measure).
    qofVal : list of dict, optional
        The QoF per horizon on the validation rows (`rollTest`; None when scored in-sample).
    reason : str, optional
        Why the candidate failed or was rejected (None when it was not).
    label : str
        The display name of a base model (or of a failed parse).
    extraDF : int
        Parameters charged by J beyond the model's own (free parameters of a library set held fixed).
    model : Forecaster, optional
        The base model itself (None for a formula: rebuilt by `FormulaScorer.fit`).
    hp : dict, optional
        The knobs of the fit ({name: value}: the scorer's penalty, or what `tune` chose under
        `Model.tuning`, ScalaTion's `Candidate.hp`); None for a model without knobs or a failed candidate.
    nParams : np.ndarray, optional
        The effective parameters per horizon J charged in-sample (the fit's `effDF`: 1 + tr H_lambda
        for a ridge); None => dfr + 1 of each QoF row.
    """
    formula: Optional[Formula]
    score: float
    qofTrain: List[Dict[str, float]] = field(default_factory=list)
    nTerms: int = 0
    qofVal: Optional[List[Dict[str, float]]] = None
    reason: Optional[str] = None
    label: str = ""
    extraDF: int = 0
    model: Optional[Forecaster] = None
    hp: Optional[Dict[str, float]] = None
    nParams: Optional[np.ndarray] = None

    # -- ScalaTion's names ------------------------------------------------------------------

    @property
    def failed(self) -> bool:
        """Whether the fit failed (or was rejected)."""
        return not math.isfinite(self.score)

    @property
    def key(self) -> str:
        """The canonical identity (`FormulaScorer.keyOf`): parameters as ?, terms sorted."""
        return self.formula.key if self.formula is not None else self.label

    @property
    def formulaStr(self) -> str:
        """The display string (fitted values), or the label of a base model."""
        return self.formula.formulaStr if self.formula is not None else self.label

    @property
    def proposalStr(self) -> str:
        """The string a proposer is shown (?w slots)."""
        return self.formula.proposalStr if self.formula is not None else self.label

    @property
    def nFree(self) -> int:
        """The number of free nonlinear parameters."""
        return self.formula.nFree if self.formula is not None else 0

    @property
    def smapeH(self) -> np.ndarray:
        """The in-sample sMAPE per horizon (fit rows)."""
        return np.array([q["smape"] for q in self.qofTrain]) if self.qofTrain else np.array([])

    @property
    def smape(self) -> float:
        """The mean in-sample sMAPE over the horizons."""
        return float(self.smapeH.mean()) if self.qofTrain else math.inf

    @property
    def qofV(self) -> List[Dict[str, float]]:
        """The validation QoF (or the fit rows' when there is none)."""
        return self.qofVal if self.qofVal is not None else self.qofTrain

    @property
    def smapeValH(self) -> np.ndarray:
        """The validation sMAPE per horizon."""
        return np.array([q["smape"] for q in self.qofV]) if self.qofTrain else np.array([])

    @property
    def smapeVal(self) -> float:
        """The mean validation sMAPE over the horizons."""
        return float(self.smapeValH.mean()) if self.qofTrain else math.inf

    @property
    def scoreH(self) -> np.ndarray:
        """J per horizon: the validation sMAPE (validating), else the in-sample sMAPE times the FPE
        factor (1 + 2 p_h / n_h), p_h = the fit's effective parameters (`nParams`, else dfr_h + 1)
        + extraDF (`Forecaster.scoreOf` per horizon)."""
        if not self.qofTrain:
            return np.array([])
        if self.qofVal is not None:
            return np.array([q["smape"] for q in self.qofVal])
        p = [q["dfr"] + 1.0 for q in self.qofTrain] if self.nParams is None \
            else np.broadcast_to(np.asarray(self.nParams, dtype=float).ravel(), (len(self.qofTrain),))
        return np.array([q["smape"] * (1.0 + 2.0 * (p_h + self.extraDF) / q["m"]) for q, p_h in zip(self.qofTrain, p)])

    @property
    def why(self) -> str:
        """The rejection reason, for display ("" when none)."""
        return f" -- {self.reason}" if self.reason else ""

    def __repr__(self) -> str:
        s = f"Candidate({self.formulaStr}, score={self.score:.4f}" if not self.failed else f"Candidate({self.formulaStr}, FAILED"
        knobs = "" if not self.hp else ", " + ", ".join(f"{k}={v:g}" for k, v in self.hp.items())
        return s + f", nTerms={self.nTerms}, nFree={self.nFree}{knobs}{self.why})"


def qof_table(qof: Sequence[Dict[str, float]], title: str = "") -> str:
    """Formats a per-horizon QoF list as ScalaTion's `showFitMap` table (one "label -> VectorD(...)"
    line per measure, the horizons as columns)."""
    out = (title + "\n") if title else ""
    for k in QOF_KEYS:
        out += f"\t\t{k} \t -> {vector_str([q[k] for q in qof])} \n"
    return out


def qof_table_full(qof: Sequence[Dict[str, float]]) -> str:
    """The per-horizon QoF list as the full `Fit.showFitMap` table (all 26 measures)."""
    return showFitMap(np.column_stack([qof_vector(q) for q in qof]))


class FormulaScorer:
    """
    The fit-and-score engine over one named-column dataset (ScalaTion's `FormulaScorer`).

    Parameters
    ----------
    data : dict of str -> np.ndarray
        The named columns (the response under its name, usually "y"), all of one length.
    hh : int, optional
        The maximum horizon (defaults to hp["hh"]).
    hp : dict, optional
        Overrides of `FormulaScorer.hp` (the copy a scorer carries is `scorer.hp`); the map is a
        constructor argument, so two scorers cannot interfere.
    scaled : bool, default=True
        Whether a formula's regression standardizes its term columns, so the ridge penalty is unit-free.
    freqs : sequence of float, optional
        The frequencies `wave(k)` draws on (defaults to the top hp["nFreqs"] of the training
        response's periodogram, `EDA4TS.topFreqs`).
    validating : bool, optional
        The switch of this scorer (None => `Model.validating` as it stands at construction).  The
        LLM-NARX-SR harness builds its scorers with the switch ON: the search's score is defined on
        the validation rows; the in-sample J is for series with no rows to spare.  `Model.tuning`
        (read at construction too, `tuning`) makes every candidate pick its own knobs
        (`Forecaster.tune`), carried in `Candidate.hp` and the model name's tag.

    Attributes
    ----------
    vars : list of str
        The legal variable names.
    fullSet, trainSet : dict
        Every row; the training rows' columns.
    m, trSize, teSize, fitSize, valSize : int
        Rows in the series, training rows (`Model`'s split), testing rows, fit rows (training rows
        minus the validation rows; the training rows when not validating), validation rows.
    trRng, teRng, fitRng, valRng : range
        The training, testing, fit and validation rows.
    skip : int
        The rows every in-sample diagnosis skips, hp["maxLag"]: the same for all candidates.
    rc : int
        The retraining cycle of the rolling passes (`Forecaster.RC`).
    evaluations : int
        The number of model fits made so far (the search budget in fits).
    """

    #: The search hyper-parameters (ScalaTion's `FormulaScorer.hp`, aliased as `LLM_NARX_SR.hp`): the ONE
    #: map every arm and the harness read.  A scorer copies it at construction (`scorer.hp`, with the
    #: constructor's overrides), so set an entry here BEFORE building the scorer (e.g.,
    #: `FormulaScorer.hp["maxLag"] = 7` for a daily series).
    hp: Dict[str, Any] = {
        "hh": 6,              # maximum forecasting horizon (h = 1..hh)
        "maxLag": 6,          # maximum lag for proposals, libraries and the parser
        "lambda": 1.0,        # penalty carried by every model (the knob `Model.tuning` turns per candidate; l1_ratio: args' default)
        "perRound": 5,        # proposals requested per round
        "maxRounds": 20,      # maximum rounds per loop
        "patience": 5,        # stagnant rounds before a loop stops
        "topK": 5,            # candidates fed back each round
        "nFreqs": 2,          # periodogram frequencies offered to wave(k)
        "exoK": 2,            # exogenous variables kept by the screening (ExoMode.Auto)
        "maxCols": 12,        # largest column set the grammar CEM samples
        "minTerms": 3,        # smallest proposal a trimmer touches
        "nlTerms": 1,         # 1 => proposers and grammar libraries use the free-parameter forms
        "maxFree": 2,         # free nonlinear parameters allowed per candidate (-1 => no cap)
        "boundTol": 0.01,     # a fitted parameter within this fraction of its box of a bound is rejected
        "maxVIF": 10.0,       # a free-parameter column with a larger VIF is rejected (0 => off)
        "fitLib": 1,          # 1 => library terms with free parameters are fitted one by one before selection
        "refitTop": 5,        # CEM: the best sets of this many rounds are refitted with their free parameters
        "cemSample": 300,     # CEM: column sets per round on a library (-1 => CEM_Selector's default)
        "cemIter": 15,        # CEM: maximum rounds on a library
        "maxFits": 0,         # model fits a loop may make per run (0 => no budget)
        "optimizer": None,    # the optimizer of the nonlinear parameters (an `OptimizerNL` or its name); None => `OPTIMIZER_NL`
        "nneg": True,         # forecasts clipped at zero (count data); set False for series that go negative
    }
    CACHE_SZ = 64                                                      # fitted models kept (oldest evicted)

    def __init__(self, data: Dict[str, np.ndarray], hh: Optional[int] = None,
                 hp: Optional[Dict[str, Any]] = None, scaled: bool = True,
                 freqs: Optional[Sequence[float]] = None, validating: Optional[bool] = None) -> None:
        self.hp = dict(FormulaScorer.hp)
        if hp:
            unknown = sorted(set(hp) - set(FormulaScorer.hp))
            if unknown:                                                  # a stale setting (valFrac, valFolds, rc, skip, hSel) is an error
                raise ValueError(f"FormulaScorer: unknown hp {unknown}; the settings are {sorted(FormulaScorer.hp)} "
                                 "(the split and the switch are Model's, the retraining cycle Forecaster.RC, the skip hp['maxLag'])")
            self.hp.update(hp)
        self.data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.vars = list(self.data.keys())
        self.hh = int(hh if hh is not None else self.hp["hh"])
        self.scaled = scaled
        self.validating = bool(Model.validating if validating is None else validating)
        self.tuning = bool(Model.tuning) and self.validating               # every candidate at its own best knobs (`Forecaster.tune`)
        self.m = len(next(iter(self.data.values())))
        self.trSize = Model.trSize(self.m)                               # ScalaTion's `Model.trSize` (Model.TE_RATIO)
        self.teSize = self.m - self.trSize
        self.trRng = range(0, self.trSize)
        self.teRng = range(self.trSize, self.m)
        self.skip = int(self.hp["maxLag"])                               # one skip for every candidate
        self.rc = Forecaster.RC
        self.ridge = float(self.hp["lambda"])
        self.valSize = Model.valSize(self.m) if self.validating else 0
        self.fitSize = self.trSize - self.valSize
        self.fitRng = range(0, self.fitSize)
        self.valRng = range(self.fitSize, self.trSize)
        self.maxFree = int(self.hp["maxFree"])
        self.boundTol = float(self.hp["boundTol"])
        self.maxVIF = float(self.hp["maxVIF"])
        self.optimizer = OptimizerNL.of(self.hp["optimizer"])              # the optimizer of the free parameters
        self.evaluations = 0
        self._fitCache: "OrderedDict[str, NARX_SR_D]" = OrderedDict()   # key -> fitted model (fit rows)
        if self.trSize <= self.skip + 2:
            raise ValueError(f"FormulaScorer: trSize = {self.trSize} too small for skip = {self.skip}")
        if self.validating and self.fitSize <= self.skip + self.hh + 1:
            raise ValueError(f"FormulaScorer: fit rows = {self.fitSize} too few for skip = {self.skip} and hh = {self.hh}: lower Model.VAL_RATIO")
        if freqs is not None:
            self.freqs = list(freqs)
        else:
            resp = "y" if "y" in self.data else self.vars[0]
            self.freqs = list(topFreqs(self.data[resp][: self.trSize], int(self.hp["nFreqs"])))

    # -- ScalaTion's names of the rows ------------------------------------------------------

    @property
    def fullSet(self) -> Dict[str, np.ndarray]:
        """Every row (the dataset)."""
        return self.data

    @property
    def trainSet(self) -> Dict[str, np.ndarray]:
        """The training rows' columns."""
        return {k: v[: self.trSize] for k, v in self.data.items()}

    @property
    def splitStr(self) -> str:
        """The split as the banners print it: "fit / validation / test" (e.g., "68 / 24 / 24"; the
        validation block is 0 when not validating)."""
        return f"{self.fitSize} / {self.valSize} / {self.teSize}"

    def fs_rng(self, rows: range) -> range:
        """The FS rows of a block of leading rows: the rows whose hh targets all lie inside it,
        rows.start until rows.stop - hh + 1 -- the rows a direct model is fitted AND diagnosed on
        in-sample (the boundary rule, ScalaTion's `Forecaster_D.fs_rng` / `FormulaScorer.fs_rng`)."""
        return range(rows.start, rows.stop - self.hh + 1)

    def shortlist(self, cfg: DatasetConfig) -> List[str]:
        """
        The exogenous variables of a configuration the arms build on (ScalaTion's
        `FormulaScorer.shortlist (cfg)`): the list itself (`Given`) or the screening's shortlist of
        the pool (`Auto`) on this scorer's data and training rows, hp["exoK"] of them, leads up to
        hp["maxLag"] (`DatasetConfig.shortlist`).
        """
        return cfg.shortlist(self.fullSet, self.trSize, int(self.hp["exoK"]), int(self.hp["maxLag"]))

    def args(self, **changes: Any) -> Args:
        """The settings a model built here carries (ScalaTion's `hparam (nvs*)`): the defaults with this
        scorer's ridge penalty, scaling, optimizer and non-negativity, and the given settings written
        over them."""
        return Args(lam=self.ridge, scaled=self.scaled, optimizer=self.optimizer,
                    nneg=bool(self.hp["nneg"])).update(**changes)

    # -- parsing and fitting -------------------------------------------------------------

    def parse(self, text: str) -> Formula:
        """Parses a formula string against this dataset (variables, maxLag, frequencies)."""
        return NARX_FormulaParser.parseFormula(text, self.vars, int(self.hp["maxLag"]), self.freqs)

    def _formula(self, f: Union[str, Formula]) -> Formula:
        return self.parse(f) if isinstance(f, str) else Formula.of(f)

    def build(self, f: Union[str, Formula], size: Optional[int] = None, fit_params: bool = True,
              hp: Optional[Dict[str, float]] = None) -> NARX_SR_D:
        """
        Builds (untrained) the model of a formula on the full series: one column per term,
        transforms and free nonlinear parameters fitted on the first `size` rows (default: the
        fit rows; `trSize` for the final model), the parameters then held fixed
        (`NARX_SR_D.fromTerms`); at the knobs `hp` (None => the scorer's penalty; a candidate's own
        knobs under `Model.tuning`, ScalaTion's `build (f, size, hp2)`).
        """
        size = self.fitSize if size is None else size
        changes = {} if hp is None else {("lam" if k == "lambda" else k): float(v) for k, v in hp.items()}
        return NARX_SR_D.fromTerms(self._formula(f), self.data, self.hh, self.args(tune=fit_params, **changes),
                                   trSize=size, maxLag=int(self.hp["maxLag"]), freqs=self.freqs)

    def fresh(self, f: Union[str, Formula]) -> Formula:
        """A copy of a formula with its OWN free parameters (`NARX_Formula.fresh`): what the validation
        pass and the final model are built from, so the candidate keeps the values of the fit the
        score belongs to."""
        return self._formula(f).fresh()

    def tuned(self, mod: Forecaster) -> Forecaster:
        """The model at its own best knobs when the scorer tunes (`Model.tuning` under the switch):
        `Forecaster.tune` over the regression's knobs on the validation rows, then the copy at the
        knobs found (`withHP`); the model itself otherwise.  The tuner's evaluations (one rolling pass
        per grid point) count toward the budget."""
        if not self.tuning or not mod.knobs:
            return mod
        if Model.validating:
            hp2 = mod.tune()[0]
        else:                                                            # the tuner needs the switch on
            with Model.withValidation(Model.VAL_RATIO, Model.tuning):
                hp2 = mod.tune()[0]
        self.evaluations += int(np.prod([len(k.grid) for k in mod.knobs]))
        return mod.withHP(hp2)

    def buildFixed(self, f: Union[str, Formula]) -> NARX_SR_D:
        """
        Builds (untrained) the model of a formula with its free parameters held at their CURRENT
        values (start values, or the values of an earlier fit), without fitting them: the design
        matrix a selection stage scores column subsets of (`library`), where a fit per subset
        would be too costly.  Unset parameters are resolved from the fit rows first.
        """
        f = self._formula(f)
        f.prepare(self.data, self.fitSize)                              # resolves the unset parameters
        copies: Dict[int, Par] = {id(p): Par.const(p.value) for p in f.params()}
        frozen = Formula(f.resp, [t.fresh(copies) for t in f.preds], f.tFormT)
        return self.build(frozen, self.fitSize, fit_params=False)

    def train(self, mod: Forecaster, rows: Optional[range] = None) -> Forecaster:
        """
        Trains a model on a block of leading rows (the fit rows of the search, or all training rows
        for a final model) and sets the fixed skip; counts one fit (ScalaTion's `train`).  A direct
        model is fitted on the block's FS rows (`fs_rng`: no target beyond the block, as
        `Forecaster_D.fitCols` and `_trainWindow` fit); a model without an input matrix on the
        block's response values.
        """
        rows = self.trRng if rows is None else rows
        X = getattr(mod, "X", None)
        if X is not None and getattr(mod, "Y", None) is not None:
            fs = self.fs_rng(rows)                                       # the rows whose hh targets all lie in the block
            mod.train(mod.Y[fs.start:fs.stop], X[fs.start:fs.stop])
        elif X is not None:
            mod.train(mod.y[rows.start:rows.stop], X[rows.start:rows.stop])
        else:
            mod.train(mod.y[rows.start:rows.stop], None)
        mod.forecastAll()                                                # the forecast matrix for EVERY row (`qofOn` reads it)
        mod.setSkip(self.skip)
        self.evaluations += 1
        return mod

    def fit(self, f: Union[str, Formula]) -> NARX_SR_D:
        """
        Fits a formula on the fit rows and returns its model, reusing the cached model when this
        structure was fitted before (by its canonical key, which ignores the values of the free
        parameters); the formula's parameters then take the cached fitted values.  Under
        `Model.tuning` the structure is fitted at its own best knobs (`tuned`).
        """
        f = self._formula(f)
        key = self.keyOf(f)
        mod = self._fitCache.get(key)
        if mod is None:
            mod = self.train(self.tuned(self.build(f)), self.fitRng)
            if len(self._fitCache) >= self.CACHE_SZ:
                self._fitCache.popitem(last=False)                       # evict the oldest
            self._fitCache[key] = mod
        elif mod.formula is not None and mod.formula is not f:
            for p, q in zip(f.params(), mod.formula.params()):           # the same structure: adopt the fit
                if p is not q:
                    p.w0, p.lb, p.ub = q.w0, q.lb, q.ub                   # the resolved box of the cached fit
                    p.set(q.value)
        return mod

    def fitFull(self, f: Union[str, Formula], hp: Optional[Dict[str, float]] = None) -> NARX_SR_D:
        """The final model: a fresh copy of the formula, parameters and coefficients fitted on ALL the
        training rows (not cached), at the knobs `hp` (a candidate's own under `Model.tuning`, else the
        scorer's), the model `showBest` reports; `tnt` builds its own and lets `tnT_Test` train it."""
        return self.train(self.build(self.fresh(f), self.trSize, hp=hp), self.trRng)

    # -- scoring -----------------------------------------------------------------------------

    def score(self, f: Union[str, Formula, Forecaster], label: Optional[str] = None) -> Candidate:
        """
        Fits and scores one formula (a string is parsed first): its QoF on the fit rows, its QoF on
        the validation rows when validating (a fresh copy rolled by `rollTest`), and the selection
        score J (`scoreOf` of the validation QoF, or of the in-sample QoF); a failed fit ranks last, as
        does a rejected one (`rejection`), with its reason.  A base model is scored under
        `label` (`scoreModel`).
        """
        if isinstance(f, Forecaster):
            return self.scoreModel(f, label if label is not None else f.modelName)
        if isinstance(f, str):
            try:
                f = self.parse(f)
            except FormulaError as e:
                return Candidate(None, math.inf, reason=f"parse: {e}", label=f.strip())
        f = Formula.of(f)
        nf = f.nFree
        if 0 <= self.maxFree < nf:
            return self.failed(f, f.nTerms, reason=f"{nf} free parameters, at most {self.maxFree} allowed")
        try:
            mod = self.fit(f)
            why = self.rejection(f, mod)
            if why is not None:
                return self.failed(f, f.nTerms, reason=why)
            q = self.qofIn(mod, self.fitRng)                               # in-sample: the FS rows of the fit rows
            qv = self.qofValidation(self.build(self.fresh(f), self.fitSize, hp=self._knobsOf(mod))) if self.validating else None   # a copy at the fit's knobs
            return Candidate(f, self.scoreOf(q if qv is None else qv, nParams=mod.effDF), q, f.nTerms, qofVal=qv, hp=self._knobsOf(mod),
                             nParams=mod.effDF)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError, ZeroDivisionError) as e:
            return self.failed(f, f.nTerms, reason=f"fit failed: {e}")

    def scoreModel(self, mod: Forecaster, label: str) -> Candidate:
        """
        Fits and scores a base model (e.g., a `RandomWalk`) the same way a formula is scored,
        under the given label (ScalaTion's `score (mod, label)`): trained on the fit rows, the
        same object rolled over the validation rows when validating, then left as trained on the
        fit rows.
        """
        try:
            self.train(mod, self.fitRng)
            q = self.qofIn(mod, self.fitRng)                               # in-sample: the FS rows of the fit rows
            qv = None
            if self.validating:
                qv = self.qofValidation(mod)                                # rolls the model itself ...
                self.train(mod, self.fitRng)                                # ... then leaves it trained on the fit rows
            X = getattr(mod, "X", None)
            k = X.shape[1] if X is not None else 0
            return Candidate(None, self.scoreOf(q if qv is None else qv, nParams=getattr(mod, "effDF", None)), q, k,
                             qofVal=qv, label=label, model=mod, nParams=getattr(mod, "effDF", None),
                             hp=self._knobsOf(mod))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError, ZeroDivisionError) as e:
            return self.failed(None, 0, label, reason=f"fit failed: {e}")

    def failed(self, f: Optional[Formula], nTerms: int, label: Optional[str] = None,
               reason: Optional[str] = None) -> Candidate:
        """A failed candidate (score inf, no QoF) with its reason."""
        return Candidate(f, math.inf, [], nTerms, label=label or "", reason=reason)

    def rejection(self, f: Formula, mod: NARX_SR_D) -> Optional[str]:
        """
        Why a fitted formula is rejected, or None: a free parameter fitted within `boundTol` of
        its box (the nonlinear term has collapsed onto a linear or constant limit), an exponent
        within 0.05 of 1 (a plain lag), or a free-parameter term whose fitted column has a
        variance inflation factor above `maxVIF` against the other columns on the fit rows.
        """
        pars = [p for p in f.params() if p.free]
        for p in pars:
            rng = p.ub - p.lb
            if math.isfinite(rng) and rng > 0.0:
                tol = self.boundTol * rng
                if p.value - p.lb <= tol:
                    return f"{p.role.value} {p.value:.4g} at its lower bound {p.lb:.4g}"
                if p.ub - p.value <= tol:
                    return f"{p.role.value} {p.value:.4g} at its upper bound {p.ub:.4g}"
            if p.role is ParRole.POWER and abs(p.value - 1.0) < 0.05:
                return f"exponent {p.value:.4g} is a plain lag"
        if self.maxVIF > 0.0 and pars and f.nTerms > 1:
            X = mod.X[: self.fitSize]
            Xc = X - X.mean(axis=0)
            for j, t in enumerate(f.preds):
                if any(p.free for p in t.params()):
                    v = self.vif(Xc, j)
                    if v > self.maxVIF:
                        return f"term {t.label()} is collinear with the others (VIF {v:.1f})"
        return None

    @staticmethod
    def vif(Xc: np.ndarray, j: int) -> float:
        """The variance inflation factor of column j of a centered matrix: 1 / (1 - R^2) of the
        regression of x_j on the other columns (no intercept); infinite when x_j is spanned by them."""
        if Xc.shape[1] < 2:
            return 1.0
        xj = Xc[:, j]
        sst = float(xj @ xj)
        if sst <= 0.0:
            return math.inf
        Xo = np.delete(Xc, j, axis=1)
        b = np.linalg.lstsq(Xo, xj, rcond=None)[0]
        sse = float(np.sum((xj - Xo @ b) ** 2))
        r2 = 1.0 - sse / sst
        return math.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)

    def scoreH(self, f: Union[str, Formula]) -> np.ndarray:
        """The sMAPE per horizon of a formula on the rows its score is defined on: the validation rows
        (a fresh copy rolled, `qofValidation`) when validating, the fit rows otherwise (exceptions
        propagate)."""
        mod = self.fit(f)
        q = self.qofValidation(self.build(self.fresh(f), self.fitSize, hp=self._knobsOf(mod))) if self.validating else self.qofIn(mod, self.fitRng)
        return np.array([qh["smape"] for qh in q])

    @staticmethod
    def _knobsOf(mod: Forecaster) -> Optional[Dict[str, float]]:
        """The knobs of a model as {name: value} (None for a model without knobs): what `Candidate.hp`
        carries and `build (.., hp)` takes."""
        knobs = getattr(mod, "knobs", ())
        if not knobs:
            return None
        h = mod.hparameter
        return {k.name: float(h[k.name]) for k in knobs}

    def notes(self, cand: Candidate, k: Optional[int] = None) -> str:
        """
        The diagnostic notes of a fitted candidate for a proposer: the autocorrelation of the
        h = 1 residuals on the fit rows at lags 1..k (what the equation leaves unexplained) and
        the terms ranked by their standardized h = 1 effect (`Explainer.importance`); "" for a
        failed or base candidate.
        """
        if cand.failed or cand.formula is None:
            return ""
        k = int(self.hp["maxLag"]) if k is None else k
        try:
            # deferred: `explainer` imports this module (through `llm_proposer`), a genuine cycle
            from modeling.forecasting.nonlinear.llm_narx_sr.explainer import Explainer
            mod = self.fit(cand.formula)
            e = self.residuals(mod, self.fitRng)                          # h = 1 residuals on the fit rows
            acf = "[" + ", ".join(f"{acorr(e, l):.2f}" for l in range(1, k + 1)) + "]"
            imp = Explainer.importance(mod, cand.formula, self.fitSize)
            tot = sum(abs(i.effect) for i in imp)
            top = ", ".join(f"{i.label} {100 * abs(i.effect) / tot if tot > 0 else 0.0:.0f}%"
                            for i in sorted(imp, key=lambda i: -abs(i.effect))[:5])
            return f"residual autocorrelation of the best at lags 1..{k}: {acf}; its terms by importance: {top}"
        except Exception:
            return ""

    # -- the final evaluation -----------------------------------------------------------------

    def tnt(self, f: Union[str, Formula, Forecaster], hp: Optional[Dict[str, float]] = None):
        """
        Train-and-Test: for a formula, a fresh copy built on ALL training rows (`build (f, trSize)`,
        at the knobs `hp`: a candidate's own under `Model.tuning`, else the scorer's), then the
        base's `tnT_Test` on the testing rows (`rollTest`: trained on the training rows, sliding
        window refit every `rc` rows, no skip); returns (model, QoF per horizon on the testing
        rows).  For a base model, its QoF per horizon on the testing rows alone (`tntModel`).
        """
        if isinstance(f, Forecaster):
            return self.tntModel(f)
        mod = self.build(self.fresh(f), self.trSize, hp=hp)
        return mod, self.tntModel(mod)

    def tntModel(self, mod: Forecaster) -> List[Dict[str, float]]:
        """The `tnT_Test` of a model (quiet) as a QoF list per horizon on the testing rows; one fit per
        retraining is counted."""
        qof = mod.tnT_Test(self.rc, verbose=False)
        self.evaluations += 1
        return self._qofList(qof, mod)

    def qofValidation(self, mod: Forecaster) -> List[Dict[str, float]]:
        """The `rollTest` of a model over the validation rows (quiet: trained on the fit rows, refit
        every `rc` rows, original units, no skip) as a QoF list per horizon; one fit is counted.  The
        fit rows' QoF when the scorer is not validating."""
        if not self.validating:
            return self.qofIn(mod, self.fitRng)
        qof = mod.rollTest(self.valRng, self.rc, "Validation", verbose=False)
        self.evaluations += 1
        return self._qofList(qof, mod)

    def _qofList(self, qof: np.ndarray, mod: Forecaster) -> List[Dict[str, float]]:
        """A QoF matrix (one row per horizon, `rollTest`) as the per-horizon QoF list the tables and
        `scoreOf` read, the DF per horizon the model's (`_dof`)."""
        out = []
        for h in range(self.hh):
            q = {k: float(v) for k, v in zip(qoF_names, np.asarray(qof[h], dtype=float))}
            q["dfr"] = float(self._dof(mod))
            out.append(q)
        return out

    # -- QoF (rev2.3: the scorer's own alignment -- `diagnoseH`, `qofByH`, `qofTrain`, `qofFit` -- is gone; the
    #    base's `qofOn` on the FS rows of a block is the in-sample table, as in ScalaTion) ----------------------

    @staticmethod
    def _dof(mod: Forecaster) -> int:
        """The degrees of freedom a model takes per horizon (the slopes plus any fitted nonlinear parameters)."""
        if hasattr(mod, "dof"):
            return int(mod.dof)
        try:
            return int(mod.nparams)
        except (ValueError, AttributeError):
            return 0

    def qofIn(self, mod: Forecaster, rows: range) -> List[Dict[str, float]]:
        """The in-sample QoF per horizon of a model trained on a block of leading rows (the fit rows, or
        the training rows): the base's `qofOn` on the block's FS rows (`fs_rng`; the first `skip` rows
        dropped), as a QoF list."""
        return self._qofList(mod.qofOn(self.fs_rng(rows)), mod)

    def residuals(self, mod: Forecaster, rows: range, noSkip: bool = False) -> np.ndarray:
        """The h = 1 residuals y - yf(., 1) of a trained model over a block of rows (original units), the
        first `skip` rows dropped unless `noSkip`."""
        start = rows.start if noSkip else rows.start + self.skip
        return mod.getY()[start:rows.stop] - mod.Yf[start:rows.stop, 1]

    @staticmethod
    def keyOf(f: Formula) -> str:
        """The canonical key of a formula: every free parameter a bare ?, so a formula and its
        fitted version are one candidate."""
        return Formula.of(f).key

    def scoreOf(self, qof: Sequence[Dict[str, float]], extra: int = 0, nParams: Optional[Any] = None) -> float:
        """
        The selection score J of a QoF list (`Forecaster.scoreOf`): the mean over the horizons of
        the sMAPE, times the FPE factor (1 + 2 (p_h + extra) / n_h) when the rows were fitted (not
        validating) -- p_h the model's effective parameters (`nParams`, its `effDF`) or dfr_h + 1 --,
        plain on the validation rows; inf when undefined.
        """
        return Forecaster.scoreOf(np.array([qof_vector(q) for q in qof]), inSample=not self.validating, extra=extra,
                                  nParams=nParams)

    # -- libraries (the cross-entropy stage, see `TermLibrary`) ------------------------------

    def library(self, resp, terms: Sequence, tFormT=None) -> NARX_SR_D:
        """
        The library model of a term list (untrained): one column per term, the design matrix the
        cross-entropy stage selects from.  A term with free parameters enters at their current
        values (the role start values resolved on the fit rows, or the values of the fit of the
        candidate it came from); the selected equation is then scored by `score`, which fits them.
        """
        return self.buildFixed(Formula(resp, list(terms), tFormT))

    def evalCols(self, lib: NARX_SR_D, cols: Cols, nFree: int = 0, qk: int = QoF.smapeC.ordinal) -> BestStep:
        """
        Evaluates a column subset of a library model (`library`): the sub-model is trained on the
        fit rows and scored as `score` does, returning a `BestStep` whose QoF vector is the h = 1
        QoF vector with the selection score at index qk and the mean sMAPE over horizons at
        `QoF.smape` -- the evaluator `SubsetSearch` calls (`TermLibrary.selectBy`).  A failed fit
        returns a None QoF (ranked last by the selector).  J is minimized, so qk must be a
        minimized slot.  Over the free-parameter cap a set stays evaluable but its score is
        steepened, so backward / CEM can move off it.
        """
        if qk in MAXI:
            raise ValueError(f"evalCols: the score J is minimized, qk = {qk} is a maximized slot")
        cols = list(cols)
        if not cols:
            return BestStep(-1, None, None, cols, qk)                     # the mean model is not a candidate
        try:
            sub = self.train(self.tuned(lib.getModel(cols)), self.fitRng)   # the set at its own best knobs under tuning
            qv = None
            if self.validating:                                          # a fresh sub-model at those knobs is rolled
                fresh = lib.getModel(cols)
                qv = self.qofValidation(fresh.withHP(self._knobsOf(sub)) if self.tuning else fresh)
            q = self.qofIn(sub, self.fitRng)
            s0 = self.scoreOf(q if qv is None else qv, nFree, sub.effDF)
            s = s0 * (1.0 + (nFree - self.maxFree)) if (0 <= self.maxFree < nFree and math.isfinite(s0)) else s0
            if not math.isfinite(s):
                return BestStep(-1, None, None, cols, qk)
            qof = qof_vector(q[0])
            qof[QoF.smape.ordinal] = float(np.mean([qh["smape"] for qh in q]))
            qof[qk] = s
            return BestStep(-1, qof, None, cols, qk, s)
        except Exception:
            return BestStep(-1, None, None, cols, qk)

    def coefImportance(self, lib: NARX_SR_D, cols: Cols) -> np.ndarray:
        """
        The STANDARDIZED COEFFICIENTS of the library sub-model on `cols`, |b_jh| sd(x_j) averaged
        over the horizons, as an importance per column of the library (NaN for the columns not in
        the set): the sub-model is trained on the fit rows as `evalCols` trains it, and its slopes
        are in original units, so the product is unit-free.  The default ranker of the Ranked
        technique in `TermLibrary.selectBy` (recursive feature elimination over the library); a
        failed fit ranks every column of the set alike (zeros).
        """
        cols = list(cols)
        imp = np.full(lib.X.shape[1], np.nan)
        try:
            sub = self.train(lib.getModel(cols), self.fitRng)
            b = np.abs(np.asarray(sub.slopes, dtype=float)).mean(axis=1)  # (n_sub,): mean |b_jh| over the horizons
            sd = np.std(sub.X[self.fitRng.start:self.fitRng.stop], axis=0)
            imp[cols] = b * sd
        except Exception:
            imp[cols] = 0.0
        return imp

    def tImportance(self, lib: NARX_SR_D, cols: Cols) -> np.ndarray:
        """
        The t-STATISTICS of the library sub-model on `cols`, |b_jh| / se_jh averaged over the horizons
        (the ridge's own standard errors, `stdErrOf`), as an importance per column of the library
        (NaN outside the set): the ranker of elimination by p-value in `TermLibrary.selectBy
        (Ranked, rank = "t")`; a failed fit ranks every column alike.
        """
        cols = list(cols)
        imp = np.full(lib.X.shape[1], np.nan)
        try:
            sub = self.train(lib.getModel(cols), self.fitRng)
            fit = self.fitRng
            Y_, X_ = sub.fit_rows(sub.Y[fit.start:fit.stop], sub.X[fit.start:fit.stop])
            resid = Y_ - sub._predict(X_)
            t = np.zeros(len(cols))
            for h in range(Y_.shape[1]):
                with np.errstate(invalid="ignore", divide="ignore"):
                    t += np.abs(np.asarray(sub.slopes, dtype=float)[:, h] / sub.stdErrOf(X_, resid[:, h], h)[1:])
            imp[cols] = t / Y_.shape[1]
        except Exception:
            imp[cols] = 0.0
        return imp

    def vifImportance(self, lib: NARX_SR_D, cols: Cols) -> np.ndarray:
        """MINUS the variance inflation factors of the library columns `cols` on the fit rows (given an
        intercept), as an importance per column (NaN outside the set): the most collinear column
        leaves first in `TermLibrary.selectBy (Ranked, rank = "vif")`."""
        cols = list(cols)
        imp = np.full(lib.X.shape[1], np.nan)
        fit = self.fitRng
        x1 = np.column_stack((np.ones(fit.stop - fit.start), lib.X[fit.start:fit.stop][:, cols]))
        imp[cols] = -vif(x1)
        return imp

    # -- reports ------------------------------------------------------------------------------

    def report(self, cand: Candidate, tnt: bool = True) -> str:
        """A text report of a candidate: formula, score, in-sample (and, when validating, validation)
        sMAPE, J per horizon, and (optionally) the rolling TnT sMAPE on the testing rows."""
        lam = "" if not cand.hp or "lambda" not in cand.hp else f", lambda = {cand.hp['lambda']:.1f}"
        lines = [f"formula:    {cand.formulaStr}", f"key:        {cand.key}",
                 f"score J:    {cand.score:.4f}   (nTerms = {cand.nTerms}, nFree = {cand.nFree}{lam}){cand.why}"]
        if not cand.failed:
            lines.append("fit sMAPE:  " + " ".join(f"{v:7.3f}" for v in cand.smapeH))
            if cand.qofVal is not None:
                lines.append("val sMAPE:  " + " ".join(f"{q['smape']:7.3f}" for q in cand.qofVal))
            lines.append("J per h:    " + " ".join(f"{v:7.3f}" for v in cand.scoreH))
            if tnt:
                qt = self.tnt(cand.formula, cand.hp)[1] if cand.formula is not None else self.tnt(cand.model)
                lines.append("TnT sMAPE:  " + " ".join(f"{q['smape']:7.3f}" for q in qt) +
                             f"   mean {np.mean([q['smape'] for q in qt]):.3f}")
        return "\n".join(lines)

