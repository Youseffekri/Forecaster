"""
FormulaScorer: the Fit-and-Score Engine of LLM-NARX-SR

One engine takes a formula (a string or a `NARX_Formula`), fits it as a direct multi-horizon
NARX model (`NARX_SR_D.fromTerms`: coefficients by ridge, free nonlinear parameters by VarPro,
L-BFGS-B or a joint fit) and scores it with the selection score J used by every search arm:

    J_h = sMAPE_h (validation rows) * (1 + 2 p / n),     p = terms + 1 + fitted parameters

the FPE-form parsimony adjustment of the out-of-sample sMAPE, averaged over the horizons
(or taken at h = 1 when hp["hSel"] = 1).  The rows are ScalaTion's: training rows
`trRng` = 0 until `trSize` (92 of the 116 Covid-19 rows), testing rows `teRng` (the last 24).
The validation rows are the last hp["valFrac"] of the TRAINING rows, split into hp["valFolds"]
rolling-origin blocks: fold i refits the candidate on the rows before block i and scores it on
the block, so J never sees the testing rows (the final rolling validation on the testing rows is
`tnt`).  With hp["valFrac"] = 0, J is the in-sample FPE-sMAPE on the training rows.

Rejection rules (the reason travels with the `Candidate`, so a proposer can be told):
more than hp["maxFree"] free parameters; a fitted parameter within hp["boundTol"] of its
box; an exponent within 0.05 of 1 (a plain lag); a free-parameter column with a VIF above
hp["maxVIF"] on the fit rows (collinear with the others).

Any failure (non-finite column, singular fit, ...) scores inf: a search generates degenerate
formulas and they must rank last, never crash.  Quiet by construction: nothing here prints a
REPORT or plots; `LLM_NARX_SR.showBest` does that for the winners.

The Python counterpart of ScalaTion's `llm_narx_sr/FormulaScorer.scala` (rev4.0).  Differences:
the QoF "matrices" are lists of one dict per horizon keyed by ScalaTion's QoF names (`QOF_KEYS`
are the ones the tables show); the hyper-parameters are a dict (`DEFAULT_HP`, ScalaTion's
`FormulaScorer.hp`) copied into every scorer (`scorer.hp`), so a harness sets them on
`DEFAULT_HP` BEFORE building a scorer; every formula (linear or not) is built by
`NARX_SR_D.fromTerms` (a formula without free parameters is the plain direct ridge model on
its terms, which is what `ARX_D.fromTerms` gives in ScalaTion); the fit cache hands a formula
the fitted values of the cached model of its structure (ScalaTion returns the cached model
only); the validation folds refit fresh copies of a formula, so the candidate keeps the fit on
the fit rows (ScalaTion refits the same parameter objects per block).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from Util.tools import vector_str
from modeling.feature_selection import BestStep, Cols
from modeling.fit import Fit, MAXI, QoF, fitMapDict, qoF_names
from modeling.forecasting.forecaster import Forecaster
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula as Formula, Par, ParRole
from modeling.forecasting.formula_parser import FormulaError
from modeling.forecasting.nonlinear.narx_formula_parser import parse_formula
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D


#: The search hyper-parameters (ScalaTion's `FormulaScorer.hp`, aliased as `LLM_NARX_SR.hp`): the
#: one map every arm and the harness read.  A scorer copies it at construction, so set an entry
#: here before building the scorer (e.g., `DEFAULT_HP["maxLag"] = 7` for a daily series).
DEFAULT_HP: Dict[str, Any] = {
    "hh": 6,              # maximum forecasting horizon (h = 1..hh)
    "maxLag": 6,          # maximum lag for proposals, libraries and the parser
    "skip": 6,            # rows skipped by every in-sample diagnosis (the same for all candidates)
    "rc": 5,              # retraining cycle of rollValidate on the testing rows
    "hSel": 0,            # 0 => score = mean FPE-sMAPE over the horizons, 1 => h = 1 only
    "lambda": 1.0,        # ridge penalty carried by every model
    "perRound": 5,        # proposals requested per round
    "maxRounds": 20,      # maximum rounds per loop
    "patience": 5,        # stagnant rounds before a loop stops
    "topK": 5,            # candidates fed back each round
    "nFreqs": 2,          # periodogram frequencies offered to wave(k)
    "exoK": 2,            # exogenous variables kept by the screening (ExoMode.Auto)
    "maxCols": 12,        # largest column set the grammar CEM samples
    "minTerms": 3,        # smallest proposal a trimmer touches
    "nlTerms": 1,         # 1 => proposers and grammar libraries use the free-parameter forms
    "valFrac": 0.25,      # share of the training rows held out for the selection score (0 => in-sample J)
    "valFolds": 3,        # rolling-origin blocks of the validation rows (k fits per candidate; 1 => one block)
    "maxFree": 2,         # free nonlinear parameters allowed per candidate (-1 => no cap)
    "boundTol": 0.01,     # a fitted parameter within this fraction of its box of a bound is rejected
    "maxVIF": 10.0,       # a free-parameter column with a larger VIF is rejected (0 => off)
    "fitLib": 1,          # 1 => library terms with free parameters are fitted one by one before selection
    "refitTop": 5,        # CEM: the best sets of this many rounds are refitted with their free parameters
    "cemSample": 300,     # CEM: column sets per round on a library (-1 => CEM_Selector's default)
    "cemIter": 15,        # CEM: maximum rounds on a library
    "maxFits": 0,         # model fits a loop may make per run (0 => no budget)
    "optimizer": "varpro",  # the optimizer of the nonlinear parameters: varpro, lbfgsb, joint
    "nneg": True,         # forecasts clipped at zero (count data); set False for series that go negative
}

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
    qof_train : list of dict
        The QoF per horizon on the fit rows (ScalaTion's `qofTrain`, hh x N_QoF; empty when failed).
    n_terms : int
        The number of terms / columns (the parsimony measure).
    qof_val : list of dict, optional
        The QoF per horizon on the validation rows, averaged over the blocks (None => the fit rows').
    reason : str, optional
        Why the candidate failed or was rejected (None when it was not).
    label : str
        The display name of a base model (or of a failed parse).
    extra_df : int
        Parameters charged by J beyond the model's own (free parameters of a library set held fixed).
    model : Forecaster, optional
        The base model itself (None for a formula: rebuilt by `FormulaScorer.fit`).
    """
    formula: Optional[Formula]
    score: float
    qof_train: List[Dict[str, float]] = field(default_factory=list)
    n_terms: int = 0
    qof_val: Optional[List[Dict[str, float]]] = None
    reason: Optional[str] = None
    label: str = ""
    extra_df: int = 0
    model: Optional[Forecaster] = None

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
    def nTerms(self) -> int:
        return self.n_terms

    @property
    def nFree(self) -> int:
        """The number of free nonlinear parameters."""
        return self.formula.nFree if self.formula is not None else 0

    @property
    def qofTrain(self) -> List[Dict[str, float]]:
        return self.qof_train

    @property
    def qofVal(self) -> Optional[List[Dict[str, float]]]:
        return self.qof_val

    @property
    def extraDF(self) -> int:
        return self.extra_df

    @property
    def smapeH(self) -> np.ndarray:
        """The in-sample sMAPE per horizon (fit rows)."""
        return np.array([q["smape"] for q in self.qof_train]) if self.qof_train else np.array([])

    @property
    def smape(self) -> float:
        """The mean in-sample sMAPE over the horizons."""
        return float(self.smapeH.mean()) if self.qof_train else math.inf

    @property
    def qofV(self) -> List[Dict[str, float]]:
        """The validation QoF (or the fit rows' when there is none)."""
        return self.qof_val if self.qof_val is not None else self.qof_train

    @property
    def smapeValH(self) -> np.ndarray:
        """The validation sMAPE per horizon."""
        return np.array([q["smape"] for q in self.qofV]) if self.qof_train else np.array([])

    @property
    def smapeVal(self) -> float:
        """The mean validation sMAPE over the horizons."""
        return float(self.smapeValH.mean()) if self.qof_train else math.inf

    @property
    def scoreH(self) -> np.ndarray:
        """J per horizon (`FormulaScorer.jOf`)."""
        return FormulaScorer.jOf(self.qof_train, self.qofV, self.extra_df) if self.qof_train else np.array([])

    @property
    def why(self) -> str:
        """The rejection reason, for display ("" when none)."""
        return f" -- {self.reason}" if self.reason else ""

    # -- the former snake_case names ---------------------------------------------------------

    @property
    def formula_str(self) -> str:
        return self.formulaStr

    @property
    def proposal_str(self) -> str:
        return self.proposalStr

    @property
    def n_free(self) -> int:
        return self.nFree

    @property
    def smape_h(self) -> np.ndarray:
        return self.smapeH

    @property
    def qof_v(self) -> List[Dict[str, float]]:
        return self.qofV

    @property
    def smape_val(self) -> float:
        return self.smapeVal

    @property
    def score_h(self) -> np.ndarray:
        return self.scoreH

    def __repr__(self) -> str:
        s = f"Candidate({self.formulaStr}, score={self.score:.4f}" if not self.failed else f"Candidate({self.formulaStr}, FAILED"
        return s + f", n_terms={self.n_terms}, n_free={self.nFree}{self.why})"


def qof_table(qof: Sequence[Dict[str, float]], title: str = "") -> str:
    """Formats a per-horizon QoF list as ScalaTion's `showFitMap` table (one "label -> VectorD(...)"
    line per measure, the horizons as columns)."""
    out = (title + "\n") if title else ""
    for k in QOF_KEYS:
        out += f"\t\t{k} \t -> {vector_str([q[k] for q in qof])} \n"
    return out


def qof_table_full(qof: Sequence[Dict[str, float]]) -> str:
    """The per-horizon QoF list as the full `Fit.showFitMap` table (all 26 measures)."""
    from modeling.fit import showFitMap
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
        Overrides of `DEFAULT_HP` (the copy a scorer carries is `scorer.hp`).
    scaled : bool, default=True
        Whether a formula's regression standardizes its term columns, so the ridge penalty is unit-free.
    freqs : sequence of float, optional
        The frequencies `wave(k)` draws on (defaults to the top hp["nFreqs"] of the training
        response's periodogram, `EDA4TS.topFreqs`).

    Attributes
    ----------
    vars : list of str
        The legal variable names.
    fullSet, trainSet : dict
        Every row; the training rows' columns.
    m, trSize, teSize, fitSize, nVal : int
        Rows in the series, training rows (ScalaTion's split), testing rows, fit rows (training
        rows minus the validation rows), validation rows.
    trRng, teRng, fitRng, valRng : range
        The training, testing, fit and validation rows.
    blocks : list of range
        The rolling-origin validation blocks (empty when hp["valFrac"] = 0).
    evaluations : int
        The number of model fits made so far (the search budget in fits).
    """

    #: ScalaTion's `FormulaScorer.hp`: the module-level map a harness edits before building a scorer.
    hp_defaults = DEFAULT_HP
    CACHE_SZ = 64                                                      # fitted models kept (oldest evicted)

    def __init__(self, data: Dict[str, np.ndarray], hh: Optional[int] = None,
                 hp: Optional[Dict[str, Any]] = None, scaled: bool = True,
                 freqs: Optional[Sequence[float]] = None) -> None:
        self.hp = dict(DEFAULT_HP)
        if hp:
            self.hp.update(hp)
        self.data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.vars = list(self.data.keys())
        self.hh = int(hh if hh is not None else self.hp["hh"])
        self.scaled = scaled
        self.m = len(next(iter(self.data.values())))
        self.tr_size = _tr_size(self.m)                                   # ScalaTion's `Model.trSize`
        self.te_size = self.m - self.tr_size
        self.tr_rows = range(0, self.tr_size)
        self.te_rows = range(self.tr_size, self.m)
        self.skip = int(self.hp["skip"])
        self.h_sel = int(self.hp["hSel"])
        self.rc = int(self.hp["rc"])
        self.ridge = float(self.hp["lambda"])
        self.val_frac = float(self.hp["valFrac"])
        self.val_folds = max(1, int(self.hp["valFolds"]))
        self.n_val = 0 if self.val_frac <= 0.0 else max(self.val_folds * (self.hh + 1), int(round(self.val_frac * self.tr_size)))
        self.fit_size = self.tr_size - self.n_val
        self.fit_rows = range(0, self.fit_size)
        self.val_rows = range(self.fit_size, self.tr_size)
        if self.n_val == 0:
            self.blocks: List[range] = []
        else:
            b = self.n_val // self.val_folds
            self.blocks = [range(self.fit_size + i * b, self.tr_size if i == self.val_folds - 1 else self.fit_size + (i + 1) * b)
                           for i in range(self.val_folds)]
        self.max_free = int(self.hp["maxFree"])
        self.bound_tol = float(self.hp["boundTol"])
        self.max_vif = float(self.hp["maxVIF"])
        self.optimizer = str(self.hp["optimizer"])
        self.evaluations = 0
        self._fitCache: "OrderedDict[str, NARX_SR_D]" = OrderedDict()   # key -> fitted model (fit rows)
        if self.tr_size <= self.skip + 2:
            raise ValueError(f"FormulaScorer: trSize = {self.tr_size} too small for skip = {self.skip}")
        if self.n_val > 0 and self.fit_size <= self.skip + self.hh + 1:
            raise ValueError(f"FormulaScorer: fit rows = {self.fit_size} too few for skip = {self.skip} and hh = {self.hh}: lower valFrac")
        if freqs is not None:
            self.freqs = list(freqs)
        else:
            from modeling.forecasting.nonlinear.llm_narx_sr.eda4ts import topFreqs
            resp = "y" if "y" in self.data else self.vars[0]
            self.freqs = list(topFreqs(self.data[resp][: self.tr_size], int(self.hp["nFreqs"])))

    # -- ScalaTion's names of the rows ------------------------------------------------------

    @property
    def fullSet(self) -> Dict[str, np.ndarray]:
        """Every row (the dataset)."""
        return self.data

    @property
    def trainSet(self) -> Dict[str, np.ndarray]:
        """The training rows' columns."""
        return {k: v[: self.tr_size] for k, v in self.data.items()}

    @property
    def trSize(self) -> int:
        return self.tr_size

    @property
    def teSize(self) -> int:
        return self.te_size

    @property
    def fitSize(self) -> int:
        return self.fit_size

    @property
    def nVal(self) -> int:
        return self.n_val

    @property
    def trRng(self) -> range:
        return self.tr_rows

    @property
    def teRng(self) -> range:
        return self.te_rows

    @property
    def fitRng(self) -> range:
        return self.fit_rows

    @property
    def valRng(self) -> range:
        return self.val_rows

    @property
    def valFrac(self) -> float:
        return self.val_frac

    @property
    def valFolds(self) -> int:
        return self.val_folds

    @property
    def maxFree(self) -> int:
        return self.max_free

    @property
    def boundTol(self) -> float:
        return self.bound_tol

    @property
    def maxVIF(self) -> float:
        return self.max_vif

    @property
    def hSel(self) -> int:
        return self.h_sel

    def hparam(self, **nvs: Any) -> Dict[str, Any]:
        """The hyper-parameter map a model built here carries: the scorer's ridge penalty with the
        given entries written over it (ScalaTion's `hparam`)."""
        return {"lambda": self.ridge, **nvs}

    # -- parsing and fitting -------------------------------------------------------------

    def parse(self, text: str) -> Formula:
        """Parses a formula string against this dataset (variables, maxLag, frequencies)."""
        return parse_formula(text, self.vars, int(self.hp["maxLag"]), self.freqs)

    def _formula(self, f: Union[str, Formula]) -> Formula:
        return self.parse(f) if isinstance(f, str) else Formula.of(f)

    def build(self, f: Union[str, Formula], size: Optional[int] = None, fit_params: bool = True) -> NARX_SR_D:
        """
        Builds (untrained) the model of a formula on the full series: one column per term,
        transforms and free nonlinear parameters fitted on the first `size` rows (default: the
        fit rows; `trSize` for the final model), the parameters then held fixed
        (`NARX_SR_D.fromTerms`).
        """
        size = self.fit_size if size is None else size
        return NARX_SR_D.fromTerms(self._formula(f), self.data, self.hh, trSize=size, lam=self.ridge,
                                   scaled=self.scaled, optimizer=self.optimizer,
                                   maxLag=int(self.hp["maxLag"]), freqs=self.freqs,
                                   nneg=bool(self.hp["nneg"]), fit_params=fit_params)

    def buildFixed(self, f: Union[str, Formula]) -> NARX_SR_D:
        """
        Builds (untrained) the model of a formula with its free parameters held at their CURRENT
        values (start values, or the values of an earlier fit), without fitting them: the design
        matrix a selection stage scores column subsets of (`library`), where a fit per subset
        would be too costly.  Unset parameters are resolved from the fit rows first.
        """
        f = self._formula(f)
        f.prepare(self.data, self.fit_size)                              # resolves the unset parameters
        copies: Dict[int, Par] = {id(p): Par.const(p.value) for p in f.params()}
        frozen = Formula(f.resp, [t.fresh(copies) for t in f.preds], f.tFormT)
        return self.build(frozen, self.fit_size, fit_params=False)

    def train(self, mod: Forecaster, rows: Optional[range] = None) -> Forecaster:
        """
        Trains a model on the given leading rows (the fit rows of the search, or all training
        rows for a final model) and sets the fixed skip; counts one fit (ScalaTion's `train`).
        A direct model trains on its (Y, X) rows, a model without an input matrix on its y rows.
        """
        rows = self.tr_rows if rows is None else rows
        sl = slice(rows.start, rows.stop)
        X = getattr(mod, "X", None)
        if X is not None and getattr(mod, "Y", None) is not None:
            mod.train(mod.Y[sl], X[sl])
        elif X is not None:
            mod.train(mod.y[sl], X[sl])
        else:
            mod.train(mod.y[sl], None)
        mod.setSkip(self.skip)
        self.evaluations += 1
        return mod

    def fit(self, f: Union[str, Formula]) -> NARX_SR_D:
        """
        Fits a formula on the fit rows and returns its model, reusing the cached model when this
        structure was fitted before (by its canonical key, which ignores the values of the free
        parameters); the formula's parameters then take the cached fitted values.
        """
        f = self._formula(f)
        key = self.keyOf(f)
        mod = self._fitCache.get(key)
        if mod is None:
            mod = self.train(self.build(f), self.fit_rows)
            if len(self._fitCache) >= self.CACHE_SZ:
                self._fitCache.popitem(last=False)                       # evict the oldest
            self._fitCache[key] = mod
        elif mod.formula is not None and mod.formula is not f:
            for p, q in zip(f.params(), mod.formula.params()):           # the same structure: adopt the fit
                if p is not q:
                    p.w0, p.lb, p.ub = q.w0, q.lb, q.ub                   # the resolved box of the cached fit
                    p.set(q.value)
        return mod

    def fitFull(self, f: Union[str, Formula]) -> NARX_SR_D:
        """The final model: parameters and coefficients fitted on ALL the training rows (not cached)."""
        return self.train(self.build(f, self.tr_size), self.tr_rows)

    def fit_full(self, f: Union[str, Formula]) -> NARX_SR_D:
        return self.fitFull(f)

    # -- scoring -----------------------------------------------------------------------------

    def score(self, f: Union[str, Formula, Forecaster], label: Optional[str] = None) -> Candidate:
        """
        Fits and scores one formula (a string is parsed first): its QoF on the fit rows and on
        the validation rows, and the selection score J (`scoreOf`); a failed fit ranks last, as
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
        if 0 <= self.max_free < nf:
            return self.failed(f, f.nTerms, reason=f"{nf} free parameters, at most {self.max_free} allowed")
        try:
            mod = self.fit(f)
            why = self.rejection(f, mod)
            if why is not None:
                return self.failed(f, f.nTerms, reason=why)
            q = self.qofFit(mod)
            qv = self.qofValidationAll(mod, lambda size: self.build(f.fresh(), size))   # own copies: the candidate keeps its fit
            return Candidate(f, self.scoreOf(q, qv), q, f.nTerms, qof_val=qv)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError, ZeroDivisionError) as e:
            return self.failed(f, f.nTerms, reason=f"fit failed: {e}")

    def scoreModel(self, mod: Forecaster, label: str) -> Candidate:
        """
        Fits and scores a base model (e.g., a `RandomWalk`) the same way a formula is scored,
        under the given label (ScalaTion's `score (mod, label)`): trained on the fit rows, the
        same object retrained per validation block, then left as trained on the fit rows.
        """
        try:
            self.train(mod, self.fit_rows)
            q = self.qofFit(mod)
            qv = self.qofValidationAll(mod, lambda size: mod)
            if self.val_folds > 1 and self.n_val > 0:
                self.train(mod, self.fit_rows)
            X = getattr(mod, "X", None)
            k = X.shape[1] if X is not None else 0
            return Candidate(None, self.scoreOf(q, qv), q, k, qof_val=qv, label=label, model=mod)
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
                tol = self.bound_tol * rng
                if p.value - p.lb <= tol:
                    return f"{p.role.value} {p.value:.4g} at its lower bound {p.lb:.4g}"
                if p.ub - p.value <= tol:
                    return f"{p.role.value} {p.value:.4g} at its upper bound {p.ub:.4g}"
            if p.role is ParRole.POWER and abs(p.value - 1.0) < 0.05:
                return f"exponent {p.value:.4g} is a plain lag"
        if self.max_vif > 0.0 and pars and f.nTerms > 1:
            X = mod.X[: self.fit_size]
            Xc = X - X.mean(axis=0)
            for j, t in enumerate(f.preds):
                if any(p.free for p in t.params()):
                    v = self.vif(Xc, j)
                    if v > self.max_vif:
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
        """The validation sMAPE per horizon of a formula (exceptions propagate)."""
        return np.array([q["smape"] for q in self.qofValidation(self.fit(f))])

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
            from modeling.forecasting.nonlinear.llm_narx_sr.eda4ts import acorr
            from modeling.forecasting.nonlinear.llm_narx_sr.explainer import Explainer
            mod = self.fit(cand.formula)
            yt, yp = self._true_pred(mod, self.fit_rows, 1, False)
            e = yt - yp                                                   # h = 1 residuals on the fit rows
            acf = "[" + ", ".join(f"{acorr(e, l):.2f}" for l in range(1, k + 1)) + "]"
            imp = Explainer.importance(mod, cand.formula, self.fit_size)
            tot = sum(abs(i.effect) for i in imp)
            top = ", ".join(f"{i.label} {100 * abs(i.effect) / tot if tot > 0 else 0.0:.0f}%"
                            for i in sorted(imp, key=lambda i: -abs(i.effect))[:5])
            return f"residual autocorrelation of the best at lags 1..{k}: {acf}; its terms by importance: {top}"
        except Exception:
            return ""

    # -- the final evaluation -----------------------------------------------------------------

    def tnt(self, f: Union[str, Formula, Forecaster]):
        """
        Train-and-Test: for a formula, a fresh model fitted on ALL training rows (`fitFull`), then
        the base's rolling validation on the testing rows (sliding window, refit every hp["rc"]
        rows, no skip -- as `tnT_Test`); returns (model, QoF per horizon on the testing rows).
        For a trained base model, its QoF per horizon on the testing rows alone (`tntModel`).
        """
        if isinstance(f, Forecaster):
            return self.tntModel(f)
        mod = self.fitFull(f)
        return mod, self.tntModel(mod)

    def tntModel(self, mod: Forecaster) -> List[Dict[str, float]]:
        """The rolling validation of a trained model on the testing rows and its QoF per horizon there."""
        mod.setSkip(0)
        yf = mod.rollValidate(self.rc)
        return self.qofTnt(mod, yf)

    def qofTnt(self, mod: Forecaster, yf: np.ndarray) -> List[Dict[str, float]]:
        """The QoF per horizon of the rolling forecasts `yf` (teSize-by-hh, model units) of the testing rows."""
        out = []
        te = self.te_size
        for h in range(1, self.hh + 1):
            n = te - h + 1
            idx = np.arange(self.tr_size, self.tr_size + n)
            y_true = mod.y[idx + h - 1]
            y_pred = yf[:n, h - 1]
            if mod.yForm is not None:
                y_true = mod.yForm.inverse_transform(y_true.reshape(-1, 1)).ravel()
                y_pred = mod.yForm.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            out.append(self._diagnose(y_true, y_pred, self._dof(mod), len(y_true)))
        return out

    def qof_tnt(self, mod: Forecaster, yf: np.ndarray) -> List[Dict[str, float]]:
        return self.qofTnt(mod, yf)

    # -- QoF -----------------------------------------------------------------------------------

    @staticmethod
    def _dof(mod: Forecaster) -> int:
        """The degrees of freedom a model takes per horizon (the slopes plus any fitted nonlinear parameters)."""
        if hasattr(mod, "dof"):
            return int(mod.dof)
        try:
            return int(mod.nparams)
        except (ValueError, AttributeError):
            return 0

    @staticmethod
    def _diagnose(y_true: np.ndarray, y_pred: np.ndarray, dfr: int, size: int) -> Dict[str, float]:
        """The QoF dict of a prediction with the DF reset to `size` rows (ScalaTion's `mod_resetDF`
        then `diagnose`), keyed by the QoF names."""
        fit = Fit(float(dfr), 1.0)
        fit.resetDF_k(int(size), int(dfr))
        q = fitMapDict(fit.diagnose(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))
        q["dfr"] = float(dfr)
        return q

    @staticmethod
    def _predict(mod: Forecaster, idx: np.ndarray) -> np.ndarray:
        """The forecasts (len(idx), hh) of the given rows in model units (`predict_rows`, or the
        forecast matrix of a model without it)."""
        if hasattr(mod, "predict_rows"):
            return mod.predict_rows(idx)
        return np.asarray(mod.forecast(int(idx[0]), int(idx[-1]) + 1), dtype=float)

    def _true_pred(self, mod: Forecaster, rows: range, h: int, no_skip: bool):
        start = rows.start if no_skip else rows.start + self.skip
        stop = min(rows.stop, self.m - h + 1)
        if stop <= start:
            return None, None
        idx = np.arange(start, stop)
        y_true = mod.y[idx + h - 1]
        y_pred = self._predict(mod, idx)[:, h - 1]
        if mod.yForm is not None:
            y_true = mod.yForm.inverse_transform(y_true.reshape(-1, 1)).ravel()
            y_pred = mod.yForm.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_true, y_pred

    def diagnoseH(self, mod: Forecaster, rows: range, h: int, noSkip: bool = False) -> Dict[str, float]:
        """
        Diagnoses ONE horizon of a model on a row range: yf(t, h) is paired with y(t + h - 1), the
        first `skip` rows dropped unless `noSkip` (an out-of-sample range), a response transform
        inverted, the DF reset as ScalaTion's `diagnoseH` does (the fit rows' size on a validation
        block); returns the QoF dict of that horizon.
        """
        y_true, y_pred = self._true_pred(mod, rows, h, noSkip)
        if y_true is None:
            raise ValueError(f"diagnoseH: no rows for horizon {h} in {rows}")
        size = max(len(y_true), self.fit_size - self.skip) if noSkip else len(y_true)
        return self._diagnose(y_true, y_pred, self._dof(mod), size)

    def diagnose_h(self, mod: Forecaster, rows: range, h: int, no_skip: bool = False) -> Dict[str, float]:
        return self.diagnoseH(mod, rows, h, no_skip)

    def qofByH(self, mod: Forecaster, rng: range, noSkip: bool = False) -> List[Dict[str, float]]:
        """The QoF per horizon (a list of hh dicts) of a model over the given rows."""
        return [self.diagnoseH(mod, rng, h, noSkip) for h in range(1, self.hh + 1)]

    def qof_by_h(self, mod: Forecaster, rows: range, no_skip: bool = False) -> List[Dict[str, float]]:
        return self.qofByH(mod, rows, no_skip)

    def qofTrain(self, mod: Forecaster) -> List[Dict[str, float]]:
        """The in-sample QoF of a model trained on the training rows (the first `skip` rows dropped)."""
        return self.qofByH(mod, self.tr_rows)

    def qof_train(self, mod: Forecaster) -> List[Dict[str, float]]:
        return self.qofTrain(mod)

    def qofFit(self, mod: Forecaster) -> List[Dict[str, float]]:
        """The in-sample QoF of a model trained on the fit rows (the first `skip` rows dropped)."""
        return self.qofByH(mod, self.fit_rows)

    def qof_fit(self, mod: Forecaster) -> List[Dict[str, float]]:
        return self.qofFit(mod)

    def qofValidation(self, mod: Forecaster) -> List[Dict[str, float]]:
        """The QoF on the first validation block, every forecast there out of sample, no row
        skipped (the fit rows' QoF when there are no validation rows)."""
        return self.qofFit(mod) if self.n_val == 0 else self.qofByH(mod, self.blocks[0], noSkip=True)

    def qof_validation(self, mod: Forecaster) -> List[Dict[str, float]]:
        return self.qofValidation(mod)

    def qofValidationAll(self, mod: Forecaster, fresh: Callable[[int], Forecaster]) -> List[Dict[str, float]]:
        """
        The validation QoF over ALL rolling-origin blocks: block 0 from the model trained on the
        fit rows, block i from a fresh model `fresh(size)` trained on every row before the block,
        the QoF averaged over the blocks; with one block it is `qofValidation`.
        """
        q0 = self.qofValidation(mod)
        if self.val_folds == 1 or self.n_val == 0:
            return q0
        qs = [q0]
        for i in range(1, self.val_folds):
            rows = range(0, self.blocks[i].start)
            m_i = self.train(fresh(rows.stop), rows)
            qs.append(self.qofByH(m_i, self.blocks[i], noSkip=True))
        return [{k: float(np.mean([q[h][k] for q in qs])) for k in qs[0][h]} for h in range(self.hh)]

    def qof_validation_all(self, mod: Forecaster, fresh) -> List[Dict[str, float]]:
        return self.qofValidationAll(mod, fresh)

    @staticmethod
    def keyOf(f: Formula) -> str:
        """The canonical key of a formula: every free parameter a bare ?, so a formula and its
        fitted version are one candidate."""
        return Formula.of(f).key

    @staticmethod
    def jOf(qof: Sequence[Dict[str, float]], qofV: Sequence[Dict[str, float]], extra: int = 0) -> np.ndarray:
        """
        The FPE-form parsimony-adjusted sMAPE per horizon: sMAPE_val,h (1 + 2 p_h / n_h),
        p_h = dfr_h + 1 + extra (the slopes, the intercept, the fitted nonlinear parameters a
        `NARX_SR_D` counts in dfr, and `extra` for parameters held fixed) and n_h the rows
        diagnosed on the fit rows.
        """
        return np.array([qv["smape"] * (1.0 + 2.0 * (q["dfr"] + 1.0 + extra) / q["m"]) for q, qv in zip(qof, qofV)])

    @staticmethod
    def j_of(qof, qof_v, extra: int = 0) -> np.ndarray:
        return FormulaScorer.jOf(qof, qof_v, extra)

    def scoreOf(self, qof: Sequence[Dict[str, float]], qofV: Optional[Sequence[Dict[str, float]]] = None,
                extra: int = 0) -> float:
        """The selection score J: the mean over horizons of `jOf`, or its h = 1 value when
        hp["hSel"] = 1; inf when undefined."""
        sj = self.jOf(qof, qof if qofV is None else qofV, extra)
        s = float(sj[0]) if self.h_sel == 1 else float(sj.mean())
        return s if math.isfinite(s) else math.inf

    def score_of(self, qof, qof_v=None, extra: int = 0) -> float:
        return self.scoreOf(qof, qof_v, extra)

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
            sub = self.train(lib.getModel(cols), self.fit_rows)
            q = self.qofFit(sub)
            qv = self.qofValidationAll(sub, lambda size: lib.getModel(cols))
            s0 = self.scoreOf(q, qv, nFree)
            s = s0 * (1.0 + (nFree - self.max_free)) if (0 <= self.max_free < nFree and math.isfinite(s0)) else s0
            if not math.isfinite(s):
                return BestStep(-1, None, None, cols, qk)
            qof = qof_vector(q[0])
            qof[QoF.smape.ordinal] = float(np.mean([qh["smape"] for qh in q]))
            qof[qk] = s
            return BestStep(-1, qof, None, cols, qk, s)
        except Exception:
            return BestStep(-1, None, None, cols, qk)

    # -- reports ------------------------------------------------------------------------------

    def report(self, cand: Candidate, tnt: bool = True) -> str:
        """A text report of a candidate: formula, score, in-sample and validation sMAPE, and
        (optionally) the rolling TnT sMAPE on the testing rows."""
        lines = [f"formula:    {cand.formulaStr}", f"key:        {cand.key}",
                 f"score J:    {cand.score:.4f}   (n_terms = {cand.n_terms}, n_free = {cand.nFree}){cand.why}"]
        if not cand.failed:
            lines.append("fit sMAPE:  " + " ".join(f"{v:7.3f}" for v in cand.smapeH))
            lines.append("val sMAPE:  " + " ".join(f"{q['smape']:7.3f}" for q in cand.qofV))
            lines.append("J per h:    " + " ".join(f"{v:7.3f}" for v in cand.scoreH))
            if tnt:
                qt = self.tnt(cand.formula.fresh())[1] if cand.formula is not None else self.tnt(cand.model)
                lines.append("TnT sMAPE:  " + " ".join(f"{q['smape']:7.3f}" for q in qt) +
                             f"   mean {np.mean([q['smape'] for q in qt]):.3f}")
        return "\n".join(lines)


def _tr_size(m: int) -> int:
    """ScalaTion's `Model.trSize`."""
    from modeling.forecasting.forecaster import trSize
    return trSize(m, NARX_SR_D.TE_RATIO)
