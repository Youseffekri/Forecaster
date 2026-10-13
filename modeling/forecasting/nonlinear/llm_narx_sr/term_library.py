"""
TermLibrary: Term Library and Selection (CEM, Backward, ...) for LLM-NARX-SR

A `TermLibrary` is a set of candidate terms (one design column each) from which ONE equation is
selected by a technique of `SubsetSearch` (`selectBy`: the cross-entropy method -- the search's
selection stage --, backward elimination, forward, stepwise or beam), every set scored by
`FormulaScorer.evalCols` -- the selection criterion of the scorer on the training rows (J, the
mean FPE-form validation sMAPE over horizons), the same criterion every candidate of the search is
ranked by.  The techniques differ only in how sets are explored, never in what is scored.  Two
ways to build a library: `TermLibrary.from_` takes the union of the terms of scored candidates
(what an LLM loop proposed: the LLM writes the hypothesis space), with `frequencyPrior` turning
how often each term appeared into the CEM's starting inclusion probabilities;
`TermLibrary.fromGrammar` expands the grammar itself (every variable x lag x function), the
library an LLM did not write -- the control that isolates what the LLM's library is worth.  A
sin/cos wave pair is kept whole when only one member is selected.

Terms with FREE NONLINEAR PARAMETERS: a library term enters the design matrix at its parameter's
current value (the role start value, or the value of the fit of the candidate it came from) --
the subsets are scored without refitting the parameters --, and the selected equation gets its
own parameters (`fresh`) and is fitted and scored by the engine like any proposal.  Two copies of
a term that differ only in start or fitted values are one term.  The grammar library has the
free-parameter forms when `nlTerms` (pow(x~k, ?), log(? + x~k), sat(x~k, ?), hinge(x~k, ?),
ema(x~k, ?)).

The Python counterpart of ScalaTion's `llm_narx_sr/TermLibrary.scala`.  Differences: Scala's
`TermLibrary.from` is `from_` here (a keyword); `selectBy` returns the probabilities as a numpy
array (None for a non-CEM technique); the random stream of the CEM sampler is `CEM_Selector`'s
(ScalaTion's `Random` port).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from modeling.cem_selector import CEM_Selector
from modeling.feature_selection import Cols, SelectionTech
from modeling.fit import QoF
from modeling.forecasting.arx_formula import FormulaVar, LabelMode, Term, WaveTerm, fn, wave as wave_k
from modeling.forecasting.nonlinear.narx_formula import (NARX_Formula, Par, ParWaveTerm, decay, ema, half, hinge,
                                                        pow as pow_p, power, sat, shift, thresh)
from modeling.forecasting.arx_formula import log, pow as pow_fixed
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.subset_search import SubsetSearch


# ---------------------------------------------------------------------------------------------
#  Term helpers (ScalaTion's `NARX_Formula.params (terms)`, `NARX_Formula.fresh (terms)`)
# ---------------------------------------------------------------------------------------------

def params_of(terms: Sequence[Term]) -> List[Par]:
    """The distinct parameters (by identity) of a term list, in order of appearance."""
    out: List[Par] = []
    seen: Set[int] = set()
    for t in terms:
        for p in t.params():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def free_count(terms: Sequence[Term]) -> int:
    """The number of FREE parameters of a term list."""
    return sum(1 for p in params_of(terms) if p.free)


def has_free(t: Term) -> bool:
    """Whether a term carries a free parameter."""
    return any(p.free for p in t.params())


def fresh_terms(terms: Sequence[Term]) -> List[Term]:
    """
    Copies of the terms whose free parameters are NEW `Par` objects (one copy per parameter, so a
    parameter shared by several terms stays shared), with the numbers as given, so that a formula
    built from the terms of other formulas (a crossover, a library selection) fits its own
    parameters without touching theirs.  Terms without parameters are returned as they are.
    """
    copies: Dict[int, Par] = {}
    return [t.fresh(copies) for t in terms]


def slot_label(t: Term) -> str:
    """The label of a term with every free parameter written as ? (ScalaTion's `Par.LabelMode.Slot`)."""
    return t.label(LabelMode.SLOT)


# ---------------------------------------------------------------------------------------------
#  TermLibrary
# ---------------------------------------------------------------------------------------------

class TermLibrary:
    """
    The candidate terms of a selection problem (ScalaTion's `TermLibrary`).

    Parameters
    ----------
    resp : FormulaVar or str
        The response variable.
    terms : sequence of Term
        The candidate terms (distinct by label; one design column each).
    tFormT : TransformT, optional
        The response/base-column transform the selected equation carries (None => none).

    Attributes
    ----------
    size : int
        The number of candidate terms / columns.
    labels : list of str
        The column names (free parameters as ?).
    """

    #: The functions and powers the grammar library applies to every lagged variable.
    LIB_FUNCTIONS = ["log1p", "sq"]
    LIB_POWERS = [1.5]

    #: The free-parameter forms the grammar library applies to every lagged variable when `nlTerms`
    #: (each term gets its own parameter): a free exponent, a shifted log, a saturation, a hinge
    #: (a threshold) and an exponentially weighted moving average.
    LIB_NL_FORMS: List[Callable[[Term], Term]] = [
        lambda t: pow_p(t, power()),
        lambda t: log(shift() + t),
        lambda t: sat(t, half()),
        lambda t: hinge(t, thresh()),
        lambda t: ema(t, decay()),
    ]

    def __init__(self, resp, terms: Sequence[Term], tFormT=None) -> None:
        self.resp = resp if isinstance(resp, FormulaVar) else FormulaVar(str(resp))
        self.terms: List[Term] = list(terms)
        self.tFormT = tFormT
        self.size = len(self.terms)
        self.labels: List[str] = [slot_label(t) for t in self.terms]
        self._index = {lab: j for j, lab in reversed(list(enumerate(self.labels)))}   # first occurrence wins

    def indexOf(self, label: str) -> int:
        """The column of a term label, or -1 if the library lacks it; a label with start values (?w)
        matches, since the library's labels write free parameters as ?."""
        return self._index.get(NARX_Formula.slots(label), -1)

    def completeWaves(self, cols: Sequence[int]) -> Cols:
        """The columns with a wave pair completed: when one member of a sin/cos pair (same
        `WaveTerm.idx`, or the same free frequency) is in cols, its partner is added (library order)."""
        cols = set(cols)
        idxs = {self.terms[j].idx for j in cols if isinstance(self.terms[j], WaveTerm)}
        freqs = [self.terms[j].freq for j in cols if isinstance(self.terms[j], ParWaveTerm)]
        out: Cols = []
        for j, t in enumerate(self.terms):
            partner = (isinstance(t, WaveTerm) and t.idx in idxs) or \
                      (isinstance(t, ParWaveTerm) and any(f is t.freq for f in freqs))
            if j in cols or partner:
                out.append(j)
        return out

    def formulaOf(self, cols: Sequence[int]) -> NARX_Formula:
        """The formula made of the given columns (library order, wave pairs completed, own parameters, the library's transform)."""
        keep = set(self.completeWaves(cols))
        return NARX_Formula(self.resp, fresh_terms([t for j, t in enumerate(self.terms) if j in keep]), self.tFormT)

    @property
    def full(self) -> NARX_Formula:
        """The formula containing every term of the library (own parameters)."""
        return NARX_Formula(self.resp, fresh_terms(self.terms), self.tFormT)

    def finite(self, dataset: Dict[str, np.ndarray]) -> "TermLibrary":
        """The library without the terms whose data column is not finite on the given rows (e.g.,
        log of a zero, a power of a negative value), so the library model can be built."""
        ok = []
        for t in self.terms:
            if not all(v in dataset for v in t.vars):
                continue
            try:
                col = t.column(dataset)
            except Exception:
                continue
            if np.all(np.isfinite(col)):
                ok.append(t)
        return TermLibrary(self.resp, ok, self.tFormT)

    def unitOf(self, j: int) -> Set[int]:
        """The unit of a column: the sin/cos pair of a wave term (the two columns enter and leave a
        set together), the column itself otherwise."""
        tj = self.terms[j]

        def sameWave(k: int) -> bool:
            tk = self.terms[k]
            if isinstance(tj, WaveTerm) and isinstance(tk, WaveTerm):
                return tj.idx == tk.idx
            if isinstance(tj, ParWaveTerm) and isinstance(tk, ParWaveTerm):
                return tj.freq is tk.freq
            return False

        return {k for k in range(self.size) if k == j or sameWave(k)}

    def freeCount(self, cols: Sequence[int]) -> int:
        """The number of free nonlinear parameters in a column set."""
        return sum(sum(1 for p in self.terms[j].params() if p.free) for j in cols)

    def withinCap(self, cols: Sequence[int], maxFree: int) -> Cols:
        """A column set brought within the free-parameter cap: while it holds more than `maxFree`
        free parameters, the free-parameter column added last (highest index) is dropped."""
        out = list(cols)
        if maxFree < 0:
            return out
        while self.freeCount(out) > maxFree:
            j = max(k for k in out if has_free(self.terms[k]))
            out.remove(j)
        return out

    def selectBy(self, tech: SelectionTech, scorer: FormulaScorer, prior: Optional[Sequence[float]] = None,
                 stream: int = 0, maxCols: int = -1, nSample: int = CEM_Selector.N_SAMPLE,
                 maxIter: int = CEM_Selector.MAX_ITER, minCols: int = 1
                 ) -> Tuple[Candidate, Optional[np.ndarray], int]:
        """
        Selects one equation from the library by a selection technique of `SubsetSearch` -- the
        cross-entropy method (`CEM_Selector`), backward elimination, forward selection, stepwise
        selection or beam search -- every set scored by the criterion of the scorer
        (`FormulaScorer.evalCols`: the fit rows, the ridge penalty, J) with wave pairs completed
        first and kept whole (`unitOf`).  Returns the selected equation scored as a `Candidate`, the
        final inclusion probability per column (CEM; None otherwise) and the number of column sets
        evaluated.  After a CEM run the best sets of the last hp["refitTop"] rounds are re-scored
        with their free parameters fitted, and a selected equation whose free parameter is
        degenerate in context falls back to its plain terms.

        Parameters
        ----------
        tech : SelectionTech
            The selection technique.
        scorer : FormulaScorer
            The fit-and-score engine (its training rows, ridge and criterion).
        prior : sequence of float, optional
            The prior inclusion probability per column (CEM; None => 0.5 each).
        stream : int, default=0
            The random number stream of the sampler (CEM).
        maxCols : int, default=-1
            The maximum number of columns in a set (-1 => no cap; a completed wave pair may add one).
        nSample : int
            The number of column sets sampled per round (CEM; -1 => 5 x columns).
        maxIter : int
            The maximum number of rounds (CEM).
        minCols : int, default=1
            The smallest set kept (Backward).
        """
        if self.size == 0:
            raise ValueError("selectBy: the library is empty")
        qk = QoF.smapeC.ordinal                                           # the QoF slot carrying the selection score
        lib = scorer.library(self.resp, self.terms, self.tFormT)

        def eval_(cols: Cols):
            keep = self.completeWaves(cols)
            return scorer.evalCols(lib, keep, self.freeCount(keep), qk)

        maxFree = int(scorer.hp["maxFree"])
        freeCol = [j for j in range(self.size) if has_free(self.terms[j])]
        prior_ = None if prior is None else np.asarray(prior, dtype=float).copy()
        if 0 <= maxFree < len(freeCol):                                   # CEM: a free-parameter column starts as likely as the cap allows
            p = np.full(self.size, 0.5) if prior_ is None else prior_
            pf = min(0.5, 0.8 * maxFree / len(freeCol))
            for j in freeCol:
                p[j] = min(p[j], pf)
            prior_ = p
        if tech is SelectionTech.CEM:
            path = SubsetSearch.cem(self.size, eval_, [], prior_, nSample, maxIter, maxCols, stream, qk=qk)
        elif tech is SelectionTech.Forward:
            path = SubsetSearch.forward(self.size, eval_, [], self.unitOf, maxCols, qk=qk)
        elif tech is SelectionTech.Backward:
            path = SubsetSearch.backward(self.size, eval_, [], self.unitOf, minCols, qk=qk)
        elif tech is SelectionTech.Stepwise:
            path = SubsetSearch.stepwise(self.size, eval_, [], self.unitOf, qk=qk)
        elif tech is SelectionTech.Beam:
            path = SubsetSearch.beam(self.size, eval_, [], self.unitOf, qk=qk)
        else:
            raise ValueError(f"selectBy: unknown technique {tech}")

        f = self.formulaOf(self.withinCap(path.cols, maxFree)) if path.found else self.full
        cand = scorer.score(f)                                            # re-scored as a formula (same fit path)
        top = int(scorer.hp["refitTop"])
        if tech is SelectionTech.CEM and top > 0 and path.found and self.freeCount(path.cols) > 0:
            seen: List[frozenset] = []
            sets: List[Cols] = []
            for s in reversed([st for st in path.steps if st.mod_cols is not None]):
                c = self.withinCap(s.mod_cols, maxFree)
                if frozenset(c) not in seen:
                    seen.append(frozenset(c))
                    sets.append(c)
                if len(sets) >= top:
                    break
            for s in sets:
                if set(s) != set(path.cols):
                    c = scorer.score(self.formulaOf(s))
                    if not c.failed and (cand.failed or c.score < cand.score):
                        cand = c
        if cand.failed and path.found and self.freeCount(path.cols) > 0:  # a free parameter degenerate in context
            plain = [j for j in path.cols if not has_free(self.terms[j])]
            if plain:
                c = scorer.score(self.formulaOf(plain))
                if not c.failed:
                    cand = c
        return cand, path.probs, path.nEval

    def fitted(self, scorer: FormulaScorer) -> "TermLibrary":
        """
        The library with every free-parameter term fitted alone against the response (the formula
        resp ~= term, `FormulaScorer.score`, fit rows), its parameters then holding the fitted
        values the selection stage builds the design matrix at; a term whose fit fails or is
        rejected (a parameter on its bound) is dropped with its reason.  Terms without free
        parameters pass unchanged.  One fit per free-parameter term.
        """
        keep: List[Term] = []
        nDrop = 0
        for t in self.terms:
            if has_free(t):
                c = scorer.score(NARX_Formula(self.resp, [t], self.tFormT))
                if c.failed:
                    nDrop += 1
                else:
                    keep.append(c.formula.preds[0])                       # the term with its fitted parameter
            else:
                keep.append(t)
        if nDrop > 0:
            print(f"TermLibrary.fitted: {self.size - nDrop} of {self.size} terms kept ({nDrop} free-parameter terms dropped: degenerate fits)")
        return TermLibrary(self.resp, keep, self.tFormT)

    def showProbs(self, probs: Sequence[float], k: int = 10) -> str:
        """The inclusion probabilities next to the term labels, highest first."""
        probs = np.asarray(probs, dtype=float)
        order = sorted(range(self.size), key=lambda j: -probs[j])[:k]
        return ", ".join(f"{self.labels[j]} {probs[j]:.2f}" for j in order)

    def __repr__(self) -> str:
        s = f"TermLibrary ({self.resp.name}, {', '.join(self.labels)}"
        return s + (")" if self.tFormT is None else f", TFORM: {self.tFormT.name})")

    # -- builders (ScalaTion's companion object) ------------------------------------------------

    @staticmethod
    def from_(cands: Sequence[Candidate]) -> "TermLibrary":
        """The union library of the terms of the given candidates (failed candidates excluded); the
        transform is the best candidate's (ScalaTion's `TermLibrary.from`)."""
        ok = [c for c in cands if not c.failed and c.formula is not None]
        if not ok:
            raise ValueError("from: no fitted candidates to build a library from")
        terms = NARX_Formula._distinct([t for c in ok for t in c.formula.preds])   # by structure: fitted values do not count
        return TermLibrary(ok[0].formula.resp, terms, ok[0].formula.tFormT)

    @staticmethod
    def frequencyPrior(cands: Sequence[Candidate], lib: "TermLibrary", eps: float = CEM_Selector.EPS) -> np.ndarray:
        """The prior inclusion probability per library column: the share of candidates whose
        formula contains the term, clamped to [eps, 1 - eps]."""
        ok = [c for c in cands if not c.failed and c.formula is not None]
        p = np.zeros(lib.size)
        for c in ok:
            for t in c.formula.preds:
                j = lib.indexOf(slot_label(t))
                if j >= 0:
                    p[j] += 1.0 / len(ok)
        return np.clip(p, eps, 1.0 - eps)

    @staticmethod
    def fromGrammar(resp: str, exo: Sequence[str], maxLag: int, funcs: Optional[Sequence[str]] = None,
                    powers: Optional[Sequence[float]] = None, freqs: Optional[Sequence[float]] = None,
                    nWaves: int = 0, tFormT=None, nlTerms: bool = False) -> "TermLibrary":
        """
        The grammar expanded into a library: every variable (response first) at lags 1..maxLag,
        each such term under every function and power, the free-parameter forms when `nlTerms`,
        and the seasonal wave pairs.

        Parameters
        ----------
        resp : str
            The response variable name.
        exo : sequence of str
            The exogenous variable names.
        maxLag : int
            The maximum lag.
        funcs : sequence of str, optional
            The function names applied to each lagged variable (defaults to `LIB_FUNCTIONS`).
        powers : sequence of float, optional
            The exponents applied to each lagged variable (defaults to `LIB_POWERS`).
        freqs : sequence of float, optional
            The wave frequencies (None => no waves).
        nWaves : int, default=0
            The number of wave pairs (capped by len(freqs)).
        tFormT : TransformT, optional
            The response/base-column transform (None => none).
        nlTerms : bool, default=False
            Whether to add the free-parameter forms (`LIB_NL_FORMS`) of each lagged variable.
        """
        funcs = TermLibrary.LIB_FUNCTIONS if funcs is None else list(funcs)
        powers = TermLibrary.LIB_POWERS if powers is None else list(powers)
        vars_ = [resp] + list(exo)
        terms: List[Term] = []
        for v in vars_:
            for k in range(1, maxLag + 1):
                terms.append(FormulaVar(v, k))
        for v in vars_:
            for k in range(1, maxLag + 1):
                for f in funcs:
                    terms.append(fn(f, FormulaVar(v, k)))
        for v in vars_:
            for k in range(1, maxLag + 1):
                for p in powers:
                    terms.append(pow_fixed(FormulaVar(v, k), p))
        if nlTerms:
            for v in vars_:
                for k in range(1, maxLag + 1):
                    for form in TermLibrary.LIB_NL_FORMS:
                        terms.append(form(FormulaVar(v, k)))
        if freqs is not None and nWaves > 0 and len(freqs) > 0:
            terms += wave_k(min(nWaves, len(freqs)), list(freqs))
        return TermLibrary(FormulaVar(resp), NARX_Formula._distinct(terms), tFormT)
