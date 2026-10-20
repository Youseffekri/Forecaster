"""
GeneticProposer: Random Terms and the Genetic Proposer (no LLM) for LLM-NARX-SR

`RandomTerms` draws terms and formulas from the grammar with a seeded random stream, so a run is
reproduced by its stream number -- with `nonlinear` the free-parameter forms as well (pow(x~k, ?),
log(? + x~k), exp(? * x~k), sat, hinge, trans, ema, dq, wave(?), see `NARX_FormulaParser`), each
draw with its own parameter.  `GeneticProposer` is the zero-LLM update rule: offspring of the
population's best formulas by crossover (union split) and mutation (add, drop, replace a term,
shift a lag, toggle the log1p transform); an offspring gets its own copies of the inherited
parameters (`fresh_terms`), so fitting it never touches a parent.  With `random = True` it ignores
the population and samples the grammar uniformly (the no-feedback control).  Both emit
`NARX_Formula`s directly -- there is nothing to parse -- and never a reject.

The Python counterpart of ScalaTion's `llm_narx_sr/GeneticProposer.scala`.  Difference: the
uniform random numbers come from numpy's `default_rng (stream)` (ScalaTion: its own
`scalation.random.Random (stream)`, whose port is `modeling.cem_selector.Random`), so a stream
reproduces a Python run, not the Scala draw sequence.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from Util.transform import TransformT
from modeling.forecasting.arx_formula import (FormulaVar, PowerTerm, ProductTerm, Term, TransformedTerm, WaveTerm,
                                              exp, fn, log, pow as pow_fixed, wave as wave_k)
from modeling.forecasting.nonlinear.narx_formula import (DqTerm, EmaTerm, HingeTerm, NARX_Formula, ParPowerTerm, ParTerm,
                                                        ParWaveTerm, SatTerm, ScaleTerm, ShiftTerm, TransTerm, cap as cap_p,
                                                        decay, dq, ema, freq, half, hinge, pow as pow_p, power, rate, sat,
                                                        shift, thresh, trans, wave as wave_p, width)
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import Source, Trace
from modeling.forecasting.nonlinear.llm_narx_sr.search_loop import Proposal, Proposer
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import fresh_terms, has_free, params_of


def _tform(tFormT):
    """The Log1p transform type when none is given (for the transformed proposals)."""
    return TransformT.Log1p if tFormT is None else None


class RandomTerms:
    """
    Draws terms and formulas from the grammar (ScalaTion's `RandomTerms`).

    Parameters
    ----------
    resp : str
        The response variable name.
    exo : sequence of str
        The exogenous variable names.
    maxLag : int
        The maximum lag.
    freqs : sequence of float, optional
        The wave frequencies (None or empty => no wave terms).
    stream : int
        The random number stream (numpy's `default_rng (stream)`).
    funcs : sequence of str
        The function names a term may be wrapped in.
    powers : sequence of float
        The exponents a term may be raised to.
    nonlinear : bool, optional
        Whether to draw the free-parameter forms as well (defaults to hp["nlTerms"] = 1).
    """

    def __init__(self, resp: str, exo: Sequence[str], maxLag: int, freqs: Optional[Sequence[float]], stream: int,
                 funcs: Sequence[str] = ("log1p", "sq"), powers: Sequence[float] = (0.5, 1.5, 2.0),
                 nonlinear: Optional[bool] = None) -> None:
        self.resp, self.exo, self.maxLag = resp, list(exo), int(maxLag)
        self.freqs = [] if freqs is None else list(freqs)
        self.stream = int(stream)
        self.rng = np.random.default_rng(self.stream)
        self.funcs, self.powers = list(funcs), list(powers)
        self.nonlinear = (int(FormulaScorer.hp["nlTerms"]) == 1) if nonlinear is None else bool(nonlinear)
        self.nFreq = len(self.freqs)

        #: The free-parameter forms a draw may produce (each with a new parameter).
        self.NL_FORMS: List[Callable[[Term], List[Term]]] = [
            lambda t: [pow_p(t, power())],
            lambda t: [log(shift() + t)],
            lambda t: [exp(rate() * t)],
            lambda t: [sat(t, half())],
            lambda t: [hinge(t, thresh())],
            lambda t: [trans(t, thresh(), width())],
            lambda t: [ema(t, decay())],
            lambda t: [dq(t, cap_p())],
            lambda t: wave_p(freq()),                                     # a free-frequency pair (no source term)
        ]

    def randInt(self, n: int) -> int:
        """A uniform random integer in 0 until n."""
        return min(n - 1, int(self.uniform() * n))

    def uniform(self) -> float:
        """A uniform random number in (0, 1)."""
        return float(self.rng.random())

    def atom(self) -> FormulaVar:
        """A lagged variable: the response with probability 1/2, else an exogenous variable, at a lag in 1..maxLag."""
        v = self.resp if (not self.exo or self.uniform() < 0.5) else self.exo[self.randInt(len(self.exo))]
        return FormulaVar(v, 1 + self.randInt(self.maxLag))

    def term(self) -> List[Term]:
        """
        One term: a lagged variable (1/2), a function of one (1/5), a power of one (1/10), a product
        of two (1/10) or a wave pair's member (1/10, when frequencies exist; both members are
        returned so the pair stays whole); with `nonlinear`, 15 % of the lagged-variable draws
        become a free-parameter form (`NL_FORMS`) of the variable instead.
        """
        u = self.uniform()
        if u < 0.5:
            if self.nonlinear and u < 0.15:
                return self.NL_FORMS[self.randInt(len(self.NL_FORMS))](self.atom())
            return [self.atom()]
        if u < 0.7:
            return [fn(self.funcs[self.randInt(len(self.funcs))], self.atom())]
        if u < 0.8:
            return [pow_fixed(self.atom(), self.powers[self.randInt(len(self.powers))])]
        if u < 0.9 or self.nFreq == 0:
            return [ProductTerm(self.atom(), self.atom())]
        return wave_k(1 + self.randInt(self.nFreq), self.freqs)          # pairs 1..k: the grammar's wave(k)

    def normalizeWaves(self, preds: Sequence[Term]) -> List[Term]:
        """The wave terms of a term list normalized to the grammar's wave(k): when any wave term is
        present, the pairs 1..k (k the largest pair index present) replace them, so the formula
        string stays parseable."""
        waves = [t for t in preds if isinstance(t, WaveTerm)]
        rest = [t for t in preds if not isinstance(t, WaveTerm)]
        if not waves or self.nFreq == 0:
            return rest
        k = min(self.nFreq, max(t.idx for t in waves) + 1)
        return rest + wave_k(k, self.freqs)

    def formula(self, n: int, pTForm: float = 0.2) -> NARX_Formula:
        """A formula with about n terms (duplicates collapsed) and, with probability pTForm, the log1p response transform."""
        preds: List[Term] = []
        for _ in range(n):
            preds += self.term()
        tf = _tform(None) if self.uniform() < pTForm else None
        return NARX_Formula(FormulaVar(self.resp), self.normalizeWaves(preds), tf)


class GeneticProposer(Proposer):
    """
    Proposes offspring of the best formulas by crossover and mutation (or random grammar samples
    when `random`), ScalaTion's `GeneticProposer`.

    Parameters
    ----------
    resp : str
        The response variable name.
    exo : sequence of str
        The exogenous variable names.
    maxLag : int
        The maximum lag.
    freqs : sequence of float, optional
        The wave frequencies (None => none).
    stream : int
        The random number stream.
    maxTerms : int, default=12
        The maximum number of terms per offspring.
    random : bool, default=False
        Whether to ignore the population and sample the grammar.
    pCross : float, default=0.5
        The probability that an offspring is a crossover of two parents.
    nonlinear : bool, optional
        Whether the draws include the free-parameter forms (defaults to hp["nlTerms"] = 1).
    maxFree : int, optional
        The free-parameter cap an offspring is brought within (defaults to hp["maxFree"]).
    """

    def __init__(self, resp: str, exo: Sequence[str], maxLag: int, freqs: Optional[Sequence[float]], stream: int,
                 maxTerms: int = 12, random: bool = False, pCross: float = 0.5, nonlinear: Optional[bool] = None,
                 maxFree: Optional[int] = None) -> None:
        self.rt = RandomTerms(resp, exo, maxLag, freqs, stream, nonlinear=nonlinear)
        self.resp, self.maxLag = resp, int(maxLag)
        self.maxTerms, self.random, self.pCross = int(maxTerms), bool(random), float(pCross)
        self.maxFree = int(FormulaScorer.hp["maxFree"]) if maxFree is None else int(maxFree)

    @property
    def source(self) -> Source:
        return Source.Random if self.random else Source.Genetic

    def propose(self, top: Sequence[Trace], n: int) -> List[Proposal]:
        """n formulas: random grammar samples when `random` or the population is empty, else
        offspring of parents drawn rank-biased from the top (the KEPT equations of the traces)."""
        out: List[Proposal] = []
        for _ in range(n):
            f = self.rt.formula(2 + self.rt.randInt(5)) if (self.random or not top) else self.offspring(top)
            out.append(self.cap(f))
        return out

    def offspring(self, top: Sequence[Trace]) -> NARX_Formula:
        """One offspring: a parent (crossed with a second parent with probability pCross), then mutated."""
        a = self.parent(top)
        child = self.crossover(a, self.parent(top)) if (len(top) > 1 and self.rt.uniform() < self.pCross) else a
        return self.mutate(child)

    def parent(self, top: Sequence[Trace]) -> NARX_Formula:
        """A parent from the top, rank-biased (the best twice as likely as the second, and so on down:
        index = floor of a squared uniform)."""
        u = self.rt.uniform()
        return top[min(len(top) - 1, int(u * u * len(top)))].fTrim

    def crossover(self, a: NARX_Formula, b: NARX_Formula) -> NARX_Formula:
        """A random split of the union of the two parents' terms (each term kept with probability 1/2,
        at least one kept); the transform of parent a; own copies of the inherited parameters."""
        union = NARX_Formula._distinct(list(a.preds) + list(b.preds))
        kept = [t for t in union if self.rt.uniform() < 0.5]
        preds = kept if kept else [union[self.rt.randInt(len(union))]]
        return NARX_Formula(a.resp, fresh_terms(preds), a.tFormT)

    def mutate(self, f: NARX_Formula) -> NARX_Formula:
        """One random structural move: add a term (1/4), drop a term (1/4), replace a term (1/5),
        shift the lag of a lagged variable (1/5) or toggle the log1p transform (1/10)."""
        preds = fresh_terms(f.preds)                                     # own copies of the parameters
        u = self.rt.uniform()
        if u < 0.25 or (len(preds) <= 1 and u < 0.5):
            preds += self.rt.term()
        elif u < 0.5:
            self._drop(preds)
        elif u < 0.7:
            self._drop(preds)
            preds += self.rt.term()
        elif u < 0.9:
            j = self.rt.randInt(len(preds))
            preds[j] = self.shift(preds[j])
        else:
            return NARX_Formula(f.resp, preds, _tform(f.tFormT))
        return NARX_Formula(f.resp, self.rt.normalizeWaves(preds), f.tFormT)

    def _drop(self, preds: List[Term]) -> None:
        """Drops one random term; dropping a wave member drops its pair and every higher pair (wave(k)
        becomes wave(k-1)); a free-frequency member drops its pair."""
        if not preds:
            return
        w = preds[self.rt.randInt(len(preds))]
        if isinstance(w, WaveTerm):
            preds[:] = [t for t in preds if not (isinstance(t, WaveTerm) and t.idx >= w.idx)]
        elif isinstance(w, ParWaveTerm):
            preds[:] = [t for t in preds if not (isinstance(t, ParWaveTerm) and t.freq is w.freq)]
        else:
            preds.remove(w)

    def shift(self, t: Term) -> Term:
        """The lag of a lagged variable shifted by one step (within 1..maxLag), through a function,
        power, product or free-parameter wrapper; other terms are returned unchanged."""
        if isinstance(t, FormulaVar):
            l = t.lag
            if l >= self.maxLag:
                l2 = l - 1
            elif l <= 1:
                l2 = l + 1
            else:
                l2 = l - 1 if self.rt.uniform() < 0.5 else l + 1
            return FormulaVar(t.name, l2)
        if isinstance(t, TransformedTerm):
            return TransformedTerm(self.shift(t.source), t.op)
        if isinstance(t, PowerTerm):
            return PowerTerm(self.shift(t.source), t.p, t.safe)
        if isinstance(t, ProductTerm):
            return ProductTerm(self.shift(t.left), t.right) if self.rt.uniform() < 0.5 else ProductTerm(t.left, self.shift(t.right))
        if isinstance(t, ShiftTerm):
            return ShiftTerm(self.shift(t.source), t.s)
        if isinstance(t, ScaleTerm):
            return ScaleTerm(self.shift(t.source), t.a)
        if isinstance(t, ParPowerTerm):
            return ParPowerTerm(self.shift(t.source), t.p)
        if isinstance(t, SatTerm):
            return SatTerm(self.shift(t.source), t.k)
        if isinstance(t, HingeTerm):
            return HingeTerm(self.shift(t.source), t.c)
        if isinstance(t, TransTerm):
            return TransTerm(self.shift(t.source), t.c, t.w)
        if isinstance(t, EmaTerm):
            return EmaTerm(self.shift(t.source), t.d)
        if isinstance(t, DqTerm):
            return DqTerm(self.shift(t.source), t.mx)
        return t

    def cap(self, f: NARX_Formula) -> NARX_Formula:
        """The formula with its number of terms capped (random drops beyond maxTerms), its free
        parameters brought within the cap (a random free-parameter term becomes the plain term under
        the parameter) and at least one term."""
        preds = list(f.preds)
        while len(preds) > self.maxTerms:
            self._drop(preds)
        while self.maxFree >= 0 and sum(1 for p in params_of(preds) if p.free) > self.maxFree:
            js = [j for j, t in enumerate(preds) if has_free(t)]
            j = js[self.rt.randInt(len(js))]
            t = preds[j]
            if isinstance(t, ParTerm):
                preds[j] = t.source                                       # the plain term under the parameter
            elif isinstance(t, ParWaveTerm):
                preds = [s for s in preds if not (isinstance(s, ParWaveTerm) and s.freq is t.freq)]
            else:
                preds.pop(j)                                              # e.g., log(? + x): the term goes
        if not preds:
            preds += self.rt.term()
        return NARX_Formula(f.resp, self.rt.normalizeWaves(NARX_Formula._distinct(preds)), f.tFormT)
