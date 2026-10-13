"""
SearchLoop: the Propose-Score-Feed-Back Loop of LLM-NARX-SR

A `SearchLoop` keeps a de-duplicated population of scored formulas (keyed by `NARX_Formula.key`)
and runs rounds of

    propose (top-K traces shown) -> [trim] -> score -> add to population

for any `Proposer` (an LLM, a genetic rule, random grammar sampling) -- the proposals may carry
free nonlinear parameters, which the scorer fits; two proposals that differ only in the start or
fitted values of their parameters are one candidate (`FormulaScorer.keyOf`) --, with an ADAPTIVE
length: the loop stops after `patience` consecutive rounds without a better best score, when a
round yields nothing new twice in a row, when the fit budget (`maxFits` > 0) is spent, or at
`maxRounds`.  Every entry into the population -- a seed, a proposal, the selection stage's
equation -- is recorded as a `Trace` (who proposed what, in which round, and what the trimmer
kept), so the origin of every equation can be read back.  With a trimmer (`SelectionTech`) each
legal proposal f with at least hp["minTerms"] terms is scored, trimmed to fTrim by that selection
technique over its own terms -- one CEM round, backward elimination, forward, stepwise or beam,
all on the scorer's criterion (`TermLibrary.selectBy`; f itself when no proper subset scores
better, so trimming never worsens a candidate) -- and fTrim is what the population holds and the
proposer sees next (with the dropped terms); validity counts f.  It counts proposals, legal
proposals, fit failures and logs the best score per round, so every arm reports the same budget
and convergence columns.  No arm logic lives here.

The Python counterpart of ScalaTion's `llm_narx_sr/SearchLoop.scala`.  Differences: a proposal is
a `NARX_Formula` or a `Reject` (Scala's `Either [String, ARX_Formula]`); the loop's debug line per
round is printed when `verbose`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from modeling.feature_selection import SelectionTech
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import ConvRow, Source, Trace
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import TermLibrary


@dataclass(frozen=True)
class Reject:
    """A proposal that could not be turned into a legal formula (Scala's `Left (reason)`)."""
    reason: str

    def __str__(self) -> str:
        return self.reason


#: A proposal: a legal formula or a reject with its reason.
Proposal = Union[NARX_Formula, Reject]


class Proposer:
    """
    Generates new formulas, given the best traces so far (ScalaTion's `Proposer` trait).  A
    proposal that could not be turned into a legal formula (a rejected string) is returned as a
    `Reject (reason)`, so validity can be counted.
    """

    @property
    def source(self) -> Source:
        """Who this proposer is (recorded in every trace it causes)."""
        raise NotImplementedError

    def propose(self, top: Sequence[Trace], n: int) -> List[Proposal]:
        """
        Proposes n new formulas (or rejects with reasons).

        Parameters
        ----------
        top : sequence of Trace
            The origin traces of the current best candidates, best first (each carries the
            proposed f, the kept fTrim and their scores).
        n : int
            The number of proposals requested.
        """
        raise NotImplementedError

    def feedback(self, failed: Sequence[Tuple[str, str]], notes: str) -> None:
        """
        Takes the round's feedback: the proposals that were legal but failed or were rejected by
        the scorer, each with its reason, and diagnostic notes on the current best
        (`FormulaScorer.notes`); a proposer may ignore it.
        """
        return None


class SearchLoop:
    """
    Runs propose -> [trim] -> score -> feed back over a de-duplicated population of scored formulas.

    Parameters
    ----------
    scorer : FormulaScorer
        The fit-and-score engine.
    topK : int, optional
        The number of best candidates shown to the proposer each round (defaults to hp["topK"]).
    trimmer : SelectionTech, optional
        The selection technique that trims every proposal (None => full version).
    stream : int, default=0
        The random number stream of a stochastic trimmer (CEM).
    maxFits : int, optional
        The model fits this loop may make (0 => no budget; defaults to hp["maxFits"]).
    verbose : bool, default=True
        Whether to print one line per round (ScalaTion's debug output).
    """

    def __init__(self, scorer: FormulaScorer, topK: Optional[int] = None, trimmer: Optional[SelectionTech] = None,
                 stream: int = 0, maxFits: Optional[int] = None, verbose: bool = True) -> None:
        self.scorer = scorer
        self.topK = int(scorer.hp["topK"]) if topK is None else int(topK)
        self.trimmer = trimmer
        self.stream = stream
        self.maxFits = int(scorer.hp["maxFits"]) if maxFits is None else int(maxFits)
        self.verbose = verbose
        self._scored: Dict[str, Candidate] = {}                          # key -> every formula scored (f and fTrim)
        self._population: Dict[str, Candidate] = {}                      # key -> kept candidates (insertion order)
        self._traceLog: List[Trace] = []                                 # every entry, in order
        self._convLog: List[ConvRow] = []                                # best-so-far per round
        self._minTerms = int(scorer.hp["minTerms"])                      # smallest proposal a trimmer touches
        self._maxFree = int(scorer.hp["maxFree"])                        # free parameters allowed per proposal (-1 => no cap)
        self._fits0 = scorer.evaluations                                 # fits made before this loop
        self._bestSoFar = math.inf                                       # best score seen (seeds included)
        self._nRounds = 0
        self._nProposed = self._nValid = self._nFailed = 0               # counters (seeds excluded)
        self._roundFailed: List[Tuple[str, str]] = []                    # this round's failed/rejected proposals with reasons
        self._error: Optional[str] = None                                # the proposer's failure that ended the loop, if any

    # -- the population --------------------------------------------------------------------------

    def top(self, k: Optional[int] = None) -> List[Candidate]:
        """The population's best candidates, best (lowest score) first; failed fits are excluded
        (they stay in the population so they are not scored again)."""
        k = self.topK if k is None else k
        return sorted([c for c in self._population.values() if not c.failed], key=lambda c: c.score)[:k]

    def topTraces(self, k: Optional[int] = None) -> List[Trace]:
        """The origin traces of the best candidates, best first (what a proposer is shown)."""
        out = []
        for c in self.top(k):
            t = self.originOf(c)
            if t is not None:
                out.append(t)
        return out

    @property
    def best(self) -> Optional[Candidate]:
        """The best candidate, if any fit succeeded."""
        t = self.top(1)
        return t[0] if t else None

    @property
    def all(self) -> List[Candidate]:
        """Every kept candidate (fitted ones first, best first)."""
        return sorted(self._population.values(), key=lambda c: c.score)

    def originOf(self, cand: Candidate) -> Optional[Trace]:
        """The first trace that led to a candidate (as f or as fTrim)."""
        for t in self._traceLog:
            if t.candTrim.key == cand.key or t.cand.key == cand.key:
                return t
        return None

    @property
    def traces(self) -> List[Trace]:
        return list(self._traceLog)

    @property
    def convergence(self) -> List[ConvRow]:
        return list(self._convLog)

    @property
    def rounds(self) -> int:
        return self._nRounds

    @property
    def nProposed(self) -> int:
        return self._nProposed

    @property
    def nValid(self) -> int:
        return self._nValid

    @property
    def nFailed(self) -> int:
        return self._nFailed

    @property
    def size(self) -> int:
        """The distinct candidates kept."""
        return len(self._population)

    @property
    def error(self) -> Optional[str]:
        """Why the proposer stopped the loop (None => it did not)."""
        return self._error

    @property
    def fits(self) -> int:
        """The model fits made since this loop was created."""
        return self.scorer.evaluations - self._fits0

    # -- entering formulas ---------------------------------------------------------------------

    def seed(self, *fs: Tuple[NARX_Formula, Source]) -> None:
        """Seeds the population with formulas (e.g., the plain ARX baseline), each with its source
        (`Seed`, or `Selected` for the equation a selection stage handed back); seeds are not
        counted as proposals."""
        for f, src in fs:
            self.enter(f, src, 0)
        b = self.best
        self._bestSoFar = b.score if b is not None else math.inf

    def candidateOf(self, f: NARX_Formula) -> Candidate:
        """Scores a formula once (per key) and returns its candidate."""
        key = FormulaScorer.keyOf(f)
        c = self._scored.get(key)
        if c is None:
            c = self.scorer.score(f)
            self._scored[key] = c
        return c

    def trim(self, f: NARX_Formula, tech: SelectionTech) -> NARX_Formula:
        """
        Trims a formula by a selection technique over its own terms and returns the kept equation:
        the best-scoring subset the technique explored, or the formula itself when no proper subset
        scores better (so trimming never worsens a candidate) or no subset could be fitted.  The
        CEM trimmer runs one round from a flat prior, the others their full search.
        """
        lib = TermLibrary(f.resp, f.preds, f.tFormT)                     # the kept equation gets its own parameters
        if tech is SelectionTech.CEM:
            cand = lib.selectBy(tech, self.scorer, stream=self.stream, maxIter=1)[0]
        else:
            cand = lib.selectBy(tech, self.scorer, stream=self.stream)[0]
        if cand.failed:
            print(f"SearchLoop.trim: {tech} left no fitted equation -- f kept")
            return f
        if cand.score < self.candidateOf(f).score:
            return cand.formula                                           # a proper subset scores better
        return f                                                          # nothing to gain: keep the proposal

    def enter(self, f: NARX_Formula, source: Source, round_: int) -> Trace:
        """Enters a formula: scores it, trims it when a trimmer is set, records the trace and keeps
        the (trimmed) candidate in the population unless already there; returns the trace."""
        cand = self.candidateOf(f)
        tech, kept = None, cand
        if self.trimmer is not None and not cand.failed and f.nTerms >= self._minTerms:
            try:
                fT = self.trim(f, self.trimmer)
            except Exception as e:
                print(f"SearchLoop.enter: trim failed: {e}")
                fT = f
            if FormulaScorer.keyOf(fT) != FormulaScorer.keyOf(f):
                tech, kept = self.trimmer, self.candidateOf(fT)
        tr = Trace(round_, source, cand, tech, kept)
        if kept.key not in self._population:
            self._population[kept.key] = kept
        self._traceLog.append(tr)
        return tr

    def evaluate(self, props: Sequence[Proposal], source: Source, round_: int) -> int:
        """Scores a batch of proposals, updating the population and the counters; returns the
        number of new (not yet kept) legal candidates."""
        nNew = 0
        self._roundFailed = []
        for p in props:
            self._nProposed += 1
            if isinstance(p, Reject):
                continue
            f = NARX_Formula.of(p)
            if 0 <= self._maxFree < f.nFree:
                self._roundFailed.append((f.formulaStr, f"{f.nFree} free parameters (?), at most {self._maxFree} allowed"))
                continue
            self._nValid += 1
            before = len(self._population)
            tr = self.enter(f, source, round_)
            if len(self._population) > before:
                nNew += 1
                if tr.candTrim.failed:
                    self._nFailed += 1
            if tr.candTrim.failed and tr.candTrim.reason is not None:
                self._roundFailed.append((f.formulaStr, tr.candTrim.reason))
        return nNew

    def run(self, proposer: Proposer, perRound: int, maxRounds: int, patience: int) -> int:
        """
        Runs rounds of propose -> score until the stop rule fires; returns the number of rounds run
        in this call.  The loop also stops when the fit budget (`maxFits` > 0) is spent, and when
        the proposer fails (an LLM that cannot be reached): the failure is kept in `error`, the
        population stays as it is.  After every round the proposer gets the round's feedback.

        Parameters
        ----------
        proposer : Proposer
            The proposal source.
        perRound : int
            The number of proposals requested per round.
        maxRounds : int
            The maximum number of rounds.
        patience : int
            The consecutive rounds without a better best score that stop the loop.
        """
        stagnant = dry = done = 0
        stop = False
        while not stop and done < maxRounds:
            try:
                props = proposer.propose(self.topTraces(), perRound)
            except Exception as e:
                self._error = f"{proposer.source} proposer failed in round {self._nRounds + 1}: {e!r}"
                print(f"SearchLoop.run: {self._error}")
                return done
            nNew = self.evaluate(props, proposer.source, self._nRounds + 1)
            done += 1
            self._nRounds += 1
            b = self.best
            if b is not None:
                if b.score < self._bestSoFar:
                    self._bestSoFar, stagnant = b.score, 0
                else:
                    stagnant += 1
                self._convLog.append(ConvRow(self._nRounds, self._nProposed, b.score, b.smape))
                if self.verbose:
                    print(f"SearchLoop.run: round {self._nRounds:3d}: best = {b.smape:8.3f} (validation {b.smapeVal:8.3f}, "
                          f"score {b.score:8.3f})  new {nNew}  valid {self._nValid}/{self._nProposed}  pop {self.size}  fits {self.fits}")
                proposer.feedback(list(self._roundFailed), self.scorer.notes(b))
            else:
                stagnant += 1
                self._convLog.append(ConvRow(self._nRounds, self._nProposed, -0.0, -0.0))
                proposer.feedback(list(self._roundFailed), "")
            dry = dry + 1 if nNew == 0 else 0
            stop = stagnant >= patience or dry >= 2 or (self.maxFits > 0 and self.fits >= self.maxFits)
        return done
