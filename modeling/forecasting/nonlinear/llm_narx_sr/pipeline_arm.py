"""
PipelineArm: the Searching Arms of LLM-NARX-SR -- one class, driven by an `ArmSpec`

    Loop { proposer -> parse -> fit & score [-> trimmer] -> population + traces }
         -> term library -> [selector] -> best equation  [-> core / seed back to the proposer, next round]

Every searching arm of the paper is this pipeline with the knobs of its `ArmSpec`:  llm-loop = an
LLM proposer, no selector (V1);  llm-cem = an LLM proposer, the union of its proposals as the
library, the cross-entropy method as the selector (V4, the paper's model), with a trimmer inside
the loop (V3), without the EDA block ("blind") or with LLM <-> selector rounds (V5) as ablations;
genetic / random = the population rule / grammar sampling as the proposer, no selector (V8 / V9);
stat-cem = no loop, the grammar expanded in full as the library, the same selector (V6, the library
no LLM wrote).  The seed of every loop is the plain-ARX baseline on the exogenous shortlist (the
`Baseline` anchor's equation), so an arm is credited only with what its search added; the selector
may pick any technique of `SubsetSearch` (`TermLibrary.selectBy`), and a CEM selector's inclusion
probabilities are the stable core an LLM is shown in the next round.  The `llm` factory gives every
LLM run a fresh, counted client (calls, tokens, cost per run); a `CachedClient` inside the factory
makes a run reproducible and free to repeat.

The Python counterpart of ScalaTion's `llm_narx_sr/PipelineArm.scala`.  Difference: the knob
defaults are read from the SCORER's hyper-parameters at run time when not given (ScalaTion reads
the global `FormulaScorer.hp` at construction).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from modeling.feature_selection import SelectionTech
from modeling.llm_client import CountingClient, LLMClient
from modeling.forecasting.nonlinear.narx_formula import NARX_Formula
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import DatasetConfig
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.genetic_proposer import GeneticProposer
from modeling.forecasting.nonlinear.llm_narx_sr.llm_proposer import LLMProposer
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import (ArmSpec, ConvRow, ExpRun, LibraryKind, Method,
                                                                  ProposerKind, SearchArm, Source, Trace)
from modeling.forecasting.nonlinear.llm_narx_sr.search_loop import Proposer, SearchLoop
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import TermLibrary

#: A factory for a fresh LLM client per run.
ClientFactory = Callable[[], LLMClient]


class PipelineArm(SearchArm):
    """
    Runs the search pipeline an `ArmSpec` describes (ScalaTion's `PipelineArm`).

    Parameters
    ----------
    spec : ArmSpec
        The arm: proposer, trimmer, selector, library, rounds, prompt options.
    llm : callable, optional
        The factory for a fresh LLM client per run (required by an LLM proposer).
    maxLag, nFreqs, perRound, maxRounds, patience, topK : int, optional
        The maximum lag in proposals and libraries, the wave frequencies available, the proposals
        requested per round, the maximum loop rounds (LLM calls) per loop, the stagnant rounds that
        stop a loop, the best candidates shown to the proposer; each defaults to the scorer's hp entry.
    maxTerms : int, default=12
        The maximum number of terms per genetic / random proposal.
    maxCols : int, optional
        The maximum number of columns in a sampled CEM set (-1 => none; None => hp["maxCols"] on
        the grammar library, none on a loop's).
    coreThresh : float, default=0.8
        The inclusion probability from which a term counts as stable core.
    prices : (float, float)
        The USD per million prompt / reply tokens for the cost column.
    funcs, powers : sequence, optional
        The functions and exponents applied to every lagged variable of the grammar library.
    nlTerms : bool, optional
        Whether the proposers and the grammar library use the free-parameter forms (defaults to hp["nlTerms"] = 1).
    verbose : bool, default=True
        Whether to print the arm's debug lines.
    """

    def __init__(self, spec: ArmSpec, llm: Optional[ClientFactory] = None, maxLag: Optional[int] = None,
                 nFreqs: Optional[int] = None, perRound: Optional[int] = None, maxRounds: Optional[int] = None,
                 patience: Optional[int] = None, topK: Optional[int] = None, maxTerms: int = 12,
                 maxCols: Optional[int] = None, coreThresh: float = 0.8, prices: Tuple[float, float] = (0.0, 0.0),
                 funcs: Optional[Sequence[str]] = None, powers: Optional[Sequence[float]] = None,
                 nlTerms: Optional[bool] = None, verbose: bool = True) -> None:
        if spec.proposer is ProposerKind.LLM and llm is None:
            raise ValueError(f"PipelineArm: {spec.name} needs an LLM client factory")
        self.spec, self.llm = spec, llm
        self._maxLag, self._nFreqs, self._perRound = maxLag, nFreqs, perRound
        self._maxRounds, self._patience, self._topK = maxRounds, patience, topK
        self.maxTerms, self._maxCols, self.coreThresh, self.prices = int(maxTerms), maxCols, float(coreThresh), prices
        self.funcs = TermLibrary.LIB_FUNCTIONS if funcs is None else list(funcs)
        self.powers = TermLibrary.LIB_POWERS if powers is None else list(powers)
        self._nlTerms = nlTerms
        self.verbose = verbose

    @property
    def method(self) -> Method:
        return self.spec.method

    @property
    def variant(self) -> str:
        return self.spec.variant

    def _knobs(self, scorer: FormulaScorer) -> dict:
        """The knobs, the scorer's hp filling the ones not given."""
        hp = scorer.hp
        g = lambda v, k: int(hp[k]) if v is None else int(v)
        return {"maxLag": g(self._maxLag, "maxLag"), "nFreqs": g(self._nFreqs, "nFreqs"),
                "perRound": g(self._perRound, "perRound"), "maxRounds": g(self._maxRounds, "maxRounds"),
                "patience": g(self._patience, "patience"), "topK": g(self._topK, "topK"),
                "cap": (int(hp["maxCols"]) if self.spec.library is LibraryKind.Grammar else -1) if self._maxCols is None else int(self._maxCols),
                "cemSample": int(hp["cemSample"]), "cemIter": int(hp["cemIter"]),
                "fitLib": int(hp["fitLib"]) == 1, "maxFits": int(hp["maxFits"]),
                "nlTerms": (int(hp["nlTerms"]) == 1) if self._nlTerms is None else bool(self._nlTerms)}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"PipelineArm.run: {msg}")

    def run(self, cfg: DatasetConfig, scorer: FormulaScorer, stream: int) -> ExpRun:
        """Runs the arm once and returns its record (the stream is the random number stream / LLM run tag)."""
        k = self._knobs(scorer)
        eval0 = scorer.evaluations
        exo = cfg.shortlist(scorer)
        freqs = self.waveFreqs(cfg, scorer, k["nFreqs"])
        base = NARX_FormulaParser.parseFormula(cfg.baselineFormulaStr(k["maxLag"], exo), scorer.vars, k["maxLag"], list(freqs))
        if self.spec.proposer is ProposerKind.NoLoop:
            return self._selectOnly(cfg, scorer, stream, exo, freqs, base, eval0, k)
        return self._loopThenSelect(cfg, scorer, stream, exo, freqs, base, eval0, k)

    def _selectOnly(self, cfg: DatasetConfig, scorer: FormulaScorer, stream: int, exo: Sequence[str], freqs: np.ndarray,
                    base: NARX_Formula, eval0: int, k: dict) -> ExpRun:
        """No loop: the grammar expanded into the library and one equation selected from it once (a
        uniform prior for CEM); the selected equation is traced as `Source.Selected`."""
        tech = self.spec.selector
        lib0 = TermLibrary.fromGrammar(cfg.resp, exo, k["maxLag"], self.funcs, self.powers, list(freqs), k["nFreqs"],
                                       nlTerms=k["nlTerms"]).finite(scorer.fullSet)
        lib = lib0.fitted(scorer) if (k["fitLib"] and k["nlTerms"]) else lib0
        self._log(f"{self.name}: grammar library of {lib.size} terms over {', '.join(exo)}" +
                  (f" ({lib0.size - lib.size} degenerate free-parameter terms dropped)" if lib.size < lib0.size else "") +
                  f"; wave frequencies {np.round(freqs, 5).tolist()}")
        sel, probs, nEv = lib.selectBy(tech, scorer, None, stream, k["cap"], k["cemSample"], k["cemIter"])
        self._log(f"{self.name}: {tech} evaluated {nEv} sets => {sel.formulaStr} @ {sel.score}" +
                  (f"; core: {lib.showProbs(probs, 8)}" if probs is not None else ""))
        best = scorer.score(base) if sel.failed else sel
        trace = Trace(1, Source.Selected, sel, tech, sel)
        return self.record(scorer, cfg, stream, best, [best], [trace]).copy(
            nEval=scorer.evaluations - eval0, probs=probs, labels=list(lib.labels))

    def _loopThenSelect(self, cfg: DatasetConfig, scorer: FormulaScorer, stream: int, exo: Sequence[str],
                        freqs: np.ndarray, base: NARX_Formula, eval0: int, k: dict) -> ExpRun:
        """`rounds` times: the loop with the spec's proposer and trimmer, seeded with the baseline (and
        the previous round's selected equation), then, with a selector, the library of the loop's
        proposals and the selection; the answer is the best of the loop's best and the selected equations."""
        spec = self.spec
        isLLM = spec.proposer is ProposerKind.LLM
        profile = self.edaOf(cfg, scorer, cfg.exo, k["maxLag"], k["nFreqs"]) if (isLLM and spec.eda) else None   # the whole pool
        core: List[str] = []                                              # stable-core labels shown to the LLM
        seeds: List[Tuple[NARX_Formula, Source]] = []                     # the previous round's selected equation
        best: Optional[Candidate] = None
        top: List[Candidate] = []
        probs: Optional[np.ndarray] = None
        labels: Optional[List[str]] = None
        nProp = nVal = nFail = calls = 0
        tokens, cost = 0, 0.0
        conv: List[ConvRow] = []
        traces: List[Trace] = []
        error: Optional[str] = None
        freqList = list(freqs)

        for r in range(1, spec.rounds + 1):
            if error is not None:
                break
            client = CountingClient(self.llm(), self.prices[0], self.prices[1]) if isLLM else None
            if spec.proposer is ProposerKind.LLM:
                proposer: Proposer = LLMProposer(client, cfg, scorer.vars, k["maxLag"], freqList, profile, spec.named,
                                                 tag=f"[run {stream}] [{self.name}]", nonlinear=k["nlTerms"],
                                                 maxFree=int(scorer.hp["maxFree"]))
                if core:
                    proposer.setCore(core)
            elif spec.proposer is ProposerKind.Genetic:
                proposer = GeneticProposer(cfg.resp, exo, k["maxLag"], freqList, stream, self.maxTerms,
                                           nonlinear=k["nlTerms"], maxFree=int(scorer.hp["maxFree"]))
            elif spec.proposer is ProposerKind.Random:
                proposer = GeneticProposer(cfg.resp, exo, k["maxLag"], freqList, stream, self.maxTerms, random=True,
                                           nonlinear=k["nlTerms"], maxFree=int(scorer.hp["maxFree"]))
            else:
                raise RuntimeError("PipelineArm: a loop needs a proposer")
            budget = 0 if k["maxFits"] <= 0 else max(1, k["maxFits"] - (scorer.evaluations - eval0))   # fits left for this loop
            loop = SearchLoop(scorer, k["topK"], spec.trimmer, stream, budget, verbose=self.verbose)
            loop.seed((base, Source.Seed), *seeds)
            loop.run(proposer, k["perRound"], k["maxRounds"], k["patience"])
            if loop.error is not None:
                error = loop.error                                        # the arm goes on with what it has
            nProp += loop.nProposed
            nVal += loop.nValid
            nFail += loop.nFailed
            if client is not None:
                calls += client.calls
                tokens += client.tokens
                cost += client.cost
            conv += loop.convergence
            traces += loop.traces
            loopBest = loop.best

            if spec.selector is None:                                     # the loop's best is the answer
                if loopBest is not None and (best is None or loopBest.score < best.score):
                    best = loopBest
                top = loop.top(k["topK"])
            else:
                tech = spec.selector
                cands = loop.top(k["topK"]) if spec.library is LibraryKind.TopK else [c for c in loop.all if not c.failed]
                if not cands:
                    self._log(f"{self.name} round {r}: no fitted candidate -- nothing to select from")
                    continue
                lib = TermLibrary.from_(cands).finite(scorer.fullSet)
                prior = TermLibrary.frequencyPrior(loop.top(k["topK"]), lib)
                self._log(f"{self.name} round {r}: library of {lib.size} terms from {len(cands)} candidates; prior {lib.showProbs(prior, 8)}")
                try:                                                      # the selection stage, guarded
                    c, p, nEv = lib.selectBy(tech, scorer, prior, stream, k["cap"], k["cemSample"], k["cemIter"])
                    self._log(f"{self.name} round {r}: {tech} evaluated {nEv} sets => {c.formulaStr} @ {c.score}")
                    probs, labels = p, list(lib.labels)
                    core = [] if p is None else [lib.labels[j] for j in range(lib.size) if p[j] >= self.coreThresh]
                    sel = c
                except Exception as e:
                    self._log(f"{self.name} round {r}: selection failed ({e}) -- keeping the loop's best")
                    sel = loopBest if loopBest is not None else cands[0]
                traces.append(Trace(loop.rounds + 1, Source.Selected, sel, tech, sel))
                if not sel.failed and (loopBest is None or sel.score <= loopBest.score):
                    roundBest = sel
                else:
                    roundBest = loopBest if loopBest is not None else sel
                if best is None or roundBest.score < best.score:
                    best = roundBest
                seeds = [(roundBest.formula, Source.Selected)] if roundBest.formula is not None else []
                pool = [sel] + loop.top(k["topK"])
                seen: set = set()
                top = []
                for c in sorted([c for c in pool if not c.failed], key=lambda c: c.score):
                    if c.key not in seen:
                        seen.add(c.key)
                        top.append(c)
                top = top[: k["topK"]]

        if best is None:
            best = scorer.score(base)                                     # nothing fitted at all: the baseline
        return self.record(scorer, cfg, stream, best, top, traces).copy(
            nProposed=nProp, nValid=nVal, nFailed=nFail, nEval=scorer.evaluations - eval0,
            llmCalls=calls, tokens=tokens, cost=cost, conv=conv, probs=probs, labels=labels, error=error)
