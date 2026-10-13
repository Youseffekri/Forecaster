"""
SearchArm: the Search Arms of LLM-NARX-SR -- the Contract and the Records (`Trace`, `ExpRun`)

Every version of the method compared in the paper is a `SearchArm`: one `run (cfg, scorer, stream)`
call, one `ExpRun` record.  The anchors are `BaselineArm`; every searching arm is a `PipelineArm`
driven by an `ArmSpec` -- who proposes (`ProposerKind`), the trimmer and the selector (the base's
`SelectionTech`), the library the selector works on (`LibraryKind`), the rounds and the prompt
options -- so the arm table of the paper is a table of specs (`LLM_NARX_SR.mainArms`,
`ablationArms`).  The harness only loops over arms x streams x folds and tabulates the records;
anything computed FROM the records (tests, tables) lives in modules that depend on this one, never
the other way round.  What an experiment runs ON is `DatasetConfig` in `datasets.py`.  Names that
select behaviour are enums (`Method`, `Source`, `ProposerKind`, `LibraryKind`, `SelectionTech`,
`ExoMode`), so a misspelt one is an error; a spec's `method` and row label are derived from it.
A `Trace` is the EVENT "in round r, source s proposed f, trimmer t kept fTrim"; a `Candidate` is
the scored unit the population holds; several traces may lead to one candidate (a formula proposed
twice, two proposals trimmed to the same equation), never fewer than one.

The Python counterpart of ScalaTion's `llm_narx_sr/SearchArm.scala` (rev4.0).  Differences: the
QoF matrices are lists of per-horizon dicts (`qofTest`), an `Option` is None, a `MatrixD.fill
(..., -0.0)` of a failed run is an empty list; `ArmSpec.copy (...)` is `ArmSpec.copy (**changes)`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.2
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from Util.tools import banner
from modeling.feature_selection import SelectionTech
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import DatasetConfig
from modeling.forecasting.nonlinear.llm_narx_sr.eda4ts import EDA4TS, topFreqs
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer, qof_table_full


class Method(Enum):
    """The arms (the method column of the results table), each with its display tag."""
    Baseline = "baseline"           # V0: plain ARX, p = px = maxLag
    TunedARX = "tuned-arx"          # V0: ARX grid over p x px
    SelARX = "sel-arx"              # V0: the baseline's lags under the base's feature selection
    RandomWalk = "random-walk"      # V0: the base's `RandomWalk`
    Random = "random"               # V9: grammar sampling through the loop
    Genetic = "genetic"             # V8: crossover / mutation through the loop
    StatCEM = "stat-cem"            # V6: grammar library + selector, no loop, no LLM
    LLMLoop = "llm-loop"            # V1: LLM proposes, no selection stage
    LLMCEM = "llm-cem"              # V4: LLM loop -> union library -> selector

    @property
    def tag(self) -> str:
        """The display tag."""
        return self.value

    def __str__(self) -> str:
        return self.name


#: The classical anchors.
ANCHORS = {Method.Baseline, Method.TunedARX, Method.SelARX, Method.RandomWalk}
Method.anchors = ANCHORS


class Source(Enum):
    """Who put a formula into a population (`Selected`: the selection stage's equation)."""
    Seed = 0
    LLM = 1
    Genetic = 2
    Random = 3
    Selected = 4

    def __str__(self) -> str:
        return self.name


class ProposerKind(Enum):
    """Who proposes structures inside the search loop: an LLM, the genetic rule, random grammar
    sampling -- or `NoLoop`: no proposals at all, the selector works on the grammar library once."""
    LLM = 0
    Genetic = 1
    Random = 2
    NoLoop = 3

    def __str__(self) -> str:
        return self.name


class LibraryKind(Enum):
    """The term library the selector works on: the union of every fitted candidate of the loop, the
    union of its top-K only, or the grammar expanded in full (`TermLibrary.fromGrammar`)."""
    Union = 0
    TopK = 1
    Grammar = 2

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ArmSpec:
    """
    One searching arm of the pipeline

        Loop { proposer -> parse -> fit & score [-> trimmer] -> population } -> library -> [selector]

    as a choice of its knobs; `PipelineArm` runs it.  The `method` (the row of the results table)
    and the `variant` label are DERIVED from the knobs.

    Attributes
    ----------
    proposer : ProposerKind
        Who proposes inside the loop (`NoLoop` => no loop, the grammar library is selected from once).
    trimmer : SelectionTech, optional
        The selection technique trimming every proposal inside the loop (None => off).
    selector : SelectionTech, optional
        The selection technique over the library after the loop (None => the loop's best is the answer).
    library : LibraryKind
        The library the selector works on (`Grammar` iff `NoLoop`).
    rounds : int
        The number of loop -> selector alternations (> 1 => the selected equation seeds the next
        loop and, for an LLM, the stable core is shown to it).
    eda : bool
        Whether the EDA profile of the training rows is in the LLM prompt.
    named : bool
        Whether the LLM prompt shows the real names and the domain sentence.
    tag : str
        An extra label appended to the variant (e.g., "ft" for a fine-tuned proposer).
    """
    proposer: ProposerKind
    trimmer: Optional[SelectionTech] = None
    selector: Optional[SelectionTech] = None
    library: LibraryKind = LibraryKind.Union
    rounds: int = 1
    eda: bool = True
    named: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("ArmSpec: rounds must be at least 1")
        if (self.proposer is ProposerKind.NoLoop) != (self.library is LibraryKind.Grammar):
            raise ValueError("ArmSpec: the grammar library is selected from without a loop, and a loop's library is its proposals")
        if self.proposer is ProposerKind.NoLoop and not (self.selector is not None and self.trimmer is None and self.rounds == 1):
            raise ValueError("ArmSpec: without a loop there is only the selector, once")
        if self.rounds > 1 and self.selector is None:
            raise ValueError("ArmSpec: rounds > 1 need a selector to seed the next loop")

    @property
    def method(self) -> Method:
        """The method (the row of the results table) the knobs amount to."""
        if self.proposer is ProposerKind.LLM:
            return Method.LLMCEM if self.selector is not None else Method.LLMLoop
        if self.proposer is ProposerKind.Genetic:
            return Method.Genetic
        if self.proposer is ProposerKind.Random:
            return Method.Random
        return Method.StatCEM

    @property
    def variant(self) -> str:
        """The variant label: every knob that departs from the method's standard version, in a fixed
        order ("" for the standard version), e.g., "blind", "trim", "trim-backward", "sel-backward",
        "2r", "blind-trim"; a CEM selector is part of the LLMCEM and StatCEM methods, so it is named
        ("sel-cem") only on a genetic or random arm."""
        parts: List[str] = []
        if not self.eda:
            parts.append("blind")
        if self.named:
            parts.append("named")
        if self.trimmer is not None:
            parts.append("trim" if self.trimmer is SelectionTech.CEM else f"trim-{self.trimmer.name.lower()}")
        if self.selector is not None:
            if self.selector is not SelectionTech.CEM or self.method in (Method.Genetic, Method.Random):
                parts.append(f"sel-{self.selector.name.lower()}")
        if self.library is LibraryKind.TopK:
            parts.append("topk")
        if self.rounds > 1:
            parts.append(f"{self.rounds}r")
        if self.tag:
            parts.append(self.tag)
        return "-".join(parts)

    @property
    def name(self) -> str:
        """The row label: the method tag with the variant appended."""
        return self.method.tag if not self.variant else f"{self.method.tag}-{self.variant}"

    def copy(self, **changes) -> "ArmSpec":
        """A copy with the given knobs changed (Scala's `copy`), e.g., `ArmSpec.llmCEM.copy (eda = False)`."""
        return replace(self, **changes)


#: The standard specs of the paper's arms; ablations are copies of them.
ArmSpec.llmLoop = ArmSpec(ProposerKind.LLM)                                                   # V1
ArmSpec.llmCEM = ArmSpec(ProposerKind.LLM, selector=SelectionTech.CEM)                       # V4 (the paper's model)
ArmSpec.statCEM = ArmSpec(ProposerKind.NoLoop, selector=SelectionTech.CEM, library=LibraryKind.Grammar)   # V6
ArmSpec.genetic = ArmSpec(ProposerKind.Genetic)                                              # V8
ArmSpec.random = ArmSpec(ProposerKind.Random)                                                # V9


@dataclass
class ConvRow:
    """
    One row of a search's convergence log.

    Attributes
    ----------
    round : int
        The round number (1-based).
    nProposed : int
        The cumulative number of proposals made so far.
    bestScore : float
        The best-so-far selection score.
    bestSmape : float
        The best-so-far mean in-sample sMAPE.
    """
    round: int
    nProposed: int
    bestScore: float
    bestSmape: float


@dataclass
class Trace:
    """
    One proposal and what selection made of it.

    Attributes
    ----------
    round : int
        The loop round the proposal was made in (0 for a seed).
    source : Source
        Who proposed f.
    cand : Candidate
        f as proposed, scored.
    trimmer : SelectionTech, optional
        The selection technique that trimmed f (None => full version).
    candTrim : Candidate
        The kept equation fTrim, scored (the same object as cand when untrimmed).
    """
    round: int
    source: Source
    cand: Candidate
    trimmer: Optional[SelectionTech]
    candTrim: Candidate

    @property
    def f(self):
        """The formula as proposed."""
        return self.cand.formula

    @property
    def fTrim(self):
        """The formula as kept."""
        return self.candTrim.formula

    @property
    def fScore(self) -> float:
        return self.cand.score

    @property
    def fTrimScore(self) -> float:
        return self.candTrim.score

    @property
    def trimmed(self) -> bool:
        """Whether selection changed f."""
        return self.trimmer is not None and self.candTrim is not self.cand

    @property
    def dropped(self) -> list:
        """The terms removed (by label)."""
        if not self.trimmed or self.f is None or self.fTrim is None:
            return []
        kept = {t.label() for t in self.fTrim.preds}
        return [t for t in self.f.preds if t.label() not in kept]

    @property
    def gain(self) -> float:
        """How much trimming helped (never < 0, see `SearchLoop.trim`)."""
        return self.fScore - self.fTrimScore

    @property
    def best(self) -> Candidate:
        return self.candTrim if self.fTrimScore <= self.fScore else self.cand

    @property
    def key(self) -> str:
        """The population identity."""
        return self.candTrim.key

    def __str__(self) -> str:
        tr = self.trimmer.name if self.trimmer is not None else "none"
        s = f"{self.round:3d}  {self.source.name:<8s}  {tr:<8s}  {self.fScore:8.3f} -> {self.fTrimScore:8.3f}  {self.cand.formulaStr}"
        if self.trimmed:
            s += f"  ->  {self.candTrim.formulaStr}  (dropped: {', '.join(t.label() for t in self.dropped)})"
        return s


@dataclass
class ExpRun:
    """
    The outcome of one arm run on one dataset (ScalaTion's `ExpRun`).

    Attributes
    ----------
    method : Method
        The arm.
    variant : str
        The ablation tag of the arm ("" for the standard version).
    stream : int
        The random number stream of the run (-1 for a deterministic arm).
    best : Candidate
        The best candidate (formula or base model, in-sample QoF).
    qofTest : list of dict
        The QoF per horizon of the best model on the testing rows (`rollValidate` forecasts,
        `FormulaScorer.tnt`); empty when the run failed.
    nProposed, nValid, nFailed, nEval : int
        The proposals made (seeds excluded), those that parsed and were legal, the legal
        candidates whose fit failed, the model fits made by the run.
    llmCalls, tokens, cost : int, int, float
        The LLM consultations consumed (0 => no LLM), the estimated tokens, the estimated USD cost.
    conv : list of ConvRow
        The convergence log (empty for a run without rounds).
    top : list of Candidate
        The best equations at the end (best first).
    traces : list of Trace
        Every proposal of the run in order (empty for an anchor).
    exoUsed : list of str
        The exogenous variables the best equation uses.
    probs : np.ndarray, optional
        The inclusion probability per library term left by a CEM stage (None when the arm has none).
    labels : list of str, optional
        The library term labels the probabilities refer to (None when none).
    error : str, optional
        What cut the run short or made it fail (an unreachable LLM, an exception; None when nothing did).
    """
    method: Method
    variant: str
    stream: int
    best: Candidate
    qofTest: List[Dict[str, float]]
    nProposed: int = 0
    nValid: int = 0
    nFailed: int = 0
    nEval: int = 0
    llmCalls: int = 0
    tokens: int = 0
    cost: float = 0.0
    conv: List[ConvRow] = field(default_factory=list)
    top: List[Candidate] = field(default_factory=list)
    traces: List[Trace] = field(default_factory=list)
    exoUsed: List[str] = field(default_factory=list)
    probs: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None
    error: Optional[str] = None

    @property
    def name(self) -> str:
        """The row label: the method tag with the variant appended."""
        return self.method.tag if not self.variant else f"{self.method.tag}-{self.variant}"

    @property
    def validity(self) -> float:
        """The share of proposals that parsed and were legal (-0.0 if none were made)."""
        return self.nValid / self.nProposed if self.nProposed > 0 else -0.0

    @property
    def smapeTrain(self) -> np.ndarray:
        """The in-sample sMAPE per horizon of the best model (the fit rows)."""
        return self.best.smapeH

    @property
    def valSmape(self) -> float:
        """The mean validation sMAPE over horizons of the best model."""
        return self.best.smapeVal

    @property
    def failed(self) -> bool:
        """Whether the run has no fitted result (its best failed or the arm threw)."""
        return self.best.failed

    @property
    def noImprovement(self) -> bool:
        """Whether a searching run's best is still a seed (nothing proposed beat it)."""
        o = self.originOf(self.best)
        return self.nProposed > 0 and o is not None and o.round == 0 and o.source is Source.Seed

    @property
    def smapeTest(self) -> np.ndarray:
        """The TnT sMAPE per horizon of the best model (testing rows, rolling); NaN when failed."""
        if not self.qofTest:
            return np.full(len(self.best.qofTrain) if self.best.qofTrain else 1, math.nan)
        return np.array([q["smape"] for q in self.qofTest])

    @property
    def testSmape(self) -> float:
        """The mean TnT sMAPE over horizons (the number a comparison uses)."""
        return float(np.mean(self.smapeTest))

    def showQoF(self) -> None:
        """Prints the QoF of the best model the way the base prints them after `diagnoseAll`
        ("fitMap QoF = " then `Fit.showFitMap`, one column per horizon): in-sample on the fit rows,
        then TnT on the testing rows."""
        banner(f"{self.name} (stream {self.stream}): {self.best.formulaStr.replace(chr(10), ' ')}")
        print("In-Sample fitMap QoF = ")
        print(qof_table_full(self.best.qofTrain) if self.best.qofTrain else "(failed)")
        print("TnT fitMap QoF = ")
        print(qof_table_full(self.qofTest) if self.qofTest else "(failed)")

    def tracesOf(self, cand: Candidate) -> List[Trace]:
        """Every trace that led to a candidate (as f or as fTrim), in order."""
        return [t for t in self.traces if t.cand.key == cand.key or t.candTrim.key == cand.key]

    def originOf(self, cand: Candidate) -> Optional[Trace]:
        """The first trace that led to a candidate: when and by whom it appeared."""
        ts = self.tracesOf(cand)
        return ts[0] if ts else None

    @property
    def equations(self) -> str:
        """The run's best equations with their scores, one per line, best first."""
        return "\n".join(f"{c.score:8.3f}  {c.smape:8.3f}  {c.nTerms:2d}  {c.formulaStr}" for c in self.top)

    def copy(self, **changes) -> "ExpRun":
        """A copy with the given fields changed (Scala's `copy`)."""
        return replace(self, **changes)


class SearchArm:
    """The contract every version of the method fulfils (ScalaTion's `SearchArm` trait)."""

    @property
    def method(self) -> Method:
        """The arm's method (the method column of the results table)."""
        raise NotImplementedError

    @property
    def variant(self) -> str:
        """The arm's ablation tag ("" for the standard version)."""
        return ""

    @property
    def name(self) -> str:
        """The arm's row label (method tag plus variant)."""
        return self.method.tag if not self.variant else f"{self.method.tag}-{self.variant}"

    @property
    def deterministic(self) -> bool:
        """Whether the arm is deterministic (run once, stream ignored)."""
        return False

    def run(self, cfg: DatasetConfig, scorer: FormulaScorer, stream: int) -> ExpRun:
        """
        Runs the arm once and returns its record.

        Parameters
        ----------
        cfg : DatasetConfig
            The dataset configuration.
        scorer : FormulaScorer
            The fit-and-score engine of the dataset (its training rows are the only rows the arm may learn from).
        stream : int
            The random number stream of this run.
        """
        raise NotImplementedError

    @staticmethod
    def waveFreqs(cfg: DatasetConfig, scorer: FormulaScorer, k: int) -> np.ndarray:
        """The wave frequencies from the periodogram of the TRAINING rows of the response (the only
        frequencies a proposal may use)."""
        return topFreqs(scorer.trainSet[cfg.resp], k)

    @staticmethod
    def edaOf(cfg: DatasetConfig, scorer: FormulaScorer, exo: Sequence[str], maxLag: int, nFreqs: int) -> EDA4TS:
        """The EDA profile of the training rows for the given exogenous variables (the shortlist or the pool)."""
        data = scorer.fullSet
        return EDA4TS(data[cfg.resp], {v: data[v] for v in exo}, scorer.trSize, maxLag, nFreqs, cfg.resp)

    @staticmethod
    def exoOf(cfg: DatasetConfig, cand: Candidate) -> List[str]:
        """The exogenous variables a candidate's formula uses (none for a base model)."""
        if cand.formula is None:
            return []
        return sorted(cand.formula.vars - {cfg.resp})

    def record(self, scorer: FormulaScorer, cfg: DatasetConfig, stream: int, best: Candidate,
               top: Sequence[Candidate] = (), traces: Sequence[Trace] = ()) -> ExpRun:
        """Wraps a best candidate as a record with its Train-and-Test QoF attached (the base's
        `rollValidate` on the testing rows, `FormulaScorer.tnt`)."""
        if best.failed:
            qofTest: List[Dict[str, float]] = []
        elif best.formula is not None:
            qofTest = scorer.tnt(best.formula.fresh())[1]
        else:
            qofTest = scorer.tnt(best.model)
        return ExpRun(self.method, self.variant, stream, best, qofTest, top=list(top), traces=list(traces),
                      exoUsed=self.exoOf(cfg, best))
