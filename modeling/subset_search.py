"""
SubsetSearch Module: Subset Search Algorithms for Feature Selection, written once

The Python counterpart of ScalaTion's `modeling/SubsetSearch.scala`.  The greedy feature-selection
techniques (`Forward`, `Backward`, `Stepwise`, `Beam`) and the rank-driven one (`Ranked`) as pure
algorithms over an EVALUATOR, in the way `CEM_Selector` is written: the algorithm decides which
column sets to try, the host (a regression, a forecaster at one horizon or jointly over horizons,
a formula term library) supplies `eval: cols -> BestStep`, which builds, trains and scores one
column set and returns a `BestStep` whose qof holds the criterion qk (qof None => the fit failed
and the set is skipped).  Nothing here prints, builds a model or knows about rows, spaces or
horizons -- all of that is the evaluator's, so the contract "every QoF compared is on the original
scale, on the selection rows" is met in ONE place per host.  `run` dispatches on `SelectionTech`;
`cem` wraps `CEM_Selector`.  Every search returns a `Path`: the best-so-far after each step (the
trace hosts turn into their key-metrics table), the best step found and the number of sets
evaluated.

`ranked` (recursive feature elimination) takes a second host callback, `rank: cols -> importance per
column` (or one fixed importance vector): the ranker decides which column leaves at each step and
the evaluator only where to stop, n evaluations instead of n(n+1)/2.  The hosts' rankers are the
standardized coefficients and the permutation importance of the sub-model on the selection rows
(`Forecaster_Reg.coefImportance`, `permImportance`); any other importance (an attention score, a
gradient x input score, a CEM inclusion probability) is passed the same way.

Conventions (shared by all techniques):
    fixed   columns in every set (e.g., the intercept column 0; empty for a `fromTerms` matrix
            or a term library)
    units   columns that enter or leave together (a sin/cos wave pair); defaults to singletons
    ties    the first candidate found wins (as in `Predictor.forwardSel`)

Author: Yousef Fekri Dabanloo
Date: 2026-10-15
Version: 1.1
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import isnan
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from modeling.cem_selector import CEM_Selector
from modeling.feature_selection import BestStep, Cols, SelectionTech, given_qk, slack_base
from modeling.fit import MAXI, extreme

#: The evaluator of a column set: builds, trains and scores the model and returns its `BestStep`.
Evaluator = Callable[[Cols], Optional[BestStep]]

#: The grouping of columns into units: column j -> the set of columns that move with it.
Units = Callable[[int], Set[int]]

#: The ranker of a column set (`ranked`): the current columns -> an importance per column, a length-n
#: vector of which only the entries of the columns in the set are read (higher = more important;
#: NaN = least).  A static ranking (one vector for every set) is passed as an array instead.
Ranker = Callable[[Cols], np.ndarray]


def _singleton(j: int) -> Set[int]:
    return {j}


class SubsetSearch:
    """
    The subset search algorithms behind the feature-selection techniques (ScalaTion's
    `SubsetSearch` object): `forward`, `backward`, `ranked`, `stepwise`, `beam`, `cem` and the
    dispatcher `run`.
    """

    #: The move set of the beam search: (columns added, columns removed) per move.
    BEAM_MOVES: Sequence[Tuple[int, int]] = ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 1))

    @dataclass
    class Path:
        """
        The result of a search: the best-so-far after each step (the trace), the best step found,
        the number of column sets evaluated and, for the cross-entropy method, the final inclusion
        probabilities (None otherwise).

        Parameters
        ----------
        steps : list of BestStep
            The best-so-far after each step, in order.
        best : BestStep
            The best step found (qof None when nothing could be evaluated).
        nEval : int
            The number of column sets evaluated.
        probs : np.ndarray, optional
            The inclusion probability per column (CEM only).
        """
        steps: List[BestStep] = field(default_factory=list)
        best: BestStep = field(default_factory=BestStep)
        nEval: int = 0
        probs: Optional[np.ndarray] = None

        @property
        def cols(self) -> Cols:
            """The columns of the best step (empty when nothing was found)."""
            return [] if self.best.mod_cols is None else list(self.best.mod_cols)

        @property
        def found(self) -> bool:
            """Whether a set could be evaluated."""
            return self.best.qof is not None

    # -- helpers ------------------------------------------------------------------------------

    @staticmethod
    def pick(cur: Optional[BestStep], cand: Optional[BestStep], qk: Optional[int] = None) -> BestStep:
        """
        The better of two steps, tolerating failed (qof None) steps: a failed candidate never
        replaces the current one, and any fitted candidate replaces a failed current one.  Ties
        keep the current step.
        """
        qk = given_qk(qk)
        if cand is None or cand.qof is None:
            return cur
        if cur is None or cur.qof is None:
            return cand
        return cand if cand.gt(float(cur.qof[qk])) else cur

    @staticmethod
    def _step(eval: Evaluator, j: int, cols: Cols, qk: int) -> BestStep:
        """Evaluates one column set and returns the step with its column index and set recorded."""
        s = eval(cols)
        if s is None or s.qof is None:
            return BestStep(j, None, None, cols, qk)
        return BestStep(j, s.qof, s.mod, cols, qk, float(s.qof[qk]))

    @staticmethod
    def _freeUnits(n: int, fixed: Cols, units: Units) -> List[Set[int]]:
        """The free columns of a set (those not fixed), one per unit, in column order."""
        seen: Set[int] = set()
        out: List[Set[int]] = []
        for j in range(n):
            if j not in fixed and j not in seen:
                u = {k for k in units(j) if k not in fixed}
                seen |= u
                out.append(u)
        return out

    @staticmethod
    def _plus(cols: Cols, u: Set[int]) -> Cols:
        """A copy of a column set with a unit added (in column order)."""
        out = list(cols)
        for j in sorted(u):
            if j not in out:
                out.append(j)
        return out

    @staticmethod
    def _minus(cols: Cols, u: Set[int]) -> Cols:
        """A copy of a column set with a unit removed."""
        return [j for j in cols if j not in u]

    # -- T H E   T E C H N I Q U E S ------------------------------------------------------------

    @classmethod
    def forward(cls, n: int, eval: Evaluator, fixed: Optional[Sequence[int]] = None,
                units: Units = _singleton, maxCols: int = -1, qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        FORWARD SELECTION: starts from the fixed columns and at each step adds the unit whose
        addition gives the best criterion; goes on until every column is in (or `maxCols` is
        reached) and returns the best set seen along the path (it does not stop at the first step
        that fails to improve).  The first step of the trace is the fixed set itself when it is
        non-empty (e.g., the intercept-only model).

        Parameters
        ----------
        n : int
            The number of columns 0 until n.
        eval : callable
            The evaluator of a column set.
        fixed : sequence of int, optional
            The columns in every set (defaults to the intercept column [0]; pass [] for none).
        units : callable, default=singletons
            The grouping of columns into units.
        maxCols : int, default=-1
            The largest set allowed (-1 => no cap).
        qk : int, optional
            The index of the QoF measure to compare on.
        """
        qk = given_qk(qk)
        fixed = [0] if fixed is None else list(fixed)
        cap = n if maxCols < 0 else maxCols
        steps: List[BestStep] = []
        nEval = 0
        cols = list(fixed)
        if not cols:
            best = BestStep(qk=qk)
        else:
            nEval += 1
            best = cls._step(eval, -1, cols, qk)
        if best.qof is not None:
            steps.append(best)
        free = cls._freeUnits(n, fixed, units)
        go = len(cols) < cap
        while go:
            cand = BestStep(qk=qk)
            for u in free:
                if all(j not in cols for j in u):
                    cj = cls._plus(cols, u)
                    if len(cj) <= cap:
                        nEval += 1
                        cand = cls.pick(cand, cls._step(eval, min(u), cj, qk), qk)
            if cand.qof is None:
                go = False                                                  # nothing to add (or every addition failed)
            else:
                cols = cand.mod_cols
                steps.append(cand)
                best = cls.pick(best, cand, qk)
                go = len(cols) < cap and len(cols) < n
        return cls.Path(steps, best, nEval)

    @classmethod
    def backward(cls, n: int, eval: Evaluator, fixed: Optional[Sequence[int]] = None,
                 units: Units = _singleton, minCols: int = 1, qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        BACKWARD ELIMINATION: starts from every column and at each step removes the unit whose
        removal gives the best criterion; goes on down to `minCols` columns (never below the fixed
        ones) and returns the best set seen along the path.  The first step of the trace is the
        full set.

        Parameters
        ----------
        n, eval, fixed, units, qk
            As in `forward`.
        minCols : int, default=1
            The smallest set kept.
        """
        qk = given_qk(qk)
        fixed = [0] if fixed is None else list(fixed)
        floor = max(minCols, len(fixed))
        steps: List[BestStep] = []
        nEval = 1
        cols = list(range(n))
        best = cls._step(eval, -1, cols, qk)
        if best.qof is not None:
            steps.append(best)
        free = cls._freeUnits(n, fixed, units)
        go = len(cols) > floor
        while go:
            cand = BestStep(qk=qk)
            for u in free:
                if all(j in cols for j in u):
                    cj = cls._minus(cols, u)
                    if len(cj) >= floor:
                        nEval += 1
                        cand = cls.pick(cand, cls._step(eval, min(u), cj, qk), qk)
            if cand.qof is None:
                go = False                                                  # nothing removable (or every removal failed)
            else:
                cols = cand.mod_cols
                steps.append(cand)
                best = cls.pick(best, cand, qk)
                go = len(cols) > floor
        return cls.Path(steps, best, nEval)

    @classmethod
    def ranked(cls, n: int, eval: Evaluator, rank: Union[Ranker, Sequence[float], np.ndarray],
               fixed: Optional[Sequence[int]] = None, units: Units = _singleton, minCols: int = 1,
               qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        RANKED ELIMINATION (recursive feature elimination, Guyon et al. 2002): starts from every
        column and at each step removes the unit of LOWEST RANK among the remaining columns -- the
        ranker decides which column leaves, the criterion only where to stop -- down to `minCols`
        columns (never below the fixed ones), and returns the best set seen along the path.  The
        first step of the trace is the full set.  One evaluation per step, n in all, against the
        n(n+1)/2 of `backward`: the ranker, not the evaluator, does the elimination work, so the
        path is only as good as the ranking (a ranking must be computed on the selection rows,
        never on the testing rows, and averaged over seeds when it is stochastic).

        Parameters
        ----------
        n, eval, fixed, units, qk
            As in `forward`.
        rank : callable or array
            The importance per column (length n; higher = more important, NaN = least): an array
            is ONE ranking of the full set (a fixed order, e.g., an attention ranking of the full
            library); a callable of the current column set is re-evaluated after every removal
            (true RFE, e.g., the standardized coefficients of the sub-model re-fit on the remaining
            columns).  A unit ranks as its best member.
        minCols : int, default=1
            The smallest set kept.
        """
        qk = given_qk(qk)
        fixed = [0] if fixed is None else list(fixed)
        floor = max(minCols, len(fixed))
        if callable(rank):
            score = rank
        else:
            fixed_r = np.asarray(rank, dtype=float)
            if fixed_r.shape != (n,):
                raise ValueError(f"SubsetSearch.ranked: rank must have one entry per column ({n}), got {fixed_r.shape}")
            score = lambda _cols: fixed_r                                  # noqa: E731  -- one ranking for every set
        steps: List[BestStep] = []
        nEval = 1
        cols = list(range(n))
        best = cls._step(eval, -1, cols, qk)
        if best.qof is not None:
            steps.append(best)
        free = cls._freeUnits(n, fixed, units)
        while len(cols) > floor:
            r = np.nan_to_num(np.asarray(score(cols), dtype=float), nan=-np.inf)
            left = [u for u in free if all(j in cols for j in u) and len(cols) - len(u) >= floor]
            if not left:
                break                                                       # only fixed columns remain (or a unit would cross the floor)
            drop = min(left, key=lambda u: (max(r[j] for j in u), min(u)))  # lowest rank; ties: the lowest column
            cols = cls._minus(cols, drop)
            nEval += 1
            s = cls._step(eval, min(drop), cols, qk)
            if s.qof is not None:
                steps.append(s)
                best = cls.pick(best, s, qk)
        return cls.Path(steps, best, nEval)

    @classmethod
    def stepwise(cls, n: int, eval: Evaluator, fixed: Optional[Sequence[int]] = None,
                 units: Units = _singleton, swap: bool = True, slack: float = float("nan"),
                 qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        STEPWISE SELECTION: at each step compares the best forward move with the best backward
        move and takes the forward one when it is at least as good (within `slack`, which favours
        additions early on: slack / l^2 at step l) and improves on the last accepted criterion
        (within `slack`), else the backward one when it improves, else -- when `swap` -- the swap
        of the two columns when that improves; stops when no move improves, or when the addition
        would re-add the unit just removed (the search has converged and would otherwise cycle).
        The rule and the slack are those of ScalaTion's `Predictor.stepwiseSelAll`.

        Parameters
        ----------
        n, eval, fixed, units, qk
            As in `forward`.
        swap : bool, default=True
            Whether to allow a swap step (one column out, one in).
        slack : float, default=nan
            The slack base (positive when qk is maximized, negative when minimized; nan => the
            magnitude of the package's `slack_base` in the direction of qk, i.e., 25 or -25).
        """
        qk = given_qk(qk)
        fixed = [0] if fixed is None else list(fixed)
        if not isnan(slack):
            slack_ = slack
        else:
            slack_ = abs(slack_base) if qk in MAXI else -abs(slack_base)
        steps: List[BestStep] = []
        nEval = 0
        cols = list(fixed)
        best = BestStep(qk=qk)
        last_q = extreme(qk)                                                # the last accepted criterion
        prev = (0, -1)                                                      # the previous move: (+1 added / -1 removed, its unit)
        go = True
        l = 1
        units_ = cls._freeUnits(n, fixed, units)
        while go and l < n:
            bestf = BestStep(qk=qk)                                         # best forward move
            bestb = BestStep(qk=qk)                                         # best backward move
            for u in units_:
                if all(j not in cols for j in u):
                    nEval += 1
                    bestf = cls.pick(bestf, cls._step(eval, min(u), cls._plus(cols, u), qk), qk)
            for u in units_:
                if all(j in cols for j in u) and len(cols) - len(u) >= max(1, len(fixed)):
                    nEval += 1
                    bestb = cls.pick(bestb, cls._step(eval, min(u), cls._minus(cols, u), qk), qk)
            sl = slack_ / (l * l)                                           # slack shrinks with the step
            fOk = bestf.qof is not None
            bOk = bestb.qof is not None

            if fOk and (not bOk or bestf.ge(float(bestb.qof[qk]) - sl)) and bestf.gt(last_q - sl):
                if prev == (-1, bestf.col):
                    go = False                                              # would re-add the unit just removed: converged
                else:
                    cols = bestf.mod_cols                                   # ADD the forward unit
                    last_q = float(bestf.qof[qk])
                    steps.append(bestf)
                    best = cls.pick(best, bestf, qk)
                    prev = (1, bestf.col)
            elif bOk and bestb.gt(last_q):
                cols = bestb.mod_cols                                       # REMOVE the backward unit
                last_q = float(bestb.qof[qk])
                steps.append(bestb)
                best = cls.pick(best, bestb, qk)
                prev = (-1, bestb.col)
            elif swap and fOk and bOk:
                cj = cls._plus(cls._minus(cols, units(bestb.col)), units(bestf.col))
                nEval += 1
                bestfb = cls._step(eval, bestf.col, cj, qk)
                if bestfb.qof is not None and bestfb.gt(last_q):
                    cols = cj
                    last_q = float(bestfb.qof[qk])
                    steps.append(bestfb)
                    best = cls.pick(best, bestfb, qk)
                    prev = (0, -1)                                          # a swap is not undone by one move
                else:
                    go = False
            else:
                go = False
            l += 1
        return cls.Path(steps, best, nEval)

    @classmethod
    def beam(cls, n: int, eval: Evaluator, fixed: Optional[Sequence[int]] = None,
             units: Units = _singleton, bk: int = 3, moves: Sequence[Tuple[int, int]] = BEAM_MOVES,
             patience: int = 4, qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        BEAM SEARCH: phase I evaluates every single unit added to the fixed columns and keeps the
        `bk` best sets (the beam); phase II expands every set of the beam by the `moves` (columns
        added, columns removed), keeps the `bk` best new sets, and repeats until `patience` rounds
        pass without a better best or no new set can be formed.  The beam is ordered by the
        criterion in its own direction (maximized or minimized).

        Parameters
        ----------
        n, eval, fixed, units, qk
            As in `forward`.
        bk : int, default=3
            The beam width (the sets kept per round).
        moves : sequence of (int, int), default=BEAM_MOVES
            The move set, (units added, units removed) per move.
        patience : int, default=4
            The rounds without a better best after which the search stops.
        """
        qk = given_qk(qk)
        fixed = [0] if fixed is None else list(fixed)
        maxi = qk in MAXI

        def rank(s: BestStep) -> float:                                     # ascending = better first
            return -float(s.qof[qk]) if maxi else float(s.qof[qk])

        def top(cands: List[BestStep]) -> List[BestStep]:
            return sorted([s for s in cands if s.qof is not None], key=rank)[:bk]

        def keep(pool: List[BestStep], s: BestStep) -> None:                # bounded pool
            if s.qof is not None:
                pool.append(s)
                if len(pool) > bk:                                          # drop the worst (the latest among ties,
                    worst = max(rank(x) for x in pool)                      # as a stable sort would)
                    pool.pop(max(i for i, x in enumerate(pool) if rank(x) == worst))

        units_ = cls._freeUnits(n, fixed, units)
        steps: List[BestStep] = []
        seen: Set[frozenset] = set()
        nEval = 0

        # phase I: one unit each
        first: List[BestStep] = []
        for u in units_:
            cj = cls._plus(list(fixed), u)
            seen.add(frozenset(cj))
            nEval += 1
            keep(first, cls._step(eval, min(u), cj, qk))

        beam_ = top(first)
        if not beam_:
            print("SubsetSearch.beam: no single-unit set could be evaluated")
            return cls.Path(steps, BestStep(qk=qk), nEval)
        best = beam_[0]
        steps.append(best)

        # phase II: local search by the move set
        left = patience
        go = True
        while go and left > 0:
            nxt: List[BestStep] = []
            for s in beam_:
                cols = s.mod_cols
                in_ = [u for u in units_ if all(j not in cols for j in u)]
                out_ = [u for u in units_ if all(j in cols for j in u)]
                for nIn, nOut in moves:
                    if len(in_) >= nIn and len(out_) >= nOut:
                        for added in combinations(in_, nIn):
                            for removed in combinations(out_, nOut):
                                cj = cols
                                for u in removed:
                                    cj = cls._minus(cj, u)
                                for u in added:
                                    cj = cls._plus(cj, u)
                                key = frozenset(cj)
                                if len(cj) > len(fixed) and key not in seen:
                                    seen.add(key)
                                    nEval += 1
                                    j = min(added[0]) if added else min(removed[0])
                                    keep(nxt, cls._step(eval, j, cj, qk))
            nextBeam = top(nxt)
            if not nextBeam:
                go = False
            else:
                cand = nextBeam[0]
                if cand.gt(float(best.qof[qk])):
                    best = cand
                    left = patience
                else:
                    left -= 1
                beam_ = nextBeam
                steps.append(best)
        return cls.Path(steps, best, nEval)

    @classmethod
    def cem(cls, n: int, eval: Evaluator, fixed: Optional[Sequence[int]] = None,
            prior: Optional[Sequence[float]] = None, nSample: int = CEM_Selector.N_SAMPLE,
            maxIter: int = CEM_Selector.MAX_ITER, maxCols: int = -1, stream: int = 0,
            qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        CROSS-ENTROPY SELECTION: runs `CEM_Selector` on the evaluator and returns its trace
        (best-so-far per round), best set, evaluations and final inclusion probabilities.

        Parameters
        ----------
        n, eval, fixed, qk
            As in `forward`.
        prior : sequence of float, optional
            The prior inclusion probability per column (None => 0.5 each).
        nSample : int
            The number of column sets sampled per round.
        maxIter : int
            The maximum number of rounds.
        maxCols : int, default=-1
            The largest set allowed (-1 => no cap).
        stream : int, default=0
            The random number stream.
        """
        qk = given_qk(qk)
        fixed = [0] if fixed is None else list(fixed)
        sel = CEM_Selector(n, eval, fixed, prior, nSample, maxIter=maxIter, maxCols=maxCols, stream=stream, qk=qk)
        best = sel.run()
        return cls.Path(list(sel.bestByRound), best, sel.nEvaluated, sel.probs)

    @classmethod
    def run(cls, tech: SelectionTech, n: int, eval: Evaluator, fixed: Optional[Sequence[int]] = None,
            units: Units = _singleton, swap: bool = True, bk: int = 3,
            prior: Optional[Sequence[float]] = None, stream: int = 0,
            rank: Optional[Union[Ranker, Sequence[float], np.ndarray]] = None,
            qk: Optional[int] = None) -> "SubsetSearch.Path":
        """
        Runs the search of a selection technique (the dispatcher hosts call).

        Parameters
        ----------
        tech : SelectionTech
            The selection technique.
        n, eval, fixed, units, qk
            As in `forward`.
        swap : bool, default=True
            Whether stepwise may swap.
        bk : int, default=3
            The beam width.
        prior : sequence of float, optional
            The prior inclusion probabilities (CEM).
        stream : int, default=0
            The random number stream (CEM).
        rank : callable or array, optional
            The importance per column (Ranked; see `ranked`).
        """
        qk = given_qk(qk)
        if tech is SelectionTech.Forward:
            return cls.forward(n, eval, fixed, units, qk=qk)
        if tech is SelectionTech.Backward:
            return cls.backward(n, eval, fixed, units, qk=qk)
        if tech is SelectionTech.Stepwise:
            return cls.stepwise(n, eval, fixed, units, swap, qk=qk)
        if tech is SelectionTech.Beam:
            return cls.beam(n, eval, fixed, units, bk, qk=qk)
        if tech is SelectionTech.CEM:
            return cls.cem(n, eval, fixed, prior, stream=stream, qk=qk)
        if tech is SelectionTech.Ranked:
            if rank is None:
                raise ValueError("SubsetSearch.run: the Ranked technique needs a rank (an importance per column)")
            return cls.ranked(n, eval, rank, fixed, units, qk=qk)
        raise ValueError(f"SubsetSearch.run: unknown technique {tech}")
