"""
CEM_Selector Module: Cross-Entropy Method (CEM) for Feature Selection

The Python counterpart of ScalaTion's `modeling/CEM_Selector.scala`.  Feature selection as a search
over inclusion PROBABILITIES rather than over feature sets: column j carries a probability p_j of
belonging to a good model.  Each round samples `nSample` distinct column sets (column j included
with probability p_j), evaluates each with the caller's `eval` function (build, train, test),
keeps the elite fraction and moves p towards the elite's inclusion frequencies,

    p_j <- (1 - alpha) p_j + alpha freq_j,      p_j kept in [eps, 1 - eps],

so no column is ever ruled in or out for good.  Settings follow the CE tutorial (de Boer et al.
2005, Remarks 2.3 and 2.6): sample size N = c n with c = 5, elite fraction rho = ln(n)/n (kept in
[0.05, 0.3]), smoothing alpha = 0.7, and the stopping rule on the elite THRESHOLD gamma (the worst
elite score): stop when gamma has not improved for `patience` = 5 rounds, when every p_j has
reached its floor or ceiling (degenerate distribution), after `maxIter` rounds, or when the
sampler cannot find an unseen set.  Two additions from iCEM (Pinneri et al. 2021): a fraction
`keep` of the previous round's elites joins the next round's ranking (elite memory, no refit
needed), and the incumbent best set always counts as an elite (elitism).  No set is fitted twice.
The result is the best set evaluated, never the mean of the distribution.

Compared with the greedy Forward/Backward/Stepwise selectors it is not greedy (a column dropped in
one round can return), it keeps diversity by sampling instead of enumerating neighbours, and it
takes a PRIOR: p starts from `prior` when one is given, so the final p measures how far the
evidence moved that prior.  The final p is a result in itself: columns near one are the stable core.

The random numbers come from `Random`, a port of ScalaTion's `scalation.random.Random` (the
MRG31k3p generator of L'Ecuyer with ScalaTion's seed table), so the same `stream` draws the same
column sets as the Scala run.

References
----------
de Boer, Kroese, Mannor and Rubinstein, "A Tutorial on the Cross-Entropy Method", Annals of
Operations Research 134 (2005) 19-67.
Pinneri et al., "Sample-efficient Cross-Entropy Method for Real-time Planning", CoRL 2020
(PMLR 155, 2021), for elite memory.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from math import ceil, log
from typing import Callable, List, Optional, Sequence, Set

import numpy as np

from modeling.feature_selection import BestStep, Cols, given_qk
from modeling.fit import MAXI, extreme


class Random:
    """
    The MRG31k3p random number generator of ScalaTion's `scalation.random.Random` (L'Ecuyer and
    Touzin's combined multiple recursive generator), with the first `N_STREAMS` seed vectors of
    ScalaTion's `RandomSeeds` so that `Random (stream).gen ()` draws ScalaTion's sequence.

    Parameters
    ----------
    stream : int, default=0
        The random number stream (taken modulo `N_STREAMS`).
    """

    M1 = 2147483647                      # modulus for MRG1 (2^31 - 1)
    M2 = 2147462579                      # modulus for MRG2 (2^31 - 21069)
    MASK12 = 511                         # mask to extract  9 lsb's (2^9 - 1)
    MASK13 = 16777215                    # mask to extract 24 lsb's (2^24 - 1)
    MASK21 = 65535                       # mask to extract 16 lsb's (2^16 - 1)
    MULT2 = 21069                        # multiplier
    NORM = 4.656612873077393e-10         # 1.0 / (M1 + 1.0) normalization to (0, 1)

    #: The first 32 of ScalaTion's 1000 seed vectors (`RandomSeeds.seeds`).
    SEEDS = (
        (12345, 12345, 12345, 12345, 12345, 12345),
        (336690377, 597094797, 1245771585, 85196284, 523477687, 2094976052),
        (502033783, 1322587635, 1964121530, 1949818481, 1607232546, 1462898381),
        (739421137, 1475938232, 730262207, 1630192198, 324551134, 795289868),
        (1719768226, 483121100, 630243355, 233387880, 1309486499, 955444484),
        (1071170538, 983367249, 1747714525, 1834354612, 1112958853, 1834520262),
        (668619019, 509577539, 471222478, 1684281489, 2023503286, 740874994),
        (194910996, 480553001, 167231637, 1590543982, 1105556857, 564564228),
        (1367468043, 509255716, 1843495086, 673221214, 401872223, 1271230434),
        (2121930688, 722207388, 1672097838, 567900934, 276898816, 1283772973),
        (1038210110, 1207855773, 2105199287, 1465402426, 1568031300, 1220015950),
        (2107420544, 1862810234, 1008829961, 471248946, 1766957949, 2011878418),
        (203171444, 703718217, 521087742, 1591997419, 1074695786, 212589429),
        (2005735039, 583570263, 2051555158, 513657610, 1940511377, 1312057937),
        (1011808945, 1331083829, 430507875, 353428573, 1932853315, 1014817719),
        (1300738479, 1916653767, 1489706578, 1985198640, 1453394643, 756305654),
        (913957902, 1851489016, 947993814, 1921546598, 916518617, 1757169865),
        (1195185732, 700359740, 792443106, 1781553352, 881793544, 1265680856),
        (1974367046, 567870939, 1592004547, 261176487, 1458611882, 992379967),
        (535631132, 1969978499, 407096239, 449762071, 984898672, 1069075015),
        (1386846636, 1034966850, 1281573087, 707488115, 620232396, 571263101),
        (1185099363, 820770467, 2055442602, 247655741, 485376535, 282790300),
        (520297187, 1116524896, 7215430, 26564296, 1101160204, 608471554),
        (1224997511, 1850787006, 1033199908, 115830823, 1589325787, 1757316387),
        (173334637, 1673232248, 1031025473, 2006124949, 1993609511, 1752510874),
        (234166222, 433432833, 1297147641, 39889038, 1626026400, 1803287512),
        (676345933, 921807125, 342259702, 1861649344, 1877877245, 1769838614),
        (1147008840, 1674990696, 1365465906, 1525424136, 791743513, 106511429),
        (544330724, 237997753, 1451311974, 820380123, 89846203, 1202720677),
        (1712417343, 373981490, 1894384939, 982314500, 1903094282, 2138345582),
        (1467690409, 326550619, 1900627694, 1840600565, 1593705522, 1212274195),
        (65899610, 111686131, 2139189103, 368973355, 229622761, 1159638732),
    )
    N_STREAMS = len(SEEDS)

    def __init__(self, stream: int = 0) -> None:
        self.stream = int(stream)
        x = self.SEEDS[self.stream % self.N_STREAMS]                   # 6-dimensional vector of seed values
        self._x11, self._x12, self._x13 = x[0], x[1], x[2]            # x_i-1, x_i-2, x_i-3 for MRG1
        self._x21, self._x22, self._x23 = x[3], x[4], x[5]            # x_i-1, x_i-2, x_i-3 for MRG2

    def _step(self) -> int:
        """Advances both recurrences and returns the combined integer in [0, M1)."""
        M1, M2 = self.M1, self.M2
        x12, x13 = self._x12, self._x13
        y1 = (((x12 & self.MASK12) << 22) + (x12 >> 9)) + (((x13 & self.MASK13) << 7) + (x13 >> 24))
        if y1 > M1:
            y1 -= M1
        y1 += x13
        if y1 > M1:
            y1 -= M1
        self._x13, self._x12, self._x11 = x12, self._x11, y1

        x21, x23 = self._x21, self._x23
        y1 = ((x21 & self.MASK21) << 15) + self.MULT2 * (x21 >> 16)
        if y1 > M2:
            y1 -= M2
        y2 = ((x23 & self.MASK21) << 15) + self.MULT2 * (x23 >> 16)
        if y2 > M2:
            y2 -= M2
        y2 += x23
        if y2 > M2:
            y2 -= M2
        y2 += y1
        if y2 > M2:
            y2 -= M2
        self._x23, self._x22, self._x21 = self._x22, x21, y2

        x11, x21 = self._x11, self._x21
        return x11 - x21 + M1 if x11 <= x21 else x11 - x21

    def gen(self) -> float:
        """The next uniform random number in (0, 1)."""
        return self._step() * self.NORM

    def igen(self) -> int:
        """The next random integer in [0, M1)."""
        return self._step()


class CEM_Selector:
    """
    Selects a subset of the columns 0 until n by the Cross-Entropy Method, using the caller's
    evaluator to build and score the model of a column set (ScalaTion's `CEM_Selector`; see
    `FeatureSelection_Reg.cemSelAll` for the wiring into feature selection).

    Parameters
    ----------
    n : int
        The number of columns (candidate features) 0 until n.
    eval : callable
        The evaluator: builds, trains and tests the model on a column set (a list of int) and
        returns its `BestStep` (col is not used; qof must hold the metric qk).
    fixed : sequence of int, default=()
        The columns always included (e.g., the intercept column 0).
    prior : np.ndarray, optional
        The prior inclusion probability per column (None => 0.5 each; entries for fixed columns are ignored).
    nSample : int, default=N_SAMPLE
        The number of distinct column sets sampled per round (-1 => C_N times the number of
        free columns, at least MIN_SAMPLE).
    elite : float, default=ELITE
        The fraction of a round's sets kept as the elite (-1 => ln(n)/n in [0.05, 0.3]).
    alpha : float, default=ALPHA
        The smoothing rate of the probability update.
    eps : float, default=EPS
        The floor/ceiling keeping probabilities inside (0, 1).
    maxIter : int, default=MAX_ITER
        The maximum number of rounds.
    patience : int, default=PATIENCE
        The rounds without improving the elite threshold after which the run stops.
    maxCols : int, default=-1
        The maximum number of columns in a sampled set (-1 => n).
    keep : float, default=KEEP
        The fraction of the previous round's elites carried into the next ranking.
    stream : int, default=0
        The random number stream.
    qk : int, optional
        The index of the QoF measure to compare on (defaults to the given `qk`).

    Attributes
    ----------
    nS : int
        The sample size per round actually used.
    rho : float
        The elite fraction actually used.
    """

    N_SAMPLE = -1                        # sets per round: -1 => C_N x free columns
    C_N = 5.0                            # sample size multiplier (tutorial: 5 to 10)
    MIN_SAMPLE = 40                      # smallest sample size per round
    ELITE = -1.0                         # elite fraction: -1 => ln(n)/n in [0.05, 0.3]
    ALPHA = 0.7                          # smoothing rate (tutorial: 0.4 to 0.9)
    EPS = 0.05                           # probability floor / ceiling
    MAX_ITER = 30                        # maximum number of rounds
    PATIENCE = 5                         # rounds without a better elite threshold (tutorial: d = 5)
    KEEP = 0.3                           # fraction of elites remembered (iCEM: 0.3)
    GAMMA_TOL = 1e-4                     # relative improvement of gamma that resets patience
    MAX_TRIES = 200                      # draws before the sampler gives up

    def __init__(self, n: int, eval: Callable[[Cols], Optional[BestStep]], fixed: Sequence[int] = (),
                 prior: Optional[Sequence[float]] = None,
                 nSample: int = N_SAMPLE, elite: float = ELITE, alpha: float = ALPHA, eps: float = EPS,
                 maxIter: int = MAX_ITER, patience: int = PATIENCE, maxCols: int = -1,
                 keep: float = KEEP, stream: int = 0, qk: Optional[int] = None) -> None:
        self.n = int(n)
        self.eval = eval
        self.fixed: Cols = list(fixed)
        self.alpha, self.eps = float(alpha), float(eps)
        self.maxIter, self.patience, self.keep = int(maxIter), int(patience), float(keep)
        self.qk = given_qk(qk)

        self._rng = Random(stream)                                                    # uniform (0, 1) random number generator
        self._free = [j for j in range(self.n) if j not in self.fixed]                # the columns the sampler decides on
        self._cap = self.n if maxCols < 0 else int(maxCols)                           # the largest set allowed
        self._maxi = self.qk in MAXI                                                  # whether the metric is maximized
        self._seen: Set[frozenset] = set()                                            # the sets evaluated so far
        self._trace: List[BestStep] = []                                              # the best-so-far after each round
        self._bestStep = BestStep(qk=self.qk)                                         # the best set found so far

        nf = len(self._free)
        self.nS: int = int(nSample) if nSample > 0 else max(self.MIN_SAMPLE, int(self.C_N * nf))
        self.rho: float = float(elite) if elite > 0.0 else min(0.3, max(0.05, log(max(2, nf)) / max(2, nf)))

        if not (self.n > 0 and all(0 <= j < self.n for j in self.fixed)):
            raise ValueError(f"CEM_Selector: fixed columns must be in 0 until {self.n}")
        if len(self.fixed) > self._cap:
            raise ValueError(f"CEM_Selector: {len(self.fixed)} fixed columns exceed maxCols = {self._cap}")
        if prior is not None and len(prior) != self.n:
            raise ValueError(f"CEM_Selector: prior.dim {len(prior)} != n = {self.n}")
        if not (self.rho <= 1.0 and 0.0 < self.alpha <= 1.0 and 0.0 < self.eps < 0.5 and 0.0 <= self.keep <= 1.0):
            raise ValueError("CEM_Selector: need 0 < elite <= 1, 0 < alpha <= 1, 0 < eps < 0.5, 0 <= keep <= 1")

        # the inclusion probabilities, one per column (fixed columns stay at 1)
        self._p = np.array([1.0 if j in self.fixed else (0.5 if prior is None else self._clamp(float(prior[j])))
                            for j in range(self.n)])

    # -- results ----------------------------------------------------------------------------

    @property
    def probs(self) -> np.ndarray:
        """A copy of the current inclusion probabilities."""
        return self._p.copy()

    @property
    def best(self) -> BestStep:
        """The best step found so far (qof is None before `run`)."""
        return self._bestStep

    @property
    def bestByRound(self) -> List[BestStep]:
        """The best-so-far step after each completed round (the trace)."""
        return list(self._trace)

    @property
    def nEvaluated(self) -> int:
        """The number of distinct column sets evaluated (models fitted)."""
        return len(self._seen)

    # -- the algorithm ----------------------------------------------------------------------

    def _clamp(self, v: float) -> float:
        """Keeps a probability inside [eps, 1 - eps]."""
        return min(1.0 - self.eps, max(self.eps, v))

    def _score(self, s: Optional[BestStep]) -> float:
        """The metric of a step, with a failed fit (None or NaN) scored as worst."""
        if s is None or s.qof is None or np.isnan(s.qof[self.qk]):
            return extreme(self.qk)
        return float(s.qof[self.qk])

    def _better(self, a: float, b: float) -> bool:
        """Whether metric value a is better than b (direction from `MAXI`)."""
        return a > b if self._maxi else a < b

    def _draw(self) -> Cols:
        """
        Draws one column set from the current probabilities: each free column with probability
        p_j, at least one free column, at most `cap` columns (the least likely extras dropped),
        plus the fixed columns; in ascending column order.
        """
        p = self._p
        cols = [j for j in self._free if self._rng.gen() < p[j]]
        if not cols and self._free:                                        # never an empty draw
            cols.append(max(self._free, key=lambda j: p[j]))
        while len(cols) + len(self.fixed) > self._cap:                     # enforce the size cap
            cols.remove(min(cols, key=lambda j: p[j]))
        return sorted(self.fixed + cols)

    def _sample(self) -> Optional[Cols]:
        """A column set not evaluated before, or None when MAX_TRIES draws in a row hit sets already seen."""
        for _ in range(self.MAX_TRIES):
            cols = self._draw()
            if frozenset(cols) not in self._seen:
                return cols
        return None

    def _evaluate(self, cols: Cols) -> BestStep:
        """Evaluates a column set, scoring a failed build as worst rather than stopping the run."""
        try:
            step = self.eval(cols)
            if step is None:
                step = BestStep(-1, None, None, cols, self.qk)
        except Exception:                                                   # a sampled set may be singular
            step = BestStep(-1, None, None, cols, self.qk)
        self._seen.add(frozenset(cols))
        return step

    def _degenerate(self) -> bool:
        """Whether every free column's probability sits at the floor or the ceiling."""
        return all(self._p[j] <= self.eps + 1e-9 or self._p[j] >= 1.0 - self.eps - 1e-9 for j in self._free)

    def run(self) -> BestStep:
        """
        Runs the Cross-Entropy Method and returns the best step found: its model, its QoF vector
        and its column set (`mod_cols`).
        """
        stagnant = 0
        gamma = extreme(self.qk)                                            # the elite threshold (worst elite score)
        memory: List[BestStep] = []                                         # elites carried over from the last round
        for it in range(1, self.maxIter + 1):
            round_: List[BestStep] = []                                     # this round's freshly evaluated sets
            dry = False
            while len(round_) < self.nS and not dry:
                cols = self._sample()
                if cols is None:
                    dry = True
                else:
                    round_.append(self._evaluate(cols))
            if not round_:
                break                                                       # no unseen sets left

            pool = round_ + memory                                          # fresh samples plus remembered elites
            ranked = sorted(pool, key=(lambda s: -self._score(s)) if self._maxi else self._score)
            nElite = max(1, int(ceil(self.rho * len(round_))))
            top = ranked[:nElite]
            best = self._bestStep
            if best.qof is not None and not any(frozenset(t.mod_cols) == frozenset(best.mod_cols) for t in top):
                elites = top + [best]                                       # elitism: the incumbent always counts
            else:
                elites = top

            for j in self._free:                                            # move p towards the elite frequencies
                freq = sum(1 for e in elites if j in e.mod_cols) / len(elites)
                self._p[j] = self._clamp((1.0 - self.alpha) * self._p[j] + self.alpha * freq)

            newGamma = self._score(top[-1])                                 # the elite threshold of this round
            tol = self.GAMMA_TOL * (1.0 + abs(newGamma))                    # relative improvement that counts
            if self._better(newGamma, gamma) and abs(newGamma - gamma) > tol:
                gamma, stagnant = newGamma, 0
            else:
                stagnant += 1
            if self._better(self._score(ranked[0]), self._score(self._bestStep)):
                self._bestStep = ranked[0]
            memory = top[:max(1, int(ceil(self.keep * nElite)))]            # elite memory for the next round
            self._trace.append(self._bestStep)
            if dry or stagnant >= self.patience or self._degenerate():
                break
        if self._bestStep.qof is None:
            print("CEM_Selector.run: no column set could be evaluated")
        return self._bestStep
