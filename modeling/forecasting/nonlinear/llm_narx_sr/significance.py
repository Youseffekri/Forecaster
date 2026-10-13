"""
Significance: Significance Analysis of LLM-NARX-SR Runs

Reads `ExpRun` records (`search_arm.py`) and nothing else from the harness.  The statistics: the
paired Wilcoxon signed-rank test (`wilcoxon_test`, ScalaTion's `mathstat.wilcoxon_test`: W = min
(W+, W-), its continuity-corrected, tie-corrected normal approximation z and the two-sided p-value)
and Spearman's rank correlation.  Here: the stream-paired comparison of two arms, the one-sample
comparison of an arm against a fixed value (a deterministic arm), the model-vs-every-arm report of
a grid, the per-fold table of `runFolds` (signed-rank over streams x folds against the fold's
baseline) and the selection check (does the selection criterion on the training rows rank
structures as the TnT rolling validation on the testing rows does?).

The Python counterpart of ScalaTion's `llm_narx_sr/Significance.scala`.  `wilcoxon_test` gives the
same (W, z, p) as ScalaTion's; its p-value equals scipy's `wilcoxon (a, b, method="approx",
correction=True)` for n >= 2 (scipy is not used, so a tiny n gives the same number as the Scala run).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Util.tools import banner
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import ExpRun, Method


def wilcoxon_test(a: Sequence[float], b: Sequence[float], show: bool = False) -> Tuple[float, float, float]:
    """
    The two-tailed Wilcoxon signed-rank test of two PAIRED samples (ScalaTion's `wilcoxon_test`):
    returns (W, z, p-value) with W = min (W+, W-), its normal approximation z (continuity-corrected,
    tie-corrected variance) and the two-sided p-value.  Pairs with a(i) == b(i) are dropped before
    ranking; if none remain, (0, 0, 1).  The normal approximation is the usual one for n >= 10 pairs.
    """
    from scipy.stats import norm, rankdata
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        print("wilcoxon_test: samples must have the same length")
        return 0.0, 0.0, 1.0
    d = (a - b)[(a - b) != 0.0]                                           # non-zero paired differences
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0
    rk = rankdata(np.abs(d))                                              # mid-ranks of |d| (ties averaged)
    wp = float(np.sum(rk[d > 0.0]))                                       # W+ = sum of ranks of positive differences
    wm = float(np.sum(rk[d < 0.0]))                                       # W- = sum of ranks of negative differences
    w = min(wp, wm)
    _, counts = np.unique(rk, return_counts=True)
    tcor = float(np.sum(counts.astype(float) ** 3 - counts))              # sum (t^3 - t) over the tie groups of |d|
    meanW = n * (n + 1) / 4.0
    varW = n * (n + 1) * (2 * n + 1) / 24.0 - tcor / 48.0
    if varW <= 0.0:
        return w, 0.0, 1.0                                                # all |d| tied => no test
    z = (w - meanW + (0.5 if w < meanW else -0.5)) / math.sqrt(varW)      # continuity correction toward the mean
    p = min(1.0, 2.0 * (1.0 - norm.cdf(abs(z))))
    if show:
        print(f"wilcoxon_test: n = {n}, W+ = {wp}, W- = {wm}, W = {w}, z = {z}, p-value = {p}")
    return w, z, p


def scorr(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rank correlation (ScalaTion's `VectorD.scorr`)."""
    from scipy.stats import spearmanr
    return float(spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).statistic)


def _testSmape(r: ExpRun) -> float:
    return r.testSmape


@dataclass
class SigResult:
    """
    The outcome of one paired comparison.

    Attributes
    ----------
    nPairs : int
        The number of pairs (streams) compared.
    meanA, sdA, meanB, sdB : float
        The mean and standard deviation of method A's and method B's paired metric.
    wStat : float
        The Wilcoxon W statistic (min of W+, W-).
    zStat : float
        The normal-approximation z-score (with continuity correction).
    pValue : float
        The two-sided p-value.
    """
    nPairs: int
    meanA: float
    sdA: float
    meanB: float
    sdB: float
    wStat: float
    zStat: float
    pValue: float

    def __str__(self) -> str:
        return (f"n={self.nPairs}  A: mean={self.meanA:8.4f} sd={self.sdA:8.4f}  B: mean={self.meanB:8.4f} sd={self.sdB:8.4f}  "
                f"W={self.wStat:6.1f}  z={self.zStat:6.3f}  p={self.pValue:7.4f}" +
                ("  *significant at 0.05*" if self.pValue < 0.05 else ""))


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


class Significance:
    """Compares arms from their run records (ScalaTion's `Significance` object)."""

    @staticmethod
    def compareMethods(runsA: Sequence[ExpRun], runsB: Sequence[ExpRun],
                       metric: Callable[[ExpRun], float] = _testSmape) -> SigResult:
        """The stream-paired Wilcoxon signed-rank test between two methods' per-stream metric (by
        default `ExpRun.testSmape`); only streams present in both are used."""
        byA = {r.stream: metric(r) for r in runsA}
        byB = {r.stream: metric(r) for r in runsB}
        streams = sorted(set(byA) & set(byB))
        if not streams:
            raise ValueError("compareMethods: no common streams between the two run sets")
        a = np.array([byA[s] for s in streams])
        b = np.array([byB[s] for s in streams])
        w, z, p = wilcoxon_test(a, b)
        return SigResult(len(streams), float(a.mean()), _sd(a), float(b.mean()), _sd(b), w, z, p)

    @staticmethod
    def compareToConstant(runsA: Sequence[ExpRun], reference: float,
                          metric: Callable[[ExpRun], float] = _testSmape) -> SigResult:
        """The one-sample signed-rank test of a method's per-stream metric against a fixed reference
        value (e.g., a deterministic arm's single run)."""
        if not runsA:
            raise ValueError("compareToConstant: runsA is empty")
        a = np.array([metric(r) for r in runsA])
        w, z, p = wilcoxon_test(a, np.full(len(a), reference))
        return SigResult(len(a), float(a.mean()), _sd(a), float(reference), 0.0, w, z, p)

    @classmethod
    def reportSignificance(cls, title: str, runs: Sequence[ExpRun], model: Method = Method.LLMCEM,
                           metric: Callable[[ExpRun], float] = _testSmape) -> None:
        """Prints the comparison of the model arm against every other arm in `runs`: stream-paired
        when the other arm has several runs, one-sample against its single value when it has one."""
        ok = [r for r in runs if not r.failed]                           # failed runs carry no metric
        mine = [r for r in ok if r.method is model and r.variant == ""]
        label = model.tag
        banner(f"{title} -- significance of {label} vs each other arm (TnT sMAPE)")
        if not mine:
            print(f"{label}: no fitted runs in this set")
            return
        for m in dict.fromkeys(r.name for r in ok):
            if m == label:
                continue
            other = [r for r in ok if r.name == m]
            if len(other) == 1:
                a = np.array([metric(r) for r in mine])
                line = (f"(deterministic): n={len(a)}  A: mean={a.mean():8.4f} sd={_sd(a):8.4f}  B: {metric(other[0]):8.4f}  "
                        f"diff={a.mean() - metric(other[0]):+8.4f}  -- no test (one value; see the fold table)")
            else:
                try:
                    line = f"(stream-paired): {cls.compareMethods(mine, other, metric)}"
                except ValueError as e:
                    line = f"skipped ({e})"
            print(f"{label} vs {m:<14s} {line}")

    @staticmethod
    def reportFolds(title: str, cuts: Sequence[int], folds: Sequence[Sequence[ExpRun]],
                    refs: Sequence[Method] = (Method.Baseline, Method.SelARX)) -> None:
        """Prints, per arm, the TnT sMAPE of each fold (mean over streams), its mean over folds, the
        mean difference to the fold's reference arms and the two-sided signed-rank p-value of those
        differences over streams x folds (n pairs)."""
        banner(f"{title} -- TnT sMAPE per fold (mean over streams), over folds, and vs the fold's {' / '.join(r.tag for r in refs)}")
        methods = list(dict.fromkeys(r.name for rs in folds for r in rs))
        hdr = " ".join(f"{'..' + str(c):>9s}" for c in cuts)
        rHdr = " ".join(f"{'vs ' + r.tag:>12s} {'n':>4s} {'p':>8s}" for r in refs)
        print(f"{'method':<14s} {hdr} {'mean':>9s} {rHdr}")
        for m in methods:
            perFold = []
            for rs in folds:
                xs = [r.testSmape for r in rs if r.name == m and not r.failed]
                perFold.append(math.nan if not xs else sum(xs) / len(xs))
            means = [v for v in perFold if not math.isnan(v)]
            mean = math.nan if not means else sum(means) / len(means)
            cells = " ".join(f"{v:9.3f}" for v in perFold)
            vs = []
            for ref in refs:
                diffs: List[float] = []
                for rs in folds:
                    b = [r.testSmape for r in rs if r.method is ref and not r.failed]
                    if b:
                        diffs += [r.testSmape - b[0] for r in rs if r.name == m and not r.failed]
                mDiff = math.nan if not diffs else sum(diffs) / len(diffs)
                p = math.nan if (m == ref.tag or len(diffs) < 2) else wilcoxon_test(diffs, np.zeros(len(diffs)))[2]
                vs.append(f"{mDiff:+12.3f} {len(diffs):4d} {p:8.4f}" + ("*" if (not math.isnan(p) and p < 0.05) else " "))
            print(f"{m:<14s} {cells} {mean:9.3f} {' '.join(vs)}")

    @staticmethod
    def selectionCheck(runs: Sequence[ExpRun]) -> float:
        """Spearman's rank correlation between the selection score (training rows) and the TnT sMAPE
        (testing rows) over the fitted runs (NaN if fewer than three): +1 means the criterion ranks
        structures as the rolling validation does, <= 0 that selecting on the training rows does not carry over."""
        scored = [r for r in runs if not r.best.failed and not math.isnan(r.testSmape)]
        if len(scored) < 3:
            return math.nan
        return scorr([r.best.score for r in scored], [r.testSmape for r in scored])
