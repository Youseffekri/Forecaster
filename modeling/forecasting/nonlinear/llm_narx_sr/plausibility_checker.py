"""
PlausibilityChecker: Plausibility Checks of LLM-NARX-SR Equations

Reads `ExpRun` records and their equations (refitted through the scorer) and reports what a domain
reader asks of an equation: (1) sign consistency -- the NET effect of each exogenous variable (the
sum of the standardized h = 1 coefficients of its terms) has the sign domain knowledge expects
(`DatasetConfig.expectedSigns`); (2) residual cleanliness -- the residual tests of
`EDA4TS.residualReport` (Ljung-Box whiteness, ACF spikes, collinearity, heteroscedasticity);
(3) structural stability -- how similar the best equations of one arm are across streams (mean
pairwise Jaccard similarity of their term sets).  Nothing here influences a search; it is the
post-hoc table of the paper.

The Python counterpart of ScalaTion's `llm_narx_sr/PlausibilityChecker.scala`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Util.tools import banner
from modeling.forecasting.eda4ts import EDA4TS
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import ExpRun


@dataclass
class PlausibilityReport:
    """
    The checks of one equation.

    Attributes
    ----------
    formulaStr : str
        The equation (formula string).
    signConsistency : float
        The share of expected-sign variables whose net effect agrees (NaN if none).
    nSignChecked : int
        The number of variables checked.
    netEffects : dict of str -> float
        The standardized net effect per variable (h = 1).
    cleanliness : float
        The share of residual tests passed.
    notes : list of str
        The residual findings in words.
    """
    formulaStr: str
    signConsistency: float
    nSignChecked: int
    netEffects: Dict[str, float]
    cleanliness: float
    notes: List[str] = field(default_factory=list)

    @property
    def tableRow(self) -> str:
        """One formatted line for a per-equation table."""
        eff = " ".join(f"{v} {e:+.2f}" for v, e in sorted(self.netEffects.items()))
        return (f"{self.formulaStr.replace(chr(10), ' ')[:56]:<56s}  sign={self.signConsistency:5.2f} (n={self.nSignChecked})  "
                f"clean={self.cleanliness:5.2f}  net: {eff}")


class PlausibilityChecker:
    """
    Checks equations against domain expectations and residual tests (ScalaTion's `PlausibilityChecker`).

    Parameters
    ----------
    expectedSigns : dict of str -> int, optional
        The expected sign (+1/-1) of each variable's net effect.
    """

    def __init__(self, expectedSigns: Optional[Dict[str, int]] = None) -> None:
        self.expectedSigns = dict(expectedSigns or {})

    @staticmethod
    def netEffects(mod, f) -> Dict[str, float]:
        """The standardized net effect of each variable at h = 1: the sum over the single-variable
        terms of the variable of slope x standard deviation of the term's column (products and waves
        are not attributed)."""
        b = np.asarray(mod.parameter, dtype=float)                        # h = 1 slopes in original units
        x = mod.X
        eff: Dict[str, float] = {}
        for j, t in enumerate(f.preds):
            if len(t.vars) == 1:
                v = next(iter(t.vars))
                eff[v] = eff.get(v, 0.0) + float(b[j] * np.std(x[:, j], ddof=1))
        return eff

    def signConsistency(self, effects: Dict[str, float]) -> Tuple[float, int]:
        """The sign-consistency score and the number of variables checked."""
        checked = [(v, s) for v, s in self.expectedSigns.items() if v in effects]
        if not checked:
            return math.nan, 0
        agree = sum(1 for v, s in checked if math.copysign(1.0, effects[v]) == float(s) and effects[v] != 0.0)
        return agree / len(checked), len(checked)

    def evaluate(self, cand: Candidate, scorer: FormulaScorer, eda: EDA4TS) -> PlausibilityReport:
        """Checks one equation: refit through the scorer (fit rows), its net effects read and the
        residual tests run on its training residuals."""
        f = cand.formula
        if f is None:                                                     # a base model (anchor): no formula to read
            return PlausibilityReport(cand.formulaStr, math.nan, 0, {}, math.nan, ["base model: not checked"])
        try:
            mod = scorer.fit(f)
            eff = self.netEffects(mod, f)
            sc, n = self.signConsistency(eff)
            skip = max(f.maxLag, scorer.skip)
            tr = scorer.trSize
            idx = np.arange(skip, tr)
            yt, yp = scorer._true_pred(mod, range(0, tr), 1, True)
            e = yt[skip:] - yp[skip:]
            notes, clean = eda.residualReport(e, mod.X[idx], mod.fname, yt[skip:])
            return PlausibilityReport(f.formulaStr, sc, n, eff, clean, notes)
        except Exception as ex:
            return PlausibilityReport(f.formulaStr, math.nan, 0, {}, math.nan, [f"not checked: {ex}"])

    @staticmethod
    def stability(cands: Sequence[Candidate]) -> float:
        """The structural stability of a set of equations: the mean pairwise Jaccard similarity of
        their term-label sets (NaN for fewer than two)."""
        sets = [{t.label() for t in c.formula.preds} | ({"TFORM"} if c.formula.tFormT is not None else set())
                for c in cands if c.formula is not None]
        if len(sets) < 2:
            return math.nan
        pairs = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                pairs.append(1.0 if not u else len(sets[i] & sets[j]) / len(u))
        return sum(pairs) / len(pairs)

    def report(self, title: str, runs: Sequence[ExpRun], scorer: FormulaScorer, eda: EDA4TS) -> None:
        """Prints the plausibility table of a grid: per arm, the stability of its best equations over
        streams and the mean sign consistency and cleanliness; then one row per run."""
        banner(f"{title} -- plausibility (stability over streams, sign consistency, residual cleanliness)")
        reports = [(r, self.evaluate(r.best, scorer, eda)) for r in runs]
        print(f"{'method':<14s} {'n':>3s} {'stability':>9s} {'sign':>6s} {'clean':>6s}")
        for m in dict.fromkeys(r.name for r in runs):
            rs = [(r, p) for r, p in reports if r.name == m]
            sign = [p.signConsistency for _, p in rs if not math.isnan(p.signConsistency)]
            cln = [p.cleanliness for _, p in rs if not math.isnan(p.cleanliness)]
            mS = math.nan if not sign else sum(sign) / len(sign)
            mC = math.nan if not cln else sum(cln) / len(cln)
            print(f"{m:<14s} {len(rs):3d} {self.stability([r.best for r, _ in rs]):9.2f} {mS:6.2f} {mC:6.2f}")
        banner(f"{title} -- per run")
        for r, p in reports:
            print(f"{r.name:<14s} {r.stream:3d}  {p.tableRow}")
            for n in p.notes:
                print(f"                    - {n}")
