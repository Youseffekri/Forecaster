"""
BaselineArm: the Classical Baseline Arms of LLM-NARX-SR (V0)

Four deterministic anchors, the classical structures WITHOUT SEARCH, each fitted and scored by the
same engine, rows and ridge penalty as every candidate of the search arms (`FormulaScorer`):
`Baseline` = the plain ARX structure  y ~= y~(L,1) + x1~(L,1) + ...  with L = hp["maxLag"] on the
shortlisted drivers -- the very seed of every search, so an arm can only be credited with what its
search added;  `TunedARX` = the best of the ARX structures over p in 1..L response lags and px in
0..L exogenous lags by the selection criterion (the classical alternative to searching structure --
the anchor the paper must beat);  `SelARX` = the baseline's lags under the base's feature selection
(forward and backward, the better kept);  `RandomWalk` = the base's `RandomWalk (y, hh)` (no model
at all).  The exogenous variables are the configuration's shortlist (`DatasetConfig.shortlist`:
given, or picked by the screening).  Each runs once.

Why the anchors are formulas and not `ARX_D (xe, y, hh)` built by the base's factory: the
factory's regression centers its columns but does not standardize them, so one ridge penalty acts
differently on columns of different scales, while the search engine standardizes every term column
(`FormulaScorer.scaled`); an anchor fitted by another regression than the candidates would confound
the comparison.

The Python counterpart of ScalaTion's `llm_narx_sr/BaselineArm.scala`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import List, Sequence

from modeling.feature_selection import SelectionTech
from modeling.forecasting.random_walk import RandomWalk
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.nonlinear.llm_narx_sr.datasets import DatasetConfig
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import ANCHORS, ExpRun, Method, SearchArm
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import TermLibrary


def arxFormulaStr(resp: str, p: int, px: int, exo: Sequence[str]) -> str:
    """
    The ARX formula string with p response lags and px exogenous lags,
    y ~= y~(p,1) + x1~(px,1) + ...  (no exogenous terms when px = 0); with p = px = L it is
    `DatasetConfig.baselineFormulaStr (L, exo)`, the seed.
    """
    expo = "".join(f" + {v}~({px},1)" for v in exo) if px > 0 else ""
    return f"{resp} ~= {resp}~({p},1){expo}"


class BaselineArm(SearchArm):
    """
    One of the classical anchors (ScalaTion's `BaselineArm`).

    Parameters
    ----------
    kind : Method, default=Method.Baseline
        Which anchor: `Method.Baseline`, `Method.TunedARX`, `Method.SelARX` or `Method.RandomWalk`.
    verbose : bool, default=True
        Whether to print the arm's debug lines.
    """

    def __init__(self, kind: Method = Method.Baseline, verbose: bool = True) -> None:
        if kind not in ANCHORS:
            raise ValueError(f"BaselineArm: {kind} is not an anchor")
        self.kind = kind
        self.verbose = verbose

    @property
    def method(self) -> Method:
        return self.kind

    @property
    def deterministic(self) -> bool:
        return True

    def run(self, cfg: DatasetConfig, scorer: FormulaScorer, stream: int = -1) -> ExpRun:
        """Scores the anchor once (the stream is ignored)."""
        eval0 = scorer.evaluations
        maxLag = int(scorer.hp["maxLag"])
        exo = cfg.shortlist(scorer)
        if self.verbose:
            print(f"BaselineArm.run: {cfg.name}: exogenous variables = {', '.join(exo)}")

        if self.kind is Method.Baseline:
            px = 0 if not exo else maxLag
            cand = self.arx(scorer, cfg, maxLag, px, exo)
            return self.record(scorer, cfg, -1, cand).copy(nEval=scorer.evaluations - eval0, exoUsed=list(exo) if px > 0 else [])
        if self.kind is Method.TunedARX:
            cands = [(self.arx(scorer, cfg, p, px, exo), px)
                     for p in range(1, maxLag + 1) for px in range(0, (maxLag if exo else 0) + 1)]
            best, pxBest = min(cands, key=lambda cp: cp[0].score)
            top = sorted([c for c, _ in cands], key=lambda c: c.score)[:5]
            return self.record(scorer, cfg, -1, best, top).copy(nEval=scorer.evaluations - eval0,
                                                              exoUsed=list(exo) if pxBest > 0 else [])
        if self.kind is Method.SelARX:
            px = 0 if not exo else maxLag
            f = NARX_FormulaParser.parseFormula(arxFormulaStr(cfg.resp, maxLag, px, exo), scorer.vars, maxLag)
            lib = TermLibrary(f.resp, f.preds, f.tFormT)
            runs: List[Candidate] = []
            for tech in (SelectionTech.Forward, SelectionTech.Backward):
                c, _, nEv = lib.selectBy(tech, scorer)
                if self.verbose:
                    print(f"BaselineArm.run: sel-arx {tech}: {nEv} sets => {c.formulaStr} @ {c.score}")
                runs.append(c)
            fitted = [c for c in runs if not c.failed]
            best = scorer.score(f) if not fitted else min(fitted, key=lambda c: c.score)
            return self.record(scorer, cfg, -1, best, sorted(fitted, key=lambda c: c.score)).copy(nEval=scorer.evaluations - eval0)
        # Method.RandomWalk
        y = scorer.fullSet[cfg.resp]
        cand = scorer.score(RandomWalk(None, y, scorer.hh), "RandomWalk: y_t+h = y_t-1")
        return self.record(scorer, cfg, -1, cand).copy(nEval=scorer.evaluations - eval0)

    @staticmethod
    def arx(scorer: FormulaScorer, cfg: DatasetConfig, p: int, px: int, exo: Sequence[str]) -> Candidate:
        """The ARX structure with p response lags and px exogenous lags on the shortlisted drivers,
        y ~= y~(p,1) + x1~(px,1) + ..., built and scored through the engine (one column per lag,
        standardized, the scorer's ridge penalty), exactly as a proposal of a search arm is scored."""
        f = NARX_FormulaParser.parseFormula(arxFormulaStr(cfg.resp, p, px, exo), scorer.vars, int(scorer.hp["maxLag"]))
        return scorer.score(f)


BaselineArm.arxFormulaStr = staticmethod(arxFormulaStr)
