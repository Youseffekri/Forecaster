"""
Explainer: Grounded Explanations of LLM-NARX-SR Equations with an Automatic Faithfulness Check

A fitted equation is handed to an LLM with its coefficients and standardized effects (coefficient
x standard deviation of the term's column on the training rows, h = 1) and the LLM writes an
explanation for a domain reader.  The reply must begin with one machine-readable CLAIM line per
term (sign of the effect, lag, importance rank) before the narrative, and every claim is checked
against the fitted model: the share of claims that are right is the explanation's faithfulness, the
share of terms claimed is its coverage.  Domain expectations (`DatasetConfig.expectedSigns`) that
the fitted signs contradict are put in the prompt, so the narrative has to address them rather
than paper over them.  `Explainer.importance` (no LLM) gives the term-importance table that
`LLM_NARX_SR.showBest` prints under the coefficient tables -- and the one upward edge of the
package: `FormulaScorer.notes` reads it for the proposer's feedback.

The Python counterpart of ScalaTion's `llm_narx_sr/Explainer.scala` (the prompt texts kept verbatim).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from modeling.llm_client import LLMClient
from modeling.forecasting.datasets import DatasetConfig
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import Candidate, FormulaScorer
from modeling.forecasting.nonlinear.llm_narx_sr.llm_proposer import genericSchema, rename

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CLAIM = re.compile(r"TERM:\s*(.+?)\s*\|\s*SIGN:\s*([+-])\s*\|\s*LAG:\s*(\d+)\s*\|\s*RANK:\s*(\d+)", re.IGNORECASE)
_FN_PREFIX = re.compile(r"^(log1p|log|sqrt|sq|dq|exp|expm1|sin|cos|tanh|pow|powr|sat|hinge|trans|ema)\(")
_AFFINE = re.compile(r"^\??[-+.0-9eE]*\s*[+*]\s*")


@dataclass
class TermEffect:
    """
    One term of a fitted equation at h = 1.

    Attributes
    ----------
    label : str
        The term label (real names).
    lag : int
        The term's lag.
    coef : float
        The coefficient in original units.
    effect : float
        The standardized effect: coef x std of the term's column (training rows).
    """
    label: str
    lag: int
    coef: float
    effect: float

    @property
    def sign(self) -> int:
        return -1 if self.effect < 0.0 else 1


@dataclass
class Claim:
    """
    What the explanation says about one term, checked.

    Attributes
    ----------
    term : str
        The term label (real names).
    sign, lag, rank : int
        The claimed sign of the effect (+1/-1), lag and importance rank (1 = largest |effect|).
    signOK, lagOK, rankOK : bool
        Whether the sign agrees with the fitted effect, the lag is the term's lag, the rank is the term's rank by |effect|.
    """
    term: str
    sign: int
    lag: int
    rank: int
    signOK: bool
    lagOK: bool
    rankOK: bool

    @property
    def ok(self) -> bool:
        return self.signOK and self.lagOK and self.rankOK

    def __str__(self) -> str:
        mark = lambda b: "ok " if b else "BAD"
        return (f"{self.term:<24s} sign {'+' if self.sign > 0 else '-'} {mark(self.signOK)}  lag {self.lag:2d} {mark(self.lagOK)}"
                f"  rank {self.rank:2d} {mark(self.rankOK)}")


@dataclass
class Explanation:
    """
    The LLM's narrative with its checked claims.

    Attributes
    ----------
    text : str
        The narrative for the reader.
    claims : list of Claim
        The claims found in the reply, matched to terms and checked.
    nTerms : int
        The number of terms of the equation.
    """
    text: str
    claims: List[Claim]
    nTerms: int

    @property
    def coverage(self) -> float:
        return -0.0 if self.nTerms == 0 else len(self.claims) / self.nTerms

    def _rate(self, p: Callable[[Claim], bool]) -> float:
        return -0.0 if not self.claims else sum(1 for c in self.claims if p(c)) / len(self.claims)

    @property
    def faithfulness(self) -> float:
        """The share of claims right in every respect."""
        return self._rate(lambda c: c.ok)

    @property
    def signRate(self) -> float:
        return self._rate(lambda c: c.signOK)

    @property
    def lagRate(self) -> float:
        return self._rate(lambda c: c.lagOK)

    @property
    def rankRate(self) -> float:
        return self._rate(lambda c: c.rankOK)

    @property
    def tableRow(self) -> str:
        """One formatted line for a table of explanations."""
        return (f"coverage {self.coverage:5.2f}  faithful {self.faithfulness:5.2f}  (sign {self.signRate:4.2f}  lag {self.lagRate:4.2f}"
                f"  rank {self.rankRate:4.2f})  claims {len(self.claims):2d} / {self.nTerms:2d} terms")


class Explainer:
    """
    Asks an LLM for a grounded explanation of a fitted equation and checks its claims against the
    model (ScalaTion's `Explainer`).

    Parameters
    ----------
    llm : LLMClient
        The completion client.
    cfg : DatasetConfig
        The dataset configuration (names, domain sentence, expected signs).
    named : bool, default=True
        Whether the real variable names and the domain sentence are shown (the search is over, so
        naming leaks nothing; blind for the ablation).
    """

    def __init__(self, llm: LLMClient, cfg: DatasetConfig, named: bool = True) -> None:
        self.llm, self.cfg, self.named = llm, cfg, named
        self.toShown: Dict[str, str] = {} if named else dict(cfg.blindNames)
        self.toReal: Dict[str, str] = {v: k for k, v in self.toShown.items()}
        self._calls = 0
        self._last = ""
        self.rn = self.shown(cfg.resp)
        about = cfg.schema if named else genericSchema(len(cfg.exo), cfg.freq)
        self.system = (
            "You explain a fitted symbolic time-series model to a domain reader.\n"
            f"{about}\n"
            f"The model forecasts {self.rn} one step ahead as an intercept plus a weighted sum of terms; VAR~k is the value\n"
            "of VAR k steps back, so a term with lag k is a delayed effect of k steps.  The standardized effect of\n"
            "a term (coefficient x standard deviation of the term's values) says how much the term moves the\n"
            "forecast; its sign says the direction.  Rank the terms by |standardized effect|, rank 1 the largest.\n"
            "Reply in exactly this form and nothing else:\n"
            "TERM: <term as written> | SIGN: <+ or -> | LAG: <k> | RANK: <r>\n"
            "   (one such line per term, all terms, in any order)\n"
            "EXPLANATION: <one paragraph in plain language: what carries the forecast, which effects are delayed and\n"
            "by how many steps, what each exogenous variable does and in which direction, and, where the fitted\n"
            "sign contradicts domain expectation, say so and offer a reason>\n")

    @property
    def calls(self) -> int:
        """The number of prompts sent so far."""
        return self._calls

    @property
    def lastPrompt(self) -> str:
        """The last user prompt sent."""
        return self._last

    def prompt(self, f, effects: Sequence[TermEffect], intercept: float) -> str:
        """The user prompt for a fitted equation: the equation with its h = 1 coefficients, the term
        table (lag, coefficient, standardized effect) and any conflict between a fitted net effect and
        the expected sign."""
        eq = " ".join(f"{e.coef:+.4g}*{self.toShownText(e.label)}" for e in effects)
        rows = "\n".join(f"{self.toShownText(e.label):<28s} {e.lag:3d}   {e.coef:+12.4g}   {e.effect:+12.4g}" for e in effects)
        net: Dict[str, float] = {}
        for e in effects:
            if "~" in e.label and "*" not in e.label:                     # single-variable terms
                v = self.variableOf(e.label)
                net[v] = net.get(v, 0.0) + e.effect
        conflicts = []
        for v, s in self.cfg.expectedSigns.items():
            if v in net and math.copysign(1.0, net[v]) != float(s):
                e = net[v]
                conflicts.append(f"Domain expectation: {self.shown(v)} should have a {'positive' if s > 0 else 'negative'} net effect; "
                                 f"the fitted net effect is {e:+.4g} ({'negative' if e < 0 else 'positive'}).")
        tail = "" if not conflicts else "\n".join(conflicts) + "\n"
        return (f"Formula: {self.toShownText(f.formulaStr)}\n"
                f"Fitted model (h = 1):  {self.rn} = {intercept:.4g} {eq}\n\n"
                "term                         lag    coefficient   standardized effect\n"
                f"{rows}\n{tail}")

    def parse(self, reply: str, effects: Sequence[TermEffect]) -> Explanation:
        """A reply parsed: the CLAIM lines matched to the terms (by label, shown names mapped back)
        and checked, and the narrative after EXPLANATION:."""
        order = sorted(effects, key=lambda e: -abs(e.effect))
        rankOf = {e.label: i + 1 for i, e in enumerate(order)}
        byLbl = {self.norm(e.label): e for e in effects}
        claims: List[Claim] = []
        for m in _CLAIM.finditer(reply):
            e = byLbl.get(self.norm(self.toRealText(m.group(1))))
            if e is None:
                continue
            sign = 1 if m.group(2) == "+" else -1
            lag, rank = int(m.group(3)), int(m.group(4))
            if not any(c.term == e.label for c in claims):
                claims.append(Claim(e.label, sign, lag, rank, sign == e.sign, lag == e.lag, rank == rankOf[e.label]))
        i = reply.upper().find("EXPLANATION:")
        text = reply.strip() if i < 0 else reply[i + len("EXPLANATION:"):].strip()
        return Explanation(self.toRealText(text), claims, len(effects))

    def explain(self, cand: Candidate, scorer: FormulaScorer) -> Explanation:
        """Explains a scored equation: refit through the scorer (fit rows), its effects read, the
        LLM prompted and the reply checked.  A base model (no formula) gets a note instead."""
        f = cand.formula
        if f is None:
            return Explanation(f"{cand.formulaStr}: a base model, not explained", [], 0)
        mod = scorer.fit(f)
        eff = Explainer.importance(mod, f, scorer.trSize)
        self._last = self.prompt(f, eff, float(mod.intercept[0]))
        self._calls += 1
        reply = self.llm.complete(self.system, self._last)
        return self.parse(reply, eff)

    def shown(self, v: str) -> str:
        return self.toShown.get(v, v)

    def toShownText(self, s: str) -> str:
        return s if self.named else rename(s, self.toShown)

    def toRealText(self, s: str) -> str:
        return s if self.named else rename(s, self.toReal)

    @staticmethod
    def norm(label: str) -> str:
        """A term label without spaces, for matching."""
        return re.sub(r"\s+", "", label)

    @staticmethod
    def variableOf(label: str) -> str:
        """The variable of a single-variable term label, e.g., log1p(icu~2) -> icu."""
        s = _AFFINE.sub("", _FN_PREFIX.sub("", label))
        m = _IDENT.search(s)
        return m.group(0) if m else label

    # -- the companion object -------------------------------------------------------------------

    @staticmethod
    def importance(mod, f, tr: int) -> List[TermEffect]:
        """
        The effect of every term of a fitted formula model at h = 1: the coefficient in original
        units and the standardized effect, coefficient x standard deviation of the term's column on
        the first `tr` rows.
        """
        b = np.asarray(mod.parameter, dtype=float)                        # h = 1 slopes in original units
        x = mod.X
        return [TermEffect(t.label(), t.lag, float(b[j]), float(b[j] * np.std(x[:tr, j], ddof=1)))
                for j, t in enumerate(f.preds)]

    @staticmethod
    def importanceTable(effects: Sequence[TermEffect]) -> str:
        """The importance table of a fitted model: terms by decreasing |standardized effect| with their share of the total."""
        total = sum(abs(e.effect) for e in effects)
        hdr = f"{'term':<28s} {'lag':>3s}   {'coefficient':>12s}   {'std. effect':>12s}   {'share':>6s}"
        rows = [f"{e.label:<28s} {e.lag:3d}   {e.coef:+12.4g}   {e.effect:+12.4g}   {100 * abs(e.effect) / total if total > 0 else 0.0:5.1f}%"
                for e in sorted(effects, key=lambda e: -abs(e.effect))]
        return "\n".join([hdr] + rows)
