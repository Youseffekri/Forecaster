"""
LLMProposer: the LLM Proposer of LLM-NARX-SR

Prompts an LLM for formula strings and parses the reply.  What the LLM sees, by default (the
contamination policy of the paper): the variables under BLIND names (y, x1, x2, ...), a generic
dataset sentence ("a weekly series y with 2 covariates"), the grammar with a note that the
intercept is implicit and, when `nonlinear`, the free-parameter forms of `NARX_FormulaParser`
(pow(x1~1, ?), log(? + x1~1), sat(x1~1, ?), ...: a ? is a parameter the engine fits on the training
rows; a fitted equation is shown back with its values as start values, ?w), optionally the EDA
profile of the training rows in words (`LLMProposer.promptBlock` of an `EDA4TS`; over the whole pool of candidate
variables when the system picks the exogenous variables), the scored population (each equation
with its validation and in-sample sMAPE; when a trimmer ran, the proposal, the kept equation and
the dropped terms), last round's rejects with their reasons and, when a cross-entropy stage ran,
the stable core of terms it found.  It never sees data rows, the dataset's name, the real variable
names or anything from the testing rows.  `named = True` restores the domain sentence and the real
names (the contamination ablation).

LLMs wrap output in prose or code fences despite instructions, so lines are kept by grammar prefix
("y ~=", plus a following "TFORM:" line) rather than trusting the format; a line that does not
parse is returned as a reject with its reason, which the search counts as invalid and shows back
next round.

The Python counterpart of ScalaTion's `llm_narx_sr/LLMProposer.scala` (the prompt texts are kept
verbatim).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modeling.llm_client import LLMClient
from modeling.forecasting.datasets import DatasetConfig
from modeling.forecasting.eda4ts import EDA4TS
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import DEFAULT_HP
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import Source, Trace
from modeling.forecasting.nonlinear.llm_narx_sr.search_loop import Proposal, Proposer, Reject

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")                           # identifier pattern


def rename(s: str, mapping: Dict[str, str]) -> str:
    """Renames the identifiers of a text through a map (identity for names not in it)."""
    if not mapping:
        return s
    return _IDENT.sub(lambda m: mapping.get(m.group(0), m.group(0)), s)


def _g3(v: float) -> str:
    """Three significant digits as Java's "%.3g" (trailing zeros kept)."""
    return "%#.3g" % v


def maxFreeShown(maxFree: Optional[int] = None) -> str:
    """The free-parameter cap as the prompt states it (hp["maxFree"]; "any number" when there is no cap)."""
    k = int(DEFAULT_HP["maxFree"]) if maxFree is None else int(maxFree)
    return "any number" if k < 0 else str(k)


def genericSchema(nExo: int, freq: str = "weekly") -> str:
    """The generic dataset sentence: the sampling frequency and the number of covariates, nothing
    that identifies the series."""
    covs = "no covariates" if nExo == 0 else ("one covariate x1" if nExo == 1 else f"{nExo} covariates x1 .. x{nExo}")
    return f"Dataset: a {freq} series y with {covs}; all values are non-negative counts or rates."


class LLMProposer(Proposer):
    """
    Prompts an LLM for formulas (ScalaTion's `LLMProposer`).

    Parameters
    ----------
    llm : LLMClient
        The completion client (wrap in `CachedClient` / `CountingClient` for experiments).
    cfg : DatasetConfig
        The dataset configuration (variables, generic / domain sentence).
    vars_ : sequence of str
        The legal variable names (the dataset's keys).
    maxLag : int
        The maximum lag advertised in the grammar.
    freqs : sequence of float, optional
        The wave frequencies (None or empty => omit wave).
    eda : EDA4TS, optional
        The EDA profile whose prompt block is shown (None => none).
    named : bool, default=False
        Whether the real variable names and the domain sentence are shown.
    tag : str, default=""
        A text folded into the prompt (e.g., "[run 3]") so a cache keyed on the prompt gives each run its own transcript.
    nonlinear : bool, optional
        Whether the grammar shown includes the free-parameter forms (defaults to hp["nlTerms"] = 1).
    maxFree : int, optional
        The free-parameter cap stated in the prompt (defaults to hp["maxFree"]).
    """

    def __init__(self, llm: LLMClient, cfg: DatasetConfig, vars_: Sequence[str], maxLag: int,
                 freqs: Optional[Sequence[float]], eda: Optional[EDA4TS] = None, named: bool = False, tag: str = "",
                 nonlinear: Optional[bool] = None, maxFree: Optional[int] = None) -> None:
        self.llm, self.cfg, self.vars, self.maxLag = llm, cfg, list(vars_), int(maxLag)
        self.freqs = [] if freqs is None else list(freqs)
        self.eda, self.named, self.tag = eda, named, tag
        self.nonlinear = (int(DEFAULT_HP["nlTerms"]) == 1) if nonlinear is None else bool(nonlinear)
        self.nFreqs = len(self.freqs)
        self.toShown: Dict[str, str] = {} if named else dict(cfg.blindNames)   # real -> shown name
        self.toReal: Dict[str, str] = {v: k for k, v in self.toShown.items()}  # shown -> real name
        self.rejects: List[Tuple[str, str]] = []                          # last round's (formulaStr, reason)
        self.core: List[str] = []                                         # stable-core terms from a CEM stage
        self.failedProps: List[Tuple[str, str]] = []                      # last round's fitted-badly / rejected proposals
        self.notes = ""                                                   # diagnostic notes on the current best
        self._calls = 0
        self._last = ""
        self.rn = self.shown(cfg.resp)                                    # names as the LLM sees them
        self.x1 = self.shown(cfg.exo[0]) if cfg.exo else self.rn
        self.x2 = self.shown(cfg.exo[1]) if len(cfg.exo) > 1 else self.x1
        self.system = self._system(maxFreeShown(maxFree))

    @property
    def source(self) -> Source:
        return Source.LLM

    def _system(self, cap: str) -> str:
        """The system prompt: task, variables, grammar, rules, examples in the shown names."""
        rn, x1, x2 = self.rn, self.x1, self.x2
        about = self.cfg.schema if self.named else genericSchema(len(self.cfg.exo), self.cfg.freq)
        waveEx = f"\n  {rn} ~= {rn}~(2,1) + {x1}~1 + wave(1)" if self.nFreqs > 0 else ""
        waveRl = f" wave(k) with 1 <= k <= {self.nFreqs} adds k seasonal sin/cos pairs." if self.nFreqs > 0 else ""
        nlEx = f"\n  {rn} ~= {rn}~(2,1) + pow({rn}~1, ?) + log(? + {x1}~1) + sat({x2}~1, ?)" if self.nonlinear else ""
        nlRl = "\n" + NARX_FormulaParser.grammarLines(x1) if self.nonlinear else ""
        nlCap = (f"  A free parameter (?) costs as much as a term and at most {cap} are allowed per formula; "
                 "a fit that ends on a parameter's bound is rejected.") if self.nonlinear else ""
        names = ", ".join(self.shown(v) for v in [self.cfg.resp] + list(self.cfg.exo))
        return (f"You propose formulas for a symbolic time-series model that forecasts {rn}.\n"
                f"{about}\n"
                f"Variables: {names}.  VAR~k is the value of VAR k steps back, 1 <= k <= {self.maxLag}.\n"
                "The model is linear in its terms with an implicit intercept: never write a constant term.\n"
                "Write one formula per line, nothing else on the line, exactly in this style:\n"
                f"  {rn} ~= {rn}~1 + {rn}~2 + {x1}~1\n"
                f"  {rn} ~= {rn}~(3,1) + log1p({x1}~2) + sq({x2}~1)\n"
                f"  {rn} ~= {rn}~1 + pow({x1}~1, 1.5) + {rn}~1 * {x1}~1{waveEx}{nlEx}\n"
                "Allowed terms: VAR~k, VAR~(max,min) (all lags from max down to min), log1p(term), log(term), sq(term),\n"
                "exp(term), pow(term, p) and powr(term, p) with 0.2 <= p <= 3.0, a product of exactly two single\n"
                f"lagged variables written like {rn}~1 * {x1}~2 (never a range, a function or wave inside a product).{waveRl}{nlRl}\n"
                "Use the variable names exactly as given; never write VAR or term literally.\n"
                "Non-negative data: log(0) is not finite, so prefer log1p.  Formulas with 3 to 8 terms do best\n"
                f"(each term costs a small penalty).{nlCap}\n"
                "Optionally end a formula with a second line:  TFORM: Log1p\n"
                "Each proposal should test ONE idea -- one new term, one changed lag, one transform -- against the best\n"
                "formulas shown, so that what helped can be told from what did not; the ideas should differ between proposals.\n"
                "Formulas are judged by their forecast error on rows they were NOT fitted on (a validation block at the end\n"
                "of the training rows, all horizons), so prefer parsimonious structures that generalize over close fits.\n"
                "Propose formulas likely to have LOWER validation sMAPE than those shown.\n")

    @property
    def calls(self) -> int:
        """The number of prompts sent so far."""
        return self._calls

    @property
    def lastPrompt(self) -> str:
        """The last user prompt sent (for inspection and the paper's appendix)."""
        return self._last

    def setCore(self, labels: Sequence[str]) -> None:
        """Tells the proposer which terms a cross-entropy stage found stable (shown in the next
        prompts as the core to build on; see `PipelineArm`)."""
        self.core = list(labels)

    def feedback(self, failed: Sequence[Tuple[str, str]], notes: str) -> None:
        """Keeps the round's failed/rejected proposals (shown next round with their reasons, as the
        parse rejects are) and the diagnostic notes on the best."""
        self.failedProps = list(failed)
        self.notes = notes

    def propose(self, top: Sequence[Trace], n: int) -> List[Proposal]:
        """
        n formulas from the LLM, prompted with the scored population, last round's rejects and
        failures, the diagnostic notes on the best, and the optional EDA block and core; the reply
        parsed.  Context lines are prefixed with "# " so a model that echoes them back is filtered
        out rather than mis-parsed.  A trimmed trace is shown as "proposed f -> kept fTrim
        (dropped: ...)" with the kept equation's score.
        """
        def line(t: Trace) -> str:                                        # a fitted equation as a template: ?w slots
            c = t.candTrim
            head = (f"# validation sMAPE = {c.smapeVal:7.2f}  (in-sample {c.smape:7.2f})  by horizon = {self._byH(c.smapeValH)}"
                    f"  terms = {c.nTerms:2d}  :  ")
            if t.trimmed:
                return (head + f"proposed {self.toShownText(t.cand.proposalStr)}  ->  kept {self.toShownText(c.proposalStr)}"
                        f"  (dropped: {', '.join(self.toShownText(d.label()) for d in t.dropped)})")
            return head + self.toShownText(c.proposalStr)

        shownTop = "# (none yet)" if not top else "\n".join(line(t) for t in top)
        profile = "" if self.eda is None else self.toShownText(LLMProposer.promptBlock(self.eda)) + "\n"
        coreTxt = "" if not self.core else \
            f"Terms found stable by the selection stage (build on them): {', '.join(self.toShownText(c) for c in self.core)}\n"
        bad = "" if not (self.rejects or self.failedProps) else \
            "Rejected last round -- do NOT repeat these:\n" + \
            "\n".join(f"# {self.toShownText(s).replace(chr(10), ' ')}  -- {self.toShownText(r)}" for s, r in self.rejects + self.failedProps) + "\n"
        diag = "" if not self.notes else f"# {self.toShownText(self.notes)}\n"
        user = (f"{self.tag}\n"
                f"{profile}Best formulas so far (lower validation sMAPE is better; do NOT repeat these, they are EXAMPLES not proposals):\n"
                f"{shownTop}\n"
                f"{diag}{coreTxt}{bad}Propose {n} NEW, DIVERSE formulas, each testing one idea.  One per line, grammar only, no prose, no numbering.\n")
        self._last = user
        raw = self.llm.complete(self.system, user)
        self._calls += 1

        lines: List[str] = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln.startswith("```"):
                ln = ln[3:].strip()
            if not ln:
                continue
            if ln.startswith(f"{self.rn} ~="):
                lines.append(self.toRealText(ln))
            elif ln.upper().startswith("TFORM:") and lines:
                lines[-1] = lines[-1] + " " + ln                           # attach to the preceding formula
        rej: List[Tuple[str, str]] = []
        out: List[Proposal] = []
        for s in lines[:n]:
            f, reason = NARX_FormulaParser.tryParseFormula(s, self.vars, self.maxLag, self.freqs)
            if f is not None:
                out.append(f)
            else:
                why = (reason or "").replace("parse: ", "", 1).split(" in formula: ")[0]
                rej.append((s, why))
                out.append(Reject(f"{s} -- {why}"))
        self.rejects = rej                                                # remembered for the next prompt
        return out

    @staticmethod
    def promptBlock(eda: EDA4TS, names: Optional[Dict[str, str]] = None) -> str:
        """
        Renders an EDA profile in words for a prompt (ScalaTion's `LLMProposer.promptBlock`, moved from
        `EDA4TS`).  Variables are shown under the given display names (the real names when absent;
        the proposer passes its blind map through `toShownText`), nothing identifies the series, and
        only training-row statistics appear.

        Parameters
        ----------
        eda : EDA4TS
            The profile of the training rows.
        names : dict of str -> str, optional
            The display name per variable.
        """
        names = {} if names is None else names

        def nm(v: str) -> str:
            return names.get(v, v)

        yn = nm(eda.respName)
        pac = ", ".join(f"{k} ({r:.2f})" for k, r in eda.pacfSpikes()[:3])
        sb = ["Data profile (training rows only):\n",
              f"- {yn}: level {_g3(eda.yMean)}, spread {_g3(eda.yStd)}, range {_g3(eda.yMin)} to {_g3(eda.yMax)}" +
              (f", {eda.zeroShare * 100:.0f}% zeros (log is unsafe, prefer log1p)" if eda.zeroShare > 0.0 else "") + "\n",
              f"- {yn} autocorrelation at lags 1..{eda.maxLag}: " +
              ", ".join(f"{eda.acf[k]:.2f}" for k in range(1, eda.maxLag + 1)) + "\n",
              f"- {yn} partial autocorrelation spikes (lag (value)): " + (pac if pac else "none") + "\n"]
        if len(eda.freqs) > 0:
            sb.append("- seasonality: strongest periods " +
                      ", ".join(f"{eda.periods[i]:.1f} steps (wave({i + 1}))" for i in range(len(eda.freqs))) + "\n")
        for v, k, r in eda.bestLags():
            sb.append(f"- {nm(v)}: strongest correlation with {yn} at lag {k} (r = {r:.2f}" +
                      (", weak)" if abs(r) <= eda.band else (", positive)" if r > 0.0 else ", negative)")) + "\n")
        if not math.isnan(eda.kpssP):
            sb.append(f"- KPSS level-stationarity p-value {eda.kpssP:.2f}" +
                      (f" (non-stationary: lags of {yn} matter)" if eda.kpssP < 0.05 else " (stationary)") + "\n")
        return "".join(sb)

    def shown(self, v: str) -> str:
        """The shown name of a variable (blind unless `named`)."""
        return self.toShown.get(v, v)

    def toShownText(self, s: str) -> str:
        """The real variables renamed to the shown names in a text (identity when `named`)."""
        return s if self.named else rename(s, self.toShown)

    def toRealText(self, s: str) -> str:
        """The shown names renamed back to the real variables in a text (identity when `named`)."""
        return s if self.named else rename(s, self.toReal)

    @staticmethod
    def _byH(sH: np.ndarray) -> str:
        """An sMAPE per horizon vector for the prompt, e.g., [11.4, 12.4]."""
        return "[" + ", ".join(f"{v:.1f}" for v in np.asarray(sH, dtype=float)) + "]"


LLMProposer.genericSchema = staticmethod(genericSchema)
LLMProposer.maxFreeShown = staticmethod(maxFreeShown)
