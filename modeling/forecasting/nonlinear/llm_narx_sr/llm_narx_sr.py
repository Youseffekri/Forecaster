"""
LLM_NARX_SR: LLM-Guided Symbolic Regression over NARX Models for Time-Series Forecasting --
the experiment harness

The method searches the STRUCTURE space of `NARX_Formula` models:

    data -> exo screening -> EDA profile
         -> Loop { proposer -> parse -> fit & score [-> trimmer] -> population + traces }
         -> term library -> selector -> best equation  [-> stable core back to the proposer, next round]
         -> rolling validation on the testing rows -> plausibility, explanation, Pareto front

The proposer is an LLM, a genetic rule or random grammar sampling (`Proposer`); the trimmer
(optional, per proposal) and the selector (over the library of every proposed term) are selection
techniques on term columns (`SelectionTech`, `SubsetSearch`: the cross-entropy method
`CEM_Selector` as the selector, any technique as either).  Every searching arm is one `ArmSpec` run
by `PipelineArm` (`mainArms`, `ablationArms`); the anchors are `BaselineArm` (V0, no search).  This
module holds the hyper-parameters (`hp`, the same dict as `FormulaScorer.hp`), the
client factories, the arm sets (main table and ablation table), runs arms x streams (`runGrid`) and
nested prefixes of the series (`runFolds`), prints the results the way the rest of the project
prints them -- the cross-arm table, and for the winners `showBest`: the REPORT block, the
coefficient tables per horizon, the in-sample fitMap on the training rows and the base's own
`rollValidate` + `diagnoseAll` on the testing rows -- and lists the lineage of a run (`showTraces`).

Row protocol (`FormulaScorer`): ScalaTion's split -- the search learns from the training rows
(`Model.trSize`), the reported number is the base's rolling validation on the testing rows.  LLM
runs are reproducible through `CachedClient` (each stream's transcript is keyed by its prompt, which
carries a "[run <stream>]" tag).  Results dir `llm_narx_sr_results` (one CSV per grid, a row
appended as each run finishes); environment variables `LLM_NARX_SR_DATASET` (the dataset of
`mainTable`), `LLM_NARX_SR_FT_ENDPOINT` / `LLM_NARX_SR_FT_MODEL` (a served fine-tuned proposer),
`OPENAI_API_KEY`; cache dir `llm_cache`.

The Python counterpart of ScalaTion's `llm_narx_sr/LLM_NARX_SR.scala` (rev4.0).  Differences: the
per-horizon coefficient tables of `showBest` are plain tables (ScalaTion prints `summary (h)` with
standard errors), nothing is plotted (ScalaTion's `PlotM` of the convergence and `rollValidate`'s
plot), the dataset is passed explicitly where Scala uses a `given`, and `runGrid` takes `csv=False`
to skip the CSV file.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Util.tools import banner, matrix_str, vector_str
from modeling.feature_selection import SelectionTech
from modeling.llm_client import CachedClient, LocalClient, OpenAIClient
from modeling.forecasting.forecaster import Forecaster
from modeling.forecasting.datasets import DatasetConfig, Datasets
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import BaselineArm
from modeling.forecasting.nonlinear.llm_narx_sr.explainer import Explainer
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import (Candidate, FormulaScorer, qof_table_full,
                                                                      qof_vector)
from modeling.forecasting.nonlinear.llm_narx_sr.pipeline_arm import ClientFactory, PipelineArm
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import ArmSpec, ExpRun, Method, SearchArm
from modeling.forecasting.nonlinear.llm_narx_sr.significance import Significance


class LLMModel(Enum):
    """The cloud models the experiments may call, each with its provider id and its USD per million
    prompt / reply tokens (update from the provider's price list), so a model and its prices cannot come apart."""
    GPT4oMini = ("gpt-4o-mini", 0.15, 0.60)
    GPT4o = ("gpt-4o", 2.50, 10.00)

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def priceIn(self) -> float:
        return self.value[1]

    @property
    def priceOut(self) -> float:
        return self.value[2]

    @property
    def prices(self) -> Tuple[float, float]:
        """The pair the cost column uses."""
        return self.priceIn, self.priceOut


class LLM_NARX_SR:
    """The experiment harness: hyper-parameters, client factories, the arm sets, the grid and fold
    runners, the report and the ARX_D-style display of the winners (ScalaTion's `LLM_NARX_SR` object)."""

    #: The search hyper-parameters (the same dict as `FormulaScorer.hp`, as in ScalaTion, so
    #: `LLM_NARX_SR.hp["maxLag"] = 7` reaches every scorer built afterwards; the split and the switch are
    #: `Model`'s, the retraining cycle `Forecaster.RC`).
    hp: Dict = FormulaScorer.hp

    STREAMS = [0, 1, 2, 3, 4, 5, 6, 7]                                   # random streams / LLM run tags
    LLM_MODEL = LLMModel.GPT4oMini                                       # OpenAI model for the cloud experiments
    CACHE_DIR = "llm_cache"                                              # CachedClient directory (a paper artifact)
    RESULTS_DIR = "llm_narx_sr_results"                                  # one CSV per grid, a row appended as each run finishes
    VERBOSE = True                                                       # the arms' debug lines

    # -- client factories ----------------------------------------------------------------------

    @classmethod
    def cloudLLM(cls, model: LLMModel = None, temp: float = 0.8) -> ClientFactory:
        """A factory for cached cloud clients (OpenAI; the key is read from the environment by
        `OpenAIClient`).  Every call builds a fresh client; the cache directory makes repeated runs free."""
        model = cls.LLM_MODEL if model is None else model
        return lambda: CachedClient(OpenAIClient(model.id, temp), cls.CACHE_DIR)

    @classmethod
    def localLLM(cls) -> Optional[ClientFactory]:
        """A factory for cached local-server clients (e.g., ollama), or None when no server answers."""
        if LocalClient().reachable():
            return lambda: CachedClient(LocalClient(), cls.CACHE_DIR)
        return None

    @classmethod
    def ftLLM(cls) -> Optional[ClientFactory]:
        """A factory for a FINE-TUNED proposer served through an OpenAI-compatible endpoint (vLLM,
        ollama, ...), or None: the endpoint and the model name are read from the environment
        (LLM_NARX_SR_FT_ENDPOINT, LLM_NARX_SR_FT_MODEL) and the server must answer."""
        ep, mdl = os.environ.get("LLM_NARX_SR_FT_ENDPOINT"), os.environ.get("LLM_NARX_SR_FT_MODEL")
        if ep and mdl and LocalClient(ep, mdl).reachable():
            return lambda: CachedClient(LocalClient(ep, mdl), cls.CACHE_DIR, tag="ft")
        return None

    @staticmethod
    def probe(llm: ClientFactory) -> Optional[str]:
        """Tries one small completion on a client factory: None when it answers, else the failure."""
        try:
            reply = llm().complete("Reply with the single word OK.", "connection check")
            return "empty reply" if reply is None else None
        except Exception as e:
            return repr(e)

    # -- arm sets ------------------------------------------------------------------------------

    @classmethod
    def anchors(cls) -> List[SearchArm]:
        """The classical anchors (the ARX structures without search and the random walk, fitted and
        scored by the same engine as every candidate)."""
        return [BaselineArm(Method.Baseline, cls.VERBOSE), BaselineArm(Method.TunedARX, cls.VERBOSE),
                BaselineArm(Method.SelARX, cls.VERBOSE), BaselineArm(Method.RandomWalk, cls.VERBOSE)]

    @classmethod
    def armsOf(cls, specs: Sequence[ArmSpec], llm: Optional[ClientFactory] = None,
               prices: Tuple[float, float] = (0.0, 0.0)) -> List[SearchArm]:
        """The arms of a list of specs on one client factory (`PipelineArm`)."""
        return [PipelineArm(s, llm, prices=prices, verbose=cls.VERBOSE) for s in specs]

    @classmethod
    def zeroLLMArms(cls) -> List[SearchArm]:
        """The zero-LLM arms: the anchors, random grammar sampling, genetic search and the grammar-library CEM."""
        return cls.anchors() + cls.armsOf([ArmSpec.random, ArmSpec.genetic, ArmSpec.statCEM])

    @classmethod
    def llmArms(cls, llm: ClientFactory, prices: Optional[Tuple[float, float]] = None) -> List[SearchArm]:
        """The LLM arms for a client factory: the LLM loop alone (V1), the model llm-cem (V4) and its
        ablations without the EDA block ("blind") and with per-proposal trimming ("trim", V3)."""
        prices = cls.LLM_MODEL.prices if prices is None else prices
        return cls.armsOf([ArmSpec.llmLoop, ArmSpec.llmCEM, ArmSpec.llmCEM.copy(eda=False),
                           ArmSpec.llmCEM.copy(trimmer=SelectionTech.CEM)], llm, prices)

    @classmethod
    def ftArms(cls, llm: ClientFactory) -> List[SearchArm]:
        """The fine-tuned-proposer arms (tag "ft", no API cost): the LLM loop and llm-cem driven by a served fine-tuned model."""
        return cls.armsOf([ArmSpec.llmLoop.copy(tag="ft"), ArmSpec.llmCEM.copy(tag="ft")], llm)

    @classmethod
    def mainArms(cls, llm: Optional[ClientFactory] = None, prices: Optional[Tuple[float, float]] = None) -> List[SearchArm]:
        """The arms of the paper's MAIN table: anchors, stat-cem, genetic and, with a client, llm-loop and llm-cem."""
        prices = cls.LLM_MODEL.prices if prices is None else prices
        arms = cls.anchors() + cls.armsOf([ArmSpec.statCEM, ArmSpec.genetic])
        if llm is not None:
            arms += cls.armsOf([ArmSpec.llmLoop, ArmSpec.llmCEM], llm, prices)
        return arms

    @classmethod
    def ablationArms(cls, llm: Optional[ClientFactory] = None, prices: Optional[Tuple[float, float]] = None) -> List[SearchArm]:
        """The arms of the ABLATION table: random sampling and, with a client, llm-cem without the
        EDA block, with trimming inside the loop, and with two LLM <-> selector rounds."""
        prices = cls.LLM_MODEL.prices if prices is None else prices
        arms = cls.armsOf([ArmSpec.random])
        if llm is not None:
            arms += cls.armsOf([ArmSpec.llmCEM.copy(eda=False), ArmSpec.llmCEM.copy(trimmer=SelectionTech.CEM),
                                ArmSpec.llmCEM.copy(rounds=2)], llm, prices)
        return arms

    # -- runners -------------------------------------------------------------------------------

    #: The harness scores the search on the VALIDATION rows (`Model.validating` for its scorers): the
    #: selection score of the arms is defined on rows never fitted; set False for a series with no
    #: rows to spare (the in-sample FPE-form J).
    VALIDATING = True

    @classmethod
    def scorerOf(cls, dataset: Dict[str, np.ndarray], hp: Optional[Dict] = None,
                 validating: Optional[bool] = None) -> FormulaScorer:
        """The scorer of a dataset (`Model`'s split, the hp settings with optional overrides, the
        switch `validating`: None => `VALIDATING`)."""
        h = dict(cls.hp)
        if hp:
            h.update(hp)
        return FormulaScorer(dataset, int(h["hh"]), hp=h, validating=cls.VALIDATING if validating is None else validating)

    @classmethod
    def runGrid(cls, title: str, cfg: DatasetConfig, arms: Sequence[SearchArm], dataset: Optional[Dict[str, np.ndarray]] = None,
                streams: Optional[Sequence[int]] = None, show: bool = True, explainer: Optional[Explainer] = None,
                csv: bool = True, hp: Optional[Dict] = None) -> List[ExpRun]:
        """
        Runs a grid of arms over streams on one dataset (deterministic arms run once), prints the
        report and, when `show`, the ARX_D-style output (`showBest`) and the traces of the overall
        best run -- the best run of every method is listed in one line each; returns the runs.

        Parameters
        ----------
        title : str
            The experiment title (dataset + configuration).
        cfg : DatasetConfig
            The dataset configuration.
        arms : sequence of SearchArm
            The arms to run.
        dataset : dict, optional
            The named-column map (defaults to `cfg.load ()`).
        streams : sequence of int, optional
            The random streams / run tags (defaults to `STREAMS`).
        show : bool, default=True
            Whether to call `showBest`, `showPareto` and `showTraces` for the winner.
        explainer : Explainer, optional
            The explainer of the winning equation (None => none).
        csv : bool, default=True
            Whether to append the rows to a CSV file in `RESULTS_DIR` as they come.
        hp : dict, optional
            Hyper-parameter overrides for this grid's scorer.
        """
        dataset = cfg.load() if dataset is None else dataset
        streams = cls.STREAMS if streams is None else list(streams)
        scorer = cls.scorerOf(dataset, hp)
        runs: List[ExpRun] = []
        banner(f"{title}: {cfg.name}, exogenous mode {cfg.exoMode}, using {', '.join(scorer.shortlist(cfg))} of {', '.join(cfg.exo)}")
        print(f"split fit / validation / test = {scorer.splitStr} (validating = {scorer.validating}, rc = {scorer.rc}): "
              f"training rows {scorer.trRng} = fit rows {scorer.fitRng} + validation rows {scorer.valRng}; testing rows {scorer.teRng}")
        file = cls.csvFile(title) if csv else None
        for arm in arms:
            for s in ([-1] if arm.deterministic else streams):
                banner(f"{cfg.name}: {arm.name} run, stream = {s}")
                try:
                    r = arm.run(cfg, scorer, s)
                except Exception as e:
                    print(f"ERROR @ LLM_NARX_SR.runGrid: {arm.name} (stream {s}) failed: {e!r} -- recorded as a failed run")
                    r = ExpRun(arm.method, arm.variant, s, scorer.failed(None, 0, arm.name, repr(e)), [], error=repr(e))
                if r.error is not None:
                    print(f"NOTE @ LLM_NARX_SR.runGrid: {arm.name} (stream {s}): {r.error}")
                runs.append(r)
                if file is not None:
                    cls.appendCSV(file, r, len(runs) == 1)
        cls.report(title, runs)
        fitted = [r for r in runs if not r.failed]
        if show and fitted:
            banner(f"{title} -- best run per method (by the selection score)")
            for m, rs in cls.byMethod(fitted):
                b = min(rs, key=lambda r: r.best.score)
                print(f"{m:<14s} stream {b.stream:2d}  score {b.best.score:8.3f}  val sMAPE {b.valSmape:8.3f}  TnT sMAPE {b.testSmape:8.3f}  {b.best.formulaStr}")
            top = cls.bestOf(fitted)
            banner(f"{title} -- overall best: {top.name} (stream {top.stream})")
            cls.showBest(top, scorer, explainer)
            cls.showPareto(top, scorer)
            cls.showTraces(top)
        return runs

    @classmethod
    def csvFile(cls, title: str) -> Path:
        """The CSV file of a grid in `RESULTS_DIR` (created when absent): the title with every
        character outside [A-Za-z0-9._-] replaced by '_', and a time stamp."""
        d = Path(cls.RESULTS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
        return d / f"{re.sub(r'[^A-Za-z0-9._-]', '_', title)}_{stamp}.csv"

    @staticmethod
    def appendCSV(file: Path, r: ExpRun, header: bool) -> None:
        """Appends one run as a CSV row (the header first when `header`): the row label, the stream,
        the validation share and the retraining cycle of the run's scorer, the in-sample, validation
        and TnT sMAPE (per horizon and mean), the terms, free parameters,
        budget columns, the exogenous variables, the error (if any) and the equation.  Never raises."""
        try:
            hh = len(r.smapeTrain)
            q = lambda s: '"' + ("" if s is None else str(s).replace("\n", " ").replace('"', "'")) + '"'
            with open(file, "a", encoding="utf-8") as w:
                if header:
                    w.write(",".join(["method", "stream", "val_ratio", "rc"] + [f"train_h{h}" for h in range(1, hh + 1)] +
                                     ["train_mean", "val_mean"] + [f"tnt_h{h}" for h in range(1, hh + 1)] +
                                     ["tnt_mean", "terms", "nfree", "proposed", "valid",
                                     "failed", "fits", "llm", "ktok", "cost", "exo", "error", "equation"]) + "\n")
                row = ([r.name, str(r.stream), str(r.valRatio), str(r.rc)] + [str(v) for v in r.smapeTrain] +
                       [str(r.best.smape), str(r.valSmape)] + [str(v) for v in r.smapeTest] +
                       [str(v) for v in (r.testSmape, r.best.nTerms, r.best.nFree, r.nProposed, r.nValid, r.nFailed, r.nEval,
                                         r.llmCalls, r.tokens / 1000.0, r.cost)] +
                       [q(" ".join(r.exoUsed)), q(r.error), q(r.best.formulaStr + r.best.why)])
                w.write(",".join(row) + "\n")
        except Exception as e:
            print(f"ERROR @ LLM_NARX_SR.appendCSV: {e}")

    @classmethod
    def runFolds(cls, title: str, cfg: DatasetConfig, arms: Sequence[SearchArm], cuts: Sequence[int],
                 dataset: Optional[Dict[str, np.ndarray]] = None, streams: Optional[Sequence[int]] = None,
                 csv: bool = True, hp: Optional[Dict] = None) -> List[List[ExpRun]]:
        """Runs the grid on nested prefixes of the series: for each cut the series is its first `cut`
        rows, split by `Model.trSize` of its own length, so every fold tests a different stretch of
        time.  Returns the runs per fold (each fold's own results table is printed by `runGrid`, without `showBest`)."""
        dataset = cfg.load() if dataset is None else dataset
        out = []
        for cut in cuts:
            sub = {k: np.asarray(v)[:cut] for k, v in dataset.items()}
            out.append(cls.runGrid(f"{title} [rows 0..{cut - 1}]", cfg, arms, sub, streams, show=False, csv=csv, hp=hp))
        return out

    @staticmethod
    def byMethod(runs: Sequence[ExpRun]) -> List[Tuple[str, List[ExpRun]]]:
        """The runs grouped by row label (method tag + variant), in first-seen order."""
        return [(m, [r for r in runs if r.name == m]) for m in dict.fromkeys(r.name for r in runs)]

    @staticmethod
    def bestOf(runs: Sequence[ExpRun], method: Optional[Method] = None, variant: Optional[str] = None) -> ExpRun:
        """The run with the best selection score (the criterion the search itself used -- never the
        testing rows), over all runs or over one method (and variant)."""
        pool = [r for r in runs if not r.failed and (method is None or r.method is method) and (variant is None or r.variant == variant)]
        if not pool:
            raise ValueError(f"bestOf: no fitted runs{'' if method is None else f' for {method}'}")
        return min(pool, key=lambda r: r.best.score)

    # -- output --------------------------------------------------------------------------------

    @staticmethod
    def toModel(cand: Candidate, scorer: FormulaScorer) -> Forecaster:
        """A candidate as a trained model (forecasts filled, skip set): the base model it holds, or
        its formula fitted by the scorer on ALL the training rows."""
        return scorer.train(cand.model) if cand.model is not None else scorer.fitFull(cand.formula, cand.hp)   # a fresh copy at its knobs

    @classmethod
    def showBest(cls, run: ExpRun, scorer: FormulaScorer, explainer: Optional[Explainer] = None) -> None:
        """
        Shows a run's best model the way the base shows a model: the REPORT block (name,
        hyper-parameters, features, parameters, h = 1 QoF), the coefficient tables per horizon, the
        in-sample fitMap on the training rows, then the base's own `tnT_Test` on the testing rows
        (which prints the TnT fitMap); in between, the term-importance table
        (`Explainer.importance`) and, with an explainer, the LLM's checked explanation.
        """
        banner(f"{run.name} (stream {run.stream}): {run.best.formulaStr}")
        if run.best.failed:
            print("the best candidate failed to fit -- nothing to show")
            return
        mod = cls.toModel(run.best, scorer)
        q = scorer.qofIn(mod, scorer.trRng)                               # the FS rows of the training rows (the base's `qofOn`)
        print(mod.report(qof_vector(q[0])))                               # REPORT block (h = 1 QoF)
        if hasattr(mod, "summary"):
            for h in range(1, scorer.hh + 1):
                mod.qofOn(scorer.fs_rng(scorer.trRng), h)                 # `summary (h)` reads the last diagnose: horizon h's
                print(mod.summary() if h == 1 else mod.summary(h))
        if getattr(mod, "nFree", 0) > 0:
            print(f"fitted nonlinear parameters ({mod.nFree} free, by {mod.optimizer}) = {vector_str(mod.w_nl)}")
        if getattr(mod, "X", None) is not None:
            params = mod.params
            if params.ndim == 2:
                banner("coefficients per horizon (intercept first)")
                print(f"{'term':<28s} " + " ".join(f"{'h=' + str(h):>12s}" for h in range(1, params.shape[1] + 1)))
                for j, nm in enumerate(["intercept"] + list(mod.fname)):
                    print(f"{nm:<28s} " + " ".join(f"{params[j, h]:12.5g}" for h in range(params.shape[1])))
            if run.best.formula is not None:
                banner("term importance at h = 1 (coefficient x std of the term's column on the training rows)")
                print(Explainer.importanceTable(Explainer.importance(mod, run.best.formula, scorer.trSize)))
        if explainer is not None:
            try:                                                          # an unreachable LLM does not end the report
                ex = explainer.explain(run.best, scorer)
                banner(f"the LLM's explanation -- {ex.tableRow}")
                for c in ex.claims:
                    print(c)
                print(ex.text)
            except Exception as e:
                print(f"ERROR @ LLM_NARX_SR.showBest: no explanation -- {e!r}")
        banner(f"In-Sample fitMap QoF on the FS rows of the training rows {scorer.fs_rng(scorer.trRng)} (skip = {scorer.skip})")
        print(qof_table_full(q))
        banner(f"TnT: tnT_Test (rc = {scorer.rc}) on the testing rows {scorer.teRng}")
        mod.tnT_Test(scorer.rc)                                           # rollTest over the testing rows, QoF table printed
        if run.conv:
            print("convergence (best score per round): " + vector_str([c.bestScore for c in run.conv]))

    @staticmethod
    def paretoFront(run: ExpRun) -> List[Candidate]:
        """The complexity-accuracy front of a run: over every equation the run scored (traces, as
        proposed and as kept, the final population, the winner), the best score at each number of
        terms, keeping only the points no smaller equation beats -- fewest terms first."""
        cands: List[Candidate] = []
        seen = set()
        for c in [x for t in run.traces for x in (t.cand, t.candTrim)] + list(run.top) + [run.best]:
            if not c.failed and c.key not in seen:
                seen.add(c.key)
                cands.append(c)
        byK: Dict[int, Candidate] = {}
        for c in cands:
            if c.nTerms not in byK or c.score < byK[c.nTerms].score:
                byK[c.nTerms] = c
        front: List[Candidate] = []
        best = math.inf
        for k in sorted(byK):
            if byK[k].score < best:
                front.append(byK[k])
                best = byK[k].score
        return front

    @classmethod
    def showPareto(cls, run: ExpRun, scorer: FormulaScorer, tnt: bool = True) -> None:
        """Prints the complexity-accuracy front of a run: per point its terms, in-sample sMAPE,
        validation sMAPE and score and, when `tnt`, its TnT sMAPE (one refit per point), so the
        reader sees where the selection criterion placed the winner (*)."""
        front = cls.paretoFront(run)
        banner(f"{run.name} (stream {run.stream}) -- complexity vs accuracy front ({len(front)} points, * = the winner)")
        print(f"{'terms':>5s} {'train sMAPE':>11s} {'val sMAPE':>9s} {'score':>8s} {('TnT sMAPE'.rjust(10) + ' ') if tnt else ''}  equation")
        for c in front:
            te = ""
            if tnt:
                q = scorer.tnt(cls.toModel(c, scorer)) if c.model is not None else scorer.tnt(c.formula, c.hp)[1]
                te = f"{np.mean([x['smape'] for x in q]):10.3f} "
            print(f"{c.nTerms:5d} {c.smape:11.3f} {c.smapeVal:9.3f} {c.score:8.3f} {te}{'*' if c.key == run.best.key else ' '} "
                  f"{c.formulaStr.replace(chr(10), ' ')}")

    @staticmethod
    def resultsTable(runs: Sequence[ExpRun]) -> Tuple[np.ndarray, List[str]]:
        """The cross-arm results table as a matrix with a header (one row per run): in-sample sMAPE
        per horizon and its mean, the validation mean, TnT sMAPE per horizon and its mean, terms,
        budget columns and the number of exogenous variables used."""
        hh = max(len(r.smapeTrain) for r in runs)
        hdr = ([f"train_h{h}" for h in range(1, hh + 1)] + ["train_mean", "val_mean"] +
               [f"tnt_h{h}" for h in range(1, hh + 1)] + ["tnt_mean", "terms", "nfree", "valid", "fail", "fits", "llm", "ktok", "cost", "n_exo"])
        tab = np.full((len(runs), len(hdr)), math.nan)
        for i, r in enumerate(runs):
            tr = np.resize(r.smapeTrain, hh) if len(r.smapeTrain) else np.full(hh, math.nan)
            te = np.resize(r.smapeTest, hh) if len(r.smapeTest) else np.full(hh, math.nan)
            tab[i] = np.concatenate([tr, [r.best.smape, r.valSmape], te,
                                     [r.testSmape, r.best.nTerms, r.best.nFree, r.validity, r.nFailed, r.nEval,
                                      r.llmCalls, r.tokens / 1000.0, r.cost, len(r.exoUsed)]])
        return tab, hdr

    @classmethod
    def report(cls, title: str, runs: Sequence[ExpRun], conv: bool = True) -> None:
        """Prints the results: one line per run (in-sample and TnT sMAPE by horizon, means, terms,
        budget, exogenous variables and the best equation), the per-method means over streams, the
        results table and the convergence tables."""
        hh = max(len(r.smapeTrain) for r in runs)
        banner(f"{title} -- results (in-sample on the fit rows | validation | TnT rolling on the testing rows)")
        hHdr = " ".join(f"{'h=' + str(h):>7s}" for h in range(1, hh + 1))
        print(f"{'method':<14s} {'strm':>4s} | {hHdr} {'mean':>7s} | {'val':>7s} | {hHdr} {'mean':>7s} | {'terms':>5s} {'free':>4s} {'valid':>6s} "
              f"{'fail':>4s} {'fits':>6s} {'LLM':>4s} {'ktok':>6s} {'cost':>7s}  exo  best equation")
        for r in runs:
            tr = " ".join(f"{v:7.3f}" for v in np.resize(r.smapeTrain, hh)) if len(r.smapeTrain) else " ".join(f"{'--':>7s}" for _ in range(hh))
            te = " ".join(f"{v:7.3f}" for v in np.resize(r.smapeTest, hh))
            vd = f"{'--':>6s}" if r.nProposed == 0 else f"{r.validity * 100:5.1f}%"
            if r.failed:
                eq = f"FAILED: {r.error if r.error is not None else r.best.reason}"
            elif r.noImprovement:
                eq = f"no improvement over the seed ({r.best.formulaStr.replace(chr(10), ' ')})"
            else:
                eq = r.best.formulaStr.replace("\n", " ")
            print(f"{r.name:<14s} {r.stream:4d} | {tr} {r.best.smape:7.3f} | {r.valSmape:7.3f} | {te} {r.testSmape:7.3f} | "
                  f"{r.best.nTerms:5d} {r.best.nFree:4d} {vd} {r.nFailed:4d} {r.nEval:6d} {r.llmCalls:4d} {r.tokens / 1000.0:6.1f} {r.cost:7.4f}  "
                  f"{','.join(r.exoUsed)}  {eq}")

        banner(f"{title} -- per-method means over streams (fitted runs)")
        print(f"{'method':<14s} {'n':>3s} | {'train':>7s} {'val':>7s} {'tnt':>7s} | {hHdr} (tnt) | {'terms':>6s} {'fits':>8s} {'LLM':>6s} {'cost':>8s}")
        for m, rs0 in cls.byMethod(runs):
            rs = [r for r in rs0 if not r.failed]                        # failed runs are left out of the means
            if not rs:
                print(f"{m:<14s} {0:3d} | every run failed")
                continue
            n = len(rs)
            teH = " ".join(f"{np.mean([r.smapeTest[h] for r in rs]):7.3f}" for h in range(hh))
            print(f"{m:<14s} {n:3d} | {np.mean([r.best.smape for r in rs]):7.3f} {np.mean([r.valSmape for r in rs]):7.3f} "
                  f"{np.mean([r.testSmape for r in rs]):7.3f} | {teH} | {np.mean([r.best.nTerms for r in rs]):6.1f} "
                  f"{np.mean([r.nEval for r in rs]):8.1f} {np.mean([r.llmCalls for r in rs]):6.1f} {np.mean([r.cost for r in rs]):8.4f}")

        tab, hdr = cls.resultsTable(runs)
        banner(f"{title} -- results table (MatrixD, header: {', '.join(hdr)})")
        print(matrix_str(tab))

        if conv:
            banner(f"{title} -- convergence (best-so-far per round)")
            for r in runs:
                if r.conv:
                    print(f"# {r.name} stream={r.stream}")
                    print("round  proposed  bestScore  bestSMAPE")
                    for c in r.conv:
                        print(f"{c.round:5d}  {c.nProposed:8d}  {c.bestScore:9.3f}  {c.bestSmape:9.3f}")

    @staticmethod
    def showTraces(run: ExpRun, k: int = -1) -> None:
        """Prints the lineage of a run: every proposal in order (round, source, trimmer, score of f
        -> score of fTrim, f -> fTrim with the dropped terms), and the origin of the winner."""
        banner(f"{run.name} (stream {run.stream}) -- traces ({len(run.traces)} entries)")
        print("rnd  source    trimmer      f score ->  fTrim score  proposed  [->  kept  (dropped)]")
        for t in (run.traces if k < 0 else run.traces[:k]):
            print(t)
        t = run.originOf(run.best)
        if t is not None:
            print(f"winner {run.best.formulaStr}: first proposed in round {t.round} by {t.source}" +
                  (f", trimmed by {t.trimmer} from {t.f.formulaStr}" if t.trimmed else "") +
                  f"; reached {len(run.tracesOf(run.best))} time(s)")
        else:
            print(f"winner {run.best.formulaStr}: no trace (a base model)")

    # -- the main table on a dataset chosen by the environment (ScalaTion's `lLM_NARX_SRTest9`) ------

    @classmethod
    def datasetFromEnv(cls, default: str = "ili") -> DatasetConfig:
        """The dataset configuration named by LLM_NARX_SR_DATASET: ili (default), covid2, etth1 (first year) or gef14."""
        which = os.environ.get("LLM_NARX_SR_DATASET", default).strip().lower()
        table = {"covid2": Datasets.COVID2, "etth1": Datasets.ETTH1_1Y, "gef14": Datasets.GEF14, "ili": Datasets.ILI}
        if which not in table:
            print(f"LLM_NARX_SR_DATASET = {which} not known (ili, covid2, etth1, gef14) -- using ili")
            return Datasets.ILI
        return table[which]

    @classmethod
    def mainTable(cls, cfg: Optional[DatasetConfig] = None, streams: Sequence[int] = (0, 1, 2, 3, 4),
                  maxFits: int = 100) -> List[ExpRun]:
        """
        The MAIN TABLE (every arm, the LLM ones with a probed cached cloud client, or a local server
        when one answers and no key is set) on a dataset (by default the one LLM_NARX_SR_DATASET
        names); GEFCom2014 runs with lags 1..7; the fit budget of every loop is `maxFits`, so the LLM
        is compared at an equal number of model fits; the stream-paired significance report follows.
        The hp entries touched are restored at the end.
        """
        cfg = cls.datasetFromEnv() if cfg is None else cfg
        try:
            data = cfg.load()
        except Exception as e:
            print(f"{cfg.name}: data file not available -- skipping: {e}")
            return []
        candidate: Optional[Tuple[ClientFactory, Tuple[float, float], str]] = None
        if os.environ.get("OPENAI_API_KEY"):
            candidate = (cls.cloudLLM(), cls.LLM_MODEL.prices, f"OpenAI {cls.LLM_MODEL.id}")
        else:
            loc = cls.localLLM()
            if loc is not None:
                candidate = (loc, (0.0, 0.0), "local model server")
        llm = None
        if candidate is None:
            print("no OPENAI_API_KEY and no local model server -- running the zero-LLM arms only")
        else:
            why = cls.probe(candidate[0])
            if why is None:
                print(f"LLM client: {candidate[2]} answers")
                llm = (candidate[0], candidate[1])
            else:
                print(f"LLM client: {candidate[2]} does NOT answer -- {why} -- running the zero-LLM arms only")
        saved = {k: cls.hp[k] for k in ("maxLag", "maxFits")}
        try:
            if cfg is Datasets.GEF14:
                cls.hp["maxLag"] = 7                                      # a daily series: lags 1..7 (and the skip)
            cls.hp["maxFits"] = maxFits
            prices = llm[1] if llm is not None else (0.0, 0.0)
            explainer = Explainer(llm[0](), cfg) if llm is not None else None
            title = f"{cfg.name} (main table)"
            main = cls.runGrid(title, cfg, cls.mainArms(llm[0] if llm else None, prices), data, streams, explainer=explainer)
            Significance.reportSignificance(title, main, Method.LLMCEM if llm is not None else Method.Genetic)
            top = cls.bestOf(main, Method.LLMCEM if llm is not None else Method.StatCEM)
            banner(f"{top.name}: best stream {top.stream}, in-sample {top.best.smape:.3f}, validation {top.valSmape:.3f}, TnT {top.testSmape:.3f}")
            print(top.best.formulaStr)
            return main
        finally:
            cls.hp.update(saved)
