"""
Protocol Module: the Measurement Protocol (rolling-origin error tables, paired tests, summary tables)

The measurement protocol of LLM-NARX-SR (the roadmap's Stage 1, item 2.2; rev2.2): every row of a long
test span is a forecast origin, the per-origin and per-horizon errors are kept in ONE long table, and
models are compared on those errors with paired tests -- not on one mean sMAPE per model.

    the long table    one row per (dataset, model, origin t, horizon h): y, yhat, e, ae, sape, ase
                      (`Forecaster.errorTable` after `rollTest` over the testing rows), the testing
                      span set by `te_ratio` (0.4: the last 40 % of a series, refit every rc rows)
    the summary       per dataset, model and horizon (and over all horizons): n origins, sMAPE (the
                      mean sape) and MASE (the mean ase) with a 95 % interval from a HAC standard
                      error (Newey-West, Bartlett weights, lag h - 1: the h-step errors of neighbouring
                      origins overlap)
    the comparisons   model A against model B per dataset and horizon: the mean loss difference, the
                      Diebold-Mariano statistic with the Harvey-Leybourne-Newbold small-sample
                      correction (HAC lag h - 1, Student t with n - 1 degrees of freedom) and the
                      Wilcoxon signed-rank test on the same paired losses (`significance.wilcoxon_test`;
                      `dm_test` and `hacVar` live there too)
    the runner        `Protocol.runModels` (named model factories on a dataset) and `Protocol.runArms`
                      (the arms of the harness: each run's best model rolled over the testing rows,
                      `FormulaScorer.tnt`), the rows appended to the protocol's table; `write` saves
                      errors / summary / comparisons as CSV

MASE here scales by the naive one-step MAE of the rows BEFORE the block (Hyndman & Koehler 2006; the
QoF table's `mase` scales by the block itself, ScalaTion's `Fit.mase`); sMAPE is the QoF table's.
The protocol lives in the `llm_narx_sr` package (it reads the harness and the base, never the
reverse); `Forecaster.errorTable` is the base's.  Python first; its ScalaTion counterpart is a later
revision.

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from Util.tools import banner
from modeling.forecasting.forecaster import Forecaster, Model
from modeling.forecasting.nonlinear.llm_narx_sr.llm_narx_sr import LLM_NARX_SR
from modeling.forecasting.nonlinear.llm_narx_sr.significance import dm_test, hacVar, wilcoxon_test

__all__ = ["Protocol", "ERROR_COLUMNS"]

#: The columns of the long error table.
ERROR_COLUMNS = ("dataset", "model", "stream", "t", "h", "y", "yhat", "e", "ae", "sape", "ase")


def _ci(x: np.ndarray, lag: int, level: float = 0.95) -> Tuple[float, float, float]:
    """The mean of x and its (1 - level) t-interval from the HAC standard error (lag `lag`)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(x.mean())
    if n < 3:
        return mu, float("nan"), float("nan")
    se = math.sqrt(hacVar(x, lag))
    q = float(student_t.ppf(0.5 + level / 2.0, n - 1))
    return mu, mu - q * se, mu + q * se


# -- the protocol -------------------------------------------------------------------------------------

class Protocol:
    """
    The measurement protocol: the long error table of every (dataset, model, origin, horizon), the
    summary tables with intervals and the paired comparisons.

    Parameters
    ----------
    te_ratio : float, default=0.4
        The share of each series rolled over as the testing span (every row of it a forecast origin,
        refit every `rc` rows); `Model.TE_RATIO` is set to it while a run is made and restored after.
    rc : int, optional
        The retraining cycle of the rolling passes (None => `Forecaster.RC`).
    hh : int, default=6
        The maximum horizon.
    loss : {"sape", "ase", "ae"}, default="sape"
        The per-origin loss the comparisons test (sape: the sMAPE's term; ase: the MASE's; ae: |e|).

    Attributes
    ----------
    errors : pd.DataFrame
        The long table (`ERROR_COLUMNS`); every run appends to it.
    """

    def __init__(self, te_ratio: float = 0.4, rc: Optional[int] = None, hh: int = 6, loss: str = "sape") -> None:
        if loss not in ("sape", "ase", "ae"):
            raise ValueError(f"Protocol: loss must be sape, ase or ae, got {loss!r}")
        self.te_ratio = float(te_ratio)
        self.rc = Forecaster.RC if rc is None else int(rc)
        self.hh = int(hh)
        self.loss = loss
        self.errors = pd.DataFrame(columns=list(ERROR_COLUMNS))

    # -- building the table ---------------------------------------------------------------------

    def add(self, mod: Forecaster, rng: range, dataset: str, model: str, stream: int = -1) -> pd.DataFrame:
        """Appends the error table of a model whose forecast matrix holds a rolling pass over `rng`
        (`Forecaster.errorTable`), tagged with the dataset, the model's label and the stream."""
        tab = mod.errorTable(rng)
        tab.insert(0, "stream", int(stream))
        tab.insert(0, "model", model)
        tab.insert(0, "dataset", dataset)
        self.errors = pd.concat([self.errors, tab], ignore_index=True) if len(self.errors) else tab
        return tab

    def runModels(self, dataset: str, y: np.ndarray, xe: Optional[np.ndarray],
                  models: Dict[str, Callable[[np.ndarray, Optional[np.ndarray], int], Forecaster]],
                  verbose: bool = True) -> pd.DataFrame:
        """
        Rolls every named model over the testing span of a series and appends its errors: `models`
        maps a label to a factory (y, xe, hh) -> Forecaster (built with the protocol's te_ratio: the
        factory should pass `Args (te_ratio = ...)` or leave it None, since `Model.TE_RATIO` is set
        for the duration).  Returns the rows added.
        """
        te0 = Model.TE_RATIO
        rows = []
        try:
            Model.TE_RATIO = self.te_ratio
            m = len(y)
            rng = Model.teRng(m)
            for name, make in models.items():
                mod = make(np.asarray(y, dtype=float), xe, self.hh)
                mod.rollTest(rng, self.rc, "TnT", verbose=False)
                tab = self.add(mod, rng, dataset, name)
                rows.append(tab)
                if verbose:
                    print(f"protocol: {dataset:<24s} {name:<28s} {mod.modelName:<28s} origins {rng.start}..{rng.stop - 1}  "
                          f"mean sMAPE {tab['sape'].mean():7.3f}  MASE {tab['ase'].mean():6.3f}")
        finally:
            Model.TE_RATIO = te0
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(ERROR_COLUMNS))

    def runArms(self, cfg: Any, arms: Sequence[Any], streams: Sequence[int] = (0,), hp: Optional[Dict] = None,
                validating: Optional[bool] = None, verbose: bool = True) -> pd.DataFrame:
        """
        Runs the arms of the harness on a dataset configuration (`DatasetConfig`) under the protocol's
        testing span and appends, per run, the errors of the run's best model rolled over the testing
        rows (`FormulaScorer.tnt` at the candidate's knobs; a deterministic arm runs once).  The scorer
        is the harness's (`LLM_NARX_SR.scorerOf`: the switch as `validating`, None => `VALIDATING`).
        Returns the rows added.
        """
        te0 = Model.TE_RATIO
        rows = []
        try:
            Model.TE_RATIO = self.te_ratio
            data = cfg.load()
            h0 = dict(LLM_NARX_SR.hp)
            h0.update(hp or {})
            h0["hh"] = self.hh
            scorer = LLM_NARX_SR.scorerOf(data, h0, validating)
            for arm in arms:
                for s in ([-1] if arm.deterministic else list(streams)):
                    r = arm.run(cfg, scorer, s)
                    if r.failed:
                        print(f"protocol: {cfg.name}: {arm.name} (stream {s}) failed -- skipped")
                        continue
                    if r.best.formula is not None:
                        mod, _ = scorer.tnt(r.best.formula, r.best.hp)
                    else:
                        mod = r.best.model
                        scorer.tnt(mod)
                    tab = self.add(mod, scorer.teRng, cfg.name, r.name, s)
                    rows.append(tab)
                    if verbose:
                        print(f"protocol: {cfg.name:<24s} {r.name:<14s} stream {s:3d}  {r.best.formulaStr[:60]:<60s}  "
                              f"mean sMAPE {tab['sape'].mean():7.3f}  MASE {tab['ase'].mean():6.3f}")
        finally:
            Model.TE_RATIO = te0
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(ERROR_COLUMNS))

    # -- the tables -------------------------------------------------------------------------------

    def summary(self, errors: Optional[pd.DataFrame] = None, level: float = 0.95) -> pd.DataFrame:
        """
        Per dataset, model and horizon (and "all": the mean over the horizons of the per-horizon
        means, the number every chooser compares): the origins n, sMAPE with its HAC t-interval
        (lag h - 1), MASE with its interval.  Streams of one arm are pooled (their origins are the
        same rows; a stochastic arm's spread shows in the interval).
        """
        tab = self.errors if errors is None else errors
        out = []
        for (ds, model), g in tab.groupby(["dataset", "model"], sort=False):
            perH = []
            for h, gh in g.groupby("h"):
                lag = int(h) - 1
                s, slo, shi = _ci(gh["sape"].to_numpy(), lag, level)
                a, alo, ahi = _ci(gh["ase"].to_numpy(), lag, level)
                perH.append((ds, model, int(h), int(gh["t"].nunique()), s, slo, shi, a, alo, ahi))
            out.extend(perH)
            if perH:
                arr = np.array([p[4:] for p in perH], dtype=float)
                n = int(np.mean([p[3] for p in perH]))
                out.append((ds, model, 0, n, *arr.mean(axis=0)))             # 0 => all horizons (the mean of the means)
        cols = ["dataset", "model", "h", "n", "smape", "smape_lo", "smape_hi", "mase", "mase_lo", "mase_hi"]
        return pd.DataFrame(out, columns=cols)

    def compare(self, modelA: str, modelB: str, dataset: Optional[str] = None,
                errors: Optional[pd.DataFrame] = None, loss: Optional[str] = None) -> pd.DataFrame:
        """
        Model A against model B on the PAIRED per-origin losses (the same dataset, origin and
        horizon; streams averaged per origin first), per dataset and horizon and pooled over the
        horizons ("all", lag hh - 1): n pairs, the mean losses, the mean difference A - B, the
        Diebold-Mariano DM* and its p-value, the Wilcoxon W and its p-value.  A negative difference
        with a small p says A forecasts better.
        """
        tab = self.errors if errors is None else errors
        loss = self.loss if loss is None else loss
        if dataset is not None:
            tab = tab[tab["dataset"] == dataset]
        out = []
        for ds, g in tab.groupby("dataset", sort=False):
            a = g[g["model"] == modelA].groupby(["t", "h"])[loss].mean()
            b = g[g["model"] == modelB].groupby(["t", "h"])[loss].mean()
            both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
            if both.empty:
                continue
            groups = [(int(h), both.xs(h, level="h")) for h in sorted(both.index.get_level_values("h").unique())]
            groups.append((0, both))                                      # 0 => pooled over the horizons
            for h, gh in groups:
                la, lb = gh["a"].to_numpy(), gh["b"].to_numpy()
                lag_h = h if h > 0 else self.hh
                dbar, dm, p_dm = dm_test(la, lb, lag_h)
                w, _, p_w = wilcoxon_test(la, lb)
                out.append((ds, modelA, modelB, h, len(la), float(la.mean()), float(lb.mean()), dbar, dm, p_dm, w, p_w))
        cols = ["dataset", "modelA", "modelB", "h", "n", "lossA", "lossB", "diff", "dm", "p_dm", "wilcoxon_w", "p_w"]
        return pd.DataFrame(out, columns=cols)

    def compareAll(self, reference: str, errors: Optional[pd.DataFrame] = None, loss: Optional[str] = None) -> pd.DataFrame:
        """Every other model against the reference model (`compare` for each), one table."""
        tab = self.errors if errors is None else errors
        others = [m for m in tab["model"].unique() if m != reference]
        parts = [self.compare(m, reference, errors=tab, loss=loss) for m in others]
        return pd.concat(parts, ignore_index=True) if parts else self.compare(reference, reference, errors=tab.iloc[0:0])

    # -- output -------------------------------------------------------------------------------------

    def report(self, reference: Optional[str] = None, errors: Optional[pd.DataFrame] = None) -> None:
        """Prints the summary table (sMAPE and MASE with intervals, per horizon and over all) and, with
        a reference model, every model's comparison with it."""
        tab = self.errors if errors is None else errors
        summ = self.summary(tab)
        banner(f"protocol: te_ratio = {self.te_ratio}, rc = {self.rc}, hh = {self.hh}, loss = {self.loss} -- summary (mean over origins, 95 % HAC interval)")
        print(f"{'dataset':<24s} {'model':<28s} {'h':>3s} {'n':>4s} {'sMAPE':>8s} {'[lo':>8s} {'hi]':>8s} {'MASE':>7s} {'[lo':>7s} {'hi]':>7s}")
        for r in summ.itertuples(index=False):
            hs = "all" if r.h == 0 else str(r.h)
            print(f"{r.dataset:<24s} {r.model:<28s} {hs:>3s} {r.n:4d} {r.smape:8.3f} {r.smape_lo:8.3f} {r.smape_hi:8.3f} "
                  f"{r.mase:7.3f} {r.mase_lo:7.3f} {r.mase_hi:7.3f}")
        if reference is not None:
            cmp_ = self.compareAll(reference, tab)
            banner(f"protocol: every model against {reference} on the paired per-origin {self.loss} (Diebold-Mariano with HLN correction; Wilcoxon signed-rank)")
            print(f"{'dataset':<24s} {'model':<28s} {'h':>3s} {'n':>4s} {'lossA':>8s} {'lossRef':>8s} {'diff':>8s} {'DM*':>7s} {'p':>7s} {'W':>8s} {'p':>7s}")
            for r in cmp_.itertuples(index=False):
                hs = "all" if r.h == 0 else str(r.h)
                flag = " *" if r.p_dm < 0.05 else ""
                print(f"{r.dataset:<24s} {r.modelA:<28s} {hs:>3s} {r.n:4d} {r.lossA:8.3f} {r.lossB:8.3f} {r.diff:8.3f} "
                      f"{r.dm:7.3f} {r.p_dm:7.4f} {r.wilcoxon_w:8.1f} {r.p_w:7.4f}{flag}")

    def write(self, directory: str, prefix: str = "protocol", reference: Optional[str] = None) -> List[Path]:
        """Writes the long table, the summary and (with a reference) the comparisons as CSV files
        `<prefix>_errors.csv`, `<prefix>_summary.csv`, `<prefix>_comparisons.csv`; returns the paths."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = [d / f"{prefix}_errors.csv", d / f"{prefix}_summary.csv"]
        self.errors.to_csv(paths[0], index=False)
        self.summary().to_csv(paths[1], index=False)
        if reference is not None:
            paths.append(d / f"{prefix}_comparisons.csv")
            self.compareAll(reference).to_csv(paths[2], index=False)
        return paths
