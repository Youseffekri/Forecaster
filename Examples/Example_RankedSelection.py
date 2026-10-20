"""
Example_RankedSelection: the Ranked technique and the rankings that drive it

The experiment that decides whether the attention ranker earns its ScalaTion port: on the Covid-19
sel-arx library (18 lag columns) and on the ARX_D matrix (19 columns), build the rankings of the
columns on the fit rows -- the standardized coefficients (`coefImportance`), the permutation
importance (`permImportance`) and, when PyTorch is installed, the attention score and the gradient
x input score of `Attn_Ranker` and the paper's `MHAttn_Regressor` as written ("paper") -- then run
the `Ranked` technique with each, and with the t-statistics ("t": elimination by p-value) and minus
the variance inflation factors ("vif"), against Backward and
CEM, under the same selection score J with `Model.validating` off and on, and report for every
run the score, the number of sets evaluated, the chosen set and its rolling TnT sMAPE on the
testing rows; and the Spearman correlation between the rankings.  Decision rule: the attention
ranking earns the port if its path beats the coefficient ranking on test sMAPE.

Run:  python Examples/Example_RankedSelection.py [--seeds 5] [--epochs 100]
"""

from __future__ import annotations

import argparse
import io
import contextlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Util.tools import banner                                            # noqa: E402
from modeling.feature_selection import SelectionTech, qk as QK           # noqa: E402
from modeling.forecasting.args import Args                               # noqa: E402
from modeling.forecasting.arx_d import ARX_D                             # noqa: E402
from modeling.forecasting.datasets import Datasets                       # noqa: E402
from modeling.forecasting.forecaster import Forecaster, Model            # noqa: E402
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import arxFormulaStr   # noqa: E402
from modeling.forecasting.nonlinear.llm_narx_sr.formula_scorer import FormulaScorer   # noqa: E402
from modeling.forecasting.nonlinear.llm_narx_sr.term_library import TermLibrary       # noqa: E402
from modeling.forecasting.nonlinear.narx_formula_parser import NARX_FormulaParser    # noqa: E402

try:
    from modeling.attn_ranker import Attn_Ranker
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def quiet(f: Callable, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return f(*a, **k)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation of two importance vectors (NaN entries dropped)."""
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1]) if ok.sum() > 2 else float("nan")


def rankings_of(x: np.ndarray, y: np.ndarray, coef: np.ndarray, perm: Optional[np.ndarray],
                seeds: int, epochs: int) -> Dict[str, np.ndarray]:
    """The rankings to compare: coefficients, permutation and, with torch, attention and gradient x input."""
    out = {"coef": coef}
    if perm is not None:
        out["perm"] = perm
    if HAVE_TORCH:
        r = Attn_Ranker(x, y, seeds=range(seeds), epochs=epochs)
        r.run()
        out["attn"] = r.scores
        out["grad"] = r.gradScores
        print(f"Attn_Ranker: {r}; final losses {np.round(r.losses, 4)}")
        from modeling.forecasting.neuralforecasting.mhattn_regressor import MHAttn_Regressor
        paper = Attn_Ranker(x, y, seeds=range(seeds), epochs=epochs,                     # the paper's regressor as written
                            regressor=lambda n: MHAttn_Regressor(input_dim=n, d_model=72, num_heads=4))
        out["paper"] = paper.run()
    return out


def library_experiment(seeds: int, epochs: int) -> None:
    """The sel-arx library (18 columns): rankings, Ranked paths, Backward and CEM, with the switch off and on."""
    cfg = Datasets.COVID2
    for validating in (False, True):
        banner(f"sel-arx library on Covid-19, validating = {validating}")
        sc = FormulaScorer(cfg.load(), hh=6, validating=validating)
        maxLag = int(sc.hp["maxLag"])
        f = NARX_FormulaParser.parseFormula(arxFormulaStr(cfg.resp, maxLag, maxLag, list(cfg.exo)), sc.vars, maxLag)
        lib = TermLibrary(f.resp, f.preds, f.tFormT)
        libm = sc.library(lib.resp, lib.terms, lib.tFormT)
        fit = sc.fitRng
        coef = sc.coefImportance(libm, list(range(lib.size)))
        ranks = rankings_of(libm.X[fit.start:fit.stop], libm.y[fit.start:fit.stop], coef, None, seeds, epochs)
        names = list(ranks)
        print("Spearman between rankings:")
        for i, a in enumerate(names):
            print("   " + "  ".join(f"{a}/{b} {spearman(ranks[a], ranks[b]):5.2f}" for b in names[i + 1:]))
        print(f"{'technique':<22s} {'J':>8s} {'sets':>6s}  {'TnT h=1..6':<40s} {'mean':>7s}  equation")
        runs: List[Tuple[str, Dict]] = [("Backward", dict(tech=SelectionTech.Backward)),
                                        ("CEM", dict(tech=SelectionTech.CEM, stream=1)),
                                        ("Ranked coef (RFE)", dict(tech=SelectionTech.Ranked)),
                                        ("Ranked t (p-value)", dict(tech=SelectionTech.Ranked, rank="t")),
                                        ("Ranked vif", dict(tech=SelectionTech.Ranked, rank="vif"))]
        runs += [(f"Ranked {k} (static)", dict(tech=SelectionTech.Ranked, rank=v)) for k, v in ranks.items()]
        for name, kw in runs:
            c, _, nEv = quiet(lib.selectBy, scorer=sc, **kw)
            _, qt = quiet(sc.tnt, c.formula.fresh())
            sm = [q["smape"] for q in qt]
            print(f"{name:<22s} {c.score:8.3f} {nEv:6d}  {' '.join(f'{v:5.1f}' for v in sm):<40s} {np.mean(sm):7.2f}  {c.formulaStr}")


def arx_d_experiment(seeds: int, epochs: int) -> None:
    """The ARX_D matrix (p = 6, px = 6, spec = 1, 19 columns), joint selection over the horizons."""
    cfg = Datasets.COVID2
    data = cfg.load()
    y = data[cfg.resp]
    xe = np.column_stack([data[v] for v in cfg.exo])
    for validating in (False, True):
        banner(f"ARX_D_6_6_2 on Covid-19 (joint selection), validating = {validating}")
        Model.validating = validating
        try:
            mod = ARX_D(y, 6, xe, Args(p=6, px=6, spec=1, lam=1.0, method="ridge"))
            fs = mod.fs_rng
            fs1 = mod.getModel_h(h=1)
            coef = mod.coefImportance(mod.mcols)
            perm = quiet(fs1.permImportance, fs1.mcols, nRep=5)
            ranks = rankings_of(mod.X[fs.start:fs.stop], mod.y[fs.start:fs.stop], coef, perm, seeds, epochs)
            names = list(ranks)
            print("Spearman between rankings:")
            for i, a in enumerate(names):
                print("   " + "  ".join(f"{a}/{b} {spearman(ranks[a], ranks[b]):5.2f}" for b in names[i + 1:]))
            print(f"{'technique':<22s} {'J':>8s} {'steps':>6s} {'k':>3s}  {'TnT mean sMAPE':>14s}  columns")
            runs = [("Backward", dict(tech=SelectionTech.Backward)), ("CEM", dict(tech=SelectionTech.CEM, stream=1)),
                    ("Ranked coef (RFE)", dict(tech=SelectionTech.Ranked)),
                    ("Ranked t (p-value)", dict(tech=SelectionTech.Ranked, rank="t")),
                    ("Ranked vif", dict(tech=SelectionTech.Ranked, rank="vif"))]
            runs += [(f"Ranked {k} (static)", dict(tech=SelectionTech.Ranked, rank=v)) for k, v in ranks.items()]
            for name, kw in runs:
                cols, rSq = quiet(mod.selectFeatures, **kw)
                best = mod.getBest()
                _, _, qt = quiet(mod.reportSelection, cols, rSq, inSample=False, showTrace=False)
                print(f"{name:<22s} {float(best.qof[QK]):8.3f} {len(rSq[0]):6d} {len(best.mod_cols):3d}  "
                      f"{Forecaster.scoreOf(qt, False):14.2f}  {[mod.fname[j] for j in best.mod_cols]}")
        finally:
            Model.validating = False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", type=int, default=5, help="trainings averaged by the attention ranker")
    ap.add_argument("--epochs", type=int, default=100, help="epochs per training")
    a = ap.parse_args()
    if not HAVE_TORCH:
        print("PyTorch is not installed: the attention and gradient rankings are skipped (pip install torch)")
    library_experiment(a.seeds, a.epochs)
    arx_d_experiment(a.seeds, a.epochs)
