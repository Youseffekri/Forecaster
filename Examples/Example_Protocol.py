"""
Example_Protocol: the "before" tables of the measurement protocol (rev2.2, roadmap Stage 1 item 2.2)

Every row of the last 40 % of a series is a forecast origin (refit every rc = 5 rows, h = 1..6); the
per-origin errors of the deterministic arms (baseline, tuned-arx, sel-arx, random-walk: the harness's
anchors under the validation J) and of the base models (ARX_D and ARX at the ridge lambda = 1, AR_6,
the random walk) go into ONE long table per dataset; the summary (sMAPE and MASE per horizon with a
95 % HAC interval) and every model's Diebold-Mariano / Wilcoxon comparison with the random walk are
printed and written as CSV (results/protocol/<dataset>_{errors,summary,comparisons}.csv).  These are
the tables every later stage is measured against.

Run from the repository root:  python Examples/Example_Protocol.py [covid|ili|gef14|etth1|all] [--arms] [--out DIR]
(the arms are run only with --arms: sel-arx's backward selection rolls every column set over the
validation block, minutes on ILI and GEFCom, longer on ETTh1).

Author: Yousef Fekri Dabanloo
Date: 2026-10-20
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Util.tools import banner
from modeling.forecasting.ar import AR
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.datasets import Datasets
from modeling.forecasting.nonlinear.llm_narx_sr.baseline_arm import BaselineArm
from modeling.forecasting.nonlinear.llm_narx_sr.protocol import Protocol
from modeling.forecasting.nonlinear.llm_narx_sr.search_arm import Method
from modeling.forecasting.random_walk import RandomWalk

HH, RC, TE_RATIO = 6, 5, 0.4
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "protocol")

DATASETS = {"covid": (Datasets.COVID2, 6), "ili": (Datasets.ILI, 6), "gef14": (Datasets.GEF14, 7), "etth1": (Datasets.ETTH1_1Y, 6)}


def baseModels(maxLag: int):
    """The base models of the "before" table, built on the series (y, xe, hh): the two ridge ARX
    structures at lambda = 1 (the harness's penalty, raw columns), the recursive AR and the random walk."""
    return {
        f"ARX_D_{maxLag}_{maxLag} ridge L1": lambda y, xe, hh: ARX_D(y, hh, xe, Args(p=maxLag, px=maxLag, spec=1, lam=1.0, method="ridge")),
        f"ARX_{maxLag}_{maxLag} ridge L1":   lambda y, xe, hh: ARX(y, hh, xe, Args(p=maxLag, px=maxLag, spec=1, lam=1.0, method="ridge")),
        f"AR_{maxLag}":                       lambda y, xe, hh: AR(y, hh, Args(p=maxLag)),
        "RandomWalk":                         lambda y, xe, hh: RandomWalk(y, hh),
    }


def before(which: str, arms: bool = False, out: str = OUT) -> Protocol:
    """The "before" table of one dataset: the base models, then (with `arms`) the deterministic arms."""
    cfg, maxLag = DATASETS[which]
    data = cfg.load()
    y = data[cfg.resp]
    xe = np.column_stack([data[v] for v in cfg.exo]) if cfg.exo else None
    banner(f"protocol 'before' table: {cfg.name} ({len(y)} rows, exo {cfg.exo}), te_ratio = {TE_RATIO}, rc = {RC}, hh = {HH}")
    P = Protocol(te_ratio=TE_RATIO, rc=RC, hh=HH)
    t0 = time.time()
    P.runModels(cfg.name, y, xe, baseModels(maxLag))
    if arms:
        kinds = (Method.Baseline, Method.TunedARX, Method.SelARX, Method.RandomWalk)
        P.runArms(cfg, [BaselineArm(k, verbose=False) for k in kinds], hp={"maxLag": maxLag})
    P.report(reference="RandomWalk")
    paths = P.write(out, which, reference="RandomWalk")
    print(f"written: {', '.join(os.path.relpath(str(p)) for p in paths)}  ({time.time() - t0:.1f} s)")
    return P


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0].lower() if args else "covid"
    arms = "--arms" in sys.argv
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT
    for w in (list(DATASETS) if which == "all" else [which]):
        before(w, arms, out)
