"""
Runner: One Forecasting Run from an Args

The one-object style of experiment scripts: every choice of a run -- the dataset, the exogenous
columns, the model, its hyper-parameters, the horizon, the test -- in one `Args`:

    args = Args()
    args.dataset = "Covid-19"
    args.model = "ARX_D"
    args.p, args.px, args.spec = 6, 6, 0
    args.validation = "tnT"
    mod = run(args)                  # loads the data, builds the model, runs the test(s)
    mod.Yf                           # the forecast matrix (original units)

    loadData (args)                  (xe, y) of args.dataset with the exogenous columns args.exo
                                     (None: the dataset's default; []: none), as the loaders return them
    makeModel (args, y, xe)          args.model on your own data: `rescale` when args.tFormT is set
                                     (AR: NormForm, ScalaTion's `AR.rescale`; RandomWalk and ARIMA
                                     are not rescaled), `NARX_SR_D.apply` / `rescale` for NARX_SR_D
    run (args)                       the In-Sample test, the TnT test (rolling validation retraining
                                     every args.rc rows) or both, each on a fresh model; returns the
                                     last model

The models never load data, make folders or plot: `run` does the loading, so a model built by hand,
`ARX_D (y, hh, xe, args)`, behaves exactly as the one `run` builds.  Python only (ScalaTion runs its
models from the test mains of its examples).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from Util.tools import banner
from modeling.forecasting.ar import AR
from modeling.forecasting.arima import ARIMA
from modeling.forecasting.args import Args
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_sr import ARX_SR
from modeling.forecasting.arx_sr_d import ARX_SR_D
from modeling.forecasting.datasets import Example_Covid, Example_ETTh1, Example_GEFCom, Example_ILI
from modeling.forecasting.forecaster import Forecaster
from modeling.forecasting.nonlinear.narx_sr_d import NARX_SR_D
from modeling.forecasting.random_walk import RandomWalk

#: The loader of each dataset (`Args.dataset`) and its default exogenous columns (those of the Examples).
LOADERS = {"Covid-19": Example_Covid, "ILI": Example_ILI, "ETTh1": Example_ETTh1, "GEFCom2014": Example_GEFCom}
DEFAULT_EXO = {"Covid-19": tuple(Example_Covid.BEST2_EXO), "ILI": ("%WEIGHTED ILI", "TOTAL PATIENTS"),
               "ETTh1": tuple(Example_ETTh1.LOADS), "GEFCom2014": ("temp", "tmax")}

#: The model classes `Args.model` names.
MODEL_CLASSES = {"RandomWalk": RandomWalk, "AR": AR, "ARIMA": ARIMA, "ARX": ARX, "ARX_D": ARX_D,
                 "ARX_SR": ARX_SR, "ARX_SR_D": ARX_SR_D, "NARX_SR_D": NARX_SR_D}
NO_EXO = ("RandomWalk", "AR", "ARIMA")                               # the models of y alone


def loadData(args: Any) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    The data of a run: the exogenous columns args.exo (None => the dataset's default; [] => none) and
    the response of args.dataset, as the loaders return them (Covid clipped to its 116 weeks).

    Returns
    -------
    (np.ndarray or None, np.ndarray)
        xe (m, n_exo), None without exogenous columns, and y (m,).

    Raises
    ------
    ValueError
        If args.dataset is not set.
    """
    args = Args.of(args)
    if args.dataset is None:
        raise ValueError("run: set args.dataset (" + " | ".join(LOADERS) + "), or call makeModel (args, y, xe) with your data")
    loader = LOADERS[args.dataset]
    exo = DEFAULT_EXO[args.dataset] if args.exo is None else args.exo
    xe, y = loader.loadData(list(exo), loader.response)
    if loader is Example_Covid:
        xe, y = Example_Covid.clip((xe, y))
    return (xe if xe.shape[1] > 0 else None), y


def makeModel(args: Any, y: np.ndarray, xe: Optional[np.ndarray] = None) -> Forecaster:
    """
    Builds the model args.model on (y, xe) with the settings of args (horizon args.hh): through
    `rescale` when args.tFormT is set (AR: its NormForm; RandomWalk and ARIMA: not rescaled), with
    `NARX_SR_D.apply` / `NARX_SR_D.rescale` (their default transform types) for NARX_SR_D.
    """
    args = Args.of(args)
    name, hh, rescale = args.model, args.hh, args.tFormT is not None
    cls = MODEL_CLASSES[name]
    if name in NO_EXO:
        return AR.rescale(y, hh, args) if name == "AR" and rescale else cls(y, hh, args)
    if name == "NARX_SR_D":
        return NARX_SR_D.rescale(y, hh, xe, args) if rescale else NARX_SR_D.apply(y, hh, xe, args)
    return cls.rescale(y, hh, xe, args) if rescale else cls(y, hh, xe, args)


def run(args: Any) -> Forecaster:
    """
    One forecasting run: loads the data (`loadData`), builds the model (`makeModel`) and makes the
    test(s) of args.validation -- "inSample" (`inSample_Test`), "tnT" (`tnT_Test (args.rc)`) or
    "both" (each on a fresh model) -- printing ScalaTion's output.

    Returns
    -------
    Forecaster
        The model of the last test (its `Yf` holds the forecasts in original units).
    """
    args = Args.of(args)
    xe, y = loadData(args)
    banner(f"run: {args.model} on {args.dataset}")
    print(f"{args}\ny.shape = {y.shape}, xe.shape = {None if xe is None else xe.shape}, hh = {args.hh}")
    mod = None
    if args.validation in ("inSample", "both"):
        mod = makeModel(args, y, xe)
        mod.inSample_Test()
    if args.validation in ("tnT", "both"):
        mod = makeModel(args, y, xe)
        mod.tnT_Test(args.rc)
    return mod
