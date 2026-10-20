"""
Args: the Settings of a Forecasting Model (and of a Run)

One object holds everything a model is configured with -- the hyper-parameters of ScalaTion's `AR.hp`
(p, d, q, P, D, Q, s, px, spec, crx, nneg), the Python-only ones (skip, lam), the model settings
(method, scaled, l1_ratio, optimizer, tune, te_ratio) and the settings of a run (`run (args)`: dataset, exo,
model, tFormT, hh, validation, rc) -- the dotdict style of experiment scripts, but CHECKED:

    args = Args()                       # every setting at its default
    args.p = 6                          # checked when set: name, type and range
    args.px = 4
    mod = ARX_D(y, hh, xe, args)        # the data in the call, the settings in args

    args.pX = 4      -> AttributeError: Args has no setting 'pX'; did you mean 'px'?
    args.p = -1      -> ValueError: Args.p = -1: must be an integer >= 0
    args.help()      -> the table of the settings (value, default, kind, meaning; * = changed)
    Args.help(ARX_D) -> the settings ARX_D reads
    args2 = args.copy(px=2)             # a variant (args is unchanged); Args(p=6, px=4) also works
    args.save("covid.json"); Args.load("covid.json")   # a preset (the changed settings)

A model takes a private SNAPSHOT of its args (`Args.of`), so later changes to args do not reach a
model that exists; its hyper-parameters (`hparameter`, a copy) are `AR.hp` with the values of args
(`Args.hparam`), printed in the REPORT as ScalaTion prints the hp it is given.  `AR.hp` (`hp` here,
re-exported by `forecaster.py`) is read-only, so no model or script changes the defaults of the
models made after it.  A setting a model does not read is ignored by it, so one args can drive a
RandomWalk, an ARIMA and an ARX_D side by side.

Python only: ScalaTion passes `HyperParameter` objects (`hp("p") = 6`); `lam` is ScalaTion's
`RidgeRegression.hp("lambda")` ("lambda" is a Python keyword: `Args (**{"lambda": 1.0})` is accepted
as an alias, the HyperParameter keeps the name "lambda"); `te_ratio` is ScalaTion's global
`Model.TE_RATIO`, here per model (None, the default: `Model.TE_RATIO` when the model is built;
0: no testing set, a competition's dataset).

Author: Yousef Fekri Dabanloo
Date: 2026-10-19
Version: 1.8
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import difflib
import functools
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

from Util.transform import TransformT
from modeling.centered_ridge import CenteredRidge
from modeling.hyper_parameter import HyperParameter


REGRESSION_METHODS = ("sm_ols", "sk_lr", "ridge", "enet")            # ARX, ARX_D, ARX_SR, ARX_SR_D
PENALIZED_METHODS = ("ridge", "enet")                                # the methods with knobs to tune (tune, withHP)
AR_METHODS = ("adjusted", "mle", "sm_ols")                           # AR: Yule-Walker (adjusted / MLE) or OLS
ARIMA_METHODS = ("statespace", "innovations_mle", "hannan_rissanen", "burg", "innovations", "yule_walker")
OPTIMIZERS = ("VarPro", "LBFGS_B", "Joint")                          # the names of `narx_sr_d.OptimizerNL`
MODELS = ("RandomWalk", "AR", "ARIMA", "ARX", "ARX_D", "ARX_SR", "ARX_SR_D", "NARX_SR_D")
DATASETS = ("Covid-19", "ILI", "ETTh1", "GEFCom2014")
VALIDATIONS = ("inSample", "tnT", "tvT", "both")                  # the tests run (args) makes

SCALA_HP = "hyper-parameters (ScalaTion's AR.hp, printed in the REPORT)"
PYTHON_HP = "hyper-parameters (Python only)"
MODEL = "model settings"
RUN = "run settings (read by run (args))"
SECTIONS = (SCALA_HP, PYTHON_HP, MODEL, RUN)


# ---------------------------------------------------------------------------------------------
# The checks: each validates and normalizes a value, raising ValueError with a plain message
# ---------------------------------------------------------------------------------------------

def _key(s: str) -> str:
    """A name in a form that ignores case, spaces, '-' and '_' (for matching choices)."""
    return "".join(ch for ch in str(s).lower() if ch not in " -_")


class _Check:
    """A check of one setting: `kind` describes the allowed values (shown by `help`)."""

    def __init__(self, kind: str, fn: Callable[[str, Any], Any]) -> None:
        self.kind = kind
        self.fn = fn

    def __call__(self, name: str, value: Any) -> Any:
        return self.fn(name, value)


def _bad(name: str, value: Any, kind: str) -> ValueError:
    return ValueError(f"Args.{name} = {value!r}: must be {kind}")


def _int(lo: int, hi: Optional[int] = None, none: bool = False) -> _Check:
    kind = (f"an integer >= {lo}" if hi is None else f"an integer in {lo}..{hi}") + (" or None" if none else "")

    def fn(name: str, v: Any) -> Any:
        if v is None and none:
            return None
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            v = int(v)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < lo or (hi is not None and v > hi):
            raise _bad(name, v, kind)
        return int(v)
    return _Check(kind.replace("an integer", "int"), fn)


def _float(lo: float, hi: Optional[float] = None, open_: bool = False, none: bool = False) -> _Check:
    if hi is None:
        kind = f"a number >= {lo}"
    else:
        kind = f"a number in ({lo}, {hi})" if open_ else f"a number in [{lo}, {hi}]"
    kind += " or None" if none else ""

    def fn(name: str, v: Any) -> Any:
        if v is None and none:
            return None
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise _bad(name, v, kind)
        v = float(v)
        ok = v >= lo if hi is None else ((lo < v < hi) if open_ else (lo <= v <= hi))
        if not ok or not np.isfinite(v):
            raise _bad(name, v, kind)
        return v
    return _Check(kind.replace("a number", "float"), fn)


def _bool() -> _Check:
    kind = "True / False (or 1 / 0)"

    def fn(name: str, v: Any) -> Any:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)) and v in (0, 1):
            return bool(v)
        raise _bad(name, v, kind)
    return _Check("bool", fn)


def _choice(values: Iterable[str], none: bool = False, aliases: Optional[Mapping[str, str]] = None) -> _Check:
    values = tuple(values)
    table = {_key(v): v for v in values}
    table.update({_key(a): v for a, v in (aliases or {}).items()})
    kind = " | ".join(values) + (" | None" if none else "")

    def fn(name: str, v: Any) -> Any:
        if v is None and none:
            return None
        if isinstance(v, str) and _key(v) in table:
            return table[_key(v)]
        raise _bad(name, v, "one of " + kind)
    return _Check(("None | " if none else "") + " | ".join(values), fn)


def _transformT() -> _Check:
    kind = "None, a TransformT or its name (Norm, MinMax, Log1p, ...)"

    def fn(name: str, v: Any) -> Any:
        if v is None or isinstance(v, TransformT):
            return v
        if isinstance(v, str):
            k = _key(v)
            k = k[:-4] if k.endswith("form") else k
            for t in TransformT:
                if _key(t.name) == k:
                    return t
        raise _bad(name, v, kind)
    return _Check("None | TransformT", fn)


def _optimizer() -> _Check:
    kind = "None, an OptimizerNL or its name (" + ", ".join(OPTIMIZERS) + ")"
    table = {_key(o): o for o in OPTIMIZERS}

    def fn(name: str, v: Any) -> Any:                                # kept as the name (`OptimizerNL.of` reads it)
        if v is None:
            return None
        if isinstance(v, Enum) and v.name in OPTIMIZERS:
            return v.name
        if isinstance(v, str) and _key(v) in table:
            return table[_key(v)]
        raise _bad(name, v, kind)
    return _Check("None | " + " | ".join(OPTIMIZERS), fn)


def _names() -> _Check:
    kind = "None or a list of column names"

    def fn(name: str, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            return (v,)
        if isinstance(v, (list, tuple)) and all(isinstance(c, str) for c in v):
            return tuple(v)                                          # a tuple: the snapshots stay unchanged
        raise _bad(name, v, kind)
    return _Check("None | list of str", fn)


# ---------------------------------------------------------------------------------------------
# The settings table: the one source of the names, defaults, checks and meanings
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    """One setting of `Args`: its name, default, check, meaning and section."""
    name: str
    default: Any
    check: _Check
    meaning: str
    section: str


_ROWS = (
    #        name         default                       check                         meaning
    Setting("p",          1,                            _int(0),                      "number of Auto-Regressive (AR) parameters (lags of y)", SCALA_HP),
    Setting("d",          1,                            _int(0),                      "number of Differences to take", SCALA_HP),
    Setting("q",          1,                            _int(0),                      "number of Moving-Average (MA) parameters", SCALA_HP),
    Setting("P",          1,                            _int(0),                      "number of Seasonal Auto-Regressive (AR) parameters", SCALA_HP),
    Setting("D",          1,                            _int(0),                      "number of Seasonal Differences to take", SCALA_HP),
    Setting("Q",          1,                            _int(0),                      "number of Seasonal Moving-Average (MA) parameters", SCALA_HP),
    Setting("s",          7,                            _int(1),                      "length of the Seasonal Period", SCALA_HP),
    Setting("px",         1,                            _int(0),                      "number of lags for the eXogenous variables", SCALA_HP),
    Setting("spec",       1,                            _int(0, 5),                   "trend terms: 0 implicit constant via centering, 1 linear, 2 + quadratic, 3 + cubic (4, 5: + sin, cos)", SCALA_HP),
    Setting("crx",        False,                        _bool(),                      "ENDO-EXO cross terms (the SR models)", SCALA_HP),
    Setting("nneg",       True,                         _bool(),                      "predictions must be non-negative", SCALA_HP),
    Setting("skip",       None,                         _int(0, none=True),           "rows skipped by the In-Sample diagnosis (None: the model's rule, usually p)", PYTHON_HP),
    Setting("lam",        CenteredRidge.hp["lambda"],   _float(0.0),                  "the ridge penalty lambda of method ridge (ScalaTion's RidgeRegression.hp)", PYTHON_HP),
    Setting("method",     None,                         _choice(REGRESSION_METHODS + AR_METHODS[:2] + ARIMA_METHODS, none=True),
            "the estimation method (None: the model's own, e.g., sm_ols for ARX_D, adjusted for AR)", MODEL),
    Setting("scaled",     False,                        _bool(),                      "the penalized regression (ridge, enet) standardizes the columns", MODEL),
    Setting("l1_ratio",   0.5,                          _float(0.0, 1.0),             "the share of the L1 penalty of method enet (0: ridge, 1: lasso)", MODEL),
    Setting("optimizer",  None,                         _optimizer(),                 "the optimizer of the nonlinear parameters (None: OPTIMIZER_NL)", MODEL),
    Setting("tune",       True,                         _bool(),                      "fit the nonlinear parameters of NARX_SR_D (False: their defaults)", MODEL),
    Setting("te_ratio",   None,                         _float(0.0, 0.95, none=True), "the fraction of rows in the testing set (None: Model.TE_RATIO when the model is built; 0: no testing set)", MODEL),
    Setting("dataset",    None,                         _choice(DATASETS, none=True, aliases={"covid": "Covid-19", "gefcom": "GEFCom2014",
                                                                                              "gef14": "GEFCom2014", "etth": "ETTh1"}),
            "the dataset of run (args)", RUN),
    Setting("exo",        None,                         _names(),                     "the exogenous columns (None: the dataset's default; []: none)", RUN),
    Setting("model",      "ARX_D",                      _choice(MODELS),              "the model run (args) builds", RUN),
    Setting("tFormT",     None,                         _transformT(),                "the transform type run (args) rescales by, passed to the model's rescale (None: no rescaling)", RUN),
    Setting("hh",         6,                            _int(1),                      "the maximum forecasting horizon of run (args)", RUN),
    Setting("validation", "both",                       _choice(VALIDATIONS, aliases={"in-sample": "inSample", "in_st": "inSample"}),
            "the test(s) run (args) makes: In-Sample, TnT (rolling validation on the testing rows), tvT (validation rows, then TnT) or both (In-Sample and TnT)", RUN),
    Setting("rc",         None,                         _int(1, none=True),           "the retraining cycle of the rolling tests (None: Forecaster.RC, 5)", RUN),
)

SETTINGS: Dict[str, Setting] = {s.name: s for s in _ROWS}

#: Other names a setting is set or read by ("lambda" is a Python keyword).
ALIASES = {"lambda": "lam"}

#: Hints for names other tools use (the message of an unknown name).
HINTS = {"max_horizon": ("hh", ""), "horizon": ("hh", ""), "horizons": ("hh", ""),
         "training_ratio": ("te_ratio", " (the TESTING fraction: 1 - the training ratio)"),
         "train_ratio": ("te_ratio", " (the TESTING fraction: 1 - the training ratio)"),
         "test_ratio": ("te_ratio", ""), "normalization": ("tFormT", " (e.g., TransformT.Norm, the run setting run (args) passes to rescale)"),
         "model_name": ("model", ""), "lags": ("p", ""), "exo_lags": ("px", ""), "ridge": ("lam", ""),
         "alpha": ("lam", ""), "lamda": ("lam", "")}

#: ScalaTion's `AR.hp`: the hyper-parameters every forecaster starts from (value, default), READ-ONLY.
hp = HyperParameter()
for _s in _ROWS:
    if _s.section == SCALA_HP:
        hp.add(_s.name, int(_s.default) if isinstance(_s.default, bool) else _s.default)
del _s
hp.freeze("AR.hp is read-only: set the hyper-parameters of a model in its Args (args.p = 6), or copy it (AR.hp.copy ())")


def unknown(name: str) -> str:
    """The message for a name that is not a setting: a hint or the closest names."""
    if name in HINTS:
        to, note = HINTS[name]
        return f"Args has no setting '{name}'; did you mean '{to}'{note}?  (Args.help () lists the settings)"
    names = list(SETTINGS) + list(ALIASES)
    low = {n.lower(): n for n in names}
    near = [name.lower()] if name.lower() in low else difflib.get_close_matches(name.lower(), list(low), n=2, cutoff=0.6)
    sugg = " or ".join(f"'{low[n]}'" for n in near)
    hint = f"; did you mean {sugg}?" if sugg else ""
    return f"Args has no setting '{name}'{hint}  (Args.help () lists the settings)"


class _hybrid:
    """A method callable on the class (`Args.help (ARX_D)`: the defaults) or on an instance
    (`args.help ()`: its values)."""

    def __init__(self, fn: Callable) -> None:
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __get__(self, obj: Any, cls: type) -> Callable:
        return functools.partial(self.fn, cls if obj is None else obj)


def _restore(values: Dict[str, Any]) -> "Args":
    """Unpickles an Args."""
    return Args(**values)


def _jsonable(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, tuple):
        return list(v)
    return v


class Args:
    """
    The checked settings of a forecasting model (and of a run): attribute access as a dotdict, but
    every name is a known setting and every value is checked when it is set (see the module).

    Parameters
    ----------
    **settings
        Settings to change from their defaults, e.g., `Args (p=6, px=4)`.

    Raises
    ------
    AttributeError
        For a name that is not a setting (with the closest names).
    ValueError
        For a value of the wrong type or out of range.
    """

    __slots__ = ("_v",)

    def __init__(self, **settings: Any) -> None:
        object.__setattr__(self, "_v", {n: s.default for n, s in SETTINGS.items()})
        for k, v in settings.items():
            setattr(self, k, v)

    # -- attribute access ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:                         # only when the normal lookup fails
        if name.startswith("__"):
            raise AttributeError(name)
        v = object.__getattribute__(self, "_v")
        key = ALIASES.get(name, name)
        if key in v:
            return v[key]
        raise AttributeError(unknown(name))

    def __setattr__(self, name: str, value: Any) -> None:
        key = ALIASES.get(name, name)
        if key not in SETTINGS:
            raise AttributeError(unknown(name))
        self._v[key] = SETTINGS[key].check(key, value)

    def __delattr__(self, name: str) -> None:
        """`del args.p` puts p back to its default."""
        self.reset(name)

    def __dir__(self) -> List[str]:                                 # tab completion lists the settings
        return sorted(set(SETTINGS) | {n for n in type(self).__dict__ if not n.startswith("_")})

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    # -- copies, changes, defaults ---------------------------------------------------------

    @classmethod
    def of(cls, args: Any = None) -> "Args":
        """A new Args from None (the defaults), an Args (a copy) or a dict of settings (checked)."""
        if args is None:
            return cls()
        if isinstance(args, Args):
            return args.copy()
        if isinstance(args, Mapping):
            return cls(**dict(args))
        raise TypeError(f"args must be an Args, a dict of settings or None, not {type(args).__name__}")

    def copy(self, **changes: Any) -> "Args":
        """A copy with the given settings changed (checked); this Args is unchanged."""
        out = Args()
        out._v.update(self._v)
        for k, v in changes.items():
            setattr(out, k, v)
        return out

    def update(self, other: Any = None, **changes: Any) -> "Args":
        """Sets, in place, the settings of `other` (a dict, or an Args: the settings it changed from their
        defaults) and the keywords; returns self."""
        if isinstance(other, Args):
            other = other.changed()
        for k, v in {**dict(other or {}), **changes}.items():
            setattr(self, k, v)
        return self

    def reset(self, *names: str) -> "Args":
        """Puts the given settings (all when none) back to their defaults; returns self."""
        for n in names or tuple(SETTINGS):
            key = ALIASES.get(n, n)
            if key not in SETTINGS:
                raise AttributeError(unknown(n))
            self._v[key] = SETTINGS[key].default
        return self

    @staticmethod
    def default(name: str) -> Any:
        """The default of a setting."""
        key = ALIASES.get(name, name)
        if key not in SETTINGS:
            raise AttributeError(unknown(name))
        return SETTINGS[key].default

    def changed(self) -> Dict[str, Any]:
        """The settings that differ from their defaults (in table order)."""
        return {n: v for n, v in self._v.items() if not _same(v, SETTINGS[n].default)}

    def toDict(self) -> Dict[str, Any]:
        """All the settings."""
        return dict(self._v)

    # -- presets -----------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves the changed settings as JSON (a preset)."""
        with open(path, "w") as fh:
            json.dump({k: _jsonable(v) for k, v in self.changed().items()}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Args":
        """Loads a preset saved by `save` (every setting checked)."""
        with open(path) as fh:
            return cls(**json.load(fh))

    # -- the model's HyperParameter ----------------------------------------------------------

    def hparam(self) -> HyperParameter:
        """
        The hyper-parameters of a model built with these settings: `AR.hp` (its defaults kept) with
        the values of the Scala hyper-parameters, crx / nneg as 0 / 1 (as ScalaTion keeps them), and,
        when set, the Python-only "skip" (unregistered: not printed).  The knobs of the regression
        (`lam`, `l1_ratio`) stay in the args: `Forecaster.hparameter` merges them (rev2.3).
        """
        h = hp.copy()
        for n in hp.registered():
            v = self._v[n]
            h[n] = int(v) if isinstance(v, bool) else v
        if self._v["skip"] is not None:
            h["skip"] = self._v["skip"]
        return h

    # -- help ----------------------------------------------------------------------------------

    @_hybrid
    def describe(self, model: Any = None) -> str:
        """
        The table of the settings: name, value, default, allowed values and meaning, by section, a
        "*" marking a changed value (on the class: the defaults); with a model (class or instance),
        only the settings it reads (its `ARGS`).
        """
        values = self._v if isinstance(self, Args) else {n: s.default for n, s in SETTINGS.items()}
        names = list(SETTINGS)
        title = "Args"
        if model is not None:
            reads = getattr(model, "ARGS", None)
            if reads is None:
                raise TypeError(f"describe: {model!r} lists no ARGS")
            names = [n for n in names if n in reads]
            title = f"Args read by {getattr(model, '__name__', type(model).__name__)}"
        lines = [f"{title}  (* = changed from the default)"]
        for sec in SECTIONS:
            rows = [SETTINGS[n] for n in names if SETTINGS[n].section == sec]
            if not rows:
                continue
            lines.append(f"  {sec}")
            for s in rows:
                v = values[s.name]
                mark = "*" if not _same(v, s.default) else " "
                kind, meaning = s.check.kind, s.meaning
                if s.name == "method" and getattr(model, "METHODS", None):  # the model's own methods and default
                    kind = "None | " + " | ".join(model.METHODS)
                    meaning = f"the estimation method (None: {model.METHOD or 'the library default'})"
                lines.append(f"  {mark} {s.name:<10} = {_show(v):<10} default {_show(s.default):<9} {meaning}  [{kind}]")
        return "\n".join(lines)

    @_hybrid
    def help(self, model: Any = None) -> None:
        """Prints `describe` (on the class: the defaults; on an instance: its values)."""
        print(self.describe(model))

    # -- printing, comparing, pickling -------------------------------------------------------

    def __repr__(self) -> str:
        return "Args(" + ", ".join(f"{k}={_show(v)}" for k, v in self.changed().items()) + ")"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Args) and all(_same(self._v[n], other._v[n]) for n in SETTINGS)

    __hash__ = None                                                   # mutable

    def __reduce__(self):
        return _restore, (self.changed(),)


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b if isinstance(a, (bool, Enum)) or isinstance(b, (bool, Enum)) else a == b


def _show(v: Any) -> str:
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, tuple):
        return repr(list(v))
    if isinstance(v, str):
        return repr(v)
    return str(v)
