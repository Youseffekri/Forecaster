"""
Datasets: the Example Loaders and the Dataset Configurations

The example datasets of the forecasting package and what an experiment runs on:

    Example_Covid    weekly US Covid-19 (data/covid_19_weekly.csv, 169 weeks; `clip` keeps the first
                     116, where the series flattens out): new deaths, with icu patients, test
                     positivity rate, ... as drivers
    Example_ILI      weekly US influenza-like illness, CDC ILINet (data/ILINetNov25.csv, 1,208 weeks):
                     ILITOTAL, with the weighted ILI percentage, the patient visits, ...
    Example_ETTh1    hourly oil temperature OT of an electricity transformer (data/ETTh1.csv, 17,420
                     rows, Zhou et al. 2021) with six load series as drivers
    Example_GEFCom   daily peak load of a US utility (data/gefcom2014_load_daily.csv, 2,099 days, built
                     by tools/build_gefcom_daily.py from GEFCom2014's L1-train.csv) with temperatures

Each loader has ScalaTion's members: `fileName`, `header`, `response`, `loadData (x_strs, y_str, ...)`
returning (xe, y), `loadData_y`, `loadData_yy` (Covid, ILI) and `context (x_strs, names, ...)`, the
named-column map of the formula layer ("y" and one short name per exogenous column), with the
constants of each object (Covid's `NO_EXO`, `BEST_EXO`, `BEST2_EXO` and `clip`; ETTh1's `LOADS`,
`LOAD_NAMES`; GEFCom's `TEMPS`).

`DatasetConfig` names what an experiment runs ON: the response, the POOL of exogenous variables,
the domain sentence a named prompt shows, the sampling-frequency word, the coefficient signs domain
knowledge expects, and who picks the exogenous variables actually used (`ExoMode`: `Given` takes the
list as is, `Auto` lets `ExoScreen` pick k of the pool on the training rows, `shortlist (data,
trSize, k, maxLag)`; `FormulaScorer.shortlist (cfg)` passes the scorer's).  `Datasets` holds the
configurations of the experiments: Covid-19 (three pools and the file-based `COVID_FILE`), ILI,
ETTh1 (all rows and the first year) and GEFCom2014.

The Python counterpart of ScalaTion's `forecasting/Example_Covid.scala`, `Example_ILI.scala`,
`Example_ETTh1.scala`, `Example_GEFCom.scala` (their objects; their test mains are the scenarios of
Examples/Example_*.py) and `forecasting/Datasets.scala`.  Differences: one file holds the four loaders
and the configurations (ScalaTion keeps each loader object with its test mains); ILI reads
data/ILINetNov25.csv (ScalaTion: national_illness_clip.csv), whose "TOTAL PATIENTS" column is the "ot"
of the ILI configuration (ScalaTion's "OT" column); `loadH` stands for `MatrixD.loadH`;
`Example_GEFCom.v_table` (the V table example_GEFComTest prints), `Datasets.ALL` and `Datasets.get`
are Python helpers.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modeling.forecasting.exo_screen import ExoScreen
from modeling.model import Model

#: The data directory: <repo>/data (override with the FORECASTER_DATA environment variable).
DATA_DIR = Path(os.environ.get("FORECASTER_DATA", Path(__file__).resolve().parents[2] / "data"))

XeY = Tuple[Optional[np.ndarray], np.ndarray]                        # (xe, y), ScalaTion's (MatrixD, VectorD)


def _frame(fileName: str, trim: int = 0) -> pd.DataFrame:
    """The CSV file under DATA_DIR (header line first), the first `trim` data rows dropped."""
    path = DATA_DIR / fileName
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (set FORECASTER_DATA to the directory holding it)")
    df = pd.read_csv(path)
    return df.iloc[trim:].reset_index(drop=True) if trim > 0 else df


def loadH(fileName: str, skipCol: int = 1) -> Tuple[np.ndarray, List[str]]:
    """
    The numeric matrix of a CSV file with a header line and its column names, the first `skipCol`
    columns (e.g., a date) skipped (ScalaTion's `MatrixD.loadH`).
    """
    df = _frame(fileName).iloc[:, skipCol:]
    return df.to_numpy(dtype=float), list(df.columns)


def _columns(loader: type, names: Sequence[str]) -> List[str]:
    """The column names checked against the loader's header (ScalaTion looks them up in `header`)."""
    names = list(names)
    bad = [s for s in names if s not in loader.header]
    if bad:
        raise KeyError(f"{loader.__name__}: unknown column(s) {bad}; the header is {list(loader.header)}")
    return names


def _load(loader: type, x_strs: Sequence[str], y_str: str, trim: int = 0, rng: Optional[range] = None) -> XeY:
    """(xe, y) of a loader's file: the columns x_strs and y_str, after `trim` rows, within `rng`."""
    x_cols, (y_col,) = _columns(loader, x_strs), _columns(loader, [y_str])
    df = _frame(loader.fileName, trim)
    if rng is not None:
        df = df.iloc[rng.start:rng.stop]
    return df[x_cols].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float)


def _context(xe: np.ndarray, y: np.ndarray, x_strs: Sequence[str], names: Optional[Sequence[str]]) -> Dict[str, np.ndarray]:
    """The named-column map: the response under "y", each exogenous column under its short name."""
    nms = list(x_strs) if names is None else list(names)
    if len(nms) != len(x_strs):
        raise ValueError("context: one short name per exogenous column")
    return {"y": y, **{nm: xe[:, j] for j, nm in enumerate(nms)}}


# ---------------------------------------------------------------------------------------------
# The example loaders (ScalaTion's `Example_Covid`, `Example_ILI`, `Example_ETTh1`, `Example_GEFCom`)
# ---------------------------------------------------------------------------------------------

class Example_Covid:
    """
    Weekly US Covid-19 data (ScalaTion's `Example_Covid`): new deaths as the response; the series
    flattens out after about 116 weeks, so the experiments `clip` to the first 116 rows.
    Usage:  xe, y = Example_Covid.clip(Example_Covid.loadData(Example_Covid.BEST2_EXO)).
    """
    fileName = "covid_19_weekly.csv"
    header = ("new_cases", "new_deaths", "reproduction_rate", "icu_patients", "hosp_patients", "new_tests",
              "positive_rate", "tests_per_case", "people_vaccinated", "people_fully_vaccinated", "total_boosters",
              "new_vaccinations", "excess_mortality_cumulative_absolute", "excess_mortality_cumulative",
              "excess_mortality", "excess_mortality_cumulative_per_million")
    response = "new_deaths"                                          # main response/output variable
    NO_EXO: Tuple[str, ...] = ()                                     # no exogenous variables
    BEST_EXO = ("icu_patients",)                                     # exo with the best lagged correlation with the response
    BEST2_EXO = ("icu_patients", "positive_rate")                    # top two exo variables (best combination of 2)

    @staticmethod
    def clip(xe_y: XeY, rng: range = range(0, 116)) -> XeY:
        """
        Clips the response to the range and the exogenous variables too, if any (xe may be None):
        for Covid the series flattens out at about 116 weeks.  Usage: _, y = clip((None, loadData_y())).
        """
        xe, y = xe_y
        return (None if xe is None else xe[rng.start:rng.stop]), y[rng.start:rng.stop]

    @classmethod
    def loadData(cls, x_strs: Sequence[str], y_str: str = response, trim: int = 0) -> XeY:
        """(xe, y): the exogenous columns x_strs and the response y_str, the first `trim` rows dropped."""
        return _load(cls, x_strs, y_str, trim)

    @classmethod
    def loadData_y(cls, y_str: str = response, trim: int = 0) -> np.ndarray:
        """The response column y_str, the first `trim` rows dropped."""
        return _load(cls, (), y_str, trim)[1]

    @classmethod
    def loadData_yy(cls, y_strs: Sequence[str], trim: int = 0) -> np.ndarray:
        """The matrix of the columns y_strs (e.g., for a VAR model), the first `trim` rows dropped."""
        return _frame(cls.fileName, trim)[_columns(cls, y_strs)].to_numpy(dtype=float)

    @classmethod
    def context(cls, x_strs: Sequence[str], names: Optional[Sequence[str]] = None, y_str: str = response,
                rng: range = range(0, 116)) -> Dict[str, np.ndarray]:
        """The named-column map ("y" and the short names, default x_strs), clipped to rng (`clip`).
        Usage: context(BEST2_EXO, ["icu", "pRate"])."""
        xe, y = cls.clip(cls.loadData(x_strs, y_str), rng)
        return _context(xe, y, x_strs, names)


class Example_ILI:
    """
    Weekly US influenza-like illness from the CDC's ILINet (ScalaTion's `Example_ILI`; here
    data/ILINetNov25.csv, 1,208 weeks from 2002-10-06 to 2025-11-23, the Covid years kept): the total
    ILI patient count ILITOTAL as the response; "TOTAL PATIENTS" plays ScalaTion's "OT" (patient visits).
    """
    fileName = "ILINetNov25.csv"
    header = ("%WEIGHTED ILI", "%UNWEIGHTED ILI", "AGE 0-4", "AGE 25-49", "AGE 5-24", "AGE 65", "ILITOTAL",
              "NUM. OF PROVIDERS", "TOTAL PATIENTS", "WEEK", "WEEKSqr")
    response = "ILITOTAL"                                            # main response/output variable

    @classmethod
    def loadData(cls, x_strs: Sequence[str], y_str: str, trim: int = 0) -> XeY:
        """(xe, y): the exogenous columns x_strs and the response y_str, the first `trim` rows dropped."""
        return _load(cls, x_strs, y_str, trim)

    @classmethod
    def loadData_y(cls, y_str: str, trim: int = 0) -> np.ndarray:
        """The response column y_str, the first `trim` rows dropped."""
        return _load(cls, (), y_str, trim)[1]

    @classmethod
    def loadData_yy(cls, y_strs: Sequence[str], trim: int = 0) -> np.ndarray:
        """The matrix of the columns y_strs (e.g., for a VAR model), the first `trim` rows dropped."""
        return _frame(cls.fileName, trim)[_columns(cls, y_strs)].to_numpy(dtype=float)

    @classmethod
    def context(cls, x_strs: Sequence[str], names: Optional[Sequence[str]] = None, y_str: str = response,
                trim: int = 0) -> Dict[str, np.ndarray]:
        """The named-column map ("y" and the short names, default x_strs).
        Usage: context(["%WEIGHTED ILI", "TOTAL PATIENTS"], ["wili", "ot"])."""
        xe, y = cls.loadData(x_strs, y_str, trim)
        return _context(xe, y, x_strs, names)


class Example_ETTh1:
    """
    ETTh1 (ScalaTion's `Example_ETTh1`; Zhou et al. 2021, Informer): hourly oil temperature OT of an
    electricity transformer, 2016-07-01 .. 2018-06-26 (17,420 rows), with six load series HUFL, HULL,
    MUFL, MULL, LUFL, LULL (high / middle / low useful / useless load); OT goes below zero.
    """
    fileName = "ETTh1.csv"
    header = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
    response = "OT"                                                  # main response/output variable
    LOADS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")         # the six loads (the candidate drivers)
    LOAD_NAMES = ("hufl", "hull", "mufl", "mull", "lufl", "lull")    # and their short formula names

    @classmethod
    def loadData(cls, x_strs: Sequence[str], y_str: str, rng: Optional[range] = None) -> XeY:
        """(xe, y): the exogenous columns x_strs and the response y_str within rng (None => all rows)."""
        return _load(cls, x_strs, y_str, rng=rng)

    @classmethod
    def loadData_y(cls, y_str: str = response, rng: Optional[range] = None) -> np.ndarray:
        """The response column y_str within rng (None => all rows)."""
        return _load(cls, (), y_str, rng=rng)[1]

    @classmethod
    def context(cls, x_strs: Sequence[str] = LOADS, names: Optional[Sequence[str]] = LOAD_NAMES,
                y_str: str = response, rng: Optional[range] = None) -> Dict[str, np.ndarray]:
        """The named-column map ("y" and the short names) within rng.  Usage: context(rng=range(0, 8760))."""
        xe, y = cls.loadData(x_strs, y_str, rng)
        return _context(xe, y, x_strs, names)


class Example_GEFCom:
    """
    GEFCom2014 load track (ScalaTion's `Example_GEFCom`; Hong et al. 2016), aggregated to days: the daily
    peak load of a US utility, 2005-01-01 .. 2010-09-30 (2,099 days), with the daily mean load (avg),
    the daily mean / max / min of the 25-station mean temperature (temp, tmax, tmin, deg F) and a
    weekend flag.  The load-temperature relation is V-shaped (heating below ~55 F, cooling above ~65 F).
    """
    fileName = "gefcom2014_load_daily.csv"
    header = ("peak", "avg", "temp", "tmax", "tmin", "weekend")
    response = "peak"                                                # main response/output variable
    TEMPS = ("temp", "tmax", "tmin")                                 # the temperature columns (the candidate drivers)

    @classmethod
    def loadData(cls, x_strs: Sequence[str], y_str: str, rng: Optional[range] = None) -> XeY:
        """(xe, y): the exogenous columns x_strs and the response y_str within rng (None => all days)."""
        return _load(cls, x_strs, y_str, rng=rng)

    @classmethod
    def loadData_y(cls, y_str: str = response, rng: Optional[range] = None) -> np.ndarray:
        """The response column y_str within rng (None => all days)."""
        return _load(cls, (), y_str, rng=rng)[1]

    @classmethod
    def context(cls, x_strs: Sequence[str] = TEMPS, names: Optional[Sequence[str]] = None,
                y_str: str = response, rng: Optional[range] = None) -> Dict[str, np.ndarray]:
        """The named-column map ("y" and the short names, default x_strs) within rng."""
        xe, y = cls.loadData(x_strs, y_str, rng)
        return _context(xe, y, x_strs, names)

    @classmethod
    def v_table(cls, bands=((20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90))) -> List[float]:
        """The mean peak load per daily-mean-temperature band (the V shape example_GEFComTest prints)."""
        xe, y = cls.loadData(["temp"], cls.response)
        temp = xe[:, 0]
        return [float(y[(temp >= a) & (temp < b)].mean()) for a, b in bands]


# ---------------------------------------------------------------------------------------------
# The dataset configurations (ScalaTion's `Datasets.scala`)
# ---------------------------------------------------------------------------------------------

class ExoMode(Enum):
    """Who picks the exogenous variables (ScalaTion's `ExoMode`): `Given` uses the configuration's
    list as is, `Auto` lets the screening (`ExoScreen`) pick k of them (the pool)."""
    Given = 0
    Auto = 1

    def __str__(self) -> str:
        return self.name


@dataclass
class DatasetConfig:
    """
    What an experiment runs on (ScalaTion's `DatasetConfig`): the response, the POOL of exogenous
    variables, the domain sentence a named prompt shows, the sampling-frequency word of the blind
    prompt, the coefficient signs domain knowledge expects, and who picks the exogenous variables
    actually used (`ExoMode`).

    Attributes
    ----------
    name : str
        For banners and titles (never sent to an LLM).
    resp : str
        The response variable name (the key "y").
    exo : list of str
        The exogenous variable names available (keys of the map).
    schema : str
        The domain sentence (only a named prompt shows it).
    expectedSigns : dict
        The expected coefficient sign per variable (+1 / -1; unknown => absent).
    freq : str
        The sampling-frequency word for the blind prompt ("weekly", "hourly", "daily").
    load : callable
        Builds the named-column map (full series).
    exoMode : ExoMode
        `Given` => the arms use `exo` as is; `Auto` => the screening picks k of them.
    """
    name: str
    resp: str
    exo: List[str]
    schema: str
    expectedSigns: Dict[str, int]
    freq: str
    load: Callable[[], Dict[str, np.ndarray]]
    exoMode: ExoMode = ExoMode.Given

    def shortlist(self, data: Dict[str, np.ndarray], trSize: int, k: int, maxLag: int) -> List[str]:
        """
        The exogenous variables an experiment builds on: the list itself (`Given`) or the screening's
        shortlist (`Auto`): `ExoScreen.shortlist` of the pool on the first trSize rows of data, the k
        best by lead cross-correlation up to maxLag (`FormulaScorer.shortlist (cfg)` passes the
        scorer's data, training rows, hp["exoK"] and hp["maxLag"]).

        Parameters
        ----------
        data : dict of str -> np.ndarray
            The named-column map (the response under `resp`, the pool under their names).
        trSize : int
            The number of training rows the screening looks at.
        k : int
            The number of exogenous variables kept.
        maxLag : int
            The largest lead examined.
        """
        if self.exoMode is ExoMode.Given:
            return list(self.exo)
        return ExoScreen.shortlist(data[self.resp], {v: data[v] for v in self.exo}, maxLag, trSize, k)[1]

    def baselineFormulaStr(self, maxLag: int, exo: Optional[Sequence[str]] = None) -> str:
        """The plain-ARX baseline: full contiguous lag windows on the response and the exogenous
        variables, no transforms, e.g., "y ~= y~(6,1) + icu~(6,1) + pRate~(6,1)"."""
        exo = self.exo if exo is None else list(exo)
        return f"{self.resp} ~= " + " + ".join(f"{v}~({maxLag},1)" for v in [self.resp] + exo)

    @property
    def blindNames(self) -> Dict[str, str]:
        """The blind display name per variable (y, x1, x2, ...) over the pool."""
        return {self.resp: "y", **{v: f"x{i + 1}" for i, v in enumerate(self.exo)}}

    def copy(self, **changes) -> "DatasetConfig":
        """A copy with the given fields changed (Scala's `copy`)."""
        return replace(self, **changes)


class _Lazy:
    """A class attribute computed when first read (ScalaTion's `lazy val`)."""

    def __init__(self, make: Callable[[type], DatasetConfig]) -> None:
        self.make = make
        self.value: Optional[DatasetConfig] = None

    def __get__(self, obj, cls: type) -> DatasetConfig:
        if self.value is None:
            self.value = self.make(cls)
        return self.value


class Datasets:
    """The dataset configurations of the experiments (ScalaTion's `Datasets`); each `load` builds the
    named-column map from the example loaders (or straight from a CSV file, `fromFile`)."""

    COVID2_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; icu (patients in intensive care) "
                     "and pRate (share of tests that are positive) are leading indicators of deaths.")
    COVID_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; hosp and icu (patients "
                    "in hospital / intensive care) are leading indicators of deaths.")
    COVID_ALL_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; the candidate drivers are new cases, the "
                        "reproduction rate, icu and hospital patients, tests, the positivity rate, tests per case, "
                        "new vaccinations and excess mortality.")
    ILI_SCHEMA = ("Dataset: weekly US influenza-like illness (CDC ILINet, 2002-2025); y = total ILI patient count; "
                  "wili = population-weighted ILI percentage; ot = total patient visits.")

    #: Covid-19 weekly deaths with icu patients and test positivity rate (`Example_Covid.BEST2_EXO`): we choose.
    COVID2 = DatasetConfig("Covid-19 (icu, pRate)", "y", ["icu", "pRate"], COVID2_SCHEMA,
                           {"icu": 1, "pRate": 1}, "weekly",
                           lambda: Example_Covid.context(Example_Covid.BEST2_EXO, ["icu", "pRate"]))
    #: Covid-19 weekly deaths with hospital and icu patients.
    COVID = DatasetConfig("Covid-19 (hosp, icu)", "y", ["hosp", "icu"], COVID_SCHEMA,
                          {"hosp": 1, "icu": 1}, "weekly",
                          lambda: Example_Covid.context(["hosp_patients", "icu_patients"], ["hosp", "icu"]))
    #: The pool of Covid-19 columns an automatic selection may pick from (the non-cumulative columns of
    #: `Example_Covid.header`) and their short names.
    COVID_POOL = ["new_cases", "reproduction_rate", "icu_patients", "hosp_patients", "new_tests",
                  "positive_rate", "tests_per_case", "new_vaccinations", "excess_mortality"]
    COVID_NAMES = ["cases", "rRate", "icu", "hosp", "tests", "pRate", "tpc", "vacc", "exMort"]
    #: Covid-19 weekly deaths with the whole pool: the system chooses (`ExoMode.Auto`).
    COVID_ALL = DatasetConfig("Covid-19 (auto exo)", "y", list(COVID_NAMES), COVID_ALL_SCHEMA,
                              {"cases": 1, "rRate": 1, "icu": 1, "hosp": 1, "pRate": 1, "tpc": -1, "vacc": -1, "exMort": 1},
                              "weekly", lambda: Example_Covid.context(Datasets.COVID_POOL, Datasets.COVID_NAMES), ExoMode.Auto)
    #: Influenza-like illness weekly counts with the weighted ILI percentage and total visits.
    ILI = DatasetConfig("ILI", "y", ["wili", "ot"], ILI_SCHEMA, {"wili": 1}, "weekly",
                        lambda: Example_ILI.context(["%WEIGHTED ILI", "TOTAL PATIENTS"], ["wili", "ot"]))

    @classmethod
    def fromFile(cls, name: str, fileName: str, resp: str, schema: str, freq: str, rng: Optional[range] = None,
                 exclude: Sequence[str] = (), skipCol: int = 1) -> DatasetConfig:
        """
        A configuration straight from a CSV file with a header line: the pool is every column
        `ExoScreen.pool` finds eligible (not the response, not excluded, finite, neither constant
        nor cumulative on the training rows), under its own column name, and the system chooses
        among them (`ExoMode.Auto`).  No expected signs are known for such a pool.

        Parameters
        ----------
        name, fileName, resp, schema, freq : str
            The dataset name, the CSV file (relative to DATA_DIR), the response column, the domain sentence, the frequency word.
        rng : range, optional
            The rows to keep (None => all rows).
        exclude : sequence of str
            The columns domain knowledge rules out (proxies of the response).
        skipCol : int, default=1
            The leading columns to skip (e.g., a date column).
        """
        all_, header = loadH(fileName, skipCol)
        data = all_ if rng is None else all_[rng.start:rng.stop]
        p = ExoScreen.pool(data, header, resp, Model.trSize(len(data)), exclude)
        for nm, why in p.dropped:
            print(f"Datasets.fromFile: {nm} dropped -- {why}")
        mp = {"y": p.y, **p.xe}
        return DatasetConfig(name, "y", list(p.xe.keys()), schema, {}, freq, lambda: mp, ExoMode.Auto)

    COVID_FILE_SCHEMA = "Dataset: weekly US Covid-19; y = new deaths; the other columns of the file are the candidate drivers."

    #: Covid-19 weekly deaths straight from the file (the rows of `Example_Covid.clip`): the pool is whatever
    #: the screening finds eligible, minus excess mortality (another count of the same deaths) and new cases
    #: (not comparable across testing regimes); the system chooses.  Lazy: the file is read when first used.
    COVID_FILE = _Lazy(lambda cls: cls.fromFile("Covid-19 (file, auto exo)", Example_Covid.fileName, Example_Covid.response,
                                                cls.COVID_FILE_SCHEMA, "weekly", range(0, 116),
                                                ["excess_mortality", "new_cases"]))

    ETTH1_SCHEMA = ("Dataset: hourly electricity transformer (ETTh1); y = oil temperature of the transformer; "
                    "the candidate drivers are the high, middle and low useful/useless power loads "
                    "(hufl, hull, mufl, mull, lufl, lull); the oil warms with load and cools slowly.")
    GEF14_SCHEMA = ("Dataset: daily electric load of a US utility (GEFCom2014); y = daily peak load; "
                    "temp and tmax are the daily mean and maximum temperature (deg F); load is lowest at "
                    "mild temperatures and rises in cold (heating) and hot (cooling) weather.")

    @classmethod
    def ettH1(cls, rng: Optional[range] = None) -> DatasetConfig:
        """The ETTh1 configuration: hourly oil temperature with the six loads as the pool, the
        system choosing k of them (`ExoMode.Auto`); every load is expected to warm the oil (+1)."""
        nm = "ETTh1 (auto exo)" if rng is None else f"ETTh1 rows {rng.start}..{rng.stop - 1} (auto exo)"
        return DatasetConfig(nm, "y", list(Example_ETTh1.LOAD_NAMES), cls.ETTH1_SCHEMA,
                             {v: 1 for v in Example_ETTh1.LOAD_NAMES}, "hourly",
                             lambda: Example_ETTh1.context(rng=rng), ExoMode.Auto)

    #: GEFCom2014 daily peak load with the daily mean and maximum temperature: we choose (the relation is
    #: V-shaped, so no sign is expected).
    GEF14 = DatasetConfig("GEFCom2014 load (temp, tmax)", "y", ["temp", "tmax"], GEF14_SCHEMA,
                          {}, "daily", lambda: Example_GEFCom.context(["temp", "tmax"]))

    @classmethod
    def get(cls, name: str) -> DatasetConfig:
        """A configuration by short name: covid2, covid, covid_all, covid_file, ili, etth1, etth1_1y, gef14."""
        key = name.strip().lower()
        if key == "covid_file":
            return cls.COVID_FILE
        if key not in cls.ALL:
            raise KeyError(f"dataset '{name}' unknown: {', '.join(list(cls.ALL) + ['covid_file'])}")
        return cls.ALL[key]


#: ETTh1, the full series (17420 hourly rows) and its first year (8760 rows, 2016-07-01 .. 2017-06-30: one
#: cycle of the seasons at a tenth of the cost).
Datasets.ETTH1 = Datasets.ettH1()
Datasets.ETTH1_1Y = Datasets.ettH1(range(0, 8760))
Datasets.ALL = {"covid2": Datasets.COVID2, "covid": Datasets.COVID, "covid_all": Datasets.COVID_ALL, "ili": Datasets.ILI,
                "etth1": Datasets.ETTH1, "etth1_1y": Datasets.ETTH1_1Y, "gef14": Datasets.GEF14}
