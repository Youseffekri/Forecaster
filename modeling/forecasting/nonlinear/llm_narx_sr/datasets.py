"""
Datasets of the LLM-NARX-SR Experiments

The example datasets as named-column maps (dict name -> np.ndarray) for the formula layer,
and `DatasetConfig`: what an experiment runs ON -- the response, the POOL of exogenous
variables, the domain sentence a named prompt shows, the sampling-frequency word, the
coefficient signs domain knowledge expects, and who picks the exogenous variables actually used
(`ExoMode`: `Given` takes the list as is, `Auto` lets the screening pick hp["exoK"] of the pool
on the training rows -- `ExoScreen` of `modeling/forecasting/exo_screen.py`, ScalaTion's
`forecasting/ExoScreen.scala`: `pool`, `rank`, `shortlist`).  `Datasets` holds the configurations of
the paper's experiments: Covid-19 (three pools and the file-based one), ILI, ETTh1 (all rows and
the first year, the system choosing two of the six loads) and GEFCom2014.

    Covid-19     weekly US deaths, 116 weeks (data/covid_19_weekly.csv), icu patients and
                 test positivity rate as drivers
    ILI          weekly US influenza-like illness, CDC ILINet (data/ILINetNov25.csv, 1,208 weeks)
    ETTh1        hourly oil temperature (OT) of an electricity transformer, 17,420 rows,
                 six load series as drivers (data/ETTh1.csv, Zhou et al. 2021)
    GEFCom2014   daily peak load of a US utility, 2,099 days, with the daily mean and max
                 temperature (data/gefcom2014_load_daily.csv, built by tools/build_gefcom_daily.py
                 from the competition's L1-train.csv; Hong et al. 2016)

The Python counterpart of ScalaTion's `Example_Covid` / `Example_ILI` / `Example_ETTh1` /
`Example_GEFCom` loaders and of `llm_narx_sr/Datasets.scala` (rev4.0).  Differences: the lazy
`COVID_FILE` is the method `Datasets.covidFile()`; ILI's `ot` is the TOTAL PATIENTS column of the
ILINet file; the configurations and their `load` are kept from the earlier core.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from modeling.forecasting.exo_screen import ExoScreen
from modeling.forecasting.forecaster import trSize

#: The data directory: <repo>/data (override with the FORECASTER_DATA environment variable).
DATA_DIR = Path(os.environ.get("FORECASTER_DATA", Path(__file__).resolve().parents[4] / "data"))


def _read(file_name: str) -> pd.DataFrame:
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (set FORECASTER_DATA to the directory holding it)")
    return pd.read_csv(path)


def _context(df: pd.DataFrame, y_str: str, x_strs: Sequence[str], names: Optional[Sequence[str]] = None,
             rng: Optional[range] = None) -> Dict[str, np.ndarray]:
    """The named-column map: the response under "y", each exogenous column under its short name."""
    names = list(x_strs) if names is None else list(names)
    if len(names) != len(x_strs):
        raise ValueError("context: names and x_strs must have the same length")
    if rng is not None:
        df = df.iloc[rng.start:rng.stop]
    ctx = {"y": df[y_str].to_numpy(dtype=float)}
    for col, nm in zip(x_strs, names):
        ctx[nm] = df[col].to_numpy(dtype=float)
    return ctx


class Example_Covid:
    """
    Weekly US Covid-19 data (data/covid_19_weekly.csv): new deaths as the response, with the
    first 116 weeks used (the rows of the ScalaTion tests and of Examples/Example_Covid.py).
    """
    FILE = "covid_19_weekly.csv"
    RESPONSE = "new_deaths"
    N_ROWS = 116
    BEST2_EXO = ["icu_patients", "positive_rate"]
    SHORT = {"icu_patients": "icu", "positive_rate": "pRate", "hosp_patients": "hosp", "new_cases": "cases",
             "reproduction_rate": "rRate", "new_tests": "tests", "tests_per_case": "tpc",
             "new_vaccinations": "vacc", "excess_mortality": "exMort"}

    @classmethod
    def frame(cls, n_rows: Optional[int] = N_ROWS) -> pd.DataFrame:
        df = _read(cls.FILE)
        return df.iloc[:n_rows].reset_index(drop=True) if n_rows else df

    @classmethod
    def loadData(cls, x_strs: Sequence[str] = ("icu_patients", "positive_rate"), n_rows: Optional[int] = N_ROWS):
        """(y, xe): the response and the exogenous matrix (n_rows-by-len(x_strs))."""
        df = cls.frame(n_rows)
        return df[cls.RESPONSE].to_numpy(dtype=float), df[list(x_strs)].to_numpy(dtype=float)

    @classmethod
    def context(cls, x_strs: Sequence[str] = ("icu_patients", "positive_rate"), names: Optional[Sequence[str]] = None,
                n_rows: Optional[int] = N_ROWS) -> Dict[str, np.ndarray]:
        """The named-column map for the formula layer: "y" and the short names (icu, pRate, ...)."""
        names = [cls.SHORT.get(c, c) for c in x_strs] if names is None else names
        return _context(cls.frame(n_rows), cls.RESPONSE, x_strs, names)


class Example_ILI:
    """
    Weekly US influenza-like illness from the CDC's ILINet (data/ILINetNov25.csv, 1,208 weeks from
    2002-10-06 to 2025-11-23): the total ILI patient count ILITOTAL as the response, with the
    population-weighted ILI percentage (%WEIGHTED ILI -> wili), the total patient visits (TOTAL
    PATIENTS -> ot) and the number of reporting providers (NUM. OF PROVIDERS -> prov) as the
    drivers (the columns of ScalaTion's `Example_ILI`, whose "OT" is TOTAL PATIENTS here).
    The Covid years are kept; `trim` drops leading rows.
    """
    FILE = "ILINetNov25.csv"
    RESPONSE = "ILITOTAL"
    HEADER = ["%WEIGHTED ILI", "%UNWEIGHTED ILI", "AGE 0-4", "AGE 25-49", "AGE 5-24", "AGE 65", "ILITOTAL",
              "NUM. OF PROVIDERS", "TOTAL PATIENTS", "WEEK", "WEEKSqr"]
    EXO = ["%WEIGHTED ILI", "TOTAL PATIENTS"]
    NAMES = ["wili", "ot"]
    SHORT = {"%WEIGHTED ILI": "wili", "%UNWEIGHTED ILI": "uwili", "AGE 0-4": "age0", "AGE 25-49": "age25",
             "AGE 5-24": "age5", "AGE 65": "age65", "NUM. OF PROVIDERS": "prov", "TOTAL PATIENTS": "ot",
             "WEEK": "week", "WEEKSqr": "weeksq"}
    N_ROWS = 1208

    @classmethod
    def frame(cls, trim: int = 0) -> pd.DataFrame:
        """The data frame (the first `trim` rows dropped)."""
        df = _read(cls.FILE)
        return df.iloc[trim:].reset_index(drop=True) if trim > 0 else df

    @classmethod
    def loadData(cls, x_strs: Sequence[str] = EXO, y_str: str = RESPONSE, trim: int = 0):
        """(y, xe): the response vector and the exogenous matrix (ScalaTion's `loadData`)."""
        df = cls.frame(trim)
        return df[y_str].to_numpy(dtype=float), df[list(x_strs)].to_numpy(dtype=float)

    @classmethod
    def loadData_y(cls, y_str: str = RESPONSE, trim: int = 0) -> np.ndarray:
        """The response vector alone (ScalaTion's `loadData_y`)."""
        return cls.frame(trim)[y_str].to_numpy(dtype=float)

    @classmethod
    def context(cls, x_strs: Sequence[str] = EXO, names: Optional[Sequence[str]] = None,
                trim: int = 0) -> Dict[str, np.ndarray]:
        """The named-column map for the formula layer: "y" (ILITOTAL) and the short names (wili, ot, ...)."""
        names = [cls.SHORT.get(c, c) for c in x_strs] if names is None else names
        return _context(cls.frame(trim), cls.RESPONSE, x_strs, names)


class Example_ETTh1:
    """
    ETTh1 (Zhou et al. 2021, Informer): hourly oil temperature OT of an electricity transformer,
    2016-07-01 .. 2018-06-26 (17,420 rows), with six load series HUFL, HULL, MUFL, MULL, LUFL,
    LULL (high / middle / low useful / useless load).  data/ETTh1.csv is the original file.
    """
    FILE = "ETTh1.csv"
    RESPONSE = "OT"
    LOADS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    LOAD_NAMES = ["hufl", "hull", "mufl", "mull", "lufl", "lull"]
    N_ROWS = 17420
    ONE_YEAR = range(0, 8760)

    @classmethod
    def frame(cls) -> pd.DataFrame:
        return _read(cls.FILE)

    @classmethod
    def loadData(cls, x_strs: Sequence[str] = LOADS, rng: Optional[range] = None):
        df = cls.frame()
        if rng is not None:
            df = df.iloc[rng.start:rng.stop]
        return df[cls.RESPONSE].to_numpy(dtype=float), df[list(x_strs)].to_numpy(dtype=float)

    @classmethod
    def context(cls, x_strs: Sequence[str] = LOADS, names: Optional[Sequence[str]] = None,
                rng: Optional[range] = None) -> Dict[str, np.ndarray]:
        """The named-column map ("y" = OT, hufl .. lull); `rng` keeps a row range (e.g., the first year)."""
        if names is None:
            names = [cls.LOAD_NAMES[cls.LOADS.index(c)] if c in cls.LOADS else c.lower() for c in x_strs]
        return _context(cls.frame(), cls.RESPONSE, x_strs, names, rng)


class Example_GEFCom:
    """
    GEFCom2014 load track (Hong et al. 2016), aggregated to days by tools/build_gefcom_daily.py:
    the daily peak load of a US utility, 2005-01-01 .. 2010-09-30 (2,099 days), with the daily
    mean load (avg), the daily mean / max / min of the 25-station mean temperature (temp, tmax,
    tmin, deg F) and a weekend flag.  The load-temperature relation is V-shaped (heating below
    ~55 F, cooling above ~65 F): mean peak per 10 F band 272, 226, 185, 148, 136, 214, 273.
    """
    FILE = "gefcom2014_load_daily.csv"
    RESPONSE = "peak"
    TEMPS = ["temp", "tmax", "tmin"]
    N_ROWS = 2099

    @classmethod
    def frame(cls) -> pd.DataFrame:
        return _read(cls.FILE)

    @classmethod
    def loadData(cls, x_strs: Sequence[str] = ("temp", "tmax")):
        df = cls.frame()
        return df[cls.RESPONSE].to_numpy(dtype=float), df[list(x_strs)].to_numpy(dtype=float)

    @classmethod
    def context(cls, x_strs: Sequence[str] = ("temp", "tmax"), names: Optional[Sequence[str]] = None,
                rng: Optional[range] = None) -> Dict[str, np.ndarray]:
        return _context(cls.frame(), cls.RESPONSE, x_strs, names, rng)

    @classmethod
    def v_table(cls, bands=((20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90))) -> List[float]:
        """The mean peak load per temperature band (the V shape)."""
        df = cls.frame()
        return [float(df[(df.temp >= a) & (df.temp < b)].peak.mean()) for a, b in bands]


class ExoMode(Enum):
    """Who picks the exogenous variables (ScalaTion's `ExoMode`): `Given` uses the configuration's
    list as is, `Auto` lets the screening (`ExoScreen`) pick hp["exoK"] of them (the pool)."""
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
    expected_signs : dict
        The expected coefficient sign per variable (+1 / -1; unknown => absent), ScalaTion's `expectedSigns`.
    freq : str
        The sampling-frequency word for the blind prompt ("weekly", "hourly", "daily").
    load : callable
        Builds the named-column map (full series).
    exoMode : ExoMode
        `Given` => the arms use `exo` as is; `Auto` => the screening picks hp["exoK"] of them.
    """
    name: str
    resp: str
    exo: List[str]
    schema: str
    expected_signs: Dict[str, int]
    freq: str
    load: Callable[[], Dict[str, np.ndarray]]
    exoMode: ExoMode = ExoMode.Given

    @property
    def expectedSigns(self) -> Dict[str, int]:
        return self.expected_signs

    def shortlist(self, scorer) -> List[str]:
        """
        The exogenous variables the arms build on: the list itself (`Given`) or the screening's
        shortlist from the training rows (`Auto`, hp["exoK"] of the pool).

        Parameters
        ----------
        scorer : FormulaScorer
            The fit-and-score engine (its data, training rows and hyper-parameters).
        """
        if self.exoMode is ExoMode.Given:
            return list(self.exo)
        k = int(scorer.hp["exoK"])
        data = scorer.fullSet
        return ExoScreen.shortlist(data[self.resp], {v: data[v] for v in self.exo},
                                   int(scorer.hp["maxLag"]), scorer.trSize, k)[1]

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


class Datasets:
    """The dataset configurations of the paper's experiments (ScalaTion's `Datasets`); each `load`
    builds the named-column map from the example loaders."""

    COVID2_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; icu (patients in intensive care) "
                     "and pRate (share of tests that are positive) are leading indicators of deaths.")
    COVID_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; hosp and icu (patients "
                    "in hospital / intensive care) are leading indicators of deaths.")
    COVID_ALL_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; the candidate drivers are new cases, the "
                        "reproduction rate, icu and hospital patients, tests, the positivity rate, tests per case, "
                        "new vaccinations and excess mortality.")
    COVID_FILE_SCHEMA = "Dataset: weekly US Covid-19; y = new deaths; the other columns of the file are the candidate drivers."
    ILI_SCHEMA = ("Dataset: weekly US influenza-like illness (CDC ILINet, 2002-2025); y = total ILI patient count; "
                  "wili = population-weighted ILI percentage; ot = total patient visits.")
    ETTH1_SCHEMA = ("Dataset: hourly electricity transformer (ETTh1); y = oil temperature of the transformer; "
                    "the candidate drivers are the high, middle and low useful/useless power loads "
                    "(hufl, hull, mufl, mull, lufl, lull); the oil warms with load and cools slowly.")
    GEF14_SCHEMA = ("Dataset: daily electric load of a US utility (GEFCom2014); y = daily peak load; "
                    "temp and tmax are the daily mean and maximum temperature (deg F); load is lowest at "
                    "mild temperatures and rises in cold (heating) and hot (cooling) weather.")

    #: Covid-19 weekly deaths with icu patients and test positivity rate (`Example_Covid.BEST2_EXO`): we choose.
    COVID2 = DatasetConfig("Covid-19 (icu, pRate)", "y", ["icu", "pRate"], COVID2_SCHEMA,
                           {"icu": 1, "pRate": 1}, "weekly",
                           lambda: Example_Covid.context(["icu_patients", "positive_rate"]))
    #: Covid-19 weekly deaths with hospital and icu patients.
    COVID = DatasetConfig("Covid-19 (hosp, icu)", "y", ["hosp", "icu"], COVID_SCHEMA,
                          {"hosp": 1, "icu": 1}, "weekly",
                          lambda: Example_Covid.context(["hosp_patients", "icu_patients"]))
    #: The pool of Covid-19 columns an automatic selection may pick from and their short names.
    COVID_POOL = ["new_cases", "reproduction_rate", "icu_patients", "hosp_patients", "new_tests",
                  "positive_rate", "tests_per_case", "new_vaccinations", "excess_mortality"]
    COVID_NAMES = ["cases", "rRate", "icu", "hosp", "tests", "pRate", "tpc", "vacc", "exMort"]
    #: Covid-19 weekly deaths with the whole pool: the system chooses (`ExoMode.Auto`).
    COVID_ALL = DatasetConfig("Covid-19 (auto exo)", "y", list(COVID_NAMES), COVID_ALL_SCHEMA,
                              {"cases": 1, "rRate": 1, "icu": 1, "hosp": 1, "pRate": 1, "tpc": -1, "vacc": -1, "exMort": 1},
                              "weekly", lambda: Example_Covid.context(Datasets.COVID_POOL, Datasets.COVID_NAMES), ExoMode.Auto)
    #: Influenza-like illness weekly counts with the weighted ILI percentage and total visits.
    ILI = DatasetConfig("ILI", "y", ["wili", "ot"], ILI_SCHEMA, {"wili": 1}, "weekly",
                        lambda: Example_ILI.context())
    #: GEFCom2014 daily peak load with the daily mean and maximum temperature: we choose (the relation is V-shaped, no sign expected).
    GEF14 = DatasetConfig("GEFCom2014 load (temp, tmax)", "y", ["temp", "tmax"], GEF14_SCHEMA,
                          {}, "daily", lambda: Example_GEFCom.context(["temp", "tmax"]))

    @classmethod
    def ettH1(cls, rng: Optional[range] = None) -> DatasetConfig:
        """The ETTh1 configuration: hourly oil temperature with the six loads as the pool, the
        system choosing hp["exoK"] of them (`ExoMode.Auto`); every load is expected to warm the oil (+1)."""
        nm = "ETTh1 (auto exo)" if rng is None else f"ETTh1 rows {rng.start}..{rng.stop - 1} (auto exo)"
        return DatasetConfig(nm, "y", list(Example_ETTh1.LOAD_NAMES), cls.ETTH1_SCHEMA,
                             {v: 1 for v in Example_ETTh1.LOAD_NAMES}, "hourly",
                             lambda: Example_ETTh1.context(rng=rng), ExoMode.Auto)

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
        df = _read(fileName).iloc[:, skipCol:]
        if rng is not None:
            df = df.iloc[rng.start:rng.stop].reset_index(drop=True)
        pool = ExoScreen.pool(df, None, resp, trSize(len(df)), exclude)
        y, xe, dropped = pool.y, pool.xe, pool.dropped
        for nm, why in dropped:
            print(f"Datasets.fromFile: {nm} dropped -- {why}")
        data = {"y": y, **xe}
        return DatasetConfig(name, "y", list(xe.keys()), schema, {}, freq, lambda: data, ExoMode.Auto)

    _covid_file: Optional[DatasetConfig] = None

    @classmethod
    def covidFile(cls) -> DatasetConfig:
        """
        Covid-19 weekly deaths straight from the file (the 116 rows of `Example_Covid`): the pool is
        whatever the screening finds eligible, minus excess mortality (another count of the same
        deaths) and new cases (not comparable across testing regimes); the system chooses.  Lazy:
        the file is read when first used (ScalaTion's `lazy val COVID_FILE`).
        """
        if cls._covid_file is None:
            cls._covid_file = cls.fromFile("Covid-19 (file, auto exo)", Example_Covid.FILE, Example_Covid.RESPONSE,
                                           cls.COVID_FILE_SCHEMA, "weekly", range(0, Example_Covid.N_ROWS),
                                           ["excess_mortality", "new_cases"])
        return cls._covid_file

    @classmethod
    def get(cls, name: str) -> DatasetConfig:
        """A configuration by short name: covid2, covid, covid_all, covid_file, ili, etth1, etth1_1y, gef14."""
        key = name.strip().lower()
        if key == "covid_file":
            return cls.covidFile()
        if key not in cls.ALL:
            raise KeyError(f"dataset '{name}' unknown: {', '.join(list(cls.ALL) + ['covid_file'])}")
        return cls.ALL[key]


#: ETTh1, the full series (17420 hourly rows) and its first year (8760 rows: one cycle of the seasons).
Datasets.ETTH1 = Datasets.ettH1()
Datasets.ETTH1_1Y = Datasets.ettH1(Example_ETTh1.ONE_YEAR)
Datasets.ALL = {"covid2": Datasets.COVID2, "covid": Datasets.COVID, "covid_all": Datasets.COVID_ALL, "ili": Datasets.ILI,
                "etth1": Datasets.ETTH1, "etth1_1y": Datasets.ETTH1_1Y, "gef14": Datasets.GEF14}
