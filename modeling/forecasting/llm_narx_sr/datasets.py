"""
Datasets of the LLM-NARX-SR Experiments

The example datasets as named-column maps (dict name -> np.ndarray) for the formula layer,
and `DatasetConfig`: what an experiment runs ON -- the response, the pool of exogenous
variables, the domain sentence a named prompt shows, the sampling-frequency word, and the
coefficient signs domain knowledge expects.

    Covid-19     weekly US deaths, 116 weeks (data/covid_19_weekly.csv), icu patients and
                 test positivity rate as drivers
    ILI          weekly US influenza-like illness (data/national_illness_clip.csv) -- the
                 file is not in the repository; `Example_ILI` tells you where it goes
    ETTh1        hourly oil temperature (OT) of an electricity transformer, 17,420 rows,
                 six load series as drivers (data/ETTh1.csv, Zhou et al. 2021)
    GEFCom2014   daily peak load of a US utility, 2,099 days, with the daily mean and max
                 temperature (data/gefcom2014_load_daily.csv, built by tools/build_gefcom_daily.py
                 from the competition's L1-train.csv; Hong et al. 2016)

The Python counterpart of ScalaTion's `Example_Covid` / `Example_ILI` / `Example_ETTh1` /
`Example_GEFCom` loaders and of `llm_narx_sr.Datasets`.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: The data directory: <repo>/data (override with the FORECASTER_DATA environment variable).
import os
DATA_DIR = Path(os.environ.get("FORECASTER_DATA", Path(__file__).resolve().parents[3] / "data"))


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
    first 116 weeks used (the rows of the ScalaTion tests and of Examples/Example_Covid19.py).
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
    def load_data(cls, x_strs: Sequence[str] = ("icu_patients", "positive_rate"), n_rows: Optional[int] = N_ROWS):
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
    Weekly US influenza-like illness (data/national_illness_clip.csv, the file of the
    ScalaTion ILI tests): the total ILI patient count as the response, with the weighted ILI
    percentage (wili) and total patient visits (ot) as drivers.  The file is NOT shipped with
    the repository: copy yours into data/ (or point FORECASTER_DATA at its directory).
    """
    FILE = "national_illness_clip.csv"
    RESPONSE = "ILITOTAL"
    EXO = ["%WEIGHTED ILI", "OT"]
    NAMES = ["wili", "ot"]

    @classmethod
    def frame(cls) -> pd.DataFrame:
        return _read(cls.FILE)

    @classmethod
    def context(cls, x_strs: Sequence[str] = EXO, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        names = cls.NAMES if names is None and list(x_strs) == cls.EXO else names
        return _context(cls.frame(), cls.RESPONSE, x_strs, names)


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
    def load_data(cls, x_strs: Sequence[str] = LOADS, rng: Optional[range] = None):
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
    def load_data(cls, x_strs: Sequence[str] = ("temp", "tmax")):
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


@dataclass
class DatasetConfig:
    """
    What an experiment runs on.

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
        The expected coefficient sign per variable (+1 / -1; unknown => absent).
    freq : str
        The sampling-frequency word for the blind prompt ("weekly", "hourly", "daily").
    load : callable
        Builds the named-column map (full series).
    """
    name: str
    resp: str
    exo: List[str]
    schema: str
    expected_signs: Dict[str, int]
    freq: str
    load: Callable[[], Dict[str, np.ndarray]]

    def baseline_formula_str(self, max_lag: int, exo: Optional[Sequence[str]] = None) -> str:
        """The plain-ARX baseline: full contiguous lag windows on the response and the exogenous
        variables, no transforms, e.g., "y ~= y~(6,1) + icu~(6,1) + pRate~(6,1)"."""
        exo = self.exo if exo is None else list(exo)
        return f"{self.resp} ~= " + " + ".join(f"{v}~({max_lag},1)" for v in [self.resp] + exo)

    def blind_names(self) -> Dict[str, str]:
        """The blind display name per variable (y, x1, x2, ...)."""
        return {self.resp: "y", **{v: f"x{i + 1}" for i, v in enumerate(self.exo)}}


class Datasets:
    """The dataset configurations of the paper's experiments."""

    COVID2_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; icu (patients in intensive care) "
                     "and pRate (share of tests that are positive) are leading indicators of deaths.")
    COVID_SCHEMA = ("Dataset: weekly US Covid-19; y = new deaths; hosp and icu (patients "
                    "in hospital / intensive care) are leading indicators of deaths.")
    ILI_SCHEMA = ("Dataset: weekly US influenza-like illness; y = total ILI patient count; "
                  "wili = population-weighted ILI percentage; ot = total patient visits.")
    ETTH1_SCHEMA = ("Dataset: hourly oil temperature of an electricity transformer (ETTh1); y = OT; the six "
                    "load series (high / middle / low useful and useless load) heat the oil with a delay.")
    GEF14_SCHEMA = ("Dataset: daily peak electricity load of a US utility (GEFCom2014); y = peak load; temp and "
                    "tmax are the daily mean and maximum temperature in deg F; the load-temperature relation is "
                    "V-shaped (heating below ~55 F, cooling above ~65 F).")

    COVID2 = DatasetConfig("Covid-19 (icu, pRate)", "y", ["icu", "pRate"], COVID2_SCHEMA,
                           {"icu": 1, "pRate": 1}, "weekly",
                           lambda: Example_Covid.context(["icu_patients", "positive_rate"]))
    COVID = DatasetConfig("Covid-19 (hosp, icu)", "y", ["hosp", "icu"], COVID_SCHEMA,
                          {"hosp": 1, "icu": 1}, "weekly",
                          lambda: Example_Covid.context(["hosp_patients", "icu_patients"]))
    ILI = DatasetConfig("ILI", "y", ["wili", "ot"], ILI_SCHEMA, {"wili": 1, "ot": 1}, "weekly",
                        lambda: Example_ILI.context())
    ETTH1 = DatasetConfig("ETTh1 (six loads)", "y", list(Example_ETTh1.LOAD_NAMES), ETTH1_SCHEMA,
                          {nm: 1 for nm in Example_ETTh1.LOAD_NAMES}, "hourly",
                          lambda: Example_ETTh1.context())
    ETTH1_1Y = DatasetConfig("ETTh1 first year (six loads)", "y", list(Example_ETTh1.LOAD_NAMES), ETTH1_SCHEMA,
                             {nm: 1 for nm in Example_ETTh1.LOAD_NAMES}, "hourly",
                             lambda: Example_ETTh1.context(rng=Example_ETTh1.ONE_YEAR))
    GEF14 = DatasetConfig("GEFCom2014 load (temp, tmax)", "y", ["temp", "tmax"], GEF14_SCHEMA,
                          {}, "daily", lambda: Example_GEFCom.context(["temp", "tmax"]))

    ALL = {"covid2": COVID2, "covid": COVID, "ili": ILI, "etth1": ETTH1, "etth1_1y": ETTH1_1Y, "gef14": GEF14}

    @classmethod
    def get(cls, name: str) -> DatasetConfig:
        """A configuration by short name: covid2, covid, ili, etth1, etth1_1y, gef14."""
        key = name.strip().lower()
        if key not in cls.ALL:
            raise KeyError(f"dataset '{name}' unknown: {', '.join(cls.ALL)}")
        return cls.ALL[key]
