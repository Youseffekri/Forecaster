"""
Build data/gefcom2014_load_daily.csv from the GEFCom2014 load-track file L1-train.csv

The GEFCom2014 load forecasting track (Hong et al., 2016, IJF) gives the hourly load of a
US utility for 2005-2010 together with 25 hourly temperature stations (w1..w25).  This
script aggregates every day (hour-ending 1..24, so the midnight row belongs to the day
before) into:

    date, peak (daily peak load), avg (daily mean load), temp (daily mean of the 25-station
    mean temperature, deg F), tmax, tmin (its daily max / min), weekend (0/1)

Only complete days (24 hours with a load value) are kept: 2,099 days, 2005-01-01 to
2010-09-30.  The timestamps of the source are M/D/YYYY without separators or zero padding
("112005 1:00" = Jan 1 2005, hour-ending 1), so a 3-digit month-day prefix is ambiguous
(1112005: Jan 11 or Nov 1) and is resolved by the chronological order of the rows.

Source file (one public copy): https://raw.githubusercontent.com/w33ab00/IRDM2016/master/data/L1-train.csv

Usage:  python tools/build_gefcom_daily.py L1-train.csv data/gefcom2014_load_daily.csv

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
License: MIT License
Note: AI assisted code
"""

import datetime as dt
import sys

import pandas as pd


def build(src: str, dst: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    df = df[df["LOAD"].notna()].reset_index(drop=True)
    dates, prev = [], None
    for ts in df["TIMESTAMP"].astype(str):
        d, h = ts.split(" ")
        year, pre, hour = int(d[-4:]), d[:-4], int(h.split(":")[0])
        cands = []
        for k in (1, 2):                                   # month of 1 or 2 digits
            if len(pre) > k:
                m, dd = int(pre[:k]), int(pre[k:])
                if 1 <= m <= 12 and 1 <= dd <= 31:
                    try:
                        cands.append(dt.date(year, m, dd))
                    except ValueError:
                        pass
        if prev is not None:
            cands = [c for c in cands if c >= prev] or cands
        c = cands[0] if prev is None else min(cands, key=lambda c: (c - prev).days)
        if hour == 0:                                      # hour-ending 24 of the previous day
            c = c - dt.timedelta(days=1)
        dates.append(c)
        prev = max(prev, c) if prev else c
    df["date"] = pd.to_datetime(dates)
    df["tmean"] = df[[f"w{i}" for i in range(1, 26)]].mean(axis=1)
    g = df.groupby("date")
    daily = pd.DataFrame({"peak": g["LOAD"].max(), "avg": g["LOAD"].mean(), "temp": g["tmean"].mean(),
                          "tmax": g["tmean"].max(), "tmin": g["tmean"].min()})
    daily["weekend"] = (daily.index.dayofweek >= 5).astype(int)
    daily = daily[g["LOAD"].count() == 24]
    daily.round(4).to_csv(dst, index_label="date", date_format="%Y-%m-%d")
    return daily


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "L1-train.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/gefcom2014_load_daily.csv"
    daily = build(src, dst)
    print(f"{len(daily)} days, {daily.index.min().date()} .. {daily.index.max().date()} -> {dst}")
    bands = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
    print("mean peak per 10 F band:", [round(daily[(daily.temp >= a) & (daily.temp < b)].peak.mean()) for a, b in bands])
