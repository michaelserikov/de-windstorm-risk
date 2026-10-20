"""From station gusts to a storm event set with a loss index.

Loss index (Klawa and Ulbrich, 2003): at each station, damage starts when the daily maximum gust
exceeds the local 98th percentile v98 and grows with the cube of the excess,
    l_j = max(0, v_j / v98_j - 1) ** 3,
which accounts for local adaptation of buildings to their usual wind climate. The national index
weights stations by the population they represent: each state's population (Eurostat, 2024) is
split equally between the selected stations in that state.

Season: extended winter, October to March, labelled by the year in which it ends (2007 = Oct
2006 to Mar 2007). Events: storm days (index above zero) closer than `gap_days` apart are merged,
up to `max_days` per event (a 72-hour clause); the event footprint is each station's maximum gust
over the event days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINTER = (10, 11, 12, 1, 2, 3)


def season_of(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return np.where(d.month >= 10, d.year + 1, d.year)


def gust_matrix(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Days x stations matrix of daily maximum gust (m/s) for winter days in [start, end]."""
    d = daily[(daily["date"] >= start) & (daily["date"] <= end) & daily["date"].dt.month.isin(WINTER)]
    return d.pivot_table(index="date", columns="station_id", values="fx")


def select_stations(gusts: pd.DataFrame, min_coverage: float = 0.9) -> list[int]:
    """Stations with gust data on at least `min_coverage` of winter days."""
    cov = gusts.notna().mean()
    return cov[cov >= min_coverage].index.tolist()


def station_weights(stations: pd.DataFrame, selected: list[int], population: pd.Series) -> pd.Series:
    """Weight per station = population of its state / number of selected stations in that state; sums to 1."""
    st = stations.set_index("station_id").loc[selected]
    n = st.groupby("state").size()
    w = st["state"].map(population) / st["state"].map(n)
    if w.isna().any():
        raise ValueError(f"no population for states: {sorted(st.loc[w.isna(), 'state'].unique())}")
    return (w / w.sum()).rename("weight")


def loss_terms(gusts: pd.DataFrame, v98: pd.Series) -> pd.DataFrame:
    """Station loss terms max(0, v/v98 - 1)^3; missing gusts count as zero."""
    ratio = gusts.div(v98, axis=1)
    return (ratio - 1).clip(lower=0).pow(3).fillna(0.0)


def daily_index(gusts: pd.DataFrame, v98: pd.Series, weights: pd.Series) -> pd.Series:
    """National daily loss index, renormalised to the stations reporting that day."""
    terms = loss_terms(gusts, v98)
    reporting = gusts.notna().mul(weights, axis=1).sum(axis=1)
    return (terms.mul(weights, axis=1).sum(axis=1) / reporting.replace(0, np.nan)).fillna(0.0).rename("loss_index")


def events(gusts: pd.DataFrame, v98: pd.Series, weights: pd.Series, gap_days: int = 1,
           max_days: int = 3, min_daily: float = 0.0) -> pd.DataFrame:
    """Merge storm days into events and compute each event's footprint loss index."""
    daily = daily_index(gusts, v98, weights)
    storm_days = daily[daily > min_daily].index.sort_values()
    groups, current = [], []
    for d in storm_days:
        if current and ((d - current[-1]).days > gap_days + 1 or (d - current[0]).days >= max_days):
            groups.append(current)
            current = []
        current.append(d)
    if current:
        groups.append(current)
    rows = []
    for g in groups:
        footprint = gusts.loc[g].max()                       # station maximum over the event days
        terms = loss_terms(footprint.to_frame().T, v98).iloc[0]
        reporting = footprint.notna().mul(weights).sum()
        peak = daily.loc[g].idxmax()
        rows.append({"start": g[0], "end": g[-1], "peak_day": peak, "days": len(g),
                     "loss_index": float(terms.mul(weights).sum() / reporting),
                     "stations_above_v98": int((footprint > v98).sum()),
                     "max_gust": float(footprint.max())})
    ev = pd.DataFrame(rows)
    ev["season"] = season_of(ev["peak_day"])
    return ev.sort_values("loss_index", ascending=False).reset_index(drop=True)
