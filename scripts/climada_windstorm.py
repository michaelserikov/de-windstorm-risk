"""German windstorm damage with CLIMADA: station footprints, population exposure, cubic impact function.

Hazard: maximum gust per event as a ratio to the station's 98th percentile.
Exposure: population weights of the stations, scaled so that total value is 1.
Impact function: mean damage ratio proportional to (ratio - 1) to a power, zero below 1.
Calibration: one scale factor to GDV insured property losses (prices and portfolio of 2020),
checked by leave-one-out. Illustrative: no building data, no vulnerability classes.

Run with a Python environment that has CLIMADA.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from climada.engine import ImpactCalc
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Centroids, Hazard
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "climada"
# GDV, Sachversicherung, million EUR, portfolio and prices 2020, status February 2022.
# Ylenia, Zeynep, Antonia and Sabine are preliminary. Dates identify the event in the catalogue.
GDV = {"2007-01-18": ("Kyrill", 3350), "2002-10-27": ("Jeanett", 1340), "2022-02-18": ("Ylenia, Zeynep, Antonia", 1250),
       "2018-01-18": ("Friederike", 1160), "2015-03-31": ("Niklas", 730), "2010-02-28": ("Xynthia", 730),
       "2020-02-10": ("Sabine", 600), "2008-03-01": ("Emma", 610)}


def build():
    st = pd.read_csv(DIR / "stations.csv", index_col=0)
    ev = pd.read_csv(DIR / "events.csv", parse_dates=["start", "end", "peak_day"])
    gust = pd.read_csv(DIR / "event_max_gust.csv", index_col=0)
    gust.columns = gust.columns.astype(int)
    ratio = gust[st.index].div(st["v98"]).fillna(0.0)
    centroids = Centroids(lat=st["lat"].to_numpy(), lon=st["lon"].to_numpy())
    # one event per row, frequency from 35 winters
    n = len(ev)
    haz = Hazard(haz_type="WS", units="ratio", centroids=centroids, event_id=np.arange(1, n + 1),
                 event_name=[str(i) for i in range(n)], date=ev["peak_day"].map(pd.Timestamp.toordinal).to_numpy(),
                 frequency=np.full(n, 1 / 35),
                 intensity=sparse.csr_matrix(ratio.to_numpy()), fraction=sparse.csr_matrix(ratio.shape))
    exp = Exposures(pd.DataFrame({"latitude": st["lat"].to_numpy(), "longitude": st["lon"].to_numpy(),
                                  "value": st["weight"].to_numpy() / st["weight"].sum(),
                                  "impf_WS": 1, "centr_WS": np.arange(len(st))}))
    return st, ev, haz, exp


def impact(haz, exp, power):
    x = np.linspace(0, 4, 401)
    mdd = np.clip(x - 1, 0, None) ** power
    fn = ImpactFunc(haz_type="WS", id=1, intensity=x, mdd=mdd / mdd.max(), paa=np.ones_like(x), intensity_unit="ratio")
    imp = ImpactCalc(exp, ImpactFuncSet([fn]), haz).impact(assign_centroids=False)
    return imp.at_event


def match(ev):
    out = {}
    for day, (name, loss) in GDV.items():
        d = pd.Timestamp(day)
        hit = ev.index[(ev["start"] <= d) & (ev["end"] >= d)]
        if len(hit):
            out[int(hit[0])] = (name, loss)
    return out


def main():
    st, ev, haz, exp = build()
    pairs = match(ev)
    idx = list(pairs)
    obs = np.array([pairs[i][1] for i in idx], dtype=float)
    results = {}
    tables = {}
    for power in (2, 3, 4):
        model = impact(haz, exp, power)
        m = model[idx]
        scale = np.exp(np.mean(np.log(obs / m)))
        loo = np.array([np.exp(np.mean(np.log(np.delete(obs, k) / np.delete(m, k)))) * m[k] for k in range(len(idx))])
        logerr = np.log(loo / obs)
        annual = pd.Series(model * scale).groupby(ev["season"]).sum()
        results[power] = {"scale_million_eur": float(scale), "storms_matched": len(idx),
                          "loo_median_abs_error_factor": float(np.exp(np.median(np.abs(logerr)))),
                          "loo_max_abs_error_factor": float(np.exp(np.max(np.abs(logerr)))),
                          "rank_corr": float(pd.Series(m).corr(pd.Series(obs), method="kendall")),
                          "corr_with_loss_index": float(pd.Series(model).corr(ev["loss_index"])),
                          "mean_annual_loss_mio": float(annual.mean()),
                          "largest_event_mio": float((model * scale).max())}
        tables[power] = pd.DataFrame({"storm": [pairs[i][0] for i in idx], "gdv_mio": obs, "model_mio": m * scale,
                                      "leave_one_out_mio": loo})
    best = min(results, key=lambda p: results[p]["loo_median_abs_error_factor"])
    tables[best].round(0).to_csv(DIR / "calibration_table.csv", index=False)
    ep = pd.read_csv(ROOT / "results" / "ep_table.csv")
    summary = {"best_power": best, "by_power": results, "ep_table_columns": ep.columns.tolist()}
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(tables[best].round(0).to_string(index=False))
    print(ep.to_string(index=False))
    print(st["weight"].sum())


if __name__ == "__main__":
    main()
