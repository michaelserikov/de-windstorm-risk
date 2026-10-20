"""Run the v0.3 compound wind and rainfall analysis (needs the v0.1 event set: run `windrisk.run` first).

    python -m windrisk.compound_run

Writes results/compound/ and figures/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import compound as cp, data, hazard as hz, plots
from .run import END, ROOT, START, load

MIN_COVERAGE = 0.9
PERIODS = (2, 5, 10, 20, 50)


def main() -> dict:
    cache, res, out, figs = ROOT / "data" / "raw", ROOT / "results", ROOT / "results" / "compound", ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    st, daily = load(cache)
    stations = pd.read_csv(res / "stations.csv").set_index("station_id")
    events = pd.read_csv(res / "events.csv", parse_dates=["start", "end", "peak_day"])
    sel = stations.index.tolist()

    d = daily[daily["station_id"].isin(sel)]
    gust = hz.gust_matrix(d, START, END)[sel]
    rain = d[(d["date"] >= START) & (d["date"] <= END) & d["date"].dt.month.isin(hz.WINTER)].pivot_table(
        index="date", columns="station_id", values="rsk")
    both = [s for s in sel if s in rain.columns and rain[s].notna().mean() >= MIN_COVERAGE]
    gust, rain = gust[both], rain[both]

    # station level
    chi = cp.station_chi(gust, rain, 0.98).join(stations[["lat", "lon", "state"]])
    chi.to_csv(out / "station_chi.csv", float_format="%.5f")
    chi_levels = []
    for q in (0.90, 0.95, 0.98, 0.99):
        c = cp.station_chi(gust, rain, q)
        chi_levels.append({"q": q, "independence": 1 - q, "chi_median": float(c["chi"].median()),
                           "chi_p10": float(c["chi"].quantile(0.1)), "chi_p90": float(c["chi"].quantile(0.9)),
                           "ratio_to_independence_median": float(c["ratio_to_independence"].median())})
    chi_levels = pd.DataFrame(chi_levels)
    chi_levels.to_csv(out / "chi_by_level.csv", index=False, float_format="%.5f")

    # event level
    r98 = rain.quantile(0.98)
    w = stations.loc[both, "weight"]
    events["rain_ratio"] = cp.event_rain(events, rain, r98, w)
    events.to_csv(out / "events_wind_rain.csv", index=False, float_format="%.6f")
    u, v = cp.pseudo_obs(events["loss_index"].to_numpy()), cp.pseudo_obs(events["rain_ratio"].to_numpy())
    tau, p_tau = kendalltau(events["loss_index"], events["rain_ratio"])
    theta, ll_gum = cp.fit_gumbel(u, v)
    rho, ll_gau = cp.fit_gaussian(u, v)
    aic = {"independence": 0.0, "gaussian": 2 - 2 * ll_gau, "gumbel": 2 - 2 * ll_gum}
    n_seasons = len(set(events["season"]))
    rate = len(events) / n_seasons
    jrp = cp.joint_return_periods(rate, PERIODS, theta, rho, u, v)
    jrp.to_csv(out / "joint_return_periods.csv", index=False, float_format="%.4f")
    diag = cp.tail_diagnostic(u, v, theta, rho)
    diag.to_csv(out / "tail_diagnostic.csv", index=False, float_format="%.2f")

    top = events.nlargest(10, "loss_index")
    rain_rank = events["rain_ratio"].rank(ascending=False)
    summary = {
        "stations": len(both), "events": int(len(events)), "events_per_winter": rate,
        "station_level": chi_levels.round(4).to_dict(orient="records"),
        "event_level": {"kendall_tau": float(tau), "p": float(p_tau),
                        "gumbel_theta": theta, "gumbel_lambda_upper": float(2 - 2 ** (1 / theta)),
                        "gaussian_rho": rho, "aic": aic, "best_by_aic": min(aic, key=aic.get)},
        "joint_return_periods": jrp.round(2).to_dict(orient="records"),
        "tail_diagnostic": diag.round(2).to_dict(orient="records"),
        "top10_wind_events_rain_rank": {str(r["start"].date()) + (f" {r['storm']}" if isinstance(r["storm"], str) else ""):
                                        int(rain_rank[i]) for i, r in top.iterrows()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    plots.chi_map(chi, figs / "compound_chi_map.png")
    plots.wind_rain_scatter(events, figs / "compound_events.png")
    plots.joint_rp(jrp, figs / "compound_joint_return_periods.png")
    return summary


if __name__ == "__main__":
    main()
