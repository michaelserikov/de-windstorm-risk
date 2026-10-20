"""Run the windstorm model: DWD gusts -> event set -> loss index -> EVT -> annual loss metrics.

    python -m windrisk.run        # downloads DWD and Eurostat data on first run (about 180 MB, cached)

Writes results/ (CSV, JSON) and figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme, kendalltau

from . import data, hazard as hz, plots, risk

ROOT = Path(__file__).resolve().parents[2]
START, END = "1990-10-01", "2025-03-31"   # 35 extended winters, 1990/91 to 2024/25
MIN_COVERAGE = 0.9
THRESHOLD_Q = 0.90                         # POT threshold: 90th percentile of event losses
# Named storms used to check the event set (peak dates from public storm records)
NAMED = {"1999-12-26": "Lothar", "2002-10-27": "Jeanett", "2007-01-18": "Kyrill", "2008-03-01": "Emma",
         "2015-03-31": "Niklas", "2018-01-18": "Friederike", "2020-02-10": "Sabine", "2022-02-18": "Ylenia/Zeynep"}


def load(cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = data.stations(cache)
    idx = data.file_index(cache)
    cand = st[(st["start"] <= "1990-01-01") & (st["end"] >= "2025-12-31") & st["station_id"].isin(idx)]
    daily = pd.concat([data.station_daily(i, idx[i], cache) for i in cand["station_id"]], ignore_index=True)
    return st, daily


def name_events(ev: pd.DataFrame) -> pd.Series:
    names = pd.Series("", index=ev.index)
    for d, n in NAMED.items():
        hit = (ev["start"] <= d) & (ev["end"] >= d)
        names[hit] = n
    return names


def main() -> dict:
    cache, out, figs = ROOT / "data" / "raw", ROOT / "results", ROOT / "figures"
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    st, daily = load(cache)
    winter = hz.gust_matrix(daily, START, END)
    sel = hz.select_stations(winter, MIN_COVERAGE)
    winter = winter[sel]
    all_days = daily[(daily["date"] >= START) & (daily["date"] <= END)].pivot_table(
        index="date", columns="station_id", values="fx")[sel]
    v98 = all_days.quantile(0.98)
    pop = data.population(cache)
    w = hz.station_weights(st, sel, pop)
    stations = st.set_index("station_id").loc[sel].assign(v98=v98, weight=w)
    stations.to_csv(out / "stations.csv", float_format="%.5f")

    ev = hz.events(winter, v98, w)
    ev["storm"] = name_events(ev)
    ev.to_csv(out / "events.csv", index=False, float_format="%.6f")

    seasons = list(range(int(hz.season_of(pd.Series([pd.Timestamp(START)]))[0]),
                         int(hz.season_of(pd.Series([pd.Timestamp(END)]))[0]) + 1))
    n_seasons = len(seasons)
    loss = ev["loss_index"].to_numpy()

    # threshold stability of the GPD shape
    stab = []
    for q in (0.80, 0.85, 0.90, 0.95):
        u_q = float(np.quantile(loss, q))
        xi_q, s_q = risk.fit_gpd(loss[loss > u_q] - u_q)
        sim_q = risk.simulate_seasons(loss, n_seasons, u_q, xi=xi_q, sigma=s_q, seed=5)
        stab.append({"quantile": q, "threshold": u_q, "n_above": int((loss > u_q).sum()), "xi": xi_q, "sigma": s_q,
                     "aal": float(sim_q["aggregate"].mean()), "aep_200": float(sim_q["aggregate"].quantile(0.995))})
    stab = pd.DataFrame(stab)
    stab.to_csv(out / "gpd_threshold_stability.csv", index=False, float_format="%.5f")

    u = float(np.quantile(loss, THRESHOLD_Q))
    xi, sigma = risk.fit_gpd(loss[loss > u] - u)
    rate = (loss > u).sum() / n_seasons
    sim = risk.simulate_seasons(loss, n_seasons, u, xi=xi, sigma=sigma)
    ep = risk.ep_table(sim)
    season_tot = ev.groupby("season")["loss_index"].agg(["sum", "max", "count"]).reindex(seasons, fill_value=0.0)
    aal_emp = float(season_tot["sum"].mean())
    aal = float(sim["aggregate"].mean())
    ep["aep_x_aal"] = ep["aep"] / aal
    ep["oep_x_aal"] = ep["oep"] / aal
    c, loc, scale = risk.fit_gev_maxima(season_tot["max"].to_numpy())
    ep["oep_gev"] = [float(genextreme.ppf(1 - 1 / t, c, loc=loc, scale=scale)) for t in ep["return_period"]]
    ep.to_csv(out / "ep_table.csv", index=False, float_format="%.5f")
    season_tot.to_csv(out / "seasons.csv", float_format="%.6f")

    boot = risk.bootstrap_1in200(ev, seasons, u)
    boot.to_csv(out / "bootstrap.csv", index=False, float_format="%.6f")

    kyrill = float(ev.loc[ev["storm"] == "Kyrill", "loss_index"].iloc[0])
    rp_kyrill_model = float(1 / (sim["max_event"] >= kyrill).mean())
    tau_sum, p_sum = kendalltau(seasons, season_tot["sum"])
    tau_cnt, p_cnt = kendalltau(seasons, (ev[ev["loss_index"] > u].groupby("season").size()
                                          .reindex(seasons, fill_value=0)))

    summary = {
        "period": [START, END], "seasons": n_seasons, "stations": len(sel), "events": int(len(ev)),
        "threshold": {"quantile": THRESHOLD_Q, "u": u, "events_above": int((loss > u).sum()), "rate_per_season": rate},
        "gpd": {"xi": xi, "sigma": sigma},
        "gev_seasonal_max": {"c_scipy": c, "loc": loc, "scale": scale},
        "aal": {"model": aal, "empirical": aal_emp},
        "aep_1in200": float(sim["aggregate"].quantile(0.995)), "oep_1in200": float(sim["max_event"].quantile(0.995)),
        "tvar_995_aggregate": risk.tvar(sim["aggregate"]),
        "bootstrap": {k: {"p05": float(boot[k].quantile(0.05)), "median": float(boot[k].median()),
                          "p95": float(boot[k].quantile(0.95))} for k in ("xi", "aal", "aep_200", "oep_200")},
        "kyrill": {"loss_index": kyrill, "model_return_period_occurrence": rp_kyrill_model,
                   "empirical_rank": int(ev.index[ev["storm"] == "Kyrill"][0]) + 1},
        "named_storm_ranks": {n: int(ev.index[ev["storm"] == n][0]) + 1 for n in NAMED.values() if (ev["storm"] == n).any()},
        "trend_kendall": {"seasonal_sum": {"tau": float(tau_sum), "p": float(p_sum)},
                          "events_above_u": {"tau": float(tau_cnt), "p": float(p_cnt)}},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    plots.seasons(season_tot, ev, figs / "seasonal_losses.png")
    plots.ep_curves(sim, season_tot, boot, aal, figs / "ep_curves.png")
    footprints = {n: winter.loc[ev.loc[ev["storm"] == n, "start"].iloc[0]:ev.loc[ev["storm"] == n, "end"].iloc[0]].max()
                  for n in ("Kyrill", "Lothar")}
    plots.footprints(footprints, stations, figs / "footprints.png")
    return summary


if __name__ == "__main__":
    main()
