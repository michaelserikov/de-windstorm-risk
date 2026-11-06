"""Parametric storm cover and a cat bond layer on the open-data storm model.

Part 1, basis risk. A hypothetical insurer holds 10% of the residential storm market in four northern states
(Hamburg, Bremen, Niedersachsen, Schleswig-Holstein). Its event loss is the regional part of the calibrated loss index
(population-weighted station terms of those states, times the euro scale fitted to GDV losses, times the share).
It buys a per-event layer (attachment exceeded by one event in 5 winters, exhaustion by one in 25) and compares
indemnity with four index triggers: the national index (an industry-loss trigger), one station (Hamburg), three city
stations and all regional stations with equal weights, plus a three-step cover on the Hamburg gust. Each trigger is
turned into a modelled loss by a power curve matched to the quantiles of the 25 largest trigger values and losses in
half of the winters and tested on the other half, and pays the same layer on that modelled loss. Because the loss proxy is
built from the same station gusts, this is the basis risk of the trigger design alone, a lower bound; a sensitivity
adds a per-event vulnerability error with the spread of the GDV calibration residuals.

Part 2, cat bond. A per-occurrence layer on the national calibrated loss, attaching at the 1-in-50 and exhausting at
the 1-in-200 occurrence loss of the central model: probability of attachment, probability of exhaustion and expected
loss, and how much these move across the bootstrap of winters and the calibration factor.

    python -m windrisk.parametric
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import genpareto, spearmanr
from scipy.stats import t as student_t

from . import calibration as cal
from .risk import simulate_seasons
from .run import ROOT, THRESHOLD_Q

REGION = ("Hamburg", "Bremen", "Niedersachsen", "Schleswig-Holstein")
SHARE = 0.10
HAMBURG, CITIES = 1975, (1975, 691, 2014)        # Hamburg-Fuhlsbuettel, Bremen, Hannover
ATTACH_RP, EXHAUST_RP = 5, 25                     # winters between events above attachment and exhaustion
STEPS = (5, 10, 25)                               # step cover on the Hamburg gust: one third at each level
N_SIM = 100_000
SEED = 31
BOND_ATTACH, BOND_EXHAUST = 50, 200
TOP = 25                                         # events used to fit each trigger's loss curve


def station_terms(gust: pd.DataFrame, v98: pd.Series) -> pd.DataFrame:
    """Events x stations: max(0, gust / v98 - 1)^3, missing gusts as zero (a missing reading pays nothing)."""
    return (gust.div(v98, axis=1) - 1).clip(lower=0).pow(3).fillna(0.0)


def layer(loss: np.ndarray, attach: float, exhaust: float) -> np.ndarray:
    return np.clip(loss - attach, 0.0, exhaust - attach)


def level_for_rate(values: np.ndarray, winters: int, period: float) -> float:
    """Event value exceeded on average once in `period` winters (empirical, linear interpolation)."""
    k = winters / period
    s = np.sort(values)[::-1]
    lo = int(np.floor(k))
    if lo < 1:
        return float(s[0])
    frac = k - lo
    return float(s[lo - 1] + frac * (s[lo] - s[lo - 1]))


def power_fit(train_x: np.ndarray, train_y: np.ndarray, top: int = TOP) -> tuple[float, float]:
    """Quantile matching with a power curve: log y = a + b log x fitted to the sorted `top` largest values of each.

    Matching quantiles, not regressing y on x, keeps the payout frequency of each layer equal to that of the loss;
    a regression would shrink large events towards the mean and the cover would rarely reach its attachment.
    """
    x = np.sort(train_x[train_x > 0])[::-1][:top]
    y = np.sort(train_y[train_y > 0])[::-1][:top]
    n = min(len(x), len(y))
    b, a = np.polyfit(np.log(x[:n]), np.log(y[:n]), 1)
    return float(a), float(b)


def cross_fit(trigger: np.ndarray, loss: np.ndarray, season: np.ndarray, top: int = TOP) -> np.ndarray:
    """Out-of-sample modelled loss from the trigger: odd winters fitted on even ones and the reverse."""
    out = np.zeros_like(loss)
    odd = season % 2 == 1
    for test in (odd, ~odd):
        a, b = power_fit(trigger[~test], loss[~test], top)
        x = trigger[test]
        out[test] = np.where(x > 0, np.exp(a) * np.power(np.where(x > 0, x, 1.0), b), 0.0)
    return out


def step_payout(gust: np.ndarray, levels: np.ndarray, limit: float) -> np.ndarray:
    """One third of the limit for each gust level reached; a missing gust pays nothing."""
    g = np.nan_to_num(gust, nan=-np.inf)
    return limit * (g[:, None] >= levels[None, :]).sum(axis=1) / len(levels)


def annual_hedge(loss: np.ndarray, payout: np.ndarray, rate: float, n_sim: int = N_SIM, seed: int = SEED) -> dict:
    """Winters built from Poisson counts of resampled historical events; retained = loss - payout + mean payout."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate, n_sim)
    pick = rng.integers(0, len(loss), n.sum())
    w = np.repeat(np.arange(n_sim), n)
    gross, paid = np.zeros(n_sim), np.zeros(n_sim)
    np.add.at(gross, w, loss[pick])
    np.add.at(paid, w, payout[pick])
    net = gross - paid + paid.mean()
    v = np.quantile(gross, 0.99)
    vn = np.quantile(net, 0.99)
    return {"var_reduction": float(1 - net.var() / gross.var()),
            "var99_gross": float(v), "var99_net": float(vn),
            "tvar99_gross": float(gross[gross >= v].mean()), "tvar99_net": float(net[net >= vn].mean())}


def basis_metrics(name: str, loss: np.ndarray, indemnity: np.ndarray, payout: np.ndarray, raw: np.ndarray,
                  rate: float) -> dict:
    hit = indemnity > 0
    big = loss >= np.quantile(loss, 0.9)
    return {"trigger": name,
            "spearman_top10pct_events": float(spearmanr(raw[big], loss[big])[0]),
            "events_in_layer": int(hit.sum()),
            "missed_events": int((hit & (payout == 0)).sum()),
            "false_payouts": int((~hit & (payout > 0)).sum()),
            "shortfall_share": float(np.clip(indemnity - payout, 0, None).sum() / indemnity.sum()),
            "overpayment_share": float(np.clip(payout - indemnity, 0, None).sum() / indemnity.sum()),
            **annual_hedge(loss, payout, rate)}


def simulate_layer(event_losses: np.ndarray, seasons: int, u: float, attach: float, exhaust: float,
                   n_sim: int = 20_000, seed: int = SEED, log_noise: float = 0.0) -> dict:
    """Annual principal loss of a per-occurrence layer (aggregate capped at the limit), GPD tail as in risk.py."""
    rng = np.random.default_rng(seed)
    above, below = event_losses[event_losses > u], event_losses[event_losses <= u]
    xi, _, sigma = genpareto.fit(above - u, floc=0)
    n_a = rng.poisson(len(above) / seasons, n_sim)
    n_b = rng.poisson(len(below) / seasons, n_sim)
    ev = np.concatenate([u + genpareto.rvs(xi, scale=sigma, size=n_a.sum(), random_state=rng),
                         rng.choice(below, n_b.sum())])
    who = np.concatenate([np.repeat(np.arange(n_sim), n_a), np.repeat(np.arange(n_sim), n_b)])
    lim = exhaust - attach
    idx_loss = layer(ev, attach, exhaust)
    actual = ev * np.exp(log_noise * rng.standard_normal(len(ev))) if log_noise else ev
    ind_loss = layer(actual, attach, exhaust)
    a, b = np.zeros(n_sim), np.zeros(n_sim)
    np.add.at(a, who, idx_loss)
    np.add.at(b, who, ind_loss)
    a, b = np.minimum(a, lim) / lim, np.minimum(b, lim) / lim
    return {"p_attach": float((a > 0).mean()), "p_exhaust": float((a >= 1 - 1e-12).mean()),
            "expected_loss": float(a.mean()), "mismatch_given_attach": float(((a > 0) != (b > 0)).sum() / max((a > 0).sum(), 1)),
            "xi": float(xi)}


def main() -> dict:
    res = ROOT / "results"
    out_dir = res / "parametric"
    out_dir.mkdir(exist_ok=True)
    st = pd.read_csv(res / "stations.csv").set_index("station_id")
    events = pd.read_csv(res / "events.csv", parse_dates=["start", "end"])
    gust = pd.read_csv(res / "climada" / "event_max_gust.csv", index_col=0)
    gust.columns = gust.columns.astype(int)
    fp = pd.read_csv(res / "climada" / "events.csv", parse_dates=["start"])
    if not (fp["start"].values == events["start"].values).all():
        raise ValueError("footprints and event set are not in the same order")
    st = st.loc[gust.columns]
    terms = station_terms(gust, st["v98"])
    season = events["season"].to_numpy()
    winters = int(events["season"].nunique())
    rate = len(events) / winters

    m = cal.match_storms(events)
    scale, se_log, dof = cal.fit_scale(m["gdv_mio"].to_numpy(), m["loss_index"].to_numpy())
    resid_sd = se_log * np.sqrt(dof + 1)

    reg = st["state"].isin(REGION).to_numpy()
    loss = SHARE * scale * (terms.loc[:, reg] * st.loc[reg, "weight"]).sum(axis=1).to_numpy()   # million EUR
    attach = level_for_rate(loss, winters, ATTACH_RP)
    exhaust = level_for_rate(loss, winters, EXHAUST_RP)
    limit = exhaust - attach
    indemnity = layer(loss, attach, exhaust)

    triggers = {
        "industry loss (national index)": events["loss_index"].to_numpy(),
        "one station (Hamburg)": terms[HAMBURG].to_numpy(),
        "three cities (Hamburg, Bremen, Hannover)": terms[list(CITIES)].mean(axis=1).to_numpy(),
        "all 35 regional stations, equal weights": terms.loc[:, reg].mean(axis=1).to_numpy(),
    }
    raw = {"indemnity": loss, **triggers}
    rows = [basis_metrics("indemnity", loss, indemnity, indemnity, loss, rate)]
    mapped = {}
    for name, x in triggers.items():
        mapped[name] = layer(cross_fit(x, loss, season), attach, exhaust)
        rows.append(basis_metrics(name, loss, indemnity, mapped[name], x, rate))
    hh = gust[HAMBURG].to_numpy()
    levels = np.array([level_for_rate(np.nan_to_num(hh, nan=0.0), winters, p) for p in STEPS])
    mapped["step cover on the Hamburg gust"] = step_payout(hh, levels, limit)
    raw["step cover on the Hamburg gust"] = np.nan_to_num(hh, nan=0.0)
    rows.append(basis_metrics("step cover on the Hamburg gust", loss, indemnity, mapped["step cover on the Hamburg gust"],
                              raw["step cover on the Hamburg gust"], rate))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "basis_risk.csv", index=False, float_format="%.4f")
    sens = [{"top": k, "trigger": name, **{c: basis_metrics(name, loss, indemnity,
                                                         layer(cross_fit(x, loss, season, k), attach, exhaust), x, rate)[c]
                                          for c in ("missed_events", "shortfall_share", "overpayment_share", "var_reduction")}}
            for k in (15, 25, 40, 100) for name, x in triggers.items()]
    pd.DataFrame(sens).to_csv(out_dir / "fit_points_sensitivity.csv", index=False, float_format="%.4f")

    # vulnerability error: the actual loss differs from the proxy by a lognormal factor per event
    rng = np.random.default_rng(SEED)
    actual = loss * np.exp(resid_sd * rng.standard_normal(len(loss)))
    ind_actual = layer(actual, attach, exhaust)
    noisy = [basis_metrics(n, actual, ind_actual, p, raw[n], rate)
             for n, p in [("indemnity", ind_actual)] + list(mapped.items())]
    pd.DataFrame(noisy).to_csv(out_dir / "basis_risk_with_vulnerability_error.csv", index=False, float_format="%.4f")

    top = events.assign(regional_loss_mio=loss, indemnity_mio=indemnity,
                        **{f"payout_{k.split(' (')[0].replace(' ', '_')}": v for k, v in mapped.items()})
    top.nlargest(15, "regional_loss_mio").drop(columns=["end", "peak_day", "days"]).to_csv(
        out_dir / "largest_regional_events.csv", index=False, float_format="%.2f")

    # Part 2: cat bond layer on the national calibrated loss
    nat = events["loss_index"].to_numpy() * scale / 1000          # bn EUR
    u = float(np.quantile(nat, THRESHOLD_Q))
    central = simulate_seasons(nat, winters, u, seed=SEED)["max_event"]
    b_att, b_exh = float(central.quantile(1 - 1 / BOND_ATTACH)), float(central.quantile(1 - 1 / BOND_EXHAUST))
    base = simulate_layer(nat, winters, u, b_att, b_exh, n_sim=N_SIM)
    noisy_bond = simulate_layer(nat, winters, u, b_att, b_exh, n_sim=N_SIM, log_noise=resid_sd)
    boot_rng = np.random.default_rng(SEED + 1)
    seasons = sorted(events["season"].unique())
    by_season = {s: nat[season == s] for s in seasons}
    boot = []
    for b in range(200):
        pick = boot_rng.choice(seasons, len(seasons), replace=True)
        x = np.concatenate([by_season[s] for s in pick])
        if (x > u).sum() < 10:
            continue
        f = np.exp(se_log * student_t.rvs(dof, random_state=boot_rng))
        r = simulate_layer(x * f, winters, u * f, b_att, b_exh, seed=SEED + 2 + b)
        boot.append(r)
    boot = pd.DataFrame(boot)
    boot.to_csv(out_dir / "cat_bond_bootstrap.csv", index=False, float_format="%.5f")

    summary = {
        "region": list(REGION), "share": SHARE, "eur_mio_per_index_unit": scale, "vulnerability_log_sd": resid_sd,
        "layer_mio_eur": {"attachment": attach, "exhaustion": exhaust, "limit": limit,
                          "events_in_layer": int((indemnity > 0).sum())},
        "step_gust_levels_ms": levels.tolist(),
        "missing_hamburg_gust_events": int(np.isnan(hh).sum()),
        "cat_bond": {"attachment_bn_eur": b_att, "exhaustion_bn_eur": b_exh, "central": base,
                     "with_vulnerability_error": noisy_bond,
                     "bootstrap_with_calibration": {k: {"p05": float(boot[k].quantile(0.05)), "median": float(boot[k].median()),
                                                        "p95": float(boot[k].quantile(0.95))}
                                                    for k in ("p_attach", "p_exhaust", "expected_loss")}},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.round(3).to_string())
    print(pd.DataFrame(noisy).round(3).to_string())
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
