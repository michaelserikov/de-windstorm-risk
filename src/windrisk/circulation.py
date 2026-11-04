"""Compound wind and rain extremes conditioned on the North Atlantic circulation (v0.3).

Winter days are grouped by the NOAA daily NAO index and by weather regimes from ERA5 mean sea
level pressure. For each group: conditional chi, storm rates and trends. An exploratory first
step for one country: associations, not causes.

    python -m windrisk.circulation
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.stats import kendalltau, theilslopes

from . import hazard as hz
from .run import END, ROOT, START, load

NAO_URL = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii"
Q = 0.98
N_BOOT = 1000
N_REGIMES = 4
N_PCS = 10
SEED = 7
MIN_COVERAGE = 0.9


def load_nao(cache: Path) -> pd.Series:
    path = cache / "nao_daily_noaa.txt"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(NAO_URL, timeout=120).read())
    # Missing values are written as -99.000 and run into the day column, so parse by pattern
    t = path.read_text().splitlines()
    t = pd.Series(t).str.extract(r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s*(-?\d+\.\d+)").dropna().astype(float)
    t.columns = ["year", "month", "day", "nao"]
    t.index = pd.to_datetime(t[["year", "month", "day"]].astype(int))
    return t["nao"].where(t["nao"] > -90).dropna()


def winter_of(index: pd.DatetimeIndex) -> np.ndarray:
    """Winter labelled by the year of its January, autumn months go to the following winter."""
    return index.year + (index.month >= 7).astype(int)


def exceedances(gust: pd.DataFrame, rain: pd.DataFrame, q: float = Q) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Station-wise exceedances of each station's own quantile, computed over all winter days."""
    days = gust.index.intersection(rain.index)
    g, r = gust.loc[days], rain.loc[days]
    return g.gt(g.quantile(q)).where(g.notna()), r.gt(r.quantile(q)).where(r.notna())


def conditional_chi(ge: pd.DataFrame, re_: pd.DataFrame, mask: np.ndarray) -> dict:
    """Pooled P(rain extreme | gust extreme) within a group of days, and its ratio to independence."""
    g = ge.to_numpy(dtype=float)[mask]
    r = re_.to_numpy(dtype=float)[mask]
    valid = ~np.isnan(g) & ~np.isnan(r)
    gust_n = np.sum((g == 1) & valid)
    joint = np.sum((g == 1) & (r == 1) & valid)
    rain_rate = np.sum((r == 1) & valid) / max(valid.sum(), 1)
    chi = joint / gust_n if gust_n else np.nan
    return {"chi": chi, "gust_rate": gust_n / max(valid.sum(), 1), "rain_rate": rain_rate,
            "joint_rate": joint / max(valid.sum(), 1),
            "ratio_to_independence": chi / rain_rate if rain_rate else np.nan}


def by_group(ge: pd.DataFrame, re_: pd.DataFrame, labels: pd.Series, seed: int = SEED) -> pd.DataFrame:
    """Conditional chi per group, with a 90% interval from resampling whole winters."""
    winters = winter_of(ge.index)
    unique = np.unique(winters)
    positions = {w: np.where(winters == w)[0] for w in unique}
    rng = np.random.default_rng(seed)
    values = labels.reindex(ge.index).to_numpy()
    groups = sorted(pd.Series(values).dropna().unique())
    rows = []
    boot = {g: [] for g in groups}
    for _ in range(N_BOOT):
        idx = np.concatenate([positions[w] for w in rng.choice(unique, len(unique))])
        gb, rb = ge.iloc[idx], re_.iloc[idx]
        vb = values[idx]
        for g in groups:
            boot[g].append(conditional_chi(gb, rb, vb == g)["ratio_to_independence"])
    for g in groups:
        mask = values == g
        row = {"group": g, "days": int(mask.sum()), "share_of_days": float(mask.mean())}
        row.update(conditional_chi(ge, re_, mask))
        row["ratio_p05"], row["ratio_p95"] = np.nanquantile(boot[g], [0.05, 0.95])
        rows.append(row)
    return pd.DataFrame(rows)


def storm_rates(events: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Storm peak days per group against the group's share of days."""
    peak = labels.reindex(events["peak_day"]).to_numpy()
    e = events.assign(group=peak).dropna(subset=["group"])
    share = labels.value_counts(normalize=True)
    out = e.groupby("group").agg(events=("loss_index", "size"), mean_loss=("loss_index", "mean"),
                                 total_loss=("loss_index", "sum"))
    out["share_of_days"] = share
    out["share_of_events"] = out["events"] / out["events"].sum()
    out["share_of_loss"] = out["total_loss"] / out["total_loss"].sum()
    out["event_rate_ratio"] = out["share_of_events"] / out["share_of_days"]
    return out.reset_index()


def trend(series: pd.Series) -> dict:
    """Mann-Kendall test through Kendall's tau, with the Theil-Sen slope per decade."""
    s = series.dropna()
    tau, p = kendalltau(s.index, s.to_numpy())
    slope, _, low, high = theilslopes(s.to_numpy(), s.index)
    return {"tau": float(tau), "p": float(p), "sen_per_decade": float(10 * slope),
            "sen_low": float(10 * low), "sen_high": float(10 * high), "n": int(len(s))}


def load_mslp(folder: Path, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Daily mean pressure on the requested winter days, in hPa."""
    fields, stamps, lat = [], [], None
    origin = pd.Timestamp("1900-01-01")
    for path in sorted(folder.glob("mslp_*.npz")):
        d = np.load(path)
        lat = d["lat"]
        time = origin + pd.to_timedelta(d["time"], unit="h")
        frame = pd.Series(np.arange(len(time)), index=time.floor("D"))
        keep = frame.index.intersection(days).unique()
        for day in keep:
            fields.append(d["msl"][frame.loc[[day]].to_numpy()].mean(axis=0) / 100.0)
            stamps.append(day)
    return np.stack(fields), lat, pd.DatetimeIndex(stamps)


def regimes(field: np.ndarray, lat: np.ndarray, days: pd.DatetimeIndex, seed: int = SEED):
    """Weather regimes: anomalies from a smoothed calendar climatology, EOFs, then k-means."""
    doy = days.dayofyear.to_numpy()
    anomalies = field.copy()
    for d in np.unique(doy):
        near = np.abs(((doy - d + 182) % 365) - 182) <= 15
        anomalies[doy == d] -= field[near].mean(axis=0)
    weight = np.sqrt(np.cos(np.deg2rad(lat)))[None, :, None]
    x = (anomalies * weight).reshape(len(days), -1)
    u, s, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    pcs = u[:, :N_PCS] * s[:N_PCS]
    explained = s**2 / np.sum(s**2)
    _, labels = kmeans2(pcs / pcs.std(axis=0)[0], N_REGIMES, seed=seed, minit="++", iter=100)
    return labels, u[:, 0] * s[0], explained[:N_PCS], anomalies


def main() -> dict:
    cache = ROOT / "data" / "raw"
    res, out = ROOT / "results", ROOT / "results" / "circulation"
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
    ge, re_ = exceedances(gust[both], rain[both])

    nao = load_nao(cache).reindex(ge.index)
    terciles = pd.qcut(nao, 3, labels=["NAO low", "NAO middle", "NAO high"]).astype(object)
    nao_table = by_group(ge, re_, terciles)
    nao_table.to_csv(out / "chi_by_nao.csv", index=False, float_format="%.5f")
    nao_storms = storm_rates(events, terciles)
    nao_storms.to_csv(out / "storms_by_nao.csv", index=False, float_format="%.5f")

    field, lat, days = load_mslp(cache / "era5_mslp_cloud", ge.index)
    labels, pc1, explained, anomalies = regimes(field, lat, days)
    regime = pd.Series([f"Regime {i + 1}" for i in labels], index=days)
    regime_table = by_group(ge.loc[days], re_.loc[days], regime)
    regime_table.to_csv(out / "chi_by_regime.csv", index=False, float_format="%.5f")
    regime_storms = storm_rates(events, regime)
    regime_storms.to_csv(out / "storms_by_regime.csv", index=False, float_format="%.5f")
    corr_pc1 = float(np.corrcoef(pc1, nao.reindex(days).to_numpy())[0, 1]) if nao.reindex(days).notna().all() else float(
        pd.Series(pc1, index=days).corr(nao.reindex(days)))
    regime_nao = nao.reindex(days).groupby(regime).mean().to_dict()

    seam = {str(y): float(field[days.year == y].std()) for y in (2021, 2022, 2023, 2024)}
    winters = winter_of(ge.index)
    joint_by_winter = ((ge == 1) & (re_ == 1)).sum(axis=1).groupby(winters).sum()
    seasons = events.groupby("season")["loss_index"].sum()
    nao_winter = nao.groupby(winters).mean()
    summary = {
        "stations": len(both), "winter_days": int(len(ge)), "pc1_nao_correlation": corr_pc1,
        "explained_variance_first_pcs": np.round(explained, 3).tolist(),
        "regime_mean_nao": {k: round(float(v), 3) for k, v in regime_nao.items()},
        "pressure_std_by_year_seam_check": seam,
        "trends": {"joint_exceedances_per_winter": trend(joint_by_winter),
                   "seasonal_loss_index": trend(seasons), "winter_mean_nao": trend(nao_winter)},
        "corr_winter_nao_joint": float(kendalltau(nao_winter.reindex(joint_by_winter.index), joint_by_winter)[0]),
        "corr_winter_nao_joint_p": float(kendalltau(nao_winter.reindex(joint_by_winter.index), joint_by_winter)[1]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for t in (nao_table, nao_storms, regime_table, regime_storms):
        print(t.round(4).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
