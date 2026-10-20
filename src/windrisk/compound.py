"""Compound wind and rainfall extremes in winter storms.

Two levels of evidence:
- Stations: on winter days, how often does the daily maximum gust exceed its local 98th
  percentile together with daily precipitation above its local 98th percentile, compared with
  independence? The conditional exceedance chi(q) = P(rain > r_q | gust > v_q) is the empirical
  tail-dependence coefficient at level q (2% under independence for q = 0.98).
- Storm events: for each event of the wind event set, a rain measure (population-weighted mean of
  each station's highest daily precipitation over the event days relative to its 98th percentile)
  next to the wind loss index. Their dependence is modelled with copulas fitted to rank-based
  pseudo-observations (independence, Gaussian, Gumbel; the Gumbel copula has upper-tail
  dependence lambda_U = 2 - 2^(1/theta)), compared by AIC. The quantity of interest is how much
  dependence shortens the joint return period of an event that is extreme in both wind and rain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm, rankdata


# --- station level -------------------------------------------------------------------------------

def station_chi(gust: pd.DataFrame, rain: pd.DataFrame, q: float = 0.98) -> pd.DataFrame:
    """Per station: joint exceedance frequency, its ratio to independence and chi(q)."""
    both_ok = gust.notna() & rain.notna()
    g = gust.where(both_ok)
    r = rain.where(both_ok)
    vq, rq = g.quantile(q), r.quantile(q)
    hi_g, hi_r = g.gt(vq), r.gt(rq)
    n = both_ok.sum()
    joint = (hi_g & hi_r).sum() / n
    indep = (hi_g.sum() / n) * (hi_r.sum() / n)
    return pd.DataFrame({"joint_freq": joint, "ratio_to_independence": joint / indep,
                         "chi": (hi_g & hi_r).sum() / hi_g.sum(), "days": n})


# --- event level ---------------------------------------------------------------------------------

def event_rain(events: pd.DataFrame, rain: pd.DataFrame, r98: pd.Series, weights: pd.Series) -> pd.Series:
    """Population-weighted mean of (event maximum daily rain / local 98th percentile) per event."""
    w = weights.reindex(rain.columns)
    out = []
    for _, e in events.iterrows():
        fp = rain.loc[e["start"]:e["end"]].max()
        ok = fp.notna()
        out.append(float((fp[ok] / r98[ok] * w[ok]).sum() / w[ok].sum()))
    return pd.Series(out, index=events.index, name="rain_ratio")


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / (len(x) + 1)


def gumbel_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Log density of the Gumbel copula (theta >= 1)."""
    x, y = -np.log(u), -np.log(v)
    s = x ** theta + y ** theta
    a = s ** (1 / theta)
    return (-a - np.log(u) - np.log(v) + (theta - 1) * (np.log(x) + np.log(y))
            + (1 / theta - 2) * np.log(s) + np.log(a + theta - 1))


def gumbel_cdf(u: float, v: float, theta: float) -> float:
    return float(np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)))


def fit_gumbel(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """MLE theta and maximised log-likelihood."""
    res = minimize_scalar(lambda t: -gumbel_logpdf(u, v, t).sum(), bounds=(1.0001, 20), method="bounded")
    return float(res.x), float(-res.fun)


def fit_gaussian(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Correlation of normal scores and the Gaussian-copula log-likelihood."""
    z1, z2 = norm.ppf(u), norm.ppf(v)
    rho = float(np.corrcoef(z1, z2)[0, 1])
    ll = (-0.5 * np.log(1 - rho ** 2) - (rho ** 2 * (z1 ** 2 + z2 ** 2) - 2 * rho * z1 * z2) / (2 * (1 - rho ** 2))).sum()
    return rho, float(ll)


def joint_exceedance(p: float, cdf_pp: float) -> float:
    """P(U > p, V > p) from the copula value C(p, p)."""
    return 1 - 2 * p + cdf_pp


def joint_return_periods(rate: float, periods, theta: float, rho: float, u: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """For each T: return period (winters) of an event exceeding the T-winter level in both variables.

    Marginal level: exceeded by one event in T winters on average, so per-event probability
    q = 1 / (rate * T). Empirical values count observed events above both empirical levels.
    """
    rows = []
    for t in periods:
        q = 1 / (rate * t)
        p = 1 - q
        indep = q * q
        gum = joint_exceedance(p, gumbel_cdf(p, p, theta))
        gau = joint_exceedance(p, float(multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([norm.ppf(p)] * 2)))
        emp = float(((u > p) & (v > p)).mean())
        rows.append({"marginal_return_period": t, "per_event_marginal_prob": q,
                     "joint_rp_independence": 1 / (rate * indep), "joint_rp_gaussian": 1 / (rate * gau),
                     "joint_rp_gumbel": 1 / (rate * gum),
                     "joint_rp_empirical": (1 / (rate * emp)) if emp > 0 else np.nan,
                     "events_both_above_observed": int(((u > p) & (v > p)).sum())})
    return pd.DataFrame(rows)


def tail_diagnostic(u: np.ndarray, v: np.ndarray, theta: float, rho: float,
                    probs=(0.5, 0.25, 0.1, 0.05, 0.02)) -> pd.DataFrame:
    """Observed vs model counts of events above the per-event quantile 1 - q in both variables."""
    n = len(u)
    rows = []
    for q in probs:
        p = 1 - q
        gau = joint_exceedance(p, float(multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([norm.ppf(p)] * 2)))
        rows.append({"per_event_prob_each": q, "observed": int(((u > p) & (v > p)).sum()),
                     "independence": n * q * q, "gaussian": n * gau,
                     "gumbel": n * joint_exceedance(p, gumbel_cdf(p, p, theta))})
    return pd.DataFrame(rows)
