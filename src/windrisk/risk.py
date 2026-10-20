"""Extreme value statistics of storm losses and insurance metrics.

Frequency-severity model: events above a threshold u arrive as a Poisson process (rate per
season), their excess over u follows a generalised Pareto distribution (GPD, maximum likelihood);
events below u are resampled from the observed ones with their own Poisson rate. Simulated
seasons give the annual aggregate loss (AEP) and the annual maximum event (OEP) distributions,
average annual loss (AAL), 1-in-200 values (VaR 99.5%, the Solvency II calibration level) and
tail value at risk. A GEV fit to seasonal maxima is a cross-check of the occurrence curve, and a
bootstrap over seasons measures parameter uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto


def fit_gpd(excess: np.ndarray) -> tuple[float, float]:
    """MLE shape xi and scale sigma of a GPD for positive excesses (location fixed at 0)."""
    xi, _, sigma = genpareto.fit(excess, floc=0)
    return float(xi), float(sigma)


def gpd_return_level(u: float, rate: float, xi: float, sigma: float, period: float) -> float:
    """Event loss exceeded on average once in `period` seasons (rate = events above u per season)."""
    m = rate * period
    if m <= 1:
        return u
    if abs(xi) < 1e-9:
        return u + sigma * np.log(m)
    return u + sigma / xi * (m ** xi - 1)


def fit_gev_maxima(maxima: np.ndarray) -> tuple[float, float, float]:
    """MLE GEV for seasonal maxima; returns (c, loc, scale) in SciPy's convention (c = -xi)."""
    c, loc, scale = genextreme.fit(maxima)
    return float(c), float(loc), float(scale)


def simulate_seasons(event_losses: np.ndarray, seasons: int, u: float, n_sim: int = 100_000,
                     xi: float | None = None, sigma: float | None = None, seed: int = 0) -> pd.DataFrame:
    """Simulated seasons: columns aggregate (sum of event losses) and max_event."""
    rng = np.random.default_rng(seed)
    above = event_losses[event_losses > u]
    below = event_losses[event_losses <= u]
    if xi is None:
        xi, sigma = fit_gpd(above - u)
    n_above = rng.poisson(len(above) / seasons, n_sim)
    n_below = rng.poisson(len(below) / seasons, n_sim)
    agg = np.zeros(n_sim)
    mx = np.zeros(n_sim)
    sev = u + genpareto.rvs(xi, scale=sigma, size=n_above.sum(), random_state=rng)
    small = rng.choice(below, size=n_below.sum(), replace=True) if len(below) else np.zeros(0)
    ia = np.repeat(np.arange(n_sim), n_above)
    ib = np.repeat(np.arange(n_sim), n_below)
    np.add.at(agg, ia, sev)
    np.add.at(agg, ib, small)
    np.maximum.at(mx, ia, sev)
    np.maximum.at(mx, ib, small)
    return pd.DataFrame({"aggregate": agg, "max_event": mx})


def ep_table(sim: pd.DataFrame, periods=(2, 5, 10, 25, 50, 100, 200, 250)) -> pd.DataFrame:
    """Return-period table of the aggregate (AEP) and occurrence (OEP) losses."""
    rows = []
    for t in periods:
        q = 1 - 1 / t
        rows.append({"return_period": t, "aep": float(sim["aggregate"].quantile(q)),
                     "oep": float(sim["max_event"].quantile(q))})
    return pd.DataFrame(rows)


def tvar(x: pd.Series, level: float = 0.995) -> float:
    """Mean loss in the worst (1 - level) share of seasons."""
    v = x.quantile(level)
    return float(x[x >= v].mean())


def bootstrap_1in200(events: pd.DataFrame, seasons: list[int], u: float, n_boot: int = 200,
                     n_sim: int = 20_000, seed: int = 1) -> pd.DataFrame:
    """Resample whole seasons (with their events), refit and resimulate: spread of the 1-in-200 losses and AAL."""
    rng = np.random.default_rng(seed)
    by_season = {s: events.loc[events["season"] == s, "loss_index"].to_numpy() for s in seasons}
    rows = []
    for b in range(n_boot):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        losses = np.concatenate([by_season[s] for s in pick])
        if (losses > u).sum() < 10:
            continue
        sim = simulate_seasons(losses, len(seasons), u, n_sim=n_sim, seed=seed + b)
        xi, _ = fit_gpd(losses[losses > u] - u)
        rows.append({"xi": xi, "aal": sim["aggregate"].mean(),
                     "aep_200": sim["aggregate"].quantile(0.995), "oep_200": sim["max_event"].quantile(0.995)})
    return pd.DataFrame(rows)
