"""Offline tests for the compound wind-rain module: copula densities against their CDFs, fits on
simulated data, joint exceedance and the station and event measures."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from windrisk import compound as cp


@pytest.mark.parametrize("theta", [1.2, 1.5, 3.0])
@pytest.mark.parametrize("u,v", [(0.3, 0.6), (0.9, 0.8), (0.05, 0.5)])
def test_gumbel_density_is_mixed_derivative_of_cdf(theta, u, v):
    h = 1e-4
    c = lambda a, b: cp.gumbel_cdf(a, b, theta)  # noqa: E731
    numeric = (c(u + h, v + h) - c(u + h, v - h) - c(u - h, v + h) + c(u - h, v - h)) / (4 * h * h)
    assert np.exp(cp.gumbel_logpdf(np.array([u]), np.array([v]), theta))[0] == pytest.approx(numeric, rel=1e-4)


def _gumbel_sample(theta, n, seed):
    """Marshall-Olkin sampling with a positive stable frailty (Chambers-Mallows-Stuck)."""
    rng = np.random.default_rng(seed)
    a = 1 / theta
    w, e = rng.uniform(0, np.pi, n), rng.exponential(1, n)
    s = (np.sin(a * w) / np.sin(w) ** (1 / a)) * (np.sin((1 - a) * w) / e) ** ((1 - a) / a)
    x = rng.exponential(1, (n, 2))
    return np.exp(-(x / s[:, None]) ** a).T


def test_gumbel_fit_recovers_theta():
    u, v = _gumbel_sample(1.8, 4000, 3)
    theta, _ = cp.fit_gumbel(cp.pseudo_obs(u), cp.pseudo_obs(v))
    assert theta == pytest.approx(1.8, abs=0.1)


def test_gaussian_fit_and_loglik_match_scipy():
    rng = np.random.default_rng(4)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 3000)
    u, v = norm.cdf(z[:, 0]), norm.cdf(z[:, 1])
    rho, ll = cp.fit_gaussian(u, v)
    assert rho == pytest.approx(0.6, abs=0.05)
    ref = (multivariate_normal([0, 0], [[1, rho], [rho, 1]]).logpdf(z) - norm.logpdf(z).sum(axis=1)).sum()
    assert ll == pytest.approx(ref, rel=1e-6)


def test_joint_exceedance_under_independence():
    p = 0.9
    assert cp.joint_exceedance(p, p * p) == pytest.approx(0.01)
    assert cp.joint_exceedance(p, cp.gumbel_cdf(p, p, 1.0)) == pytest.approx(0.01)  # theta = 1 is independence


def test_station_chi_near_tail_probability_for_independent_series():
    rng = np.random.default_rng(5)
    g = pd.DataFrame(rng.gumbel(20, 5, (40_000, 2)))
    r = pd.DataFrame(rng.exponential(3, (40_000, 2)))
    chi = cp.station_chi(g, r, 0.98)
    assert chi["chi"].to_numpy() == pytest.approx([0.02, 0.02], abs=0.006)
    assert chi["ratio_to_independence"].to_numpy() == pytest.approx([1, 1], abs=0.35)


def test_event_rain_uses_event_maximum_and_weights():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rain = pd.DataFrame({1: [0, 10, 20, 0.0], 2: [5, 5, np.nan, 5.0]}, index=dates)
    ev = pd.DataFrame({"start": [dates[1]], "end": [dates[2]]})
    out = cp.event_rain(ev, rain, pd.Series({1: 10.0, 2: 10.0}), pd.Series({1: 0.75, 2: 0.25}))
    assert out.iloc[0] == pytest.approx(0.75 * 2.0 + 0.25 * 0.5)
