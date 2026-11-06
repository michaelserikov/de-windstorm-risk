"""Offline tests of the parametric cover and cat bond layer (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from windrisk import parametric as pm


def test_layer_clips_at_attachment_and_exhaustion():
    assert pm.layer(np.array([5.0, 15.0, 40.0]), 10.0, 30.0).tolist() == [0.0, 5.0, 20.0]


def test_level_for_rate_counts_events_per_winter():
    values = np.arange(1.0, 101.0)            # 100 events in 10 winters
    assert pm.level_for_rate(values, 10, 5) == pytest.approx(99.0)     # two events above in 10 winters
    assert pm.level_for_rate(values, 10, 4) == pytest.approx(98.5)     # 2.5 events: interpolated


def test_power_fit_recovers_an_exact_power_law():
    x = np.linspace(0.01, 1.0, 200)
    a, b = pm.power_fit(x, 3.0 * x ** 2, top=50)
    assert np.exp(a) == pytest.approx(3.0)
    assert b == pytest.approx(2.0)


def test_power_fit_matches_quantiles_not_pairs():
    x = np.linspace(0.1, 1.0, 100)
    shuffled = np.random.default_rng(0).permutation(5.0 * x)
    a, b = pm.power_fit(x, shuffled, top=30)
    assert np.exp(a) == pytest.approx(5.0) and b == pytest.approx(1.0)


def test_cross_fit_predicts_each_half_from_the_other():
    season = np.repeat(np.arange(10), 20)
    x = np.tile(np.linspace(0.05, 1.0, 20), 10)
    loss = 2.0 * x
    loss[season % 2 == 0] *= 10                # even winters ten times larger
    pred = pm.cross_fit(x, loss, season, top=20)
    odd = season % 2 == 1
    assert pred[odd] == pytest.approx(20.0 * x[odd])
    assert pred[~odd] == pytest.approx(2.0 * x[~odd])


def test_step_payout_and_missing_gust():
    pay = pm.step_payout(np.array([20.0, 31.0, 40.0, np.nan]), np.array([30.0, 35.0, 38.0]), 9.0)
    assert pay.tolist() == [0.0, 3.0, 9.0, 0.0]


def test_station_terms_cubic_and_missing_zero():
    g = pd.DataFrame({1: [30.0, np.nan], 2: [10.0, 40.0]})
    t = pm.station_terms(g, pd.Series({1: 20.0, 2: 20.0}))
    assert t[1].tolist() == pytest.approx([0.125, 0.0])
    assert t[2].tolist() == pytest.approx([0.0, 1.0])


def test_cat_bond_layer_probabilities_are_ordered():
    rng = np.random.default_rng(1)
    losses = rng.pareto(2.0, 2000) + 0.01
    u = float(np.quantile(losses, 0.9))
    r = pm.simulate_layer(losses, 100, u, attach=3.0, exhaust=6.0, n_sim=20_000)
    assert 0 < r["p_exhaust"] <= r["expected_loss"] <= r["p_attach"] < 1
    assert r["mismatch_given_attach"] == 0.0
    far = pm.simulate_layer(losses, 100, u, attach=1e9, exhaust=2e9, n_sim=5_000)
    assert far["p_attach"] == 0.0 and far["expected_loss"] == 0.0
