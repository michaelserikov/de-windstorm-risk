"""Offline tests for the circulation module: NAO parsing, winter labels, exceedances, conditional
chi, storm rates, trends and regimes, all on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from windrisk import circulation as cc


def test_load_nao_parses_values_and_drops_missing(tmp_path):
    (tmp_path / "nao_daily_noaa.txt").write_text(
        " 2000  1  1   0.500\n"
        " 2000  1  2  -1.250\n"
        " 2000  1 3-99.000\n"
        " 2000  1  4   2.000\n"
    )
    nao = cc.load_nao(tmp_path)
    assert list(nao.index.strftime("%Y-%m-%d")) == ["2000-01-01", "2000-01-02", "2000-01-04"]
    assert nao.tolist() == [0.5, -1.25, 2.0]


def test_winter_of_assigns_autumn_to_the_following_january():
    days = pd.to_datetime(["2000-03-31", "2000-10-01", "2000-12-31", "2001-01-01"])
    assert cc.winter_of(days).tolist() == [2000, 2001, 2001, 2001]


def test_exceedances_use_each_stations_own_quantile_and_keep_missing():
    days = pd.date_range("2000-01-01", periods=100)
    gust = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 10})
    rain = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    gust.index = rain.index = days
    gust.iloc[0, 0] = np.nan
    ge, re_ = cc.exceedances(gust, rain, q=0.9)
    assert np.isnan(ge.iloc[0, 0])
    assert ge["a"].sum() == ge["b"].sum() == 10
    assert re_["a"].sum() == 10


def _frame(values):
    return pd.DataFrame({"s": values}, index=pd.date_range("2000-01-01", periods=len(values)))


def test_conditional_chi_known_answer():
    ge = _frame([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    re_ = _frame([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = cc.conditional_chi(ge, re_, np.ones(10, dtype=bool))
    assert r["chi"] == pytest.approx(0.25)
    assert r["rain_rate"] == pytest.approx(0.2)
    assert r["gust_rate"] == pytest.approx(0.4)
    assert r["joint_rate"] == pytest.approx(0.1)
    assert r["ratio_to_independence"] == pytest.approx(1.25)


def test_conditional_chi_ignores_missing_and_applies_mask():
    ge = _frame([1.0, np.nan, 1.0, 0.0])
    re_ = _frame([1.0, 1.0, np.nan, 0.0])
    r = cc.conditional_chi(ge, re_, np.array([True, True, True, False]))
    assert r["chi"] == pytest.approx(1.0)
    assert r["rain_rate"] == pytest.approx(1.0)


def test_conditional_chi_without_gust_extremes_is_nan():
    zeros = _frame([0.0, 0.0])
    r = cc.conditional_chi(zeros, zeros, np.ones(2, dtype=bool))
    assert np.isnan(r["chi"]) and np.isnan(r["ratio_to_independence"])


def test_by_group_separates_a_planted_dependence(monkeypatch):
    """Rain follows the gust only in group A, so only A exceeds independence."""
    monkeypatch.setattr(cc, "N_BOOT", 100)
    rng = np.random.default_rng(1)
    days = pd.date_range("2000-01-01", periods=4000)
    gust = rng.random(4000) < 0.1
    rain = rng.random(4000) < 0.1
    group = np.where(np.arange(4000) % 2 == 0, "A", "B")
    rain = np.where(group == "A", gust, rain)
    ge = pd.DataFrame({"s": gust.astype(float)}, index=days)
    re_ = pd.DataFrame({"s": rain.astype(float)}, index=days)
    t = cc.by_group(ge, re_, pd.Series(group, index=days)).set_index("group")
    assert t.loc["A", "chi"] == pytest.approx(1.0)
    assert t.loc["B", "ratio_to_independence"] == pytest.approx(1.0, abs=0.5)
    assert t["days"].sum() == 4000
    assert t.loc["A", "ratio_p05"] <= t.loc["A", "ratio_to_independence"] <= t.loc["A", "ratio_p95"]


def test_storm_rates_shares_and_ratio():
    days = pd.date_range("2000-01-01", periods=4)
    labels = pd.Series(["x", "x", "x", "y"], index=days)
    events = pd.DataFrame({"peak_day": [days[0], days[3], days[3]], "loss_index": [1.0, 2.0, 1.0]})
    t = cc.storm_rates(events, labels).set_index("group")
    assert t.loc["y", "events"] == 2
    assert t.loc["y", "share_of_loss"] == pytest.approx(0.75)
    assert t.loc["y", "event_rate_ratio"] == pytest.approx((2 / 3) / 0.25)
    assert t["share_of_events"].sum() == pytest.approx(1.0)


def test_trend_recovers_a_slope_and_finds_none_in_noise():
    years = np.arange(1990, 2025)
    up = cc.trend(pd.Series(2.0 * years, index=years))
    assert up["tau"] == pytest.approx(1.0) and up["sen_per_decade"] == pytest.approx(20.0)
    flat = cc.trend(pd.Series(np.random.default_rng(3).normal(size=len(years)), index=years))
    assert flat["p"] > 0.05 and flat["sen_low"] < 0 < flat["sen_high"]


def test_regimes_recover_two_planted_patterns():
    """Two opposite pressure patterns plus noise; the labels must split the days along them."""
    rng = np.random.default_rng(5)
    n, ny, nx = 400, 6, 8
    lat = np.linspace(30, 75, ny)
    days = pd.date_range("2000-01-01", periods=n)
    pattern = rng.normal(size=(ny, nx))
    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    field = 1010 + 8 * sign[:, None, None] * pattern + rng.normal(0, 0.5, (n, ny, nx))
    labels, pc1, explained, anomalies = cc.regimes(field, lat, days)
    assert labels.shape == (n,) and anomalies.shape == field.shape
    assert explained[0] > 0.8
    assert abs(np.corrcoef(pc1, sign)[0, 1]) > 0.99
    # Each regime is dominated by one sign of the planted pattern
    for k in np.unique(labels):
        assert abs(sign[labels == k].mean()) > 0.9


def test_saved_results_are_consistent_with_the_stated_shares():
    """The committed tables: groups cover all days and all 861 events."""
    root = cc.ROOT / "results" / "circulation"
    if not (root / "chi_by_nao.csv").exists():
        pytest.skip("results not present")
    for name in ("nao", "regime"):
        chi = pd.read_csv(root / f"chi_by_{name}.csv")
        storms = pd.read_csv(root / f"storms_by_{name}.csv")
        assert chi["share_of_days"].sum() == pytest.approx(1.0, abs=1e-3)
        assert storms["events"].sum() == 861
        assert storms["share_of_loss"].sum() == pytest.approx(1.0, abs=1e-3)
