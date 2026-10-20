"""Offline tests: DWD station-list parsing, loss index, event merging, EVT formulas and loss metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from windrisk import data, hazard as hz, risk


def test_station_list_parsing_handles_names_with_spaces_and_missing_flag(tmp_path):
    text = ("Stations_id von_datum bis_datum Stationshoehe geoBreite geoLaenge Stationsname Bundesland Abgabe\n"
            "----------- --------- --------- ------------- --------- --------- ----------------------------------------- ---------- ------\n"
            "00011 19800901 20261019            680     47.9736    8.5205 Donaueschingen (Landeplatz)              Baden-Württemberg                        Frei\n"
            "00044 19690101 20261019             44     52.9336    8.2370 Großenkneten                             Niedersachsen\n")
    (tmp_path / data.STATION_FILE).write_bytes(text.encode("latin-1"))
    st = data.stations(tmp_path)
    assert st["state"].tolist() == ["Baden-Württemberg", "Niedersachsen"]
    assert st.loc[0, "name"] == "Donaueschingen (Landeplatz)"
    assert st.loc[1, "height"] == 44.0


def test_loss_terms_zero_below_v98_and_cubic_above():
    g = pd.DataFrame({1: [20.0, 30.0, 40.0], 2: [np.nan, 25.0, 50.0]})
    t = hz.loss_terms(g, pd.Series({1: 20.0, 2: 25.0}))
    assert t[1].tolist() == pytest.approx([0.0, 0.125, 1.0])
    assert t[2].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_station_weights_split_state_population_and_sum_to_one():
    st = pd.DataFrame({"station_id": [1, 2, 3], "state": ["A", "A", "B"]})
    w = hz.station_weights(st, [1, 2, 3], pd.Series({"A": 300.0, "B": 100.0}))
    assert w.sum() == pytest.approx(1.0)
    assert w.loc[1] == pytest.approx(0.375) and w.loc[3] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        hz.station_weights(st, [1, 2, 3], pd.Series({"A": 300.0}))


def test_events_merge_consecutive_days_cap_length_and_use_footprint_max():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    g = pd.DataFrame({1: [10, 30, 10, 10, 10, 30, 31, 32, 33, 10.0], 2: [10, 10, 40, 10, 10, 10, 10, 10, 10, 10.0]},
                     index=dates)
    v98 = pd.Series({1: 20.0, 2: 20.0})
    ev = hz.events(g, v98, pd.Series({1: 0.5, 2: 0.5}), gap_days=0, max_days=3).sort_values("start")
    assert ev["start"].dt.day.tolist() == [2, 6, 9]          # days 2-3, 6-8 (capped at 3), then 9
    first = ev.iloc[0]                                         # footprint: station 1 max 30, station 2 max 40
    assert first["loss_index"] == pytest.approx(0.5 * 0.5 ** 3 + 0.5 * 1.0)


def test_season_labels():
    assert hz.season_of(pd.Series(pd.to_datetime(["2006-10-01", "2007-03-31", "2007-01-18"]))).tolist() == [2007, 2007, 2007]


def test_gpd_fit_recovers_parameters_and_return_level_matches_quantile():
    x = genpareto.rvs(0.3, scale=2.0, size=20_000, random_state=np.random.default_rng(1))
    xi, sigma = risk.fit_gpd(x)
    assert xi == pytest.approx(0.3, abs=0.05) and sigma == pytest.approx(2.0, rel=0.05)
    u, rate, T = 5.0, 3.0, 100.0
    assert risk.gpd_return_level(u, rate, 0.3, 2.0, T) == pytest.approx(u + genpareto.ppf(1 - 1 / (rate * T), 0.3, scale=2.0))


def test_simulated_aal_matches_frequency_times_severity():
    rng = np.random.default_rng(2)
    seasons, u = 50, 1.0
    losses = np.r_[rng.uniform(0, 1, 200), 1 + rng.exponential(2.0, 100)]
    sim = risk.simulate_seasons(losses, seasons, u, n_sim=200_000, xi=0.0, sigma=2.0)
    expected = (200 / seasons) * 0.5 + (100 / seasons) * (1 + 2.0)
    assert sim["aggregate"].mean() == pytest.approx(expected, rel=0.02)
    assert (sim["max_event"] <= sim["aggregate"] + 1e-12).all()
    assert risk.tvar(sim["aggregate"]) >= sim["aggregate"].quantile(0.995)
