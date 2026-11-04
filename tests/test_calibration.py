"""Offline tests of the calibration of the loss index to insured losses."""

import numpy as np
import pandas as pd
import pytest

from windrisk import calibration as cal


def test_scale_is_exact_without_noise():
    index = np.array([0.1, 0.05, 0.02])
    scale, se, dof = cal.fit_scale(1000 * index, index)
    assert scale == pytest.approx(1000.0)
    assert se == pytest.approx(0.0, abs=1e-12)
    assert dof == 2


def test_scale_is_the_geometric_mean():
    scale, se, _ = cal.fit_scale(np.array([2.0, 8.0]), np.array([1.0, 1.0]))
    assert scale == pytest.approx(4.0)
    assert se == pytest.approx(np.log(4) / np.sqrt(2) / np.sqrt(2), rel=1e-9)


def test_match_storms_uses_the_event_window():
    events = pd.DataFrame({"start": pd.to_datetime(["2007-01-17", "2000-01-01"]), "end": pd.to_datetime(["2007-01-19", "2000-01-02"]),
                           "loss_index": [0.15, 0.01]})
    m = cal.match_storms(events)
    assert m["storm"].tolist() == ["Kyrill"]
    assert m["loss_index"].iloc[0] == 0.15


def test_calibration_uncertainty_widens_the_range():
    draws = np.exp(np.random.default_rng(1).normal(0, 0.5, 2000))
    narrow = cal.combined_range(draws, 10.0, 1e-9, 7, draws=50000)
    wide = cal.combined_range(draws, 10.0, 0.3, 7, draws=50000)
    assert np.quantile(wide, 0.95) / np.quantile(wide, 0.05) > np.quantile(narrow, 0.95) / np.quantile(narrow, 0.05)
    assert np.median(wide) == pytest.approx(np.median(narrow), rel=0.05)


def test_results_reproduce_the_stated_factor():
    from windrisk.run import ROOT
    p = ROOT / "results" / "events.csv"
    if not p.exists():
        pytest.skip("results not present")
    m = cal.match_storms(pd.read_csv(p, parse_dates=["start", "end"]))
    assert len(m) == 8
    assert cal.fit_scale(m["gdv_mio"].to_numpy(), m["loss_index"].to_numpy())[0] == pytest.approx(16100.0, rel=0.02)   # the CLIMADA run gives 16,206 through its own treatment of missing stations
