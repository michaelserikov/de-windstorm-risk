"""The 1-in-200 loss in euros with its calibration uncertainty included.

The cubic loss index is scaled to GDV insured losses of eight storms by one factor (geometric mean of the ratios).
The factor's sampling uncertainty (Student t on the log ratios) is combined with the existing bootstrap of the 1-in-200
index, because the bootstrap alone leaves the calibration out. No CLIMADA needed: the cubic impact function is the index.

    python -m windrisk.calibration
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .run import ROOT

# GDV, Sachversicherung, million EUR, portfolio and prices 2020, status February 2022; source in results/climada/summary.json
GDV = {"2007-01-18": ("Kyrill", 3350), "2002-10-27": ("Jeanett", 1340), "2022-02-18": ("Ylenia, Zeynep, Antonia", 1250),
       "2018-01-18": ("Friederike", 1160), "2015-03-31": ("Niklas", 730), "2010-02-28": ("Xynthia", 730),
       "2020-02-10": ("Sabine", 600), "2008-03-01": ("Emma", 610)}
SEED = 21
DRAWS = 200000


def match_storms(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, (name, loss) in GDV.items():
        d = pd.Timestamp(day)
        hit = events[(events["start"] <= d) & (events["end"] >= d)]
        if len(hit):
            rows.append({"storm": name, "gdv_mio": float(loss), "loss_index": float(hit.iloc[0]["loss_index"])})
    return pd.DataFrame(rows)


def fit_scale(gdv: np.ndarray, index: np.ndarray) -> tuple[float, float, int]:
    """Geometric mean of GDV / index, the standard error of its logarithm and the degrees of freedom."""
    logs = np.log(gdv / index)
    return float(np.exp(logs.mean())), float(logs.std(ddof=1) / np.sqrt(len(logs))), len(logs) - 1


def combined_range(index_draws: np.ndarray, scale: float, se_log: float, dof: int, seed: int = SEED,
                   draws: int = DRAWS) -> np.ndarray:
    """Draws of the loss in the units of the scale: bootstrap index times uncertain scale."""
    rng = np.random.default_rng(seed)
    index = rng.choice(index_draws, draws)
    return index * scale * np.exp(se_log * student_t.rvs(dof, size=draws, random_state=rng))


def main() -> dict:
    res = ROOT / "results"
    events = pd.read_csv(res / "events.csv", parse_dates=["start", "end"])
    boot = pd.read_csv(res / "bootstrap.csv")["aep_200"].to_numpy()
    point = json.loads((res / "summary.json").read_text(encoding="utf-8"))["aep_1in200"]
    m = match_storms(events)
    scale, se, dof = fit_scale(m["gdv_mio"].to_numpy(), m["loss_index"].to_numpy())
    all_draws = combined_range(boot, scale, se, dof)
    fixed = boot * scale
    out = {"storms": len(m), "eur_mio_per_index_unit": scale, "se_log_scale": se, "aep_1in200_bn_eur": point * scale / 1000,
           "range_bootstrap_only_bn_eur": [float(np.quantile(fixed, q) / 1000) for q in (0.05, 0.95)],
           "range_with_calibration_bn_eur": [float(np.quantile(all_draws, q) / 1000) for q in (0.05, 0.95)]}
    (res / "climada" / "calibration_uncertainty.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
