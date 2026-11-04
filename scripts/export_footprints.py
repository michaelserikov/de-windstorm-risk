"""Export event footprints (maximum gust per station and event) for the CLIMADA step."""
from pathlib import Path

import pandas as pd

from windrisk import hazard as hz
from windrisk.run import END, ROOT, START, load

cache, res = ROOT / "data" / "raw", ROOT / "results"
out = res / "climada"
out.mkdir(parents=True, exist_ok=True)
_, daily = load(cache)
stations = pd.read_csv(res / "stations.csv").set_index("station_id")
events = pd.read_csv(res / "events.csv", parse_dates=["start", "end", "peak_day"])
gust = hz.gust_matrix(daily[daily["station_id"].isin(stations.index)], START, END)[stations.index.tolist()]
rows = {}
for i, e in events.iterrows():
    rows[i] = gust.loc[e["start"]:e["end"]].max()
pd.DataFrame(rows).T.to_csv(out / "event_max_gust.csv", float_format="%.2f")
stations[["lat", "lon", "v98", "weight"]].to_csv(out / "stations.csv", float_format="%.5f")
events.to_csv(out / "events.csv", index=False)
print(len(events), len(stations))
