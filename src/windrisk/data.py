"""Open input data, downloaded once and cached in data/raw/.

- DWD Climate Data Center, daily station observations ("KL"): daily maximum wind gust FX (m/s)
  and precipitation RSK (mm). Licence CC BY 4.0, source: Deutscher Wetterdienst.
- Eurostat demo_r_d2jan: population on 1 January by NUTS-1 region (the 16 German states).
  Licence CC BY 4.0, source: Eurostat.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

DWD = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"
STATION_FILE = "KL_Tageswerte_Beschreibung_Stationen.txt"
EUROSTAT = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_r_d2jan"
            "?format=JSON&lang=EN&sex=T&age=TOTAL&unit=NR&time={year}&" +
            "&".join(f"geo={g}" for g in ("DE1 DE2 DE3 DE4 DE5 DE6 DE7 DE8 DE9 DEA DEB DEC DED DEE DEF DEG").split()))

_session = requests.Session()
_session.headers["User-Agent"] = "de-windstorm-risk (research; github.com/michaelserikov)"


def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        r = _session.get(url, timeout=120)
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(url)


def stations(cache: Path) -> pd.DataFrame:
    """DWD station list for daily KL data (fixed-width text, Latin-1)."""
    path = cache / STATION_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(DWD + STATION_FILE))
    rows = []
    for ln in path.read_bytes().decode("latin-1").splitlines()[2:]:
        m = re.match(r"\s*(\d+)\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+(-?[\d.]+)\s+(-?[\d.]+) (.*)$", ln)
        if not m:
            continue
        rest = m.group(7)  # station name in a 41-character field, then the state (one hyphenated word)
        state = rest[41:].split()
        rows.append({"station_id": int(m.group(1)), "start": pd.Timestamp(m.group(2)), "end": pd.Timestamp(m.group(3)),
                     "height": float(m.group(4)), "lat": float(m.group(5)), "lon": float(m.group(6)),
                     "name": rest[:41].strip(), "state": state[0] if state else ""})
    return pd.DataFrame(rows)


def file_index(cache: Path) -> dict[int, str]:
    """Station id -> name of its historical zip file on the DWD server."""
    path = cache / "kl_historical_index.html"
    if not path.exists():
        path.write_bytes(_get(DWD))
    html = path.read_text(encoding="latin-1")
    out = {}
    for name in pd.Series(html.split('href="')[1:]).str.split('"').str[0]:
        if name.startswith("tageswerte_KL_") and name.endswith("_hist.zip"):
            out[int(name.split("_")[2])] = name
    return out


def station_daily(station_id: int, zip_name: str, cache: Path) -> pd.DataFrame:
    """Daily FX (max gust, m/s) and RSK (precipitation, mm) of one station; -999 marks missing values."""
    path = cache / "kl" / zip_name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(DWD + zip_name))
        time.sleep(0.2)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.startswith("produkt_klima_tag"))
        df = pd.read_csv(io.BytesIO(z.read(name)), sep=";", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"date": pd.to_datetime(df["MESS_DATUM"].astype(str), format="%Y%m%d"),
                        "fx": pd.to_numeric(df["FX"], errors="coerce"),
                        "rsk": pd.to_numeric(df["RSK"], errors="coerce")})
    out.loc[out["fx"] < 0, "fx"] = float("nan")
    out.loc[out["rsk"] < 0, "rsk"] = float("nan")
    out["station_id"] = station_id
    return out


def population(cache: Path, year: int = 2024) -> pd.Series:
    """Population on 1 January of `year` by German state (Eurostat NUTS-1), indexed by state name."""
    path = cache / f"eurostat_population_{year}.json"
    if not path.exists():
        path.write_bytes(_get(EUROSTAT.format(year=year)))
    d = pd.read_json(path, typ="series")
    geo = d["dimension"]["geo"]["category"]
    values = d["value"]
    return pd.Series({geo["label"][g]: float(values[str(i)]) for g, i in geo["index"].items()}, name="population")
