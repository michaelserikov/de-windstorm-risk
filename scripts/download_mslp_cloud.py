"""Download 6-hourly ERA5 mean sea level pressure over the North Atlantic from public cloud copies.

Sources, no account and no queue:
- 1990-2022: WeatherBench 2 copy of ERA5, 1.5 degree conservative regridding.
- 2023-2025: ARCO-ERA5, 0.25 degree, averaged here to the same 1.5 degree boxes is NOT done:
  the nearest 1.5 degree points are taken, so these years are point samples.

Output: one .npz per year in data/raw/era5_mslp_cloud/ with time (hours since 1900-01-01),
lat, lon and msl (Pa). Finished years are skipped, so the script can be rerun.

Usage: python download_mslp_cloud.py [--probe]
"""
import argparse
import datetime as dt
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np

BASE = "https://storage.googleapis.com/"
WB2 = BASE + "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr/"
ARCO = BASE + "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "era5_mslp_cloud"
LAT_RANGE = (20.0, 80.0)
LON_WEST, LON_EAST = -90.0, 40.0
WORKERS = 24


def fetch(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == tries - 1:
                raise


def meta(root):
    return json.loads(fetch(root + ".zmetadata"))["metadata"]


def decode(raw, array):
    codec = numcodecs.get_codec(array["compressor"])
    data = np.frombuffer(codec.decode(raw), dtype=array["dtype"])
    return data.reshape(array["chunks"], order=array.get("order", "C"))


def coord(root, name, m):
    array = m[f"{name}/.zarray"]
    parts = []
    for i in range(int(np.ceil(array["shape"][0] / array["chunks"][0]))):
        chunk = decode(fetch(f"{root}{name}/{i}"), {**array, "chunks": array["chunks"]})
        parts.append(chunk)
    return np.concatenate(parts)[: array["shape"][0]]


def lon_index(lon):
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    keep = np.where((wrapped >= LON_WEST) & (wrapped <= LON_EAST))[0]
    return keep[np.argsort(wrapped[keep])], np.sort(wrapped[keep])


def hours_since_1900(year_start, year_end):
    start = dt.datetime(1900, 1, 1)
    first = int((dt.datetime(year_start, 1, 1) - start).total_seconds() // 3600)
    last = int((dt.datetime(year_end, 1, 1) - start).total_seconds() // 3600)
    return np.arange(first, last, 6)


def wb2_year(year, m, lat, lon, lat_keep, lon_keep, lon_sorted):
    array = m["mean_sea_level_pressure/.zarray"]
    step = array["chunks"][0]
    base = int((dt.datetime(year, 1, 1) - dt.datetime(1959, 1, 1)).total_seconds() // 21600)
    end = int((dt.datetime(year + 1, 1, 1) - dt.datetime(1959, 1, 1)).total_seconds() // 21600)
    end = min(end, array["shape"][0])
    chunks = range(base // step, (end - 1) // step + 1)

    def one(c):
        return c, decode(fetch(f"{WB2}mean_sea_level_pressure/{c}.0.0"), array)

    with ThreadPoolExecutor(WORKERS) as pool:
        got = dict(pool.map(one, chunks))
    field = np.concatenate([got[c] for c in chunks])[base - chunks[0] * step : end - chunks[0] * step]
    # WB2 order is time, lon, lat
    field = field[:, lon_keep][:, :, lat_keep].transpose(0, 2, 1)
    return hours_since_1900(year, year + 1)[: len(field)], field


def arco_year(year, m, lat_target, lon_target):
    array = m["mean_sea_level_pressure/.zarray"]
    lat = coord(ARCO, "latitude", m)
    lon = coord(ARCO, "longitude", m)
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    lat_i = [int(np.argmin(np.abs(lat - v))) for v in lat_target]
    lon_i = [int(np.argmin(np.abs(lon_wrapped - v))) for v in lon_target]
    times = hours_since_1900(year, year + 1)

    def one(t):
        chunk = decode(fetch(f"{ARCO}mean_sea_level_pressure/{t}.0.0"), array)
        return chunk[0][np.ix_(lat_i, lon_i)]

    with ThreadPoolExecutor(WORKERS) as pool:
        field = np.stack(list(pool.map(one, times.tolist())))
    return times, field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m_wb2 = meta(WB2)
    lat = coord(WB2, "latitude", m_wb2)
    lon = coord(WB2, "longitude", m_wb2)
    lat_keep = np.where((lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1]))[0]
    lon_keep, lon_sorted = lon_index(lon)
    lat_target = lat[lat_keep]
    m_arco = meta(ARCO)

    if args.probe:
        started = dt.datetime.now()
        t, f = wb2_year(2000, m_wb2, lat, lon, lat_keep, lon_keep, lon_sorted)
        print("wb2 2000", f.shape, "seconds", (dt.datetime.now() - started).seconds,
              "mean hPa", round(float(np.nanmean(f)) / 100, 1))
        array = m_arco["mean_sea_level_pressure/.zarray"]
        started = dt.datetime.now()
        hour = int(hours_since_1900(2025, 2026)[-1])
        raw = fetch(f"{ARCO}mean_sea_level_pressure/{hour}.0.0")
        chunk = decode(raw, array)
        print("arco last 2025 step MB", round(len(raw) / 1e6, 2), "seconds", (dt.datetime.now() - started).seconds,
              "mean hPa", round(float(np.nanmean(chunk)) / 100, 1), "nan share", float(np.isnan(chunk).mean()))
        return

    for year in range(1990, 2026):
        target = OUT / f"mslp_{year}.npz"
        if target.exists():
            continue
        if year <= 2022:
            times, field = wb2_year(year, m_wb2, lat, lon, lat_keep, lon_keep, lon_sorted)
        else:
            times, field = arco_year(year, m_arco, lat_target, lon_sorted)
        part = target.with_suffix(".tmp.npz")
        np.savez_compressed(part, time=times, lat=lat_target, lon=lon_sorted, msl=field.astype("float32"))
        part.replace(target)
        print(dt.datetime.now().isoformat(timespec="seconds"), year, field.shape, flush=True)


if __name__ == "__main__":
    sys.exit(main())
