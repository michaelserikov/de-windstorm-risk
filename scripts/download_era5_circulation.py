"""Download ERA5 fields for the circulation analysis (October to March, 1990-2025).

Fields go to data/raw/era5_circulation/, one file per variable and calendar year.
Finished files are skipped, so the script can be stopped and started again at any time.
"""

from __future__ import annotations

import time
from pathlib import Path

import cdsapi

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "era5_circulation"
LOG = OUT / "download.log"
YEARS = range(1990, 2026)
MONTHS = ["01", "02", "03", "10", "11", "12"]
DAYS = [f"{d:02d}" for d in range(1, 32)]
SIX_HOURLY = ["00:00", "06:00", "12:00", "18:00"]
HOURLY = [f"{h:02d}:00" for h in range(24)]
NORTH_ATLANTIC = [80, -90, 20, 40]
EUROPE = [72, -15, 35, 30]

# dataset, file prefix, request fields specific to the variable
JOBS = [
    ("reanalysis-era5-single-levels", "mslp",
     {"variable": ["mean_sea_level_pressure"], "time": SIX_HOURLY, "area": NORTH_ATLANTIC}),
    ("reanalysis-era5-pressure-levels", "z500",
     {"variable": ["geopotential"], "pressure_level": ["500"], "time": SIX_HOURLY, "area": NORTH_ATLANTIC}),
    ("reanalysis-era5-single-levels", "gust",
     {"variable": ["10m_wind_gust_since_previous_post_processing"], "time": HOURLY, "area": EUROPE}),
    ("reanalysis-era5-single-levels", "precip",
     {"variable": ["total_precipitation"], "time": HOURLY, "area": EUROPE}),
]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True, progress=False)
    todo = [(dataset, prefix, fields, year) for dataset, prefix, fields in JOBS for year in YEARS]
    for number, (dataset, prefix, fields, year) in enumerate(todo, start=1):
        target = OUT / f"{prefix}_{year}.nc"
        if target.exists() and target.stat().st_size > 0:
            continue
        request = {
            "product_type": ["reanalysis"],
            "year": [str(year)],
            "month": MONTHS,
            "day": DAYS,
            "grid": [1.0, 1.0],
            "data_format": "netcdf",
            "download_format": "unarchived",
            **fields,
        }
        for attempt in range(1, 6):
            part = target.with_suffix(".part")
            try:
                log(f"[{number}/{len(todo)}] start {target.name} (attempt {attempt})")
                client.retrieve(dataset, request).download(str(part))
                part.replace(target)
                log(f"[{number}/{len(todo)}] done {target.name} {target.stat().st_size} bytes")
                break
            except Exception as error:  # the service fails transiently; retry, then move on
                log(f"[{number}/{len(todo)}] failed {target.name}: {error}")
                time.sleep(60 * attempt)
    missing = [f"{p}_{y}.nc" for _, p, _, y in todo if not (OUT / f"{p}_{y}.nc").exists()]
    log(f"finished, missing files: {len(missing)} {missing[:10]}")


if __name__ == "__main__":
    main()
