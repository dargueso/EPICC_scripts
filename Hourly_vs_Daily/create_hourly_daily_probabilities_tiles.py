#!/usr/bin/env python
"""
Compute hourly & daily rainfall probability distributions *per tile*
and write one output file per tile, in parallel with joblib.

Original statistical logic is unchanged – all updates are in the
tile discovery / job-control section.
"""

import re, glob
import time
from pathlib import Path
from itertools import product

import xarray as xr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as cfg
import synthetic_future_utils as sf

# ---------------------------------------------------------------------
# User-configurable section
# ---------------------------------------------------------------------
# Pattern of your pre-split tile files  (edit if the path changes)
pattern_tiles = (
    "{path}/{wrun}/split_files_tiles_{tsize}_025buffer/"
    "UIB_01H_RAIN_20??-??_{ytile}y-{xtile}x_025buffer.nc"
)
tile_size   = 50          # number of native gridpoints per tile
buffer_lab  = "025buffer" # used only for output filenames
N_JOBS      = 1         # parallel workers (set 1 to run serially)

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def discover_tiles(path_template, wrun):
    """
    Scan disk for all available y/x tile IDs for a given experiment.
    Returns a sorted list of (ytile, xtile) strings, e.g. ('000', '001').
    """
    # grab one month so we don’t list 120× the same tile names
    sample_glob = (
        f"{cfg.path_in}/{wrun}/split_files_tiles_{tile_size}_{buffer_lab}"
        "/UIB_01H_RAIN_20??-01_*y-*x_*.nc"
    )
    tiles = set()
    for fp in Path(cfg.path_in).glob(
        f"{wrun}/split_files_tiles_{tile_size}_{buffer_lab}/UIB_01H_RAIN_20??-01_*y-*x_*.nc"
    ):
        m = re.search(r"_(\d{3})y-(\d{3})x_", fp.name)
        if m:
            tiles.add((m.group(1), m.group(2)))
    return sorted(tiles)


def build_file_list(wrun, ytile, xtile):
    """Return the list of monthly NetCDFs belonging to one tile."""
    pattern = pattern_tiles.format(
        path=cfg.path_in,
        wrun=wrun,
        tsize=tile_size,
        ytile=ytile,
        xtile=xtile,
    )
    files = sorted(glob.glob(pattern))
    print(f"[{wrun}] Found {len(files)} files for tile {ytile}y-{xtile}x")
    return files


def process_tile(wrun, ytile, xtile):
    """Worker function executed in parallel."""
    files = build_file_list(wrun, ytile, xtile)
    if not files:
        print(f"[{wrun}] – no files found for tile {ytile}y-{xtile}x, skipping.")
        return

    t0 = time.time()
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        chunks={"time": 24},  # keep memory small
    )

    # ---------------- statistical core (unchanged) ------------------
    ds_h = ds.RAIN.where(ds.RAIN > cfg.WET_VALUE_H, 0.0)
    ds_d = ds_h.resample(time="1D").sum().where(lambda x: x > cfg.WET_VALUE_D)

    wet_days = ds_d > cfg.WET_VALUE_D
    wet_days_hourly = wet_days.reindex(time=ds_h.time, method="ffill")
    ds_h_wet_days = ds_h.where(wet_days_hourly)

    wet_hour_fraction = (
        ds_h_wet_days.where(ds_h_wet_days > 0)
        .resample(time="1D")
        .count() / 24.0
    ).where(lambda x: x > 0)

    hdist, wdist, n_samples = sf.calculate_wet_hour_intensity_distribution(
        ds_h_wet_days,
        ds_d,
        wet_hour_fraction,
        drain_bins=cfg.drain_bins,
        hrain_bins=cfg.hrain_bins,
    )

    # ---------------- write result ------------------
    fout = (
        f"{cfg.path_out}/{wrun}/probability_hourly_"
        f"{ytile}y-{xtile}x_{buffer_lab}.nc"
    )
    sf.save_probability_data(
        hdist, wdist, n_samples, cfg.drain_bins, cfg.hrain_bins, fout=fout
    )

    ds.close()
    print(
        f"[{wrun}] tile {ytile}y-{xtile}x done in {time.time()-t0:.1f}s → {fout}"
    )


def main():
    wrf_runs = ["EPICC_2km_ERA5", "EPICC_2km_ERA5_CMIP6anom"]

    for wrun in wrf_runs:
        tiles = discover_tiles(pattern_tiles, wrun)
        if not tiles:
            raise RuntimeError(f"No tiles found for run {wrun}")

        print(
            f"[{wrun}] Found {len(tiles)} tiles – launching with {N_JOBS} jobs\n"
        )
        Parallel(n_jobs=N_JOBS)(
            delayed(process_tile)(wrun, y, x) for y, x in tiles
        )


if __name__ == "__main__":
    main()
