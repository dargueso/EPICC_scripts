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
from glob import glob

import config as cfg
import synthetic_future_utils as sf

# ---------------------------------------------------------------------
# User-configurable section
# ---------------------------------------------------------------------
# Pattern of your pre-split tile files  (edit if the path changes)

tile_size   = 50          # number of native gridpoints per tile
buffer_lab  = "025buffer" # used only for output filenames
N_JOBS      = 20         # parallel workers (set 1 to run serially)

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def process_tile(filespath,ny, nx, wrun):
    """Worker function executed in parallel."""

    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    print (f'Analyzing tile y: {ny} x: {nx}')
    ds = xr.open_mfdataset(
        filesin,
        combine="by_coords"
    )

    t0 = time.time()

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

        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/UIB_01H_RAIN_20??-??.nc'))
        files_ref = xr.open_dataset(filesin[0])
        nlats = files_ref.sizes['y']
        nlons = files_ref.sizes['x']
        lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
        latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

        xytiles=list(product(latsteps, lonsteps))
        filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/UIB_01H_RAIN_20??-??'
        print(f'Ej: {filespath}_000y-000x.nc')

        Parallel(n_jobs=N_JOBS)(delayed(process_tile)(filespath,xytile[0],xytile[1],wrun) for xytile in xytiles)
        


if __name__ == "__main__":
    main()
