#!/usr/bin/env python
"""
Create synthetic future hourly rainfall data *per tile*
and write one output file per tile, in parallel with joblib.

Original logic is unchanged – all updates are in the
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
from numba.typed import List
from numba import njit, float32, prange 

import config as cfg
import synthetic_future_utils as sf

# ---------------------------------------------------------------------
# User-configurable section
# ---------------------------------------------------------------------

# Pattern of your pre-split tile files  (edit if the path changes)

pattern_tiles = (
    "{path}/{wrun}/split_files_tiles_{tsize}_025buffer/"
    "UIB_DAY_RAIN_20??-??_{ytile}y-{xtile}x_025buffer.nc"
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
    start_time = time.time()
    files = build_file_list(wrun, ytile, xtile)
    if not files:
        print(f"[{wrun}] – no files found for tile {ytile}y-{xtile}x, skipping.")
        return

    t0 = time.time()
    finf = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        chunks={"time": 24},  # keep memory small
    )

    ds_dist = (
        f"{cfg.path_in}/{wrun}/probability_hourly_"
        f"{ytile}y-{xtile}x_{buffer_lab}.nc"
    )

    ## Calculation of synthetic future hourly data
    rain_daily_future = finf.RAIN.where(finf.RAIN > cfg.WET_VALUE_D)
    rain_arr = rain_daily_future.data.astype(np.float32, order='C')

    # Get time dimension info
    n_time = rain_arr.shape[0]
    time_coords = rain_daily_future.time

    # convert daily total to its bin index once so the kernel re-uses it fast
    bin_idx = (np.digitize(rain_arr, cfg.drain_bins) - 1).astype(np.int16)

    ny, nx = rain_arr.shape[1:]
    ix0, ix1 = buffer, nx - buffer
    iy0, iy1 = buffer, ny - buffer
    n_cells = (iy1 - iy0) * (ix1 - ix0)

    sample_buffers = []
    for sample in range(cfg.n_samples):
        buffers = List()
        for _ in range(n_cells):
            buffers.append(List.empty_list(float32))
        sample_buffers.append(buffers)

    wet_hour_dist_present = rainfall_probability.wet_hours_distribution.data
    hourly_intensity_dist_present = rainfall_probability.hourly_distribution.data
    samples_per_bin_present = rainfall_probability.samples_per_bin.data

    wet_hour_dist_present = np.nan_to_num(wet_hour_dist_present, nan=0.0)
    hourly_intensity_dist_present = np.nan_to_num(hourly_intensity_dist_present, nan=0.0)

    whdp_weighted = wet_hour_dist_present * samples_per_bin_present[:, np.newaxis, :, :]
    hidp_weighted = hourly_intensity_dist_present * samples_per_bin_present[:, np.newaxis, :, :]

    comp_samp = sf._window_sum(samples_per_bin_present, cfg.buffer)
    comp_wWet = sf._window_sum(whdp_weighted, cfg.buffer)
    comp_wHr = sf._window_sum(hidp_weighted, cfg.buffer)
    comp_wWet /= comp_samp[:, None]
    comp_wHr /= comp_samp[:, None]

    wet_cdf = np.cumsum(comp_wWet, axis=1).astype(np.float32, order="C")
    hour_cdf = np.cumsum(comp_wHr, axis=1).astype(np.float32, order="C")

    # Generate synthetic data for each sample
    for sample in range(cfg.n_samples):
        print(f"Processing sample {sample + 1}/{cfg.n_samples}")
        sf.generate_dmax_hourly_values_per_timestep(
            rain_arr, wet_cdf, hour_cdf, bin_idx, 
            cfg.hrain_bins.astype(np.float32),
            sample_buffers[sample],
            iy0, iy1, ix0, ix1,
            thresh=cfg.WET_VALUE_H,
            seed=123 + sample)   

    # Calculate quantiles along time dimension for each sample and cell
    qs_values = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)
    n_quantiles = len(qs_values)

    # Result array: (n_samples, n_quantiles, ny, nx)
    result_quantiles = np.full((cfg.n_samples, n_quantiles, ny, nx), np.nan, dtype=np.float32)
    full_output = np.zeros((cfg.n_samples, n_time, ny, nx), dtype=np.float32)

    for sample in range(cfg.n_samples):
        print(f"Calculating quantiles for sample {sample + 1}/{cfg.n_samples}")
        c = 0
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                # Get time series for this cell and sample
                time_series = np.array(sample_buffers[sample][c], dtype=np.float32)
                
                # Only calculate quantiles if we have wet values
                if len(time_series) > 0:
                    # Filter values above WET_VALUE (should already be filtered, but double-check)
                    wet_values = time_series[time_series > cfg.WET_VALUE_H]
                    full_output[sample,:,iy,ix] = time_series
                    if len(wet_values) > 0:
                        result_quantiles[sample, :, iy, ix] = np.quantile(
                            wet_values, qs_values, method="linear")
                c += 1


    # Create xarray DataArray with proper coordinates
    future_synthetic_quant = xr.DataArray(
        result_quantiles,
        dims=("sample", "quantile", "y", "x"),
        coords=dict(
            sample=np.arange(cfg.n_samples),
            quantile=qs_values,
            y=rain_daily_future.y,
            x=rain_daily_future.x,
        ),
        attrs={
            'description': 'Synthetic future hourly rainfall quantiles per sample',
            'units': 'mm/hour',
            'wet_threshold_daily': cfg.WET_VALUE_D,
            'wet_threshold_hourly': cfg.WET_VALUE_H,
            'note': 'Quantiles calculated along time dimension for each sample, only for values > wet_threshold'
        }
    )
    future_synthetic_dmax = xr.DataArray(
        full_output,
        dims=("sample", "time", "y", "x"),
        coords=dict(
            sample=np.arange(cfg.n_samples),
            time=time_coords,
            y=rain_daily_future.y,
            x=rain_daily_future.x,
        ),
        attrs={
            'description': 'Synthetic future daily maximum of hourly rainfall',
            'units': 'mm/hour',
            'wet_threshold_daily': cfg.WET_VALUE_D,
            'wet_threshold_hourly': cfg.WET_VALUE_H,
            'note': 'Daily maximum hourly values for each timestep, only for values > wet_threshold'
        }
    )
    
    # Save results
    fout = (
        f"{cfg.path_out}/{wrun}/probability_hourly_"
        f"{ytile}y-{xtile}x_{buffer_lab}.nc"
    )


    future_synthetic_dmax.to_netcdf(f'{cfg.path_out}/{wrun}/future_synthetic_dmax_{ytile}y-{xtile}x_{buffer_lab}.nc')
    future_synthetic_quant.to_netcdf(f'{cfg.path_out}/{wrun}/future_synthetic_quant_per_sample_{ytile}y-{xtile}x_{buffer_lab}.nc')
    future_synthetic_quant = future_synthetic_quant.rename({'quantile': 'qs_time'})    
    future_synthetic_quant_confidence = future_synthetic_quant.quantile(q=[0.025, 0.975], dim='sample')
    future_synthetic_quant_confidence.to_netcdf(f'{cfg.path_out}/{wrun}/future_synthetic_quant_confidence_{ytile}y-{xtile}x_{buffer_lab}.nc')

    print(f"Result shape: {future_synthetic_quant.shape}")
    print(f"Dimensions: {future_synthetic_quant.dims}")
    print(f"Quantiles: {qs_values}")
    
    end_time = time.time()
    print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')


def main():
    wrf_runs = ["EPICC_2km_ERA5_CMIP6anom"]

    for wrun in wrf_runs:
        tiles = discover_tiles(pattern_tiles, wrun)
        if not tiles:
            raise RuntimeError(f"No tiles found for run {wrun}")

        print(
            f"[{wrun}] Found {len(tiles)} tiles – launching with {N_JOBS} jobs\n"
        )
        Parallel(n_jobs=N_JOBS)(
            delayed(process_tile)(wrun, y, x) for y, x in tiles[0]
        )


if __name__ == "__main__":
    main()
