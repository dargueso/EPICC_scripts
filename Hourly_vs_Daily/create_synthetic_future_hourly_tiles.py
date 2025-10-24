#!/usr/bin/env python
"""
Memory-efficient version that calculates quantiles directly without storing full time series.
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DRIVER'] = 'core'  # Use memory driver to avoid file corruption
os.environ['NETCDF4_USE_CACHE'] = '0'  # Disable NetCDF caching

import re, glob
import time
from pathlib import Path
import xarray as xr
import numpy as np
from joblib import Parallel, delayed
import config as cfg
import synthetic_future_utils as sf
import warnings
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
# ---------------------------------------------------------------------
# User-configurable section
# ---------------------------------------------------------------------

tile_size   = 50          # number of native gridpoints per tile
buffer_lab  = "025buffer" # used only for output filenames
N_JOBS      = 20         # Can now use more jobs since memory usage is much lower

# Pattern of your pre-split tile files
pattern_tiles = (
    "{path}/{wrun}/split_files_tiles_{tsize}_{buffer_lab}/"
    "UIB_DAY_RAIN_20??-??_{ytile}y-{xtile}x_{buffer_lab}.nc"
)

def discover_tiles(path_template, wrun):
    """Scan disk for all available y/x tile IDs."""
    tiles = set()
    for fp in Path(cfg.path_in).glob(
        f"{wrun}/split_files_tiles_{tile_size}_{buffer_lab}/UIB_DAY_RAIN_20??-??_*y-*x_{buffer_lab}.nc"
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
        buffer_lab=buffer_lab,
        ytile=ytile,
        xtile=xtile,
    )
    files = sorted(glob.glob(pattern))
    print(f"[{wrun}] Found {len(files)} files for tile {ytile}y-{xtile}x")
    return files

def process_tile_safe(wrun, ytile, xtile, min_ytile, max_ytile, min_xtile, max_xtile):
    """Safe wrapper around process_tile with error handling."""
    try:
        return process_tile(wrun, ytile, xtile,min_ytile, max_ytile, min_xtile, max_xtile)
    except Exception as e:
        print(f"ERROR processing tile {ytile}y-{xtile}x: {e}")
        import traceback
        traceback.print_exc()
        return None
def process_tile(wrun, ytile, xtile, min_ytile, max_ytile, min_xtile, max_xtile):
    """Worker function executed in parallel - now much more memory efficient."""
    import gc
    
    start_time = time.time()
    files = build_file_list(wrun, ytile, xtile)
    if not files:
        print(f"[{wrun}] — no files found for tile {ytile}y-{xtile}x, skipping.")
        return
    
    # Skip if output already exists
    output_file = f'{cfg.path_out}/{wrun}/future_synthetic_quant_per_sample_{ytile}y-{xtile}x_{buffer_lab}.nc'
    if Path(output_file).exists():
        print(f"[{wrun}] — output already exists for tile {ytile}y-{xtile}x, skipping.")
        return

    # Initialize variables for cleanup
    finf = None
    rainfall_probability = None
    
    try:
        print(f"[{wrun}] Loading data for tile {ytile}y-{xtile}x...")
        
        # CRITICAL: Use cache=False to avoid file caching issues
        finf = xr.open_mfdataset(
            files,
            combine="by_coords",
            parallel=False,  # Disable parallel to avoid HDF5 conflicts
            chunks={"time": 12},
            engine='netcdf4',
            cache=False,  # Disable caching to avoid corruption
        )

        ds_dist = (
            f"{cfg.path_in}/EPICC_2km_ERA5/split_files_tiles_{tile_size}_{buffer_lab}/"
            f"rainfall_probability_optimized_conditional_5mm_bins_{ytile}y-{xtile}x_{buffer_lab}.nc"
        )
        
        # Load the rainfall probability dataset
        rainfall_probability = xr.open_dataset(ds_dist, cache=False)

        ## Calculation of synthetic future hourly data
        rain_daily_future = finf.RAIN.where(finf.RAIN > cfg.WET_VALUE_D)
        
        # Force immediate computation and conversion to numpy
        print(f"[{wrun}] Computing rain array...")
        rain_arr = rain_daily_future.values.astype(np.float32, copy=True)
        lat_arr = finf.lat.isel(time=0).values.copy()
        lon_arr = finf.lon.isel(time=0).values.copy()

        # convert daily total to its bin index
        print(f"[{wrun}] Computing bin indices...")
        bin_idx = (np.digitize(rain_arr, cfg.drain_bins) - 1).astype(np.int16)

        # Load and prepare probability distributions
        print(f"[{wrun}] Loading probability distributions...")
        wet_hour_dist_present = rainfall_probability.wet_hours_distribution.values.copy()
        hourly_intensity_dist_present = rainfall_probability.hourly_distribution.values.copy()
        samples_per_bin_present = rainfall_probability.samples_per_bin.values.copy()

        # Close files explicitly BEFORE processing
        print(f"[{wrun}] Closing input files...")
        rainfall_probability.close()
        finf.close()
        rainfall_probability = None
        finf = None
        gc.collect()
        
        # NOW process the data with files safely closed
        print(f"[{wrun}] Processing distributions...")
        wet_hour_dist_present = np.nan_to_num(wet_hour_dist_present, nan=0.0)
        hourly_intensity_dist_present = np.nan_to_num(hourly_intensity_dist_present, nan=0.0)
        samples_per_bin_present = np.nan_to_num(samples_per_bin_present, nan=0.0)

        whdp_weighted = wet_hour_dist_present * samples_per_bin_present[:, np.newaxis, :, :]
        hidp_weighted = hourly_intensity_dist_present * samples_per_bin_present[:, np.newaxis, :, :]

        comp_samp = sf._window_sum(samples_per_bin_present, cfg.buffer)
        comp_wWet = sf._window_sum(whdp_weighted, cfg.buffer)
        comp_wHr = sf._window_sum(hidp_weighted, cfg.buffer)

        # Safe division
        nonzero_mask = comp_samp != 0
        comp_wWet = np.where(nonzero_mask[:, None], 
                            comp_wWet / comp_samp[:, None], 
                            0.0)
        comp_wHr = np.where(nonzero_mask[:, None], 
                            comp_wHr / comp_samp[:, None], 
                            0.0)
        wet_cdf = np.cumsum(comp_wWet, axis=1).astype(np.float32, order="C")
        hour_cdf = np.cumsum(comp_wHr, axis=1).astype(np.float32, order="C")

        # Clean up intermediate arrays
        del wet_hour_dist_present, hourly_intensity_dist_present, samples_per_bin_present
        del whdp_weighted, hidp_weighted, comp_samp, comp_wWet, comp_wHr, nonzero_mask
        gc.collect()

        # Define quantiles
        qs_values = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)
        n_quantiles = len(qs_values)

        ny, nx = rain_arr.shape[1:]

        ix0, ix1 = cfg.buffer, nx - cfg.buffer
        iy0, iy1 = cfg.buffer, ny - cfg.buffer

        if xtile == "000": ix0 = 0
        if ytile == "000": iy0 = 0
        if xtile == max_xtile: ix1 = nx
        if ytile == max_ytile: iy1 = ny


        
        # Calculate actual inner tile dimensions
        ny_inner = iy1 - iy0
        nx_inner = ix1 - ix0
        
        print(f"[{wrun}] Tile {ytile}y-{xtile}x: {ny}x{nx} -> inner: {ny_inner}x{nx_inner}")

        output_ny = ny_inner
        output_nx = nx_inner

        result_quantiles = np.full((cfg.n_samples, n_quantiles, output_ny, output_nx), 
                                   np.nan, dtype=np.float32)

        print(f"[{wrun}] Generating {cfg.n_samples} samples...")
        for sample in range(cfg.n_samples):
            if sample % 200 == 0:  # Print every 200 samples
                print(f"  Sample {sample + 1}/{cfg.n_samples}")
            
            sample_quantiles = sf.generate_quantiles_directly(
                rain_arr, wet_cdf, hour_cdf, bin_idx, 
                cfg.hrain_bins.astype(np.float32),
                qs_values,
                iy0, iy1, ix0, ix1,
                thresh=cfg.WET_VALUE_H,
                seed=123 + sample)
            result_quantiles[sample, :, :, :] = sample_quantiles
        
        # Edge padding
        if xtile == min_xtile: result_quantiles[:,:,:,:cfg.buffer]=np.nan
        if xtile == max_xtile: result_quantiles[:,:,:,-cfg.buffer:]=np.nan
        if ytile == min_ytile: result_quantiles[:,:,:cfg.buffer,:]=np.nan
        if ytile == max_ytile: result_quantiles[:,:,-cfg.buffer:,:]=np.nan


        # Create xarray Dataset
        print(f"[{wrun}] Creating output dataset...")
        future_synthetic_quant = xr.Dataset(
            data_vars=dict(
                precipitation=(("sample", "quantile", "y", "x"), result_quantiles),
                lat=(("y", "x"), lat_arr[iy0:iy1, ix0:ix1]),
                lon=(("y", "x"), lon_arr[iy0:iy1, ix0:ix1]),
            ),
            coords=dict(
                sample=np.arange(cfg.n_samples),
                quantile=qs_values,
                y=np.arange(lat_arr[iy0:iy1, ix0:ix1].shape[0]),
                x=np.arange(lon_arr[iy0:iy1, ix0:ix1].shape[1]),
            ),
            attrs={
                'description': 'Synthetic future hourly rainfall quantiles per sample',
                'units': 'mm/hour',
                'wet_threshold_daily': cfg.WET_VALUE_D,
                'wet_threshold_hourly': cfg.WET_VALUE_H,
                'note': 'Quantiles calculated directly from synthetic hourly values, only for values > wet_threshold. Edge tiles padded with NaN.',
                'tile_size': tile_size,
                'buffer_size': cfg.buffer,
                'actual_inner_size': f'{ny_inner}x{nx_inner}'
            }
        )
        
        # Save results
        print(f"[{wrun}] Saving output files...")
        future_synthetic_quant.to_netcdf(output_file)
        
        # Calculate confidence intervals
        future_synthetic_quant_renamed = future_synthetic_quant.rename({'quantile': 'qs_time'})    
        future_synthetic_quant_confidence = future_synthetic_quant_renamed.quantile(q=[0.025, 0.975], dim='sample')
        confidence_file = f'{cfg.path_out}/{wrun}/future_synthetic_quant_confidence_{ytile}y-{xtile}x_{buffer_lab}.nc'
        future_synthetic_quant_confidence.to_netcdf(confidence_file)

        print(f"[{wrun}] Tile {ytile}y-{xtile}x complete: {dict(future_synthetic_quant.sizes)}")
        
        # Clean up
        del future_synthetic_quant, future_synthetic_quant_renamed, future_synthetic_quant_confidence
        
    except Exception as e:
        print(f"[{wrun}] ERROR in tile {ytile}y-{xtile}x: {e}")
        raise
        
    finally:
        # CRITICAL: Ensure all files are closed and memory freed
        try:
            if rainfall_probability is not None:
                rainfall_probability.close()
        except:
            pass
        try:
            if finf is not None:
                finf.close()
        except:
            pass
        
        # Aggressive cleanup
        try:
            del rain_arr, bin_idx, result_quantiles, lat_arr, lon_arr
            del wet_cdf, hour_cdf
        except:
            pass
        
        gc.collect()
    
    end_time = time.time()
    print(f'[{wrun}] ======> Tile {ytile}y-{xtile}x DONE in {(end_time-start_time):.2f} seconds \n')

def main():
    wrf_runs = ["EPICC_2km_ERA5_CMIP6anom"]

    for wrun in wrf_runs:
        tiles = discover_tiles(pattern_tiles, wrun)
        if not tiles:
            raise RuntimeError(f"No tiles found for run {wrun}")

        # Find min/max tile indices
        ytiles = sorted(set(y for y, x in tiles))
        xtiles = sorted(set(x for y, x in tiles))
        
        min_ytile = ytiles[0]
        max_ytile = ytiles[-1]
        min_xtile = xtiles[0]
        max_xtile = xtiles[-1]

        print(f"[{wrun}] Found {len(tiles)} tiles")
        print(f"  y-range: {min_ytile} to {max_ytile}")
        print(f"  x-range: {min_xtile} to {max_xtile}")
        print(f"Processing tiles...\n")


        # Process tiles serially with error handling
        # for y, x in tiles:
        #     result = process_tile_safe(wrun, y, x)
        #     if result is None:
        #         print(f"Skipping failed tile {y}y-{x}x")
                
        # Alternative: if you want to use parallel processing, uncomment below and comment out the serial loop above
        Parallel(n_jobs=N_JOBS)(
            delayed(process_tile_safe)(wrun, y, x, min_ytile, max_ytile, min_xtile, max_xtile) for y, x in tiles
        )

if __name__ == "__main__":
    main()
