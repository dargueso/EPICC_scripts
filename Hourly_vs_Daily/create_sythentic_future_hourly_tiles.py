#!/usr/bin/env python
"""
Memory-efficient version that calculates quantiles directly without storing full time series.
"""

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
N_JOBS      = 1          # Can now use more jobs since memory usage is much lower

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

def process_tile(wrun, ytile, xtile):
    """Worker function executed in parallel - now much more memory efficient."""
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

    # Load data with smaller chunks
    finf = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        chunks={"time": 12},  # Smaller chunks
    )

    ds_dist = (
        f"{cfg.path_in}/EPICC_2km_ERA5/split_files_tiles_{tile_size}_{buffer_lab}/"
        f"rainfall_probability_optimized_conditional_5mm_bins_{ytile}y-{xtile}x_{buffer_lab}.nc"
    )
    
    # Load the rainfall probability dataset
    rainfall_probability = xr.open_dataset(ds_dist)

    ## Calculation of synthetic future hourly data
    rain_daily_future = finf.RAIN.where(finf.RAIN > cfg.WET_VALUE_D)
    rain_arr = rain_daily_future.data.astype(np.float32, order='C')

    # Get time dimension info
    time_coords = rain_daily_future.time

    # convert daily total to its bin index once so the kernel re-uses it fast
    bin_idx = (np.digitize(rain_arr, cfg.drain_bins) - 1).astype(np.int16)

    # Convert dask arrays to numpy arrays before passing to numba
    if hasattr(rain_arr, 'compute'):
        rain_arr = rain_arr.compute()
    if hasattr(bin_idx, 'compute'):
        bin_idx = bin_idx.compute()

    ny, nx = rain_arr.shape[1:]
    ix0, ix1 = cfg.buffer, nx - cfg.buffer
    iy0, iy1 = cfg.buffer, ny - cfg.buffer
    
    # Calculate actual inner tile dimensions
    ny_inner = iy1 - iy0
    nx_inner = ix1 - ix0
    
    print(f"Full tile dimensions: {ny}x{nx}")
    print(f"Buffer size: {cfg.buffer}")
    print(f"Inner tile dimensions: {ny_inner}x{nx_inner}")
    
    # Always output tile_size x tile_size (50x50), padding with zeros if needed
    output_ny = tile_size
    output_nx = tile_size

    # Load and prepare probability distributions
    wet_hour_dist_present = rainfall_probability.wet_hours_distribution.data
    hourly_intensity_dist_present = rainfall_probability.hourly_distribution.data
    samples_per_bin_present = rainfall_probability.samples_per_bin.data

    # Convert to numpy if needed and handle NaN values
    if hasattr(wet_hour_dist_present, 'compute'):
        wet_hour_dist_present = wet_hour_dist_present.compute()
    if hasattr(hourly_intensity_dist_present, 'compute'):
        hourly_intensity_dist_present = hourly_intensity_dist_present.compute()
    if hasattr(samples_per_bin_present, 'compute'):
        samples_per_bin_present = samples_per_bin_present.compute()

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

    # Define quantiles
    qs_values = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)
    n_quantiles = len(qs_values)

    # NEW: Generate quantiles directly for each sample - MUCH more memory efficient
    # Always output consistent tile_size x tile_size arrays, padding with NaN in correct positions
    result_quantiles = np.full((cfg.n_samples, n_quantiles, output_ny, output_nx), np.nan, dtype=np.float32)

    # Determine padding positions based on tile location
    # For edge tiles, figure out where the real data should be placed
    pad_top = max(0, tile_size - ny_inner) if ytile == "000" else 0
    pad_left = max(0, tile_size - nx_inner) if xtile == "000" else 0
    pad_bottom = max(0, tile_size - ny_inner) if ytile != "000" and ny_inner < tile_size else 0
    pad_right = max(0, tile_size - nx_inner) if xtile != "000" and nx_inner < tile_size else 0
    
    # Calculate where to place the real data in the output array
    y_start = pad_top
    y_end = y_start + ny_inner
    x_start = pad_left
    x_end = x_start + nx_inner
    
    print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
    print(f"Real data placement: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")

    for sample in range(cfg.n_samples):
        print(f"Processing sample {sample + 1}/{cfg.n_samples}")
        
        # Generate quantiles directly for inner tile only (no buffer processing)
        sample_quantiles = sf.generate_quantiles_directly(
            rain_arr, wet_cdf, hour_cdf, bin_idx, 
            cfg.hrain_bins.astype(np.float32),
            qs_values,
            iy0, iy1, ix0, ix1,  # inner tile boundaries
            thresh=cfg.WET_VALUE_H,
            seed=123 + sample)
        
        # Place real data in correct position with appropriate padding
        result_quantiles[sample, :, y_start:y_end, x_start:x_end] = sample_quantiles

    # Create coordinate arrays for the full tile_size x tile_size output
    # Generate coordinates that match the padding strategy
    if ny_inner == tile_size and nx_inner == tile_size:
        # Normal interior tile
        y_coords = rain_daily_future.y[iy0:iy1]
        x_coords = rain_daily_future.x[ix0:ix1]
    else:
        # Edge tile - create coordinates matching the data placement
        full_y = rain_daily_future.y
        full_x = rain_daily_future.x
        
        # Y coordinates
        if pad_top > 0:
            # Top edge tile - pad at top
            y_coords = np.concatenate([
                np.full(pad_top, np.nan),
                full_y[iy0:iy1]
            ])
        elif pad_bottom > 0:
            # Bottom edge tile - pad at bottom  
            y_coords = np.concatenate([
                full_y[iy0:iy1],
                np.full(pad_bottom, np.nan)
            ])
        else:
            y_coords = full_y[iy0:iy1]
            
        # X coordinates
        if pad_left > 0:
            # Left edge tile - pad at left
            x_coords = np.concatenate([
                np.full(pad_left, np.nan),
                full_x[ix0:ix1]
            ])
        elif pad_right > 0:
            # Right edge tile - pad at right
            x_coords = np.concatenate([
                full_x[ix0:ix1],
                np.full(pad_right, np.nan)
            ])
        else:
            x_coords = full_x[ix0:ix1]

    # Create xarray DataArray with proper coordinates (always tile_size x tile_size)
    future_synthetic_quant = xr.DataArray(
        result_quantiles,
        dims=("sample", "quantile", "y", "x"),
        coords=dict(
            sample=np.arange(cfg.n_samples),
            quantile=qs_values,
            y=y_coords,
            x=x_coords,
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

    future_synthetic_quant.name = "precipitation"
    
    # Save results
    future_synthetic_quant.to_netcdf(output_file)
    
    # Calculate confidence intervals
    future_synthetic_quant_renamed = future_synthetic_quant.rename({'quantile': 'qs_time'})    
    future_synthetic_quant_confidence = future_synthetic_quant_renamed.quantile(q=[0.025, 0.975], dim='sample')
    confidence_file = f'{cfg.path_out}/{wrun}/future_synthetic_quant_confidence_{ytile}y-{xtile}x_{buffer_lab}.nc'
    future_synthetic_quant_confidence.to_netcdf(confidence_file)

    print(f"Result shape: {future_synthetic_quant.shape}")
    print(f"Dimensions: {future_synthetic_quant.dims}")
    print(f"Quantiles: {qs_values}")
    
    # Clean up memory and explicitly close datasets
    import gc
    finf.close()
    rainfall_probability.close()
    del rain_arr, bin_idx, result_quantiles
    del future_synthetic_quant, rainfall_probability, finf
    gc.collect()
    
    end_time = time.time()
    print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')

def main():
    wrf_runs = ["EPICC_2km_ERA5_CMIP6anom"]

    for wrun in wrf_runs:
        tiles = discover_tiles(pattern_tiles, wrun)
        if not tiles:
            raise RuntimeError(f"No tiles found for run {wrun}")

        print(f"[{wrun}] Found {len(tiles)} tiles — launching with {N_JOBS} jobs\n")
        
        Parallel(n_jobs=N_JOBS)(
            delayed(process_tile)(wrun, y, x) for y, x in tiles
        )

if __name__ == "__main__":
    main()