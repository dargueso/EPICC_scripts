#!/usr/bin/env python
"""
Ultra-conservative rainfall analysis script with parallel processing
- Uses minimal memory and very conservative file handling
- Runs each tile as completely separate subprocess
- Parallel processing with proper process isolation
- Each tile gets fresh Python interpreter
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NETCDF4_PLUGIN_PATH'] = ''

import subprocess
import sys

import time
from glob import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configuration
PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN = "EPICC_2km_ERA5_CMIP6anom"

# Parallel processing configuration
N_JOBS = 1  # Number of tiles to process simultaneously (adjust based on your system)

def discover_tiles(file_pattern):
    """Discover all available tiles"""
    print("Discovering available tiles...")
    
    files = glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    tile_pattern = r'(\d{3})y-(\d{3})x'
    tiles = set()
    
    for file in files:
        match = re.search(tile_pattern, os.path.basename(file))
        if match:
            y_coord = match.group(1)
            x_coord = match.group(2)
            tiles.add((y_coord, x_coord))
    
    tiles = sorted(list(tiles))
    print(f"Found {len(tiles)} tiles: {tiles[:5]}{'...' if len(tiles) > 5 else ''}")
    
    return tiles

def check_files_exist(y_coord, x_coord):
    """Check if files exist for a tile before processing"""
    tile_id = f"{y_coord}y-{x_coord}x"
    file_pattern = f'{PATH_IN}/{WRUN}/split_files_tiles_50/UIB_10MIN_RAIN_20??-??_{tile_id}.nc'
    files = glob(file_pattern)
    
    if len(files) == 0:
        return False, "No files found"
    
    # Quick check with ncdump to validate a few files
    test_files = files[:2]  # Test first 2 files
    for test_file in test_files:
        try:
            result = subprocess.run(['ncdump', '-h', test_file], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False, f"ncdump failed on {os.path.basename(test_file)}"
        except Exception as e:
            return False, f"Error testing {os.path.basename(test_file)}: {str(e)}"
    
    return True, f"Found {len(files)} files, validation passed"

def run_single_tile_script(y_coord, x_coord):
    """
    Run the original working script for a single tile by modifying it on the fly
    """
    tile_id = f"{y_coord}y-{x_coord}x"
    
    # Create a temporary script for this specific tile
    temp_script = f"temp_single_tile_{tile_id}_{os.getpid()}.py"  # Add PID to avoid conflicts
    
    # Read the original working script
    original_script = """#!/usr/bin/env python

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NETCDF4_PLUGIN_PATH'] = ''

import xarray as xr
import numpy as np
import time
from glob import glob

# Configuration
PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WET_VALUE_M = 0.1
WET_VALUE_D = 0.1
DRAIN_BINS = np.arange(0, 105, 5)
MRAIN_BINS = np.arange(0, 101, 1)

try:
    from numba import jit
    NUMBA_AVAILABLE = True
    
    @jit(nopython=True, cache=True)
    def compute_histogram_numba(values, bins):
        valid_values = values[~np.isnan(values)]
        valid_values = valid_values[valid_values > 0]
        
        if len(valid_values) == 0:
            return np.zeros(len(bins) - 1, dtype=np.float64)
        
        hist = np.zeros(len(bins) - 1, dtype=np.float64)
        
        for val in valid_values:
            for i in range(len(bins) - 1):
                if bins[i] <= val < bins[i + 1]:
                    hist[i] += 1.0
                    break
            if val >= bins[-1]:
                hist[-1] += 1.0
        
        total = np.sum(hist)
        if total > 0:
            hist = hist / total 
        
        return hist

except ImportError:
    NUMBA_AVAILABLE = False

def compute_histogram_numpy(values, bins):
    valid = values[values > 0]
    if len(valid) == 0:
        return np.zeros(len(bins) - 1)
    
    hist, _ = np.histogram(valid, bins=bins)
    overflow_values = valid[valid >= bins[-1]]
    if len(overflow_values) > 0:
        hist[-1] += len(overflow_values)
    
    total = hist.sum()
    if total > 0:
        hist = hist.astype(float) / total
    
    return hist


def calculate_gini_coefficient(rainfall_data):

    # Get the raw numpy array
    rr_arr = rainfall_data.values  # Shape: (total_10min, y, x)
    
    total_10min, y, x = rr_arr.shape
    
    # Check if we have complete hours (6 10-min intervals per hour)
    if total_10min % 6 != 0:
        # Truncate to complete hours
        complete_10min = (total_10min // 6) * 6
        rr_arr = rr_arr[:complete_10min, :, :]
        total_10min = complete_10min
    
    nhours = total_10min // 6
    
    # Reshape to (nhours, 6, y, x) - group 10min intervals into hours
    rr_reshaped = rr_arr.reshape((nhours, 6, y, x))
    
    # Sort along the 10min axis (axis=1)
    arr_sorted = np.sort(np.nan_to_num(rr_reshaped, nan=0), axis=1)
    
    # Count non-NaN values per hour
    n = np.sum(~np.isnan(rr_reshaped), axis=1)  # Shape: (nhours, y, x)
    
    # Sum of values per hour
    sum_vals = np.nansum(rr_reshaped, axis=1)  # Shape: (nhours, y, x)
    sum_vals = np.where(sum_vals == 0, np.nan, sum_vals)
    
    # Create rank array for the 6 10-min intervals
    r = np.arange(1, 7)  # ranks 1 to 6
    
    # Reshape r to broadcast: (1, 6, 1, 1)
    r = r.reshape(1, 6, 1, 1)
    
    # Expand n to broadcast: (nhours, 1, y, x)
    n_broadcasted = np.expand_dims(n, axis=1)
    
    # Calculate Gini coefficient
    numerator = np.sum((2 * r - n_broadcasted - 1) * arr_sorted, axis=1)
    
    # Avoid division by zero
    denominator = n * sum_vals
    gini = np.where(denominator > 0, numerator / denominator, np.nan)
    
    return gini  # Shape: (nhours, y, x)


def calculate_wet_interval_intensity_distribution_optimized(ds_m_wet_hours, ds_d, wet_interval_fraction,
                                                      drain_bins=DRAIN_BINS, mrain_bins=MRAIN_BINS):
    print("Computing rainfall distributions...")
    
    ny = ds_m_wet_hours.sizes['y']
    nx = ds_m_wet_hours.sizes['x']
    n_drain_bins = len(drain_bins) - 1
    n_mrain_bins = len(mrain_bins) - 1

    wet_interval_fraction_arr = np.zeros((n_drain_bins, ny, nx))
    samples_per_bin = np.zeros((n_drain_bins, ny, nx))
    gini_coeff_per_bin = np.zeros((n_drain_bins, ny, nx))
    interval_distribution_bin = np.zeros((n_drain_bins, n_mrain_bins, ny, nx))
    wet_interval_distribution_bin = np.zeros((n_drain_bins, 6, ny, nx))

    for ibin in range(n_drain_bins):
        if ibin == n_drain_bins - 1:
            lower_bound = drain_bins[ibin]
            upper_bound = np.inf
            print(f"Processing bin {ibin+1}/{n_drain_bins}: {lower_bound}+ mm")
        else:
            lower_bound = drain_bins[ibin]
            upper_bound = drain_bins[ibin + 1]
            print(f"Processing bin {ibin+1}/{n_drain_bins}: {lower_bound}-{upper_bound} mm")

        if upper_bound == np.inf:
            bin_hours = ds_d >= lower_bound
        else:
            bin_hours = (ds_d >= lower_bound) & (ds_d < upper_bound)
        
        total_samples = bin_hours.sum()
        if total_samples == 0:
            print(f"  No samples in bin {ibin+1}, skipping...")
            continue
            
        print(f"  Found {total_samples.values} total samples in bin {ibin+1}")
        
        bin_hours_intervals = bin_hours.reindex(time=ds_m_wet_hours.time, method='ffill')
        bin_ds_m_masked = ds_m_wet_hours.where(bin_hours_intervals)
        
        wet_intervals = bin_ds_m_masked.where(bin_ds_m_masked > 0).count(dim=['time'])
        total_intervals_in_bin = bin_hours_intervals.sum(dim=['time'])
        
        wet_interval_fraction_arr[ibin, :, :] = xr.where(
            total_intervals_in_bin > 0,
            wet_intervals / total_intervals_in_bin,
            0
        ).values
        
        samples_per_bin[ibin, :, :] = bin_hours.sum(dim=['time']).values
        bin_wet_intervals = wet_interval_fraction.where(bin_hours)

        # Gini coefficient
        gini_per_hour = calculate_gini_coefficient(bin_ds_m_masked) 
        gini_coeff_per_bin[ibin, :, :] = np.nanmean(gini_per_hour, axis=0)
 
        # Histograms for interval intensities
        masked_intervals = bin_ds_m_masked.where(bin_ds_m_masked > 0)
        hist_func = compute_histogram_numba if NUMBA_AVAILABLE else compute_histogram_numpy
        
        use_dask = hasattr(masked_intervals.data, 'chunks') and masked_intervals.data.chunks is not None
        if use_dask:
            masked_intervals = masked_intervals.chunk(dict(time=-1))
        
        hist_intervals = xr.apply_ufunc(
            hist_func,
            masked_intervals,
            input_core_dims=[["time"]],
            output_core_dims=[["bin"]],
            kwargs={"bins": mrain_bins},
            vectorize=True,
            dask="parallelized" if use_dask else "forbidden",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"bin": n_mrain_bins}} if use_dask else {},
        )
        
        interval_distribution_bin[ibin, :, :, :] = hist_intervals.transpose("bin", "y", "x").values
        
        # Histogram for wet intervals per hour
        masked_wet_intervals = bin_wet_intervals * 6.0
        use_dask_wet = hasattr(masked_wet_intervals.data, 'chunks') and masked_wet_intervals.data.chunks is not None
        if use_dask_wet:
            masked_wet_intervals = masked_wet_intervals.chunk(dict(time=-1))
        
        hist_wet_intervals = xr.apply_ufunc(
            hist_func,
            masked_wet_intervals,
            input_core_dims=[["time"]],
            output_core_dims=[["bin"]],
            kwargs={"bins": np.arange(1, 8, 1)},
            vectorize=True,
            dask="parallelized" if use_dask_wet else "forbidden",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"bin": 6}} if use_dask_wet else {},
        )
        
        wet_interval_distribution_bin[ibin, :, :, :] = hist_wet_intervals.transpose("bin", "y", "x").values

    return interval_distribution_bin, wet_interval_distribution_bin, samples_per_bin, gini_coeff_per_bin

wrun = "EPICC_2km_ERA5_CMIP6anom"
TILE_ID = "REPLACE_TILE_ID"
file_pattern = f'{PATH_IN}/{wrun}/split_files_tiles_50/UIB_10MIN_RAIN_20??-??_{TILE_ID}.nc'
output_file = f'{PATH_OUT}/{wrun}/rainfall_probability_optimized_conditional_10min_hourly_{TILE_ID}.nc'

print(f"Processing tile {TILE_ID}")
print("Loading data...")

files = sorted(glob(file_pattern))
chunk_dict = {'time': 6*7, 'y': -1, 'x': -1}

ds = xr.open_mfdataset(files, combine='by_coords', parallel=True, chunks=chunk_dict)

ds_m = ds.RAIN.where(ds.RAIN > WET_VALUE_M, 0.0)
ds_d = ds_m.resample(time='1H').sum()
ds_d = ds_d.where(ds_d > WET_VALUE_D)

wet_hours = ds_d > WET_VALUE_D
wet_hours_intervals = wet_hours.reindex(time=ds_m.time, method='ffill')
ds_m_wet_hours = ds_m.where(wet_hours_intervals)

wet_interval_fraction = ds_m_wet_hours.where(ds_m_wet_hours > 0).resample(time='1H').count() / 6.0
wet_interval_fraction = wet_interval_fraction.where(wet_interval_fraction > 0.0)

print("Loading into memory...")
ds_m_wet_hours = ds_m_wet_hours.load()
ds_d = ds_d.load()
wet_interval_fraction = wet_interval_fraction.load()

interval_dist, wet_interval_dist, samples, gini_coeff = calculate_wet_interval_intensity_distribution_optimized(
    ds_m_wet_hours, ds_d, wet_interval_fraction, DRAIN_BINS, MRAIN_BINS
)

# Create output dataset
ny, nx = interval_dist.shape[2], interval_dist.shape[3]
y_coords = ds.y if 'y' in ds.coords else np.arange(ny)
x_coords = ds.x if 'x' in ds.coords else np.arange(nx)

mrain_bin_centers = (MRAIN_BINS[:-1] + MRAIN_BINS[1:]) / 2
interval_vec = np.arange(1, 7)  # 1 to 7

interval_da = xr.DataArray(
    data=interval_dist,
    dims=('drain_bin', 'mrain_bin', 'y', 'x'),
    coords={'drain_bin': DRAIN_BINS[:-1], 'mrain_bin': mrain_bin_centers, 'y': y_coords, 'x': x_coords}
)

wet_interval_da = xr.DataArray(
    data=wet_interval_dist,
    dims=('drain_bin', 'interval', 'y', 'x'),
    coords={'drain_bin': DRAIN_BINS[:-1], 'interval': interval_vec, 'y': y_coords, 'x': x_coords}
)

samples_da = xr.DataArray(
    data=samples,
    dims=('drain_bin', 'y', 'x'),
    coords={'drain_bin': DRAIN_BINS[:-1], 'y': y_coords, 'x': x_coords}
)

gini_da = xr.DataArray(
    data=gini_coeff,
    dims=('drain_bin', 'y', 'x'),
    coords={'drain_bin': DRAIN_BINS[:-1], 'y': y_coords, 'x': x_coords}
)

output_ds = xr.Dataset({
    'interval_distribution': interval_da,
    'wet_interval_distribution': wet_interval_da,
    'samples_per_bin': samples_da,
    'gini_coefficient': gini_da
})

print(f"Saving to {output_file}...")
encoding = {var: {'zlib': True, 'complevel': 4} for var in output_ds.data_vars}
output_ds.to_netcdf(output_file, encoding=encoding)

ds.close()
print(f"Tile {TILE_ID} completed successfully!")
"""
    
    # Write the temporary script with the correct tile ID
    script_content = original_script.replace("REPLACE_TILE_ID", tile_id)
    
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        # Run the script as a separate process with timeout
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=10000)  # 10 minute timeout
        
        if result.returncode == 0:
            return "success", result.stdout
        else:
            return "failed", result.stderr
            
    except subprocess.TimeoutExpired:
        return "timeout", "Process exceeded 10 minute timeout"
    except Exception as e:
        return "error", str(e)
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)

def process_tile_wrapper(tile_coords):
    """
    Wrapper function for parallel processing - processes a single tile
    """
    y_coord, x_coord = tile_coords
    tile_id = f"{y_coord}y-{x_coord}x"
    
    try:
        # Check if output already exists
        output_file = f'{PATH_OUT}/{WRUN}/rainfall_probability_optimized_conditional_10min_hourly_{tile_id}.nc'
        if os.path.exists(output_file):
            return tile_id, "skipped", 0.0, "Already exists"
        
        # Check if files exist and are valid
        files_ok, msg = check_files_exist(y_coord, x_coord)
        if not files_ok:
            return tile_id, "failed", 0.0, f"Files not valid: {msg}"
        
        # Process the tile
        tile_start = time.time()
        status, output = run_single_tile_script(y_coord, x_coord)
        tile_time = time.time() - tile_start
        
        return tile_id, status, tile_time, output[:200] if len(output) > 200 else output
        
    except Exception as e:
        return tile_id, "error", 0.0, str(e)

def main():
    """Main processing function with parallel processing"""
    print("=" * 80)
    print("ULTRA-CONSERVATIVE PARALLEL TILE PROCESSING")
    print("Running tiles as independent processes in parallel")
    print("=" * 80)
    
    total_start = time.time()
    
    # Discover tiles
    file_pattern_discovery = f'{PATH_IN}/{WRUN}/split_files_tiles_50/UIB_10MIN_RAIN_20??-??_*y-*x.nc'
    tiles = discover_tiles(file_pattern_discovery)
    
    if not tiles:
        raise ValueError("No tiles found!")
    
    print(f"Found {len(tiles)} tiles to process")
    print(f"Using {N_JOBS} parallel workers")
    print(f"Each tile runs in completely isolated subprocess")
    
    # Use ProcessPoolExecutor for true process isolation
    successful = 0
    failed = 0
    skipped = 0
    completed = 0
    
    print(f"\nStarting parallel processing...")
    
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        # Submit all tiles for processing
        future_to_tile = {
            executor.submit(process_tile_wrapper, tile): tile 
            for tile in tiles
        }
        
        # Process completed tiles as they finish
        for future in as_completed(future_to_tile):
            tile_coords = future_to_tile[future]
            completed += 1
            
            try:
                tile_id, status, tile_time, output = future.result()
                
                if status == "success":
                    successful += 1
                    print(f"✓ {tile_id}: SUCCESS ({tile_time:.1f}s)")
                elif status == "skipped":
                    skipped += 1
                    print(f"- {tile_id}: SKIPPED ({output})")
                else:
                    failed += 1
                    print(f"✗ {tile_id}: {status}")
                    if len(output) < 100:
                        print(f"    Error: {output}")
                
                # Progress update
                progress = 100 * completed / len(tiles)
                print(f"    Progress: {completed}/{len(tiles)} ({progress:.1f}%)")
                
                # Estimate remaining time
                if completed > 3:  # Wait for a few samples
                    elapsed = time.time() - total_start
                    avg_time_per_tile = elapsed / completed
                    remaining_tiles = len(tiles) - completed
                    est_remaining = remaining_tiles * avg_time_per_tile / N_JOBS  # Account for parallelism
                    print(f"    Estimated time remaining: {est_remaining/60:.1f} minutes")
                
            except Exception as e:
                failed += 1
                tile_id = f"{tile_coords[0]}y-{tile_coords[1]}x"
                print(f"✗ {tile_id}: EXCEPTION - {str(e)}")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total tiles: {len(tiles)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Wall clock time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    if N_JOBS > 1:
        theoretical_sequential_time = total_time * N_JOBS
        speedup = theoretical_sequential_time / total_time
        print(f"Parallel speedup: {speedup:.1f}x (with {N_JOBS} workers)")
    
    success_rate = (successful / len(tiles)) * 100 if len(tiles) > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if failed > 0:
        print(f"\nNote: Failed tiles may have corrupted files or insufficient memory.")
        print(f"Consider reducing N_JOBS if memory issues persist.")

if __name__ == "__main__":
    main()
