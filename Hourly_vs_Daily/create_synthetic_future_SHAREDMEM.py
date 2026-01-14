#!/usr/bin/env python
"""
Memory-efficient full-domain version with multiprocessing.
Uses shared memory for zero-copy data access across workers.
FIXED: Correct buffer handling + FAST multiprocessing via shared memory
"""
import os
import time
import numpy as np
import xarray as xr
from multiprocessing import Pool, shared_memory
from numba import njit
from scipy.ndimage import uniform_filter
import warnings
import shutil
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN_PRESENT = "EPICC_2km_ERA5"
WRUN_FUTURE = "EPICC_2km_ERA5_CMIP6anom"

# Frequency configuration
FREQ_HIGH = '1H'   # High frequency (e.g., '10MIN', '1H')
FREQ_LOW = 'D'     # Low frequency (e.g., '1H', '3H', '6H', '12H', 'D')

# Wet thresholds
WET_VALUE_HIGH = 0.1  # mm per high-freq interval
WET_VALUE_LOW = 1.0   # mm per low-freq interval

# Bins - adjust based on frequencies
BINS_HIGH = np.arange(0, 101, 1)  # For hourly: 0-100mm in 1mm steps
BINS_LOW = np.arange(0, 105, 5)   # For daily: 0-100mm in 5mm steps

# Quantiles to calculate from synthetic samples
QUANTILES = np.array([0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85,
                      0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0], 
                     dtype=np.float32)

# Bootstrap confidence levels to compute across samples
BOOTSTRAP_QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.9, 0.95, 0.975, 0.99], dtype=np.float32)

# Processing parameters
TILE_SIZE = 50
BUFFER = 10
N_SAMPLES = 1000
N_PROCESSES = 32  # Adjust based on your system

# Frequency mapping (10-min intervals per period)
FREQ_TO_10MIN = {
    '10MIN': 1,
    '1H': 6,
    '3H': 18,
    '6H': 36,
    '12H': 72,
    'D': 144
}

# Frequency name mapping for file patterns
FREQ_TO_NAME = {
    '10MIN': '10MIN',
    '1H': '01H',
    '3H': '3HOUR',
    '6H': '6HOUR',
    '12H': '12HOUR',
    'D': 'DAY'
}

intervals_high = FREQ_TO_10MIN[FREQ_HIGH]
intervals_low = FREQ_TO_10MIN[FREQ_LOW]
n_interval = intervals_low // intervals_high  # e.g., 24 for hourly given daily

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _window_sum(arr, radius):
    """Sliding-window sum over last two spatial axes."""
    k = 2 * radius + 1
    size = [1] * (arr.ndim - 2) + [k, k]
    return uniform_filter(arr, size=size, mode="nearest") * (k * k)

@njit(cache=True)
def _calculate_quantiles_streaming(values_array, quantiles, thresh):
    """Calculate quantiles from values above threshold."""
    valid_values = values_array[values_array > thresh]
    
    if len(valid_values) == 0:
        return np.full(len(quantiles), np.nan, dtype=np.float32)
    
    valid_values = np.sort(valid_values)
    n = len(valid_values)
    result = np.empty(len(quantiles), dtype=np.float32)
    
    for i, q in enumerate(quantiles):
        if q == 0.0:
            result[i] = valid_values[0]
        elif q == 1.0:
            result[i] = valid_values[-1]
        else:
            index = q * (n - 1)
            lower_idx = int(np.floor(index))
            upper_idx = int(np.ceil(index))
            
            if lower_idx == upper_idx:
                result[i] = valid_values[lower_idx]
            else:
                weight = index - lower_idx
                result[i] = valid_values[lower_idx] * (1 - weight) + valid_values[upper_idx] * weight
    
    return result

@njit(cache=True)
def _process_cell_for_quantiles(rain, bin_idx, wet_cdf, hour_cdf, hr_edges, 
                                thresh, iy, ix, rng_state, quantiles, n_interval):
    """Process a single cell and return quantiles."""
    n_t = rain.shape[0]
    temp_values = np.empty(n_t * n_interval, dtype=np.float32)
    value_count = 0
    
    for t in range(n_t):
        R = rain[t, iy, ix]
        if R <= 0 or not np.isfinite(R):
            continue
        
        b = bin_idx[t, iy, ix]
        if b < 0 or b >= wet_cdf.shape[0]:
            continue
        
        # Sample number of wet intervals
        wet_cdf_slice = wet_cdf[b, :, iy, ix]
        if not np.any(wet_cdf_slice > 0):
            continue
        
        Nh = np.searchsorted(wet_cdf_slice, rng_state.random()) + 1
        if Nh <= 0:
            continue
        
        # Sample interval intensities
        cdf_hr = hour_cdf[b, :, iy, ix]
        if not np.any(cdf_hr > 0):
            continue
        
        idx_bins = np.empty(Nh, np.int64)
        for k in range(Nh):
            idx_bins[k] = np.searchsorted(cdf_hr, rng_state.random())
        
        # Generate intensities
        intens = np.empty(Nh, np.float32)
        for k in range(Nh):
            if idx_bins[k] >= len(hr_edges) - 1:
                idx_bins[k] = len(hr_edges) - 2
            lo = hr_edges[idx_bins[k]]
            hi = hr_edges[idx_bins[k] + 1]
            intens[k] = lo + (hi - lo) * rng_state.random()
        
        s_int = intens.sum()
        if s_int == 0.0:
            continue
        
        values = R * intens / s_int
        
        # Filter by threshold
        if np.any(values < thresh):
            mask = values >= thresh
            if not np.any(mask):
                continue
            values = values[mask]
            values *= R / values.sum()
        
        # Store values
        if len(values) > 0:
            n_values = len(values)
            if value_count + n_values > len(temp_values):
                break
            
            for k in range(n_values):
                temp_values[value_count] = values[k]
                value_count += 1
    
    if value_count > 0:
        return _calculate_quantiles_streaming(temp_values[:value_count], quantiles, thresh)
    else:
        return np.full(len(quantiles), np.nan, dtype=np.float32)

def generate_quantiles_for_tile(rain_arr, wet_cdf, hour_cdf, bin_idx, hr_edges,
                                quantiles, iy0, iy1, ix0, ix1, n_interval, 
                                thresh, seed):
    """Generate quantiles for a tile region."""
    rng = np.random.Generator(np.random.PCG64(seed))
    
    ny_inner = iy1 - iy0
    nx_inner = ix1 - ix0
    n_quantiles = len(quantiles)
    
    result = np.full((n_quantiles, ny_inner, nx_inner), np.nan, dtype=np.float32)
    
    for i, iy in enumerate(range(iy0, iy1)):
        for j, ix in enumerate(range(ix0, ix1)):
            cell_quantiles = _process_cell_for_quantiles(
                rain_arr, bin_idx, wet_cdf, hour_cdf, hr_edges, 
                thresh, iy, ix, rng, quantiles, n_interval)
            result[:, i, j] = cell_quantiles
    
    return result

# =============================================================================
# TILE PROCESSING FUNCTION - SHARED MEMORY VERSION
# =============================================================================

def process_tile(tile_info):
    """Process a single tile - accesses data from shared memory."""
    (iy_tile, ix_tile, TILE_SIZE, config, checkpoint_dir, shm_info) = tile_info
    
    # Unpack config
    BUFFER = config['buffer']
    QUANTILES = config['quantiles']
    BOOTSTRAP_QUANTILES = config['bootstrap_quantiles']
    BINS_HIGH = config['bins_high']
    WET_VALUE_HIGH = config['wet_value_high']
    n_interval = config['n_interval']
    N_SAMPLES = config['n_samples']
    ny_tiles = config['ny_tiles']
    nx_tiles = config['nx_tiles']
    
    n_quantiles = len(QUANTILES)
    n_bootstrap = len(BOOTSTRAP_QUANTILES)
    
    # Attach to shared memory
    shm_rain = shared_memory.SharedMemory(name=shm_info['rain_name'])
    shm_bin = shared_memory.SharedMemory(name=shm_info['bin_name'])
    shm_wet = shared_memory.SharedMemory(name=shm_info['wet_name'])
    shm_hour = shared_memory.SharedMemory(name=shm_info['hour_name'])
    
    # Reconstruct arrays from shared memory
    rain_arr = np.ndarray(shm_info['rain_shape'], dtype=np.float32, buffer=shm_rain.buf)
    bin_idx = np.ndarray(shm_info['bin_shape'], dtype=np.int16, buffer=shm_bin.buf)
    wet_cdf = np.ndarray(shm_info['wet_shape'], dtype=np.float32, buffer=shm_wet.buf)
    hour_cdf = np.ndarray(shm_info['hour_shape'], dtype=np.float32, buffer=shm_hour.buf)
    
    ny, nx = rain_arr.shape[1:]
    
    # Calculate tile boundaries with buffer
    y_start = iy_tile * TILE_SIZE
    y_end = min(y_start + TILE_SIZE, ny)
    x_start = ix_tile * TILE_SIZE
    x_end = min(x_start + TILE_SIZE, nx)
    
    ny_inner = y_end - y_start
    nx_inner = x_end - x_start
    
    # Extract tile with buffer for smoothing context
    y_start_buf = max(0, y_start - BUFFER)
    y_end_buf = min(ny, y_end + BUFFER)
    x_start_buf = max(0, x_start - BUFFER)
    x_end_buf = min(nx, x_end + BUFFER)
    
    rain_tile = rain_arr[:, y_start_buf:y_end_buf, x_start_buf:x_end_buf].copy()
    bin_idx_tile = bin_idx[:, y_start_buf:y_end_buf, x_start_buf:x_end_buf].copy()
    wet_cdf_tile = wet_cdf[:, :, y_start_buf:y_end_buf, x_start_buf:x_end_buf].copy()
    hour_cdf_tile = hour_cdf[:, :, y_start_buf:y_end_buf, x_start_buf:x_end_buf].copy()
    
    # Close shared memory (don't unlink - other processes need it)
    shm_rain.close()
    shm_bin.close()
    shm_wet.close()
    shm_hour.close()
    
    # Coordinates within buffered tile for the actual tile region
    iy0 = y_start - y_start_buf
    iy1 = iy0 + ny_inner
    ix0 = x_start - x_start_buf
    ix1 = ix0 + nx_inner
    
    # Generate N_SAMPLES for this tile
    all_samples = np.full((N_SAMPLES, n_quantiles, ny_inner, nx_inner), 
                          np.nan, dtype=np.float32)
    
    # Generate samples
    for sample in range(N_SAMPLES):
        sample_quantiles = generate_quantiles_for_tile(
            rain_tile, wet_cdf_tile, hour_cdf_tile, bin_idx_tile,
            BINS_HIGH.astype(np.float32), QUANTILES,
            iy0, iy1, ix0, ix1, n_interval, WET_VALUE_HIGH,
            seed=123 + sample)
        
        all_samples[sample, :, :, :] = sample_quantiles
    
    # Compute bootstrap quantiles across samples
    result_quantiles = np.full((n_bootstrap, n_quantiles, ny_inner, nx_inner), 
                               np.nan, dtype=np.float32)
    
    for iq in range(n_quantiles):
        for iy in range(ny_inner):
            for ix in range(nx_inner):
                sample_values = all_samples[:, iq, iy, ix]
                valid_mask = ~np.isnan(sample_values)
                if np.sum(valid_mask) > 0:
                    valid_samples = sample_values[valid_mask]
                    result_quantiles[:, iq, iy, ix] = np.quantile(valid_samples, BOOTSTRAP_QUANTILES)
    
    # Apply edge masking
    if iy_tile == 0:
        result_quantiles[:, :, :BUFFER, :] = np.nan
    if iy_tile == ny_tiles - 1:
        result_quantiles[:, :, -BUFFER:, :] = np.nan
    if ix_tile == 0:
        result_quantiles[:, :, :, :BUFFER] = np.nan
    if ix_tile == nx_tiles - 1:
        result_quantiles[:, :, :, -BUFFER:] = np.nan
    
    # Save checkpoint
    checkpoint_file = f'{checkpoint_dir}/tile_{iy_tile:03d}_{ix_tile:03d}.npy'
    np.save(checkpoint_file, result_quantiles)
    
    return (iy_tile, ix_tile, result_quantiles, ny_inner, nx_inner)

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    print("="*60)
    print(f"Synthetic Future {FREQ_HIGH} from {FREQ_LOW}")
    print("="*60)
    
    total_start = time.time()
    
    # Load present-day probability distributions
    print("\n1. Loading probability distributions...")
    prob_file = f'{PATH_IN}/{WRUN_PRESENT}/condprob_{FREQ_HIGH}_given_{FREQ_LOW}_full_domain.nc'
    ds_prob = xr.open_dataset(prob_file)
    
    # Load future low-frequency rainfall
    print(f"\n2. Loading future {FREQ_LOW} rainfall...")
    freq_name_low = FREQ_TO_NAME[FREQ_LOW]
    future_file = f'{PATH_IN}/{WRUN_FUTURE}/UIB_{freq_name_low}_RAIN.zarr'
    
    try:
        ds_future = xr.open_zarr(future_file, consolidated=True)
    except FileNotFoundError:
        print(f"   ERROR: Could not find {future_file}")
        print(f"   Please ensure your {FREQ_LOW} rainfall data is available")
        raise
    
    # Get data arrays
    rain_low_freq = ds_future.RAIN.where(ds_future.RAIN > WET_VALUE_LOW).values.astype(np.float32, copy=True)
    lat = ds_future.lat.isel(time=0).values.copy()
    lon = ds_future.lon.isel(time=0).values.copy()
    
    ny, nx = rain_low_freq.shape[1:]
    print(f"   Domain size: {ny} x {nx}")
    print(f"   Time steps: {rain_low_freq.shape[0]}")
    
    # Compute bin indices
    print("\n3. Computing bin indices...")
    bin_idx = (np.digitize(rain_low_freq, BINS_LOW) - 1).astype(np.int16)
    
    # Prepare probability distributions
    print("\n4. Preparing probability distributions...")
    wet_dist = ds_prob.cond_prob_n_wet.values.copy()
    hour_dist = ds_prob.cond_prob_intensity.values.copy()
    n_events = ds_prob.n_events.values.copy()
    
    ds_prob.close()
    ds_future.close()
    
    # Convert to proper format and handle NaNs
    wet_dist = np.nan_to_num(wet_dist, nan=0.0)
    hour_dist = np.nan_to_num(hour_dist, nan=0.0)
    n_events = np.nan_to_num(n_events, nan=0.0)
    
    # Transpose to match expected format
    wet_dist = np.transpose(wet_dist, (1, 0, 2, 3))
    hour_dist = np.transpose(hour_dist, (1, 0, 2, 3))
    
    # Apply smoothing with buffer
    print("\n5. Applying spatial smoothing...")
    wet_weighted = wet_dist * n_events[:, np.newaxis, :, :]
    hour_weighted = hour_dist * n_events[:, np.newaxis, :, :]
    
    comp_samp = _window_sum(n_events, BUFFER)
    comp_wet = _window_sum(wet_weighted, BUFFER)
    comp_hour = _window_sum(hour_weighted, BUFFER)
    
    # Safe division
    nonzero_mask = comp_samp != 0
    comp_wet = np.where(nonzero_mask[:, None], comp_wet / comp_samp[:, None], 0.0)
    comp_hour = np.where(nonzero_mask[:, None], comp_hour / comp_samp[:, None], 0.0)
    
    # Create CDFs
    wet_cdf = np.cumsum(comp_wet, axis=1).astype(np.float32, order="C")
    hour_cdf = np.cumsum(comp_hour, axis=1).astype(np.float32, order="C")
    
    # Cleanup
    del wet_dist, hour_dist, n_events, wet_weighted, hour_weighted
    del comp_samp, comp_wet, comp_hour, nonzero_mask
    
    # Create shared memory blocks
    print("\n6. Creating shared memory blocks...")
    shm_rain = shared_memory.SharedMemory(create=True, size=rain_low_freq.nbytes)
    shm_bin = shared_memory.SharedMemory(create=True, size=bin_idx.nbytes)
    shm_wet = shared_memory.SharedMemory(create=True, size=wet_cdf.nbytes)
    shm_hour = shared_memory.SharedMemory(create=True, size=hour_cdf.nbytes)
    
    # Copy data to shared memory
    shm_rain_arr = np.ndarray(rain_low_freq.shape, dtype=np.float32, buffer=shm_rain.buf)
    shm_bin_arr = np.ndarray(bin_idx.shape, dtype=np.int16, buffer=shm_bin.buf)
    shm_wet_arr = np.ndarray(wet_cdf.shape, dtype=np.float32, buffer=shm_wet.buf)
    shm_hour_arr = np.ndarray(hour_cdf.shape, dtype=np.float32, buffer=shm_hour.buf)
    
    shm_rain_arr[:] = rain_low_freq[:]
    shm_bin_arr[:] = bin_idx[:]
    shm_wet_arr[:] = wet_cdf[:]
    shm_hour_arr[:] = hour_cdf[:]
    
    shm_info = {
        'rain_name': shm_rain.name,
        'bin_name': shm_bin.name,
        'wet_name': shm_wet.name,
        'hour_name': shm_hour.name,
        'rain_shape': rain_low_freq.shape,
        'bin_shape': bin_idx.shape,
        'wet_shape': wet_cdf.shape,
        'hour_shape': hour_cdf.shape
    }
    
    # Can now delete original arrays
    del rain_low_freq, bin_idx, wet_cdf, hour_cdf
    
    # Calculate tiles
    ny_tiles = (ny + TILE_SIZE - 1) // TILE_SIZE
    nx_tiles = (nx + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = ny_tiles * nx_tiles
    
    # Setup checkpoint directory
    checkpoint_dir = f'{PATH_OUT}/{WRUN_FUTURE}/checkpoint_tiles'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoint tiles
    existing_tiles = set()
    if os.path.exists(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith('tile_') and fname.endswith('.npy'):
                parts = fname.replace('.npy', '').split('_')
                if len(parts) == 3:
                    try:
                        iy = int(parts[1])
                        ix = int(parts[2])
                        existing_tiles.add((iy, ix))
                    except ValueError:
                        continue
    
    if existing_tiles:
        print(f"   Found {len(existing_tiles)} existing checkpoint tiles - will resume")
    
    print(f"\n7. Processing {total_tiles} tiles ({ny_tiles} x {nx_tiles})...")
    print(f"   Using {N_PROCESSES} processes")
    print(f"   Generating {N_SAMPLES} samples per tile")
    
    # Create config dict
    config = {
        'buffer': BUFFER,
        'quantiles': QUANTILES,
        'bootstrap_quantiles': BOOTSTRAP_QUANTILES,
        'bins_high': BINS_HIGH,
        'wet_value_high': WET_VALUE_HIGH,
        'n_interval': n_interval,
        'n_samples': N_SAMPLES,
        'ny_tiles': ny_tiles,
        'nx_tiles': nx_tiles
    }
    
    # Create tile tasks - skip completed tiles
    tile_tasks = []
    skipped_count = 0
    for iy_tile in range(ny_tiles):
        for ix_tile in range(nx_tiles):
            if (iy_tile, ix_tile) in existing_tiles:
                skipped_count += 1
                continue
            tile_tasks.append((iy_tile, ix_tile, TILE_SIZE, config, checkpoint_dir, shm_info))
    
    if skipped_count > 0:
        print(f"   Skipping {skipped_count} already completed tiles")
    print(f"   Processing {len(tile_tasks)} remaining tiles")
    
    # Process tiles in parallel with progress tracking
    print(f"   Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    completed = [skipped_count]
    
    def progress_callback(result):
        completed[0] += 1
        elapsed = time.time() - total_start
        tiles_per_min = completed[0] / (elapsed / 60) if elapsed > 0 else 0
        remaining_tiles = total_tiles - completed[0]
        eta_min = remaining_tiles / tiles_per_min if tiles_per_min > 0 else 0
        print(f"   Completed: {completed[0]:4d}/{total_tiles} tiles "
              f"({100*completed[0]/total_tiles:5.1f}%) | "
              f"Rate: {tiles_per_min:.2f} tiles/min | "
              f"ETA: {eta_min:.1f} min")
    
    try:
        results_list = []
        if len(tile_tasks) > 0:
            with Pool(processes=N_PROCESSES) as pool:
                async_results = []
                for tile_task in tile_tasks:
                    result = pool.apply_async(process_tile, args=(tile_task,), 
                                            callback=lambda r: progress_callback(r))
                    async_results.append(result)
                
                # Wait for all tasks to complete
                for r in async_results:
                    results_list.append(r.get())
    
    finally:
        # Clean up shared memory
        print("\n   Cleaning up shared memory...")
        shm_rain.close()
        shm_bin.close()
        shm_wet.close()
        shm_hour.close()
        shm_rain.unlink()
        shm_bin.unlink()
        shm_wet.unlink()
        shm_hour.unlink()
    
    # Load all results (new + checkpointed)
    print("\n8. Loading all tile results (including checkpoints)...")
    results = results_list
    for iy_tile, ix_tile in existing_tiles:
        checkpoint_file = f'{checkpoint_dir}/tile_{iy_tile:03d}_{ix_tile:03d}.npy'
        tile_data = np.load(checkpoint_file)
        
        y_start = iy_tile * TILE_SIZE
        y_end = min(y_start + TILE_SIZE, ny)
        x_start = ix_tile * TILE_SIZE
        x_end = min(x_start + TILE_SIZE, nx)
        ny_inner = y_end - y_start
        nx_inner = x_end - x_start
        
        results.append((iy_tile, ix_tile, tile_data, ny_inner, nx_inner))
    
    # Combine results
    print("\n9. Combining tile results...")
    n_quantiles = len(QUANTILES)
    n_bootstrap = len(BOOTSTRAP_QUANTILES)
    result_full = np.full((n_bootstrap, n_quantiles, ny, nx), np.nan, dtype=np.float32)
    
    for result in results:
        if result is not None:
            iy_tile, ix_tile, tile_data, ny_inner, nx_inner = result
            
            y_start = iy_tile * TILE_SIZE
            y_end = y_start + ny_inner
            x_start = ix_tile * TILE_SIZE
            x_end = x_start + nx_inner
            
            result_full[:, :, y_start:y_end, x_start:x_end] = tile_data
    
    # Create output dataset
    print("\n10. Creating output dataset...")
    ds_output = xr.Dataset(
        data_vars=dict(
            precipitation=(("bootstrap_quantile", "quantile", "y", "x"), result_full),
            lat=(("y", "x"), lat),
            lon=(("y", "x"), lon),
        ),
        coords=dict(
            bootstrap_quantile=BOOTSTRAP_QUANTILES,
            quantile=QUANTILES,
            y=np.arange(ny),
            x=np.arange(nx),
        ),
        attrs={
            'description': f'Bootstrap confidence intervals for synthetic future {FREQ_HIGH} rainfall quantiles from {FREQ_LOW} totals',
            'units': f'mm/{FREQ_HIGH}',
            'wet_threshold_high': WET_VALUE_HIGH,
            'wet_threshold_low': WET_VALUE_LOW,
            'freq_high': FREQ_HIGH,
            'freq_low': FREQ_LOW,
            'n_samples': N_SAMPLES,
            'buffer_size': BUFFER,
            'bootstrap_quantiles': BOOTSTRAP_QUANTILES.tolist(),
            'note': f'Bootstrap quantiles (confidence levels) calculated from {N_SAMPLES} synthetic realizations of {FREQ_HIGH} values > {WET_VALUE_HIGH} mm. Edge buffer of {BUFFER} pixels masked with NaN.'
        }
    )
    
    # Save
    output_file = f'{PATH_OUT}/{WRUN_FUTURE}/synthetic_future_{FREQ_HIGH}_from_{FREQ_LOW}_confidence.nc'
    print(f"\n11. Saving to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds_output.to_netcdf(output_file)
    
    # Clean up checkpoint directory
    print(f"\n12. Cleaning up checkpoint tiles...")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print(f"Complete! Total time: {total_time/60:.2f} minutes")
    print("="*60)

if __name__ == "__main__":
    main()
