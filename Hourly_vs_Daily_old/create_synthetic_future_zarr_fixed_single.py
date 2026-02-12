#!/usr/bin/env python
"""
Memory-efficient full-domain version with multiprocessing.
Calculates synthetic future high-frequency rainfall quantiles from low-frequency totals.
FIXED: Correct buffer handling at domain edges
"""
import os
import time
import numpy as np
import xarray as xr
from multiprocessing import Pool
from numba import njit
from scipy.ndimage import uniform_filter
import warnings
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
BOOTSTRAP_QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975, 0.99], dtype=np.float32)

# Processing parameters
TILE_SIZE = 50
BUFFER = 10
N_SAMPLES = 1000
N_PROCESSES = 32  # Adjust based on your system

# Single tile testing mode
SINGLE_TILE_MODE = True  # Set to True to test a single tile, False for full domain
SINGLE_TILE_Y = 5  # Tile index in y direction (0-indexed)
SINGLE_TILE_X = 11  # Tile index in x direction (0-indexed)
# For tile 005y-011x: y_range = [250-299], x_range = [550-599]

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
    '1H': 'HOUR',
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
# TILE PROCESSING FUNCTION - FIXED VERSION
# =============================================================================

def process_tile(tile_info):
    """Process a single tile with CORRECTED edge buffer handling."""
    (iy_tile, ix_tile, rain_arr, bin_idx, wet_cdf, hour_cdf, 
     tile_size, config) = tile_info
    
    # Unpack config
    BUFFER = config['buffer']
    QUANTILES = config['quantiles']
    BINS_HIGH = config['bins_high']
    WET_VALUE_HIGH = config['wet_value_high']
    n_interval = config['n_interval']
    N_SAMPLES = config['n_samples']
    ny_tiles = config['ny_tiles']
    nx_tiles = config['nx_tiles']
    
    # Full domain dimensions
    ny_full, nx_full = rain_arr.shape[1:]
    
    # Tile bounds WITHOUT buffer (what we want to output)
    y_start = iy_tile * tile_size
    y_end = min(ny_full, y_start + tile_size)
    x_start = ix_tile * tile_size
    x_end = min(nx_full, x_start + tile_size)
    
    # Tile bounds WITH buffer (what we extract for processing)
    y_start_buf = max(0, y_start - BUFFER)
    y_end_buf = min(ny_full, y_end + BUFFER)
    x_start_buf = max(0, x_start - BUFFER)
    x_end_buf = min(nx_full, x_end + BUFFER)
    
    # Extract tile with buffer
    rain_tile = rain_arr[:, y_start_buf:y_end_buf, x_start_buf:x_end_buf]
    bin_idx_tile = bin_idx[:, y_start_buf:y_end_buf, x_start_buf:x_end_buf]
    wet_cdf_tile = wet_cdf[:, :, y_start_buf:y_end_buf, x_start_buf:x_end_buf]
    hour_cdf_tile = hour_cdf[:, :, y_start_buf:y_end_buf, x_start_buf:x_end_buf]
    
    # Calculate inner region indices relative to the buffered tile
    # These are the actual data indices within the extracted tile
    iy0 = y_start - y_start_buf  # Offset from buffer start to actual tile start
    iy1 = iy0 + (y_end - y_start)  # Inner tile height
    ix0 = x_start - x_start_buf  # Offset from buffer start to actual tile start
    ix1 = ix0 + (x_end - x_start)  # Inner tile width
    
    ny_inner = iy1 - iy0
    nx_inner = ix1 - ix0
    
    n_quantiles = len(QUANTILES)
    BOOTSTRAP_QUANTILES = config['bootstrap_quantiles']
    n_bootstrap = len(BOOTSTRAP_QUANTILES)
    
    # Store all samples temporarily to compute bootstrap quantiles
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
    
    # Compute bootstrap quantiles across samples for each (quantile, y, x)
    # Result shape: (n_bootstrap, n_quantiles, ny_inner, nx_inner)
    result_quantiles = np.full((n_bootstrap, n_quantiles, ny_inner, nx_inner), 
                               np.nan, dtype=np.float32)
    
    for iq in range(n_quantiles):
        for iy in range(ny_inner):
            for ix in range(nx_inner):
                sample_values = all_samples[:, iq, iy, ix]
                # Only compute bootstrap quantiles where we have valid samples
                valid_mask = ~np.isnan(sample_values)
                if np.sum(valid_mask) > 0:
                    valid_samples = sample_values[valid_mask]
                    result_quantiles[:, iq, iy, ix] = np.quantile(valid_samples, BOOTSTRAP_QUANTILES)
    
    # CRITICAL FIX: Apply edge masking to output, not during extraction
    # Mask the outer BUFFER pixels for edge tiles
    if iy_tile == 0:  # Top edge
        result_quantiles[:, :, :BUFFER, :] = np.nan
    if iy_tile == ny_tiles - 1:  # Bottom edge
        result_quantiles[:, :, -BUFFER:, :] = np.nan
    if ix_tile == 0:  # Left edge
        result_quantiles[:, :, :, :BUFFER] = np.nan
    if ix_tile == nx_tiles - 1:  # Right edge
        result_quantiles[:, :, :, -BUFFER:] = np.nan
    
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
    
    # Load future low-frequency rainfall (flexible frequency)
    print(f"\n2. Loading future {FREQ_LOW} rainfall...")
    freq_name_low = FREQ_TO_NAME[FREQ_LOW]
    future_file = f'{PATH_IN}/{WRUN_FUTURE}/UIB_{freq_name_low}_RAIN.zarr'
    
    # Try to open, if not found, suggest correct path
    try:
        ds_future = xr.open_zarr(future_file, consolidated=True)
    except FileNotFoundError:
        print(f"   ERROR: Could not find {future_file}")
        print(f"   Please ensure your {FREQ_LOW} rainfall data is available")
        print(f"   Expected file: UIB_{freq_name_low}_RAIN.zarr")
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
    wet_dist = ds_prob.cond_prob_n_wet.values.copy()  # (n_wet_timesteps, bin_low, y, x)
    hour_dist = ds_prob.cond_prob_intensity.values.copy()  # (bin_high, bin_low, y, x)
    n_events = ds_prob.n_events.values.copy()  # (bin_low, y, x)
    
    ds_prob.close()
    ds_future.close()
    
    # Convert to proper format and handle NaNs
    wet_dist = np.nan_to_num(wet_dist, nan=0.0)
    hour_dist = np.nan_to_num(hour_dist, nan=0.0)
    n_events = np.nan_to_num(n_events, nan=0.0)
    
    # Transpose to match expected format: (bin_low, n_intervals, y, x)
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
    
    # Calculate tiles
    ny_tiles = (ny + TILE_SIZE - 1) // TILE_SIZE
    nx_tiles = (nx + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = ny_tiles * nx_tiles

    mytiles_x = np.ceil(nx / TILE_SIZE).astype(int)
    mytiles_y = np.ceil(ny / TILE_SIZE).astype(int)
    
    if SINGLE_TILE_MODE:
        print(f"\n6. SINGLE TILE MODE: Processing tile {SINGLE_TILE_Y:03d}y-{SINGLE_TILE_X:03d}x")
        y_start_tile = SINGLE_TILE_Y * TILE_SIZE
        y_end_tile = min(y_start_tile + TILE_SIZE, ny)
        x_start_tile = SINGLE_TILE_X * TILE_SIZE
        x_end_tile = min(x_start_tile + TILE_SIZE, nx)
        print(f"   Tile dimensions: {y_end_tile - y_start_tile} x {x_end_tile - x_start_tile}")
        print(f"   Grid point range: Y[{y_start_tile}-{y_end_tile-1}], X[{x_start_tile}-{x_end_tile-1}]")
        print(f"   Generating {N_SAMPLES} samples")
    else:
        print(f"\n6. Total tiles to process: {mytiles_x} x {mytiles_y} = {mytiles_x * mytiles_y}")
    
    print(f"\n6. Processing tiles...")
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
    
    # Create tile tasks
    tile_tasks = []
    if SINGLE_TILE_MODE:
        # Process only the specified tile
        tile_tasks.append((SINGLE_TILE_Y, SINGLE_TILE_X, rain_low_freq, bin_idx, 
                         wet_cdf, hour_cdf, TILE_SIZE, config))
    else:
        # Process all tiles
        for iy_tile in range(ny_tiles):
            for ix_tile in range(nx_tiles):
                tile_tasks.append((iy_tile, ix_tile, rain_low_freq, bin_idx, 
                                 wet_cdf, hour_cdf, TILE_SIZE, config))
    
    # Process tiles in parallel
    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(process_tile, tile_tasks)
    
    # Combine results
    print("\n7. Combining tile results...")
    n_quantiles = len(QUANTILES)
    n_bootstrap = len(BOOTSTRAP_QUANTILES)
    
    if SINGLE_TILE_MODE:
        # Extract only the single tile data for output
        result = results[0]
        iy_tile, ix_tile, tile_data, ny_inner, nx_inner = result
        
        y_start = iy_tile * TILE_SIZE
        y_end = y_start + ny_inner
        x_start = ix_tile * TILE_SIZE
        x_end = x_start + nx_inner
        
        # Output only the tile region
        result_full = tile_data
        lat_out = lat[y_start:y_end, x_start:x_end]
        lon_out = lon[y_start:y_end, x_start:x_end]
        
        ny_out = ny_inner
        nx_out = nx_inner
        y_coords = np.arange(y_start, y_end)
        x_coords = np.arange(x_start, x_end)
        
        extra_attrs = {
            'tile_y_index': SINGLE_TILE_Y,
            'tile_x_index': SINGLE_TILE_X,
            'tile_y_range': f'{y_start}-{y_end-1}',
            'tile_x_range': f'{x_start}-{x_end-1}'
        }
    else:
        # Full domain
        result_full = np.full((n_bootstrap, n_quantiles, ny, nx), np.nan, dtype=np.float32)
        
        for result in results:
            if result is not None:
                iy_tile, ix_tile, tile_data, ny_inner, nx_inner = result
                
                y_start = iy_tile * TILE_SIZE
                y_end = y_start + ny_inner
                x_start = ix_tile * TILE_SIZE
                x_end = x_start + nx_inner
                
                result_full[:, :, y_start:y_end, x_start:x_end] = tile_data
        
        lat_out = lat
        lon_out = lon
        ny_out = ny
        nx_out = nx
        y_coords = np.arange(ny)
        x_coords = np.arange(nx)
        extra_attrs = {}
    
    # Create output dataset
    print("\n8. Creating output dataset...")
    ds_output = xr.Dataset(
        data_vars=dict(
            precipitation=(("bootstrap_quantile", "quantile", "y", "x"), result_full),
            lat=(("y", "x"), lat_out),
            lon=(("y", "x"), lon_out),
        ),
        coords=dict(
            bootstrap_quantile=BOOTSTRAP_QUANTILES,
            quantile=QUANTILES,
            y=y_coords,
            x=x_coords,
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
            'note': f'Bootstrap quantiles (confidence levels) calculated from {N_SAMPLES} synthetic realizations of {FREQ_HIGH} values > {WET_VALUE_HIGH} mm. Edge buffer of {BUFFER} pixels masked with NaN. Use bootstrap_quantile dimension to assess uncertainty.',
            **extra_attrs
        }
    )
    
    # Save
    if SINGLE_TILE_MODE:
        output_file = f'{PATH_OUT}/{WRUN_FUTURE}/synthetic_future_{FREQ_HIGH}_from_{FREQ_LOW}_confidence_tile_{SINGLE_TILE_Y:03d}y_{SINGLE_TILE_X:03d}x.nc'
    else:
        output_file = f'{PATH_OUT}/{WRUN_FUTURE}/synthetic_future_{FREQ_HIGH}_from_{FREQ_LOW}_confidence.nc'
    
    print(f"\n9. Saving to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds_output.to_netcdf(output_file)
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    if SINGLE_TILE_MODE:
        print(f"Single tile {SINGLE_TILE_Y:03d}y-{SINGLE_TILE_X:03d}x complete!")
        print(f"Tile dimensions: {ny_out} x {nx_out}")
        print(f"Grid point range: Y[{y_coords[0]}-{y_coords[-1]}], X[{x_coords[0]}-{x_coords[-1]}]")
    else:
        print(f"Full domain complete!")
        print(f"Domain size: {ny_out} x {nx_out}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print("="*60)

if __name__ == "__main__":
    main()
