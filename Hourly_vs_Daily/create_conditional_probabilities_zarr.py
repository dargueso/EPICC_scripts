import os 
import time
import numpy as np
import xarray as xr
import dask
from multiprocessing import Pool, cpu_count

# Configure dask
dask.config.set(scheduler='threads', num_workers=8)  # Reduced since we're using multiprocessing

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN = "EPICC_2km_ERA5"

# Wet thresholds - values below these are considered "dry" and excluded
WET_VALUE_HIFREQ = 0.1  # mm (for high frequency)
WET_VALUE_LOFREQ = 1.0  # mm (for low frequency)

#####################################################################
# CONFIGURATION - Change these for different frequency comparisons
#####################################################################

# Frequency pair to compare
FREQ_HIGH = '01H'      # Options: '10MIN', '01H'
FREQ_LOW = 'DAY'        # Options: '01H', 'DAY' (daily)

# Bins for each frequency (will be used for both axes)
BINS_HIGH = np.arange(0, 100, 1)  # For hourly: 0-100mm in 1mm steps
BINS_LOW = np.arange(0, 100, 5)  # For daily: 0-100mm in 5mm steps

# Add infinity as the last bin edge to catch all values above max
BINS_HIGH = np.append(BINS_HIGH, np.inf)
BINS_LOW = np.append(BINS_LOW, np.inf)

tile_size = 50

# Number of parallel processes to use
N_PROCESSES = 32  # Adjust based on your system

#####################################################################
# FREQUENCY CONVERSION MAPPING
#####################################################################

# Map frequency strings to number of 10-min intervals
# {
#     '10MIN': 1,
#     '1H': 6,
#     '3H': 18,
#     '6H': 36,
#     '12H': 72,
#     'D': 144  # 24 hours
# }

if FREQ_HIGH == '10MIN':
    FREQ_TO_ORIG = {
        '10MIN': 1,
        '01H': 6,
        'DAY': 144  # 24 hours
    }
elif FREQ_HIGH == '01H':
    FREQ_TO_ORIG = {
        '01H': 1,
        'DAY': 24  # 24 hours
    }




# Validate frequency choices
if FREQ_HIGH not in FREQ_TO_ORIG:
    raise ValueError(f"FREQ_HIGH must be one of {list(FREQ_TO_ORIG.keys())}")
if FREQ_LOW not in FREQ_TO_ORIG:
    raise ValueError(f"FREQ_LOW must be one of {list(FREQ_TO_ORIG.keys())}")

intervals_high = FREQ_TO_ORIG[FREQ_HIGH]
intervals_low = FREQ_TO_ORIG[FREQ_LOW]

if intervals_high >= intervals_low:
    raise ValueError(f"FREQ_HIGH ({FREQ_HIGH}) must be finer than FREQ_LOW ({FREQ_LOW})")

# Calculate how many high-freq intervals per low-freq interval
repeats = intervals_low // intervals_high

#####################################################################
# GINI COEFFICIENT FUNCTION
#####################################################################

def gini_coefficient(x):
    """
    Calculate Gini coefficient for an array.
    
    Parameters:
    -----------
    x : array-like
        Values to calculate Gini coefficient for (including zeros)
        
    Returns:
    --------
    float
        Gini coefficient (0 = perfect equality, 1 = maximum inequality)
    """
    # Remove any NaN values
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return np.nan
    
    # Handle all zeros case
    if np.all(x == 0):
        return 0.0
    
    # Sort values
    sorted_x = np.sort(x)
    n = len(sorted_x)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_x)
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1.0) / n
    
    return gini

#####################################################################
# TILE PROCESSING FUNCTION
#####################################################################

def process_tile(tile_info):
    """
    Process a single tile and return histograms.
    
    Parameters:
    -----------
    tile_info : tuple
        (iy_tile, ix_tile, zarr_path, tile_size, config)
    
    Returns:
    --------
    tuple
        (iy_tile, ix_tile, hist_2d_intensity, hist_2d_n_wet, gini_by_bin, n_events_by_bin)
    """
    iy_tile, ix_tile, zarr_path, tile_size, config = tile_info
    
    # Unpack config
    BINS_HIGH = config['BINS_HIGH']
    BINS_LOW = config['BINS_LOW']
    WET_VALUE_HIFREQ = config['WET_VALUE_HIFREQ']
    WET_VALUE_LOFREQ = config['WET_VALUE_LOFREQ']
    intervals_high = config['intervals_high']
    intervals_low = config['intervals_low']
    repeats = config['repeats']
    ny_full = config['ny_full']
    nx_full = config['nx_full']
    
    # Define tile bounds
    y_start = iy_tile * tile_size
    y_end = min(y_start + tile_size, ny_full)  # Don't exceed domain
    x_start = ix_tile * tile_size
    x_end = min(x_start + tile_size, nx_full)  # Don't exceed domain
    
    # Get actual tile dimensions (may be smaller than tile_size at edges)
    actual_ny = y_end - y_start
    actual_nx = x_end - x_start
    
    tile_id = f"{iy_tile:03d}y-{ix_tile:03d}x"
    
    try:
        # Open zarr and extract tile
        ds = xr.open_zarr(zarr_path, consolidated=True)
        ds_tile = ds.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
        ds_tile_loaded = ds_tile.load()
        
        # Get orig data
        rain_orig = ds_tile_loaded.RAIN.values
        
        # Calculate number of complete intervals for low frequency
        n_intervals_low = len(ds_tile_loaded.time) // intervals_low
        n_timesteps_use = n_intervals_low * intervals_low
        rain_orig_trimmed = rain_orig[:n_timesteps_use]
        #Set all values below wet threshold to zero
        rain_orig_trimmed[rain_orig_trimmed < WET_VALUE_HIFREQ] = 0.0

        # Resample to high frequency
        n_intervals_high = n_timesteps_use // intervals_high
        rain_high = rain_orig_trimmed.reshape(n_intervals_high, intervals_high, actual_ny, actual_nx).sum(axis=1)


        # Resample to low frequency
        rain_low = rain_orig_trimmed.reshape(n_intervals_low, intervals_low, actual_ny, actual_nx).sum(axis=1)
        
        # Keep high-freq data in low-freq blocks
        rain_high_blocks = rain_high.reshape(n_intervals_low, repeats, actual_ny, actual_nx)
        
        # Initialize arrays
        nbins_high = len(BINS_HIGH) - 1
        nbins_low = len(BINS_LOW) - 1
        
        hist_2d_intensity = np.zeros((nbins_high, nbins_low, actual_ny, actual_nx), dtype=np.float32)
        hist_2d_n_wet = np.zeros((repeats, nbins_low, actual_ny, actual_nx), dtype=np.float32)
        gini_by_bin = np.full((nbins_low, actual_ny, actual_nx), np.nan, dtype=np.float32)
        n_events_by_bin = np.zeros((nbins_low, actual_ny, actual_nx), dtype=np.int32)
        
        # Process each grid point in tile
        for iy in range(actual_ny):
            for ix in range(actual_nx):
                # Get timeseries for this grid point
                rain_high_point = rain_high[:, iy, ix]
                rain_low_point = np.repeat(rain_low[:, iy, ix], repeats)
                
                rain_high_blocks_point = rain_high_blocks[:, :, iy, ix]
                rain_low_blocks_point = rain_low[:, iy, ix]
                
                # Trim to same length
                min_len = min(len(rain_high_point), len(rain_low_point))
                rain_high_point = rain_high_point[:min_len]
                rain_low_point = rain_low_point[:min_len]
                
                # Filter to keep only WET HOURS in WET DAYS
                wet_mask_daily = rain_low_point >= WET_VALUE_LOFREQ
                wet_mask_hourly = rain_high_point >= WET_VALUE_HIFREQ
                wet_mask = wet_mask_daily & wet_mask_hourly
                
                rain_high_wet = rain_high_point[wet_mask]
                rain_low_wet = rain_low_point[wet_mask]
                
                # For n_wet histogram
                wet_mask_blocks = rain_low_blocks_point > WET_VALUE_LOFREQ
                rain_high_blocks_wet = rain_high_blocks_point[wet_mask_blocks, :]
                rain_low_blocks_wet = rain_low_blocks_point[wet_mask_blocks]
                
                ####################################################################
                # HISTOGRAM 1: P(intensity_high | intensity_low)
                ####################################################################
                if len(rain_high_wet) > 0:
                    hist_counts_intensity, _, _ = np.histogram2d(
                        rain_high_wet,
                        rain_low_wet,
                        bins=[BINS_HIGH, BINS_LOW]
                    )
                    
                    for j in range(nbins_low):
                        col_sum = hist_counts_intensity[:, j].sum()
                        if col_sum > 0:
                            hist_2d_intensity[:, j, iy, ix] = hist_counts_intensity[:, j] / col_sum
                
                ####################################################################
                # HISTOGRAM 2: P(n_wet_timesteps | intensity_low)
                ####################################################################
                if len(rain_low_blocks_wet) > 0:
                    n_wet_timesteps = (rain_high_blocks_wet > WET_VALUE_HIFREQ).sum(axis=1)
                    
                    hist_counts_n_wet, _, _ = np.histogram2d(
                        n_wet_timesteps,
                        rain_low_blocks_wet,
                        bins=[np.arange(0.5, repeats + 1.5, 1), BINS_LOW]
                    )
                    
                    for j in range(nbins_low):
                        col_sum = hist_counts_n_wet[:, j].sum()
                        if col_sum > 0:
                            hist_2d_n_wet[:, j, iy, ix] = hist_counts_n_wet[:, j] / col_sum
                
                ####################################################################
                # GINI COEFFICIENT & N_EVENTS
                ####################################################################
                if len(rain_low_blocks_wet) > 0:
                    gini_per_day = np.array([gini_coefficient(rain_high_blocks_wet[i, :]) 
                                             for i in range(len(rain_low_blocks_wet))])
                    
                    for j in range(nbins_low):
                        in_bin = (rain_low_blocks_wet >= BINS_LOW[j]) & (rain_low_blocks_wet < BINS_LOW[j+1])
                        n_in_bin = in_bin.sum()
                        
                        if n_in_bin > 0:
                            gini_by_bin[j, iy, ix] = np.nanmean(gini_per_day[in_bin])
                            n_events_by_bin[j, iy, ix] = n_in_bin
        
        return (iy_tile, ix_tile, hist_2d_intensity, hist_2d_n_wet, gini_by_bin, n_events_by_bin)
    
    except Exception as e:
        print(f"Error processing tile {tile_id}: {e}")
        return None

#####################################################################
print("="*60)
print(f"Creating Conditional Probability Histograms - P({FREQ_HIGH} | {FREQ_LOW})")
print("="*60)

# Zarr file path
zarr_path = f'{PATH_IN}/{WRUN}/UIB_{FREQ_HIGH}_RAIN.zarr'

print(f"\nConfiguration:")
print(f"High-freq bins ({FREQ_HIGH}): {len(BINS_HIGH)-1} bins")
print(f"Low-freq bins ({FREQ_LOW}): {len(BINS_LOW)-1} bins")
print(f"Max wet timesteps per {FREQ_LOW} period: {repeats}")
print(f"Wet threshold ({FREQ_HIGH}): >= {WET_VALUE_HIFREQ} mm")
print(f"Wet threshold ({FREQ_LOW}): >= {WET_VALUE_LOFREQ} mm")
print(f"Aggregation: {intervals_high} x {FREQ_HIGH} -> {FREQ_HIGH}, {intervals_low} x {FREQ_HIGH} -> {FREQ_LOW}")
print(f"Parallel processes: {N_PROCESSES}")
# Open zarr once to get dimensions
print("\n1. Opening Zarr dataset...")
t0 = time.time()
ds_full = xr.open_zarr(zarr_path, consolidated=True)
t1 = time.time()
print(f"   Opened in {t1-t0:.2f}s")
print(f"   Full domain shape: {ds_full.RAIN.shape}")
print(f"   Dimensions: y={len(ds_full.y)}, x={len(ds_full.x)}")

# Calculate number of tiles (including partial tiles at edges)
ny_tiles = int(np.ceil(len(ds_full.y) / tile_size))
nx_tiles = int(np.ceil(len(ds_full.x) / tile_size))
total_tiles = ny_tiles * nx_tiles

print(f"\n2. Processing {total_tiles} tiles ({ny_tiles} x {nx_tiles})...")

# Initialize arrays for full domain
nbins_high = len(BINS_HIGH) - 1
nbins_low = len(BINS_LOW) - 1

hist_2d_intensity_full = np.zeros((nbins_high, nbins_low, len(ds_full.y), len(ds_full.x)), dtype=np.float32)
hist_2d_n_wet_full = np.zeros((repeats, nbins_low, len(ds_full.y), len(ds_full.x)), dtype=np.float32)
gini_by_bin_full = np.full((nbins_low, len(ds_full.y), len(ds_full.x)), np.nan, dtype=np.float32)
n_events_by_bin_full = np.zeros((nbins_low, len(ds_full.y), len(ds_full.x)), dtype=np.int32)

# Create list of tile tasks
config = {
    'BINS_HIGH': BINS_HIGH,
    'BINS_LOW': BINS_LOW,
    'WET_VALUE_HIFREQ': WET_VALUE_HIFREQ,
    'WET_VALUE_LOFREQ': WET_VALUE_LOFREQ,
    'intervals_high': intervals_high,
    'intervals_low': intervals_low,
    'repeats': repeats,
    'ny_full': len(ds_full.y),
    'nx_full': len(ds_full.x)
}

tile_tasks = []
for iy_tile in range(ny_tiles):
    for ix_tile in range(nx_tiles):
        tile_tasks.append((iy_tile, ix_tile, zarr_path, tile_size, config))

# Process tiles in parallel
total_start = time.time()

print(f"   Starting parallel processing with {N_PROCESSES} workers...")
with Pool(processes=N_PROCESSES) as pool:
    results = pool.map(process_tile, tile_tasks)

# Combine results
print("\n3. Combining tile results...")
processed_count = 0
for result in results:
    if result is not None:
        iy_tile, ix_tile, hist_intensity, hist_n_wet, gini, n_events = result
        
        y_start = iy_tile * tile_size
        y_end = y_start + tile_size
        x_start = ix_tile * tile_size
        x_end = x_start + tile_size
        
        hist_2d_intensity_full[:, :, y_start:y_end, x_start:x_end] = hist_intensity
        hist_2d_n_wet_full[:, :, y_start:y_end, x_start:x_end] = hist_n_wet
        gini_by_bin_full[:, y_start:y_end, x_start:x_end] = gini
        n_events_by_bin_full[:, y_start:y_end, x_start:x_end] = n_events
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"   Combined {processed_count}/{total_tiles} tiles")

total_time = time.time() - total_start
print(f"\n   All tiles processed in {total_time/60:.2f} minutes")
print(f"   Successfully processed: {processed_count}/{total_tiles} tiles")

# Create bin labels for coordinates
bin_centers_high = (BINS_HIGH[:-2] + BINS_HIGH[1:-1]) / 2
bin_centers_high = np.append(bin_centers_high, BINS_HIGH[-2] + 5)

bin_centers_low = (BINS_LOW[:-2] + BINS_LOW[1:-1]) / 2
bin_centers_low = np.append(bin_centers_low, BINS_LOW[-2] + 5)

n_wet_values = np.arange(1, repeats + 1)

# Save to netCDF
print("\n4. Saving to NetCDF...")
t0 = time.time()

ds_hist = xr.Dataset(
    {
        'cond_prob_intensity': ([f'bin_{FREQ_HIGH}', f'bin_{FREQ_LOW}', 'y', 'x'], hist_2d_intensity_full),
        'cond_prob_n_wet': (['n_wet_timesteps', f'bin_{FREQ_LOW}', 'y', 'x'], hist_2d_n_wet_full),
        'gini_coefficient': ([f'bin_{FREQ_LOW}', 'y', 'x'], gini_by_bin_full),
        'n_events': ([f'bin_{FREQ_LOW}', 'y', 'x'], n_events_by_bin_full)
    },
    coords={
        'y': ds_full.y.values,
        'x': ds_full.x.values,
        f'bin_{FREQ_HIGH}': bin_centers_high,
        f'bin_{FREQ_LOW}': bin_centers_low,
        'n_wet_timesteps': n_wet_values
    },
    attrs={
        'description': f'Conditional probabilities for {FREQ_HIGH} given {FREQ_LOW} precipitation (wet periods only)',
        'freq_high': FREQ_HIGH,
        'freq_low': FREQ_LOW,
        'intervals_high': intervals_high,
        'intervals_low': intervals_low,
        'repeats': repeats,
        'wet_threshold_high': WET_VALUE_HIFREQ,
        'wet_threshold_low': WET_VALUE_LOFREQ,
        f'bin_edges_{FREQ_HIGH}': BINS_HIGH[:-1].tolist(),
        f'bin_edges_{FREQ_LOW}': BINS_LOW[:-1].tolist(),
        'note': f'Last bin includes all values > {BINS_HIGH[-2]:.0f} mm for {FREQ_HIGH} and > {BINS_LOW[-2]:.0f} mm for {FREQ_LOW}',
        'normalization': f'Both probability variables are conditional probabilities that sum to 1.0 for each {FREQ_LOW} bin'
    }
)

ds_hist['cond_prob_intensity'].attrs = {
    'long_name': f'P({FREQ_HIGH} intensity | {FREQ_LOW} intensity)',
    'description': f'Conditional probability of wet {FREQ_HIGH} rainfall intensity given {FREQ_LOW} rainfall intensity (only wet hours in wet days)',
    'units': 'probability (0-1)'
}

ds_hist['cond_prob_n_wet'].attrs = {
    'long_name': f'P(number of wet {FREQ_HIGH} timesteps | {FREQ_LOW} intensity)',
    'description': f'Conditional probability of having N wet {FREQ_HIGH} timesteps ( >= {WET_VALUE_HIFREQ} mm) given {FREQ_LOW} rainfall intensity',
    'units': 'probability (0-1)',
    'note': f'n_wet_timesteps ranges from 1 to {repeats}'
}

ds_hist['gini_coefficient'].attrs = {
    'long_name': f'Mean Gini coefficient of {FREQ_HIGH} rainfall distribution',
    'description': f'Mean Gini coefficient measuring inequality in {FREQ_HIGH} rainfall distribution for each {FREQ_LOW} rainfall bin. Calculated as the mean of per-period Gini coefficients (0 = perfect equality, 1 = maximum inequality)',
    'units': 'dimensionless (0-1)',
    'note': f'Gini calculated per wet {FREQ_LOW} period using all {FREQ_HIGH} timesteps (including dry hours), then averaged across all periods in each bin'
}

ds_hist['n_events'].attrs = {
    'long_name': f'Number of wet {FREQ_LOW} events',
    'description': f'Number of wet {FREQ_LOW} periods ( >= {WET_VALUE_LOFREQ} mm) in each {FREQ_LOW} rainfall bin',
    'units': 'count'
}

output_file = f'{PATH_OUT}/{WRUN}/condprob_{FREQ_HIGH}_given_{FREQ_LOW}_full_domain.nc'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
ds_hist.to_netcdf(output_file)

t1 = time.time()
print(f"   Saved to: {output_file}")
print(f"   Save time: {t1-t0:.2f}s")

print("\n" + "="*60)
print("Complete!")
print(f"Total processing time: {(time.time() - total_start)/60:.2f} minutes")
print("="*60)
