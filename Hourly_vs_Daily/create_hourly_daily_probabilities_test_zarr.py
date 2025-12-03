import os 
import time
import numpy as np
import xarray as xr
import dask

# Configure dask
dask.config.set(scheduler='threads', num_workers=64)

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
FREQ_HIGH = '1H'      # Options: '10MIN', '1H', '3H', '6H', '12H'
FREQ_LOW = 'D'        # Options: '1H', '3H', '6H', '12H', 'D' (daily)

# Bins for each frequency (will be used for both axes)
# Adjust these based on your frequency choice
BINS_HIGH = np.arange(0, 101, 1)  # For hourly: 0-100mm in 1mm steps
BINS_LOW = np.arange(0, 105, 5)  # For daily: 0-100mm in 5mm steps

# Add infinity as the last bin edge to catch all values above max
BINS_HIGH = np.append(BINS_HIGH, np.inf)
BINS_LOW = np.append(BINS_LOW, np.inf)

tile_size = 50

#####################################################################
# FREQUENCY CONVERSION MAPPING
#####################################################################

# Map frequency strings to number of 10-min intervals
FREQ_TO_10MIN = {
    '10MIN': 1,
    '1H': 6,
    '3H': 18,
    '6H': 36,
    '12H': 72,
    'D': 144  # 24 hours
}

# Validate frequency choices
if FREQ_HIGH not in FREQ_TO_10MIN:
    raise ValueError(f"FREQ_HIGH must be one of {list(FREQ_TO_10MIN.keys())}")
if FREQ_LOW not in FREQ_TO_10MIN:
    raise ValueError(f"FREQ_LOW must be one of {list(FREQ_TO_10MIN.keys())}")

intervals_high = FREQ_TO_10MIN[FREQ_HIGH]
intervals_low = FREQ_TO_10MIN[FREQ_LOW]

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
    # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    cumsum = np.cumsum(sorted_x)
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1.0) / n
    
    return gini

#####################################################################
print("="*60)
print(f"Creating Conditional Probability Histograms - P({FREQ_HIGH} | {FREQ_LOW})")
print("="*60)

ny = '005'
nx = '011'
tile_id = f"{ny}y-{nx}x"

# Define tile bounds
y_start = int(ny) * tile_size
y_end = y_start + tile_size
x_start = int(nx) * tile_size
x_end = x_start + tile_size

# Zarr file path
zarr_path = f'{PATH_IN}/{WRUN}/UIB_10MIN_RAIN.zarr'

print(f"\nProcessing tile: {tile_id}")
print(f"Tile bounds: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
print(f"High-freq bins ({FREQ_HIGH}): {len(BINS_HIGH)-1} bins")
print(f"Low-freq bins ({FREQ_LOW}): {len(BINS_LOW)-1} bins")
print(f"Max wet timesteps per {FREQ_LOW} period: {repeats}")
print(f"Wet threshold ({FREQ_HIGH}): >= {WET_VALUE_HIFREQ} mm")
print(f"Wet threshold ({FREQ_LOW}): >= {WET_VALUE_LOFREQ} mm")
print(f"Aggregation: {intervals_high} x 10min -> {FREQ_HIGH}, {intervals_low} x 10min -> {FREQ_LOW}")

# Open and extract tile
print("\n1. Loading data...")
t0 = time.time()
fin_hf = xr.open_zarr(zarr_path, consolidated=True)
fin_hf = fin_hf.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
fin_hf_loaded = fin_hf.load()
t1 = time.time()
print(f"   Loaded in {t1-t0:.2f}s")
print(f"   Shape: {fin_hf_loaded.RAIN.shape}")

# Resample to both frequencies
print(f"\n2. Resampling to {FREQ_HIGH} and {FREQ_LOW}...")
t0 = time.time()

# Get 10-min data
rain_10min = fin_hf_loaded.RAIN.values

# Calculate number of complete intervals for low frequency
n_intervals_low = len(fin_hf_loaded.time) // intervals_low
n_timesteps_use = n_intervals_low * intervals_low
rain_10min_trimmed = rain_10min[:n_timesteps_use]

# Resample to high frequency
n_intervals_high = n_timesteps_use // intervals_high
rain_high = rain_10min_trimmed.reshape(n_intervals_high, intervals_high, tile_size, tile_size).sum(axis=1)

# Resample to low frequency
rain_low = rain_10min_trimmed.reshape(n_intervals_low, intervals_low, tile_size, tile_size).sum(axis=1)

# Also keep high-freq data in low-freq blocks for counting wet timesteps
# Reshape to (n_intervals_low, repeats, tile_size, tile_size)
rain_high_blocks = rain_high.reshape(n_intervals_low, repeats, tile_size, tile_size)

t1 = time.time()
print(f"   Resampled in {t1-t0:.2f}s")
print(f"   {FREQ_HIGH} shape: {rain_high.shape}")
print(f"   {FREQ_LOW} shape: {rain_low.shape}")
print(f"   {FREQ_HIGH} blocks shape: {rain_high_blocks.shape}")

# Create 2D histogram for each grid point
print(f"\n3. Creating conditional probabilities and Gini coefficients...")
t0 = time.time()

# Initialize arrays to store histograms
nbins_high = len(BINS_HIGH) - 1
nbins_low = len(BINS_LOW) - 1

# P(intensity_high | intensity_low) - dimensions: (nbins_high, nbins_low, y, x)
hist_2d_intensity = np.zeros((nbins_high, nbins_low, tile_size, tile_size), dtype=np.float32)

# P(n_wet_timesteps | intensity_low) - dimensions: (repeats, nbins_low, y, x)
hist_2d_n_wet = np.zeros((repeats, nbins_low, tile_size, tile_size), dtype=np.float32)

# Mean Gini coefficient for each low-freq bin - dimensions: (nbins_low, y, x)
gini_by_bin = np.full((nbins_low, tile_size, tile_size), np.nan, dtype=np.float32)

# Number of events in each low-freq bin - dimensions: (nbins_low, y, x)
n_events_by_bin = np.zeros((nbins_low, tile_size, tile_size), dtype=np.int32)

# For each grid point, create histograms
for iy in range(tile_size):
    for ix in range(tile_size):
        # Get timeseries for this grid point
        rain_high_point = rain_high[:, iy, ix]
        rain_low_point = np.repeat(rain_low[:, iy, ix], repeats)
        
        # Also get the blocked high-freq data for counting wet timesteps
        rain_high_blocks_point = rain_high_blocks[:, :, iy, ix]  # (n_intervals_low, repeats)
        rain_low_blocks_point = rain_low[:, iy, ix]  # (n_intervals_low,)
        
        # Trim to same length
        min_len = min(len(rain_high_point), len(rain_low_point))
        rain_high_point = rain_high_point[:min_len]
        rain_low_point = rain_low_point[:min_len]
        
        # *** KEY FIX: Filter to keep only WET HOURS in WET DAYS ***
        wet_mask_daily = rain_low_point >= WET_VALUE_LOFREQ  # Wet days
        wet_mask_hourly = rain_high_point >= WET_VALUE_HIFREQ  # Wet hours
        
        # Combine masks: keep only wet hours in wet days
        wet_mask = wet_mask_daily & wet_mask_hourly
        
        rain_high_wet = rain_high_point[wet_mask]
        rain_low_wet = rain_low_point[wet_mask]
        
        # For n_wet histogram, filter only by wet days (keep all hours to count them)
        wet_mask_blocks = rain_low_blocks_point >= WET_VALUE_LOFREQ
        rain_high_blocks_wet = rain_high_blocks_point[wet_mask_blocks, :]
        rain_low_blocks_wet = rain_low_blocks_point[wet_mask_blocks]
        
        # Track statistics for first grid point
        if iy == 0 and ix == 0:
            print(f"\n   DIAGNOSTICS for grid point [0,0]:")
            print(f"   Total {FREQ_HIGH} timesteps: {len(rain_high_point)}")
            print(f"   Wet {FREQ_LOW} periods: {wet_mask_daily.sum()} ({100*wet_mask_daily.sum()/len(rain_high_point):.1f}%)")
            print(f"   Wet {FREQ_HIGH} timesteps: {wet_mask_hourly.sum()} ({100*wet_mask_hourly.sum()/len(rain_high_point):.1f}%)")
            print(f"   Wet {FREQ_HIGH} in wet {FREQ_LOW}: {wet_mask.sum()} ({100*wet_mask.sum()/len(rain_high_point):.1f}%)")
        
        ####################################################################
        # HISTOGRAM 1: P(intensity_high | intensity_low)
        # Only wet hours in wet days
        ####################################################################
        if len(rain_high_wet) > 0:
            hist_counts_intensity, _, _ = np.histogram2d(
                rain_high_wet,
                rain_low_wet,
                bins=[BINS_HIGH, BINS_LOW]
            )
            
            # Normalize to conditional probability: P(high | low)
            for j in range(nbins_low):
                col_sum = hist_counts_intensity[:, j].sum()
                if col_sum > 0:
                    hist_2d_intensity[:, j, iy, ix] = hist_counts_intensity[:, j] / col_sum
        
        ####################################################################
        # HISTOGRAM 2: P(n_wet_timesteps | intensity_low)
        # For wet days, count how many hours are wet
        ####################################################################
        if len(rain_low_blocks_wet) > 0:
            # For each wet low-freq period, count how many high-freq timesteps are wet
            n_wet_timesteps = (rain_high_blocks_wet >= WET_VALUE_HIFREQ).sum(axis=1)  # Count per period
            
            # Create 2D histogram: (n_wet_timesteps, low_freq_intensity)
            # n_wet_timesteps ranges from 1 to repeats
            hist_counts_n_wet, _, _ = np.histogram2d(
                n_wet_timesteps,
                rain_low_blocks_wet,
                bins=[np.arange(0.5, repeats + 1.5, 1), BINS_LOW]  # Bins centered on 1, 2, ..., repeats
            )
            
            # Normalize to conditional probability: P(n_wet | low)
            for j in range(nbins_low):
                col_sum = hist_counts_n_wet[:, j].sum()
                if col_sum > 0:
                    hist_2d_n_wet[:, j, iy, ix] = hist_counts_n_wet[:, j] / col_sum
            
            # Diagnostic for first grid point
            if iy == 0 and ix == 0:
                print(f"   N_wet timesteps min/max: {n_wet_timesteps.min()} / {n_wet_timesteps.max()}")
                print(f"   Non-zero LOW-FREQ bins (intensity): {(hist_counts_intensity.sum(axis=0) > 0).sum()}")
                print(f"   Non-zero LOW-FREQ bins (n_wet): {(hist_counts_n_wet.sum(axis=0) > 0).sum()}")
        
        ####################################################################
        # GINI COEFFICIENT & N_EVENTS: Mean Gini per day for each low-freq bin
        # Calculate Gini for EACH wet day, then average by bin
        # Also count number of events in each bin
        ####################################################################
        if len(rain_low_blocks_wet) > 0:
            # Calculate Gini coefficient for each wet day
            gini_per_day = np.array([gini_coefficient(rain_high_blocks_wet[i, :]) 
                                     for i in range(len(rain_low_blocks_wet))])
            
            # For each low-freq bin, average the Gini values and count events
            for j in range(nbins_low):
                # Find wet days that fall in this bin
                in_bin = (rain_low_blocks_wet >= BINS_LOW[j]) & (rain_low_blocks_wet < BINS_LOW[j+1])
                
                n_in_bin = in_bin.sum()
                if n_in_bin > 0:
                    # Average Gini coefficients for days in this bin
                    gini_by_bin[j, iy, ix] = np.nanmean(gini_per_day[in_bin])
                    
                    # Count number of events in this bin
                    n_events_by_bin[j, iy, ix] = n_in_bin
        
        # Diagnostic for first grid point
        if iy == 0 and ix == 0:
            print(f"   Mean Gini coefficients and N events by bin (first non-empty bins):")
            for j in range(nbins_low):
                if not np.isnan(gini_by_bin[j, iy, ix]):
                    print(f"      Bin {j} [{BINS_LOW[j]:.0f}-{BINS_LOW[j+1]:.0f} mm]: Mean Gini = {gini_by_bin[j, iy, ix]:.4f}, N events = {n_events_by_bin[j, iy, ix]}")
                    if j >= 4:  # Show first 5 non-empty bins
                        break

t1 = time.time()
print(f"\n   Histograms created in {t1-t0:.2f}s")

# Create bin labels for coordinates
bin_centers_high = (BINS_HIGH[:-2] + BINS_HIGH[1:-1]) / 2
bin_centers_high = np.append(bin_centers_high, BINS_HIGH[-2] + 5)

bin_centers_low = (BINS_LOW[:-2] + BINS_LOW[1:-1]) / 2
bin_centers_low = np.append(bin_centers_low, BINS_LOW[-2] + 5)

# For n_wet, use actual counts: 1, 2, 3, ..., repeats
n_wet_values = np.arange(1, repeats + 1)

# Save to netCDF
print("\n4. Saving histograms...")
t0 = time.time()

# Create xarray dataset with y, x as LAST dimensions
ds_hist = xr.Dataset(
    {
        'cond_prob_intensity': ([f'bin_{FREQ_HIGH}', f'bin_{FREQ_LOW}', 'y', 'x'], hist_2d_intensity),
        'cond_prob_n_wet': (['n_wet_timesteps', f'bin_{FREQ_LOW}', 'y', 'x'], hist_2d_n_wet),
        'gini_coefficient': ([f'bin_{FREQ_LOW}', 'y', 'x'], gini_by_bin),
        'n_events': ([f'bin_{FREQ_LOW}', 'y', 'x'], n_events_by_bin)
    },
    coords={
        'y': fin_hf_loaded.y.values,
        'x': fin_hf_loaded.x.values,
        f'bin_{FREQ_HIGH}': bin_centers_high,
        f'bin_{FREQ_LOW}': bin_centers_low,
        'n_wet_timesteps': n_wet_values
    },
    attrs={
        'description': f'Conditional probabilities for {FREQ_HIGH} given {FREQ_LOW} precipitation (wet periods only)',
        'tile_id': tile_id,
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

# Add variable attributes
ds_hist['cond_prob_intensity'].attrs = {
    'long_name': f'P({FREQ_HIGH} intensity | {FREQ_LOW} intensity)',
    'description': f'Conditional probability of wet {FREQ_HIGH} rainfall intensity given {FREQ_LOW} rainfall intensity (only wet hours in wet days)',
    'units': 'probability (0-1)'
}

ds_hist['cond_prob_n_wet'].attrs = {
    'long_name': f'P(number of wet {FREQ_HIGH} timesteps | {FREQ_LOW} intensity)',
    'description': f'Conditional probability of having N wet {FREQ_HIGH} timesteps (>= {WET_VALUE_HIFREQ} mm) given {FREQ_LOW} rainfall intensity',
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
    'description': f'Number of wet {FREQ_LOW} periods (>= {WET_VALUE_LOFREQ} mm) in each {FREQ_LOW} rainfall bin',
    'units': 'count'
}

output_file = f'{PATH_OUT}/{WRUN}/histograms/condprob_{FREQ_HIGH}_given_{FREQ_LOW}_{tile_id}.nc'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
ds_hist.to_netcdf(output_file)

t1 = time.time()
print(f"   Saved to: {output_file}")
print(f"   Save time: {t1-t0:.2f}s")

print("\n" + "="*60)
print("Complete!")
print("="*60)