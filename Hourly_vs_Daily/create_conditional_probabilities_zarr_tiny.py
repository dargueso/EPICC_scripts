import os
import time
import numpy as np
import xarray as xr
import dask
import zarr

PATH_IN = "/home/dargueso/postprocessed/EPICC/"
PATH_OUT = "/home/dargueso/postprocessed/EPICC/"
WRUN = "EPICC_2km_ERA5"
test_suffix = "_test_21x21"


# Wet thresholds - values below these are considered "dry" and excluded
WET_VALUE_HIFREQ = 0.1  # mm (for high frequency)
WET_VALUE_LOFREQ = 1.0  # mm (for low frequency)

# Frequency pair to compare
FREQ_HIGH = "01H"  # Options: '10MIN', '01H'
FREQ_LOW = "DAY"  # Options: '01H', 'DAY' (daily)
intervals_high = 1  # Number of high-frequency intervals in one low-frequency interval (e.g., 1 hour in a day)
intervals_low = 24  # Number of high-frequency intervals in one low-frequency interval (e.g., 24 hours in a day)
repeats = intervals_low // intervals_high

# Bins for each frequency (will be used for both axes)
BINS_HIGH = np.arange(0, 100, 1)  # For hourly: 0-100mm in 1mm steps
BINS_LOW = np.arange(0, 100, 5)  # For daily: 0-100mm in 5mm steps

# Add infinity as the last bin edge to catch all values above max
BINS_HIGH = np.append(BINS_HIGH, np.inf)
BINS_LOW = np.append(BINS_LOW, np.inf)

nbins_high = len(BINS_HIGH) - 1
nbins_low = len(BINS_LOW) - 1

zarr_path = f"{PATH_IN}/{WRUN}/UIB_{FREQ_HIGH}_RAIN{test_suffix}.zarr"

ds = xr.open_zarr(zarr_path, consolidated=False, zarr_format=2)
rain_hour = ds.RAIN

# 1. Daily rainfall from hourly, >= 1mm only (else NaN)
daily_rain = rain_hour.resample(time="1D").sum()
daily_rain_wet = daily_rain.where(daily_rain >= WET_VALUE_LOFREQ)

# 2. Hourly rainfall for those same wet days (else NaN)
wet_days_mask = daily_rain >= WET_VALUE_LOFREQ
hourly_rain_wet_days = rain_hour.where(
    wet_days_mask.reindex(time=rain_hour.time, method="ffill")
)

# 3. Daily rainfall replicated to hourly frequency
daily_rain_wet_hourly = daily_rain_wet.reindex(time=rain_hour.time, method="ffill")

# 4. Wet hours on wet days (hourly >= 0.1mm and daily >= 1mm)
# # Mask for wet hours (>=0.1mm) within wet days (>=1mm)
wet_hours_on_wet_days_mask = (
    wet_days_mask.reindex(time=rain_hour.time, method="ffill") & 
    (rain_hour >= WET_VALUE_HIFREQ)
)

hourly_rain_wet_hours_wet_days = rain_hour.where(wet_hours_on_wet_days_mask)

rain_high_wet = hourly_rain_wet_days.values.flatten(order='F')
rain_low_wet = daily_rain_wet_hourly.values.flatten(order='F')
rain_high_wet_hwet = hourly_rain_wet_hours_wet_days.values.flatten(order='F')

rain_low_wet_clean = rain_low_wet[~np.isnan(rain_high_wet_hwet)]
rain_high_wet_hwet_clean = rain_high_wet_hwet[~np.isnan(rain_high_wet_hwet)]

####################################################################
# HISTOGRAM 1: P(intensity_high | intensity_low)
####################################################################
hist_2d_intensity = np.zeros((nbins_high, nbins_low, 1, 1), dtype=np.float32)
if len(rain_high_wet) > 0:
    hist_counts_intensity, _, _ = np.histogram2d(rain_high_wet, rain_low_wet, bins=[BINS_HIGH, BINS_LOW])
for j in range(nbins_low):
    col_sum = hist_counts_intensity[:, j].sum()
    if col_sum > 0:
        hist_2d_intensity[:, j, 0, 0] = hist_counts_intensity[:, j] / col_sum


####################################################################
# HISTOGRAM 2: P(n_wet_timesteps | intensity_low)
####################################################################

n_wet_hours=hourly_rain_wet_hours_wet_days.resample(time="1D").count().where(wet_days_mask).values.flatten(order='F')
n_wet_hours_clean = n_wet_hours[~np.isnan(daily_rain_wet.values.flatten(order='F'))]
daily_rain_wet_clean = daily_rain_wet.values.flatten(order='F')[~np.isnan(daily_rain_wet.values.flatten(order='F'))]

hist_2d_n_wet = np.zeros((repeats, nbins_low, 1, 1), dtype=np.float32)

if len(daily_rain_wet_clean) > 0:

    hist_counts_n_wet, _, _ = np.histogram2d(
        n_wet_hours_clean,
        daily_rain_wet_clean,
        bins=[np.arange(0.5, repeats + 1.5, 1), BINS_LOW]
    )
    
    for j in range(nbins_low):
        col_sum = hist_counts_n_wet[:, j].sum()
        if col_sum > 0:
            hist_2d_n_wet[:, j, 0, 0] = hist_counts_n_wet[:, j] / col_sum


#####################################################################
#####################################################################
# BUILD SYNTHETIC FUTURE FOR A SAMPLE DAY
#####################################################################
#####################################################################

# Function to sample hourly from daily
def sample_hourly_from_daily(daily_value, hist_2d_n_wet, hist_2d_intensity, BINS_LOW, BINS_HIGH, n_bootstrap=1000):
    """Sample possible hourly structures for a given daily rainfall amount."""
    bin_idx = np.digitize(daily_value, BINS_LOW) - 1
    bin_idx = min(bin_idx, nbins_low - 1)
    
    p_n_wet = hist_2d_n_wet[:, bin_idx, 0, 0]
    p_intensity = hist_2d_intensity[:, bin_idx, 0, 0]
    
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        n_wet = np.random.choice(np.arange(1, repeats + 1), p=p_n_wet / p_n_wet.sum())
        
        # Sample bin indices based on probability
        bin_indices = np.random.choice(
            nbins_high,
            size=n_wet,
            p=p_intensity / p_intensity.sum()
        )
        
        # Sample random values WITHIN each bin
        intensities = np.zeros(n_wet)
        for i, idx in enumerate(bin_indices):
            lower = BINS_HIGH[idx]
            upper = BINS_HIGH[idx + 1]
            # Handle infinity in last bin
            if np.isinf(upper):
                intensities[i] = lower + np.random.exponential(scale=10)  # Exponential tail for values above last bin
            else:
                intensities[i] = np.random.uniform(lower, upper)
        
        scale_factor = daily_value / intensities.sum()
        intensities_scaled = intensities * scale_factor
        
        hourly = np.zeros(24)
        wet_positions = np.random.choice(24, size=n_wet, replace=False)
        hourly[wet_positions] = intensities_scaled
        bootstrap_samples.append(hourly)
    
    return np.array(bootstrap_samples)



# Function to get hourly indices for a day
def get_hourly_indices_for_day(day_idx, n_wet_hours_clean):
    """Find hourly array indices for a given day."""
    cumsum = np.cumsum(n_wet_hours_clean.astype(int))
    start_idx = 0 if day_idx == 0 else cumsum[day_idx - 1]
    end_idx = cumsum[day_idx]
    return int(start_idx), int(end_idx)

# Test with random day
random_day_idx = np.random.randint(len(daily_rain_wet_clean))
test_daily_value = daily_rain_wet_clean[random_day_idx]
test_n_wet_obs = int(n_wet_hours_clean[random_day_idx])

start_idx, end_idx = get_hourly_indices_for_day(random_day_idx, n_wet_hours_clean)
test_hourly_obs = rain_high_wet_hwet_clean[start_idx:end_idx]

print(f"Day index: {random_day_idx}")
print(f"Hourly indices: {start_idx}:{end_idx}")
print(f"Daily: {test_daily_value:.1f}mm, Obs wet hours: {test_n_wet_obs}")
print(f"Obs hourly: {test_hourly_obs}, sum={test_hourly_obs.sum():.1f}mm")

# Update the call to include BINS_HIGH
bootstrap_samples = sample_hourly_from_daily(
    test_daily_value, hist_2d_n_wet, hist_2d_intensity, BINS_LOW, BINS_HIGH, n_bootstrap=10000
)

print(f"\nBootstrap n_wet: {(bootstrap_samples > 0).sum(axis=1).mean():.1f} ± {(bootstrap_samples > 0).sum(axis=1).std():.1f}")
print(f"Bootstrap max: {bootstrap_samples.max(axis=1).mean():.1f} ± {bootstrap_samples.max(axis=1).std():.1f}mm")


# Compare observed vs bootstrap using percentiles
def check_compatibility(test_hourly_obs, bootstrap_samples, alpha=0.05):
    """
    Check if observed hourly rainfall is compatible with bootstrap estimates.
    Returns percentiles where observed values fall.
    """
    # Get bootstrap statistics
    boot_n_wet = (bootstrap_samples > 0).sum(axis=1)
    boot_max = bootstrap_samples.max(axis=1)
    boot_mean_wet = [bs[bs > 0].mean() for bs in bootstrap_samples]
    
    # Observed statistics
    obs_n_wet = len(test_hourly_obs)
    obs_max = test_hourly_obs.max()
    obs_mean_wet = test_hourly_obs.mean()
    
    # Calculate percentiles
    p_n_wet = (boot_n_wet < obs_n_wet).sum() / len(boot_n_wet) * 100
    p_max = (boot_max < obs_max).sum() / len(boot_max) * 100
    p_mean = (boot_mean_wet < obs_mean_wet).sum() / len(boot_mean_wet) * 100
    
    # Get confidence intervals
    ci_low, ci_high = alpha/2 * 100, (1 - alpha/2) * 100
    
    print(f"Observed n_wet: {obs_n_wet} (percentile: {p_n_wet:.1f}%)")
    print(f"  Bootstrap CI: [{np.percentile(boot_n_wet, ci_low):.0f}, {np.percentile(boot_n_wet, ci_high):.0f}]")
    
    print(f"Observed max: {obs_max:.3f}mm (percentile: {p_max:.1f}%)")
    print(f"  Bootstrap CI: [{np.percentile(boot_max, ci_low):.1f}, {np.percentile(boot_max, ci_high):.1f}]mm")
    
    print(f"Observed mean_wet: {obs_mean_wet:.3f}mm (percentile: {p_mean:.1f}%)")
    print(f"  Bootstrap CI: [{np.percentile(boot_mean_wet, ci_low):.1f}, {np.percentile(boot_mean_wet, ci_high):.1f}]mm")
    
    # Check compatibility (within 5-95% range)
    compatible = (ci_low < p_n_wet < ci_high and 
                  ci_low < p_max < ci_high and 
                  ci_low < p_mean < ci_high)
    
    print(f"\nCompatible: {compatible}")
    return compatible

# Run check
check_compatibility(test_hourly_obs, bootstrap_samples)