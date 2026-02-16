import os
import time
import numpy as np
import xarray as xr
import dask
import zarr
import pandas as pd

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
print(f"Original data shape: {rain_hour.shape}, time range: {rain_hour.time.min().values} to {rain_hour.time.max().values}")

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

# Check what probabilities the histogram has for low daily rainfall bins
print("Checking histogram for 1-2mm daily rainfall:")

# Find the bin index for 1.5mm (middle of 1-2mm range)
test_value = 1.5
bin_idx = np.digitize(test_value, BINS_LOW) - 1
print(f"\nDaily value: {test_value}mm falls in bin {bin_idx}")
print(f"Bin range: {BINS_LOW[bin_idx]}-{BINS_LOW[bin_idx+1]}mm")

# Show the probability distribution for n_wet_hours
p_n_wet = hist_2d_n_wet[:, bin_idx, 0, 0]
print(f"\nP(n_wet_hours | daily={BINS_LOW[bin_idx]}-{BINS_LOW[bin_idx+1]}mm):")
for n in range(1, min(10, repeats+1)):
    print(f"  {n} wet hours: {p_n_wet[n-1]:.4f} ({p_n_wet[n-1]*100:.1f}%)")

# Check actual observed data in that bin
mask = (daily_rain_wet_clean >= BINS_LOW[bin_idx]) & (daily_rain_wet_clean < BINS_LOW[bin_idx+1])
observed_n_wet_in_bin = n_wet_hours_clean[mask]
print(f"\nActual observed data for this bin ({mask.sum()} days):")
print(f"  n_wet_hours distribution:")
for n in range(1, min(10, repeats+1)):
    count = (observed_n_wet_in_bin == n).sum()
    print(f"    {n} wet hours: {count} days ({count/mask.sum()*100:.1f}%)")
    

for j in range(nbins_low):
        col_sum = hist_counts_n_wet[:, j].sum()
        if col_sum > 0:
            hist_2d_n_wet[:, j, 0, 0] = hist_counts_n_wet[:, j] / col_sum

# ADD THE DIAGNOSTIC CODE HERE
print("\nChecking P(n_wet=1) for low daily rainfall bins:")
for i in range(min(5, nbins_low)):
    p_1_wet = hist_2d_n_wet[0, i, 0, 0]  # First row is n_wet=1
    print(f"Bin {BINS_LOW[i]}-{BINS_LOW[i+1]}mm: P(n_wet=1) = {p_1_wet:.4f}")
    
# Count how many zeros in the histogram
n_zeros_n_wet = (hist_2d_n_wet == 0).sum()
n_zeros_intensity = (hist_2d_intensity == 0).sum()
print(f"\nZeros in hist_2d_n_wet: {n_zeros_n_wet}/{hist_2d_n_wet.size}")
print(f"Zeros in hist_2d_intensity: {n_zeros_intensity}/{hist_2d_intensity.size}")


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


#####################################################################
# TEST SINGLE RANDOM DAY
#####################################################################

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
    test_daily_value, hist_2d_n_wet, hist_2d_intensity, BINS_LOW, BINS_HIGH, n_bootstrap=1000
)

print(f"\nBootstrap n_wet: {(bootstrap_samples > 0).sum(axis=1).mean():.1f} ± {(bootstrap_samples > 0).sum(axis=1).std():.1f}")
print(f"Bootstrap max: {bootstrap_samples.max(axis=1).mean():.1f} ± {bootstrap_samples.max(axis=1).std():.1f}mm")


# Compare observed vs bootstrap using percentiles
# Check if observed values have reasonable probability
def check_compatibility(test_hourly_obs, bootstrap_samples, alpha=0.05):
    """
    Check if observed hourly rainfall is compatible with bootstrap estimates.
    Uses frequency-based test instead of percentile rank.
    """
    # Get bootstrap statistics
    boot_n_wet = (bootstrap_samples > 0).sum(axis=1)
    boot_max = bootstrap_samples.max(axis=1)
    boot_mean_wet = np.array([bs[bs > 0].mean() for bs in bootstrap_samples])
    
    # Observed statistics
    obs_n_wet = len(test_hourly_obs)
    obs_max = test_hourly_obs.max()
    obs_mean_wet = test_hourly_obs.mean()
    
    # Calculate FREQUENCIES (what % of bootstrap samples match observed)
    freq_n_wet = (boot_n_wet == obs_n_wet).sum() / len(boot_n_wet) * 100
    
    # For continuous values, use tolerance window
    tol = 0.2  # mm tolerance
    freq_max = ((boot_max >= obs_max - tol) & (boot_max <= obs_max + tol)).sum() / len(boot_max) * 100
    freq_mean = ((boot_mean_wet >= obs_mean_wet - tol) & (boot_mean_wet <= obs_mean_wet + tol)).sum() / len(boot_mean_wet) * 100
    
    # Compatible if ANY metric has frequency > 1% (appears in bootstrap)
    min_freq = 1.0  # at least 1% of bootstrap samples
    compatible = (freq_n_wet >= min_freq) or (freq_max >= min_freq) or (freq_mean >= min_freq)
    
    print(f"Observed n_wet: {obs_n_wet} (appears in {freq_n_wet:.1f}% of bootstrap)")
    print(f"Observed max: {obs_max:.3f}mm (appears in {freq_max:.1f}% of bootstrap)")
    print(f"Observed mean_wet: {obs_mean_wet:.3f}mm (appears in {freq_mean:.1f}% of bootstrap)")
    print(f"\nCompatible: {compatible} (needs ≥{min_freq}% in at least one metric)")
    
    return compatible

# Run check
check_compatibility(test_hourly_obs, bootstrap_samples)


#####################################################################
# CHECK ALL RAINY DAYS
#####################################################################

def check_all_days(daily_rain_wet_clean, n_wet_hours_clean, rain_high_wet_hwet_clean, 
                   hist_2d_n_wet, hist_2d_intensity, BINS_LOW, BINS_HIGH, 
                   n_bootstrap=1000, min_freq=1.0):
    """Check compatibility using frequency-based test."""
    
    n_days = len(daily_rain_wet_clean)
    results = {
        'day_idx': [], 'daily_value': [], 'n_wet_obs': [], 'compatible': [],
        'freq_n_wet': [], 'freq_max': [], 'freq_mean': []
    }
    
    print(f"\nChecking {n_days} rainy days...")
    
    for day_idx in range(n_days):
        if day_idx % 1000 == 0:
            print(f"  Processing day {day_idx}/{n_days}")
        
        test_daily_value = daily_rain_wet_clean[day_idx]
        test_n_wet_obs = int(n_wet_hours_clean[day_idx])
        
        if test_n_wet_obs == 0:
            continue
        
        start_idx, end_idx = get_hourly_indices_for_day(day_idx, n_wet_hours_clean)
        test_hourly_obs = rain_high_wet_hwet_clean[start_idx:end_idx]
        
        bootstrap_samples = sample_hourly_from_daily(
            test_daily_value, hist_2d_n_wet, hist_2d_intensity, 
            BINS_LOW, BINS_HIGH, n_bootstrap=n_bootstrap
        )
        
        # FREQUENCY-BASED statistics (not percentiles!)
        boot_n_wet = (bootstrap_samples > 0).sum(axis=1)
        boot_max = bootstrap_samples.max(axis=1)
        boot_mean_wet = np.array([bs[bs > 0].mean() for bs in bootstrap_samples])
        
        obs_n_wet = len(test_hourly_obs)
        obs_max = test_hourly_obs.max()
        obs_mean_wet = test_hourly_obs.mean()
        
        freq_n_wet = (boot_n_wet == obs_n_wet).sum() / len(boot_n_wet) * 100
        tol = 0.2
        freq_max = ((boot_max >= obs_max - tol) & (boot_max <= obs_max + tol)).sum() / len(boot_max) * 100
        freq_mean = ((boot_mean_wet >= obs_mean_wet - tol) & (boot_mean_wet <= obs_mean_wet + tol)).sum() / len(boot_mean_wet) * 100
        
        compatible = (freq_n_wet >= min_freq) or (freq_max >= min_freq) or (freq_mean >= min_freq)
        
        results['day_idx'].append(day_idx)
        results['daily_value'].append(test_daily_value)
        results['n_wet_obs'].append(obs_n_wet)
        results['compatible'].append(compatible)
        results['freq_n_wet'].append(freq_n_wet)
        results['freq_max'].append(freq_max)
        results['freq_mean'].append(freq_mean)
    
    results_df = pd.DataFrame(results)
    print(f"\n=== SUMMARY ===")
    print(f"Total days checked: {len(results_df)} (skipped {n_days - len(results_df)} days with 0 wet hours)")
    print(f"Compatible: {results_df['compatible'].sum()} ({results_df['compatible'].mean()*100:.1f}%)")
    print(f"Incompatible: {(~results_df['compatible']).sum()} ({(~results_df['compatible']).mean()*100:.1f}%)")
    
    return results_df

# Run for all days
results_df = check_all_days(
    daily_rain_wet_clean, n_wet_hours_clean, rain_high_wet_hwet_clean,
    hist_2d_n_wet, hist_2d_intensity, BINS_LOW, BINS_HIGH, 
    n_bootstrap=1000
)

# Inspect incompatible days
print("\nSample of incompatible days:")
print(results_df[~results_df['compatible']].head(10))

#####################################################################
# Test 99.9th percentile preservation
#####################################################################

# 1. Observed 99.9th percentile of wet hours
p999_observed = np.percentile(rain_high_wet_hwet_clean, 99.9)

print(f"Observed 99.9th percentile: {p999_observed:.2f}mm")

# 2. Generate large synthetic dataset and calculate p99.9 for each bootstrap
print("\nGenerating synthetic data for all days...")
n_bootstrap_members = 1000  # Number of complete datasets to generate

p999_bootstrap = []

for boot_idx in range(n_bootstrap_members):
    if boot_idx % 10 == 0:
        print(f"  Bootstrap member {boot_idx}/{n_bootstrap_members}")
    
    synthetic_hourly_all = []
    
    # Generate synthetic hourly for all days
    for day_idx in range(len(daily_rain_wet_clean)):
        daily_val = daily_rain_wet_clean[day_idx]
        
        # Generate one realization
        synth = sample_hourly_from_daily(
            daily_val, hist_2d_n_wet, hist_2d_intensity, 
            BINS_LOW, BINS_HIGH, n_bootstrap=1
        )[0]  # Take first (and only) sample
        
        # Extract wet hours
        wet_hours = synth[synth > 0]
        synthetic_hourly_all.extend(wet_hours)
    
    # Calculate p99.9 for this bootstrap member
    p999_bootstrap.append(np.percentile(synthetic_hourly_all, 99.9))

p999_bootstrap = np.array(p999_bootstrap)

# 3. Calculate 2.5-97.5 percentile range of bootstrap p99.9 values
ci_low = np.percentile(p999_bootstrap, 2.5)
ci_high = np.percentile(p999_bootstrap, 97.5)

print(f"\n=== RESULTS ===")
print(f"Observed p99.9: {p999_observed:.2f}mm")
print(f"Bootstrap p99.9 - Mean: {p999_bootstrap.mean():.2f}mm, Std: {p999_bootstrap.std():.2f}mm")
print(f"Bootstrap p99.9 - 95% CI: [{ci_low:.2f}, {ci_high:.2f}]mm")

in_range = ci_low <= p999_observed <= ci_high
print(f"\nObserved within 95% CI: {in_range}")
print(f"Bias: {(p999_bootstrap.mean() - p999_observed):.2f}mm ({(p999_bootstrap.mean() - p999_observed)/p999_observed*100:.1f}%)")

import matplotlib.pyplot as plt

#####################################################################
# Compare multiple percentiles: Observed vs Bootstrap
#####################################################################

percentiles_to_check = [90, 95, 98, 99, 99.5, 99.9]

# 1. Calculate observed percentiles
percentiles_observed = [np.percentile(rain_high_wet_hwet_clean, p) for p in percentiles_to_check]

print("Calculating bootstrap percentile ranges...")

# 2. Calculate same percentiles for each bootstrap member
percentiles_bootstrap = {p: [] for p in percentiles_to_check}

for boot_idx in range(n_bootstrap_members):
    if boot_idx % 10 == 0:
        print(f"  Bootstrap member {boot_idx}/{n_bootstrap_members}")
    
    synthetic_hourly_all = []
    
    for day_idx in range(len(daily_rain_wet_clean)):
        daily_val = daily_rain_wet_clean[day_idx]
        synth = sample_hourly_from_daily(
            daily_val, hist_2d_n_wet, hist_2d_intensity, 
            BINS_LOW, BINS_HIGH, n_bootstrap=1
        )[0]
        wet_hours = synth[synth > 0]
        synthetic_hourly_all.extend(wet_hours)
    
    # Calculate all percentiles for this bootstrap member
    for p in percentiles_to_check:
        percentiles_bootstrap[p].append(np.percentile(synthetic_hourly_all, p))

# 3. Get 95% CI for each percentile
percentiles_ci_low = [np.percentile(percentiles_bootstrap[p], 2.5) for p in percentiles_to_check]
percentiles_ci_high = [np.percentile(percentiles_bootstrap[p], 97.5) for p in percentiles_to_check]
percentiles_mean = [np.mean(percentiles_bootstrap[p]) for p in percentiles_to_check]

# 4. Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Bootstrap range as shaded area
ax.fill_between(percentiles_to_check, percentiles_ci_low, percentiles_ci_high, 
                alpha=0.3, color='blue', label='Bootstrap 95% CI')

# Bootstrap mean
ax.plot(percentiles_to_check, percentiles_mean, 'b--', linewidth=2, label='Bootstrap mean')

# Observed
ax.plot(percentiles_to_check, percentiles_observed, 'ro-', linewidth=2, 
        markersize=8, label='Observed')

ax.set_xlabel('Percentile', fontsize=12)
ax.set_ylabel('Hourly rainfall (mm)', fontsize=12)
ax.set_title('Observed vs Bootstrap: Wet Hour Intensity Percentiles', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(percentiles_to_check)

plt.tight_layout()
plt.savefig('percentiles_comparison.png', dpi=300, bbox_inches='tight')
#plt.show()

# Print summary
print("\n=== PERCENTILE COMPARISON ===")
print(f"{'Percentile':<12} {'Observed':<12} {'Bootstrap Mean':<15} {'95% CI':<25} {'In Range?'}")
print("-" * 80)
for i, p in enumerate(percentiles_to_check):
    in_range = percentiles_ci_low[i] <= percentiles_observed[i] <= percentiles_ci_high[i]
    print(f"{p:<12.1f} {percentiles_observed[i]:<12.2f} {percentiles_mean[i]:<15.2f} "
          f"[{percentiles_ci_low[i]:.2f}, {percentiles_ci_high[i]:.2f}]    {in_range}")



# ADD DIAGNOSTIC HERE
print("\n=== SCALING BIAS DIAGNOSTIC ===")
test_daily = 15.0
samples = sample_hourly_from_daily(test_daily, hist_2d_n_wet, hist_2d_intensity, 
                                    BINS_LOW, BINS_HIGH, n_bootstrap=1000)
print(f"For {test_daily}mm daily:")
print(f"Bootstrap max values - Mean: {samples.max(axis=1).mean():.2f}mm, "
      f"Max: {samples.max():.2f}mm")

# Check what the observed max is for similar daily values
similar_days = (daily_rain_wet_clean >= 14) & (daily_rain_wet_clean <= 16)
print(f"Observed for {test_daily}±1mm days: {similar_days.sum()} days")
if similar_days.sum() > 0:
    obs_maxes = []
    for day_idx in np.where(similar_days)[0]:
        start_idx, end_idx = get_hourly_indices_for_day(day_idx, n_wet_hours_clean)
        obs_maxes.append(rain_high_wet_hwet_clean[start_idx:end_idx].max())
    print(f"Observed max - Mean: {np.mean(obs_maxes):.2f}mm, Max: {np.max(obs_maxes):.2f}mm")

import pdb; pdb.set_trace()  # fmt: skip