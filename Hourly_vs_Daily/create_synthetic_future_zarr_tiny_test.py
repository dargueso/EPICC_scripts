#!/usr/bin/env python
"""
Memory-efficient full-domain version with multiprocessing.
Uses shared memory for zero-copy data access across workers.
FIXED: Correct buffer handling + FAST multiprocessing via shared memory
"""
import os
import sys
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
WRUN_FUTURE = "EPICC_2km_ERA5"
test_suffix = "_test_3x3"
# Frequency configuration
FREQ_HIGH = '01H'   # High frequency (e.g., '10MIN', '1H')
FREQ_LOW = 'DAY'     # Low frequency (e.g., '1H', '3H', '6H', '12H', 'D')

# Wet thresholds
WET_VALUE_HIGH = 0.1  # mm per high-freq interval
WET_VALUE_LOW = 1   # mm per low-freq interval

# Bins - adjust based on frequencies
BINS_HIGH = np.arange(0, 101, 1)  # For hourly: 0-100mm in 1mm steps
BINS_LOW = np.arange(0, 105, 5)   # For daily: 0-100mm in 5mm steps

# Quantiles to calculate from synthetic samples
QUANTILES = np.array([0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85,
                      0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0], 
                     dtype=np.float32)

# Bootstrap confidence levels to compute across samples
BOOTSTRAP_QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99], dtype=np.float32)

# Processing parameters
TILE_SIZE = 50
BUFFER = 1
N_SAMPLES = 1000
N_PROCESSES = 4  # Adjust based on your system

if FREQ_HIGH == '10MIN':
    FREQ_TO_HIGH = {
        '10MIN': 1,
        '01H': 6,
        'DAY': 144  # 24 hours
    }
elif FREQ_HIGH == '01H':
    FREQ_TO_HIGH = {
        '01H': 1,
        'DAY': 24  # 24 hours
    }

intervals_high = FREQ_TO_HIGH[FREQ_HIGH]
intervals_low = FREQ_TO_HIGH[FREQ_LOW]
n_interval = intervals_low // intervals_high  # e.g., 24 for hourly given daily

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _window_sum(arr, radius):
    """Sliding-window sum over last two spatial axes."""
    k = 2 * radius + 1
    size = [1] * (arr.ndim - 2) + [k, k]
    return uniform_filter(arr, size=size, mode="nearest") * (k * k)


def sample_hourly_from_daily(daily_value, hist_2d_n_wet, hist_2d_intensity, hist_2d_maxhour,
                              BINS_LOW, BINS_HIGH, n_bootstrap=1000, daily_noise=0.0):
    bootstrap_samples = []
    max_hourly_samples = []
    nbins_high = len(BINS_HIGH) - 1

    for _ in range(n_bootstrap):
        if daily_noise > 0:
            daily_noisy = daily_value + np.random.normal(0, daily_noise)
            daily_noisy = max(daily_noisy, WET_VALUE_LOW)
        else:
            daily_noisy = daily_value

        bin_idx = np.digitize(daily_noisy, BINS_LOW) - 1
        p_n_wet     = hist_2d_n_wet[bin_idx, :]
        p_intensity = hist_2d_intensity[bin_idx, :]
        p_maxhour   = hist_2d_maxhour[bin_idx, :]

        n_wet = np.random.choice(np.arange(1, n_interval + 1), p=p_n_wet / p_n_wet.sum())
        hourly = np.zeros(24)

        # --- Sample max hourly value ---
        max_bin_idx = np.random.choice(nbins_high, p=p_maxhour / p_maxhour.sum())
        lo, hi = BINS_HIGH[max_bin_idx], BINS_HIGH[max_bin_idx + 1]
        max_hourly = lo + np.random.exponential(scale=10) if np.isinf(hi) else np.random.uniform(lo, hi)
        max_hourly = min(max_hourly, daily_noisy)
        max_hourly = max(max_hourly, WET_VALUE_HIGH)

        if n_wet == 1:
            hourly[np.random.randint(24)] = daily_noisy
            bootstrap_samples.append(hourly)
            max_hourly_samples.append(daily_noisy)
            continue

        remaining = daily_noisy - max_hourly

        if remaining <= 0:
            wet_pos = np.random.randint(24)
            hourly[wet_pos] = daily_noisy
            bootstrap_samples.append(hourly)
            max_hourly_samples.append(daily_noisy)
            continue

        # --- Sample and scale remaining n_wet-1 hours ---
        # Reduce n_wet until n_remaining * WET_VALUE_HIGH <= remaining
        n_remaining = n_wet - 1
        while n_remaining > 0 and n_remaining * WET_VALUE_HIGH > remaining:
            n_remaining -= 1

        if n_remaining == 0:
            # Can't fit any more wet hours above threshold - put everything in max hour
            wet_pos = np.random.randint(24)
            hourly[wet_pos] = daily_noisy
            bootstrap_samples.append(hourly)
            max_hourly_samples.append(daily_noisy)
            continue

        # Sample intensities for remaining hours
        bin_indices = np.random.choice(nbins_high, size=n_remaining,
                                       p=p_intensity / p_intensity.sum())
        intensities = np.zeros(n_remaining)
        for i, idx in enumerate(bin_indices):
            lo, hi = BINS_HIGH[idx], BINS_HIGH[idx + 1]
            intensities[i] = lo + np.random.exponential(scale=10) if np.isinf(hi) else np.random.uniform(lo, hi)

        # Shift-then-scale: preserves shape, enforces min=WET_VALUE_HIGH, sums to remaining
        excess_budget = remaining - n_remaining * WET_VALUE_HIGH  # must be >= 0
        intensity_residual = intensities - intensities.min()       # min=0, shape preserved

        if intensity_residual.sum() > 0:
            intensities_scaled = intensity_residual * (excess_budget / intensity_residual.sum()) + WET_VALUE_HIGH
        else:
            # All sampled values identical - distribute evenly
            intensities_scaled = np.full(n_remaining, remaining / n_remaining)

        # Final check: should sum to remaining and min >= WET_VALUE_HIGH
        assert abs(intensities_scaled.sum() - remaining) < 1e-4
        assert intensities_scaled.min() >= WET_VALUE_HIGH - 1e-6

        wet_positions = np.random.choice(24, size=n_remaining + 1, replace=False)
        hourly[wet_positions[0]] = max_hourly
        hourly[wet_positions[1:]] = intensities_scaled

        bootstrap_samples.append(hourly)
        max_hourly_samples.append(max_hourly)


    return np.array(bootstrap_samples), np.array(max_hourly_samples)


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
    prob_file = f'{PATH_IN}/{WRUN_PRESENT}/condprob_{FREQ_HIGH}_given_{FREQ_LOW}{test_suffix}.nc'
    ds_prob = xr.open_dataset(prob_file)

    # VALIDATE BIN COMPATIBILITY
    prob_bins_high = ds_prob.attrs.get(f'bin_edges_{FREQ_HIGH}')
    prob_bins_low = ds_prob.attrs.get(f'bin_edges_{FREQ_LOW}')

    if prob_bins_high is not None:
        # Script 1 stores bins without the np.inf edge in attributes
        expected_bins_high = BINS_HIGH[:-1]  # Remove inf for comparison
        if not np.allclose(prob_bins_high, expected_bins_high):
            raise ValueError(f"BINS_HIGH mismatch!\n"
                            f"  Expected: {expected_bins_high[:5]}...\n"
                            f"  Got: {prob_bins_high[:5]}...")
        print(f"   ✓ High-frequency bins match ({len(prob_bins_high)} bins)")

    if prob_bins_low is not None:
        expected_bins_low = BINS_LOW[:-1]
        if not np.allclose(prob_bins_low, expected_bins_low):
            raise ValueError(f"BINS_LOW mismatch!\n"
                            f"  Expected: {expected_bins_low}\n"
                            f"  Got: {prob_bins_low}")
        print(f"   ✓ Low-frequency bins match ({len(prob_bins_low)} bins)")

    # VALIDATE THRESHOLDS
    prob_wet_high = ds_prob.attrs.get('wet_threshold_high')
    prob_wet_low = ds_prob.attrs.get('wet_threshold_low')

    if prob_wet_high != WET_VALUE_HIGH:
        raise ValueError(f"WET_VALUE_HIGH mismatch! Expected {prob_wet_high}, got {WET_VALUE_HIGH}")
    if prob_wet_low != WET_VALUE_LOW:
        raise ValueError(f"WET_VALUE_LOW mismatch! Expected {prob_wet_low}, got {WET_VALUE_LOW}")
    print(f"   ✓ Wet thresholds match (high={WET_VALUE_HIGH}, low={WET_VALUE_LOW})")

    # Load future low-frequency rainfall
    print(f"\n2. Loading future {FREQ_LOW} rainfall...")
    freq_name_low = FREQ_LOW
    future_file = f'{PATH_IN}/{WRUN_FUTURE}/UIB_{freq_name_low}_RAIN{test_suffix}.zarr'

    try:
        ds_future = xr.open_zarr(future_file, consolidated=False)
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
    n_wet_dist = ds_prob.cond_prob_n_wet.values.copy()
    intens_dist = ds_prob.cond_prob_intensity.values.copy()
    max_int_dist = ds_prob.cond_prob_max_intensity.values.copy()
    n_events = ds_prob.n_events.values.copy()

    ds_prob.close()
    ds_future.close()

    # Convert to proper format and handle NaNs
    n_wet_dist = np.nan_to_num(n_wet_dist, nan=0.0)
    intens_dist = np.nan_to_num(intens_dist, nan=0.0)
    max_int_dist = np.nan_to_num(max_int_dist, nan=0.0)
    n_events = np.nan_to_num(n_events, nan=0.0)


    # Transpose to match expected format
    n_wet_dist = np.transpose(n_wet_dist, (1, 0, 2, 3))
    intens_dist = np.transpose(intens_dist, (1, 0, 2, 3))
    max_int_dist = np.transpose(max_int_dist, (1, 0, 2, 3))
    
    # Apply smoothing with buffer
    print("\n5. Applying spatial smoothing...")
    wet_weighted = n_wet_dist * n_events[:, np.newaxis, :, :]
    intens_weighted = intens_dist * n_events[:, np.newaxis, :, :]
    max_int_weighted = max_int_dist * n_events[:, np.newaxis, :, :]

    comp_samp = _window_sum(n_events.astype(np.float64), BUFFER)
    comp_wet = _window_sum(wet_weighted.astype(np.float64), BUFFER)
    comp_hour = _window_sum(intens_weighted.astype(np.float64), BUFFER)
    comp_max = _window_sum(max_int_weighted.astype(np.float64), BUFFER)

    # Safe division
    nonzero_mask = comp_samp != 0
    comp_wet = np.where(nonzero_mask[:, None], comp_wet / comp_samp[:, None], 0.0)
    comp_hour = np.where(nonzero_mask[:, None], comp_hour / comp_samp[:, None], 0.0)
    comp_max = np.where(nonzero_mask[:, None], comp_max / comp_samp[:, None], 0.0)

    # Create CDFs
    wet_cdf = np.cumsum(comp_wet, axis=1).astype(np.float32, order="C")
    hour_cdf = np.cumsum(comp_hour, axis=1).astype(np.float32, order="C")
    max_cdf = np.cumsum(comp_max, axis=1).astype(np.float32, order="C")


    hourly_intensity_quantiles = np.zeros((N_SAMPLES,len(QUANTILES), ny, nx), dtype=np.float32)
    max_hourly_intensity_quantiles = np.zeros((N_SAMPLES,len(QUANTILES), ny, nx), dtype=np.float32)

    for i, iy in enumerate(range(BUFFER, ny - BUFFER)):
        for j, ix in enumerate(range(BUFFER, nx - BUFFER)):
            # Update the CDFs with the new values
            rain_arr_clean = rain_low_freq[:, iy, ix]
            rain_arr_clean = rain_arr_clean[~np.isnan(rain_arr_clean)]
            rain_arr_clean = rain_arr_clean[rain_arr_clean >= WET_VALUE_LOW]

            hourly_intensity_bootstrap = np.zeros((N_SAMPLES,rain_arr_clean.shape[0], n_interval), dtype=np.float32)
            max_hourly_bootstrap = np.zeros((N_SAMPLES, rain_arr_clean.shape[0]), dtype=np.float32)

            for t in range(rain_arr_clean.shape[0]):
                hourly_intensity_bootstrap[:, t, :], max_hourly_bootstrap[:, t] = sample_hourly_from_daily(
                    rain_arr_clean[t], comp_wet[:,:, iy, ix], comp_hour[:,:, iy, ix], comp_max[:,:,iy, ix],
                    BINS_LOW, BINS_HIGH, n_bootstrap=N_SAMPLES, daily_noise=0.0
                )
        

        for b, q in enumerate(range(N_SAMPLES)):
            
            this_sample = hourly_intensity_bootstrap[b, :,:]
            wet_only_bootstrap = this_sample[this_sample >= WET_VALUE_HIGH]
            hourly_intensity_quantiles[b, :, iy, ix] = np.quantile(wet_only_bootstrap, QUANTILES)
            max_hourly_intensity_quantiles[b, :, iy, ix] = np.quantile(max_hourly_bootstrap[b, :], QUANTILES)

        hrly_estimates = np.quantile(hourly_intensity_quantiles, QUANTILES, axis=0)
        max_hrly_estimates = np.quantile(max_hourly_intensity_quantiles, QUANTILES, axis=0)
        

        import pdb; pdb.set_trace()  # fmt: skip

    # Cleanup
    del wet_weighted, intens_weighted, max_int_weighted, n_events
    del comp_samp, comp_wet, comp_hour, comp_max, nonzero_mask

    # Create shared memory blocks
    print("\n6. Creating shared memory blocks...")
    shm_rain = shared_memory.SharedMemory(create=True, size=rain_low_freq.nbytes)
    shm_bin = shared_memory.SharedMemory(create=True, size=bin_idx.nbytes)
    shm_wet = shared_memory.SharedMemory(create=True, size=wet_cdf.nbytes)
    shm_hour = shared_memory.SharedMemory(create=True, size=hour_cdf.nbytes)
    shm_max = shared_memory.SharedMemory(create=True, size=max_cdf.nbytes)



if __name__ == "__main__":
    main()