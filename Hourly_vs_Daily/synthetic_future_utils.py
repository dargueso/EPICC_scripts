import xarray as xr
import numpy as np
from numba import njit, float32
from scipy.ndimage import uniform_filter        # fast moving-window sum
from tqdm.auto import tqdm    
import config as cfg


drain_bins = cfg.drain_bins
hrain_bins = cfg.hrain_bins
buffer = cfg.buffer
n_samples = cfg.n_samples

def compute_histogram(values, bins):
    # values: 1D array over time
    valid = values[values > 0]
    hist, _ = np.histogram(valid, bins=bins, density=True)
    return hist * np.diff(bins)


def _window_sum(arr, radius):
    """
    Sliding-window *sum* over the last two spatial axes (ny, nx).
    Works for both 3-D and 4-D arrays by building the size tuple at run-time.
    """
    k = 2 * radius + 1
    size = [1] * (arr.ndim - 2) + [k, k]   # e.g. (1,1,k,k) or (1,k,k)
    return uniform_filter(arr, size=size, mode="nearest") * (k * k)

#####################################################################
# NEW FUNCTIONS FOR TIME-BASED QUANTILES - SERIAL VERSION
#####################################################################

@njit(cache=True)
def _calculate_quantiles_streaming(values_array, quantiles, thresh):
    """
    Calculate quantiles from an array of values above threshold.
    Much more memory efficient than storing all values.
    """
    # Filter values above threshold
    valid_values = values_array[values_array > thresh]
    
    if len(valid_values) == 0:
        return np.full(len(quantiles), np.nan, dtype=np.float32)
    
    # Sort values for quantile calculation
    valid_values = np.sort(valid_values)
    n = len(valid_values)
    
    result = np.empty(len(quantiles), dtype=np.float32)
    
    for i, q in enumerate(quantiles):
        if q == 0.0:
            result[i] = valid_values[0]
        elif q == 1.0:
            result[i] = valid_values[-1]
        else:
            # Linear interpolation for quantiles
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
                               thresh, iy, ix, rng_state, quantiles,n_interval):
    """
    Process a single cell and return quantiles directly.
    Memory efficient - doesn't store all values.
    """
    n_t = rain.shape[0]
    temp_values = np.empty(n_t * n_interval, dtype=np.float32)  # Worst case: 24 values per day (if hourly), 144 if 10min
    #temp_values = None
    value_count = 0
    
    for t in range(n_t):
        
        R = rain[t, iy, ix]
        if R <= 0 or not np.isfinite(R):
            continue
        b = bin_idx[t, iy, ix]
        
        if b < 0 or b >= wet_cdf.shape[0]:
            continue

        # 1 — wet-hour count
        wet_cdf_slice = wet_cdf[b, :, iy, ix]
        if not np.any(wet_cdf_slice > 0):
            continue
            
        Nh = np.searchsorted(wet_cdf_slice, rng_state.random()) + 1
        if Nh <= 0:
            continue

        # 2 — hourly-intensity bins
        cdf_hr = hour_cdf[b, :, iy, ix]
        if not np.any(cdf_hr > 0):
            continue
            
        idx_bins = np.empty(Nh, np.int64)
        for k in range(Nh):
            idx_bins[k] = np.searchsorted(cdf_hr, rng_state.random())

        # 3 — intensities inside bins
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

        if np.any(values < thresh):
            mask = values >= thresh
            if not np.any(mask):
                continue
            values = values[mask]
            values *= R / values.sum()

        # Store the maximum hourly value for this timestep
        if len(values) > 0:
            n_values = len(values)
            max_val = np.max(values)
            min_val = np.min(values)
            if min_val > thresh:
                for k in range(n_values):
                    temp_values[value_count] = values[k]
                    value_count += 1
            else:
                raise ValueError("Unexpected value below threshold after masking.")

    if value_count > 0:
        return _calculate_quantiles_streaming(temp_values[:value_count], quantiles, thresh)
    else:
        return np.full(len(quantiles), np.nan, dtype=np.float32)

def generate_quantiles_directly(
        rain_arr,            # (time, ny, nx)  float32
        wet_cdf, hour_cdf,   # cum-sums (ready)
        bin_idx,             # (time, ny, nx)  int16
        hr_edges,            # (n_hour_bins+1,)
        quantiles,           # quantiles to calculate
        iy0, iy1, ix0, ix1,  # inner tile boundaries (no buffer)
        thresh=0.1,
        seed=None):
    """
    Generate quantiles directly without storing full time series.
    Only processes the inner tile region (no buffer).
    Much more memory efficient.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    
    ny_inner = iy1 - iy0
    nx_inner = ix1 - ix0
    n_quantiles = len(quantiles)
    
    # Result array: (n_quantiles, ny_inner, nx_inner) - only inner tile
    result = np.full((n_quantiles, ny_inner, nx_inner), np.nan, dtype=np.float32)
    
    # Process only the inner tile cells (no buffer)
    for i, iy in enumerate(range(iy0, iy1)):
        for j, ix in enumerate(range(ix0, ix1)):
            cell_quantiles = _process_cell_for_quantiles(
                rain_arr, bin_idx, wet_cdf, hour_cdf, hr_edges, 
                thresh, iy, ix, rng, quantiles, n_interval=24)
            
            result[:, i, j] = cell_quantiles
    
    return result
