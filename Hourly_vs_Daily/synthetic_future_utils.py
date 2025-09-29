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

def calculate_wet_hour_intensity_distribution(ds_h_wet_days, 
                              ds_d, 
                              wet_hour_fraction,
                              drain_bins = np.arange(0,55,5), 
                              hrain_bins = np.arange(0, 105, 5),
                              ):
    """
    Calculate the wet hour intensity distribution and other statistics for hourly rainfall.
    """
    ny = ds_h_wet_days.sizes['y']
    nx = ds_h_wet_days.sizes['x']
    nhbins = len(hrain_bins) - 1
    ndbins = len(drain_bins)

    wet_hours_fraction = np.zeros((ndbins, ny, nx))
    samples_per_bin = np.zeros((ndbins, ny, nx))
    hourly_distribution_bin = np.zeros((ndbins, nhbins, ny, nx))
    wet_hours_distribution_bin = np.zeros((ndbins, 24, ny, nx))

    for ibin in tqdm(range(ndbins), desc="Processing rainfall bins"):
        if ibin == ndbins-1:
            upper_bound = np.inf
        else:
            lower_bound = drain_bins[ibin]
            upper_bound = drain_bins[ibin + 1]   

        bin_days = (ds_d >= lower_bound) & (ds_d < upper_bound) 
        bin_days_hourly = bin_days.reindex(time=ds_h_wet_days.time, method='ffill')
        bin_ds_h_masked = ds_h_wet_days.where(bin_days_hourly)

        wet_hours = bin_ds_h_masked.where(bin_ds_h_masked > 0).count(dim=['time'])
        wet_hours_fraction[ibin,:,:] = wet_hours / bin_days_hourly.sum(dim=['time'])
        samples_per_bin[ibin,:,:] = np.sum(bin_days.values, axis=(0))
        
        masked = bin_ds_h_masked.where(bin_ds_h_masked > 0)

        # Fix: Check if masked has chunks to determine if it's a dask array
        use_dask = hasattr(masked.data, 'chunks') and masked.data.chunks is not None
        
        # If using dask, ensure time dimension is in a single chunk
        if use_dask:
            masked = masked.chunk(dict(time=-1))
        
        hist_hourint = xr.apply_ufunc(
            compute_histogram,
            masked,
            input_core_dims=[["time"]],
            output_core_dims=[["bin"]],
            kwargs={"bins": hrain_bins},
            vectorize=True,
            dask="parallelized" if use_dask else "forbidden",
            output_dtypes=[float],
            dask_gufunc_kwargs={
            "output_sizes": {"bin": len(hrain_bins) - 1}
            } if use_dask else {},
        )
        hourly_distribution_bin[ibin, :, :, :] = hist_hourint.transpose("bin", "y", "x").data

        masked_wethour = wet_hour_fraction.where(bin_days)* 24.0  # Convert to hours

        # Fix: Check if masked_wethour has chunks to determine if it's a dask array
        use_dask_wethour = hasattr(masked_wethour.data, 'chunks') and masked_wethour.data.chunks is not None

        # If using dask, ensure time dimension is in a single chunk
        if use_dask_wethour:
            masked_wethour = masked_wethour.chunk(dict(time=-1))

        hist_wethours = xr.apply_ufunc(
            compute_histogram,
            masked_wethour,
            input_core_dims=[["time"]],
            output_core_dims=[["bin"]],
            kwargs={"bins": np.arange(1, 26, 1)},
            vectorize=True,
            dask="parallelized" if use_dask_wethour else "forbidden",
            output_dtypes=[float],
            dask_gufunc_kwargs={
            "output_sizes": {"bin": 24}
            } if use_dask_wethour else {},
        )

        wet_hours_distribution_bin [ibin, :, :, :] = hist_wethours.transpose("bin", "y", "x").data

    return hourly_distribution_bin, wet_hours_distribution_bin, samples_per_bin

def save_probability_data(hourly_distribution_bin, 
                          wet_hours_distribution_bin, 
                          samples_per_bin, 
                          drain_bins, 
                          hrain_bins,
                          fout='rainfall_probabilities.nc'):
    """
    Build an xarray with probabilities and save to a pickle file.
    """
    # ------------------------------------------------------------------
    # 1.  Coordinate vectors
    # ------------------------------------------------------------------
    drain_bin_edges = drain_bins                       # 11 edges, 0 … 50 mm
    hrain_bin_edges = hrain_bins                       # 21 edges, 0 … 100 mm
    hour_vec        = np.arange(1,25)                  # 1 … 24
    ny = hourly_distribution_bin.shape[2]  # Fix: changed from shape[3] to shape[2]
    nx = hourly_distribution_bin.shape[3]  # Fix: changed from shape[4] to shape[3]

    # For the hourly-rain axis we usually want *bin centres*
    # rather than edges, so take the midpoint between each pair:
    hrain_bin_mid = (hrain_bin_edges[:-1] + hrain_bin_edges[1:]) / 2   # 20 values

    # ------------------------------------------------------------------
    # 2.  Wrap each array in a DataArray
    # ------------------------------------------------------------------
    hourly_da = xr.DataArray(
        data   = hourly_distribution_bin,              # shape (11, 20, 2)
        dims   = ('drain_bin', 'hrain_bin','y', 'x'),
        coords = {
            'drain_bin' : drain_bin_edges,             # mm
            'hrain_bin' : hrain_bin_mid,               # mm h⁻¹ (bin centres)
            'y': np.arange(ny),                # y grid points
            'x': np.arange(nx)                 # x grid points
        },
        name   = 'hourly_distribution',
        attrs  = {'description': 'Hourly rainfall distribution per drain-bin'}
    )

    wet_hours_da = xr.DataArray(
        data   = wet_hours_distribution_bin,           # shape (11, 24, 2)
        dims   = ('drain_bin', 'hour','y', 'x'),
        coords = {
            'drain_bin' : drain_bin_edges,
            'hour'      : hour_vec,                    # 1 … 24 (local hour)
            'y': np.arange(ny),                # y grid points
            'x': np.arange(nx)                 # x grid points
        },
        name   = 'wet_hours_distribution',
        attrs  = {'description': 'Number of wet hours per drain-bin'}
    )

    samples_da = xr.DataArray(
        data   = samples_per_bin,                      # shape (11, 2)
        dims   = ( 'drain_bin', 'y', 'x'),
        coords = {
            'drain_bin' : drain_bin_edges,
            'y': np.arange(ny),                # y grid points
            'x': np.arange(nx)                 # x grid points
        },
        name   = 'samples_per_bin',
        attrs  = {'description': 'Sample count per bin-bin'}
    )

    # ------------------------------------------------------------------
    # 3.  Merge into one Dataset
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        data_vars = {
            'hourly_distribution'   : hourly_da,
            'wet_hours_distribution': wet_hours_da,
            'samples_per_bin'       : samples_da
        },
        coords = {
            'drain_bin' : ('drain_bin', drain_bin_edges, {'units': 'mm'}),
            'hrain_bin' : ('hrain_bin', hrain_bin_mid,   {'units': 'mm h-1'}),
            'hour'      : ('hour', hour_vec),
            'y': ('y', np.arange(ny)),                # y grid points
            'x': ('x', np.arange(nx)),                 # x grid points
        },
        attrs = {
            'title'      : 'Rainfall bin statistics',
            'created_by' : 'merge-arrays-into-xarray.py',
            'note'       : 'hrain_bin coordinate uses bin centres; change to edges if preferred'
        }
    )

    # ------------------------------------------------------------------
    # 4.  (Optional) quick sanity check and save
    # ------------------------------------------------------------------
    print(ds)
    ds.to_netcdf(fout)
    return (ds)

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
                               thresh, iy, ix, rng_state, quantiles):
    """
    Process a single cell and return quantiles directly.
    Memory efficient - doesn't store all values.
    """
    n_t = rain.shape[0]
    temp_values = np.empty(n_t * 24, dtype=np.float32)  # Worst case: 24 values per day
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
            max_val = np.max(values)
            if max_val > thresh:
                temp_values[value_count] = max_val
                value_count += 1
    
    # Calculate quantiles from collected values
    if value_count > 0:
        actual_values = temp_values[:value_count]
        return _calculate_quantiles_streaming(actual_values, quantiles, thresh)
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
                thresh, iy, ix, rng, quantiles)
            
            result[:, i, j] = cell_quantiles
    
    return result