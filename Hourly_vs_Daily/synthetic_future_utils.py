import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter        # fast moving-window sum
from tqdm.auto import tqdm    
import config as cfg

y_idx=cfg.y_idx
x_idx=cfg.x_idx
WET_VALUE_H = cfg.WET_VALUE_H  # mm
WET_VALUE_D = cfg.WET_VALUE_D  # mm
drain_bins = cfg.drain_bins
hrain_bins = cfg.hrain_bins
buffer = cfg.buffer
n_samples = cfg.n_samples

# ... (keep all existing functions: compute_histogram, calculate_wet_hour_intensity_distribution, 
#      save_probability_data, _window_sum, _arr_append) ...

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
    nexp = ds_h_wet_days.sizes['exp']
    ny = ds_h_wet_days.sizes['y']
    nx = ds_h_wet_days.sizes['x']
    nhbins = len(hrain_bins) - 1
    ndbins = len(drain_bins)

    wet_hours_fraction = np.zeros((2,ndbins, ny, nx))
    samples_per_bin = np.zeros((2,ndbins, ny, nx))
    hourly_distribution_bin = np.zeros((nexp, ndbins, nhbins, ny, nx))
    wet_hours_distribution_bin = np.zeros((nexp, ndbins, 24, ny, nx))

    for ibin in range(ndbins):
        if ibin == ndbins-1:
            upper_bound = np.inf
        else:
            lower_bound = drain_bins[ibin]
            upper_bound = drain_bins[ibin + 1]   

        bin_days = (ds_d >= lower_bound) & (ds_d < upper_bound) 
        bin_days_hourly = bin_days.reindex(time=ds_h_wet_days.time, method='ffill')
        bin_ds_h_masked = ds_h_wet_days.where(bin_days_hourly)

        wet_hours = bin_ds_h_masked.where(bin_ds_h_masked > 0).count(dim=['time'])
        wet_hours_fraction[:,ibin,:,:] = wet_hours / bin_days_hourly.sum(dim=['time'])
        samples_per_bin[:,ibin,:,:] = np.sum(bin_days.values, axis=(1))
        
        masked = bin_ds_h_masked.where(bin_ds_h_masked > 0)

        hist_hourint = xr.apply_ufunc(
            compute_histogram,
            masked,
            input_core_dims=[["time"]],
            output_core_dims=[["bin"]],
            kwargs={"bins": hrain_bins},
            vectorize=True,
            dask="parallelized" if isinstance(masked.data, xr.core.dataarray.DataArray) and hasattr(masked.data, 'chunks') else False,
            output_dtypes=[float],
        )
        hourly_distribution_bin[:, ibin, :, :, :] = hist_hourint.transpose("exp", "bin", "y", "x").data

        masked_wethour = wet_hour_fraction.where(bin_days)* 24.0  # Convert to hours

        hist_wethours = xr.apply_ufunc(
            compute_histogram,
            masked_wethour,
            input_core_dims=[["time"]],
            output_core_dims=[["bin"]],
            kwargs={"bins": np.arange(1, 26, 1)},
            vectorize=True,
            dask="parallelized" if isinstance(masked_wethour.data, xr.core.dataarray.DataArray) and hasattr(masked_wethour.data, 'chunks') else False,
            output_dtypes=[float],
        )

        wet_hours_distribution_bin [:, ibin, :, :, :] = hist_wethours.transpose("exp", "bin", "y", "x").data

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
    exp_vec         = ['Present', 'Future']            # ['Present', 'Future']
    ny = hourly_distribution_bin.shape[3]  # number of y grid points
    nx = hourly_distribution_bin.shape[4]  # number of x grid points

    # For the hourly-rain axis we usually want *bin centres*
    # rather than edges, so take the midpoint between each pair:
    hrain_bin_mid = (hrain_bin_edges[:-1] + hrain_bin_edges[1:]) / 2   # 20 values

    # ------------------------------------------------------------------
    # 2.  Wrap each array in a DataArray
    # ------------------------------------------------------------------
    hourly_da = xr.DataArray(
        data   = hourly_distribution_bin,              # shape (11, 20, 2)
        dims   = ('experiment', 'drain_bin', 'hrain_bin','y', 'x'),
        coords = {
            'experiment': exp_vec,
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
        dims   = ('experiment','drain_bin', 'hour','y', 'x'),
        coords = {
            'experiment': exp_vec,
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
        dims   = ('experiment', 'drain_bin', 'y', 'x'),
        coords = {
            'drain_bin' : drain_bin_edges,
            'experiment': exp_vec,
            'y': np.arange(ny),                # y grid points
            'x': np.arange(nx)                 # x grid points
        },
        name   = 'samples_per_bin',
        attrs  = {'description': 'Sample count per drain-bin'}
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
            'experiment': ('experiment', exp_vec),
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

def window_sum(arr, radius):
    """
    Sliding-window *sum* over the last two spatial axes (ny, nx).
    Works for both 3-D and 4-D arrays by building the size tuple at run-time.
    """
    k = 2 * radius + 1
    size = [1] * (arr.ndim - 2) + [k, k]   # e.g. (1,1,k,k) or (1,k,k)
    return uniform_filter(arr, size=size, mode="nearest") * (k * k)


#####################################################################
#####################################################################


def generate_consistent_synthetic_data(rain_arr, wet_cdf, hour_cdf, bin_idx, 
                                     hr_edges, iy0, iy1, ix0, ix1, n_time,
                                     thresh=0.1, seed=None):
    """
    Generate synthetic hourly data consistently for both full data and quantiles.
    Returns a list of arrays, one per cell, with values for each timestep.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n_t = rain_arr.shape[0]
    width = ix1 - ix0
    n_cells = (iy1 - iy0) * width
    
    # Initialize data structure: one array per cell with n_time values
    cell_data = []
    for _ in range(n_cells):
        cell_data.append(np.zeros(n_time, dtype=np.float32))
    
    # Generate data for each timestep
    for t in range(n_t):
        c = 0
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                R = rain_arr[t, iy, ix]
                if R <= 0 or not np.isfinite(R):
                    c += 1
                    continue
                    
                b = bin_idx[t, iy, ix]
                if b < 0:
                    c += 1
                    continue

                # 1 - wet-hour count
                Nh = np.searchsorted(wet_cdf[b, :, iy, ix], rng.random()) + 1
                
                # 2 - hourly-intensity bins
                cdf_hr = hour_cdf[b, :, iy, ix]
                idx_bins = np.empty(Nh, dtype=np.int64)
                for k in range(Nh):
                    idx_bins[k] = np.searchsorted(cdf_hr, rng.random())

                # 3 - intensities inside bins
                intens = np.empty(Nh, dtype=np.float32)
                for k in range(Nh):
                    lo = hr_edges[idx_bins[k]]
                    hi = hr_edges[idx_bins[k] + 1]
                    intens[k] = lo + (hi - lo) * rng.random()

                s_int = intens.sum()
                if s_int == 0.0:
                    c += 1
                    continue
                    
                values = R * intens / s_int
                
                if np.any(values < thresh):
                    values = values[values >= thresh]
                    values *= R / values.sum()
                    Nh = len(values)
                    

                # 4 - choose hours & find maximum value above threshold
                sel = np.empty(Nh, dtype=np.int64)
                filled = 0
                for hour in range(24):
                    need = Nh - filled
                    if need == 0:
                        break
                    if rng.random() < need / (24 - hour):
                        sel[filled] = hour
                        filled += 1
                


                # Store the maximum hourly value for this timestep
                max_val = np.max(values)
                if max_val <= thresh:
                    raise ValueError(f"Maximum value {max_val} below threshold {thresh} at timestep {t}, cell ({iy},{ix})")
                

                # Store in cell data (0 if no wet value above threshold)
                cell_data[c][t] = max_val if max_val > thresh else 0.0
                

                c += 1
    
    return cell_data
