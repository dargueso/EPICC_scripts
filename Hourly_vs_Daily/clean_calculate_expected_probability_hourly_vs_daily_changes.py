#!/usr/bin/env python
'''
@File    :  clean_calculate_expected_probability_hourly_vs_daily_changes.py
@Time    :  2025/07/09 15:34:17
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  None
@Desc    :  Script to calculate the expected probability of hourly vs daily changes in rainfall
            and determine if changes in hourly rainfall can be explained by changes in daily rainfall.
'''


import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from numba import njit, prange
from scipy.ndimage import uniform_filter        # fast moving-window sum
from tqdm.auto import tqdm                      # nice, notebook-friendly bar

WET_VALUE = 0.1  # mm

y_idx=258; x_idx=559 #1.01 km away from Palma University station 

qs = xr.DataArray([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dims='q', name='quantile')


# ###############################################################################
# ##### FUNCTIONS
# ###############################################################################

def calculate_hourly_to_daily_ratio(ds_h, ds_d):

    daily_expanded = ds_d.reindex_like(ds_h, method='ffill')
    rainfall_ratios = ds_h / daily_expanded

    return rainfall_ratios

def calculate_ratio_coincident_extremes(ds_hx, ds_d, qtiles_hx, qtiles_d, quantile=0.95):

    ds_hx_ext = ds_hx > qtiles_hx.sel(quantile=quantile)
    ds_d_ext = ds_d > qtiles_d.sel(quantile=quantile)


    tot_h = ds_hx_ext.sum(dim='time')  # Total number of days with hourly max above quantile
    tot_d = ds_d_ext.sum(dim='time')  # Total number of days with daily rainfall above quantile

    both = (ds_hx_ext) & (ds_d_ext)  # Coincident extremes
    both_tot = both.sum(dim='time')
    ratio = both_tot / (tot_h)

    if np.all(tot_h != tot_d):
        raise ValueError("Total number of days with hourly max and daily rainfall above quantile do not match.")
    
    return ratio

def calculate_gini_coefficient(rainfall_ratios):
    """
    Calculate the GINI coefficient for the hourly to daily rainfall ratio.
    """
    
    rr_arr = rainfall_ratios.values
    exp, total_hours, y, x = rr_arr.shape
    assert total_hours % 24 == 0, "Total hours must be a multiple of 24 for daily aggregation."
    ndays = total_hours // 24

    axis = 2
    # Reshape the array to have dimensions (exp, ndays, y, x)
    rr_reshaped = rr_arr.reshape((exp, ndays, 24, y, x))
   
    arr_sorted = np.sort(np.nan_to_num(rr_reshaped, nan=0), axis=axis)
    n = np.sum(~np.isnan(rr_reshaped), axis=axis)
    sum_vals = np.nansum(rr_reshaped, axis=axis)
    sum_vals = np.where(sum_vals == 0, np.nan, sum_vals)

    
    r = np.arange(1, rr_reshaped.shape[axis] + 1)
    
    shape = [1] * rr_reshaped.ndim
    shape[axis] = rr_reshaped.shape[axis]
    r = r.reshape(shape)                         # shape = e.g. (1, 1, 24, 1, 1)
    n_broadcasted = np.expand_dims(n, axis=axis)  # e.g. (2, 3653, 1, 11, 11)

    numerator = np.sum((2 * r - n_broadcasted - 1) * arr_sorted, axis=axis)
    gini = numerator / (n * sum_vals)
    return gini


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
    Parameters:
    - ds_h_wet_days: Hourly dataset (masked for wet values).
    - ds_d: Daily dataset.

    - drain_bins: Bins for daily rainfall distribution.
    - hrain_bins: Bins for hourly rainfall distribution.
    Returns:
    - hourly_distribution_bin: Hourly rainfall distribution for each bin.
    - wet_hours_distribution_bin: Wet hours distribution for each bin.
    - samples_per_bin: Number of samples in each bin.

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

    return hourly_distribution_bin, wet_hours_distribution_bin,samples_per_bin
# ------------------------------------------------------------------
# Fast square-window sum (reflective borders) → used to merge the
# controls of the surrounding cells defined by `buffer`
def _window_sum(arr, radius):
    """
    Sliding-window *sum* over the last two spatial axes (ny, nx).
    Works for both 3-D and 4-D arrays by building the size tuple at run-time.
    """
    k = 2 * radius + 1
    size = [1] * (arr.ndim - 2) + [k, k]   # e.g. (1,1,k,k) or (1,k,k)
    return uniform_filter(arr, size=size, mode="nearest") * (k * k)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _fill_hourly_sample(out_hourly,
                        rain, bin_idx,
                        wet_cdf, hour_cdf,
                        hrain_edges,
                        rng_state,          # one Generator
                        iy0, iy1, ix0, ix1):

    n_t, ny, nx = rain.shape

    for iy in prange(iy0, iy1):
        for ix in prange(ix0, ix1):

            for t in range(n_t):
                R = rain[t, iy, ix]
                if R <= 0 or not np.isfinite(R):
                    continue

                b = bin_idx[t, iy, ix]
                if b < 0:                  # no training data
                    continue

                # ---- 1. wet-hour count ------------------------------------------
                cdf_wet = wet_cdf[b, :, iy, ix]        # 1-D (24,)
                Nh = np.searchsorted(cdf_wet, rng_state.random()) + 1  # 1–24

                # ---- 2. choose hourly-intensity bins ----------------------------
                cdf_hr  = hour_cdf[b, :, iy, ix]       # 1-D (n_hourly_bins,)
                idx_bins = np.empty(Nh, np.int64)
                for k in range(Nh):
                    idx_bins[k] = np.searchsorted(cdf_hr, rng_state.random())

                # ---- 3. draw intensities inside each bin ------------------------
                intens = np.empty(Nh, np.float32)
                for k in range(Nh):
                    lo  = hrain_edges[idx_bins[k]]
                    hi  = hrain_edges[idx_bins[k] + 1]
                    intens[k] = lo + (hi - lo) * rng_state.random()

                s_int = intens.sum()
                if s_int == 0.0:
                    continue
                values = R * intens / s_int

                # ---- 4. scatter into the 24-h vector -------------------------------
                # sample `Nh` unique hours without replacement (reservoir sampling)
                sel = np.empty(Nh, np.int64)
                filled = 0
                for hour in range(24):
                    remaining = 24 - hour
                    needed    = Nh - filled
                    if needed == 0:
                        break
                    # probability that we take this `hour`
                    if rng_state.random() < needed / remaining:
                        sel[filled] = hour
                        filled += 1

                tgt = out_hourly[t, iy, ix]          # view (24,)
                tgt[:] = 0.0
                for k in range(Nh):
                    tgt[sel[k]] = values[k]




# ------------------------------------------------------------------
def generate_hourly_synthetic(
        rain_daily: xr.DataArray,
        wet_hour_distribution_bin: np.ndarray,
        hourly_distribution_bin  : np.ndarray,
        samples_per_bin          : np.ndarray,
        hrain_bins               : np.ndarray,
        drain_bins               : np.ndarray,
        buffer: int = 1,
        n_samples: int = 100,
        seed: int | None = None,
) -> xr.DataArray:
    """
    Fast (Numba-accelerated) replacement – progress bar shows samples × points.
    """
    # ---------- 1.  training data → compound distributions ----------
    wet_hour_distribution_bin = np.nan_to_num(wet_hour_distribution_bin, nan=0.0)
    hourly_distribution_bin   = np.nan_to_num(hourly_distribution_bin , nan=0.0)

    weighted_wet_hour_distribution_bin = wet_hour_distribution_bin * samples_per_bin[:, np.newaxis, :, :]
    weighted_hourly_distribution_bin   = hourly_distribution_bin   * samples_per_bin[:, np.newaxis, :, :]

    comp_samp = _window_sum(samples_per_bin,          buffer)
    comp_wWet = _window_sum(weighted_wet_hour_distribution_bin, buffer)
    comp_wHr  = _window_sum(weighted_hourly_distribution_bin , buffer)

    comp_wWet /= comp_samp[:, None]
    comp_wHr  /= comp_samp[:, None]

    wet_cdf  = np.cumsum(comp_wWet , axis=1).astype(np.float32, order="C")
    hour_cdf = np.cumsum(comp_wHr  , axis=1).astype(np.float32, order="C")

    # ---------- 2.  drivers ----------
    rain_arr = rain_daily.data.astype(np.float32, order="C")
    bin_idx  = (np.digitize(rain_arr, drain_bins) - 1).astype(np.int16, order="C")

    n_time, ny, nx = rain_arr.shape
    hourly = np.zeros((n_samples, n_time, ny, nx, 24), dtype=np.float32)

    # inside-window bounds (skip the edge band defined by `buffer`)
    ix0, ix1 = buffer, nx - buffer
    iy0, iy1 = buffer, ny - buffer
    n_pts = (ix1 - ix0) * (iy1 - iy0)

    # thread-safe RNGs
    base_rng = np.random.default_rng(seed)
    rng_pool = [np.random.Generator(np.random.PCG64(base_rng.integers(2**32)))
                for _ in range(n_samples)]

    # ---------- 3.  main loop over samples with one tqdm -----------
    with tqdm(total=n_samples * n_pts, desc="samples × points") as bar:
        for s in range(n_samples):
            _fill_hourly_sample(hourly[s],
                                rain_arr, bin_idx,
                                wet_cdf[..., iy0:iy1, ix0:ix1],
                                hour_cdf[..., iy0:iy1, ix0:ix1],
                                hrain_bins.astype(np.float32),
                                rng_pool[s],
                                iy0, iy1, ix0, ix1)
            bar.update(n_pts)

    # ---------- 4.  wrap back into xarray --------------------------
    hourly = hourly.transpose(0, 1, 4, 2, 3)   # (sample, time, hour, y, x)
    return xr.DataArray(
        hourly,
        dims=("sample", "time", "hour", "y", "x"),
        coords=dict(
            sample=np.arange(n_samples),
            time=rain_daily.time,
            hour=np.arange(24),
            y=rain_daily.y,
            x=rain_daily.x,
        ),
    )



def save_probability_data(hourly_distribution_bin, 
                          wet_hours_distribution_bin, 
                          samples_per_bin, 
                          drain_bins, 
                          hrain_bins):
    
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
        dims   = ('experiment', 'drain_bin',     'y', 'x'),
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
    ds.to_netcdf("rainfall_probability_hourly_vs_daily.nc")
    return (ds)

# ###############################################################################
# ##### STEP 0: Load and prepare data
# ###############################################################################

start_time = time.time()

finp = xr.open_dataset(f'UIB_01H_RAIN_{y_idx}y-{x_idx}x_Present_large.nc')
finf = xr.open_dataset(f'UIB_01H_RAIN_{y_idx}y-{x_idx}x_Future_large.nc')

finp = finp.RAIN.where(finp.RAIN > WET_VALUE, 0.0)  
finf = finf.RAIN.where(finf.RAIN > WET_VALUE, 0.0)

ds = xr.concat([finp, finf], dim=pd.Index(['Present', 'Future'], name='exp'))

ds_h = ds.copy()
ds_d = ds_h.resample(time='1D').sum()
ds_d = ds_d.where(ds_d > WET_VALUE)

wet_days = ds_d > WET_VALUE
wet_days_hourly = wet_days.reindex(time=ds_h.time, method='ffill')
ds_h_wet_days = ds_h.where(wet_days_hourly)
ds_hx_wet_days = ds_h_wet_days.resample(time='1D').max()


# ###############################################################################
# ##### STEP 1: Calculate basic statistics
# ###############################################################################

# 1.1 Ratio of hourly to daily rainfall

rainfall_ratios = calculate_hourly_to_daily_ratio(ds_h, ds_d)

# 1.2 Calculate wet-hour fraction

wet_hour_fraction = ds_h_wet_days.where(ds_h_wet_days>0).resample(time='1D').count() / 24.0
wet_hour_fraction = wet_hour_fraction.where(wet_hour_fraction > 0.0)

# 1.3 Calculate quantiles

qtiles_h = ds_h_wet_days.where(ds_h_wet_days>0).quantile(qs, dim='time')
qtiles_d = ds_d.quantile(qs, dim='time')
qtiles_d_all = ds_d.quantile(qs, dim=['time','x','y'])
qtiles_hx = ds_hx_wet_days.quantile(qs, dim='time')

# 1.4 Calculate ratio of coincident extremes

ratio_coincident_extremes = calculate_ratio_coincident_extremes(ds_hx_wet_days, ds_d, qtiles_hx, qtiles_d, quantile=0.95)

print(ratio_coincident_extremes.values)
print(ratio_coincident_extremes.median(dim=['y','x']).values)


# 1.5 Calculate GINI coefficient for hourly rainfall ratio

gini_coeff = calculate_gini_coefficient(ds_h_wet_days)

print(f"Present: {np.nanmean(gini_coeff,axis=(1,2,3))[0]:.3f} \u00B1 {np.nanstd(gini_coeff,axis=(1,2,3))[0]:.3f} ")
print(f"Future: {np.nanmean(gini_coeff,axis=(1,2,3))[1]:.3f} \u00B1 {np.nanstd(gini_coeff,axis=(1,2,3))[1]:.3f} ")


# 1.6 Calcualte GINI coefficient only for extreme days

qt_days = ds_d > qtiles_d_all.sel(quantile=0.99)
qt_days_hourly = qt_days.reindex(time=ds_h.time, method='ffill')
qt_ds_h_masked = ds_h.where(qt_days_hourly)
gini_coeff_extreme = calculate_gini_coefficient(qt_ds_h_masked)

print(f"Present: {np.nanmean(gini_coeff_extreme,axis=(1,2,3))[0]:.3f} \u00B1 {np.nanstd(gini_coeff_extreme,axis=(1,2,3))[0]:.3f} ")
print(f"Future: {np.nanmean(gini_coeff_extreme,axis=(1,2,3))[1]:.3f} \u00B1 {np.nanstd(gini_coeff_extreme,axis=(1,2,3))[1]:.3f} ")

###############################################################################
##### STEP 2: Generate future synthetic future hourly data
###############################################################################


# 2.1 Calculate the wet hour intensity distribution and other statistics for hourly rainfall
# Generate future synthetic hourly data based on CTRL data (relationship between hourly and daily rainfall)
drain_bins = np.arange(0,105,5)
hrain_bins = np.concatenate((np.arange(0,10,1),np.arange(10,105,5)))

hourly_distribution_bin, wet_hours_distribution_bin,samples_per_bin = calculate_wet_hour_intensity_distribution (ds_h_wet_days,
                                                                                                 ds_d, 
                                                                                                 wet_hour_fraction,
                                                                                                 drain_bins = drain_bins, 
                                                                                                 hrain_bins = hrain_bins,  
                                                                                                 )
future_synthetic_ensemble = generate_hourly_synthetic(ds_d.sel(exp='Future'), 
                                                      wet_hours_distribution_bin[0,:].squeeze(),
                                                      hourly_distribution_bin[0,:].squeeze(),
                                                      samples_per_bin[0,:].squeeze(),
                                                      hrain_bins = hrain_bins,
                                                      drain_bins = drain_bins,
                                                      buffer = 5,  # mm
                                                      n_samples=100)
                                                        

rainfall_probability = save_probability_data(hourly_distribution_bin, wet_hours_distribution_bin, samples_per_bin, drain_bins, hrain_bins)

future_synthetic_ensemble = future_synthetic_ensemble.where(future_synthetic_ensemble>WET_VALUE).quantile(q=qs, dim='time', keep_attrs=True)
future_synthetic_ensemble.to_netcdf("synthetic_future_hourly_rainfall_new.nc")
end_time = time.time()
print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')
# ###############################################################################
# ##### STEP 3: Build a null model for PGW hours
# ###############################################################################

'''
Under the null hypothesis “sub-daily structure is unchanged”, an expected PGW hourly series can be generated by resampling CTRL hourly-fraction vectors and “pouring” each future day’s rain into them:

for each wet PGW day k:
    draw with replacement q* from {q(j)}CTRL
    P̂_h_i (k) = q*_i · P_d_PGW(k)   for i = 1,…,24
Do that N ≈ 500–1000 times to obtain a bootstrap ensemble of “CTRL-informed but PGW-scaled” hourly data.

This captures:

CTRL sub-daily variability
PGW daily totals (mean shift, variance change, seasonality)
Nothing else.'''


# 3.1 Prepare the data for the model

q_ctrl = rainfall_ratios.sel(exp='Present').stack(time_flat=('time',))

q_ctrl = q_ctrl.assign_coords(
    day=('time_flat', pd.to_datetime(rainfall_ratios.time.values).date),
    hour=('time_flat', pd.to_datetime(rainfall_ratios.time.values).hour)
)

q_ctrl = q_ctrl.set_index(time_flat=['day', 'hour']).unstack('time_flat').transpose('day', 'hour', 'y', 'x')


end_time = time.time()
print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')


# def main():
    
    









# ###############################################################################
# ##### __main__  scope
# ###############################################################################

# if __name__ == "__main__":

#     main()

