import xarray as xr
import numpy as np
import pandas as pd


import xarray as xr
import scipy.stats as stats    # SciPy ≥ 1.9 plays well with dask-arrays

def moments_above_quantiles(data: xr.DataArray,
                            qtiles: xr.DataArray,
                            *,
                            what=("mean", "var", "skew", "kurt"),
                            adim = 'time') -> xr.Dataset:
    """
    Parameters
    ----------
    data
        e.g. ds_h_masked_max  – dims ('exp','time','y','x')
    qtiles
        e.g. ds_h_masked_max_qtiles – dims ('quantile','exp','y','x')
    what
        Iterable of moments you want.
        Allowed names: 'mean', 'var', 'std', 'skew', 'kurt'
    Returns
    -------
    xr.Dataset
        dims ('quantile','exp','y','x') with one variable per requested moment
    """
    # 1. Broadcast quantile thresholds over the time axis
    #    Result dims: ('quantile','exp','time','y','x')
    above = data > qtiles                   # boolean mask
    masked = data.where(above)              # keep only values above each threshold
    out = {}

    if "mean" in what:
        out["mean"] = masked.mean(adim, skipna=True)

    if "var" in what or "std" in what:
        var = masked.var(adim, skipna=True, ddof=0)
        if "var" in what:
            out["var"] = var
        if "std" in what:
            out["std"] = var**0.5           # cheaper than .std() twice

    # Higher-order moments need SciPy; xr.apply_ufunc wires it to Dask & keeps metadata
    if "skew" in what:
        out["skew"] = xr.apply_ufunc(
            stats.skew,
            masked,
            input_core_dims=[[adim]],
            output_core_dims=[[]],
            kwargs={"axis": -1, "nan_policy": "omit", "bias": False},
            dask="parallelized",
            output_dtypes=[float],
        )

    if "kurt" in what:
        out["kurt"] = xr.apply_ufunc(
            stats.kurtosis,
            masked,
            input_core_dims=[[adim]],
            output_core_dims=[[]],
            kwargs={"axis": -1, "nan_policy": "omit", "fisher": True, "bias": False},
            dask="parallelized",
            output_dtypes=[float],
        )

    return xr.Dataset(out)


def gini_coefficient(arr, axis):
    """Compute Gini coefficient along given axis (e.g., hour=2), ignoring NaNs."""
    arr_sorted = np.sort(np.nan_to_num(arr, nan=0), axis=axis)
    n = np.sum(~np.isnan(arr), axis=axis)
    sum_vals = np.nansum(arr, axis=axis)
    sum_vals = np.where(sum_vals == 0, np.nan, sum_vals)

    r = np.arange(1, arr.shape[axis] + 1)

    # Broadcast r and n correctly
    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    r = r.reshape(shape)                         # shape = e.g. (1, 1, 24, 1, 1)
    n_broadcasted = np.expand_dims(n, axis=axis)  # e.g. (2, 3653, 1, 11, 11)

    numerator = np.sum((2 * r - n_broadcasted - 1) * arr_sorted, axis=axis)
    gini = numerator / (n * sum_vals)
    return gini

## Define constants

DAILY_WET_THRESHOLD = 0.1  # mm
HOURLY_WET_THRESHOLD = 0.1  # mm

y_idx=258; x_idx=559 #1.01 km away from Palma University station 
#y_idx=250; x_idx=423 # 0.76 km away from Turis
#y_idx=384; x_idx=569 #  Spanish Med Pyrenees

## First we load present and future datasets and combine them

finp = xr.open_dataset(f'UIB_01H_RAIN_{y_idx}y-{x_idx}x_Present_large.nc')
finf = xr.open_dataset(f'UIB_01H_RAIN_{y_idx}y-{x_idx}x_Future_large.nc')

finp = finp.RAIN.where(finp.RAIN > HOURLY_WET_THRESHOLD, 0.0)  
finf = finf.RAIN.where(finf.RAIN > HOURLY_WET_THRESHOLD, 0.0)  

ds = xr.concat([finp, finf], dim=pd.Index(['Present', 'Future'], name='exp'))

#ds = ds.sel(time=ds.time.dt.month.isin([9, 10, 11] ))


## Create daily statistics

ds_h = ds.copy()
ds_d = ds_h.resample(time='1D').sum()
ds_d = ds_d.where(ds_d > DAILY_WET_THRESHOLD)


# Create a mask for wet days in the daily dataset
# and reindex it to match the hourly dataset
# This will allow us to mask the hourly dataset accordingly

wet_days = ds_d > DAILY_WET_THRESHOLD
wet_days_hourly = wet_days.reindex(time=ds_h.time, method='ffill')
ds_h_masked = ds_h.where(wet_days_hourly)

ds_h_masked_max = ds_h_masked.resample(time='1D').max() # Maximum hourly rainfall per day
ds_h_masked_mean = ds_h_masked.where(ds_h_masked>0).resample(time='1D').mean() # Mean hourly rainfall per day - only wet hours
wet_hour_fraction = ds_h_masked.where(ds_h_masked>0).resample(time='1D').count() / 24.0 
wet_hour_fraction = wet_hour_fraction.where(wet_hour_fraction > 0.0) # Fraction of wet hours per day - only wet days

qtiles = [50, 75, 90, 95, 99]

ds_h_masked_max_qtiles = ds_h_masked_max.quantile(q=np.array(qtiles) / 100.0, dim='time')
ds_h_masked_qtiles = ds_h_masked.where(ds_h_masked>0).quantile(q=np.array(qtiles) / 100.0, dim='time')
ds_h_masked_mean_qtiles = ds_h_masked_mean.quantile(q=np.array(qtiles) / 100.0, dim='time')
ds_d_masked_qtiles = ds_d.quantile(q=np.array(qtiles) / 100.0, dim='time')


# mom_h_max = moments_above_quantiles(
#     ds_h_masked_max, ds_h_masked_max_qtiles,
# )

# mom_h = moments_above_quantiles(
#     ds_h_masked, ds_h_masked_qtiles,
# )

# mom_d = moments_above_quantiles(
#     ds_d, ds_d_masked_qtiles                    # same structure as -h- pair
# )

## Reshape the hourly dataset to meet the structure days, hours, y, x
## The array will be masked to only include wet days

arr_h_masked = ds_h_masked.values
exp, total_hours, y, x = arr_h_masked.shape
assert total_hours % 24 == 0, "Time dimension is not divisible by 24"
ndays = total_hours // 24
ds_h_daily = arr_h_masked.reshape((exp, ndays, 24, y, x))


gini_daily = xr.apply_ufunc(
    gini_coefficient,
    ds_h_daily,
    input_core_dims=[["hour"]],
    output_core_dims=[[]],
    vectorize=True,
    kwargs={"axis": 2},  # 2 = hour axis
    dask="parallelized",  # only if using Dask
    output_dtypes=[float],
)


# Check what percentage of daily quantiles coincide with the hourly ones

# ── inputs ──────────────────────────────────────────────────────────────
# ds_h_masked_max              dims: (exp, time, y, x)    ← daily-hourly maxima
# ds_d_masked_max              dims: (exp, time, y, x)    ← daily maxima
# ds_h_masked_max_qtiles       dims: (quantile, exp, y, x)
# ds_d_masked_max_qtiles       dims: (quantile, exp, y, x)

# ── 1. identify exceedances ────────────────────────────────────────────

h_exc = ds_h_masked_max  > ds_h_masked_max_qtiles          # (quantile,time,exp,y,x)
d_exc = ds_d  > ds_d_masked_qtiles          # (quantile,time,exp,y,x)

# ── 2. coincidence mask ───────────────────────────────────────────────
both_exc = h_exc & d_exc                                   # True where both exceed

# ── 3. percentages ────────────────────────────────────────────────────
# % of hourly exceedances that also exceed the daily threshold
pct_of_hourly = 100 * both_exc.sum("time") / h_exc.sum("time")

# % of daily exceedances that also exceed the hourly threshold
pct_of_daily  = 100 * both_exc.sum("time") / d_exc.sum("time")

# ── 4. tidy up (optional) ──────────────────────────────────────────────
# Avoid divide-by-zero warnings and leave NaN where there are no exceedances:
pct_of_hourly = pct_of_hourly.where(np.isfinite(pct_of_hourly))
pct_of_daily  = pct_of_daily.where(np.isfinite(pct_of_daily))

# Result: DataArrays with dims (quantile, exp, y, x)
pct_of_hourly.name = "pct_hourly_events_also_daily"
pct_of_daily.name  = "pct_daily_events_also_hourly"


# ── 5. Example: how many hourly events exceed the daily threshold? ──────
# This is the maximum number of hourly events that exceed the quantile threshold in a single day
# Thresholds are calculated from the daily maxima

h_exc_all = ds_h_masked > ds_h_masked_max_qtiles
h_exc_all.resample(time='1D').sum().max('time').max(dim=['x','y'])

print("Percentage of hourly events that also exceed the daily threshold:"
      f"\n{pct_of_daily.median(dim=['x','y'])}\n")
print("Maximum number of hourly events that exceed the quantile threshold in a single day:")
print(h_exc_all.resample(time='1D').sum().max('time').max(dim=['x','y']))

#####################################################################
#####################################################################

## Analysis by region: statistics use the entire region, not just the point

# A reference of how to calculate the daily statistics from the hourly dataset
# ds_h = ds.copy()
# ds_d = ds_h.resample(time='1D').sum()
# ds_d = ds_d.where(ds_d > DAILY_WET_THRESHOLD)
# wet_days = ds_d > DAILY_WET_THRESHOLD
# wet_days_hourly = wet_days.reindex(time=ds_h.time, method='ffill')
# ds_h_masked = ds_h.where(wet_days_hourly)
# ds_h_masked_max = ds_h_masked.resample(time='1D').max() # Maximum hourly rainfall per day
# ds_h_masked_mean = ds_h_masked.where(ds_h_masked>0).resample(time='1D').mean() # Mean hourly rainfall per day - only wet hours
# wet_hour_fraction = ds_h_masked.where(ds_h_masked>0).resample(time='1D').count() / 24.0 
# wet_hour_fraction = wet_hour_fraction.where(wet_hour_fraction > 0.0) # Fraction of wet hours per day - only wet days

ds_h_masked_max_qtiles_all = ds_h_masked_max.quantile(q=np.array(qtiles) / 100.0, dim=['time','x','y'])
ds_h_masked_qtiles_all = ds_h_masked.where(ds_h_masked > 0).quantile(q=np.array(qtiles) / 100.0, dim=['time','x','y'])
ds_d_masked_qtiles_all = ds_d.quantile(q=np.array(qtiles) / 100.0, dim=['time','x','y'])  

mom_h_max = moments_above_quantiles(
    ds_h_masked_max.stack(events=("time", "y", "x")).dropna(dim='events'), ds_h_masked_max_qtiles_all, adim='events'
)

mom_h = moments_above_quantiles(
    ds_h_masked.stack(events=("time", "y", "x")).dropna(dim='events'), ds_h_masked_qtiles_all,adim='events'
)

mom_d = moments_above_quantiles(
    ds_d.stack(events=("time", "y", "x")).dropna(dim='events'), ds_d_masked_qtiles_all,adim='events'              # same structure as -h- pair
)

# Calculate Gini coefficients for the days where daily rainfall is above quantile

qt_days = ds_d > ds_d_masked_qtiles_all.sel(quantile=0.99)  # Select days above the 99th quantile
qt_days_hourly = qt_days.reindex(time=ds_h.time, method='ffill')
qt_ds_h_masked = ds_h.where(qt_days_hourly)

# qt_ds_h_masked.count(dim=['time','y','x'])
# This can be very different for each experiment! 

arr_h_masked = qt_ds_h_masked.values

exp, total_hours, y, x = arr_h_masked.shape
assert total_hours % 24 == 0, "Time dimension is not divisible by 24"
ndays = total_hours // 24
ds_h_daily = arr_h_masked.reshape((exp, ndays, 24, y, x))


gini_daily_reg = xr.apply_ufunc(
    gini_coefficient,
    ds_h_daily,
    input_core_dims=[["hour"]],
    output_core_dims=[[]],
    vectorize=True,
    kwargs={"axis": 2},  # 2 = hour axis
    dask="parallelized",  # only if using Dask
    output_dtypes=[float],
)

# This is telling how inequally the rainfall is distributed across the hours of the day
print(f"Mean Gini coefficient (ow inequally the rainfall is distributed across the hours of the day):")
print(f"Present: {np.nanmean(gini_daily_reg,axis=(1,2,3))[0]:.3f} \u00B1 {np.nanstd(gini_daily_reg,axis=(1,2,3))[0]:.3f} ")
print(f"Future: {np.nanmean(gini_daily_reg,axis=(1,2,3))[1]:.3f} \u00B1 {np.nanstd(gini_daily_reg,axis=(1,2,3))[1]:.3f} ")


import pdb; pdb.set_trace()  # fmt: skip
