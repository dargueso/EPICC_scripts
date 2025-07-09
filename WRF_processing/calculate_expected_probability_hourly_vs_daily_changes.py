import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import scipy.stats as stats    # SciPy ≥ 1.9 plays well with dask-arrays
from dask import delayed
from tqdm.auto import trange 
import matplotlib.pyplot as plt


#from copulas.bivariate import Clayton, Gumbel

from scipy.stats import rankdata,norm,spearmanr



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

def generate_hourly_distribution(
    R_day: float,
    wet_hour_probs: np.ndarray,         # shape (24,), sum = 1
    intensity_probabilities: np.ndarray,  # shape (n_bins,), sum = 1
    hourly_bins: np.ndarray             # shape (n_bins + 1,), bin edges
) -> np.ndarray:
    """
    Generate a stochastic 24-hour rainfall distribution for a given daily total.

    Parameters
    ----------
    R_day : float
        Total daily rainfall (mm).
    wet_hour_probs : np.ndarray
        Probabilities for number of wet hours (1 to 24), shape (24,).
    intensity_probabilities : np.ndarray
        Probabilities for hourly rainfall intensities (e.g., 0.1–50 mm/h), shape (n_bins,).
    hourly_bins : np.ndarray
        Bin edges corresponding to intensity_probabilities, shape (n_bins + 1,).

    Returns
    -------
    hourly_rain : np.ndarray
        24-element array with hourly rainfall values summing to R_day.
    """
    # 1. Sample number of wet hours: N in [1, 24]
    N_wet = np.random.choice(np.arange(1, 25), p=wet_hour_probs)

    # 2. Sample N_wet intensity values from the histogram bins
    bin_choices = np.random.choice(len(intensity_probabilities), size=N_wet, p=intensity_probabilities)
    # For each bin, sample uniformly within the bin range
    intensities = np.array([
        np.random.uniform(low=hourly_bins[b], high=hourly_bins[b+1]) for b in bin_choices
    ])

    # 3. Normalize intensities and scale to R_day
    fractions = intensities / intensities.sum()
    wet_values = fractions * R_day

    # 4. Randomly assign wet_values to 24-hour slots
    hourly_rain = np.zeros(24)
    wet_slots = np.random.choice(24, size=N_wet, replace=False)
    hourly_rain[wet_slots] = wet_values

    return hourly_rain


def generate_hourly_ensemble(
    rain_daily: xr.DataArray,
    wet_hour_distribution_bin: np.ndarray,     # (n_daily_bins, 24)
    hourly_distribution_bin: np.ndarray,       # (n_daily_bins, n_hourly_bins)
    hourly_bins: np.ndarray,                   # (n_hourly_bins + 1,)
    daily_bin_edges: np.ndarray,               # (n_daily_bins + 1,)
    n_samples: int
) -> xr.DataArray:
    """
    Generate an ensemble of stochastic hourly rainfall distributions per daily rainfall event.

    Parameters
    ----------
    rain_daily : xr.DataArray
        Daily precipitation (time,) or (time, y, x), with NaNs on dry days.
    wet_hour_distribution_bin : np.ndarray
        Probabilities for number of wet hours per daily bin (n_daily_bins, 24).
    hourly_distribution_bin : np.ndarray
        Hourly intensity probabilities per daily bin (n_daily_bins, n_hourly_bins).
    hourly_bins : np.ndarray
        Edges of intensity bins (n_hourly_bins + 1).
    daily_bin_edges : np.ndarray
        Edges of daily rainfall bins (n_daily_bins + 1).
    n_samples : int
        Number of synthetic realizations to generate per day.

    Returns
    -------
    xr.DataArray
        Hourly rainfall (sample, time, hour) or (sample, time, hour, y, x).
    """
    shape = rain_daily.shape
    dims = rain_daily.dims
    coords = rain_daily.coords
    is_3d = len(shape) == 3
    # Flatten for easier processing
    # Flatten for easier processing
    if is_3d:
        rain_flat = rain_daily.stack(sample_dim=dims[1:]).values  # (time, n_points)
    else:
        rain_flat = rain_daily.values[:, np.newaxis]              # (time, 1)

    n_time, n_points = rain_flat.shape
    hourly_output = np.zeros((n_samples, n_time, 24, n_points), dtype=np.float32)

    # Digitize daily rainfall to bins
    bin_indices = np.digitize(rain_flat, daily_bin_edges) - 1  # shape (time, n_points)
    # Iterate through each sample
    for s in range(n_samples):
        for t in range(n_time):
            for p in range(n_points):
                R = rain_flat[t, p]
                if np.isnan(R) or R <= 0:
                    continue

                b = bin_indices[t, p]
                if b < 0 or b >= wet_hour_distribution_bin.shape[0]:
                    continue

                # 1. Sample number of wet hours
                Nh = np.random.choice(np.arange(1, 25), p=wet_hour_distribution_bin[b])

                # 2. Sample Nh intensity values from hourly PDF
                intensity_probs = hourly_distribution_bin[b]
                idx_bins = np.random.choice(len(intensity_probs), size=Nh, p=intensity_probs)
                if np.any(idx_bins >= len(hourly_bins) - 1):
                    import pdb; pdb.set_trace()  # fmt: skip
                intensities = np.random.uniform(low=hourly_bins[idx_bins],high=hourly_bins[idx_bins + 1])

                # 3. Normalize & scale
                fractions = intensities / np.sum(intensities)
                values = R * fractions

                # 4. Assign to random hours
                hours = np.zeros(24, dtype=np.float32)
                slots = np.random.choice(24, size=Nh, replace=False)
                hours[slots] = values

                hourly_output[s, t, :, p] = hours

    # Unstack back if needed
    if is_3d:
        coords_new = {
            "sample": np.arange(n_samples),
            "time": rain_daily.coords["time"],
            "hour": np.arange(24),
            **{dim: coords[dim] for dim in dims[1:]}
        }
        hourly_output = hourly_output.reshape((n_samples, n_time, 24, *shape[1:]))
        return xr.DataArray(hourly_output, dims=("sample", "time", "hour", *dims[1:]), coords=coords_new)
    else:
        coords_new = {
            "sample": np.arange(n_samples),
            "time": rain_daily.coords["time"],
            "hour": np.arange(24)
        }
        return xr.DataArray(hourly_output[:, :, :, 0], dims=("sample", "time", "hour"), coords=coords_new)

def empirical_cdf(x):
    return rankdata(x, method='average') / (len(x) + 1)


def calculate_ratio_simultaneous_extremes(ds_h_dmax,ds_d):
    """
    Calculate the percentage of times that both the hourly daily max 
    and daily rainfall exceed their respective quantiles the same day
    """

    tot_h = ds_h_dmax.count(dim='time')  # Total number of days with hourly max above quantile
    tot_d = ds_d.count(dim='time')  # Total number of days with daily rainfall above quantile

    both = (ds_h_dmax > 0) & (ds_d > 0)  # Both exceed the quantile
    both_tot = both.sum(dim='time')  # Total number of days where both exceed the quantile

    ratio = both_tot / (tot_h)  # Percentage of days where both exceed the quantile

    if np.all(tot_h != tot_d):
        raise ValueError("Total number of days with hourly max and daily rainfall above quantile do not match.")
    
    return ratio

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

#####################################################################
#####################################################################

## STEP 5 

qs = xr.DataArray([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dims='q', name='quantile')

P_d_CTRL = ds_d.sel(exp='Present').where(ds_d.sel(exp='Present') > DAILY_WET_THRESHOLD)  # Mask present daily data
P_d_PGW  = ds_d.sel(exp='Future').where(ds_d.sel(exp='Future') > DAILY_WET_THRESHOLD)  # Mask future daily data
P_h_CTRL = ds_h.sel(exp='Present').where(ds_h.sel(exp='Present') > HOURLY_WET_THRESHOLD)  # Mask present hourly data
P_h_PGW  = ds_h.sel(exp='Future').where(ds_h.sel(exp='Future') > HOURLY_WET_THRESHOLD)  # Mask future hourly data

Qd_CTRL = P_d_CTRL.quantile(qs, dim='time')      # (q, y, x) or area-mean
Qd_PGW  = P_d_PGW .quantile(qs, dim='time')
Qh_CTRL = P_h_CTRL.quantile(qs, dim='time')
Qh_PGW  = P_h_PGW .quantile(qs, dim='time')

Cd = Qd_PGW / Qd_CTRL      # daily change factors
Ch = Qh_PGW / Qh_CTRL      # hourly change factors


NBOOT = 750            # 500–1000 is typical
rng   = np.random.default_rng(seed=404)

def _sample(da):
    """Resample with replacement along the first dim (time/day)."""
    n = da.sizes['time']
    idx = xr.DataArray(rng.integers(0, n, n), dims='time')
    return da.isel(time=idx)

# # pre-allocate arrays: (boot, q)
# ny, nx = P_d_CTRL.sizes['y'], P_d_CTRL.sizes['x']

# Cd_boot = np.empty((NBOOT, qs.size, ny, nx), np.float32)
# Ch_boot = np.empty_like(Cd_boot)

# for b in range(NBOOT):
#     Cd_boot[b] = (
#         _sample(P_d_PGW ).quantile(qs, dim='time') /
#         _sample(P_d_CTRL).quantile(qs, dim='time')
#     ).values                                 # shape (8, ny, nx)

#     Ch_boot[b] = (
#         _sample(P_h_PGW ).quantile(qs, dim='time') /
#         _sample(P_h_CTRL).quantile(qs, dim='time')
#     ).values

Cd_boot = np.empty((NBOOT, qs.size), np.float32)
Ch_boot = np.empty_like(Cd_boot)

for b in trange(NBOOT, desc="Bootstrap draws", unit="draw"):
    Cd_boot[b] = (
        _sample(P_d_PGW ).quantile(qs, dim='time') /
        _sample(P_d_CTRL).quantile(qs, dim='time')
    ).mean(dim=('y', 'x')).values         # ← spatial mean here

    Ch_boot[b] = (
        _sample(P_h_PGW ).quantile(qs, dim='time') /
        _sample(P_h_CTRL).quantile(qs, dim='time')
    ).mean(dim=('y', 'x')).values


# convert to xarray for convenience
Cd_boot = xr.DataArray(Cd_boot, dims=('boot','q'), coords={'q':qs})
Ch_boot = xr.DataArray(Ch_boot, dims=('boot','q'), coords={'q':qs})


# 95 % envelopes
Cd_lo, Cd_hi = Cd_boot.quantile([.025,.975], dim='boot')
Ch_lo, Ch_hi = Ch_boot.quantile([.025,.975], dim='boot')


fig, ax = plt.subplots(figsize=(6, 4))

# ribbons first so lines sit on top
ax.fill_between(qs, Cd_lo, Cd_hi, alpha=.3, label='Daily 95 % CI')
ax.fill_between(qs, Ch_lo, Ch_hi, alpha=.3, label='Hourly 95 % CI')

ax.plot(qs, Cd, marker='o',  lw=2, label='Daily change factor')
ax.plot(qs, Ch, marker='s',  lw=2, label='Hourly change factor')

ax.set_xlabel('Quantile q')
ax.set_ylabel('PGW / CTRL')
ax.set_xscale('linear')          # or 'log' if you prefer
ax.grid(True, ls=':')
ax.legend(frameon=False, fontsize=9)
ax.set_title('Quantile-Scaling Quick Check')

plt.tight_layout()
plt.show()




# Calculate ratio of daily total rainfall in hourly data

daily_expanded = ds_d.reindex_like(ds_h, method='ffill')
rainfall_ratios = ds_h / daily_expanded  
#rainfall_ratios = rainfall_ratios.fillna(0) 

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

sameday_ext_ratio = calculate_ratio_simultaneous_extremes(ds_h_masked_max.where(ds_h_masked_max > ds_h_masked_max_qtiles.sel(quantile =0.95)),ds_d.where(ds_d > ds_d_masked_qtiles.sel(quantile=0.95)))

print(sameday_ext_ratio.values)
print(sameday_ext_ratio.median(dim=['y','x']).values)

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


combqtiles = np.arange(0,101,1) / 100.0  # Combine quantiles for the analysis
rainfall_bins = np.arange(0, 55, 5)
hourly_rainfall_bins = np.arange(0, 105, 5)  # Bins for hourly rainfall
bin_gini_mean = np.zeros((len(rainfall_bins), 2))  # Two experiments: Present and Future
wet_hours_fraction = np.zeros((len(rainfall_bins), 2))  # Two experiments: Present and Future
hourly_mean = np.zeros((len(rainfall_bins), 2))  # Two experiments: Present and Future
samples_per_bin = np.zeros((len(rainfall_bins), 2))  # Two experiments: Present and Future
quantiles_bin = np.zeros((len(rainfall_bins), len(combqtiles), 2))  # Two experiments: Present and Future
hourly_distribution_bin = np.zeros((len(rainfall_bins),len(hourly_rainfall_bins)-1,2))  # Two experiments: Present and Future
wet_hours_distribution_bin = np.zeros((len(rainfall_bins), 24, 2))  # Two experiments: Present and Future

for ibin in range(len(rainfall_bins)):

    if ibin == len(rainfall_bins)-1:
        upper_bound = np.inf
    else:
        lower_bound = rainfall_bins[ibin]
        upper_bound = rainfall_bins[ibin + 1]

    bin_days = (ds_d >= lower_bound) & (ds_d < upper_bound)  
    bin_days_hourly = bin_days.reindex(time=ds_h.time, method='ffill')
    bin_ds_h_masked = ds_h.where(bin_days_hourly)

    hourly_mean[ibin,:] = bin_ds_h_masked.mean(dim=['time','x','y'])
    wet_hours = bin_ds_h_masked.where(bin_ds_h_masked > 0).count(dim=['time','x','y'])
    wet_hours_fraction[ibin,:] = wet_hours / bin_days_hourly.sum(dim=['time','x','y'])

    bin_gini = gini_daily.copy()
    bin_gini[~bin_days.values] = np.nan  # Mask out the days that are not in the bi
    bin_gini_mean[ibin,:] = np.nanmean(bin_gini, axis=(1, 2, 3))

    samples_per_bin[ibin, :] = np.sum(bin_days.values, axis=(1, 2, 3))  # Count of days in the Present experiment
    quantiles_bin [ibin, :, :] = bin_ds_h_masked.where(bin_ds_h_masked > 0).quantile(q=combqtiles, dim=['time', 'x', 'y']).values
    for iexp in range(2):  # Two experiments: Present and Future
        if np.sum(bin_days.values[iexp, :, :, :]) > 0:
            hourly_distribution_bin[ibin, :, iexp] = np.histogram(bin_ds_h_masked.where(bin_ds_h_masked > 0).isel(exp=iexp),bins=hourly_rainfall_bins,density=1)[0]*np.diff(hourly_rainfall_bins)
            wet_hours_distribution_bin [ibin, :, iexp] = np.histogram(wet_hour_fraction.where(bin_days).isel(exp=iexp)*24., bins= np.arange(1, 26, 1),density=True)[0]


# Rd = 17.5
# bin_index = np.digitize(Rd, rainfall_bins)-1
# wet_hours_prob = wet_hours_distribution_bin[bin_index, :, 0]
# hour_intensity_prob = hourly_distribution_bin[bin_index, :, 0]  # Present experiment
# future_synthetic = generate_hourly_distribution(Rd, wet_hours_prob, hour_intensity_prob, hourly_rainfall_bins)


#test = ds_d.isel(exp=0,x=25,y=25,time=slice(0,3))
#future_synthetic_ensemble = generate_hourly_ensemble(test, wet_hours_distribution_bin[:, :, 0].squeeze(), hourly_distribution_bin[:, :, 0].squeeze(), hourly_rainfall_bins, rainfall_bins,1)  # Reshape to (1, 24) for consistency
future_synthetic_ensemble = generate_hourly_ensemble(ds_d.sel(exp='Future'), wet_hours_distribution_bin[:, :, 0].squeeze(), hourly_distribution_bin[:, :, 0].squeeze(), hourly_rainfall_bins, rainfall_bins,10) # Reshape to (1, 24) for consistency


#####################################################################
#####################################################################

# Calculating Copulas statistics

D_xy = ds_d.sel(exp='Present', y=25, x=25).values
H_xy = ds_h_masked_max.sel(exp='Present', y=25, x=25).values

D_xy = D_xy[~np.isnan(D_xy)]
H_xy = H_xy[~np.isnan(H_xy)]

u = empirical_cdf(D_xy)
v = empirical_cdf(H_xy)

# Estimate copula parameter (Gaussian copula → correlation in normal space)
# Use Spearman's rho as a rank-based proxy, then convert to Pearson
rho_s, _ = spearmanr(u, v)
rho = 2 * np.sin(np.pi * rho_s / 6)

z_u = norm.ppf(u)
z_v = norm.ppf(v)

D_xy_future = ds_d.sel(exp='Future', y=25, x=25).values
H_xy_future = ds_h_masked_max.sel(exp='Future', y=25, x=25).values

D_xy_future = D_xy_future[~np.isnan(D_xy_future)]
H_xy_future = H_xy_future[~np.isnan(H_xy_future)]

uf = empirical_cdf(D_xy_future)
vf = empirical_cdf(H_xy_future)

z_uf = norm.ppf(uf)
z_vf = norm.ppf(vf)

mu_cond = rho * z_uf
std_cond = np.sqrt(1 - rho**2)
z_diff = (z_vf - mu_cond) / std_cond

p_exceed = 1 - norm.cdf(z_diff)  # shape: (n_time,)

# import pdb; pdb.set_trace()  # fmt: skip


# # Estimate expected v for future P_D values
# D_xy_future = ds_d.sel(exp='Future', y=5, x=5).values
# H_xy_future = ds_h_masked_max.sel(exp='Future', y=5, x=5).values

# D_xy_future = D_xy_future[~np.isnan(D_xy_future)]
# H_xy_future = H_xy_future[~np.isnan(H_xy_future)]

# u_future = np.searchsorted(np.sort(D_xy), D_xy_future) / (len(D_xy) + 1)
# v_future = np.searchsorted(np.sort(H_xy), H_xy_future) / (len(H_xy) + 1)
# expected_v = []

# tolerance = 0.005

# for uf in u_future:
#     idx = np.where(np.abs(u - uf) < tolerance)[0]    # Tolerance of 0.02
#     if len(idx) > 0:
#         expected_v.append(np.mean(v[idx]))
#     else:
#         expected_v.append(np.nan)

    
#####################################################################
#####################################################################

# Build a null model for PGW hours

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

#####################################################################
#####################################################################

# ---------------------------------------------------------------------
# 0.  Handy aliases (adapt to your real names) -------------------------
# ---------------------------------------------------------------------
q_ctrl = rainfall_ratios.sel(exp='Present').stack(time_flat=('time',))
q_ctrl = q_ctrl.assign_coords(
    day=('time_flat', pd.to_datetime(rainfall_ratios.time.values).date),
    hour=('time_flat', pd.to_datetime(rainfall_ratios.time.values).hour)
)
q_ctrl = q_ctrl.set_index(time_flat=['day', 'hour']).unstack('time_flat').transpose('day', 'hour', 'y', 'x')  
                                                                        # (day, hour, y, x)  float32
daily_pgw = ds_d.sel(exp='Future').rename({'time': 'day'})              # (day, y, x) masked ≤0.1 mm

daily_pgw_hour = daily_pgw.broadcast_like(q_ctrl)   # (day,h,y,x)
r_ctrl         = ds_h_masked_max                    # (day,y,x)    CTRL r
h_pgw          = ds_h.sel(exp='Future')             # (time,y,x)
r_pgw          = (h_pgw.groupby('time.day')
                       .max()                       # daily max hour total
                 / ds_d.sel(exp='Future'))          # / daily total

NBOOT   = 500
SUBSAMP = 50_000            # for KS/AD
rng     = np.random.default_rng(123)
nday = min(q_ctrl.sizes['day'], daily_pgw.sizes['day'])

# ---------------------------------------------------------------------
# 1.  Reference samples that never change -----------------------------
# ---------------------------------------------------------------------
h_pgw_all = h_pgw.stack(t=('time','y','x')).dropna('t')
idx_ref   = rng.choice(h_pgw_all.sizes['t'], size=SUBSAMP, replace=False)
sample_ref = h_pgw_all.isel(t=idx_ref).values.astype('float32')

# ---------------------------------------------------------------------
# 2.  Containers for bootstrap statistics -----------------------------
# ---------------------------------------------------------------------
boot_q99        = []
boot_q999       = []
boot_wetfreq    = []
boot_r_median   = []
boot_r_q95      = []
boot_ks         = []
boot_ad         = []

# ---------------------------------------------------------------------
# 3.  Bootstrap loop ---------------------------------------------------
# ---------------------------------------------------------------------
for b in range(NBOOT):
    # (i)  resample CTRL days
    day_idx = rng.integers(0, nday, size=nday)
    q_samp = (q_ctrl.isel(day=day_idx).assign_coords(day=('day', np.arange(nday))))
    pgw_samp = (daily_pgw_hour.assign_coords(day=q_samp.day))

    # (ii) expected hourly PGW field
    p_exp_h = q_samp * pgw_samp                      # lazy

    # (iii) METRIC 1 – extreme quantiles ------------------
    boot_q99 .append(p_exp_h.quantile(0.99 , dim=('day','hour')))
    boot_q999.append(p_exp_h.quantile(0.999, dim=('day','hour')))

    # (iv)  METRIC 2 – wet-hour frequency -----------------
    nhours_wet = p_exp_h.notnull().sum(dim=('day','hour'))
    nhours_all = xr.full_like(nhours_wet, p_exp_h.sizes['day']*24)
    boot_wetfreq.append( (nhours_wet / nhours_all).astype('float32') )
    # (v)   METRIC 3 – peak-hour ratio stats --------------
    r_boot = (p_exp_h.max('hour') / daily_pgw_hour.assign_coords(day=q_samp.day)).astype('float32')
    boot_r_median.append( r_boot.median(dim='day') )
    boot_r_q95.append( r_boot.quantile(0.95 , dim='day') )

    # (vi)  METRIC 4 – whole-CDF KS / AD ------------------
    flat   = p_exp_h.stack(t=('day','hour','y','x')).dropna('t')
    idx    = rng.choice(flat.sizes['t'], size=SUBSAMP, replace=False)
    sample = flat.isel(t=idx).values.astype('float32')
    ks_D, _  = stats.ks_2samp(sample_ref, sample)
    ad_stat  = stats.anderson_ksamp([sample_ref, sample]).statistic
    boot_ks.append(ks_D)
    boot_ad.append(ad_stat)

# ---------------------------------------------------------------------
# 4.  Convert lists → xarray / numpy for easy maths -------------------
# ---------------------------------------------------------------------
boot_q99     = xr.concat(boot_q99    , dim='boot')
boot_q999    = xr.concat(boot_q999   , dim='boot')
boot_wetfreq = xr.concat(boot_wetfreq, dim='boot')
boot_r_med   = xr.concat(boot_r_median, dim='boot')
boot_r_q95   = xr.concat(boot_r_q95  , dim='boot')
boot_ks      = np.asarray(boot_ks)
boot_ad      = np.asarray(boot_ad)


# Hourly quantiles
q99_pgw  = h_pgw.quantile(0.99 , dim='time')
q999_pgw = h_pgw.quantile(0.999, dim='time')

# Wet-hour frequency
wetfreq_pgw = h_pgw.notnull().sum(dim='time') / h_pgw.sizes['time']

# Peak-hour ratio stats
r_med_pgw = r_pgw.median(dim='day')
r_q95_pgw = r_pgw.quantile(0.95 , dim='day')

# Whole-CDF metrics were done in loop: ks_real, ad_real
ks_real = boot_ks[-1]     # by construction: last slot = PGW itself
ad_real = boot_ad[-1]
boot_ks = boot_ks[:-1]    # drop the PGW entry for envelope calc
boot_ad = boot_ad[:-1]

lo = boot_q99.quantile(0.025, dim='boot')
hi = boot_q99.quantile(0.975, dim='boot')
sig_q99 = (q99_pgw < lo) | (q99_pgw > hi)          # boolean (y,x)

#Do the same for q999, wetfreq, r_med, r_q95.

p_ks = (np.sum(boot_ks >= ks_real) + 1) / (NBOOT + 1)
p_ad = (np.sum(boot_ad >= ad_real) + 1) / (NBOOT + 1)



print(f"KS statistic = {ks_real:.4f}  "
      f"(95 % envelope {lo:.4f}–{hi:.4f})  "
      f"=> {'COMPATIBLE' if compatible else 'NOT compatible'}")

import pdb; pdb.set_trace()  # fmt: skip


