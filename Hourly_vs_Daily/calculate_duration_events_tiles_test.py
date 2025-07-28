#!/usr/bin/env python
'''
@File    :  calculate_duration_events.py
@Time    :  2025/07/21 18:48:29
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  Hourly vs Daily Precipitation
@Desc    :  Detect events and calculate their duration
'''


import xarray as xr
import numpy as np
import time as ttime
import config as cfg
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from itertools import product

# ---------------------------------------------------------------------
# User-configurable section
# ---------------------------------------------------------------------

# Pattern of your pre-split tile files  (edit if the path changes)

tile_size   = 50          # number of native gridpoints per tile
N_JOBS      = 10         # parallel workers (set 1 to run serially)



# ---------------------------------------------------------------------
# Calculation of spell lengths
# ---------------------------------------------------------------------


# ────────────────────────────────────────────────────────────────────
# Optional: Numba for C‑speed.  We fall back gracefully if it’s absent.
# ────────────────────────────────────────────────────────────────────
try:
    from numba import njit
except ImportError:                # keep the exact call signature
    def njit(*args, **kwargs):
        def wrapper(func):         # noqa: D401
            return func            # no‑op decorator
        return wrapper


@njit
def _spells_1d(a):
    """
    Inner kernel: for a 1‑D boolean array *a*, return an int32 array
    with spell lengths at the start of each True run, zeros elsewhere.
    """
    n = a.size
    out = np.zeros(n, dtype=np.int32)
    i = 0
    while i < n:
        if a[i]:
            j = i + 1
            while j < n and a[j]:
                j += 1
            out[i] = j - i         # inclusive length
            i = j                  # skip to end of run
        else:
            i += 1
    return out


def spells_at_start(da: xr.DataArray, *, dim: str = "time") -> xr.DataArray:
    """
    Compute spell lengths along *dim* of a 3‑D boolean DataArray.

    Parameters
    ----------
    da  : xr.DataArray
        Boolean DataArray shaped (time, y, x) (name of *dim* can vary).
    dim : str, default "time"
        Dimension along which to measure spells.

    Returns
    -------
    xr.DataArray
        Same shape/dtype=int32: spell length at spell start, 0 elsewhere.

    Notes
    -----
    * Works with dask‑backed arrays—the kernel is applied
      chunk‑wise and then reassembled.
    * If Numba is installed the inner loop runs close to C speed.
    """
    if da.dtype != bool:
        da = da.astype(bool)

    return xr.apply_ufunc(
        _spells_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,           # broadcasts over the (y, x) plane
        dask="parallelized",      # keep lazy when da is dask‑backed
        output_dtypes=[np.int32],
        dask_gufunc_kwargs={'allow_rechunk': True}
    ).transpose(*da.dims)

def event_max_cumul_1d(p, srun):
    """
    p   : precip  (time,)
    srun : srun    (time,)  -- 0 except at event start where it holds length
    returns a 1‑D array the same length as `time` whose non‑NaNs are
    the max precip of the event and sit on the first hour of that event.
    """
    out_max = np.full_like(p, np.nan, dtype=float)
    out_cumul = np.full_like(p, np.nan, dtype=float)

    starts = np.where(srun > 0)[0]         # event beginnings
    for i in starts:
        L = int(srun[i])                   # duration in hours
        if L:                             # be safe if srun contains zeros
            out_max[i] = np.nanmax(p[i : i + L])
            out_cumul[i] = np.nansum(p[i : i + L])
    return out_max, out_cumul

def max_cumul_at_start(precipitation, srun):

    return xr.apply_ufunc(
        event_max_cumul_1d,
        precipitation,
        srun,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[precipitation.dtype, precipitation.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

# -----------------------------------------------------------------------
def top_k_precip(
    precip: xr.DataArray,
    k: int = 100,
    *,
    time_dim: str = "time",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Extract the k largest precipitation values per grid‑cell along *time*.

    Parameters
    ----------
    precip   : xr.DataArray
        3‑D array (time, y, x) of precipitation values.
    k        : int, default 100
        Number of largest values to keep.
    time_dim : str, default "time"
        Name of the time dimension in *precip*.

    Returns
    -------
    top_vals  : xr.DataArray  (rank, y, x)
        The k largest values, sorted descending.
    top_idx   : xr.DataArray  (rank, y, x)
        Integer indices along the *time* axis of those values.
    top_times : xr.DataArray  (rank, y, x)
        Datetime64 (or whatever coordinates) corresponding to *top_idx*.
    """
    if precip.ndim != 3 or precip.dims[0] != time_dim:
        raise ValueError("precip must be (time, y, x) with time as first dim")

    arr = precip.data                        # numpy or dask array
    T, Y, X = arr.shape
    k = min(k, T)                            # can’t take more than T

    # ---- 1. argpartition gives us the k largest indices, unordered ----
    idx_part = np.argpartition(-arr, k - 1, axis=0)[:k, :, :]  # (k, Y, X)

    # ---- 2. gather the corresponding values --------------------------
    val_part = np.take_along_axis(arr, idx_part, axis=0)        # (k, Y, X)

    # ---- 3. sort the k‑block so values are descending ---------------
    order = np.argsort(-val_part, axis=0)                       # (k, Y, X)
    top_idx  = np.take_along_axis(idx_part, order, axis=0)
    top_vals = np.take_along_axis(val_part, order, axis=0)

    # ---- 4. map indices → timestamps --------------------------------
    time_coord = precip[time_dim].values                        # (T,)
    top_times = time_coord[top_idx]                             # (k, Y, X)

    # ---- 5. wrap results back into DataArrays -----------------------
    rank = np.arange(k)
    dims_out = ("event",) + precip.dims[1:]                      # ('event', 'y', 'x')
    coords = {
        "event": rank,
        precip.dims[1]: precip.coords[precip.dims[1]],
        precip.dims[2]: precip.coords[precip.dims[2]],
    }

    top_vals_da  = xr.DataArray(top_vals,  dims=dims_out, coords=coords, name="top_values")
    top_idx_da   = xr.DataArray(top_idx,   dims=dims_out, coords=coords, name="time_index")
    top_times_da = xr.DataArray(top_times, dims=dims_out, coords=coords, name="time")

    return top_vals_da, top_idx_da, top_times_da



def process_tile(filespath,ny, nx, wrun):
    """Worker function executed in parallel."""

    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    print (f'Analyzing tile y: {ny} x: {nx}')
    finp = xr.open_mfdataset(
        filesin,
        combine="by_coords"
    )

    lat = finp.lat.isel(time=0).data.squeeze()
    lon = finp.lon.isel(time=0).data.squeeze()

    precipitation = finp.RAIN.where(finp.RAIN>cfg.WET_VALUE_H, 0.0)
    wet_hours = precipitation > cfg.WET_VALUE_H
    srun = spells_at_start(wet_hours, dim='time')

    n_events = srun.where(srun > 0).count(dim='time')
    mean_duration = srun.where(srun > 0).mean(dim='time')
    mean_intensity = precipitation.where(wet_hours).mean(dim='time')
    totpr = precipitation.where(wet_hours, 0.0).sum(dim='time')

    peak_at_start, cumul_at_start = max_cumul_at_start(precipitation, srun)
    peak_at_start = peak_at_start.transpose(*precipitation.dims)
    cumul_at_start = cumul_at_start.transpose(*precipitation.dims)

    # precip:  (time, y, x) DataArray
    top_vals_peak, top_idx_peak, top_times_peak = top_k_precip(peak_at_start, k=100)
    top_vals_cumul, top_idx_cumul, top_times_cumul = top_k_precip(cumul_at_start, k=100)

    top_duration_cumul = srun.isel(time=top_idx_cumul)
    top_duration_peak = srun.isel(time=top_idx_peak)


    ds_stats = xr.Dataset({
        'cumul': (['event','y','x'], top_vals_cumul.data.squeeze()),
        'cumul_duration': (['event','y','x'], top_duration_cumul.data.squeeze()),
        'cumul_time': (['event','y','x'], top_times_cumul.data.squeeze()),
        'peak': (['event','y','x'], top_vals_peak.data.squeeze()),
        'peak_duration': (['event','y','x'], top_duration_peak.data.squeeze()),
        'peak_time': (['event','y','x'], top_times_peak.data.squeeze()),
        'lat':(['y','x'],lat),
        'lon':(['y','x'],lon),
        })

    fout = f"{cfg.path_out}/{wrun}/split_files_tiles_{tile_size}/Hourly_decomposition_top100_NDI_{ny}y-{nx}x.nc"
    ds_stats.to_netcdf(fout, mode='w', format='NETCDF4')

    #Build xarray dataset with results
    ds_results = xr.Dataset({
        'n_events': (['y','x'],n_events.data.squeeze()),
        'mean_duration': (['y','x'],mean_duration.data.squeeze()),
        'mean_intensity': (['y','x'],mean_intensity.data.squeeze()),
        'total_precipitation': (['y','x'],totpr.data.squeeze()),
        'peak_at_start': (['y','x'], peak_at_start.mean(dim='time').data.squeeze()),
        'lat':(['y','x'],lat),
        'lon':(['y','x'],lon),
        })
    
    fout = f"{cfg.path_out}/{wrun}/split_files_tiles_{tile_size}/Hourly_decomposition_NDI_{ny}y-{nx}x.nc"
    ds_results.to_netcdf(fout, mode='w', format='NETCDF4')  



def main():

    wrf_runs = ["EPICC_2km_ERA5", "EPICC_2km_ERA5_CMIP6anom"]

    for wrun in wrf_runs:

        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/UIB_01H_RAIN_20??-??.nc'))
        files_ref = xr.open_dataset(filesin[0])
        nlats = files_ref.sizes['y']
        nlons = files_ref.sizes['x']
        lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
        latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

        xytiles=list(product(latsteps, lonsteps))
        filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/UIB_01H_RAIN_20??-??'
        print(f'Ej: {filespath}_000y-000x.nc')

        Parallel(n_jobs=N_JOBS)(delayed(process_tile)(filespath,xytile[0],xytile[1],wrun) for xytile in xytiles)
        

if __name__ == "__main__":
    main()
