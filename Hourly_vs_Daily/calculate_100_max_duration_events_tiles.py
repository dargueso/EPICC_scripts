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
from xarray_extras.sort import topk, argtopk, take_along_dim 
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
N_JOBS      = 1         # parallel workers (set 1 to run serially)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


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
  
    n_events, mean_duration, mean_intensity, srun = decompose_precipitation(finp.RAIN, cfg.WET_VALUE_H)
    totpr = finp.RAIN.where(finp.RAIN > cfg.WET_VALUE_H, 0.0).sum(dim='time') 

    # peak_at_start, cumul_at_start = xr.apply_ufunc(
    # event_max_cumul_1d,
    # finp.RAIN,     # (time, y, x)
    # srun,              # (time, y, x)
    # input_core_dims=[['time'], ['time']],
    # output_core_dims=[['time']],
    # vectorize=True,        # loop in compiled NumPy, not Python
    # dask='parallelized',   # work on each dask block independently
    # output_dtypes=[finp.RAIN.dtype],
    # dask_gufunc_kwargs={'allow_rechunk': True}
    # )

    peak_at_start, cumul_at_start = xr.apply_ufunc(
        event_max_cumul_1d,
        finp.RAIN,
        srun,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[finp.RAIN.dtype, finp.RAIN.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    peak_at_start = peak_at_start.transpose(*finp.RAIN.dims)
    cumul_at_start = cumul_at_start.transpose(*finp.RAIN.dims)


    mask = srun>0
    cumul = cumul_at_start.where(mask)
    peak  = peak_at_start.where(mask)

    ds_cumul = top_k_events(cumul, srun, k=100, new_dim='event')
    ds_peak  = top_k_events(peak , srun, k=100, new_dim='event')
    import pdb; pdb.set_trace()  # fmt: skip

    #Build xarray dataset with results
    ds_results = xr.Dataset({
        'cumul': (['event','y','x'], ds_cumul.value.data.squeeze()),
        'cumul_duration': (['event','y','x'], ds_cumul.duration.data.squeeze()),
        'cumul_time': (['event','y','x'], ds_cumul.time.data),
        'peak': (['event','y','x'], ds_peak.value.data.squeeze()),
        'peak_duration': (['event','y','x'], ds_peak.duration.data.squeeze()),
        'peak_time': (['event','y','x'], ds_peak.time.data.squeeze()),
        'lat':(['y','x'],lat),
        'lon':(['y','x'],lon),
        })
    
    fout = f"{cfg.path_out}/{wrun}/split_files_tiles_{tile_size}/Hourly_decomposition_top100_NDI_{ny}y-{nx}x.nc"
    ds_results.to_netcdf(fout, mode='w', format='NETCDF4')


    for seas in ['DJF','MAM','JJA','SON']:

        fin_seas = finp.sel(time=finp.time.dt.season == seas)
  
        srun_seas = srun.sel(time=finp.time.dt.season == seas)
        # peak_at_start, cumul_at_start = xr.apply_ufunc(
        #     event_max_cumul_1d,
        #     fin_seas.RAIN,     # (time, y, x)
        #     srun_seas,              # (time, y, x)
        #     input_core_dims=[['time'], ['time']],
        #     output_core_dims=[['time']],
        #     vectorize=True,        # loop in compiled NumPy, not Python
        #     dask='parallelized',   # work on each dask block independently
        #     output_dtypes=[finp.RAIN.dtype],
        #     dask_gufunc_kwargs={'allow_rechunk': True}
        # )

        peak_at_start, cumul_at_start = xr.apply_ufunc(
            event_max_cumul_1d,
            fin_seas.RAIN,
            srun_seas,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['time'], ['time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[finp.RAIN.dtype, finp.RAIN.dtype],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )

        peak_at_start = peak_at_start.transpose(*fin_seas.RAIN.dims)
        cumul_at_start = cumul_at_start.transpose(*fin_seas.RAIN.dims)

        import pdb; pdb.set_trace()  # fmt: skip
        mask = srun_seas>0
        cumul = cumul_at_start.where(mask)
        peak  = peak_at_start.where(mask)

        ds_cumul = top_k_events(cumul, srun_seas, k=100, new_dim='event')
        ds_peak  = top_k_events(peak , srun_seas, k=100, new_dim='event')


        #Build xarray dataset with results
        # Note: 'srun' is not calculated for seasonal data, as it is not needed

        ds_seas = xr.Dataset({
            'cumul': (['event','y','x'], ds_cumul.value.data.squeeze()),
            'cumul_duration': (['event','y','x'], ds_cumul.duration.data.squeeze()),
            'cumul_time': (['event','y','x'], ds_cumul.time.data),
            'peak': (['event','y','x'], ds_peak.value.data.squeeze()),
            'peak_duration': (['event','y','x'], ds_peak.duration.data.squeeze()),
            'peak_time': (['event','y','x'], ds_peak.time.data.squeeze()),
            'lat':(['y','x'],lat),
            'lon':(['y','x'],lon),
        })
        fout_seas = f"{cfg.path_out}/{wrun}/split_files_tiles_{tile_size}/Hourly_decomposition_top100_NDI_{seas}_{ny}y-{nx}x.nc"
        ds_seas.to_netcdf(fout_seas, mode='w', format='NETCDF4')
         
def top_k_events(value_da, srun_da, k=100, new_dim='event'):
    """
    value_da : DataArray with the metric to rank (accumulated or peak)
    srun_da  : DataArray with spell durations (same dims as value_da)
    k        : how many events to keep
    Returns  : Dataset with value, duration, time on new_dim
    """
    # (1) Find the top‑k values and their integer indices along 'time'
    vals = topk(value_da, k, dim="time") 
    idx  = argtopk(value_da, k, dim="time")
    #vals = value_da.topk(k, dim='time', largest=True, sorted=True)
    #idx  = value_da.argtopk(k, dim='time', largest=True, sorted=True)

    # (2) Convert integer indices → real time coordinates
    # times = value_da['time'].isel(time=idx)
    times = value_da["time"].isel(time=idx)

    # (3) Pick the spell duration that corresponds to those indices
    durs = srun_da.isel(time=idx)

    # (4) Assemble a tidy Dataset
    return xr.Dataset(
        data_vars=dict(
            value     = (('event', 'y', 'x'), vals.data),
            duration  = (('event', 'y', 'x'), durs.data),
            time      = (('event', 'y', 'x'), times.data),
        ),
        coords=dict(
            event = np.arange(k),
            y     = value_da['y'],
            x     = value_da['x'],
        ),
        attrs=value_da.attrs,   # keep units, etc.
    ).rename_dims({'event': new_dim})


def decompose_precipitation(precipitation, wet_value):

    wet_hours = precipitation > wet_value
    srun = simple_spell_duration(wet_hours)
    n_events = srun.where(srun > 0).count(dim='time')
    mean_duration = srun.where(srun > 0).mean(dim='time')
    mean_intensity = precipitation.where(wet_hours).mean(dim='time')

    return n_events, mean_duration, mean_intensity, srun

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

def simple_spell_duration(bool_array):
    """
    Simple approach using apply_ufunc for better performance.
    """
    def process_timeseries(ts):
        """Process a single time series."""
        result = np.zeros_like(ts, dtype=int)
        
        if not ts.any():
            return result
            
        # Find transitions
        padded = np.concatenate(([0], ts, [0]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Assign durations to start positions
        for start, end in zip(starts, ends):
            result[start] = end - start
            
        return result
    
    return xr.apply_ufunc(
        process_timeseries,
        bool_array,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int],
        dask_gufunc_kwargs={'allow_rechunk': True}
    ).transpose(*bool_array.dims)    


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

        Parallel(n_jobs=N_JOBS)(delayed(process_tile)(filespath,xytile[0],xytile[1],wrun) for xytile in xytiles[34:36])
        

if __name__ == "__main__":
    main()
