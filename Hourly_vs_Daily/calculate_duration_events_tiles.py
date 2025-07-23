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
N_JOBS      = 20         # parallel workers (set 1 to run serially)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def process_tile(filespath,ny, nx, wrun):
    """Worker function executed in parallel."""

    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    finp = xr.open_mfdataset(
        filesin,
        combine="by_coords"
    )

    lat = finp.lat.isel(time=0).data.squeeze()
    lon = finp.lon.isel(time=0).data.squeeze()
  
    n_events, mean_duration, mean_intensity, srun = decompose_precipitation(finp.RAIN, cfg.WET_VALUE_H)
    totpr = finp.RAIN.where(finp.RAIN > cfg.WET_VALUE_H, 0.0).sum(dim='time') 
  
    #Build xarray dataset with results
    ds_results = xr.Dataset({
        'n_events': (['y','x'],n_events.data.squeeze()),
        'mean_duration': (['y','x'],mean_duration.data.squeeze()),
        'mean_intensity': (['y','x'],mean_intensity.data.squeeze()),
        'total_precipitation': (['y','x'],totpr.data.squeeze()),
        #'srun': (['time','y','x'],  srun.data.squeeze().astype(bool)),
        'lat':(['y','x'],lat),
        'lon':(['y','x'],lon),
        })
    
    fout = f"{cfg.path_out}/{wrun}/split_files_tiles_{tile_size}/Hourly_decomposition_NDI_{ny}y-{nx}x.nc"
    ds_results.to_netcdf(fout, mode='w', format='NETCDF4')
    
def decompose_precipitation(precipitation, wet_value):

    wet_hours = precipitation > wet_value
    srun = simple_spell_duration(wet_hours)
    n_events = srun.where(srun > 0).count(dim='time')
    mean_duration = srun.where(srun > 0).mean(dim='time')
    mean_intensity = precipitation.where(wet_hours).mean(dim='time')


    return n_events, mean_duration, mean_intensity, srun

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

    wrf_runs = ["EPICC_2km_ERA5"]#, "EPICC_2km_ERA5_CMIP6anom"]

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
