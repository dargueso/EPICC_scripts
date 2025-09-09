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

from turtle import shape
import xarray as xr
import numpy as np
import time as ttime
import config as cfg
import pandas as pd



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
        output_dtypes=[int]
    ).transpose(*bool_array.dims)  

def main():

    start_time = ttime.time()
    finp = xr.open_dataset(f'UIB_01H_RAIN_258y-559x_Present_large.nc')

    n_events, mean_duration, mean_intensity, srun = decompose_precipitation(finp.RAIN, cfg.WET_VALUE_H)

    totpr = finp.RAIN.where(finp.RAIN > cfg.WET_VALUE_H, 0.0).sum(dim='time') 
    end_time = ttime.time()
    print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')
if __name__ == "__main__":
    main()