#!/usr/bin/env python
'''
Modified version to calculate quantiles along time dimension for each sample
Result shape: (n_samples, quantile, y, x)
'''

import xarray as xr
import numpy as np
import synthetic_future_utils as sf
import time
import pandas as pd
import config as cfg
from glob import glob

path_in = cfg.path_in
WET_VALUE_H = cfg.WET_VALUE_H  # mm
WET_VALUE_D = cfg.WET_VALUE_D  # mm
qs = cfg.qs
drain_bins = cfg.drain_bins
hrain_bins = cfg.hrain_bins

def main():
    start_time = time.time()
    filesinp = sorted(glob(f'{path_in}/EPICC_2km_ERA5/split_files_tiles_50_025buffer/UIB_01H_RAIN_20??-??_000y-000x_*.nc'))
    #filesinp = sorted(glob(f'{cfg.path_in}/EPICC_2km_ERA5/UIB_01H_RAIN_????-??.nc'))
    # filesinf = sorted(glob(f'{cfg.path_in}/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_????-??.nc'))
                      
                      
    # Open all files as a single dataset
    finp = xr.open_mfdataset(
        filesinp,
        combine='by_coords',  # Assumes time coordinates align or continue across files
        parallel=True,        # Enables Dask parallel reads (good for large datasets)
        chunks={'time': 24}   # Optional: use chunking if you want to enable lazy loading
    )

    # finf = xr.open_mfdataset(
    #     filesinf,
    #     combine='by_coords',  # Assumes time coordinates align or continue across files
    #     parallel=True,        # Enables Dask parallel reads (good for large datasets)
    #     chunks={'time': 24}   # Optional: use chunking if you want to enable lazy loading
    # )

    ds_h = finp.RAIN.where(finp.RAIN > WET_VALUE_H, 0.0)  
    # finf = finf.RAIN.where(finf.RAIN > WET_VALUE_H, 0.0)

    # ds_h = xr.concat([finp, finf], dim=pd.Index(['Present', 'Future'], name='exp'))

    ds_d = ds_h.resample(time='1D').sum()
    ds_d = ds_d.where(ds_d > WET_VALUE_D)

    wet_days = ds_d > WET_VALUE_D
    wet_days_hourly = wet_days.reindex(time=ds_h.time, method='ffill')
    ds_h_wet_days = ds_h.where(wet_days_hourly)


    wet_hour_fraction = ds_h_wet_days.where(ds_h_wet_days>0).resample(time='1D').count() / 24.0
    wet_hour_fraction = wet_hour_fraction.where(wet_hour_fraction > 0.0)

    hourly_intensity_dist, wet_hour_dist, samples_per_bin = sf.calculate_wet_hour_intensity_distribution(
        ds_h_wet_days, ds_d, wet_hour_fraction,
        drain_bins=drain_bins, hrain_bins=hrain_bins)
    
    sf.save_probability_data(
        hourly_intensity_dist, wet_hour_dist, samples_per_bin, 
        drain_bins, hrain_bins, fout='rainfall_probability_hourly_vs_daily.nc')                                                                                                                                

      
    end_time = time.time()
    print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')

if __name__ == "__main__":
    main()