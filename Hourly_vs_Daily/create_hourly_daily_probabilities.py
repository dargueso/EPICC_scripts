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

y_idx=cfg.y_idx
x_idx=cfg.x_idx
WET_VALUE_H = cfg.WET_VALUE_H  # mm
WET_VALUE_D = cfg.WET_VALUE_D  # mm
qs = cfg.qs
drain_bins = cfg.drain_bins
hrain_bins = cfg.hrain_bins

def main():
    start_time = time.time()
    finp = xr.open_dataset(f'UIB_01H_RAIN_{y_idx}y-{x_idx}x_Present_large.nc')
    finf = xr.open_dataset(f'UIB_01H_RAIN_{y_idx}y-{x_idx}x_Future_large.nc')

    finp = finp.RAIN.where(finp.RAIN > WET_VALUE_H, 0.0)  
    finf = finf.RAIN.where(finf.RAIN > WET_VALUE_H, 0.0)

    ds_h = xr.concat([finp, finf], dim=pd.Index(['Present', 'Future'], name='exp'))

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