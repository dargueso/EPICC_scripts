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
from numba import njit, float32, prange 
from numba.typed import List
import config as cfg

y_idx=cfg.y_idx
x_idx=cfg.x_idx
WET_VALUE_H = cfg.WET_VALUE_H  # mm
WET_VALUE_D = cfg.WET_VALUE_D  # mm
qs = cfg.qs
drain_bins = cfg.drain_bins
hrain_bins = cfg.hrain_bins
buffer = cfg.buffer
n_samples = cfg.n_samples

def main():

    start_time = time.time()
    finf = xr.open_dataset(f'UIB_DAY_RAIN_{y_idx}y-{x_idx}x_Future_large.nc')

    rainfall_probability = xr.open_dataset('rainfall_probability_hourly_vs_daily.nc')                                                                                                                  
    
    ## Calculation of synthetic future hourly data
    rain_daily_future = finf.RAIN.where(finf.RAIN > WET_VALUE_D)
    rain_arr = rain_daily_future.data.astype(np.float32, order='C')
    
    # Get time dimension info
    n_time = rain_arr.shape[0]
    time_coords = rain_daily_future.time

    # convert daily total to its bin index once so the kernel re-uses it fast
    bin_idx = (np.digitize(rain_arr, drain_bins) - 1).astype(np.int16)                                                                                    

    ny, nx = rain_arr.shape[1:]
    ix0, ix1 = buffer, nx - buffer
    iy0, iy1 = buffer, ny - buffer
    n_cells = (iy1 - iy0) * (ix1 - ix0)
    # Modified: Create buffers for each sample, storing time series per cell
    # Shape: [n_samples][n_cells][time_values]
    sample_buffers = []
    for sample in range(n_samples):
        buffers = List()
        for _ in range(n_cells):
            buffers.append(List.empty_list(float32))
        sample_buffers.append(buffers)

    wet_hour_dist_present = rainfall_probability.wet_hours_distribution.data[0,:].squeeze()
    hourly_intensity_dist_present = rainfall_probability.hourly_distribution.data[0,:].squeeze() 
    samples_per_bin_present = rainfall_probability.samples_per_bin.data[0,:].squeeze()

    wet_hour_dist_present = np.nan_to_num(wet_hour_dist_present, nan=0.0)
    hourly_intensity_dist_present = np.nan_to_num(hourly_intensity_dist_present, nan=0.0)

    whdp_weighted = wet_hour_dist_present * samples_per_bin_present[:, np.newaxis, :, :]
    hidp_weighted = hourly_intensity_dist_present * samples_per_bin_present[:, np.newaxis, :, :]

    comp_samp = sf._window_sum(samples_per_bin_present, buffer)
    comp_wWet = sf._window_sum(whdp_weighted, buffer)
    comp_wHr = sf._window_sum(hidp_weighted, buffer)
    comp_wWet /= comp_samp[:, None]
    comp_wHr /= comp_samp[:, None]

    wet_cdf = np.cumsum(comp_wWet, axis=1).astype(np.float32, order="C")
    hour_cdf = np.cumsum(comp_wHr, axis=1).astype(np.float32, order="C")

    # Generate synthetic data for each sample
    for sample in range(n_samples):
        print(f"Processing sample {sample + 1}/{n_samples}")
        sf.generate_dmax_hourly_values_per_timestep(
            rain_arr, wet_cdf, hour_cdf, bin_idx, 
            hrain_bins.astype(np.float32),
            sample_buffers[sample],
            iy0, iy1, ix0, ix1,
            thresh=WET_VALUE_H,
            seed=123 + sample)

    # Calculate quantiles along time dimension for each sample and cell
    qs_values = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)
    n_quantiles = len(qs_values)
    
    # Result array: (n_samples, n_quantiles, ny, nx)
    result_quantiles = np.full((n_samples, n_quantiles, ny, nx), np.nan, dtype=np.float32)
    full_output = np.zeros((n_samples, n_time, ny, nx), dtype=np.float32)

    for sample in range(n_samples):
        print(f"Calculating quantiles for sample {sample + 1}/{n_samples}")
        c = 0
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                # Get time series for this cell and sample
                time_series = np.array(sample_buffers[sample][c], dtype=np.float32)
                
                # Only calculate quantiles if we have wet values
                if len(time_series) > 0:
                    # Filter values above WET_VALUE (should already be filtered, but double-check)
                    wet_values = time_series[time_series > WET_VALUE_H]
                    full_output[sample,:,iy,ix] = time_series
                    if len(wet_values) > 0:
                        result_quantiles[sample, :, iy, ix] = np.quantile(
                            wet_values, qs_values, method="linear")
                c += 1

    # Create xarray DataArray with proper coordinates
    future_synthetic_quant = xr.DataArray(
        result_quantiles,
        dims=("sample", "quantile", "y", "x"),
        coords=dict(
            sample=np.arange(n_samples),
            quantile=qs_values,
            y=rain_daily_future.y,
            x=rain_daily_future.x,
        ),
        attrs={
            'description': 'Synthetic future hourly rainfall quantiles per sample',
            'units': 'mm/hour',
            'wet_threshold_daily': WET_VALUE_D,
            'wet_threshold_hourly': WET_VALUE_H,
            'note': 'Quantiles calculated along time dimension for each sample, only for values > wet_threshold'
        }
    )

    future_synthetic_dmax = xr.DataArray(
        full_output,
        dims=("sample", "time", "y", "x"),
        coords=dict(
            sample=np.arange(n_samples),
            time=time_coords,
            y=rain_daily_future.y,
            x=rain_daily_future.x,
        ),
        attrs={
            'description': 'Synthetic future daily maximum of hourly rainfall',
            'units': 'mm/hour',
            'wet_threshold_daily': WET_VALUE_D,
            'wet_threshold_hourly': WET_VALUE_H,
            'note': 'Daily maximum hourly values for each timestep, only for values > wet_threshold'
        }
    )
    
    # Save results
    future_synthetic_dmax.to_netcdf('future_synthetic_dmax.nc')
    future_synthetic_quant.to_netcdf('future_synthetic_quant_per_sample.nc')
    future_synthetic_quant = future_synthetic_quant.rename({'quantile': 'qs_time'})    
    future_synthetic_quant_confidence = future_synthetic_quant.quantile(q=[0.025, 0.975], dim='sample')
    future_synthetic_quant_confidence.to_netcdf('future_synthetic_quant_confidence.nc')

    print(f"Result shape: {future_synthetic_quant.shape}")
    print(f"Dimensions: {future_synthetic_quant.dims}")
    print(f"Quantiles: {qs_values}")
    
    end_time = time.time()
    print(f'======> DONE in {(end_time-start_time):.2f} seconds \n')

if __name__ == "__main__":
    main()