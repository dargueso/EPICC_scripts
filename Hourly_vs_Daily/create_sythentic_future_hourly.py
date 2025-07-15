#!/usr/bin/env python
'''
Modified version to calculate both full synthetic data and quantiles from the same generation
This ensures consistency between the two outputs.
Result shapes: 
- Full data: (n_samples, time, y, x) 
- Quantiles: (n_samples, quantile, y, x)
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
buffer = cfg.buffer
n_samples = cfg.n_samples

def main():

    calc_distributions = True
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
    ds_hx_wet_days = ds_h_wet_days.resample(time='1D').max()

    if calc_distributions:
        wet_hour_fraction = ds_h_wet_days.where(ds_h_wet_days>0).resample(time='1D').count() / 24.0
        wet_hour_fraction = wet_hour_fraction.where(wet_hour_fraction > 0.0)

        hourly_intensity_dist, wet_hour_dist, samples_per_bin = sf.calculate_wet_hour_intensity_distribution(
            ds_h_wet_days, ds_d, wet_hour_fraction,
            drain_bins=drain_bins, hrain_bins=hrain_bins)
        
        rainfall_probability = sf.save_probability_data(
            hourly_intensity_dist, wet_hour_dist, samples_per_bin, 
            drain_bins, hrain_bins, fout='rainfall_probability_hourly_vs_daily.nc')                                                                                                                                
    else:
        rainfall_probability = xr.open_dataset('rainfall_probability_hourly_vs_daily.nc')                                                                                                                  
    
    ## Calculation of synthetic future hourly data
    rain_daily_future = ds_d.sel(exp='Future')       
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

    wet_hour_dist_present = wet_hour_dist[0,:].squeeze()
    hourly_intensity_dist_present = hourly_intensity_dist[0,:].squeeze() 
    samples_per_bin_present = samples_per_bin[0,:].squeeze()

    wet_hour_dist_present = np.nan_to_num(wet_hour_dist_present, nan=0.0)
    hourly_intensity_dist_present = np.nan_to_num(hourly_intensity_dist_present, nan=0.0)

    whdp_weighted = wet_hour_dist_present * samples_per_bin_present[:, np.newaxis, :, :]
    hidp_weighted = hourly_intensity_dist_present * samples_per_bin_present[:, np.newaxis, :, :]

    comp_samp = sf.window_sum(samples_per_bin_present, buffer)
    comp_wWet = sf.window_sum(whdp_weighted, buffer)
    comp_wHr = sf.window_sum(hidp_weighted, buffer)
    comp_wWet /= comp_samp[:, None]
    comp_wHr /= comp_samp[:, None]

    wet_cdf = np.cumsum(comp_wWet, axis=1).astype(np.float32, order="C")
    hour_cdf = np.cumsum(comp_wHr, axis=1).astype(np.float32, order="C")

    # ============================================================================
    # Initialize arrays for both full data and quantiles
    # ============================================================================
    
    # Full synthetic data array: (n_samples, n_time, ny, nx)
    full_synthetic_data = np.zeros((n_samples, n_time, ny, nx), dtype=np.float32)
    
    # Quantiles setup
    qs_values = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)
    n_quantiles = len(qs_values)
    result_quantiles = np.full((n_samples, n_quantiles, ny, nx), np.nan, dtype=np.float32)

    # ============================================================================
    # Generate synthetic data ONCE and populate both arrays
    # ============================================================================
    
    for sample in range(n_samples):
        print(f"Processing sample {sample + 1}/{n_samples}")
        
        # Generate data with full timestep tracking
        timestep_data = sf.generate_consistent_synthetic_data(
            rain_arr, wet_cdf, hour_cdf, bin_idx, 
            hrain_bins.astype(np.float32),
            iy0, iy1, ix0, ix1,
            n_time,
            thresh=WET_VALUE_H,
            seed=123 + sample)
        
        # Fill full data array and collect values for quantiles
        c = 0
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                # Get all values for this cell across all timesteps
                all_values = []

                for t in range(n_time):
                    value = timestep_data[c][t]
                    if value > WET_VALUE_H:
                        full_synthetic_data[sample, t, iy, ix] = value
                        all_values.append(value)
                
                # Calculate quantiles from the same values used in full data
                if len(all_values) > 0:
                    result_quantiles[sample, :, iy, ix] = np.quantile(
                        all_values, qs_values, method="linear")
                
                c += 1

    # ============================================================================
    # Create xarray DataArrays and save results
    # ============================================================================
    
    # Full synthetic data
    future_synthetic_full = xr.DataArray(
        full_synthetic_data,
        dims=("sample", "time", "y", "x"),
        coords=dict(
            sample=np.arange(n_samples),
            time=time_coords,
            y=ds_d.y,
            x=ds_d.x,
        ),
        attrs={
            'description': 'Full synthetic future hourly rainfall data',
            'units': 'mm/hour',
            'wet_threshold_daily': WET_VALUE_D,
            'wet_threshold_hourly': WET_VALUE_H,
            'note': 'Daily maximum hourly values for each timestep'
        }
    )
    
    # Quantiles
    future_synthetic_quant = xr.DataArray(
        result_quantiles,
        dims=("sample", "quantile", "y", "x"),
        coords=dict(
            sample=np.arange(n_samples),
            quantile=qs_values,
            y=ds_d.y,
            x=ds_d.x,
        ),
        attrs={
            'description': 'Synthetic future hourly rainfall quantiles per sample',
            'units': 'mm/hour',
            'wet_threshold_daily': WET_VALUE_D,
            'wet_threshold_hourly': WET_VALUE_H,
            'note': 'Quantiles calculated from the same data as full synthetic dataset'
        }
    )
    
    # Save results
    print("Saving full synthetic data...")
    future_synthetic_full.to_netcdf('future_synthetic_full_data.nc')
    
    print("Saving quantiles...")
    future_synthetic_quant.to_netcdf('future_synthetic_quant_per_sample.nc')
    


    # ============================================================================
    # Consistency check
    # ============================================================================
    print("\n=== CONSISTENCY CHECK ===")
    
    # Check that quantiles match between the two datasets
    for sample in range(min(3, n_samples)):  # Check first 3 samples
        for iy in range(iy0, min(iy0+3, iy1)):  # Check first 3 cells
            for ix in range(ix0, min(ix0+3, ix1)):
                # Extract wet values from full data
                full_wet_values = full_synthetic_data[sample, :, iy, ix]
                full_wet_values = full_wet_values[full_wet_values > WET_VALUE_H]
                
                if len(full_wet_values) > 0:
                    # Calculate quantiles from full data
                    full_quantiles = np.quantile(full_wet_values, qs_values, method="linear")
                    stored_quantiles = result_quantiles[sample, :, iy, ix]
                    
                    # Check if they match (within floating point precision)
                    if np.allclose(full_quantiles, stored_quantiles, rtol=1e-6):
                        print(f"✓ Sample {sample}, cell ({iy},{ix}): Quantiles match")
                    else:
                        print(f"✗ Sample {sample}, cell ({iy},{ix}): Quantiles MISMATCH!")
                        print(f"  Full data quantiles: {full_quantiles}")
                        print(f"  Stored quantiles: {stored_quantiles}")
    
    print(f"\nFull data shape: {future_synthetic_full.shape}")
    print(f"Quantiles shape: {future_synthetic_quant.shape}")
    print(f"Full data dimensions: {future_synthetic_full.dims}")
    print(f"Quantiles dimensions: {future_synthetic_quant.dims}")
    print(f"Quantiles: {qs_values}")
    
    end_time = time.time()
    print(f'\n======> DONE in {(end_time-start_time):.2f} seconds')



if __name__ == "__main__":
    main()