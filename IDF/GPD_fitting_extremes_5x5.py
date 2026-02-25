#!/usr/bin/env python
'''
@File    :  Untitled-1
@Time    :  2025/03/29 09:51:18
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  None
@Desc    :  None
'''

import xarray as xr
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


wruns = ['EPICC_2km_ERA5','EPICC_2km_ERA5_CMIP6anom']
wrun = wruns[0]

all_locs = ['Mallorca', 'Turis', 'Pyrenees']

xidx = {'Mallorca': 559, 'Turis': 423, 'Pyrenees': 569}
yidx = {'Mallorca': 258, 'Turis': 250, 'Pyrenees': 384}

#####################################################################
#####################################################################
def fit_genpareto(values):
    values = values[~np.isnan(values)] 
    shape, loc, scale = stats.genpareto.fit(values,floc=0)
    return np.array([shape, loc, scale])  # Return all parameters


def main():
        
    for loc in all_locs:
        
        fin = xr.open_dataset(f'{wrun}/UIB_10MIN_RAIN_{yidx[loc]:03d}y-{xidx[loc]:03d}x.nc')
        fin_resampled = fin.resample(time='1H').sum().resample(time='1D').max()
        print(f"Maximum rainfall in your resample data is: {fin_resampled.RAIN.max().values} mm")
        
        fin_5x5 = fin_resampled.isel(y=slice(5-2,5+3), x=slice(5-2,5+3)).max(dim=['y','x'])
        
        
        # Get annual maximum for each grid point
        annual_max = fin_resampled.groupby('time.year').max()
        annual_max_5x5 = fin_5x5.groupby('time.year').max()
        print(f"Maximum rainfall in your annual data is: {annual_max.RAIN.max().values} mm")
        
        # Get Peak Over Threshold (POT) for each grid point
        
        POT = annual_max.min('year')
        POT_5x5 = annual_max_5x5.min('year')
        
        rain_POT = fin_resampled.where(fin_resampled >= POT, drop=True)
        rain_exceedance = rain_POT-POT
        rain_exceedance_5x5 = fin_5x5.where(fin_5x5 >= POT_5x5, drop=True)
        rain_exceedance_5x5 = rain_exceedance_5x5 - POT_5x5
        print(f"Maximum rainfall in your POT data is: {rain_exceedance.RAIN.max().values} mm")
        
        
        results = xr.apply_ufunc(
        fit_genpareto, rain_exceedance,
        input_core_dims=[["time"]],  # Apply along 'time'
        output_core_dims=[["params"]],  # Add new dimension 'params'
        vectorize=True,  # Vectorize over lat/lon
        dask="parallelized",  # Enable parallel computation if using Dask
        output_dtypes=[float],  # Output type
        )
        param_names = ["shape", "loc", "scale"]    
        results = results.assign_coords(params=param_names)
        
        results_5x5 = fit_genpareto(rain_exceedance_5x5.RAIN.values)
        
        tdim = rain_exceedance.sizes['time']
        ydim = rain_exceedance.sizes['y']
        xdim = rain_exceedance.sizes['x']
        
        nbootstrap = 1000
        shape_bt_5x5 = np.zeros(nbootstrap)
        scale_bt_5x5 = np.zeros(nbootstrap)
        
        for bt in range(nbootstrap):
            
            random_y = np.random.randint(0, ydim, size=tdim)  # Random row index (0 to 4)
            random_x = np.random.randint(0, xdim, size=tdim)  # Random column index (0 to 4)

            fin_bt_5x5 = fin_resampled.isel(y=("time", random_y), x=("time", random_x))
            annual_max_bt_5x5 = fin_bt_5x5.groupby('time.year').max()
            POT_bt_5x5 = annual_max_bt_5x5.min('year')
            rain_exceedance_bt_5x5 = fin_bt_5x5.where(fin_bt_5x5 >= POT_bt_5x5, drop=True)
            rain_exceedance_bt_5x5 = rain_exceedance_bt_5x5 - POT_bt_5x5
            results_bt_5x5 = fit_genpareto(rain_exceedance_bt_5x5.RAIN.values)
            shape_bt_5x5[bt] = results_bt_5x5[0]
            scale_bt_5x5[bt] = results_bt_5x5[2]
            
        #####################################################################
        #####################################################################
        #Plotting
        # Extract shape and scale parameters
        shape_values = results.RAIN.sel(params="shape").values.flatten()
        scale_values = results.RAIN.sel(params="scale").values.flatten()

        shape_values_neighbours = results.RAIN.sel(params="shape").isel(y=slice(5-2,5+3), x=slice(5-2,5+3)).values.flatten()
        scale_values_neighbours = results.RAIN.sel(params="scale").isel(y=slice(5-2,5+3), x=slice(5-2,5+3)).values.flatten()
        
        mean_scale_values_neighbours = np.nanmean(scale_values_neighbours)
        mean_shape_values_neighbours = np.nanmean(shape_values_neighbours)
        
        shape_5x5 = results_5x5[0]
        scale_5x5 = results_5x5[2]
        # Remove NaNs (optional, if needed)
        valid = ~np.isnan(shape_values) & ~np.isnan(scale_values)
        shape_values = shape_values[valid]
        scale_values = scale_values[valid]

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(shape_bt_5x5, scale_bt_5x5, alpha=0.2, color='g', marker='o', s=10)
        plt.scatter(shape_values, scale_values, alpha=0.3, edgecolor='k')
        plt.scatter(shape_values_neighbours, scale_values_neighbours, alpha=0.7, color='r')
        plt.scatter(mean_shape_values_neighbours, mean_scale_values_neighbours, alpha=0.7, color='r', marker='x', s=100)
        plt.scatter(shape_5x5, scale_5x5, alpha=0.7, color='g', marker='x', s=100)
        
        plt.xlabel("Shape Parameter")
        plt.ylabel("Scale Parameter")
        plt.title("Scatter Plot of Shape vs. Scale Parameters")
        plt.grid(True)
        plt.savefig(f"scatter_shape_scale_{loc}.png")
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
