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
This requires the data to be previously processed Calc_RFA_clusters_Wards.py
'''

import xarray as xr
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
import epicc_config as cfg


wruns = ['EPICC_2km_ERA5','EPICC_2km_ERA5_CMIP6anom']
wrunp = wruns[0]
wrunf = wruns[1]

all_locs = ['Mallorca', 'Turis', 'Pyrenees']

xidx = {'Mallorca': 559, 'Turis': 423, 'Pyrenees': 569}
yidx = {'Mallorca': 258, 'Turis': 250, 'Pyrenees': 384}

nside = 2

freqs = ['10min','30min','1h','6h','12h','1d']

#####################################################################
#####################################################################
def fit_genpareto(values):
    values = values[~np.isnan(values)] 
    shape, loc, scale = stats.genpareto.fit(values,floc=0)
    return np.array([shape, loc, scale])  # Return all parameters

def resample_data(ds, fq):
    time_steps = {'30min': 3, '1h': 6, '6h': 36, '12h': 72, '1d': 144}
    
    if fq == '10min':
        return ds.resample(time='1D').max()
    else:
        return ds.coarsen(time=time_steps[fq], boundary="trim").sum().coarsen(time=144//time_steps[fq], boundary="trim").max()
        
    
    

def main():
    for loc in all_locs:
        print(f"Processing location: {loc}")
        filesinp = sorted(glob(f'{cfg.path_in}/{wrunp}/{cfg.patt_in}_10MIN_RAIN_20??_daymax_cluster_{loc}_*.nc'))
        
        finp= xr.open_mfdataset(filesinp, concat_dim="time", combine="nested")
        #finf = xr.open_mfdataset(f'{wrunf}/UIB_10MIN_RAIN_????-??.nc')
        rain_p = finp.RAIN.values - finp.RAIN.min('time').values
        #finp = finp.chunk({'time': -1})
        param_names = ["shape", "loc", "scale"]  
        results_p = xr.apply_ufunc(
        fit_genpareto, rain_p.squeeze(),
        vectorize=True,  # Vectorize over lat/lon
        dask="parallelized",  # Enable parallel computation if using Dask
        dask_gufunc_kwargs={'allow_rechunk': True}
        )

        import pdb; pdb.set_trace()  # fmt: skip
        print(results_p)
        
        shape,location,scale = results_p[0], results_p[1], results_p[2]
        
        print(f"GPD Fit Parameters: Shape (ξ) = {shape:.4f}, Scale (σ) = {scale:.4f}, Location (μ) = {location:.4f}")

        #Plotting
        xx = np.linspace(0, np.max(rain_p), num=100)
        #xx = np.linspace(min(rain_over_threshold), max(rain_over_threshold), 100)
        pdf_gep = stats.genpareto.pdf(xx, shape, loc=location, scale=scale)
        plt.hist(rain_p, bins=10, density=True, alpha=0.5, label="Model")
        plt.plot(xx, pdf_gep, 'r-', lw=2, label="GEP PDF (MLE)")
        plt.xlabel("Rainfall exceedances over POT threshold (mm)")
        plt.ylabel("Density")
        plt.title("GEP Fit using MLE")
        plt.legend()
        plt.savefig(f"gpd_fit_{loc}.png")
            


            
            
        #     # Future individual points
        #     results_f = xr.apply_ufunc(
        #     fit_genpareto, rain_f,
        #     input_core_dims=[["time"]],  # Apply along 'time'
        #     output_core_dims=[["params"]],  # Add new dimension 'params'
        #     vectorize=True,  # Vectorize over lat/lon
        #     dask="parallelized",  # Enable parallel computation if using Dask
        #     output_dtypes=[float],  # Output type
        #     )
        #     results_f = results_f.assign_coords(params=param_names)
            
        #     # Present buffer area
            
            
        #     results_buffer_p = fit_genpareto(rain_buffer_p.values)
        #     results_buffer_f = fit_genpareto(rain_buffer_f.values)
            
        #     tdim = finp_resampled.sizes['time']
            
        #     nbootstrap = 1000
        #     shape_btp = np.zeros(nbootstrap)
        #     scale_btp = np.zeros(nbootstrap)

            
        #     for bt in tqdm(range(nbootstrap), desc="Present bootstrap", leave=False):
                
        #         random_y = np.random.randint(y_center-nside, y_center+nside+1, size=tdim)  # Random row index (0 to 4)
        #         random_x = np.random.randint(x_center-nside, x_center+nside+1, size=tdim)  # Random column index (0 to 4)

        #         fin_bt = finp_resampled.isel(y=("time", random_y), x=("time", random_x))
        #         annual_max_bt = fin_bt.groupby('time.year').max()
        #         POT_bt = annual_max_bt.min('year')
        #         rain_bt = fin_bt.where(fin_bt >= POT_bt, drop=True)
        #         rain_bt = rain_bt - POT_bt
        #         results_bt = fit_genpareto(rain_bt.values)
        #         shape_btp[bt] = results_bt[0]
        #         scale_btp[bt] = results_bt[2]
            
        #     tdim = finf_resampled.sizes['time']
            
        #     shape_btf = np.zeros(nbootstrap)
        #     scale_btf = np.zeros(nbootstrap)
            
        #     for bt in tqdm(range(nbootstrap), desc="Future bootstrap", leave=False):
                
        #         random_y = np.random.randint(y_center-nside, y_center+nside+1, size=tdim)  # Random row index (0 to 4)
        #         random_x = np.random.randint(x_center-nside, x_center+nside+1, size=tdim)  # Random column index (0 to 4)

        #         fin_bt = finf_resampled.isel(y=("time", random_y), x=("time", random_x))
        #         annual_max_bt = fin_bt.groupby('time.year').max()
        #         POT_bt = annual_max_bt.min('year')
        #         rain_bt = fin_bt.where(fin_bt >= POT_bt, drop=True)
        #         rain_bt = rain_bt - POT_bt
        #         results_bt = fit_genpareto(rain_bt.values)
        #         shape_btf[bt] = results_bt[0]
        #         scale_btf[bt] = results_bt[2]
            
            
                
        #     #####################################################################
        #     #####################################################################
        #     #Plotting
        #     # Extract shape and scale parameters
            
        #     shape_center_p = results_p.sel(params="shape").isel(y=5,x=5).values
        #     scale_center_p = results_p.sel(params="scale").isel(y=5,x=5).values
        #     shape_center_f = results_f.sel(params="shape").isel(y=5,x=5).values
        #     scale_center_f = results_f.sel(params="scale").isel(y=5,x=5).values
            
        #     shape_values_p = results_p.sel(params="shape").values.flatten()
        #     scale_values_p = results_p.sel(params="scale").values.flatten()
        #     shape_values_f = results_f.sel(params="shape").values.flatten()
        #     scale_values_f = results_f.sel(params="scale").values.flatten()

        #     shape_values_neigh_p = results_p.sel(params="shape").isel(y=slice(y_center-nside,y_center+nside+1), x=slice(x_center-nside,x_center+nside+1)).values.flatten()
        #     scale_values_neigh_p = results_p.sel(params="scale").isel(y=slice(y_center-nside,y_center+nside+1), x=slice(x_center-nside,x_center+nside+1)).values.flatten()
            
        #     shape_values_neigh_f = results_f.sel(params="shape").isel(y=slice(y_center-nside,y_center+nside+1), x=slice(x_center-nside,x_center+nside+1)).values.flatten()
        #     scale_values_neigh_f = results_f.sel(params="scale").isel(y=slice(y_center-nside,y_center+nside+1), x=slice(x_center-nside,x_center+nside+1)).values.flatten()
            
            
        #     mean_scale_values_neighbours_p = np.nanmean(scale_values_neigh_p)
        #     mean_shape_values_neighbours_p = np.nanmean(shape_values_neigh_p)
            
        #     mean_scale_values_neighbours_f = np.nanmean(scale_values_neigh_f)
        #     mean_shape_values_neighbours_f = np.nanmean(shape_values_neigh_f)
            

        #     # # Scatter plot
        #     # plt.figure(figsize=(8, 6))
        #     ax = axs[nfq]
        #     ax.set_title(f"{fq} {loc}", fontsize='large')
        #     # First plot the bootstrap points
        #     ax.scatter(scale_btp, shape_btp, alpha=0.5, color='lavender', marker='o', s=10, label='Bootstrap Present')
        #     ax.scatter(scale_btf, shape_btf, alpha=0.5, color='moccasin', marker='o', s=10, label='Bootstrap Future')


        #     # All region points
        #     #ax.scatter(scale_values_p, shape_values_p, alpha=0.7, color='mediumpurple', marker='o', s=10)
        #     #ax.scatter(scale_values_f, shape_values_f, alpha=0.7, color='darkorange', marker='o', s=10)

        #     ax.scatter(scale_values_neigh_p, shape_values_neigh_p, alpha=0.7, color='indigo', s=20,label='Neighbours Present')
        #     ax.scatter(scale_values_neigh_f, shape_values_neigh_f, alpha=0.7, color='orangered', s=20, label='Neighbours Future')

        #     ax.scatter(mean_scale_values_neighbours_p, mean_shape_values_neighbours_p, alpha=0.7, color='darkviolet', marker='d', s=100, label='Mean Neighbours Present')
        #     ax.scatter(mean_scale_values_neighbours_f, mean_shape_values_neighbours_f, alpha=0.7, color='red', marker='d', s=100, label='Mean Neighbours Future')

        #     # Maximum of buffer area
        #     ax.scatter(results_buffer_p[2], results_buffer_p[0], alpha=0.7, color='purple', marker='x', s=100, label='Maximum of neighbours Present')
        #     ax.scatter(results_buffer_f[2], results_buffer_f[0], alpha=0.7, color='tomato', marker='x', s=100, label='Maximum of neighbours Future')

        #     # This is the actual point (central point)
        #     ax.scatter(scale_center_p, shape_center_p, alpha=0.9, color='cyan', marker='*', s=100, label='Central Point Present')
        #     ax.scatter(scale_center_f, shape_center_f, alpha=0.9, color='magenta', marker='*', s=100, label='Central Point Future')

        #     # Swap axis labels
        #     ax.set_xlabel("Scale Parameter")
        #     ax.set_ylabel("Shape Parameter")

        #     # Add title and grid
            
        #     ax.grid(True)

        #     ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
          
            
        # plt.tight_layout()
        # # Add legend outside the plot
        # #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        # #plt.suptitle("Scatter Plot of Scale vs. Shape Parameters")
        # plt.savefig(f"scatter_scale_shape_{loc}_allfqs.png")

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
