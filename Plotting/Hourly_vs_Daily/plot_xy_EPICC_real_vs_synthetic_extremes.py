#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_real_vs_synthetic_extremes.py
@Time    :  2025/11/04 14:44:32
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  None
@Desc    :  None
'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


locs_x_idx = [559,423,569,795,638,821,1091,335,989]#,433,866]
locs_y_idx = [258,250,384,527,533,407,174,119,425]#,254,506]
locs_names = ['Mallorca','Turis','Pyrenees','Rosiglione', 'Ardeche','Corte','Catania','Almeria',"L'Aquila"]#,'Valencia','Barga']


#Loading synthetic future qtiles

filein_syn = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/future_synthetic_quant_confidence_025buffer.nc'
fin_syn_all = xr.open_dataset(filein_syn).squeeze()
fin_syn_all = fin_syn_all.isel(qs_time=slice(10,-2))
qtiles= fin_syn_all.qs_time.values
cl_hi = 0.95
cl_lo = round(1 - cl_hi, 2)

#####################################################################
#####################################################################

for loc in range(len(locs_names)):

    loc_name = locs_names[loc]
    print(loc_name)
    xloc = locs_x_idx[loc]
    yloc = locs_y_idx[loc]

    filein_pres = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/UIB_01H_RAIN_{yloc:3d}y-{xloc:3d}x_025buffer.nc'
    fin_pres = xr.open_dataset(filein_pres).squeeze()

    filein_fut = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_{yloc:3d}y-{xloc:3d}x_025buffer.nc'
    fin_fut = xr.open_dataset(filein_fut).squeeze()

    fin_syn = fin_syn_all.isel(y=yloc,x=xloc).squeeze()

    #fin_pres_loc = fin_pres.where(fin_pres.RAIN>0.1).isel(y=25,x=25).dropna(dim="time")
    #fin_fut_loc = fin_fut.where(fin_fut.RAIN>0.1).isel(y=25,x=25).dropna(dim="time")

    fin_pres_loc = fin_pres.where(fin_pres.RAIN>0.1).stack(xyt=("time","y","x")).dropna(dim="xyt")
    fin_fut_loc = fin_fut.where(fin_fut.RAIN>0.1).stack(xyt=("time","y","x")).dropna(dim="xyt")

    fin_pres_qtiles = fin_pres_loc.quantile(qtiles, dim='xyt', skipna=True)
    fin_fut_qtiles = fin_fut_loc.quantile(qtiles, dim='xyt', skipna=True)

    fig = plt.figure(figsize=(10, 10), constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)

    # Plot with improved styling
    ax.plot(qtiles, fin_pres_qtiles.RAIN, label='Present-day observations', 
            color='#2E86AB', linewidth=2, marker='o', markersize=4)
    ax.plot(qtiles, fin_fut_qtiles.RAIN, label='Future observations', 
            color="#E50C0C", linewidth=2, marker='s', markersize=4)
    ax.plot(qtiles, fin_syn.sel(quantile=cl_hi).precipitation.squeeze(), 
            label='Future synthetic', color='#F18F01', linewidth=1, 
            linestyle='--', marker=None)
    ax.plot(qtiles, fin_syn.sel(quantile=cl_lo).precipitation.squeeze(), 
            color='#F18F01', linewidth=1, 
            linestyle='--', marker=None)
    ax.fill_between(qtiles, fin_syn.sel(quantile=cl_lo).precipitation.squeeze(), 
                    fin_syn.sel(quantile=cl_hi).precipitation.squeeze(), 
                    color='#F18F01', alpha=0.2)

    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels and title with better formatting
    ax.set_xlabel('Quantiles', fontsize=12, fontweight='bold')
    ax.set_ylabel('1-hour precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'1-hour Precipitation Quantiles at {loc_name}', 
                fontsize=14, fontweight='bold', pad=20)

    # Improved legend
    ax.legend(frameon=True, fancybox=True, shadow=True, 
            fontsize=11, loc='upper left')

    # Enhanced grid
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add minor ticks for log scale
    ax.minorticks_on()

    plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/quantiles_1H_precipitation_{loc_name}_{cl_hi}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')