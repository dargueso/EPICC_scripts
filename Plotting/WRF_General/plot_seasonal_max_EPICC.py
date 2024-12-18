#!/usr/bin/env python
'''
@File    :  plot_seasonal_max_EPICC.py
@Time    :  2024/10/15 16:46:59
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2024, Daniel Argüeso
@Project :  None
@Desc    :  None
'''

import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cartopy.crs as crs
import cartopy.feature as cfeature
import wrf as wrf
import pandas as pd
import seaborn as sns
import time


#Measuring time
start = time.time()

fq = '10MIN'
pr = pd.period_range(start=f'2011-01',end=f'2020-12', freq='M')
yearmonths=tuple([(period.month,period.year) for period in pr])

path_in = '/scratch0/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/'

#Loading geographical info

geo_filename = '/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc'
geo_file_xr = xr.open_dataset(geo_filename)
geo_file_nc = nc.Dataset(geo_filename)

geo_proj = wrf.get_cartopy(wrfin=geo_file_nc)
xbounds = wrf.cartopy_xlim(wrfin=geo_file_nc)
ybounds = wrf.cartopy_ylim(wrfin=geo_file_nc)
lats = geo_file_xr.XLAT_M.squeeze().values
lons = geo_file_xr.XLONG_M.squeeze().values
ter = geo_file_xr.HGT_M.squeeze().values

#Loading data
total_rain=[]
for month, year in yearmonths:
    print(f'Processing month {month} and year {year}')
    filein = f'{path_in}/zarr/bymonth/UIB_{fq}_RAIN_{year}-{month:02d}.zarr'
    with xr.open_zarr(filein) as  fin_rain:
        total_rain.append(fin_rain['RAIN'])

total_rain_xr = xr.concat(total_rain, dim='time')

seasonal_max = total_rain_xr.groupby('time.season').max(dim='time')


all_max = total_rain_xr.max().values
lmax = MaxNLocator(nbins=15).tick_values(0,all_max)
cmap = sns.color_palette("icefire", as_cmap=True)


#Plotting

fig,axs = plt.subplots(2,2,layout='constrained',figsize=(20,20),subplot_kw=dict(projection=geo_proj))
# oce10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m')
# lakes10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m')
# rivers10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
seasons = seasonal_max.season.values
axes_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Loop through each season and plot its data
for seas, pos in zip(seasons, axes_positions):
    ax = axs[pos]
    
    ct = ax.contourf(lons, 
                     lats, 
                     seasonal_max.sel(season=seas), 
                     levels = lmax,
                     cmap=cmap,
                     extend="max",
                     transform=crs.PlateCarree(),
                     zorder=100
                     )
    ax.set_title(f'Max Rainfall - {seas}')  # Set the title as the season name
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.spines['geo'].set_linewidth(1)
    ax.coastlines(linewidth=0.5, zorder=102, resolution="50m")
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(104)

    gl = ax.gridlines(
        crs=crs.PlateCarree(),
        xlocs=range(-180, 181, 5),
        ylocs=range(-80, 81, 5),
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        linewidth=0.2,
        color="k",
        alpha=1,
        linestyle="-",
        zorder=103
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xpadding = -10
    gl.ypadding = -10
    gl.xlabel_style = {"size": 12, "color": "black", "zorder" : 103}
    gl.ylabel_style = {"size": 12, "color": "black", "zorder" : 103}

cbar = plt.colorbar(ct, ax=axs, ticks=lmax, aspect=25, orientation="horizontal", shrink=0.5)
cbar.set_label(label='10-min precipitation (mm)', size='x-large', weight='bold')
cbar.ax.tick_params(labelsize='large')

# Adjust layout for better spacing
# plt.tight_layout()

plt.savefig('seasonal_max_rainfall.png')

#Measuring time
end = time.time()
print(f'Elapsed time: {end-start}')