#!/usr/bin/env python
'''
@File    :  plot_map_EPICC_real_vs_synthetic_extremes.py
@Time    :  2025/10/27 11:20:49
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  None
'''

import xarray as xr
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors
from matplotlib.colors import BoundaryNorm
from matplotlib import colormaps as cmaps
import string
import seaborn as sns
from glob import glob


from wrf import (to_np, getvar,get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)



###########################################################
###########################################################
geo_file_name = "/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
geo_file = xr.open_dataset(geo_file_name)
lm_is=geo_file.LANDMASK.squeeze()

# Create and add gray border zone (50 grid points from edge)
border_width = 50
border_mask = np.zeros_like(lm_is.values)
border_mask[:border_width, :] = 1  # Top
border_mask[-border_width:, :] = 1  # Bottom
border_mask[:, :border_width] = 1  # Left
border_mask[:, -border_width:] = 1  # Right

#####################################################################
#####################################################################

def get_geoinfo():

    fileref = nc.Dataset(geo_file_name)
    hgt = getvar(fileref, "HGT_M", timeidx=0)
    hgt = hgt.where(hgt>=0,0)
    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)

    return cart_proj,lats,lons,hgt

#####################################################################
#####################################################################

cart_proj,lats,lons,hgt = get_geoinfo()
mbounds = None
cart_proj._threshold /= 100.
#####################################################################
#####################################################################


qtile = 0.7
qs_values = np.array([0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85,0.90, 0.95,0.98, 0.99,0.995, 0.999,0.9999,1])
mylevels=np.arange(0, 45, 5)
cmap = sns.color_palette("icefire", as_cmap=True)
norm = BoundaryNorm(mylevels, ncolors=cmap.N, extend="max")

cmap_diff = cmaps["BrBG"]
norm_diff = BoundaryNorm(np.arange(-20, 22, 2), ncolors=cmap_diff.N, extend="both")
#Load data

#Loading CTL percentiles

# filesin_pres = sorted(glob('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/UIB_01H_RAIN_????_daymax.nc'))
# filein_pres = xr.open_mfdataset(filesin_pres).compute()
# fin_pres = filein_pres.where(filein_pres.RAIN>0.1).quantile(qs_values,dim='time')
# fin_pres.to_netcdf('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/UIB_01H_RAIN_allyears_daymax_qtiles.nc')

fin_pres = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/UIB_01H_RAIN_allyears_daymax_qtiles.nc').sel(quantile = qtile).squeeze()

#Loading PGW percentiles
# filesin_fut = sorted(glob('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_????_daymax.nc'))
# filein_fut = xr.open_mfdataset(filesin_fut).compute()
# fin_fut = filein_fut.where(filein_fut.RAIN>0.1).quantile(qs_values,dim='time')
# fin_fut.to_netcdf('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_allyears_daymax_qtiles.nc')

fin_fut = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_allyears_daymax_qtiles.nc').sel(quantile = qtile).squeeze()


#Loading SYN confidence intervals

filein_syn = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/future_synthetic_quant_confidence_025buffer.nc'
fin_syn = xr.open_dataset(filein_syn).sel(qs_time = qtile).squeeze()
#####################################################################
#####################################################################

fig,axs = plt.subplots(2,2,layout='constrained',figsize=(20,20),subplot_kw=dict(projection=cart_proj))

data = [fin_pres.RAIN, fin_fut.RAIN, fin_syn.precipitation.isel(quantile=1).squeeze()]
data_diff = fin_fut.RAIN - fin_pres.RAIN
for ns,ax in enumerate(axs.flat[:-1]):
    ax.set_title(f"{string.ascii_lowercase[ns]}", size='x-large', weight='bold',loc="left")
    ax.add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=102)


    ctrain = ax.pcolormesh(
            lons,
            lats,
            data[ns],
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            zorder = 101
        )
    
    ax.contourf(to_np(lons), to_np(lats), border_mask,
                    levels=[0.5, 1.5],
                    colors=['gray'],
                    alpha=0.5,
                    transform=ccrs.PlateCarree(),
                    zorder=103)

    ax.set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
    ax.set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
    gl=ax.gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False,zorder=103)
    gl.right_labels=False
    gl.top_labels=False

axs[1,1].set_title(f"{string.ascii_lowercase[3]}", size='x-large', weight='bold',loc="left")
axs[1,1].add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=102)

dtrain = axs[1,1].pcolormesh(
            lons,
            lats,
            data_diff,
            cmap=cmap_diff,
            norm=norm_diff,
            transform=ccrs.PlateCarree(),
            zorder = 101
        )
axs[1,1].contourf(to_np(lons), to_np(lats), border_mask,
                levels=[0.5, 1.5],
                colors=['gray'],
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=103)
axs[1,1].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs[1,1].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl=axs[1,1].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False,zorder=103)
gl.right_labels=False
gl.top_labels=False

cbar_ax = fig.add_axes([0.1, 0.05, 0.4, 0.03])
cbar = fig.colorbar(
    ctrain, cax=cbar_ax, orientation="horizontal", shrink=0.5
)
cbar.set_label("Precipitation (mm)", fontsize=10)
cbar.ax.tick_params(labelsize=10)

cbard_ax = fig.add_axes([0.6, 0.05, 0.4, 0.03])
cbard = fig.colorbar(
    dtrain, cax=cbard_ax, orientation="horizontal", shrink=0.5
)
cbard.set_label("Precipitation (mm)", fontsize=10)
cbard.ax.tick_params(labelsize=10)

plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/EPICC_real_vs_synthetic_extremes_dmax_q{qtile}th.png',dpi=150)