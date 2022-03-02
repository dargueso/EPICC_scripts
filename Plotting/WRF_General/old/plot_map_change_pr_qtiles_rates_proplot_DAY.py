#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-03-17T09:40:51+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-17T09:40:54+01:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files:
#
#####################################################################
"""


import xarray as xr
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import proplot as plot

import datetime as dt
import sys

import string
from glob import glob

import epicc_config as cfg

from matplotlib.ticker import MaxNLocator

from wrf import (to_np, getvar,get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)



#### READING INPUT FILE ######
### Options



varname = 'RAIN'
reg = 'EPICC'


###########################################################
###########################################################

def get_geoinfo():

    fileref = nc.Dataset(f'{cfg.path_wrfout}/{wrun_pre}/out/{cfg.file_ref}')
    hgt = getvar(fileref, "ter")
    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)

    return cart_proj,lats,lons,hgt

def map_bounds(reg):

    if reg=='EPICC':
        mbounds = None
    else:
        mbounds = GeoBounds(CoordPair(lat=cfg.reg_coords[reg][0], lon=cfg.reg_coords[reg][1]),
                               CoordPair(lat=cfg.reg_coords[reg][2], lon=cfg.reg_coords[reg][3]))
    return mbounds



###########################################################
###########################################################
qtile = 0.99
wrun_pre = cfg.wrf_runs[0]
wrun_fut = wrun_pre.replace("ERA5","ERA5_CMIP6anom")

cmap = plot.Colormap('DryWet')
cmapp = plot.Colormap('IceFire')
cart_proj,lats,lons,hgt = get_geoinfo()

mbounds = map_bounds(reg)
filesinp = sorted(glob(f'{cfg.path_in}/{wrun_pre}/UIB_DAY_RAIN_*.nc'))
filesinf = sorted(glob(f'{cfg.path_in}/{wrun_fut}/UIB_DAY_RAIN_*.nc'))

fin_pre = xr.open_mfdataset(filesinp,concat_dim="time", combine="nested").load().sel(time=slice(str(cfg.syear),str(cfg.eyear)))
fin_fut = xr.open_mfdataset(filesinf,concat_dim="time", combine="nested").load().sel(time=slice(str(cfg.syear),str(cfg.eyear)))

lmean = MaxNLocator(nbins=15).tick_values(0,100)
lmax = MaxNLocator(nbins=15).tick_values(-100,100)



###########################################################
###########################################################

#Plotting
# Create a figure
fig, axs = plot.subplots(width=12,height=5,ncols=2,nrows=1,proj=cart_proj)


axs[0].add_feature(cfeature.COASTLINE,linewidth=0.5)
axs[0].add_feature(cfeature.BORDERS,linewidth=0.5)
axs[0].text(0.5,1.02,f'Present', fontsize='x-large', horizontalalignment='center', transform=axs[0].transAxes)
xseas_pre = fin_pre.RAIN.quantile(qtile,dim=['time']).squeeze()
xseas_fut = fin_fut.RAIN.quantile(qtile,dim=['time']).squeeze()

m0=axs[0].contourf(to_np(lons), to_np(lats), xseas_pre ,
                 transform=ccrs.PlateCarree(),
                 cmap=cmapp,extend='both')
axs[0].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs[0].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl0=axs[0].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False)
gl0.right_labels=False
gl0.top_labels=False     


axs[1].add_feature(cfeature.COASTLINE,linewidth=0.5)
axs[1].add_feature(cfeature.BORDERS,linewidth=0.5)
axs[1].text(0.5,1.02,f'Change', fontsize='x-large', horizontalalignment='center', transform=axs[1].transAxes)
xseas_pre = fin_pre.RAIN.quantile(qtile,dim=['time']).squeeze()
xseas_fut = fin_fut.RAIN.quantile(qtile,dim=['time']).squeeze()
dplot1 = (xseas_fut - xseas_pre)*100./xseas_pre
m1=axs[1].contourf(to_np(lons), to_np(lats), dplot1,levels=lmax,
                   transform=ccrs.PlateCarree(),
                    cmap=cmap,extend='both')

axs[1].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs[1].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl1=axs[1].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True, x_inline=False, y_inline=False)#,xlocs=range(-10,10,1), ylocs=range(20,60,1))
gl1.right_labels=False
gl1.top_labels=False

fig.colorbar(m0,length=0.7, loc='b',label='mm',col=1)
fig.colorbar(m1,length=0.7, loc='b',label='%',col=2)

plt.savefig(f'{cfg.path_out}/WRF_General/Change_RAIN_DAY_{qtile*100}th_ptile_{cfg.syear}-{cfg.eyear}_{reg}.png',dpi=150)
