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
sdate  = dt.datetime(2013,1,1)
edate  = dt.datetime(2021,1,1)
reg = -'EPICC'
freq= ['10MIN'] #['10MIN','01H','DAY','MON']

###########################################################
###########################################################


labeltop={'10MIN': f'{sdate.strftime("%H:%M %d %b %Y")}-{edate.strftime("%H:%M %d %b %Y")}',
              '01H'  : f'{sdate.strftime("%H:00 %d %b %Y")}-{edate.strftime("%H:00 %d %b %Y")}',
              'DAY'  : f'{sdate.strftime("%d %b %Y")}-{edate.strftime("%d %b %Y")}',
              'MON'  : f'{sdate.strftime("%b %Y")}-{edate.strftime("%d %b %Y")}'}
units = {'10MIN': f'mm/10min',
              '01H'  : f'mm hr-1',
              'DAY'  : f'mm day-1',
              'MON'  : f'mm month-1'}

def get_geoinfo():

    fileref = nc.Dataset(f'{cfg.path_wrfout}/{wrun}/out/{cfg.file_ref}')
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

wrun = cfg.wrf_runs[0]

cmap = plot.Colormap('IceFire')
cart_proj,lats,lons,hgt = get_geoinfo()

mbounds = map_bounds(reg)

filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_????-??.nc'))
fin_all = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")
fin = fin_all.sel(time=slice(sdate,edate)).squeeze()

filespt = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_2013-2020_qtiles_all.nc')
finpt = xr.open_dataset(filespt)


xmax = finpt.RAIN.max(skipna=True).values
mmax = fin.RAIN.mean(dim='time').max(skipna=True).values

lmean = MaxNLocator(nbins=15).tick_values(0,mmax)
lmax = MaxNLocator(nbins=15).tick_values(0,xmax)



###########################################################
###########################################################

#Plotting
# Create a figure
fig, axs = plot.subplots(width=12,height=15,ncols=2,nrows=4,proj=cart_proj)
#plot.rc.abc = True
#fig.suptitle.size='x-large'

axs.format(
        suptitle="Precipitation",
        suptitlesize='xx-large',
        abc=True, abcloc='ul',
        #grid=False, xticks=25, yticks=5
    )

###########################################################
###########################################################

for ns,season in enumerate(['DJF','MAM','JJA','SON']):
    axs[ns*2].add_feature(cfeature.COASTLINE,linewidth=0.5)
    axs[ns*2].add_feature(cfeature.BORDERS,linewidth=0.5)
    if ns == 0:
        axs[ns*2].text(0.5,1.02,f'Mean', fontsize='x-large', horizontalalignment='center', transform=axs[ns*2].transAxes)
    #axs[0].text(0.98,0.92,f'{labeltop[freq]}', fontsize='medium', horizontalalignment='right', transform=axs[0].transAxes)
    #CS = axs[0].contour(to_np(lons), to_np(lats), fin[varname].mean('time')*tot_seconds,levels=11,linewidth=0)
    dplot0= fin[varname].groupby('time.season').mean(dim='time').sel(season=season)
    m0=axs[ns*2].contourf(to_np(lons), to_np(lats), dplot0,levels=lmean,
                 transform=ccrs.PlateCarree(),
                 cmap=cmap,extend='max')
    axs[ns*2].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
    axs[ns*2].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
    gl0=axs[ns*2].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False)#,xlocs=range(-10,10,1), ylocs=range(20,60,1))
    gl0.right_labels=False
    gl0.top_labels=False
    #axs[ns*2].colorbar(m0,length=0.7, loc='b',label='mm day-1')
    ###########################################################
    ###########################################################


    axs[ns*2+1].add_feature(cfeature.COASTLINE,linewidth=0.5)
    axs[ns*2+1].add_feature(cfeature.BORDERS,linewidth=0.5)
    if ns == 0:
        axs[ns*2+1].text(0.5,1.02,f'{freq} Maximum Rate', fontsize='x-large', horizontalalignment='center', transform=axs[ns*2+1].transAxes)

    axs[ns*2+1].text(0.98,1.02,f'{season}', fontsize='x-large', horizontalalignment='right', transform=axs[ns*2+1].transAxes)


    filespt_seas = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_2013-2020_qtiles_all_{season}.nc')
    finpt_seas = xr.open_dataset(filespt_seas)
    dplot1 = finpt_seas.RAIN.squeeze().sel(quantile=qtile).squeeze()
    m1=axs[ns*2+1].contourf(to_np(lons), to_np(lats), dplot1.where(dplot1>0.1),levels=lmax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,extend='max')

    axs[ns*2+1].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
    axs[ns*2+1].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))


    gl1=axs[ns*2+1].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True, x_inline=False, y_inline=False)#,xlocs=range(-10,10,1), ylocs=range(20,60,1))
    gl1.right_labels=False
    gl1.top_labels=False
    #axs[ns*2+1].colorbar(m1,length=0.7, loc='b',label=units[freq])

fig.colorbar(m0,length=0.7, loc='b',label=units[freq],col=1)
fig.colorbar(m1,length=0.7, loc='b',label=units[freq],col=2)


#fig.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.15,wspace=0.1,hspace=0.2)
plt.savefig(f'{cfg.path_out}/WRF_General/{varname}_{freq}_{sdate.strftime("%Y-%m-%d_%H-%M")}-{edate.strftime("%Y-%m-%d_%H-%M")}_{reg}_seasons_{wrun}.png', dpi=150)
