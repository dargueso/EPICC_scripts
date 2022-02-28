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
import argparse
import dateparser
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
freq= '01H'


###########################################################
###########################################################


units = {'10MIN': f'%',
              '01H'  : f'%',
              'DAY'  : f'%',
              'MON'  : f'%'}

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
qtile = 0.999

wrun_pre = cfg.wrf_runs[0]
wrun_fut = wrun_pre.replace("ERA5","ERA5_CMIP6anom")

cmap = plot.Colormap('DryWet')
cart_proj,lats,lons,hgt = get_geoinfo()

mbounds = map_bounds(reg)

filesin_pre = sorted(glob(f'{cfg.path_in}/{wrun_pre}/{cfg.patt_in}_DAY_{varname}_????-??.nc'))
fin_all_pre = xr.open_mfdataset(filesin_pre,concat_dim="time", combine="nested")

filesin_fut = sorted(glob(f'{cfg.path_in}/{wrun_fut}/{cfg.patt_in}_DAY_{varname}_????-??.nc'))
fin_all_fut = xr.open_mfdataset(filesin_fut,concat_dim="time", combine="nested")


fin_pre = fin_all_pre.sel(time=slice(str(cfg.syear),str(cfg.eyear))).squeeze()
fin_fut = fin_all_fut.sel(time=slice(str(cfg.syear),str(cfg.eyear))).squeeze()



#tot_seconds = int((fin.isel(time=-1).time-fin.isel(time=0).time)*1e-9)
if reg!='EPICC':
    fin_reg_pre =  fin_pre.where((fin_pre.lat>=cfg.reg_coords[reg][0]) &\
                         (fin_pre.lat<=cfg.reg_coords[reg][2]) &\
                        (fin_pre.lon>=cfg.reg_coords[reg][1]) &\
                        (fin_pre.lon<=cfg.reg_coords[reg][3]),
                        drop=True)
    fin_reg_fut =  fin_fut.where((fin_fut.lat>=cfg.reg_coords[reg][0]) &\
                    (fin_fut.lat<=cfg.reg_coords[reg][2]) &\
                    (fin_fut.lon>=cfg.reg_coords[reg][1]) &\
                    (fin_fut.lon<=cfg.reg_coords[reg][3]),
                    drop=True)
else:
    fin_reg_pre = fin_pre
    fin_reg_fut = fin_fut



lmean = MaxNLocator(nbins=15).tick_values(-100,100)
lmax = MaxNLocator(nbins=15).tick_values(-100,100)

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
    mseas_pre = fin_pre[varname].groupby('time.season').mean(dim='time').sel(season=season)
    mseas_fut = fin_fut[varname].groupby('time.season').mean(dim='time').sel(season=season)

    dplot0= (mseas_fut - mseas_pre)*100./mseas_pre
    m0=axs[ns*2].contourf(to_np(lons), to_np(lats), dplot0,levels=lmean,
                 transform=ccrs.PlateCarree(),
                 cmap=cmap,extend='both')
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
        axs[ns*2+1].text(0.5,1.02,f'{freq} {qtile*100:0.2f}th percentile', fontsize='x-large', horizontalalignment='center', transform=axs[ns*2+1].transAxes)

    axs[ns*2+1].text(0.98,1.02,f'{season}', fontsize='x-large', horizontalalignment='right', transform=axs[ns*2+1].transAxes)
    #axs[ns*2+1].text(0.98,0.02,f'{labeltop[freq]}', fontsize='medium', horizontalalignment='right', transform=axs[ns*2+1].transAxes)
    #CS = axs[1].contour(to_np(lons), to_np(lats), fin[varname].max('time')*tot_seconds,levels=11,linewidth=0)

    fin_pre_seas = xr.open_dataset(f'{cfg.path_in}/{wrun_pre}/UIB_10MIN_RAIN_2013-2020_qtiles_all_{season}.nc')
    fin_fut_seas = xr.open_dataset(f'{cfg.path_in}/{wrun_fut}/UIB_10MIN_RAIN_2013-2020_qtiles_all_{season}.nc')
    xseas_pre = fin_pre_seas.RAIN.squeeze().sel(quantile=qtile).squeeze()
    xseas_fut = fin_fut_seas.RAIN.squeeze().sel(quantile=qtile).squeeze()

    dplot1 = (xseas_fut - xseas_pre)*100./xseas_pre
    m1=axs[ns*2+1].contourf(to_np(lons), to_np(lats), dplot1,levels=lmax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,extend='both')

    axs[ns*2+1].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
    axs[ns*2+1].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))


    gl1=axs[ns*2+1].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True, x_inline=False, y_inline=False)#,xlocs=range(-10,10,1), ylocs=range(20,60,1))
    gl1.right_labels=False
    gl1.top_labels=False
    #axs[ns*2+1].colorbar(m1,length=0.7, loc='b',label=units[freq])

fig.colorbar(m0,length=0.7, loc='b',label=units[freq],col=1)
fig.colorbar(m1,length=0.7, loc='b',label=units[freq],col=2)


#fig.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.15,wspace=0.1,hspace=0.2)
plt.savefig(f'{cfg.path_out}/WRF_General/Change_{varname}_{freq}_2013-2020_{reg}_{qtile*100:0.2f}th_seasons.png',dpi=150)
