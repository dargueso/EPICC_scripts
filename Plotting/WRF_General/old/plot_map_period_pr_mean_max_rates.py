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
#import proplot as plot
from optparse import OptionParser
import dateparser
import datetime as dt
import sys

import string
from glob import glob
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from mylibs import visualization as vis
from mylibs import io,utils
import epicc_config as cfg

from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)



#### READING INPUT FILE ######
### Options

parser = OptionParser()

parser.add_option("-s", "--start", dest="sdatestr",
help="Starting date  of the period to plot in format 2019-09-11 09:00\n [default: 2019-09-11 09:00]",metavar="DATE",default='2019-09-11 09:00')

parser.add_option("-e", "--end", dest="edatestr",
help="Ending date  of the period to plot in format 2019-09-15 09:30\n [default: 2019-09-15 09:30]",metavar="DATE",default='2019-09-15 09:30')

parser.add_option("-f", "--freq", dest="freq",
help="Frequency to plot from 10min to monthly\n [default: hourly]",metavar="FREQ",default='01H')

parser.add_option("-v", "--var", dest="var",
help="Variable to plot \n [default: PRNC]",metavar="VAR",default='PRNC')

parser.add_option("-r", "--reg", dest="reg",
help="Region to plot \n [default: EPICC]",metavar="REG",default='EPICC')

(opts, args) = parser.parse_args()


varname = opts.var
sdatestr  = opts.sdatestr
edatestr  = opts.edatestr
reg = opts.reg
freq=opts.freq


freq_seconds = {'10MIN':600.,'01H':3600.,'DAY':86400}

# sdate=dt.datetime.strptime(sdatestr,'%Y-%m-%d %H:%M')
# edate=dt.datetime.strptime(edatestr,'%Y-%m-%d %H:%M')
sdate=dateparser.parse(sdatestr)
edate=dateparser.parse(edatestr)



if freq not in ['10MIN','01H','DAY','MON']:
    sys.exit("Frequency not available, choose between 10MIN,01H,DAY and MON")
if ((reg not in cfg.reg_coords.keys()) & (reg!='EPICC')):
    print(f"Currently available regions: {list(cfg.reg_coords.keys())}")
    sys.exit("Region not defined, check regions in epicc_config.py file or use default")

if freq == '10MIN':
    labeltop=f'{sdate.strftime("%H:%M %d %b %Y")}-{edate.strftime("%H:%M %d %b %Y")}'
elif freq == '01H':
    labeltop=f'{sdate.strftime("%H:00 %d %b %Y")}-{edate.strftime("%H:00 %d %b %Y")}'
elif freq == 'DAY':
    labeltop=f'{sdate.strftime("%d %b %Y")}-{edate.strftime("%d %b %Y")}'
else:
    labeltop=f'{sdate.strftime("%b %Y")}-{edate.strftime("%d %b %Y")}'

wrun = cfg.wrf_runs[0]

cmap = utils.rgb2cmap(f'{cfg.path_cmaps}/precip_11lev.rgb')
#cmap = plot.Colormap('IceFire')
fileref = nc.Dataset(f'{cfg.path_wrfout}/{wrun}/out/{cfg.file_ref}')
# Get the sea level pressure
hgt = getvar(fileref, "ter")
# Get the latitude and longitude points
lats, lons = latlon_coords(hgt)
# Get the cartopy mapping object
cart_proj = get_cartopy(hgt)

if sdate.year != edate.year:
    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_*.nc'))
else:
    if sdate.month != edate.month:
        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_{sdate.year}-*.nc'))
    else:
        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_{sdate.year}-{sdate.month:02d}.nc'))

fin_all = xr.open_mfdataset(filesin,combine='by_coords')
fin = fin_all.sel(time=slice(sdate,edate)).squeeze()
tot_seconds = (fin.isel(time=-1).time-fin.isel(time=0).time)*1e-9

# Create a figure
fig = plt.figure(figsize=(12,5))
spec = fig.add_gridspec(ncols=2, nrows=1)
# Set the GeoAxes to the projection used by WRF
ax0 = fig.add_subplot(spec[0, 0],projection=cart_proj)
#ax0.coastlines('10m', linewidth=0.8)
ax0.add_feature(cfeature.COASTLINE,linewidth=0.5)
ax0.add_feature(cfeature.BORDERS,linewidth=0.5)
ax0.text(0.5,1.02,f'Mean Rate', fontsize='x-large', horizontalalignment='center', transform=ax0.transAxes)
ax0.text(0.98,0.92,f'{labeltop}', fontsize='medium', horizontalalignment='right', transform=ax0.transAxes)
#ax0.plot(-0.9480,38.0856,marker='o',mfc='r',mec=None,transform=ccrs.PlateCarree())
CS = ax0.contour(to_np(lons), to_np(lats), fin[varname].mean('time')*tot_seconds,linewidth=0)
plt.contourf(to_np(lons), to_np(lats), fin[varname].mean('time')*tot_seconds, CS.levels[1:],
                transform=ccrs.PlateCarree(),
                cmap=cmap)

# Set the map bounds

if reg != 'EPICC':

    reg_bounds = GeoBounds(CoordPair(lat=cfg.reg_coords[reg][0], lon=cfg.reg_coords[reg][1]),
                           CoordPair(lat=cfg.reg_coords[reg][2], lon=cfg.reg_coords[reg][3]))
    ax0.set_xlim(cartopy_xlim(hgt, geobounds=reg_bounds))
    ax0.set_ylim(cartopy_ylim(hgt, geobounds=reg_bounds))
else:
    ax0.set_xlim(cartopy_xlim(hgt))
    ax0.set_ylim(cartopy_ylim(hgt))
# Add the gridlines
ax0.gridlines(color="black", linestyle="dotted")
plt.colorbar(ax=ax0, shrink=.98, orientation='horizontal')

ax1 = fig.add_subplot(spec[0, 1],projection=cart_proj)
ax1.coastlines('10m', linewidth=0.8)
ax1.text(0.5,1.02,f'{freq} Maximum Rate', fontsize='x-large', horizontalalignment='center', transform=ax1.transAxes)
ax1.text(0.98,0.92,f'{labeltop}', fontsize='medium', horizontalalignment='right', transform=ax1.transAxes)

CS = ax1.contour(to_np(lons), to_np(lats), fin[varname].max('time')*tot_seconds)
plt.contourf(to_np(lons), to_np(lats), fin[varname].max('time')*tot_seconds, CS.levels[1:],
                transform=ccrs.PlateCarree(),
                cmap=cmap)


if reg != 'EPICC':

    reg_bounds = GeoBounds(CoordPair(lat=cfg.reg_coords[reg][0], lon=cfg.reg_coords[reg][1]),
                           CoordPair(lat=cfg.reg_coords[reg][2], lon=cfg.reg_coords[reg][3]))
    ax1.set_xlim(cartopy_xlim(hgt, geobounds=reg_bounds))
    ax1.set_ylim(cartopy_ylim(hgt, geobounds=reg_bounds))
else:
    ax1.set_xlim(cartopy_xlim(hgt))
    ax1.set_ylim(cartopy_ylim(hgt))
# Add the gridlines
ax1.gridlines(color="black", linestyle="dotted")
plt.colorbar(ax=ax1, shrink=.98, orientation='horizontal')


plt.suptitle("Precipitation (mm)")
fig.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.15,wspace=0.1,hspace=0.2)
plt.savefig(f'{cfg.path_out}/WRF_General/{varname}_{freq}_{sdate.strftime("%Y-%m-%d_%H-%M")}-{edate.strftime("%Y-%m-%d_%H-%M")}_{reg}.png')
