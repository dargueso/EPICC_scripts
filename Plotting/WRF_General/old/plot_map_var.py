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

import string
from glob import glob
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from mylibs import visualization as vis
from mylibs import io,utils
import epicc_config as cfg

from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)

varname = ['PRNC']
wrun = cfg.wrf_runs[0]

cmap = utils.rgb2cmap(f'{cfg.path_cmaps}/precip_11lev.rgb')
fileref = nc.Dataset(f'{cfg.path_wrfout}/{wrun}/out/{cfg.file_ref}')
# Get the sea level pressure
hgt = getvar(fileref, "ter")
# Get the latitude and longitude points
lats, lons = latlon_coords(hgt)

# Get the cartopy mapping object
cart_proj = get_cartopy(hgt)


filesin = sorted(glob(f'{cfg.path_in}/{wrun}/20??/{cfg.patt_in}_MON_PRNC*.nc'))
fin = xr.open_mfdataset(filesin,combine='by_coords')


# Create a figure
fig = plt.figure(figsize=(12,6))
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m",
                             facecolor="none",
                             name="admin_1_states_provinces_shp")
ax.add_feature(states, linewidth=.5, edgecolor="black")
ax.coastlines('10m', linewidth=0.8)

# Make the contour outlines and filled contours for the smoothed sea level
# pressure.
#plt.contour(to_np(lons), to_np(lats), to_np(hgt), 10, colors="black",
#            transform=ccrs.PlateCarree())
plt.contourf(to_np(lons), to_np(lats), fin.PRNC.sum('time')*3600., 10,
             transform=ccrs.PlateCarree(),
             cmap=cmap)

# Add a color bar
plt.colorbar(ax=ax, shrink=.98)

# Set the map bounds


# reg_bounds = GeoBounds(CoordPair(lat=cfg.reg_coords['BA'][0], lon=cfg.reg_coords['BA'][1]),
#                        CoordPair(lat=cfg.reg_coords['BA'][2], lon=cfg.reg_coords['BA'][3]))
# ax.set_xlim(cartopy_xlim(hgt, geobounds=reg_bounds))
# ax.set_ylim(cartopy_ylim(hgt, geobounds=reg_bounds))
ax.set_xlim(cartopy_xlim(hgt))
ax.set_ylim(cartopy_ylim(hgt))

# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

plt.title("Precipitation (mm hr-1)")

plt.savefig(f'{cfg.path_out}/tests/pr.png')

# geo_proj,xbounds,ybounds,lats,lons = vis.get_info_from_geo_em(cfg.geofile_ref)

#
# filesin = sorted(glob(f'{cfg.path_in}/{wrun}/20??/{cfg.patt_in}_MON_PRNC*.nc'))
# fin = xr.open_mfdataset(filesin,combine='by_coords')

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1,1,1,projection=geo_proj)
# ax.set_title(f'{string.ascii_lowercase[0]}',loc='left',fontsize='xx-large',fontweight='bold')
# ax.coastlines(linewidth=1,zorder=102,resolution='50m')
# ct=ax.contourf(lons,lats,fin.PRNC.mean('time')*3600.,levels=np.arange(0,11,1),cmap=cmap,extend='both',transform=ccrs.PlateCarree())

# oce50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
# lakes50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')
#
# ax.add_feature(oce50m , zorder=100,facecolor=[24/255,  116/255,  205/255])
# ax.add_feature(lakes50m, zorder=100,linewidth=0.5,edgecolor='k',facecolor=[24/255,  116/255,  205/255])
#
# # # *must* call draw in order to get the axis boundary used to add ticks:
# fig.canvas.draw()

# Define gridline locations and draw the lines using cartopy's built-in gridliner:
# xticks = [-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
# yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# ax.gridlines(xlocs=xticks, ylocs=yticks)
#
# # Label the end-points of the gridlines using the custom tick makers:
# ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
# ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
# vis.lambert_xticks(ax, xticks)
# vis.lambert_yticks(ax, yticks)
#
# # Set the map bounds
# ax.set_xlim(xbounds)
# ax.set_ylim(ybounds)

# Adding colorbar

# cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.025])
# cbar=plt.colorbar(ct, cax=cbar_ax,orientation="horizontal")
# cbar.set_label ('Precipitation (mm)')
# fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.18,wspace=0.1,hspace=0.5)
# plt.savefig(f'{cfg.path_out}/tests/pr.png')
