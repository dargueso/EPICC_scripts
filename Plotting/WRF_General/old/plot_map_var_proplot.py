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
from matplotlib import colors
import proplot as plot

from glob import glob

import epicc_config as cfg

from wrf import (to_np, getvar, get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)

varname = ['PRNC']
wrun = cfg.wrf_runs[0]

def get_geoinfo():

    fileref = nc.Dataset(f'{cfg.path_wrfout}/{wrun}/out/{cfg.file_ref}')
    hgt = getvar(fileref, "ter")
    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)

    return cart_proj,lats,lons

###########################################################
###########################################################
#GEOGRAFICAL INFO
#cmap = utils.rgb2cmap(f'{cfg.path_cmaps}/precip_11lev.rgb')
cmap = plot.Colormap('IceFire')
#cmap = colors.ListedColormap(['#ffffff','#ffffe0','#edfac2','#cdffcd','#99f0b2','#53bd9f','#32a696','#3296b4','#0570b0','#05508c','#0a1f96','#2c0246','#6a2c5a','#db252b'])

cart_proj,lats,lons = get_geoinfo()
import pdb; pdb.set_trace()


filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_MON_PRNC*.nc'))
fin = xr.open_mfdataset(filesin,combine='by_coords')
tot_seconds = (fin.isel(time=-1).time-fin.isel(time=0).time)*1e-9

# Create a figure
fig, ax = plot.subplots(proj=cart_proj)

# Download and add the states and coastlines
ax.add_feature(states, linewidth=.1, edgecolor="black")
ax.coastlines('10m', linewidth=0.5)

levels = plot.arange(600, 1500, 50)
m = ax.contourf(to_np(lons), to_np(lats), fin.PRNC.mean('time')*tot_seconds, levels=levels,
             transform=ccrs.PlateCarree(),
             cmap=cmap,extend='max')

# Add the gridlines
ax.gridlines(color="black", linestyle="dotted",linewidth=0.5)

reg='BAL'
if reg != 'EPICC':

    reg_bounds = GeoBounds(CoordPair(lat=cfg.reg_coords[reg][0], lon=cfg.reg_coords[reg][1]),
                           CoordPair(lat=cfg.reg_coords[reg][2], lon=cfg.reg_coords[reg][3]))
    ax.set_xlim(cartopy_xlim(hgt, geobounds=reg_bounds))
    ax.set_ylim(cartopy_ylim(hgt, geobounds=reg_bounds))
else:
    ax.set_xlim(cartopy_xlim(hgt))
    ax.set_ylim(cartopy_ylim(hgt))

# Add a color bar
#fig.colorbar(m, shrink=.98, orientation='horizontal')
fig.colorbar(m, label='mm', loc='b', length=0.7)

#
# plt.title("Precipitation (mm hr-1)")

plt.savefig(f'{cfg.path_out}/tests/pr_proplot.pdf')
