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
import matplotlib.patches as mpatches
import proplot as plot

import epicc_config as cfg

from matplotlib.ticker import MaxNLocator

from wrf import (to_np, getvar,get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)


wrun = cfg.wrf_runs[0]
reg = 'EPICC'
###########################################################
###########################################################
geo_file = xr.open_dataset(cfg.geofile_ref)
lm_is=geo_file.LANDMASK.squeeze()
#####################################################################
#####################################################################

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

cart_proj,lats,lons,hgt = get_geoinfo()
mbounds = map_bounds(reg)

cart_proj._threshold /= 100.
###########################################################
###########################################################

#Plotting
# Create a figure
fig, axs = plot.subplots(width=12,height=8,ncols=1,nrows=1,proj=cart_proj)
axs.format(
        suptitle="Elevation",
        suptitlesize='20',
        abc=False
    )

axs[0].add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=101)
axs[0].add_feature(cfeature.BORDERS,linewidth=0.5,zorder=101)

cmap = plot.Colormap('IceFire')
cmap_kw = {'left': 0}
m0=axs[0].contourf(to_np(lons), to_np(lats), hgt.where(lm_is>0,0),
                 transform=ccrs.PlateCarree(),cmap=cmap,
                 cmap_kw=cmap_kw,levels = range(0,3200,200),extend='both')

for reg in ['BAL']:#['SWM','NWM']:

    lon1=cfg.reg_coords[reg][1]
    lon2=cfg.reg_coords[reg][3]
    lat1=cfg.reg_coords[reg][0]
    lat2=cfg.reg_coords[reg][2]

    dom_sq = axs[0].add_patch(mpatches.Rectangle(xy=[lon1, lat1], width=lon2-lon1, height=lat2-lat1,
                                edgecolor='red',
                                facecolor='none',
                                linewidth=2,
                                zorder=102,
                                transform=ccrs.PlateCarree()))

    if reg == 'WME':
        axs[0].text(lon1+0.5, lat2-1, reg,size=25, color='red', zorder=102,ha='left',transform=ccrs.PlateCarree())
    else:
        axs[0].text(lon2-0.5, lat2-1, reg,size=20, color='red', zorder=102,ha='right',transform=ccrs.PlateCarree())



axs[0].colorbar(m0, loc='b', label='m')
axs[0].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs[0].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl0=axs[0].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False)
gl0.right_labels=False
gl0.top_labels=False

plt.savefig(f'{cfg.path_out}/WRF_General/EPICC_domain_with_regs.png',dpi=150)
