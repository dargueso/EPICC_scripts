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


###########################################################
###########################################################

#Plotting
# Create a figure
fig, axs = plot.subplots(width=12,height=8,ncols=1,nrows=1,proj=cart_proj)
axs.format(
        suptitle="WRF DOMAIN",
        suptitlesize='40',
        abc=False
    )

axs[0].add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=101)
axs[0].add_feature(cfeature.BORDERS,linewidth=0.5,zorder=101)

# oce50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
# lakes50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')
# axs[0].add_feature(oce50m , zorder=100,facecolor=[197/255,  230/255,  219/255])
# axs[0].add_feature(lakes50m, zorder=100,linewidth=0.5,edgecolor='k',facecolor=[197/255,  230/255,  219/255])

cmap = plot.Colormap('IceFire')

cmap_kw = {'left': 0}
m0=axs[0].contourf(to_np(lons), to_np(lats), hgt.where(lm_is>0,0),
                 transform=ccrs.PlateCarree(),cmap=cmap,
                 cmap_kw=cmap_kw,levels = range(0,3200,200),extend='both')
axs[0].colorbar(m0, loc='b', label='m')
axs[0].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs[0].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl0=axs[0].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False)
gl0.right_labels=False
gl0.top_labels=False    

plt.savefig(f'{cfg.path_out}/WRF_General/EPICC_domain{reg}.png',dpi=150)


#####################################################################
#####################################################################

# #NO PROPLOT
# fig = plt.figure(figsize=(12,10))
# ax = plt.axes(projection=cart_proj)

# mbounds = GeoBounds(CoordPair(lat=cfg.reg_coords[reg][0], lon=cfg.reg_coords[reg][1]),
#                     CoordPair(lat=cfg.reg_coords[reg][2], lon=cfg.reg_coords[reg][3]))


# ax.set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
# ax.set_ylim(cartopy_ylim(hgt,geobounds=mbounds))

# states = cfeature.NaturalEarthFeature(category="cultural", scale="50m",
#                              facecolor="none",
#                              name="admin_1_states_provinces_shp")
# ax.add_feature(states, linewidth=.5, edgecolor="black")
# ax.coastlines('10m', linewidth=0.8)

# plt.pcolormesh(to_np(lons), to_np(lats), hgt.where(lm_is>0,0).squeeze(),
#                  transform=ccrs.PlateCarree(),
#                  cmap='terrain')
# ax.set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
# ax.set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
# #plt.colorbar(ax=ax, shrink=.98,label='m')
# ax.gridlines(color="black", linestyle="dotted")

# plt.title("Precipitation (mm hr-1)")
# plt.savefig(f'{cfg.path_out}/WRF_General/EPICC_domain_{reg}.pdf')
