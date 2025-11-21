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
import matplotlib.colors
import proplot as plot

from matplotlib.ticker import MaxNLocator

from wrf import (to_np, getvar,get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)


###########################################################
###########################################################
geo_file_name = "/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
geo_file = xr.open_dataset(geo_file_name)
lm_is=geo_file.LANDMASK.squeeze()
#####################################################################
#####################################################################

def get_geoinfo():

    fileref = nc.Dataset(geo_file_name)
    hgt = getvar(fileref, "HGT_M", timeidx=0)
    hgt = hgt.where(hgt>=0,0)
    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)

    return cart_proj,lats,lons,hgt

###########################################################
###########################################################

cart_proj,lats,lons,hgt = get_geoinfo()

med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc')
region_sq = np.zeros_like(med_mask['combined_mask'].values)
region_sq[80:669,193:1169]=1

mbounds = None

# Locations

locs_x_idx = [559,423,569,795,638,821,1091,989]#,335,433,866]
locs_y_idx = [258,250,384,527,533,407,174,425]#,119,254,506]
locs_names = ['Mallorca','Turis','Pyrenees','Rosiglione', 'Ardeche','Corte','Catania',"L'Aquila"]#'Almeria','Valencia','Barga']



cart_proj._threshold /= 100.
###########################################################
###########################################################

#Plotting
# Create a figure
fig, axs = plot.subplots(width=12,height=8,ncols=1,nrows=1,proj=cart_proj)
# axs.format(
#         suptitle="",
#         suptitlesize='20',
#         abc=False
#     )

axs[0].add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=102)
axs[0].add_feature(cfeature.BORDERS,linewidth=0.5,zorder=102)
axs[0].add_feature(cfeature.OCEAN, color='#80b3ff', zorder=100)

# Create terrain colormap
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
terrain_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors_land)
terrain_cmap.set_under([7/255, 146/255, 12/255])

# Define terrain levels (0 to 1500m)
terrain_levels = np.arange(0, 3200, 200)  # 0, 100, 200, ..., 1500
m0=axs[0].contourf(to_np(lons), to_np(lats), hgt.where(lm_is>0,0),
                 transform=ccrs.PlateCarree(),levels=terrain_levels, cmap=terrain_cmap, 
                 extend='max',zorder=99)

nx = hgt.shape[1]
ny = hgt.shape[0]

subregs = np.zeros_like(hgt.values)


for loc in range(len(locs_names)):
    locname = locs_names[loc]
    locx = lons[locs_y_idx[loc],locs_x_idx[loc]].values
    locy = lats[locs_y_idx[loc],locs_x_idx[loc]].values

    nyc = locs_y_idx[loc]
    nxc = locs_x_idx[loc]

    subregs[nyc-10:nyc+11,nxc-10:nxc+11]=1
    

    axs[0].plot(locx,locy,'ro',transform=ccrs.PlateCarree(),zorder=103)
    axs[0].text(locx+0.1,locy-0.1,locname,color='black',fontsize=10,transform=ccrs.PlateCarree(),zorder=103,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0))

axs[0].contourf(to_np(lons), to_np(lats), subregs,
                levels=[0.5, 1.5],
                colors='none',
                edgecolor='red',
                linewidth=1,
                transform=ccrs.PlateCarree(),
                zorder=101)

axs[0].contourf(to_np(lons), to_np(lats), region_sq,
                levels=[0.5, 1.5],
                colors='none',
                edgecolor='red',
                linewidth=2,
                transform=ccrs.PlateCarree(),
                zorder=101)

cs = axs[0].contourf(to_np(lons), to_np(lats), med_mask['combined_mask'].values,
                levels=[1.5, 2.5, 3.5],
                hatches=['////','....'],
                colors='none',
                edgecolor='black',
                linewidth=0.5,
                transform=ccrs.PlateCarree(),
                zorder=101)

# for i, collection in enumerate(cs.collections):
#     collection.set_edgecolor('red')


# Create and add gray border zone (50 grid points from edge)
border_width = 50
border_mask = np.zeros_like(hgt.values)
border_mask[:border_width, :] = 1  # Top
border_mask[-border_width:, :] = 1  # Bottom
border_mask[:, :border_width] = 1  # Left
border_mask[:, -border_width:] = 1  # Right

axs[0].contourf(to_np(lons), to_np(lats), border_mask,
                levels=[0.5, 1.5],
                colors=['gray'],
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=101)


axs[0].colorbar(m0, shrink=0.5, loc='b', label='Elevation (m)')
axs[0].set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs[0].set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl0=axs[0].gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False,zorder=103)
gl0.right_labels=False
gl0.top_labels=False

plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/EPICC_domain_with_regs.png',dpi=150)
