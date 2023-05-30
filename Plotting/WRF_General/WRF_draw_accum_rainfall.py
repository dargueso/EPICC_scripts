#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-02-14T13:02:06+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-02-14T13:02:13+01:00
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



from glob import glob

import xarray as xr
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wrf
import proplot as plt
#import matplotlib.pyplot as plt

###########################################################
###########################################################

###########################################################
############# USER MODIF ##################################
path_in = '/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/EPICC_2km_ERA5_HVC_GWD'
geopath_in = '/home/dargueso/share/geo_em_files/EPICC/'


############# END OF USER MODIF ###########################
###########################################################


###########################################################
###########################################################
#GEOGRAPHICAL INFO

geofile_ref = nc.Dataset(f'{geopath_in}/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc')
xbounds = wrf.cartopy_xlim(wrfin=geofile_ref)
ybounds = wrf.cartopy_ylim(wrfin=geofile_ref)
geo_proj = wrf.get_cartopy(wrfin=geofile_ref)


hgt = wrf.getvar(geofile_ref, "ter")
lats, lons = wrf.latlon_coords(hgt)


###########################################################
###########################################################

filesin = sorted(glob(f'{path_in}/UIB_01H_RAIN_20??-??.nc'))
fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")
varplot = fin.RAIN.sum(dim='time').squeeze()
###########################################################
###########################################################

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1,1,1,projection=geo_proj)


oce10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m')
lakes10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m')
rivers10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
ax.add_feature(oce10m , zorder=100,facecolor='lightgray')#facecolor=[24/255,  116/255,  205/255])
ax.add_feature(lakes10m, zorder=100,linewidth=0.5,edgecolor='k',facecolor='lightgray')
               #facecolor=[24/255,  116/255,  205/255])
ax.add_feature(rivers10m, edgecolor='lightgray',facecolor='none')#, facecolor=[24/255,  116/255,  205/255])
ax.coastlines(linewidth=0.4,zorder=100,resolution='10m')
ct=ax.contourf(lons,lats,varplot,cmap='Dense',extend='max',vmin=5,
               transform=ccrs.PlateCarree(),zorder=101)

gl=ax.gridlines(crs=ccrs.PlateCarree(), xlocs=range(-180,181,5), ylocs=range(-80,81,5),
                draw_labels=True, zorder=103,linewidth=0.2, color='k', alpha=1, linestyle='-')

ax.set_xlim(xbounds)
ax.set_ylim(ybounds)

ax.colorbar(ct, loc='b')
# cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.025])
# cbar=plt.colorbar(ct, cax=cbar_ax,orientation="horizontal")
# cbar.set_label ('Accumulated rainfall (mm)')

fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.18,wspace=0.1,hspace=0.5)
fig.savefig("./Accum_rainfall_test.png", bbox_inches='tight',dpi=300)
plt.close()
