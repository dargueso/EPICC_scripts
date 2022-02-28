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
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wrf
import proplot as plt
#import matplotlib.pyplot as plt
#from matplotlib.colors import BoundaryNorm

###########################################################
############# USER MODIF ##################################
path_in = '/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/EPICC_2km_ERA5_HVC_GWD/'
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

filesin = sorted(glob(f'{path_in}/UIB_DAY_RAIN_20??-??.nc'))
fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")

varcolor = fin.RAIN.compute().quantile(0.99,dim=['time'])
varsize = fin.RAIN.where(fin.RAIN>varcolor).sum('time')/fin.RAIN.sum('time')

varsizenorm = (varsize - varsize.min())/(varsize.max()-varsize.min())
###########################################################
###########################################################

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1,1,1,projection=geo_proj)


ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m'), zorder=100,facecolor='lightgray')
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m'), zorder=100,facecolor='darkgray')
ax.coastlines(linewidth=0.4,zorder=103,resolution='50m')

# cmap=rgb2cmap("/home/dargueso/share/colormaps/precip_11lev.rgb")
# norm = BoundaryNorm(np.arange(0,100,10), ncolors=cmap.N, clip=False)


# ct=ax.scatter(lons[::10,::10],lats[::10,::10],c=varcolor.values[::10,::10],s=20**(varsize.values[::10,::10]*3),cmap='IceFire',
#                transform=ccrs.PlateCarree(),zorder=101)

ct=ax.scatter(lons[::10,::10],lats[::10,::10],c=varcolor.values[::10,::10],s=varsize.values[::10,::10],smin=0,smax=500,cmap='IceFire',
              transform=ccrs.PlateCarree(),zorder=101)

gl=ax.gridlines(crs=ccrs.PlateCarree(), xlocs=range(-180,181,5), ylocs=range(-80,81,5),
                draw_labels=True, zorder=103,linewidth=0.2, color='k', alpha=1, linestyle='-')

# ax.set_xlim(xbounds)
# ax.set_ylim(ybounds)

ax.set_extent([-5,10,35,45])

#fig.colorbar(ct, ax=ax)
#ax.colorbar(ct, loc='b')

fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.18,wspace=0.1,hspace=0.5)
fig.savefig("./Scatter_P99TOT_test.png", bbox_inches='tight',dpi=300)
plt.close()
