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
import numpy as np
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
path_in = '/vg6/dargueso-NO-BKUP/OBS_DATA/EOBS/'


############# END OF USER MODIF ###########################
###########################################################


fin_all = xr.open_dataset(f'{path_in}/rr_ens_mean_0.1deg_reg_v24.0e.nc')
fin = fin_all.sel(longitude=slice(-15,30),latitude=slice(35,65))


lats = fin.latitude.values
lons = fin.longitude.values
X,Y=np.meshgrid(lons,lats)

varcolor = fin.rr.compute().quantile(0.99,dim=['time'])
varsize = fin.rr.where(fin.rr>varcolor).sum('time')*100/fin.rr.sum('time')

varsizenorm = (varsize - varsize.min())/(varsize.max()-varsize.min())
###########################################################
###########################################################

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())


#oce10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m')
#lakes10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m')
#rivers10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
#ax.add_feature(oce10m , zorder=100,facecolor='lightgray')#facecolor=[24/255,  116/255,  205/255])
#ax.add_feature(lakes10m, zorder=100,linewidth=0.5,edgecolor='k',facecolor='lightgray')
               #facecolor=[24/255,  116/255,  205/255])
#ax.add_feature(rivers10m, edgecolor='lightgray',facecolor='none')#, facecolor=[24/255,  116/255,  205/255])
ax.coastlines(linewidth=0.4,zorder=103,resolution='50m')
ct=ax.scatter(X[::2,::2],Y[::2,::2],c=varcolor.values[::2,::2],s=varsizenorm.values[::2,::2]**(2)*0.1,cmap='IceFire',
               transform=ccrs.PlateCarree(),zorder=101)

gl=ax.gridlines(crs=ccrs.PlateCarree(), xlocs=range(-180,181,5), ylocs=range(-80,81,5),
                draw_labels=True, zorder=103,linewidth=0.2, color='k', alpha=1, linestyle='-')

ax.set_extent([-15,30,35,65])

#ax.colorbar(ct, loc='b')

fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.18,wspace=0.1,hspace=0.5)
fig.savefig("./EOBS_Scatter_P99TOT.png", bbox_inches='tight',dpi=300)
plt.close()
