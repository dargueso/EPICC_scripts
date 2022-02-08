#!/usr/bin/env python

""" Script_name.py

Author: Daniel Argueso @ CCRC, UNSW. Sydney (Australia)
email: d.argueso@ unsw.edu.au
Created: Mon Jun  1 11:06:59 AEST 2015

"""

import xarray as xr
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob as glob
import re

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import wrf
import string

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

###########################################################
###########################################################
def rgb2cmap(filename,base='255'):
  """Function to read a rgb file (NCL colortables) and convert them to matplotlib colormap

     Author: Daniel Argueso @ CCRC, UNSW. Sydney (Australia)
  """

  filein=open(filename)
  lines=filein.readlines()
  colors=[]

  for line in lines[2:]:
      line=re.sub('\s+',' ',line)
      li=line.strip()
      if li:
          values=li.split(' ')
          if base == '255':
            new_values=[i/255. for i in map(int,values[:3])]
          else:
            new_values=[i for i in map(float,values[:3])]
          colors.append(new_values)
  cmap=ListedColormap(colors)
  cmap.set_over(colors[-1])
  cmap.set_under(colors[0])

  return cmap
###########################################################
###########################################################



domains=['d01','d02','d03']#,'d04']
cmap=rgb2cmap("/home/dargueso/share/colormaps/OceanLakeLandSnow.rgb")





for domain in domains:

    geofile_ref="./geo_em.%s.nc" %(domain)

    if domain=='d01':

        #getting geographical info for plotting
        geo_file = xr.open_dataset(geofile_ref)
        geo_file_lm = geo_file.HGT_M.squeeze()
        xbounds = wrf.cartopy_xlim(wrfin=nc.Dataset(geofile_ref))
        ybounds = wrf.cartopy_ylim(wrfin=nc.Dataset(geofile_ref))
        geo_proj = wrf.get_cartopy(wrfin=nc.Dataset(geofile_ref))
        lat = geo_file.XLAT_M.squeeze().values
        lon = geo_file.XLONG_M.squeeze().values


        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(1,1,1,projection=geo_proj)
        ax.coastlines(linewidth=0.4,zorder=102,resolution='50m')
        ct=ax.contourf(lon,lat,geo_file_lm,levels=np.arange(-800,3400,200),cmap='terrain',extend='both',transform=ccrs.PlateCarree(),zorder=101)
        gl=ax.gridlines(crs=ccrs.PlateCarree(), xlocs=range(-180,181,5), ylocs=range(-80,81,5),
                         draw_labels=True, zorder=102,
                         linewidth=0.2, color='k', alpha=1, linestyle='-')

        oce50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
        lakes50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')

        ax.add_feature(oce50m , zorder=100,facecolor=[24/255,  116/255,  205/255])
        ax.add_feature(cfeature.LAKES, zorder=100,linewidth=0.5,edgecolor='k',facecolor=[24/255,  116/255,  205/255])


        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xpadding = 10

        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)



    else:
        geo_file = xr.open_dataset(geofile_ref)
        lat = geo_file.XLAT_M.squeeze().values
        lon = geo_file.XLONG_M.squeeze().values
        border_lat=np.concatenate((lat[:,0],lat[-1,:],lat[::-1,-1],lat[0,::-1]))
        border_lon=np.concatenate((lon[:,0],lon[-1,:],lon[::-1,-1],lon[0,::-1]))
        ax.plot(border_lon,border_lat,'k',transform=ccrs.PlateCarree(),zorder=102)

    geo_file.close()

cbar_ax = fig.add_axes([0.25, 0.1, 0.3, 0.025])
cbar=plt.colorbar(ct, cax=cbar_ax,orientation="horizontal")
cbar.set_label ('Elevation (m)')

fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.18,wspace=0.1,hspace=0.5)
fig.savefig("./WRF_DOMAINS.png", bbox_inches='tight',dpi=300)
plt.close()
