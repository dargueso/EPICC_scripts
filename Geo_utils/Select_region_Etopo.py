#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-04-26T16:36:39+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-04-26T16:36:42+02:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files: ETOPO1_Bed_g_gmt4.grd
#
#####################################################################
"""

import xarray as xr
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-lons", "--longitudes" ,dest="lons",nargs=2, type=float,help="Range of longitudes to select in format lon1,lon2 (-180,180)",metavar=('lon1', 'lon2'))
parser.add_argument("-lats", "--latitudes"  ,dest="lats",nargs=2, type=float,help="Range of latitudes to select in format lat1,lat2 (-90,90)",metavar=('lat1', 'lat2'))
parser.add_argument("-o", "--fileout", dest='fileout',type=str,help="Output filename",metavar=('fileout'),default = "ETOPO1_clipped.nc")
args = parser.parse_args()



lon1,lon2 = args.lons
lat1,lat2 = args.lats
fileout = args.fileout


fin = xr.open_dataset("ETOPO1_Bed_g_gmt4.grd")

fin_sel = fin.sel(x=slice(lon1,lon2),y=slice(lat1,lat2),drop=True)

fin_sel.to_netcdf(fileout)
