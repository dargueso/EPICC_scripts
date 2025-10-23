#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-06-17T11:53:02+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-06-17T11:53:43+02:00
#
# @Project@ EPICC
# Version: 1.0
# Description: Script to calculate percentiles and wet-percentiles from postprocessed files
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import pdb
import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
import time
import subprocess as subprocess
from joblib import Parallel, delayed


wrun = cfg.wrf_runs[0]
tile_size = 50
freq = 'DAY'
buffer= 25 
###########################################################
###########################################################

def main():

    """  Split files into tiles """

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/rainfall_probability_optimized_conditional_5mm_bins.nc'))
    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']
    files_ref.close()

    Parallel(n_jobs=1)(delayed(split_files)(fin,nlons,nlats,tile_size, buffer) for fin in filesin)

    #split_files(f'{cfg.path_in}/{wrun}/UIB_LANDMASK.nc',nlons,nlats,tile_size)

    #Then concatenate using:
    # for ny in $(seq -s " " -f %03g 0 10); do for nx in $(seq -s " " -f %03g 0 10); do ncrcat UIB_10MIN_RAIN_*_${ny}y-${nx}x.nc UIB_10MIN_RAIN_2011-2020_${ny}y-${nx}x.nc ;done done

###########################################################
###########################################################

def split_files(fin,nlons,nlats,tile_size, buffer=0):

    """Split files based on longitude using xarray"""
    print(fin)

    finxr = xr.open_dataset(fin).load()


    for nnlon,stlon in enumerate(range(0,nlons,tile_size)):
      for nnlat,stlat in enumerate(range(0,nlats,tile_size)):
        
        if buffer == 0:
            fout = fin.replace(".nc",f'_{nnlat:03d}y-{nnlon:03d}x.nc')
        else:
            fout = fin.replace(".nc",f'_{nnlat:03d}y-{nnlon:03d}x_{buffer:03d}buffer.nc')
       
        print("nnlat, nnlon:", nnlat, nnlon)
        print("stlon, stlat before buffer:", stlon, stlat)

        slon = stlon - buffer
        slat = stlat - buffer

        elon = stlon + tile_size + 2*buffer
        elat = stlat + tile_size + 2*buffer



        if elon > nlons: elon=nlons
        if elat > nlats: elat=nlats
        if slon < 0: slon=0
        if slat < 0: slat=0

        print("slon, slat after buffer:", slon, slat)
        fin_tile = finxr.isel(x=slice(slon,elon),y=slice(slat,elat))
        fin_tile.to_netcdf(fout)

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
