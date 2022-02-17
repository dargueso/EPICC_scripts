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
freq = '01H'
###########################################################
###########################################################

def main():

    """  Split files into tiles """

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??.nc'))
    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']
    files_ref.close()

    Parallel(n_jobs=20)(delayed(split_files)(fin,nlons,nlats,tile_size) for fin in filesin)


    #Then concatenate using:
    # for ny in $(seq -s " " -f %03g 0 10); do for nx in $(seq -s " " -f %03g 0 10); do ncrcat UIB_10MIN_RAIN_*_${ny}y-${nx}x.nc UIB_10MIN_RAIN_2011-2020_${ny}y-${nx}x.nc ;done done


###########################################################
###########################################################

def split_files(fin,nlons,nlats,tile_size):

    """Split files based on longitude using xarray"""
    print(fin)

    finxr = xr.open_dataset(fin).load()


    for nnlon,slon in enumerate(range(0,nlons,tile_size)):
      for nnlat,slat in enumerate(range(0,nlats,tile_size)):

        fout = fin.replace(".nc",f'_{nnlat:03d}y-{nnlon:03d}x.nc')

        elon = slon + tile_size
        elat = slat + tile_size

        if elon > nlons: elon=nlons
        if elat > nlats: elat=nlats

        fin_tile = finxr.isel(x=slice(slon,elon),y=slice(slat,elat))
        fin_tile.to_netcdf(fout)

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
