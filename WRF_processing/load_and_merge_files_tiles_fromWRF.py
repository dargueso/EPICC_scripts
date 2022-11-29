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

import xarray as xr
import numpy as np
from glob import glob


pathin = '/vg5/dargueso/postprocessed/EPICC/temp/EPICC_2km_ERA5'
nproc_x = 12
nproc_y = 48
var = 'HGT'
dim_x = "west_east"
dim_y = "south_north"


###########################################################
###########################################################

def main():

    """  Split files into tiles """

    filespath = f'{pathin}/UIB_HGT_2010-12-22'

    latlongrid = []
    for ytile in range(nproc_y):
        print(ytile,ytile*nproc_x)
        latlongrid.append([f'{filespath}_{(ytile*nproc_x+xtile):04d}.nc' for xtile in range(nproc_x)])

    # latlongrid = []
    # latlongrid.append([f'{filespath}_{ntile:04d}.nc' for ntile in range(nproc_x*nproc_y)])

    fin_all = xr.open_mfdataset(latlongrid,combine='nested',concat_dim=["y", "x"]).load()
    print(f'Output file: {filespath}.nc')
    fin_all.to_netcdf(f'{filespath}.nc')

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
