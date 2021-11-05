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
import epicc_config as cfg
from glob import glob

wrun = cfg.wrf_runs[0]
tile_size = 50
mode = 'wetonly'
freq = '01H'

###########################################################
###########################################################

def main():

    """  Split files into tiles """

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??.nc'))
    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']

    # filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_2011-2020_???y-???x_qtiles_wetonly.nc'))
    # fin_all = xr.open_mfdataset(filesin,combine='by_coords')

    filespath = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_2011-2020'
    latlongrid = []
    for nnlat in range(nlats//tile_size+1):
      print(nnlat)
      latlongrid.append([f'{filespath}_{nnlat:03d}y-{nnlon:03d}x_qtiles_{mode}.nc' for nnlon in range(nlons//tile_size+1)])
    fin_all = xr.open_mfdataset(latlongrid,combine='nested',concat_dim=["y", "x"])
    
    print(f'Ej: {filespath}_000y-000x_qtiles_{mode}.nc')
    fout = latlongrid[0][0].replace("_000y-000x_",f'_')
    fin_all.to_netcdf(fout)


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
