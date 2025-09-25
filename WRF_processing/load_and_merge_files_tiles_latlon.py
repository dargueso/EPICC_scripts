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
#wrun = 'EPICC_2km_ERA5_CMIP6anom'
tile_size = 50

# #filespath = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_2013-2020'
# filespath = f'{cfg.path_in}/{wrun}/ptiles_tiles_50/UIB_10MIN_RAIN_2013-2020'
# #filessuffix = f'qtiles_{mode}'
# filessuffix = '_qtiles_all_DJF'

###########################################################
###########################################################

def main():

    """  Split files into tiles """
    filespatterns = [#f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_NDI',
                     #f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_top100_NDI',
                     f'{cfg.path_in}/{wrun}/rainfall_probability_optimized_conditional',
                    #  f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_NDI_DJF',
                    #  f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_NDI_MAM',
                    #  f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_NDI_JJA',
                    #  f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_NDI_SON',
                     ]
    for filespath in filespatterns:
      #filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/Hourly_decomposition_NDI_SON'

      filessuffix = f''
      filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_20??-??.nc'))
      files_ref = xr.open_dataset(filesin[0])
      nlats = files_ref.sizes['y']
      nlons = files_ref.sizes['x']

      latlongrid = []
      for nnlat in range(nlats//tile_size+1):
        print(nnlat)
        latlongrid.append([f'{filespath}_{nnlat:03d}y-{nnlon:03d}x{filessuffix}.nc' for nnlon in range(nlons//tile_size+1)])
      fin_all = xr.open_mfdataset(latlongrid,combine='nested',concat_dim=["y", "x"],engine='netcdf4').load()
      print(f'Ej: {filespath}_000y-000x{filessuffix}.nc')
      fout = latlongrid[0][0].replace("_000y-000x",f'')
      print(f'Output file: {fout}')
      fin_all.to_netcdf(fout)

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
