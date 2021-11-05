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
from itertools import product
from joblib import Parallel, delayed

wrun = cfg.wrf_runs[0]
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = cfg.wet_value
tile_size = 50
freq = '01H'
###########################################################
###########################################################

def main():

    """ Calculating percentiles using a loop"""

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??.nc'))
    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']

    
    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]
    
    xytiles=list(product(latsteps, lonsteps))

    filespath = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??'
    print(f'Ej: {filespath}_000y-000x.nc')


    Parallel(n_jobs=20)(delayed(calc_percentile)(filespath,xytile[0],xytile[1],qtiles,mode) for xytile in xytiles)
    



#####################################################################
#####################################################################

def calc_percentile(filespath,ny,nx,qtiles,mode='wetonly'):
  print (f'Analyzing tile y: {ny} x: {nx}')
  filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))


  if len(filesin)==1:
    fin = xr.open_dataset(filesin[0]).load()
  else:
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested").load()

  if mode == 'wetonly':
    ptiles = fin.RAIN.where(fin.RAIN>wet_value).quantile(qtiles,dim=['time'])
  else:
    ptiles = fin.RAIN.quantile(qtiles,dim=['time'])
  
  
  fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}.nc'
  ptiles.to_netcdf(fout)
  fin.close()
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
