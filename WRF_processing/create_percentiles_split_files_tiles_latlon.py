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

wrf_runs = ['EPICC_2km_ERA5','EPICC_2km_ERA5_CMIP6anom']
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = cfg.wet_value
tile_size = 50
###########################################################
###########################################################

def main():

    """ Calculating percentiles using a loop"""
    for wrun in wrf_runs:

        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_20??-??.nc'))
        files_ref = xr.open_dataset(filesin[0])
        nlats = files_ref.sizes['y']
        nlons = files_ref.sizes['x']

        lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
        latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

        xytiles=list(product(latsteps, lonsteps))

        filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_10MIN_RAIN_20??-??'
        print(f'Ej: {filespath}_000y-000x.nc')


        Parallel(n_jobs=20)(delayed(calc_percentile)(filespath,xytile[0],xytile[1],qtiles,wrun,mode) for xytile in xytiles)




#####################################################################
#####################################################################

def calc_percentile(filespath,ny,nx,qtiles,wrun,mode='wetonly'):
  print (f'Analyzing tile y: {ny} x: {nx}')
  filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))

  if len(filesin)==1:
    fin = xr.open_dataset(filesin[0]).load()
  else:
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested").load().sel(time=slice(str(cfg.syear),str(cfg.eyear)))

  for fq in ['10MIN','01H','DAY']:
    if fq=='10MIN':
      fin_freq = fin
    elif fq=='01H':
      fin_freq = fin.resample(time=f"1H").sum('time')
    elif fq=='DAY':
      fin_freq = fin.resample(time=f"D").sum('time')



    #Year
    if mode == 'wetonly':
      ptiles = fin_freq.RAIN.where(fin_freq.RAIN>wet_value).quantile(qtiles,dim=['time'])
    else:
      ptiles = fin_freq.RAIN.quantile(qtiles,dim=['time'])
    fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}.nc'
    ptiles.to_netcdf(fout)


    #Season
    for ns,season in enumerate(['DJF','MAM','JJA','SON']):
      if mode == 'wetonly':
        ptiles = fin_freq.RAIN.where(fin_freq.RAIN>wet_value).groupby('time.season').quantile(qtiles,dim=['time']).sel(season=season)
      else:
        ptiles = fin_freq.RAIN.groupby('time.season').quantile(qtiles,dim=['time']).sel(season=season)
      fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}_{season}.nc'
      ptiles.to_netcdf(fout)

  fin.close()
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
