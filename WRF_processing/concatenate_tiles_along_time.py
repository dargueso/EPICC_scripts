#!/usr/bin/env python
'''
@File    :  concatenate_location_box_postprocessed.py
@Time    :  2025/11/08 23:55:08
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  None
@Desc    :  None
'''

import xarray as xr
import epicc_config as cfg
from glob import glob
import os
from itertools import product
from joblib import Parallel, delayed
wrun = cfg.wrf_runs[1]
###########################################################
###########################################################
tile_size = 50
freq = '01H'
buffer = 0
wrf_runs = ['EPICC_2km_ERA5','EPICC_2km_ERA5_CMIP6anom']
###########################################################
###########################################################

def main():

    for wrun in wrf_runs:

        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/RAIN/{cfg.patt_in}_01H_RAIN_20??-??.nc'))
        files_ref = xr.open_dataset(filesin[0])
        nlats = files_ref.sizes['y']
        nlons = files_ref.sizes['x']
        lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
        latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]
        xytiles=list(product(latsteps, lonsteps))

        filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_{tile_size}/{cfg.patt_in}_{freq}_RAIN_20??-??'
        print(f'Ej: {filespath}_000y-000x.nc')

        Parallel(n_jobs=20)(delayed(concatenate_tiles_time)(filespath,xytile[0],xytile[1],wrun,buffer) for xytile in xytiles)



def concatenate_tiles_time(filespath,ny,nx,wrun,buffer=0):

    
    if buffer !=0:
        buffer_str = f'_{buffer:03d}buffer'
    else:
        buffer_str = ''
    print (f'Analyzing tile y: {ny} x: {nx}')
    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x{buffer_str}.nc'))
    fout = f'{os.path.dirname(filesin[0])}/{cfg.patt_in}_{freq}_RAIN_{ny}y-{nx}x{buffer_str}.nc'
    if os.path.exists(fout):
        print (f'File {ny}y-{nx}x exists. Skipping...')
        return
    data_list = []
    for fin in filesin:
        ds = xr.open_dataset(fin).squeeze()
        data_list.append(ds)
    
    combined = xr.concat(data_list, dim='time')
    combined.to_netcdf(fout)


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
