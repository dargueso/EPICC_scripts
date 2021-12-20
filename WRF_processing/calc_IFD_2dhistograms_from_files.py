#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-11-30T13:10:07+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-11-30T13:10:32+01:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import xarray as xr
import epicc_config as cfg
from glob import glob
import numpy as np
from itertools import product
from itertools import groupby
from joblib import Parallel, delayed
from xhistogram.xarray import histogram


# wrun = cfg.wrf_runs[0]
wrun = 'EPICC_2km_ERA5_HVC_GWD'
tile_size = 50

fileref = xr.open_dataset(f'{cfg.path_wrfout}/{cfg.wrf_runs[0]}/out/{cfg.file_ref}')
nlats=fileref.south_north.size
nlons=fileref.west_east.size


#filespath = f'{cfg.path_in}/{wrun}/hist2d_IFD_tiles_50/hist2d_IFD_spell_2013-2020'

#I_bins_spell = np.asarray(list(range(1,6,1)) + list(range(6,144,6)) + list(range(144,576,144)))
I_bins_spell = np.asarray(list(range(1,6,1)) + list(range(6,582,6)))
I_bins_intensity=np.array(list(range(1,251,2))+[1000])
# I_bins_intensity=np.array(list(range(1,30,1))+list(range(30,100,2))+list(range(100,255,5))+[1000])
#I_bins_intensity = np.arange(0,605,5)

#####################################################################
#####################################################################

def main():

    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

    xytiles=list(product(latsteps, lonsteps))

    filespath = f'{cfg.path_in}/{wrun}/hist2d_IFD_tiles_50/hist2d_IFD_spell_2013-2020'
    print(f'Ej: {filespath}_000y-000x.nc')



    Parallel(n_jobs=10)(delayed(calc_IFD_hist2d)(filespath,xytile[0],xytile[1],I_bins_spell,I_bins_intensity) for xytile in xytiles)

    #calc_IFD_hist2d(filespath,'010','010',I_bins_spell,I_bins_intensity)

#####################################################################
#####################################################################


def calc_IFD_hist2d(filespath,ny,nx,I_bins_spell,I_bins_intensity):
    print (f'Analyzing tile y: {ny} x: {nx}')
    filesin = f'{filespath}_{ny}y-{nx}x.nc'
    # filespr = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_10MIN_RAIN_2013-01_{ny}y-{nx}x.nc'
    # fpr = xr.open_dataset(filespr)

    fin = xr.open_dataset(filesin)
    hist2d = histogram(fin.spell, fin.intensity, bins=[I_bins_spell,I_bins_intensity],dim=['time'])

    fout = filesin.replace('IFD_spell','IFD_spell_hist2d')


    hist2d.to_netcdf(fout)
    fin.close()



#####################################################################
#####################################################################

if __name__ == "__main__":

    main()

#####################################################################
#####################################################################
