#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-10-31T17:43:27+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-10-31T17:46:20+01:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies: Files must be previously resampled using create_time_resample_files.py
#
# Files:
#
#####################################################################
"""

import pandas as pd
import xarray as xr
import epicc_config as cfg
from glob import glob
import numpy as np
from itertools import product
from itertools import groupby
from joblib import Parallel, delayed
from xhistogram.xarray import histogram

#####################################################################
#####################################################################


wrun = 'EPICC_2km_ERA5_HVC_GWD'

fileref = xr.open_dataset(f'{cfg.path_wrfout}/{cfg.wrf_runs[0]}/out/{cfg.file_ref}')
nlats=fileref.south_north.size
nlons=fileref.west_east.size
RSmins = [10,20,30,60,120,180,240,360,480,720,1440]

#I_bins_edges=np.array(list(range(1,251,2))+[1000])
#I_bins_edges=np.array([0,0.1]+list(range(1,10))+list(range(10,20,2))+list(range(20,100,5))+list(range(100,200,10))+list(range(200,620,20)))
I_bins_edges = np.asarray([1,5] + list(range(10,505,5)))
I_bins_centers=I_bins_edges[:-1]+np.diff(I_bins_edges)/2
tile_size = 50
wet_th = cfg.wet_value
#####################################################################
#####################################################################

def main():


    """ Calculating IFD by tiles """

    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

    xytiles=list(product(latsteps, lonsteps))
    filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_10MIN_RAIN_20??-??'
    print(f'Ej: {filespath}_000y-000x.nc')
    Parallel(n_jobs=20)(delayed(calc_IFD_resample)(filespath,xytile[0],xytile[1],RSmins,I_bins_edges,I_bins_centers) for xytile in xytiles)

#####################################################################
#####################################################################

def calc_IFD_resample(filespath,ny,nx,RSmins,I_bins_edges,I_bins_centers):
    print (f'Analyzing tile y: {ny} x: {nx}')
    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested",data_vars="minimal").sel(time=slice(str(cfg.syear),str(cfg.eyear))).load()
    ylen = fin.sizes['y']
    xlen = fin.sizes['x']
    lons = fin.lon.squeeze()
    lats = fin.lat.squeeze()
    hist_array = np.zeros((len(RSmins),len(I_bins_centers),ylen,xlen),dtype=np.int32)

    for nm,m in enumerate(RSmins):
        print (m)

        if m == 10:
            fin_resample = fin.squeeze()

        else:
            fin_resample = fin.resample(time=f"{m}T").sum('time').squeeze()

        fin_resample_wet = fin_resample.where(fin_resample>=wet_th)
        hist_array[nm,:,:,:] = histogram(fin_resample.RAIN,bins=I_bins_edges,dim=['time']).transpose("RAIN_bin", "y", "x").values


    hist_o = xr.Dataset({'FRQ':(['RSmins','Ibins','y','x'],hist_array),'lat':(['y','x'],lats.squeeze()),'lon':(['y','x'],lons.squeeze())},coords={'Ibins':I_bins_centers,'RSmins':RSmins})

    fout = f'{cfg.path_in}/{wrun}/hist2d_resample_time_IFD_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_{wet_th}mm.nc'
    hist_o.to_netcdf(fout)
    fin.close()


#####################################################################
#####################################################################

if __name__ == "__main__":

    main()

#####################################################################
#####################################################################
