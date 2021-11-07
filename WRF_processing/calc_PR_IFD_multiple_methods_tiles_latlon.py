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
from joblib import Parallel, delayed

#####################################################################
#####################################################################


wrun = cfg.wrf_runs[0]

fileref = xr.open_dataset(f'{cfg.path_wrfout}/{cfg.wrf_runs[0]}/out/{cfg.file_ref}')
nlats=fileref.south_north.size
nlons=fileref.west_east.size
RSmins = [10,20,30]
RShours = [1,2,3,6,8,12]
RSdays  = [24,48,72]

I_bins_edges=np.array([0,0.1]+list(range(1,10))+list(range(10,20,2))+list(range(20,100,5))+list(range(100,200,10))+list(range(200,620,20)))
I_bins_centers=I_bins_edges[:-1]+np.diff(I_bins_edges)/2
I_bins_centers[1]=0.5
tile_size = 50
#####################################################################
#####################################################################

def main():


    """ Calculating IFD by tiles """

    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

    xytiles=list(product(latsteps, lonsteps))
    filespath = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_20??-??'
    print(f'Ej: {filespath}_000y-000x.nc')

    Parallel(n_jobs=20)(delayed(calc_IFD)(filespath,xytile[0],xytile[1],RSmins,RShours,RSdays,I_bins_edges,I_bins_centers) for xytile in xytiles)


#####################################################################
#####################################################################

def calc_IFD(filespath,ny,nx,RSmins,RShours,RSdays,I_bins_edges,I_bins_centers):
    print (f'Analyzing tile y: {ny} x: {nx}')
    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested").load()
    ylen = fin.sizes['y']
    xlen = fin.sizes['x']
    lons = fin.lon.squeeze()
    lats = fin.lat.squeeze()
    hist_array = np.zeros((len(I_bins_centers),len(RSmins)+len(RShours)+len(RSdays),ylen,xlen),dtype=np.int32)

    for nm,m in enumerate(RSmins):
        #print (m)
        if m == 10: 
            fin_resample = fin.squeeze()
        else: 
            fin_resample = fin.resample(time=f"{m}T").sum('time').squeeze()

        for nint in range(len(I_bins_centers)):
            hist_array[nint,nm,:,:] = fin_resample.where((fin_resample>=I_bins_edges[nint]*m/60.) & (fin_resample<I_bins_edges[nint+1]*m/60.)).RAIN.count(dim='time').values

    for nh,hr in enumerate(RShours):
        #print(hr)
        fin_resample = fin.resample(time=f"{hr}H").sum('time').squeeze()
        for nint in range(len(I_bins_centers)):
            hist_array[nint,nh+len(RSmins),:,:]= fin_resample.where((fin_resample>=I_bins_edges[nint]*hr) & (fin_resample<I_bins_edges[nint+1]*hr)).RAIN.count(dim='time').values

    for nday,dhour in enumerate(RSdays):
        #print(dhour)
        nloc = nday+len(RSmins)+len(RShours)
        fin_resample = fin.resample(time=f"{hr}H").sum('time').squeeze()
        for nint in range(len(I_bins_centers)):
            hist_array[nint,nloc,:,:] = fin_resample.where((fin_resample>=I_bins_edges[nint]*dhour) & (fin_resample<I_bins_edges[nint+1]*dhour)).RAIN.count(dim='time').values

    hist_o = xr.Dataset({'FRQ':(['Ibins','RStime','y','x'],hist_array),'lat':(['y','x'],lats.isel(time=0).squeeze()),'lon':(['y','x'],lons.isel(time=0).squeeze())},coords={'Ibins':I_bins_centers,'RStime':RSmins+RShours+RSdays})
    
    fout = f'{cfg.path_in}/{wrun}/hist2d_IFD_resample_time_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x.nc'
    hist_o.to_netcdf(fout)
    fin.close()

#####################################################################
#####################################################################

if __name__ == "__main__":

    main()

#####################################################################
#####################################################################
