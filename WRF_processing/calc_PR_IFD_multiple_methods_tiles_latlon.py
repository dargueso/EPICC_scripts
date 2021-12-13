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


wrun = 'EPICC_2km_ERA5_CMIP6anom_HVC_GWD'

fileref = xr.open_dataset(f'{cfg.path_wrfout}/{cfg.wrf_runs[0]}/out/{cfg.file_ref}')
nlats=fileref.south_north.size
nlons=fileref.west_east.size
RSmins = [10,20,30,60,120,180,360,480,720,1440]

I_bins_edges=np.array(list(range(1,251,2))+[1000])
#I_bins_edges=np.array([0,0.1]+list(range(1,10))+list(range(10,20,2))+list(range(20,100,5))+list(range(100,200,10))+list(range(200,620,20)))
I_bins_centers=I_bins_edges[:-1]+np.diff(I_bins_edges)/2
I_bins_centers[1]=1
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

    Parallel(n_jobs=10)(delayed(calc_IFD_resample)(filespath,xytile[0],xytile[1],RSmins,I_bins_edges,I_bins_centers) for xytile in xytiles)
    #Parallel(n_jobs=20)(delayed(calc_IFD_spell)(filespath,xytile[0],xytile[1]) for xytile in xytiles)


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


        hist_array[nm,:,:,:] = histogram(fin_resample.RAIN,bins=I_bins_edges,dim=['time']).transpose("RAIN_bin", "y", "x").values


    hist_o = xr.Dataset({'FRQ':(['RSmins','Ibins','y','x'],hist_array),'lat':(['y','x'],lats.squeeze()),'lon':(['y','x'],lons.squeeze())},coords={'Ibins':I_bins_centers,'RSmins':RSmins})

    fout = f'{cfg.path_in}/{wrun}/hist2d_IFD_resample_time_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x.nc'
    hist_o.to_netcdf(fout)
    fin.close()

#####################################################################
#####################################################################

def calc_IFD_spell(filespath,ny,nx):
    print (f'Analyzing tile y: {ny} x: {nx}')
    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested").sel(time=slice(str(cfg.syear),str(cfg.eyear))).load()
    ylen = fin.sizes['y']
    xlen = fin.sizes['x']
    lons = fin.lon.squeeze()
    lats = fin.lat.squeeze()

    series=(fin.RAIN>wet_th).astype(dtype=np.int32).values
    spell=np.zeros((len(fin.time)-1,len(fin.y),len(fin.x)),dtype=np.int32)
    intensity = np.zeros((len(fin.time)-1,len(fin.y),len(fin.x)),dtype=float)
    for iy in range(ylen):
        for ix in range(xlen):
            if ((series[0,iy,ix]==1) | (series[-1,iy,ix]==1)):
                series[0,iy,ix]=0
                series[-1,iy,ix]=0
            srun=-np.diff(series[:,iy,ix])
            L=(series[:,iy,ix]).tolist()
            groups_rain = []
            for k,g in groupby(L):
                if k==1:
                    b=list(g)
                    groups_rain.append(sum(b))
            if np.any(srun==1):
                spell[srun==1,iy,ix]=np.asarray(groups_rain)

            for t in range(len(fin.time)-1):
                if spell[t,iy,ix]!=0:
                    intensity[t,iy,ix]=fin.isel(time=slice(t,t+spell[t,iy,ix]),y=iy,x=ix).RAIN.sum().item()

    spell = np.append(np.zeros((1,len(fin.y),len(fin.x)),dtype=np.int32), spell, axis=0)
    intensity = np.append(np.zeros((1,len(fin.y),len(fin.x)),dtype=float), intensity, axis=0)




    fino=xr.Dataset({'spell':(['time','y','x'],spell),'intensity':(['time','y','x'],intensity),'lat':(['y','x'],fin.lat.isel(time=0).squeeze()),'lon':(['y','x'],fin.lon.isel(time=0).squeeze())},\
        coords={'time':fin.time.values})

    fout = f'{cfg.path_in}/{wrun}/hist2d_IFD_spell_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x.nc'
    fino.to_netcdf(fout,mode='w',encoding={'spell':{'zlib': True,'complevel': 5},'intensity':{'zlib': True,'complevel': 5}})
    fin.close()


#####################################################################
#####################################################################

if __name__ == "__main__":

    main()

#####################################################################
#####################################################################
