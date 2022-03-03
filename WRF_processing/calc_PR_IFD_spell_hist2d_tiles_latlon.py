#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-03-01T00:48:54+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-03-01T00:49:03+01:00
#
# @Project@ EPICC
# Version: 0.1 (Beta)
# Description: This script calculates spells of rain from postprocessed
# files that have previously been tiled. The purpose is to compute 2D histograms
# once the spells are calculated and joined.
#
# Dependencies:
#
# Files:
#
#####################################################################
"""
from glob import glob
import numpy as np
import xarray as xr
from itertools import product
from itertools import groupby
from joblib import Parallel, delayed
from xhistogram.xarray import histogram

import epicc_config as cfg


###########################################################
###########################################################

wrun = 'EPICC_2km_ERA5_HVC_GWD'
tile_size = 50
fileref = xr.open_dataset(f'{cfg.path_wrfout}/{cfg.wrf_runs[0]}/out/{cfg.file_ref}')
nlats=fileref.south_north.size
nlons=fileref.west_east.size
wet_th = cfg.wet_value
save_spell_file = False
###########################################################
###########################################################

I_bins_spell = np.asarray(list(range(1,6,1)) + list(range(6,582,6)))
I_bins_intensity = np.arange(0,1005,5)
I_bins_centers=I_bins_intensity[:-1]+np.diff(I_bins_intensity)/2

###########################################################
###########################################################
def main():
    """ Calculating spells by tiles """


    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

    xytiles=list(product(latsteps, lonsteps))
    filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_10MIN_RAIN_20??-??'
    print(f'Ej: {filespath}_000y-000x.nc')

    Parallel(n_jobs=10)(delayed(calc_IFD_spell)(filespath,xytile[0],xytile[1]) for xytile in xytiles)

###########################################################
###########################################################

def calc_IFD_spell(filespath,ny,nx):
    print (f'Analyzing tile y: {ny} x: {nx}')
    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested",data_vars="minimal").sel(time=slice(str(cfg.syear),str(cfg.eyear))).load()
    ylen = fin.sizes['y']
    xlen = fin.sizes['x']
    lons = fin.lon.squeeze()
    lats = fin.lat.squeeze()

    spell=np.zeros((len(fin.time),len(fin.y),len(fin.x)),dtype=np.int32)
    intensity = np.zeros((len(fin.time),len(fin.y),len(fin.x)),dtype=float)

    for iy in range(ylen):
        for ix in range(xlen):
            #print(f'y: {iy}; x: {ix}')
            srain = fin.RAIN.isel(y=iy,x=ix).where(fin.isel(y=iy,x=ix).RAIN>wet_th,0)
            srain = srain.to_dataframe()
            srain = srain.rename(columns = {'RAIN':'pr'})
            srain['pr'] = srain['pr'].where(srain['pr']>=wet_th,0)
            srain['event_start'] = (srain['pr'].astype(bool).shift() != srain['pr'].astype(bool)).where(srain['pr']>=wet_th, False)
            srain['event_end'] = (srain['pr'].astype(bool).shift(-1) != srain['pr'].astype(bool)).where(srain['pr']>=wet_th, False)
            srain['event'] = srain['event_start'].cumsum().where(srain['pr']>=wet_th)

            wet_event_intensity = srain.groupby('event')['pr'].sum()
            wet_event_duration = srain.groupby('event')['pr'].count()
            srain['event_pr'] = srain['event'].map(wet_event_intensity).where(srain['event_end'])
            srain['event_dur'] =  srain['event'].map(wet_event_duration).where(srain['event_end'])

            spell[:,iy,ix] = srain['event_dur']
            intensity[:,iy,ix] = srain['event_pr']

    fino=xr.Dataset({'spell':(['time','y','x'],spell),'intensity':(['time','y','x'],intensity),'lat':(['y','x'],fin.lat.squeeze()),'lon':(['y','x'],fin.lon.squeeze())},coords={'time':fin.time.values})


    if save_spell_file:
        fout = f'{cfg.path_in}/{wrun}/spell_IFD_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_{wet_th}mm.nc'
        fino.to_netcdf(fout,mode='w',encoding={'spell':{'zlib': True,'complevel': 5},'intensity':{'zlib': True,'complevel': 5}})

    fin.close()

    hist2d = histogram(fino.spell, fino.intensity, bins=[I_bins_spell,I_bins_intensity],dim=['time'])
    fout2d = f'{cfg.path_in}/{wrun}/hist2d_spell_IFD_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_{wet_th}mm.nc'
    hist2d.to_netcdf(fout2d)


###########################################################
###########################################################
if __name__ == "__main__":

    main()

###########################################################
###########################################################
