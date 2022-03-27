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
import time

import epicc_config as cfg
from tqdm import tqdm


###########################################################
###########################################################

wruns = ['EPICC_2km_ERA5_CMIP6anom_HVC_GWD']
tile_size = 50
fileref = xr.open_dataset(f'{cfg.path_wrfout}/{cfg.wrf_runs[0]}/out/{cfg.file_ref}')
nlats=fileref.south_north.size
nlons=fileref.west_east.size
wet_th = cfg.wet_value
###########################################################
###########################################################

I_bins_spell = np.asarray(list(range(1,6,1)) + list(range(6,582,6)))
I_bins_intensity = np.asarray([1,5] + list(range(10,505,5)))
I_bins_centers= np.arange(2.5,500,5)
###########################################################
###########################################################
def main():
    """ Calculating spells by tiles """


    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

    xytiles=list(product(latsteps, lonsteps))
    for wrun in wruns:
        print(f"Analyzing {wrun}")
        filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_10MIN_RAIN_20??-??'
        print(f'Ej: {filespath}_000y-000x.nc')
        Parallel(n_jobs=10)(delayed(calc_IFD_spell)(filespath,xytile[0],xytile[1],wrun) for xytile in xytiles)

###########################################################
###########################################################

def calc_IFD_spell(filespath,ny,nx,wrun):
    print (f'Analyzing tile y: {ny} x: {nx}')
    # Check initial time
    start_time = time.time()
    filesin = sorted(glob(f'{filespath}_{ny}y-{nx}x.nc'))
    fin = xr.open_mfdataset(filesin,concat_dim="time", combine="nested",data_vars="minimal").sel(time=slice(str(cfg.syear),str(cfg.eyear))).load()
    ylen = fin.sizes['y']
    xlen = fin.sizes['x']
    lons = fin.lon.squeeze()
    lats = fin.lat.squeeze()

    hist2d=np.zeros((len(I_bins_spell),len(I_bins_intensity)-1,len(fin.y),len(fin.x)),dtype=np.int32)


    for iy in range(ylen):
        #start_timey = time.time()
        for ix in range(xlen):

            #print(f'y: {iy}; x: {ix}')
            srain = fin.RAIN.isel(y=iy,x=ix).where(fin.isel(y=iy,x=ix).RAIN>=wet_th,0)
            srain = srain.to_dataframe()
            srain = srain.rename(columns = {'RAIN':'pr'})
            srain['event_start'] = (srain['pr'].astype(bool).shift() != srain['pr'].astype(bool))
            srain['event'] = srain['event_start'].cumsum()
            wet_event_duration = srain.groupby('event')['pr'].count()
            srain['event_dur_expand'] =  srain['event'].map(wet_event_duration)

            srain_hist = srain.where(srain['pr']>=wet_th).dropna(subset=['pr']).pr
            hist2d[0,:,iy,ix]=np.histogram(srain.pr,bins=I_bins_intensity)[0]


            for nlspell,len_spell in enumerate(I_bins_spell[1:]):
                #print(len_spell)
                srain['prcum'] = srain.pr.rolling(len_spell).sum()
                srain_sel = srain.where((srain['pr']>=wet_th) & (srain['event_dur_expand']>=len_spell)).dropna(subset=['pr','event_dur_expand'])


                if srain_sel.empty:
                    #print(f"No events equal or longer than {len_spell*10} minutes ({len_spell/6.} hours)")
                    break
                hist2d[nlspell+1,:,iy,ix]=np.histogram(srain_sel.prcum,bins=I_bins_intensity)[0]
        #end_timey = time.time()
        #print(f'======> DONE 1 row in {(end_timey-start_timey):.2f} seconds \n')
    fino=xr.Dataset({'hist2d':(['duration','intensity','y','x'],hist2d),'lat':(['y','x'],fin.lat.squeeze()),'lon':(['y','x'],fin.lon.squeeze())},coords={'duration':I_bins_spell,'intensity':I_bins_centers})
    fout = f'{cfg.path_in}/{wrun}/hist2d_spell_sumevents_IFD_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x.nc'
    fino.to_netcdf(fout)
    end_time = time.time()
    print(f'======> DONE 1 tile in {(end_time-start_time):.2f} seconds \n')



###########################################################
###########################################################
if __name__ == "__main__":

    main()

###########################################################
###########################################################
