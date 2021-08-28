#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-06-07T16:53:49+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-06-07T16:53:56+02:00
#
# @Project@ EPICC
# Version: 1.0 (Beta)
# Description:
#
# Dependencies: Intersection of available models created with Get_CMIP6_Monthly_PGW_NCI.py
#
# Files: Monthly files of CMIP6 models, and list of available models SearchLocations_intersection_cmip6_mon.txt
#
#####################################################################
"""



import numpy as np
import netCDF4 as nc
import subprocess as subprocess
from glob import glob
import pandas as pd
import xarray as xr
import os



variables=['ua','va','zg']#,'uas','vas','tas','ts','hurs','ps','psl']
experiments = ['historical','ssp585']

#model_list = open (f'SearchLocations_intersection_cmip6_mon_Amon.txt',"r")
model_list = sorted(glob("*r1i1p1f?"))
calc_ACycle = True
calc_CC  = True

if not os.path.exists("./AnnualCycle"):
    os.makedirs("./AnnualCycle")

if not os.path.exists("./AnnualCycle_change"):
    os.makedirs("./AnnualCycle_change")

def main():

    for mod_mem in model_list:
        for vn,varname in enumerate(variables):

            if calc_ACycle == True:
                calculate_annual_cycle(mod_mem,varname)
            if calc_CC == True:
                ctime0=checkpoint(0)
                calculate_CC_signal(mod_mem,varname)

            #Bash command for regridding
            #for file in $(ls *_AnnualCycle.nc); do cdo -remapcon,era5_grid ${file} regrid_era5/${file};done
            # if regrid_era5 == True:
            #
            #     regrid_file_to_era5(mod_mem,varname)



###########################################################
###########################################################

def checkpoint(ctime,msg='task'):
  import time

  """ Computes the spent time from the last checkpoint

  Input: a given time
  Output: present time
  Print: the difference between the given time and present time
  Author: Alejandro Di Luca
  Modified: Daniel Argüeso
  Created: 07/08/2013
  Last Modification: 06/07/2021

  """
  if ctime==0:
    ctime=time.time()
    dtime=0
  else:
    dtime=time.time()-ctime
    ctime=time.time()
    print(f'{msg}')
    print(f'======> DONE in {dtime:0.2f} seconds',"\n")
  return ctime

###########################################################
###########################################################
def calculate_annual_cycle(mod_mem,varname):
    """ For a given model, member and variable,
    Calculate annual cycle """

    model =  mod_mem.split('_')[0]
    member =  mod_mem.split('_')[1]

    print(f'{varname} {model} {member}')

    ctime_i=checkpoint(0)

    filenames_all = sorted(glob(f'./{model}_{member}/{varname}_*'))
    finall= xr.open_mfdataset(filenames_all,concat_dim="time", combine="nested")

    ctime1=checkpoint(ctime_i, 'Read files: done')


    if not os.path.exists(f'AnnualCycle/{varname}_{model}_{member}_historical_1990-2014_AnnualCycle.nc'):
        ctime00 = checkpoint(0)
        fin_p = finall.sel(time=slice('1990','2014'))

        if varname == 'hus':
            fin_p = fin_p.where((fin_p.hus>=0) & (fin_p.hus<100))
        elif varname == 'ta':
            fin_p = fin_p.where((fin_p.ta>=0) & (fin_p.ta<400))
        elif varname == 'ua':
            fin_p = fin_p.where((fin_p.ua>-500) & (fin_p.ua<500))
        elif varname == 'va':
            fin_p = fin_p.where((fin_p.va>-500) & (fin_p.va<500))
        elif varname == 'zg':
            fin_p = fin_p.where((fin_p.zg>-1000) & (fin_p.zg<60000))


        fin_p_mm = fin_p.groupby('time.month').mean('time')
        fin_p_mm.to_netcdf(f'AnnualCycle/{varname}_{model}_{member}_historical_1990-2014_AnnualCycle.nc')
        ctime2=checkpoint(ctime00, 'historical file done')
    else:
        print(f' Historical {varname} {model} {member} Already processed')

    if not os.path.exists(f'AnnualCycle/{varname}_{model}_{member}_ssp585_2076-2100_AnnualCycle.nc'):
        ctime00 = checkpoint(0)
        fin_f = finall.sel(time=slice('2076','2100'))

        if varname == 'hus':
            fin_f = fin_f.where((fin_f.hus>=0) & (fin_f.hus<100))
        elif varname == 'ta':
            fin_f = fin_f.where((fin_f.ta>=0) & (fin_f.ta<400))
        elif varname == 'ua':
            fin_f = fin_f.where((fin_f.ua>-500) & (fin_f.ua<500))
        elif varname == 'va':
            fin_f = fin_f.where((fin_f.va>-500) & (fin_f.va<500))
        elif varname == 'zg':
            fin_f = fin_f.where((fin_f.zg>-1000) & (fin_f.zg<60000))



        fin_f_mm = fin_f.groupby('time.month').mean('time')
        fin_f_mm.to_netcdf(f'AnnualCycle/{varname}_{model}_{member}_ssp585_2076-2100_AnnualCycle.nc')
        ctime2=checkpoint(ctime00, 'ssp585 file done')
    else:
        print(f' ssp585 {varname} {model} {member} Already processed')

    ctimef = checkpoint(ctime_i,f'Done Acycle {varname} {model} {member}')
    finall.close()


###########################################################
###########################################################
def calculate_CC_signal(mod_mem,varname):
    """ From present and future annual cycle
    calculate CC signal for every month"""

    ctime00=checkpoint(0)

    model =  mod_mem.split('_')[0]
    member =  mod_mem.split('_')[1]

    if not os.path.exists(f'AnnualCycle_change/{varname}_{model}_{member}_CC_2076-2100_1990-2014_AnnualCycle.nc'):

        fin_p = xr.open_dataset(f'AnnualCycle/{varname}_{model}_{member}_historical_1990-2014_AnnualCycle.nc')
        fin_f = xr.open_dataset(f'AnnualCycle/{varname}_{model}_{member}_ssp585_2076-2100_AnnualCycle.nc')

        fin_d = fin_f - fin_p



        if 'plev_bnds' in fin_d.keys():
            fin_d = fin_d.drop(('plev_bnds'))
        if 'lon_bnds' in fin_d.keys():
            fin_d = fin_d.drop(('lon_bnds'))
        if 'lat_bnds' in fin_d.keys():
            fin_d = fin_d.drop(('lat_bnds'))

        datelist = pd.date_range(f'1990-01-01',periods=12,freq='MS')

        foutclean  = fin_d.rename({'month': 'time'})
        foutclean  = foutclean.assign_coords({"time": datelist})
        foutclean.to_netcdf(f'AnnualCycle_change/{varname}_{model}_{member}_CC_2076-2100_1990-2014_AnnualCycle.nc',unlimited_dims='time')
        ctime2=checkpoint(ctime00, 'historical file done')

        fin_p.close()
        fin_f.close()
    else:
        print(f' CC file {varname} {model} {member} Already processed')


    ctime1=checkpoint(ctime00,f'Done CC {varname} {model} {member}')


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###############################################################################
