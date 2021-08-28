#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-08-25T18:41:42+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-08-25T18:41:44+02:00
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
import numpy as np
from glob import glob

CMIP6_AnnualCycle_CC="/vg6/dargueso-NO-BKUP/BDY_DATA/CMIP6_ANOMALY/CLIMATO_ATM/AnnualCycle_change/regrid_era5/"

vars = ['hus','ta','ua','va','zg']#,'hurs','tas','uas','vas','ps','psl','ts']

plvs=np.asarray([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000,
    20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100 ])

correct_plevs = True

if correct_plevs:
    for var in vars:
        filesin = sorted(glob(f'{CMIP6_AnnualCycle_CC}/{var}*'))
        for filepath in filesin:
            filename = filepath.split('/')[-1]
            print(filename)
            fin = xr.open_dataset(filepath)
            fin.coords['plev']=plvs
            fin.to_netcdf(f'{CMIP6_AnnualCycle_CC}/correct_plevs/{filename}')

for var in vars:
    print(var)
    filesin = sorted(glob(f'{CMIP6_AnnualCycle_CC}/correct_plevs/{var}*'))
    fin = xr.open_mfdataset(filesin,concat_dim='model',combine='nested')
    fin_ensmean = fin.mean(dim='model').squeeze()
    fin_ensmean.to_netcdf(f'{var}_CC_signal_ssp585_2076-2100_1990-2014.nc')
