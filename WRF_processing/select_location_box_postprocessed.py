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

import pdb
import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
import time
import subprocess as subprocess
from joblib import Parallel, delayed


wrun = cfg.wrf_runs[0]
#y_idx=258; x_idx=559 #1.01 km away from Palma University station 
#y_idx=250; x_idx=423 # 0.76 km away from Turis
y_idx=384; x_idx=569 #  Spanish Med Pyrenees
freq = '10MIN'
###########################################################
###########################################################

def main():

    """  Select box around location """

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??.nc'))
    files_ref = xr.open_dataset(filesin[0])
 

    Parallel(n_jobs=20)(delayed(select_files)(fin,y_idx,x_idx) for fin in filesin)



#####################################################################
#####################################################################

def select_files(fin,y_idx,x_idx,buffer=5):
    """Select box around location using xarray"""

    print(fin)
    finxr = xr.open_dataset(fin).load()
    fout = fin.replace(".nc",f'_{y_idx:03d}y-{x_idx:03d}x.nc')

    finxr = finxr.isel(y=slice(int(y_idx)-buffer,int(y_idx)+buffer+1),x=slice(int(x_idx)-buffer,int(x_idx)+buffer+1))
    finxr.to_netcdf(fout)

    finxr.close()
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
