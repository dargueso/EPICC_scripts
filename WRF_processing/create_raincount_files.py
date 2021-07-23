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

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob

wrun = cfg.wrf_runs[0]

###########################################################
###########################################################

def main():

    """ Count number of timesteps above a threshold for percentile purposes"""

    # Check initial time
    ctime0=checkpoint(0)

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_20??-??.nc'))
    fin_all = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")

    ctime1=checkpoint(ctime0,"Loaded all data at once")

    fin_all_th = fin_all.where(fin_all.RAIN>0.1).count(dim='time')
    ctime2=checkpoint(ctime1,"Counted events")
    fin_all.close()

    fin_all_th.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_2011-2020-countwet_0.1mm.nc')
    ctime3=checkpoint(ctime2,"Saved file")


    print("Loading each file at a time")

    ctime0=checkpoint(0)
    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_20??-??.nc'))

    for n,file in enumerate(filesin):
        #print(file)
        if n==0:
            fin = xr.open_dataset(file)
            aux = fin.where(fin.RAIN>0.1).count(dim='time').RAIN.values
        else:

            fin = xr.open_dataset(file)
            aux += fin.where(fin.RAIN>0.1).count(dim='time').RAIN.values


        fin.close()
    ctime1=checkpoint(ctime0,"Loaded and counted events")

    fin_all = xr.open_dataset(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_2011-2020-countwet_0.1mm.nc')
    fin_all.update({"RAIN": (("y","x"),aux)})
    fin_all.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_2011-2020-countwet_0.1mm_loop.nc')
    ctime2=checkpoint(ctime1,"Saved file")


###########################################################
###########################################################
def checkpoint(ctime,msg='task'):
  import time

  """ Computes the spent time from the last checkpoint

  Input: a given time
  Output: present time
  Print: the difference between the given time and present time
  Author: Alejandro Di Luca
  Created: 07/08/2013
  Last Modification: 14/08/2013

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
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
