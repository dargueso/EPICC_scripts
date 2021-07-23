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
qtiles = np.asarray(cfg.qtiles)
mode = 'wet'
###########################################################
###########################################################

def main():

    """ Calculating percentiles using a loop"""

    files_ref = xr.open_dataset(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_2011-01.nc')
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']

    files_ref.close()

    ctime00=checkpoint(0)
    datasets = []
    for nn in range(0,int(np.ceil(nlons/10))):
        ctime0=checkpoint(0)
        print(nn)
        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/split_files/{cfg.patt_in}_10MIN_RAIN_20??-??_{nn:03d}.nc'))
        ds = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")
        if mode == 'wet':
            ptiles = ds.RAIN.where(ds.RAIN>0.1).load().quantile(qtiles,dim=['time'])
        else:
            ptiles = ds.RAIN.load().quantile(qtiles,dim=['time'])
        ds.close()
        datasets.append(ptiles)
        ctime1=checkpoint(ctime0,f'Split file {nn}/{int(np.ceil(nlons/10))}')

    ctot=checkpoint(ctime00,f'All files computed')
    combined = xr.concat(datasets, dim='x')
    combined.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_2011-2020-qtiles_splitmethod_{mode}.nc',mode='w')
    csaved=checkpoint(ctime00,f'All files computed and saved')

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
