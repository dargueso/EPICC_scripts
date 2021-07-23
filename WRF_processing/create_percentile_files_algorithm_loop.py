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

###########################################################
###########################################################

def main():

    """ Calculating percentiles using a loop"""




    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_20??-??.nc'))
    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']
    qtiles_array = np.zeros((len(qtiles),nlats,nlons),dtype=np.float)

    for ny in range(0,nlats,100):
        for nx in range(0,nlons,100):
            ctime0=checkpoint(0)
            print(ny,nx)
            files_all = xr.open_mfdataset(filesin, concat_dim='time')
            aux = files_all.RAIN.isel(y=slice(ny,ny+100),x=slice(nx,nx+100)).values
            ctime1=checkpoint(ctime0,f'Loaded gridpoint {ny}-{ny+100}, {nx}-{nx+100}')
            qtiles_array[:,ny:ny+100,nx:nx+100]=np.percentile(aux, qtiles*100.,axis=0)
            files_all.close()

    fino=xr.Dataset({'prtile':(['qtiles','y','x'],qtiles_array)},coords={'qtiles':qtiles,'y':range(nlats),'x':range(nlons)})
    fino.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_2011-2020-qtiles_algorithm_loop.nc',mode='w')


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
