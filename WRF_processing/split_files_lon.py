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
import subprocess as subprocess
from joblib import Parallel, delayed

wrun = cfg.wrf_runs[0]

###########################################################
###########################################################

def main():

    """  Split files into tiles """

    # Check initial time
    ctime0=checkpoint(0)

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_20??-??.nc'))

    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']

    files_ref.close()

    ctime00=checkpoint(0)


    for nn in range(0,int(np.ceil(nlons/10))):
        ctime0=checkpoint(0)
        Parallel(n_jobs=10)(delayed(split_files)(fin,nn,nlons) for fin in filesin)
        ctime1=checkpoint(ctime0,f'Split file {nn}/{int(np.ceil(nlons/10))}')




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

def split_files(fin,nn,nlons):

    """Split files based on longitude using ncks"""
    print(fin,nn)
    fout = fin.replace(".nc",f'_{nn:03d}.nc')

    if nn*10+10>nlons:

        subprocess.call(f"ncks -d x,{nn*10},{nlons-1} {fin} {fout}",shell=True)

    else:

        subprocess.call(f"ncks -d x,{nn*10},{nn*10+10} {fin} {fout}",shell=True)

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
