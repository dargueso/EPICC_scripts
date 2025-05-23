#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-07-22T10:06:12+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-07-22T10:06:24+02:00
#
# @Project@ EPICC
# Version: 1.0
# Description: Script to interpolate from input data pressure levels to ERA5 pressure levels
# This is required to add the climate change signal from CMIP6 to ERA5
#
# Dependencies:
#
# Files: Annual cycle change files from Calculate_CMIP6_Annual_cycle-CC_change-regrid_ERA5.py and ERA5 sample.
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

ERA5_dir = "/home/dargueso/BDY_DATA/ERA5/ERA5_netcdf"
CMIP6anom_dir = "/home/dargueso/BDY_DATA/CMIP6/"

variables = ["ta", "ua", "va", "zg", "hus"]
era5_ref = xr.open_dataset(f"{ERA5_dir}/era5_daily_pl_20160101.nc")
era5_plev = era5_ref.plev.values


if not os.path.exists(f"{CMIP6anom_dir}/interp_plevs/"):
    os.makedirs(f"{CMIP6anom_dir}/interp_plevs/")


def main():
    ctime_i = checkpoint(0)
    for vn, varname in enumerate(variables):
        ctime_00 = checkpoint(0)

        if not os.path.exists(
            f"{CMIP6anom_dir}/interp_plevs/{varname}_CC_signal_ssp585_2070-2099_1985-2014_pinterp.nc"
        ):
            fin = xr.open_dataset(
                f"{CMIP6anom_dir}/{varname}_CC_signal_ssp585_2070-2099_1985-2014.nc"
            )
            fin.reindex(plev=fin.plev[::-1])
            fin_pinterp = fin.interp(
                plev=era5_plev, kwargs={"fill_value": "extrapolate"}
            )
            fin_pinterp.to_netcdf(
                f"{CMIP6anom_dir}/interp_plevs/{varname}_CC_signal_ssp585_2070-2099_1985-2014_pinterp.nc",
                unlimited_dims="time",
            )
        ctime1 = checkpoint(ctime_00, f"{varname} file interpolated")

    ctime_e = checkpoint(ctime_i, f"Done CC vertical interpolation to ERA5 plevs")


###########################################################
###########################################################


def checkpoint(ctime, msg="task"):
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
    if ctime == 0:
        ctime = time.time()
        dtime = 0
    else:
        dtime = time.time() - ctime
        ctime = time.time()
        print(f"{msg}")
        print(f"======> DONE in {dtime:0.2f} seconds", "\n")
    return ctime


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    main()

###############################################################################
