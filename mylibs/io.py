#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-03-17T10:32:00+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-17T10:32:24+01:00
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
from glob import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
import pdb; pdb.set_trace()

import epicc_config as cfg

def load_wrf(wrun,freq,var,levs=[]):

    if levs != []:
        patt_in = cfg.patt_in + "_" + levs

    else:
        patt_in = cfg.patt_in

    filesin = sorted(glob('%s/%s/20??/%s_%s_%s_*.nc' %(cfg.path_in,wrun,patt_in,freq,var)))
    fin = xr.open_mfdataset(filesin,combine='by_coords')

    fin.coords['y']=fin.lat.values[0,:,0]
    fin.coords['x']=fin.lon.values[0,0,:]
    if var == 'PRNC':
        fin.PR.values = fin.PR.values*3600.
        fin.PR.attrs['units']='mm -hr-1'

    return fin
