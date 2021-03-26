#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2018-02-14T15:07:45+11:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2018-02-14T15:07:47+11:00
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



import os as os
import sys
import datetime as dt
from glob import glob
import itertools
import wrf as wrf
import xarray as xr
import netCDF4 as nc
import numpy as np

import compute_vars as cvars
from wrf_utils import wrftime2date,sel_wrfout_files


import wrf_utils as wrfu
import calendar

from joblib import Parallel, delayed
import epicc_config as cfg

###########################################################
###########################################################


WRF_runs=['EPICC_2km_ERA5_HVC_GWD']
varnames=['TC']#'TC','TD','RH','UA','Z','VA','WA']

numvar=len(varnames)
path_out = "/vg5/dargueso-NO-BKUP/postprocessed/plevs/EPICC"
path_in="/vg6/dargueso-NO-BKUP/WRF_OUT/EPICC/"
path_proc="/vg5/dargueso-NO-BKUP/postprocessed/unified/EPICC"

smonth = 8
emonth = 12

periods=[2020]

plevs=cfg.plevs

patt='UIB'
patt_wrf='wrf3hrly'
dom='d01'
patt_inst="UIB_PLEVS"

###########################################################
###########################################################

for wrun in WRF_runs:
    for syear in periods:
        eyear = syear

        fullpathin = path_in + "/" + wrun + "/out"
        fullpathout = path_out + "/" + wrun + "/" + str(syear) 

        if not os.path.exists(fullpathout):
            os.makedirs(fullpathout)

        #Parallel(n_jobs=numvar)(delayed(wrfu.plevs_interp)(path_in,path_out,path_geo,syear,eyear,smonth,emonth,plevs,patt,patt_wrf,dom,wrun,varn) for varn in varnames)
        for varn in varnames:
            d1 = dt.datetime(int(syear),smonth,1)
            d2 = dt.datetime(int(eyear),emonth,calendar.monthrange(int(eyear), emonth)[1])+dt.timedelta(days=1)
            total_hours = (d2-d1).days*24+(d2-d1).seconds//3600.
            total_days = (d2-d1).days
            date_list= [d1 + dt.timedelta(days=x) for x in range(0, total_days)]

            Parallel(n_jobs=10)(delayed(wrfu.plevs_interp_byday)(fullpathin,fullpathout,cfg.geofile_ref,date,plevs,patt,patt_wrf,dom,wrun,varn) for date in date_list)
