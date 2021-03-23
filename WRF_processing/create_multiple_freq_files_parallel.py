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



import os
import sys
import datetime as dt
import glob as glob
import itertools
import subprocess as subprocess
from joblib import Parallel, delayed
import xarray as xr
from dateutil.relativedelta import relativedelta
import numpy as np
import wrf_utils as wrfu


###########################################################
###########################################################




WRF_runs=['EPICC_2km_ERA5_HVC_GWD']#,'Oned_4km_ERA5_CMIP5anom_HVC_NC']#,'Oned_16km_ERA5_HVC','Oned_8km_ERA5_HVC','Oned_4km_ERA5_HVC']#,'Oned_4km_ERA5_HVC']#,'Allres_8km_ERA5_HVC','Allres_16km_ERA5_HVC','Allres_32km_ERA5_HVC','Allres_32km_ERA5_HVC_SN']
#varnames=['UA','VA','TC','Z','SPECHUM','ET','TAS','PR','PRNC','PSL','HUSS','SST','OLR','CLOUDFRAC','WDIR10','WSPD10']
varnames_hfreq=['PRNC']
varnames_lfreq=[]
varnames = varnames_hfreq + varnames_lfreq

frequencies=['10MIN','01H','DAY','MON','DCYCLE']


path_in = "/vg5/dargueso-NO-BKUP/postprocessed/EPICC"
path_out = "/vg5/dargueso-NO-BKUP/postprocessed/unified/EPICC"
periods=[2020]#,2014,2015]
smonth = 7
emonth = 8

patt_inst="UIB"




for wrun in WRF_runs:
    for syear in periods:
        eyear = syear+1

        fullpathin = "%s/%s/%s/" %(path_in,wrun,syear)
        fullpathout = "%s/%s/%s/" %(path_out,wrun,syear)

        if not os.path.exists(fullpathout):
            os.makedirs(fullpathout)



        for freq in frequencies:
            print(syear,eyear,wrun,freq)


            if freq == '10MIN':
                patt="%s_%s"%(patt_inst,'10MIN')
                Parallel(n_jobs=10)(delayed(wrfu.create_10min_files)(fullpathin,fullpathout,syear,eyear,smonth,emonth,patt_inst,varn) for varn in varnames_hfreq)

            if freq == '01H':
                patt="%s_%s"%(patt_inst,'10MIN')
                Parallel(n_jobs=10)(delayed(wrfu.create_hourly_files_cdo)(fullpathin,fullpathout,syear,eyear,smonth,emonth,patt_inst,varn) for varn in varnames_hfreq)

                patt="%s_%s"%(patt_inst,'01H')
                Parallel(n_jobs=10)(delayed(wrfu.create_hourly_files)(fullpathin,fullpathout,syear,eyear,smonth,emonth,patt_inst,varn) for varn in varnames_lfreq)

            if freq == 'DAY':
                patt="%s_%s"%(patt_inst,'01H')
                Parallel(n_jobs=10)(delayed(wrfu.create_daily_files)(fullpathout,syear,eyear,smonth,emonth,patt,varn) for varn in varnames)

            if freq == 'MON':
                patt="%s_%s"%(patt_inst,'DAY')
                Parallel(n_jobs=10)(delayed(wrfu.create_monthly_files)(fullpathout,syear,eyear,smonth,emonth,patt,varn) for varn in varnames)

            if freq == 'DCYCLE':
                patt="%s_%s"%(patt_inst,'01H')
                Parallel(n_jobs=10)(delayed(wrfu.create_diurnalcycle_files_cdo)(fullpathout,syear,eyear,smonth,emonth,patt,varn) for varn in varnames)
