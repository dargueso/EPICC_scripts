#!/usr/bin/env python

"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2018-02-14T13:30:48+11:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-16T17:48:56+01:00
#
# @Project@ EPICC_2km_ERA5_HVC_GWD
# Version: 1.0
# Description: Script to generate postprocessed files from EPICC wrfouts and wrfprecs
#
# Dependencies: atmopy
#
# Files: wrfout_* or wrfprec_*from EPICC runs
#
#####################################################################
"""

import netCDF4 as nc
import numpy as np
from glob import glob
import datetime as dt
import calendar
import os
from optparse import OptionParser
import re
from dateutil.relativedelta import relativedelta
import sys

from joblib import Parallel, delayed

import compute_vars as cvars
from wrf_utils import wrftime2date,sel_wrfout_files

import EPICC_post_config as cfg


###########################################################
###########################################################
def checkpoint(ctime):
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
    print('======> DONE in ',float('%.2g' %(dtime)),' seconds',"\n")
  return ctime

###########################################################
###########################################################

def postproc_var_byday(wrun,varn,date):

    institution=cfg.institution
    path_in = cfg.path_in
    path_out = cfg.path_out
    patt    = cfg.patt
    dom = cfg.dom
    fullpathin = path_in + "/" + wrun + "/out"
    fullpathout = path_out + "/" + wrun + "/" + syear
    file_refname = fullpathin+"/"+cfg.file_ref

    ctime_var=checkpoint(0)


    y = date.year
    m = date.month
    d = date.day

    print(y,m,d)

    sdate="%s-%s-%s" %(y,str(m).rjust(2,"0"),str(d).rjust(2,"0"))
    filesin = sorted(glob('%s/%s_%s_%s*' %(fullpathin,patt,dom,sdate)))



    x=[]
    t=[]

    if len(filesin) == 1:
        print(filesin)
        ncfile = nc.Dataset(filesin[0])
        if patt == 'wrf3hrly':
            fwrf2d = nc.Dataset(filesin[0].replace('wrf3hrly','wrfout'))
            for varname in fwrf2d.variables.keys():
                ncfile.variables[varname]=fwrf2d.variables[varname]

        varout,atts = cvars.compute_WRFvar(ncfile,varn)
        otimes =  wrftime2date(filesin[0].split())[:]

    else:

        for n,filename in enumerate(filesin):

            tFragment = wrftime2date(filename.split())[:]
            ncfile = nc.Dataset(filesin_wrf[thour])
            if patt == 'wrf3hrly':
                fwrf2d = nc.Dataset(filesin_wrf[thour].replace('wrf3hrly','wrfout'))
                for varname in fwrf2d.variables.keys():
                    ncfile.variables[varname]=fwrf2d.variables[varname]

            xFragment,atts = cvars.compute_WRFvar(ncfile,varn)

            if len(tFragment)==1:
                if len(xFragment.shape) == 3:
                    xFragment=np.expand_dims(xFragment,axis=0)
                if len(xFragment.shape) == 2:
                    xFragment=np.expand_dims(xFragment,axis=0)

            ncfile.close()
            if patt == 'wrf3hrly': fwrf2d.close()

            x.append(xFragment)
            t.append(tFragment)

        varout = np.concatenate(x, axis=0)
        otimes = np.concatenate(t, axis=0)

    ###########################################################
    ###########################################################

    # ## Creating netcdf files
    fileout = "%s/%s_%s_%s.nc" %(fullpathout,institution,varn,str(sdate))
    ref_file = nc.Dataset(file_refname)
    lat=ref_file.variables['XLAT'][0,:]
    lon=ref_file.variables['XLONG'][0,:]

    varinfo = { 'values': varout,
                'varname': varn,
                'atts':atts,
                'lat': lat,
                'lon': lon,
                'times': otimes}

    cvars.create_netcdf(varinfo,fileout)



    #edate = dt.datetime(y,m,d) + dt.timedelta(days=1)
    print(otimes[-1].strftime("%Y-%m-%d"))
    ctime=checkpoint(ctime_var)

###########################################################
###########################################################

# Check initial time
ctime_i=checkpoint(0)
ctime=checkpoint(0)

varnames = cfg.variables

###########################################################
###########################################################

wrun_all = cfg.wruns
path_in = cfg.path_in
path_out = cfg.path_out
syear_all = cfg.syears
eyear_all = cfg.eyears
smonth = cfg.smonth
emonth = cfg.emonth

for wrun in wrun_all:
    for n,syear in enumerate(syear_all):
        eyear = eyear_all[n]
        fullpathin = path_in + "/" + wrun + "/out"
        fullpathout = path_out + "/" + wrun + "/" + syear



        if not os.path.exists(fullpathout):
            os.makedirs(fullpathout)
        datenow=dt.datetime.now().strftime("%Y-%m-%d_%H:%M")

        for varn in varnames:

            d1 = dt.datetime(int(syear),smonth,1)
            d2 = dt.datetime(int(eyear),emonth,calendar.monthrange(int(eyear), emonth)[1])+dt.timedelta(days=1)
            total_hours = (d2-d1).days*24+(d2-d1).seconds//3600
            total_days = (d2-d1).days
            date_list= [d1 + dt.timedelta(days=x) for x in range(0, total_days)]
            Parallel(n_jobs=10)(delayed(postproc_var_byday)(wrun,varn,date) for date in date_list)

        ctime=checkpoint(ctime_i)

        #sys.stdout.close()
