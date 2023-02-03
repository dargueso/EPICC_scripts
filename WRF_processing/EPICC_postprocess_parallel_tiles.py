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


def main():

    """ POSTPROCESS REQUESTES VARIABLES FROM WRF OUTPUTS"""

    # Check initial time
    ctime_i=checkpoint(0)
    ctime=checkpoint(0)

    for wrun in cfg.wruns:

        init_date = dt.datetime(cfg.syear,cfg.smonth,1)
        year = init_date.year
        month= init_date.month
        day = init_date.day

        while (year < cfg.eyear or (year == cfg.eyear and month < cfg.emonth)):

            end_date   = dt.datetime(year,month,day) + relativedelta(years=1)
            if end_date>dt.datetime(cfg.eyear,cfg.emonth,1):
                end_date = dt.datetime(cfg.eyear,cfg.emonth,calendar.monthrange(int(cfg.eyear), cfg.emonth)[1])+dt.timedelta(days=1)

            fullpathout = cfg.path_proc + "/" + wrun
            if not os.path.exists(fullpathout):
                os.makedirs(fullpathout)

            for varn in cfg.variables:
                d1 = dt.datetime(year,month,22)
                d2 = dt.datetime(year,month,23)
                #d2 = dt.datetime(end_date.year,end_date.month,end_date.day)
                total_days = (d2-d1).days
                date_list= [d1 + dt.timedelta(days=x) for x in range(0, total_days)]
                Parallel(n_jobs=1)(delayed(postproc_var_byday)(wrun,varn,date) for date in date_list)

            ctime=checkpoint(ctime_i)

            year  = end_date.year
            month = end_date.month
            day   = end_date.day

            #sys.stdout.close()


###########################################################
###########################################################

def postproc_var_byday(wrun,varn,date):


    patt    = cfg.patt
    dom = cfg.dom
    fullpathin = cfg.path_wrfo + "/" + wrun
    fullpathout = cfg.path_proc + "/" + wrun

    ctime_var=checkpoint(0)


    y = date.year
    m = date.month
    d = date.day

    sdate="%s-%s-%s" %(y,str(m).rjust(2,"0"),str(d).rjust(2,"0"))

    for ntile in range(cfg.nproc_x*cfg.nproc_y):

        file_refname = f"{fullpathin}/{cfg.file_ref}_{ntile:04d}"
        filesin = sorted(glob(f'{fullpathin}/{patt}_{dom}_{sdate}*_{ntile:04d}'))


        x=[]
        t=[]

        if len(filesin) == 1:
            print(filesin)
            ncfile = nc.Dataset(filesin[0])
            if patt == 'wrf3hrly':
                fwrf2d = nc.Dataset(filesin[0].replace('wrf3hrly','wrfout'))
                fwrfgeo = nc.Dataset(f'{cfg.path_geo}/{cfg.file_geo}')
                for varname in fwrf2d.variables.keys():
                    ncfile.variables[varname]=fwrf2d.variables[varname]
                for varname in fwrfgeo.variables.keys():
                    ncfile.variables[varname]=fwrfgeo.variables[varname]

            varout,atts = cvars.compute_WRFvar(ncfile,varn)
            otimes =  wrftime2date(filesin[0].split())[:]

        else:

            for n,filename in enumerate(filesin):

                tFragment = wrftime2date(filename.split())[:]
                ncfile = nc.Dataset(filesin[n])
                if patt == 'wrf3hrly':
                    fwrf2d = nc.Dataset(filesin[n].replace('wrf3hrly','wrfout'))
                    fwrfgeo = nc.Dataset(f'{cfg.path_geo}/{cfg.file_geo}')
                    for varname in fwrf2d.variables.keys():
                        ncfile.variables[varname]=fwrf2d.variables[varname]
                    for varname in fwrfgeo.variables.keys():
                        ncfile.variables[varname]=fwrfgeo.variables[varname]

                xFragment,atts = cvars.compute_WRFvar(ncfile,varn)

                if len(tFragment)==1:
                    if len(xFragment.shape) == 3:
                        xFragment=np.expand_dims(xFragment,axis=0)
                    if len(xFragment.shape) == 2:
                        xFragment=np.expand_dims(xFragment,axis=0)

                ncfile.close()
                if patt == 'wrf3hrly':
                    fwrf2d.close()
                    fwrfgeo.close()
                x.append(xFragment)
                t.append(tFragment)

            varout = np.concatenate(x, axis=0)
            otimes = np.concatenate(t, axis=0)

        ###########################################################
        ###########################################################

        # ## Creating netcdf files
        fileout = f"{fullpathout}/{cfg.institution}_{varn}_{sdate}_{ntile:04d}.nc"
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

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###############################################################################
