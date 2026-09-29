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
import EPICC_post_config as cfg
import calendar
import pandas as pd


###########################################################
###########################################################

varnames_hfreq=['PRNC']
varnames_mfreq=[]
varnames_lfreq=[]
varnames = varnames_hfreq + varnames_mfreq + varnames_lfreq
#frequencies=['10MIN','01H','03H','DAY','MON','DCYCLE']
frequencies=['10MIN','01H','DAY','MON']
path_in = cfg.path_proc
path_out = cfg.path_unif
patt_inst=cfg.institution
njobs = 12

def main():



    eday = calendar.monthrange(cfg.eyear,cfg.emonth)[1]
    datelist = pd.date_range(f'{cfg.syear}-{cfg.smonth}-01',f'{cfg.eyear}-{cfg.emonth}-{eday}',freq='MS').strftime("%Y-%m").tolist()
    datelist = datelist[:-1]


    for varn in varnames:
        for wrun in cfg.wruns:

            fullpathin = "%s/%s/" %(path_in,wrun)
            fullpathout = "%s/%s/" %(path_out,wrun)
            if not os.path.exists(fullpathout):
                os.makedirs(fullpathout)



            for freq in frequencies:

                if freq == '10MIN':
                    if varn in varnames_hfreq:
                        patt="%s_%s"%(patt_inst,'10MIN')
                        Parallel(n_jobs=njobs)(delayed(create_10min_files_from_pp)(fullpathin,fullpathout,yearmonth,patt_inst,varn) for yearmonth in datelist)

                if freq == '01H':
                    if varn in varnames_hfreq:
                        patt="%s_%s"%(patt_inst,'10MIN')
                        Parallel(n_jobs=njobs)(delayed(create_hourly_files)(fullpathout,yearmonth,patt,varn) for yearmonth in datelist)

                    if varn in varnames_mfreq:
                        patt="%s_%s"%(patt_inst,'01H')
                        Parallel(n_jobs=njobs)(delayed(create_hourly_files_from_pp)(fullpathin,fullpathout,yearmonth,patt_inst,varn) for yearmonth in datelist)

                if freq == '03H':

                    if varn in varnames_lfreq:
                        patt="%s_%s"%(patt_inst,'03H')
                        Parallel(n_jobs=njobs)(delayed(create_3hourly_files_from_pp)(fullpathin,fullpathout,yearmonth,patt_inst,varn) for yearmonth in datelist)


                if freq == 'DAY':
                    patt="%s_%s"%(patt_inst,'01H')
                    Parallel(n_jobs=njobs)(delayed(create_daily_files)(fullpathout,yearmonth,patt,varn) for yearmonth in datelist)

                if freq == 'MON':
                    patt="%s_%s"%(patt_inst,'DAY')
                    Parallel(n_jobs=njobs)(delayed(create_monthly_files)(fullpathout,yearmonth,patt,varn) for yearmonth in datelist)

###########################################################
###########################################################

def create_10min_files_from_pp(fullpathin,fullpathout,yearmonth,patt_inst,varn):

    """Create 10min files from original postprocessed"""

    fin = f'{fullpathin}/{patt_inst}_{varn}_{yearmonth}*'
    fout = f'{fullpathout}/{patt_inst}_10MIN_{varn}_{yearmonth}.nc'
    print(fin)
    subprocess.call(f"ncrcat {fin} {fout}",shell=True)

def create_hourly_files_from_pp(fullpathin,fullpathout,yearmonth,patt_inst,varn):

    """Create hourly files from original postprocessed"""

    fin = f'{fullpathin}/{patt_inst}_{varn}_{yearmonth}*'
    fout = f'{fullpathout}/{patt_inst}_01H_{varn}_{yearmonth}.nc'
    print(fin)
    subprocess.call(f"ncrcat {fin} {fout}",shell=True)

# Variables accumulated over the LAST output interval (WRF PREC_ACC_NC over
# prec_acc_dt): the value stamped t is rain from t-dt to t.
ACCUMULATED = ('RAIN', 'PRNC')


def create_hourly_files(fullpathout,yearmonth,patt,varn):

    """Create hourly files from 10min files

    Accumulated variables (RAIN, PRNC) are stamped at the END of their 10-min
    interval, so a plain `cdo hoursum` (grouping stamps HH:00..HH:50) gives
    rain from HH-1:50 to HH:50, 10 min early. Checked against RAINNC in the
    EPICC wrfout of 2020-01-10: stamps HH:10..HH+1:00 reproduce the exact
    hourly total, HH:00..HH:50 do not. So for those variables: append the
    first value of the next month (needed for the hour 23:00-00:00), move
    every stamp back by 10 min (to the START of its interval), keep this
    month, then sum. The last hour of the record has no closing value and
    comes out incomplete.
    Instantaneous variables are averaged as before.
    """

    fin = f'{fullpathout}/{patt}_{varn}_{yearmonth}.nc'
    fout = fin.replace("10MIN_%s" %(varn),"01H_%s" %(varn))
    print("Input: ", fin)
    print("Output: ", fout)
    if varn in ACCUMULATED:
        month = int(yearmonth[5:7])
        nextym = (pd.Timestamp(f"{yearmonth}-01") + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        fnext = f'{fullpathout}/{patt}_{varn}_{nextym}.nc'
        if os.path.exists(fnext):
            src = f"-mergetime {fin} -seltimestep,1 {fnext}"
        else:
            print(f"WARNING: {fnext} missing, the last hour of {yearmonth} is incomplete")
            src = fin
        subprocess.call(f"cdo hoursum -selmon,{month} -shifttime,-10min {src} {fout}",shell=True)
        # shifttime moves input time_bnds too: set the true clock-hour bounds
        set_hour_bounds(fout)
    else:
        subprocess.call(f"cdo hourmean {fin} {fout}",shell=True)

def set_hour_bounds(fout):
    """Set time_bnds to [HH:00, HH+1:00) for each hourly time stamp."""
    import netCDF4 as nc
    with nc.Dataset(fout, 'r+') as f:
        t = f.variables['time']
        cal = getattr(t, 'calendar', 'standard')
        stamps = pd.to_datetime([str(x) for x in nc.num2date(t[:], t.units, cal)]).round('s')
        start = stamps.floor('1h')
        end = start + pd.Timedelta('1h')
        bname = getattr(t, 'bounds', 'time_bnds')
        if bname not in f.variables:
            if 'bnds' not in f.dimensions:
                f.createDimension('bnds', 2)
            f.createVariable(bname, 'd', ('time', 'bnds'))
            t.bounds = bname
        f.variables[bname][:] = np.stack(
            [nc.date2num(list(x.to_pydatetime()), t.units, cal) for x in (start, end)], axis=1)


def create_3hourly_files_from_pp(fullpathin,fullpathout,yearmonth,patt_inst,varn):

    """Create hourly files from original postprocessed"""

    fin = f'{fullpathin}/{patt_inst}_{varn}_{yearmonth}*'
    fout = f'{fullpathout}/{patt_inst}_03H_{varn}_{yearmonth}.nc'
    print(fin)
    subprocess.call(f"ncrcat {fin} {fout}",shell=True)

def create_daily_files(fullpathout,yearmonth,patt,varn):
    """Create daily files from hourly files"""

    fin = f'{fullpathout}/{patt}_{varn}_{yearmonth}.nc'
    fout = fin.replace("01H_%s" %(varn),"DAY_%s" %(varn))
    print("Input: ", fin)
    print("Output: ", fout)
    if varn == 'RAIN':
        subprocess.call(f"cdo daysum {fin} {fout}",shell=True)
    else:
        subprocess.call(f"cdo daymean {fin} {fout}",shell=True)

def create_monthly_files(fullpathout,yearmonth,patt,varn):
    """Create monthly files from daily files"""
    fin = f'{fullpathout}/{patt}_{varn}_{yearmonth}.nc'
    fout = fin.replace("DAY_%s" %(varn),"MON_%s" %(varn))
    print("Input: ", fin)
    print("Output: ", fout)
    if varn == 'RAIN':
        subprocess.call(f"cdo monsum {fin} {fout}",shell=True)
    else:
        subprocess.call(f"cdo monmean {fin} {fout}",shell=True)

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###############################################################################
