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

varnames_hfreq=['RAIN']
varnames_lfreq=[]
varnames = varnames_hfreq + varnames_lfreq
frequencies=['10MIN','01H','DAY','MON','DCYCLE']
path_in = "/vg5/dargueso-NO-BKUP/postprocessed/EPICC"
path_out = "/vg5/dargueso-NO-BKUP/postprocessed/unified/EPICC"
patt_inst=cfg.institution

def main():



    eday = calendar.monthrange(cfg.eyear,cfg.emonth)[1]
    datelist = pd.date_range(f'{cfg.syear}-{cfg.smonth}-01',f'{cfg.eyear}-{cfg.emonth}-{eday}',freq='MS').strftime("%Y-%m").tolist()



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
                        Parallel(n_jobs=12)(delayed(create_10min_files_from_pp)(fullpathin,fullpathout,yearmonth,patt_inst,varn) for yearmonth in datelist)

                if freq == '01H':
                    if varn in varnames_hfreq:
                        patt="%s_%s"%(patt_inst,'10MIN')
                        Parallel(n_jobs=12)(delayed(create_hourly_files)(fullpathout,yearmonth,patt,varn) for yearmonth in datelist)

                    if varn in varnames_lfreq:
                        patt="%s_%s"%(patt_inst,'01H')
                        Parallel(n_jobs=12)(delayed(create_hourly_files_from_pp)(fullpathin,fullpathout,yearmonth,patt_inst,varn) for yearmonth in datelist)

                if freq == 'DAY':
                    patt="%s_%s"%(patt_inst,'01H')
                    Parallel(n_jobs=12)(delayed(create_daily_files)(fullpathout,yearmonth,patt,varn) for yearmonth in datelist)

                if freq == 'MON':
                    patt="%s_%s"%(patt_inst,'DAY')
                    Parallel(n_jobs=12)(delayed(create_monthly_files)(fullpathout,yearmonth,patt,varn) for yearmonth in datelist)

###########################################################
###########################################################

def create_10min_files_from_pp(fullpathin,fullpathout,yearmonth,patt_inst,varn):

    """Create 10min files from original postprocessed"""

    fin = f'{fullpathin}/{yearmonth[:4]}/{patt_inst}_{varn}_{yearmonth}*'
    fout = f'{fullpathout}/{patt_inst}_10MIN_{varn}_{yearmonth}.nc'
    print(fin)
    subprocess.call(f"ncrcat {fin} {fout}",shell=True)

def create_hourly_files_from_pp(fullpathin,fullpathout,yearmonth,patt_inst,varn):

    """Create hourly files from original postprocessed"""

    fin = f'{fullpathin}/{yearmonth[:4]}/{patt_inst}_{varn}_{yearmonth}*'
    fout = f'{fullpathout}/{patt_inst}_01H_{varn}_{yearmonth}.nc'
    print(fin)
    subprocess.call(f"ncrcat {fin} {fout}",shell=True)

def create_hourly_files(fullpathout,yearmonth,patt,varn):

    """Create hourly files from 10min files"""

    fin = f'{fullpathout}/{patt}_{varn}_{yearmonth}.nc'
    fout = fin.replace("10MIN_%s" %(varn),"01H_%s" %(varn))
    print("Input: ", fin)
    print("Output: ", fout)
    if varn == 'RAIN':
        subprocess.call(f"cdo hoursum {fin} {fout}",shell=True)
    else:
        subprocess.call(f"cdo hourmean {fin} {fout}",shell=True)

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
