#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-05-14T10:29:20+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-05-14T10:30:03+02:00
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

###########################################################
###########################################################

import xarray as xr
import numpy as np
import os
from glob import glob
import subprocess as subprocess
from joblib import Parallel, delayed
import argparse
from argparse import RawTextHelpFormatter
import dateparser
import datetime as dt
import epicc_config as cfg

###########################################################
###########################################################


def main():

    """ Calculate statistics from WRF postprocessed files for optimize plotting"""

    #INPUT ARGUMENTS
    parser = argparse.ArgumentParser(
        description="PURPOSE: Calculate statistics from WRF postprocessed files\n \n"
                    "OUTPUT:\n"
                    "- *_{period}_{stat}.nc: File with stat over the selected period.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-s", "--start",dest="sdatestr",type=str,help="Starting date  of the period to plot in format 2019-09-11 09:00\n partial datestrings such as 2019-09 also valid\n [default: 2019-09-11 09:00]",metavar="DATE",default=f'{cfg.syear}-01-01 00:00')
    parser.add_argument("-e", "--end"  ,dest="edatestr",type=str,help="Ending date  of the period to plot in format 2019-09-11 09:00\n partial datestrings such as 2019-09 also valid\n [default: 2019-09-15 09:30]",metavar="DATE",default=f'{cfg.eyear}-12-31 23:59')
    parser.add_argument("-v", "--var", dest="var", help="Variable to plot \n [default: RAIN]",metavar="VAR",default='RAIN')
    parser.add_argument("-t", "--thres"  ,dest="thres",type=str,help="Threshold to define extreme rainfall event",metavar="THRES",default='10')
    parser.add_argument("-f", "--freq", dest="freq",help="Frequency to plot from 10min to monthly\n [default: hourly]",metavar="FREQ",default='01H',choices=['10MIN','01H','DAY','MON'])
    parser.add_argument("-r", "--reg", dest="reg", help="Region to plot \n [default: EPICC]",metavar="REG",default='EPICC',choices=cfg.reg_coords.keys())

    args = parser.parse_args()

    sdate=dateparser.parse(args.sdatestr)
    edate=dateparser.parse(args.edatestr)
    varname = args.var
    freq= args.freq
    reg = args.reg
    pr_thres = int(args.thres)
    wrun = cfg.wrf_runs[0]


    files_all = sorted(glob(f'{cfg.path_postproc}/{wrun}/{cfg.institution}_{freq}_{varname}_????-??.nc'))
    filesin = sel_files(files_all,sdate.year,edate.year)

    print(f'Processing dates: {args.sdatestr} to {args.edatestr}')

    Parallel(n_jobs=12)(delayed(select_extreme_dates)(fin,reg,pr_thres) for fin in filesin)


###########################################################
###########################################################

def select_extreme_dates(fin_name,reg,pr_thres):

    print("Input: ", fin_name)
    fout_name = fin_name.replace(".nc",f"-{pr_thres}mm_{reg}.nc")

    if not os.path.isfile(fout_name):

        fin = xr.open_dataset(fin_name)

        if reg!='EPICC':
            fin_reg =  fin.where((fin.lat>=cfg.reg_coords[reg][0]) &\
                                 (fin.lat<=cfg.reg_coords[reg][2]) &\
                                (fin.lon>=cfg.reg_coords[reg][1]) &\
                                (fin.lon<=cfg.reg_coords[reg][3]),
                                drop=True)
        else:
            fin_reg = fin


        fout = fin.sel(time=(fin_reg.max(dim=['x','y']).RAIN>pr_thres))

        if len(fout.time)!=0:
            fout.to_netcdf(fout_name)
        else:
            print(f'No extreme events above {pr_thres}mm found in {fin_name}')


    else:
        print (f"File exist already: {fout_name}")
        #continue

def sel_files(filelist,syear,eyear):
    year=np.array([fname.split("_")[-1][0:4] for fname in filelist],np.int64)
    sel_files=[n for i,n in enumerate(filelist) if  ((year[i]>= syear) & (year[i]<= eyear))]
    return sel_files

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###############################################################################
