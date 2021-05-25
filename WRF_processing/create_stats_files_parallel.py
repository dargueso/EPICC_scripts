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
import EPICC_post_config as cfg

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
    parser.add_argument("-t", "--stat"  ,dest="stat",type=str,help="Statistic metric to calculate",metavar="STAT",default='max',choices=['max','min','mean'])
    parser.add_argument("-a", "--seas"  ,dest="seas",type=str,help="Season to calculate stat over",metavar="SEAS",default='year',choices=['year','DJF','MAM','JJA','SON','all'])
    parser.add_argument("-f", "--freq", dest="freq",help="Frequency to plot from 10min to monthly\n [default: hourly]",metavar="FREQ",default='10MIN',choices=['10MIN','01H','DAY','MON'])

    args = parser.parse_args()

    sdate=dateparser.parse(args.sdatestr)
    edate=dateparser.parse(args.edatestr)
    varname = args.var
    season = args.seas
    stat = args.stat
    freq= args.freq
    wrun = cfg.wruns[0]

    files_all = sorted(glob(f'{cfg.path_unif}/{wrun}/{cfg.institution}_{freq}_{varname}_????-??.nc'))
    filesin = sel_files(files_all,sdate.year,edate.year)

    print(f'Processing dates: {args.sdatestr} to {args.edatestr}')

    Parallel(n_jobs=12)(delayed(compute_stat_cdo)(fin,stat) for fin in filesin)

    files_all_stat = sorted(glob(f'{cfg.path_unif}/{wrun}/{cfg.institution}_{freq}_{varname}_????-??-{stat}.nc'))
    files_stat = sel_files(files_all_stat,sdate.year,edate.year)

    merge_files_season(files_stat,season,sdate.year,edate.year,stat)


###########################################################
###########################################################

def merge_files_season(files_stat,season,syear,eyear,stat):

    print(f"Merging files for season {season}")
    files_stat_season = sel_files_season(files_stat,season)
    ending = files_stat_season[0].split('_')[-1]
    fout = files_stat_season[0].replace(f"{ending}",f"{syear}-{eyear}-{stat}-{season}.nc")
    if not os.path.isfile(fout):

        subprocess.call(f"cdo ens{stat} {' '.join(files_stat_season)} {fout}",shell=True)
    else:
        print (f"File exist already: {fout}")

def compute_stat_cdo(fin,stat):

    print("Input: ", fin)
    fout = fin.replace(".nc",f"-{stat}.nc")
    if not os.path.isfile(fout):
        subprocess.call(f"cdo tim{stat} {fin} {fout}",shell=True)

    else:
        print (f"File exist already: {fout}")
        #continue

def sel_files(filelist,syear,eyear):
    year=np.array([fname.split("_")[-1][0:4] for fname in filelist],np.int64)
    sel_files=[n for i,n in enumerate(filelist) if  ((year[i]>= syear) & (year[i]<= eyear))]
    return sel_files
def sel_files_season(filelist,season):
    month = np.array([fname.split("_")[-1][5:7] for fname in filelist],np.int64)
    seas_months = {'DJF': [1,2,12],
                   'MAM': list(range(3,6)),
                   'JJA': list(range(6,9)),
                   'SON': list(range(9,12)),
                   'YEAR': list(range(13))}

    sel_files=[n for i,n in enumerate(filelist) if  ((month[i] in seas_months[season]))]
    return sel_files

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###############################################################################
