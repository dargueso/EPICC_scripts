#!/usr/bin/env python
'''
@File    :  check_completeness_simulation.py
@Time    :  2023/02/20 18:11:40
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2023, Daniel Argüeso
@Project :  EPICC
@Desc    :  None
'''

import argparse
import calendar
import os
import pandas as pd
import datetime as dt
from tqdm.auto import tqdm


def parse_args():

    parser = argparse.ArgumentParser(description='Check completeness of EPICC simulation')

    parser.add_argument(
        '-p', 
        '--path', 
        dest = 'path', 
        help="directory where EPICC raw files are stored", 
        type=str, 
        default='.',
    )

    parser.add_argument(
    '-s', 
    '--syear', 
    dest = 'syear', 
    help="first year to check", 
    type=int, 
    default=2011,
    )

    parser.add_argument(
    '-e', 
    '--eyear', 
    dest = 'eyear', 
    help="last year to check", 
    type=int, 
    default=2020,
    )


    args = parser.parse_args()

    return args

def check_missing(path, prefix, syear, eyear):

    sdate = dt.datetime(syear,1,1,0,0,0)
    edate = dt.datetime(eyear,12,31,0,0,0)
    n=0
    date_list = pd.date_range(sdate, edate, freq='D').strftime("%Y-%m-%d_%H:%M:%S").tolist()
    for date in tqdm(date_list, desc=f"Looping dates", leave="False"):
        file = f"{path}/{prefix}{date}"
        if not os.path.isfile(file):
            print(f"File {file} is missing")
            n +=1
    print (f"There are {n} {prefix} missing files of {len(date_list)})")
    

    
def main():

    args = parse_args()

    path = args.path
    syear = args.syear
    eyear = args.eyear

    print(path)
    print(f"Starting year: {syear}")
    print(f"Ending year: {eyear}")

    nyears = eyear - syear + 1
    nmonths = nyears * 12
    ndays = 365 * nyears + calendar.leapdays(syear, eyear)
    nhours = ndays * 24 * 60

    n = False

    check_missing(path, "wrfout_d01_", syear, eyear)
    check_missing(path, "wrf3hrly_d01_", syear, eyear)
    check_missing(path, "wrfprec_d01_", syear, eyear)

    # if n == False:
    #     print("There are no files missing for the period ", syear, "-", eyear)

    #     check_completeness("wrfout", "1")
    #     check_completeness("wrf3hrly", "1")
    #     check_completeness("wrfprec", "1")
    
    # else:
    #     print("There are missing files for the period ", syear, "-", eyear)
        



###############################################################################
# __main__  scope
###############################################################################

if __name__ == "__main__":
    raise SystemExit(main())

###############################################################################

