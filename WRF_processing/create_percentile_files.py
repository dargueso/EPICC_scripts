#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-06-17T11:53:02+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-06-17T11:53:43+02:00
#
# @Project@ EPICC
# Version: 1.0
# Description: Script to calculate percentiles and wet-percentiles from postprocessed files
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import xarray as xr
import numpy as np

import epicc_config as cfg

import argparse
import dateparser
from glob import glob


###########################################################
###########################################################

def main():

    """ Calculate quantiles from WRF postprocessed files for optimize plotting"""

    #INPUT ARGUMENTS

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start",dest="sdatestr",type=str,help="Starting date  of the period to plot in format 2019-09-11 09:00\n partial datestrings such as 2019-09 also valid\n [default: 2019-09-11 09:00]",metavar="DATE",default='2019-09-11 09:00')
    parser.add_argument("-e", "--end"  ,dest="edatestr",type=str,help="Ending date  of the period to plot in format 2019-09-11 09:00\n partial datestrings such as 2019-09 also valid\n [default: 2019-09-15 09:30]",metavar="DATE",default='2019-09-15 09:30')
    parser.add_argument("-f", "--freq", dest="freq",help="Frequency to plot from 10min to monthly\n [default: hourly]",metavar="FREQ",default='01H',choices=['10MIN','01H','DAY','MON'])
    parser.add_argument("-v", "--var", dest="var", help="Variable to plot \n [default: RAIN]",metavar="VAR",default='RAIN')
    parser.add_argument("-w", "--wet", dest="wet", help="Whether to use all values (dry) or wet-only values (wet) \n [default: wet]",metavar="WET",default='wet')
    args = parser.parse_args()


    varname = args.var
    sdatestr  = args.sdatestr
    edatestr  = args.edatestr
    freq = args.freq
    wet = args.wet
    wet_value = cfg.wet_value
    wrun = cfg.wrf_runs[0]


    qtiles = np.asarray(cfg.qtiles)


    sdate=dateparser.parse(sdatestr)
    edate=dateparser.parse(edatestr)

    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_????-??.nc'))
    fin_all = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")
    fin = fin_all.sel(time=slice(sdate,edate)).squeeze()


    if wet == 'wet':
        qtilesp = fin[varname].load().where(fin[varname]>wet_value).quantile(qtiles,dim=['time'])
        qtilesp.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_{sdate.year}-{eyear.year}-qtiles_wetonly.nc')

    else:
        qtilesp = fin[varname].load().quantile(qtiles,dim=['time'])
        qtilesp.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_{sdate.year}-{edate.year}-qtiles.nc')

    fin_all.close()
    qtilesp.close()

    # for season in ['DJF','MAM','JJA','SON']:
    #     filesin_seas = sel_files_season(filesin,season)
    #     fin_all_seas = xr.open_mfdataset(filesin_seas,concat_dim="time", combine="nested")
    #     fin_seas = fin_all_seas.sel(time=slice(sdate,edate)).squeeze()
    #     if wet == True:
    #         qtilesp = fin_seas[varname].compute().where(fin_seas[varname]>wet_value).quantile(qtiles,dim=['time'])
    #         qtilesp.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_{syear}-{eyear}-qtiles_wetonly-{season}.nc')
    #
    #     else:
    #         qtilesp = fin_seas[varname].compute().quantile(qtiles,dim=['time'])
    #         qtilesp.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_{varname}_{syear}-{eyear}-qtiles-{season}.nc')
    #



###########################################################
###########################################################

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
