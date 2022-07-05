#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2019-11-06T11:09:25+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2019-11-06T11:09:27+01:00
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





from glob import glob
import xarray as xr
import os



wruns=['EPICC_2km_ERA5_HVC_GWD','EPICC_2km_ERA5_CMIP6anom_HVC_GWD']

path_in = "/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/"
path_out = "/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC"

patt_in="UIB_03H_PLEVS"

var='WA'

for n,wrun in enumerate(wruns):
    fullpathin = "%s/%s/" %(path_in,wrun)
    fullpathout = "%s/%s/" %(path_out,wrun)

    if not os.path.exists(fullpathout):
        os.makedirs(fullpathout)


    filesin = sorted(glob(f'{fullpathin}/{patt_in}_{var}*'))
    for file in filesin:
        print(file)
    
        fin=xr.open_dataset(file)
        fin.coords['lev']=fin.levels

        finavg=fin.sel(lev=slice(900,600)).mean('lev').squeeze()
        fileout = file.replace(patt_in,'UIB_VAVG_900-600hPA_03H')
        fin.to_netcdf(fileout)
