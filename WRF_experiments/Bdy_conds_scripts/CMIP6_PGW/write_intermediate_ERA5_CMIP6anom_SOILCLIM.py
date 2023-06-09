#!/usr/bin/env python

""" write_intermediate_ERA5_CMIP5anom.py
run write_intermediate_ERA5_CMIP5anom.py -s 2007 -e 2007

Authors: Daniel Argueso- Alejandro Di Luca @ CCRC, UNSW. Sydney (Australia)
email: a.diluca@unsw.edu.au

Created: Wed Jun 17 14:08:31 AEST 2015




Modified: March 23 2016
 - I added a conversion from relative humidity to specific humidity for the 3-d variable (Alejandro)
 - I added a mask to the surface temperature so the output looks the same as the ERA-Interm field (Alejandro)

Modified: May 30 2016
 - (Alejandro) I modified the calculation of the specific humidity in two ways:
    1) For vertical levels in the stratosphere (p<50 hPa) I assume that the saturation pressure is
    zero in the denominator in the calculation of the saturation mixing ratio. Otherwise, saturation
    vapor pressure becomes large than the total pressure!!!
    2) I set all specific humidity values smaller than zero to zero. Generally there are no values smaller
    than zero.

Modified March 27 2018
 - Adapted to ERA5 from a version for ERA-Interim (Daniel)

Modified Sept 6 2018 (from Alejandro July 9 2018 ERA-Interim version write_intermediate_ERAI_CMIP5anom.py)
 - (Alejandro) Netcdf files in /srv/ccrc/data19/z3393020/ERA-interim_CMIP5anom/ have different dimensions for
 files after EIN201401_an_pl.nc (see below).
 In files BEFORE EIN201401_an_pl.nc the pressure level variables is called lev
 In files AFTER EIN201401_an_pl.nc the pressure level variables is called plev

[z3444417@monsoon ERA-interim_CMIP5anom]$ ncdump -h EIN201401_an_pl.nc | grep plev
        plev = 37 ;
        double plev(plev) ;

[z3444417@monsoon ERA-interim_CMIP5anom]$ ncdump -h EIN201312_an_pl.nc | grep lev
        lev = 37 ;
        double lev(lev) ;
                lev:standard_name = "air_pressure" ;
                lev:long_name = "pressure" ;

 The name is not the only difference. The main difference is that in files BEFORE EIN201401_an_pl.nc the levels
 were order from the minima to the maxima while in files AFTER EIN201401_an_pl.nc the levels were order from the maximum
 to the minimum. So all calculation were wrong using the original script for events after 01-2014

I have now modified the script so it checks the name and order of the pressure vertical levels.

Modified July 7 2021
 - Adapted to CMIP6 and ERA5 (Daniel Argüeso)
 - Part of the EPICC project



"""

import netCDF4 as nc
import numpy as np
import glob as glob
from optparse import OptionParser
import calendar
import outputInter_soil as f90
import datetime as dt
import sys
import matplotlib.pyplot as plt
import copy as cp
import pdb
import os


### Options
parser = OptionParser()
parser.add_option(
    "-s",
    "--syear",
    type="int",
    dest="syear",
    help="first year to process",
    metavar="input argument",
)
parser.add_option(
    "-e",
    "--eyear",
    type="int",
    dest="eyear",
    help="last year to process",
    metavar="input argument",
)

(opts, args) = parser.parse_args()
###

overwrite_file = True
create_figs = False
syear = opts.syear
eyear = opts.eyear
nyears = eyear - syear + 1
month_i = 1
month_f = 12

varsoil = [
    "LANDSEA",
    "ST000007",
    "ST007028",
    "ST028100",
    "ST100289",
    "SM000007",
    "SM007028",
    "SM028100",
    "SM100289",
]
varsoil_codes = {
    "ST000007": "stl1",
    "ST007028": "stl2",
    "ST028100": "stl3",
    "ST100289": "stl4",
    "SM000007": "swvl1",
    "SM007028": "swvl2",
    "SM028100": "swvl3",
    "SM100289": "swvl4",
    "LANDSEA": "lsm",
}
var_units_era5 = {
    "LANDSEA": "0/1 Flag",
    "ST000007": "K",
    "ST007028": "K",
    "ST028100": "K",
    "ST100289": "K",
    "SM000007": "1",
    "SM007028": "1",
    "SM028100": "1",
    "SM100289": "1",
}

nfieldssoil = len(varsoil)


CMIP6anom_dir = "/home/dargueso/BDY_DATA/ERA5_CMIP6anom/"
ERA5_dir = "/home/dargueso/BDY_DATA/ERA5/ERA5_netcdf"
figs_path = "/home/dargueso/BDY_DATA/ERA5_CMIP6anom/Figs"

nlat = 601
nlon = 1200


file_ref = nc.Dataset(f"{ERA5_dir}/sfcclim.nc", "r")
lat = file_ref.variables["lat"][:]
lon = file_ref.variables["lon"][:]

olon, olat = np.meshgrid(lon, lat)


for y in range(nyears):
    year = y + syear
    ferasfc = nc.Dataset(f"{ERA5_dir}/sfcclim.nc", "r")
    Y = str(year)
    M = "12"
    D = "22"
    filedate = f"{Y}-{M}-{D}_00-00-00"

    vout = {}

    for var in varsoil:
        print("Processing variable %s" % (var))

        if var == "LANDSEA":
            sst = ferasfc.variables["sst"][:, :]
            vout[var] = np.int32(np.ma.getmask(sst))
        else:
            vout[var] = ferasfc.variables["%s" % (varsoil_codes[var])][:, :]

    fieldssoil = np.ndarray(
        shape=(nfieldssoil, nlat, nlon), dtype="float32"
    )  # ,order='Fortran')

    startlat = lat[0]
    startlon = lon[0]
    deltalon = 0.30
    deltalat = -0.30

    fieldssoil[0, :, :] = np.float32(vout["LANDSEA"])
    fieldssoil[1, :, :] = np.float32(vout["ST000007"])
    fieldssoil[2, :, :] = np.float32(vout["ST007028"])
    fieldssoil[3, :, :] = np.float32(vout["ST028100"])
    fieldssoil[4, :, :] = np.float32(vout["ST100289"])
    fieldssoil[5, :, :] = np.float32(vout["SM000007"])
    fieldssoil[6, :, :] = np.float32(vout["SM007028"])
    fieldssoil[7, :, :] = np.float32(vout["SM028100"])
    fieldssoil[8, :, :] = np.float32(vout["SM100289"])

    f90.writeintsoil(
        fieldssoil, filedate, nlat, nlon, startlat, startlon, deltalon, deltalat
    )
