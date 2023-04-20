#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2017-11-28T18:25:06+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2017-11-28T18:25:21+01:00
#
# @Project@ REHIPRE
# Version: x.0 (Beta)
# Description: Functions to calculate diagnostics from WRF outputs
#
# Dependencies: wrf-python, netCDF4, numpy
#
# Files:
#
#####################################################################
"""

import netCDF4 as nc
import numpy as np
import datetime as dt
import wrf as wrf
from constants import const as const
from scipy.ndimage import gaussian_filter
import wrf_utils as wrfu
import EPICC_post_config as cfg

# wrf.set_cache_size(0)
wrf.disable_xarray()


###########################################################
###########################################################
def compute_WRFvar(ncfile, varname):
    """Function to compute a variable name from wrf outputs
    filename: wrfout (or other file name used as input)
    varname : variable to be extracted or computed from WRF outputs
    """
    if varname in list(ncfile.variables.keys()):
        varval = ncfile.variables[varname][:]
        varatt = {}
        for att in ncfile.variables[varname].ncattrs():
            varatt[att] = getattr(ncfile.variables[varname], att)

    else:
        method_name = "compute_%s" % (varname)
        possibles = globals().copy()
        possibles.update(locals())
        compute = possibles.get(method_name)
        varval, varatt = compute(ncfile)

    return varval, varatt


###########################################################
###########################################################
def create_netcdf(var, filename):
    print((("\n Create output file %s") % (filename)))

    otimes = var["times"]
    outfile = nc.Dataset(
        filename, "w", format="NETCDF4_CLASSIC", zlib=True, complevel=5
    )
    outfile.createDimension("time", None)
    # outfile.createDimension('bnds',2)
    if var["values"].ndim == 4:
        outfile.createDimension("y", var["values"].shape[2])
        outfile.createDimension("x", var["values"].shape[3])
        outfile.createDimension("lev", var["values"].shape[1])

        outvar = outfile.createVariable(
            var["varname"],
            "f",
            ("time", "lev", "y", "x"),
            zlib=True,
            complevel=5,
            fill_value=const.missingval,
        )

    if var["values"].ndim == 3:
        outfile.createDimension("y", var["values"].shape[1])
        outfile.createDimension("x", var["values"].shape[2])

        outvar = outfile.createVariable(
            var["varname"],
            "f",
            ("time", "y", "x"),
            zlib=True,
            complevel=5,
            fill_value=const.missingval,
        )

    outtime = outfile.createVariable(
        "time", "d", "time", zlib=True, complevel=5, fill_value=const.missingval
    )
    # outtime_bnds = outfile.createVariable('time_bnds','f8',('time','bnds'),fill_value=const.missingval)
    outlat = outfile.createVariable(
        "lat", "f", ("y", "x"), zlib=True, complevel=5, fill_value=const.missingval
    )
    outlon = outfile.createVariable(
        "lon", "f", ("y", "x"), zlib=True, complevel=5, fill_value=const.missingval
    )

    if var["values"].ndim == 4:
        outlev = outfile.createVariable(
            "levels", "f", ("lev"), zlib=True, complevel=5, fill_value=const.missingval
        )
        if var["varname"] == "cloudfrac":
            setattr(outlev, "standard_name", "cloud-level")
            setattr(outlev, "long_name", "Clouds level")
            setattr(outlev, "units", "")
            setattr(outlev, "_CoordinateAxisType", "z")
        else:
            setattr(outlev, "standard_name", "model-level")
            setattr(outlev, "long_name", "Model level")
            setattr(outlev, "units", "eta levels")
            setattr(outlev, "_CoordinateAxisType", "z")

    setattr(outlat, "standard_name", "latitude")
    setattr(outlat, "long_name", "Latitude")
    setattr(outlat, "units", "degrees_north")
    setattr(outlat, "_CoordinateAxisType", "Lat")

    setattr(outlon, "standard_name", "longitude")
    setattr(outlon, "long_name", "Longitude")
    setattr(outlon, "units", "degrees_east")
    setattr(outlon, "_CoordinateAxisType", "Lon")

    setattr(outtime, "standard_name", "time")
    setattr(outtime, "long_name", "Time")
    setattr(outtime, "units", "seconds since 1949-12-01 00:00:00")
    setattr(outtime, "calendar", "standard")

    outtime[:] = nc.date2num(
        [otimes[x] for x in range(len(otimes))],
        units="seconds since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outlat[:] = var["lat"][:]
    outlon[:] = var["lon"][:]

    outvar[:] = var["values"][:]

    for outatt in list(var["atts"].keys()):
        setattr(outvar, outatt, var["atts"][outatt])

    setattr(outfile, "creation_date", dt.datetime.today().strftime("%Y-%m-%d"))
    setattr(outfile, "author", "Daniel Argueso @UIB")
    setattr(outfile, "contact", "d.argueso@uib.es")
    # setattr(outfile,'comments','files created from wrf outputs %s/%s' %(path_in,patt))

    outfile.close()


###########################################################
###########################################################


def compute_PRNC(ncfile):
    """Function to calculate non-convective precipitation flux from a wrf output
    It also provides variable attribute CF-Standard
    """

    ## Specific to PR
    if hasattr(ncfile, "PREC_ACC_DT"):
        accum_dt = getattr(ncfile, "PREC_ACC_DT")
    else:
        print(("NO PREC_ACC_DT in input file. Set to default %s min" % (cfg.acc_dt)))
        accum_dt = int(cfg.acc_dt)

    ## Computing diagnostic
    prnc_acc = ncfile.variables["PREC_ACC_NC"][:]

    ## Deacumulating over prac_acc_dt (namelist entry)
    prnc = prnc_acc / (accum_dt * 60.0)

    atts = {
        "standard_name": "non_convective_precipitation_flux",
        "long_name": "non-convective total precipitation flux",
        "units": "kg m-2 s-1",
    }

    return prnc, atts


def compute_RAIN(ncfile):
    """Function to calculate non-convective precipitation flux from a wrf output
    It also provides variable attribute CF-Standard
    """

    ## Computing diagnostic
    if "PREC_ACC" in ncfile.variables.keys():
        pr_acc = ncfile.variables["PREC_ACC_NC"][:] + ncfile.variables["PREC_ACC"][:]
    else:
        prnc_acc = ncfile.variables["PREC_ACC_NC"][:]

    ## Deacumulating over prac_acc_dt (namelist entry)
    rain = prnc_acc

    atts = {
        "standard_name": "Accumulated rainfall",
        "long_name": "Accumulated rainfall",
        "units": "mm",
    }

    return rain, atts


def compute_TAS(ncfile):
    """Function to calculate 2-m temperature from WRF OUTPUTS
    It also provides variable attributes CF-Standard
    """

    t2 = ncfile.variables["T2"][:]

    atts = {
        "standard_name": "air_temperature",
        "long_name": "Surface air temperature",
        "units": "K",
        "hgt": "2 m",
    }

    return t2, atts


def compute_TD2(ncfile):
    """Function to calculate 2-m dewpoint temperature from WRF OUTPUTS
    It also provides variable attributes CF-Standard
    """

    td2 = wrf.getvar(ncfile, "td2", wrf.ALL_TIMES, units="K")

    atts = {
        "standard_name": "air_dewpoint_temperature",
        "long_name": "Surface air dewpoint temperature",
        "units": "K",
        "hgt": "2 m",
    }

    return td2, atts


###########################################################
###########################################################
def compute_TC(ncfile):
    """Function to calculate temperature in degC at model full levels from WRF outputs
    It also provides variable attributes CF-Standard
    """

    tc = wrf.getvar(ncfile, "tc", wrf.ALL_TIMES)

    atts = {
        "standard_name": "air_temperature",
        "long_name": "Air temperature",
        "units": "degC",
        "hgt": "full_model_level",
    }

    return tc, atts


###########################################################
###########################################################
def compute_WA(ncfile):
    """Function to calculate vertical wind at model full levels from WRF outputs
    It also provides variable attributes CF-Standard
    """

    wa = wrf.getvar(ncfile, "wa", wrf.ALL_TIMES)

    atts = {
        "standard_name": "vertical_wind_speed",
        "long_name": "Vertical wind speed",
        "units": "m s-1",
        "hgt": "full_model_level",
    }

    return wa, atts


###########################################################
###########################################################


def compute_HUSS(ncfile):
    """Function to calculate specific humidity near surface from WRF outputs
    It also provides variable attributes CF-Standard
    """

    q2 = ncfile.variables["Q2"][:]
    huss = q2 / (1 + q2)

    atts = {
        "standard_name": "specific_humidity",
        "long_name": "Surface specific humidity",
        "units": "kg/kg",
        "hgt": "2 m",
    }

    return huss, atts


def compute_RSDS(ncfile):
    """Function to calculate downward shortwave surface radiation
    It also provides variable attributes CF-Standard
    """
    swdown = ncfile.variables["SWDOWN"][:]

    atts = {
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "long_name": "Downward SW surface radiation",
        "units": "W m-2",
    }

    return swdown, atts


def compute_RLDS(ncfile):
    """Function to calculate downward longwave surface radiation
    It also provides variable attributes CF-Standard
    """
    glw = ncfile.variables["GLW"][:]

    atts = {
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "long_name": "Downward LW surface radiation",
        "units": "W m-2",
    }

    return glw, atts


def compute_PSFC(ncfile):
    """Function to calculate surface pressure from WRF OUTPUTS
    It also provides variable attributes CF-Standard
    """

    psfc = ncfile.variables["PSFC"][:]

    atts = {
        "standard_name": "surface_pressure",
        "long_name": "Surface pressure",
        "units": "Pa",
    }

    return psfc, atts


def compute_PSL(ncfile):
    """Function to calculate PSL using wrf-python diagnostics
    It also provides variable attribute CF-Standard
    """
    # Get the sea level pressure using wrf-python
    psl = wrf.getvar(ncfile, "slp", wrf.ALL_TIMES)

    atts = {
        "standard_name": "air_pressure_at_mean_sea_level",
        "long_name": "Sea Level Pressure",
        "units": "Pa",
    }

    return psl, atts


def compute_PVO(ncfile):
    """Function to calculate PVU (potential vorticity) using wrf-python diagnostics
    It also provides variable attribute CF-Standard
    """

    # Get the sea level pressure using wrf-python
    psl = wrf.getvar(ncfile, "pvo", wrf.ALL_TIMES)

    atts = {
        "standard_name": "air_pressure_at_mean_sea_level",
        "long_name": "Sea Level Pressure",
        "units": "PVU",
    }

    return psl, atts


def compute_U10MET(ncfile):
    """Function to calculate zonal 10-m windspeed rotated to Earth coordinates
    from WRF OUTPUTS
    It also provides variable attributes CF-Standard
    """

    # Get vertical wind using wrf-python

    ua = np.squeeze(wrf.getvar(ncfile, "uvmet10", wrf.ALL_TIMES)[0, :])

    atts = {
        "standard_name": "zonal_wind_speed",
        "long_name": "Zonal earth coordinates wind speed",
        "units": "m s-1",
        "hgt": "10 m",
    }

    return ua, atts


def compute_V10MET(ncfile):
    """Function to calculate meridional 10-m windspeed rotated to Earth coordinates
    from WRF OUTPUTS
    It also provides variable attributes CF-Standard
    """
    # Get vertical wind using wrf-python

    va = np.squeeze(wrf.getvar(ncfile, "uvmet10", wrf.ALL_TIMES)[1, :])

    atts = {
        "standard_name": "meridional_wind_speed",
        "long_name": "Meridional earth coordinates wind speed",
        "units": "m s-1",
        "hgt": "10 m",
    }

    return va, atts


def compute_WSPD10(ncfile):
    """Function to calculate 10-m wind speed on Earth coordinates
    from WRF OUTPUTS
    It also provides variable attributes CF-Standard
    """

    # Get wind speed (this function calculates both speed and direction)
    # We select speed only

    uvmet10_wind = np.squeeze(
        wrf.getvar(ncfile, "uvmet10_wspd_wdir", wrf.ALL_TIMES)[0, :]
    )

    atts = {
        "standard_name": "wind_speed",
        "long_name": "Earth coordinates wind speed",
        "units": "m s-1",
        "hgt": "10 m",
    }

    return uvmet10_wind, atts


def compute_PW(ncfile):
    """Function to calculate precipitable water in the column"""

    pw = wrf.getvar(ncfile, "pw", wrf.ALL_TIMES)

    atts = {
        "standard_name": "precipitable_water",
        "long_name": "column precipitable water",
        "units": "kg m-2",
    }

    return pw, atts


###########################################################
###########################################################


def compute_CAPE2D(ncfile):
    """Function to calculate CAPE using methods described in:
    http://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.cape_2d.html
    This is NOT CF-compliant in any way. cape2d contains 4 variables distributed by levels: MCAPE [J kg-1], MCIN[J kg-1], LCL[m] and LFC[m]
    """
    pres_hpa = wrf.getvar(ncfile, "pressure", wrf.ALL_TIMES)
    tkel = wrf.getvar(ncfile, "tk", wrf.ALL_TIMES)
    qv = wrf.getvar(ncfile, "QVAPOR", wrf.ALL_TIMES)
    z = wrf.getvar(ncfile, "geopotential", wrf.ALL_TIMES) / const.g
    psfc = wrf.getvar(ncfile, "PSFC", wrf.ALL_TIMES) / 100.0  # Converto to hPA
    terrain = wrf.getvar(ncfile, "ter", wrf.ALL_TIMES)
    ter_follow = True

    cape2d = wrf.cape_2d(
        pres_hpa,
        tkel,
        qv,
        z,
        terrain,
        psfc,
        ter_follow,
        missing=const.missingval,
        meta=False,
    )

    atts = {
        "standard_name": "cape2d_variables",
        "long_name": "mcape mcin lcl lfc",
        "units": "SI",
    }

    return cape2d, atts
