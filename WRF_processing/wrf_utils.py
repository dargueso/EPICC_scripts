#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2018-02-13T10:26:27+11:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2018-02-13T10:26:29+11:00
#
# @Project@
# Version: x.0 (Beta)
# Description: Utilities to deal with WRF outputs
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import datetime as dt
import netCDF4 as nc
import numpy as np
import xarray as xr
import os as os
from glob import glob
from dateutil.relativedelta import relativedelta
import wrf as wrf
import compute_vars as cvars
import calendar
import subprocess


def sel_wrfout_files(filelist, sdate, edate):
    """Module to select files from a file list that have records between two
    given dates
    ----
    Output: list of files
    """

    d1 = dt.date(np.int64(sdate[0:4]), np.int64(sdate[5:7]), np.int64(sdate[8:10]))
    d2 = dt.date(np.int64(edate[0:4]), np.int64(edate[5:7]), np.int64(edate[8:10]))

    years = np.array(
        [fname.split("/")[-1].split("_")[2][0:4] for fname in filelist], np.int64
    )
    months = np.array(
        [fname.split("/")[-1].split("_")[2][5:7] for fname in filelist], np.int64
    )
    days = np.array(
        [fname.split("/")[-1].split("_")[2][8:10] for fname in filelist], np.int64
    )

    file_dates = np.array(
        [dt.date(years[i], months[i], days[i]) for i in range(len(years))]
    )

    selec_files = [
        filelist[i] for i, n in enumerate(file_dates) if ((n >= d1) & (n <= d2))
    ]

    return selec_files


###########################################################
###########################################################


def wrftime2date(files):
    """
    Conversion of dates from a wrf file or a list of wrf files
    format: [Y] [Y] [Y] [Y] '-' [M] [M] '-' [D] [D] '_' [H] [H] ':' [M] [M] ':' [S] [S]
    to a datetime object.
    """

    if len(files) == 1:
        fin = nc.Dataset(str(files[0]), "r")
        times = fin.variables["Times"]
    else:
        fin = nc.MFDataset(files[:])
        times = fin.variables["Times"]

    year = np.zeros(len(times), dtype=np.int64)
    month = year.copy()
    day = year.copy()
    hour = year.copy()
    minute = year.copy()
    second = year.copy()

    for i in range(len(times)):
        listdate = times[i]
        year[i] = (
            int(listdate[0]) * 1000
            + int(listdate[1]) * 100
            + int(listdate[2]) * 10
            + int(listdate[3])
        )
        month[i] = int(listdate[5]) * 10 + int(listdate[6])
        day[i] = int(listdate[8]) * 10 + int(listdate[9])
        hour[i] = int(listdate[11]) * 10 + int(listdate[12])
        minute[i] = int(listdate[14]) * 10 + int(listdate[15])
        second[i] = int(listdate[17]) * 10 + int(listdate[18])

    dates = [
        dt.datetime(year[i], month[i], day[i], hour[i], minute[i], second[i])
        for i in range(len(times))
    ]
    return dates


###########################################################
###########################################################


def plevs_interp(
    path_in,
    path_out,
    geofile_ref,
    syear,
    eyear,
    smonth,
    emonth,
    plevs,
    patt,
    patt_wrf,
    dom,
    wrun,
    varn,
):
    fullpathin = path_in + "/" + wrun + "/out"
    fullpathout = path_out + "/" + wrun + "/" + str(syear) + "-" + str(eyear)

    inputinf = {}
    inputinf["geofile"] = geofile_ref
    geofile = nc.Dataset(geofile_ref)

    y = syear
    m = smonth
    d = 1

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s-%s" % (y, str(m).rjust(2, "0"), str(d).rjust(2, "0"))

        filesin_wrf = sorted(
            glob("%s/%s/out/%s_%s_%s*" % (path_in, wrun, patt_wrf, dom, sdate))
        )

        z = []
        t = []

        for thour in range(len(filesin_wrf)):
            fwrf3d = nc.Dataset(filesin_wrf[thour])
            fwrf2d = nc.Dataset(filesin_wrf[thour].replace(patt_wrf, "wrfout"))

            fwrf3d.variables["F"] = geofile.variables["F"]
            for varname in fwrf2d.variable.keys():
                fwrf3d.variables[varname] = fwrf2d.variables[varname]

            tFragment = wrftime2date(filesin_wrf[thour].split())[:]

            field, atts = cvars.compute_WRFvar(fwrf3d, varn)
            zFragment = wrf.vinterp(
                fwrf3d, np.squeeze(field), vert_coord="pressure", interp_levels=plevs
            )

            zFragment = np.expand_dims(zFragment, axis=0)
            z.append(zFragment)
            t.append(tFragment)

        fieldint = np.concatenate(z, axis=0)
        otimes = np.concatenate(t, axis=0)

        varinfo = {
            "values": fieldint,
            "varname": varn,
            "plevs": plevs,
            "atts": atts,
            "lat": fwrf.variables["XLAT"][0, :],
            "lon": fwrf.variables["XLONG"][0, :],
            "times": otimes,
        }

        fileout = "%s/%s_PLEVS_%s_%s.nc" % (fullpathout, patt, varn, sdate)
        create_plevs_netcdf(varinfo, fileout)

        edate = otimes[-1] + dt.timedelta(days=1)

        y = edate.year
        m = edate.month
        d = edate.day


###########################################################
###########################################################


def plevs_interp_byday(
    fullpathin, fullpathout, geofile_ref, date, plevs, patt, patt_wrf, dom, wrun, varn
):
    inputinf = {}
    inputinf["geofile"] = geofile_ref
    geofile = nc.Dataset(geofile_ref)

    y = date.year
    m = date.month
    d = date.day

    print(y, m, d)
    sdate = "%s-%s-%s" % (y, str(m).rjust(2, "0"), str(d).rjust(2, "0"))

    filein_wrf3d = "%s/%s_%s_%s_00:00:00" % (fullpathin, patt_wrf, dom, sdate)
    filein_wrf2d = filein_wrf3d.replace(patt_wrf, "wrfout")
    filein_aux = f"./aux_mientras2/aux_{sdate}.nc"

    os.system(f"ncks -d Time,0,23,3 {filein_wrf2d} {filein_aux}")

    # try:
    #     print(f"ncks -d Time,0,23,3 {filein_wrf2d} {filein_aux}")
    #     subprocess.check_output(f"ncks -d Time,0,23,3 {filein_wrf2d} {filein_aux}")
    # except Exception:
    #     raise SystemExit(f"ERROR: Could not select times for {filein_wrf2d}")

    fwrf3d = nc.Dataset(filein_wrf3d)
    fwrf2d = nc.Dataset(filein_aux)
    otimes = wrftime2date(filein_wrf3d.split())[:]

    # fwrf3d.variables["F"] = geofile.variables["F"]
    ntimes = fwrf3d.dimensions['Time'].size
    fwrf3d.variables['F'] = np.broadcast_to(geofile.variables['F'][:],(ntimes,) + geofile.variables['F'][:].squeeze().shape)

    for varname in fwrf2d.variables.keys():
        fwrf3d.variables[varname] = fwrf2d.variables[varname]

    field, atts = cvars.compute_WRFvar(fwrf3d, varn)
    fieldint = wrf.vinterp(
        fwrf3d,
        np.squeeze(field),
        vert_coord="pressure",
        interp_levels=plevs,
        timeidx=wrf.ALL_TIMES,
    )
    # zFragment = np.expand_dims(zFragment, axis=0)
    # z.append(zFragment)
    # t.append(tFragment)

    # fieldint = np.concatenate(z, axis=0)
    # otimes = np.concatenate(t, axis=0)

    varinfo = {
        "values": fieldint,
        "varname": varn,
        "plevs": plevs,
        "atts": atts,
        "lat": fwrf2d.variables["XLAT"][0, :],
        "lon": fwrf2d.variables["XLONG"][0, :],
        "times": otimes,
    }

    fileout = "%s/%s_PLEVS_%s_%s.nc" % (fullpathout, patt, varn, sdate)
    create_plevs_netcdf(varinfo, fileout)
    fwrf3d.close()
    fwrf2d.close()
    os.remove(filein_aux)


###########################################################
###########################################################
def create_plevs_netcdf(var, filename):
    print(("\n Create output file %s") % (filename))

    otimes = var["times"]
    outfile = nc.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    outfile.createDimension("time", None)
    outfile.createDimension("bnds", 2)

    outfile.createDimension("y", var["values"].shape[2])
    outfile.createDimension("x", var["values"].shape[3])
    outfile.createDimension("lev", var["values"].shape[1])

    outvar = outfile.createVariable(
        var["varname"], "f", ("time", "lev", "y", "x"), fill_value=1e20
    )

    outtime = outfile.createVariable("time", "f8", "time", fill_value=1e20)
    outtime_bnds = outfile.createVariable(
        "time_bnds", "f8", ("time", "bnds"), fill_value=1e20
    )
    outlat = outfile.createVariable("lat", "f", ("y", "x"), fill_value=1e20)
    outlon = outfile.createVariable("lon", "f", ("y", "x"), fill_value=1e20)
    outlev = outfile.createVariable("levels", "f", ("lev"), fill_value=1e20)

    setattr(outlev, "standard_name", "pressure")
    setattr(outlev, "long_name", "pressure_level")
    setattr(outlev, "units", "hPa")
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
    setattr(outtime, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime, "calendar", "standard")

    setattr(outtime_bnds, "standard_name", "time_bnds")
    setattr(outtime_bnds, "long_name", "time_bounds")
    setattr(outtime_bnds, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime_bnds, "calendar", "standard")

    step_seconds = np.int64((otimes[1] - otimes[0]).total_seconds())

    outtime[:] = nc.date2num(
        [otimes[x] for x in range(len(otimes))],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outtime_bnds[:, 0] = nc.date2num(
        [
            otimes[x] - dt.timedelta(seconds=step_seconds / 2.0)
            for x in range(len(otimes))
        ],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )
    outtime_bnds[:, 1] = nc.date2num(
        [
            otimes[x] + dt.timedelta(seconds=step_seconds / 2.0)
            for x in range(len(otimes))
        ],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outlat[:] = var["lat"][:]
    outlon[:] = var["lon"][:]
    outlev[:] = np.asarray(var["plevs"])

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


###########################################################
###########################################################
def create_zlevs_netcdf(var, filename):
    print(("\n Create output file %s") % (filename))

    otimes = var["times"]
    outfile = nc.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    outfile.createDimension("time", None)
    outfile.createDimension("bnds", 2)

    outfile.createDimension("y", var["values"].shape[2])
    outfile.createDimension("x", var["values"].shape[3])
    outfile.createDimension("lev", var["values"].shape[1])

    outvar = outfile.createVariable(
        var["varname"], "f", ("time", "lev", "y", "x"), fill_value=1e20
    )

    outtime = outfile.createVariable("time", "f8", "time", fill_value=1e20)
    outtime_bnds = outfile.createVariable(
        "time_bnds", "f8", ("time", "bnds"), fill_value=1e20
    )
    outlat = outfile.createVariable("lat", "f", ("y", "x"), fill_value=1e20)
    outlon = outfile.createVariable("lon", "f", ("y", "x"), fill_value=1e20)
    outlev = outfile.createVariable("levels", "f", ("lev"), fill_value=1e20)

    setattr(outlev, "standard_name", "height")
    setattr(outlev, "long_name", "height_above_ground_level")
    setattr(outlev, "units", "m")
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
    setattr(outtime, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime, "calendar", "standard")

    setattr(outtime_bnds, "standard_name", "time_bnds")
    setattr(outtime_bnds, "long_name", "time_bounds")
    setattr(outtime_bnds, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime_bnds, "calendar", "standard")

    step_seconds = np.int64((otimes[1] - otimes[0]).total_seconds())

    outtime[:] = nc.date2num(
        [otimes[x] for x in range(len(otimes))],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outtime_bnds[:, 0] = nc.date2num(
        [
            otimes[x] - dt.timedelta(seconds=step_seconds / 2.0)
            for x in range(len(otimes))
        ],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )
    outtime_bnds[:, 1] = nc.date2num(
        [
            otimes[x] + dt.timedelta(seconds=step_seconds / 2.0)
            for x in range(len(otimes))
        ],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outlat[:] = var["lat"][:]
    outlon[:] = var["lon"][:]
    outlev[:] = np.asarray(var["zlevs"]) * 1000.0

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


def create_netcdf_vcross(var, filename):
    print(("\n Create output file %s") % (filename))

    otimes = var["times"]
    outfile = nc.Dataset(
        filename, "w", format="NETCDF4_CLASSIC", zlib=True, complevel=5
    )

    outfile.createDimension("time", None)
    outfile.createDimension("bnds", 2)
    outfile.createDimension("lev", var["values"].shape[1])
    outfile.createDimension("x", var["values"].shape[2])

    outvar = outfile.createVariable(
        var["varname"], "f", ("time", "lev", "x"), fill_value=1e20
    )

    outtime = outfile.createVariable("time", "f8", "time", fill_value=1e20)
    outtime_bnds = outfile.createVariable(
        "time_bnds", "f8", ("time", "bnds"), fill_value=1e20
    )
    outlat = outfile.createVariable("lat", "f", ("x"), fill_value=1e20)
    outlon = outfile.createVariable("lon", "f", ("x"), fill_value=1e20)
    outlev = outfile.createVariable("levels", "f", ("lev"), fill_value=1e20)

    setattr(outlat, "standard_name", "latitude")
    setattr(outlat, "long_name", "Latitude")
    setattr(outlat, "units", "degrees_north")
    setattr(outlat, "_CoordinateAxisType", "Lat")

    setattr(outlon, "standard_name", "longitude")
    setattr(outlon, "long_name", "Longitude")
    setattr(outlon, "units", "degrees_east")
    setattr(outlon, "_CoordinateAxisType", "Lon")
    if var["vcross"] == "p":
        setattr(outlev, "standard_name", "pressure")
        setattr(outlev, "long_name", "pressure_level")
        setattr(outlev, "units", "hPa")
        setattr(outlev, "_CoordinateAxisType", "z")
    elif var["vcross"] == "z":
        setattr(outlev, "standard_name", "height")
        setattr(outlev, "long_name", "height_above_ground_level")
        setattr(outlev, "units", "m")
        setattr(outlev, "_CoordinateAxisType", "z")

    setattr(outtime, "standard_name", "time")
    setattr(outtime, "long_name", "Time")
    setattr(outtime, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime, "calendar", "standard")

    setattr(outtime_bnds, "standard_name", "time_bnds")
    setattr(outtime_bnds, "long_name", "time_bounds")
    setattr(outtime_bnds, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime_bnds, "calendar", "standard")

    step_seconds = np.int64((otimes[1] - otimes[0]).total_seconds())

    outtime[:] = nc.date2num(
        [otimes[x] for x in range(len(otimes))],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outtime_bnds[:, 0] = nc.date2num(
        [
            otimes[x] - dt.timedelta(seconds=step_seconds / 2.0)
            for x in range(len(otimes))
        ],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )
    outtime_bnds[:, 1] = nc.date2num(
        [
            otimes[x] + dt.timedelta(seconds=step_seconds / 2.0)
            for x in range(len(otimes))
        ],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    outlat[:] = var["lat"][:]
    outlon[:] = var["lon"][:]
    outlev[:] = var["vlevs"][:]

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


def create_netcdf(var, filename):
    print(("\n Create output file %s") % (filename))

    otimes = var["times"]
    outfile = nc.Dataset(
        filename, "w", format="NETCDF4_CLASSIC", zlib=True, complevel=5
    )

    outfile.createDimension("time", None)
    # outfile.createDimension('bnds',2)

    outfile.createDimension("y", var["values"].shape[1])
    outfile.createDimension("x", var["values"].shape[2])

    outvar = outfile.createVariable(
        var["varname"], "f", ("time", "y", "x"), fill_value=1e20
    )

    outtime = outfile.createVariable("time", "f8", "time", fill_value=1e20)
    outtime_bnds = outfile.createVariable(
        "time_bnds", "f8", ("time", "bnds"), fill_value=1e20
    )
    # outlat  = outfile.createVariable('lat','f',('y','x'),fill_value=1e20)
    # outlon  = outfile.createVariable('lon','f',('y','x'),fill_value=1e20)

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
    setattr(outtime, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime, "calendar", "standard")

    setattr(outtime_bnds, "standard_name", "time_bnds")
    setattr(outtime_bnds, "long_name", "time_bounds")
    setattr(outtime_bnds, "units", "hours since 1949-12-01 00:00:00")
    setattr(outtime_bnds, "calendar", "standard")

    step_seconds = np.int64((otimes[1] - otimes[0]).total_seconds())

    outtime[:] = nc.date2num(
        [otimes[x] for x in range(len(otimes))],
        units="hours since 1949-12-01 00:00:00",
        calendar="standard",
    )

    # outtime_bnds[:,0]=nc.date2num([otimes[x]-dt.timedelta(seconds=step_seconds/2.) for x in range(len(otimes))],units='hours since 1949-12-01 00:00:00',calendar='standard')
    # outtime_bnds[:,1]=nc.date2num([otimes[x]+dt.timedelta(seconds=step_seconds/2.) for x in range(len(otimes))],units='hours since 1949-12-01 00:00:00',calendar='standard')

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
def zlevs_interp(
    path_in,
    path_out,
    geofile_ref,
    syear,
    eyear,
    smonth,
    emonth,
    zlevs,
    patt,
    patt_wrf,
    dom,
    wrun,
    varn,
):
    fullpathin = path_in + "/" + wrun + "/out"
    fullpathout = path_out + "/" + wrun + "/" + str(syear) + "-" + str(eyear)

    inputinf = {}
    inputinf["geofile"] = geofile_ref
    geofile = nc.Dataset(geofile_ref)

    y = syear
    m = smonth
    d = 1

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s-%s" % (y, str(m).rjust(2, "0"), str(d).rjust(2, "0"))

        filesin_wrf = sorted(
            glob("%s/%s/out/%s_%s_%s*" % (path_in, wrun, patt_wrf, dom, sdate))
        )

        z = []
        t = []

        for thour in range(len(filesin_wrf)):
            fwrf = nc.Dataset(filesin_wrf[thour])
            fwrf.variables["F"] = geofile.variables["F"]
            tFragment = wrftime2date(filesin_wrf[thour].split())[:]
            field, atts = cvars.compute_WRFvar(filesin_wrf[thour], varn)
            zFragment = wrf.vinterp(
                fwrf, np.squeeze(field), vert_coord="ght_agl", interp_levels=zlevs
            )

            zFragment = np.expand_dims(zFragment, axis=0)
            z.append(zFragment)
            t.append(tFragment)

        fieldint = np.concatenate(z, axis=0)
        otimes = np.concatenate(t, axis=0)

        varinfo = {
            "values": fieldint,
            "varname": varn,
            "zlevs": zlevs,
            "atts": atts,
            "lat": fwrf.variables["XLAT"][0, :],
            "lon": fwrf.variables["XLONG"][0, :],
            "times": otimes,
        }

        fileout = "%s/%s_ZLEVS_%s_%s.nc" % (fullpathout, patt, varn, sdate)
        create_zlevs_netcdf(varinfo, fileout)

        edate = otimes[-1] + dt.timedelta(days=1)

        y = edate.year
        m = edate.month
        d = edate.day


###########################################################
###########################################################


def zlevs_interp_byday(
    fullpathin, fullpathout, geofile_ref, date, zlevs, patt, patt_wrf, dom, wrun, varn
):
    # fullpathin = path_in + "/" + wrun + "/out"
    # fullpathout = path_out + "/" + wrun + "/" + str(syear) + "-" + str(eyear)
    inputinf = {}
    inputinf["geofile"] = geofile_ref
    geofile = nc.Dataset(geofile_ref)

    y = date.year
    m = date.month
    d = date.day

    sdate = "%s-%s-%s" % (y, str(m).rjust(2, "0"), str(d).rjust(2, "0"))

    filesin_wrf = sorted(glob("%s/%s_%s_%s*" % (fullpathin, patt_wrf, dom, sdate)))

    z = []
    t = []

    for thour in range(len(filesin_wrf)):
        fwrf = nc.Dataset(filesin_wrf[thour])
        fwrf.variables["F"] = geofile.variables["F"]
        tFragment = wrftime2date(filesin_wrf[thour].split())[:]
        field, atts = cvars.compute_WRFvar(filesin_wrf[thour], varn)
        zFragment = wrf.vinterp(
            fwrf, np.squeeze(field), vert_coord="ght_agl", interp_levels=zlevs
        )

        zFragment = np.expand_dims(zFragment, axis=0)
        z.append(zFragment)
        t.append(tFragment)

    fieldint = np.concatenate(z, axis=0)
    otimes = np.concatenate(t, axis=0)

    varinfo = {
        "values": fieldint,
        "varname": varn,
        "zlevs": zlevs,
        "atts": atts,
        "lat": fwrf.variables["XLAT"][0, :],
        "lon": fwrf.variables["XLONG"][0, :],
        "times": otimes,
    }

    fileout = "%s/%s_ZLEVS_%s_%s.nc" % (fullpathout, patt, varn, sdate)
    create_zlevs_netcdf(varinfo, fileout)


###########################################################
###########################################################


def create_10min_files(
    fullpathin, fullpathout, syear, eyear, smonth, emonth, patt_inst, varn
):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s*" % (fullpathin, patt_inst, varn, sdate))

        fin = "%s/%s_%s_%s*" % (fullpathin, patt_inst, varn, sdate)
        fout = "%s/%s_10MIN_%s_%s.nc" % (fullpathout, patt_inst, varn, sdate)

        os.system("ncrcat %s %s" % (fin, fout))

        edate = dt.datetime(y, m, 0o1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################
def create_hourly_files_cdo(fullpathout, syear, eyear, smonth, emonth, patt, varn):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate))

        fin = "%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate)

        fout = fin.replace("10MIN_%s" % (varn), "01H_%s" % (varn))

        print("Input: ", fin)
        print("Output: ", fout)

        os.system("cdo hourmean %s %s" % (fin, fout))

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################
def create_hourly_files(
    fullpathin, fullpathout, syear, eyear, smonth, emonth, patt_inst, varn
):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s*" % (fullpathin, patt_inst, varn, sdate))

        fin = "%s/%s_%s_%s*" % (fullpathin, patt_inst, varn, sdate)
        fout = "%s/%s_01H_%s_%s.nc" % (fullpathout, patt_inst, varn, sdate)

        os.system("ncrcat %s %s" % (fin, fout))

        edate = dt.datetime(y, m, 0o1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################


def create_hourly_files_byday(
    fullpathin, fullpathout, syear, eyear, smonth, emonth, patt_inst, varn
):
    y = syear
    m = smonth
    d = 1

    while y < eyear or (y == eyear and m <= emonth):
        for d in range(1, calendar.monthrange(y, m)[1] + 1):
            sdate = "%s-%02d-%02d" % (y, m, d)
            print("%s/%s_%s_%s*" % (fullpathin, patt_inst, varn, sdate))
            fin = "%s/%s_%s_%s*" % (fullpathin, patt_inst, varn, sdate)
            fout = "%s/%s_01H_%s_%s.nc" % (fullpathout, patt_inst, varn, sdate)

            os.system("ncrcat %s %s" % (fin, fout))

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################


def create_daily_files(fullpathout, syear, eyear, smonth, emonth, patt, varn):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate))

        fin = "%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate)

        fout = fin.replace("01H_%s" % (varn), "DAY_%s" % (varn))

        print("Input: ", fin)
        print("Output: ", fout)

        os.system("cdo daymean %s %s" % (fin, fout))

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################
def create_daily_files_byday(fullpathout, syear, eyear, smonth, emonth, patt, varn):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        for d in range(1, calendar.monthrange(y, m)[1] + 1):
            sdate = "%s-%02d_%02d" % (y, m, d)
            sdate_m = "%s-%s" % (y, str(m).rjust(2, "0"))
            print("%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate))
            fin = "%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate)
            fout = fin.replace("01H_%s" % (varn), "DAY_%s" % (varn))

            print("Input: ", fin)
            print("Output: ", fout)

            os.system("cdo daymean %s %s" % (fin, fout))

        fin_all_aux = "%s/%s_%s_%s_*" % (fullpathout, patt, varn, sdate_m)
        fin_all = fin_all_aux.replace("01H_%s" % (varn), "DAY_%s" % (varn))
        fout_day = fin_all.replace("-*", ".nc")
        os.system("cdo cat %s %s" % (fin_all, fout_day))
        os.system("rm -f %s" % (fin_all))

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)
        y = edate.year
        m = edate.month


###########################################################
###########################################################


def create_monthly_files(fullpathout, syear, eyear, smonth, emonth, patt, varn):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate))

        fin = "%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate)

        fout = fin.replace("DAY_%s" % (varn), "MON_%s" % (varn))

        print("Input: ", fin)
        print("Output: ", fout)

        os.system("cdo monmean %s %s" % (fin, fout))

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################


def create_diurnalcycle_files(fullpathout, syear, eyear, smonth, emonth, patt, varn):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate))
        fin = "%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate)

        fout = fin.replace("01H_%s" % (varn), "DCYCLE_%s" % (varn))

        print(fin, fout)

        fin_xr = xr.open_dataset(fin)
        fout_xr = np.squeeze(fin_xr.groupby("time.hour").mean("time"))

        fout_xr.to_netcdf(fout, mode="w", format="NETCDF4_CLASSIC")

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################


def create_diurnalcycle_files_cdo(
    fullpathout, syear, eyear, smonth, emonth, patt, varn
):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))

        print("%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate))
        fin = "%s/%s_%s_%s.nc" % (fullpathout, patt, varn, sdate)

        fout = fin.replace("01H_%s" % (varn), "DCYCLE_%s" % (varn))

        print(fin, fout)
        print("%s/dcycle_temp_%s%s" % (fullpathout, varn, sdate))

        if not os.path.exists("%s/dcycle_temp_%s%s" % (fullpathout, varn, sdate)):
            os.makedirs("%s/dcycle_temp_%s%s" % (fullpathout, varn, sdate))

        os.system(
            "cdo splitday %s %s/dcycle_temp_%s%s/aux" % (fin, fullpathout, varn, sdate)
        )
        os.system(
            "cdo ensmean %s/dcycle_temp_%s%s/aux* %s" % (fullpathout, varn, sdate, fout)
        )

        # auxfiles = sorted(glob("%s/dcycle_temp_%s%s/aux??.nc" %(fullpathout,varn,sdate)))
        # for auxf in auxfiles:
        #
        #     aux_tm = auxf.replace("aux","aux_timmean")
        #     os.system('cdo timmean %s %s' %(auxf,aux_tm))
        #
        # os.system('cdo mergetime %s/dcycle_temp_%s%s/aux_timmean* %s' %(fullpathout,varn,sdate,fout))

        os.system("rm -fr %s/dcycle_temp_%s%s/" % (fullpathout, varn, sdate))

        print(fin, fout)

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month


###########################################################
###########################################################


def create_diurnalcycle_files_cdo_hourlyfiles(
    fullpathout, syear, eyear, smonth, emonth, patt, varn
):
    y = syear
    m = smonth

    while y < eyear or (y == eyear and m <= emonth):
        cwd = os.getcwd()
        os.chdir(fullpathout)
        sdate = "%s-%s" % (y, str(m).rjust(2, "0"))
        finpat = "%s_%s_%s.nc" % (patt, varn, sdate)
        fout = finpat.replace("01H_%s" % (varn), "DCYCLE_%s" % (varn))
        print(finpat, fout)
        os.system("cdo ensmean %s_%s_%s*  %s" % (patt, varn, sdate, fout))

        os.chdir(cwd)

        edate = dt.datetime(y, m, 1) + relativedelta(months=1)

        y = edate.year
        m = edate.month
