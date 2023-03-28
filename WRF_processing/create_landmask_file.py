#!/usr/bin/env python
"""
@File    :  create_landmask_file.py
@Time    :  2023/03/28 16:07:21
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  EPICC   
@Desc    :  None
"""

import netCDF4 as nc
import numpy as np
import xarray as xr
import datetime as dt

import wrf as wrf
import EPICC_post_config as cfg
from constants import const as const


def main():
    var = {}
    varname = "LANDMASK"
    fwrfgeo = xr.open_dataset(f"{cfg.path_geo}/{cfg.file_geo}")
    var["values"] = fwrfgeo.variables[varname].values.squeeze()
    var["lat"] = fwrfgeo.variables["XLAT_M"].values.squeeze()
    var["lon"] = fwrfgeo.variables["XLONG_M"].values.squeeze()
    var["varname"] = varname

    fullpathout = cfg.path_proc + "/" + cfg.wruns[0]
    fileout = f"{fullpathout}/{cfg.institution}_{varname}.nc"
    create_netcdf(var, fileout)


###########################################################
###########################################################
def create_netcdf(var, filename):
    print((("\n Create output file %s") % (filename)))

    outfile = nc.Dataset(
        filename, "w", format="NETCDF4_CLASSIC", zlib=True, complevel=5
    )
    outfile.createDimension("y", var["values"].shape[0])
    outfile.createDimension("x", var["values"].shape[1])

    outvar = outfile.createVariable(
        var["varname"],
        "f",
        ("y", "x"),
        zlib=True,
        complevel=5,
        fill_value=const.missingval,
    )

    outlat = outfile.createVariable(
        "lat", "f", ("y", "x"), zlib=True, complevel=5, fill_value=const.missingval
    )
    outlon = outfile.createVariable(
        "lon", "f", ("y", "x"), zlib=True, complevel=5, fill_value=const.missingval
    )

    setattr(outlat, "standard_name", "latitude")
    setattr(outlat, "long_name", "Latitude")
    setattr(outlat, "units", "degrees_north")
    setattr(outlat, "_CoordinateAxisType", "Lat")

    setattr(outlon, "standard_name", "longitude")
    setattr(outlon, "long_name", "Longitude")
    setattr(outlon, "units", "degrees_east")
    setattr(outlon, "_CoordinateAxisType", "Lon")

    outlat[:] = var["lat"][:]
    outlon[:] = var["lon"][:]

    outvar[:] = var["values"][:]

    setattr(outvar, "description", "Land-sea mask")
    setattr(outvar, "units", "1/0")

    setattr(outfile, "creation_date", dt.datetime.today().strftime("%Y-%m-%d"))
    setattr(outfile, "author", "Daniel Argueso @UIB")
    setattr(outfile, "contact", "d.argueso@uib.es")
    # setattr(outfile,'comments','files created from wrf outputs %s/%s' %(path_in,patt))

    outfile.close()


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    main()

###############################################################################
