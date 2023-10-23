#!/usr/bin/env python
import datetime as dt
from dateutil.relativedelta import relativedelta
import cdsapi
import calendar

c = cdsapi.Client()


def retrieve_era5():
    """
    A function to demonstrate how to iterate efficiently over several years and months etc
    for a particular era5 request.
    Change the variables below to adapt the iteration to your needs.
    You can use the variable 'target' to organise the requested data in files as you wish.
    In the example below the data are organised in files per day. (eg "era5_daily_20151001.grb")
    """
    yearStart = 1970 
    yearEnd = 2023
    monthStart =1 
    monthEnd = 12

    y = yearStart
    m = monthStart
    d = 1 

    while y < yearEnd or (y == yearEnd and m <= monthEnd):

        target = "era5_daily_MSLP_%04d%02d%02d.nc" % (y, m, d)
        era5_request(y, m, d, target)

        edate = dt.datetime(y, m, d) + dt.timedelta(days=1)
        y = edate.year
        m = edate.month
        d = edate.day


def era5_request(year, month, day, target):
    """
    An ERA5 request for analysis pressure level data.
    Change the keywords below to adapt it to your needs.
    (eg to add or to remove  levels, parameters, times etc)
    Request cost per day is 112 fields, 14.2326 Mbytes
    """

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable":"mean_sea_level_pressure",
            "year": "%s" % (year),
            "month": "%02d" % (month),
            "day": "%02d" % (day),
            "time": ["00:00", "03:00", "06:00","09:00", "12:00","15:00", "18:00","21:00"],
        },
        target,
    )


if __name__ == "__main__":
    retrieve_era5()
