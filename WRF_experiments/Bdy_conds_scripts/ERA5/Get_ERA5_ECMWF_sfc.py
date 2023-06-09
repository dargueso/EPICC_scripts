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
    yearStart = 2010
    yearEnd = 2011
    monthStart =12 
    monthEnd = 1

    y = yearStart
    m = monthStart
    d = 22

    while y < yearEnd or (y == yearEnd and m <= monthEnd):

        target = "era5_daily_sfc_%04d%02d%02d.grb" % (y, m, d)
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
            "format": "grib",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "land_sea_mask",
                "mean_sea_level_pressure",
                "sea_ice_cover",
                "sea_surface_temperature",
                "skin_temperature",
                "snow_depth",
                "soil_temperature_level_1",
                "soil_temperature_level_2",
                "soil_temperature_level_3",
                "soil_temperature_level_4",
                "surface_pressure",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
            ],
            "year": "%s" % (year),
            "month": "%02d" % (month),
            "day": "%02d" % (day),
            "time": ["00:00", "03:00", "06:00","09:00", "12:00","15:00", "18:00","21:00"], 
            "grid": [0.3, 0.3],
        },
        target,
    )


if __name__ == "__main__":
    retrieve_era5()
