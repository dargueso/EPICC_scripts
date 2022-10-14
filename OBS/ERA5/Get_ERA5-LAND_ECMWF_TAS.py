#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso @ UIB
# Date:   2017-10-05T17:20:56+02:00
# Email:  d.argueso@uib.es
# Last modified by:   Daniel Argueso
# Last modified time: 2017-10-06T11:14:19+02:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:  cdsapi (needs API)
#
# Files:
#
#####################################################################
"""

#!/usr/bin/env python
import cdsapi
c = cdsapi.Client()


def retrieve_era5():
    """
       A function to demonstrate how to iterate efficiently over several years and months etc
       for a particular eraint request.
       Change the variables below to adapt the iteration to your needs.
       You can use the variable 'target' to organise the requested data in files as you wish.
       In the example below the data are organised in files per month. (eg "era5_hourly_201510.nc")
    """
    yearStart = 1950 
    yearEnd = 2022
    monthStart = 1
    monthEnd = 12
    for year in list(range(yearStart, yearEnd + 1)):
        for month in list(range(monthStart, monthEnd + 1)):

            target = f"era5_land_TAS_{year}{month:02d}.nc"
            era5_request(year,month, target)

def era5_request(year,month,target):
    """
        An era5 request for analysis surface level data.
        Change the keywords below to adapt it to your needs.
        (eg to add or to remove  levels, parameters, times etc)
    """
    c.retrieve(
        'reanalysis-era5-land',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': '2m_temperature',
            'year': [f'{year}'],
            'month': [f'{month:02d}'],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                80, -25, 10,
                50,
            ],
        },
        target)

if __name__ == '__main__':
    retrieve_era5()
