#!/usr/bin/env python
"""
@File    :  get_cyclone_precip.py
@Time    :  2023/07/28 13:38:32
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  EPICC
@Desc    :  This program looks for precipitation objects within a distance from the cyclone track
"""

import xarray as xr
from scipy.ndimage import filters
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np


class const:
    earth_radius = 6371000  # Earth radius in m


def haversine(lat1, lon1, lat2, lon2):
    """Function to calculate grid distances lat-lon
    This uses the Haversine formula
    lat,lon : input coordinates (degrees) - array or float
    dist_m : distance (m)
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    # convert decimal degrees to radians
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    dist_m = c * const.earth_radius
    return dist_m


#####################################################################
#####################################################################

# PARAMETERS

pr_threshold = 0.1  # in mm the threshold to delimit precipation objects (areas where it rains above this threshold)
max_dist_to_cyc = (
    1000  # Precip object center must be within 1000 km to the cyclone track
)

#####################################################################
#####################################################################
##
# I used this file to look for a particularly intense cyclone as an example
# I chose a cyclone in 2012-04-14 near Italian peninsula
ds = xr.open_dataset("cyclones_2011-2020_subdmn_CL8.nc")
index = 1125
id = 865
lon_cyc = 14.146
lat_cyc = 40.236
date = "2012-04-14 04"

#####################################################################
#####################################################################

pr = xr.open_dataset("era5_hourly_PR_201204.nc")
PR_time = pr.sel(time=date)
pr_mask = PR_time.tp >= (
    pr_threshold * 1e-3
)  # Convert to m - original unit in netCDF file

plt.pcolormesh(PR_time.tp)
plt.savefig("pr_cyc.png")

plt.pcolormesh(pr_mask)
plt.savefig("pr_mask.png")

#####################################################################
#####################################################################
# Detect objects

objects_id_pr, num_objects = ndimage.label(pr_mask)

# Calculate "mass-center" of objects - used to determine whether the system is within the area of inlfuence or not
# Other methods must be tested (e.g. any part of the object is within the area)

obj_mass_center = np.array(
    [
        ndimage.center_of_mass(objects_id_pr == (tt + 1))
        for tt in range(objects_id_pr.max())
    ]
)


# Select object within distance

obj_id_pr_cyc = objects_id_pr.copy() * 0
for id_pr in range(objects_id_pr.max()):
    dist_cyc = (
        haversine(
            lat_cyc,
            lon_cyc,
            pr.latitude[int(obj_mass_center[id_pr][0])],
            pr.longitude[int(obj_mass_center[id_pr][1])],
        )
        / 1000
    )  # Convert to km

    if dist_cyc.values < max_dist_to_cyc:
        obj_id_pr_cyc[objects_id_pr == (id_pr + 1)] = id_pr + 1


plt.pcolormesh(objects_id_pr)
plt.savefig("pr_objects.png")
plt.pcolormesh(obj_id_pr_cyc)
plt.savefig("pr_objects_cyclone.png")
