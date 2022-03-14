import sys

import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from scipy.spatial import Delaunay
import cartopy.feature as cfeature

from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

import geocat.viz as viz
import geocat.viz.util as gvutil


def normalize_coords(lat, lon):
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    lon[lon > 180.0] -= 360
    return lat, lon

print('Argv[1]: ',sys.argv[1])
mesh = Dataset(sys.argv[1])

ter = mesh.variables['ter']
latCell = mesh.variables['latCell'][:]
lonCell = mesh.variables['lonCell'][:]
latCell, lonCell = normalize_coords(latCell, lonCell)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()

gvutil.set_axes_limits_and_ticks(ax,
                                 xlim=(-180, 180),
                                 ylim=(-90, 90),
                                 xticks=np.linspace(-180,180,13),
                                 yticks=np.linspace(-90,90,7))

gvutil.add_major_minor_ticks(ax, labelsize=10)
gvutil.add_lat_lon_ticklabels(ax)

gvutil.set_titles_and_labels(ax,
                             maintitle="Terrain Tri-Countourf Plot",
                             lefttitle=ter.long_name,
                             lefttitlefontsize=13,
                             righttitle=ter.units,
                             righttitlefontsize=13,
                             xlabel="",
                             ylabel="")

contour = ax.tricontourf(lonCell, latCell, ter[:], 
                         levels=100,
                         cmap='terrain', 
                         transform=ccrs.PlateCarree())

fig.colorbar(contour, orientation="horizontal")


plt.savefig('mpas_terrain.png')
