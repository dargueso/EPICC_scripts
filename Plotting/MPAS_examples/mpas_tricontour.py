import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('var', type=str, help="Desired variable to plot")
parser.add_argument('vlevel', type=int, help="Desired vertical level")
parser.add_argument('file',
                    type=str,
                    help='''File you want to plot from''')
args = parser.parse_args()

mesh = Dataset(args.file)
varName = args.var
vlevel = args.vlevel

var = mesh.variables[varName]
field = var[:]
field = field.squeeze()

latCell = mesh.variables['latCell'][:]
lonCell = mesh.variables['lonCell'][:]
zgrid = mesh.variables['zgrid'][:]
latCell, lonCell = normalize_coords(latCell, lonCell)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
# set_global() is currently needed for global plots. It seems Cartopy (or Matplotlib) currently has
# issues with setting the extent. See https://github.com/SciTools/cartopy/issues/1345
ax.set_global()
ax.add_feature(cfeature.LAND, color='silver')

gvutil.set_axes_limits_and_ticks(ax,
                                 xlim=(-180, 180),
                                 ylim=(-90, 90),
                                 xticks=np.linspace(-180,180,13),
                                 yticks=np.linspace(-90,90,7))

gvutil.add_major_minor_ticks(ax)
gvutil.add_lat_lon_ticklabels(ax)

ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

gvutil.set_titles_and_labels(ax,
                             maintitle="{0} at {1:.0f} m".format(var.name, zgrid[0,vlevel]),
                             lefttitle=var.long_name,
                             lefttitlefontsize=13,
                             righttitle=var.units,
                             righttitlefontsize=13,
                             xlabel="",
                             ylabel="")

contour = ax.tricontour(lonCell, latCell, field[:,vlevel], 
                        colors='black',
                        levels=7,
                        linewidths=0.5,
                        linestyles='-',
                        transform=ccrs.PlateCarree())

ax.clabel(contour, contour.levels, fontsize=12, fmt="%.0f")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig('mpas_{0}_l{1}.png'.format(var.name, vlevel))
