#!/usr/bin/env python
'''
@File    :  Seasonal_distribution_topN.py
@Time    :  2025/07/24 12:33:07
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  None
@Desc    :  Modified to add transparency based on dominance strength
'''

import xarray as xr
import numpy as np
import time as ttime
import config as cfg
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as crs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colormaps as cmaps
from wrf import (
    to_np,
    getvar,
    get_cartopy,
    cartopy_xlim,
    GeoBounds,
    CoordPair,
    cartopy_ylim,
    latlon_coords,
)
import cmocean


topN = 20
mode = 'peak'  # 'peak'
output_filename = f"{mode}_top{topN}_seasonal_dist.png"
output_filename2 = f"{mode}_top{topN}_seasonal_dom.png"
output_filename3 = f"{mode}_top{topN}_seasonal_dom_transparent.png"

finp = xr.open_dataset(f"{cfg.path_out}/EPICC_2km_ERA5/Hourly_decomposition_top100_NDI.nc")
finf = xr.open_dataset(f"{cfg.path_out}/EPICC_2km_ERA5_CMIP6anom/Hourly_decomposition_top100_NDI.nc")

Ip = finp[mode].isel(event=slice(0,topN)).mean(dim='event')
Dp = finp[f'{mode}_duration'].isel(event=slice(0,topN)).mean(dim='event')

If = finf[mode].isel(event=slice(0,topN+1)).mean(dim='event')
Df = finf[f'{mode}_duration'].isel(event=slice(0,topN)).mean(dim='event')

# Seasonal distribution present
da = finp[f'{mode}_time'].isel(event=slice(0, topN)).dt.season  # (event, y, x)
da = da.assign_coords(event=np.arange(da.sizes['event']))
seasons = ['DJF', 'MAM', 'JJA', 'SON']
percent = xr.concat(
    [(da == s).sum(dim='event') / da.sizes['event'] * 100 for s in seasons],
    dim='season'
)
percent = percent.assign_coords(season=seasons)

# Seasonal distribution future
da_f = finf[f'{mode}_time'].isel(event=slice(0, topN)).dt.season  # (event, y, x)
da_f = da_f.assign_coords(event=np.arange(da_f.sizes['event']))
percent_f = xr.concat(
    [(da_f == s).sum(dim='event') / da_f.sizes['event'] * 100 for s in seasons],
    dim='season'
)
percent_f = percent_f.assign_coords(season=seasons)

## PLOTTING ##

geo_filename = "/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
print(f"Loading geo file: {geo_filename}")
geo_file = xr.open_dataset(geo_filename)

with nc.Dataset(geo_filename) as fileref:
    ds = getvar(fileref, "LU_INDEX")
    lats, lons = latlon_coords(ds)
    cart_proj = get_cartopy(ds)
    xbounds = cartopy_xlim(ds)
    ybounds = cartopy_ylim(ds)
    terrain = getvar(fileref, "ter")
    terrain = xr.where(terrain < 0, 0, terrain)

def create_border_mask(data_array, border_width=5):
    """Create a mask that sets border pixels to NaN"""
    mask = np.ones_like(data_array, dtype=bool)
    mask[border_width:-border_width, border_width:-border_width] = False
    return mask

# Original seasonal distribution plot
fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(15,12),subplot_kw=dict(projection=cart_proj))

mylevels = np.arange(0, 112.5, 12.5)
cmap = cmocean.cm.deep  # , len(mylevels_diff)+1)
norm = BoundaryNorm(mylevels, ncolors=cmap.N)
axs = axs.ravel()

for ns, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
    axs[ns].set_title(season, fontsize=14, fontweight='bold')
    axs[ns].coastlines("10m", linewidth=0.8, zorder=102)
    m0 = axs[ns].pcolormesh(
        to_np(lons), to_np(lats),
        percent.sel(season=season).squeeze(),
        transform=crs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True
    )
    gl1 = axs[ns].gridlines(color="black", linestyle="dotted", draw_labels=True,
                    x_inline=False, y_inline=False, zorder=102)

    # Add gridlines
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    axs[ns].set_xlim(xbounds)
    axs[ns].set_ylim(ybounds)

cbar = fig.colorbar(m0, ax=axs, ticks=np.arange(0, 125, 25), orientation='horizontal', fraction=0.05, pad=0.08)
cbar.set_label('%', fontsize=12)

fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.30, wspace=0.1, hspace=0.1)

print(f"Saving figure to: {output_filename}")
fig.savefig(output_filename, bbox_inches="tight", dpi=300)
print("Seasonal distribution plot completed!")

# Calculate dominance with transparency based on difference from second-most dominant
# Get the dominant season and its index
dominant_season = percent.idxmax(dim='season')
max_idx = percent.argmax(dim='season')

# Calculate the difference between first and second most dominant seasons
# Get the maximum value (first highest)
first_max = percent.max(dim='season')

# To get second highest, we'll mask the maximum and find the max of the remaining
# Create a mask where the maximum values are
max_mask = (percent == first_max)
# Set maximum values to a very small number and find the new maximum
percent_masked = percent.where(~max_mask, -999)
second_max = percent_masked.max(dim='season')

# Calculate the dominance strength (difference between 1st and 2nd)
dominance_strength = first_max - second_max

# Normalize dominance strength to alpha values (0-1)
# You can adjust these thresholds based on your data
min_diff = 0    # Minimum difference (most transparent)
max_diff = 30   # Maximum difference (fully opaque) - adjust based on your data range

# Create alpha values: 0.1 (very transparent) to 1.0 (fully opaque)
alpha_values = np.clip((dominance_strength - min_diff) / (max_diff - min_diff), 0.1, 1.0)

# Define the 4 seasons and their colors
season_labels = ['DJF', 'MAM', 'JJA', 'SON']
season_colors = ['#1f78b4',  # DJF (blue)
                 '#33a02c',  # MAM (green)
                 '#ff7f00',  # JJA (orange)
                 '#6a3d9a']  # SON (purple)

cmap = ListedColormap(season_colors)
bounds = [0, 1, 2, 3, 4]
norm = BoundaryNorm(bounds, cmap.N)

# Create the transparent dominant season plot
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(15, 12), 
                       subplot_kw=dict(projection=cart_proj))

axs.set_title('Dominant Season (Transparency = Dominance Strength)', fontsize=14, fontweight='bold')
axs.coastlines("10m", linewidth=0.8, zorder=102)

# Plot with alpha based on dominance strength
m0 = axs.pcolormesh(
    to_np(lons), to_np(lats),
    max_idx,
    alpha=to_np(alpha_values),  # Apply transparency based on dominance strength
    transform=crs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    shading="auto",
    rasterized=True,
)

gl1 = axs.gridlines(color="black", linestyle="dotted", draw_labels=True,
                   x_inline=False, y_inline=False, zorder=102)

# Add gridlines
gl1.top_labels = False
gl1.right_labels = False
gl1.xformatter = LONGITUDE_FORMATTER
gl1.yformatter = LATITUDE_FORMATTER
gl1.xpadding = 10

# Set bounds
axs.set_xlim(xbounds)
axs.set_ylim(ybounds)

# Create colorbar for seasons
cbar = fig.colorbar(m0, ax=axs, ticks=np.arange(0.5, 4.5, 1), 
                   orientation='horizontal', fraction=0.05, pad=0.08)
cbar.set_ticklabels(season_labels)
cbar.set_label('Dominant Season', fontsize=12)

# Add text annotation explaining the transparency
fig.text(0.5, 0.02, 'Transparency indicates dominance strength: opaque = strong dominance, transparent = weak dominance', 
         ha='center', fontsize=10, style='italic')

print(f"Saving transparent figure to: {output_filename3}")
fig.savefig(output_filename3, bbox_inches="tight", dpi=300)

# Print some statistics about dominance strength
print(f"Dominance strength statistics:")
print(f"  Mean difference: {float(dominance_strength.mean()):.1f}%")
print(f"  Max difference: {float(dominance_strength.max()):.1f}%")
print(f"  Min difference: {float(dominance_strength.min()):.1f}%")
print(f"  Std difference: {float(dominance_strength.std()):.1f}%")

plt.show()

print("Visualization with transparency completed successfully!")